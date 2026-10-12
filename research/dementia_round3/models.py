"""Patient-level research models. No pretrained weights or downloads are implicit."""
from dataclasses import asdict, dataclass
import torch
from torch import nn

MODEL_VERSION = 'round3-patient-models-v1'
ANATOMICAL_FEATURES = ('left_hippocampus', 'right_hippocampus', 'left_lateral_ventricle',
                      'right_lateral_ventricle', 'left_cortex', 'right_cortex',
                      'left_cerebral_white_matter', 'right_cerebral_white_matter')


@dataclass(frozen=True)
class ModelConfig:
    kind: str = 'multiplanar'
    width: int = 16
    dropout: float = 0.1
    slice_chunk: int = 32
    features: tuple = ANATOMICAL_FEATURES

    def validate(self):
        if self.kind not in {'multiplanar', 'resnet3d18', 'anatomical_logistic'}:
            raise ValueError('Unsupported model family')
        if self.width < 4 or self.width % 4 or not 0 <= self.dropout < 1 or self.slice_chunk < 1:
            raise ValueError('Invalid model configuration')
        if not self.features or len(self.features) != len(set(self.features)):
            raise ValueError('Named anatomical features must be unique')


class ResidualBlock(nn.Module):
    def __init__(self, dim, incoming, outgoing, stride=1):
        super().__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.main = nn.Sequential(conv(incoming, outgoing, 3, stride, 1, bias=False),
                                  nn.GroupNorm(4, outgoing), nn.ReLU(),
                                  conv(outgoing, outgoing, 3, 1, 1, bias=False),
                                  nn.GroupNorm(4, outgoing))
        self.skip = (nn.Identity() if incoming == outgoing and stride == 1 else
                     nn.Sequential(conv(incoming, outgoing, 1, stride, bias=False),
                                   nn.GroupNorm(4, outgoing)))

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.stem = nn.Sequential(conv(1, width, 3, 2, 1, bias=False),
                                  nn.GroupNorm(4, width), nn.ReLU())
        stages = []
        incoming = width
        for i, outgoing in enumerate([width, width * 2, width * 4, width * 8]):
            stages.append(nn.Sequential(ResidualBlock(dim, incoming, outgoing, 1 if i == 0 else 2),
                                        ResidualBlock(dim, outgoing, outgoing)))
            incoming = outgoing
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        return self.stages(self.stem(x))


def validate_volume_tensor(x):
    if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 16:
        raise ValueError('Expected complete [patients,1,X,Y,Z] volumes, each dimension >=16')
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite model input')


class ResNet3D18(nn.Module):
    """Compact 18-layer residual network; one loss/output per complete volume."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(3, config.width)
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.width * 8, 2))

    def explanation_forward(self, x):
        validate_volume_tensor(x)
        features = self.encoder(x)
        logits = self.classifier(features.mean((2, 3, 4)))
        return logits, [features], {'kind': '3d', 'shape': list(x.shape[2:])}

    def forward(self, x):
        return self.explanation_forward(x)[0]


class MultiPlanarModel(nn.Module):
    """Shared 2D encoder of every full-FOV slice in all three canonical axes.

    Mean pooling across each entire axis and concatenation produce one patient
    representation. No slice selection, slice labels or central crop is used.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(2, config.width)
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.width * 24, 2))

    def explanation_forward(self, x):
        validate_volume_tensor(x)
        batch = x.shape[0]
        maps, vectors, counts = [], [], []
        for axis in range(3):
            d = axis + 2
            other = [i for i in (2, 3, 4) if i != d]
            slices = x.permute(0, d, 1, *other).reshape(-1, 1, x.shape[other[0]], x.shape[other[1]])
            features = torch.cat([self.encoder(part) for part in slices.split(self.config.slice_chunk)], 0)
            maps.append(features)
            count = x.shape[d]
            counts.append(count)
            vectors.append(features.mean((2, 3)).reshape(batch, count, -1).mean(1))
        logits = self.classifier(torch.cat(vectors, 1))
        return logits, maps, {'kind': 'multiplanar', 'batch': batch,
                              'shape': list(x.shape[2:]), 'slice_counts': counts}

    def forward(self, x):
        return self.explanation_forward(x)[0]


class AnatomicalLogistic(nn.Module):
    """Logistic baseline on real named segmentation volumes / intracranial volume.

    Features cannot be generated by an intensity threshold or by this model.
    Standardization buffers must be fitted on TRAIN patients only.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('mean', torch.zeros(len(config.features)))
        self.register_buffer('scale', torch.ones(len(config.features)))
        self.register_buffer('normalizer_fitted', torch.tensor(False))
        self.classifier = nn.Linear(len(config.features), 1)

    def fit_normalizer(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.config.features) or not torch.isfinite(x).all():
            raise ValueError('Invalid anatomical training features')
        with torch.no_grad():
            self.mean.copy_(x.mean(0))
            self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
            self.normalizer_fitted.fill_(True)

    def forward(self, x):
        if not bool(self.normalizer_fitted):
            raise ValueError('Anatomical normalizer has not been fitted on TRAIN')
        if x.ndim != 2 or x.shape[1] != len(self.config.features) or not torch.isfinite(x).all():
            raise ValueError('Invalid anatomical feature tensor')
        positive = self.classifier((x - self.mean) / self.scale)
        return torch.cat([torch.zeros_like(positive), positive], 1)


def build_model(config):
    config = ModelConfig(**config) if isinstance(config, dict) else config
    config.validate()
    return {'multiplanar': MultiPlanarModel, 'resnet3d18': ResNet3D18,
            'anatomical_logistic': AnatomicalLogistic}[config.kind](config)


def model_metadata(model):
    return {'version': MODEL_VERSION, 'config': asdict(model.config),
            'parameters': sum(p.numel() for p in model.parameters()),
            'unit': 'one complete patient volume or named anatomical feature vector'}
