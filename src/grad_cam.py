"""Hook-based Grad-CAM for the tumor and dementia specialist classifiers.

Grad-CAM needs a backward pass, but every inference path in
`radiology_report_gui.py` runs under `@torch.no_grad()` for speed. This
module is intentionally decoupled from that path: callers pass in an
already-loaded model and a single input tensor, and `GradCAM.generate`
runs its own short-lived autograd pass, including when its caller is in
`torch.no_grad()` or `torch.inference_mode()`. Parameter gradients are left
untouched.

Failures (including an unsupported backward op on an unusual device backend
such as DirectML) are caught and reported as `None` rather than raised, so a
Grad-CAM failure never breaks the classification result it is explaining.
"""

from __future__ import annotations

import logging
import operator
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last spatial block for architectures supported by the pipeline."""

    features = getattr(model, "features", None)
    if isinstance(features, (nn.Sequential, nn.ModuleList)) and len(features) > 0:
        return features[-1]
    # experiment_pipeline also supports ResNet specialists.
    if isinstance(getattr(model, "layer4", None), nn.Module):
        return model.layer4
    raise ValueError(
        f"Don't know how to find a Grad-CAM target layer for {type(model).__name__}; "
        "expected `.features` (EfficientNet / MobileNet) or `.layer4` (ResNet)."
    )


class GradCAM:
    """Hook-based Grad-CAM against a single target layer of `model`."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
        """Return a (H, W) float array in [0, 1], or None on failure.

        `input_tensor` must be a single unbatched or batch-of-1 floating-point
        tensor already on the model's device. Invalid/flat/nonfinite results
        return None. The caller's input, gradients, and module modes are preserved.
        """

        activations: dict[str, torch.Tensor] = {}
        def forward_hook(_module, _inputs, output):
            activations["value"] = output

        # Restoring only model.training with model.train(...) would overwrite
        # deliberately mixed modes, such as a frozen BatchNorm in a training model.
        training_modes = [(module, module.training) for module in self.model.modules()]
        forward_handle = None
        try:
            class_idx = operator.index(class_idx)
            if not isinstance(input_tensor, torch.Tensor) or not input_tensor.is_floating_point():
                raise ValueError("Grad-CAM requires a floating-point input tensor")
            if input_tensor.ndim not in (3, 4) or (input_tensor.ndim == 4 and input_tensor.shape[0] != 1):
                raise ValueError("Grad-CAM requires one image with shape (C,H,W) or (1,C,H,W)")
            self.model.eval()
            forward_handle = self.target_layer.register_forward_hook(forward_hook)

            # enable_grad alone does not override inference_mode. Clone inside
            # this context so an inference tensor becomes autograd-compatible.
            with torch.inference_mode(False), torch.enable_grad():
                x = input_tensor.clone().detach()
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                x.requires_grad_(True)

                output = self.model(x)
                if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
                    raise ValueError("Grad-CAM requires logits with shape (1, number_of_classes)")
                if class_idx < 0 or class_idx >= output.shape[1]:
                    logger.warning("Grad-CAM class_idx %s out of range for output shape %s", class_idx, output.shape)
                    return None
                if not bool(torch.isfinite(output).all()):
                    logger.warning("Grad-CAM received nonfinite logits; skipping heatmap")
                    return None
                if "value" not in activations:
                    logger.warning("Grad-CAM target layer was not reached in the forward graph")
                    return None

                captured = activations["value"]
                if not isinstance(captured, torch.Tensor) or captured.ndim != 4 or captured.shape[0] != 1:
                    raise ValueError("Grad-CAM target must produce a spatial feature tensor (1,C,H,W)")
                # Direct differentiation avoids module backward hooks, whose
                # output views conflict with in-place activations. Unlike
                # score.backward(), it also never populates parameter .grad.
                gradient, = torch.autograd.grad(output[0, class_idx], captured)
                activation = captured.detach()[0].float()  # (C, h, w)
                gradient = gradient.detach()[0].float()

                weights = gradient.mean(dim=(1, 2))  # (C,)
                cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))  # (h, w)

                if not bool(torch.isfinite(cam).all()):
                    logger.warning("Grad-CAM produced a nonfinite map; skipping heatmap")
                    return None

                cam_min = cam.min()
                cam_max = cam.max()
                if float(cam_max - cam_min) < 1e-12:
                    logger.warning("Grad-CAM produced a flat map (max - min ~= 0); skipping heatmap")
                    return None
                cam = (cam - cam_min) / (cam_max - cam_min)

                return cam.cpu().numpy().astype(np.float32)

        except Exception:
            logger.exception("Grad-CAM generation failed")
            return None
        finally:
            if forward_handle is not None:
                forward_handle.remove()
            for module, was_training in training_modes:
                module.training = was_training


def overlay_cam_on_image(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Upsample `cam` to `pil_image`'s resolution and alpha-blend a jet colormap over it."""

    import matplotlib

    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not cam.size or not np.isfinite(cam).all():
        raise ValueError("The heatmap must be a nonempty, finite 2D array")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be finite and between 0 and 1")

    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size

    cam_tensor = torch.from_numpy(np.ascontiguousarray(cam)).unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    resized = F.interpolate(cam_tensor, size=(height, width), mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).squeeze(0).clamp(0, 1).numpy()  # (H, W) in [0, 1]

    colormap = matplotlib.colormaps["jet"]
    heatmap_rgba = colormap(resized)  # (H, W, 4) floats in [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.float32)

    base_rgb = np.asarray(rgb_image, dtype=np.float32)
    blended = base_rgb * (1.0 - alpha) + heatmap_rgb * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")
