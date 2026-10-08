"""Headless checkpoint/preprocessing checks; no real weights or display needed."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

from src.experiment_pipeline import build_transforms


def tiny_classifier(*_args):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))


@unittest.skipIf(torch is None, "PyTorch is not installed in the artifact-only environment")
class GuiCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reporting integrations are unrelated to model loading. Stub only these
        # optional packages so the real GUI loader can run without an LLM/PDF stack.
        optional_modules = {}
        for name, exports in {
            "ollama": [],
            "tkhtmlview": ["HTMLLabel"],
            "tkcalendar": ["DateEntry"],
            "weasyprint": ["HTML"],
            "pypdf": ["PdfReader", "PdfWriter"],
        }.items():
            module = types.ModuleType(name)
            for export in exports:
                setattr(module, export, Mock())
            optional_modules[name] = module
        previous = {name: sys.modules.get(name) for name in optional_modules}
        sys.modules.update(optional_modules)
        try:
            cls.gui = importlib.import_module("src.radiology_report_gui")
        finally:
            # Restore only the optional packages, retaining modules imported by
            # torch/torchvision (which register native operators on first import).
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def setUp(self):
        self.app = types.SimpleNamespace(
            device=torch.device("cpu"),
            tumor_model=None,
            alz_model=None,
            image_path=None,
            model_status_label=Mock(),
            generate_btn=Mock(),
        )
        self.image = Image.new("RGB", (83, 57), color=(23, 119, 211))

    def checkpoint(self, arch, classes, image_size=None):
        result = {
            "arch": arch,
            "class_names": classes,
            "model_state": tiny_classifier().state_dict(),
        }
        if image_size is not None:
            result["image_size"] = image_size
        return result

    def load(self, tumor, dementia):
        checkpoints = {
            self.gui.TUMOR_MODEL_PATH: tumor,
            self.gui.ALZHEIMERS_MODEL_PATH: dementia,
        }
        with (
            patch.object(self.gui.os.path, "isfile", side_effect=lambda path: path in checkpoints),
            patch.object(self.gui.torch, "load", side_effect=lambda path, **_: checkpoints[path]),
            patch.object(self.gui, "build_tumor_model", side_effect=tiny_classifier),
            patch.object(self.gui, "build_alzheimers_model", side_effect=tiny_classifier),
        ):
            self.gui.App.load_models(self.app)
        status = self.app.model_status_label.configure.call_args.kwargs["text"]
        self.assertIn("Tumor: Ready", status)
        self.assertIn("Alzheimer's: Ready", status)

    def assert_preprocessing(self, tumor_size, dementia_size):
        for transform, size in [
            (self.app.tumor_tfms, tumor_size),
            (self.app.alz_tfms, dementia_size),
        ]:
            actual = transform(self.image)
            self.assertEqual(tuple(actual.shape), (3, size, size))
            torch.testing.assert_close(
                actual, build_transforms(train=False, image_size=size)(self.image),
                rtol=0, atol=0,
            )

    def test_each_specialist_uses_its_checkpoint_size_and_training_normalization(self):
        self.load(
            self.checkpoint("efficientnet_b3", self.gui.TUMOR_CLASSES_4, 288),
            self.checkpoint("mobilenet_v3_large", self.gui.ALZ_CLASSES_4, 160),
        )
        self.assert_preprocessing(288, 160)

    def test_reload_legacy_models_resets_previous_checkpoint_preprocessing(self):
        self.load(
            self.checkpoint("efficientnet_b3", self.gui.TUMOR_CLASSES_4, 288),
            self.checkpoint("mobilenet_v3_large", self.gui.ALZ_CLASSES_4, 160),
        )
        self.load(tiny_classifier(), tiny_classifier())
        self.assert_preprocessing(224, 224)

    def test_grad_cam_reuses_the_tensor_preprocessing_used_for_classification(self):
        self.load(
            self.checkpoint("efficientnet_b3", self.gui.TUMOR_CLASSES_4, 288),
            self.checkpoint("mobilenet_v3_large", self.gui.ALZ_CLASSES_4, 160),
        )
        self.app.gatekeeper_mode = "binary"
        self.app.gatekeeper_classes = ["tumor", "dementia"]
        self.app.gate_tfms = build_transforms(train=False, image_size=224)
        with tempfile.TemporaryDirectory() as temporary:
            self.app.image_path = str(Path(temporary) / "image.png")
            self.image.save(self.app.image_path)
            for index, model in enumerate([self.app.tumor_model, self.app.alz_model]):
                with self.subTest(domain=self.app.gatekeeper_classes[index]):
                    logits = torch.full((1, 2), -10.0)
                    logits[0, index] = 10.0
                    self.app.gatekeeper_model = Mock(return_value=logits)
                    observed = []
                    handle = model.register_forward_pre_hook(
                        lambda _model, inputs: observed.append(inputs[0].clone())
                    )
                    try:
                        result = self.gui.App._run_cascaded_classification(self.app)
                    finally:
                        handle.remove()
                    self.assertIs(result.cam_model, model)
                    cam_input = result.cam_transform(result.source_image).unsqueeze(0)
                    torch.testing.assert_close(cam_input, observed[0], rtol=0, atol=0)

    def test_checkpoint_without_size_retains_default_224(self):
        self.load(
            self.checkpoint("efficientnet_b3", self.gui.TUMOR_CLASSES_4),
            self.checkpoint("mobilenet_v3_large", self.gui.ALZ_CLASSES_4),
        )
        self.assert_preprocessing(224, 224)


if __name__ == "__main__":
    unittest.main()
