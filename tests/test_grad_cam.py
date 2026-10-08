"""Analytical Grad-CAM contracts; no training, checkpoints, or network.

The artifact-only CI environment omits torch. Run scripts/test_grad_cam_contract.py
in the model environment to require these checks instead of skipping them.
"""
import unittest
import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image


@unittest.skipIf(torch is None, "PyTorch is not installed in the artifact-only environment")
class GradCAMTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        with torch.no_grad():
            self.model[0].weight.fill_(1)
        self.x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        self.expected = self.x[0, 0].numpy() / 15

    def assert_clean(self, model=None):
        model = self.model if model is None else model
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._backward_hooks)

    def test_analytical_cam_repeatability_under_no_grad(self):
        for _ in range(3):
            with torch.no_grad():
                result = GradCAM(self.model, self.model[0]).generate(self.x, 0)
            np.testing.assert_allclose(result, self.expected, atol=1e-7)
            self.assert_clean()

    def test_inference_mode_and_inference_tensor(self):
        with torch.inference_mode():
            x = self.x.clone()
            self.assertTrue(torch.is_inference(x))
            result = GradCAM(self.model, self.model[0]).generate(x, 0)
            self.assertTrue(torch.is_inference_mode_enabled())
        np.testing.assert_allclose(result, self.expected, atol=1e-7)
        self.assert_clean()

    def test_inplace_activation_after_target(self):
        model = nn.Sequential(self.model[0], nn.ReLU(inplace=True), self.model[1], self.model[2])
        x = self.x - 5
        result = GradCAM(model, model[0]).generate(x, 0)
        np.testing.assert_allclose(result, x[0, 0].clamp_min(0).numpy() / 10, atol=1e-7)
        self.assert_clean(model)

    def test_preserves_gradients_input_and_mixed_module_modes(self):
        self.model.train()
        self.model[0].eval()
        modes = [module.training for module in self.model.modules()]
        self.model[0].weight.grad = torch.full_like(self.model[0].weight, 7)
        original_grad = self.model[0].weight.grad
        self.x.requires_grad_(True)
        self.x.grad = torch.full_like(self.x, 3)
        original_input = self.x.detach().clone()
        result = GradCAM(self.model, self.model[0]).generate(self.x, 0)
        np.testing.assert_allclose(result, self.expected, atol=1e-7)
        self.assertIs(self.model[0].weight.grad, original_grad)
        self.assertTrue(torch.equal(original_grad, torch.full_like(original_grad, 7)))
        self.assertTrue(torch.equal(self.x, original_input))
        self.assertTrue(torch.equal(self.x.grad, torch.full_like(self.x, 3)))
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_unbatched_input_and_frozen_parameters(self):
        self.model.requires_grad_(False)
        result = GradCAM(self.model, self.model[0]).generate(self.x[0], np.int64(0))
        np.testing.assert_allclose(result, self.expected, atol=1e-7)
        self.assertIsNone(self.model[0].weight.grad)

    def test_invalid_classes_shapes_and_flat_maps_fail_safely(self):
        invalid = [
            (self.x, -1), (self.x, 1), (self.x, 0.5),
            (self.x.repeat(2, 1, 1, 1), 0), (self.x[0, 0], 0),
            (self.x.long(), 0), (torch.zeros_like(self.x), 0),
            (torch.full_like(self.x, float("nan")), 0),
        ]
        for x, target in invalid:
            with self.subTest(shape=x.shape, target=target), self.assertLogs("src.grad_cam"):
                self.assertIsNone(GradCAM(self.model, self.model[0]).generate(x, target))
            self.assert_clean()
            self.assertTrue(self.model.training)

    def test_unreachable_target_and_failure_cleanup(self):
        with self.assertLogs("src.grad_cam"):
            self.assertIsNone(GradCAM(self.model, nn.Conv2d(1, 1, 1)).generate(self.x, 0))

        class FailingModel(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, x):
                self.layer(x)
                raise RuntimeError("Deliberate forward failure")

        model = FailingModel(self.model[0])
        model.layer.eval()
        with self.assertLogs("src.grad_cam"):
            self.assertIsNone(GradCAM(model, model.layer).generate(self.x, 0))
        self.assertTrue(model.training)
        self.assertFalse(model.layer.training)
        self.assert_clean(model)

    def test_keeps_existing_forward_hooks(self):
        calls = []
        handle = self.model[0].register_forward_hook(lambda *_: calls.append(True))
        try:
            result = GradCAM(self.model, self.model[0]).generate(self.x, 0)
            np.testing.assert_allclose(result, self.expected, atol=1e-7)
            self.assertEqual(calls, [True])
            self.assertEqual(list(self.model[0]._forward_hooks), [handle.id])
        finally:
            handle.remove()

    def test_nonfinite_gradients_never_produce_an_overlay(self):
        class InfiniteGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * float("inf")

        class InvalidBackward(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, x):
                return InfiniteGradient.apply(self.layer(x)).mean((2, 3))

        model = InvalidBackward(self.model[0])
        with self.assertLogs("src.grad_cam"):
            self.assertIsNone(GradCAM(model, model.layer).generate(self.x, 0))
        self.assert_clean(model)

    def test_target_layer_selection(self):
        model = nn.Module()
        model.features = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU())
        self.assertIs(get_target_layer(model), model.features[-1])
        model = nn.Module()
        model.layer4 = nn.Sequential(nn.Conv2d(1, 1, 1))
        self.assertIs(get_target_layer(model), model.layer4)
        with self.assertRaises(ValueError):
            get_target_layer(nn.Linear(1, 1))

    def test_overlay_dimensions_alpha_and_negative_strides(self):
        source = Image.new("RGB", (27, 15), color=(70, 80, 90))
        cam = self.expected[:, ::-1]
        overlay = overlay_cam_on_image(source, cam)
        self.assertEqual(overlay.size, source.size)
        self.assertEqual(overlay.mode, "RGB")
        np.testing.assert_array_equal(overlay_cam_on_image(source, cam, alpha=0), source)
        for invalid_cam in [np.zeros((0, 2)), np.zeros(3), np.full((2, 2), np.nan)]:
            with self.assertRaises(ValueError):
                overlay_cam_on_image(source, invalid_cam)
        for alpha in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                overlay_cam_on_image(source, cam, alpha=alpha)


if __name__ == "__main__":
    unittest.main()
