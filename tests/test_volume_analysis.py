"""Volume analysis contracts: slice preparation, stacked Grad-CAM and 3D geometry.

Everything here runs on synthetic volumes and randomly initialised networks:
no patient data, trained checkpoints or network. The tests pin *mechanics*
(geometry, aggregation, scaling, provenance); they say nothing about how well
any trained model performs.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

try:
    import nibabel as nib
    import scipy  # noqa: F401
except ModuleNotFoundError:
    nib = None

try:
    import pydicom
except ModuleNotFoundError:
    pydicom = None

from src.mri_volume import MRIVolume

if nib is not None:
    from src import volume_analysis as va


def head_phantom(shape=(88, 104, 92), spacing=2.0, seed=0):
    """Integer 'head': bright scalp rim, dark skull, textured brain, darker ventricles."""

    rng = np.random.default_rng(seed)
    axes = [(np.arange(n) - (n - 1) / 2) * spacing for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt((x / 72.0) ** 2 + (y / 90.0) ** 2 + (z / 80.0) ** 2)
    volume = np.zeros(shape, dtype=np.float64)
    volume[radius < 1.0] = 900  # scalp
    volume[radius < 0.93] = 60  # skull
    volume[radius < 0.86] = 520 + 6 * (x[radius < 0.86] / spacing) % 37  # brain with left-right texture
    ventricles = (np.abs(x) < 14) & (np.abs(y) < 26) & (np.abs(z - 8) < 12) & (np.abs(x) > 3)
    volume[ventricles] = 150
    volume[(radius < 0.86)] += rng.integers(0, 25, size=int((radius < 0.86).sum()))
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = [-(n - 1) / 2 * spacing for n in shape]
    return np.rint(volume).astype(np.int16), affine


@unittest.skipIf(nib is None, "nibabel/scipy are not installed")
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.affine = head_phantom()
        cls.volume = MRIVolume.from_array(cls.data, cls.affine)
        cls.prepared = va.prepare_clinical_volume(cls.volume)

    def test_grid_and_protocol_slices_follow_the_frozen_recipe(self):
        prepared = self.prepared
        self.assertEqual(prepared.recipe, "exploratory_t1_axial_v1")
        np.testing.assert_allclose(np.sqrt((prepared.grid_affine[:3, :3] ** 2).sum(0)), [1.5, 1.5, 1.5])
        self.assertEqual(nib.aff2axcodes(prepared.grid_affine), ("R", "A", "S"))
        zlo, zhi = prepared.cranial_extent
        expected = [int(round(zlo + fraction * (zhi - zlo))) for fraction in np.linspace(0.45, 0.85, 9)]
        self.assertEqual(prepared.protocol_indices, expected)
        self.assertEqual(len(set(expected)), 9)
        self.assertEqual(prepared.band_indices, list(range(expected[0], expected[-1] + 1)))
        x0, x1, y0, y1 = prepared.crop
        self.assertTrue(0 <= x0 < x1 <= prepared.grid.shape[0] and 0 <= y0 < y1 <= prepared.grid.shape[1])
        self.assertFalse(prepared.notes)

    def test_rendered_slices_are_model_inputs(self):
        images = self.prepared.render_slices(self.prepared.protocol_indices)
        self.assertEqual(images.shape, (9, 224, 224))
        self.assertEqual(images.dtype, np.uint8)
        self.assertGreater(int(images.max()), 200)
        tensor = va.images_to_tensor(images) if torch is not None else None
        if tensor is not None:
            self.assertEqual(tuple(tensor.shape), (9, 3, 224, 224))
            torch.testing.assert_close(tensor[:, 0] * 0.229 + 0.485, tensor[:, 2] * 0.225 + 0.406, atol=1e-6, rtol=0)
            torch.testing.assert_close(tensor[:, 0] * 0.229 + 0.485, torch.from_numpy(images).float() / 255, atol=1e-6, rtol=0)

    def test_display_convention_anterior_up_patient_left_on_image_left(self):
        # Mark one voxel far anterior-left of centre on the grid and find it in the image.
        prepared = self.prepared
        x0, x1, y0, y1 = prepared.crop
        z = prepared.protocol_indices[4]
        marked = va.PreparedVolume(**{**prepared.__dict__, "grid": prepared.grid.copy()})
        marked.grid[:, :, z] = prepared.window[0]
        left_x, anterior_y = x0 + 6, y1 - 7  # low x = patient left (RAS), high y = anterior
        marked.grid[left_x - 1 : left_x + 2, anterior_y - 1 : anterior_y + 2, z] = prepared.window[1]
        image = marked.render_slice(z)
        row, column = np.unravel_index(int(np.argmax(image)), image.shape)
        self.assertLess(row, 112)  # anterior at the top
        self.assertLess(column, 112)  # patient left on image left (neurological)

    def test_same_anatomy_in_any_voxel_order_or_dtype_gives_identical_slices(self):
        reference = self.prepared.render_slices(self.prepared.protocol_indices)
        permuted = np.flip(np.transpose(self.data, (2, 0, 1)), axis=1)
        affine = self.affine[:, [2, 0, 1, 3]].copy()
        affine[:3, 3] += affine[:3, 1] * (permuted.shape[1] - 1)
        affine[:3, 1] *= -1
        for candidate in (
            MRIVolume.from_array(permuted, affine),  # sagittal-style storage order
            MRIVolume.from_array(self.data.astype(np.float32), self.affine),
            MRIVolume.from_array(self.data.astype(np.float64), self.affine),
        ):
            prepared = va.prepare_clinical_volume(candidate)
            self.assertEqual(prepared.protocol_indices, self.prepared.protocol_indices)
            np.testing.assert_array_equal(prepared.render_slices(prepared.protocol_indices), reference)

    def test_integer_scans_are_interpolated_like_the_frozen_nifti_pipeline(self):
        from nibabel.processing import resample_to_output

        # The frozen script resamples an int16 NIfTI in int16; fractional values use float64.
        aligned = resample_to_output(nib.Nifti1Image(self.data, self.affine), voxel_sizes=(1.5, 1.5, 1.5), order=1, mode="constant", cval=0)
        np.testing.assert_array_equal(self.prepared.grid, np.asarray(aligned.dataobj, dtype=np.float32))
        fractional = self.data.astype(np.float64) * 0.75
        aligned = resample_to_output(nib.Nifti1Image(fractional, self.affine), voxel_sizes=(1.5, 1.5, 1.5), order=1, mode="constant", cval=0)
        prepared = va.prepare_clinical_volume(MRIVolume.from_array(fractional, self.affine))
        np.testing.assert_array_equal(prepared.grid, np.asarray(aligned.dataobj, dtype=np.float32))

    def test_unusable_volumes_are_refused(self):
        with self.assertRaisesRegex(va.PreparationError, "not a 3D head volume"):
            va.prepare_clinical_volume(MRIVolume(np.ones((64, 64, 1), dtype=np.float32), np.eye(4)))
        with self.assertRaisesRegex(va.PreparationError, "Too little positive signal"):
            va.prepare_clinical_volume(MRIVolume(np.zeros((40, 40, 40), dtype=np.float32), np.eye(4)))
        thin = np.zeros((60, 60, 60), dtype=np.float32)
        thin[10:50, 10:50, 28:34] = 500  # a slab, not a head
        with self.assertRaises(va.PreparationError):
            va.prepare_clinical_volume(MRIVolume(thin, np.diag([2.0, 2.0, 2.0, 1.0])))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_maps_return_to_the_voxels_they_were_computed_from(self):
        prepared = self.prepared
        x0, x1, y0, y1 = prepared.crop
        indices = prepared.band_indices[:3]
        for target in [(x0 + 5, y0 + 9), (x1 - 8, y1 - 6), ((x0 + x1) // 2, (y0 + y1) // 2), (x0 + 20, y1 - 30)]:
            # Forward: light one voxel, render, locate it in model-input pixels.
            marked = va.PreparedVolume(**{**prepared.__dict__, "grid": np.full_like(prepared.grid, prepared.window[0])})
            marked.grid[target[0], target[1], indices[1]] = prepared.window[1]
            image = marked.render_slice(indices[1]).astype(np.float32)
            # Inverse: a map that is the rendered image itself must peak at that voxel.
            maps = np.zeros((3, 224, 224), dtype=np.float32)
            maps[1] = image / image.max()
            volume = prepared.maps_to_grid(maps, indices)
            peak = np.unravel_index(int(np.argmax(volume)), volume.shape)
            self.assertEqual(peak[2], indices[1])
            self.assertLessEqual(abs(peak[0] - target[0]), 1)
            self.assertLessEqual(abs(peak[1] - target[1]), 1)
        self.assertEqual(volume.shape, prepared.grid.shape)
        self.assertEqual(float(volume[:, :, indices[0]].max()), 0.0)  # an all-zero slice map stays zero
        outside = np.ones(prepared.grid.shape, dtype=bool)
        outside[x0:x1, y0:y1, :] = False
        self.assertEqual(float(prepared.maps_to_grid(np.ones((3, 7, 7), np.float32), indices)[outside].max()), 0.0)
        with self.assertRaises(ValueError):
            prepared.maps_to_grid(np.ones((2, 7, 7), np.float32), indices)

    def test_resample_array_round_trip_and_identity(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(12, 14, 10)).astype(np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        np.testing.assert_allclose(va.resample_array(data, affine, affine, data.shape), data, atol=1e-6)
        shifted = affine.copy()
        shifted[:3, 3] = [2.0, 0.0, 0.0]  # target grid starts one voxel further right
        moved = va.resample_array(data, affine, shifted, data.shape)
        np.testing.assert_allclose(moved[:-1], data[1:], atol=1e-6)
        np.testing.assert_array_equal(moved[-1], 0)


@unittest.skipIf(torch is None or nib is None, "PyTorch/nibabel are not installed")
class SliceStackGradCamTests(unittest.TestCase):
    def analytical_model(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        with torch.no_grad():
            model[0].weight.fill_(1)
        return model

    def test_maps_are_raw_so_slices_stay_comparable(self):
        model = self.analytical_model()
        base = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        stack = torch.cat([base, 3 * base, -base - 1])  # weak, strong and negative-evidence slices
        maps, logits = va.slice_stack_grad_cam(model, model[0], stack, 0)
        expected = base[0, 0].numpy() / 16  # weight = mean gradient = 1 / (h * w)
        np.testing.assert_allclose(maps[0], expected, atol=1e-7)
        np.testing.assert_allclose(maps[1], 3 * expected, atol=1e-6)  # not rescaled to its own maximum
        np.testing.assert_array_equal(maps[2], 0)  # nothing to show stays dark
        np.testing.assert_allclose(logits[:, 0], stack.mean(dim=(1, 2, 3)).numpy(), atol=1e-6)

    def small_network(self):
        torch.manual_seed(0)
        network = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.BatchNorm2d(12), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(12, 3),
        )
        network.eval()
        with torch.no_grad():  # non-trivial running statistics
            network[1].running_mean.normal_()
            network[4].running_var.uniform_(0.5, 2.0)
        return network

    def test_matches_the_single_image_grad_cam_slice_by_slice(self):
        from src.grad_cam import GradCAM

        network = self.small_network()
        stack = torch.randn(5, 3, 16, 16)
        for class_index in range(3):
            maps, _ = va.slice_stack_grad_cam(network, network[5], stack, class_index, batch_size=2)
            for position in range(len(stack)):
                single = GradCAM(network, network[5]).generate(stack[position], class_index)
                if single is None:  # flat map: the stack's raw version must be flat too
                    self.assertLess(float(np.ptp(maps[position])), 1e-6)
                    continue
                normalised = (maps[position] - maps[position].min()) / (maps[position].max() - maps[position].min())
                np.testing.assert_allclose(normalised, single, atol=1e-5)

    def test_batch_size_and_no_grad_contexts_do_not_change_the_result(self):
        network = self.small_network()
        stack = torch.randn(7, 3, 16, 16)
        reference, reference_logits = va.slice_stack_grad_cam(network, network[5], stack, 1, batch_size=7)
        for batch_size in (1, 3):
            maps, logits = va.slice_stack_grad_cam(network, network[5], stack, 1, batch_size=batch_size)
            np.testing.assert_allclose(maps, reference, atol=1e-6)
            np.testing.assert_allclose(logits, reference_logits, atol=1e-6)
        with torch.no_grad():
            np.testing.assert_allclose(va.slice_stack_grad_cam(network, network[5], stack, 1)[0], reference, atol=1e-6)
        with torch.inference_mode():
            np.testing.assert_allclose(va.slice_stack_grad_cam(network, network[5], stack.clone(), 1)[0], reference, atol=1e-6)

    def test_model_state_is_left_untouched(self):
        network = self.small_network()
        network.train()
        network[1].eval()  # deliberately mixed modes, e.g. frozen BatchNorm
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = {name: value.clone() for name, value in network.state_dict().items()}
        modes = [module.training for module in network.modules()]
        va.slice_stack_grad_cam(network, network[5], torch.randn(3, 3, 16, 16), 0)
        self.assertEqual([module.training for module in network.modules()], modes)
        for name, value in network.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for parameter in network.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)
        for module in network.modules():
            self.assertFalse(module._forward_hooks)

    def test_margin_score_explains_the_logit_difference(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([1.0, 2.0]).view(2, 1, 1, 1))
            model[3].weight.copy_(torch.tensor([[1.0, 0.5], [0.25, 2.0]]))
        x = torch.rand(2, 1, 4, 4) + 0.1
        logit_maps, _ = va.slice_stack_grad_cam(model, model[0], x, 1, score="logit")
        margin_maps, _ = va.slice_stack_grad_cam(model, model[0], x, 1, score="margin")
        activation = torch.cat([x, 2 * x], dim=1).numpy()  # the two feature channels
        np.testing.assert_allclose(logit_maps, np.maximum((0.25 * activation[:, 0] + 2.0 * activation[:, 1]) / 16, 0), atol=1e-6)
        np.testing.assert_allclose(
            margin_maps, np.maximum(((0.25 - 1.0) * activation[:, 0] + (2.0 - 0.5) * activation[:, 1]) / 16, 0), atol=1e-6
        )

    def test_invalid_requests_are_rejected_and_hooks_removed(self):
        model = self.analytical_model()
        stack = torch.zeros(2, 1, 4, 4)
        for arguments in [dict(class_index=5), dict(class_index=0, score="probability")]:
            with self.assertRaises(ValueError):
                va.slice_stack_grad_cam(model, model[0], stack, **arguments)
        with self.assertRaises(ValueError):
            va.slice_stack_grad_cam(model, model[0], stack[0], 0)
        with self.assertRaises(ValueError):
            va.slice_stack_grad_cam(model, model[2], stack, 0)  # not a spatial feature map
        self.assertFalse(model[0]._forward_hooks or model[2]._forward_hooks)

    def test_refuses_to_explain_a_function_that_differs_from_the_forward_pass(self):
        class Unstable(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.calls = 0

            def forward(self, x):
                self.calls += 1  # behaves differently on every call
                return self.conv(x).mean(dim=(2, 3)) + self.calls

        model = Unstable()
        with self.assertRaisesRegex(RuntimeError, "differ from the plain forward pass"):
            va.slice_stack_grad_cam(model, model.conv, torch.ones(2, 1, 4, 4), 0)


@unittest.skipIf(torch is None or nib is None, "PyTorch/nibabel are not installed")
class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="volume_analysis_test_"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)
        cls.checkpoints = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            model = va.build_clinical_slice_model()
            with torch.no_grad():  # make the two classes separable enough to be interesting
                model.head.weight.mul_(40)
            path = cls.directory / f"clinical_seed{seed}.pt"
            torch.save({"model": model.state_dict(), "architecture": va.CLINICAL_ARCHITECTURE, "seed": seed, "epoch": 3}, path)
            cls.checkpoints.append(path)
        cls.data, cls.affine = head_phantom()
        cls.volume = MRIVolume.from_array(cls.data, cls.affine, acquisition={"contrast_guess": {"label": "T1w"}})
        cls.analyzer = va.DementiaVolumeAnalyzer(cls.checkpoints, device=torch.device("cpu"))
        cls.result = cls.analyzer.analyze(cls.volume)

    def test_prediction_is_softmax_of_mean_protocol_logits_then_member_mean(self):
        result = self.result
        images = result.prepared.render_slices(result.prepared.protocol_indices)
        tensor = va.images_to_tensor(images)
        expected = []
        with torch.no_grad():
            for model in self.analyzer.models:
                expected.append(torch.softmax(model(tensor).mean(dim=0), dim=0).numpy())
        np.testing.assert_allclose(result.member_probabilities, np.stack(expected), atol=1e-6)
        np.testing.assert_allclose(result.probabilities, np.mean(expected, axis=0), atol=1e-6)
        self.assertAlmostEqual(float(result.probabilities.sum()), 1.0, places=6)
        self.assertEqual(result.class_names, ("CN", "AD"))
        self.assertEqual(result.predicted_label, "AD" if result.probabilities[1] >= 0.5 else "CN")
        self.assertEqual([member["seed"] for member in result.members], [1, 2])
        self.assertTrue(all(len(member["sha256"]) == 64 for member in result.members))

    def test_attention_volume_is_scaled_once_and_confined_to_the_analysed_slab(self):
        result = self.result
        cam = result.cam_grid
        self.assertEqual(cam.shape, result.prepared.grid.shape)
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)
        self.assertGreater(float(cam.max()), 0.5)
        self.assertEqual(result.cam_indices, result.prepared.band_indices)
        inside = np.zeros(cam.shape[2], dtype=bool)
        inside[result.cam_indices] = True
        self.assertEqual(float(cam[:, :, ~inside].max()), 0.0)
        x0, x1, y0, y1 = result.prepared.crop
        outside = np.ones(cam.shape, dtype=bool)
        outside[x0:x1, y0:y1] = False
        self.assertEqual(float(cam[outside].max()), 0.0)
        per_slice_peak = cam[:, :, result.cam_indices].max(axis=(0, 1))
        self.assertGreater(float(per_slice_peak.max() - per_slice_peak.min()), 0.0)  # slices are not individually maxed out
        self.assertEqual(result.slice_probabilities.shape, (len(result.cam_indices), 2))
        np.testing.assert_allclose(result.slice_probabilities.sum(axis=1), 1.0, atol=1e-6)

        native = result.cam_on_source_grid(self.volume)
        self.assertEqual(native.shape, self.volume.shape)
        # Centre of mass must be the same world position on both grids.
        def centre_of_mass(volume, affine):
            index = [(volume.sum(axis=tuple(a for a in range(3) if a != axis)) * np.arange(n)).sum() / volume.sum()
                     for axis, n in enumerate(volume.shape)]
            return (affine @ [*index, 1])[:3]

        np.testing.assert_allclose(
            centre_of_mass(native, self.volume.affine), centre_of_mass(cam, result.prepared.grid_affine), atol=1.5
        )

    def test_protocol_only_and_explicit_target(self):
        sparse = self.analyzer.analyze(self.volume, dense=False, cam_target="CN", cam_score="margin")
        self.assertEqual(sparse.cam_indices, sparse.prepared.protocol_indices)
        self.assertEqual(sparse.class_names[sparse.cam_target_index], "CN")
        np.testing.assert_allclose(sparse.probabilities, self.result.probabilities, atol=1e-6)
        plain = self.analyzer.analyze(self.volume, grad_cam=False)
        self.assertIsNone(plain.cam_grid)
        self.assertIsNone(plain.summary()["grad_cam"])
        with self.assertRaises(ValueError):
            plain.cam_on_source_grid(self.volume)

    def test_saved_outputs_overlay_the_scan_and_carry_the_research_notice(self):
        output = self.directory / "saved"
        written = self.result.save(output, self.volume)
        for name in ("result.json", "anatomy.nii.gz", "gradcam.nii.gz", "gradcam_native.nii.gz", "montage.png", "projections.png",
                     "viewer.html"):
            self.assertTrue((output / name).is_file(), name)
            self.assertIn(name, written)
        anatomy, cam, native = (nib.load(str(output / name)) for name in ("anatomy.nii.gz", "gradcam.nii.gz", "gradcam_native.nii.gz"))
        np.testing.assert_allclose(anatomy.affine, cam.affine)
        np.testing.assert_allclose(cam.affine, self.result.prepared.grid_affine, atol=1e-4)
        self.assertEqual(cam.shape, anatomy.shape)
        np.testing.assert_allclose(native.affine, self.volume.affine, atol=1e-4)
        self.assertEqual(native.shape, self.volume.shape)
        np.testing.assert_allclose(cam.get_fdata(dtype=np.float32), self.result.cam_grid, atol=1e-7)
        summary = json.loads((output / "result.json").read_text())
        self.assertEqual(summary["class_names"], ["CN", "AD"])
        self.assertIn("not a diagnosis", summary["notice"])
        self.assertIn("specificity 0/10", summary["notice"])
        self.assertAlmostEqual(sum(summary["grad_cam"]["slice_share_of_attention"]), 1.0, places=4)
        self.assertEqual(summary["preparation"]["protocol_slice_indices"], self.result.prepared.protocol_indices)

    def test_saved_viewer_shows_the_same_map_at_the_same_place(self):
        from src import volume_viewer

        output = self.directory / "saved_viewer"
        self.result.save(output, self.volume)
        page = (output / "viewer.html").read_text(encoding="utf-8")
        payload = json.loads(page.split('<script type="application/json" id="vv-data">')[1].split("</script>")[0])
        offset = payload["offset"]
        peak = tuple(int(value) + int(start) for value, start in zip(payload["peak"], offset))
        self.assertEqual(float(self.result.cam_grid[peak]), float(self.result.cam_grid.max()))  # same strongest voxel
        self.assertEqual(payload["slab_grid"], [self.result.cam_indices[0], self.result.cam_indices[-1]])
        self.assertEqual(payload["scored"], self.result.prepared.protocol_indices)  # Round 2: the nine protocol slices
        np.testing.assert_allclose(payload["affine"], self.result.prepared.grid_affine)
        self.assertEqual(payload["notice"], self.result.notice)
        self.assertEqual(payload["target_class"], self.result.class_names[self.result.cam_target_index])
        # The saved folder alone is enough to rebuild the same page.
        self.assertEqual(volume_viewer.payload_from_saved(output), payload)
        # A failure while writing the viewer must not cost the volumes.
        with patch.object(volume_viewer, "build_viewer", side_effect=RuntimeError("no viewer today")):
            with self.assertLogs("src.volume_analysis", level="ERROR"):
                written = self.result.save(self.directory / "saved_without_viewer", self.volume)
        self.assertNotIn("viewer.html", written)
        self.assertTrue((self.directory / "saved_without_viewer" / "gradcam.nii.gz").is_file())

    def test_overlays_have_the_model_input_geometry(self):
        z = self.result.cam_indices[len(self.result.cam_indices) // 2]
        overlay = self.result.slice_overlay(z)
        self.assertEqual(overlay.size, (224, 224))
        self.assertEqual(overlay.mode, "RGB")
        plain = self.result.slice_overlay(self.result.prepared.cranial_extent[0])  # outside the slab: anatomy only
        array = np.asarray(plain)
        np.testing.assert_array_equal(array[:, :, 0], array[:, :, 1])
        self.assertEqual(self.result.montage(columns=8, tile=32).size[0], 8 * 32)
        self.assertEqual(self.result.projections(scale=1).height, max(self.result.prepared.grid.shape[1:]))

    def test_warnings_for_inputs_outside_the_training_domain(self):
        flair = MRIVolume.from_array(self.data, self.affine, acquisition={"contrast_guess": {"label": "FLAIR"}, "contrast_agent": "Gd"})
        result = self.analyzer.analyze(flair, grad_cam=False)
        joined = " ".join(result.warnings)
        self.assertIn("looks like FLAIR", joined)
        self.assertIn("Contrast agent", joined)
        self.assertIn("Research output only", joined)
        unknown = MRIVolume.from_array(self.data, self.affine, acquisition={"contrast_guess": {"label": "unknown"}})
        self.assertIn("does not identify this series as T1-weighted", " ".join(self.analyzer.analyze(unknown, grad_cam=False).warnings))
        computed_tomography = MRIVolume.from_array(self.data, self.affine, acquisition={"modality": "CT"})
        with self.assertRaisesRegex(va.PreparationError, "CT series"):
            self.analyzer.analyze(computed_tomography)

    def test_explanation_failure_does_not_lose_the_prediction(self):
        with patch.object(va, "slice_stack_grad_cam", side_effect=RuntimeError("backward op unsupported")):
            with self.assertLogs("src.volume_analysis", level="ERROR"):
                result = self.analyzer.analyze(self.volume)
        self.assertIsNone(result.cam_grid)
        np.testing.assert_allclose(result.probabilities, self.result.probabilities, atol=1e-6)
        self.assertTrue(any("Grad-CAM could not be computed" in note for note in result.warnings))

    def test_checkpoint_metadata_sets_policy_classes_and_notice(self):
        def save(name, seed, **extra):
            torch.manual_seed(seed)
            path = self.directory / name
            torch.save({"model": va.build_clinical_slice_model().state_dict(), "architecture": va.CLINICAL_ARCHITECTURE,
                        "seed": seed, "epoch": 1, **extra}, path)
            return path

        notice = "Research output only. Cross-validated on synthetic people."
        extra = dict(slice_policy="band", class_names=["NonDemented", "Demented"], threshold=0.4, notice=notice)
        band = [save(f"band{seed}.pt", seed, **extra) for seed in (11, 12)]
        analyzer = va.DementiaVolumeAnalyzer(band, device=torch.device("cpu"))
        result = analyzer.analyze(self.volume)
        # Band policy: the score comes from every slice of the band, the same slices the map explains.
        tensor = va.images_to_tensor(result.prepared.render_slices(result.prepared.band_indices))
        with torch.no_grad():
            expected = np.stack([torch.softmax(model(tensor).mean(dim=0), dim=0).numpy() for model in analyzer.models])
        np.testing.assert_allclose(result.member_probabilities, expected, atol=1e-6)
        self.assertEqual(result.cam_indices, result.prepared.band_indices)
        self.assertEqual(result.class_names, ("NonDemented", "Demented"))
        self.assertEqual(result.predicted_label, "Demented" if result.probabilities[1] >= 0.4 else "NonDemented")
        self.assertEqual(result.notice, notice)
        self.assertEqual(result.warnings[-1], notice)
        summary = result.summary()
        self.assertEqual(summary["notice"], notice)
        self.assertNotIn("specificity 0/10", json.dumps(summary))  # not the Round 2 record
        self.assertEqual(summary["positive_class"], "Demented")
        self.assertEqual(summary["threshold_on_positive_class_probability"], 0.4)
        self.assertIn("trained-band", summary["aggregation"])
        self.assertTrue(all("notice" not in member and member["slice_policy"] == "band" for member in summary["members"]))
        self.assertEqual(analyzer.analyze(self.volume, grad_cam=False, cam_target=None).predicted_label, result.predicted_label)
        with self.assertRaisesRegex(ValueError, "must be one of .*NonDemented"):
            analyzer.analyze(self.volume, cam_target="AD")  # not one of this model's classes

        # Round 2 files carry none of these keys and keep that round's frozen protocol and record.
        self.assertEqual(self.analyzer.slice_policy, "protocol9")
        self.assertEqual(self.analyzer.class_names, ("CN", "AD"))
        self.assertEqual(self.analyzer.notice, va.RESEARCH_USE_NOTICE)
        self.assertIn("nine protocol", self.result.summary()["aggregation"])

        with self.assertRaisesRegex(ValueError, "Ensemble members disagree on"):
            va.DementiaVolumeAnalyzer([band[0], self.checkpoints[0]], device=torch.device("cpu"))
        only_policy_differs = save("policy.pt", 13, **{**extra, "slice_policy": "protocol9"})
        with self.assertRaisesRegex(ValueError, "disagree on slice_policy"):
            va.DementiaVolumeAnalyzer([band[0], only_policy_differs], device=torch.device("cpu"))
        for bad in (dict(slice_policy="everything"), dict(class_names=["a", "b", "c"]), dict(threshold=1.5)):
            with self.assertRaisesRegex(ValueError, "unsupported class names, slice policy or threshold"):
                va.load_clinical_checkpoint(save("bad.pt", 3, **bad))

    def test_placed_ensemble_takes_precedence_over_round2_defaults(self):
        placed = self.directory / "placed"
        with patch.object(va, "TRAINED_VOLUME_MODEL_DIR", placed):
            self.assertEqual(va.default_checkpoints(), list(va.DEFAULT_CLINICAL_CHECKPOINTS))  # nothing placed
            placed.mkdir()
            (placed / "notes.txt").write_text("not a checkpoint")
            self.assertEqual(va.default_checkpoints(), list(va.DEFAULT_CLINICAL_CHECKPOINTS))
            for number, source in enumerate(reversed(self.checkpoints)):
                shutil.copyfile(source, placed / f"fold{number}.pt")
            self.assertEqual([path.name for path in va.default_checkpoints()], ["fold0.pt", "fold1.pt"])
            analyzer = va.DementiaVolumeAnalyzer(device=torch.device("cpu"))
            self.assertEqual([member["file"] for member in analyzer.members], ["fold0.pt", "fold1.pt"])

    def test_checkpoint_loading_is_strict(self):
        pointer = self.directory / "pointer.pt"
        pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n")
        with self.assertRaisesRegex(FileNotFoundError, "git lfs pull"):
            va.load_clinical_checkpoint(pointer)
        wrong = self.directory / "wrong.pt"
        torch.save({"model": va.build_clinical_slice_model().state_dict(), "architecture": "something_else", "pad": "x" * 2048}, wrong)
        with self.assertRaisesRegex(ValueError, "unexpected architecture"):
            va.load_clinical_checkpoint(wrong)
        stage = self.directory / "stage.pt"
        torch.save({"model_state": {}, "arch": "mobilenet_v3_large", "pad": "x" * 2048}, stage)
        with self.assertRaisesRegex(ValueError, "not a clinical patient-bag checkpoint"):
            va.load_clinical_checkpoint(stage)
        # Opening a checkpoint must not be able to run code: only tensors and plain metadata are unpickled.
        whole = self.directory / "whole_model.pt"
        torch.save(torch.nn.Linear(64, 64), whole)  # a whole pickled module, the legacy scripts' format
        with self.assertRaisesRegex(ValueError, "not a loadable slice-bag checkpoint"):
            va.load_clinical_checkpoint(whole)
        marker = self.directory / "code_ran.txt"

        class RunsCodeWhenUnpickled:
            def __reduce__(self):
                return (open, (str(marker), "w"))

        hostile = self.directory / "hostile.pt"
        torch.save({"model": va.build_clinical_slice_model().state_dict(), "architecture": va.CLINICAL_ARCHITECTURE,
                    "notice": RunsCodeWhenUnpickled()}, hostile)
        with self.assertRaisesRegex(ValueError, "not a loadable slice-bag checkpoint"):
            va.load_clinical_checkpoint(hostile)
        self.assertFalse(marker.exists())

    @unittest.skipIf(pydicom is None, "pydicom is not installed")
    def test_dicom_and_nifti_inputs_give_the_same_answer_and_no_phi_leaves(self):
        try:
            import dicom_synth as synth
        except ModuleNotFoundError:
            from tests import dicom_synth as synth

        # Same scan stored two ways: NIfTI on disk, and a sagittal DICOM series.
        nifti = self.directory / "scan.nii.gz"
        nib.save(nib.Nifti1Image(self.data, self.affine), str(nifti))
        sagittal, sagittal_affine = synth.reoriented(self.data, self.affine, (1, 2, 0), (True, False, True))
        synth.write_series(self.directory / "dicom", sagittal, sagittal_affine, shuffle=5)

        from_nifti = va.load_any_volume(nifti)
        from_dicom = va.load_any_volume(self.directory / "dicom")
        np.testing.assert_array_equal(from_dicom.data, from_nifti.data)
        first = self.analyzer.analyze(from_nifti, dense=False)
        second = self.analyzer.analyze(from_dicom, dense=False)
        np.testing.assert_array_equal(second.probabilities, first.probabilities)
        np.testing.assert_array_equal(second.cam_grid, first.cam_grid)

        output = self.directory / "dicom_saved"
        second.save(output, from_dicom)
        text = (output / "result.json").read_text()
        viewer = (output / "viewer.html").read_text(encoding="utf-8")
        for needle in ("SYNTHETIC", "PHANTOM", "FAKE-ID", "19400229", "FAKE-ACC", "Nowhere", "FAKESCANNER", "t1_mprage_sag_p2_iso"):
            self.assertNotIn(needle, text)
            self.assertNotIn(needle, viewer)
        self.assertEqual(json.loads(text)["source"]["format"], "dicom_series")
        self.assertEqual(json.loads(text)["acquisition"]["contrast_guess"]["label"], "T1w")

        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = va.main([str(self.directory / "dicom"), "--output", str(self.directory / "cli"), "--protocol-slices-only",
                            "--checkpoints", *map(str, self.checkpoints)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue())["predicted_label"], second.predicted_label)
        self.assertTrue((self.directory / "cli" / "gradcam_native.nii.gz").is_file())
        with self.assertLogs("src.volume_analysis", level="ERROR"):
            self.assertEqual(va.main([str(self.directory / "missing"), "--output", str(self.directory / "cli2")]), 2)


if __name__ == "__main__":
    unittest.main()
