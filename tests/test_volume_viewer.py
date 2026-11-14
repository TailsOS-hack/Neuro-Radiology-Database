"""Interactive viewer contracts: what the page embeds, where it puts it, and what it must not contain.

The payload tests need only NumPy and Pillow. The browser test runs when Node.js
with Playwright is available and is skipped otherwise.
"""

import base64
import contextlib
import io
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src import volume_viewer as vv

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

NOTICE = "Research output only, not a diagnosis. A synthetic ensemble for tests."
SHAPE = (40, 48, 36)
EXPLAINED = list(range(14, 26))
PROTOCOL = [14, 17, 20, 23, 25]
PEAK = (13, 30, 20)
MARKER = (9, 37, 27)  # a bright voxel toward the patient's left, front and top: fixes the orientation of every view
BROWSER_CHECK = Path(__file__).with_name("viewer_browser_check.js")


def synthetic_inputs(with_map=True, band=False):
    """A small head-like volume on a 1.5 mm RAS grid, an attention volume and a matching summary."""

    rng = np.random.default_rng(7)
    x, y, z = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
    inside = ((x - 20) / 12) ** 2 + ((y - 25) / 15) ** 2 + ((z - 18) / 11) ** 2 < 1
    grid = np.zeros(SHAPE, dtype=np.float32)
    grid[inside] = 300 + 300 * rng.random(int(inside.sum()))
    grid[MARKER] = 800.0
    cam = np.zeros(SHAPE, dtype=np.float32)
    cam[10:30, 14:36, EXPLAINED[0] : EXPLAINED[-1] + 1] = rng.random((20, 22, len(EXPLAINED))) * 0.6
    cam[PEAK] = 1.0
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    affine[:3, 3] = [-30.0, -36.0, -27.0]
    energy = cam.sum(axis=(0, 1))[EXPLAINED]
    summary = {
        "class_names": ["CN", "AD"],
        "probabilities": [0.6, 0.4],
        "predicted_label": "CN",
        "threshold_on_positive_class_probability": 0.5,
        "member_probabilities": [[0.7, 0.3], [0.5, 0.5]],
        "aggregation": f"softmax(mean of the {'trained-band' if band else 'nine protocol'} slice logits) per member; "
        "equal-weight mean of member probabilities",
        "preparation": {
            "recipe": "exploratory_t1_axial_v1",
            "grid_shape": list(SHAPE),
            "grid_spacing_mm": [1.5, 1.5, 1.5],
            "grid_affine": affine.tolist(),
            "crop_x0_x1_y0_y1": [4, 36, 6, 44],
            "cranial_extent_z": [7, 29],
            "intensity_window": [0.0, 800.0],
            "protocol_slice_indices": PROTOCOL,
        },
        "grad_cam": {
            "target_class": "CN",
            "score": "logit",
            "slice_indices": EXPLAINED,
            "slice_share_of_attention": [float(value) for value in energy / energy.sum()],
            "slice_probabilities": [[1 - p, p] for p in np.linspace(0.2, 0.8, len(EXPLAINED)).tolist()],
        },
        "source": {"format": "dicom_series", "note": "SOURCE-MARKER"},
        "acquisition": {"note": "ACQUISITION-MARKER"},
        "warnings": ["The header does not identify this series as T1-weighted.", NOTICE],
        "notice": NOTICE,
    }
    if not with_map:
        summary["grad_cam"] = None
    return grid, (cam if with_map else None), summary, affine


def decode_sprite(uri, shape, columns):
    """Inverse of the sprite layout, done independently of the module: tile z at (z % columns, z // columns)."""

    sheet = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
    nx, ny, nz = shape
    volume = np.zeros(shape, dtype=np.uint8)
    for z in range(nz):
        top, left = (z // columns) * ny, (z % columns) * nx
        volume[:, :, z] = sheet[top : top + ny, left : left + nx].T
    return volume


def embedded_payload(page):
    return json.loads(page.split('<script type="application/json" id="vv-data">')[1].split("</script>")[0])


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.grid, self.cam, self.summary, self.affine = synthetic_inputs()
        self.payload = vv.viewer_payload(self.grid, self.cam, self.summary)

    def test_embedded_volumes_round_trip_exactly(self):
        payload = self.payload
        window = tuple(slice(start, start + size) for start, size in zip(payload["offset"], payload["shape"]))
        expected_anatomy = np.rint(np.clip(self.grid / 800.0, 0, 1) * 255).astype(np.uint8)
        expected_map = np.rint(np.clip(self.cam, 0, 1) * 255).astype(np.uint8)
        anatomy = decode_sprite(payload["anatomy_png"], payload["shape"], payload["tiles"][0])
        attention = decode_sprite(payload["attention_png"], payload["shape"], payload["tiles"][0])
        np.testing.assert_array_equal(anatomy, expected_anatomy[window])
        np.testing.assert_array_equal(attention, expected_map[window])
        self.assertEqual(payload["checksums"], [int(anatomy.sum(dtype=np.int64)), int(attention.sum(dtype=np.int64))])
        # Cropping to the head loses neither tissue nor attention.
        outside = np.ones(SHAPE, dtype=bool)
        outside[window] = False
        self.assertFalse((expected_anatomy[outside] > 12).any())
        self.assertFalse(expected_map[outside].any())
        self.assertLess(int(np.prod(payload["shape"])), int(np.prod(SHAPE)))

    def test_positions_refer_to_the_model_grid(self):
        payload = self.payload
        offset = payload["offset"]
        self.assertEqual(tuple(value + start for value, start in zip(payload["peak"], offset)), PEAK)
        self.assertEqual(payload["slab_grid"], [EXPLAINED[0], EXPLAINED[-1]])
        self.assertEqual(payload["slab"], [EXPLAINED[0] - offset[2], EXPLAINED[-1] - offset[2]])
        self.assertEqual(payload["explained"], EXPLAINED)
        self.assertEqual(payload["scored"], PROTOCOL)
        np.testing.assert_allclose(payload["affine"], self.affine)
        self.assertEqual(payload["spacing"], [1.5, 1.5, 1.5])
        # The page turns (cropped index + offset) into millimetres with this affine.
        world = self.affine @ [*PEAK, 1.0]
        np.testing.assert_allclose(world[:3], [-30 + 1.5 * PEAK[0], -36 + 1.5 * PEAK[1], -27 + 1.5 * PEAK[2]])
        self.assertEqual(len(payload["slice_share"]), len(EXPLAINED))
        self.assertAlmostEqual(sum(payload["slice_share"]), 1.0, places=6)
        self.assertEqual(payload["slice_positive_probability"][0], 0.2)
        self.assertIn("5 axial slices", payload["scored_text"])

    def test_band_policy_scores_every_explained_slice(self):
        grid, cam, summary, _ = synthetic_inputs(band=True)
        payload = vv.viewer_payload(grid, cam, summary)
        self.assertEqual(payload["scored"], EXPLAINED)
        self.assertIn("every one of the 12", payload["scored_text"])
        self.assertIn("Every slice", payload["scored_chart_text"])

    def test_page_is_self_contained_and_names_nobody(self):
        page = vv._render(self.payload, title="3D attention map", heading="h1", fragment=False)
        self.assertTrue(page.lstrip().lower().startswith("<!doctype html>"))
        # No request leaves the page: the only URL in it is the SVG namespace string.
        self.assertEqual(set(re.findall(r"https?://[^\s\"'<>)]+", page)), {"http://www.w3.org/2000/svg"})
        for forbidden in ("SOURCE-MARKER", "ACQUISITION-MARKER", "fetch(", "XMLHttpRequest", "<link", "@import", "src=\"http"):
            self.assertNotIn(forbidden, page)
        payload = embedded_payload(page)
        self.assertEqual(payload["notice"], NOTICE)
        self.assertEqual(payload["warnings"], ["The header does not identify this series as T1-weighted."])  # the notice is not repeated
        self.assertNotIn("__VV_", page)
        self.assertEqual(sorted(key for key in payload if key not in self.payload), [])

    def test_text_from_the_analysis_cannot_break_out_of_the_page(self):
        hostile = 'Notice </script><script>window.pwned = 1</script> <img src=x onerror="window.pwned = 2"> __VV_TITLE__'
        self.summary["notice"] = hostile
        self.summary["warnings"] = [hostile + " again"]
        payload = vv.viewer_payload(self.grid, self.cam, self.summary)
        template = vv.TEMPLATE_PATH.read_text(encoding="utf-8")
        page = vv._render(payload, title='Map <b>"x"</b>', heading="h1", fragment=False)
        self.assertEqual(page.count("</script"), template.count("</script"))  # no extra script element was opened or closed
        self.assertEqual(page.count("<script"), template.count("<script"))
        self.assertEqual(embedded_payload(page)["notice"], hostile)  # the text itself survives intact, as data
        block = page.split('<script type="application/json" id="vv-data">')[1].split("</script>")[0]
        self.assertFalse(set("<>&") & set(block))  # nothing in the data can open or close markup
        self.assertIn("Map &lt;b&gt;&quot;x&quot;&lt;/b&gt;", page)
        self.assertNotIn('Map <b>"x"</b>', page)

    def test_page_does_not_depend_on_how_its_encoding_is_declared(self):
        # Pure ASCII: a host that mislabels the charset cannot garble a degree sign or a minus.
        page = vv._render(self.payload, title="3D attention map", heading="h1", fragment=False)
        self.assertTrue(page.isascii())
        self.assertTrue(vv.TEMPLATE_PATH.read_text(encoding="utf-8").isascii())

    def test_fragment_embeds_in_another_page(self):
        fragment = vv.viewer_fragment(payload=self.payload, title="Explore the result")
        self.assertIsNone(re.search(r"<(?:!doctype|html|head|body|main)[\s>]", fragment, flags=re.IGNORECASE))
        self.assertTrue(fragment.lstrip().startswith("<style>"))
        self.assertIn('<h2 class="vv-title">Explore the result</h2>', fragment)
        self.assertIn('id="vv-data"', fragment)
        with self.assertRaises(vv.ViewerError):
            vv.viewer_fragment(payload=self.payload, heading="h4")

    def test_scan_without_a_map_still_gets_a_viewer(self):
        grid, _, summary, _ = synthetic_inputs(with_map=False)
        payload = vv.viewer_payload(grid, None, summary)
        self.assertIsNone(payload["attention_png"])
        self.assertNotIn("slab", payload)
        self.assertIn("anatomy_png", payload)
        self.assertIn("vv-data", vv._render(payload, title="t", heading="h1", fragment=False))
        # A map that is zero everywhere has nothing to show either.
        grid, cam, summary, _ = synthetic_inputs()
        self.assertIsNone(vv.viewer_payload(grid, np.zeros_like(cam), summary)["attention_png"])

    def test_inconsistent_inputs_are_refused(self):
        with self.assertRaisesRegex(vv.ViewerError, "grid shape"):
            vv.viewer_payload(self.grid[:-1], self.cam[:-1], self.summary)
        with self.assertRaisesRegex(vv.ViewerError, "same grid"):
            vv.viewer_payload(self.grid, self.cam[:, :, :-1], self.summary)
        broken = self.cam.copy()
        broken[0, 0, 0] = np.nan
        with self.assertRaisesRegex(vv.ViewerError, "nonfinite"):
            vv.viewer_payload(self.grid, broken, self.summary)
        self.summary["preparation"]["intensity_window"] = [5.0, 5.0]
        with self.assertRaisesRegex(vv.ViewerError, "window"):
            vv.viewer_payload(self.grid, self.cam, self.summary)
        with self.assertRaisesRegex(vv.ViewerError, "3D volume"):
            vv.viewer_payload(self.grid[0], None, self.summary)


@unittest.skipIf(nib is None, "nibabel is not installed")
class SavedFolderTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="volume_viewer_test_"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.grid, self.cam, self.summary, self.affine = synthetic_inputs()
        nib.save(nib.Nifti1Image(self.grid, self.affine), str(self.directory / "anatomy.nii.gz"))
        nib.save(nib.Nifti1Image(self.cam, self.affine), str(self.directory / "gradcam.nii.gz"))
        (self.directory / "result.json").write_text(json.dumps(self.summary))

    def test_viewer_is_rebuilt_from_a_saved_folder(self):
        self.assertEqual(vv.payload_from_saved(self.directory), vv.viewer_payload(self.grid, self.cam, self.summary))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(vv.main([str(self.directory)]), 0)
        written = self.directory / vv.VIEWER_FILENAME
        self.assertEqual(buffer.getvalue().strip(), str(written))
        self.assertEqual(embedded_payload(written.read_text(encoding="utf-8"))["peak"], vv.viewer_payload(self.grid, self.cam, self.summary)["peak"])
        other = self.directory / "elsewhere" / "map.html"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vv.main([str(self.directory), "--output", str(other)]), 0)
        self.assertTrue(other.is_file())

    def test_folder_without_a_map_or_without_results(self):
        (self.directory / "gradcam.nii.gz").unlink()
        self.assertIsNone(vv.payload_from_saved(self.directory)["attention_png"])
        with self.assertLogs("src.volume_viewer", level="ERROR"):
            self.assertEqual(vv.main([str(self.directory / "missing")]), 2)
        (self.directory / "result.json").write_text("{}")
        with self.assertLogs("src.volume_viewer", level="ERROR"):
            self.assertEqual(vv.main([str(self.directory)]), 2)


def _browser_available():
    node = shutil.which("node")
    if node is None or not BROWSER_CHECK.is_file():
        return False
    try:
        probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


@unittest.skipUnless(_browser_available(), "Node.js with Playwright is not available")
class BrowserTests(unittest.TestCase):
    """The page in a real browser: exact decoding, the orientation of every view, and no errors."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="volume_viewer_browser_"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)
        grid, cam, summary, _ = synthetic_inputs()
        cls.payload = vv.viewer_payload(grid, cam, summary)
        page = cls.directory / "viewer.html"
        page.write_text(vv._render(cls.payload, title="3D attention map", heading="h1", fragment=False), encoding="utf-8")
        marker = [value - start for value, start in zip(MARKER, cls.payload["offset"])]
        run = subprocess.run(["node", str(BROWSER_CHECK), str(page), json.dumps(marker)], capture_output=True, text=True, timeout=180)
        if run.returncode != 0:
            raise AssertionError(f"browser check failed to run:\n{run.stdout}\n{run.stderr}")
        cls.report = json.loads(run.stdout)

    def test_loads_without_errors_or_requests(self):
        self.assertEqual(self.report["problems"], [])
        self.assertTrue(self.report["status_hidden"])
        self.assertEqual(self.report["overflow"], 0)

    def test_browser_decodes_the_volumes_exactly(self):
        self.assertEqual(self.report["checksums"], self.payload["checksums"])

    def test_every_view_puts_the_marker_where_its_edge_letters_say(self):
        # The marker voxel is the brightest in the scan. With the crosshair on it, each slice view must draw it at
        # the pixel that its edge letters imply: left on the left, anterior and superior up, face to the right.
        nx, ny, nz = self.payload["shape"]
        mx, my, mz = (value - start for value, start in zip(MARKER, self.payload["offset"]))
        expected = {"vv-axial": [mx, ny - 1 - my], "vv-coronal": [mx, nz - 1 - mz], "vv-sagittal": [my, nz - 1 - mz]}
        self.assertEqual(self.report["brightest"], expected)
        self.assertEqual(self.report["edges"], {"vv-axial": ["L", "R", "A", "P"], "vv-coronal": ["L", "R", "S", "I"],
                                                "vv-sagittal": ["P", "A", "S", "I"]})
        # x below the grid centre is the patient's left (RAS+), y above it anterior, z above it superior.
        self.assertIn("left", self.report["readout"])
        self.assertIn("anterior", self.report["readout"])
        self.assertIn("superior", self.report["readout"])

    def test_controls_drive_the_views(self):
        self.assertEqual(self.report["peak_idx"], self.payload["peak"])
        self.assertIn("attention 1.00", self.report["peak_readout"])
        self.assertIn("outside the analysed slab", self.report["outside_readout"])
        self.assertTrue(self.report["map_changes_pixels"])
        self.assertTrue(self.report["rotation_changes_3d"])
        self.assertEqual(self.report["chart_marks"], [len(EXPLAINED), len(EXPLAINED)])
        self.assertEqual(self.report["table_rows"], len(EXPLAINED))


if __name__ == "__main__":
    unittest.main()
