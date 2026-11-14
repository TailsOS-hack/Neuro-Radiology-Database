"""Interactive viewer contracts: what the page embeds, where it puts it, what it says and what it must not contain.

The payload tests need only NumPy. The saved-folder tests need nibabel. The
browser tests run when Node.js with Playwright and its Chromium are available
and are skipped otherwise.
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
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
MARKER = (9, 37, 27)  # a bright voxel in air toward the patient's left, front and top: fixes the orientation of every view
STRAY = (4, 6, 18)  # attention outside the head's bounding box: cropping to the head must not lose it
BROWSER_CHECK = Path(__file__).with_name("viewer_browser_check.js")
#: Every key a page may embed. Anything else would be data nobody decided to publish.
PAYLOAD_KEYS = {
    "shape", "offset", "spacing", "affine", "encoding", "anatomy_z", "attention_z", "checksums", "class_names",
    "probabilities", "member_probabilities", "predicted_label", "threshold", "notice", "warnings", "target_class",
    "scored", "scored_text", "no_map_text",
}
MAP_KEYS = {
    "explained", "slab_grid", "slab", "peak", "slice_share", "slice_positive_probability", "map_short", "map_text",
    "scored_chart_text", "slab_text",
}


def synthetic_inputs(with_map=True, policy="protocol9", explained=None, record="scored"):
    """A small head-like volume on a 1.5 mm RAS grid, an attention volume and a matching summary.

    ``policy`` is how the score is computed (the protocol slices, or every slice
    of the band), ``explained`` the slices that have a map (default: the whole
    band) and ``record`` how the summary states the policy: "scored" as the
    analysis writes it now, "members" and "text" as older result.json files did.
    """

    explained = list(EXPLAINED if explained is None else explained)
    rng = np.random.default_rng(7)
    x, y, z = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
    inside = ((x - 20) / 12) ** 2 + ((y - 25) / 15) ** 2 + ((z - 18) / 11) ** 2 < 1
    grid = np.zeros(SHAPE, dtype=np.float32)
    grid[inside] = 300 + 300 * rng.random(int(inside.sum()))
    grid[MARKER] = 800.0
    cam = np.zeros(SHAPE, dtype=np.float32)
    cam[10:30, 14:36, EXPLAINED[0] : EXPLAINED[-1] + 1] = rng.random((20, 22, len(EXPLAINED))) * 0.6
    cam[PEAK] = 0.8  # the strongest voxel is below 1: the page shows the map relative to it
    cam[STRAY] = 0.3
    without_map = np.ones(SHAPE[2], dtype=bool)
    without_map[explained] = False
    cam[:, :, without_map] = 0.0
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    affine[:3, 3] = [-30.0, -36.0, -27.0]
    energy = cam.sum(axis=(0, 1))[explained]
    alone = 0.2 + 0.5 * rng.random(len(explained))
    alone[min(3, len(explained) - 1)] = 0.9  # one slice clearly above the rest, away from both ends
    band = policy == "band"
    summary = {
        "class_names": ["CN", "AD"],
        "probabilities": [0.6, 0.4],
        "predicted_label": "CN",
        "threshold_on_positive_class_probability": 0.5,
        "member_probabilities": [[0.7, 0.3], [0.5, 0.5]],
        "members": [{"seed": 1, "slice_policy": policy}, {"seed": 2, "slice_policy": policy}],
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
            "scored_slice_indices": list(EXPLAINED) if band else PROTOCOL,
        },
        "grad_cam": {
            "target_class": "CN",
            "score": "logit",
            "slice_indices": explained,
            "slice_share_of_attention": [float(value) for value in energy / energy.sum()],
            "slice_probabilities": [[1 - float(p), float(p)] for p in alone],
        },
        "source": {"format": "dicom_series", "note": "SOURCE-MARKER"},
        "acquisition": {"note": "ACQUISITION-MARKER"},
        "warnings": ["The header does not identify this series as T1-weighted.", NOTICE],
        "notice": NOTICE,
    }
    if record in ("members", "text"):
        del summary["preparation"]["scored_slice_indices"]
    if record == "text":
        del summary["members"]
    if not with_map:
        summary["grad_cam"] = None
    return grid, (cam if with_map else None), summary, affine


def unpack(text, shape):
    """Inverse of the embedded encoding, written independently of the module."""

    nx, ny, nz = shape
    steps = np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype=np.uint8).reshape(nz, ny, nx)
    return (np.cumsum(steps.astype(np.int64), axis=2) % 256).astype(np.uint8).transpose(2, 1, 0)


def embedded_payload(page):
    return json.loads(page.split('<script type="application/json" id="vv-data">')[1].split("</script>")[0])


def render(payload, **options):
    return vv._render(payload, heading=options.pop("heading", "h1"), fragment=options.pop("fragment", False), **options)


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.grid, self.cam, self.summary, self.affine = synthetic_inputs()
        self.payload = vv.viewer_payload(self.grid, self.cam, self.summary)

    def test_embedded_volumes_round_trip_exactly(self):
        payload = self.payload
        window = tuple(slice(start, start + size) for start, size in zip(payload["offset"], payload["shape"]))
        expected_anatomy = np.rint(np.clip(self.grid / 800.0, 0, 1) * 255).astype(np.uint8)
        # Attention is shown as a fraction of the strongest voxel, here 0.8.
        expected_map = np.rint(np.clip(self.cam.astype(np.float64) / float(self.cam.max()), 0, 1) * 255).astype(np.uint8)
        anatomy = unpack(payload["anatomy_z"], payload["shape"])
        attention = unpack(payload["attention_z"], payload["shape"])
        np.testing.assert_array_equal(anatomy, expected_anatomy[window])
        np.testing.assert_array_equal(attention, expected_map[window])
        self.assertEqual(int(attention.max()), 255)
        self.assertEqual(payload["checksums"], [int(anatomy.sum(dtype=np.int64)), int(attention.sum(dtype=np.int64))])
        # Cropping to the head loses neither tissue nor attention, including attention outside the head's box.
        outside = np.ones(SHAPE, dtype=bool)
        outside[window] = False
        self.assertFalse((expected_anatomy[outside] > 12).any())
        self.assertFalse(expected_map[outside].any())
        self.assertGreater(int(expected_map[STRAY]), 0)
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

    def test_numbers_are_the_ones_in_the_summary(self):
        payload, summary = self.payload, self.summary
        self.assertEqual(payload["class_names"], ["CN", "AD"])
        self.assertEqual(payload["probabilities"], [0.6, 0.4])
        self.assertEqual(payload["member_probabilities"], [[0.7, 0.3], [0.5, 0.5]])
        self.assertEqual(payload["predicted_label"], "CN")
        self.assertEqual(payload["threshold"], 0.5)
        self.assertEqual(payload["target_class"], "CN")
        self.assertEqual(payload["slice_share"], summary["grad_cam"]["slice_share_of_attention"])
        self.assertEqual(payload["slice_positive_probability"], [row[1] for row in summary["grad_cam"]["slice_probabilities"]])
        # Neither profile is symmetric, so a reversed or shifted list cannot pass.
        self.assertNotEqual(payload["slice_share"], payload["slice_share"][::-1])
        self.assertEqual(int(np.argmax(payload["slice_positive_probability"])), 3)
        self.assertAlmostEqual(sum(payload["slice_share"]), 1.0, places=6)

    def test_wording_follows_which_slices_are_scored_and_which_have_a_map(self):
        # Round 2 ensemble with the default map: five slices scored, all twelve of the slab explained.
        payload = self.payload
        self.assertEqual(
            payload["scored_text"],
            "The score uses 5 axial slices; the map covers all 12 slices of the slab between the lowest and the highest of them.",
        )
        self.assertIn("only 5 of its 12 slices fed the score", payload["map_short"])
        self.assertIn("On the other 7 it shows what the same model responds to", payload["map_text"])
        self.assertIn("did not enter the score", payload["map_text"])
        self.assertEqual(payload["scored_chart_text"], "Only the 5 larger dots are the slices the score is computed from.")
        self.assertIn("the 12 axial slices between the two marked levels (slices 14 to 25 of the 1.5 mm grid)", payload["slab_text"])

        # A model that scores every slice of the band, with the default map: the two sets are the same.
        band = vv.viewer_payload(*synthetic_inputs(policy="band")[:3])
        self.assertEqual(band["scored"], EXPLAINED)
        self.assertEqual(band["scored_text"], "The score uses every one of the 12 axial slices of the slab, and the map covers each of them.")
        self.assertIn("all of which count toward the score", band["map_text"])
        self.assertNotIn("did not enter the score", band["map_text"])
        self.assertEqual(band["scored_chart_text"], "Every slice shown counts toward the score.")

        # The map limited to the protocol slices: separate slices, not a slab.
        sparse = vv.viewer_payload(*synthetic_inputs(explained=PROTOCOL)[:3])
        self.assertEqual(sparse["explained"], PROTOCOL)
        self.assertEqual(
            sparse["scored_text"],
            "The score uses 5 axial slices, and the map covers exactly those; the slices between them were not analysed.",
        )
        self.assertIn("5 separate axial slices", sparse["slab_text"])
        self.assertIn("The slices between them", sparse["slab_text"])

        # A band model with the map limited to the protocol slices: the score uses more slices than have a map.
        partial = vv.viewer_payload(*synthetic_inputs(policy="band", explained=PROTOCOL)[:3])
        self.assertEqual(partial["scored"], EXPLAINED)
        self.assertEqual(partial["scored_text"], "The score uses 12 axial slices; the map covers 5 of them.")
        self.assertEqual(
            partial["scored_chart_text"],
            "All 5 slices shown count toward the score, which also uses 7 slices that have no map.",
        )

        # No map at all: the count still follows the policy.
        grid, _, summary, _ = synthetic_inputs(with_map=False, policy="band")
        self.assertEqual(vv.viewer_payload(grid, None, summary)["scored_text"], "The score uses 12 axial slices.")
        grid, _, summary, _ = synthetic_inputs(with_map=False)
        self.assertEqual(vv.viewer_payload(grid, None, summary)["scored_text"], "The score uses 5 axial slices.")

    def test_slice_policy_is_read_from_older_summaries_too(self):
        for record in ("scored", "members", "text"):
            with self.subTest(record=record):
                self.assertEqual(vv.viewer_payload(*synthetic_inputs(policy="band", record=record)[:3])["scored"], EXPLAINED)
                self.assertEqual(vv.viewer_payload(*synthetic_inputs(record=record)[:3])["scored"], PROTOCOL)
        # What the analysis recorded wins over anything inferred.
        grid, cam, summary, _ = synthetic_inputs(policy="band")
        summary["preparation"]["scored_slice_indices"] = [14, 20]
        self.assertEqual(vv.viewer_payload(grid, cam, summary)["scored"], [14, 20])

    def test_page_is_self_contained_and_names_nobody(self):
        page = render(self.payload)
        self.assertTrue(page.lstrip().lower().startswith("<!doctype html>"))
        # No request leaves the page: the only URL in it is the SVG namespace string.
        self.assertEqual(set(re.findall(r"https?://[^\s\"'<>)]+", page)), {"http://www.w3.org/2000/svg"})
        for forbidden in ("SOURCE-MARKER", "ACQUISITION-MARKER", "fetch(", "XMLHttpRequest", "<link", "@import", "src=\"http",
                          "<img", "new Image"):
            self.assertNotIn(forbidden, page)
        embedded = embedded_payload(page)
        self.assertEqual(embedded, self.payload)
        self.assertEqual(embedded["notice"], NOTICE)
        self.assertEqual(embedded["warnings"], ["The header does not identify this series as T1-weighted."])  # the notice is not repeated
        self.assertNotIn("__VV_", page)
        # The page carries the listed fields and nothing else from the summary (no source, members or acquisition block).
        self.assertEqual(set(self.payload), PAYLOAD_KEYS | MAP_KEYS)
        self.assertEqual(set(vv.viewer_payload(*synthetic_inputs(with_map=False)[:3])), PAYLOAD_KEYS)

    def test_text_from_the_analysis_cannot_break_out_of_the_page(self):
        hostile = 'Notice </script><script>window.pwned = 1</script> <img src=x onerror="window.pwned = 2"> R&D &lt; __VV_TITLE__'
        self.summary["notice"] = hostile
        self.summary["warnings"] = [hostile + " again"]
        payload = vv.viewer_payload(self.grid, self.cam, self.summary)
        template = vv.TEMPLATE_PATH.read_text(encoding="utf-8")
        page = render(payload, title='Map <b>"x"</b>')
        self.assertEqual(page.count("</script"), template.count("</script"))  # no extra script element was opened or closed
        self.assertEqual(page.count("<script"), template.count("<script"))
        self.assertEqual(embedded_payload(page)["notice"], hostile)  # the text itself survives intact, as data
        block = page.split('<script type="application/json" id="vv-data">')[1].split("</script>")[0]
        self.assertFalse(set("<>&") & set(block))  # nothing in the data can open or close markup
        self.assertIn("\\u0026", block)
        self.assertIn("Map &lt;b&gt;&quot;x&quot;&lt;/b&gt;", page)
        self.assertNotIn('Map <b>"x"</b>', page)

    def test_a_title_is_text_and_nothing_else(self):
        # A placeholder inside the title is shown as it is; the data is not inserted a second time.
        page = render(self.payload, title="Map __VV_DATA__ __VV_HEADING__")
        self.assertEqual(page.count(self.payload["anatomy_z"]), 1)
        self.assertIn("<h1 class=\"vv-title\">Map __VV_DATA__ __VV_HEADING__</h1>", page)
        # Pure ASCII whatever the title holds: a host that mislabels the charset cannot garble anything.
        page = render(self.payload, title="Carte d'attention 3D – résumé")
        self.assertTrue(page.isascii())
        self.assertIn("r&#233;sum&#233;", page)
        self.assertTrue(render(self.payload).isascii())
        self.assertTrue(vv.TEMPLATE_PATH.read_text(encoding="utf-8").isascii())

    def test_fragment_embeds_in_another_page(self):
        fragment = vv.viewer_fragment(payload=self.payload, title="Explore the result")
        self.assertIsNone(re.search(r"<(?:!doctype|html|head|body|main)[\s>]", fragment, flags=re.IGNORECASE))
        self.assertTrue(fragment.lstrip().startswith("<style>"))
        self.assertIn('<h2 class="vv-title">Explore the result</h2>', fragment)
        self.assertEqual(fragment.count('<h3 class="vv-sub"'), 3)  # section headings sit one level below the title
        self.assertIn('id="vv-data"', fragment)
        page = render(self.payload)
        self.assertIn(f'<h1 class="vv-title">{vv.DEFAULT_TITLE}</h1>', page)
        self.assertEqual(page.count('<h2 class="vv-sub"'), 3)
        with self.assertRaises(vv.ViewerError):
            vv.viewer_fragment(payload=self.payload, heading="h4")

    def test_scan_without_a_map_still_gets_a_viewer(self):
        grid, _, summary, _ = synthetic_inputs(with_map=False)
        payload = vv.viewer_payload(grid, None, summary)
        self.assertIsNone(payload["attention_z"])
        self.assertEqual(payload["checksums"][1], 0)
        self.assertIn("No attention map is available", payload["no_map_text"])
        page = render(payload)
        self.assertIn(f"<title>{vv.NO_MAP_TITLE}</title>", page)  # not called an attention map
        self.assertNotIn(f"<title>{vv.DEFAULT_TITLE}</title>", page)
        # A map that is zero everywhere has nothing to show either, and the page says which case it is.
        grid, cam, summary, _ = synthetic_inputs()
        empty = vv.viewer_payload(grid, np.zeros_like(cam), summary)
        self.assertIsNone(empty["attention_z"])
        self.assertIn("Grad-CAM produced no attention", empty["no_map_text"])

    def test_slab_outside_the_head_is_kept(self):
        # A slab that reaches beyond the head: its marked levels and its slices must stay inside the picture.
        grid, cam, summary, _ = synthetic_inputs()
        wide = list(range(2, 35))
        summary["grad_cam"]["slice_indices"] = wide
        summary["grad_cam"]["slice_share_of_attention"] = [1 / len(wide)] * len(wide)
        summary["grad_cam"]["slice_probabilities"] = [[0.5, 0.5]] * len(wide)
        payload = vv.viewer_payload(grid, cam, summary)
        self.assertEqual(payload["slab_grid"], [2, 34])
        self.assertGreaterEqual(payload["slab"][0], 0)
        self.assertLess(payload["slab"][1], payload["shape"][2])

    def test_inconsistent_inputs_are_refused(self):
        with self.assertRaisesRegex(vv.ViewerError, "grid shape"):
            vv.viewer_payload(self.grid[:-1], self.cam[:-1], self.summary)
        with self.assertRaisesRegex(vv.ViewerError, "same grid"):
            vv.viewer_payload(self.grid, self.cam[:, :, :-1], self.summary)
        for volume in ("cam", "grid"):
            broken = getattr(self, volume).copy()
            broken[0, 0, 0] = np.nan
            with self.assertRaisesRegex(vv.ViewerError, "nonfinite"):
                vv.viewer_payload(*((self.grid, broken) if volume == "cam" else (broken, self.cam)), self.summary)
        with self.assertRaisesRegex(vv.ViewerError, "3D volume"):
            vv.viewer_payload(self.grid[0], None, self.summary)
        _, _, unordered, _ = synthetic_inputs()
        unordered["grad_cam"]["slice_indices"] = EXPLAINED[::-1]
        with self.assertRaisesRegex(vv.ViewerError, "ascending"):
            vv.viewer_payload(self.grid, self.cam, unordered)
        _, _, three, _ = synthetic_inputs()
        three["class_names"] = ["CN", "MCI", "AD"]
        with self.assertRaisesRegex(vv.ViewerError, "two-class"):
            vv.viewer_payload(self.grid, self.cam, three)
        _, _, nan, _ = synthetic_inputs()
        nan["probabilities"] = [float("nan"), 0.4]
        with self.assertRaisesRegex(vv.ViewerError, "nonfinite numbers"):
            render(vv.viewer_payload(self.grid, self.cam, nan))
        self.summary["preparation"]["intensity_window"] = [5.0, 5.0]
        with self.assertRaisesRegex(vv.ViewerError, "window"):
            vv.viewer_payload(self.grid, self.cam, self.summary)

    def test_a_failed_write_leaves_the_earlier_file_whole(self):
        directory = Path(tempfile.mkdtemp(prefix="volume_viewer_write_"))
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        target = directory / "viewer.html"
        vv._write_whole(target, "first page")
        with patch.object(vv.os, "replace", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                vv._write_whole(target, "second page")
        self.assertEqual(target.read_text(encoding="utf-8"), "first page")
        self.assertEqual([path.name for path in directory.iterdir()], ["viewer.html"])  # no partial file is left behind


@unittest.skipIf(nib is None, "nibabel is not installed")
class SavedFolderTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="volume_viewer_test_"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.grid, self.cam, self.summary, self.affine = synthetic_inputs()
        nib.save(nib.Nifti1Image(self.grid, self.affine), str(self.directory / "anatomy.nii.gz"))
        nib.save(nib.Nifti1Image(self.cam, self.affine), str(self.directory / "gradcam.nii.gz"))
        (self.directory / "result.json").write_text(json.dumps(self.summary))

    def run_cli(self, *arguments, expect):
        """Run the command; a failure must be one logged line and exit code 2, never a traceback."""

        if expect == 0:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(vv.main([str(argument) for argument in arguments]), 0)
            return output.getvalue().strip()
        with self.assertLogs("src.volume_viewer", level="ERROR") as logs:
            self.assertEqual(vv.main([str(argument) for argument in arguments]), expect)
        self.assertEqual(len(logs.records), 1)
        return logs.records[0].getMessage()

    def test_viewer_is_rebuilt_from_a_saved_folder(self):
        expected = vv.viewer_payload(self.grid, self.cam, self.summary)
        self.assertEqual(vv.payload_from_saved(self.directory), expected)
        written = self.directory / vv.VIEWER_FILENAME
        self.assertEqual(self.run_cli(self.directory, expect=0), str(written))
        self.assertEqual(embedded_payload(written.read_text(encoding="utf-8")), expected)
        other = self.directory / "elsewhere" / "map.html"
        self.run_cli(self.directory, "--output", other, expect=0)
        self.assertTrue(other.is_file())

    def test_geometry_comes_from_the_files_own_headers(self):
        expected = vv.viewer_payload(self.grid, self.cam, self.summary)
        # The same map stored in another voxel order (left-right flipped, with the matching affine) is the same
        # map in the world: it must be drawn where it was, not mirrored.
        flip = np.diag([-1.0, 1.0, 1.0, 1.0])
        flip[0, 3] = SHAPE[0] - 1
        nib.save(nib.Nifti1Image(self.cam[::-1], self.affine @ flip), str(self.directory / "gradcam.nii.gz"))
        self.assertEqual(vv.payload_from_saved(self.directory), expected)
        nib.save(nib.Nifti1Image(self.grid[::-1], self.affine @ flip), str(self.directory / "anatomy.nii.gz"))
        self.assertEqual(vv.payload_from_saved(self.directory), expected)
        # A file that sits somewhere else, or on another grid, is refused: result.json no longer describes it.
        moved = self.affine.copy()
        moved[0, 3] += 30.0
        nib.save(nib.Nifti1Image(self.cam, moved), str(self.directory / "gradcam.nii.gz"))
        self.assertIn("gradcam.nii.gz is not on the grid recorded in result.json", self.run_cli(self.directory, expect=2))
        nib.save(nib.Nifti1Image(self.cam, self.affine), str(self.directory / "gradcam.nii.gz"))
        nib.save(nib.Nifti1Image(self.grid[:, :, :-2], self.affine), str(self.directory / "anatomy.nii.gz"))
        self.assertIn("anatomy.nii.gz is not on the grid", self.run_cli(self.directory, expect=2))

    def test_folder_without_a_map_or_without_results(self):
        (self.directory / "gradcam.nii.gz").unlink()
        self.assertIsNone(vv.payload_from_saved(self.directory)["attention_z"])
        self.assertIn("not found", self.run_cli(self.directory / "missing", expect=2))
        (self.directory / "result.json").write_text("{}")
        self.assertIn("result.json lacks 'preparation'", self.run_cli(self.directory, expect=2))

    def test_damaged_folders_end_in_one_error_line(self):
        good = json.dumps(self.summary)
        cases = {
            "not JSON": lambda: (self.directory / "result.json").write_text("{not json"),
            "a list": lambda: (self.directory / "result.json").write_text("[]"),
            "not UTF-8": lambda: (self.directory / "result.json").write_bytes(b"\xff\xfe\x00"),
            "a NaN score": lambda: (self.directory / "result.json").write_text(good.replace("[0.6, 0.4]", "[NaN, 0.4]")),
            "one class": lambda: (self.directory / "result.json").write_text(good.replace('["CN", "AD"]', '["CN"]')),
            "garbage volume": lambda: (self.directory / "anatomy.nii.gz").write_bytes(b"not a NIfTI file"),
            "cut volume": lambda: (self.directory / "gradcam.nii.gz").write_bytes(
                (self.directory / "gradcam.nii.gz").read_bytes()[:200]
            ),
        }
        originals = {name: (self.directory / name).read_bytes() for name in ("result.json", "anatomy.nii.gz", "gradcam.nii.gz")}
        for label, damage in cases.items():
            with self.subTest(case=label):
                for name, content in originals.items():
                    (self.directory / name).write_bytes(content)
                damage()
                message = self.run_cli(self.directory, expect=2)
                self.assertTrue(message.startswith("Cannot build the viewer from "), message)
                self.assertFalse((self.directory / vv.VIEWER_FILENAME).exists())
        for name, content in originals.items():
            (self.directory / name).write_bytes(content)
        # An output path that cannot be written is reported the same way.
        self.run_cli(self.directory, "--output", self.directory, expect=2)
        with patch.object(vv, "TEMPLATE_PATH", self.directory / "no_template.html"):
            self.assertIn("template", self.run_cli(self.directory, expect=2))


def _browser_available():
    node = shutil.which("node")
    if node is None or not BROWSER_CHECK.is_file():
        return False
    # Start the browser once: Playwright can be installed without it.
    probe = "require('playwright').chromium.launch().then((browser) => browser.close()).catch(() => process.exit(3))"
    try:
        return subprocess.run([node, "-e", probe], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


HOSTILE = '</script><script>window.pwned1 = 1</script><img src=x onerror="window.pwned2 = 1"> R&D'
HOSTILE_CLASS = 'AD"><img src=x onerror="window.pwned3 = 1">'


@unittest.skipUnless(_browser_available(), "Node.js with Playwright and its browser is not available")
class BrowserTests(unittest.TestCase):
    """The page in a real browser: what it draws, what it says, and that it draws nothing when it cannot be right."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="volume viewer #1 "))  # a path that needs escaping in a URL
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)
        grid, cam, summary, cls.affine = synthetic_inputs()
        cls.summary = summary
        cls.payload = vv.viewer_payload(grid, cam, summary)
        cls.plain = vv.viewer_payload(*synthetic_inputs(with_map=False)[:3])
        cls.sparse = vv.viewer_payload(*synthetic_inputs(explained=PROTOCOL)[:3])

        _, _, hostile_summary, _ = synthetic_inputs()
        hostile_summary["notice"] = HOSTILE
        hostile_summary["warnings"] = [HOSTILE + " again"]
        hostile_summary["class_names"] = ["<b>CN</b>", HOSTILE_CLASS]
        hostile_summary["predicted_label"] = "<b>CN</b>"
        hostile_summary["grad_cam"]["target_class"] = "<b>CN</b>"
        truncated = dict(cls.payload)
        truncated["attention_z"] = truncated["attention_z"][: len(truncated["attention_z"]) // 12 * 4]
        altered = dict(cls.payload)
        altered["checksums"] = [altered["checksums"][0] + 1, altered["checksums"][1]]
        pages = {
            "main": render(cls.payload),
            "nomap": render(cls.plain),
            "hostile": render(vv.viewer_payload(grid, cam, hostile_summary), title=HOSTILE),
            "sparse": render(cls.sparse),
            "truncated": render(truncated),
            "altered": render(altered),
        }
        spec = {"pages": {}, "marker": [value - start for value, start in zip(MARKER, cls.payload["offset"])]}
        for name, text in pages.items():
            (cls.directory / f"{name}.html").write_text(text, encoding="utf-8")
            spec["pages"][name] = str(cls.directory / f"{name}.html")
        cls.sparse_slices = list(range(PROTOCOL[0] - 1, PROTOCOL[-1] + 2))
        spec["sparse_steps"] = [z - cls.sparse["offset"][2] for z in cls.sparse_slices]
        (cls.directory / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        run = subprocess.run(["node", str(BROWSER_CHECK), str(cls.directory / "spec.json")], capture_output=True, text=True, timeout=600)
        if run.returncode != 0:
            raise AssertionError(f"browser check failed to run:\n{run.stdout}\n{run.stderr}")
        cls.report = json.loads(run.stdout)

    def world(self, index):
        return (self.affine @ [*index, 1.0])[:3]

    def test_loads_without_errors_or_requests(self):
        main = self.report["main"]
        self.assertEqual(main["problems"], [])
        self.assertTrue(main["status_hidden"])
        self.assertEqual(main["overflow"], 0)
        self.assertEqual(main["checksums"], self.payload["checksums"])  # the browser unpacked exactly the saved values
        self.assertEqual(main["start_idx"], self.payload["peak"])
        self.assertEqual(main["heading_levels"], ["H1", "H2", "H2", "H2"])
        self.assertEqual(main["touch_action"], "pan-y")  # a finger on a view can still scroll the page
        self.assertEqual(main["spin_label"], "Rotate")
        self.assertIn("with the attention map drawn over it", main["axial_label"])

    def test_every_view_draws_the_marker_where_its_edge_letters_say(self):
        # The marker voxel is the brightest in the scan and lies in air toward the patient's left, front and top.
        # Read back from the visible canvases: it is bright at the place the convention puts it and dark at the
        # mirror images of that place, and the letter drawn on its side of each view names its true side.
        left, anterior, superior = (value < 0 for value in (self.world(MARKER) * [1, -1, -1]))
        self.assertTrue(left and anterior and superior)
        expected_letters = {
            "vv-axial": {"left": "L", "right": "R", "top": "A", "bottom": "P"},
            "vv-coronal": {"left": "L", "right": "R", "top": "S", "bottom": "I"},
            "vv-sagittal": {"left": "P", "right": "A", "top": "S", "bottom": "I"},  # face to the right
        }
        true_side = {"vv-axial": ("L", "A"), "vv-coronal": ("L", "S"), "vv-sagittal": ("A", "S")}
        for view, seen in self.report["main"]["screen"].items():
            with self.subTest(view=view):
                self.assertGreater(seen["marker"], 150)
                self.assertLess(seen["mirrored_across"], 60)
                self.assertLess(seen["mirrored_up_down"], 60)
                self.assertEqual(seen["letters"], expected_letters[view])
                across, up_down = seen["marker_side"]
                self.assertEqual((seen["letters"][across], seen["letters"][up_down]), true_side[view])

    def test_positions_and_slice_numbers_are_those_of_the_model_grid(self):
        main = self.report["main"]
        x, y, z = self.world(MARKER)
        self.assertEqual(
            main["marker_readout"],
            f"Crosshair {abs(x):.1f} mm left {abs(y):.1f} mm anterior {abs(z):.1f} mm superior outside the analysed slices",
        )
        self.assertEqual(main["marker_slices"], [f"slice {MARKER[0]}", f"slice {MARKER[1]}", f"slice {MARKER[2]}"])
        self.assertEqual(main["marker_now"], "The crosshair is outside the analysed slices.")
        self.assertTrue(main["outside_readout"].endswith("outside the analysed slices"))
        x, y, z = self.world(PEAK)
        expected = f"Crosshair {abs(x):.1f} mm left {abs(y):.1f} mm anterior {abs(z):.1f} mm superior attention 1.00"
        self.assertEqual(main["start_readout"], expected)
        self.assertEqual(main["peak_idx"], self.payload["peak"])
        self.assertEqual(main["peak_readout"], expected)  # the strongest voxel is 1.00 by definition of the scale
        self.assertEqual(main["slider_valuetext"], f"slice {PEAK[2]}, {abs(z):.1f} mm superior")
        self.assertEqual(main["bar_click_slice"], EXPLAINED[3])

    def test_numbers_on_the_page_are_the_ones_in_the_summary(self):
        main, payload = self.report["main"], self.payload
        self.assertEqual(
            main["facts"],
            "Ensemble P(AD) 0.400, label CN at threshold 0.5 (2 members: 0.30, 0.50). " + payload["scored_text"] + " " + payload["map_short"],
        )
        self.assertEqual(main["table_head"], ["Slice", "Height (mm)", "Share of map", "P(AD) alone", "Used for the score"])
        rows = [
            [str(z), f"{self.world((0, 0, z))[2]:.1f}", f"{share * 100:.2f}%", f"{alone:.3f}", "yes" if z in PROTOCOL else ""]
            for z, share, alone in zip(EXPLAINED, payload["slice_share"], payload["slice_positive_probability"])
        ]
        self.assertEqual(main["table_rows"], rows)
        self.assertEqual(int(np.argmax(main["bar_heights"])), int(np.argmax(payload["slice_share"])))
        self.assertEqual(int(np.argmin(main["dot_heights"])), int(np.argmax(payload["slice_positive_probability"])))
        self.assertEqual(main["scored_dots"], [EXPLAINED.index(z) for z in PROTOCOL])
        self.assertIn(payload["scored_chart_text"], main["prob_text"])
        alone = payload["slice_positive_probability"][EXPLAINED.index(PEAK[2])]
        self.assertEqual(main["start_now"], f"At the crosshair: slice {PEAK[2]}, {self.world(PEAK)[2]:.1f} mm: P(AD) {alone:.2f}, used for the score")
        self.assertIn(payload["map_text"], main["read"])
        self.assertIn("seen from the patient's right side, so the face points to the right", main["read"])

    def test_controls_and_pointers(self):
        main = self.report["main"]
        red, green, blue = main["colour_with_map"]
        self.assertGreater(blue - red, 20)  # the map is drawn over the strongest voxel
        self.assertEqual(len(set(main["colour_without_map"])), 1)  # and gone when it is switched off
        for tip in main["chart_tips"]:
            self.assertEqual((tip["shown"], tip["inside"], tip["overflow"]), (True, True, 0), tip)
        self.assertTrue(main["chart_tips"][0]["text"].startswith(f"slice {EXPLAINED[0]}, "))
        self.assertTrue(main["chart_tips"][1]["text"].startswith(f"slice {EXPLAINED[-1]}, "))
        self.assertEqual((main["view_tip"]["shown"], main["view_tip"]["inside"]), (True, True), main["view_tip"])
        self.assertFalse(main["right_button_moves"])
        self.assertEqual(main["drag_past_edge"][0], main["drag_past_edge"][1])  # a drag past the image reaches the last voxel
        self.assertTrue(self.report["touch"]["tap_places_crosshair"])
        self.assertEqual(self.report["touch"]["problems"], [])

    def test_3d_view_shows_the_head_and_turns_with_the_pointer(self):
        view = self.report["main"]["view3d"]
        self.assertGreater(view["head_share"], 0.15)  # the projection itself, not only the orientation marker
        self.assertTrue(view["picture_changes"])
        # Dragging right or down, or pressing the right arrow, moves the near side of the head the same way.
        self.assertAlmostEqual(view["drag_right_yaw"], -0.4, places=6)
        self.assertAlmostEqual(view["drag_down_pitch"], 0.3, places=6)
        self.assertAlmostEqual(view["arrow_right_yaw"], -0.12, places=6)
        self.assertEqual(view["caption_near_full_turn"], "turn 0° tilt −17°")

    def test_rotation_runs_only_while_the_view_is_on_screen(self):
        motion = self.report["motion"]
        self.assertEqual(motion["problems"], [])
        self.assertEqual(motion["turning_after_load"], "true")
        self.assertTrue(motion["turns_while_visible"])
        self.assertEqual(motion["off_screen_fraction"], 0)
        self.assertFalse(motion["turns_while_off_screen"])
        self.assertTrue(motion["turns_again_when_back"])
        # The one automatic turn waits until most of the view can be seen.
        self.assertLess(motion["short_window"]["fraction"], 0.6)
        self.assertEqual(motion["short_window"]["pressed"], "false")
        self.assertEqual(motion["short_window"]["pressed_once_in_view"], "true")

    def test_page_without_a_map_works_and_does_not_mention_one(self):
        plain = self.report["plain"]
        self.assertEqual(plain["problems"], [])
        self.assertTrue(plain["status_hidden"])
        self.assertEqual(plain["overflow"], 0)
        self.assertTrue(plain["scan_drawn"])
        self.assertEqual((plain["title"], plain["document_title"]), (vv.NO_MAP_TITLE, vv.NO_MAP_TITLE))
        self.assertEqual(plain["attention_controls_shown"], 0)
        self.assertFalse(any("attention" in label for label in plain["labels"]))
        centre = [n // 2 + start for n, start in zip(self.plain["shape"], self.plain["offset"])]
        x, y, z = self.world(centre)
        sides = [("left", "right"), ("posterior", "anterior"), ("inferior", "superior")]
        parts = [f"{abs(value):.1f} mm {names[0 if value < 0 else 1]}" for value, names in zip((x, y, z), sides)]
        self.assertEqual(plain["readout"], "Crosshair " + " ".join(parts))
        self.assertEqual(plain["facts"], "Ensemble P(AD) 0.400, label CN at threshold 0.5 (2 members: 0.30, 0.50). The score uses 5 axial slices.")
        self.assertIn("No attention map. " + self.plain["no_map_text"], plain["read"])

    def test_text_from_the_analysis_is_shown_as_text(self):
        hostile = self.report["hostile"]
        self.assertEqual(hostile["problems"], [])
        self.assertEqual(hostile["pwned"], [])
        self.assertEqual(hostile["scripts"], 2)  # the data block and the page's own script
        self.assertEqual(hostile["injected_elements"], 0)
        self.assertEqual(hostile["handler_attributes"], 0)
        self.assertEqual(hostile["notice"], HOSTILE)
        self.assertEqual(hostile["warnings"], [HOSTILE + " again"])
        self.assertEqual(hostile["title"], HOSTILE)
        self.assertIn(f"Ensemble P({HOSTILE_CLASS}) 0.400, label <b>CN</b>", hostile["facts"])
        self.assertIn(f"P({HOSTILE_CLASS}) ", hostile["tip"])
        self.assertEqual(hostile["table_head"][3], f"P({HOSTILE_CLASS}) alone")

    def test_slices_without_a_map_are_not_reported_as_zero_attention(self):
        sparse = self.report["sparse"]
        self.assertEqual(sparse["problems"], [])
        self.assertIn(self.sparse["scored_text"], sparse["facts"])
        self.assertIn("5 separate axial slices", sparse["read"])
        for z, step in zip(self.sparse_slices, sparse["steps"]):
            with self.subTest(slice=z):
                self.assertEqual(step["slice"], f"slice {z}")
                if z in PROTOCOL:
                    self.assertRegex(step["note"], r"^attention \d\.\d\d$")
                    self.assertTrue(step["now"].startswith(f"At the crosshair: slice {z}, "), step["now"])
                elif PROTOCOL[0] < z < PROTOCOL[-1]:
                    self.assertEqual(step["note"], "no map on this slice")
                    self.assertEqual(step["now"], "The crosshair is on a slice without a map.")
                else:
                    self.assertEqual(step["note"], "outside the analysed slices")
                    self.assertEqual(step["now"], "The crosshair is outside the analysed slices.")

    def test_a_page_that_cannot_be_right_draws_nothing_and_says_so(self):
        expected = {"truncated": "could not be unpacked", "altered": "does not match its checksum", "old_browser": "cannot unpack"}
        for name, words in expected.items():
            with self.subTest(page=name):
                refused = self.report[name]
                self.assertIn(words, refused["status"])
                self.assertFalse(refused["status_hidden"])
                self.assertEqual(refused["drawn"], 0)
                self.assertFalse(refused["exposed"])
                self.assertEqual(refused["problems"], [])  # using the controls afterwards raises no error


if __name__ == "__main__":
    unittest.main()
