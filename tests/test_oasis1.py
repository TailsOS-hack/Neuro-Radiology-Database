"""OASIS-1 index and loader contracts on a synthetic tree (no OASIS data, no network)."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

if nib is not None:
    from src import oasis1

CSV = """ID,M/F,Hand,Age,Educ,SES,MMSE,CDR,eTIV,nWBV,ASF,Delay
OAS1_0001_MR1,F,R,74,2,3,29,0,1344,0.743,1.306,N/A
OAS1_0002_MR1,F,R,55,4,1,29,0,1147,0.810,1.531,N/A
OAS1_0003_MR1,F,R,80,4,3,27,0.5,1454,0.708,1.207,N/A
OAS1_0004_MR1,M,R,22,,,,,1588,0.803,1.105,N/A
OAS1_0005_MR1,M,R,70,2,2,21,1,1500,0.700,1.170,N/A
OAS1_0006_MR1,M,R,88,1,4,15,2,1400,0.650,1.250,N/A
OAS1_0001_MR2,F,R,74,,,,0,1344,0.743,1.306,30
"""


def write_session(root, disc, session_id, shape=(20, 24, 16), marker=None, with_image=True):
    folder = root / disc / session_id / "PROCESSED" / "MPRAGE" / "SUBJ_111"
    folder.mkdir(parents=True)
    (root / disc / session_id / "RAW").mkdir()
    if not with_image:
        return
    data = np.full(shape, 100, dtype=np.int16)
    if marker is not None:
        data[marker] = 2000
    # OASIS stores each image as a 4D Analyze volume with a single frame.
    nib.save(nib.AnalyzeImage(data[..., None], np.eye(4)), str(folder / f"{session_id}_mpr_n4_anon_sbj_111.img"))


@unittest.skipIf(nib is None, "nibabel is not installed")
class Oasis1Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="oasis1_test_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "oasis_cross-sectional.csv").write_text(CSV)
        (self.root / "oasis_cross-sectional-reliability.csv").write_text("ID,Age,CDR\nOAS1_0099_MR1,70,0\n")
        write_session(self.root, "disc1", "OAS1_0001_MR1", marker=(17, 20, 2))
        write_session(self.root, "disc1", "OAS1_0002_MR1")
        write_session(self.root, "disc2", "OAS1_0003_MR1")
        write_session(self.root, "disc2", "OAS1_0004_MR1")
        write_session(self.root, "disc2", "OAS1_0005_MR1", with_image=False)  # clinical row, no image
        write_session(self.root, "disc3", "OAS1_0006_MR1")
        write_session(self.root, "disc3", "OAS1_0001_MR2")  # same person, repeat session
        write_session(self.root, "disc3", "OAS1_0009_MR1")  # image, no clinical row

    def test_index_joins_clinical_rows_and_folders(self):
        sessions = {session.session_id: session for session in oasis1.load_index(self.root)}
        self.assertEqual(len(sessions), 8)
        first = sessions["OAS1_0001_MR1"]
        self.assertEqual((first.subject_id, first.visit, first.age, first.cdr, first.mmse, first.sex), ("OAS1_0001", 1, 74.0, 0.0, 29.0, "F"))
        self.assertEqual(first.image_path.name, "OAS1_0001_MR1_mpr_n4_anon_sbj_111.img")
        self.assertIsNone(sessions["OAS1_0004_MR1"].cdr)  # younger adult: CDR not assessed
        self.assertIsNone(sessions["OAS1_0005_MR1"].image_path)
        self.assertEqual(sessions["OAS1_0001_MR2"].subject_id, "OAS1_0001")
        self.assertEqual(sessions["OAS1_0001_MR2"].visit, 2)
        self.assertIsNone(sessions["OAS1_0009_MR1"].age)
        self.assertNotIn("OAS1_0099_MR1", sessions)  # the reliability table is not the clinical table

    def test_eligibility_is_older_adults_with_a_rating_first_visit_only(self):
        sessions = oasis1.load_index(self.root)
        eligible = [session.session_id for session in oasis1.eligible_sessions(sessions)]
        # too young (0002, 0004), no image (0005), repeat visit (0001_MR2) and no clinical row (0009) are out
        self.assertEqual(eligible, ["OAS1_0001_MR1", "OAS1_0003_MR1", "OAS1_0006_MR1"])
        self.assertEqual([s.session_id for s in oasis1.eligible_sessions(sessions, min_age=50)],
                         ["OAS1_0001_MR1", "OAS1_0002_MR1", "OAS1_0003_MR1", "OAS1_0006_MR1"])
        summary = oasis1.summarize(sessions)
        self.assertEqual(summary["eligible_people"], 3)
        self.assertEqual(summary["eligible_by_cdr"], {"0": 1, "0.5": 1, "2": 1})
        self.assertEqual((summary["eligible_nondemented_cdr0"], summary["eligible_demented_cdr_above_0"]), (1, 2))
        self.assertEqual(summary["repeat_sessions"], 1)
        self.assertEqual(summary["sessions_without_clinical_row"], 1)
        self.assertIs(summary["matches_documented_dementia_counts"], False)  # a three-person tree is not the cohort
        self.assertEqual(summary["documented_dementia_counts_by_cdr"], {"0.5": 70, "1": 28, "2": 2})
        self.assertNotIn("OAS1_", json.dumps(summary))  # aggregate only

    def test_orientation_matches_the_provider_derived_mapping(self):
        # For the documented grid the mapping is exactly the matrix the Clinica converter hard-codes.
        np.testing.assert_array_equal(
            oasis1.subj_111_affine((256, 256, 160)),
            [[0, 0, -1, 80], [1, 0, 0, -128], [0, 1, 0, -128], [0, 0, 0, 1]],
        )
        session = next(s for s in oasis1.load_index(self.root) if s.session_id == "OAS1_0001_MR1")
        volume = oasis1.load_session_volume(session)
        self.assertEqual(volume.orientation, ("R", "A", "S"))
        self.assertEqual(volume.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(volume.shape, (16, 20, 24))  # stored (anterior, superior, left) -> canonical (R, A, S)
        index = np.unravel_index(int(np.argmax(volume.data)), volume.data.shape)
        # Stored voxel (17, 20, 2): anterior of centre, superior of centre, and toward the right (low k).
        np.testing.assert_allclose((volume.affine @ [*index, 1])[:3], [16 / 2 - 2, 17 - 20 / 2, 20 - 24 / 2])
        self.assertEqual(volume.acquisition["contrast_guess"]["label"], "T1w")
        self.assertEqual(volume.source["dataset"], "OASIS-1")
        self.assertNotIn("OAS1_", json.dumps(volume.describe()))
        self.assertTrue(any("not the documented" in note for note in volume.warnings))  # synthetic grid is small

    def test_image_is_found_by_folder_name_whatever_the_nesting(self):
        other = self.root / "disc4" / "OAS1_0010_MR1" / "SOMEWHERE" / "SUBJ_111"
        other.mkdir(parents=True)
        nib.save(nib.AnalyzeImage(np.zeros((8, 8, 8, 1), dtype=np.int16), np.eye(4)), str(other / "image_b.img"))
        nib.save(nib.AnalyzeImage(np.zeros((8, 8, 8, 1), dtype=np.int16), np.eye(4)), str(other / "image_a.img"))
        decoy = self.root / "disc4" / "OAS1_0010_MR1" / "PROCESSED" / "MPRAGE" / "T88_111"
        decoy.mkdir(parents=True)
        nib.save(nib.AnalyzeImage(np.zeros((8, 8, 8, 1), dtype=np.int16), np.eye(4)), str(decoy / "atlas.img"))
        session = next(s for s in oasis1.load_index(self.root) if s.session_id == "OAS1_0010_MR1")
        self.assertEqual(session.image_path, other / "image_a.img")  # first by name, never the atlas-space image

    def test_problems_are_reported(self):
        session = next(s for s in oasis1.load_index(self.root) if s.session_id == "OAS1_0005_MR1")
        with self.assertRaisesRegex(oasis1.OasisError, "no PROCESSED/MPRAGE/SUBJ_111 image"):
            oasis1.load_session_volume(session)
        write_session(self.root, "copy", "OAS1_0003_MR1")
        with self.assertRaisesRegex(oasis1.OasisError, "appears twice"):
            oasis1.load_index(self.root)
        shutil.rmtree(self.root / "copy")
        (self.root / "oasis_cross-sectional.csv").write_text("ID,Age\nOAS1_0001_MR1,74\n")
        with self.assertRaisesRegex(oasis1.OasisError, "lacks the column"):
            oasis1.load_index(self.root)
        (self.root / "oasis_cross-sectional.csv").unlink()
        (self.root / "oasis_cross-sectional.xlsx").write_bytes(b"not really a spreadsheet")
        with self.assertRaisesRegex(oasis1.OasisError, "save it as CSV"):
            oasis1.load_index(self.root)
        with self.assertRaises(oasis1.OasisError):
            oasis1.load_index(self.root / "missing")

    def test_command_line_summary_and_orientation_sheet(self):
        from PIL import Image

        sheet = self.root / "qc" / "sheet.png"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(oasis1.main([str(self.root), "--qc", str(sheet)]), 0)
        summary = json.loads(buffer.getvalue())
        self.assertEqual(summary["eligible_people"], 3)
        with Image.open(sheet) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (16 + 4 + 16 + 4 + 20 + 4, 3 * (24 + 4)))  # axial | coronal | sagittal, three rows
        with self.assertLogs("src.oasis1", level="ERROR"):
            self.assertEqual(oasis1.main([str(self.root / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
