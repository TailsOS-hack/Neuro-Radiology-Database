"""Headless contracts for scan display and header-derived form values (numpy + Pillow only)."""

import types
import unittest
from datetime import date

import numpy as np

from src import scan_view
from src.mri_volume import MRIVolume


def volume_with_marker(shape=(40, 60, 20), spacing=(1.0, 1.0, 1.0)):
    data = np.full(shape, 100.0, dtype=np.float32)
    if shape[2] > 7:
        data[2:6, shape[1] - 6 : shape[1] - 2, 7] = 1000.0  # patient left (low x), anterior (high y), slice 7
    return MRIVolume(data, np.diag([*spacing, 1.0]))


class ScanViewTests(unittest.TestCase):
    def test_axial_slice_is_anterior_up_and_left_on_left(self):
        volume = volume_with_marker()
        image = np.asarray(scan_view.axial_slice_image(volume, 7, annotate=False))
        self.assertEqual(image.shape, (60, 40, 3))  # rows follow the anterior-posterior axis
        rows, columns = np.where(image[:, :, 0] == image[:, :, 0].max())
        self.assertLess(rows.mean(), 10)  # anterior at the top
        self.assertLess(columns.mean(), 10)  # patient left on image left
        other = np.asarray(scan_view.axial_slice_image(volume, 3, annotate=False))
        self.assertEqual(int(other.max()), int(other.min()))  # the marker is only in slice 7
        clipped = scan_view.axial_slice_image(volume, 999, annotate=False)  # out of range -> last slice
        self.assertEqual(clipped.size, (40, 60))

    def test_anisotropic_voxels_keep_physical_aspect(self):
        volume = volume_with_marker(shape=(40, 30, 20), spacing=(1.0, 2.0, 5.0))
        self.assertEqual(scan_view.axial_slice_image(volume, 0, annotate=False).size, (40, 60))

    def test_window_is_shared_by_all_slices_and_robust_to_flat_data(self):
        volume = volume_with_marker()
        low, high = scan_view.display_window(volume)
        self.assertLess(low, high)
        flat = MRIVolume(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))
        low, high = scan_view.display_window(flat)
        self.assertLess(low, high)
        self.assertEqual(int(np.asarray(scan_view.axial_slice_image(flat, 3, annotate=False)).max()), 0)

    def test_orientation_letters_do_not_touch_the_classified_image(self):
        volume = volume_with_marker(shape=(120, 120, 20))
        plain = scan_view.axial_slice_image(volume, 3, annotate=False)
        annotated = scan_view.annotate_orientation(plain)
        self.assertEqual(int(np.asarray(plain).max()), int(np.asarray(plain).min()))  # untouched input
        self.assertGreater(int((np.asarray(annotated) != np.asarray(plain)).sum()), 0)
        tiny = scan_view.axial_slice_image(volume_with_marker(shape=(20, 20, 4)), 0, annotate=True)
        self.assertEqual(tiny.size, (20, 20))  # too small to label: returned as is

    def test_person_name_and_dates(self):
        self.assertEqual(scan_view.format_person_name("DOE^JANE^Q"), "JANE Q DOE")
        self.assertEqual(scan_view.format_person_name("DOE^JANE=ideographic^x"), "JANE DOE")
        self.assertEqual(scan_view.format_person_name("Single"), "Single")
        self.assertEqual(scan_view.format_person_name(""), "")
        self.assertEqual(scan_view.parse_dicom_date("19400229"), date(1940, 2, 29))
        self.assertEqual(scan_view.parse_dicom_date("1940.02.29"), date(1940, 2, 29))
        for bad in ("", "1940", "19401332", "abcdefgh", None):
            self.assertIsNone(scan_view.parse_dicom_date(bad))

    def test_technique_and_contrast_come_only_from_recorded_parameters(self):
        volume = MRIVolume(
            np.zeros((176, 256, 256), dtype=np.float32), np.eye(4),
            acquisition={"modality": "MR", "field_strength_t": 3.0, "mr_acquisition_type": "3D", "repetition_time_ms": 2300.0,
                         "echo_time_ms": 2.98, "inversion_time_ms": 900.0, "flip_angle_deg": 9.0,
                         "contrast_guess": {"label": "T1w"}, "series_description": "SECRET FREE TEXT"},
            source={"format": "dicom_series"},
        )
        technique = scan_view.technique_from_header(volume)
        self.assertEqual(
            technique, "MR 3 T 3D T1w (TR 2300 ms, TE 2.98 ms, TI 900 ms, flip 9 deg); 176 x 256 x 256 voxels at 1 x 1 x 1 mm"
        )
        self.assertNotIn("SECRET", technique + scan_view.scan_caption(volume, 4))
        self.assertIn("axial slice 5 of 256", scan_view.scan_caption(volume, 4))
        self.assertEqual(scan_view.contrast_from_header(volume), "None recorded in image header")
        volume.acquisition["contrast_agent"] = "Gadovist"
        self.assertEqual(scan_view.contrast_from_header(volume), "Gadovist")
        bare = MRIVolume(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))
        self.assertEqual(scan_view.technique_from_header(bare), "MR; 16 x 16 x 16 voxels at 1 x 1 x 1 mm")

    def test_volume_eligibility(self):
        self.assertTrue(scan_view.is_volume(MRIVolume(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))))
        self.assertFalse(scan_view.is_volume(MRIVolume(np.zeros((64, 64, 1), dtype=np.float32), np.eye(4))))

    def test_volume_report_states_the_limits_and_escapes_user_text(self):
        result = types.SimpleNamespace(
            class_names=("CN", "AD"),
            probabilities=np.array([0.536, 0.464]),
            member_probabilities=np.array([[0.59, 0.41], [0.45, 0.55], [0.56, 0.44]]),
            predicted_label="CN",
            warnings=["The series looks like FLAIR, not T1-weighted", "Research output only, not a diagnosis. ..."],
            cam_grid=np.zeros((2, 2, 2)),
        )
        text = scan_view.volume_report_html(result, {"name": "<b>Jane</b> Doe", "patient_id": "A&B", "dob": ""}, "MR 3 T")
        self.assertIn("research score, not a diagnosis", text)
        self.assertIn("specificity 0/10", text)
        self.assertIn("must not inform patient care", text)
        self.assertIn("<b>Ensemble P(AD):</b> 0.464", text)
        self.assertIn("0.41, 0.55, 0.44", text)
        self.assertIn("&lt;b&gt;Jane&lt;/b&gt; Doe", text)  # user-typed text is escaped
        self.assertIn("A&amp;B", text)
        self.assertIn("<b>Date of birth:</b> Not provided", text)
        self.assertIn("looks like FLAIR", text)
        self.assertEqual(text.count("Research output only"), 0)  # the notice is not repeated as a scan note
        self.assertIn("slice by slice", text)
        result.cam_grid = None
        self.assertNotIn("slice by slice", scan_view.volume_report_html(result))


if __name__ == "__main__":
    unittest.main()
