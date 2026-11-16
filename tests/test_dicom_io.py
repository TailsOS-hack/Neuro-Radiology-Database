"""DICOM ingestion contracts: geometry, intensities, failure modes and PHI separation.

Synthetic series are written by ``dicom_synth`` so every acquisition plane,
slice direction and file order can be demanded to read back to the *same*
canonical volume. Real scanner files bundled with pydicom cover pixel decoding.
No network, models or patient data. Skipped where pydicom is not installed.
"""

import itertools
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

try:
    import pydicom
except ModuleNotFoundError:
    pydicom = None

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

if pydicom is not None:
    try:
        import dicom_synth as synth  # unittest discovery / pytest put tests/ on sys.path
    except ModuleNotFoundError:
        from tests import dicom_synth as synth  # python -m unittest tests.test_dicom_io

    from src import dicom_io
    from src.mri_volume import MRIVolume, PatientIdentity, axis_codes, reorient_to_ras

FAKE_PHI = ("SYNTHETIC", "PHANTOM", "FAKE-ID", "19400229", "FAKE-ACC", "Nowhere", "FAKESCANNER", "SN-000")


def bundled(name):
    """Path of a DICOM file shipped inside the pydicom wheel (never downloaded)."""

    path = Path(pydicom.data.__file__).parent / "test_files" / name
    return path if path.exists() else None


@unittest.skipIf(pydicom is None, "pydicom is not installed")
class DicomGeometryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="dicom_io_test_"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.volume = synth.phantom((14, 18, 11))
        self.affine = synth.oblique_affine()
        self.expected, self.expected_affine = reorient_to_ras(self.volume.astype(np.float32), self.affine)

    def assert_reads_back(self, path, expected=None, **load):
        loaded = dicom_io.load_volume(path, **load)
        np.testing.assert_array_equal(loaded.data, self.expected if expected is None else expected)
        np.testing.assert_allclose(loaded.affine, self.expected_affine, rtol=0, atol=1e-5)
        self.assertEqual(loaded.orientation, ("R", "A", "S"))
        return loaded

    def test_every_acquisition_orientation_reads_back_to_the_same_volume(self):
        # 6 axis orders x 8 flip patterns: axial, coronal and sagittal stacks in both directions.
        for case, (permutation, flips) in enumerate(
            itertools.product(itertools.permutations(range(3)), itertools.product([False, True], repeat=3))
        ):
            with self.subTest(permutation=permutation, flips=flips):
                data, affine = synth.reoriented(self.volume, self.affine, permutation, flips)
                folder = self.directory / f"case{case}"
                synth.write_series(folder, data, affine, shuffle=case, reverse_instance_numbers=bool(case % 2))
                self.assert_reads_back(folder)

    def test_axis_aligned_volume_has_expected_world_coordinates(self):
        # Voxel (i, j, k) of the canonical array must sit at a known RAS position.
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        affine[:3, 3] = (-10.0, -20.0, -30.0)
        data, written_affine = synth.reoriented(self.volume, affine, (2, 0, 1), (True, True, False))
        synth.write_series(self.directory / "aligned", data, written_affine)
        loaded = dicom_io.load_volume(self.directory / "aligned")
        np.testing.assert_allclose(loaded.affine, affine, atol=1e-6)
        self.assertEqual(loaded.spacing, (2.0, 3.0, 4.0))
        index = np.unravel_index(int(np.argmax(self.volume)), self.volume.shape)
        self.assertEqual(float(loaded.data[index]), float(self.volume.max()))
        np.testing.assert_allclose(loaded.affine @ [*index, 1], [-10 + 2 * index[0], -20 + 3 * index[1], -30 + 4 * index[2], 1])

    def test_slice_order_ignores_file_names_and_instance_numbers(self):
        first = self.directory / "a"
        second = self.directory / "b"
        synth.write_series(first, self.volume, self.affine, shuffle=1)
        synth.write_series(second, self.volume, self.affine, shuffle=99, reverse_instance_numbers=True)
        np.testing.assert_array_equal(dicom_io.load_volume(first).data, dicom_io.load_volume(second).data)

    def test_slice_step_comes_from_positions_not_slice_thickness(self):
        folder = self.directory / "thick"
        for path in synth.write_series(folder, self.volume, self.affine):
            dataset = pydicom.dcmread(path)
            dataset.SliceThickness = "9.9"  # thickness != spacing between slice centres
            dataset.save_as(path)
        self.assert_reads_back(folder)

    def test_enhanced_multiframe_uses_per_frame_positions(self):
        synth.write_enhanced(self.directory / "enhanced" / "e.dcm", self.volume, self.affine)
        loaded = self.assert_reads_back(self.directory / "enhanced")
        self.assertEqual(loaded.source["format"], "dicom_multiframe")
        self.assertEqual(loaded.source["files"], 1)
        self.assertEqual(loaded.acquisition["repetition_time_ms"], 2300.0)
        self.assertEqual(loaded.acquisition["echo_time_ms"], 2.98)
        self.assertEqual(loaded.acquisition["inversion_time_ms"], 900.0)

        order = np.random.default_rng(3).permutation(self.volume.shape[2])
        synth.write_enhanced(self.directory / "enhanced2" / "e.dcm", self.volume, self.affine, frame_order=order, per_frame_orientation=True)
        self.assert_reads_back(self.directory / "enhanced2")

    def test_zip_archive_matches_folder(self):
        folder = self.directory / "series"
        synth.write_series(folder, self.volume, self.affine)
        archive = self.directory / "series.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in sorted(folder.iterdir()):
                handle.write(path, f"nested/{path.name}")
        from_folder = dicom_io.load_volume(folder)
        from_zip = self.assert_reads_back(archive)
        self.assertEqual(from_zip.source["content_sha256"], from_folder.source["content_sha256"])

    def test_single_file_and_extensionless_files(self):
        folder = self.directory / "noext"
        files = synth.write_series(folder, self.volume.astype(np.int16), self.affine, extension="", transfer_syntax="implicit")
        self.assert_reads_back(folder)
        single = dicom_io.load_volume(files[0])
        self.assertEqual(min(single.shape), 1)
        self.assertTrue(dicom_io.is_dicom_file(files[0]))

    def test_legacy_files_without_preamble_or_file_meta(self):
        for syntax in ("implicit", "explicit"):
            with self.subTest(syntax=syntax):
                folder = self.directory / f"bare_{syntax}"
                files = synth.write_series(folder, self.volume, self.affine, transfer_syntax=syntax, bare=True, extension="")
                self.assertNotEqual(files[0].read_bytes()[128:132], b"DICM")
                self.assertTrue(dicom_io.is_dicom_file(files[0]))
                self.assert_reads_back(folder)

    def test_orientation_jitter_between_slices_does_not_split_the_series(self):
        folder = self.directory / "jitter"
        files = synth.write_series(folder, self.volume, self.affine, shuffle=None)
        for number, path in enumerate(files):  # converters sometimes re-serialise cosines slice by slice
            dataset = pydicom.dcmread(path)
            cosines = [float(value) + (4e-4 if number % 2 else -4e-4) for value in dataset.ImageOrientationPatient]
            dataset.ImageOrientationPatient = [f"{value:.7f}" for value in cosines]
            dataset.save_as(path)
        series = dicom_io.discover_series(folder)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].n_slices, self.volume.shape[2])
        np.testing.assert_array_equal(dicom_io.load_series(series[0]).data, self.expected)

    def test_folder_scan_is_capped(self):
        from unittest.mock import patch

        folder = self.directory / "many"
        synth.write_series(folder, self.volume, self.affine)
        with patch.object(dicom_io, "_MAX_FILES_SCANNED", 5):
            with self.assertRaisesRegex(dicom_io.DicomError, "select the study or series folder"):
                dicom_io.discover_series(folder)

    def test_duplicate_copies_are_ignored(self):
        folder = self.directory / "dup"
        synth.write_series(folder, self.volume, self.affine)
        shutil.copytree(folder, folder / "copy")
        self.assert_reads_back(folder)

    def test_missing_slice_is_an_error_not_a_silent_gap(self):
        folder = self.directory / "missing"
        synth.write_series(folder, self.volume, self.affine, skip_slices=[4])
        with self.assertRaisesRegex(dicom_io.DicomGeometryError, "1 slice\\(s\\) are missing"):
            dicom_io.load_volume(folder)

    def test_irregular_spacing_is_an_error(self):
        folder = self.directory / "irregular"
        files = synth.write_series(folder, self.volume, self.affine, shuffle=None)
        dataset = pydicom.dcmread(files[5])
        position = np.array([float(value) for value in dataset.ImagePositionPatient])
        normal = np.cross(*np.array([float(value) for value in dataset.ImageOrientationPatient]).reshape(2, 3))
        dataset.ImagePositionPatient = [f"{value:.6f}" for value in position + 0.4 * normal]
        dataset.save_as(files[5])
        with self.assertRaisesRegex(dicom_io.DicomGeometryError, "not uniform"):
            dicom_io.load_volume(folder)

    def test_multi_echo_series_is_split_into_volumes(self):
        folder = self.directory / "echoes"
        series_uid, study_uid = synth.generate_uid(), synth.generate_uid()
        second = (self.volume // 2).astype(np.uint16)
        synth.write_series(folder, self.volume, self.affine, series_uid=series_uid, study_uid=study_uid, echo_time=2.0, filename_prefix="A")
        synth.write_series(folder, second, self.affine, series_uid=series_uid, study_uid=study_uid, echo_time=6.0, filename_prefix="B")
        first = self.assert_reads_back(folder)
        self.assertTrue(any("2 volumes" in note for note in first.warnings))
        self.assert_reads_back(folder, expected=reorient_to_ras(second.astype(np.float32), self.affine)[0], volume_index=1)
        with self.assertRaises(dicom_io.DicomGeometryError):
            dicom_io.load_volume(folder, volume_index=2)

    def test_incomplete_second_volume_is_rejected(self):
        folder = self.directory / "partial"
        series_uid, study_uid = synth.generate_uid(), synth.generate_uid()
        synth.write_series(folder, self.volume, self.affine, series_uid=series_uid, study_uid=study_uid, filename_prefix="A")
        synth.write_series(folder, self.volume, self.affine, series_uid=series_uid, study_uid=study_uid, filename_prefix="B", skip_slices=[0, 1])
        with self.assertRaisesRegex(dicom_io.DicomGeometryError, "inconsistent number"):
            dicom_io.load_volume(folder)

    @unittest.skipIf(nib is None, "nibabel is not installed")
    def test_canonical_form_matches_nibabel(self):
        for permutation, flips in [((0, 1, 2), (False, False, False)), ((2, 0, 1), (True, False, True)), ((1, 2, 0), (False, True, True))]:
            data, affine = synth.reoriented(self.volume, self.affine, permutation, flips)
            reference = nib.as_closest_canonical(nib.Nifti1Image(data.astype(np.float32), affine))
            canonical, canonical_affine = reorient_to_ras(data.astype(np.float32), affine)
            np.testing.assert_array_equal(canonical, reference.get_fdata(dtype=np.float32))
            np.testing.assert_allclose(canonical_affine, reference.affine, atol=1e-12)
            self.assertEqual(axis_codes(canonical_affine), nib.aff2axcodes(reference.affine))


@unittest.skipIf(pydicom is None, "pydicom is not installed")
class DicomIntensityTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="dicom_io_test_"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.volume = synth.phantom((12, 10, 8))
        self.affine = synth.oblique_affine()

    def test_rescale_is_applied_per_slice(self):
        count = self.volume.shape[2]
        slopes, intercepts = np.linspace(0.5, 2.0, count), np.linspace(-10.0, 30.0, count)
        synth.write_series(self.directory, self.volume, self.affine, slopes=slopes, intercepts=intercepts)
        loaded = dicom_io.load_volume(self.directory)
        expected = reorient_to_ras(self.volume.astype(np.float64) * slopes + intercepts, self.affine)[0]
        np.testing.assert_allclose(loaded.data, expected, rtol=1e-9, atol=1e-9)
        self.assertEqual(loaded.data.dtype, np.float64)  # real-valued rescale keeps double precision
        self.assertTrue(loaded.acquisition["rescale"]["varies_between_slices"])

    def test_unit_rescale_keeps_integers_exact(self):
        synth.write_series(self.directory, self.volume, self.affine)
        loaded = dicom_io.load_volume(self.directory)
        self.assertEqual(loaded.data.dtype, np.float32)
        np.testing.assert_array_equal(loaded.data, np.rint(loaded.data))
        self.assertFalse(loaded.acquisition["rescale"]["varies_between_slices"])

    def test_monochrome1_is_inverted(self):
        synth.write_series(self.directory, self.volume, self.affine, photometric="MONOCHROME1")
        loaded = dicom_io.load_volume(self.directory)
        inverted = float(self.volume.max()) + float(self.volume.min()) - self.volume.astype(np.float64)
        np.testing.assert_array_equal(loaded.data, reorient_to_ras(inverted, self.affine)[0])
        self.assertTrue(any("MONOCHROME1" in note for note in loaded.warnings))

    def test_rle_compressed_series(self):
        synth.write_series(self.directory, self.volume, self.affine, transfer_syntax="rle")
        np.testing.assert_array_equal(
            dicom_io.load_volume(self.directory).data, reorient_to_ras(self.volume.astype(np.float32), self.affine)[0]
        )

    def test_real_scanner_file_decodes_identically_in_every_transfer_syntax(self):
        reference_path = bundled("MR_small.dcm")
        if reference_path is None:
            self.skipTest("pydicom test files are not bundled in this installation")
        reference = dicom_io.load_volume(reference_path)
        self.assertEqual(reference.shape, (64, 64, 1))
        self.assertEqual(reference.acquisition["modality"], "MR")
        np.testing.assert_array_equal(
            np.sort(reference.data.ravel()), np.sort(pydicom.dcmread(reference_path).pixel_array.astype(np.float32).ravel())
        )
        checked = 0
        for name in ("MR_small_implicit.dcm", "MR_small_bigendian.dcm", "MR_small_expb.dcm", "MR_small_padded.dcm",
                     "MR_small_RLE.dcm", "MR_small_jp2klossless.dcm", "MR_small_jpeg_ls_lossless.dcm"):
            path = bundled(name)
            if path is None:
                continue
            with self.subTest(file=name):
                try:
                    loaded = dicom_io.load_volume(path)
                except dicom_io.DicomDependencyError as error:
                    self.assertIn("pylibjpeg", str(error))  # actionable message when a codec is absent
                    continue
                np.testing.assert_array_equal(loaded.data, reference.data)
                np.testing.assert_allclose(loaded.affine, reference.affine)
                checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_truncated_pixel_data_is_reported(self):
        path = bundled("MR_truncated.dcm")
        if path is None:
            self.skipTest("pydicom test files are not bundled in this installation")
        with self.assertRaisesRegex(dicom_io.DicomError, "unreadable or truncated"):
            dicom_io.load_volume(path)


@unittest.skipIf(pydicom is None, "pydicom is not installed")
class DicomDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="dicom_io_test_"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.affine = synth.oblique_affine()

    def write_study(self):
        study = synth.generate_uid()
        synth.write_series(self.directory / "t2", synth.phantom((12, 10, 30)), self.affine, study_uid=study, tags=synth.T2_TSE, series_number=3)
        synth.write_series(self.directory / "t1", synth.phantom((12, 10, 64)), self.affine, study_uid=study, series_number=5)
        synth.write_series(
            self.directory / "loc", synth.phantom((12, 10, 3)), self.affine, study_uid=study, series_number=1,
            tags={"SeriesDescription": "localizer", "ProtocolName": "localizer"},
        )
        (self.directory / "notes.txt").write_text("not dicom")
        (self.directory / "junk.bin").write_bytes(b"\x00" * 300)
        (self.directory / "tiny").write_bytes(b"x")

    def test_structural_t1_is_ranked_first_and_selected(self):
        self.write_study()
        series = dicom_io.discover_series(self.directory)
        self.assertEqual(series[0].contrast.label, "T1w")
        self.assertEqual(sorted(entry.contrast.label for entry in series[1:]), ["T2w", "localizer"])
        self.assertEqual(series[0].number, 5)
        self.assertEqual(series[0].contrast.confidence, "high")
        self.assertTrue(all(series[0].t1_structural_score() > other.t1_structural_score() for other in series[1:]))
        volume = dicom_io.load_volume(self.directory)
        self.assertEqual(volume.shape[2] * volume.shape[1] * volume.shape[0], 12 * 10 * 64)
        note = next(note for note in volume.warnings if "3 series found" in note)
        # A fact for whoever reads the result; how to choose another series is for the caller to say.
        self.assertRegex(note, r"^3 series found; loaded series 5: [^.]*\.$")
        t2 = next(entry for entry in series if entry.contrast.label == "T2w")
        chosen = dicom_io.load_volume(self.directory, series_key=t2.key)
        self.assertEqual(chosen.acquisition["contrast_guess"]["label"], "T2w")
        self.assertEqual(chosen.shape[0] * chosen.shape[1] * chosen.shape[2], 12 * 10 * 30)
        with self.assertRaises(dicom_io.NotDicomError):
            dicom_io.select_series(series, key="zzzz")

    def test_three_plane_localizer_is_split_by_orientation(self):
        series_uid, study_uid = synth.generate_uid(), synth.generate_uid()
        volume = synth.phantom((10, 10, 4))
        for number, permutation in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
            affine = np.eye(4)
            affine[:, :3] = np.eye(4)[:, list(permutation)]
            synth.write_series(self.directory, volume, affine, series_uid=series_uid, study_uid=study_uid, filename_prefix=f"P{number}_")
        series = dicom_io.discover_series(self.directory)
        self.assertEqual(len(series), 3)
        self.assertEqual({entry.n_slices for entry in series}, {4})
        self.assertEqual(len({entry.key for entry in series}), 3)

    def test_non_dicom_inputs_are_rejected(self):
        (self.directory / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
        (self.directory / "random.dat").write_bytes(np.random.default_rng(0).bytes(4096))
        # Starts like a bare data set (group 0008) but is not one: must be skipped, not crash the scan.
        (self.directory / "lookalike").write_bytes(b"\x08\x00\x08\x00" + np.random.default_rng(1).bytes(2048))
        self.assertEqual(dicom_io.discover_series(self.directory), [])
        with self.assertRaises(dicom_io.NotDicomError):
            dicom_io.load_volume(self.directory)
        with self.assertRaises(dicom_io.NotDicomError):
            dicom_io.load_volume(self.directory / "does_not_exist")
        self.assertFalse(dicom_io.is_dicom_file(self.directory / "image.png"))

    def test_unsupported_objects_fail_with_a_reason(self):
        colour = bundled("SC_rgb_small_odd.dcm")
        cine = bundled("rtdose.dcm")
        report = bundled("reportsi.dcm")
        if colour is None or cine is None or report is None:
            self.skipTest("pydicom test files are not bundled in this installation")
        with self.assertRaisesRegex(dicom_io.UnsupportedDicomError, "not a grayscale MRI"):
            dicom_io.load_volume(colour)
        with self.assertRaisesRegex(dicom_io.UnsupportedDicomError, "not a spatial MRI volume"):
            dicom_io.load_volume(cine)
        with self.assertRaises(dicom_io.NotDicomError):
            dicom_io.load_volume(report)  # a structured report has no image

    def test_mosaic_is_refused(self):
        files = synth.write_series(self.directory, synth.phantom((8, 8, 4)), self.affine)
        for path in files:
            dataset = pydicom.dcmread(path)
            dataset.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND", "MOSAIC"]
            dataset.save_as(path)
        with self.assertRaisesRegex(dicom_io.UnsupportedDicomError, "mosaic"):
            dicom_io.load_volume(self.directory)

    def test_non_mr_modality_is_flagged(self):
        path = bundled("CT_small.dcm")
        if path is None:
            self.skipTest("pydicom test files are not bundled in this installation")
        volume = dicom_io.load_volume(path)
        self.assertEqual(volume.acquisition["contrast_guess"]["label"], "not_mr")
        self.assertTrue(any("not MR" in note for note in volume.warnings))

    def test_contrast_heuristics(self):
        cases = [
            # Siemens MPRAGE: long TR is the inversion repetition
            ({"modality": "MR", "repetition_time_ms": 2300, "echo_time_ms": 2.98, "inversion_time_ms": 900, "flip_angle_deg": 9,
              "series_description": "t1_mprage_sag"}, "T1w", "high"),
            # GE BRAVO / Philips TFE: TR is the echo spacing; no helpful text
            ({"modality": "MR", "repetition_time_ms": 8.2, "echo_time_ms": 3.2, "inversion_time_ms": 450, "flip_angle_deg": 12}, "T1w", "medium"),
            ({"modality": "MR", "repetition_time_ms": 5000, "echo_time_ms": 95, "series_description": "t2_tse_tra"}, "T2w", "high"),
            ({"modality": "MR", "repetition_time_ms": 9000, "echo_time_ms": 120, "inversion_time_ms": 2500, "series_description": "t2_flair"}, "FLAIR", "high"),
            ({"modality": "MR", "series_description": "ep2d_diff_mddw_20"}, "other", "medium"),
            ({"modality": "MR", "series_description": "AAHead_Scout"}, "localizer", "medium"),
            ({"modality": "MR"}, "unknown", "low"),
            ({"modality": "CT", "series_description": "t1"}, "not_mr", "high"),
            # text says T1 but timing says T2: report the physics, with low confidence
            ({"modality": "MR", "repetition_time_ms": 5000, "echo_time_ms": 95, "series_description": "T1 mislabelled"}, "T2w", "low"),
        ]
        for acquisition, label, confidence in cases:
            with self.subTest(acquisition=acquisition):
                guess = dicom_io.guess_contrast(acquisition)
                self.assertEqual((guess.label, guess.confidence), (label, confidence))
                self.assertTrue(guess.reasons)


@unittest.skipIf(pydicom is None, "pydicom is not installed")
class DicomPrivacyAndExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="dicom_io_test_"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.volume = synth.phantom((16, 18, 20))
        self.affine = synth.oblique_affine()
        synth.write_series(self.directory / "series", self.volume, self.affine)
        self.loaded = dicom_io.load_volume(self.directory / "series")

    def assert_no_phi(self, text):
        for needle in FAKE_PHI:
            self.assertNotIn(needle, text)

    def test_identity_is_available_but_never_printed(self):
        identity = self.loaded.identity
        self.assertIsInstance(identity, PatientIdentity)
        self.assertEqual(identity.name, "SYNTHETIC^PHANTOM")
        self.assertEqual(identity.patient_id, "FAKE-ID-0042")
        self.assertEqual(identity.birth_date, "19400229")
        self.assertEqual(identity.study_date, "20260102")
        self.assert_no_phi(repr(identity))
        self.assert_no_phi(repr(self.loaded.identity) + str(self.loaded.identity))
        self.assert_no_phi(json.dumps(self.loaded.describe()))
        series = dicom_io.discover_series(self.directory / "series")[0]
        self.assert_no_phi(repr(series) + series.label + series.technical_label)

    def test_acquisition_metadata_is_technical_only(self):
        acquisition = self.loaded.acquisition
        self.assertEqual(acquisition["field_strength_t"], 3.0)
        self.assertEqual(acquisition["mr_acquisition_type"], "3D")
        self.assertEqual(acquisition["contrast_guess"]["label"], "T1w")
        self.assertNotIn("patient", " ".join(acquisition).replace("patient_position", ""))
        public = self.loaded.describe(include_free_text=False)["acquisition"]
        self.assertNotIn("series_description", public)
        self.assertNotIn("protocol_name", public)

    @unittest.skipIf(nib is None, "nibabel is not installed")
    def test_nifti_export_is_deterministic_private_and_round3_hash_compatible(self):
        first = self.loaded.to_nifti(self.directory / "out" / "scan.nii.gz")
        second = self.loaded.to_nifti(self.directory / "out2" / "scan.nii.gz")
        self.assertEqual(first["volume_sha256"], second["volume_sha256"])
        self.assertEqual(first["orientation"], ["R", "A", "S"])

        # research/dementia_round3/data.py recomputes this hash from the file like so:
        image = nib.as_closest_canonical(nib.load(first["path"]))
        data = np.asarray(image.get_fdata(dtype=np.float32), dtype="<f4")
        import hashlib

        recomputed = hashlib.sha256(data.tobytes(order="C") + np.asarray(image.affine, dtype="<f8").tobytes()).hexdigest()
        self.assertEqual(first["canonical_volume_sha256"], recomputed)
        np.testing.assert_array_equal(data, self.loaded.data)
        np.testing.assert_allclose(image.affine, self.loaded.affine, atol=1e-4)
        qform, qform_code = image.get_qform(coded=True)
        sform, sform_code = image.get_sform(coded=True)
        self.assertEqual((int(qform_code), int(sform_code)), (1, 1))
        np.testing.assert_allclose(qform, sform, atol=1e-3)

        sidecar = Path(first["sidecar"]).read_text()
        self.assert_no_phi(sidecar)
        self.assertNotIn("t1_mprage_sag_p2_iso", sidecar)  # free text is dropped by default
        self.assert_no_phi(nib.load(first["path"]).header.binaryblock.decode("latin-1"))
        kept = self.loaded.to_nifti(self.directory / "out3" / "scan.nii", include_free_text=True)
        self.assertIn("t1_mprage_sag_p2_iso", Path(kept["sidecar"]).read_text())

    @unittest.skipIf(nib is None, "nibabel is not installed")
    def test_command_line_lists_and_converts(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(dicom_io.main([str(self.directory / "series")]), 0)
        self.assertIn("MR T1w, 20 slices", buffer.getvalue())
        self.assert_no_phi(buffer.getvalue())

        target = self.directory / "cli" / "scan.nii.gz"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(dicom_io.main([str(self.directory / "series"), "--to-nifti", str(target)]), 0)
        record = json.loads(buffer.getvalue())
        self.assertTrue(target.is_file())
        self.assertEqual(record["shape"], list(self.loaded.shape))
        with self.assertLogs("src.dicom_io", level="ERROR"):
            self.assertEqual(dicom_io.main([str(self.directory / "nothing_here")]), 2)


class VolumeContainerTests(unittest.TestCase):
    """Pure numpy contracts; run even without pydicom."""

    def test_volume_validation(self):
        from src.mri_volume import MRIVolume, VolumeError

        good = MRIVolume(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
        self.assertEqual(good.shape, (2, 3, 4))
        self.assertEqual(good.orientation, ("R", "A", "S"))
        self.assertEqual(good.obliquity_degrees(), 0.0)
        for data, affine in [
            (np.zeros((2, 3)), np.eye(4)),
            (np.zeros((2, 3, 4)), np.eye(3)),
            (np.full((2, 3, 4), np.nan), np.eye(4)),
            (np.zeros((2, 3, 4)), np.zeros((4, 4))),
            (np.zeros((2, 3, 4)), np.diag([1.0, 1.0, 0.0, 1.0])),
        ]:
            with self.assertRaises(VolumeError):
                MRIVolume(data, affine)

    def test_reorientation_preserves_world_coordinates(self):
        from src.mri_volume import axis_codes, reorient_to_ras

        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 6, 7)).astype(np.float32)
        affine = np.array([[0.0, 0.0, -1.5, 10.0], [-2.0, 0.0, 0.0, 20.0], [0.0, 3.0, 0.0, 30.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(axis_codes(affine), ("P", "S", "L"))
        canonical, canonical_affine = reorient_to_ras(data, affine)
        self.assertEqual(axis_codes(canonical_affine), ("R", "A", "S"))
        for index in [(0, 0, 0), (4, 5, 6), (2, 1, 3)]:
            world = affine @ [*index, 1]
            voxel = np.rint(np.linalg.inv(canonical_affine) @ world).astype(int)[:3]
            self.assertEqual(float(canonical[tuple(voxel)]), float(data[index]))

    def test_precision_policy(self):
        from src.mri_volume import as_float_voxels

        self.assertEqual(as_float_voxels(np.arange(8, dtype=np.int16)).dtype, np.float32)
        self.assertEqual(as_float_voxels(np.array([2**24 + 1], dtype=np.int32)).dtype, np.float64)
        self.assertEqual(as_float_voxels(np.zeros(3, dtype=np.float64)).dtype, np.float64)
        self.assertEqual(as_float_voxels(np.zeros(3, dtype=np.float32)).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
