"""Write standards-conformant synthetic DICOM MR series for tests and validation.

The writer is the inverse of ``src.dicom_io``: it takes a voxel array plus a
RAS+ affine and emits one file per slice along the third voxel axis (or one
Enhanced multi-frame file). Because the acquisition plane, slice direction,
file order, rescale and photometric interpretation are all parameters, a test
can demand that every variant reads back to the *same* canonical volume.

Patient fields are filled with obviously fake values so PHI-separation tests
have something to look for. No real data is created here.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, RLELossless, generate_uid

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
ENHANCED_MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4.1"

FAKE_PATIENT = {
    "PatientName": "SYNTHETIC^PHANTOM",
    "PatientID": "FAKE-ID-0042",
    "PatientBirthDate": "19400229",
    "PatientSex": "O",
    "PatientAge": "086Y",
    "AccessionNumber": "FAKE-ACC-7",
    "InstitutionName": "Nowhere General Test Hospital",
    "StationName": "FAKESCANNER1",
    "DeviceSerialNumber": "SN-000-FAKE",
    "StudyDescription": "Synthetic study for SYNTHETIC^PHANTOM",
}

T1_MPRAGE = {
    "SeriesDescription": "t1_mprage_sag_p2_iso",
    "ProtocolName": "t1_mprage_sag_p2_iso",
    "SequenceName": "*tfl3d1_16ns",
    "ScanningSequence": ["GR", "IR"],
    "SequenceVariant": ["SK", "SP", "MP"],
    "MRAcquisitionType": "3D",
    "RepetitionTime": 2300.0,
    "EchoTime": 2.98,
    "InversionTime": 900.0,
    "FlipAngle": 9.0,
    "MagneticFieldStrength": 3.0,
    "Manufacturer": "SyntheticVendor",
    "ManufacturerModelName": "Phantom 3T",
}

T2_TSE = {
    "SeriesDescription": "t2_tse_tra",
    "ProtocolName": "t2_tse_tra",
    "SequenceName": "*tse2d1_15",
    "ScanningSequence": "SE",
    "MRAcquisitionType": "2D",
    "RepetitionTime": 5000.0,
    "EchoTime": 95.0,
    "FlipAngle": 150.0,
    "MagneticFieldStrength": 3.0,
    "Manufacturer": "SyntheticVendor",
    "ManufacturerModelName": "Phantom 3T",
}


def lps_affine(affine_ras: np.ndarray) -> np.ndarray:
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(affine_ras, dtype=np.float64)


def _base_dataset(sop_class: str, sop_uid: str, syntax, study_uid: str, series_uid: str, tags: dict) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = syntax
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.StudyDate = "20260102"
    dataset.StudyTime = "101500"
    dataset.SeriesDate = "20260102"
    dataset.StudyID = "1"
    dataset.PatientPosition = "HFS"
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    for keyword, value in {**FAKE_PATIENT, **tags}.items():
        setattr(dataset, keyword, value)
    return dataset


def _pixel_module(dataset: Dataset, rows: int, columns: int, dtype: np.dtype, photometric: str) -> None:
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.Rows, dataset.Columns = rows, columns
    dataset.BitsAllocated = dtype.itemsize * 8
    dataset.BitsStored = dtype.itemsize * 8
    dataset.HighBit = dtype.itemsize * 8 - 1
    dataset.PixelRepresentation = 1 if np.issubdtype(dtype, np.signedinteger) else 0


def _format(values: Sequence[float]) -> list:
    # DICOM DS is at most 16 characters; keep as much precision as fits.
    return [f"{float(value):.10g}"[:16] for value in values]


def write_series(
    directory,
    stored: np.ndarray,
    affine_ras: np.ndarray,
    *,
    tags: Optional[dict] = None,
    series_number: int = 2,
    study_uid: Optional[str] = None,
    series_uid: Optional[str] = None,
    slopes: Optional[Sequence[float]] = None,
    intercepts: Optional[Sequence[float]] = None,
    photometric: str = "MONOCHROME2",
    transfer_syntax: str = "explicit",
    shuffle: Optional[int] = 0,
    reverse_instance_numbers: bool = False,
    skip_slices: Sequence[int] = (),
    extension: str = ".dcm",
    echo_time: Optional[float] = None,
    filename_prefix: str = "IM",
    bare: bool = False,
) -> list:
    """Write ``stored[i, j, k]`` as a classic single-frame series, one file per ``k``.

    ``stored`` holds the integer values to put in PixelData; the value a reader
    must recover for slice ``k`` is ``stored * slopes[k] + intercepts[k]``.
    File names are shuffled (seeded) so nothing can rely on name order.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stored = np.asarray(stored)
    if stored.ndim != 3 or not np.issubdtype(stored.dtype, np.integer):
        raise ValueError("stored must be a 3D integer array")
    affine = lps_affine(affine_ras)
    column_step, row_step, slice_step, origin = affine[:3, 0], affine[:3, 1], affine[:3, 2], affine[:3, 3]
    column_spacing, row_spacing = np.linalg.norm(column_step), np.linalg.norm(row_step)
    orientation = np.concatenate([column_step / column_spacing, row_step / row_spacing])
    study_uid = study_uid or generate_uid()
    series_uid = series_uid or generate_uid()
    syntax = {"explicit": ExplicitVRLittleEndian, "implicit": ImplicitVRLittleEndian, "rle": RLELossless}[
        transfer_syntax
    ]
    count = stored.shape[2]
    names = [f"{filename_prefix}{index:05d}{extension}" for index in range(count)]
    if shuffle is not None:
        random.Random(shuffle).shuffle(names)

    written = []
    for k in range(count):
        if k in skip_slices:
            continue
        sop_uid = generate_uid()
        base_syntax = ExplicitVRLittleEndian if transfer_syntax == "rle" else syntax
        dataset = _base_dataset(MR_IMAGE_STORAGE, sop_uid, base_syntax, study_uid, series_uid, {**T1_MPRAGE, **(tags or {})})
        dataset.SeriesNumber = series_number
        dataset.InstanceNumber = (count - k) if reverse_instance_numbers else (k + 1)
        dataset.ImagePositionPatient = _format(origin + k * slice_step)
        dataset.ImageOrientationPatient = _format(orientation)
        dataset.PixelSpacing = _format([row_spacing, column_spacing])
        dataset.SliceThickness = f"{np.linalg.norm(slice_step):.6g}"
        dataset.SliceLocation = f"{float((origin + k * slice_step) @ np.cross(orientation[:3], orientation[3:])):.6f}"
        if echo_time is not None:
            dataset.EchoTime = echo_time
        if slopes is not None:
            dataset.RescaleSlope = f"{float(slopes[k]):.10g}"
            dataset.RescaleIntercept = f"{float(intercepts[k] if intercepts is not None else 0.0):.10g}"
            dataset.RescaleType = "US"
        plane = np.ascontiguousarray(stored[:, :, k].T)  # pixel_array is [row, column]
        _pixel_module(dataset, plane.shape[0], plane.shape[1], plane.dtype, photometric)
        dataset.PixelData = plane.tobytes()
        if transfer_syntax == "rle":
            dataset.compress(RLELossless, plane)
        path = directory / names[k]
        if bare:  # legacy export: the data set alone, no 128-byte preamble or file meta group
            with path.open("wb") as handle:
                pydicom.dcmwrite(handle, Dataset(dataset), implicit_vr=syntax == ImplicitVRLittleEndian, little_endian=True,
                                 enforce_file_format=False)
        else:
            dataset.save_as(path, enforce_file_format=True)
        written.append(path)
    return written


def write_enhanced(
    path,
    stored: np.ndarray,
    affine_ras: np.ndarray,
    *,
    tags: Optional[dict] = None,
    slopes: Optional[Sequence[float]] = None,
    intercepts: Optional[Sequence[float]] = None,
    frame_order: Optional[Sequence[int]] = None,
    per_frame_orientation: bool = False,
) -> Path:
    """Write the same volume as one Enhanced MR multi-frame object.

    Frames are stored in ``frame_order`` (default: reversed) so a reader has to
    use the per-frame PlanePositionSequence rather than the frame index.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = np.asarray(stored)
    affine = lps_affine(affine_ras)
    column_step, row_step, slice_step, origin = affine[:3, 0], affine[:3, 1], affine[:3, 2], affine[:3, 3]
    column_spacing, row_spacing = np.linalg.norm(column_step), np.linalg.norm(row_step)
    orientation = np.concatenate([column_step / column_spacing, row_step / row_spacing])
    count = stored.shape[2]
    order = list(frame_order) if frame_order is not None else list(range(count))[::-1]

    acquisition = {**T1_MPRAGE, **(tags or {})}
    timing = {key: acquisition.pop(key) for key in ("RepetitionTime", "FlipAngle") if key in acquisition}
    echo_time = acquisition.pop("EchoTime", None)
    inversion = acquisition.pop("InversionTime", None)
    acquisition.pop("ScanningSequence", None)  # classic-only attributes
    acquisition.pop("SequenceVariant", None)
    acquisition.pop("SequenceName", None)
    dataset = _base_dataset(
        ENHANCED_MR_IMAGE_STORAGE, generate_uid(), ExplicitVRLittleEndian, generate_uid(), generate_uid(), acquisition
    )
    dataset.SeriesNumber = 7
    dataset.InstanceNumber = 1
    dataset.PulseSequenceName = "tfl3d1"
    dataset.NumberOfFrames = len(order)

    measures = Dataset()
    measures.PixelSpacing = _format([row_spacing, column_spacing])
    measures.SliceThickness = f"{np.linalg.norm(slice_step):.6g}"
    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = _format(orientation)
    shared = Dataset()
    shared.PixelMeasuresSequence = DicomSequence([measures])
    if not per_frame_orientation:
        shared.PlaneOrientationSequence = DicomSequence([plane_orientation])
    timing_item = Dataset()
    for keyword, value in timing.items():
        setattr(timing_item, keyword, value)
    shared.MRTimingAndRelatedParametersSequence = DicomSequence([timing_item])
    if echo_time is not None:
        echo_item = Dataset()
        echo_item.EffectiveEchoTime = echo_time
        shared.MREchoSequence = DicomSequence([echo_item])
    if inversion is not None:
        modifier = Dataset()
        modifier.InversionRecovery = "YES"
        modifier.InversionTimes = [inversion]
        shared.MRModifierSequence = DicomSequence([modifier])
    dataset.SharedFunctionalGroupsSequence = DicomSequence([shared])

    frames, per_frame = [], []
    for stack_position, k in enumerate(order):
        item = Dataset()
        position = Dataset()
        position.ImagePositionPatient = _format(origin + k * slice_step)
        item.PlanePositionSequence = DicomSequence([position])
        if per_frame_orientation:
            item.PlaneOrientationSequence = DicomSequence([plane_orientation])
        content = Dataset()
        content.StackID = "1"
        content.InStackPositionNumber = stack_position + 1
        item.FrameContentSequence = DicomSequence([content])
        if slopes is not None:
            transformation = Dataset()
            transformation.RescaleSlope = f"{float(slopes[k]):.10g}"
            transformation.RescaleIntercept = f"{float(intercepts[k] if intercepts is not None else 0.0):.10g}"
            transformation.RescaleType = "US"
            item.PixelValueTransformationSequence = DicomSequence([transformation])
        per_frame.append(item)
        frames.append(np.ascontiguousarray(stored[:, :, k].T))
    dataset.PerFrameFunctionalGroupsSequence = DicomSequence(per_frame)
    pixels = np.stack(frames)
    _pixel_module(dataset, pixels.shape[1], pixels.shape[2], pixels.dtype, "MONOCHROME2")
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


def phantom(shape=(24, 30, 18), seed: int = 0, dtype=np.uint16) -> np.ndarray:
    """Asymmetric integer volume: every axis and direction is distinguishable."""

    rng = np.random.default_rng(seed)
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    volume = 50 + 7 * i + 13 * j + 29 * k + rng.integers(0, 40, size=shape)
    # A bright off-centre blob breaks every mirror symmetry.
    blob = ((i - shape[0] * 0.7) ** 2 + (j - shape[1] * 0.3) ** 2 + (k - shape[2] * 0.6) ** 2) < (min(shape) / 4) ** 2
    volume = volume + 900 * blob
    return volume.astype(dtype)


def oblique_affine(spacing=(0.9, 1.1, 1.3), angles_deg=(8.0, -5.0, 12.0), origin=(-90.0, 110.0, -70.0)) -> np.ndarray:
    """RAS affine with anisotropic voxels and a rotation about all three axes."""

    ax, ay, az = np.radians(angles_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = (rz @ ry @ rx) * np.asarray(spacing)
    affine[:3, 3] = origin
    return affine


def reoriented(volume: np.ndarray, affine: np.ndarray, permutation: Sequence[int], flips: Sequence[bool]):
    """Same anatomy in another voxel order: permute then flip axes, keeping world coordinates."""

    data = np.transpose(volume, permutation)
    new_affine = np.asarray(affine, dtype=np.float64).copy()
    new_affine[:, :3] = new_affine[:, list(permutation)]
    for axis, flip in enumerate(flips):
        if flip:
            data = np.flip(data, axis=axis)
            new_affine[:3, 3] += new_affine[:3, axis] * (data.shape[axis] - 1)
            new_affine[:3, axis] *= -1
    return np.ascontiguousarray(data), new_affine
