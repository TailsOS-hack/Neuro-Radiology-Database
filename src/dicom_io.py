"""DICOM MRI ingestion: files, folders or zip archives -> canonical 3D volumes.

What this module guarantees for every volume it returns:

* **Geometry comes from the header, not from file names or instance numbers.**
  Slices are ordered by their position along the slice normal
  (ImagePositionPatient . (row x column)), the inter-slice step is measured
  from those positions (never from SliceThickness), and the voxel-to-world
  affine is built from ImageOrientationPatient / PixelSpacing and converted
  from DICOM's LPS+ to the RAS+ convention used everywhere else here.
* **Intensities are the modality-LUT output.** RescaleSlope/Intercept are
  applied per slice (they can differ slice to slice), MONOCHROME1 is inverted.
* **It fails loudly instead of guessing.** Missing slices, irregular spacing,
  mixed orientations, colour secondary captures, mosaics and undecodable
  transfer syntaxes raise a :class:`DicomError` that says what is wrong.
* **PHI stays out of everything that is written or logged.** Identifying
  header fields live in ``MRIVolume.identity`` only.

Classic single-frame series and Enhanced (multi-frame) MR are both supported.
Compressed transfer syntaxes need the optional pylibjpeg plugins listed in
``requirements.txt``.

CLI::

    python -m src.dicom_io <file|folder|zip>                 # list series
    python -m src.dicom_io <path> --to-nifti scan.nii.gz     # convert the best T1 series
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import numpy as np

try:
    from src.mri_volume import MRIVolume, PatientIdentity, PathLike, VolumeError
except ModuleNotFoundError:  # executed from inside src/
    from mri_volume import MRIVolume, PatientIdentity, PathLike, VolumeError

logger = logging.getLogger(__name__)

try:  # pydicom is required for DICOM input but not for the rest of the app
    import pydicom
    from pydicom.errors import InvalidDicomError
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    pydicom = None

    class InvalidDicomError(Exception):  # type: ignore[no-redef]
        pass


# Extensions that are never DICOM; skipped without opening during folder scans.
_NON_DICOM_SUFFIXES = {
    ".json", ".txt", ".csv", ".tsv", ".xml", ".html", ".pdf", ".md", ".log", ".ini", ".cfg",
    ".nii", ".gz", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".bval", ".bvec",
    ".py", ".pt", ".pth", ".npz", ".npy", ".zip", ".tar", ".7z", ".rar", ".exe", ".dll", ".ds_store",
}
_MEDIA_DIRECTORY_SOP_CLASS = "1.2.840.10008.1.3.10"  # DICOMDIR index, not an image

#: A data set written without the Part 10 preamble starts with group 0008 (or 0002), little endian.
_BARE_DATASET_STARTS = (b"\x08\x00", b"\x02\x00")
#: Folder scans stop here; a study has thousands of files, a home directory has millions.
_MAX_FILES_SCANNED = 50_000

#: Same physical slice position if within this many mm along the slice normal.
_SAME_POSITION_MM = 0.01
#: Direction cosines must agree to this tolerance to belong to one stack.
_ORIENTATION_TOLERANCE = 1e-3


class DicomError(VolumeError):
    """Base class for every DICOM ingestion failure."""


class DicomDependencyError(DicomError):
    """pydicom or a pixel-data decoder plugin is not installed."""


class NotDicomError(DicomError):
    """The path holds no readable DICOM image."""


class DicomGeometryError(DicomError):
    """The slices cannot be assembled into one regularly spaced volume."""


class UnsupportedDicomError(DicomError):
    """A DICOM object this pipeline deliberately does not interpret."""


def _require_pydicom() -> None:
    if pydicom is None:
        raise DicomDependencyError("DICOM support needs pydicom (pip install pydicom)")


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class _Source:
    """One DICOM file on disk or inside a zip archive."""

    path: Path
    member: Optional[str] = None

    def read_bytes(self) -> bytes:
        if self.member is None:
            return self.path.read_bytes()
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(self.member)

    def open(self) -> io.BufferedIOBase:
        if self.member is None:
            return self.path.open("rb")
        return io.BytesIO(self.read_bytes())

    @property
    def label(self) -> str:
        return str(self.path) if self.member is None else f"{self.path}!{self.member}"


def _iter_sources(path: Path) -> Iterator[_Source]:
    if path.is_dir():
        for root, directories, files in os.walk(path):
            directories.sort()
            for name in sorted(files):
                yield _Source(Path(root) / name)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if not info.is_dir():
                    yield _Source(path, info.filename)
    elif path.is_file():
        yield _Source(path)
    else:
        raise NotDicomError(f"No such file or folder: {path}")


def _has_dicom_magic(handle: io.BufferedIOBase) -> bool:
    handle.seek(128)
    return handle.read(4) == b"DICM"


def is_dicom_file(path: PathLike) -> bool:
    """True if ``path`` is a DICOM Part 10 file (or a preamble-less DICOM image)."""

    path = Path(path)
    if not path.is_file() or pydicom is None:
        return False
    return _read_header(_Source(path)) is not None


def _read_header(source: _Source):
    """Header-only dataset, or None when the source is not a DICOM image object."""

    suffix = Path(source.member or source.path.name).suffix.lower()
    if suffix in _NON_DICOM_SUFFIXES:
        return None
    try:
        with source.open() as handle:
            has_magic = _has_dicom_magic(handle)
            handle.seek(0)
            if not has_magic and handle.read(2) not in _BARE_DATASET_STARTS:
                return None  # not Part 10 and does not begin with a DICOM group: never force-parse it
            handle.seek(0)
            dataset = pydicom.dcmread(handle, stop_before_pixels=True, force=not has_magic)
    except (InvalidDicomError, OSError, EOFError, ValueError, KeyError, AttributeError, zipfile.BadZipFile):
        return None
    except Exception:  # corrupt or foreign files must not abort a folder scan
        logger.debug("Skipping unreadable file during DICOM scan", exc_info=True)
        return None
    if str(dataset.get("SOPClassUID", "")) == _MEDIA_DIRECTORY_SOP_CLASS:
        return None
    if "Rows" not in dataset or "Columns" not in dataset:
        return None  # reports, structured data, presentation states, raw data...
    if not has_magic and "Modality" not in dataset and "SOPClassUID" not in dataset:
        return None  # arbitrary bytes that happened to parse
    return dataset


# --------------------------------------------------------------------------- header access


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "\\".join(_text(item) for item in value)
    return str(value).strip()


def _number(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
            value = value[0]
        result = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    return result if np.isfinite(result) else None


def _numbers(value: Any, count: int) -> Optional[tuple]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != count or not np.isfinite(result).all():
        return None
    return result


def _functional_groups(dataset, frame: Optional[int]) -> list:
    """Per-frame group first, then the shared group (Enhanced multi-frame objects)."""

    groups = []
    per_frame = dataset.get("PerFrameFunctionalGroupsSequence")
    if per_frame is not None and frame is not None and frame < len(per_frame):
        groups.append(per_frame[frame])
    shared = dataset.get("SharedFunctionalGroupsSequence")
    if shared is not None and len(shared):
        groups.append(shared[0])
    return groups


def _lookup(dataset, keyword: str, frame: Optional[int] = None, macro: Optional[str] = None) -> Any:
    """Find ``keyword`` at top level or inside the functional-group macro ``macro``."""

    if macro is None and keyword in dataset:
        return dataset.get(keyword)
    for group in _functional_groups(dataset, 0 if frame is None else frame):
        sequences = [group.get(macro)] if macro else [element.value for element in group if element.VR == "SQ"]
        for sequence in sequences:
            if sequence is not None and len(sequence) and keyword in sequence[0]:
                return sequence[0].get(keyword)
    return dataset.get(keyword) if keyword in dataset else None


_ACQUISITION_TEXT = {
    "modality": "Modality",
    "manufacturer": "Manufacturer",
    "model": "ManufacturerModelName",
    "software_versions": "SoftwareVersions",
    "mr_acquisition_type": "MRAcquisitionType",
    "scanning_sequence": "ScanningSequence",
    "sequence_variant": "SequenceVariant",
    "scan_options": "ScanOptions",
    "sequence_name": "SequenceName",
    "pulse_sequence_name": "PulseSequenceName",
    "image_type": "ImageType",
    "body_part": "BodyPartExamined",
    "patient_position": "PatientPosition",
    "contrast_agent": "ContrastBolusAgent",
    "series_description": "SeriesDescription",
    "protocol_name": "ProtocolName",
}
_ACQUISITION_NUMBERS = {
    "field_strength_t": "MagneticFieldStrength",
    "repetition_time_ms": "RepetitionTime",
    "echo_time_ms": "EchoTime",
    "inversion_time_ms": "InversionTime",
    "flip_angle_deg": "FlipAngle",
    "echo_train_length": "EchoTrainLength",
    "pixel_bandwidth_hz": "PixelBandwidth",
    "slice_thickness_mm": "SliceThickness",
    "spacing_between_slices_mm": "SpacingBetweenSlices",
    "bits_stored": "BitsStored",
    "series_number": "SeriesNumber",
}


def _acquisition_metadata(dataset) -> dict:
    """Technical, non-identifying acquisition parameters (allow-list only)."""

    metadata: dict = {}
    for name, keyword in _ACQUISITION_TEXT.items():
        value = _text(_lookup(dataset, keyword))
        if value:
            metadata[name] = value
    for name, keyword in _ACQUISITION_NUMBERS.items():
        value = _number(_lookup(dataset, keyword))
        if value is not None:
            metadata[name] = value
    # Enhanced MR spells two of the timing parameters differently.
    if "echo_time_ms" not in metadata:
        value = _number(_lookup(dataset, "EffectiveEchoTime"))
        if value is not None:
            metadata["echo_time_ms"] = value
    if "inversion_time_ms" not in metadata:
        value = _number(_lookup(dataset, "InversionTimes"))
        if value is not None:
            metadata["inversion_time_ms"] = value
    transfer_syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
    if transfer_syntax is not None:
        metadata["transfer_syntax"] = getattr(transfer_syntax, "name", str(transfer_syntax))
    return metadata


def _identity(dataset) -> PatientIdentity:
    get = lambda keyword: _text(dataset.get(keyword))  # noqa: E731
    return PatientIdentity(
        name=get("PatientName"),
        patient_id=get("PatientID"),
        birth_date=get("PatientBirthDate"),
        sex=get("PatientSex"),
        age=get("PatientAge"),
        study_date=get("StudyDate") or get("SeriesDate") or get("AcquisitionDate"),
        accession_number=get("AccessionNumber"),
        study_description=get("StudyDescription"),
        study_uid=get("StudyInstanceUID"),
        series_uid=get("SeriesInstanceUID"),
        institution=get("InstitutionName"),
        station=get("StationName"),
        device_serial=get("DeviceSerialNumber"),
    )


# --------------------------------------------------------------------------- contrast heuristics


@dataclass(frozen=True)
class ContrastGuess:
    """Best-effort image contrast from header parameters. A hint, not ground truth."""

    label: str  # "T1w", "T2w", "FLAIR", "PDw", "localizer", "other", "unknown", "not_mr"
    confidence: str  # "high", "medium", "low"
    reasons: tuple = ()

    @property
    def is_t1(self) -> bool:
        return self.label == "T1w"


_TEXT_RULES = (
    # (label, substrings) - first match wins, so specific/derived sequences come first
    ("other", ("dwi", "dti", "diff", "adc", "trace", "bold", "fmri", "ep2d", "epi", "asl", "perf", "swi",
               "swan", "t2star", "t2*", "fieldmap", "field_map", "b0map", "b1map", "mrs", "svs", "csi",
               "tof", "angio", "mra", "phase", "cine", "spectro")),
    ("localizer", ("localizer", "localiser", "scout", "survey", "3-plane", "3plane", "3 plane", "calibration",
                   "aahead", "aascout")),
    ("T1w", ("mprage", "mp-rage", "mp_rage", "mp rage", "mp2rage", "bravo", "spgr", "tfl", "t1w", "t1_", "t1-",
             "t1 ", "3dt1", "3d t1", "3d_t1", "mdeft", "fl3d", "vibe", "tfe", "t1")),
    ("FLAIR", ("flair", "dark-fluid", "dark_fluid", "da-fl")),
    ("T2w", ("t2",)),
    ("PDw", ("proton", "pdw", "pd_", "pd ")),
)


def guess_contrast(acquisition: dict) -> ContrastGuess:
    """Classify the series contrast from sequence text and TR/TE/TI/flip angle.

    Vendors report TR differently for inversion-prepared 3D gradient echo (the
    full inversion repetition on Siemens, the echo spacing on GE/Philips), so
    the timing rules accept both conventions.
    """

    modality = str(acquisition.get("modality", "")).upper()
    if modality and modality != "MR":
        return ContrastGuess("not_mr", "high", (f"Modality is {modality}",))

    text = " ".join(
        str(acquisition.get(key, "")) for key in ("series_description", "protocol_name", "sequence_name",
                                                 "pulse_sequence_name")
    ).lower()
    text_label = None
    for label, needles in _TEXT_RULES:
        hit = next((needle for needle in needles if needle in text), None)
        if hit is not None:
            # T1w is listed before FLAIR on purpose: "T1 FLAIR" is a T1-weighted
            # inversion-recovery image, whereas a T2 FLAIR never mentions T1.
            text_label, text_reason = label, f"sequence text contains '{hit.strip()}'"
            break

    repetition = acquisition.get("repetition_time_ms")
    echo = acquisition.get("echo_time_ms")
    inversion = acquisition.get("inversion_time_ms")
    flip = acquisition.get("flip_angle_deg")
    timing_label = None
    if echo is not None and repetition is not None:
        if echo <= 15:
            gradient_echo = repetition <= 60 and (flip is None or flip <= 45)
            prepared = inversion is not None and 300 <= inversion <= 1500 and repetition >= 1000 and (
                flip is None or flip <= 25
            )
            spin_echo_t1 = 250 <= repetition <= 1000 and (flip is None or flip >= 60)
            if gradient_echo or prepared or spin_echo_t1:
                timing_label = "T1w"
                timing_reason = f"short TE ({echo:g} ms) with T1-weighted TR/TI/flip"
        if timing_label is None and echo >= 70 and repetition >= 2000:
            timing_label = "FLAIR" if inversion is not None and inversion >= 1500 else "T2w"
            timing_reason = f"long TE ({echo:g} ms) and long TR ({repetition:g} ms)"
        if timing_label is None and echo <= 40 and repetition >= 1800 and inversion is None:
            timing_label = "PDw"
            timing_reason = f"short TE ({echo:g} ms) with long TR ({repetition:g} ms)"

    if text_label in ("other", "localizer"):
        return ContrastGuess(text_label, "medium", (text_reason,))
    if text_label and timing_label:
        if text_label == timing_label:
            return ContrastGuess(text_label, "high", (text_reason, timing_reason))
        return ContrastGuess(timing_label, "low", (timing_reason, f"but {text_reason}"))
    if timing_label:
        return ContrastGuess(timing_label, "medium", (timing_reason,))
    if text_label:
        return ContrastGuess(text_label, "medium", (text_reason,))
    return ContrastGuess("unknown", "low", ("no sequence text or timing parameters identify the contrast",))


# --------------------------------------------------------------------------- series discovery


@dataclass(frozen=True)
class _Slice:
    """One image plane: a single-frame file, or one frame of a multi-frame file."""

    source: _Source
    sop_uid: str
    frame: Optional[int]
    rows: int
    columns: int
    position: Optional[tuple]
    orientation: Optional[tuple]
    pixel_spacing: Optional[tuple]
    slope: float
    intercept: float
    order_key: tuple  # (temporal position, echo, acquisition number, instance number, frame)


@dataclass
class DicomSeries:
    """A set of image planes that share a series and an image-plane orientation."""

    key: str
    modality: str
    description: str
    number: Optional[int]
    rows: int
    columns: int
    n_slices: int
    is_multiframe: bool
    acquisition: dict
    contrast: ContrastGuess
    identity: PatientIdentity = field(repr=False)
    slices: list = field(repr=False, default_factory=list)

    @property
    def technical_label(self) -> str:
        """One-line summary built from coded fields only; safe to log or export."""

        number = f"series {self.number}" if self.number is not None else f"series {self.key}"
        return (
            f"{number}: {self.modality or '?'} {self.contrast.label}, "
            f"{self.n_slices} slice{'s' if self.n_slices != 1 else ''}, {self.columns}x{self.rows}"
        )

    @property
    def label(self) -> str:
        """Summary including the free-text SeriesDescription, for interactive pickers only."""

        return f"{self.technical_label} - {self.description}" if self.description else self.technical_label

    def t1_structural_score(self) -> float:
        """Ranking score for 'the 3D T1 structural series of this study'."""

        score = {
            ("T1w", "high"): 100, ("T1w", "medium"): 70, ("T1w", "low"): 35, ("unknown", "low"): 10,
        }.get((self.contrast.label, self.contrast.confidence), -100)
        if str(self.acquisition.get("mr_acquisition_type", "")).upper() == "3D":
            score += 20
        score += 20 if self.n_slices >= 100 else 10 if self.n_slices >= 60 else -40 if self.n_slices < 20 else 0
        image_type = str(self.acquisition.get("image_type", "")).upper()
        if "DERIVED" in image_type or "SECONDARY" in image_type or "MPR" in image_type:
            score -= 15  # reformats exist only when the acquired series does too
        if self.acquisition.get("contrast_agent"):
            score -= 10  # post-contrast T1 is a different image from the training domain
        return float(score)


def _slice_records(source: _Source, dataset) -> list:
    """Expand one dataset into its image planes (1 for classic files, N for multi-frame)."""

    rows, columns = int(dataset.Rows), int(dataset.Columns)
    sop_uid = _text(dataset.get("SOPInstanceUID")) or hashlib.sha1(source.label.encode()).hexdigest()
    frames = int(_number(dataset.get("NumberOfFrames")) or 1)
    base_key = (
        _number(dataset.get("TemporalPositionIdentifier")) or 0.0,
        _number(dataset.get("EchoNumbers")) or _number(dataset.get("EchoTime")) or 0.0,
        _number(dataset.get("AcquisitionNumber")) or 0.0,
        _number(dataset.get("InstanceNumber")) or 0.0,
    )
    enhanced = "PerFrameFunctionalGroupsSequence" in dataset or "SharedFunctionalGroupsSequence" in dataset
    if frames > 1 and not enhanced:
        raise UnsupportedDicomError(
            "Multi-frame DICOM without per-frame functional groups (cine/ultrasound-style object) "
            "is not a spatial MRI volume"
        )
    records = []
    for frame in range(frames) if enhanced else [None]:
        if enhanced:
            position = _numbers(_lookup(dataset, "ImagePositionPatient", frame, "PlanePositionSequence"), 3)
            orientation = _numbers(_lookup(dataset, "ImageOrientationPatient", frame, "PlaneOrientationSequence"), 6)
            spacing = _numbers(_lookup(dataset, "PixelSpacing", frame, "PixelMeasuresSequence"), 2)
            slope = _number(_lookup(dataset, "RescaleSlope", frame, "PixelValueTransformationSequence"))
            intercept = _number(_lookup(dataset, "RescaleIntercept", frame, "PixelValueTransformationSequence"))
            temporal = _number(_lookup(dataset, "TemporalPositionIndex", frame, "FrameContentSequence")) or 0.0
            echo = _number(_lookup(dataset, "EffectiveEchoTime", frame, "MREchoSequence")) or 0.0
            order_key = (temporal, echo, base_key[2], base_key[3], float(frame))
        else:
            position = _numbers(dataset.get("ImagePositionPatient"), 3)
            orientation = _numbers(dataset.get("ImageOrientationPatient"), 6)
            spacing = _numbers(dataset.get("PixelSpacing"), 2)
            slope = _number(dataset.get("RescaleSlope"))
            intercept = _number(dataset.get("RescaleIntercept"))
            order_key = (*base_key, 0.0)
        records.append(
            _Slice(
                source=source, sop_uid=sop_uid, frame=frame, rows=rows, columns=columns,
                position=position, orientation=orientation, pixel_spacing=spacing,
                slope=1.0 if slope in (None, 0.0) else slope,
                intercept=0.0 if intercept is None else intercept,
                order_key=order_key,
            )
        )
    return records


def _same_stack(record: _Slice, reference: _Slice) -> bool:
    """Same matrix size and image-plane orientation (within tolerance, no rounding buckets)."""

    if (record.rows, record.columns) != (reference.rows, reference.columns):
        return False
    if record.orientation is None or reference.orientation is None:
        return record.orientation is None and reference.orientation is None
    return max(abs(a - b) for a, b in zip(record.orientation, reference.orientation)) <= _ORIENTATION_TOLERANCE


def discover_series(path: PathLike, *, progress: Optional[Callable[[int], None]] = None) -> list:
    """Scan a file, folder (recursively) or zip and return its image series.

    Headers only are read, so a study with thousands of files is listed in
    seconds. A series holding several image-plane orientations or matrix sizes
    (a three-plane localizer, for instance) is split into one entry per stack.
    Results are sorted with the most likely 3D T1 structural series first.
    """

    _require_pydicom()
    path = Path(path)
    grouped: dict = {}  # series UID -> list of stacks, each {"dataset": first header, "slices": [...]}
    seen_instances: set = set()
    unsupported: list = []
    scanned = 0
    for source in _iter_sources(path):
        scanned += 1
        if scanned > _MAX_FILES_SCANNED:
            raise DicomError(
                f"More than {_MAX_FILES_SCANNED} files under {path}; select the study or series folder instead"
            )
        if progress is not None and scanned % 200 == 0:
            progress(scanned)
        dataset = _read_header(source)
        if dataset is None:
            continue
        try:
            records = _slice_records(source, dataset)
        except UnsupportedDicomError as exc:
            unsupported.append(str(exc))
            continue
        series_uid = _text(dataset.get("SeriesInstanceUID")) or f"{source.path.parent}|{_text(dataset.get('SeriesNumber'))}"
        stacks = grouped.setdefault(series_uid, [])
        for record in records:
            instance = (record.sop_uid, record.frame)
            if instance in seen_instances:
                continue  # the same object copied twice into the folder
            seen_instances.add(instance)
            stack = next((entry for entry in stacks if _same_stack(record, entry["slices"][0])), None)
            if stack is None:
                stack = {"dataset": dataset, "slices": []}
                stacks.append(stack)
            stack["slices"].append(record)

    result = []
    for series_uid, entry, stack_number in ((uid, entry, number) for uid, stacks in grouped.items() for number, entry in enumerate(stacks)):
        dataset, slices = entry["dataset"], entry["slices"]
        acquisition = _acquisition_metadata(dataset)
        number = _number(dataset.get("SeriesNumber"))
        signature = (slices[0].rows, slices[0].columns, stack_number)  # scan order is deterministic
        result.append(
            DicomSeries(
                key=hashlib.sha256(f"{series_uid}|{signature}".encode()).hexdigest()[:12],
                modality=_text(dataset.get("Modality")).upper(),
                description=_text(dataset.get("SeriesDescription")),
                number=None if number is None else int(number),
                rows=slices[0].rows,
                columns=slices[0].columns,
                n_slices=len(slices),
                is_multiframe=slices[0].frame is not None,
                acquisition=acquisition,
                contrast=guess_contrast(acquisition),
                identity=_identity(dataset),
                slices=slices,
            )
        )
    if not result and unsupported:
        raise UnsupportedDicomError(unsupported[0])
    result.sort(key=lambda series: (-series.t1_structural_score(), -series.n_slices, series.number or 0, series.key))
    return result


def select_series(series: Sequence[DicomSeries], *, key: Optional[str] = None) -> DicomSeries:
    """Pick a series by ``key`` or, by default, the best 3D T1 structural candidate."""

    if not series:
        raise NotDicomError("No DICOM image series found")
    if key is not None:
        matches = [candidate for candidate in series if candidate.key.startswith(key)]
        if len(matches) != 1:
            raise NotDicomError(f"Series key '{key}' matches {len(matches)} series")
        return matches[0]
    return max(series, key=lambda candidate: (candidate.t1_structural_score(), candidate.n_slices))


# --------------------------------------------------------------------------- volume assembly


def _decode(dataset, label: str) -> np.ndarray:
    """Stored pixel values of a dataset as an array, with decoder errors made actionable."""

    samples = int(dataset.get("SamplesPerPixel", 1))
    photometric = _text(dataset.get("PhotometricInterpretation")).upper()
    if samples != 1 or photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise UnsupportedDicomError(
            f"{label}: {photometric or 'colour'} image with {samples} samples per pixel is not a grayscale "
            "MRI acquisition (screenshot / secondary capture?)"
        )
    if getattr(dataset, "file_meta", None) is None or "TransferSyntaxUID" not in dataset.file_meta:
        # Written without a Part 10 header: pydicom detected the encoding while parsing.
        from pydicom.dataset import FileMetaDataset
        from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian

        implicit, little_endian = getattr(dataset, "original_encoding", (True, True))
        if getattr(dataset, "file_meta", None) is None:
            dataset.file_meta = FileMetaDataset()
        dataset.file_meta.TransferSyntaxUID = (
            ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian if little_endian else ExplicitVRBigEndian
        )
    try:
        return np.asarray(dataset.pixel_array)
    except Exception as exc:
        syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
        name = getattr(syntax, "name", str(syntax))
        if getattr(syntax, "is_compressed", False):
            raise DicomDependencyError(
                f"{label}: could not decode {name} pixel data. Compressed DICOM needs the optional decoders: "
                f"pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg. Decoder said: {exc}"
            ) from exc
        raise DicomError(f"{label}: pixel data is unreadable or truncated ({exc})") from exc


def _order_into_volume(slices: Sequence[_Slice], volume_index: int) -> tuple:
    """Sort planes along the slice normal; return (ordered slices, step vector, notes)."""

    notes: list = []
    first = slices[0]
    if any(record.position is None or record.orientation is None for record in slices):
        if len(slices) == 1:
            return list(slices), None, ["No ImagePositionPatient/ImageOrientationPatient: identity geometry assumed"]
        raise DicomGeometryError(
            "Series lacks ImagePositionPatient/ImageOrientationPatient, so its slices cannot be placed in space"
        )
    if any(record.pixel_spacing is None for record in slices):
        raise DicomGeometryError("Series lacks PixelSpacing; voxel size is unknown")
    spacings = np.array([record.pixel_spacing for record in slices])
    if np.abs(spacings - spacings[0]).max() > 1e-4:
        raise DicomGeometryError("In-plane pixel spacing differs between slices of the series")

    row_direction = np.array(first.orientation[:3])
    column_direction = np.array(first.orientation[3:])
    if abs(np.linalg.norm(row_direction) - 1) > 1e-2 or abs(np.linalg.norm(column_direction) - 1) > 1e-2 or abs(
        row_direction @ column_direction
    ) > 1e-2:
        raise DicomGeometryError("ImageOrientationPatient is not a pair of orthogonal unit vectors")
    normal = np.cross(row_direction, column_direction)
    distances = np.array([np.dot(record.position, normal) for record in slices])

    order = np.argsort(distances, kind="stable")
    groups: list = []  # planes that share one physical position
    for index in order:
        if groups and distances[index] - distances[groups[-1][0]] <= _SAME_POSITION_MM:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    counts = {len(group) for group in groups}
    if len(counts) != 1:
        raise DicomGeometryError(
            "Slice positions repeat an inconsistent number of times "
            f"({sorted(counts)} images per position); the series mixes incomplete volumes"
        )
    per_position = counts.pop()
    if per_position > 1:
        if not 0 <= volume_index < per_position:
            raise DicomGeometryError(f"volume_index {volume_index} is outside this {per_position}-volume series")
        notes.append(
            f"Series holds {per_position} volumes at the same slice positions (multi-echo / dynamic); "
            f"using volume {volume_index}"
        )
    elif volume_index != 0:
        raise DicomGeometryError("volume_index is only meaningful for multi-volume series")
    ordered = [
        slices[sorted(group, key=lambda index: slices[index].order_key)[volume_index]] for group in groups
    ]

    if len(ordered) == 1:
        return ordered, None, notes
    positions = np.array([record.position for record in ordered])
    gaps = np.diff(positions @ normal)
    step = float(np.median(gaps))
    tolerance = max(0.02, 0.02 * step)
    irregular = np.abs(gaps - step) > tolerance
    if irregular.any():
        multiples = gaps[irregular] / step
        if np.all(np.abs(multiples - np.round(multiples)) < 0.05) and np.all(np.round(multiples) >= 2):
            missing = int(np.round(multiples).sum() - irregular.sum())
            gap_text = ", ".join(f"{gap:.3f}" for gap in sorted({float(gap) for gap in np.round(gaps[irregular], 3)}))
            raise DicomGeometryError(
                f"{missing} slice(s) are missing from the series (gaps of {gap_text} mm in a {step:.3f} mm stack). "
                "Re-export the complete series."
            )
        raise DicomGeometryError(
            f"Slice spacing is not uniform (steps range {gaps.min():.3f}-{gaps.max():.3f} mm); "
            "this is not a regular 3D volume"
        )
    step_vector = (positions[-1] - positions[0]) / (len(ordered) - 1)
    tilt = float(np.degrees(np.arccos(np.clip(step_vector @ normal / np.linalg.norm(step_vector), -1, 1))))
    if tilt > 0.5:
        notes.append(f"Slice stack is sheared {tilt:.2f} degrees from the image-plane normal (gantry tilt)")
    return ordered, step_vector, notes


def load_series(series: DicomSeries, *, volume_index: int = 0) -> MRIVolume:
    """Assemble one discovered series into a canonical RAS+ :class:`MRIVolume`."""

    _require_pydicom()
    if not series.slices:
        raise NotDicomError("Series has no image planes")
    image_type = str(series.acquisition.get("image_type", "")).upper()
    if "MOSAIC" in image_type:
        raise UnsupportedDicomError("Siemens mosaic images (fMRI/DWI) are not structural volumes")

    ordered, step_vector, notes = _order_into_volume(series.slices, volume_index)

    multiframe_arrays: dict = {}  # a multi-frame file is decoded once, then indexed per frame
    files_read = 0
    planes = []
    digest = hashlib.sha256()
    inverted = False
    for record in ordered:
        stored = multiframe_arrays.get(record.source)
        if stored is None:
            raw = record.source.read_bytes()
            dataset = pydicom.dcmread(io.BytesIO(raw), force=True)
            stored = _decode(dataset, record.source.label)
            digest.update(hashlib.sha256(raw).digest())
            files_read += 1
            inverted = inverted or _text(dataset.get("PhotometricInterpretation")).upper() == "MONOCHROME1"
            if record.frame is not None:
                multiframe_arrays[record.source] = stored
            del dataset, raw
        if record.frame is not None and stored.ndim == 3:
            stored = stored[record.frame]
        if stored.shape != (record.rows, record.columns):
            raise DicomGeometryError(
                f"{record.source.label}: pixel array is {stored.shape}, header says {(record.rows, record.columns)}"
            )
        if record.slope == 1.0 and record.intercept == 0.0:
            planes.append(stored)  # stored integers are the values; kept exact
        else:
            planes.append(stored.astype(np.float64) * record.slope + record.intercept)
    multiframe_arrays.clear()

    # DICOM pixel_array is [row, column]; voxel axis 0 runs along a row (column index).
    data = np.stack(planes, axis=-1).transpose(1, 0, 2)
    if inverted:
        data = data.astype(np.float64)
        data = (data.max() + data.min()) - data
        notes.append("MONOCHROME1 pixel data inverted so that higher values are brighter")

    first = ordered[0]
    if first.position is None or first.orientation is None:
        # No spatial tags (e.g. a lone exported slice): standard axial display
        # geometry, columns toward patient left and rows toward posterior.
        affine_lps = np.eye(4)
        if first.pixel_spacing is not None:
            affine_lps[0, 0], affine_lps[1, 1] = first.pixel_spacing[1], first.pixel_spacing[0]
    else:
        row_direction = np.array(first.orientation[:3])
        column_direction = np.array(first.orientation[3:])
        row_spacing, column_spacing = first.pixel_spacing  # between rows, between columns
        if step_vector is None:
            thickness = series.acquisition.get("spacing_between_slices_mm") or series.acquisition.get(
                "slice_thickness_mm"
            ) or 1.0
            step_vector = np.cross(row_direction, column_direction) * float(thickness)
        affine_lps = np.eye(4)
        affine_lps[:3, 0] = row_direction * column_spacing
        affine_lps[:3, 1] = column_direction * row_spacing
        affine_lps[:3, 2] = step_vector
        affine_lps[:3, 3] = first.position
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps  # DICOM LPS+ -> NIfTI RAS+

    slopes = {record.slope for record in ordered}
    intercepts = {record.intercept for record in ordered}
    acquisition = dict(series.acquisition)
    acquisition["contrast_guess"] = {
        "label": series.contrast.label,
        "confidence": series.contrast.confidence,
        "reasons": list(series.contrast.reasons),
    }
    acquisition["rescale"] = {
        "slope_range": [min(slopes), max(slopes)],
        "intercept_range": [min(intercepts), max(intercepts)],
        "varies_between_slices": len(slopes) > 1 or len(intercepts) > 1,
    }
    if series.modality and series.modality != "MR":
        notes.append(f"Modality is {series.modality}, not MR")

    return MRIVolume.from_array(
        data,
        affine_ras,
        acquisition=acquisition,
        identity=series.identity,
        source={
            "format": "dicom_multiframe" if series.is_multiframe else "dicom_series",
            "files": files_read,
            "slices": len(ordered),
            "content_sha256": digest.hexdigest(),
            "series_key": series.key,
        },
        warnings=notes,
    )


def load_volume(path: PathLike, *, series_key: Optional[str] = None, volume_index: int = 0) -> MRIVolume:
    """Discover the series under ``path`` and load one (default: best 3D T1 candidate)."""

    series = discover_series(path)
    if not series:
        raise NotDicomError(f"No DICOM image series found under {path}")
    chosen = select_series(series, key=series_key)
    volume = load_series(chosen, volume_index=volume_index)
    if series_key is None and len(series) > 1:
        volume.warnings.append(
            f"{len(series)} series found; loaded {chosen.technical_label}. Pass series_key to choose another."
        )
    return volume


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List DICOM series or convert one to a canonical NIfTI volume.")
    parser.add_argument("path", help="DICOM file, folder or zip archive")
    parser.add_argument("--series", help="Series key (prefix) from the listing; default: best 3D T1 candidate")
    parser.add_argument("--volume-index", type=int, default=0, help="Which volume of a multi-volume series")
    parser.add_argument("--to-nifti", metavar="OUT.nii.gz", help="Write the selected series as NIfTI + JSON sidecar")
    parser.add_argument(
        "--keep-descriptions", action="store_true",
        help="Keep free-text SeriesDescription/ProtocolName in the sidecar (may contain identifying text)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        series = discover_series(args.path)
        if not series:
            raise NotDicomError(f"No DICOM image series found under {args.path}")
        if not args.to_nifti:
            for candidate in series:
                print(f"{candidate.key}  {candidate.label}")
            return 0
        volume = load_series(select_series(series, key=args.series), volume_index=args.volume_index)
        record = volume.to_nifti(args.to_nifti, include_free_text=args.keep_descriptions)
        record["warnings"] = volume.warnings
        print(json.dumps(record, indent=2))
        return 0
    except DicomError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
