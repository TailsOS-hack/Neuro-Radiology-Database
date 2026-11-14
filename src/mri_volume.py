"""In-memory MRI volume shared by the DICOM and NIfTI readers.

Every reader in this project hands downstream code the same object: a float32
array in closest-canonical RAS+ voxel order plus the 4x4 voxel-to-world affine
(world axes: +x Right, +y Anterior, +z Superior, millimetres; the NIfTI
convention used by nibabel and by ``research/dementia_round3``). Keeping one
geometry convention is what lets a slice-wise Grad-CAM stack be written back
as a volume that overlays the scan it explains.

Identifying header fields are held in a separate :class:`PatientIdentity`
object. Nothing in this module logs them or writes them to disk; exports carry
acquisition (technical) metadata only unless a caller asks otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: World-axis letters for the negative / positive direction of x, y, z (RAS+).
_AXIS_LETTERS = (("L", "R"), ("P", "A"), ("I", "S"))

#: Free-text acquisition fields. Technologists can type anything into them, so
#: they are dropped from exports that may leave the acquiring machine.
FREE_TEXT_ACQUISITION_FIELDS = ("series_description", "protocol_name")


class VolumeError(ValueError):
    """A volume is malformed or cannot be represented by this pipeline."""


@dataclass(frozen=True)
class PatientIdentity:
    """Identifying header content (PHI). Kept apart from acquisition metadata.

    Values are the raw header strings. They are only ever handed to the local
    GUI to pre-fill the report form; no export, log line or sidecar in this
    package includes them.
    """

    name: str = ""
    patient_id: str = ""
    birth_date: str = ""
    sex: str = ""
    age: str = ""
    study_date: str = ""
    accession_number: str = ""
    study_description: str = ""
    study_uid: str = ""
    series_uid: str = ""
    institution: str = ""
    station: str = ""
    device_serial: str = ""

    def __repr__(self) -> str:  # never leak PHI through reprs, logs or tracebacks
        populated = sorted(key for key, value in asdict(self).items() if value)
        return f"PatientIdentity(<redacted>; fields={populated})"


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    """Voxel edge lengths in mm (column norms of the affine's 3x3 part)."""

    return np.sqrt((np.asarray(affine, dtype=np.float64)[:3, :3] ** 2).sum(axis=0))


def _orientation(affine: np.ndarray) -> list[tuple[int, int]]:
    """For each voxel axis: (closest world axis, +1 or -1 direction).

    Same decision rule as ``nibabel.orientations.io_orientation``: shears are
    removed with a polar decomposition and every world axis is claimed once, so
    oblique acquisitions still receive a unique permutation.
    """

    rzs = np.asarray(affine, dtype=np.float64)[:3, :3]
    zooms = np.sqrt((rzs * rzs).sum(axis=0))
    if not np.isfinite(rzs).all() or np.any(zooms <= 0):
        raise VolumeError("Affine has a nonfinite or zero-length voxel axis")
    rotation_stretch = rzs / zooms
    left, singular, right = np.linalg.svd(rotation_stretch, full_matrices=False)
    tolerance = singular.max() * 3 * np.finfo(singular.dtype).eps
    if np.any(singular <= tolerance):
        raise VolumeError("Affine is singular; voxel axes are not independent")
    rotation = left @ right
    result: list[tuple[int, int]] = []
    for in_axis in range(3):
        column = rotation[:, in_axis]
        out_axis = int(np.argmax(np.abs(column)))
        result.append((out_axis, -1 if column[out_axis] < 0 else 1))
        rotation[out_axis, :] = 0  # each world axis is assigned exactly once
    if sorted(axis for axis, _ in result) != [0, 1, 2]:
        raise VolumeError("Could not assign voxel axes to distinct anatomical axes")
    return result


def axis_codes(affine: np.ndarray) -> tuple[str, str, str]:
    """Anatomical direction of increasing index along each voxel axis, e.g. ('R','A','S')."""

    return tuple(  # type: ignore[return-value]
        _AXIS_LETTERS[out_axis][1 if direction > 0 else 0] for out_axis, direction in _orientation(affine)
    )


def reorient_to_ras(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip/permute voxel axes to closest-canonical RAS+ without resampling.

    Voxel values are untouched and every voxel keeps its world coordinate; only
    the array index order changes. Equivalent to ``nibabel.as_closest_canonical``.
    """

    data = np.asarray(data)
    affine = np.asarray(affine, dtype=np.float64)
    if data.ndim != 3 or affine.shape != (4, 4):
        raise VolumeError("Expected a 3D array and a 4x4 affine")
    orientation = _orientation(affine)
    new_affine = affine.copy()
    for in_axis, (_, direction) in enumerate(orientation):
        if direction < 0:
            data = np.flip(data, axis=in_axis)
            # index i -> (n - 1) - i: shift the origin to the far end, reverse the step
            new_affine[:3, 3] += new_affine[:3, in_axis] * (data.shape[in_axis] - 1)
            new_affine[:3, in_axis] *= -1
    permutation = [0, 0, 0]
    for in_axis, (out_axis, _) in enumerate(orientation):
        permutation[out_axis] = in_axis
    data = np.transpose(data, permutation)
    new_affine[:, :3] = new_affine[:, permutation]
    return np.ascontiguousarray(data), new_affine


def as_float_voxels(data: np.ndarray) -> np.ndarray:
    """float32 when that is lossless, otherwise float64.

    Integer scans are exact in float32. Scans that carry a real-valued rescale
    (a NIfTI ``scl_slope``, a non-unit DICOM RescaleSlope) keep double precision
    so downstream interpolation sees the same numbers the header defines.
    """

    data = np.asarray(data)
    if data.dtype == np.float64:
        return np.ascontiguousarray(data)
    if np.issubdtype(data.dtype, np.integer) and data.size and np.abs(data.astype(np.int64)).max() >= 2**24:
        return np.ascontiguousarray(data, dtype=np.float64)
    return np.ascontiguousarray(data, dtype=np.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class MRIVolume:
    """One 3D MRI volume in closest-canonical RAS+ voxel order.

    Attributes:
        data: float32 array (float64 when the source carries a real-valued
            rescale), shape (X, Y, Z); index increases toward Right, Anterior,
            Superior respectively (up to obliquity).
        affine: 4x4 voxel-index -> RAS millimetre transform.
        acquisition: technical, non-identifying acquisition metadata.
        identity: identifying header fields, or None. Never exported.
        source: provenance (format, file count, content digest).
        warnings: human-readable notes about anything unusual found on load.
    """

    data: np.ndarray
    affine: np.ndarray
    acquisition: dict = field(default_factory=dict)
    identity: Optional[PatientIdentity] = None
    source: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = as_float_voxels(self.data)
        self.affine = np.asarray(self.affine, dtype=np.float64)
        if self.data.ndim != 3 or min(self.data.shape) < 1:
            raise VolumeError(f"Expected a 3D volume, got shape {self.data.shape}")
        if self.affine.shape != (4, 4) or not np.isfinite(self.affine).all():
            raise VolumeError("Volume affine must be a finite 4x4 matrix")
        if not np.allclose(self.affine[3], [0, 0, 0, 1]):
            raise VolumeError("Volume affine must have [0, 0, 0, 1] as its last row")
        if abs(np.linalg.det(self.affine[:3, :3])) < 1e-9:
            raise VolumeError("Volume affine is singular")
        if not np.isfinite(self.data).all():
            raise VolumeError("Volume contains nonfinite intensities")

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        affine: np.ndarray,
        *,
        acquisition: Optional[dict] = None,
        identity: Optional[PatientIdentity] = None,
        source: Optional[dict] = None,
        warnings: Optional[list] = None,
    ) -> "MRIVolume":
        """Build a volume from any voxel order; it is reoriented to RAS+ here."""

        canonical, canonical_affine = reorient_to_ras(as_float_voxels(data), affine)
        return cls(
            canonical,
            canonical_affine,
            acquisition=dict(acquisition or {}),
            identity=identity,
            source=dict(source or {}),
            warnings=list(warnings or []),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)  # type: ignore[return-value]

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel size in mm along the R, A and S voxel axes."""

        return tuple(float(v) for v in voxel_sizes(self.affine))  # type: ignore[return-value]

    @property
    def orientation(self) -> tuple[str, str, str]:
        return axis_codes(self.affine)

    @property
    def field_of_view_mm(self) -> tuple[float, float, float]:
        return tuple(float(s * n) for s, n in zip(self.spacing, self.shape))  # type: ignore[return-value]

    def obliquity_degrees(self) -> float:
        """Largest angle between a voxel axis and its nearest scanner axis."""

        directions = self.affine[:3, :3] / voxel_sizes(self.affine)
        return float(np.degrees(np.arccos(np.clip(np.abs(directions).max(axis=0), -1.0, 1.0))).max())

    def canonical_sha256(self) -> str:
        """Content hash of voxels + affine, as ``research/dementia_round3/data.py`` recomputes it."""

        return hashlib.sha256(
            np.asarray(self.data, dtype="<f4").tobytes(order="C") + np.asarray(self.affine, dtype="<f8").tobytes()
        ).hexdigest()

    def describe(self, *, include_free_text: bool = True) -> dict:
        """Geometry + acquisition summary. Never includes :class:`PatientIdentity`."""

        acquisition = {
            key: value
            for key, value in self.acquisition.items()
            if include_free_text or key not in FREE_TEXT_ACQUISITION_FIELDS
        }
        return _jsonable(
            {
                "shape": list(self.shape),
                "spacing_mm": [round(v, 6) for v in self.spacing],
                "field_of_view_mm": [round(v, 3) for v in self.field_of_view_mm],
                "orientation": list(self.orientation),
                "obliquity_degrees": round(self.obliquity_degrees(), 4),
                "affine": self.affine.tolist(),
                "intensity": {
                    "min": float(self.data.min()),
                    "max": float(self.data.max()),
                    "mean": float(self.data.mean()),
                },
                "acquisition": acquisition,
                "source": self.source,
                "warnings": list(self.warnings),
            }
        )

    def to_nifti(self, path: PathLike, *, sidecar: bool = True, include_free_text: bool = False) -> dict:
        """Write float32 NIfTI-1 (+ JSON sidecar) and verify it by reading it back.

        Returns a provenance record with ``volume_sha256`` (file bytes) and
        ``canonical_volume_sha256`` (voxels + affine after an independent reload),
        the two hashes the Round 3 patient manifest pins. The NIfTI header and the
        sidecar carry no identifying fields; free-text descriptors are omitted
        from the sidecar unless ``include_free_text`` is set.
        """

        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
            raise VolumeError("NIfTI export needs nibabel (pip install nibabel)") from exc

        path = Path(path)
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            raise VolumeError("NIfTI output must end in .nii or .nii.gz")
        path.parent.mkdir(parents=True, exist_ok=True)

        image = nib.Nifti1Image(np.asarray(self.data, dtype=np.float32), self.affine)
        image.header.set_data_dtype(np.float32)
        image.header.set_xyzt_units("mm", "sec")
        image.header["descrip"] = b"neuro-radiology-database canonical RAS"
        image.set_sform(self.affine, code=1)
        directions = self.affine[:3, :3] / voxel_sizes(self.affine)
        sheared = not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4)
        # A qform cannot represent shear; leave it unset rather than conflicting.
        image.set_qform(None if sheared else self.affine, code=0 if sheared else 1)
        nib.save(image, str(path))

        reloaded = nib.load(str(path))
        stored = np.asarray(reloaded.get_fdata(dtype=np.float32), dtype="<f4")
        if stored.shape != self.data.shape or not np.array_equal(stored, self.data.astype(np.float32)):
            raise VolumeError("NIfTI readback differs from the in-memory volume")
        if not np.allclose(reloaded.affine, self.affine, rtol=0, atol=1e-3):
            raise VolumeError("NIfTI readback affine differs from the in-memory volume")
        canonical = nib.as_closest_canonical(reloaded)
        canonical_data = np.asarray(canonical.get_fdata(dtype=np.float32), dtype="<f4")
        record = {
            "path": str(path),
            "volume_sha256": sha256_file(path),
            "canonical_volume_sha256": hashlib.sha256(
                canonical_data.tobytes(order="C") + np.asarray(canonical.affine, dtype="<f8").tobytes()
            ).hexdigest(),
            "shape": list(self.shape),
            "spacing_mm": [float(v) for v in voxel_sizes(reloaded.affine)],
            "orientation": list(axis_codes(reloaded.affine)),
        }
        if sidecar:
            sidecar_path = path.with_name(path.name.replace(".nii.gz", "").replace(".nii", "") + ".json")
            payload = {**record, "path": path.name, **self.describe(include_free_text=include_free_text)}
            sidecar_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            record["sidecar"] = str(sidecar_path)
        return record


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_nifti(path: PathLike) -> MRIVolume:
    """Read a 3D NIfTI file into the same canonical representation as DICOM."""

    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise VolumeError("Reading NIfTI needs nibabel (pip install nibabel)") from exc

    path = Path(path)
    image = nib.load(str(path))
    if len(image.shape) == 4 and image.shape[3] == 1:
        image = nib.funcs.squeeze_image(image)
    if len(image.shape) != 3:
        raise VolumeError(f"Expected a 3D NIfTI volume, got shape {tuple(image.shape)}")
    # ``dataobj`` keeps on-disk integers as integers and applies any header
    # scaling in double precision, exactly what nibabel's own resampling sees.
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all():
        raise VolumeError("NIfTI volume contains nonfinite intensities")
    return MRIVolume.from_array(
        data,
        image.affine,
        acquisition={},  # a NIfTI header does not record modality or sequence
        source={"format": "nifti", "files": 1, "content_sha256": sha256_file(path)},
    )
