"""OASIS-1 (cross-sectional) index and volume loader for patient-level training.

OASIS-1 is obtained from its provider under the provider's data use terms
(https://sites.wustl.edu/oasisbrains/). Nothing in this module downloads it,
and nothing here should be read as permission to redistribute it: keep the
extracted data under the git-ignored ``data/external/``.

Why this cohort matters here: it has what the legacy slice images lack, namely
one identified person per scan and a Clinical Dementia Rating for each older
adult, so splits can be made by person and performance can be reported per
person.

Expected layout, as extracted from the provider's disc archives (any nesting of
``disc*`` folders is fine)::

    <root>/**/OAS1_0001_MR1/**/SUBJ_111/<image>.{hdr,img}    # normally under PROCESSED/MPRAGE/
    <root>/**/oasis_cross-sectional.csv                     # needs the ID, Age and CDR columns

Image used: ``SUBJ_111``, the average of the session's co-registered T1
acquisitions in native space at 1 mm, **with skull**. That is
the kind of image a DICOM T1 series holds, so a model trained on it reads raw
scans without a brain-extraction step. The files are Analyze 7.5, whose header
carries no trustworthy orientation; the voxel-to-world mapping below is the one
the Clinica OASIS-1 converter hard-codes for these images (voxel axes run
anterior, superior, left).

CLI::

    python -m src.oasis1 data/external/oasis1                  # cohort summary (aggregate counts only)
    python -m src.oasis1 data/external/oasis1 --qc oasis_qc.png  # orientation check sheet: look before training
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from src.mri_volume import MRIVolume, PathLike, VolumeError, sha256_file
except ModuleNotFoundError:  # executed from inside src/
    from mri_volume import MRIVolume, PathLike, VolumeError, sha256_file

logger = logging.getLogger(__name__)

SESSION_PATTERN = re.compile(r"^(OAS1_\d{4})_MR(\d)$")
#: Shape of every SUBJ_111 image according to the provider and the Clinica converter.
SUBJ_111_SHAPE = (256, 256, 160)
#: People with dementia by CDR in the provider's fact sheet (100 in all, every one aged over 60).
#: Used to warn about an incomplete tree, never to force a result.
DOCUMENTED_DEMENTED_BY_CDR = {"0.5": 70, "1": 28, "2": 2}


class OasisError(VolumeError):
    """The OASIS-1 tree or its clinical table is not as expected."""


@dataclass(frozen=True)
class OasisSession:
    """One imaging session. ``subject_id`` is what must never be split across folds."""

    session_id: str  # OAS1_0001_MR1
    subject_id: str  # OAS1_0001
    visit: int  # 1, or 2 for the short-interval repeat ("reliability") session
    age: Optional[float]
    sex: str
    cdr: Optional[float]  # Clinical Dementia Rating; None where it was not assessed (younger adults)
    mmse: Optional[float]
    directory: Optional[Path]

    @property
    def image_path(self) -> Optional[Path]:
        """The session's SUBJ_111 image (``.img``), or None when absent."""

        if self.directory is None:
            return None
        # Normally PROCESSED/MPRAGE/SUBJ_111/; found by folder name so a different nesting still works.
        images = sorted(path for path in self.directory.rglob("*.img") if path.parent.name == "SUBJ_111")
        return images[0] if images else None

    @property
    def cdr_level(self) -> Optional[str]:
        return None if self.cdr is None else f"{self.cdr:g}"


def _number(value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text or text.upper() in ("N/A", "NA", "NAN"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def find_clinical_table(root: Path) -> Path:
    """Locate ``oasis_cross-sectional.csv`` under ``root`` (not the reliability table)."""

    candidates = sorted(
        (path for path in root.rglob("*.csv") if "cross-sectional" in path.name.lower() and "reliab" not in path.name.lower()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if candidates:
        return candidates[0]
    spreadsheets = [path.name for path in root.rglob("*.xls*") if "cross-sectional" in path.name.lower()]
    hint = f" Found {spreadsheets[0]}: save it as CSV next to the images." if spreadsheets else ""
    raise OasisError(f"No oasis_cross-sectional.csv found under {root}.{hint}")


def read_clinical_table(path: Path) -> dict:
    """Session ID -> {age, sex, cdr, mmse}. Column names are matched case-insensitively."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise OasisError(f"{path} has no rows")
    columns = {name.strip().lower(): name for name in rows[0]}
    missing = [name for name in ("id", "age", "cdr") if name not in columns]
    if missing:
        raise OasisError(f"{path} lacks the column(s) {missing}; expected the provider's ID, Age, CDR, ... table")
    table = {}
    for row in rows:
        session_id = (row[columns["id"]] or "").strip()
        if not SESSION_PATTERN.match(session_id):
            continue
        if session_id in table:
            raise OasisError(f"{path} lists {session_id} more than once")
        cdr = _number(row[columns["cdr"]])
        if cdr is not None and cdr not in (0.0, 0.5, 1.0, 2.0, 3.0):
            raise OasisError(f"Unexpected CDR value {cdr} for one session; the table may not be the provider's")
        table[session_id] = {
            "age": _number(row[columns["age"]]),
            "sex": (row.get(columns.get("m/f", ""), "") or "").strip(),
            "cdr": cdr,
            "mmse": _number(row.get(columns.get("mmse", ""), "")),
        }
    if not table:
        raise OasisError(f"{path} has no OAS1_xxxx_MRn session IDs")
    return table


def find_session_directories(root: Path) -> dict:
    """Session ID -> directory, for every ``OAS1_xxxx_MRn`` folder under ``root``."""

    found: dict = {}
    for current, directories, _files in os.walk(root):
        for name in sorted(directories):
            if SESSION_PATTERN.match(name):
                if name in found:
                    raise OasisError(f"Session folder {name} appears twice under {root}; remove the duplicate extraction")
                found[name] = Path(current) / name
        # Session folders are leaves for this purpose; do not descend into them.
        directories[:] = sorted(name for name in directories if not SESSION_PATTERN.match(name))
    return found


def load_index(root: PathLike) -> list:
    """Every session named in the clinical table or present on disk, as :class:`OasisSession`."""

    root = Path(root)
    if not root.is_dir():
        raise OasisError(f"No such folder: {root}")
    table = read_clinical_table(find_clinical_table(root))
    directories = find_session_directories(root)
    sessions = []
    for session_id in sorted(set(table) | set(directories)):
        subject_id, visit = SESSION_PATTERN.match(session_id).groups()
        clinical = table.get(session_id, {})
        sessions.append(
            OasisSession(
                session_id=session_id, subject_id=subject_id, visit=int(visit), age=clinical.get("age"),
                sex=clinical.get("sex", ""), cdr=clinical.get("cdr"), mmse=clinical.get("mmse"),
                directory=directories.get(session_id),
            )
        )
    return sessions


def eligible_sessions(sessions: Sequence[OasisSession], *, min_age: float = 60.0) -> list:
    """First-visit sessions of older adults with a recorded CDR and an image on disk.

    Younger adults have no CDR and are excluded on purpose: counting them as
    controls would let a model score well by recognising age. The repeat
    (``MR2``) sessions are the same people scanned again and are left out so
    every person contributes one scan.
    """

    return [
        session for session in sessions
        if session.visit == 1 and session.cdr is not None and session.age is not None and session.age >= min_age
        and session.image_path is not None
    ]


def summarize(sessions: Sequence[OasisSession], *, min_age: float = 60.0) -> dict:
    """Aggregate description of what was found. No session or subject identifiers."""

    eligible = eligible_sessions(sessions, min_age=min_age)
    by_cdr = Counter(session.cdr_level for session in eligible)
    documented = {level: by_cdr.get(level, 0) == count for level, count in DOCUMENTED_DEMENTED_BY_CDR.items()}
    return {
        "sessions_in_clinical_table_or_on_disk": len(sessions),
        "sessions_with_image": sum(session.image_path is not None for session in sessions),
        "sessions_without_clinical_row": sum(session.age is None and session.cdr is None for session in sessions),
        "repeat_sessions": sum(session.visit != 1 for session in sessions),
        "minimum_age": min_age,
        "eligible_people": len({session.subject_id for session in eligible}),
        "eligible_by_cdr": {level: by_cdr[level] for level in sorted(by_cdr, key=float)},
        "eligible_nondemented_cdr0": sum(session.cdr == 0 for session in eligible),
        "eligible_demented_cdr_above_0": sum(session.cdr > 0 for session in eligible),
        "matches_documented_dementia_counts": all(documented.values()) if min_age <= 60.0 else None,
        "documented_dementia_counts_by_cdr": DOCUMENTED_DEMENTED_BY_CDR,
    }


def subj_111_affine(shape: Sequence[int]) -> np.ndarray:
    """Voxel -> RAS mm for a SUBJ_111 image: axes run Anterior, Superior, Left; 1 mm; centred.

    For the documented 256 x 256 x 160 grid this is exactly the matrix the
    Clinica OASIS-1 converter hard-codes::

        [[0, 0, -1,   80], [1, 0, 0, -128], [0, 1, 0, -128], [0, 0, 0, 1]]
    """

    n0, n1, n2 = (int(n) for n in shape)
    return np.array(
        [[0.0, 0.0, -1.0, n2 / 2], [1.0, 0.0, 0.0, -n0 / 2], [0.0, 1.0, 0.0, -n1 / 2], [0.0, 0.0, 0.0, 1.0]]
    )


def load_session_volume(session: OasisSession) -> MRIVolume:
    """The session's SUBJ_111 image as a canonical RAS+ :class:`MRIVolume`."""

    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise OasisError("Reading OASIS-1 needs nibabel (pip install nibabel)") from exc

    path = session.image_path
    if path is None:
        raise OasisError(f"{session.session_id} has no PROCESSED/MPRAGE/SUBJ_111 image")
    data = np.asarray(nib.load(str(path)).dataobj)
    data = data.reshape(data.shape[:3]) if data.ndim == 4 and data.shape[3] == 1 else data
    if data.ndim != 3:
        raise OasisError(f"{path.name} is not a single 3D volume (shape {data.shape})")
    warnings = []
    if tuple(data.shape) != SUBJ_111_SHAPE:
        warnings.append(
            f"SUBJ_111 image is {tuple(data.shape)}, not the documented {SUBJ_111_SHAPE}; "
            "check its orientation on the QC sheet before trusting it"
        )
    return MRIVolume.from_array(
        np.rint(data) if np.issubdtype(data.dtype, np.floating) else data,
        subj_111_affine(data.shape),
        acquisition={
            "modality": "MR",
            "contrast_guess": {"label": "T1w", "confidence": "high", "reasons": ["OASIS-1 T1-weighted MRI (provider documentation)"]},
            "dataset": "OASIS-1",
            "image_kind": "SUBJ_111: average of the session's co-registered acquisitions, native space, 1 mm, with skull",
        },
        source={"format": "analyze", "dataset": "OASIS-1", "files": 1, "content_sha256": sha256_file(path)},
        warnings=warnings,
    )


def orientation_sheet(sessions: Sequence[OasisSession], path: PathLike, *, count: int = 6) -> Path:
    """Write axial / coronal / sagittal mid-slices for a few sessions, without diagnoses.

    Look at it before training. In every row the axial view must show the nose
    at the top, the coronal and sagittal views the top of the head at the top,
    and the sagittal view the face on the right. Left versus right cannot be
    judged by eye and rests on the provider-derived mapping in this module.
    """

    from PIL import Image

    rows = []
    for session in list(sessions)[:count]:
        volume = load_session_volume(session)
        low, high = np.percentile(volume.data[volume.data > 0], [1, 99.5]) if (volume.data > 0).any() else (0.0, 1.0)
        scaled = np.clip((volume.data - low) / max(high - low, 1e-6), 0, 1)
        centre = [n // 2 for n in scaled.shape]
        views = [np.rot90(np.take(scaled, centre[axis], axis=axis)) for axis in (2, 1, 0)]  # axial, coronal, sagittal
        height = max(view.shape[0] for view in views)
        padded = [np.pad(view, ((0, height - view.shape[0]), (0, 4))) for view in views]
        rows.append(np.concatenate(padded, axis=1))
    if not rows:
        raise OasisError("No sessions with an image to show")
    width = max(row.shape[1] for row in rows)
    sheet = np.concatenate([np.pad(row, ((0, 4), (0, width - row.shape[1]))) for row in rows], axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(sheet * 255).astype(np.uint8)).save(path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise an extracted OASIS-1 tree and check image orientation.")
    parser.add_argument("root", help="Folder holding the extracted OASIS-1 discs and oasis_cross-sectional.csv")
    parser.add_argument("--min-age", type=float, default=60.0, help="Youngest age included (default 60)")
    parser.add_argument("--qc", metavar="SHEET.png", help="Write an orientation check sheet for the first eligible sessions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        sessions = load_index(args.root)
        summary = summarize(sessions, min_age=args.min_age)
        if args.qc:
            summary["orientation_sheet"] = str(orientation_sheet(eligible_sessions(sessions, min_age=args.min_age), args.qc))
    except VolumeError as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
