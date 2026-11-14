"""Display and form helpers for scans loaded into the GUI (no Tk imports).

Kept apart from ``radiology_report_gui`` so the slice rendering, header-derived
form values and the research summary can be tested headlessly.
"""

from __future__ import annotations

import html
from datetime import date
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from src.mri_volume import MRIVolume
except ModuleNotFoundError:  # executed from inside src/
    from mri_volume import MRIVolume

#: Smallest extent (voxels) along every axis for a scan to count as a 3D volume.
MIN_VOLUME_EXTENT = 16


def is_volume(volume: MRIVolume) -> bool:
    return min(volume.shape) >= MIN_VOLUME_EXTENT


def display_window(volume: MRIVolume) -> tuple:
    """Robust intensity window for the whole scan (same grey scale on every slice)."""

    data = volume.data
    positive = data[data > 0]
    values = positive if positive.size > 100 else data.ravel()
    low, high = (float(value) for value in np.percentile(values, [1.0, 99.5]))
    if not high > low:
        low, high = float(data.min()), float(data.max())
    if not high > low:
        high = low + 1.0
    return low, high


def axial_slice_image(volume: MRIVolume, index: int, window: Optional[tuple] = None, *, annotate: bool = True) -> Image.Image:
    """Axial slice ``index`` as an RGB image: anterior up, patient left on image left.

    The orientation matches the slices the research model sees (neurological
    convention), and the pixel aspect follows the voxel size. ``annotate`` adds
    L / R / A / P edge letters so the convention is never ambiguous.
    """

    index = int(np.clip(index, 0, volume.shape[2] - 1))
    low, high = window or display_window(volume)
    plane = np.clip((np.asarray(volume.data[:, :, index], dtype=np.float64) - low) / (high - low), 0, 1)
    pixels = np.rint(np.rot90(plane) * 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    spacing_x, spacing_y, _ = volume.spacing
    width = max(1, int(round(image.width * spacing_x / min(spacing_x, spacing_y))))
    height = max(1, int(round(image.height * spacing_y / min(spacing_x, spacing_y))))
    if (width, height) != image.size:
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    image = image.convert("RGB")
    return annotate_orientation(image) if annotate else image


def annotate_orientation(image: Image.Image) -> Image.Image:
    """Draw L / R / A / P at the edges of an anterior-up, left-on-left axial image."""

    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if min(width, height) < 48:
        return image
    colour = (255, 214, 10)
    draw.text((3, height // 2 - 5), "L", fill=colour)
    draw.text((width - 10, height // 2 - 5), "R", fill=colour)
    draw.text((width // 2 - 3, 2), "A", fill=colour)
    draw.text((width // 2 - 3, height - 13), "P", fill=colour)
    return image


def format_person_name(value: str) -> str:
    """DICOM PN 'FAMILY^GIVEN^MIDDLE' -> 'GIVEN MIDDLE FAMILY' (first component group only)."""

    alphabetic = (value or "").split("=")[0]
    parts = [part.strip() for part in alphabetic.split("^")]
    family, rest = (parts[0] if parts else ""), [part for part in parts[1:3] if part]
    return " ".join([*rest, family]).strip()


def parse_dicom_date(value: str) -> Optional[date]:
    """'YYYYMMDD' (or legacy 'YYYY.MM.DD') -> date, or None when absent/invalid."""

    digits = "".join(character for character in (value or "") if character.isdigit())
    if len(digits) != 8:
        return None
    try:
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return None


def technique_from_header(volume: MRIVolume) -> str:
    """One-line technique statement built only from recorded acquisition parameters."""

    acquisition = volume.acquisition
    parts = []
    modality = acquisition.get("modality") or "MR"
    field_strength = acquisition.get("field_strength_t")
    parts.append(f"{modality} {field_strength:g} T" if field_strength else str(modality))
    contrast = (acquisition.get("contrast_guess") or {}).get("label")
    dimension = acquisition.get("mr_acquisition_type")
    described = " ".join(text for text in (dimension, contrast if contrast not in (None, "unknown", "other", "not_mr") else None) if text)
    if described:
        parts.append(described)
    timing = [
        f"{name} {acquisition[key]:g} {unit}"
        for name, key, unit in (("TR", "repetition_time_ms", "ms"), ("TE", "echo_time_ms", "ms"), ("TI", "inversion_time_ms", "ms"), ("flip", "flip_angle_deg", "deg"))
        if acquisition.get(key) is not None
    ]
    if timing:
        parts.append("(" + ", ".join(timing) + ")")
    spacing = " x ".join(f"{value:.2g}" for value in volume.spacing)
    parts.append(f"{volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]} voxels at {spacing} mm")
    return f"{' '.join(parts[:-1])}; {parts[-1]}"


def contrast_from_header(volume: MRIVolume) -> str:
    agent = volume.acquisition.get("contrast_agent")
    return str(agent) if agent else "None recorded in image header"


def scan_caption(volume: MRIVolume, index: int) -> str:
    """Technical one-liner for the viewer (no identifying fields)."""

    guess = volume.acquisition.get("contrast_guess") or {}
    contrast = guess.get("label", "unknown contrast")
    source = {"dicom_series": "DICOM series", "dicom_multiframe": "DICOM multi-frame", "nifti": "NIfTI"}.get(
        volume.source.get("format"), "scan"
    )
    spacing = " x ".join(f"{value:.2g}" for value in volume.spacing)
    return (
        f"{source}: {contrast}, {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]} at {spacing} mm  |  "
        f"axial slice {index + 1} of {volume.shape[2]}"
    )


def volume_report_html(result, patient_info: Optional[dict] = None, technique: str = "") -> str:
    """HTML summary of a volume analysis for the report pane and the PDF.

    Deliberately not shaped like a radiology report: it states a research
    score, how it was computed and the documented limits of the model.
    """

    patient_info = patient_info or {}
    escape = lambda value: html.escape(str(value)) if value else "Not provided"  # noqa: E731
    positive = result.class_names[1]
    probabilities = {name: float(value) for name, value in zip(result.class_names, result.probabilities)}
    members = ", ".join(f"{float(row[1]):.2f}" for row in result.member_probabilities)
    scored = "Nine axial slices at 45-85% of the cranial extent" if getattr(result, "slice_policy", "protocol9") == "protocol9" else (
        "Every 1.5 mm axial slice between 45% and 85% of the cranial extent"
    )
    rows = [
        ("Patient", escape(patient_info.get("name"))),
        ("Patient ID", escape(patient_info.get("patient_id"))),
        ("Date of birth", escape(patient_info.get("dob"))),
        ("Technique", escape(technique)),
        ("Model", f"{html.escape(positive)}/{html.escape(result.class_names[0])} research ensemble "
                  f"({len(result.member_probabilities)} MobileNetV3-Large members)"),
        ("Input", f"{scored}, 1.5 mm grid (recipe exploratory_t1_axial_v1)"),
        (f"Ensemble P({html.escape(positive)})", f"{probabilities.get(positive, float('nan')):.3f}"),
        (f"Member P({html.escape(positive)})", members),
        (f"Label at threshold {float(getattr(result, 'threshold', 0.5)):g}", html.escape(result.predicted_label)),
    ]
    # Bold-label lines rather than a table: the in-app HTML widget does not lay out tables.
    table = "".join(f"<p><b>{name}:</b> {value}</p>" for name, value in rows)
    notice = getattr(result, "notice", "") or next((note for note in result.warnings if note.startswith("Research output only")), "")
    notes = [note for note in result.warnings if note != notice and not note.startswith("Research output only")]
    note_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    cam_text = (
        "<p>The heatmap is Grad-CAM computed slice by slice across the analysed slab and scaled once for the whole "
        "volume, so slices the model relied on less stay darker. Use the slider to step through it, or save the 3D "
        "map (NIfTI) to view it over the scan in a volume viewer.</p>"
        if result.cam_grid is not None
        else ""
    )
    return (
        "<h2>Volume analysis - research score, not a diagnosis</h2>"
        f"{table}"
        "<h3>How to read this</h3>"
        f"<p>{html.escape(notice)}</p>"
        "<p>This model has not been validated for clinical use and must not inform patient care.</p>"
        f"{cam_text}"
        + (f"<h3>Notes for this scan</h3><ul>{note_html}</ul>" if note_html else "")
    )
