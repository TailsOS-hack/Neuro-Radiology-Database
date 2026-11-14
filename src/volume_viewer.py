"""Interactive viewer for a volume analysis: three linked slice views and a rotating 3D projection.

``build_viewer`` turns a :class:`volume_analysis.VolumeAnalysis` (or the files
it saved) into one HTML file that embeds the scan on the model's grid and the
stacked Grad-CAM map. The file needs no server, no network and no other file:
open it in a current browser. ``VolumeAnalysis.save`` writes it as
``viewer.html`` next to the NIfTI volumes.

What it holds: the image volume, the attention volume and the analysis summary
(scores, geometry, the ensemble's notice). It holds no patient-identifying
header fields, exactly like ``result.json``. It does hold the scan itself, so
treat the file like the scan.

CLI::

    python -m src.volume_viewer results/                 # (re)build results/viewer.html from saved outputs
    python -m src.volume_viewer results/ --output map.html
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import json
import logging
import os
import re
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

VIEWER_FILENAME = "viewer.html"
DEFAULT_TITLE = "3D attention map"
#: Title of a page for an analysis that has no attention map.
NO_MAP_TITLE = "Scan on the model grid"
TEMPLATE_PATH = Path(__file__).with_name("volume_viewer_template.html")
_FRAGMENT_START, _FRAGMENT_END = "<!--vv:fragment-start-->", "<!--vv:fragment-end-->"
_SUBHEADING = {"h1": "h2", "h2": "h3", "h3": "h4"}
#: Voxels kept around the head when the display volume is cropped to it.
_CROP_MARGIN = 3
#: Display level (0-255) above which a voxel counts as part of the head for cropping.
_FOREGROUND_LEVEL = 12
#: How far a saved volume's affine may differ from the one in result.json (mm per unit step, and mm).
_AFFINE_TOLERANCE = 1e-3


class ViewerError(ValueError):
    """The analysis outputs cannot be turned into a viewer."""


def _pack(volume: np.ndarray) -> str:
    """A uint8 (X, Y, Z) volume as base64 text of a zlib stream; x varies fastest, then y, then z.

    Every row (fixed y and z) holds its first voxel followed by each voxel's
    difference from the one before it, modulo 256: smooth images compress
    better that way. The page undoes it with a running sum, so the voxel
    values arrive exactly, without passing through an image decoder.
    """

    rows = np.ascontiguousarray(volume.transpose(2, 1, 0))
    steps = np.diff(rows.astype(np.int16), axis=2, prepend=0).astype(np.uint8)
    return base64.b64encode(zlib.compress(steps.tobytes(), 9)).decode("ascii")


def _crop_bounds(anatomy: np.ndarray, attention: Optional[np.ndarray], keep_z: Optional[tuple] = None) -> tuple:
    """Index ranges that keep the head, every voxel of the map and the axial slices ``keep_z`` (first, last)."""

    occupied = anatomy > _FOREGROUND_LEVEL
    if attention is not None:
        occupied |= attention > 0
    if not occupied.any():
        return tuple((0, n) for n in anatomy.shape)
    bounds = []
    for axis, size in enumerate(anatomy.shape):
        along = np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != axis)))
        start, stop = int(along[0]) - _CROP_MARGIN, int(along[-1]) + _CROP_MARGIN + 1
        if axis == 2 and keep_z is not None:
            start, stop = min(start, int(keep_z[0])), max(stop, int(keep_z[1]) + 1)
        bounds.append((max(0, start), min(size, stop)))
    return tuple(bounds)


def _scored_slices(summary: dict) -> list:
    """Axial grid indices the score is computed from."""

    preparation = summary["preparation"]
    recorded = preparation.get("scored_slice_indices")
    if recorded:
        return [int(z) for z in recorded]
    protocol = [int(z) for z in preparation["protocol_slice_indices"]]
    policies = {member.get("slice_policy") for member in summary.get("members") or [] if isinstance(member, dict)} - {None}
    if policies:
        band = policies == {"band"}
    else:  # a result.json from before the policy was recorded for each member
        band = "trained-band" in str(summary.get("aggregation", ""))
    return list(range(protocol[0], protocol[-1] + 1)) if band else protocol


def _slices(count: int) -> str:
    return f"{count} axial slice" + ("" if count == 1 else "s")


def _map_texts(scored: list, explained: list, target: str, spacing_mm: float) -> dict:
    """The sentences that say which slices the score uses, which ones the map covers and what the map means.

    The score and the map need not use the same slices: the Round 2 ensemble
    scores nine slices while the default map covers every slice between them,
    and a map can also be limited to the nine. The wording follows the sets.
    """

    scored_set, explained_set = set(scored), set(explained)
    both = [z for z in explained if z in scored_set]
    unscored = len(explained) - len(both)
    first, last = explained[0], explained[-1]
    contiguous = explained == list(range(first, last + 1))
    levels = f"slices {first} to {last} of the {spacing_mm:g} mm grid"
    caution = "It does not locate disease, and a tidy-looking map is not evidence of a good model."
    if unscored == 0:
        map_short = f"The map shows what pushed the {target} output up on the slices it covers."
        map_text = (
            f"The map shows what pushed this model's {target} output up on the slices it covers, "
            f"all of which count toward the score. {caution}"
        )
    else:
        map_short = (
            f"The map shows what pushes the {target} output up slice by slice; "
            f"only {len(both)} of its {len(explained)} slices fed the score."
        )
        map_text = (
            f"On the {_slices(len(both))} the score is computed from, the map shows what pushed this model's {target} "
            f"output up. On the other {unscored} it shows what the same model responds to when it is given that slice "
            f"alone; those slices did not enter the score. {caution}"
        )
    if explained_set == scored_set:
        scored_text = (
            f"The score uses every one of the {_slices(len(scored))} of the slab, and the map covers each of them."
            if contiguous
            else f"The score uses {_slices(len(scored))}, and the map covers exactly those; the slices between them were not analysed."
        )
        chart_text = "Every slice shown counts toward the score."
    elif scored_set < explained_set:
        scored_text = (
            f"The score uses {_slices(len(scored))}; the map covers "
            + (f"all {len(explained)} slices of the slab between the lowest and the highest of them." if contiguous
               else f"{len(explained)} slices, including all of them.")
        )
        chart_text = f"Only the {len(both)} larger dots are the slices the score is computed from."
    elif explained_set < scored_set:
        scored_text = f"The score uses {_slices(len(scored))}; the map covers {len(explained)} of them."
        chart_text = (
            f"All {len(explained)} slices shown count toward the score, "
            f"which also uses {len(scored) - len(both)} slices that have no map."
        )
    else:
        scored_text = (
            f"The score uses {_slices(len(scored))}; the map covers {len(explained)} slices, "
            f"{len(both)} of which count toward the score."
        )
        chart_text = f"The {len(both)} larger dots are the slices the score is computed from."
    if contiguous:
        slab_text = (
            f"The map covers the {_slices(len(explained))} between the two marked levels ({levels}). "
            "Outside them it is empty by construction, not because the model found nothing."
        )
    else:
        slab_text = (
            f"The map covers {len(explained)} separate axial slices between the two marked levels ({levels}). "
            "The slices between them, and everything outside the levels, have no map by construction, "
            "not because the model found nothing."
        )
    return {
        "map_short": map_short,
        "map_text": map_text,
        "scored_text": scored_text,
        "scored_chart_text": chart_text,
        "slab_text": slab_text,
    }


def viewer_payload(grid: np.ndarray, cam_grid: Optional[np.ndarray], summary: dict) -> dict:
    """Everything the page needs, as JSON-serialisable values.

    ``grid`` is the scan on the model grid (RAS+), ``cam_grid`` the attention
    volume on the same grid (or None), ``summary`` the dict from
    ``VolumeAnalysis.summary()`` / ``result.json``. The page shows attention
    as a fraction of the strongest voxel, so the map is divided by its own
    maximum here; an analysis already stores it on that scale.
    """

    grid = np.asarray(grid)
    if grid.ndim != 3 or min(grid.shape) < 2:
        raise ViewerError(f"Expected a 3D volume on the model grid, not shape {grid.shape}")
    if not np.isfinite(grid).all():
        raise ViewerError("The scan contains nonfinite values")
    preparation = summary["preparation"]
    if list(grid.shape) != list(preparation["grid_shape"]):
        raise ViewerError("The volume does not have the grid shape recorded in the analysis summary")
    low, high = (float(value) for value in preparation["intensity_window"])
    if not high > low:
        raise ViewerError("Degenerate intensity window in the analysis summary")
    anatomy = np.rint(np.clip((grid.astype(np.float64) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    spacing = [float(value) for value in preparation["grid_spacing_mm"]]

    explanation = summary.get("grad_cam") if cam_grid is not None else None
    attention = None
    no_map_text = "No attention map is available for this analysis, so only the scan on the model's grid is shown."
    if explanation is not None:
        cam_grid = np.asarray(cam_grid, dtype=np.float64)
        if cam_grid.shape != grid.shape:
            raise ViewerError("The attention volume and the scan are not on the same grid")
        if not np.isfinite(cam_grid).all():
            raise ViewerError("The attention volume contains nonfinite values")
        strongest = float(cam_grid.max())
        if strongest > 0:
            attention = np.rint(np.clip(cam_grid / strongest, 0, 1) * 255).astype(np.uint8)
        else:
            explanation = None  # an all-zero map has nothing to show
            no_map_text = "Grad-CAM produced no attention for this scan, so only the scan on the model's grid is shown."

    explained = [int(z) for z in explanation["slice_indices"]] if explanation is not None else []
    if explanation is not None and (not explained or explained != sorted(set(explained))):
        raise ViewerError("The analysis summary does not list the explained slices in ascending order")
    bounds = _crop_bounds(anatomy, attention, (explained[0], explained[-1]) if explained else None)
    window = tuple(slice(start, stop) for start, stop in bounds)
    offset = [start for start, _ in bounds]
    anatomy = np.ascontiguousarray(anatomy[window])
    class_names = [str(name) for name in summary["class_names"]]
    if len(class_names) != 2:
        raise ViewerError("The viewer expects a two-class analysis")
    scored = _scored_slices(summary)
    payload = {
        "shape": list(anatomy.shape),
        "offset": offset,
        "spacing": spacing,
        "affine": [[float(value) for value in row] for row in preparation["grid_affine"]],
        "encoding": "base64 of zlib; uint8, x fastest then y then z; each row as differences along x modulo 256",
        "anatomy_z": _pack(anatomy),
        "attention_z": None,
        "checksums": [int(anatomy.sum(dtype=np.int64)), 0],
        "class_names": class_names,
        "probabilities": [float(value) for value in summary["probabilities"]],
        "member_probabilities": [[float(value) for value in row] for row in summary["member_probabilities"]],
        "predicted_label": str(summary["predicted_label"]),
        "threshold": float(summary["threshold_on_positive_class_probability"]),
        "notice": str(summary["notice"]),
        "warnings": [str(text) for text in summary.get("warnings", []) if text != summary["notice"]],
        "target_class": None,
        "scored": scored,
        "scored_text": f"The score uses {_slices(len(scored))}.",
        "no_map_text": no_map_text,
    }
    if explanation is None:
        return payload

    attention = np.ascontiguousarray(attention[window])
    first, last = explained[0], explained[-1]
    peak = np.unravel_index(int(attention.argmax()), attention.shape)
    payload["attention_z"] = _pack(attention)
    payload["checksums"][1] = int(attention.sum(dtype=np.int64))
    payload.update(
        {
            "target_class": str(explanation["target_class"]),
            "explained": explained,
            "slab_grid": [first, last],
            "slab": [first - offset[2], last - offset[2]],
            "peak": [int(value) for value in peak],
            "slice_share": [float(value) for value in explanation["slice_share_of_attention"]],
            "slice_positive_probability": [float(row[1]) for row in explanation["slice_probabilities"]],
        }
    )
    payload.update(_map_texts(scored, explained, payload["target_class"], spacing[2]))
    if len(payload["slice_share"]) != len(explained) or len(payload["slice_positive_probability"]) != len(explained):
        raise ViewerError("The analysis summary is inconsistent")
    return payload


def _embed_json(payload: dict) -> str:
    """JSON that is safe inside a ``<script>`` element whatever text the analysis carries.

    No ``<``, ``>`` or ``&`` survives, so nothing in it can close the element,
    open a comment or start another tag; a JSON parser reads the escapes back
    as the original characters.
    """

    try:
        text = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    except ValueError as exc:
        raise ViewerError("The analysis summary contains nonfinite numbers") from exc
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _render(payload: dict, *, title: Optional[str] = None, heading: str, fragment: bool) -> str:
    if heading not in _SUBHEADING:
        raise ViewerError("heading must be h1, h2 or h3")
    if title is None:
        title = DEFAULT_TITLE if payload.get("attention_z") else NO_MAP_TITLE
    try:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ViewerError(f"The viewer template {TEMPLATE_PATH.name} cannot be read") from exc
    if fragment:
        start, end = template.index(_FRAGMENT_START) + len(_FRAGMENT_START), template.index(_FRAGMENT_END)
        template = template[start:end].strip() + "\n"
    values = {
        # Numeric character references keep the page ASCII whatever the title holds.
        "__VV_TITLE__": html.escape(title).encode("ascii", "xmlcharrefreplace").decode("ascii"),
        "__VV_HEADING__": heading,
        "__VV_SUBHEADING__": _SUBHEADING[heading],
        "__VV_DATA__": _embed_json(payload),
    }
    missing = [name for name in values if name not in template]
    if missing:
        raise ViewerError(f"The viewer template lacks {', '.join(missing)}")
    # One pass over the template: text that goes in is never scanned for placeholders again.
    return re.sub("|".join(re.escape(name) for name in values), lambda match: values[match.group(0)], template)


def _inputs(result) -> tuple:
    return result.prepared.grid, result.cam_grid, result.summary()


def build_viewer(result, *, title: Optional[str] = None) -> str:
    """One self-contained HTML document for a :class:`volume_analysis.VolumeAnalysis`."""

    return _render(viewer_payload(*_inputs(result)), title=title, heading="h1", fragment=False)


def viewer_fragment(result=None, *, payload: Optional[dict] = None, title: Optional[str] = None, heading: str = "h2") -> str:
    """Style, markup and script only, for embedding the viewer in another page."""

    if payload is None:
        payload = viewer_payload(*_inputs(result))
    return _render(payload, title=title, heading=heading, fragment=True)


def _write_whole(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that the file is either the complete new page or left as it was."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


def write_viewer(result, path) -> Path:
    """Write the viewer for ``result`` to ``path`` and return the path."""

    path = Path(path)
    _write_whole(path, build_viewer(result))
    return path


def _saved_volume(path: Path, grid_shape: Sequence[int], grid_affine: np.ndarray) -> np.ndarray:
    """Voxels of a saved NIfTI in canonical RAS+ order, checked against the grid in ``result.json``.

    The page takes its geometry from ``result.json``, so a file that another
    tool re-saved in a different voxel order is reoriented first, and one that
    sits on a different grid is refused rather than drawn in the wrong place.
    """

    try:
        from src.mri_volume import load_nifti
    except ModuleNotFoundError:
        from mri_volume import load_nifti

    try:
        volume = load_nifti(path)
    except Exception as exc:  # nibabel raises its own error types for damaged files
        raise ViewerError(f"{path.name} cannot be read ({type(exc).__name__}: {exc})") from exc
    same_grid = tuple(volume.shape) == tuple(int(n) for n in grid_shape) and np.allclose(
        volume.affine, grid_affine, rtol=0, atol=_AFFINE_TOLERANCE
    )
    if not same_grid:
        raise ViewerError(
            f"{path.name} is not on the grid recorded in result.json; it was changed after the analysis. "
            "Run the analysis again to get a matching set of files."
        )
    return np.asarray(volume.data, dtype=np.float32)


def payload_from_saved(results_dir) -> dict:
    """The viewer payload rebuilt from a folder written by ``VolumeAnalysis.save``."""

    results_dir = Path(results_dir)
    summary_path, anatomy_path, map_path = (results_dir / name for name in ("result.json", "anatomy.nii.gz", "gradcam.nii.gz"))
    for required in (summary_path, anatomy_path):
        if not required.is_file():
            raise ViewerError(f"{required} not found; expected a folder written by the volume analysis")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except ValueError as exc:  # invalid JSON or not UTF-8
        raise ViewerError(f"result.json cannot be read ({exc})") from exc
    try:
        preparation = summary["preparation"]
        grid_shape = [int(n) for n in preparation["grid_shape"]]
        grid_affine = np.asarray(preparation["grid_affine"], dtype=np.float64)
        if grid_affine.shape != (4, 4) or len(grid_shape) != 3:
            raise TypeError("grid_shape and grid_affine do not describe a 3D grid")
        grid = _saved_volume(anatomy_path, grid_shape, grid_affine)
        cam_grid = _saved_volume(map_path, grid_shape, grid_affine) if map_path.is_file() else None
        return viewer_payload(grid, cam_grid, summary)
    except ViewerError:
        raise
    except KeyError as exc:
        raise ViewerError(f"result.json lacks {exc}") from exc
    except (IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ViewerError(f"result.json is not in the form the volume analysis writes ({type(exc).__name__}: {exc})") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the interactive 3D viewer from saved volume-analysis outputs.")
    parser.add_argument("results", help="Folder holding result.json, anatomy.nii.gz and gradcam.nii.gz")
    parser.add_argument("--output", help=f"HTML file to write (default: <results>/{VIEWER_FILENAME})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = Path(args.output) if args.output else Path(args.results) / VIEWER_FILENAME
    try:
        _write_whole(output, _render(payload_from_saved(args.results), heading="h1", fragment=False))
    except (ViewerError, OSError) as exc:
        logger.error("Cannot build the viewer from %s: %s", args.results, exc)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
