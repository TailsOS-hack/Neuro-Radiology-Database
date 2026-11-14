"""Interactive viewer for a volume analysis: three linked slice views and a rotating 3D projection.

``build_viewer`` turns a :class:`volume_analysis.VolumeAnalysis` (or the files
it saved) into one HTML file that embeds the scan on the model's grid and the
stacked Grad-CAM map. The file needs no server, no network and no other file:
open it in a browser. ``VolumeAnalysis.save`` writes it as ``viewer.html`` next
to the NIfTI volumes.

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
import io
import json
import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

VIEWER_FILENAME = "viewer.html"
DEFAULT_TITLE = "3D attention map"
TEMPLATE_PATH = Path(__file__).with_name("volume_viewer_template.html")
_FRAGMENT_START, _FRAGMENT_END = "<!--vv:fragment-start-->", "<!--vv:fragment-end-->"
#: Voxels kept around the head when the display volume is cropped to it.
_CROP_MARGIN = 3
#: Display level (0-255) above which a voxel counts as part of the head for cropping.
_FOREGROUND_LEVEL = 12


class ViewerError(ValueError):
    """The analysis outputs cannot be turned into a viewer."""


def _sprite(volume: np.ndarray) -> tuple:
    """Tile the axial slices of a uint8 (X, Y, Z) volume into one lossless PNG.

    Slice ``z`` is the tile at column ``z % columns``, row ``z // columns``;
    inside a tile, pixel (column x, row y) is voxel (x, y, z).
    """

    nx, ny, nz = volume.shape
    columns = max(1, math.ceil(math.sqrt(nz * ny / nx)))  # roughly square sheet
    rows = math.ceil(nz / columns)
    sheet = np.zeros((rows * ny, columns * nx), dtype=np.uint8)
    for z in range(nz):
        top, left = (z // columns) * ny, (z % columns) * nx
        sheet[top : top + ny, left : left + nx] = volume[:, :, z].T
    buffer = io.BytesIO()
    Image.fromarray(sheet, mode="L").save(buffer, format="PNG", compress_level=9)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), columns, rows


def _crop_bounds(anatomy: np.ndarray, attention: Optional[np.ndarray]) -> tuple:
    occupied = anatomy > _FOREGROUND_LEVEL
    if attention is not None:
        occupied |= attention > 0
    if not occupied.any():
        return tuple((0, n) for n in anatomy.shape)
    bounds = []
    for axis, size in enumerate(anatomy.shape):
        along = np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append((max(0, int(along[0]) - _CROP_MARGIN), min(size, int(along[-1]) + _CROP_MARGIN + 1)))
    return tuple(bounds)


def viewer_payload(grid: np.ndarray, cam_grid: Optional[np.ndarray], summary: dict) -> dict:
    """Everything the page needs, as JSON-serialisable values.

    ``grid`` is the scan on the model grid (RAS+), ``cam_grid`` the attention
    volume on the same grid in [0, 1] (or None), ``summary`` the dict from
    ``VolumeAnalysis.summary()`` / ``result.json``.
    """

    grid = np.asarray(grid)
    if grid.ndim != 3 or min(grid.shape) < 2:
        raise ViewerError(f"Expected a 3D volume on the model grid, not shape {grid.shape}")
    preparation = summary["preparation"]
    if list(grid.shape) != list(preparation["grid_shape"]):
        raise ViewerError("The volume does not have the grid shape recorded in the analysis summary")
    low, high = (float(value) for value in preparation["intensity_window"])
    if not high > low:
        raise ViewerError("Degenerate intensity window in the analysis summary")
    anatomy = np.rint(np.clip((grid.astype(np.float64) - low) / (high - low), 0, 1) * 255).astype(np.uint8)

    explanation = summary.get("grad_cam") if cam_grid is not None else None
    attention = None
    if explanation is not None:
        cam_grid = np.asarray(cam_grid)
        if cam_grid.shape != grid.shape:
            raise ViewerError("The attention volume and the scan are not on the same grid")
        if not np.isfinite(cam_grid).all():
            raise ViewerError("The attention volume contains nonfinite values")
        attention = np.rint(np.clip(cam_grid, 0, 1) * 255).astype(np.uint8)
        if not attention.any():
            attention, explanation = None, None  # an all-zero map has nothing to show

    bounds = _crop_bounds(anatomy, attention)
    window = tuple(slice(start, stop) for start, stop in bounds)
    offset = [start for start, _ in bounds]
    anatomy = np.ascontiguousarray(anatomy[window])
    anatomy_png, columns, rows = _sprite(anatomy)
    class_names = list(summary["class_names"])
    positive = class_names[1]
    scored = list(preparation["protocol_slice_indices"])
    payload = {
        "shape": list(anatomy.shape),
        "offset": offset,
        "spacing": [float(value) for value in preparation["grid_spacing_mm"]],
        "affine": [[float(value) for value in row] for row in preparation["grid_affine"]],
        "tiles": [columns, rows],
        "anatomy_png": anatomy_png,
        "attention_png": None,
        "checksums": [int(anatomy.sum(dtype=np.int64)), 0],
        "class_names": class_names,
        "probabilities": [float(value) for value in summary["probabilities"]],
        "member_probabilities": [[float(value) for value in row] for row in summary["member_probabilities"]],
        "predicted_label": summary["predicted_label"],
        "threshold": float(summary["threshold_on_positive_class_probability"]),
        "notice": summary["notice"],
        "warnings": [text for text in summary.get("warnings", []) if text != summary["notice"]],
        "target_class": None,
        "scored_text": "",
    }
    if explanation is None:
        payload["scored_text"] = f"The score uses {len(scored)} axial slices."
        return payload

    attention = np.ascontiguousarray(attention[window])
    explained = [int(z) for z in explanation["slice_indices"]]
    band = "trained-band" in summary.get("aggregation", "")
    if band:
        scored = explained
    first, last = explained[0], explained[-1]
    peak = np.unravel_index(int(attention.argmax()), attention.shape)
    payload["attention_png"], _, _ = _sprite(attention)
    payload["checksums"][1] = int(attention.sum(dtype=np.int64))
    payload.update(
        {
            "target_class": explanation["target_class"],
            "explained": explained,
            "scored": [int(z) for z in scored],
            "slab_grid": [first, last],
            "slab": [first - offset[2], last - offset[2]],
            "peak": [int(value) for value in peak],
            "slice_share": [float(value) for value in explanation["slice_share_of_attention"]],
            "slice_positive_probability": [float(row[1]) for row in explanation["slice_probabilities"]],
            "slab_text": f"slices {first} to {last} of the 1.5 mm grid, {len(explained)} slices",
            "scored_text": (
                f"The score uses every one of the {len(explained)} axial slices of the slab."
                if band
                else f"The score uses {len(scored)} axial slices; the map covers all {len(explained)} slices of the slab between them."
            ),
            "scored_chart_text": (
                "Every slice of the slab counts toward the score."
                if band
                else f"Only the {len(scored)} larger dots are the slices the score is computed from."
            ),
        }
    )
    if positive not in class_names or len(payload["slice_share"]) != len(explained):
        raise ViewerError("The analysis summary is inconsistent")
    return payload


def _embed_json(payload: dict) -> str:
    """JSON that is safe inside a ``<script>`` element whatever text the analysis carries.

    No ``<``, ``>`` or ``&`` survives, so nothing in it can close the element,
    open a comment or start another tag; a JSON parser reads the escapes back
    as the original characters.
    """

    text = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _render(payload: dict, *, title: str, heading: str, fragment: bool) -> str:
    import html

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if fragment:
        start, end = template.index(_FRAGMENT_START) + len(_FRAGMENT_START), template.index(_FRAGMENT_END)
        template = template[start:end].strip() + "\n"
    if heading not in ("h1", "h2", "h3"):
        raise ViewerError("heading must be h1, h2 or h3")
    # The data goes in last so that no placeholder inside it can be touched.
    page = template.replace("__VV_TITLE__", html.escape(title)).replace("__VV_HEADING__", heading)
    return page.replace("__VV_DATA__", _embed_json(payload))


def _inputs(result) -> tuple:
    return result.prepared.grid, result.cam_grid, result.summary()


def build_viewer(result, *, title: str = DEFAULT_TITLE) -> str:
    """One self-contained HTML document for a :class:`volume_analysis.VolumeAnalysis`."""

    return _render(viewer_payload(*_inputs(result)), title=title, heading="h1", fragment=False)


def viewer_fragment(result=None, *, payload: Optional[dict] = None, title: str = DEFAULT_TITLE, heading: str = "h2") -> str:
    """Style, markup and script only, for embedding the viewer in another page."""

    if payload is None:
        payload = viewer_payload(*_inputs(result))
    return _render(payload, title=title, heading=heading, fragment=True)


def write_viewer(result, path) -> Path:
    """Write the viewer for ``result`` to ``path`` and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_viewer(result), encoding="utf-8")
    return path


def payload_from_saved(results_dir) -> dict:
    """The viewer payload rebuilt from a folder written by ``VolumeAnalysis.save``."""

    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise ViewerError("Reading saved volumes needs nibabel (pip install nibabel)") from exc

    results_dir = Path(results_dir)
    summary_path, anatomy_path, map_path = (results_dir / name for name in ("result.json", "anatomy.nii.gz", "gradcam.nii.gz"))
    for required in (summary_path, anatomy_path):
        if not required.is_file():
            raise ViewerError(f"{required} not found; expected a folder written by the volume analysis")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grid = np.asarray(nib.load(str(anatomy_path)).dataobj, dtype=np.float32)
    cam_grid = np.asarray(nib.load(str(map_path)).dataobj, dtype=np.float32) if map_path.is_file() else None
    return viewer_payload(grid, cam_grid, summary)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the interactive 3D viewer from saved volume-analysis outputs.")
    parser.add_argument("results", help="Folder holding result.json, anatomy.nii.gz and gradcam.nii.gz")
    parser.add_argument("--output", help=f"HTML file to write (default: <results>/{VIEWER_FILENAME})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        payload = payload_from_saved(args.results)
    except (ViewerError, KeyError, json.JSONDecodeError, OSError) as exc:
        logger.error("%s", exc if not isinstance(exc, KeyError) else f"result.json lacks {exc}")
        return 2
    output = Path(args.output) if args.output else Path(args.results) / VIEWER_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(payload, title=DEFAULT_TITLE, heading="h1", fragment=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
