"""Checks that the viewer's tests can fail.

Each deliberate defect listed below is applied to a temporary copy of ``src/``
and of the viewer tests, and ``tests/test_volume_viewer.py`` is run on that
copy. A defect that leaves the tests green is a hole in the tests. Nothing in
the repository is changed.

    python scripts/check_viewer_tests.py                    # every defect (several minutes)
    python scripts/check_viewer_tests.py --only mirrored    # defects whose name contains the text
    python scripts/check_viewer_tests.py --list

Most defects are in the page itself and are only seen by the browser tests,
which need Node.js with Playwright and its Chromium; without them the script
stops and says so. Exit code 0 means every defect was caught.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE, MODULE = "src/volume_viewer_template.html", "src/volume_viewer.py"

#: (what is wrong, [(file, text to replace, replacement), ...]). Every text must occur exactly once in its file.
DEFECTS = [
    # ---- text from the analysis must stay text
    ("notice inserted as markup", [(PAGE, '$("vv-notice").textContent = data.notice;', '$("vv-notice").innerHTML = data.notice;')]),
    ("warnings inserted as markup", [(PAGE, "      item.textContent = text;\n      list.append(item);", "      item.innerHTML = text;\n      list.append(item);")]),
    ("ampersand not escaped in the embedded data", [(MODULE, '.replace("&", "\\\\u0026")', "")]),
    ("a block of the summary embedded that nobody chose to publish", [(MODULE, '        "no_map_text": no_map_text,\n    }', '        "no_map_text": no_map_text,\n        "members": summary.get("members"),\n    }')]),
    # ---- where things are drawn
    ("slice views mirrored", [(PAGE, "    g.drawImage(plane.buffer, x, y, w, h);", "    g.save(); g.translate(x + w, y); g.scale(-1, 1); g.drawImage(plane.buffer, 0, 0, w, h); g.restore();")]),
    ("left and right letters swapped", [(PAGE, "    g.fillText(plane.edges[0], 8 * ratio, H / 2);\n    g.fillText(plane.edges[1], W - 8 * ratio, H / 2);", "    g.fillText(plane.edges[1], 8 * ratio, H / 2);\n    g.fillText(plane.edges[0], W - 8 * ratio, H / 2);")]),
    ("top and bottom letters swapped", [(PAGE, "    g.fillText(plane.edges[2], W / 2, 8 * ratio);\n    g.fillText(plane.edges[3], W / 2, H - 8 * ratio);", "    g.fillText(plane.edges[3], W / 2, 8 * ratio);\n    g.fillText(plane.edges[2], W / 2, H - 8 * ratio);")]),
    ("map drawn mirrored over the scan", [(PAGE, "        const t = attention ? attention[q] : 0;", "        const t = attention ? attention[q - 2 * (q % nx) + nx - 1] : 0;")]),
    ("slab levels drawn without the crop offset", [(PAGE, "      const top = y + ((nz - 1 - data.slab[1]) / nz) * h;", "      const top = y + ((nz - 1 - data.slab_grid[1]) / nz) * h;")]),
    ("3D view drawn as a mirror image", [(PAGE, "      right: [cy, sy, 0],", "      right: [-cy, -sy, 0],")]),
    ("3D projection not drawn", [(PAGE, "    g.drawImage(view3d.buffer, x0, y0, side, side);\n", "")]),
    ("sagittal view said to be seen from the left", [(PAGE, "seen from the patient's right side", "seen from the patient's left side")]),
    # ---- numbers and positions
    ("other class shown as the positive probability", [(PAGE, "fmt(data.probabilities[1], 3)", "fmt(data.probabilities[0], 3)")]),
    ("millimetres without the crop offset", [(PAGE, "const i = index[0] + OFF[0], j = index[1] + OFF[1], k = index[2] + OFF[2];", "const i = index[0], j = index[1], k = index[2];")]),
    ("slice numbers without the crop offset", [(PAGE, 'infos[axis].textContent = "slice " + (state.idx[axis] + OFF[axis]);', 'infos[axis].textContent = "slice " + state.idx[axis];')]),
    ("bars drawn in reverse order", [(PAGE, "y: yOf(shares[i] * 100), height: Math.max(0, yOf(0) - yOf(shares[i] * 100))", "y: yOf(shares[count - 1 - i] * 100), height: Math.max(0, yOf(0) - yOf(shares[count - 1 - i] * 100))")]),
    ("table columns swapped", [(PAGE, 'fmt(data.slice_share[i] * 100, 2) + "%", fmt(data.slice_positive_probability[i], 3)', 'fmt(data.slice_positive_probability[i], 3), fmt(data.slice_share[i] * 100, 2) + "%"')]),
    ("zero attention reported between mapped slices", [(PAGE, 'if (explained.has(gridZ)) return "attention "', 'if (gridZ >= data.slab_grid[0] && gridZ <= data.slab_grid[1]) return "attention "')]),
    ("probabilities reversed", [(MODULE, '"probabilities": [float(value) for value in summary["probabilities"]],', '"probabilities": [float(value) for value in summary["probabilities"]][::-1],')]),
    ("member probabilities reversed", [(MODULE, '[[float(value) for value in row] for row in summary["member_probabilities"]]', '[[float(value) for value in row][::-1] for row in summary["member_probabilities"]]')]),
    ("threshold halved", [(MODULE, 'float(summary["threshold_on_positive_class_probability"])', 'float(summary["threshold_on_positive_class_probability"]) / 2')]),
    ("slice shares reversed", [(MODULE, '[float(value) for value in explanation["slice_share_of_attention"]]', '[float(value) for value in explanation["slice_share_of_attention"]][::-1]')]),
    ("per-slice probabilities shifted", [(MODULE, '[float(row[1]) for row in explanation["slice_probabilities"]]', '[float(row[1]) for row in explanation["slice_probabilities"]][1:] + [0.5]')]),
    ("map not rescaled to its strongest voxel", [(MODULE, "np.rint(np.clip(cam_grid / strongest, 0, 1) * 255)", "np.rint(np.clip(cam_grid, 0, 1) * 255)")]),
    ("band model said to score the protocol slices only", [(MODULE, "return list(range(protocol[0], protocol[-1] + 1)) if band else protocol", "return protocol")]),
    ("one slice worded as several", [(MODULE, "    single = len(explained) == 1\n", "    single = False\n")]),
    # ---- controls
    ("chart bar goes to another slice", [(PAGE, "setIndex(2, localZ(slices[i]))", "setIndex(2, slices[i])")]),
    ("strongest-attention button goes to the corner", [(PAGE, '$("vv-peak").addEventListener("click", () => goTo(data.peak));', '$("vv-peak").addEventListener("click", () => goTo([0, 0, 0]));')]),
    ("opacity slider does nothing", [(PAGE, "    state.opacity = Number(opacity.value) / 100;\n", "")]),
    ("hide-below slider does nothing", [(PAGE, "    state.floor = Number(floor.value) / 100;\n", "")]),
    ("horizontal drag turns the head against the pointer", [(PAGE, "state.yaw = drag.yaw - (event.clientX - drag.x) * 0.01;", "state.yaw = drag.yaw + (event.clientX - drag.x) * 0.01;")]),
    ("views swallow touch scrolling", [(PAGE, "touch-action: pan-y pinch-zoom;", "touch-action: none;")]),
    ("a second finger moves the crosshair", [(PAGE, "      if (!event.isPrimary) { drag = null; return; }\n", "")]),
    ("head jumps when a resting finger starts to drag", [(PAGE, "        Object.assign(drag, { active: true, x: event.clientX, y: event.clientY, yaw: state.yaw, pitch: state.pitch });", "        drag.active = true;")]),
    ("a tap does not stop the turning", [(PAGE, "      if (drag.touch && !drag.active) takeOver(); // a tap stops the turning, as a mouse press does\n", "")]),
    ("rotation keeps running off screen", [(PAGE, "const canTick = () => state.spin && seen > 0 && !document.hidden && Boolean(anatomy);", "const canTick = () => state.spin && !document.hidden && Boolean(anatomy);")]),
    # ---- layout
    ("captions of the four views differ in height", [(PAGE, "@media (max-width: 1019px) {\n  .vv-caption { flex-direction: column; gap: 0; }", "@media (max-width: 519px) {\n  .vv-caption { flex-direction: column; gap: 0; }")]),
    ("views and their slider do not share a screen", [(PAGE, "@media (min-width: 620px) { .vv-views {", "@media (min-width: 700px) { .vv-views {")]),
    # ---- a page that cannot be right
    ("draws despite a checksum mismatch", [(PAGE, '      fail("The scan embedded in this page does not match its checksum, so nothing is shown. Build the page again.");\n      return;', '      $("vv-status").textContent = "does not match its checksum";')]),
    ("map checksum not checked", [(PAGE, "if (sums[0] !== data.checksums[0] || sums[1] !== data.checksums[1]) {", "if (sums[0] !== data.checksums[0]) {")]),
    ("checksum blind to the order of the voxels", [
        (PAGE, "    let crc = 0xffffffff;\n    for (let i = 0; i < volume.length; i++) crc = CRC[(crc ^ volume[i]) & 255] ^ (crc >>> 8);\n    return (crc ^ 0xffffffff) >>> 0;", "    let crc = 0;\n    for (let i = 0; i < volume.length; i++) crc += volume[i];\n    return crc;"),
        (MODULE, "    return zlib.crc32(np.ascontiguousarray(volume.transpose(2, 1, 0)).tobytes())", "    return int(volume.sum(dtype=np.int64))"),
    ]),
    ("failed page keeps its views and controls", [(PAGE, '    for (const node of document.querySelectorAll("#vv .vv-views, #vv .vv-readout, #vv .vv-controls, #vv .vv-charts, #vv .vv-read")) node.hidden = true;\n', "")]),
    ("page without a map crashes", [(PAGE, "const home = hasMap ? data.peak : state.idx.slice();", "const home = data.peak.slice();")]),
    # ---- what goes into the page
    ("crop ignores attention outside the head", [(MODULE, "        occupied |= attention > 0\n", "        pass\n")]),
    ("crop ignores the slab", [(MODULE, "start, stop = min(start, int(keep_z[0])), max(stop, int(keep_z[1]) + 1)", "pass")]),
    ("slices outside the grid accepted", [(MODULE, "        if explained[0] < 0 or explained[-1] >= grid.shape[2]:\n            raise ViewerError(\"The analysis summary lists explained slices outside the grid\")\n", "")]),
    ("attention on unlisted slices accepted", [(MODULE, "        if not {int(z) for z in np.flatnonzero(attention.any(axis=(0, 1)))} <= set(explained):", "        if False:")]),
    ("saved volumes read without their headers", [(MODULE, "        volume.affine, grid_affine, rtol=0, atol=_AFFINE_TOLERANCE\n    )", "        grid_affine, grid_affine, rtol=0, atol=_AFFINE_TOLERANCE\n    )")]),
    ("an infinite number in result.json ends in a traceback", [(MODULE, "AttributeError, ArithmeticError, RecursionError) as exc:", "AttributeError, RecursionError) as exc:")]),
]


def run_tests(work: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_volume_viewer"], cwd=work, capture_output=True, text=True, timeout=1800
    )


def fresh_copy(work: Path) -> None:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(REPO / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (work / "tests").mkdir()
    for name in ("test_volume_viewer.py", "viewer_browser_check.js"):
        shutil.copy(REPO / "tests" / name, work / "tests" / name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Apply deliberate defects to a copy of the viewer and check that its tests fail.")
    parser.add_argument("--only", help="Run only the defects whose name contains this text")
    parser.add_argument("--list", action="store_true", help="List the defects and exit")
    args = parser.parse_args(argv)
    chosen = [defect for defect in DEFECTS if not args.only or args.only.lower() in defect[0].lower()]
    if args.list:
        for name, _ in chosen:
            print(name)
        return 0

    with tempfile.TemporaryDirectory(prefix="viewer_defects_") as scratch:
        work = Path(scratch) / "copy"
        fresh_copy(work)
        baseline = run_tests(work)
        summary = baseline.stderr.strip().splitlines()[-1] if baseline.stderr.strip() else ""
        if baseline.returncode != 0:
            print(f"The unchanged copy does not pass its tests ({summary}); fix that first.")
            return 2
        if "skipped" in summary:
            print(f"Some viewer tests were skipped here ({summary}). The browser tests need Node.js with Playwright and its "
                  "Chromium, and the saved-folder tests need nibabel; without them most defects cannot be judged.")
            return 2
        print(f"unchanged copy: {summary}")

        survivors = []
        for name, edits in chosen:
            fresh_copy(work)
            applied = True
            for target, old, new in edits:
                text = (work / target).read_text(encoding="utf-8")
                if text.count(old) != 1:
                    applied = False
                    break
                (work / target).write_text(text.replace(old, new), encoding="utf-8")
            if not applied:
                print(f"NOT APPLIED  {name}: the code it replaces was not found exactly once; update this list")
                survivors.append(name)
                continue
            caught = run_tests(work).returncode != 0
            print(f"{'caught      ' if caught else 'NOT CAUGHT  '} {name}", flush=True)
            if not caught:
                survivors.append(name)

    print(f"{len(chosen) - len(survivors)} of {len(chosen)} defects were caught.")
    if survivors:
        print("Not caught: " + "; ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
