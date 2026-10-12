# Corner and border reliance audit

The fixed validation controls **do not support dominant input-corner reliance** by either the old model or the validation-selected new model, candidate 3. This is limited evidence from a development sample, not proof that every possible shortcut is absent or that the model is clinically robust.

Open the [native 7×7 gallery for all 115 fixed cases](results/native_gallery.html), [native score-location summary](results/native_location_mass.pdf), [validation masking comparison](results/validation_masking_controls.pdf), or [explicit input masks](results/input_mask_definitions.pdf).

## What was measured

The attribution analysis reuses and verifies all 230 original native arrays: old plus candidate 3 for original100, eight historical class-balanced examples, and seven historically selected gallery examples. No original100 or held-out test model inference was performed in this audit. Native arrays are unchanged; the gallery shows the unsmoothed 7×7 cells beside a clearly labeled display overlay. Cyan outlines identify the four native corner cells. Neither the outline nor interpolation changes any measured attribution value.

The fresh inference sample contains 160 **new-model validation** images, 40 per class and one per prepared hash group, selected by a deterministic SHA256 ordering with seed 20261012. Old-baseline historical strict exposure is 99 training, 23 validation, and 38 test images. Therefore old/new comparisons here are descriptive. Hash groups are not patient identities.

Fixed controls erase the four 32×32 corners or outer 16-pixel border of the normalized 224×224 input, or retain only those regions. Each is tested with black and ImageNet-mean fill. Matched central controls remove the exact same number of pixels, chosen by increasing radial distance from image center. These controls use approximately disk-shaped central masks, not a crop. The protocol and selected sample were written before this inference. No training, checkpoint changes, or checkpoint selection occurred.

## Findings

All 160 clean validation predictions were correct for both models.

| Control | Old: black fill | Old: mean fill | New c3: black fill | New c3: mean fill |
|---|---:|---:|---:|---:|
| Erase corners | 100% | 100% | 100% | 100% |
| Corners only | 25% | 26.25% | 25% | 25% |
| Erase equally sized center (8.16% pixels) | 86.25% | 95% | 63.125% | 81.25% |
| Erase border | 100% | 98.75% | 100% | 100% |
| Border only | 25.625% | 23.125% | 21.875% | 25% |
| Erase equally sized center (26.53% pixels) | 41.25% | 70% | 25% | 35.625% |

For candidate 3, mean probability total variation was 0.0005–0.0027 after corner erasure and 0.0013–0.0066 after border erasure. Erasing central content caused much larger changes. These artificial central occlusions also harmed candidate 3 more than the old model, so this audit does not establish that the new model is stronger under every perturbation.

On reused original100, mean native corner score mass was **2.90% old / 6.51% new**, compared with 8.16% of native grid area. Mean outer-ring mass was **25.16% / 41.66%**, compared with 48.98% of grid area. No native map's maximum occurred in a corner cell among the 230 maps. These area references are descriptive, not statistical null hypotheses. The new maps are more spatially diffuse; a corner-colored overlay alone does not establish causal reliance on input-corner pixels. Native receptive fields are broad and overlap, and per-map normalized scores are not anatomical probabilities.

A nonblack display-content proxy (mean RGB >0.08) occurred in corner masks for 23/160 images and border masks for 88/160. Some regions were entirely nonblack. This proxy includes bright padding and is **not a brain segmentation**. It does not justify assuming that all border content is dispensable.

## Consequence for the next training round

This audit does not justify cropping the field of view, forcing Grad-CAM toward the center, or aggressive corner removal. Preserve the complete image and prioritize the separately documented intensity/noise/blur robustness deficits. If background counterfactual augmentation is included, restrict it conservatively to identified low-intensity background rather than unconditionally deleting visible border content, and retain clean validation performance as a guard. Determine all choices using training/validation data; keep original100 and test results out of model selection.

## Reproduce and verify

Use the existing PyTorch research environment. From the repository root:

```bash
MPLCONFIGDIR=/tmp/kevin-corner-mpl XDG_CACHE_HOME=/tmp/kevin-corner-cache \
python docs/kevin_paper_20261011/round2_corner_audit/corner_audit.py --device mps --output docs/kevin_paper_20261011/round2_corner_audit/results
python docs/kevin_paper_20261011/round2_corner_audit/verify_and_plot.py
```

The inference output must be a new directory; the supplied results already exist and are preserved. A CPU option is available. `verify_and_plot.py` reads the default results directory and does no inference. It recomputes all 26 metric records from all 4,160 exported probability rows, checks source/checkpoint/native-array hashes, and validates mask areas and complete case alignment. The saved state checks confirm both checkpoints and in-memory model states were unchanged and no gradients accumulated. Gallery panels and summary plots were visually inspected; PDF page/text bounds passed.

Machine-readable records: [pre-inference protocol](results/protocol_before_inference.json), [validation sample](results/validation_sample.csv), [probabilities](results/validation_predictions.csv), [native map measures](results/native_map_mass.csv), [summary](results/summary.json), [verification](results/verification.json), and [visual QA](results/visual_qa.json). Existing paper figures and frozen training/evaluation sources were not modified.
