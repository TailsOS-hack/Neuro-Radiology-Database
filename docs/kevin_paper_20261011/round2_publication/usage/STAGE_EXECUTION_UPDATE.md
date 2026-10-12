<!-- Portable documentation copy; frozen original retained unchanged. -->
# Execution update after the frozen stage protocol

This note supersedes the local MPS attribution example in the preserved, hash-pinned README. No frozen runtime code or probability tolerance is changed.

Run attribution on the same cloud CUDA GPU after the completed stage evaluation, using the unchanged frozen script and the explicit `attribution_input_allowlist.txt`. The historical 230 map arrays and all 115 case identities remain unchanged. The three stage selections must be frozen before those inputs are installed. A probability mismatch at the original tolerance remains a blocked replay-review result; do not widen the tolerance.

From the cloud repository `/tmp/Neuro-Radiology-Database`:

```sh
/tmp/kevin-round2-venv/bin/python \
  docs/kevin_paper_20261011/round2_evaluation/round2_attribution.py \
  --data-root /tmp/Neuro-Radiology-Database \
  --suite /workspace/kevin_round2_results \
  --output /workspace/kevin_round2_results/round2_attribution --device cuda
```

After the complete suite, evaluation and attribution are retrieved and independently verified, generate only the statistical report locally. Set `STAGE_SUITE` to the verified local copy:

```sh
MPLCONFIGDIR=/tmp/kevin-round2-mpl \
python \
  docs/kevin_paper_20261011/round2_evaluation/report_round2.py \
  --data-root . --suite "$STAGE_SUITE" --output "$STAGE_SUITE/round2_report"
```

Review all 28 quantitative/robustness plots, both attribution summaries, and every one of the 115 returned case panels. Retain every fixed seed and both historical comparators. The stage benchmarks remain previously seen, post-hoc image benchmarks; the separate clinical experiment and fresh patient-level cohort must not be substituted into the four-class staging claims.

After source-figure visual review, assemble the 30-page bookmarked atlas without altering the figures:

```sh
python \
  docs/kevin_paper_20261011/round2_evaluation/assemble_atlas.py \
  --report "$STAGE_SUITE/round2_report" \
  --attribution "$STAGE_SUITE/round2_attribution" \
  --output "$STAGE_SUITE/round2_atlas"
```

Both report and atlas destination directories must be new. The atlas links the separate 115-panel gallery. Keep the previous historical graph atlas linked separately. Record visual QA and destination readback before publication or cloud-storage deletion.
