# Dementia publication handoff for Abid — October 11, 2026

This is one complete scientific handoff: completed model comparisons, all current and earlier publication graphs, all 115 regenerated diagnostic Grad-CAM panels, native attribution arrays and controls, raw learning curves for every seed, statistics, methods, dataset provenance, and the next-study design. It contains dementia work only.

**The improved research workflow has not produced a clinically superior or clinically ready model.** The completed Round 2 clinical ensemble predicted AD for all 19 fresh Harvard participants, including all 10 controls. Its accuracy was 9/19 versus 10/19 for the old clinical ensemble. The original production model, Round 1 representative c3, and all three Round 2 stage models scored 100/100 on the reused four-class reader benchmark, which contains 66 old-training, nine old-validation and 25 old-test images. These are different endpoints and neither result supports superiority over clinicians in practice.

## Read these first

1. **MODEL_AND_DATASET_COMPARISON.md** — what changed, all key results, actual datasets versus pending access, and the exact training hardware.
2. **FIGURE_CAPTION_GUIDANCE.md** — suggested manuscript/supplement use and scientifically defensible captions.
3. **repository/docs/kevin_paper_20261011/round2_publication/Methods.md**, **RESULTS.md**, and **DATA_AND_LIMITATIONS.md** — detailed completed study methods, all seeds, uncertainty, source limitations and failed checks.
4. **repository/docs/kevin_paper_20261011/round2_publication/combined_graphs/Round2_All_Graphs_Atlas.pdf** — all 49 new graph pages, with adjacent page index.
5. **repository/docs/kevin_paper_20261011/publication/All_Graphs_Atlas.pdf** — the preserved earlier 40-page graph atlas.
6. **FIGURE_INDEX.csv** — every supplied PNG/PDF figure, its scope, format and presentation status.

## Grad-CAM: start with the complete diagnostic gallery

Open **repository/docs/kevin_paper_20261011/round2_publication/artifacts/stage/attribution_diagnostics/gallery.html** locally. It links all 115 regenerated cases, alongside the native 7 × 7 maps and relevant controls. These are explanations of the **legacy four-class stage model**, not of the binary patient-level clinical model. The panels preserve original, earlier and Round 2 comparisons. No map was shifted, cropped, recolored or selected to look more anatomical for this handoff.

The original probability-replay tolerance of 0.0001 was exceeded in 2/100 reader cases (differences about 0.0001177 and 0.0003737), although predicted classes agreed. Those cases, the warning-bearing figures, blocked-check marker, all native arrays, all 72 parameter-randomization records, and numerical summaries remain in the package. A more central-looking heatmap is not proof of biological validity. Use the fixed complete gallery or its preselected historical class-balanced examples, with controls and limitations, rather than selecting only pleasing maps.

The fixed validation corner/outer-border erasures retained 160/160 for the selected representative; corner-only input yielded 25%. This is evidence about those limited controls, not a general declaration that shortcuts are absent.

## File organization and reproducibility

- **repository/docs/kevin_paper_20261011/round2_publication/artifacts/**: latest stage/clinical graphs, numerical plot inputs, all-model scenario tables, native attribution arrays, original100 case-aligned model/reader results, and six raw training-history directories.
- **repository/docs/kevin_paper_20261011/publication/**: earlier round figures, statistical outputs, raw per-seed histories, and reports retained as historical evidence.
- **repository/paper_package/Dementia_Only_2026-10-09/source/**: selected older dementia training/quantitative figures for the chronological record. They are not results for the new checkpoint.
- **repository/docs/kevin_paper_20261011/round3/**: the next research design and its software verification; no Round 3 trained weights, patient outcomes or clinical figures exist yet.
- **repository/models/kevin_round2_20261011/CHECKPOINTS.json**: exact identities for all six research checkpoints. Weight binaries and raw patient MRI are not duplicated in this email package.
- **SOURCE_MANIFEST.json** maps copied/sanitized scientific documents to source SHA256 identities. **FILE_MANIFEST.csv** covers the final payload. The repository source snapshot is **8bfd1dbb6bf6b29b1faf1962e6038c59f9a4f574**.

Repository: https://github.com/TailsOS-hack/Neuro-Radiology-Database/tree/8bfd1dbb6bf6b29b1faf1962e6038c59f9a4f574

Existing figures and numerical results have not been rerun or modified for this delivery. Scientific text derivatives omit operational material; their changes are marked in the source manifest. PNG/PDF figures retain their source bytes and original resolution. Raw clinical images, clinical participant records, signed access agreements and personal correspondence are excluded. The author must still verify source terms and the actual ethics/oversight determination before journal submission; none is invented here.

Round 3 currently has **109 passing software tests, no acquired eligible cohort and no real training**. NACC has acknowledged an application, but approval, actual delivery and structural-MRI entitlement remain unverified. BrainLat and MIRIAD are access candidates, not newly trained cohorts. The absence of new clinical results is deliberate and must not be filled with claims based on synthetic software tests.
