# Kevin dementia paper: portable publication materials

Six runs completed 305 epochs. The individual new models matched or slightly
underperformed the old image benchmark. A fixed stage ensemble corrected one
additional primary image, with an uncertainty interval including zero. Clinical
generalization was weak; these results do not establish an overall stronger or
clinically ready model. The production checkpoint remains unchanged.

- [Results overview](reports/Overview.md)
- [Complete six-run results and original 100-case comparison](reports/Full_Results.md)
- [Completed Methods](reports/Completed_Methods.md) and [prespecified Methods](reports/Prespecified_Methods.md)
- [Statistics and ensemble handoff](reports/Statistical_Handoff.md)
- [All figures](FIGURES.md)
- [All quantitative, stress and attribution-control graphs in one PDF](All_Graphs_Atlas.pdf)
- [All 115 old/new Grad-CAM case panels](figures/attribution/index.html)
- [Dataset sources, licenses, omissions and reproducibility](DATA_AND_REPRODUCIBILITY.md)
- [Reconstruction and exact-run boundaries](RECONSTRUCTION.md)
- [Attribution findings and replay discrepancies](figures/attribution/README.md)
- [Post-hoc stress findings](figures/robustness/FINDINGS.md)
- [Historical readiness assessment](reports/Readiness_Protocol_Snapshot.txt)
- [Aggregate retrieval and cleanup evidence](evidence/completion_summary.json)
- [Software/device versions relevant to numerical replay](evidence/runtime_versions.json)
- [Six research checkpoints and their hashes](../../../models/kevin_20261011/README.md)

The audited local sources remain unchanged. These public copies normalize links
and omit private clinical rows and operational metadata; original-source and
export hashes are separately recorded in EXPORT_MANIFEST.json. Git tracking of
legacy benchmark images is not proof of dataset license compatibility.

Attribution preserved all 230 original maps' predictions/logits, while 2 of 144
randomized maps failed or were flat. CPU replay preserved all 200 original-100
classes but 43 probability vectors exceeded the preset tolerance. The broader
stress replay also preserved classes but showed numerical probability differences.
Candidate 3 fell from 100/100 on local clean original-100 inputs to 96/100 under
the tested stronger noise setting, while the old model remained 100/100 throughout
that fixed grid. The new candidate was not uniformly more robust. These synthetic
image changes are post-hoc diagnostics, not clinical safety or acquisition tests.

Each seed's original loss plot, accuracy plot and unsmoothed history CSV is linked
in the [six-run results table](reports/Full_Results.md#all-six-training-runs-and-raw-curves).

| Additional diagnostic family | Export status |
| --- | --- |
| quantitative | complete output present |
| attribution | complete output present |
| robustness | complete output present |
| clean_replay | complete output present |

Rebuild with `python3 docs/kevin_paper_20261011/publication/export_publication.py --require-complete`.
STAGING_ALLOWLIST.txt is an explicit candidate list, not a command to bulk-stage
the working directory. Large research models require Git LFS; no cloud action is
performed by this exporter.
