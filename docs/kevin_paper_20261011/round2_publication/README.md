# Kevin dementia paper — completed Round 2 evidence

**Clinical superiority and real-world readiness were not demonstrated.** The new clinical ensemble classified all 19 fresh Harvard participants as AD, including all 10 controls: 9/19 correct, compared with 10/19 for the old fixed clinical ensemble. The small external cohort and this failure prevent a clinical-readiness claim. Better reused internal results do not resolve that limitation.

The legacy four-class image task and the separate patient-level AD/control task have different labels, models and evaluation units. The clinical model cannot be scored against the four-class reader labels. On the reused original 100-image task, the old and all three Round 2 stage models each scored 100/100. Those images include known old-development exposure.

- [Results and uncertainty](RESULTS.md)
- [All 49 new graph pages](combined_graphs/Round2_All_Graphs_Atlas.pdf) and [page index](combined_graphs/atlas_index.csv)
- [All 115 regenerated Grad-CAM diagnostic cases](artifacts/stage/attribution_diagnostics/gallery.html)
- [Stage comparison tables](artifacts/stage/report/README.md) and [clinical patient comparison tables](artifacts/clinical/report/README.md)
- [Raw stage training curves](artifacts/stage/training/) and [raw clinical training curves](artifacts/clinical/training/)
- [Methods](Methods.md), [data access and limitations](DATA_AND_LIMITATIONS.md), and [six research checkpoints](../../../models/kevin_round2_20261011/CHECKPOINTS.json)
- [Earlier 40-page graph atlas, preserved](../publication/All_Graphs_Atlas.pdf)

The attribution gallery is explicitly diagnostic: 2/100 probability replays exceeded the original 0.0001 tolerance, with no changed predicted classes. The failure, original maps, full case roster and randomization results are retained. Visual review does not establish anatomical validity or absence of every shortcut.

Training and validation selection finished on RTX PRO 6000 Blackwell. The B300 Blackwell Ultra then ran the frozen clinical comparison. All outputs and original training states were downloaded, checksum-verified and reread locally before the task's GPUs and storage were deleted. The scheduled monitor remains paused. Production classifier weights were not replaced.

This public package includes source code, aggregate statistics, research checkpoints, legacy benchmark comparisons and reviewed figure derivatives. Newly acquired clinical MRI, individual clinical predictions, participant manifests, raw headers, correspondence and cloud connection details are excluded. The findings support transparent exploratory reporting, including the external generalization failure; they do not guarantee journal acceptance.
