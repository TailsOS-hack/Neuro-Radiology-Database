# Post-hoc attribution audit

This audit recreates all 115 dementia attribution case slots from the earlier paper package: the reused original 100-image reader benchmark, eight previously fixed class-balanced specialist examples, and seven previously selected historical gallery examples. Each panel compares the unchanged old baseline with candidate_3, the stage-only seed 20261012 checkpoint selected using validation before the original evaluation. The extra examples are not independent validation samples.

All 230 original Grad-CAM maps were finite and nonflat. Explanation generation preserved predictions and logits in every case; parameter state, module modes, and checkpoint hashes were also preserved. The eight fixed specialist examples received 96 mask-control comparisons and 144 parameter-randomization comparisons. Two randomized maps failed or were flat and remain recorded as failures, rather than being silently dropped.

The mask results depend on the replacement method: the new model's top-CAM masking was less damaging than scattered random masking on average with ImageNet-mean replacement at each tested fraction, but more damaging with blurred-image replacement. These mixed results do not establish faithful anatomical localization. Native maps are coarse 7-by-7 model-feature attributions; interpolation does not create anatomical precision.

Local CPU replay reproduced all 200 original100 class predictions for the two models, but 43 probability vectors exceeded the preset absolute tolerance of 0.0001: 38 baseline and five candidate_3 cases. Maximum absolute probability differences were 0.001191 and 0.001996, respectively. The tolerance was not widened. The initial abort and the amendment to retain and report discrepancies are documented in [replay_amendment.json](replay_amendment.json). The CPU and original CUDA environments differ; the cause of the differences has not been established.

Open [the case gallery](index.html), [numeric summary](summary.json), [mask controls](perturbation_controls.csv), [randomization records](randomization.csv), and [visual and checksum verification](visual_qa.json). These post-hoc diagnostics do not change the frozen original results, select a different model, or demonstrate clinical readiness.

The audit source is [run_attribution_audit.py](../../../posthoc_explainability/run_attribution_audit.py). It requires the original result layout, the verified baseline and candidate checkpoints, and the already tracked historical JPEG examples. Use a new output directory because the script refuses to overwrite a completed audit:

```sh
python docs/kevin_paper_20261011/posthoc_explainability/run_attribution_audit.py \
  --results-root training_logs/experiments/kevin_20261011/runpod_blackwell_20261011 \
  --output /tmp/kevin-attribution-repeat
```

The committed publication export includes a separate reconstruction guide. The legacy JPEG dataset's provenance and license chain remain unresolved; this audit adds no new raw clinical MRI redistribution.
