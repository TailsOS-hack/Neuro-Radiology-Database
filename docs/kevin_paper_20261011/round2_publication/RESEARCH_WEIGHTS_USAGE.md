# Selected research checkpoint export

`export_research_weights.py` is prepared for a later local run after the complete stage and clinical results have been independently retrieved and verified. Preparation of this script is not an export or a claim that clinical training/evaluation finished. It performs no real-image inference, fitting, model selection, Git operation or cloud action.

The exact destination is `models/kevin_round2_20261011`. It contains three `stage_seed<seed>.pt` files, three `clinical_seed<seed>.pt` files and public `CHECKPOINTS.json`. The production `models/alzheimers_classifier.pt` is read only for a preservation hash and never replaced.

Required inputs are the two local result directories, their real `results_archive.py verify` JSON receipts, separately retrieved one-line archive checksum files, and independently recorded SHA256 values of the two frozen evaluation-selection documents. The script checks each receipt's destination, archive digest, manifest membership and every local result byte, then verifies completion, frozen source identities and all six selected checkpoint identities. A hand-written success flag is not a replacement for the actual verifier output. Archives and receipts remain private and are not copied to the model directory; only their integrity hashes and aggregate verification counts enter the public manifest.

After root supplies verified paths, the invocation has this form:

```sh
python docs/kevin_paper_20261011/round2_publication/export_research_weights.py \
  --stage-suite "$STAGE_SUITE" \
  --stage-retrieval-receipt "$STAGE_RETRIEVAL_RECEIPT" \
  --stage-archive-sha256-file "$STAGE_ARCHIVE_SHA256_FILE" \
  --stage-freeze-sha256 "$STAGE_FREEZE_SHA256" \
  --clinical-suite "$CLINICAL_SUITE" \
  --clinical-retrieval-receipt "$CLINICAL_RETRIEVAL_RECEIPT" \
  --clinical-archive-sha256-file "$CLINICAL_ARCHIVE_SHA256_FILE" \
  --clinical-freeze-sha256 "$CLINICAL_FREEZE_SHA256"
```

For stage models, only top-level `optimizer_state`, `scheduler_state` and `history` are removed. All inference state, the unused auxiliary tensor state, architecture/class metadata and aggregate validation metadata remain exactly unchanged. Stage-only auxiliary weights have no clinical interpretation and are never used by the equality test. Full training checkpoints remain in the verified private archives. The historical loader's `load()` remains compatible; the old Round 1 evaluation command's additional history-based selection check is not an export verification interface.

Clinical selected `best.pt` files are copied byte-for-byte if their wrapper contains only the documented model tensors and aggregate metadata. Any optimizer, unknown field, individual patient key or patient/image path causes rejection rather than silent sanitization. Neither patient manifests nor per-patient predictions are exported. Both endpoint wrappers are loaded safely and all retained tensor dtype, shape and raw bytes plus all retained metadata are compared exactly.

Each source/export pair then uses the existing stage loader or clinical model builder in CPU evaluation mode with one thread and deterministic algorithms. Inputs are fixed-seed synthetic standard-normal tensors only. All stage logits/probabilities, or clinical `softmax(mean(nine slice logits))` outputs, must match bit-for-bit. Model states must remain unchanged by the evaluation. This verifies serialization fidelity; it is not additional medical validation or an export to a clinically approved runtime.

Every prospective file is prepared and checked before destination writes. Existing files are accepted only when their bytes match the prepared output. Different or unexpected files are refused. The six weights publish before `CHECKPOINTS.json`, and every destination is read back. A failed or interrupted export must be inspected; it never authorizes replacing an existing different output.

Before committing model artifacts, root must add an explicit Git LFS rule for `models/kevin_round2_20261011/*.pt`, confirm all six paths receive the LFS filter, inspect the exact staging list and verify remote LFS upload. The script does not edit `.gitattributes` or call Git. Keep the clinical/research-use limits with the published weights, and publish all fixed seeds rather than choosing one from test outcomes.
