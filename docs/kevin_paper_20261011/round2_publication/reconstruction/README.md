# Completed-suite reconstruction integrity helper

These public copies preserve the private operational helper and its synthetic tests byte for byte. `SOURCE_COPY_PROVENANCE.json` records each original source hash. The helper's original docstring describes its private-input role; publishing this source does not publish those inputs or authorize access to them. No actual archive, retrieval receipt, cloud identity, connection information, clinical row or checkpoint is included here.

The helper creates a smaller evaluation copy of a completed clinical training suite only after verifying the full original archive, an independent archive checksum, the actual retrieval receipt, and every original local payload. It calls the unchanged clinical completed-suite validator before packaging and after installation. Only the three prescribed `last.pt` files are omitted from the copy. The complete originals retain their optimizer states; selected `best.pt`, histories and every validation record remain byte-identical. This copy cannot resume training.

Run the self-contained synthetic integrity tests from the repository root:

```sh
python -B -m unittest discover -s docs/kevin_paper_20261011/round2_publication/reconstruction -p test_migrate_completed_clinical.py -v
```

The copied synthetic verification JSON records the original ten-test run using invented archives, checksums and metadata. Its scientific validator is mocked; it does not claim to verify any real clinical experiment. The portable-copy test run is recorded separately. This helper performs no model forward pass, cloud operation or Git operation.

For lawful local reconstruction, provide the pinned clinical trainer, prepared protocol, private 43-patient development manifest, all 387 matching development PNGs and pinned ImageNet initialization. Use a compatible environment with PyTorch/torchvision, NumPy, Pillow, scikit-learn and SciPy. Private manifests must be regenerated or obtained through an authorized project workflow; their absence is not permission to bypass verification. No held-out image is needed by the completed-suite validator. If an evaluation freeze exists, its unchanged evaluation support files and private evaluation manifest are also needed.

Build after full retrieval verification:

```sh
python docs/kevin_paper_20261011/round2_publication/reconstruction/migrate_completed_clinical.py build \
  --root . --suite "$VERIFIED_ORIGINAL_SUITE" \
  --receipt "$ACTUAL_RETRIEVAL_RECEIPT" \
  --checksum "$INDEPENDENT_ARCHIVE_SHA256_FILE" \
  --output "$NEW_MIGRATION_ARCHIVE"
```

Install into new local directories only, with the compatible ML Python environment:

```sh
python docs/kevin_paper_20261011/round2_publication/reconstruction/migrate_completed_clinical.py install \
  --archive "$NEW_MIGRATION_ARCHIVE" \
  --expected-sha256 "$INDEPENDENT_MIGRATION_SHA256" \
  --root "$REPOSITORY_ROOT" --suite "$NEW_SUITE_DIRECTORY" \
  --evidence "$NEW_EVIDENCE_DIRECTORY"
```

Archive members, copied payloads and destination readback are checked before a success receipt is written. The original retrieval inventory is preserved separately from the smaller result copy, so it cannot falsely claim the omitted optimizer checkpoints are present. An already-launched clinical evaluation is refused, protecting the single-launch guard. No scientific paths or content are silently rewritten. The actual clinical comparison and its separate held-out input gate remain outside this helper.
