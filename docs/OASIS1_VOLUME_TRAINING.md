# Patient-level volume training on OASIS-1

Added 2026-11-14. **Status: prepared, not yet run on OASIS-1.** The OASIS-1
scans are not in this repository and were not available where this was built.
Everything below describes software that was checked on synthetic people and,
for mechanics only, on 20 real public T1 volumes with a non-clinical proxy
label. There is no OASIS-1 result, no trained checkpoint and no performance
claim here.

Why this cohort: the stage model's training images are anonymous 2D slices with
no patient identities, so its 99.9% benchmark cannot show how it does on a new
person, and on raw scans it fails outright (see
[DICOM_VOLUME_ANALYSIS.md](DICOM_VOLUME_ANALYSIS.md)). OASIS-1 has what those
images lack: one identified person per scan, a Clinical Dementia Rating (CDR)
for each older adult, and full 3D volumes with the skull, the kind of image a
DICOM T1 series holds.

## 1. Get the data

OASIS-1 is distributed by its provider under the provider's data use terms.
Requesting it and agreeing to those terms is for the person obtaining the data;
nothing in this repository downloads OASIS-1 or stands in for that agreement.

This project's dataset registry
(`docs/kevin_paper_20261011/publication/datasets/dataset_registry.json`) already
records how access works and what to watch for:

* Access goes through the provider's request form, with a research statement
  and acceptance of the data use terms. Start at
  <https://sites.wustl.edu/oasisbrains/home/access/> and follow the current
  instructions for **OASIS-1 (cross-sectional)**. Copies on other sites do not
  establish permission to use the data.
* The provider advises against combining OASIS-1, OASIS-2 and OASIS-3: the same
  people can appear in more than one under different IDs. Treat OASIS-1 as an
  alternative to the others, not an addition, and never present one as an
  independent test of a model trained on another.
* Compare age-matched older adults, not young adults against people with
  dementia, and count people, not acquisitions or slices.

Once access is granted, download the cross-sectional imaging archives and the
cross-sectional demographic/clinical table (`oasis_cross-sectional.csv`; if it
is offered as a spreadsheet, save it as CSV) and extract everything under
`data/external/oasis1/` (any nesting of `disc*` folders is fine).
`data/external/` is git-ignored; keep it that way and do not redistribute the
data.

## 2. Check the tree before training

```bash
python -m src.oasis1 data/external/oasis1 --qc oasis_qc.png
```

This prints aggregate counts only. The provider's fact sheet lists 100 people
with dementia, all over 60: 70 at CDR 0.5, 28 at CDR 1 and 2 at CDR 2;
`matches_documented_dementia_counts` says whether your tree agrees. The cohort
is described in the literature as 198 people aged 60 to 96, which would leave
98 without dementia; compare `eligible_people` with that. If the counts are
short, the extraction is probably incomplete.

**Open `oasis_qc.png` and look.** Each row is one scan: axial, coronal, sagittal.
The nose must be at the top of the axial view, the top of the head at the top of
the other two, and the face on the right of the sagittal view. These files are
Analyze 7.5, whose headers carry no trustworthy orientation, so the
voxel-to-world mapping is hard-coded. It is the matrix the
[Clinica OASIS-1 converter](https://github.com/aramis-lab/clinica) uses for the
same images (`SUBJ_111`, 256 x 256 x 160), and a test pins it; but it has not
been looked at on real OASIS-1 files in this project yet. Left versus right
cannot be judged by eye and rests on that mapping.

## 3. Train

```bash
python -m src.train_dementia_volume \
    --oasis-root data/external/oasis1 \
    --output training_logs/experiments/dementia_volume/oasis1
```

Use a GPU if there is one (device order is CUDA, then DirectML, then CPU). On a
4-core CPU expect on the order of one to two hours for the five folds; that is
an estimate from a timing run on 20 scans, not a measurement on OASIS-1. `--init imagenet` (the
default) downloads torchvision's MobileNetV3-Large ImageNet weights once; pass
`--init <file>` to use a local copy instead (its SHA-256 is recorded, and
compared with the official file's).

What a run does:

| Step | Choice | Why |
|---|---|---|
| People | First-visit sessions of adults aged 60+ with a recorded CDR | Younger adults have no CDR; counting them as controls would reward recognising age. The repeat (`MR2`) sessions are the same people again. |
| Label | CDR 0 versus CDR above 0 | In OASIS-1 every rated person with CDR above 0 has a clinical diagnosis of very mild to moderate Alzheimer's disease. Four classes are not attempted: CDR 2 is two people. |
| Image | `SUBJ_111`: average of the session's co-registered acquisitions, native space, 1 mm, **with skull** | The same kind of image as a DICOM T1 series, so the result needs no brain extraction at inference time. |
| Input | The preparation used at inference (`prepare_clinical_volume`, recipe `exploratory_t1_axial_v1`), every 1.5 mm axial slice between 45% and 85% of the cranial extent | One code path for training, for DICOM input and for the 3D Grad-CAM stack. |
| Model | MobileNetV3-Large features, global average pool, linear head; one score per scan = mean slice logit | Same family as the Round 2 clinical model, so the analyzer and the per-slice Grad-CAM apply unchanged. |
| Fitting | Round 2's reviewed schedule: head only for 5 epochs, then the last three feature blocks too; BatchNorm frozen; AdamW; cosine decay; patient-consistent augmentation; nine random band slices per person per step | Reuses choices that were already reviewed rather than inventing new ones. |
| Splits | Five folds **by person**, stratified by CDR level, seed 42 | No person contributes to both fitting and testing. |
| Selection | Each fold picks its epoch on validation people held out of its own fitting set | The test fold never influences fitting or selection. The 0.5 threshold is fixed, not tuned. |

Outputs in the output folder:

| File | Content |
|---|---|
| `cross_validation.json` | Aggregate results: every person scored once by a model fitted without them; sensitivity and specificity with exact 95% intervals; balanced accuracy and AUROC with intervals from resampling people; results by CDR level; per-fold rows; checkpoint hashes. No per-person rows. |
| `fold0.pt` ... `fold4.pt` | The five fold models. Together they are the ensemble. Each carries its class names, its slice policy, a notice stating what it was trained on and how it did, and separately the sentence about the attention map, which the analysis adds only to a result that has a map. |
| `fold<k>_history.json` | Loss per epoch and the selected epoch for each fold. |
| `private/` | Per-person predictions and the prepared-slice cache. It contains its own `.gitignore`; never commit it. |

An output folder is never overwritten, whether its run finished or was
interrupted: start again in a new folder. To keep the prepared slices between
attempts, pass `--cache-dir <folder outside the output>` (only while the scans
themselves are unchanged). Nothing here replaces an existing checkpoint.

## 4. Use the result

```bash
python -m src.volume_analysis /path/to/dicom/study --output results/ \
    --checkpoints training_logs/experiments/dementia_volume/oasis1/fold*.pt
```

To make a reviewed ensemble the default for the CLI and the GUI's **Analyze
Volume** button, copy its `fold*.pt` files into `models/dementia_volume/`.
Until then the default remains the Round 2 clinical ensemble. Every result
states which ensemble produced it, through that ensemble's own notice.

## 5. What the result will and will not mean

* It will be **internal cross-validation on one cohort from one study**. That
  is a real improvement in the kind of evidence, because the unit is the person,
  but it is not external validation. Performance on other scanners and sites is
  usually lower, and for the Round 2 model it collapsed.
* 70 of the 100 people with dementia are CDR 0.5 (very mild). That is a hard
  distinction on structural MRI. Expect results well short of the stage model's
  99.9%, which was measured on slices without patient identities. See the two
  published studies cited in [DICOM_VOLUME_ANALYSIS.md](DICOM_VOLUME_ANALYSIS.md)
  for what leakage-free evaluation typically shows.
* The natural next check is to score cohorts from other sites with the trained
  ensemble. The project's Besga, Harvard and OpenNeuro cohorts can serve as that
  check for a model that never trained on them, bearing in mind they were all
  inspected in earlier rounds. A manifest (`--manifest cohort.csv` with
  `subject_id`, `scan`, `label`) is how other labelled cohorts are fed to the
  trainer; scoring an existing ensemble on a cohort is not yet a single command.
* The Grad-CAM map still covers only the analysed slab, at 7 x 7 per slice. A
  whole-brain map needs a whole-brain model, which is what the gated Round 3
  design is for.

## 6. What was verified, and what was not

Verified (`tests/test_oasis1.py`, `tests/test_train_dementia_volume.py`,
`tests/test_volume_analysis.py`):

* folds are by person, stratified, deterministic, and a person's repeat scans
  stay together; validation people are held out of fitting with both classes;
* every usable scan receives exactly one out-of-fold prediction, from the fold
  model that did not see that person; a scan that cannot be prepared is counted
  and reported, not silently dropped;
* the same seed reproduces the same predictions;
* the summary contains no per-person rows;
* fold checkpoints load into the analyzer, which scores a scan exactly as the
  trainer does and explains it with the 3D Grad-CAM stack;
* the per-epoch validation shortcut (computing the never-trained early layers
  once) gives the same numbers as a full forward pass, before and after fitting;
* the OASIS-1 index applies the eligibility rules above and the loader's
  voxel-to-world mapping equals the Clinica matrix for the documented grid;
* the training command itself, end to end, on a synthetic tree laid out like
  the provider's discs (Analyze pairs, the clinical table, a repeat session, a
  young adult without a rating, a rated adult under 60): only first visits of
  rated older adults are used, one scan per person, labelled CDR 0 versus above
  0; and each person's out-of-fold score equals what the inference path gives
  for that scan with that fold's model;
* mechanics on real volumes: the real architecture was fitted for a few epochs
  on the 20 public OpenNeuro T1 scans with a **non-clinical proxy label**
  (recorded sex), end to end from preparation to the 3D map. Losses fell and the
  outputs were written. That run measured nothing about dementia and none of
  its outputs are kept.

Not verified:

* anything on OASIS-1 itself: the file layout and clinical table are handled as
  documented by the provider and as other tools read them, but no OASIS-1 file
  has been opened by this code;
* any performance figure.
