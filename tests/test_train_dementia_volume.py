"""Patient-level training mechanics on synthetic people (no real scans, no network, CPU only).

These tests pin how people are split, scored and recorded. They use random
initialisation and a few steps of fitting, so they say nothing about how well
any model performs.
"""

import csv
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

from src.mri_volume import MRIVolume, VolumeError

if torch is not None:
    from src import train_dementia_volume as tdv
    from src import volume_analysis as va


def synthetic_head(seed, enlarged, spacing=3.0, shape=(60, 70, 62)):
    """Small integer head; ``enlarged`` widens the dark ventricles (the only class signal)."""

    rng = np.random.default_rng(seed)
    axes = [(np.arange(n) - (n - 1) / 2) * spacing for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt((x / 72.0) ** 2 + (y / 90.0) ** 2 + (z / 80.0) ** 2)
    volume = np.zeros(shape)
    volume[radius < 1.0] = 900
    volume[radius < 0.93] = 60
    volume[radius < 0.86] = 520
    half_width = 34 if enlarged else 10
    volume[(np.abs(x) < half_width) & (np.abs(y) < 30) & (np.abs(z - 20) < 28)] = 120
    volume += rng.integers(0, 20, size=shape) * (radius < 0.86)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = [-(n - 1) / 2 * spacing for n in shape]
    return MRIVolume.from_array(np.rint(volume).astype(np.int16), affine)


def make_cases(people=16, repeat_scan_for=()):
    cases = []
    for number in range(people):
        label = number % 2
        stratum = ("CDR 0" if label == 0 else "CDR 1" if number % 4 == 1 else "CDR 0.5")
        subject = f"person{number:02d}"
        for scan in range(2 if number in repeat_scan_for else 1):
            cases.append(tdv.Case(case_id=f"{subject}_scan{scan}", subject_id=subject, label=label, stratum=stratum,
                                  load=lambda number=number, scan=scan, label=label: synthetic_head(1000 * scan + number, bool(label))))
    return cases


def tiny_model():
    """Stand-in with the attributes the trainer drives (features blocks, head); fast on CPU."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Sequential(nn.Conv2d(3, 8, 5, stride=4, padding=2), nn.BatchNorm2d(8), nn.ReLU()),
                nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU()),
                nn.Sequential(nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU()),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.0)
            self.head = nn.Linear(16, 2)

        def forward(self, x):
            return self.head(self.dropout(self.avgpool(self.features(x)).flatten(1)))

    return Tiny()


FAST = dict(folds=2, max_epochs=3, min_epochs=1, patience=2, warmup_epochs=1, unfreeze_from=1, slices_per_bag=4,
            batch_patients=4, head_lr=1e-2, backbone_lr=1e-3, bootstrap_repetitions=40)


@unittest.skipIf(torch is None, "PyTorch/nibabel are not installed")
class SplitAndMetricTests(unittest.TestCase):
    def test_folds_are_by_person_stratified_and_deterministic(self):
        cases = make_cases(40, repeat_scan_for=(3, 8))
        assignment = tdv.assign_folds(cases, 5, 42)
        self.assertEqual(assignment, tdv.assign_folds(list(reversed(cases)), 5, 42))  # order of input does not matter
        self.assertNotEqual(assignment, tdv.assign_folds(cases, 5, 43))
        self.assertEqual(set(assignment), {case.subject_id for case in cases})
        sizes = np.bincount(list(assignment.values()), minlength=5)
        self.assertLessEqual(sizes.max() - sizes.min(), 1)
        strata = {case.subject_id: (case.label, case.stratum) for case in cases}
        for key in set(strata.values()):  # every stratum is spread across the folds
            per_fold = np.bincount([fold for subject, fold in assignment.items() if strata[subject] == key], minlength=5)
            self.assertLessEqual(per_fold.max() - per_fold.min(), 1)
        for case in cases:  # a person's second scan goes wherever the person goes
            self.assertEqual(assignment[case.subject_id], assignment[f"person{int(case.subject_id[-2:]):02d}"])

    def test_split_rejects_inconsistent_or_too_small_cohorts(self):
        cases = make_cases(8)
        conflicting = cases + [tdv.Case("extra", cases[0].subject_id, 1 - cases[0].label, cases[0].stratum, cases[0].load)]
        with self.assertRaisesRegex(ValueError, "different labels"):
            tdv.assign_folds(conflicting, 2, 42)
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            tdv.assign_folds(cases[:3], 5, 42)
        one_class = [case for case in cases if case.label == 0]
        with self.assertRaisesRegex(ValueError, "both classes"):
            tdv.assign_folds(one_class, 2, 42)

    def test_validation_people_are_held_out_of_fitting_with_both_classes(self):
        cases = make_cases(40)
        subjects = {case.subject_id: (case.label, case.stratum) for case in cases}
        held = tdv.hold_out_validation(subjects, 0.2, 7)
        self.assertEqual(held, tdv.hold_out_validation(dict(reversed(list(subjects.items()))), 0.2, 7))
        self.assertTrue(4 <= len(held) <= 12)
        self.assertEqual({subjects[subject][0] for subject in held}, {0, 1})
        self.assertEqual({subjects[subject][0] for subject in set(subjects) - held}, {0, 1})
        with self.assertRaisesRegex(ValueError, "too small"):
            tdv.hold_out_validation({"a": (0, "x"), "b": (1, "y")}, 0.2, 7)
        # One-person strata: each class still gets someone to validate on and someone to fit on.
        singles = {"a": (0, "s1"), "b": (0, "s2"), "c": (1, "s3"), "d": (1, "s4")}
        held = tdv.hold_out_validation(singles, 0.2, 7)
        self.assertEqual(sorted(singles[subject][0] for subject in held), [0, 1])

    def test_exact_intervals_and_auroc(self):
        low, high = tdv.exact_interval(9, 10)
        self.assertAlmostEqual(low, 0.5549839, places=5)
        self.assertAlmostEqual(high, 0.9974714, places=5)
        self.assertEqual(tdv.exact_interval(0, 10)[0], 0.0)
        self.assertAlmostEqual(tdv.exact_interval(0, 10)[1], 0.3084971, places=5)  # the "0/10" bound quoted for Round 2
        self.assertEqual(tdv.exact_interval(10, 10)[1], 1.0)
        self.assertEqual(tdv.exact_interval(0, 0), (None, None))
        self.assertEqual(tdv.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(tdv.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(tdv.auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertIsNone(tdv.auroc([1, 1], [0.2, 0.3]))

    def test_binary_metrics_count_people_correctly(self):
        result = tdv.binary_metrics([1, 1, 1, 0, 0, 0, 0], [0.9, 0.6, 0.4, 0.1, 0.2, 0.7, 0.5], threshold=0.5)
        self.assertEqual(result["confusion"], {"true_positive": 2, "false_negative": 1, "true_negative": 2, "false_positive": 2})
        self.assertAlmostEqual(result["sensitivity"], 2 / 3)
        self.assertAlmostEqual(result["specificity"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], (2 / 3 + 0.5) / 2)
        self.assertEqual(result["sensitivity_exact_95ci"], list(tdv.exact_interval(2, 3)))
        only_controls = tdv.binary_metrics([0, 0], [0.9, 0.9])  # everyone called positive: specificity 0, no sensitivity
        self.assertEqual((only_controls["specificity"], only_controls["sensitivity"], only_controls["balanced_accuracy"]), (0.0, None, None))
        for labels, scores in [([0, 1], [0.5]), ([], []), ([0, 1], [0.5, 1.5]), ([0, 1], [0.5, float("nan")])]:
            with self.assertRaises(ValueError):
                tdv.binary_metrics(labels, scores)
        intervals = tdv.bootstrap_intervals([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], ["a", "a", "b", "c"], 0.5, seed=1, repetitions=50)
        self.assertEqual(intervals["balanced_accuracy"], [1.0, 1.0])

    def test_augmentation_is_patient_consistent_and_bounded(self):
        images = np.random.default_rng(0).integers(0, 255, size=(3, 64, 64), dtype=np.uint8)
        first = tdv.augment_bag(images, np.random.default_rng(5))
        again = tdv.augment_bag(images, np.random.default_rng(5))
        np.testing.assert_array_equal(first, again)
        self.assertEqual(first.shape, (3, 64, 64))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(0.0 <= first.min() and first.max() <= 1.0)
        self.assertFalse(np.array_equal(first, tdv.augment_bag(images, np.random.default_rng(6))))


@unittest.skipIf(torch is None, "PyTorch/nibabel are not installed")
class CrossValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="train_dementia_volume_test_"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)
        cls.cases = make_cases(16, repeat_scan_for=(2, 5))

        def unusable():
            raise VolumeError("simulated unreadable scan")

        cls.cases.append(tdv.Case("broken_scan0", "broken", 0, "CDR 0", unusable))
        cls.config = tdv.TrainConfig(**FAST)
        cls.cache = cls.directory / "cache"
        cls.summary = tdv.cross_validate(cls.cases, cls.directory / "run", cohort="synthetic people", config=cls.config,
                                         cache_dir=cls.cache, init="random", device=torch.device("cpu"), model_factory=tiny_model)

    def read_private(self, run="run"):
        with (self.directory / run / "private" / "out_of_fold_predictions.csv").open() as handle:
            return list(csv.DictReader(handle))

    def test_every_person_is_tested_once_by_a_model_that_never_saw_them(self):
        rows = self.read_private()
        usable = [case for case in self.cases if case.subject_id != "broken"]
        self.assertEqual(sorted(row["case_id"] for row in rows), sorted(case.case_id for case in usable))
        assignment = tdv.assign_folds(usable, self.config.folds, self.config.seed)
        for row in rows:
            self.assertEqual(int(row["fold"]), assignment[row["subject_id"]])
        by_person = {}
        for row in rows:  # both scans of a person are tested in the same fold
            self.assertEqual(by_person.setdefault(row["subject_id"], row["fold"]), row["fold"])
        for fold in self.summary["per_fold"]:
            self.assertEqual(fold["fit_people"] + fold["validation_people"] + fold["test_people"], 16)
        self.assertEqual(sum(fold["test_people"] for fold in self.summary["per_fold"]), 16)

    def test_summary_is_aggregate_and_states_its_limits(self):
        summary = self.summary
        self.assertEqual(json.loads((self.directory / "run" / "cross_validation.json").read_text())["people"], 16)
        self.assertEqual((summary["people"], summary["scans"], summary["scans_that_could_not_be_prepared"]), (16, 18, 1))
        self.assertEqual(summary["people_by_class"], {"CN": 8, "AD": 8})
        self.assertEqual(summary["people_by_stratum"], {"CDR 0": 8, "CDR 0.5": 4, "CDR 1": 4})
        pooled = summary["out_of_fold"]
        self.assertEqual(pooled["n"], 18)  # scans; people are the resampling unit
        self.assertEqual(sum(pooled["confusion"].values()), 18)
        self.assertEqual(len(pooled["sensitivity_exact_95ci"]), 2)
        self.assertIn("resamples of people", pooled["bootstrap"]["method"])
        self.assertEqual(sum(row["n"] for row in summary["by_stratum"].values()), 18)
        self.assertTrue(any("Not external validation" in limit for limit in summary["limits"]))
        self.assertIn("has not been tested on scans from any other site", summary["notice"])
        text = json.dumps(summary)
        self.assertNotIn("person0", text)  # no per-person rows outside private/
        self.assertNotIn("broken", text)
        self.assertEqual((self.directory / "run" / "private" / ".gitignore").read_text(), "*\n")
        failures = (self.directory / "run" / "private" / "preparation_failures.csv").read_text()
        self.assertIn("simulated unreadable scan", failures)

    def test_histories_and_selection(self):
        for fold in range(self.config.folds):
            history = json.loads((self.directory / "run" / f"fold{fold}_history.json").read_text())
            epochs = history["epochs"]
            self.assertEqual(epochs[0]["epoch"], 0)
            self.assertIsNone(epochs[0]["fit_loss"])  # epoch 0 is the untrained model and is eligible
            self.assertTrue(all(np.isfinite(row["validation_nll"]) for row in epochs))
            self.assertTrue(all(row["fit_loss"] is not None and np.isfinite(row["fit_loss"]) for row in epochs[1:]))
            selected = min(epochs, key=lambda row: row["validation_nll"])
            self.assertAlmostEqual(history["selected_validation_nll"], selected["validation_nll"], places=3)
            self.assertLessEqual(history["completed_epochs"], self.config.max_epochs)

    def test_same_seed_reproduces_the_predictions(self):
        tdv.cross_validate(self.cases, self.directory / "again", cohort="synthetic people", config=self.config,
                           cache_dir=self.cache, init="random", device=torch.device("cpu"), model_factory=tiny_model)
        first = {row["case_id"]: row["probability_AD"] for row in self.read_private()}
        second = {row["case_id"]: row["probability_AD"] for row in self.read_private("again")}
        self.assertEqual(first, second)

    def test_finished_runs_are_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            tdv.cross_validate(self.cases, self.directory / "run", cohort="x", config=self.config, init="random",
                               device=torch.device("cpu"), model_factory=tiny_model)
        with self.assertRaises(ValueError):
            tdv.TrainConfig(folds=1).validate()
        with self.assertRaises(ValueError):
            tdv.TrainConfig(class_names=("a", "b", "c")).validate()

    def test_validation_shortcut_is_exact_before_and_after_fitting(self):
        cases = self.cases[:4]
        cache = tdv.SliceCache(self.cache)
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = tiny_model()
        split = self.config.unfreeze_from
        stored = tdv.frozen_activations(model, cases, cache, device, split)
        np.testing.assert_allclose(tdv.predict_from_activations(model, stored, device, split),
                                   tdv.predict_cases(model, cases, cache, device), atol=1e-6)
        # Fit the trainable part for a few steps exactly as the trainer does: the stored activations stay valid.
        optimizer = torch.optim.SGD([p for block in list(model.features.children())[split:] for p in block.parameters()]
                                    + list(model.head.parameters()), lr=0.05)
        for step in range(3):
            tdv._set_trainable(model, self.config.warmup_epochs + 1, self.config)
            inputs = va.images_to_tensor(cache.images(cases[step])[:6])
            loss = nn.functional.cross_entropy(model(inputs), torch.tensor([step % 2] * 6))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        after = tdv.predict_cases(model, cases, cache, device)
        np.testing.assert_allclose(tdv.predict_from_activations(model, stored, device, split), after, atol=1e-6)
        self.assertGreater(float(np.abs(after - tdv.predict_cases(tiny_model(), cases, cache, device)).max()), 0.0)

    def test_cache_holds_the_band_slices_the_analyzer_would_render(self):
        case = self.cases[0]
        cache = tdv.SliceCache(self.cache)
        prepared = va.prepare_clinical_volume(case.load())
        np.testing.assert_array_equal(cache.images(case), prepared.render_slices(prepared.band_indices))
        self.assertTrue(any(self.cache.glob("*.npz")))
        self.assertFalse(any("person" in path.name for path in self.cache.iterdir()))  # opaque file names

    def test_shared_cache_does_not_mix_up_equally_named_scans_from_different_places(self):
        # Two cohorts can both hold "sub-01|sub-01/anat/T1w.nii.gz"; a shared cache must not hand one the other's slices.
        first, second = self.cases[0], self.cases[1]
        here = tdv.Case("sub-01|T1w.nii.gz", "sub-01", first.label, first.stratum, first.load, source="/cohort_a/T1w.nii.gz")
        there = tdv.Case("sub-01|T1w.nii.gz", "sub-01", second.label, second.stratum, second.load, source="/cohort_b/T1w.nii.gz")
        cache = tdv.SliceCache(self.directory / "shared_cache")
        np.testing.assert_array_equal(cache.images(here), tdv.SliceCache(self.cache).images(first))
        np.testing.assert_array_equal(cache.images(there), tdv.SliceCache(self.cache).images(second))
        self.assertEqual(len(list((self.directory / "shared_cache").glob("*.npz"))), 2)
        self.assertFalse(np.array_equal(cache.images(here), cache.images(there)))


@unittest.skipIf(torch is None, "PyTorch/nibabel are not installed")
class RealArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="train_dementia_volume_real_"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)
        cls.cases = make_cases(8)
        config = tdv.TrainConfig(folds=2, max_epochs=1, min_epochs=1, patience=1, warmup_epochs=0, slices_per_bag=3,
                                 batch_patients=2, bootstrap_repetitions=20)
        cls.summary = tdv.cross_validate(cls.cases, cls.directory / "run", cohort="synthetic people", config=config,
                                         init="random", device=torch.device("cpu"))

    def test_fold_checkpoints_load_into_the_analyzer_and_explain_in_3d(self):
        paths = [self.directory / "run" / entry["file"] for entry in self.summary["checkpoints"]]
        self.assertEqual([path.name for path in paths], ["fold0.pt", "fold1.pt"])
        analyzer = va.DementiaVolumeAnalyzer(paths, device=torch.device("cpu"))
        self.assertEqual((analyzer.slice_policy, analyzer.class_names, analyzer.threshold), ("band", ("CN", "AD"), 0.5))
        self.assertEqual(analyzer.notice, self.summary["notice"])
        self.assertIn("2-fold cross-validation by person", analyzer.notice)
        self.assertNotIn("specificity 0/10", analyzer.notice)  # its own record, not the Round 2 ensemble's
        for member, entry in zip(analyzer.members, self.summary["checkpoints"]):
            self.assertEqual(member["sha256"], entry["sha256"])
        volume = self.cases[0].load()
        result = analyzer.analyze(volume)
        self.assertEqual(result.cam_grid.shape, result.prepared.grid.shape)
        self.assertEqual(result.cam_indices, result.prepared.band_indices)
        # The trainer's own scoring and the analyzer agree on a scan: one definition of the model's output.
        cache = tdv.SliceCache(self.directory / "run" / "private" / "prepared")
        expected = np.mean([tdv.predict_cases(model, [self.cases[0]], cache, torch.device("cpu"))[0] for model in analyzer.models], axis=0)
        np.testing.assert_allclose(result.probabilities, expected, atol=1e-6)

    def test_checkpoint_records_what_it_was_trained_on(self):
        checkpoint = torch.load(self.directory / "run" / "fold0.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["architecture"], va.CLINICAL_ARCHITECTURE)
        self.assertEqual(checkpoint["initialization"]["kind"], "random")
        self.assertEqual(checkpoint["recipe"], va.CLINICAL_RECIPE)
        training = checkpoint["training"]
        self.assertEqual((training["cohort"], training["fold"], training["folds"], training["people"]), ("synthetic people", 0, 2, 8))
        self.assertIn("test fold never used", training["selection"])
        self.assertNotIn("person0", json.dumps({key: value for key, value in checkpoint.items() if key != "model"}))

    def test_backbone_initialisation_from_a_weights_file_is_recorded(self):
        from torchvision import models

        torch.manual_seed(3)
        source = models.mobilenet_v3_large(weights=None)
        path = self.directory / "weights.pth"
        torch.save(source.state_dict(), path)
        model = va.build_clinical_slice_model()
        record = tdv.initialise_backbone(model, str(path))
        self.assertEqual(record["kind"], "file")
        self.assertEqual(len(record["sha256"]), 64)
        self.assertFalse(record["is_official_imagenet1k_v1"])  # random weights, and the record says so
        for key, value in source.features.state_dict().items():
            torch.testing.assert_close(model.features.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaises(FileNotFoundError):
            tdv.initialise_backbone(model, str(self.directory / "missing.pth"))

    def test_manifest_cohort(self):
        folder = self.directory / "cohort"
        folder.mkdir()
        rows = [("subject_id", "scan", "label", "stratum")]
        for number, case in enumerate(self.cases[:4]):
            volume = case.load()
            nib.save(nib.Nifti1Image(np.asarray(volume.data, dtype=np.int16), volume.affine), str(folder / f"scan{number}.nii.gz"))
            rows.append((f"p{number}", f"scan{number}.nii.gz", "AD" if case.label else "0", "older" if number < 2 else ""))
        manifest = folder / "cohort.csv"
        with manifest.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        cases = tdv.cases_from_manifest(manifest, ("CN", "AD"))
        self.assertEqual([case.label for case in cases], [case.label for case in self.cases[:4]])
        self.assertEqual([case.stratum for case in cases], ["older", "older", "CN", "AD"])
        np.testing.assert_array_equal(cases[1].load().data, self.cases[1].load().data)
        self.assertEqual(cases[1].source, str((folder / "scan1.nii.gz").resolve()))  # the cache key follows the file
        manifest.write_text("subject_id,scan,label\np0,scan0.nii.gz,maybe\n")
        with self.assertRaisesRegex(ValueError, "label must be one of"):
            tdv.cases_from_manifest(manifest, ("CN", "AD"))
        manifest.write_text("subject_id,scan,label\np0,nowhere.nii.gz,AD\n")
        with self.assertRaisesRegex(ValueError, "scan not found"):
            tdv.cases_from_manifest(manifest, ("CN", "AD"))


OASIS_PEOPLE = [  # (subject number, age, CDR as written in the table)
    (1, 74, "0"), (2, 81, "0"), (3, 66, "0"), (4, 90, "0"), (5, 71, "0"),
    (6, 77, "0.5"), (7, 83, "0.5"), (8, 69, "0.5"), (9, 79, "1"), (10, 86, "2"),
]


def write_oasis_tree(root):
    """A small tree laid out like the provider's discs: Analyze pairs in stored (anterior, superior, left) order."""

    lines = ["ID,M/F,Hand,Age,Educ,SES,MMSE,CDR,eTIV,nWBV,ASF,Delay"]

    def add(session_id, age, cdr, seed, enlarged, delay="N/A"):
        lines.append(f"{session_id},F,R,{age},2,3,28,{cdr},1400,0.75,1.25,{delay}")
        folder = root / f"disc{1 + seed % 3}" / session_id / "PROCESSED" / "MPRAGE" / "SUBJ_111"
        folder.mkdir(parents=True)
        head = np.asarray(synthetic_head(seed, enlarged).data)  # (right, anterior, superior)
        stored = np.ascontiguousarray(np.transpose(head, (1, 2, 0))[:, :, ::-1]).astype(np.int16)
        nib.save(nib.AnalyzeImage(stored[..., None], np.eye(4)), str(folder / f"{session_id}_mpr_n4_anon_sbj_111.img"))

    for number, age, cdr in OASIS_PEOPLE:
        add(f"OAS1_{number:04d}_MR1", age, cdr, number, cdr != "0")
    add("OAS1_0001_MR2", 74, "0", 101, False, delay="20")  # the same person again: must not be trained or tested on
    add("OAS1_0011_MR1", 24, "", 11, False)  # young adult without a rating: not a control
    add("OAS1_0012_MR1", 48, "0", 12, False)  # rated but under 60
    (root / "oasis_cross-sectional.csv").write_text("\n".join(lines) + "\n")


@unittest.skipIf(torch is None, "PyTorch/nibabel are not installed")
class OasisCommandLineTests(unittest.TestCase):
    """The command a user runs on OASIS-1, end to end, on a synthetic tree in the provider's layout."""

    @classmethod
    def setUpClass(cls):
        import contextlib
        import io

        cls.directory = Path(tempfile.mkdtemp(prefix="train_dementia_volume_oasis_"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)
        cls.root = cls.directory / "oasis1"
        write_oasis_tree(cls.root)
        cls.output = cls.directory / "run"
        buffer = io.StringIO()
        handlers, level = logging.root.handlers[:], logging.root.level  # main() configures logging; undo it afterwards
        try:
            with contextlib.redirect_stdout(buffer):
                cls.status = tdv.main(["--oasis-root", str(cls.root), "--output", str(cls.output), "--init", "random",
                                       "--folds", "2", "--max-epochs", "1", "--device", "cpu"])
        finally:
            logging.root.handlers[:], logging.root.level = handlers, level
        cls.printed = buffer.getvalue()

    def rows(self):
        with (self.output / "private" / "out_of_fold_predictions.csv").open() as handle:
            return list(csv.DictReader(handle))

    def test_only_first_visits_of_rated_older_adults_are_used_one_per_person(self):
        self.assertEqual(self.status, 0)
        rows = self.rows()
        self.assertEqual(sorted(row["case_id"] for row in rows), [f"OAS1_{number:04d}_MR1" for number, _, _ in OASIS_PEOPLE])
        self.assertEqual(sorted(row["subject_id"] for row in rows), [f"OAS1_{number:04d}" for number, _, _ in OASIS_PEOPLE])
        expected = {f"OAS1_{number:04d}_MR1": (str(int(cdr != "0")), f"CDR {cdr}") for number, _, cdr in OASIS_PEOPLE}
        self.assertEqual({row["case_id"]: (row["label"], row["stratum"]) for row in rows}, expected)  # CDR 0 versus above 0

    def test_summary_names_the_cohort_and_holds_no_identifiers(self):
        summary = json.loads((self.output / "cross_validation.json").read_text())
        self.assertIn("OASIS-1, adults aged 60+, CDR 0 versus CDR above 0", summary["cohort"])
        self.assertEqual((summary["people"], summary["scans"], summary["scans_that_could_not_be_prepared"]), (10, 10, 0))
        self.assertEqual(summary["people_by_class"], {"CN": 5, "AD": 5})
        self.assertEqual(summary["people_by_stratum"], {"CDR 0": 5, "CDR 0.5": 3, "CDR 1": 1, "CDR 2": 1})
        self.assertEqual(sum(row["test_people"] for row in summary["per_fold"]), 10)
        printed = json.loads(self.printed)
        self.assertEqual(printed["people"], 10)
        self.assertNotIn("OAS1_", json.dumps(summary) + self.printed)
        tracked = [path.name for path in self.output.iterdir() if path.is_file()]
        self.assertEqual(sorted(tracked), ["cross_validation.json", "fold0.pt", "fold0_history.json", "fold1.pt", "fold1_history.json"])

    def test_out_of_fold_score_is_what_inference_gives_for_that_scan(self):
        # Trainer and inference share one path: OASIS loader -> canonical volume -> preparation -> band -> model.
        from src import oasis1

        sessions = {session.session_id: session for session in oasis1.load_index(self.root)}
        analyzers = {}
        for row in self.rows()[:4]:
            fold = row["fold"]
            if fold not in analyzers:
                analyzers[fold] = va.DementiaVolumeAnalyzer([self.output / f"fold{fold}.pt"], device=torch.device("cpu"))
            result = analyzers[fold].analyze(oasis1.load_session_volume(sessions[row["case_id"]]))
            self.assertAlmostEqual(float(result.probabilities[1]), float(row["probability_AD"]), places=5)

    def test_a_second_run_into_the_same_folder_is_refused(self):
        with self.assertLogs("src.train_dementia_volume", level="ERROR") as logs:
            status = tdv.main(["--oasis-root", str(self.root), "--output", str(self.output), "--init", "random"])
        self.assertEqual(status, 2)
        self.assertIn("never overwritten", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
