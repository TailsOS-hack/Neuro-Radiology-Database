"""Cohort audit must detect re-encoded exposure and reject incomplete evidence."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.audit_radiologist_cohort import audit_cohort


class RadiologistCohortAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cohort = self.root / 'cohort'
        self.cohort.mkdir()
        self.labels = self.root / 'labels.csv'
        self.manifest = self.root / 'manifest.csv'
        image = Image.new('RGB', (12, 12), (33, 65, 110))
        image.save(self.cohort / 'case.png', compress_level=0)
        image.save(self.root / 'train.png', compress_level=0)
        image.save(self.root / 'test.png', compress_level=9)
        self.write_labels([{'FileName': 'case.png', 'TrueCategory': 'MildDemented'}])
        with self.manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['path', 'split', 'domain', 'subtype'])
            writer.writeheader()
            writer.writerows([
                dict(path='train.png', split='train', domain='dementia', subtype='MildDemented'),
                dict(path='test.png', split='test', domain='dementia', subtype='ModerateDemented'),
            ])

    def write_labels(self, rows):
        with self.labels.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['FileName', 'TrueCategory'])
            writer.writeheader()
            writer.writerows(rows)

    def audit(self, expected_count=1):
        return audit_cohort(self.cohort, self.labels, [self.manifest],
                            root=self.root, expected_count=expected_count)

    def test_reencoded_match_exposes_both_splits_and_label_conflict(self):
        result = self.audit()['manifests']['manifest.csv']
        self.assertEqual(result['file_sha256']['split_combinations'], {'train': 1})
        rgb = result['decoded_rgb_sha256']
        self.assertEqual(rgb['split_combinations'], {'test,train': 1})
        self.assertEqual(rgb['matched_source_rows'], 2)
        self.assertEqual(rgb['train_exposed_images'], 1)
        self.assertEqual(rgb['test_only_images'], 0)
        self.assertEqual(rgb['label_conflict_images'], ['case.png'])

    def test_rejects_duplicate_label_rows(self):
        self.write_labels([{'FileName': 'case.png', 'TrueCategory': 'MildDemented'}] * 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate filenames'):
            self.audit(expected_count=2)

    def test_rejects_unlabelled_image(self):
        (self.cohort / 'extra.png').write_bytes((self.cohort / 'case.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'image/label mismatch'):
            self.audit()

    def test_fails_on_unreadable_source(self):
        (self.root / 'train.png').write_bytes(b'broken image')
        with self.assertRaises(OSError):
            self.audit()

    def test_unmatched_image_does_not_count_as_test_only(self):
        Image.new('RGB', (12, 12), 'red').save(self.cohort / 'case.png')
        result = self.audit()['manifests']['manifest.csv']['decoded_rgb_sha256']
        self.assertEqual(result['unmatched_images'], ['case.png'])
        self.assertEqual(result['test_only_images'], 0)


if __name__ == '__main__':
    unittest.main()
