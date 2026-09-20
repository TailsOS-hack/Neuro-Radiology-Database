"""Regression: byte-level deduplication must not imply decoded-pixel independence."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.audit_review_data import audit_rows
from src.experiment_pipeline import ImageRecord, assign_split


class ReviewDataTests(unittest.TestCase):
    def test_reencoded_pixels_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new('RGB', (12, 12), (33, 65, 110))
            image.save(root/'a.png', compress_level=0)
            image.save(root/'b.png', compress_level=9)
            rows = [dict(path=path, split=split, domain='dementia', subtype='MildDemented')
                    for path, split in [('a.png', 'train'), ('b.png', 'test')]]
            result = audit_rows(rows, root)
            self.assertEqual(result['cross_split']['file']['rows'], 0)
            self.assertEqual(result['cross_split']['pixels']['rows'], 2)
            rows[1]['split'] = 'train'
            self.assertEqual(audit_rows(rows, root)['cross_split']['pixels']['rows'], 0)

    def test_future_split_keeps_reencoded_official_test_group_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new('RGB', (12, 12), (33, 65, 110))
            image.save(root/'a.png', compress_level=0)
            image.save(root/'b.png', compress_level=9)
            records = [ImageRecord(root/'a.png', 'tumor', 'glioma', 'pool'),
                       ImageRecord(root/'b.png', 'tumor', 'glioma', 'test')]
            result = assign_split(records, seed=42, val_fraction=.1, test_fraction=.2)
            self.assertEqual(len(result), 2)
            self.assertEqual({r.split for r in result}, {'test'})

    def test_unreadable_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            row = dict(path='absent.png', split='test', domain='dementia', subtype='MildDemented')
            result = audit_rows([row], Path(directory))
            self.assertEqual(len(result['unreadable']), 1)


if __name__ == '__main__':
    unittest.main()
