"""HSI split regression checks using the local Indian Pines ground truth."""
import sys
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.hsi_splits import (split_class_locations, build_saved_split,
                                select_saved_split, read_saved_split, write_saved_split,
                                validate_saved_split)


class HSISplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load((ROOT / 'configs/datasets/indian_pines.yaml').read_text())['DATASET']
        gt_path = Path(config['ROOT']) / config['NAME'] / config['HSI']['GT_FILE']
        cls.gt = sio.loadmat(gt_path)[config['HSI']['GT_KEY']]
        cls.names = config['HSI']['CLASS_NAMES']
        cls.master = build_saved_split(cls.gt, cls.names, 'Indian_pines', seed=42)

    def test_new_classes_use_only_their_held_out_test_partition(self):
        total = 0
        for raw_label in range(9, 17):
            coords = np.argwhere(self.gt == raw_label)
            np.random.RandomState(42 + raw_label).shuffle(coords)
            full_train, full_val, expected_test = split_class_locations(coords, .6, .2, -1)
            train, val, test = split_class_locations(coords, .6, .2, 16, test_only=True)
            self.assertEqual((len(train), len(val)), (0, 0))
            np.testing.assert_array_equal(test, expected_test)
            self.assertFalse(set(map(tuple, test)) & set(map(tuple, full_train)))
            self.assertFalse(set(map(tuple, test)) & set(map(tuple, full_val)))
            total += len(test)
        self.assertEqual(total, 1202)

    def test_base_16_shot_counts_are_unchanged(self):
        totals = np.zeros(3, dtype=int)
        for raw_label in range(1, 9):
            coords = np.argwhere(self.gt == raw_label)
            groups = split_class_locations(coords, .6, .2, 16)
            totals += [len(group) for group in groups]
            sets = [set(map(tuple, group)) for group in groups]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        np.testing.assert_array_equal(totals, [128, 32, 859])

    def test_new_test_partition_is_independent_of_shot_count(self):
        coords = np.argwhere(self.gt == 9)
        expected = split_class_locations(coords, .6, .2, -1)[2]
        for shots in (-1, 1, 16, 100):
            train, val, test = split_class_locations(coords, .6, .2, shots, test_only=True)
            self.assertEqual((len(train), len(val)), (0, 0))
            np.testing.assert_array_equal(test, expected)

    def test_too_small_class_cannot_form_three_partitions(self):
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            split_class_locations(np.array([[4, 8]]), .6, .2, 16, test_only=True)

    def test_fap_test_batches_are_128_and_training_unchanged(self):
        for path in (ROOT / 'configs/trainers/FAP').glob('*.yaml'):
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config['DATALOADER']['TEST']['BATCH_SIZE'], 128)
            self.assertEqual(config['ATTACK']['HSI_PGD']['TEST_BATCH_SIZE'], 128)
            expected_train_pgd = 4 if path.name == 'hsi_vit_b32.yaml' else 16
            self.assertEqual(config['ATTACK']['HSI_PGD']['BATCH_SIZE'], expected_train_pgd)

    def test_master_partition_round_trip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split_Indian_pines.json'
            write_saved_split(path, self.master)
            loaded = read_saved_split(path, self.gt, self.names, 'Indian_pines')
            self.assertEqual(loaded, self.master)
            with self.assertRaises(FileExistsError):
                write_saved_split(path, self.master)
            self.assertEqual(read_saved_split(path, self.gt, self.names, 'Indian_pines'), loaded)

    def test_training_seed_changes_few_shot_draws_not_test(self):
        first = select_saved_split(self.master, 'all', 16, seed=0)
        second = select_saved_split(self.master, 'all', 16, seed=1)
        self.assertNotEqual(first['train'], second['train'])
        self.assertEqual(first['test'], second['test'])
        self.assertEqual(first, select_saved_split(self.master, 'all', 16, seed=0))
        self.assertEqual([len(first[key]) for key in ('train','val','test')], [252,64,2061])
        self.assertEqual(len({tuple(entry[:2]) for entry in first['train']}), len(first['train']))

    def test_saved_base_new_partition_counts_and_label_remapping(self):
        base = select_saved_split(self.master, 'base', 16, seed=1)
        new = select_saved_split(self.master, 'new', 16, seed=2)
        self.assertEqual([len(base[key]) for key in ('train','val','test')], [128,32,859])
        self.assertEqual([len(new[key]) for key in ('train','val','test')], [0,0,1202])
        self.assertEqual({entry[2] for entry in new['test']}, set(range(8)))
        self.assertEqual({entry[3] for entry in new['test']}, set(self.names[8:]))
        self.assertFalse({tuple(e[:2]) for e in base['test']} & {tuple(e[:2]) for e in new['test']})

    def test_class_filtering_preserves_same_seed_few_shot_selection(self):
        all_classes = select_saved_split(self.master, 'all', 16, seed=7)
        base = select_saved_split(self.master, 'base', 16, seed=7)
        for key in ('train','val','test'):
            self.assertEqual(base[key], [entry for entry in all_classes[key] if entry[2] < 8])

    def test_shot_count_never_changes_saved_test(self):
        expected = select_saved_split(self.master, 'all', 1, seed=0)['test']
        for shots in (2,4,16,-1):
            self.assertEqual(select_saved_split(self.master, 'all', shots, seed=9)['test'], expected)

    def test_invalid_saved_partition_is_rejected(self):
        bad = copy.deepcopy(self.master)
        bad['test'].append(bad['train'][0])
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            validate_saved_split(bad, self.gt, self.names, 'Indian_pines')
        moved = copy.deepcopy(self.master)
        moved['test'].append(moved['train'].pop())
        with self.assertRaisesRegex(ValueError, 'split ratios'):
            validate_saved_split(moved, self.gt, self.names, 'Indian_pines')
        changed_gt = self.gt.copy()
        changed_gt[0,0] = 1 if changed_gt[0,0] != 1 else 2
        with self.assertRaisesRegex(ValueError, 'gt_sha256'):
            validate_saved_split(self.master, changed_gt, self.names, 'Indian_pines')
        with self.assertRaisesRegex(ValueError, 'stride'):
            validate_saved_split(self.master, self.gt, self.names, 'Indian_pines', stride=2)

    def test_master_partition_covers_all_10249_labeled_pixels(self):
        locations = [tuple(entry[:2]) for key in ('train','val','test') for entry in self.master[key]]
        self.assertEqual(len(locations), 10249)
        self.assertEqual(len(set(locations)), 10249)
        self.assertEqual(set(locations), set(map(tuple, np.argwhere(self.gt > 0))))



if __name__ == '__main__':
    unittest.main()
