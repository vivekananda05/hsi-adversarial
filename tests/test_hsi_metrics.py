"""Metric checks against manually calculated confusion matrices."""
import importlib.util
from pathlib import Path
import unittest
import math

path = Path(__file__).resolve().parents[1] / 'Dassl.pytorch/dassl/evaluation/classification_metrics.py'
spec = importlib.util.spec_from_file_location('hsi_classification_metrics', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
classification_metrics = module.classification_metrics


class MetricTests(unittest.TestCase):
    def test_imbalanced_classes_oa_aa_and_kappa_differ(self):
        # Confusion matrix [[8, 2], [1, 1]]: OA=75%, AA=65%, Kappa=.25.
        result = classification_metrics([0]*10+[1]*2, [0]*8+[1]*2+[0, 1])
        self.assertAlmostEqual(result['OA'], 75)
        self.assertAlmostEqual(result['AA'], 65)
        self.assertAlmostEqual(result['kappa'], .25)

    def test_perfect_predictions(self):
        self.assertEqual(classification_metrics([0, 1, 2], [0, 1, 2]),
                         {'OA': 100., 'AA': 100., 'kappa': 1.})

    def test_all_wrong_balanced_predictions(self):
        self.assertEqual(classification_metrics([0, 0, 1, 1], [1, 1, 0, 0]),
                         {'OA': 0., 'AA': 0., 'kappa': -1.})

    def test_predicted_only_class_does_not_change_aa_denominator(self):
        result = classification_metrics([5, 5], [5, 9])
        self.assertEqual(result['OA'], 50.)
        self.assertEqual(result['AA'], 50.)
        self.assertEqual(result['kappa'], 0.)

    def test_undefined_single_class_kappa(self):
        result = classification_metrics([1, 1], [1, 1])
        self.assertEqual(result['OA'], 100.)
        self.assertTrue(math.isnan(result['kappa']))

    def test_invalid_inputs(self):
        for truth, pred in [([], []), ([1], [1, 2]), ([[1]], [[1]])]:
            with self.assertRaises(ValueError):
                classification_metrics(truth, pred)


if __name__ == '__main__':
    unittest.main()
