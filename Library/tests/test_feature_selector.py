import unittest
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

LIBRARY_DIR = Path(__file__).resolve().parents[1]
if str(LIBRARY_DIR) not in sys.path:
    sys.path.insert(0, str(LIBRARY_DIR))

from main import FeatureSelector as LegacyFeatureSelector
from maxrel_minred_minstra import FeatureSelector


class FeatureSelectorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_classification(
            n_samples=60,
            n_features=8,
            n_informative=4,
            n_redundant=1,
            n_classes=3,
            random_state=7,
        )

    def assert_valid_path(self, selected_path, max_features):
        self.assertEqual(len(selected_path), max_features)
        for step, subset in enumerate(selected_path, start=1):
            self.assertEqual(len(subset), step)
            self.assertEqual(len(set(subset)), step)
            self.assertTrue(all(0 <= feature < self.X.shape[1] for feature in subset))

    def test_information_methods_return_cumulative_paths(self):
        for method_name in ["mRMR", "JMI", "relax_mRMR"]:
            with self.subTest(method=method_name):
                selector = FeatureSelector(max_features=3, classes_=np.unique(self.y), random_state=0)
                selected_path = getattr(selector, method_name)(self.X, self.y)

                self.assert_valid_path(selected_path, max_features=3)
                self.assertEqual(selector.selected_features_, selected_path[-1])
                self.assertEqual(selector.all_selected_features, selected_path)

    def test_mrmr_ms_returns_cumulative_path(self):
        selector = FeatureSelector(
            max_features=3,
            classes_=np.unique(self.y),
            kernel="linear",
            random_state=0,
        )
        selected_path = selector.mRMR_MS(self.X, self.y)

        self.assert_valid_path(selected_path, max_features=3)
        self.assertEqual(selector.selected_features_, selected_path[-1])

    def test_legacy_main_import_points_to_package_selector(self):
        self.assertIs(LegacyFeatureSelector, FeatureSelector)


if __name__ == "__main__":
    unittest.main()
