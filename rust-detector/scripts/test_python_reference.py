import itertools
import unittest
from types import SimpleNamespace

try:
    from python_reference import Ensemble
except ModuleNotFoundError as error:
    if error.name not in {"polars", "numpy", "scipy", "sklearn", "croston", "kneebow", "isotree"}:
        raise
    raise unittest.SkipTest("Install requirements-validation.txt for Python reference tests") from error


class EnsembleThresholdTests(unittest.TestCase):
    def threshold(self, values):
        ensemble = object.__new__(Ensemble)
        ensemble.detectors = [
            SimpleNamespace(threshold=lambda guess, value=value: value)
            for value in values
        ]
        return ensemble.threshold(100.0)

    def test_second_smallest_in_any_member_order(self):
        for values in itertools.permutations([10.0, 20.0, 30.0]):
            with self.subTest(values=values):
                self.assertEqual(self.threshold(values), 20.0)

    def test_ties_and_unreachable_thresholds(self):
        for values, expected in [
            ([10.0, 10.0, 30.0], 10.0),
            ([10.0, 30.0, 30.0], 30.0),
            ([10.0, 20.0, float("inf")], 20.0),
            ([10.0, float("inf"), float("inf")], float("inf")),
        ]:
            with self.subTest(values=values):
                self.assertEqual(self.threshold(values), expected)


if __name__ == "__main__":
    unittest.main()
