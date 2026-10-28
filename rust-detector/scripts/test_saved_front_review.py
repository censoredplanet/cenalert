import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from review_saved_fronts import parameters, sample_front, unique_front, validate_cache, Review, VARIANTS


class FrontReviewTests(unittest.TestCase):
    def test_cache_rejects_changed_inputs_and_unversioned_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validate_cache(root, {'labels': 'original'})
            validate_cache(root, {'labels': 'original'})
            with self.assertRaises(ValueError):
                validate_cache(root, {'labels': 'updated'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'old.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_cache(root, {})

    def test_curves_use_requested_group_size(self):
        review = Review.__new__(Review)
        review.common = ['RU']
        review.labels = []
        review.cutoff, review.end, review.proximity = '2024-12-31', '2026-08-31', 3
        review.binary = Path('unused')
        review.minimum_countries = 4
        review.front = lambda c, v: [dict(front_index=0)]
        review.evaluate_point = lambda task: dict(country=task[0], variant=task[1],
            front_index=0, training_count=3)
        summary = dict(raw=5, labels_total=2, raw_covered=2, retained_covered=1,
                       notifications={'4': 3, '5': 4})
        with tempfile.TemporaryDirectory() as directory:
            review.output = Path(directory)
            with patch('review_saved_fronts.aggregate_rows', return_value=(summary, {})) as aggregate:
                review.curves(1)
            self.assertTrue(all(call.args[-1] == 4 for call in aggregate.call_args_list))
            import json
            rows = json.loads((review.output / 'curve-data.json').read_text())
            self.assertEqual(len(rows), 7 * len(VARIANTS))
            self.assertTrue(all(r['notifications'] == 3 and r['minimum_group_countries'] == 4
                                for r in rows))

    def test_parameter_rounding_matches_rust_ties_even(self):
        self.assertEqual(parameters([60.5,3,5,.1,.05],'original')['window'],60)
        self.assertEqual(parameters([61.5,3,5,.1,.05],'original')['window'],62)

    def test_unique_front_orders_by_training_count_then_impact(self):
        p=[60,3,5,.1,.05]
        q=[70,3,5,.1,.05]
        result=unique_front([(p,[2,-10]),(p,[2,-10]),(q,[2,-20])],'original')
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['training_impact'],20)
        self.assertEqual(result[1]['parameters']['window'],60)

    def test_sampling_includes_both_endpoints(self):
        front=list(range(5))
        self.assertEqual(sample_front(front,0),0)
        self.assertEqual(sample_front(front,.5),2)
        self.assertEqual(sample_front(front,1),4)
        self.assertEqual(sample_front([3],.5),3)

    def test_ensemble_parameter_order(self):
        p=parameters([50,.2,.05,3,6,.7,1.8],'ensemble')
        self.assertEqual(p['chebyshev_k'],6)
        self.assertEqual(p['isolation_forest_min_score'],.7)
        self.assertEqual(p['lof_min_score'],1.8)


if __name__=='__main__':unittest.main()
