"""Checks for member attribution at accepted anomaly starts."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from review_member_contributions import analyze


class MemberContributionTests(unittest.TestCase):
    def test_multiple_anomalies_count_a_label_once_and_sparse_votes_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def write(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write('experiment.json', dict(train_end='2024-12-31', test_end='2026-08-31', proximity_days=3))
            write('inputs/labels.json', [dict(scope='country', country_code='RU', date='2024-01-02'),
                                        dict(scope='country', country_code='RU', date='2024-02-01')])
            base = 'runs/42/RU/labels_ensemble/'
            write(base + 'result.json', dict(status='completed', country='RU', variant='labels_ensemble',
                  seed=42, first_date='2011-01-01', selections=[dict(selection='knee')]))
            vote = {m: {'voted': m != 'chebyshev'} for m in ('chebyshev', 'isolation_forest', 'lof')}
            write(base + 'knee-detection.json', dict(anomalies=[
                dict(start='2024-01-01', ensemble_vote=vote),
                dict(start='2024-01-03', ensemble_vote=vote),
                dict(start='2024-02-01', sparse=True)]))
            analyze(root, root / 'out')
            with (root / 'out/member-contributions.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            rows = {r['member']: r for r in rows if r['period'] == 'training'}
            self.assertEqual(rows['isolation_forest']['accepted_anomalies'], '2')
            self.assertEqual(rows['isolation_forest']['country_labels_supported'], '1')
            self.assertEqual(rows['croston']['country_labels_supported'], '1')
            self.assertEqual(rows['ensemble']['country_labels_supported'], '2')
            self.assertEqual(rows['chebyshev']['country_labels_supported'], '0')


if __name__ == '__main__':
    unittest.main()
