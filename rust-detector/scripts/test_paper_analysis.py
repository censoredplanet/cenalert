"""Tests of fixed-parameter ablation and label-sampling analysis."""
import unittest

from analyze_paper_results import (
    Analysis, kept, label_kind, member_parameters, rejected_episodes, sampling_threshold, severity,
)


class PaperAnalysisTests(unittest.TestCase):
    def test_standalone_sparse_threshold_preserves_training_scale(self):
        p = dict(window=60,min_residual_ratio=.2,efficiency=.05,chebyshev_z=3,
                 chebyshev_k=5,isolation_forest_min_score=.6,lof_min_score=1.2)
        for member in ('isolation_forest','lof'):
            result=member_parameters(p,member,25,100)
            self.assertEqual(result['min_residual_ratio']*100,p['min_residual_ratio']*25)
        self.assertEqual(member_parameters(p,'chebyshev',25,100)['min_residual_ratio'],.2)

    def test_rejections_exclude_accepted_anomalies_and_group_consecutive_days(self):
        def point(day, count=1, active=False):
            return dict(date=f'2025-01-{day:02}',anomaly=active,
                        ensemble_vote=dict(vote_count=count,lof=dict(voted=True)))
        rows=[point(1),point(2),point(3,active=True),point(4,count=2),point(5)]
        self.assertEqual(rejected_episodes(rows,'lof'),[
            dict(start='2025-01-01',end='2025-01-02'),dict(start='2025-01-05',end='2025-01-05')])

    def test_sampling_uses_only_drawn_labels_and_respects_no_match(self):
        refs=[dict(impact=10,labels=[0,1]),dict(impact=2,labels=[2])]
        self.assertIsNone(sampling_threshold(refs,set()))
        self.assertIsNone(sampling_threshold(refs,{3}))
        self.assertEqual(sampling_threshold(refs,{0}),10)
        self.assertEqual(sampling_threshold(refs,{0,2}),2)
        self.assertTrue(kept(1,None,5))
        self.assertFalse(kept(1,10,5))
        self.assertTrue(kept(5,10,5))

    def test_severity_bands_and_ties(self):
        reference=[0,4,8,12,16]
        self.assertEqual([severity(x,reference) for x in (0,4,8,12)],
                         ['low','medium','high','very_high'])
        self.assertEqual(severity(1,[]),'unavailable')

    def test_notification_classes_follow_labels(self):
        self.assertEqual(label_kind(dict(scope='global',description='World Cup')),'other_explained')
        self.assertEqual(label_kind(dict(scope='country',description='Reaction to US TikTok ban')),'other_explained')
        self.assertEqual(label_kind(dict(scope='country',description='Telegram',metadata={'category':'filtering'})), 'restriction_related')

    def test_subsampling_full_and_empty_sets_reconcile(self):
        a=Analysis.__new__(Analysis)
        a.train_labels=[dict(scope='country',country_code='RU',date='2024-01-01')]
        a.labels=a.train_labels+[dict(scope='country',country_code='RU',date='2025-01-01')]
        a.first_test='2025-01-01';a.config=dict(test_end='2026-08-31');a.proximity=3
        row=dict(country='RU',variant='labels_ensemble',selection='knee',high_impact_threshold=5,
                 coalesced_anomalies=[dict(start='2025-01-01',impact=1),dict(start='2025-02-01',impact=6)])
        a.context=dict(countries=[row],summary={'labels_ensemble/knee':dict(retained=1,retained_covered=0)})
        refs=[dict(country='RU',variant='labels_ensemble',selection='knee',applicable_labels=[0],
                   references=[dict(impact=10,labels=[0])])]
        result=a.sample(refs,3)
        self.assertEqual(result,a.sample(refs,3))
        empty=next(r for r in result['summary'] if r['fraction']==0)
        full=next(r for r in result['summary'] if r['fraction']==1)
        self.assertEqual(empty['filtered']['mean'],0)
        self.assertEqual(empty['test_labels_retained']['mean'],1)
        self.assertEqual(full['filtered']['mean'],1)
        self.assertEqual(full['test_labels_retained']['mean'],0)


if __name__=='__main__':unittest.main()
