"""Regression tests for event coverage, label sampling, and notification counts."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from analyze_paper_results import (
    Analysis, kept, label_kind, member_parameters, rejected_episodes, sampling_threshold, severity,
    verify_member_balance,
)
from analyze_event_coverage import ensemble_only, calibration_changes
from analyze_notification_outcomes import (
    calibration_probability, label_status, month_range, region_months, grouping_sensitivity,
)
from prepare_paper_figures import group_summary, matched_notifications, same_day_groups
from review_contextualization import grouping_key
from plot_detection_cases import plot_threshold
from repair_evaluation_dates import reconstruct
from analyze_explained_impact import analyze as analyze_impact, country_impact_rows, replace_section
from download_dashboard_series import download_country


class PaperAnalysisTests(unittest.TestCase):
    def test_impact_report_preserves_following_sections(self):
        original = '# Results\n\n## Explained impact\nold\n\n## Data reconstruction\nkeep\n'
        updated = replace_section(original, '## Explained impact', '## Explained impact\nnew')
        self.assertIn('## Data reconstruction\nkeep\n', updated)
        self.assertNotIn('old', updated)
        self.assertEqual(updated.count('## Explained impact'), 1)

    def test_dashboard_download_validates_and_sorts_observations(self):
        records = [dict(country='RU', date='2025-01-02', rate=2),
                   dict(country='RU', date='2025-01-01', rate=1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('download_dashboard_series.graphql_request',
                       return_value={'cenalertTimeseries': records}):
                summary = download_country('unused', 'RU', root, 0)
                self.assertEqual(summary['first_date'], '2025-01-01')
                self.assertEqual(summary['observations'], 2)
            saved = (root / 'RU.json').read_bytes()
            for bad in ([records[0], records[0]], [dict(country='RU', date='2025-01-01', rate=-1)]):
                with patch('download_dashboard_series.graphql_request',
                           return_value={'cenalertTimeseries': bad}):
                    with self.assertRaises(RuntimeError):
                        download_country('unused', 'RU', root, 0)
                self.assertEqual((root / 'RU.json').read_bytes(), saved)

    def test_explained_impact_counts_intervals_once_and_weights_countries_equally(self):
        labels = [dict(scope='country', country_code='AA', date='2025-01-01',
                       description='Block', metadata={'category':'filtering'}),
                  dict(scope='global', date='2025-01-01', description='Game')]
        rows = [dict(country=c, variant='labels_ensemble', selection='knee',
                     raw_anomalies=a, retained_anomalies=a) for c,a in [
            ('AA',[dict(start='2025-01-01',impact=90),dict(start='2025-02-01',impact=10)]),
            ('BB',[dict(start='2025-02-01',impact=10)])]]
        result = analyze_impact(dict(train_end='2024-12-31',test_end='2026-08-31',
                                    country_proximity_days=3,countries=rows),labels)
        s = result['summaries'][0]
        self.assertEqual(s['anomalies'],3)
        self.assertEqual(s['total_impact'],110)
        self.assertEqual(s['categories'][0]['cumulative_impact'],90)
        self.assertEqual(s['categories'][1]['anomalies'],0)
        self.assertAlmostEqual(s['mean_country_explained_share'],.45)
        plotted = country_impact_rows(result, limit=1)
        self.assertEqual([r[0] for r in plotted], ['AA', 'Remaining 1 countries'])
        self.assertEqual(sum(c['cumulative_impact'] for _, row in plotted for c in row.values()),110)
        self.assertEqual(sum(c['anomalies'] for _, row in plotted for c in row.values()),3)

    def test_repair_uses_preceding_date_pairs_and_excludes_bad_dates(self):
        original = {'2025-01-01': 2, '2025-01-02': 400, '2025-01-03': 6,
                    '2025-01-04': 80, '2025-01-05': 800}
        sample = {'2025-01-05': 1, '2025-01-04': 5, '2025-01-03': 3,
                  '2025-01-02': 1, '2025-01-01': 1}
        value, scale, pairs = reconstruct(original, sample, '2025-01-04',
                                          ['2025-01-02', '2025-01-04'], 2)
        self.assertEqual((value, scale, pairs), (10, 2, 2))
        self.assertIsNone(reconstruct(original, sample, '2025-01-04', [], 4)[0])
        self.assertEqual(reconstruct({}, {'2025-01-04': 0}, '2025-01-04', [], 10)[0], 0)

    def test_case_plot_preserves_stored_impact_threshold(self):
        self.assertEqual(plot_threshold(dict(threshold=3, diagnostic_threshold=4)), 3)
        self.assertEqual(plot_threshold(dict(threshold=None, diagnostic_threshold=4)), 4)
        self.assertIsNone(plot_threshold(dict(threshold=None)))

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
        self.assertEqual(label_kind(dict(scope='country', country_code='KZ', date='2026-08-13',
            description='Telegram outage', metadata={'category':'additional'})), 'other_explained')

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

    def test_balance_audit_counts_dense_intervals_only(self):
        rows = [dict(country='RU',configuration='labels_ensemble',selection='knee',
                     period='training', mode='standalone_dense',anomalies=n) for n in (2,5,3)]
        rows.append(dict(period='test',mode='standalone_dense',anomalies=100))
        result=verify_member_balance([dict(summary=rows)])
        self.assertEqual(result[0]['excess'],0)
        rows[1]['anomalies']=6
        self.assertEqual(verify_member_balance([dict(summary=rows)])[0]['excess'],1)


class NotificationAnalysisTests(unittest.TestCase):
    def test_calibration_counts_labels_once(self):
        reference = dict(applicable_labels=[1, 2, 3, 4],
                         references=[dict(labels=[1, 2]), dict(labels=[1])])
        self.assertEqual(calibration_probability(reference), .5)
        self.assertEqual(calibration_probability(dict(applicable_labels=[], references=[])), 0)

    def test_detected_filtered_and_retained(self):
        label = dict(scope='country', country_code='RU', date='2025-01-05')
        row = dict(raw_anomalies=[dict(start='2025-01-02')], retained_anomalies=[])
        self.assertEqual(label_status(row, label, 3), 'filtered')
        row['retained_anomalies'] = [dict(start='2025-01-01', component_starts=['2025-01-02'])]
        self.assertEqual(label_status(row, label, 3), 'retained')
        label['date'] = '2025-01-06'
        self.assertEqual(label_status(row, label, 3), 'not_detected')

    def test_region_group_is_counted_once_per_region(self):
        rows = region_months([dict(date='2025-01-01', country='IN;PK;RU')], ['2025-01','2025-02'])
        indexed = {(r['region'], r['month']):r['notifications'] for r in rows}
        self.assertEqual(indexed['APAC','2025-01'], 1)
        self.assertEqual(indexed['EECA','2025-01'], 1)
        self.assertEqual(indexed['APAC','2025-02'], 0)
        self.assertEqual(len(rows), 12)

    def test_twenty_month_evaluation(self):
        months = month_range('2025-01-01','2026-08-31')
        self.assertEqual(len(months), 20)
        self.assertEqual(months[-1], '2026-08')

    def test_grouping_evidence_counts_distinct_countries(self):
        label=dict(scope='global',date='2025-01-01',description='Outage')
        grouping=dict(grouped_notifications=[dict(date='2025-01-01',countries=['RU','UA'],anomalies=[{},{}])],
                      label_matching=dict(total=1,matched=1,groups=[dict(members=[
                          dict(country='RU',labels=[label,label]),dict(country='UA',labels=[label])])]))
        result=grouping_sensitivity(dict(available_grouping={'labels_ensemble/knee':{'5':grouping}}))
        self.assertEqual(result[0]['groups'][0]['global_matches'][0]['countries'],['RU','UA'])

    def test_grouping_selection_and_legacy_records(self):
        self.assertEqual(grouping_key({}), '5')
        self.assertEqual(grouping_key({'minimum_group_countries':4}), '4')

    def test_member_vote_is_distinct_from_independently_tuned_baseline(self):
        vote = {k:dict(voted=v) for k,v in
                [('chebyshev',False),('isolation_forest',True),('lof',True)]}
        anomaly = dict(start='2025-01-02',end='2025-01-02',impact=2,ensemble_vote=vote)
        context = dict(country_proximity_days=3, countries=[
            dict(country='AA',variant=v,selection='knee',raw_anomalies=a)
            for v,a in [('original',[anomaly]),('labels',[]),('labels_ensemble',[anomaly])]])
        labels = [dict(scope='country',country_code='AA',date='2025-01-01',description='Event'),
                  dict(scope='global',date='2025-01-01',description='Global event')]
        result = ensemble_only(context,labels,'2025-01-01','2025-12-31')
        self.assertEqual(result['intervals'],1)
        self.assertEqual(result['matched_intervals'],1)
        self.assertEqual(result['country_labels'],1)
        self.assertEqual(result['missed_by_original'],0)
        self.assertEqual(result['missed_by_labeled_single'],1)

    def test_calibration_counts_detected_anomalies(self):
        def context(n):
            return dict(countries=[dict(country='AA',variant='labels_ensemble',selection='knee',
                                       calibration=dict(matched_training_anomalies=n))])
        self.assertEqual(calibration_changes(context(1),context(0))[0]['after'],0)
        self.assertEqual(calibration_changes(context(2),context(3)),[])

    def test_distinct_countries_and_exact_date_vs_window(self):
        records = [dict(country=c, retained_anomalies=[dict(start='2025-01-01')]*n)
                   for c, n in [('AA', 2), ('BB', 1), ('CC', 1)]]
        label = dict(country_code='AA', scope='country', date='2025-01-03',
                     description='Blocking', metadata={'category':'filtering'})
        rows = same_day_groups(records, [label], 3)
        self.assertEqual(rows[0]['country_count'], 3)
        self.assertEqual(rows[0]['same_day_restriction_countries'], '')
        self.assertEqual(rows[0]['window_matched_restriction_countries'], 'AA')
        self.assertEqual(group_summary(rows)['largest_window_matched_restriction_group'], 3)
        self.assertEqual(group_summary(rows)['largest_same_day_restriction_group'], 0)
        label['date'] = '2025-01-01'
        self.assertEqual(same_day_groups(records, [label], 3)[0]['same_day_restriction_countries'], 'AA')

    def test_group_membership_and_coalesced_starts(self):
        records = [dict(country='AA', retained_anomalies=[dict(start='2025-01-01',
                                     component_starts=['2025-01-01','2025-01-04'])])]
        label = dict(country_code='BB', scope='country', date='2025-01-07',
                     description='Blocking', metadata={'category':'filtering'})
        self.assertEqual(same_day_groups(records, [label], 3)[0]['window_matched_restriction_events'], '')
        label['country_code'] = 'AA'
        row = same_day_groups(records, [label], 3)[0]
        self.assertEqual(row['window_matched_restriction_countries'], 'AA')
        self.assertEqual(row['same_day_restriction_countries'], '')

    def test_unmatched_and_global_events_do_not_imply_restrictions(self):
        records = [dict(country='AA', retained_anomalies=[dict(start='2025-01-01')])]
        labels = [dict(scope='global', date='2025-01-01', description='World Cup')]
        summary = group_summary(same_day_groups(records, labels, 3))
        self.assertEqual(summary['largest_same_day_restriction_group'], 0)
        self.assertEqual(group_summary([])['multi_country_dates'], 0)

    def test_group_is_one_matched_notification(self):
        label = dict(scope='global', date='2025-01-01', description='World Cup')
        group = dict(date='2025-01-01', anomalies=[dict(country='AA'),dict(country='BB')])
        grouping = dict(individual_notifications=[], grouped_notifications=[group],
            label_matching=dict(individuals=[], matched=1, groups=[dict(members=[
                dict(country='AA',labels=[label]),dict(country='BB',labels=[label])])]))
        context = dict(available_grouping={'labels_ensemble/knee': {'5':grouping}})
        events = [dict(analysis='notification', configuration='labels_ensemble', selection='knee',
            date='2025-01-01', country='AA;BB', category='other_explained', severity='high',event='World Cup')]
        rows = matched_notifications(context, events)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['notification_type'], 'group')
        self.assertEqual(rows[0]['category'], 'other_explained')


if __name__ == '__main__':
    unittest.main()
