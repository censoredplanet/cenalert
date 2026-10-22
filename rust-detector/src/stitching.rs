use std::collections::BTreeMap;

use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::Observation;

#[derive(Debug, Clone, Deserialize)]
pub struct SamplePoint {
    pub date: NaiveDate,
    pub value: f64,
}

#[derive(Debug, Deserialize)]
pub struct SampleWindow {
    pub source: String,
    pub collected_on: NaiveDate,
    pub samples: Vec<Vec<SamplePoint>>,
    #[serde(default)]
    pub coarse: Option<CoarseSamples>,
}

#[derive(Debug, Deserialize)]
pub struct CoarseSamples {
    pub source: String,
    pub samples: Vec<Vec<SamplePoint>>,
}

#[derive(Debug, Deserialize)]
pub struct SampleArchive {
    pub country: String,
    pub topic: String,
    pub windows: Vec<SampleWindow>,
}

#[derive(Debug, Serialize)]
pub struct WindowSummary {
    pub source: String,
    pub collected_on: NaiveDate,
    pub complete_dates: usize,
    pub incomplete_dates: usize,
    pub eligible_dates: usize,
    pub anchor_pairs: usize,
    pub scale: Option<f64>,
    pub scale_method: Option<&'static str>,
    pub coarse: Option<CoarseCalibration>,
    pub written_dates: usize,
    pub unresolved_dates: usize,
}

#[derive(Debug, Serialize)]
pub struct CoarseCalibration {
    pub source: String,
    pub anchor_pairs: usize,
    pub fine_coarse_pairs: usize,
    pub scale: Option<f64>,
    pub coarse_over_daily_mean: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct RebuildResult {
    pub country: String,
    pub replace_from: NaiveDate,
    pub through_date: NaiveDate,
    pub settling_days: u32,
    pub windows: Vec<WindowSummary>,
    pub unresolved_dates: Vec<NaiveDate>,
    /// Present only when every requested date has a calibrated value.
    pub series: Option<Vec<Observation>>,
}

#[derive(Debug, Clone, Copy)]
pub struct RebuildConfig {
    pub replace_from: NaiveDate,
    pub through_date: NaiveDate,
    pub as_of: NaiveDate,
    pub settling_days: u32,
    pub sample_count: usize,
    pub minimum_overlap: usize,
    pub minimum_coarse_overlap: Option<usize>,
}

fn combine(
    window: &SampleWindow,
    expected: usize,
) -> Result<(BTreeMap<NaiveDate, f64>, usize), String> {
    combine_samples(&window.samples, &window.source, expected, true)
}

fn combine_samples(
    samples: &[Vec<SamplePoint>],
    source: &str,
    expected: usize,
    intersection: bool,
) -> Result<(BTreeMap<NaiveDate, f64>, usize), String> {
    if expected == 0 || samples.len() != expected {
        return Err(format!("{source}: expected {expected} samples"));
    }
    let mut grouped: BTreeMap<NaiveDate, Vec<f64>> = BTreeMap::new();
    for sample in samples {
        if sample.is_empty() {
            return Err(format!("{source}: empty sample"));
        }
        let mut previous = None;
        for point in sample {
            if !point.value.is_finite()
                || point.value < 0.0
                || previous.is_some_and(|d| d >= point.date)
            {
                return Err(format!(
                    "{}: invalid value or unordered/duplicate date {}",
                    source, point.date
                ));
            }
            previous = Some(point.date);
            grouped.entry(point.date).or_default().push(point.value);
        }
    }
    let mut complete = BTreeMap::new();
    let mut incomplete = 0;
    for (day, values) in grouped {
        if values.len() != expected {
            incomplete += 1;
        } else {
            let mean = if intersection && values.contains(&0.0) {
                0.0
            } else {
                values.iter().map(|v| v / expected as f64).sum()
            };
            if !mean.is_finite() {
                return Err(format!("{source}: non-finite combined value"));
            }
            complete.insert(day, mean);
        }
    }
    Ok((complete, incomplete))
}

fn median_ratio(mut ratios: Vec<f64>, minimum: usize) -> Option<f64> {
    ratios.retain(|v| v.is_finite() && *v > 0.0);
    if ratios.len() < minimum {
        return None;
    }
    ratios.sort_by(f64::total_cmp);
    let mid = ratios.len() / 2;
    Some(if ratios.len().is_multiple_of(2) {
        ratios[mid - 1] / 2.0 + ratios[mid] / 2.0
    } else {
        ratios[mid]
    })
}

fn week_mean(values: &BTreeMap<NaiveDate, f64>, start: NaiveDate) -> Option<f64> {
    (0..7)
        .map(|offset| {
            start
                .checked_add_signed(Duration::days(offset))
                .and_then(|d| values.get(&d))
                .map(|v| v / 7.0)
        })
        .sum()
}

fn calibrate_coarse(
    window: &SampleWindow,
    values: &BTreeMap<NaiveDate, f64>,
    coarse_reference: &mut BTreeMap<NaiveDate, f64>,
    daily_scale: Option<f64>,
    cutoff: NaiveDate,
    config: RebuildConfig,
) -> Result<Option<CoarseCalibration>, String> {
    let (Some(coarse), Some(minimum)) = (&window.coarse, config.minimum_coarse_overlap) else {
        return Ok(None);
    };
    for sample in &coarse.samples {
        if sample.len() < 2
            || sample
                .windows(2)
                .any(|pair| (pair[1].date - pair[0].date).num_days() != 7)
        {
            return Err(format!(
                "{}: coarse samples must have weekly cadence",
                coarse.source
            ));
        }
    }
    let (coarse_values, _) =
        combine_samples(&coarse.samples, &coarse.source, config.sample_count, true)?;
    // Calibration uses raw daily means before intersection suppression.
    let (daily_means, _) =
        combine_samples(&window.samples, &window.source, config.sample_count, false)?;
    let eligible: BTreeMap<_, _> = coarse_values
        .into_iter()
        .filter(|(date, _)| {
            date.checked_add_signed(Duration::days(6))
                .is_some_and(|end| end <= cutoff && end <= config.through_date)
        })
        .collect();
    let fine_coarse: Vec<_> = eligible
        .iter()
        .filter_map(|(date, coarse)| {
            week_mean(&daily_means, *date)
                .filter(|fine| *fine > 0.0 && *coarse > 0.0)
                .map(|fine| coarse / fine)
        })
        .collect();
    let fine_coarse_pairs = fine_coarse.len();
    let coarse_over_daily_mean = median_ratio(fine_coarse, minimum);
    let previous_ratios: Vec<_> = eligible
        .iter()
        .filter_map(|(date, new)| {
            coarse_reference
                .get(date)
                .filter(|old| **old > 0.0 && *new > 0.0)
                .map(|old| old / new)
        })
        .collect();
    let mut anchor_pairs = previous_ratios.len();
    let scale = if let (Some(daily), Some(ratio)) = (daily_scale, coarse_over_daily_mean) {
        Some(daily / ratio)
    } else if let Some(scale) = median_ratio(previous_ratios, minimum) {
        Some(scale)
    } else {
        let ratios: Vec<_> = eligible
            .iter()
            .filter_map(|(date, coarse)| {
                week_mean(values, *date)
                    .filter(|reference| *reference > 0.0 && *coarse > 0.0)
                    .map(|reference| reference / coarse)
            })
            .collect();
        anchor_pairs = ratios.len();
        median_ratio(ratios, minimum)
    }
    .filter(|scale| scale.is_finite() && *scale > 0.0);
    if let Some(scale) = scale {
        for (date, value) in eligible {
            let scaled = value * scale;
            if scaled.is_finite() {
                coarse_reference.insert(date, scaled);
            }
        }
    }
    Ok(Some(CoarseCalibration {
        source: coarse.source.clone(),
        anchor_pairs,
        fine_coarse_pairs,
        scale,
        coarse_over_daily_mean,
    }))
}

/// Rebuild a suffix from complete, settled samples, retaining the supplied prefix.
pub fn rebuild_series(
    archive: &SampleArchive,
    baseline: &[Observation],
    config: RebuildConfig,
) -> Result<RebuildResult, String> {
    if config.replace_from > config.through_date
        || config.through_date > config.as_of
        || config.sample_count == 0
        || config.minimum_overlap == 0
        || config.minimum_coarse_overlap.is_some_and(|n| n < 3)
        || baseline.is_empty()
    {
        return Err("invalid rebuild dates, sample count, overlap, or baseline".into());
    }
    let mut previous = None;
    for point in baseline {
        if previous.is_some_and(|d| d >= point.date)
            || !point.value.is_finite()
            || point.value < 0.0
            || point.country.as_deref() != Some(archive.country.as_str())
            || point.topic.as_deref() != Some(archive.topic.as_str())
        {
            return Err(
                "baseline must be ordered, finite, and match the archive country/topic".into(),
            );
        }
        previous = Some(point.date);
    }
    if baseline[0].date >= config.replace_from {
        return Err("baseline must include a prefix before replace-from".into());
    }
    let mut values: BTreeMap<NaiveDate, f64> = baseline
        .iter()
        .filter(|p| p.date < config.replace_from)
        .map(|p| (p.date, p.value))
        .collect();
    let prefix_end = config.replace_from.pred_opt().ok_or("date underflow")?;
    let expected_prefix = (prefix_end - baseline[0].date).num_days() + 1;
    if values.len() as i64 != expected_prefix {
        return Err("baseline prefix must contain every day before replace-from".into());
    }
    let mut windows: Vec<_> = archive
        .windows
        .iter()
        .filter(|w| w.collected_on <= config.as_of)
        .collect();
    windows.sort_by(|a, b| (a.collected_on, &a.source).cmp(&(b.collected_on, &b.source)));
    if windows
        .windows(2)
        .any(|w| w[0].collected_on == w[1].collected_on && w[0].source == w[1].source)
    {
        return Err("duplicate archive window".into());
    }
    let mut summaries = Vec::new();
    let mut coarse_reference = BTreeMap::new();
    for window in windows {
        let (combined, incomplete_dates) = combine(window, config.sample_count)?;
        let cutoff = window
            .collected_on
            .checked_sub_signed(Duration::days(config.settling_days as i64))
            .ok_or("settling period exceeds supported dates")?;
        let eligible: BTreeMap<_, _> = combined
            .iter()
            .filter(|(d, _)| **d <= cutoff && **d <= config.through_date)
            .map(|(d, v)| (*d, *v))
            .collect();
        let ratios: Vec<f64> = eligible
            .iter()
            .filter_map(|(day, new)| {
                values
                    .get(day)
                    .filter(|old| **old > 0.0 && *new > 0.0)
                    .map(|old| old / new)
            })
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        let anchor_pairs = ratios.len();
        let daily_scale = median_ratio(ratios, config.minimum_overlap);
        let coarse = calibrate_coarse(
            window,
            &values,
            &mut coarse_reference,
            daily_scale,
            cutoff,
            config,
        )?;
        let coarse_scale = coarse
            .as_ref()
            .and_then(|c| Some(c.scale? * c.coarse_over_daily_mean?))
            .filter(|s| s.is_finite() && *s > 0.0);
        let scale = daily_scale.or(coarse_scale);
        let scale_method = daily_scale
            .map(|_| "daily_overlap")
            .or(coarse_scale.map(|_| "coarse_bridge"));
        let mut written_dates = 0;
        let mut unresolved_dates = 0;
        for (day, raw) in &eligible {
            if *day < config.replace_from {
                continue;
            }
            let scaled = if *raw == 0.0 {
                Some(0.0)
            } else {
                scale.map(|s| raw * s)
            };
            if let Some(value) = scaled.filter(|v| v.is_finite()) {
                values.insert(*day, value);
                written_dates += 1;
            } else {
                // A newer unresolved positive observation supersedes an older zero.
                values.remove(day);
                unresolved_dates += 1;
            }
        }
        summaries.push(WindowSummary {
            source: window.source.clone(),
            collected_on: window.collected_on,
            complete_dates: combined.len(),
            incomplete_dates,
            eligible_dates: eligible.len(),
            anchor_pairs,
            scale,
            scale_method,
            coarse,
            written_dates,
            unresolved_dates,
        });
    }
    let mut missing = Vec::new();
    let mut day = config.replace_from;
    loop {
        if !values.contains_key(&day) {
            missing.push(day);
        }
        if day == config.through_date {
            break;
        }
        day = day.succ_opt().ok_or("date overflow")?;
    }
    let series = missing.is_empty().then(|| {
        values
            .into_iter()
            .map(|(date, value)| Observation {
                date,
                value,
                country: Some(archive.country.clone()),
                topic: Some(archive.topic.clone()),
            })
            .collect()
    });
    Ok(RebuildResult {
        country: archive.country.clone(),
        replace_from: config.replace_from,
        through_date: config.through_date,
        settling_days: config.settling_days,
        windows: summaries,
        unresolved_dates: missing,
        series,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn day(n: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, n).unwrap()
    }
    fn window(name: &str, collected: u32, samples: Vec<Vec<(u32, f64)>>) -> SampleWindow {
        SampleWindow {
            coarse: None,
            source: name.into(),
            collected_on: day(collected),
            samples: samples
                .into_iter()
                .map(|s| {
                    s.into_iter()
                        .map(|(d, value)| SamplePoint {
                            date: day(d),
                            value,
                        })
                        .collect()
                })
                .collect(),
        }
    }
    fn baseline() -> Vec<Observation> {
        (1..=3)
            .map(|d| Observation {
                date: day(d),
                value: 20.0,
                country: Some("PK".into()),
                topic: Some("vpn".into()),
            })
            .collect()
    }
    fn config() -> RebuildConfig {
        RebuildConfig {
            replace_from: day(3),
            through_date: day(3),
            as_of: day(31),
            settling_days: 2,
            sample_count: 2,
            minimum_overlap: 2,
            minimum_coarse_overlap: None,
        }
    }
    fn archive(windows: Vec<SampleWindow>) -> SampleArchive {
        SampleArchive {
            country: "PK".into(),
            topic: "vpn".into(),
            windows,
        }
    }
    #[test]
    fn incomplete_dates_are_missing_and_zero_is_retained() {
        let w = window(
            "a",
            5,
            vec![vec![(1, 2.0), (2, 0.0), (3, 0.0)], vec![(1, 4.0), (2, 5.0)]],
        );
        let (v, missing) = combine(&w, 2).unwrap();
        assert_eq!(missing, 1);
        assert_eq!(v[&day(1)], 3.0);
        assert_eq!(v[&day(2)], 0.0);
        assert!(!v.contains_key(&day(3)));
    }
    #[test]
    fn later_samples_revise_history_without_changing_prefix() {
        let ws = vec![
            window("a", 5, vec![vec![(1, 10.0), (2, 10.0), (3, 50.0)]; 2]),
            window("b", 6, vec![vec![(1, 10.0), (2, 10.0), (3, 15.0)]; 2]),
        ];
        let r = rebuild_series(&archive(ws), &baseline(), config()).unwrap();
        let s = r.series.unwrap();
        assert_eq!(s[0], baseline()[0]);
        assert_eq!(s[2].value, 30.0);
    }
    #[test]
    fn recent_or_future_batches_cannot_supply_values() {
        for collected in [4, 31] {
            let a = archive(vec![window(
                "a",
                collected,
                vec![vec![(1, 10.0), (2, 10.0), (3, 15.0)]; 2],
            )]);
            let mut c = config();
            c.as_of = day(30);
            let r = rebuild_series(&a, &baseline(), c).unwrap();
            assert!(r.series.is_none());
            assert_eq!(r.unresolved_dates, vec![day(3)]);
        }
    }
    #[test]
    fn unknown_scale_blocks_positive_values_but_not_zeros() {
        for raw in [0.0, 5.0] {
            let a = archive(vec![window("a", 5, vec![vec![(3, raw)]; 2])]);
            let r = rebuild_series(&a, &baseline(), config()).unwrap();
            assert_eq!(r.series.is_some(), raw == 0.0);
        }
    }
    #[test]
    fn newer_unscaled_positive_does_not_keep_an_old_zero() {
        let a = archive(vec![
            window("a", 5, vec![vec![(3, 0.0)]; 2]),
            window("b", 6, vec![vec![(3, 5.0)]; 2]),
        ]);
        assert!(rebuild_series(&a, &baseline(), config())
            .unwrap()
            .series
            .is_none());
    }
    #[test]
    fn missing_prefix_day_fails() {
        let mut points = baseline();
        points.remove(1);
        assert!(rebuild_series(&archive(vec![]), &points, config()).is_err());
    }

    fn sparse_coarse_case(
        reference_scale: f64,
        raw_scale: f64,
        coarse_scale: f64,
    ) -> (SampleArchive, Vec<Observation>, RebuildConfig) {
        let mut sample = window(
            "daily",
            31,
            vec![
                (1..=28)
                    .map(|d| (d, if d == 25 { 4.0 * raw_scale } else { 0.0 }))
                    .collect(),
                (1..=28).map(|d| (d, 4.0 * raw_scale)).collect(),
            ],
        );
        sample.coarse = Some(CoarseSamples {
            source: "weekly".into(),
            samples: vec![
                (0..5)
                    .map(|i| SamplePoint {
                        date: day(1) + Duration::days(i * 7),
                        value: 10.0 * coarse_scale
                    })
                    .collect();
                2
            ],
        });
        let baseline = (1..=21)
            .map(|d| Observation {
                date: day(d),
                value: 2.0 * reference_scale,
                country: Some("PK".into()),
                topic: Some("vpn".into()),
            })
            .collect();
        let config = RebuildConfig {
            replace_from: day(22),
            through_date: day(28),
            minimum_coarse_overlap: Some(3),
            ..config()
        };
        (archive(vec![sample]), baseline, config)
    }

    #[test]
    fn coarse_bridge_preserves_reference_units_and_raw_normalization() {
        for reference_scale in [1.0, 10.0] {
            for (raw_scale, coarse_scale) in [(1.0, 1.0), (7.0, 1.0), (1.0, 5.0)] {
                let (archive, baseline, config) =
                    sparse_coarse_case(reference_scale, raw_scale, coarse_scale);
                let result = rebuild_series(&archive, &baseline, config).unwrap();
                assert_eq!(result.windows[0].scale_method, Some("coarse_bridge"));
                let series = result.series.unwrap();
                assert!((series[24].value - 4.0 * reference_scale).abs() < 1e-10);
                assert_eq!(series[23].value, 0.0);
                assert_eq!(&series[..21], baseline.as_slice());
            }
        }
    }

    #[test]
    fn coarse_calibration_requires_enough_complete_weeks() {
        let (archive, baseline, mut config) = sparse_coarse_case(1.0, 1.0, 1.0);
        for minimum in [None, Some(4)] {
            config.minimum_coarse_overlap = minimum;
            let result = rebuild_series(&archive, &baseline, config).unwrap();
            assert_eq!(result.unresolved_dates, vec![day(25)]);
        }
        let mut values = BTreeMap::from([(day(1), 3.0), (day(7), 3.0)]);
        assert_eq!(week_mean(&values, day(1)), None);
        values.extend((1..=7).map(|d| (day(d), 3.0)));
        assert!((week_mean(&values, day(1)).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn coarse_calibration_rejects_nonweekly_samples() {
        let (mut archive, baseline, config) = sparse_coarse_case(1.0, 1.0, 1.0);
        archive.windows[0].coarse.as_mut().unwrap().samples[0][1].date = day(9);
        assert!(rebuild_series(&archive, &baseline, config)
            .unwrap_err()
            .contains("weekly cadence"));
    }

    #[test]
    fn coarse_calibration_carries_previous_absolute_scale() {
        let (archive, _, config) = sparse_coarse_case(1.0, 1.0, 1.0);
        let mut reference: BTreeMap<_, _> =
            [1, 8, 15, 22].into_iter().map(|d| (day(d), 30.0)).collect();
        let calibration = calibrate_coarse(
            &archive.windows[0],
            &BTreeMap::new(),
            &mut reference,
            None,
            day(28),
            config,
        )
        .unwrap()
        .unwrap();
        assert_eq!(calibration.scale, Some(3.0));
        assert_eq!(reference[&day(22)], 30.0);
        assert!(!reference.contains_key(&day(29)));
    }

    #[test]
    fn duplicate_dates_and_wrong_sample_counts_fail() {
        let w = window("a", 5, vec![vec![(1, 1.0), (1, 2.0)]; 2]);
        assert!(combine(&w, 2).is_err());
        assert!(combine(&w, 45).is_err());
    }
}
