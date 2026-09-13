use std::error::Error;
use std::fmt;

use chrono::NaiveDate;
use serde::Serialize;

use crate::Anomaly;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoalescingError {
    MissingPeakValue { start: NaiveDate },
}

impl fmt::Display for CoalescingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPeakValue { start } => write!(
                formatter,
                "cannot coalesce sparse anomaly starting {start}: peak_value is missing"
            ),
        }
    }
}

impl Error for CoalescingError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CoalescingSummary {
    pub maximum_gap_days: u32,
    pub input_count: usize,
    pub output_count: usize,
    pub merged_groups: usize,
    pub absorbed_anomalies: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoalescingResult {
    pub anomalies: Vec<Anomaly>,
    pub summary: CoalescingSummary,
}

/// Combines chains of sparse anomalies separated by at most `maximum_gap_days`
/// normal days. Dense anomalies always remain separate.
pub fn coalesce_sparse_anomalies(
    mut anomalies: Vec<Anomaly>,
    maximum_gap_days: u32,
) -> Result<CoalescingResult, CoalescingError> {
    let input_count = anomalies.len();
    anomalies.sort_by_key(|anomaly| (anomaly.start, anomaly.end));

    let mut output: Vec<Anomaly> = Vec::with_capacity(input_count);
    let mut merged_groups = 0;
    let mut absorbed_anomalies = 0;

    for anomaly in anomalies {
        let Some(previous) = output.last_mut() else {
            output.push(anomaly);
            continue;
        };
        if !can_coalesce(previous, &anomaly, maximum_gap_days) {
            output.push(anomaly);
            continue;
        }

        if previous.component_starts.is_empty() {
            previous.component_starts.push(previous.start);
            merged_groups += 1;
        }
        let starts = if anomaly.component_starts.is_empty() {
            vec![anomaly.start]
        } else {
            anomaly.component_starts.clone()
        };
        previous.component_starts.extend(starts);
        previous.component_starts.sort_unstable();
        previous.component_starts.dedup();

        let previous_peak = previous
            .peak_value
            .ok_or(CoalescingError::MissingPeakValue {
                start: previous.start,
            })?;
        let anomaly_peak = anomaly
            .peak_value
            .ok_or(CoalescingError::MissingPeakValue {
                start: anomaly.start,
            })?;
        if anomaly_peak > previous_peak {
            previous.peak = anomaly.peak;
            previous.peak_value = Some(anomaly_peak);
        }

        previous.end = previous.end.max(anomaly.end);
        previous.score = previous.score.max(anomaly.score);
        previous.residual = previous.residual.max(anomaly.residual);
        previous.impact += anomaly.impact;
        absorbed_anomalies += 1;
    }

    Ok(CoalescingResult {
        summary: CoalescingSummary {
            maximum_gap_days,
            input_count,
            output_count: output.len(),
            merged_groups,
            absorbed_anomalies,
        },
        anomalies: output,
    })
}

fn can_coalesce(left: &Anomaly, right: &Anomaly, maximum_gap_days: u32) -> bool {
    if !left.sparse || !right.sparse {
        return false;
    }
    let days_between_starts = right.start.signed_duration_since(left.end).num_days();
    let normal_days_between = days_between_starts.saturating_sub(1);
    normal_days_between <= i64::from(maximum_gap_days)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anomaly(start: &str, end: &str, peak: &str, peak_value: f64, sparse: bool) -> Anomaly {
        Anomaly {
            start: NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap(),
            end: NaiveDate::parse_from_str(end, "%Y-%m-%d").unwrap(),
            peak: NaiveDate::parse_from_str(peak, "%Y-%m-%d").unwrap(),
            peak_value: Some(peak_value),
            score: 0.0,
            residual: peak_value,
            impact: peak_value,
            sparse,
            component_starts: Vec::new(),
            ensemble_vote: None,
        }
    }

    #[test]
    fn coalesces_a_chain_of_nearby_sparse_anomalies() {
        let result = coalesce_sparse_anomalies(
            vec![
                anomaly("2025-01-01", "2025-01-01", "2025-01-01", 5.0, true),
                anomaly("2025-01-03", "2025-01-03", "2025-01-03", 9.0, true),
                anomaly("2025-01-05", "2025-01-06", "2025-01-06", 7.0, true),
            ],
            1,
        )
        .unwrap();

        assert_eq!(result.anomalies.len(), 1);
        assert_eq!(result.anomalies[0].start.to_string(), "2025-01-01");
        assert_eq!(result.anomalies[0].end.to_string(), "2025-01-06");
        assert_eq!(result.anomalies[0].peak.to_string(), "2025-01-03");
        assert_eq!(result.anomalies[0].peak_value, Some(9.0));
        assert_eq!(result.anomalies[0].impact, 21.0);
        assert_eq!(result.anomalies[0].component_starts.len(), 3);
        assert_eq!(result.summary.merged_groups, 1);
        assert_eq!(result.summary.absorbed_anomalies, 2);
    }

    #[test]
    fn does_not_merge_dense_or_distant_anomalies() {
        let result = coalesce_sparse_anomalies(
            vec![
                anomaly("2025-01-01", "2025-01-01", "2025-01-01", 5.0, true),
                anomaly("2025-01-03", "2025-01-03", "2025-01-03", 9.0, false),
                anomaly("2025-01-10", "2025-01-10", "2025-01-10", 7.0, true),
            ],
            1,
        )
        .unwrap();

        assert_eq!(result.anomalies.len(), 3);
        assert_eq!(result.summary.absorbed_anomalies, 0);
    }

    #[test]
    fn requires_peak_values_only_when_a_merge_is_attempted() {
        let mut second = anomaly("2025-01-03", "2025-01-03", "2025-01-03", 9.0, true);
        second.peak_value = None;
        let error = coalesce_sparse_anomalies(
            vec![
                anomaly("2025-01-01", "2025-01-01", "2025-01-01", 5.0, true),
                second,
            ],
            1,
        )
        .unwrap_err();

        assert!(matches!(error, CoalescingError::MissingPeakValue { .. }));
    }
}
