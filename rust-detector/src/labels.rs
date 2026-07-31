use std::error::Error;
use std::fmt;
use std::io::Read;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::Anomaly;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelScope {
    Country,
    #[serde(alias = "multi_country", alias = "multi-country")]
    Global,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLabel {
    pub date: NaiveDate,
    pub scope: LabelScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelCoverage {
    pub total: usize,
    pub matched: usize,
    pub missed: usize,
}

#[derive(Debug)]
pub enum LabelInputError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Empty,
}

impl fmt::Display for LabelInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read labels: {error}"),
            Self::Json(error) => write!(formatter, "invalid label JSON: {error}"),
            Self::Empty => formatter.write_str("label list is empty"),
        }
    }
}

impl Error for LabelInputError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Empty => None,
        }
    }
}

impl From<std::io::Error> for LabelInputError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for LabelInputError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub fn parse_labels_json<R: Read>(reader: R) -> Result<Vec<EventLabel>, LabelInputError> {
    let labels: Vec<EventLabel> = serde_json::from_reader(reader)?;
    if labels.is_empty() {
        return Err(LabelInputError::Empty);
    }
    Ok(labels)
}

pub fn label_coverage(
    labels: &[EventLabel],
    anomalies: &[Anomaly],
    proximity_days: u32,
) -> LabelCoverage {
    let proximity_days = i64::from(proximity_days);
    let matched = labels
        .iter()
        .filter(|label| {
            anomalies.iter().any(|anomaly| {
                anomaly
                    .start
                    .signed_duration_since(label.date)
                    .num_days()
                    .abs()
                    <= proximity_days
            })
        })
        .count();

    LabelCoverage {
        total: labels.len(),
        matched,
        missed: labels.len() - matched,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(date: &str, scope: LabelScope) -> EventLabel {
        EventLabel {
            date: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            scope,
            description: None,
        }
    }

    fn anomaly(start: &str) -> Anomaly {
        let date = NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap();
        Anomaly {
            start: date,
            end: date,
            peak: date,
            score: 1.0,
            residual: 0.0,
            impact: 1.0,
        }
    }

    #[test]
    fn parses_country_and_global_labels() {
        let labels = parse_labels_json(
            br#"[
                {"date": "2025-01-10", "scope": "country"},
                {"date": "2025-02-10", "scope": "global", "description": "Outage"}
            ]"#
            .as_slice(),
        )
        .unwrap();

        assert_eq!(labels[0].scope, LabelScope::Country);
        assert_eq!(labels[1].scope, LabelScope::Global);
        assert_eq!(labels[1].description.as_deref(), Some("Outage"));
    }

    #[test]
    fn rejects_an_empty_label_file() {
        assert!(matches!(
            parse_labels_json(b"[]".as_slice()),
            Err(LabelInputError::Empty)
        ));
    }

    #[test]
    fn matches_start_dates_on_both_sides_of_the_boundary() {
        let labels = vec![
            label("2025-01-10", LabelScope::Country),
            label("2025-02-10", LabelScope::Global),
            label("2025-03-10", LabelScope::Country),
        ];
        let anomalies = vec![
            anomaly("2025-01-07"),
            anomaly("2025-02-13"),
            anomaly("2025-03-14"),
        ];

        assert_eq!(
            label_coverage(&labels, &anomalies, 3),
            LabelCoverage {
                total: 3,
                matched: 2,
                missed: 1,
            }
        );
    }

    #[test]
    fn zero_day_proximity_requires_the_same_start_date() {
        let labels = vec![label("2025-01-10", LabelScope::Country)];

        assert_eq!(
            label_coverage(&labels, &[anomaly("2025-01-10")], 0).missed,
            0
        );
        assert_eq!(
            label_coverage(&labels, &[anomaly("2025-01-11")], 0).missed,
            1
        );
    }
}
