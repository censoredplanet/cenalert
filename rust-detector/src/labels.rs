use std::collections::BTreeMap;
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelMatchMode {
    #[default]
    Proximity,
    Exact,
}

impl LabelMatchMode {
    fn matches(self, distance_days: i64, proximity_days: i64) -> bool {
        match self {
            Self::Proximity => distance_days.abs() <= proximity_days,
            Self::Exact => distance_days == 0,
        }
    }

    pub(crate) fn margin_days(self, proximity_days: i64) -> i64 {
        match self {
            Self::Proximity => proximity_days,
            Self::Exact => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLabel {
    pub date: NaiveDate,
    pub scope: LabelScope,
    #[serde(default, skip_serializing_if = "is_default_match_mode")]
    pub match_mode: LabelMatchMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country_code: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affected_countries: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl EventLabel {
    pub fn new(date: NaiveDate, scope: LabelScope) -> Self {
        Self {
            date,
            scope,
            match_mode: LabelMatchMode::default(),
            country_code: None,
            affected_countries: Vec::new(),
            description: None,
            metadata: BTreeMap::new(),
        }
    }
}

fn is_default_match_mode(match_mode: &LabelMatchMode) -> bool {
    *match_mode == LabelMatchMode::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelMatch {
    pub label_index: usize,
    pub distance_days: i64,
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
                let distance_days = anomaly.start.signed_duration_since(label.date).num_days();
                label.match_mode.matches(distance_days, proximity_days)
            })
        })
        .count();

    LabelCoverage {
        total: labels.len(),
        matched,
        missed: labels.len() - matched,
    }
}

pub fn match_anomaly_to_label(
    anomaly: &Anomaly,
    labels: &[EventLabel],
    country_code: Option<&str>,
    proximity_days: u32,
) -> Option<LabelMatch> {
    let proximity_days = i64::from(proximity_days);
    labels
        .iter()
        .enumerate()
        .filter(|(_, label)| label_applies_to_country(label, country_code))
        .filter_map(|(index, label)| {
            let distance_days = anomaly.start.signed_duration_since(label.date).num_days();
            label
                .match_mode
                .matches(distance_days, proximity_days)
                .then_some((index, label, distance_days))
        })
        .min_by_key(|(index, label, distance_days)| {
            (
                distance_days.abs(),
                scope_priority(label.scope),
                label.date,
                *index,
            )
        })
        .map(|(label_index, _, distance_days)| LabelMatch {
            label_index,
            distance_days,
        })
}

pub(crate) fn required_country_labels(
    labels: &[EventLabel],
    country_code: Option<&str>,
) -> Vec<EventLabel> {
    labels
        .iter()
        .filter(|label| {
            label.scope == LabelScope::Country && label_applies_to_country(label, country_code)
        })
        .cloned()
        .collect()
}

fn label_applies_to_country(label: &EventLabel, country_code: Option<&str>) -> bool {
    match label.scope {
        LabelScope::Country => label.country_code.as_deref().is_none_or(|label_country| {
            country_code.is_some_and(|country| country.eq_ignore_ascii_case(label_country))
        }),
        LabelScope::Global => {
            label.affected_countries.is_empty()
                || country_code.is_some_and(|country| {
                    label
                        .affected_countries
                        .iter()
                        .any(|affected| affected.eq_ignore_ascii_case(country))
                })
        }
    }
}

fn scope_priority(scope: LabelScope) -> u8 {
    match scope {
        LabelScope::Country => 0,
        LabelScope::Global => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(date: &str, scope: LabelScope) -> EventLabel {
        EventLabel::new(NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(), scope)
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
        assert_eq!(labels[0].match_mode, LabelMatchMode::Proximity);
        assert_eq!(labels[1].scope, LabelScope::Global);
        assert_eq!(labels[1].description.as_deref(), Some("Outage"));
    }

    #[test]
    fn parses_an_exact_match_label() {
        let labels = parse_labels_json(
            br#"[
                {"date": "2022-01-01", "scope": "country", "match_mode": "exact"}
            ]"#
            .as_slice(),
        )
        .unwrap();

        assert_eq!(labels[0].match_mode, LabelMatchMode::Exact);
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

    #[test]
    fn exact_labels_ignore_the_configured_proximity() {
        let mut exact = label("2022-01-01", LabelScope::Country);
        exact.match_mode = LabelMatchMode::Exact;

        assert_eq!(
            label_coverage(&[exact.clone()], &[anomaly("2022-01-01")], 3).missed,
            0
        );
        assert_eq!(
            label_coverage(&[exact.clone()], &[anomaly("2022-01-02")], 3).missed,
            1
        );
        assert!(match_anomaly_to_label(&anomaly("2022-01-02"), &[exact], Some("RU"), 3).is_none());
    }

    #[test]
    fn prioritizes_a_country_label_when_distance_is_tied() {
        let anomaly = anomaly("2025-01-10");
        let labels = vec![
            label("2025-01-10", LabelScope::Global),
            label("2025-01-10", LabelScope::Country),
        ];

        assert_eq!(
            match_anomaly_to_label(&anomaly, &labels, Some("RU"), 3),
            Some(LabelMatch {
                label_index: 1,
                distance_days: 0,
            })
        );
    }

    #[test]
    fn filters_country_labels_by_country_code() {
        let mut ru = label("2025-01-10", LabelScope::Country);
        ru.country_code = Some("RU".to_owned());
        let mut tr = label("2025-01-11", LabelScope::Country);
        tr.country_code = Some("TR".to_owned());
        let global = label("2025-01-12", LabelScope::Global);

        assert_eq!(
            required_country_labels(&[ru, tr, global], Some("RU")).len(),
            1
        );
    }

    #[test]
    fn limits_global_matches_when_affected_countries_are_known() {
        let anomaly = anomaly("2025-01-10");
        let mut global = label("2025-01-10", LabelScope::Global);
        global.affected_countries = vec!["RU".to_owned(), "TR".to_owned()];

        assert!(match_anomaly_to_label(&anomaly, &[global.clone()], Some("RU"), 3).is_some());
        assert!(match_anomaly_to_label(&anomaly, &[global], Some("US"), 3).is_none());
    }
}
