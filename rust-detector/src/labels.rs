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
    /// Additional countries where this country event may be used for attribution.
    /// These countries do not inherit the event as a tuning constraint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_matching_countries: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl EventLabel {
    fn matches_distance(&self, distance_days: i64, proximity_days: i64) -> bool {
        let window = match self.scope {
            LabelScope::Country => proximity_days,
            LabelScope::Global => proximity_days.min(1),
        };
        self.match_mode.matches(distance_days, window)
    }

    pub fn new(date: NaiveDate, scope: LabelScope) -> Self {
        Self {
            date,
            scope,
            match_mode: LabelMatchMode::default(),
            country_code: None,
            affected_countries: Vec::new(),
            additional_matching_countries: Vec::new(),
            description: None,
            title: None,
            id: None,
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
    InvalidCountryCode,
}

impl fmt::Display for LabelInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read labels: {error}"),
            Self::Json(error) => write!(formatter, "invalid label JSON: {error}"),
            Self::Empty => formatter.write_str("label list is empty"),
            Self::InvalidCountryCode => formatter.write_str("label country codes must be nonempty strings and agree between top level and metadata"),
        }
    }
}

impl Error for LabelInputError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Empty | Self::InvalidCountryCode => None,
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
    let mut labels: Vec<EventLabel> = serde_json::from_reader(reader)?;
    if labels.is_empty() {
        return Err(LabelInputError::Empty);
    }
    for label in &mut labels {
        let legacy = label
            .metadata
            .get("country_code")
            .map(|value| {
                value
                    .as_str()
                    .filter(|code| !code.trim().is_empty())
                    .ok_or(LabelInputError::InvalidCountryCode)
            })
            .transpose()?;
        if let Some(code) = &label.country_code {
            if code.trim().is_empty()
                || legacy.is_some_and(|other| !code.trim().eq_ignore_ascii_case(other.trim()))
            {
                return Err(LabelInputError::InvalidCountryCode);
            }
        }
        label.country_code = label
            .country_code
            .as_deref()
            .or(legacy)
            .map(|code| code.trim().to_ascii_uppercase());
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
                anomaly_start_dates(anomaly).any(|start| {
                    let distance_days = start.signed_duration_since(label.date).num_days();
                    label.matches_distance(distance_days, proximity_days)
                })
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
            anomaly_start_dates(anomaly)
                .map(|start| start.signed_duration_since(label.date).num_days())
                .filter(|distance| label.matches_distance(*distance, proximity_days))
                .min_by_key(|distance| distance.abs())
                .map(|distance| (index, label, distance))
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

fn anomaly_start_dates(anomaly: &Anomaly) -> impl Iterator<Item = NaiveDate> + '_ {
    std::iter::once(anomaly.start).chain(anomaly.component_starts.iter().copied())
}

pub(crate) fn required_country_labels(
    labels: &[EventLabel],
    country_code: Option<&str>,
) -> Vec<EventLabel> {
    labels
        .iter()
        .filter(|label| {
            label.scope == LabelScope::Country
                && label.match_mode == LabelMatchMode::Proximity
                && country_label_constrains_country(label, country_code)
        })
        .cloned()
        .collect()
}

fn label_applies_to_country(label: &EventLabel, country_code: Option<&str>) -> bool {
    match label.scope {
        LabelScope::Country => {
            label.country_code.as_deref().is_none_or(|label_country| {
                country_code.is_some_and(|country| country.eq_ignore_ascii_case(label_country))
            }) || country_code.is_some_and(|country| {
                label
                    .additional_matching_countries
                    .iter()
                    .any(|additional| additional.eq_ignore_ascii_case(country))
            })
        }
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

fn country_label_constrains_country(label: &EventLabel, country_code: Option<&str>) -> bool {
    label.country_code.as_deref().is_none_or(|label_country| {
        country_code.is_some_and(|country| country.eq_ignore_ascii_case(label_country))
    })
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
            peak_value: Some(1.0),
            score: 1.0,
            residual: 0.0,
            impact: 1.0,
            sparse: false,
            component_starts: Vec::new(),
            ensemble_vote: None,
        }
    }

    #[test]
    fn preserves_mixed_metadata_and_optional_identity() {
        let input = serde_json::json!([
            {"date":"2025-01-10", "scope":"country", "country_code":"AA",
             "title":"Example outage", "id":"example-id", "metadata":{
                "affected_services":["Example Chat"],
                "sources":[{"type":"news","name":"Example News","url":"https://example.com"}],
                "cause":"protests", "justification":"public safety"}},
            {"date":"2025-01-11", "scope":"global", "metadata":{
                "affected_services":"Example service, with legacy description", "source":"Other", "link":"https://example.org"}}
        ]);
        let labels = parse_labels_json(input.to_string().as_bytes()).unwrap();
        assert_eq!(serde_json::to_value(&labels).unwrap(), input);
        assert_eq!(required_country_labels(&labels, Some("AA")).len(), 1);
        assert!(required_country_labels(&labels, Some("BB")).is_empty());
    }

    #[test]
    fn nested_country_code_does_not_constrain_other_countries() {
        let labels = parse_labels_json(
            br#"[{"date":"2025-01-10","scope":"country","metadata":{"country_code":" aa "}}]"#
                .as_slice(),
        )
        .unwrap();
        assert_eq!(labels[0].country_code.as_deref(), Some("AA"));
        assert_eq!(required_country_labels(&labels, Some("AA")).len(), 1);
        assert!(required_country_labels(&labels, Some("BB")).is_empty());
        assert!(match_anomaly_to_label(&anomaly("2025-01-10"), &labels, Some("BB"), 3).is_none());
    }

    #[test]
    fn rejects_invalid_or_conflicting_country_codes() {
        for metadata in [
            serde_json::json!({"country_code":"BB"}),
            serde_json::json!({"country_code":[]}),
            serde_json::json!({"country_code":""}),
        ] {
            let input = serde_json::json!([{"date":"2025-01-10","scope":"country","country_code":"AA","metadata":metadata}]);
            assert!(matches!(
                parse_labels_json(input.to_string().as_bytes()),
                Err(LabelInputError::InvalidCountryCode)
            ));
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
            anomaly("2025-02-11"),
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
    fn global_window_is_one_day_on_both_sides() {
        let global = label("2025-01-10", LabelScope::Global);
        let country = label("2025-01-10", LabelScope::Country);
        for (day, expected) in [
            (7, false),
            (8, false),
            (9, true),
            (10, true),
            (11, true),
            (12, false),
            (13, false),
        ] {
            let a = anomaly(&format!("2025-01-{day:02}"));
            assert_eq!(
                label_coverage(std::slice::from_ref(&global), std::slice::from_ref(&a), 3).matched,
                usize::from(expected)
            );
            assert_eq!(
                match_anomaly_to_label(&a, std::slice::from_ref(&global), Some("RU"), 3).is_some(),
                expected
            );
            assert_eq!(
                label_coverage(std::slice::from_ref(&country), &[a], 3).matched,
                1
            );
        }
        assert!(match_anomaly_to_label(
            &anomaly("2025-01-11"),
            std::slice::from_ref(&global),
            None,
            0
        )
        .is_none());
        let mut exact = global;
        exact.match_mode = LabelMatchMode::Exact;
        assert!(match_anomaly_to_label(&anomaly("2025-01-11"), &[exact], None, 3).is_none());
    }

    #[test]
    fn matches_a_component_start_retained_by_coalescing() {
        let labels = vec![label("2025-01-03", LabelScope::Country)];
        let mut combined = anomaly("2025-01-01");
        combined.component_starts = vec![
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
        ];

        assert_eq!(label_coverage(&labels, &[combined.clone()], 0).missed, 0);
        assert_eq!(
            match_anomaly_to_label(&combined, &labels, Some("RU"), 0),
            Some(LabelMatch {
                label_index: 0,
                distance_days: 0,
            })
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
    fn exact_pseudo_labels_do_not_become_tuning_constraints() {
        let regular = label("2025-01-10", LabelScope::Country);
        let mut pseudo = label("2022-01-01", LabelScope::Country);
        pseudo.match_mode = LabelMatchMode::Exact;

        let required = required_country_labels(&[regular.clone(), pseudo], Some("RU"));

        assert_eq!(required, vec![regular]);
    }

    #[test]
    fn limits_global_matches_when_affected_countries_are_known() {
        let anomaly = anomaly("2025-01-10");
        let mut global = label("2025-01-10", LabelScope::Global);
        global.affected_countries = vec!["RU".to_owned(), "TR".to_owned()];

        assert!(match_anomaly_to_label(&anomaly, &[global.clone()], Some("RU"), 3).is_some());
        assert!(match_anomaly_to_label(&anomaly, &[global], Some("US"), 3).is_none());
    }

    #[test]
    fn country_event_can_match_an_explicit_additional_country() {
        let anomaly = anomaly("2025-01-10");
        let mut event = label("2025-01-10", LabelScope::Country);
        event.country_code = Some("RU".to_owned());
        event.additional_matching_countries = vec!["UA".to_owned()];

        assert!(match_anomaly_to_label(&anomaly, &[event.clone()], Some("RU"), 3).is_some());
        assert!(match_anomaly_to_label(&anomaly, &[event.clone()], Some("UA"), 3).is_some());
        assert!(match_anomaly_to_label(&anomaly, &[event], Some("BR"), 3).is_none());
    }

    #[test]
    fn additional_matching_country_does_not_inherit_tuning_constraint() {
        let mut event = label("2025-01-10", LabelScope::Country);
        event.country_code = Some("RU".to_owned());
        event.additional_matching_countries = vec!["UA".to_owned()];

        assert_eq!(
            required_country_labels(&[event.clone()], Some("RU")),
            vec![event.clone()]
        );
        assert!(required_country_labels(&[event], Some("UA")).is_empty());
    }
}
