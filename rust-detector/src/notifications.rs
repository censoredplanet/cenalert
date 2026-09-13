use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::Anomaly;

const REVIEW_HINT: &str =
    "investigate a shared cross-country cause before country-specific censorship";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CountryAnomaly {
    pub country: String,
    #[serde(flatten)]
    pub anomaly: Anomaly,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SameDayOccurrence {
    pub date: NaiveDate,
    pub country_count: usize,
    pub countries: Vec<String>,
    pub anomaly_count: usize,
    pub total_impact: f64,
    pub grouped_as_notification: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiCountryNotification {
    pub date: NaiveDate,
    pub countries: Vec<String>,
    pub total_impact: f64,
    pub review_hint: &'static str,
    pub anomalies: Vec<CountryAnomaly>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CrossCountryGroupingResult {
    pub minimum_countries: usize,
    pub input_anomaly_count: usize,
    pub individual_notification_count: usize,
    pub grouped_notification_count: usize,
    pub same_day_occurrences: Vec<SameDayOccurrence>,
    pub individual_notifications: Vec<CountryAnomaly>,
    pub grouped_notifications: Vec<MultiCountryNotification>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupingError {
    MinimumCountriesTooSmall,
    EmptyCountry { anomaly_start: NaiveDate },
}

impl fmt::Display for GroupingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinimumCountriesTooSmall => {
                formatter.write_str("minimum country count must be at least 2")
            }
            Self::EmptyCountry { anomaly_start } => {
                write!(
                    formatter,
                    "anomaly starting {anomaly_start} has an empty country code"
                )
            }
        }
    }
}

impl Error for GroupingError {}

/// Summarizes every exact-date co-occurrence across at least two countries and
/// replaces occurrences meeting `minimum_countries` with grouped notifications.
pub fn group_same_day_anomalies(
    anomalies: Vec<CountryAnomaly>,
    minimum_countries: usize,
) -> Result<CrossCountryGroupingResult, GroupingError> {
    if minimum_countries < 2 {
        return Err(GroupingError::MinimumCountriesTooSmall);
    }
    for anomaly in &anomalies {
        if anomaly.country.trim().is_empty() {
            return Err(GroupingError::EmptyCountry {
                anomaly_start: anomaly.anomaly.start,
            });
        }
    }

    let input_anomaly_count = anomalies.len();
    let mut by_date: BTreeMap<NaiveDate, Vec<CountryAnomaly>> = BTreeMap::new();
    for mut anomaly in anomalies {
        anomaly.country = anomaly.country.trim().to_uppercase();
        by_date
            .entry(anomaly.anomaly.start)
            .or_default()
            .push(anomaly);
    }

    let mut same_day_occurrences = Vec::new();
    let mut individual_notifications = Vec::new();
    let mut grouped_notifications = Vec::new();

    for (date, mut dated_anomalies) in by_date {
        dated_anomalies.sort_by(|left, right| {
            left.country
                .cmp(&right.country)
                .then_with(|| left.anomaly.start.cmp(&right.anomaly.start))
        });
        let countries: Vec<String> = dated_anomalies
            .iter()
            .map(|anomaly| anomaly.country.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let country_count = countries.len();
        let total_impact = dated_anomalies
            .iter()
            .map(|anomaly| anomaly.anomaly.impact)
            .sum();
        let grouped = country_count >= minimum_countries;

        if country_count >= 2 {
            same_day_occurrences.push(SameDayOccurrence {
                date,
                country_count,
                countries: countries.clone(),
                anomaly_count: dated_anomalies.len(),
                total_impact,
                grouped_as_notification: grouped,
            });
        }

        if grouped {
            grouped_notifications.push(MultiCountryNotification {
                date,
                countries,
                total_impact,
                review_hint: REVIEW_HINT,
                anomalies: dated_anomalies,
            });
        } else {
            individual_notifications.extend(dated_anomalies);
        }
    }

    Ok(CrossCountryGroupingResult {
        minimum_countries,
        input_anomaly_count,
        individual_notification_count: individual_notifications.len(),
        grouped_notification_count: grouped_notifications.len(),
        same_day_occurrences,
        individual_notifications,
        grouped_notifications,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anomaly(country: &str, start: &str, impact: f64) -> CountryAnomaly {
        let date = NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap();
        CountryAnomaly {
            country: country.to_owned(),
            anomaly: Anomaly {
                start: date,
                end: date,
                peak: date,
                peak_value: Some(impact),
                score: 2.0,
                residual: 0.0,
                impact,
                sparse: false,
                component_starts: Vec::new(),
                ensemble_vote: None,
            },
        }
    }

    #[test]
    fn reports_all_cooccurrences_and_groups_only_at_the_threshold() {
        let result = group_same_day_anomalies(
            vec![
                anomaly("RU", "2025-01-01", 10.0),
                anomaly("TR", "2025-01-01", 20.0),
                anomaly("CN", "2025-01-01", 30.0),
                anomaly("BR", "2025-01-02", 5.0),
                anomaly("AR", "2025-01-02", 6.0),
                anomaly("US", "2025-01-03", 7.0),
            ],
            3,
        )
        .unwrap();

        assert_eq!(result.same_day_occurrences.len(), 2);
        assert_eq!(result.grouped_notifications.len(), 1);
        assert_eq!(result.grouped_notifications[0].countries.len(), 3);
        assert_eq!(result.grouped_notifications[0].total_impact, 60.0);
        assert_eq!(result.individual_notifications.len(), 3);
    }

    #[test]
    fn distinct_countries_control_the_grouping_threshold() {
        let result = group_same_day_anomalies(
            vec![
                anomaly("RU", "2025-01-01", 10.0),
                anomaly("RU", "2025-01-01", 20.0),
                anomaly("TR", "2025-01-01", 30.0),
            ],
            3,
        )
        .unwrap();

        assert!(result.grouped_notifications.is_empty());
        assert_eq!(result.same_day_occurrences[0].country_count, 2);
    }

    #[test]
    fn rejects_a_threshold_below_two() {
        assert_eq!(
            group_same_day_anomalies(Vec::new(), 1),
            Err(GroupingError::MinimumCountriesTooSmall)
        );
    }
}
