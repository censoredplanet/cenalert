use serde::Serialize;

use crate::{match_anomaly_to_label, Anomaly, EventLabel};

pub struct FilterContext<'a> {
    pub labels: &'a [EventLabel],
    pub country_code: &'a str,
    pub proximity_days: u32,
}

pub trait AnomalyFilter {
    fn name(&self) -> &'static str;
    fn rejection_reason(&self) -> &'static str;
    fn apply(&self, anomalies: Vec<Anomaly>, context: &FilterContext<'_>) -> FilterApplication;
}

pub struct FilterApplication {
    pub retained: Vec<Anomaly>,
    pub discarded: Vec<Anomaly>,
    pub impact_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FilterStageSummary {
    pub filter: &'static str,
    pub input_count: usize,
    pub retained_count: usize,
    pub discarded_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impact_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiscardedAnomaly {
    pub filter: &'static str,
    pub reason: &'static str,
    pub anomaly: Anomaly,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FilteringResult {
    pub stages: Vec<FilterStageSummary>,
    pub anomalies: Vec<Anomaly>,
    pub discarded: Vec<DiscardedAnomaly>,
}

pub fn apply_anomaly_filters(
    anomalies: Vec<Anomaly>,
    context: &FilterContext<'_>,
    filters: &[Box<dyn AnomalyFilter>],
) -> FilteringResult {
    let mut retained = anomalies;
    let mut discarded = Vec::new();
    let mut stages = Vec::new();
    for filter in filters {
        let input_count = retained.len();
        let application = filter.apply(retained, context);
        stages.push(FilterStageSummary {
            filter: filter.name(),
            input_count,
            retained_count: application.retained.len(),
            discarded_count: application.discarded.len(),
            impact_threshold: application.impact_threshold,
        });
        discarded.extend(
            application
                .discarded
                .into_iter()
                .map(|anomaly| DiscardedAnomaly {
                    filter: filter.name(),
                    reason: filter.rejection_reason(),
                    anomaly,
                }),
        );
        retained = application.retained;
    }
    FilteringResult {
        stages,
        anomalies: retained,
        discarded,
    }
}

pub struct MinimumLabeledImpactFilter;

impl AnomalyFilter for MinimumLabeledImpactFilter {
    fn name(&self) -> &'static str {
        "minimum_labeled_impact"
    }

    fn rejection_reason(&self) -> &'static str {
        "impact is below the lowest-impact labeled anomaly"
    }

    fn apply(&self, anomalies: Vec<Anomaly>, context: &FilterContext<'_>) -> FilterApplication {
        let impact_threshold = anomalies
            .iter()
            .filter(|anomaly| {
                match_anomaly_to_label(
                    anomaly,
                    context.labels,
                    Some(context.country_code),
                    context.proximity_days,
                )
                .is_some()
            })
            .map(|anomaly| anomaly.impact)
            .filter(|impact| impact.is_finite())
            .min_by(f64::total_cmp);

        let Some(threshold) = impact_threshold else {
            return FilterApplication {
                retained: anomalies,
                discarded: Vec::new(),
                impact_threshold: None,
            };
        };
        let (retained, discarded) = anomalies
            .into_iter()
            .partition(|anomaly| anomaly.impact >= threshold);
        FilterApplication {
            retained,
            discarded,
            impact_threshold: Some(threshold),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;
    use crate::{EventLabel, LabelScope};

    fn anomaly(start: &str, impact: f64) -> Anomaly {
        let date = NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap();
        Anomaly {
            start: date,
            end: date,
            peak: date,
            score: 2.0,
            residual: 0.0,
            impact,
            ensemble_vote: None,
        }
    }

    fn label(country: &str, date: &str) -> EventLabel {
        let mut label = EventLabel::new(
            NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            LabelScope::Country,
        );
        label.country_code = Some(country.to_owned());
        label
    }

    #[test]
    fn discards_anomalies_below_the_weakest_labeled_anomaly() {
        let labels = vec![label("RU", "2025-01-10"), label("RU", "2025-01-20")];
        let context = FilterContext {
            labels: &labels,
            country_code: "RU",
            proximity_days: 3,
        };
        let result = apply_anomaly_filters(
            vec![
                anomaly("2025-01-01", 4.0),
                anomaly("2025-01-10", 10.0),
                anomaly("2025-01-20", 20.0),
                anomaly("2025-02-01", 12.0),
            ],
            &context,
            &[Box::new(MinimumLabeledImpactFilter)],
        );

        assert_eq!(result.stages[0].impact_threshold, Some(10.0));
        assert_eq!(result.anomalies.len(), 3);
        assert_eq!(result.discarded.len(), 1);
        assert_eq!(result.discarded[0].anomaly.impact, 4.0);
    }

    #[test]
    fn retains_everything_when_no_anomaly_matches_a_label() {
        let labels = vec![label("RU", "2025-03-01")];
        let context = FilterContext {
            labels: &labels,
            country_code: "RU",
            proximity_days: 3,
        };
        let result = apply_anomaly_filters(
            vec![anomaly("2025-01-01", 4.0)],
            &context,
            &[Box::new(MinimumLabeledImpactFilter)],
        );

        assert_eq!(result.stages[0].impact_threshold, None);
        assert_eq!(result.anomalies.len(), 1);
        assert!(result.discarded.is_empty());
    }

    #[test]
    fn explicitly_shared_country_event_can_set_the_threshold() {
        let mut russia = label("RU", "2025-01-10");
        russia.additional_matching_countries = vec!["UA".to_owned()];
        let labels = vec![russia];
        let context = FilterContext {
            labels: &labels,
            country_code: "UA",
            proximity_days: 3,
        };
        let result = apply_anomaly_filters(
            vec![anomaly("2025-01-01", 5.0), anomaly("2025-01-10", 10.0)],
            &context,
            &[Box::new(MinimumLabeledImpactFilter)],
        );

        assert_eq!(result.stages[0].impact_threshold, Some(10.0));
        assert_eq!(result.anomalies.len(), 1);
    }
}
