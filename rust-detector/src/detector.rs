use crate::chebyshev::chebyshev_score_from_sorted;
use crate::model::{
    AnnotatedObservation, Anomaly, DetectionResult, DetectorParameters, Observation,
};
use crate::statistics::{efficiency_ratio, mean};
use crate::window::{DemandCategory, SlidingWindow};

/// Common interface for concrete and ensemble anomaly detectors.
pub trait Detector {
    fn detect(&self, observations: &[Observation]) -> DetectionResult;
}

pub(crate) struct DenseEvaluation {
    pub score: f64,
    pub minimum_score: f64,
}

pub(crate) trait DenseScorer {
    fn initial_minimum_score(&self) -> f64;
    fn evaluate(&mut self, window: &mut SlidingWindow, value: f64) -> DenseEvaluation;
    fn threshold(&mut self, window: &mut SlidingWindow, value: f64) -> f64;
    fn idle_threshold(&mut self, _window: &mut SlidingWindow) -> Option<f64> {
        None
    }
}

pub struct ChebyshevDetector {
    parameters: DetectorParameters,
}

impl ChebyshevDetector {
    pub fn new(parameters: DetectorParameters) -> Self {
        Self { parameters }
    }
}

impl Detector for ChebyshevDetector {
    fn detect(&self, observations: &[Observation]) -> DetectionResult {
        run_detector(
            observations,
            self.parameters.window,
            self.parameters.min_residual,
            self.parameters.efficiency,
            ChebyshevScorer::new(self.parameters),
        )
    }
}

pub(crate) struct ChebyshevScorer {
    z: f64,
    k: f64,
    minimum_score: f64,
    threshold: f64,
    shapiro_workspace: Vec<f64>,
}

impl ChebyshevScorer {
    pub(crate) fn new(parameters: DetectorParameters) -> Self {
        Self {
            z: parameters.z,
            k: parameters.k,
            minimum_score: f64::INFINITY,
            threshold: f64::NAN,
            shapiro_workspace: Vec::with_capacity(parameters.window),
        }
    }
}

impl DenseScorer for ChebyshevScorer {
    fn initial_minimum_score(&self) -> f64 {
        f64::INFINITY
    }

    fn evaluate(&mut self, window: &mut SlidingWindow, value: f64) -> DenseEvaluation {
        let moments = (window.mean(), window.std());
        let result = chebyshev_score_from_sorted(
            window.sorted_values(),
            window.shapiro_shift(),
            &mut self.shapiro_workspace,
            value,
            self.z,
            self.k,
            moments,
        );
        self.minimum_score = result.minimum_score;
        self.threshold = result.threshold;
        DenseEvaluation {
            score: result.score,
            minimum_score: result.minimum_score,
        }
    }

    fn threshold(&mut self, _window: &mut SlidingWindow, _value: f64) -> f64 {
        self.threshold
    }

    fn idle_threshold(&mut self, window: &mut SlidingWindow) -> Option<f64> {
        Some(window.mean() + self.minimum_score * window.std())
    }
}

pub(crate) fn run_detector<S: DenseScorer>(
    observations: &[Observation],
    window: usize,
    min_residual: f64,
    efficiency: f64,
    scorer: S,
) -> DetectionResult {
    let mut state = DetectorState::new(window, min_residual, efficiency, scorer);
    let mut annotated: Vec<AnnotatedObservation> =
        observations.iter().map(AnnotatedObservation::new).collect();

    for (index, observation) in observations.iter().enumerate() {
        state.process(index, observation.value, &mut annotated);
    }

    let scale_factor = normalize(&mut annotated);
    let anomalies = collect_anomalies(&annotated);
    DetectionResult {
        annotated,
        anomalies,
        scale_factor,
    }
}

struct DetectorState<S> {
    min_residual: f64,
    efficiency: f64,
    scorer: S,
    window: SlidingWindow,
    active_anomaly: bool,
    intermittent_anomaly: bool,
    efficiency_values: Vec<f64>,
    croston_forecast: Option<f64>,
    minimum_score: f64,
}

impl<S: DenseScorer> DetectorState<S> {
    fn new(window: usize, min_residual: f64, efficiency: f64, scorer: S) -> Self {
        let minimum_score = scorer.initial_minimum_score();
        Self {
            min_residual,
            efficiency,
            scorer,
            window: SlidingWindow::new(window),
            active_anomaly: false,
            intermittent_anomaly: false,
            efficiency_values: Vec::new(),
            croston_forecast: None,
            minimum_score,
        }
    }

    fn process(&mut self, index: usize, value: f64, annotated: &mut [AnnotatedObservation]) {
        if index < self.window.capacity() {
            if value > 0.0 {
                self.window.insert(value, index + 1);
            }
            return;
        }

        if !self.active_anomaly
            && self
                .window
                .last_arrival()
                .is_some_and(|last_arrival| index - last_arrival >= self.window.capacity())
        {
            self.window.clear();
        }

        let previous_category = annotated[index - 1].demand_category;
        let category = if self.active_anomaly {
            previous_category.unwrap_or(DemandCategory::None)
        } else {
            self.window.classify_demand(index)
        };

        let sparse = is_sparse(category);
        let mut score = if self.active_anomaly {
            f64::INFINITY
        } else {
            0.0
        };
        let mut residual = score;
        let mut forecast = 0.0;
        if value > 0.0 && sparse {
            if !self.active_anomaly {
                self.croston_forecast = Some(self.window.croston_sba_forecast());
            }
            forecast = if self.window.is_empty() {
                0.0
            } else {
                self.croston_forecast.unwrap_or(0.0)
            };
            residual = value - forecast;
        } else if value > 0.0 {
            if previous_category.is_some_and(is_sparse) {
                self.window.winsorize_upper_five_percent();
            }
            let result = self.scorer.evaluate(&mut self.window, value);
            score = result.score;
            self.minimum_score = result.minimum_score;
        }

        let new_anomaly =
            !self.active_anomaly && (score >= self.minimum_score || residual >= self.min_residual);
        let return_to_normal = self.active_anomaly
            && (score < self.minimum_score || residual < self.min_residual || value.abs() <= 1e-8);
        let new_normal =
            self.active_anomaly && efficiency_ratio(&self.efficiency_values) < self.efficiency;

        self.active_anomaly =
            (new_anomaly || self.active_anomaly) && !(return_to_normal || new_normal);

        let mut threshold = None;
        if new_anomaly {
            self.intermittent_anomaly = sparse;
            threshold = Some(if self.intermittent_anomaly {
                forecast + self.min_residual
            } else {
                self.scorer.threshold(&mut self.window, value)
            });
            self.efficiency_values.push(annotated[index - 1].value);
        }
        if self.active_anomaly {
            self.efficiency_values.push(value);
        }

        if new_normal || return_to_normal {
            self.finish_anomaly(index, value, return_to_normal, new_normal);
        }

        let previous_threshold = annotated[index - 1].threshold;
        let stored_threshold = if new_anomaly {
            threshold
        } else if self.active_anomaly {
            previous_threshold
        } else {
            self.scorer.idle_threshold(&mut self.window)
        };
        let retain_demand_statistics = self.active_anomaly && !new_anomaly;
        let coefficient_of_variation_squared = if retain_demand_statistics {
            annotated[index - 1].coefficient_of_variation_squared
        } else {
            Some(self.window.coefficient_of_variation_squared())
        };
        let average_interdemand_interval = if retain_demand_statistics {
            annotated[index - 1].average_interdemand_interval
        } else {
            Some(self.window.average_interdemand_interval(index))
        };

        annotated[index].anomaly = self.active_anomaly;
        annotated[index].score = finite_or_infinite(score);
        annotated[index].residual = finite_or_infinite(residual);
        annotated[index].threshold = finite_or_infinite_option(stored_threshold);
        annotated[index].minimum_score = finite_or_infinite(self.minimum_score);
        annotated[index].coefficient_of_variation_squared =
            finite_or_infinite_option(coefficient_of_variation_squared);
        annotated[index].average_interdemand_interval =
            finite_or_infinite_option(average_interdemand_interval);
        annotated[index].demand_category = Some(category);

        if !self.active_anomaly && value > 0.0 {
            self.window.insert(value, index + 1);
        }
    }

    fn finish_anomaly(
        &mut self,
        index: usize,
        value: f64,
        return_to_normal: bool,
        new_normal: bool,
    ) {
        let anomaly = self.efficiency_values[1..].to_vec();
        let target_mean = if return_to_normal {
            self.window.mean()
        } else {
            value
        };
        let target_std = self.window.std();

        if self.intermittent_anomaly && new_normal {
            self.window.clear();
            self.window.set_last_arrival(index - anomaly.len());
        }

        let first_timestamp = index - anomaly.len() + 1;
        for (offset, point) in anomaly.iter().enumerate() {
            self.window.insert(*point, first_timestamp + offset);
        }

        if !self.intermittent_anomaly {
            self.window.recenter(target_mean, target_std);
        } else if new_normal {
            let anomaly_mean = mean(&anomaly);
            self.window.scale(target_mean / anomaly_mean);
        }

        self.intermittent_anomaly = false;
        self.efficiency_values.clear();
    }
}

fn is_sparse(category: DemandCategory) -> bool {
    matches!(
        category,
        DemandCategory::None | DemandCategory::Lumpy | DemandCategory::Intermittent
    )
}

fn finite_or_infinite(value: f64) -> Option<f64> {
    (!value.is_nan()).then_some(value)
}

fn finite_or_infinite_option(value: Option<f64>) -> Option<f64> {
    value.and_then(finite_or_infinite)
}

fn normalize(annotated: &mut [AnnotatedObservation]) -> f64 {
    let maximum = annotated
        .iter()
        .map(|point| point.value)
        .fold(0.0, f64::max);
    let scale_factor = if maximum > 100.0 {
        maximum / 100.0
    } else {
        1.0
    };

    if scale_factor > 1.0 {
        for point in annotated {
            point.value /= scale_factor;
            point.residual = point.residual.map(|value| value / scale_factor);
            point.threshold = point.threshold.map(|value| value / scale_factor);
        }
    }
    scale_factor
}

fn collect_anomalies(annotated: &[AnnotatedObservation]) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();
    let mut index = 0;
    while index < annotated.len() {
        if !annotated[index].anomaly {
            index += 1;
            continue;
        }

        let start_index = index;
        while index + 1 < annotated.len() && annotated[index + 1].anomaly {
            index += 1;
        }
        let end_index = index;
        let points = &annotated[start_index..=end_index];
        let peak = points
            .iter()
            .reduce(|peak, point| {
                if point.value > peak.value {
                    point
                } else {
                    peak
                }
            })
            .expect("anomaly group cannot be empty");
        let impact = points
            .iter()
            .map(|point| point.value - point.threshold.unwrap_or(f64::NAN))
            .sum();

        anomalies.push(Anomaly {
            start: annotated[start_index].date,
            end: annotated[end_index].date,
            peak: peak.date,
            score: annotated[start_index].score.unwrap_or(f64::NAN),
            residual: annotated[start_index].residual.unwrap_or(f64::NAN),
            impact,
        });
        index += 1;
    }
    anomalies
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;

    fn observations(values: &[f64]) -> Vec<Observation> {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| Observation {
                date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()
                    + chrono::Duration::days(index as i64),
                value: *value,
                country: Some("XX".to_owned()),
                topic: Some("vpn".to_owned()),
            })
            .collect()
    }

    fn parameters(window: usize) -> DetectorParameters {
        DetectorParameters {
            window,
            z: 2.0,
            k: 3.0,
            min_residual: 5.0,
            efficiency: 0.05,
        }
    }

    #[test]
    fn detects_and_closes_dense_anomaly() {
        let values = [
            8.0, 9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 50.0, 45.0, 10.0,
        ];

        let mut detector_parameters = parameters(5);
        // Avoid the sparse residual path during the synthetic first
        // interarrival; this test isolates the dense score path.
        detector_parameters.min_residual = 20.0;
        let result = ChebyshevDetector::new(detector_parameters).detect(&observations(&values));

        assert_eq!(result.anomalies.len(), 1);
        assert_eq!(
            result.anomalies[0].start,
            NaiveDate::from_ymd_opt(2025, 1, 9).unwrap()
        );
        assert_eq!(
            result.anomalies[0].end,
            NaiveDate::from_ymd_opt(2025, 1, 10).unwrap()
        );
        assert_eq!(
            result.anomalies[0].peak,
            NaiveDate::from_ymd_opt(2025, 1, 9).unwrap()
        );
    }

    #[test]
    fn detects_sparse_residual_anomaly() {
        let values = [0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 25.0, 0.0, 5.0];

        let result = ChebyshevDetector::new(parameters(4)).detect(&observations(&values));

        assert_eq!(result.anomalies.len(), 1);
        assert_eq!(
            result.anomalies[0].start,
            NaiveDate::from_ymd_opt(2025, 1, 9).unwrap()
        );
        assert_eq!(
            result.anomalies[0].end,
            NaiveDate::from_ymd_opt(2025, 1, 9).unwrap()
        );
        assert_eq!(result.anomalies[0].score, 0.0);
        assert!(result.anomalies[0].residual > 5.0);
    }

    #[test]
    fn rebuilds_sparse_window_when_both_end_conditions_hold() {
        let detector_parameters = parameters(4);
        let mut state = DetectorState::new(
            detector_parameters.window,
            detector_parameters.min_residual,
            detector_parameters.efficiency,
            ChebyshevScorer::new(detector_parameters),
        );
        state.window.insert(1.0, 1);
        state.window.insert(2.0, 2);
        state.intermittent_anomaly = true;
        state.efficiency_values = vec![0.0, 10.0, 20.0];

        state.finish_anomaly(10, 0.0, true, true);

        assert_eq!(state.window.len(), 2);
        assert_eq!(state.window.last_arrival(), Some(10));
    }

    #[test]
    fn normalizes_google_trends_values_above_one_hundred() {
        let values = [100.0, 120.0, 80.0, 100.0];

        let result = ChebyshevDetector::new(parameters(3)).detect(&observations(&values));

        assert_eq!(result.scale_factor, 1.2);
        assert_eq!(result.annotated[1].value, 100.0);
    }

    #[test]
    fn does_not_report_warmup_points() {
        let values = [100.0, 100.0, 100.0, 10.0, 10.0];

        let result = ChebyshevDetector::new(parameters(3)).detect(&observations(&values));

        assert!(result.annotated[..3].iter().all(|point| !point.anomaly));
    }

    #[test]
    fn supports_dynamic_dispatch_through_detector_trait() {
        let detector: Box<dyn Detector> = Box::new(ChebyshevDetector::new(parameters(3)));

        let result = detector.detect(&observations(&[1.0, 2.0, 3.0, 4.0]));

        assert_eq!(result.annotated.len(), 4);
    }
}
