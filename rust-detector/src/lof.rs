use crate::detector::{run_detector, DenseEvaluation, DenseScorer, Detector};
use crate::model::{DetectionResult, Observation, OutlierDetectorParameters};
use crate::window::SlidingWindow;

pub struct LocalOutlierFactorDetector {
    parameters: OutlierDetectorParameters,
}

impl LocalOutlierFactorDetector {
    pub fn new(parameters: OutlierDetectorParameters) -> Self {
        Self { parameters }
    }
}

impl Detector for LocalOutlierFactorDetector {
    fn detect(&self, observations: &[Observation]) -> DetectionResult {
        run_detector(
            observations,
            self.parameters.window,
            self.parameters.min_residual_ratio,
            self.parameters.efficiency,
            LocalOutlierFactorScorer {
                window: self.parameters.window,
                minimum_score: self.parameters.min_score,
            },
        )
    }
}

pub fn local_outlier_factor_score(window: &[f64], value: f64, capacity: usize) -> f64 {
    if window.is_empty() {
        return f64::NAN;
    }
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    if value <= mean {
        return 1.0;
    }

    let mut samples = Vec::with_capacity(window.len() + 1);
    samples.extend_from_slice(window);
    samples.push(value);
    let neighbor_count = capacity.saturating_sub(1).min(samples.len() - 1);
    if neighbor_count == 0 {
        return f64::NAN;
    }

    let neighbors: Vec<Vec<usize>> = (0..samples.len())
        .map(|index| nearest_neighbors(&samples, index, neighbor_count))
        .collect();
    let k_distances: Vec<f64> = neighbors
        .iter()
        .enumerate()
        .map(|(index, nearest)| {
            nearest
                .iter()
                .map(|neighbor| (samples[index] - samples[*neighbor]).abs())
                .fold(0.0, f64::max)
        })
        .collect();
    let local_reachability: Vec<f64> = neighbors
        .iter()
        .enumerate()
        .map(|(index, nearest)| {
            let mean_reachability = nearest
                .iter()
                .map(|neighbor| {
                    (samples[index] - samples[*neighbor])
                        .abs()
                        .max(k_distances[*neighbor])
                })
                .sum::<f64>()
                / neighbor_count as f64;
            1.0 / (mean_reachability + 1e-10)
        })
        .collect();

    let target = samples.len() - 1;
    neighbors[target]
        .iter()
        .map(|neighbor| local_reachability[*neighbor])
        .sum::<f64>()
        / neighbor_count as f64
        / local_reachability[target]
}

fn nearest_neighbors(samples: &[f64], index: usize, count: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .filter(|(candidate, _)| *candidate != index)
        .map(|(candidate, value)| (candidate, (samples[index] - value).abs()))
        .collect();
    distances.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    distances
        .into_iter()
        .take(count)
        .map(|(neighbor, _)| neighbor)
        .collect()
}

pub(crate) struct LocalOutlierFactorScorer {
    pub window: usize,
    pub minimum_score: f64,
}

impl DenseScorer for LocalOutlierFactorScorer {
    fn initial_minimum_score(&self) -> f64 {
        self.minimum_score
    }

    fn evaluate(&mut self, window: &mut SlidingWindow, value: f64) -> DenseEvaluation {
        let values = window.values();
        let score = local_outlier_factor_score(&values, value, self.window);
        DenseEvaluation {
            score,
            minimum_score: self.minimum_score,
        }
    }

    fn threshold(&mut self, window: &mut SlidingWindow, value: f64) -> f64 {
        let values = window.values();
        bounded_threshold(
            window.mean(),
            value,
            |candidate| local_outlier_factor_score(&values, candidate, self.window),
            self.minimum_score,
        )
    }
}

pub(crate) fn bounded_threshold(
    lower: f64,
    upper: f64,
    score: impl Fn(f64) -> f64,
    target_score: f64,
) -> f64 {
    if !lower.is_finite() || !upper.is_finite() || upper <= lower {
        return lower;
    }
    let objective = |value: f64| (score(value) - target_score).abs();
    let sqrt_epsilon = (2.2e-16_f64).sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let mut left = lower;
    let mut right = upper;
    let mut best = left + golden_mean * (right - left);
    let mut previous = best;
    let mut previous_previous = best;
    let mut best_value = objective(best);
    let mut previous_value = best_value;
    let mut previous_previous_value = best_value;
    let mut step = 0.0_f64;
    let mut previous_step = 0.0_f64;
    let mut midpoint = 0.5 * (left + right);
    let mut tolerance = sqrt_epsilon * best.abs() + 1e-5 / 3.0;
    let mut double_tolerance = 2.0 * tolerance;
    let mut evaluations = 1;

    while (best - midpoint).abs() > double_tolerance - 0.5 * (right - left) && evaluations < 500 {
        let mut use_golden = true;
        if previous_step.abs() > tolerance {
            let first = (best - previous) * (best_value - previous_previous_value);
            let second = (best - previous_previous) * (best_value - previous_value);
            let mut numerator = (best - previous_previous) * second - (best - previous) * first;
            let mut denominator = 2.0 * (second - first);
            if denominator > 0.0 {
                numerator = -numerator;
            }
            denominator = denominator.abs();
            let saved_step = previous_step;
            previous_step = step;
            if numerator.abs() < (0.5 * denominator * saved_step).abs()
                && numerator > denominator * (left - best)
                && numerator < denominator * (right - best)
            {
                step = numerator / denominator;
                let candidate = best + step;
                if candidate - left < double_tolerance || right - candidate < double_tolerance {
                    step = tolerance * if midpoint >= best { 1.0 } else { -1.0 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            previous_step = if best >= midpoint {
                left - best
            } else {
                right - best
            };
            step = golden_mean * previous_step;
        }

        let candidate = best + (if step >= 0.0 { 1.0 } else { -1.0 }) * step.abs().max(tolerance);
        let candidate_value = objective(candidate);
        evaluations += 1;
        if candidate_value <= best_value {
            if candidate >= best {
                left = best;
            } else {
                right = best;
            }
            previous_previous = previous;
            previous_previous_value = previous_value;
            previous = best;
            previous_value = best_value;
            best = candidate;
            best_value = candidate_value;
        } else {
            if candidate < best {
                left = candidate;
            } else {
                right = candidate;
            }
            if candidate_value <= previous_value || previous == best {
                previous_previous = previous;
                previous_previous_value = previous_value;
                previous = candidate;
                previous_value = candidate_value;
            } else if candidate_value <= previous_previous_value
                || previous_previous == best
                || previous_previous == previous
            {
                previous_previous = candidate;
                previous_previous_value = candidate_value;
            }
        }
        midpoint = 0.5 * (left + right);
        tolerance = sqrt_epsilon * best.abs() + 1e-5 / 3.0;
        double_tolerance = 2.0 * tolerance;
    }
    let threshold = best;
    if approximately_equal(threshold, upper) {
        lower
    } else {
        threshold
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-8 + 1e-5 * right.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_one_to_values_not_above_the_window_mean() {
        assert_eq!(local_outlier_factor_score(&[1.0, 2.0, 3.0], 2.0, 3), 1.0);
    }

    #[test]
    fn assigns_a_larger_score_to_an_isolated_high_value() {
        let ordinary = local_outlier_factor_score(&[8.0, 9.0, 10.0, 11.0], 12.0, 4);
        let isolated = local_outlier_factor_score(&[8.0, 9.0, 10.0, 11.0], 50.0, 4);

        assert!(isolated > ordinary);
        assert!(isolated > 1.0);
    }

    #[test]
    fn matches_sklearn_reference_scores() {
        let cases = [
            (&[8.0, 9.0, 10.0, 11.0][..], 50.0, 4, 15.714285713705102),
            (&[1.0, 1.0, 1.0, 2.0][..], 10.0, 4, 8.6666666659),
            (
                &[2.0, 4.0, 8.0, 16.0, 32.0][..],
                40.0,
                5,
                1.1614289072265938,
            ),
        ];

        for (window, value, capacity, expected) in cases {
            let actual = local_outlier_factor_score(window, value, capacity);
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    }

    #[test]
    fn bounded_threshold_finds_a_monotonic_target() {
        let threshold = bounded_threshold(0.0, 10.0, |value| value, 4.0);

        assert!((threshold - 4.0).abs() < 1e-4);
    }
}
