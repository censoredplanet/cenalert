use rsomics_shapiro::swilk;

use crate::statistics::{mean, population_std};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChebyshevScore {
    pub score: f64,
    pub minimum_score: f64,
    pub threshold: f64,
    pub normal: bool,
}

pub fn chebyshev_score(values: &[f64], value: f64, z: f64, k: f64) -> ChebyshevScore {
    let mean = mean(values);
    let standard_deviation = population_std(values);
    let normal = is_normal(values);
    score_from_statistics(value, z, k, mean, standard_deviation, normal)
}

pub(crate) fn chebyshev_score_from_sorted(
    sorted_values: &[f64],
    shift: f64,
    shapiro_workspace: &mut Vec<f64>,
    value: f64,
    z: f64,
    k: f64,
    moments: (f64, f64),
) -> ChebyshevScore {
    let normal = is_normal_sorted(sorted_values, shift, shapiro_workspace);
    score_from_statistics(value, z, k, moments.0, moments.1, normal)
}

fn score_from_statistics(
    value: f64,
    z: f64,
    k: f64,
    mean: f64,
    standard_deviation: f64,
    normal: bool,
) -> ChebyshevScore {
    let minimum_score = if normal { z } else { k };
    ChebyshevScore {
        score: (value - mean) / standard_deviation,
        minimum_score,
        threshold: mean + minimum_score * standard_deviation,
        normal,
    }
}

fn is_normal(values: &[f64]) -> bool {
    if values.len() < 3 || values.iter().any(|value| value.is_nan()) {
        return false;
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let shift = values[values.len() / 2];
    for value in &mut sorted {
        *value -= shift;
    }
    swilk(&sorted).pw >= 0.05
}

fn is_normal_sorted(values: &[f64], shift: f64, workspace: &mut Vec<f64>) -> bool {
    if values.len() < 3 || values.iter().any(|value| value.is_nan()) {
        return false;
    }

    workspace.clear();
    workspace.extend(values.iter().map(|value| value - shift));
    swilk(workspace).pw >= 0.05
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_z_threshold_for_a_normal_window() {
        let normal_sample = [
            -1.59, -1.30, -1.05, -0.80, -0.61, -0.44, -0.29, -0.15, 0.0, 0.15, 0.29, 0.44, 0.61,
            0.80, 1.05, 1.30, 1.59,
        ];

        let result = chebyshev_score(&normal_sample, 3.0, 3.0, 6.0);

        assert!(result.normal);
        assert_eq!(result.minimum_score, 3.0);
    }

    #[test]
    fn uses_chebyshev_threshold_for_a_non_normal_window() {
        let skewed_sample = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0,
        ];

        let result = chebyshev_score(&skewed_sample, 40.0, 3.0, 6.0);

        assert!(!result.normal);
        assert_eq!(result.minimum_score, 6.0);
    }
}
