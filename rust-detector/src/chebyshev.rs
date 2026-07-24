use rsomics_shapiro::shapiro;

use crate::statistics::{mean, population_std};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChebyshevScore {
    pub score: f64,
    pub minimum_score: f64,
    pub threshold: f64,
    pub normal: bool,
}

pub fn chebyshev_score(values: &[f64], value: f64, z: f64, k: f64) -> ChebyshevScore {
    let normal = is_normal(values);
    let minimum_score = if normal { z } else { k };
    let mean = mean(values);
    let standard_deviation = population_std(values);

    ChebyshevScore {
        score: (value - mean) / standard_deviation,
        minimum_score,
        threshold: mean + minimum_score * standard_deviation,
        normal,
    }
}

fn is_normal(values: &[f64]) -> bool {
    shapiro(values).is_ok_and(|result| result.pvalue >= 0.05)
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
