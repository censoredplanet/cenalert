pub(crate) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation, matching NumPy's `std` default (`ddof=0`).
pub(crate) fn population_std(values: &[f64]) -> f64 {
    let mean = mean(values);
    if mean.is_nan() {
        return f64::NAN;
    }
    let variance = values
        .iter()
        .map(|value| {
            let difference = value - mean;
            difference * difference
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

pub fn efficiency_ratio(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return f64::NAN;
    }

    let net_change = values[values.len() - 1] - values[0];
    let total_change: f64 = values
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .sum();
    net_change / total_change
}

/// Replaces the largest five percent of values using SciPy winsorize's
/// default truncation rule (`inclusive=true`).
pub(crate) fn winsorize_upper_five_percent(values: &mut [f64]) {
    let replacement_count = (values.len() as f64 * 0.05).floor() as usize;
    if replacement_count == 0 || replacement_count >= values.len() {
        return;
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let upper_limit = sorted[values.len() - replacement_count - 1];
    for value in values {
        if *value > upper_limit {
            *value = upper_limit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_numpy_style_population_statistics() {
        let values = [1.0, 2.0, 3.0, 4.0];

        assert_eq!(mean(&values), 2.5);
        assert!((population_std(&values) - 1.118_033_988_749_895).abs() < 1e-12);
    }

    #[test]
    fn computes_signed_efficiency_ratio() {
        assert_eq!(efficiency_ratio(&[2.0, 4.0, 3.0, 5.0]), 0.6);
        assert_eq!(efficiency_ratio(&[5.0, 3.0, 4.0, 2.0]), -0.6);
        assert!(efficiency_ratio(&[1.0]).is_nan());
    }

    #[test]
    fn winsorizes_same_number_of_upper_values_as_scipy() {
        let mut values: Vec<f64> = (1..=20).map(f64::from).collect();

        winsorize_upper_five_percent(&mut values);

        assert_eq!(values[18], 19.0);
        assert_eq!(values[19], 19.0);
    }
}
