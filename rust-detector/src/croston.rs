const ALPHA: f64 = 0.1;
const SBA_DEBIASING_FACTOR: f64 = 0.95;

/// One-step Croston SBA forecast matching StatsForecast's fixed-alpha model.
pub fn croston_sba_forecast(series: &[f64]) -> f64 {
    let demand_forecast =
        simple_exponential_forecast(series.iter().copied().filter(|value| *value > 0.0));
    if demand_forecast.is_nan() {
        return series.last().copied().unwrap_or(f64::NAN);
    }
    let mut previous_position = 0;
    let intervals = series
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .map(|(index, _)| {
            // StatsForecast extracts demand with `value > 0`, but extracts
            // intervals with `value != 0`. This distinction matters for NaNs
            // introduced by the Python detector's sparse-window rescaling.
            let position = index + 1;
            let interval = position - previous_position;
            previous_position = position;
            interval as f64
        });
    let interval_forecast = simple_exponential_forecast(intervals);
    let classic = if interval_forecast == 0.0 {
        demand_forecast
    } else {
        demand_forecast / interval_forecast
    };
    SBA_DEBIASING_FACTOR * classic
}

fn simple_exponential_forecast(mut values: impl Iterator<Item = f64>) -> f64 {
    let Some(mut forecast) = values.next() else {
        return f64::NAN;
    };
    for value in values {
        forecast = ALPHA * value + (1.0 - ALPHA) * forecast;
    }
    forecast
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecasts_regular_demand_with_sba_debiasing() {
        let forecast = croston_sba_forecast(&[10.0, 10.0, 10.0]);

        assert!((forecast - 9.5).abs() < 1e-12);
    }

    #[test]
    fn forecasts_intermittent_demand() {
        let forecast = croston_sba_forecast(&[0.0, 10.0, 0.0, 0.0, 20.0]);
        let expected_demand = 11.0;
        let expected_interval = 2.1;

        assert!((forecast - 0.95 * expected_demand / expected_interval).abs() < 1e-12);
    }

    #[test]
    fn follows_statsforecast_naive_fallback_for_no_demand() {
        assert_eq!(croston_sba_forecast(&[0.0, 0.0]), 0.0);
        assert!(croston_sba_forecast(&[]).is_nan());
    }

    #[test]
    fn follows_statsforecast_nan_interval_semantics() {
        assert!((croston_sba_forecast(&[f64::NAN, 10.0]) - 9.5).abs() < 1e-12);
    }
}
