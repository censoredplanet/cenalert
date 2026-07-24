const ALPHA: f64 = 0.1;
const SBA_DEBIASING_FACTOR: f64 = 0.95;

/// One-step Croston SBA forecast matching StatsForecast's fixed-alpha model.
pub fn croston_sba_forecast(series: &[f64]) -> f64 {
    let demands: Vec<f64> = series
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .collect();
    if demands.is_empty() {
        return series.last().copied().unwrap_or(f64::NAN);
    }

    let intervals = demand_intervals(series);
    let demand_forecast = simple_exponential_forecast(&demands, ALPHA);
    let interval_forecast = simple_exponential_forecast(&intervals, ALPHA);
    let classic = if interval_forecast == 0.0 {
        demand_forecast
    } else {
        demand_forecast / interval_forecast
    };
    SBA_DEBIASING_FACTOR * classic
}

fn demand_intervals(series: &[f64]) -> Vec<f64> {
    let mut previous_position = 0;
    let mut intervals = Vec::new();
    for (index, value) in series.iter().enumerate() {
        if *value != 0.0 {
            let position = index + 1;
            intervals.push((position - previous_position) as f64);
            previous_position = position;
        }
    }
    intervals
}

fn simple_exponential_forecast(values: &[f64], alpha: f64) -> f64 {
    let mut forecast = values[0];
    for value in &values[1..] {
        forecast = alpha * value + (1.0 - alpha) * forecast;
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
}
