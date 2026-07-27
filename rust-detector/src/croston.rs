const ALPHA: f64 = 0.1;
const SBA_DEBIASING_FACTOR: f64 = 0.95;

/// One-step Croston SBA forecast matching StatsForecast's fixed-alpha model.
pub fn croston_sba_forecast(series: &[f64]) -> f64 {
    let mut previous_position = 0;
    let arrivals = series.iter().enumerate().filter_map(|(index, value)| {
        if *value > 0.0 {
            let position = index + 1;
            let interval = position - previous_position;
            previous_position = position;
            Some((*value, interval))
        } else {
            None
        }
    });
    let forecast = croston_sba_forecast_from_pairs(arrivals);
    if forecast.is_nan() {
        return series.last().copied().unwrap_or(f64::NAN);
    }
    forecast
}

pub(crate) fn croston_sba_forecast_from_arrivals(
    demands: impl Iterator<Item = f64>,
    intervals: impl Iterator<Item = usize>,
) -> f64 {
    croston_sba_forecast_from_pairs(demands.zip(intervals))
}

fn croston_sba_forecast_from_pairs(mut arrivals: impl Iterator<Item = (f64, usize)>) -> f64 {
    let Some((mut demand_forecast, first_interval)) = arrivals.next() else {
        return f64::NAN;
    };
    let mut interval_forecast = first_interval as f64;
    for (demand, interval) in arrivals {
        demand_forecast = ALPHA * demand + (1.0 - ALPHA) * demand_forecast;
        interval_forecast = ALPHA * interval as f64 + (1.0 - ALPHA) * interval_forecast;
    }
    let classic = if interval_forecast == 0.0 {
        demand_forecast
    } else {
        demand_forecast / interval_forecast
    };
    SBA_DEBIASING_FACTOR * classic
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
