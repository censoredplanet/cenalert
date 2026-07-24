use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::io::Read;

use chrono::{DateTime, NaiveDate};
use serde::Deserialize;

use crate::model::{DetectorParameters, Observation};

#[derive(Debug)]
pub enum InputError {
    Io(std::io::Error),
    Json(serde_json::Error),
    EmptySeries,
    InvalidDate(String),
    DuplicateDate(NaiveDate),
    InvalidValue { date: String, value: f64 },
    InvalidParameters(String),
}

impl fmt::Display for InputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read input: {error}"),
            Self::Json(error) => write!(formatter, "invalid JSON: {error}"),
            Self::EmptySeries => write!(formatter, "time series is empty"),
            Self::InvalidDate(date) => write!(formatter, "invalid date: {date}"),
            Self::DuplicateDate(date) => {
                write!(formatter, "duplicate observation for date {date}")
            }
            Self::InvalidValue { date, value } => {
                write!(formatter, "invalid value {value} for date {date}")
            }
            Self::InvalidParameters(message) => {
                write!(formatter, "invalid detector parameters: {message}")
            }
        }
    }
}

impl Error for InputError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InputError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for InputError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Deserialize)]
struct DashboardRecord {
    #[serde(default)]
    country: Option<String>,
    #[serde(alias = "value")]
    rate: f64,
    date: String,
    #[serde(default)]
    topic: Option<String>,
}

/// Parses and normalizes a Dashboard-exported JSON array.
///
/// Records are sorted chronologically. Duplicate dates are rejected because
/// the detector expects exactly one observation per day.
pub fn parse_dashboard_json<R: Read>(reader: R) -> Result<Vec<Observation>, InputError> {
    let records: Vec<DashboardRecord> = serde_json::from_reader(reader)?;
    if records.is_empty() {
        return Err(InputError::EmptySeries);
    }

    let mut observations = Vec::with_capacity(records.len());
    for record in records {
        if !record.rate.is_finite() || record.rate < 0.0 {
            return Err(InputError::InvalidValue {
                date: record.date,
                value: record.rate,
            });
        }

        let date = parse_date(&record.date)?;
        observations.push(Observation {
            date,
            value: record.rate,
            country: normalize_optional_text(record.country),
            topic: normalize_optional_text(record.topic),
        });
    }

    observations.sort_unstable_by_key(|observation| observation.date);
    reject_duplicate_dates(&observations)?;
    Ok(observations)
}

/// Parses the five-element parameter arrays stored by `cenalert-paper`.
///
/// Array order: `[window, z, k, min_residual, efficiency]`.
pub fn parse_parameters_json<R: Read>(reader: R) -> Result<DetectorParameters, InputError> {
    let values: Vec<f64> = serde_json::from_reader(reader)?;
    if values.len() != 5 {
        return Err(InputError::InvalidParameters(format!(
            "expected 5 values, found {}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InputError::InvalidParameters(
            "all values must be finite".to_owned(),
        ));
    }

    // Match Python's `round()` used by `run_cenalert.py`, including its
    // round-half-to-even behavior.
    let rounded_window = values[0].round_ties_even();
    if rounded_window < 1.0 || rounded_window > usize::MAX as f64 {
        return Err(InputError::InvalidParameters(
            "window must round to a positive integer".to_owned(),
        ));
    }
    if values[1] <= 0.0 || values[2] <= 0.0 {
        return Err(InputError::InvalidParameters(
            "z and k must be positive".to_owned(),
        ));
    }
    if values[3] <= 0.0 {
        return Err(InputError::InvalidParameters(
            "min_residual must be positive".to_owned(),
        ));
    }
    if !(0.0..1.0).contains(&values[4]) {
        return Err(InputError::InvalidParameters(
            "efficiency must be in [0, 1)".to_owned(),
        ));
    }

    Ok(DetectorParameters {
        window: rounded_window as usize,
        z: values[1],
        k: values[2],
        min_residual: values[3],
        efficiency: values[4],
    })
}

fn parse_date(value: &str) -> Result<NaiveDate, InputError> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .or_else(|_| DateTime::parse_from_rfc3339(value).map(|date_time| date_time.date_naive()))
        .map_err(|_| InputError::InvalidDate(value.to_owned()))
}

fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|text| {
        let normalized = text.trim();
        if normalized.is_empty() {
            None
        } else {
            Some(normalized.to_owned())
        }
    })
}

fn reject_duplicate_dates(observations: &[Observation]) -> Result<(), InputError> {
    let mut dates = HashSet::with_capacity(observations.len());
    for observation in observations {
        if !dates.insert(observation.date) {
            return Err(InputError::DuplicateDate(observation.date));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases_iso_dates_and_sorts_records() {
        let input = br#"[
            {
                "country": " RU ",
                "rate": 2.5,
                "date": "2025-01-02T00:00:00Z",
                "topic": "vpn"
            },
            {
                "country": "RU",
                "value": 0,
                "date": "2025-01-01",
                "topic": "vpn"
            }
        ]"#;

        let observations = parse_dashboard_json(input.as_slice()).unwrap();

        assert_eq!(observations.len(), 2);
        assert_eq!(
            observations[0].date,
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()
        );
        assert_eq!(observations[0].value, 0.0);
        assert_eq!(observations[1].value, 2.5);
        assert_eq!(observations[1].country.as_deref(), Some("RU"));
    }

    #[test]
    fn rejects_duplicate_dates_after_sorting() {
        let input = br#"[
            {"rate": 1, "date": "2025-01-01"},
            {"rate": 2, "date": "2025-01-01T12:00:00Z"}
        ]"#;

        let error = parse_dashboard_json(input.as_slice()).unwrap_err();
        assert!(matches!(error, InputError::DuplicateDate(_)));
    }

    #[test]
    fn rejects_negative_values() {
        let input = br#"[{"rate": -1, "date": "2025-01-01"}]"#;

        let error = parse_dashboard_json(input.as_slice()).unwrap_err();
        assert!(matches!(error, InputError::InvalidValue { .. }));
    }

    #[test]
    fn parses_paper_parameter_array() {
        let input =
            br#"[89.64249009209439, 4.371769582031636, 6.053424084413223, 74.88743438197304, 0.05135053360365411]"#;

        let parameters = parse_parameters_json(input.as_slice()).unwrap();

        assert_eq!(parameters.window, 90);
        assert_eq!(parameters.z, 4.371769582031636);
        assert_eq!(parameters.k, 6.053424084413223);
        assert_eq!(parameters.min_residual, 74.88743438197304);
        assert_eq!(parameters.efficiency, 0.05135053360365411);
    }

    #[test]
    fn rounds_window_ties_like_python() {
        let even = parse_parameters_json(br#"[60.5, 3, 6, 1, 0.05]"#.as_slice()).unwrap();
        let odd = parse_parameters_json(br#"[61.5, 3, 6, 1, 0.05]"#.as_slice()).unwrap();

        assert_eq!(even.window, 60);
        assert_eq!(odd.window, 62);
    }

    #[test]
    fn rejects_parameter_arrays_with_wrong_length() {
        let input = br#"[60, 3, 6]"#;

        let error = parse_parameters_json(input.as_slice()).unwrap_err();
        assert!(matches!(error, InputError::InvalidParameters(_)));
    }
}
