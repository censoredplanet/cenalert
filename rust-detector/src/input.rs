use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::io::Read;

use chrono::{DateTime, NaiveDate};
use serde::Deserialize;

use crate::model::{
    DetectorParameters, EnsembleParameters, Observation, OutlierDetectorParameters,
};

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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DetectorParameterInput {
    Legacy(Vec<f64>),
    Ratio(DetectorParameters),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OutlierParameterInput {
    Legacy(Vec<f64>),
    Ratio(OutlierDetectorParameters),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyEnsembleParameters {
    window: usize,
    min_residual: f64,
    efficiency: f64,
    chebyshev_z: f64,
    chebyshev_k: f64,
    isolation_forest_min_score: f64,
    lof_min_score: f64,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EnsembleParameterInput {
    Ratio(EnsembleParameters),
    Legacy(LegacyEnsembleParameters),
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

/// Parses current ratio parameter objects or legacy arrays from `cenalert-paper`.
///
/// Legacy array order is `[window, z, k, min_residual, efficiency]`; its
/// residual in the normalized `[0, 100]` scale is divided by 100.
pub fn parse_parameters_json<R: Read>(reader: R) -> Result<DetectorParameters, InputError> {
    let input: DetectorParameterInput = serde_json::from_reader(reader)?;
    let parameters = match input {
        DetectorParameterInput::Ratio(parameters) => parameters,
        DetectorParameterInput::Legacy(values) => {
            require_length(&values, 5)?;
            require_finite(&values)?;
            DetectorParameters {
                window: rounded_window(values[0], 1)?,
                z: values[1],
                k: values[2],
                min_residual_ratio: values[3] / 100.0,
                efficiency: values[4],
            }
        }
    };
    validate_detector_parameters(parameters)
}

/// Parses current ratio parameter objects or legacy paper-compatible arrays.
pub fn parse_outlier_parameters_json<R: Read>(
    reader: R,
) -> Result<OutlierDetectorParameters, InputError> {
    let input: OutlierParameterInput = serde_json::from_reader(reader)?;
    let parameters = match input {
        OutlierParameterInput::Ratio(parameters) => parameters,
        OutlierParameterInput::Legacy(values) => {
            require_length(&values, 4)?;
            require_finite(&values)?;
            OutlierDetectorParameters {
                window: rounded_window(values[0], 2)?,
                min_score: values[1],
                min_residual_ratio: values[2] / 100.0,
                efficiency: values[3],
            }
        }
    };
    validate_outlier_parameters(parameters)
}

/// Parses current ratio parameters or the previous ensemble object format.
pub fn parse_ensemble_parameters_json<R: Read>(
    reader: R,
) -> Result<EnsembleParameters, InputError> {
    let input: EnsembleParameterInput = serde_json::from_reader(reader)?;
    let parameters = match input {
        EnsembleParameterInput::Ratio(parameters) => parameters,
        EnsembleParameterInput::Legacy(parameters) => EnsembleParameters {
            window: parameters.window,
            min_residual_ratio: parameters.min_residual / 100.0,
            efficiency: parameters.efficiency,
            chebyshev_z: parameters.chebyshev_z,
            chebyshev_k: parameters.chebyshev_k,
            isolation_forest_min_score: parameters.isolation_forest_min_score,
            lof_min_score: parameters.lof_min_score,
        },
    };
    validate_ensemble_parameters(parameters)
}

fn validate_detector_parameters(
    parameters: DetectorParameters,
) -> Result<DetectorParameters, InputError> {
    require_finite(&[
        parameters.z,
        parameters.k,
        parameters.min_residual_ratio,
        parameters.efficiency,
    ])?;
    if parameters.window == 0 {
        return Err(InputError::InvalidParameters(
            "window must be positive".to_owned(),
        ));
    }
    if parameters.z <= 0.0 || parameters.k <= 0.0 {
        return Err(InputError::InvalidParameters(
            "z and k must be positive".to_owned(),
        ));
    }
    validate_ratio(parameters.min_residual_ratio)?;
    validate_efficiency(parameters.efficiency)?;
    Ok(parameters)
}

fn validate_outlier_parameters(
    parameters: OutlierDetectorParameters,
) -> Result<OutlierDetectorParameters, InputError> {
    require_finite(&[
        parameters.min_score,
        parameters.min_residual_ratio,
        parameters.efficiency,
    ])?;
    if parameters.window < 2 {
        return Err(InputError::InvalidParameters(
            "window must be at least 2".to_owned(),
        ));
    }
    if parameters.min_score <= 0.0 {
        return Err(InputError::InvalidParameters(
            "min_score must be positive".to_owned(),
        ));
    }
    validate_ratio(parameters.min_residual_ratio)?;
    validate_efficiency(parameters.efficiency)?;
    Ok(parameters)
}

fn validate_ensemble_parameters(
    parameters: EnsembleParameters,
) -> Result<EnsembleParameters, InputError> {
    require_finite(&[
        parameters.min_residual_ratio,
        parameters.efficiency,
        parameters.chebyshev_z,
        parameters.chebyshev_k,
        parameters.isolation_forest_min_score,
        parameters.lof_min_score,
    ])?;
    if parameters.window < 2 {
        return Err(InputError::InvalidParameters(
            "window must be at least 2".to_owned(),
        ));
    }
    if parameters.chebyshev_z <= 0.0
        || parameters.chebyshev_k <= 0.0
        || parameters.isolation_forest_min_score <= 0.0
        || parameters.lof_min_score <= 0.0
    {
        return Err(InputError::InvalidParameters(
            "detector score thresholds must be positive".to_owned(),
        ));
    }
    validate_ratio(parameters.min_residual_ratio)?;
    validate_efficiency(parameters.efficiency)?;
    Ok(parameters)
}

fn require_length(values: &[f64], expected: usize) -> Result<(), InputError> {
    if values.len() != expected {
        return Err(InputError::InvalidParameters(format!(
            "expected {expected} values, found {}",
            values.len()
        )));
    }
    Ok(())
}

fn require_finite(values: &[f64]) -> Result<(), InputError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InputError::InvalidParameters(
            "all values must be finite".to_owned(),
        ));
    }
    Ok(())
}

fn rounded_window(value: f64, minimum: usize) -> Result<usize, InputError> {
    require_finite(&[value])?;
    let rounded = value.round_ties_even();
    if rounded < minimum as f64 || rounded > usize::MAX as f64 {
        return Err(InputError::InvalidParameters(format!(
            "window must round to an integer of at least {minimum}"
        )));
    }
    Ok(rounded as usize)
}

fn validate_ratio(value: f64) -> Result<(), InputError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(InputError::InvalidParameters(
            "min_residual_ratio must be in [0, 1]".to_owned(),
        ));
    }
    Ok(())
}

fn validate_efficiency(value: f64) -> Result<(), InputError> {
    if !(0.0..1.0).contains(&value) {
        return Err(InputError::InvalidParameters(
            "efficiency must be in [0, 1)".to_owned(),
        ));
    }
    Ok(())
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
        assert_eq!(parameters.min_residual_ratio, 0.7488743438197304);
        assert_eq!(parameters.efficiency, 0.05135053360365411);
    }

    #[test]
    fn parses_current_ratio_parameter_object() {
        let input = br#"{
            "window": 60,
            "z": 4,
            "k": 6,
            "min_residual_ratio": 0.05,
            "efficiency": 0.05
        }"#;

        let parameters = parse_parameters_json(input.as_slice()).unwrap();

        assert_eq!(parameters.min_residual_ratio, 0.05);
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

    #[test]
    fn parses_outlier_parameter_array() {
        let parameters =
            parse_outlier_parameters_json(br#"[60.5, 1.5, 5, 0.05]"#.as_slice()).unwrap();

        assert_eq!(parameters.window, 60);
        assert_eq!(parameters.min_score, 1.5);
        assert_eq!(parameters.min_residual_ratio, 0.05);
        assert_eq!(parameters.efficiency, 0.05);
    }

    #[test]
    fn parses_ensemble_parameter_object() {
        let input = br#"{
            "window": 60,
            "min_residual": 5,
            "efficiency": 0.05,
            "chebyshev_z": 4,
            "chebyshev_k": 6,
            "isolation_forest_min_score": 0.8,
            "lof_min_score": 1.5
        }"#;
        let parameters = parse_ensemble_parameters_json(input.as_slice()).unwrap();
        assert_eq!(parameters.window, 60);
        assert_eq!(parameters.min_residual_ratio, 0.05);
        assert_eq!(parameters.lof_min_score, 1.5);
    }

    #[test]
    fn parses_current_ensemble_ratio_object() {
        let input = br#"{
            "window": 60,
            "min_residual_ratio": 0,
            "efficiency": 0.05,
            "chebyshev_z": 4,
            "chebyshev_k": 6,
            "isolation_forest_min_score": 0.8,
            "lof_min_score": 1.5
        }"#;

        let parameters = parse_ensemble_parameters_json(input.as_slice()).unwrap();

        assert_eq!(parameters.min_residual_ratio, 0.0);
    }

    #[test]
    fn rejects_residual_ratios_above_one() {
        let input = br#"{
            "window": 60,
            "z": 4,
            "k": 6,
            "min_residual_ratio": 1.01,
            "efficiency": 0.05
        }"#;

        let error = parse_parameters_json(input.as_slice()).unwrap_err();

        assert!(error.to_string().contains("[0, 1]"));
    }
}
