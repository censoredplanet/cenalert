use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::DemandCategory;

/// One normalized daily observation consumed by the detector core.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Observation {
    pub date: NaiveDate,
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
}

/// Parameters used by the production Chebyshev/Z-Score detector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DetectorParameters {
    pub window: usize,
    pub z: f64,
    pub k: f64,
    pub min_residual: f64,
    pub efficiency: f64,
}

/// Parameters shared by Isolation Forest and Local Outlier Factor detectors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct OutlierDetectorParameters {
    pub window: usize,
    pub min_score: f64,
    pub min_residual: f64,
    pub efficiency: f64,
}

/// Three-detector ensemble with shared time-series and Croston parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnsembleParameters {
    pub window: usize,
    pub min_residual: f64,
    pub efficiency: f64,
    pub chebyshev_z: f64,
    pub chebyshev_k: f64,
    pub isolation_forest_min_score: f64,
    pub lof_min_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnnotatedObservation {
    pub date: NaiveDate,
    pub value: f64,
    pub anomaly: bool,
    pub score: Option<f64>,
    pub residual: Option<f64>,
    pub threshold: Option<f64>,
    pub minimum_score: Option<f64>,
    pub coefficient_of_variation_squared: Option<f64>,
    pub average_interdemand_interval: Option<f64>,
    pub demand_category: Option<DemandCategory>,
}

impl AnnotatedObservation {
    pub(crate) fn new(observation: &Observation) -> Self {
        Self {
            date: observation.date,
            value: observation.value,
            anomaly: false,
            score: None,
            residual: None,
            threshold: None,
            minimum_score: None,
            coefficient_of_variation_squared: None,
            average_interdemand_interval: None,
            demand_category: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Anomaly {
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub peak: NaiveDate,
    pub score: f64,
    pub residual: f64,
    pub impact: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    pub annotated: Vec<AnnotatedObservation>,
    pub anomalies: Vec<Anomaly>,
    pub scale_factor: f64,
}
