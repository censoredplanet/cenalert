mod chebyshev;
mod croston;
mod detector;
pub mod input;
pub mod model;
pub mod selection;
mod statistics;
pub mod tuning;
mod window;

pub use chebyshev::{chebyshev_score, ChebyshevScore};
pub use croston::croston_sba_forecast;
pub use detector::{ChebyshevDetector, Detector};
pub use input::{parse_dashboard_json, parse_parameters_json, InputError};
pub use model::{AnnotatedObservation, Anomaly, DetectionResult, DetectorParameters, Observation};
pub use selection::{select_parameters, SelectionError, SelectionMethod, SelectionResult};
pub use statistics::efficiency_ratio;
pub use tuning::{evaluate_parameters, tune_chebyshev, ParetoSolution, TuningConfig, TuningResult};
pub use window::{DemandCategory, SlidingWindow};
