use std::fmt;

use kneed::knee_locator::{
    InterpMethod, KneeLocator, KneeLocatorParams, ValidCurve, ValidDirection,
};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{storage::Owned, DMatrix, DVector, Dyn, Matrix, Vector};
use serde::Serialize;

use crate::tuning::ParetoSolution;

const MIN_CURVE_FIT_R2: f64 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMethod {
    ExponentialDecay,
    Reciprocal,
    NegativeLogarithm,
    PowerLawDecay,
    InverseSquareRoot,
    Polynomial,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionResult {
    pub parameters: [f64; 5],
    pub objective: [f64; 2],
    pub knee: f64,
    pub method: SelectionMethod,
    pub r_squared: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionError(String);

impl fmt::Display for SelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SelectionError {}

pub fn select_parameters(front: &[ParetoSolution]) -> Result<SelectionResult, SelectionError> {
    let unique = unique_detecting_solutions(front)?;
    if unique.len() == 1 {
        let selected = &unique[0];
        return Ok(SelectionResult {
            parameters: selected.0,
            objective: selected.1,
            knee: selected.1[0],
            method: SelectionMethod::Polynomial,
            r_squared: None,
        });
    }

    let x: Vec<f64> = unique.iter().map(|solution| solution.1[0]).collect();
    let y: Vec<f64> = unique.iter().map(|solution| solution.1[1]).collect();
    let best_fit = CurveModel::ALL
        .into_iter()
        .filter_map(|model| fit_curve(model, &x, &y))
        .max_by(|left, right| left.r_squared.total_cmp(&right.r_squared));

    let (fitted_y, method, r_squared) = match best_fit {
        Some(fit) if fit.r_squared > MIN_CURVE_FIT_R2 => (
            fit.values,
            fit.model.selection_method(),
            Some(fit.r_squared),
        ),
        _ => (
            polynomial_fit(&x, &y, 7)?,
            SelectionMethod::Polynomial,
            None,
        ),
    };

    let locator = KneeLocator::new(
        x.clone(),
        fitted_y,
        1.0,
        KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Decreasing,
            InterpMethod::Interp1d,
        ),
    )
    .map_err(|error| SelectionError(format!("could not locate Pareto knee: {error}")))?;
    let knee = locator.knee.unwrap_or(x[0]);
    let selected = unique
        .iter()
        .find(|solution| solution.1[0] == knee)
        .unwrap_or(&unique[0]);

    Ok(SelectionResult {
        parameters: selected.0,
        objective: selected.1,
        knee,
        method,
        r_squared,
    })
}

pub fn select_nth_from_left(
    front: &[ParetoSolution],
    position: usize,
) -> Result<ParetoSolution, SelectionError> {
    if position == 0 {
        return Err(SelectionError(
            "left-side position must be at least 1".into(),
        ));
    }
    let unique = unique_detecting_solutions(front)?;
    unique.get(position - 1).copied().ok_or_else(|| {
        SelectionError(format!(
            "left-side position {position} exceeds the {} unique detecting solutions",
            unique.len()
        ))
    })
}

fn unique_detecting_solutions(
    front: &[ParetoSolution],
) -> Result<Vec<ParetoSolution>, SelectionError> {
    let mut unique: Vec<ParetoSolution> = Vec::new();
    for solution in front {
        if !solution.1.iter().all(|value| value.is_finite())
            || !solution.0.iter().all(|value| value.is_finite())
        {
            return Err(SelectionError(
                "Pareto front contains a non-finite value".into(),
            ));
        }
        if let Some(existing) = unique.iter_mut().find(|existing| existing.1 == solution.1) {
            *existing = *solution;
        } else if solution.1[0] > 0.0 {
            unique.push(*solution);
        }
    }
    unique.sort_by(|left, right| left.1[0].total_cmp(&right.1[0]));
    if unique.is_empty() {
        return Err(SelectionError(
            "parameter selection requires a positive-anomaly Pareto point".into(),
        ));
    }
    Ok(unique)
}

#[derive(Debug, Clone, Copy)]
enum CurveModel {
    ExponentialDecay,
    Reciprocal,
    NegativeLogarithm,
    PowerLawDecay,
    InverseSquareRoot,
}

impl CurveModel {
    const ALL: [Self; 5] = [
        Self::ExponentialDecay,
        Self::Reciprocal,
        Self::NegativeLogarithm,
        Self::PowerLawDecay,
        Self::InverseSquareRoot,
    ];

    fn parameter_count(self) -> usize {
        if matches!(self, Self::PowerLawDecay) {
            4
        } else {
            3
        }
    }

    fn selection_method(self) -> SelectionMethod {
        match self {
            Self::ExponentialDecay => SelectionMethod::ExponentialDecay,
            Self::Reciprocal => SelectionMethod::Reciprocal,
            Self::NegativeLogarithm => SelectionMethod::NegativeLogarithm,
            Self::PowerLawDecay => SelectionMethod::PowerLawDecay,
            Self::InverseSquareRoot => SelectionMethod::InverseSquareRoot,
        }
    }

    fn evaluate(self, x: f64, parameters: &[f64]) -> Option<f64> {
        let value = match self {
            Self::ExponentialDecay => parameters[0] * (-parameters[1] * x).exp() + parameters[2],
            Self::Reciprocal => parameters[0] / (x + parameters[1]) + parameters[2],
            Self::NegativeLogarithm => -parameters[0] * (x + parameters[1]).ln() + parameters[2],
            Self::PowerLawDecay => {
                parameters[0] / (x + parameters[1]).powf(parameters[2]) + parameters[3]
            }
            Self::InverseSquareRoot => parameters[0] / (x + parameters[1]).sqrt() + parameters[2],
        };
        value.is_finite().then_some(value)
    }

    fn derivatives(self, x: f64, parameters: &[f64]) -> Option<Vec<f64>> {
        let derivatives = match self {
            Self::ExponentialDecay => {
                let exponential = (-parameters[1] * x).exp();
                vec![exponential, -parameters[0] * x * exponential, 1.0]
            }
            Self::Reciprocal => {
                let denominator = x + parameters[1];
                vec![1.0 / denominator, -parameters[0] / denominator.powi(2), 1.0]
            }
            Self::NegativeLogarithm => {
                let argument = x + parameters[1];
                vec![-argument.ln(), -parameters[0] / argument, 1.0]
            }
            Self::PowerLawDecay => {
                let base = x + parameters[1];
                let power = base.powf(-parameters[2]);
                vec![
                    power,
                    -parameters[0] * parameters[2] * base.powf(-parameters[2] - 1.0),
                    -parameters[0] * base.ln() * power,
                    1.0,
                ]
            }
            Self::InverseSquareRoot => {
                let base = x + parameters[1];
                vec![base.powf(-0.5), -0.5 * parameters[0] * base.powf(-1.5), 1.0]
            }
        };
        derivatives
            .iter()
            .all(|value| value.is_finite())
            .then_some(derivatives)
    }
}

struct CurveFit {
    model: CurveModel,
    values: Vec<f64>,
    r_squared: f64,
}

struct CurveFitProblem {
    model: CurveModel,
    x: Vec<f64>,
    y: Vec<f64>,
    parameters: DVector<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for CurveFitProblem {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, parameters: &Vector<f64, Dyn, Self::ParameterStorage>) {
        self.parameters.copy_from(parameters);
    }

    fn params(&self) -> Vector<f64, Dyn, Self::ParameterStorage> {
        self.parameters.clone()
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        let residuals: Option<Vec<f64>> = self
            .x
            .iter()
            .zip(&self.y)
            .map(|(&x, &y)| {
                self.model
                    .evaluate(x, self.parameters.as_slice())
                    .map(|prediction| prediction - y)
            })
            .collect();
        residuals.map(DVector::from_vec)
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        let mut jacobian = DMatrix::zeros(self.x.len(), self.parameters.len());
        for (row, &x) in self.x.iter().enumerate() {
            let derivatives = self.model.derivatives(x, self.parameters.as_slice())?;
            for (column, derivative) in derivatives.into_iter().enumerate() {
                jacobian[(row, column)] = derivative;
            }
        }
        Some(jacobian)
    }
}

fn fit_curve(model: CurveModel, x: &[f64], y: &[f64]) -> Option<CurveFit> {
    let problem = CurveFitProblem {
        model,
        x: x.to_vec(),
        y: y.to_vec(),
        parameters: DVector::from_element(model.parameter_count(), 1.0),
    };
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);
    if !report.termination.was_successful() {
        return None;
    }
    let values: Option<Vec<f64>> = x
        .iter()
        .map(|&value| model.evaluate(value, problem.parameters.as_slice()))
        .collect();
    let values = values?;
    if !is_convex_decreasing(x, &values) {
        return None;
    }
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let total = y.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    if total == 0.0 {
        return None;
    }
    let residual = y
        .iter()
        .zip(&values)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>();
    Some(CurveFit {
        model,
        values,
        r_squared: 1.0 - residual / total,
    })
}

fn is_convex_decreasing(x: &[f64], y: &[f64]) -> bool {
    let slopes: Option<Vec<f64>> = x
        .windows(2)
        .zip(y.windows(2))
        .map(|(x, y)| {
            let width = x[1] - x[0];
            (width > 0.0).then_some((y[1] - y[0]) / width)
        })
        .collect();
    let Some(slopes) = slopes else {
        return false;
    };
    slopes.iter().all(|slope| *slope < 0.0) && slopes.windows(2).all(|slopes| slopes[1] > slopes[0])
}

fn polynomial_fit(x: &[f64], y: &[f64], maximum_degree: usize) -> Result<Vec<f64>, SelectionError> {
    let degree = maximum_degree.min(x.len() - 1);
    let center = (x[0] + x[x.len() - 1]) / 2.0;
    let scale = ((x[x.len() - 1] - x[0]) / 2.0).max(1.0);
    let normalized: Vec<f64> = x.iter().map(|value| (value - center) / scale).collect();
    let matrix = DMatrix::from_fn(x.len(), degree + 1, |row, column| {
        normalized[row].powi((degree - column) as i32)
    });
    let coefficients = matrix
        .svd(true, true)
        .solve(&DVector::from_column_slice(y), f64::EPSILON.sqrt())
        .map_err(|error| SelectionError(format!("polynomial fit failed: {error}")))?;
    Ok(normalized
        .iter()
        .map(|value| {
            coefficients
                .iter()
                .fold(0.0, |result, coefficient| result * value + coefficient)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_a_point_from_a_convex_front() {
        let front = vec![
            ParetoSolution([1.0; 5], [1.0, -10.0]),
            ParetoSolution([2.0; 5], [2.0, -20.0]),
            ParetoSolution([3.0; 5], [3.0, -24.0]),
            ParetoSolution([4.0; 5], [4.0, -26.0]),
            ParetoSolution([5.0; 5], [5.0, -27.0]),
        ];
        let result = select_parameters(&front).unwrap();

        assert!(front.iter().any(|solution| solution.0 == result.parameters));
        assert!(front.iter().any(|solution| solution.1 == result.objective));
    }

    #[test]
    fn rejects_front_without_a_detecting_solution() {
        let error = select_parameters(&[ParetoSolution([1.0; 5], [0.0, 0.0])]).unwrap_err();
        assert!(error.to_string().contains("positive-anomaly"));
    }

    #[test]
    fn selects_the_only_detecting_solution() {
        let expected = ParetoSolution([2.0; 5], [1.0, -10.0]);
        let result = select_parameters(&[ParetoSolution([1.0; 5], [0.0, 0.0]), expected]).unwrap();

        assert_eq!(result.parameters, expected.0);
        assert_eq!(result.objective, expected.1);
    }

    #[test]
    fn selects_a_one_based_position_from_the_left() {
        let front = vec![
            ParetoSolution([3.0; 5], [3.0, -30.0]),
            ParetoSolution([1.0; 5], [1.0, -10.0]),
            ParetoSolution([2.0; 5], [2.0, -20.0]),
        ];

        assert_eq!(select_nth_from_left(&front, 2).unwrap(), front[2]);
    }

    #[test]
    fn left_selection_rejects_zero_and_out_of_range_positions() {
        let front = [ParetoSolution([1.0; 5], [1.0, -10.0])];

        assert!(select_nth_from_left(&front, 0).is_err());
        assert!(select_nth_from_left(&front, 2).is_err());
    }
}
