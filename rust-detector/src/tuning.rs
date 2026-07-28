use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use optirustic::algorithms::{Algorithm, NSGA2Arg, StoppingCondition, NSGA2};
use optirustic::core::{
    BoundedNumber, EvaluationResult, Evaluator, Individual, Objective, ObjectiveDirection, Problem,
    VariableType,
};
use serde::{Deserialize, Serialize};

use crate::{ChebyshevDetector, Detector, DetectorParameters, Observation};

const ANOMALY_COUNT: &str = "anomaly_count";
const NEGATIVE_VISIBILITY: &str = "negative_visibility";
const WINDOW: &str = "window";
const Z: &str = "z";
const K: &str = "k";
const MIN_RESIDUAL: &str = "min_residual";
const EFFICIENCY: &str = "efficiency";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuningConfig {
    pub evaluations: u32,
    pub population_size: usize,
    pub parallel: bool,
    pub seed: Option<u64>,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            evaluations: 5_000,
            population_size: 100,
            parallel: true,
            seed: None,
        }
    }
}

/// A parameter vector and its two minimization objectives.
///
/// This tuple representation matches the Pareto-front JSON files produced for
/// the paper: `[parameters, [anomaly_count, negative_visibility]]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParetoSolution(pub [f64; 5], pub [f64; 2]);

#[derive(Debug, Clone, PartialEq)]
pub struct TuningResult {
    pub solutions: Vec<ParetoSolution>,
    pub evaluations: u32,
}

#[derive(Debug)]
struct ChebyshevEvaluator {
    observations: Arc<[Observation]>,
}

impl Evaluator for ChebyshevEvaluator {
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
        let raw_parameters = parameter_values(individual)?;
        let objectives = evaluate_parameters(&self.observations, raw_parameters);

        Ok(EvaluationResult {
            constraints: None,
            objectives: HashMap::from([
                (ANOMALY_COUNT.to_owned(), objectives[0]),
                (NEGATIVE_VISIBILITY.to_owned(), objectives[1]),
            ]),
        })
    }
}

pub fn tune_chebyshev(
    observations: &[Observation],
    config: TuningConfig,
) -> Result<TuningResult, Box<dyn Error>> {
    validate_config(config)?;
    let problem = create_problem(observations)?;
    let options = NSGA2Arg {
        number_of_individuals: config.population_size,
        stopping_condition: StoppingCondition::MaxFunctionEvaluations(config.evaluations),
        crossover_operator_options: None,
        mutation_operator_options: None,
        resume_from_file: None,
        parallel: Some(config.parallel),
        export_history: None,
        seed: config.seed,
    };
    let mut algorithm = NSGA2::new(problem, options)?;
    algorithm.run()?;

    let mut solutions = algorithm
        .population()
        .individuals()
        .iter()
        .filter(|individual| {
            individual
                .get_data("rank")
                .and_then(|rank| rank.as_integer())
                .is_ok_and(|rank| rank == 1)
        })
        .map(solution_from_individual)
        .collect::<Result<Vec<_>, _>>()?;
    solutions.sort_unstable_by(compare_solutions);
    solutions.dedup();

    Ok(TuningResult {
        solutions,
        evaluations: algorithm.number_of_function_evaluations(),
    })
}

pub fn evaluate_parameters(observations: &[Observation], raw_parameters: [f64; 5]) -> [f64; 2] {
    let parameters = DetectorParameters {
        window: raw_parameters[0].round_ties_even() as usize,
        z: raw_parameters[1],
        k: raw_parameters[2],
        min_residual: raw_parameters[3],
        efficiency: raw_parameters[4],
    };
    let result = ChebyshevDetector::new(parameters).detect(observations);
    let visibility = result
        .anomalies
        .iter()
        .map(|anomaly| anomaly.impact)
        .sum::<f64>();
    [result.anomalies.len() as f64, -visibility]
}

fn create_problem(observations: &[Observation]) -> Result<Problem, Box<dyn Error>> {
    let objectives = vec![
        Objective::new(ANOMALY_COUNT, ObjectiveDirection::Minimise),
        Objective::new(NEGATIVE_VISIBILITY, ObjectiveDirection::Minimise),
    ];
    let variables = vec![
        real_variable(WINDOW, 30.0, 90.0)?,
        real_variable(Z, 3.0, 5.0)?,
        real_variable(K, 5.0, 18.0)?,
        real_variable(MIN_RESIDUAL, 1.0, 100.0)?,
        real_variable(EFFICIENCY, 0.01, 0.1)?,
    ];
    Ok(Problem::new(
        objectives,
        variables,
        None,
        Box::new(ChebyshevEvaluator {
            observations: Arc::from(observations),
        }),
    )?)
}

fn real_variable(name: &str, minimum: f64, maximum: f64) -> Result<VariableType, Box<dyn Error>> {
    Ok(VariableType::Real(BoundedNumber::new(
        name, minimum, maximum,
    )?))
}

fn parameter_values(individual: &Individual) -> Result<[f64; 5], Box<dyn Error>> {
    Ok([
        individual.get_variable_value(WINDOW)?.as_real()?,
        individual.get_variable_value(Z)?.as_real()?,
        individual.get_variable_value(K)?.as_real()?,
        individual.get_variable_value(MIN_RESIDUAL)?.as_real()?,
        individual.get_variable_value(EFFICIENCY)?.as_real()?,
    ])
}

fn solution_from_individual(individual: &Individual) -> Result<ParetoSolution, Box<dyn Error>> {
    Ok(ParetoSolution(
        parameter_values(individual)?,
        [
            individual.get_objective_value(ANOMALY_COUNT)?,
            individual.get_objective_value(NEGATIVE_VISIBILITY)?,
        ],
    ))
}

fn compare_solutions(left: &ParetoSolution, right: &ParetoSolution) -> std::cmp::Ordering {
    left.1[0]
        .total_cmp(&right.1[0])
        .then_with(|| left.1[1].total_cmp(&right.1[1]))
        .then_with(|| {
            left.0
                .iter()
                .zip(right.0)
                .find_map(|(left, right)| {
                    let ordering = left.total_cmp(&right);
                    ordering.is_ne().then_some(ordering)
                })
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn validate_config(config: TuningConfig) -> Result<(), Box<dyn Error>> {
    if config.population_size < 4 || !config.population_size.is_multiple_of(2) {
        return Err("population size must be an even number of at least 4".into());
    }
    if config.evaluations < config.population_size as u32 {
        return Err("evaluations must be at least the population size".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;

    fn observations(values: &[f64]) -> Vec<Observation> {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| Observation {
                date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()
                    + chrono::Duration::days(index as i64),
                value: *value,
                country: None,
                topic: None,
            })
            .collect()
    }

    #[test]
    fn evaluates_the_same_objectives_as_direct_detection() {
        let series = observations(&[
            8.0, 9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 50.0, 45.0, 10.0,
        ]);
        let raw = [5.0, 2.0, 3.0, 20.0, 0.05];

        let objectives = evaluate_parameters(&series, raw);
        let direct = ChebyshevDetector::new(DetectorParameters {
            window: 5,
            z: 2.0,
            k: 3.0,
            min_residual: 20.0,
            efficiency: 0.05,
        })
        .detect(&series);

        assert_eq!(objectives[0], direct.anomalies.len() as f64);
        assert_eq!(
            objectives[1],
            -direct
                .anomalies
                .iter()
                .map(|anomaly| anomaly.impact)
                .sum::<f64>()
        );
    }

    #[test]
    fn validates_population_and_evaluation_counts() {
        assert!(validate_config(TuningConfig {
            evaluations: 100,
            population_size: 3,
            ..TuningConfig::default()
        })
        .is_err());
        assert!(validate_config(TuningConfig {
            evaluations: 50,
            population_size: 100,
            ..TuningConfig::default()
        })
        .is_err());
    }
}
