use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use optirustic::algorithms::{Algorithm, NSGA2Arg, StoppingCondition, NSGA2};
use optirustic::core::{
    BoundedNumber, Constraint, EvaluationResult, Evaluator, Individual, Objective,
    ObjectiveDirection, Problem, RelationalOperator, VariableType,
};
use serde::{Deserialize, Serialize};

use crate::{
    label_coverage, ChebyshevDetector, Detector, DetectorParameters, EventLabel, Observation,
};

const ANOMALY_COUNT: &str = "anomaly_count";
const NEGATIVE_VISIBILITY: &str = "negative_visibility";
const WINDOW: &str = "window";
const Z: &str = "z";
const K: &str = "k";
const MIN_RESIDUAL: &str = "min_residual";
const EFFICIENCY: &str = "efficiency";
const MISSED_LABELS: &str = "missed_labels";

#[derive(Debug, Clone, PartialEq)]
pub struct TuningConfig {
    pub evaluations: u32,
    pub population_size: usize,
    pub parallel: bool,
    pub seed: Option<u64>,
    pub k_min: f64,
    pub k_max: f64,
    pub labels: Vec<EventLabel>,
    pub proximity_days: u32,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            evaluations: 5_000,
            population_size: 100,
            parallel: true,
            seed: None,
            k_min: 5.0,
            k_max: 18.0,
            labels: Vec::new(),
            proximity_days: 3,
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
    labels: Arc<[EventLabel]>,
    proximity_days: u32,
}

impl Evaluator for ChebyshevEvaluator {
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
        let raw_parameters = parameter_values(individual)?;
        let parameters = detector_parameters(raw_parameters);
        let result = ChebyshevDetector::new(parameters).detect(&self.observations);
        let objectives = objectives_from_anomalies(&result.anomalies);
        let constraints = if self.labels.is_empty() {
            None
        } else {
            let coverage = label_coverage(&self.labels, &result.anomalies, self.proximity_days);
            Some(HashMap::from([(
                MISSED_LABELS.to_owned(),
                coverage.missed as f64,
            )]))
        };

        Ok(EvaluationResult {
            constraints,
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
    validate_config(&config)?;
    let has_labels = !config.labels.is_empty();
    let problem = create_problem(observations, &config)?;
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
            individual.is_feasible()
                && individual
                    .get_data("rank")
                    .and_then(|rank| rank.as_integer())
                    .is_ok_and(|rank| rank == 1)
        })
        .map(solution_from_individual)
        .collect::<Result<Vec<_>, _>>()?;
    solutions.sort_unstable_by(compare_solutions);
    solutions.dedup();
    if has_labels && solutions.is_empty() {
        return Err("tuning found no parameter set that detects every label".into());
    }

    Ok(TuningResult {
        solutions,
        evaluations: algorithm.number_of_function_evaluations(),
    })
}

pub fn evaluate_parameters(observations: &[Observation], raw_parameters: [f64; 5]) -> [f64; 2] {
    let result = ChebyshevDetector::new(detector_parameters(raw_parameters)).detect(observations);
    objectives_from_anomalies(&result.anomalies)
}

fn detector_parameters(raw_parameters: [f64; 5]) -> DetectorParameters {
    DetectorParameters {
        window: raw_parameters[0].round_ties_even() as usize,
        z: raw_parameters[1],
        k: raw_parameters[2],
        min_residual: raw_parameters[3],
        efficiency: raw_parameters[4],
    }
}

fn objectives_from_anomalies(anomalies: &[crate::Anomaly]) -> [f64; 2] {
    let visibility = anomalies.iter().map(|anomaly| anomaly.impact).sum::<f64>();
    [anomalies.len() as f64, -visibility]
}

fn create_problem(
    observations: &[Observation],
    config: &TuningConfig,
) -> Result<Problem, Box<dyn Error>> {
    let objectives = vec![
        Objective::new(ANOMALY_COUNT, ObjectiveDirection::Minimise),
        Objective::new(NEGATIVE_VISIBILITY, ObjectiveDirection::Minimise),
    ];
    let variables = vec![
        real_variable(WINDOW, 30.0, 90.0)?,
        real_variable(Z, 3.0, 5.0)?,
        real_variable(K, config.k_min, config.k_max)?,
        real_variable(MIN_RESIDUAL, 1.0, 100.0)?,
        real_variable(EFFICIENCY, 0.01, 0.1)?,
    ];
    let constraints = (!config.labels.is_empty()).then(|| {
        vec![Constraint::new(
            MISSED_LABELS,
            RelationalOperator::EqualTo,
            0.0,
        )]
    });
    Ok(Problem::new(
        objectives,
        variables,
        constraints,
        Box::new(ChebyshevEvaluator {
            observations: Arc::from(observations),
            labels: Arc::from(config.labels.as_slice()),
            proximity_days: config.proximity_days,
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

fn validate_config(config: &TuningConfig) -> Result<(), Box<dyn Error>> {
    if config.population_size < 4 || !config.population_size.is_multiple_of(2) {
        return Err("population size must be an even number of at least 4".into());
    }
    if config.evaluations < config.population_size as u32 {
        return Err("evaluations must be at least the population size".into());
    }
    if !config.k_min.is_finite()
        || !config.k_max.is_finite()
        || config.k_min <= 0.0
        || config.k_min >= config.k_max
    {
        return Err("K bounds must be finite, positive, and satisfy k_min < k_max".into());
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
        assert!(validate_config(&TuningConfig {
            evaluations: 100,
            population_size: 3,
            ..TuningConfig::default()
        })
        .is_err());
        assert!(validate_config(&TuningConfig {
            evaluations: 50,
            population_size: 100,
            ..TuningConfig::default()
        })
        .is_err());
    }

    #[test]
    fn validates_k_bounds() {
        assert!(validate_config(&TuningConfig {
            k_min: 10.0,
            k_max: 10.0,
            ..TuningConfig::default()
        })
        .is_err());
        assert!(validate_config(&TuningConfig {
            k_min: 12.0,
            k_max: 10.0,
            ..TuningConfig::default()
        })
        .is_err());
    }

    #[test]
    fn labeled_tuning_rejects_a_population_that_misses_every_label() {
        let series = observations(&vec![10.0; 100]);
        let error = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![EventLabel {
                    date: NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
                    scope: crate::LabelScope::Country,
                    description: None,
                }],
                ..TuningConfig::default()
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("no parameter set"));
    }

    #[test]
    fn labeled_tuning_returns_only_feasible_solutions() {
        let mut values: Vec<f64> = (0..90).map(|index| 8.0 + (index % 5) as f64).collect();
        values.extend([100.0, 90.0, 10.0]);
        let series = observations(&values);
        let result = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![EventLabel {
                    date: NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
                    scope: crate::LabelScope::Country,
                    description: None,
                }],
                ..TuningConfig::default()
            },
        )
        .unwrap();

        assert!(!result.solutions.is_empty());
        for solution in result.solutions {
            let parameters = detector_parameters(solution.0);
            let detection = ChebyshevDetector::new(parameters).detect(&series);
            assert_eq!(
                label_coverage(
                    &[EventLabel {
                        date: NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
                        scope: crate::LabelScope::Country,
                        description: None,
                    }],
                    &detection.anomalies,
                    3,
                )
                .missed,
                0
            );
        }
    }
}
