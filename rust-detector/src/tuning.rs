use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use optirustic::algorithms::{Algorithm, AlgorithmSerialisedExport, NSGA2Arg, StoppingCondition, NSGA2};
use optirustic::core::{
    BoundedNumber, Constraint, EvaluationResult, Evaluator, Individual, Objective,
    ObjectiveDirection, Problem, RelationalOperator, VariableType, VariableValue,
};
use serde::{Deserialize, Serialize};

use crate::{
    label_coverage, labels::required_country_labels, ChebyshevDetector, Detector,
    DetectorParameters, EnsembleDetector, EnsembleParameters, EventLabel, IsolationForestDetector,
    LocalOutlierFactorDetector, Observation, OutlierDetectorParameters,
};

const ANOMALY_COUNT: &str = "anomaly_count";
const NEGATIVE_VISIBILITY: &str = "negative_visibility";
const WINDOW: &str = "window";
const Z: &str = "z";
const K: &str = "k";
const MIN_RESIDUAL_RATIO: &str = "min_residual_ratio";
const EFFICIENCY: &str = "efficiency";
const ISOLATION_FOREST_MIN_SCORE: &str = "isolation_forest_min_score";
const LOF_MIN_SCORE: &str = "lof_min_score";
const MISSED_LABELS: &str = "missed_labels";
const MEMBER_IMBALANCE: &str = "member_imbalance";

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
    /// Ensemble vectors to include in the initial population, in Pareto JSON order.
    pub initial_parameters: Vec<[f64; 7]>,
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
            initial_parameters: Vec::new(),
        }
    }
}

/// A parameter vector and its two minimization objectives.
///
/// This tuple representation matches the Pareto-front JSON files produced for
/// the paper: `[parameters, [anomaly_count, negative_visibility]]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParetoSolution(pub [f64; 5], pub [f64; 2]);

/// A seven-parameter ensemble vector and its two minimization objectives.
///
/// Parameters are ordered as window, minimum residual ratio, efficiency,
/// Chebyshev Z, Chebyshev K, Isolation Forest minimum score, and LOF minimum
/// score.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnsembleParetoSolution(pub [f64; 7], pub [f64; 2]);

#[derive(Debug, Clone, PartialEq)]
pub struct TuningResult {
    pub solutions: Vec<ParetoSolution>,
    pub evaluations: u32,
    pub required_labels: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleTuningResult {
    pub solutions: Vec<EnsembleParetoSolution>,
    pub evaluations: u32,
    pub required_labels: usize,
}

#[derive(Debug)]
struct ChebyshevEvaluator {
    observations: Arc<[Observation]>,
    labels: Arc<[EventLabel]>,
    proximity_days: u32,
}

#[derive(Debug)]
struct EnsembleEvaluator {
    observations: Arc<[Observation]>,
    labels: Arc<[EventLabel]>,
    proximity_days: u32,
}

impl Evaluator for ChebyshevEvaluator {
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
        let raw_parameters = parameter_values(individual)?;
        let parameters = chebyshev_parameters(raw_parameters);
        let result = ChebyshevDetector::new(parameters).detect(&self.observations);
        evaluation_result(&result.anomalies, &self.labels, self.proximity_days)
    }
}

impl Evaluator for EnsembleEvaluator {
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
        let parameters = ensemble_parameters(ensemble_parameter_values(individual)?);
        let result = EnsembleDetector::new(parameters).detect(&self.observations);
        let counts = ensemble_member_counts(&self.observations, parameters);
        ensemble_evaluation_result(&result.anomalies, &self.labels, self.proximity_days, counts)
    }
}

pub fn tune_chebyshev(
    observations: &[Observation],
    config: TuningConfig,
) -> Result<TuningResult, Box<dyn Error>> {
    validate_config(&config)?;
    if !config.initial_parameters.is_empty() {
        return Err("initial parameters are supported only for ensemble tuning".into());
    }
    let (required_labels, required_label_count) = tuning_labels(observations, &config)?;
    let problem = create_problem(observations, &config, &required_labels)?;
    let mut algorithm = NSGA2::new(problem, nsga_options(&config))?;
    algorithm.run()?;

    let mut solutions = algorithm
        .population()
        .individuals()
        .iter()
        .filter(is_feasible_rank_one)
        .map(solution_from_individual)
        .collect::<Result<Vec<_>, _>>()?;
    solutions.sort_unstable_by(compare_solutions);
    solutions.dedup();
    if required_label_count > 0 && solutions.is_empty() {
        return Err("tuning found no parameter set that detects every label".into());
    }

    Ok(TuningResult {
        solutions,
        evaluations: algorithm.number_of_function_evaluations(),
        required_labels: required_label_count,
    })
}

pub fn tune_ensemble(
    observations: &[Observation],
    config: TuningConfig,
) -> Result<EnsembleTuningResult, Box<dyn Error>> {
    validate_config(&config)?;
    let (required_labels, required_label_count) = tuning_labels(observations, &config)?;
    let problem = create_ensemble_problem(observations, &config, &required_labels)?;
    let draft = NSGA2::new(problem, nsga_options(&config))?;
    let (mut algorithm, initial_evaluations) = if config.initial_parameters.is_empty() {
        (draft, 0)
    } else {
        warm_start_ensemble(draft, observations, &config, &required_labels)?
    };
    if initial_evaluations == 0 {
        algorithm.run()?;
    } else {
        algorithm.initialise()?;
        while initial_evaluations + algorithm.number_of_function_evaluations() < config.evaluations {
            algorithm.evolve()?;
        }
    }

    let mut solutions = algorithm
        .population()
        .individuals()
        .iter()
        .filter(is_feasible_rank_one)
        .map(ensemble_solution_from_individual)
        .collect::<Result<Vec<_>, _>>()?;
    solutions.sort_unstable_by(compare_ensemble_solutions);
    solutions.dedup();
    if solutions.is_empty() {
        return Err("tuning found no ensemble parameter set satisfying member-balance and applicable label constraints".into());
    }

    Ok(EnsembleTuningResult {
        solutions,
        evaluations: initial_evaluations + algorithm.number_of_function_evaluations(),
        required_labels: required_label_count,
    })
}

fn warm_start_ensemble(
    draft: NSGA2,
    observations: &[Observation],
    config: &TuningConfig,
    labels: &[EventLabel],
) -> Result<(NSGA2, u32), Box<dyn Error>> {
    if config.initial_parameters.len() > config.population_size {
        return Err("initial parameter count exceeds population size".into());
    }
    let mut export = draft.get_results();
    let names = [
        WINDOW, MIN_RESIDUAL_RATIO, EFFICIENCY, Z, K,
        ISOLATION_FOREST_MIN_SCORE, LOF_MIN_SCORE,
    ];
    for (individual, values) in export.individuals.iter_mut().zip(&config.initial_parameters) {
        for (name, value) in names.iter().zip(values) {
            let VariableType::Real(variable) = export.problem.get_variable(name)? else {
                return Err("ensemble tuning requires real-valued variables".into());
            };
            let (lower, upper) = variable.bounds();
            if !value.is_finite() || !(lower..=upper).contains(value) {
                return Err(format!("initial {name}={value} is outside [{lower}, {upper}]").into());
            }
            individual.update_variable(name, VariableValue::Real(*value))?;
        }
    }
    // optirustic imports evaluated populations. Recompute every objective and
    // constraint with the current evaluator before using its population loader.
    let mut evaluations = 0;
    if config.parallel {
        NSGA2::do_parallel_evaluation(&mut export.individuals, &mut evaluations)?;
    } else {
        NSGA2::do_evaluation(&mut export.individuals, &mut evaluations)?;
    }
    let population = AlgorithmSerialisedExport {
        options: draft.algorithm_options(),
        problem: export.problem.serialise(),
        individuals: export.individuals.iter().map(Individual::serialise).collect(),
        generation: 0,
        number_of_function_evaluations: evaluations,
        algorithm: export.algorithm,
        additional_data: None,
        took: export.took,
        exported_on: chrono::Utc::now(),
    };
    let (temporary, file) = PopulationFile::create()?;
    serde_json::to_writer(file, &population)?;
    let mut options = nsga_options(config);
    options.resume_from_file = Some(temporary.0.clone());
    let problem = create_ensemble_problem(observations, config, labels)?;
    Ok((NSGA2::new(problem, options)?, evaluations))
}

struct PopulationFile(std::path::PathBuf);

impl PopulationFile {
    fn create() -> std::io::Result<(Self, std::fs::File)> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "cenalert-population-{}-{}.json",
                std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            match std::fs::OpenOptions::new()
                .write(true).create_new(true).open(&path)
            {
                Ok(file) => return Ok((Self(path), file)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

impl Drop for PopulationFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub fn evaluate_parameters(observations: &[Observation], raw_parameters: [f64; 5]) -> [f64; 2] {
    let result = ChebyshevDetector::new(chebyshev_parameters(raw_parameters)).detect(observations);
    objectives_from_anomalies(&result.anomalies)
}

pub fn evaluate_ensemble_parameters(
    observations: &[Observation],
    raw_parameters: [f64; 7],
) -> [f64; 2] {
    let result = EnsembleDetector::new(ensemble_parameters(raw_parameters)).detect(observations);
    objectives_from_anomalies(&result.anomalies)
}

pub fn chebyshev_parameters(raw_parameters: [f64; 5]) -> DetectorParameters {
    DetectorParameters {
        window: raw_parameters[0].round_ties_even() as usize,
        z: raw_parameters[1],
        k: raw_parameters[2],
        min_residual_ratio: raw_parameters[3],
        efficiency: raw_parameters[4],
    }
}

pub fn ensemble_parameters(raw_parameters: [f64; 7]) -> EnsembleParameters {
    EnsembleParameters {
        window: raw_parameters[0].round_ties_even() as usize,
        min_residual_ratio: raw_parameters[1],
        efficiency: raw_parameters[2],
        chebyshev_z: raw_parameters[3],
        chebyshev_k: raw_parameters[4],
        isolation_forest_min_score: raw_parameters[5],
        lof_min_score: raw_parameters[6],
    }
}

/// Dense anomaly intervals from independent Chebyshev, iForest, and LOF runs,
/// in that order. Each member uses the candidate's shared parameters and its
/// own evolving history, including the normal end-of-anomaly rules.
pub fn ensemble_member_counts(
    observations: &[Observation],
    parameters: EnsembleParameters,
) -> [usize; 3] {
    let outlier_parameters = |min_score| OutlierDetectorParameters {
        window: parameters.window,
        min_score,
        min_residual_ratio: parameters.min_residual_ratio,
        efficiency: parameters.efficiency,
    };
    let chebyshev = ChebyshevDetector::new(DetectorParameters {
        window: parameters.window,
        z: parameters.chebyshev_z,
        k: parameters.chebyshev_k,
        min_residual_ratio: parameters.min_residual_ratio,
        efficiency: parameters.efficiency,
    })
    .detect(observations);
    let isolation_forest =
        IsolationForestDetector::new(outlier_parameters(parameters.isolation_forest_min_score))
            .detect(observations);
    let lof = LocalOutlierFactorDetector::new(outlier_parameters(parameters.lof_min_score))
        .detect(observations);
    [chebyshev, isolation_forest, lof].map(|result| dense_interval_count(&result.anomalies))
}

fn dense_interval_count(anomalies: &[crate::Anomaly]) -> usize {
    anomalies.iter().filter(|anomaly| !anomaly.sparse).count()
}

fn member_imbalance(counts: [usize; 3]) -> usize {
    let largest = *counts.iter().max().expect("three ensemble members");
    let others = counts.iter().sum::<usize>() - largest;
    largest.saturating_sub(others)
}

fn ensemble_evaluation_result(
    anomalies: &[crate::Anomaly],
    labels: &[EventLabel],
    proximity_days: u32,
    counts: [usize; 3],
) -> Result<EvaluationResult, Box<dyn Error>> {
    let mut result = evaluation_result(anomalies, labels, proximity_days)?;
    result
        .constraints
        .get_or_insert_with(HashMap::new)
        .insert(MEMBER_IMBALANCE.to_owned(), member_imbalance(counts) as f64);
    Ok(result)
}

fn evaluation_result(
    anomalies: &[crate::Anomaly],
    labels: &[EventLabel],
    proximity_days: u32,
) -> Result<EvaluationResult, Box<dyn Error>> {
    let objectives = objectives_from_anomalies(anomalies);
    let constraints = if labels.is_empty() {
        None
    } else {
        let coverage = label_coverage(labels, anomalies, proximity_days);
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

fn objectives_from_anomalies(anomalies: &[crate::Anomaly]) -> [f64; 2] {
    let visibility = anomalies.iter().map(|anomaly| anomaly.impact).sum::<f64>();
    [anomalies.len() as f64, -visibility]
}

fn create_problem(
    observations: &[Observation],
    config: &TuningConfig,
    required_labels: &[EventLabel],
) -> Result<Problem, Box<dyn Error>> {
    let objectives = vec![
        Objective::new(ANOMALY_COUNT, ObjectiveDirection::Minimise),
        Objective::new(NEGATIVE_VISIBILITY, ObjectiveDirection::Minimise),
    ];
    let variables = vec![
        real_variable(WINDOW, 30.0, 90.0)?,
        real_variable(Z, 3.0, 5.0)?,
        real_variable(K, config.k_min, config.k_max)?,
        real_variable(MIN_RESIDUAL_RATIO, 0.0, 1.0)?,
        real_variable(EFFICIENCY, 0.01, 0.1)?,
    ];
    let constraints = (!required_labels.is_empty()).then(|| {
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
            labels: Arc::from(required_labels),
            proximity_days: config.proximity_days,
        }),
    )?)
}

fn create_ensemble_problem(
    observations: &[Observation],
    config: &TuningConfig,
    required_labels: &[EventLabel],
) -> Result<Problem, Box<dyn Error>> {
    let objectives = vec![
        Objective::new(ANOMALY_COUNT, ObjectiveDirection::Minimise),
        Objective::new(NEGATIVE_VISIBILITY, ObjectiveDirection::Minimise),
    ];
    let variables = vec![
        real_variable(WINDOW, 30.0, 90.0)?,
        real_variable(MIN_RESIDUAL_RATIO, 0.0, 1.0)?,
        // The original LOF search used 0.05 as its lower efficiency bound.
        // Since efficiency is shared, use the intersection of member ranges.
        real_variable(EFFICIENCY, 0.05, 0.1)?,
        real_variable(Z, 3.0, 5.0)?,
        real_variable(K, config.k_min, config.k_max)?,
        real_variable(ISOLATION_FOREST_MIN_SCORE, 0.6, 1.0)?,
        real_variable(LOF_MIN_SCORE, 1.2, 3.0)?,
    ];
    let mut constraints = vec![Constraint::new(
        MEMBER_IMBALANCE,
        RelationalOperator::EqualTo,
        0.0,
    )];
    if !required_labels.is_empty() {
        constraints.push(Constraint::new(
            MISSED_LABELS,
            RelationalOperator::EqualTo,
            0.0,
        ));
    }
    Ok(Problem::new(
        objectives,
        variables,
        Some(constraints),
        Box::new(EnsembleEvaluator {
            observations: Arc::from(observations),
            labels: Arc::from(required_labels),
            proximity_days: config.proximity_days,
        }),
    )?)
}

fn tuning_labels(
    observations: &[Observation],
    config: &TuningConfig,
) -> Result<(Vec<EventLabel>, usize), Box<dyn Error>> {
    let country_code = if config.labels.is_empty() {
        None
    } else {
        observation_country_code(observations)?
    };
    if country_code.is_none()
        && config
            .labels
            .iter()
            .any(|label| label.scope == crate::LabelScope::Country && label.country_code.is_some())
    {
        return Err(
            "the input series must identify its country when labels contain country codes".into(),
        );
    }
    let required_labels = required_labels_for_series(
        observations,
        &config.labels,
        country_code,
        config.proximity_days,
    );
    let count = required_labels.len();
    Ok((required_labels, count))
}

fn nsga_options(config: &TuningConfig) -> NSGA2Arg {
    NSGA2Arg {
        number_of_individuals: config.population_size,
        stopping_condition: StoppingCondition::MaxFunctionEvaluations(config.evaluations),
        crossover_operator_options: None,
        mutation_operator_options: None,
        resume_from_file: None,
        parallel: Some(config.parallel),
        export_history: None,
        seed: config.seed,
    }
}

fn is_feasible_rank_one(individual: &&Individual) -> bool {
    individual.is_feasible()
        && individual
            .get_data("rank")
            .and_then(|rank| rank.as_integer())
            .is_ok_and(|rank| rank == 1)
}

fn observation_country_code(observations: &[Observation]) -> Result<Option<&str>, Box<dyn Error>> {
    let mut country_code = None;
    for observation in observations {
        let Some(candidate) = observation.country.as_deref() else {
            continue;
        };
        if country_code.is_some_and(|country: &str| !country.eq_ignore_ascii_case(candidate)) {
            return Err("label-constrained tuning requires a single-country time series".into());
        }
        country_code = Some(candidate);
    }
    Ok(country_code)
}

fn required_labels_for_series(
    observations: &[Observation],
    labels: &[EventLabel],
    country_code: Option<&str>,
    proximity_days: u32,
) -> Vec<EventLabel> {
    let Some(first_date) = observations
        .iter()
        .map(|observation| observation.date)
        .min()
    else {
        return Vec::new();
    };
    let last_date = observations
        .iter()
        .map(|observation| observation.date)
        .max()
        .expect("a minimum date implies a maximum date");
    let proximity_days = i64::from(proximity_days);

    required_country_labels(labels, country_code)
        .into_iter()
        .filter(|label| {
            label.date.signed_duration_since(first_date).num_days() >= -proximity_days
                && label.date.signed_duration_since(last_date).num_days() <= proximity_days
        })
        .collect()
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
        individual
            .get_variable_value(MIN_RESIDUAL_RATIO)?
            .as_real()?,
        individual.get_variable_value(EFFICIENCY)?.as_real()?,
    ])
}

fn ensemble_parameter_values(individual: &Individual) -> Result<[f64; 7], Box<dyn Error>> {
    Ok([
        individual.get_variable_value(WINDOW)?.as_real()?,
        individual
            .get_variable_value(MIN_RESIDUAL_RATIO)?
            .as_real()?,
        individual.get_variable_value(EFFICIENCY)?.as_real()?,
        individual.get_variable_value(Z)?.as_real()?,
        individual.get_variable_value(K)?.as_real()?,
        individual
            .get_variable_value(ISOLATION_FOREST_MIN_SCORE)?
            .as_real()?,
        individual.get_variable_value(LOF_MIN_SCORE)?.as_real()?,
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

fn ensemble_solution_from_individual(
    individual: &Individual,
) -> Result<EnsembleParetoSolution, Box<dyn Error>> {
    Ok(EnsembleParetoSolution(
        ensemble_parameter_values(individual)?,
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

fn compare_ensemble_solutions(
    left: &EnsembleParetoSolution,
    right: &EnsembleParetoSolution,
) -> std::cmp::Ordering {
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

    fn with_country(mut observations: Vec<Observation>, country: &str) -> Vec<Observation> {
        for observation in &mut observations {
            observation.country = Some(country.to_owned());
        }
        observations
    }

    #[test]
    fn evaluates_the_same_objectives_as_direct_detection() {
        let series = observations(&[
            8.0, 9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 50.0, 45.0, 10.0,
        ]);
        let raw = [5.0, 2.0, 3.0, 0.2, 0.05];

        let objectives = evaluate_parameters(&series, raw);
        let direct = ChebyshevDetector::new(DetectorParameters {
            window: 5,
            z: 2.0,
            k: 3.0,
            min_residual_ratio: 0.2,
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
    fn evaluates_ensemble_objectives_as_direct_detection() {
        let mut values: Vec<f64> = (0..90).map(|index| 8.0 + (index % 5) as f64).collect();
        values.extend([100.0, 90.0, 10.0]);
        let series = observations(&values);
        let raw = [60.0, 0.05, 0.05, 4.0, 6.0, 0.8, 1.5];

        let objectives = evaluate_ensemble_parameters(&series, raw);
        let direct = EnsembleDetector::new(ensemble_parameters(raw)).detect(&series);

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
    fn member_balance_accepts_equality_and_zero_counts() {
        for counts in [[0, 0, 0], [1, 1, 0], [100, 120, 220], [20, 20, 20]] {
            assert_eq!(member_imbalance(counts), 0);
        }
        for counts in [[100, 120, 500], [500, 120, 100], [100, 500, 120]] {
            assert_eq!(member_imbalance(counts), 280);
        }
        assert_eq!(member_imbalance([0, 1, 0]), 1);
    }

    #[test]
    fn dense_interval_count_excludes_croston_and_ignores_duration() {
        let first = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let anomaly = crate::Anomaly {
            start: first,
            end: first + chrono::Duration::days(9),
            peak: first,
            peak_value: None,
            score: 3.0,
            residual: 1.0,
            impact: 10.0,
            sparse: false,
            component_starts: Vec::new(),
            ensemble_vote: None,
        };
        let sparse = crate::Anomaly {
            sparse: true,
            ..anomaly.clone()
        };
        assert_eq!(dense_interval_count(&[anomaly, sparse]), 1);
    }

    #[test]
    fn balance_is_enforced_with_and_without_label_constraints() {
        let series = observations(&vec![0.0; 100]);
        let label = EventLabel::new(series[50].date, crate::LabelScope::Country);
        for labels in [Vec::new(), vec![label]] {
            let problem = Arc::new(
                create_ensemble_problem(&series, &TuningConfig::default(), &labels).unwrap(),
            );
            assert!(problem
                .constraint_names()
                .contains(&MEMBER_IMBALANCE.to_owned()));
            assert_eq!(
                problem.number_of_constraints(),
                1 + usize::from(!labels.is_empty())
            );
            let individual = Individual::new(Arc::clone(&problem));
            let evaluated = problem.evaluator().evaluate(&individual).unwrap();
            let constraints = evaluated.constraints.unwrap();
            assert_eq!(constraints[MEMBER_IMBALANCE], 0.0);
            if !labels.is_empty() {
                assert_eq!(constraints[MISSED_LABELS], 1.0);
            }
            let imbalanced = ensemble_evaluation_result(&[], &labels, 3, [1, 5, 1]).unwrap();
            assert_eq!(imbalanced.constraints.unwrap()[MEMBER_IMBALANCE], 3.0);
            let single = create_problem(&series, &TuningConfig::default(), &labels).unwrap();
            assert!(!single
                .constraint_names()
                .contains(&MEMBER_IMBALANCE.to_owned()));
        }
    }

    #[test]
    fn member_counts_use_independent_interval_end_conditions() {
        let mut values = vec![10.0; 90];
        values.extend([100.0, 90.0, 80.0, 10.0]);
        let series = observations(&values);
        let parameters = ensemble_parameters([60.0, 0.05, 0.05, 4.0, 6.0, 0.6, 1.2]);
        let counts = ensemble_member_counts(&series, parameters);
        assert_eq!(counts, [1, 1, 1]);
        let mut sparse_values = vec![0.0; 90];
        sparse_values.extend([100.0, 0.0, 0.0, 90.0, 0.0]);
        let sparse = observations(&sparse_values);
        assert!(!EnsembleDetector::new(parameters)
            .detect(&sparse)
            .anomalies
            .is_empty());
        assert_eq!(ensemble_member_counts(&sparse, parameters), [0, 0, 0]);
    }

    #[test]
    fn warm_start_keeps_supplied_vectors_and_counts_initial_evaluations() {
        let seed = [37.0, 0.002, 0.07, 3.1, 15.0, 0.75, 1.3];
        for (evaluations, parallel) in [(4, false), (8, true)] {
            let result = tune_ensemble(
                &observations(&vec![0.0; 100]),
                TuningConfig {
                    evaluations,
                    population_size: 4,
                    parallel,
                    initial_parameters: vec![seed],
                    ..TuningConfig::default()
                },
            ).unwrap();
            assert_eq!(result.evaluations, evaluations);
            if evaluations == 4 {
                assert!(result.solutions.iter().any(|solution| solution.0 == seed));
            }
        }
    }

    #[test]
    fn warm_start_rechecks_current_label_constraints() {
        let result = tune_ensemble(
            &observations(&vec![0.0; 100]),
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                initial_parameters: vec![[37.0, 0.002, 0.07, 3.1, 15.0, 0.75, 1.3]],
                labels: vec![EventLabel::new(
                    NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
                    crate::LabelScope::Country,
                )],
                ..TuningConfig::default()
            },
        );
        assert!(result.unwrap_err().to_string().contains("no ensemble parameter set"));
    }

    #[test]
    fn warm_start_rejects_invalid_vectors() {
        let seed = [37.0, 0.002, 0.07, 3.1, 15.0, 0.75, 1.3];
        let series = observations(&vec![0.0; 100]);
        let config = TuningConfig {
            evaluations: 4, population_size: 4, parallel: false,
            initial_parameters: vec![seed], ..TuningConfig::default()
        };
        assert!(tune_chebyshev(&series, config.clone()).is_err());
        let mut too_many = config.clone();
        too_many.initial_parameters = vec![seed; 5];
        assert!(tune_ensemble(&series, too_many).unwrap_err().to_string()
            .contains("exceeds population size"));
        for value in [1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = config.clone();
            invalid.initial_parameters[0][6] = value;
            assert!(tune_ensemble(&series, invalid).unwrap_err().to_string()
                .contains("outside"));
        }
    }

    #[test]
    fn ensemble_tuning_returns_valid_rank_one_parameters() {
        let mut values = vec![10.0; 90];
        values.extend([100.0, 90.0, 10.0]);
        let result = tune_ensemble(
            &observations(&values),
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                seed: Some(42),
                ..TuningConfig::default()
            },
        )
        .unwrap();

        assert!(!result.solutions.is_empty());
        for solution in result.solutions {
            let parameters = ensemble_parameters(solution.0);
            assert!((30..=90).contains(&parameters.window));
            assert!((0.0..=1.0).contains(&parameters.min_residual_ratio));
            assert!((0.05..=0.1).contains(&parameters.efficiency));
            assert!((3.0..=5.0).contains(&parameters.chebyshev_z));
            assert!((5.0..=18.0).contains(&parameters.chebyshev_k));
            assert!((0.6..=1.0).contains(&parameters.isolation_forest_min_score));
            assert!((1.2..=3.0).contains(&parameters.lof_min_score));
            assert_eq!(
                member_imbalance(ensemble_member_counts(&observations(&values), parameters)),
                0
            );
        }
    }

    #[test]
    fn labeled_ensemble_tuning_rejects_a_population_that_misses_every_label() {
        let series = observations(&vec![0.0; 100]);
        let error = tune_ensemble(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                seed: Some(42),
                labels: vec![EventLabel::new(
                    NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
                    crate::LabelScope::Country,
                )],
                ..TuningConfig::default()
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("no ensemble parameter set"));
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
        let series = observations(&vec![0.0; 100]);
        let error = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                seed: Some(42),
                labels: vec![EventLabel::new(
                    NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
                    crate::LabelScope::Country,
                )],
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
                labels: vec![EventLabel::new(
                    NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
                    crate::LabelScope::Country,
                )],
                ..TuningConfig::default()
            },
        )
        .unwrap();

        assert!(!result.solutions.is_empty());
        for solution in result.solutions {
            let parameters = chebyshev_parameters(solution.0);
            let detection = ChebyshevDetector::new(parameters).detect(&series);
            assert_eq!(
                label_coverage(
                    &[EventLabel::new(
                        NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
                        crate::LabelScope::Country,
                    )],
                    &detection.anomalies,
                    3,
                )
                .missed,
                0
            );
        }
    }

    #[test]
    fn global_labels_do_not_constrain_tuning() {
        let series = with_country(observations(&vec![10.0; 100]), "RU");
        let result = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![EventLabel::new(
                    NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
                    crate::LabelScope::Global,
                )],
                ..TuningConfig::default()
            },
        )
        .unwrap();

        assert_eq!(result.required_labels, 0);
        assert!(!result.solutions.is_empty());
    }

    #[test]
    fn country_labels_only_constrain_the_matching_country() {
        let series = with_country(observations(&vec![10.0; 100]), "RU");
        let mut label = EventLabel::new(
            NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            crate::LabelScope::Country,
        );
        label.country_code = Some("TR".to_owned());
        let result = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![label],
                ..TuningConfig::default()
            },
        )
        .unwrap();

        assert_eq!(result.required_labels, 0);
    }

    #[test]
    fn country_labels_outside_the_series_do_not_constrain_tuning() {
        let series = with_country(observations(&vec![10.0; 100]), "RU");
        let mut label = EventLabel::new(
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            crate::LabelScope::Country,
        );
        label.country_code = Some("RU".to_owned());
        let result = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![label],
                ..TuningConfig::default()
            },
        )
        .unwrap();

        assert_eq!(result.required_labels, 0);
    }

    #[test]
    fn labeled_tuning_rejects_mixed_country_series() {
        let mut series = observations(&vec![10.0; 100]);
        series[0].country = Some("RU".to_owned());
        series[1].country = Some("TR".to_owned());

        let error = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![EventLabel::new(
                    NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
                    crate::LabelScope::Country,
                )],
                ..TuningConfig::default()
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("single-country"));
    }

    #[test]
    fn labeled_tuning_requires_country_metadata_for_a_unified_label_file() {
        let series = observations(&vec![10.0; 100]);
        let mut label = EventLabel::new(
            NaiveDate::from_ymd_opt(2025, 2, 1).unwrap(),
            crate::LabelScope::Country,
        );
        label.country_code = Some("RU".to_owned());

        let error = tune_chebyshev(
            &series,
            TuningConfig {
                evaluations: 4,
                population_size: 4,
                parallel: false,
                labels: vec![label],
                ..TuningConfig::default()
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("identify its country"));
    }
}
