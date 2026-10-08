use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::PathBuf;

use cenalert_detector::{
    apply_anomaly_filters, coalesce_sparse_anomalies, group_same_day_anomalies,
    parse_dashboard_json, parse_ensemble_parameters_json, parse_labels_json,
    parse_outlier_parameters_json, parse_parameters_json, select_ensemble_nth_from_left,
    select_ensemble_parameters, select_nth_from_left, select_parameters, tune_chebyshev,
    tune_ensemble, Anomaly, AnomalyFilter, ChebyshevDetector, CoalescingSummary, CountryAnomaly,
    CrossCountryGroupingResult, DetectionResult, Detector, EnsembleDetector,
    EnsembleParetoSolution, FilterContext, FilteringResult, IsolationForestDetector,
    LocalOutlierFactorDetector, MinimumLabeledImpactFilter, ParetoSolution, TuningConfig,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(version, about = "CenAlert anomaly detection and parameter tuning")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Detect anomalies using a selected parameter set.
    Detect(DetectArgs),
    /// Tune detector parameters with NSGA-II.
    Tune(TuneArgs),
    /// Select the preferred tradeoff from an NSGA-II Pareto front.
    Select(SelectArgs),
    /// Filter detector output into notification candidates.
    Filter(FilterArgs),
    /// Analyze same-day candidates and group broad cross-country occurrences.
    GroupCrossCountry(GroupCrossCountryArgs),
}

#[derive(Debug, Args)]
struct DetectArgs {
    /// Include daily values and detector diagnostics in the JSON output.
    #[arg(long)]
    include_annotated: bool,
    /// Freeze Croston and output scaling to the maximum through this training date.
    #[arg(long)]
    reference_end: Option<chrono::NaiveDate>,
    /// Dashboard-exported time-series JSON.
    #[arg(long)]
    input: PathBuf,
    /// Algorithm-specific detector parameter JSON.
    #[arg(long)]
    parameters: PathBuf,
    /// Detection algorithm whose parameter JSON is supplied.
    #[arg(long, value_enum, default_value_t = DetectionAlgorithm::Chebyshev)]
    algorithm: DetectionAlgorithm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DetectionAlgorithm {
    Chebyshev,
    IsolationForest,
    Lof,
    Ensemble,
}

#[derive(Debug, Args)]
struct TuneArgs {
    /// Dashboard-exported time-series JSON.
    #[arg(long)]
    input: PathBuf,
    /// Detector to tune.
    #[arg(long, value_enum, default_value_t = TuningAlgorithm::Chebyshev)]
    algorithm: TuningAlgorithm,
    /// Destination for the Pareto-front JSON.
    #[arg(long)]
    output: PathBuf,
    /// Maximum number of detector evaluations.
    #[arg(long, default_value_t = 5_000)]
    evaluations: u32,
    /// NSGA-II population size; must be even.
    #[arg(long, default_value_t = 100)]
    population_size: usize,
    /// Random seed for the evolutionary operators.
    #[arg(long)]
    seed: Option<u64>,
    /// Number of Rayon worker threads.
    #[arg(long, conflicts_with = "sequential")]
    threads: Option<usize>,
    /// Evaluate each NSGA-II population sequentially.
    #[arg(long)]
    sequential: bool,
    /// Optional labels; applicable proximity labels constrain feasible parameter sets.
    #[arg(long)]
    labels: Option<PathBuf>,
    /// Maximum distance for constraining proximity labels and label attribution.
    #[arg(long, requires = "labels")]
    proximity_days: Option<u32>,
    /// Lower bound of the Chebyshev K search range.
    #[arg(long, default_value_t = 5.0)]
    k_min: f64,
    /// Upper bound of the Chebyshev K search range.
    #[arg(long, default_value_t = 18.0)]
    k_max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TuningAlgorithm {
    Chebyshev,
    Ensemble,
}

#[derive(Debug, Args)]
struct SelectArgs {
    /// Pareto-front JSON produced by the tune subcommand.
    #[arg(long)]
    input: PathBuf,
    /// Detector whose Pareto front is supplied.
    #[arg(long, value_enum, default_value_t = TuningAlgorithm::Chebyshev)]
    algorithm: TuningAlgorithm,
    /// Destination for the selected algorithm-specific parameter JSON.
    #[arg(long)]
    output: PathBuf,
    /// Select the one-based nth unique solution from the low-anomaly end instead of the knee.
    #[arg(long)]
    left_position: Option<usize>,
}

#[derive(Debug, Args)]
struct FilterArgs {
    /// Detection JSON produced by the detect subcommand.
    #[arg(long)]
    input: PathBuf,
    /// Labels used to derive the per-country impact threshold.
    #[arg(long)]
    labels: PathBuf,
    /// Destination for retained and discarded anomalies.
    #[arg(long)]
    output: PathBuf,
    /// Maximum distance for label attribution.
    #[arg(long, default_value_t = 3)]
    proximity_days: u32,
    /// Merge sparse anomalies separated by at most this many normal days.
    #[arg(long)]
    coalesce_sparse_gap_days: Option<u32>,
}

#[derive(Debug, Args)]
struct GroupCrossCountryArgs {
    /// JSON array of country-tagged notification candidates.
    #[arg(long)]
    input: PathBuf,
    /// Destination for the cross-country analysis and grouped notifications.
    #[arg(long)]
    output: PathBuf,
    /// Distinct countries required to create one grouped notification.
    #[arg(long, default_value_t = 5)]
    minimum_countries: usize,
}

#[derive(Debug, Serialize)]
struct DetectorOutput<P> {
    algorithm: &'static str,
    observation_count: usize,
    first_date: String,
    last_date: String,
    countries: Vec<String>,
    topics: Vec<String>,
    parameters: P,
    scale_factor: f64,
    anomalies: Vec<Anomaly>,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotated: Option<Vec<cenalert_detector::AnnotatedObservation>>,
}

#[derive(Debug, Deserialize)]
struct FilterInput {
    countries: Vec<String>,
    anomalies: Vec<Anomaly>,
}

#[derive(Debug, Serialize)]
struct FilterOutput {
    country: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    coalescing: Option<CoalescingSummary>,
    #[serde(flatten)]
    result: FilteringResult,
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        Command::Detect(arguments) => run_detection(arguments),
        Command::Tune(arguments) => run_tuning(arguments),
        Command::Select(arguments) => run_selection(arguments),
        Command::Filter(arguments) => run_filter(arguments),
        Command::GroupCrossCountry(arguments) => run_cross_country_grouping(arguments),
    }
}

fn run_cross_country_grouping(arguments: GroupCrossCountryArgs) -> Result<(), Box<dyn Error>> {
    let anomalies: Vec<CountryAnomaly> = serde_json::from_reader(File::open(arguments.input)?)?;
    let result: CrossCountryGroupingResult =
        group_same_day_anomalies(anomalies, arguments.minimum_countries)?;
    serde_json::to_writer_pretty(BufWriter::new(File::create(arguments.output)?), &result)?;
    Ok(())
}

fn run_filter(arguments: FilterArgs) -> Result<(), Box<dyn Error>> {
    let input: FilterInput = serde_json::from_reader(File::open(arguments.input)?)?;
    let [country] = input.countries.as_slice() else {
        return Err("filter input must contain exactly one country".into());
    };
    let labels = parse_labels_json(File::open(arguments.labels)?)?;
    let context = FilterContext {
        labels: &labels,
        country_code: country,
        proximity_days: arguments.proximity_days,
    };
    let (anomalies, coalescing) = if let Some(maximum_gap_days) = arguments.coalesce_sparse_gap_days
    {
        let result = coalesce_sparse_anomalies(input.anomalies, maximum_gap_days)?;
        (result.anomalies, Some(result.summary))
    } else {
        (input.anomalies, None)
    };
    let filters: Vec<Box<dyn AnomalyFilter>> = vec![Box::new(MinimumLabeledImpactFilter)];
    let output = FilterOutput {
        country: country.clone(),
        coalescing,
        result: apply_anomaly_filters(anomalies, &context, &filters),
    };
    serde_json::to_writer_pretty(BufWriter::new(File::create(arguments.output)?), &output)?;
    Ok(())
}

fn run_selection(arguments: SelectArgs) -> Result<(), Box<dyn Error>> {
    let message = match arguments.algorithm {
        TuningAlgorithm::Chebyshev => {
            let front: Vec<ParetoSolution> = serde_json::from_reader(File::open(arguments.input)?)?;
            let output = BufWriter::new(File::create(arguments.output)?);
            if let Some(position) = arguments.left_position {
                let selected = select_nth_from_left(&front, position)?;
                serde_json::to_writer_pretty(
                    output,
                    &cenalert_detector::chebyshev_parameters(selected.0),
                )?;
                left_selection_message(position, selected.1)
            } else {
                let selected = select_parameters(&front)?;
                serde_json::to_writer_pretty(
                    output,
                    &cenalert_detector::chebyshev_parameters(selected.parameters),
                )?;
                knee_selection_message(selected.objective, selected.knee, selected.method)
            }
        }
        TuningAlgorithm::Ensemble => {
            let front: Vec<EnsembleParetoSolution> =
                serde_json::from_reader(File::open(arguments.input)?)?;
            let output = BufWriter::new(File::create(arguments.output)?);
            if let Some(position) = arguments.left_position {
                let selected = select_ensemble_nth_from_left(&front, position)?;
                serde_json::to_writer_pretty(
                    output,
                    &cenalert_detector::ensemble_parameters(selected.0),
                )?;
                left_selection_message(position, selected.1)
            } else {
                let selected = select_ensemble_parameters(&front)?;
                serde_json::to_writer_pretty(output, &selected.parameters)?;
                knee_selection_message(selected.objective, selected.knee, selected.method)
            }
        }
    };
    eprintln!("{message}");
    Ok(())
}

fn left_selection_message(position: usize, objective: [f64; 2]) -> String {
    format!(
        "selected left-side position {position} with {} anomalies",
        objective[0]
    )
}

fn knee_selection_message(
    objective: [f64; 2],
    knee: Option<f64>,
    method: cenalert_detector::SelectionMethod,
) -> String {
    if let Some(knee) = knee {
        format!(
            "selected {}-anomaly solution at knee {knee} using {method:?}",
            objective[0]
        )
    } else {
        format!(
            "no knee found; selected leftmost fallback with {} anomalies",
            objective[0]
        )
    }
}

fn run_detection(arguments: DetectArgs) -> Result<(), Box<dyn Error>> {
    let observations = parse_dashboard_json(File::open(arguments.input)?)?;
    let reference_maximum = if let Some(end) = arguments.reference_end {
        if !matches!(
            arguments.algorithm,
            DetectionAlgorithm::Chebyshev | DetectionAlgorithm::Ensemble
        ) {
            return Err("--reference-end supports chebyshev and ensemble".into());
        }
        let maximum = observations
            .iter()
            .filter(|point| point.date <= end)
            .map(|point| point.value)
            .reduce(f64::max)
            .ok_or("no observations on or before --reference-end")?;
        if maximum <= 0.0 {
            return Err("training reference maximum must be positive".into());
        }
        Some(maximum)
    } else {
        None
    };
    let parameter_file = File::open(arguments.parameters)?;

    match arguments.algorithm {
        DetectionAlgorithm::Chebyshev => {
            let parameters = parse_parameters_json(parameter_file)?;
            let mut detector = ChebyshevDetector::new(parameters);
            if let Some(maximum) = reference_maximum {
                detector = detector.with_reference_maximum(maximum);
            }
            let result = detector.detect(&observations);
            write_detection_output(
                "chebyshev",
                &observations,
                parameters,
                result,
                arguments.include_annotated,
            )
        }
        DetectionAlgorithm::Lof => {
            let parameters = parse_outlier_parameters_json(parameter_file)?;
            let result = LocalOutlierFactorDetector::new(parameters).detect(&observations);
            write_detection_output(
                "lof",
                &observations,
                parameters,
                result,
                arguments.include_annotated,
            )
        }
        DetectionAlgorithm::IsolationForest => {
            let parameters = parse_outlier_parameters_json(parameter_file)?;
            let result = IsolationForestDetector::new(parameters).detect(&observations);
            write_detection_output(
                "isolation-forest",
                &observations,
                parameters,
                result,
                arguments.include_annotated,
            )
        }
        DetectionAlgorithm::Ensemble => {
            let parameters = parse_ensemble_parameters_json(parameter_file)?;
            let mut detector = EnsembleDetector::new(parameters);
            if let Some(maximum) = reference_maximum {
                detector = detector.with_reference_maximum(maximum);
            }
            let result = detector.detect(&observations);
            write_detection_output(
                "ensemble",
                &observations,
                parameters,
                result,
                arguments.include_annotated,
            )
        }
    }
}

fn write_detection_output<P: Serialize>(
    algorithm: &'static str,
    observations: &[cenalert_detector::Observation],
    parameters: P,
    result: DetectionResult,
    include_annotated: bool,
) -> Result<(), Box<dyn Error>> {
    let mut countries: Vec<String> = observations
        .iter()
        .filter_map(|observation| observation.country.clone())
        .collect();
    countries.sort_unstable();
    countries.dedup();

    let mut topics: Vec<String> = observations
        .iter()
        .filter_map(|observation| observation.topic.clone())
        .collect();
    topics.sort_unstable();
    topics.dedup();

    let output = DetectorOutput {
        algorithm,
        observation_count: observations.len(),
        first_date: observations
            .first()
            .expect("parser rejects empty series")
            .date
            .to_string(),
        last_date: observations
            .last()
            .expect("parser rejects empty series")
            .date
            .to_string(),
        countries,
        topics,
        parameters,
        scale_factor: result.scale_factor,
        anomalies: result.anomalies,
        annotated: include_annotated.then_some(result.annotated),
    };

    serde_json::to_writer_pretty(io::stdout().lock(), &output)?;
    println!();
    Ok(())
}

fn run_tuning(arguments: TuneArgs) -> Result<(), Box<dyn Error>> {
    if arguments.threads == Some(0) {
        return Err("threads must be at least 1".into());
    }
    if let Some(threads) = arguments.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    let observations = parse_dashboard_json(File::open(arguments.input)?)?;
    let labels = arguments
        .labels
        .map(|path| parse_labels_json(File::open(path)?))
        .transpose()?
        .unwrap_or_default();
    let provided_label_count = labels.len();
    let config = TuningConfig {
        evaluations: arguments.evaluations,
        population_size: arguments.population_size,
        parallel: !arguments.sequential,
        seed: arguments.seed,
        k_min: arguments.k_min,
        k_max: arguments.k_max,
        labels,
        proximity_days: arguments.proximity_days.unwrap_or(3),
    };
    let (evaluations, solution_count, required_labels) = match arguments.algorithm {
        TuningAlgorithm::Chebyshev => {
            let result = tune_chebyshev(&observations, config)?;
            serde_json::to_writer_pretty(
                BufWriter::new(File::create(arguments.output)?),
                &result.solutions,
            )?;
            (
                result.evaluations,
                result.solutions.len(),
                result.required_labels,
            )
        }
        TuningAlgorithm::Ensemble => {
            let result = tune_ensemble(&observations, config)?;
            serde_json::to_writer_pretty(
                BufWriter::new(File::create(arguments.output)?),
                &result.solutions,
            )?;
            (
                result.evaluations,
                result.solutions.len(),
                result.required_labels,
            )
        }
    };
    if provided_label_count == 0 {
        eprintln!(
            "completed {} evaluations; wrote {} Pareto solutions",
            evaluations, solution_count
        );
    } else if required_labels == 0 {
        eprintln!(
            "completed {} evaluations without applicable country-label constraints; wrote {} Pareto solutions",
            evaluations, solution_count
        );
    } else {
        eprintln!(
            "completed {} evaluations; wrote {} feasible Pareto solutions preserving all {} labels",
            evaluations, solution_count, required_labels
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn clap_configuration_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_detection_subcommand() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "detect",
            "--input",
            "series.json",
            "--parameters",
            "RU.json",
        ])
        .unwrap();

        let Command::Detect(arguments) = parsed.command else {
            panic!("expected detect command");
        };
        assert!(matches!(arguments.algorithm, DetectionAlgorithm::Chebyshev));
    }

    #[test]
    fn parses_filter_subcommand() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "filter",
            "--input",
            "detected.json",
            "--labels",
            "labels.json",
            "--output",
            "candidates.json",
        ])
        .unwrap();

        let Command::Filter(arguments) = parsed.command else {
            panic!("expected filter subcommand")
        };
        assert_eq!(arguments.proximity_days, 3);
        assert_eq!(arguments.coalesce_sparse_gap_days, None);
    }

    #[test]
    fn parses_cross_country_grouping_subcommand() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "group-cross-country",
            "--input",
            "candidates.json",
            "--output",
            "notifications.json",
            "--minimum-countries",
            "4",
        ])
        .unwrap();

        let Command::GroupCrossCountry(arguments) = parsed.command else {
            panic!("expected group-cross-country subcommand")
        };
        assert_eq!(arguments.minimum_countries, 4);
    }

    #[test]
    fn parses_lof_detection() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "detect",
            "--input",
            "series.json",
            "--parameters",
            "LOF.json",
            "--algorithm",
            "lof",
        ])
        .unwrap();
        let Command::Detect(arguments) = parsed.command else {
            panic!("expected detect command");
        };

        assert!(matches!(arguments.algorithm, DetectionAlgorithm::Lof));
    }

    #[test]
    fn parses_isolation_forest_and_ensemble_detection() {
        for (name, expected) in [
            ("isolation-forest", DetectionAlgorithm::IsolationForest),
            ("ensemble", DetectionAlgorithm::Ensemble),
        ] {
            let parsed = Cli::try_parse_from([
                "cenalert-detector",
                "detect",
                "--input",
                "series.json",
                "--parameters",
                "parameters.json",
                "--algorithm",
                name,
            ])
            .unwrap();
            let Command::Detect(arguments) = parsed.command else {
                panic!("expected detect command");
            };
            assert_eq!(arguments.algorithm, expected);
        }
    }

    #[test]
    fn parses_tuning_defaults() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "tune",
            "--input",
            "series.json",
            "--output",
            "front.json",
        ])
        .unwrap();
        let Command::Tune(arguments) = parsed.command else {
            panic!("expected tune command");
        };

        assert_eq!(arguments.evaluations, 5_000);
        assert_eq!(arguments.algorithm, TuningAlgorithm::Chebyshev);
        assert_eq!(arguments.population_size, 100);
        assert!(!arguments.sequential);
        assert_eq!(arguments.k_min, 5.0);
        assert_eq!(arguments.k_max, 18.0);
        assert!(arguments.labels.is_none());
        assert!(arguments.proximity_days.is_none());
    }

    #[test]
    fn parses_ensemble_tuning_and_selection() {
        let tuned = Cli::try_parse_from([
            "cenalert-detector",
            "tune",
            "--algorithm",
            "ensemble",
            "--input",
            "series.json",
            "--output",
            "front.json",
        ])
        .unwrap();
        let Command::Tune(tune_arguments) = tuned.command else {
            panic!("expected tune command");
        };
        assert_eq!(tune_arguments.algorithm, TuningAlgorithm::Ensemble);

        let selected = Cli::try_parse_from([
            "cenalert-detector",
            "select",
            "--algorithm",
            "ensemble",
            "--input",
            "front.json",
            "--output",
            "selected.json",
        ])
        .unwrap();
        let Command::Select(select_arguments) = selected.command else {
            panic!("expected select command");
        };
        assert_eq!(select_arguments.algorithm, TuningAlgorithm::Ensemble);
    }

    #[test]
    fn proximity_requires_labels() {
        assert!(Cli::try_parse_from([
            "cenalert-detector",
            "tune",
            "--input",
            "series.json",
            "--output",
            "front.json",
            "--proximity-days",
            "3",
        ])
        .is_err());
    }

    #[test]
    fn parses_selection_subcommand() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "select",
            "--input",
            "front.json",
            "--output",
            "selected.json",
        ])
        .unwrap();

        assert!(matches!(parsed.command, Command::Select(_)));
    }

    #[test]
    fn parses_left_side_selection() {
        let parsed = Cli::try_parse_from([
            "cenalert-detector",
            "select",
            "--input",
            "front.json",
            "--output",
            "selected.json",
            "--left-position",
            "3",
        ])
        .unwrap();
        let Command::Select(arguments) = parsed.command else {
            panic!("expected select command");
        };

        assert_eq!(arguments.left_position, Some(3));
    }
}
