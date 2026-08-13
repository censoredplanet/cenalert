use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::PathBuf;

use cenalert_detector::{
    parse_dashboard_json, parse_labels_json, parse_parameters_json, select_parameters,
    tune_chebyshev, Anomaly, ChebyshevDetector, Detector, DetectorParameters, ParetoSolution,
    TuningConfig,
};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

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
    /// Tune Chebyshev detector parameters with NSGA-II.
    Tune(TuneArgs),
    /// Select the preferred tradeoff from an NSGA-II Pareto front.
    Select(SelectArgs),
}

#[derive(Debug, Args)]
struct DetectArgs {
    /// Dashboard-exported time-series JSON.
    #[arg(long)]
    input: PathBuf,
    /// Five-element detector parameter JSON.
    #[arg(long)]
    parameters: PathBuf,
}

#[derive(Debug, Args)]
struct TuneArgs {
    /// Dashboard-exported time-series JSON.
    #[arg(long)]
    input: PathBuf,
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
    /// Optional event labels that every feasible parameter set must detect.
    #[arg(long)]
    labels: Option<PathBuf>,
    /// Maximum distance for proximity labels; exact labels always require the same date.
    #[arg(long, requires = "labels")]
    proximity_days: Option<u32>,
    /// Lower bound of the Chebyshev K search range.
    #[arg(long, default_value_t = 5.0)]
    k_min: f64,
    /// Upper bound of the Chebyshev K search range.
    #[arg(long, default_value_t = 18.0)]
    k_max: f64,
}

#[derive(Debug, Args)]
struct SelectArgs {
    /// Pareto-front JSON produced by the tune subcommand.
    #[arg(long)]
    input: PathBuf,
    /// Destination for the selected five-element parameter JSON.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct DetectorOutput {
    algorithm: &'static str,
    observation_count: usize,
    first_date: String,
    last_date: String,
    countries: Vec<String>,
    topics: Vec<String>,
    parameters: DetectorParameters,
    scale_factor: f64,
    anomalies: Vec<Anomaly>,
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
    }
}

fn run_selection(arguments: SelectArgs) -> Result<(), Box<dyn Error>> {
    let front: Vec<ParetoSolution> = serde_json::from_reader(File::open(arguments.input)?)?;
    let selected = select_parameters(&front)?;
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(arguments.output)?),
        &selected.parameters,
    )?;
    eprintln!(
        "selected {}-anomaly solution at knee {} using {:?}",
        selected.objective[0], selected.knee, selected.method
    );
    Ok(())
}

fn run_detection(arguments: DetectArgs) -> Result<(), Box<dyn Error>> {
    let observations = parse_dashboard_json(File::open(arguments.input)?)?;
    let parameters = parse_parameters_json(File::open(arguments.parameters)?)?;

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

    let result = ChebyshevDetector::new(parameters).detect(&observations);
    let output = DetectorOutput {
        algorithm: "chebyshev",
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
    let result = tune_chebyshev(
        &observations,
        TuningConfig {
            evaluations: arguments.evaluations,
            population_size: arguments.population_size,
            parallel: !arguments.sequential,
            seed: arguments.seed,
            k_min: arguments.k_min,
            k_max: arguments.k_max,
            labels,
            proximity_days: arguments.proximity_days.unwrap_or(3),
        },
    )?;
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(arguments.output)?),
        &result.solutions,
    )?;
    if provided_label_count == 0 {
        eprintln!(
            "completed {} evaluations; wrote {} Pareto solutions",
            result.evaluations,
            result.solutions.len()
        );
    } else if result.required_labels == 0 {
        eprintln!(
            "completed {} evaluations without applicable country-label constraints; wrote {} Pareto solutions",
            result.evaluations,
            result.solutions.len()
        );
    } else {
        eprintln!(
            "completed {} evaluations; wrote {} feasible Pareto solutions preserving all {} labels",
            result.evaluations,
            result.solutions.len(),
            result.required_labels
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

        assert!(matches!(parsed.command, Command::Detect(_)));
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
        assert_eq!(arguments.population_size, 100);
        assert!(!arguments.sequential);
        assert_eq!(arguments.k_min, 5.0);
        assert_eq!(arguments.k_max, 18.0);
        assert!(arguments.labels.is_none());
        assert!(arguments.proximity_days.is_none());
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
}
