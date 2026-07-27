use std::env;
use std::error::Error;
use std::fs::File;
use std::io;
use std::path::PathBuf;

use cenalert_detector::{
    parse_dashboard_json, parse_parameters_json, Anomaly, ChebyshevDetector, Detector,
    DetectorParameters,
};
use serde::Serialize;

#[derive(Debug)]
struct Arguments {
    input: PathBuf,
    parameters: PathBuf,
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
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments(env::args().skip(1))?;
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

fn parse_arguments(arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let mut input = None;
    let mut parameters = None;
    let mut arguments = arguments;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--input" => input = Some(required_value("--input", &mut arguments)?),
            "--parameters" => parameters = Some(required_value("--parameters", &mut arguments)?),
            "--help" | "-h" => return Err(usage()),
            unknown => return Err(format!("unknown argument: {unknown}\n\n{}", usage())),
        }
    }

    Ok(Arguments {
        input: input
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing --input\n\n{}", usage()))?,
        parameters: parameters
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing --parameters\n\n{}", usage()))?,
    })
}

fn required_value(
    option: &str,
    arguments: &mut impl Iterator<Item = String>,
) -> Result<String, String> {
    arguments
        .next()
        .filter(|value| !value.starts_with("--"))
        .ok_or_else(|| format!("missing value for {option}"))
}

fn usage() -> String {
    "usage: cenalert-detector --input <dashboard.json> --parameters <parameters.json>".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_arguments() {
        let arguments = vec![
            "--input".to_owned(),
            "series.json".to_owned(),
            "--parameters".to_owned(),
            "RU.json".to_owned(),
        ];

        let parsed = parse_arguments(arguments.into_iter()).unwrap();

        assert_eq!(parsed.input, PathBuf::from("series.json"));
        assert_eq!(parsed.parameters, PathBuf::from("RU.json"));
    }

    #[test]
    fn rejects_missing_arguments() {
        let error = parse_arguments(Vec::new().into_iter()).unwrap_err();
        assert!(error.contains("missing --input"));
    }
}
