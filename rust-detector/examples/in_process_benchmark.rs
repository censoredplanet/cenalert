use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use cenalert_detector::{parse_dashboard_json, parse_parameters_json, ChebyshevDetector, Detector};
use serde::Serialize;

#[derive(Serialize)]
struct BenchmarkResult {
    evaluations: usize,
    parameter_sets: usize,
    total_seconds: f64,
    mean_milliseconds: f64,
    evaluations_per_second: f64,
    anomaly_count: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let (input, parameter_path, evaluations) = parse_arguments()?;
    let observations = parse_dashboard_json(File::open(input)?)?;
    let parameters = load_parameters(parameter_path)?;

    black_box(ChebyshevDetector::new(parameters[0]).detect(black_box(&observations)));

    let started = Instant::now();
    let mut anomaly_count = 0;
    for index in 0..evaluations {
        let detector = ChebyshevDetector::new(parameters[index % parameters.len()]);
        let result = detector.detect(black_box(&observations));
        anomaly_count = black_box(result.anomalies.len());
    }
    let elapsed = started.elapsed().as_secs_f64();

    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &BenchmarkResult {
            evaluations,
            parameter_sets: parameters.len(),
            total_seconds: elapsed,
            mean_milliseconds: elapsed * 1_000.0 / evaluations as f64,
            evaluations_per_second: evaluations as f64 / elapsed,
            anomaly_count,
        },
    )?;
    println!();
    Ok(())
}

fn load_parameters(
    path: PathBuf,
) -> Result<Vec<cenalert_detector::DetectorParameters>, Box<dyn Error>> {
    if path.is_file() {
        return Ok(vec![parse_parameters_json(File::open(path)?)?]);
    }

    let mut paths = fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    paths.sort_unstable();
    if paths.is_empty() {
        return Err("parameter directory is empty".into());
    }
    paths
        .into_iter()
        .map(|path| {
            parse_parameters_json(File::open(path)?)
                .map_err(|error| -> Box<dyn Error> { Box::new(error) })
        })
        .collect()
}

fn parse_arguments() -> Result<(PathBuf, PathBuf, usize), String> {
    let mut arguments = env::args().skip(1);
    let input = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("missing dashboard JSON path")?;
    let parameters = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("missing parameter JSON file or directory")?;
    let evaluations = arguments
        .next()
        .map(|value| value.parse())
        .transpose()
        .map_err(|_| "evaluations must be a positive integer")?
        .unwrap_or(100);
    if evaluations == 0 {
        return Err("evaluations must be a positive integer".to_owned());
    }
    Ok((input, parameters, evaluations))
}
