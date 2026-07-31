# CenAlert Rust Detector

This crate contains the incremental Rust reimplementation of CenAlert's
production anomaly-detection path.

The current implementation includes:

- Dashboard JSON parsing (`rate` or `value`);
- `YYYY-MM-DD` and RFC 3339 date parsing;
- chronological sorting;
- duplicate-date and numeric validation;
- paper parameter JSON parsing;
- the non-zero sliding window and demand categorization;
- NumPy-compatible population statistics and upper winsorization;
- SciPy-compatible Shapiro-Wilk normality decisions;
- Chebyshev/Z-score calculation;
- the fixed-alpha Croston SBA forecast used by StatsForecast 2.0.1;
- a common `Detector` trait for Chebyshev, future standalone detectors, and
  ensemble detectors;
- the production anomaly state machine and window feedback behavior; and
- anomaly interval and metadata output;
- parallel NSGA-II parameter tuning;
- optional event-label constraints for parameter tuning; and
- Pareto-front knee selection compatible with the paper workflow.

## Development environment

Install Rust with [rustup](https://rustup.rs/). This crate pins its toolchain in
`rust-toolchain.toml`, including the `rustfmt` and `clippy` components.

For a new shell installed with `rustup --no-modify-path`, load Cargo with:

```bash
source "$HOME/.cargo/env"
```

## Run detection

```bash
cargo run --release -- \
  detect \
  --input <dashboard-export.json> \
  --parameters <parameters.json>
```

The command prints JSON containing the normalized input summary, detector
parameters, scale factor, and detected anomaly intervals. Each anomaly contains
its start, end, peak, score, residual, and impact.

## Tune parameters

The `tune` subcommand reproduces the paper's Chebyshev NSGA-II search: window
size, Z-score threshold, Chebyshev threshold, minimum residual, and efficiency
are optimized against anomaly count and negative total impact. Candidate
detector evaluations run in parallel through Rayon by default.

```bash
cargo run --release -- \
  tune \
  --input <dashboard-export.json> \
  --output <pareto-front.json> \
  --evaluations 5000 \
  --population-size 100 \
  --seed 42
```

Use `--threads <count>` to set the Rayon worker count or `--sequential` for a
single-threaded comparison. The output uses the paper-compatible
`[[parameters], [objectives]]` JSON representation.

The Chebyshev K search range defaults to the paper's `5–18` range and can be
changed for experiments with `--k-min` and `--k-max`.

### Preserve labeled events

An optional label file constrains tuning to parameter sets that detect every
listed event. A label is matched when an anomaly start date is within three
days before or after its date by default:

```json
[
  {
    "date": "2022-02-24",
    "scope": "country",
    "description": "Example country-specific event"
  },
  {
    "date": "2022-03-01",
    "scope": "global",
    "description": "Example multi-country event"
  }
]
```

The `scope` field preserves the distinction between country-specific and global
events; both kinds are required constraints. Descriptions are optional.

```bash
cargo run --release -- \
  tune \
  --input <dashboard-export.json> \
  --labels <labels.json> \
  --proximity-days 3 \
  --output <feasible-pareto-front.json>
```

The NSGA-II constraint value is the number of missed labels. Only feasible
rank-one solutions with zero missed labels are written. Tuning returns an error
if it finds no feasible parameter set. Omitting `--labels` preserves the fully
unsupervised workflow.

## Select parameters

The `select` subcommand chooses a preferred count-versus-impact tradeoff from
the Pareto front. It follows the Python workflow by trying the same five
convex, decreasing curve families, falling back to a seventh-degree polynomial,
and locating the curve's knee.

```bash
cargo run --release -- \
  select \
  --input <pareto-front.json> \
  --output <selected-parameters.json>
```

The output is a five-element parameter JSON accepted directly by `detect`.
Together, the three subcommands provide this local workflow:

```text
Dashboard JSON -> tune -> Pareto front -> select -> parameters -> detect
```

## Tests

```bash
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## Validation

Validation uses public **All time** JSON exports from the CenAlert Dashboard and
the paper-era selected parameters copied into `tests/fixtures/parameters`.
Dashboard exports are recorded by checksum in
`tests/fixtures/dashboard-manifest.json`, rather than committed as large,
daily-changing snapshots.

The recorded reference environment uses:

- Dashboard exports downloaded on 2026-07-22;
- the repository's `src/lib/detection.py` as the Python reference;
- the Python dependency versions pinned by the repository;
- Rust 1.97.1; and
- Apple Silicon with macOS 26.5.2 for the runtime benchmark.

### Python parity

Create a local virtual environment, install the detector-only validation
packages, and build the optimized Rust binary:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-validation.txt
cargo build --release
```

Release builds use `opt-level = 3`, fat link-time optimization, and one codegen
unit. For a machine-specific local benchmark, native CPU instructions can be
enabled explicitly:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The native build may not run on older or different processors, so it is not the
portable default.

The comparison requires equal anomaly counts and exact start, end, and peak
dates. Score, residual, and impact must match with relative and absolute
tolerances of `1e-8`.

| Series | Observations | Zero values | Python anomalies | Rust anomalies | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Russia (`RU`) | 5,681 | 18 (0.32%) | 25 | 25 | Match |
| South Sudan (`SS`) | 5,682 | 5,021 (88.37%) | 3 | 3 | Match |

Run the comparisons from `rust-detector/`:

```bash
.venv/bin/python scripts/compare_outputs.py \
  --input <RU-dashboard-export.json> \
  --parameters tests/fixtures/parameters/RU.json \
  --rust-binary target/release/cenalert-detector

.venv/bin/python scripts/compare_outputs.py \
  --input <SS-dashboard-export.json> \
  --parameters tests/fixtures/parameters/SS.json \
  --rust-binary target/release/cenalert-detector
```

The recorded exports report `matched 25 anomalies` for Russia and
`matched 3 anomalies` for South Sudan.

### Runtime benchmark

`scripts/benchmark.py` performs one warm-up and then reports three steady-state
runs. It includes JSON input parsing and detector execution for both versions;
the Rust measurement also includes process startup and JSON output.

```bash
.venv/bin/python scripts/benchmark.py \
  --input <RU-dashboard-export.json> \
  --parameters tests/fixtures/parameters/RU.json \
  --rust-binary target/release/cenalert-detector \
  --iterations 3
```

Russia All Time results:

| Implementation | Median runtime |
| --- | ---: |
| Python | 3.779 seconds |
| Initial Rust release | 0.421 seconds |
| Speedup | **8.97×** |

This original measurement established an end-to-end CLI improvement, but it
overstates the cost relevant to parameter tuning because each Rust sample
starts a process, parses the input, and serializes output.

`examples/in_process_benchmark.rs` measures the detector in one process after
loading the series and parameter sets once. It accepts either one parameter
JSON file or a directory of parameter JSON files:

```bash
cargo run --release --example in_process_benchmark -- \
  <RU-dashboard-export.json> \
  tests/fixtures/parameters/RU.json \
  500

cargo run --release --example in_process_benchmark -- \
  <RU-dashboard-export.json> \
  <parameter-json-directory> \
  5000
```

Profiling identified repeated Shapiro-Wilk window sorting as the largest named
hotspot. The optimized window maintains sorted values, caches unchanged
moments, and computes Croston SBA directly from demand arrivals. The outputs
remain byte-for-byte equal to the initial Rust implementation for the recorded
Russia and South Sudan exports.

| In-process benchmark | Initial Rust | Optimized Rust | Improvement |
| --- | ---: | ---: | ---: |
| Russia, one parameter set | 6.83 ms/evaluation | 2.44 ms/evaluation | 2.80× |
| South Sudan, one parameter set | 0.76 ms/evaluation | 0.25 ms/evaluation | 3.10× |

A single-threaded 5,000-evaluation sweep over the Russia series, cycling through
73 paper parameter sets, completed in 9.00 seconds. This benchmark exercises
the detection/evaluation core only; it does not add or prescribe Aaron's future
parameter-tuning workflow.
