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
- the production anomaly state machine and window feedback behavior; and
- anomaly interval and metadata output.

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
  --input <dashboard-export.json> \
  --parameters <parameters.json>
```

The command prints JSON containing the normalized input summary, detector
parameters, scale factor, and detected anomaly intervals. Each anomaly contains
its start, end, peak, score, residual, and impact.

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
| Rust release | 0.421 seconds |
| Speedup | **8.97×** |

This establishes a clear core-runtime improvement on one full historical
series. It does not claim that the entire parameter-tuning workflow is 8.97×
faster, because Phase 1 excludes tuning integration.
