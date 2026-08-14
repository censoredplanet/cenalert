# CenAlert Rust Detector

This crate contains the incremental Rust reimplementation of CenAlert's
production anomaly-detection path.

Unless stated otherwise, run all commands below from the `rust-detector/`
directory.

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
- the production anomaly state machine and window feedback behavior;
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
  --input /path/to/dashboard-export.json \
  --parameters /path/to/parameters.json
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
  --input /path/to/dashboard-export.json \
  --output /tmp/pareto-front.json \
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

An optional unified label file can contain both country-specific and global
events. During tuning, only labels for the input series' country are required
constraints: a feasible parameter set must detect every one of them. Global
events remain available to the label-matching API, but do not constrain tuning
because a multi-country event is not necessarily visible in every country.
Country labels outside the input series' date range and proximity margin are
also ignored, so one multi-country label file can be reused with historical
snapshots.

Regular labels use `"match_mode": "proximity"` by default and match when an
anomaly start date is within three days before or after the label date. These
applicable country labels constrain tuning. Set `"match_mode": "exact"` for a
pseudo-label that attributes only an anomaly beginning on that exact date;
pseudo-labels never constrain tuning because a methodology change is not
necessarily detectable in every series:

```json
[
  {
    "date": "2022-02-24",
    "scope": "country",
    "country_code": "RU",
    "description": "Example country-specific event"
  },
  {
    "date": "2022-03-01",
    "scope": "global",
    "description": "Example multi-country event",
    "affected_countries": ["RU", "TR"],
    "metadata": {
      "source": "Example source"
    }
  },
  {
    "date": "2022-01-01",
    "scope": "country",
    "match_mode": "exact",
    "description": "Example series-specific methodology change"
  }
]
```

`country_code` identifies the country whose tuning is constrained.
`affected_countries` is optional and limits where a global event can be used
for attribution; an empty list means that its affected countries are not yet
known. Descriptions and string-valued metadata are optional. In the matching
API, a country event is preferred when it and a global event are equally close
to an anomaly. A country label without `country_code`, as in the exact-match
example, applies to any country series processed with that label file. This
makes series-specific pseudo-label dates configurable rather than hardcoded;
omit them when tuning data from another source. `--proximity-days` does not
affect exact-match labels, and exact-match labels are excluded from the
NSGA-II missed-label constraint.

```bash
cargo run --release -- \
  tune \
  --input /path/to/dashboard-export.json \
  --labels /path/to/labels.json \
  --proximity-days 3 \
  --output /tmp/feasible-pareto-front.json
```

The NSGA-II constraint value is the number of missed proximity-mode country
labels. Only feasible rank-one solutions with zero missed required labels are
written. Exact pseudo-labels and global labels do not contribute to this
constraint. Tuning returns an error if it finds no feasible parameter set.
Omitting `--labels` preserves the fully unsupervised workflow.

### Build a local label file

`scripts/build_labels.py` normalizes the four manually verified CSV sheets into
the unified JSON schema while retaining their additional fields as metadata. It
can also reproduce the paper-style automatic verification step:
country events from `cenalert-paper/events/custom/Community.csv` are retained
only when their start date is within the configured proximity of an anomaly
detected with the paper series and selected parameters.

From `rust-detector/`, build the detector and run:

```bash
cargo build --release

python3 scripts/build_labels.py \
  --filtering ../CenAlert_Filtering.csv \
  --shutdowns ../CenAlert_Shutdowns.csv \
  --other ../CenAlert_Other.csv \
  --global-events ../CenAlert_Global.csv \
  --community-events ../../cenalert-paper/events/custom/Community.csv \
  --series-directory ../../cenalert-paper/series/012t0g \
  --parameters-directory ../../cenalert-paper/parameters/chebyshev_selected \
  --detector target/release/cenalert-detector \
  --proximity-days 3 \
  --output /tmp/cenalert-labels.json
```

The CSV inputs and JSON output can remain outside the repository. The builder
validates required columns and dates, preserves distinct same-day records, and
reports such groups for manual review.

The manually verified sheets are included as provided; the builder does not
revalidate those events against the selected time series. Before constrained
tuning, confirm that the labels were verified against the same Google Trends
keyword or topic as the input series. A mismatched series can make the
requirement to preserve every applicable country label infeasible.

## Select parameters

The `select` subcommand chooses a preferred count-versus-impact tradeoff from
the Pareto front. It follows the Python workflow by trying the same five
convex, decreasing curve families, falling back to a seventh-degree polynomial,
and locating the curve's knee. An endpoint returned by Kneedle is not treated
as a meaningful knee. If no interior knee exists, selection explicitly reports
that condition and uses the leftmost solution as a fallback.

```bash
cargo run --release -- \
  select \
  --input /path/to/pareto-front.json \
  --output /tmp/selected-parameters.json
```

For labeled-selection experiments, `--left-position N` selects the one-based
`N`th unique solution from the low-anomaly end of the front instead of fitting
and selecting its knee. For example:

```bash
cargo run --release -- \
  select \
  --input /path/to/pareto-front.json \
  --output /tmp/left-3-parameters.json \
  --left-position 3
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
python3 -m py_compile scripts/build_labels.py
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
  --input /path/to/RU-dashboard-export.json \
  --parameters tests/fixtures/parameters/RU.json \
  --rust-binary target/release/cenalert-detector

.venv/bin/python scripts/compare_outputs.py \
  --input /path/to/SS-dashboard-export.json \
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
  --input /path/to/RU-dashboard-export.json \
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
  /path/to/RU-dashboard-export.json \
  tests/fixtures/parameters/RU.json \
  500

cargo run --release --example in_process_benchmark -- \
  /path/to/RU-dashboard-export.json \
  /path/to/parameter-json-directory \
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
73 paper parameter sets, completed in 9.00 seconds. This benchmark measures the
detection/evaluation core rather than the NSGA-II optimizer itself.
