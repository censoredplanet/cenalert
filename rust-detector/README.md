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
- an `isotree`-compatible one-dimensional Isolation Forest scorer;
- a scikit-learn-compatible one-dimensional Local Outlier Factor scorer;
- standalone Chebyshev, Isolation Forest, and Local Outlier Factor detectors
  behind a common `Detector` trait;
- a three-detector majority-vote ensemble with a shared window, Croston
  configuration, residual ratio, and efficiency threshold;
- per-member ensemble vote metadata for explaining two- and three-vote alerts;
- the production anomaly state machine and window feedback behavior;
- anomaly interval and metadata output;
- parallel NSGA-II parameter tuning;
- optional event-label constraints for parameter tuning;
- explicit cross-country label attribution without inherited tuning constraints;
- Pareto-front knee selection compatible with the paper workflow; and
- composable post-detection filtering for notification candidates.

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

Newly selected Chebyshev parameters use a named JSON object:

```json
{
  "window": 60,
  "z": 4,
  "k": 6,
  "min_residual_ratio": 0.05,
  "efficiency": 0.05
}
```

Local Outlier Factor accepts the same shared fields plus its own score
threshold:

```json
{
  "window": 60,
  "min_score": 1.5,
  "min_residual_ratio": 0.05,
  "efficiency": 0.05
}
```

```bash
cargo run --release -- \
  detect \
  --algorithm lof \
  --input /path/to/dashboard-export.json \
  --parameters /path/to/lof-parameters.json
```

Isolation Forest uses the same fields:

```bash
cargo run --release -- \
  detect \
  --algorithm isolation-forest \
  --input /path/to/dashboard-export.json \
  --parameters /path/to/isolation-forest-parameters.json
```

The deployment snapshot computes inverse-score thresholds for every outlier
detector point,
which passes reversed bounds to SciPy for ordinary below-mean observations and
raises an exception. Isolation Forest and LOF validation therefore follow the
runnable paper behavior: thresholds are material to anomaly output only when
an anomaly begins. The Rust implementation avoids that failure.

The ensemble always contains all three dense detectors, so an anomaly requires
at least two votes. Its JSON object keeps algorithm-specific score thresholds
separate while enforcing one shared window and one shared set of sparse-series
parameters:

```json
{
  "window": 60,
  "min_residual_ratio": 0.05,
  "efficiency": 0.05,
  "chebyshev_z": 4,
  "chebyshev_k": 6,
  "isolation_forest_min_score": 0.8,
  "lof_min_score": 1.5
}
```

```bash
cargo run --release -- \
  detect \
  --algorithm ensemble \
  --input /path/to/dashboard-export.json \
  --parameters /path/to/ensemble-parameters.json
```

The ensemble's reported threshold is the minimum of its three member
thresholds. Sparse observations use the single shared
Croston path rather than voting three times.

The Croston residual ratio is `(value - forecast) / time_series_maximum` and
must be in `[0, 1]`. This makes sparse detection invariant to multiplicative
rescaling and removes the need to update an absolute residual threshold when
Google Trends introduces a new maximum. Legacy paper arrays and the previous
ensemble object with `min_residual` are still accepted; their residual on the
normalized `[0, 100]` scale is divided by 100 when loaded.

The command prints JSON containing the normalized input summary, detector
parameters, scale factor, and detected anomaly intervals. Each anomaly contains
its start, end, peak, peak value, score, residual, impact, and whether it was
detected through the sparse Croston path.

Dense ensemble anomalies also contain `ensemble_vote`, which reports the
Chebyshev, Isolation Forest, and LOF scores, their minimum scores, each voting
decision, the total vote count, and the two-vote majority requirement. Sparse
Croston anomalies have no member-vote object because they use the shared sparse
path.

## Filter notification candidates

The `filter` subcommand applies composable post-detection filters. Its first
filter derives a per-country impact threshold from the lowest-impact anomaly
that matches an applicable label. It retains anomalies at or above that
threshold and records discarded anomalies with the filter decision. When none
of the country's detected anomalies matches a label, the threshold is
unavailable and all anomalies are retained.

```bash
cargo run --release -- \
  filter \
  --input /path/to/detection-output.json \
  --labels /path/to/labels.json \
  --proximity-days 3 \
  --coalesce-sparse-gap-days 1 \
  --output /path/to/notification-candidates.json
```

When `--coalesce-sparse-gap-days` is supplied, sparse anomalies separated by
no more than that number of normal days are combined before impact filtering.
The combined interval retains every component start for label matching, uses
the component with the largest observed value as its peak, and sums component
impact without assigning impact to the intervening normal days. Dense
anomalies remain separate. Detection outputs created before `peak_value` and
`sparse` were added should be regenerated before using this option.

## Analyze and group cross-country notifications

The `group-cross-country` subcommand finds notification candidates with the
same start date in at least two distinct countries. Dates meeting a configurable
country threshold become one grouped notification; candidates on other dates
remain individual. Each group retains the original country anomalies.

The input is a JSON array containing a `country` field and the anomaly fields
produced by `filter`:

```bash
cargo run --release -- \
  group-cross-country \
  --input /path/to/all-country-candidates.json \
  --minimum-countries 5 \
  --output /path/to/cross-country-notifications.json
```

The output reports every date shared by at least two countries, allowing
different grouping thresholds to be evaluated from one run.

## Tune parameters

The `tune` subcommand supports the paper's Chebyshev NSGA-II search and joint
three-detector ensemble tuning. Candidate detector evaluations run in parallel
through Rayon by default. Both searches minimize anomaly count and negative
total impact.

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

For the ensemble, add `--algorithm ensemble`. One seven-dimensional candidate
jointly tunes:

1. the window, residual ratio, and efficiency shared by all three members;
2. the Chebyshev Z and K thresholds;
3. the Isolation Forest and LOF minimum scores.

The ranges come from the paper's individual detector searches. Because
efficiency is shared, its ensemble range is the common `0.05–0.1` interval.
The residual-ratio range is `[0, 1]`.

```bash
cargo run --release -- \
  tune \
  --algorithm ensemble \
  --input /path/to/dashboard-export.json \
  --output /tmp/ensemble-pareto-front.json \
  --evaluations 5000 \
  --population-size 100 \
  --seed 42
```

Each ensemble Pareto solution has the same `[parameters, objectives]` shape as
the Chebyshev output. Its parameter order is `window`, `min_residual_ratio`,
`efficiency`, `chebyshev_z`, `chebyshev_k`,
`isolation_forest_min_score`, and `lof_min_score`.

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
  --algorithm ensemble \
  --input /path/to/dashboard-export.json \
  --labels /path/to/labels.json \
  --proximity-days 3 \
  --output /tmp/feasible-ensemble-pareto-front.json
```

Omit `--algorithm ensemble` to apply the same constraints to Chebyshev tuning.

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
  --additional-events "/path/to/New Labels 20206-08-29.csv" \
  --exclusions config/label-exclusions.csv \
  --community-events ../../cenalert-paper/events/custom/Community.csv \
  --series-directory ../../cenalert-paper/series/012t0g \
  --parameters-directory ../../cenalert-paper/parameters/chebyshev_selected \
  --detector target/release/cenalert-detector \
  --proximity-days 3 \
  --output /tmp/cenalert-labels.json
```

The CSV inputs and JSON output can remain outside the repository. Repeat
`--additional-events` to load reviewed mixed country/global sheets; the
standard nine-column header is optional for these supplemental files. The
builder validates columns and dates, preserves distinct same-day records, and
reports such groups for manual review.

The manually verified sheets are included as provided; the builder does not
revalidate those events against the selected time series. Before constrained
tuning, confirm that the labels were verified against the same Google Trends
keyword or topic as the input series. A mismatched series can make the
requirement to preserve every applicable country label infeasible.

`config/label-exclusions.csv` records label-removal decisions without changing
the source spreadsheets. Excluded events remain in their source data and do
not become tuning constraints or label matches in the generated file.

When an event should also match specific other countries, pass an optional
rule CSV through `--cross-country-matches`. Its
`additional_matching_countries` column contains semicolon-separated ISO codes.

A country label may list `additional_matching_countries` when that event is
known to affect specific other countries. Those countries may use the event for
attribution, while the event constrains tuning only for its owning
`country_code`:

```json
{
  "date": "2025-01-01",
  "scope": "country",
  "country_code": "RU",
  "additional_matching_countries": ["UA"],
  "description": "Example Russia event also relevant to Ukraine"
}
```

### Run the temporal evaluation

`scripts/run_experiment_matrix.py` runs the detector and label-constraint
comparison plus an untuned baseline on a fixed set of country series. The
default training cutoff is 2024-12-31; test results cover 2025–2026 through the
last available observation.

| Group | Detector | Labels constrain tuning |
| --- | --- | --- |
| `original` | Chebyshev/Z-score with Croston | No |
| `labels` | Chebyshev/Z-score with Croston | Yes |
| `ensemble` | Three-detector ensemble with shared Croston | No |
| `labels_ensemble` | Three-detector ensemble with shared Croston | Yes |
| `fixed_3sigma` | Previous 60 observations, mean + 3 population standard deviations | No |

Here `original` uses the current Rust implementation, including its residual
ratio parameter. It is a single-detector baseline, not an execution of the
archived Python backend.

```bash
cargo build --release
python3 scripts/run_experiment_matrix.py \
  --series-directory /path/to/country-series \
  --labels /path/to/labels.json \
  --rust-binary target/release/cenalert-detector \
  --output-directory /path/to/four-group-experiment \
  --train-end 2024-12-31 \
  --test-end 2026-12-31 \
  --evaluations 5000 \
  --population-size 100 \
  --seeds 42 43 44 \
  --workers 4 \
  --threads-per-tuning 1
```

Add `--prepare-only` to validate and snapshot inputs before starting searches.
Then run the same command with `--resume` and without `--prepare-only`.
`input-summary.json` records actual training/test coverage for each country.
Countries without either period or without a positive training maximum are
explicitly skipped. Use one consistent topic/keyword source across the cutoff.

Only training observations and labels dated on or before the cutoff participate
in tuning and selection. Detection replays historical observations to initialize
the window, keeps the selected parameters fixed, and reports intervals starting
after the cutoff. Earlier intervals remain in full detection JSON. Croston's
denominator and output normalization are frozen to the training maximum using
`detect --reference-end`; later maxima cannot change those reference values.
Dashboard stitching may incorporate later revisions, so this is a retrospective
temporal test on the downloaded snapshot.

The fixed baseline uses a strict `value > mean + 3*std` rule, with the preceding
60 observations including zeros and anomalous values. Consecutive flagged
observations form intervals. A rise above a zero-variance window triggers an
anomaly with a null score. `--baseline-window` changes the prespecified window;
60 is an initial experiment choice. This baseline has no Croston path or
parameter search. It appears once per seed with selection `fixed`.

The series directory accepts the same CSV or Dashboard JSON inputs as the
ensemble batch workflow. Use `--countries RU SS` for a subset and `--seeds 42`
for an initial run. Each country and seed runs all four groups with the same
evaluation budget and country-offset seed. The Chebyshev K range is 5–18.
The existing efficiency ranges are retained: 0.01–0.1 for Chebyshev and
0.05–0.1 for the ensemble. Seeds control the optimizer's evolutionary operators;
its initial population remains unseeded, so fresh runs can differ even with
the same seed. Saved Pareto fronts and parameters preserve each realized run.
Both knee (including its leftmost fallback) and leftmost selections are
evaluated from each Pareto front; `--selections knee` or `leftmost` limits this.
Compare rows with the same country, seed, and selection rule.

Outputs include:

- `summary.csv`: status, anomaly count, country-label coverage, impact, tuning
  time, and fixed-parameter detection time for each run and selection.
- `anomalies.csv`: raw anomaly intervals with nearest applicable label matches.
- `label-coverage.csv`: each applicable country label and its matching starts.
- `yearly-summary.csv`: test metrics and actual coverage dates for each year.
- `detector-differences.csv`: ensemble-only and Chebyshev-only starts, allowing
  the configured proximity window for a counterpart (default ±3 days).
- `report.md`: completion counts and interpretation notes.
- `runs/<seed>/<country>/<group>/`: Pareto front, selected parameters, complete
  detection JSON, selection logs, and result metadata.

Times include subprocess startup and I/O. Concurrent runs share machine
resources; use `--workers 1` for timing comparisons without concurrent jobs.

Test-period labels are used only for reporting. They are incomplete ground
truth; unmatched anomalies remain candidates for review. Raw outputs precede
coalescing, impact filtering, and notification grouping. Annual summaries match
starts and labels within each year; pooled coverage can additionally match
across year boundaries. Impact depends on each detector's threshold, so larger
impact or more detections alone does not establish better detection quality.

Keep the experiment directory outside Git. It retains input series, labels,
the executable, scripts, and settings so `--resume` can validate them before
reusing completed runs. Changed inputs, settings, or code require a new output
directory. Failed executions are retried; infeasible searches are recorded as
results. Missing artifacts in completed runs trigger a rerun. A zero-anomaly
Pareto front selects a deterministic representative and records zero detections.

Run the experiment regression tests after building the release binary:

```bash
python3 -m unittest discover -s scripts -p test_experiment_matrix.py -v
```

### Run all-country ensemble tuning

`scripts/batch_ensemble_tuning.py` runs the complete constrained workflow for
each country series: tune, select the knee or documented leftmost fallback, run
detection, and compare that result with the first Pareto-front solution. For an
infeasible country, it records labels missed by the best simultaneous-coverage
unconstrained Pareto solution and distinguishes labels never detected by any
diagnostic Pareto solution. It also exports anomalies that have no applicable
country or global label.

The series directory may contain paper-format CSV files or single-country JSON
exports downloaded from the Dashboard. For each feasible country, the work
directory retains the knee and leftmost detector outputs plus
`notification-candidates.json`. The report directory includes a combined
`ensemble-notification-candidates.csv`. By default, sparse anomalies separated
by at most one normal day are combined before minimum labeled-impact filtering;
use `--coalesce-sparse-gap-days` to change that gap. The workflow also writes
machine-readable candidates and a cross-country analysis, grouping dates shared
by five countries by default. Use `--multi-country-threshold` to change that
threshold. The report directory includes `cross-country-occurrences.csv` for
reviewing every date shared by at least two countries.

The work directory stores intermediate inputs and Pareto fronts so an
interrupted run can continue with `--resume`. Keep it outside the repository:

```bash
.venv/bin/python scripts/batch_ensemble_tuning.py \
  --series-directory ../../cenalert-paper/series/012t0g \
  --labels /tmp/cenalert-labels.json \
  --rust-binary target/release/cenalert-detector \
  --output-directory reports/ensemble-tuning-paper-series \
  --work-directory /tmp/cenalert-ensemble-tuning \
  --evaluations 5000 \
  --population-size 100 \
  --workers 10 \
  --threads-per-tuning 1 \
  --seed 42
```

For current Dashboard exports, point `--series-directory` to the directory of
All Time JSON files and describe the snapshot in the generated report:

```bash
.venv/bin/python scripts/batch_ensemble_tuning.py \
  --series-directory /path/to/dashboard-all-time-json \
  --labels /tmp/cenalert-labels.json \
  --rust-binary target/release/cenalert-detector \
  --output-directory reports/ensemble-tuning-dashboard-current \
  --work-directory /tmp/cenalert-dashboard-tuning \
  --evaluations 5000 \
  --population-size 100 \
  --workers 10 \
  --threads-per-tuning 1 \
  --seed 42 \
  --dataset-description "CenAlert Dashboard vpn keyword series through YYYY-MM-DD"
```

Use `--countries RU SS` for a smaller validation run. The Markdown report
contains the configuration, findings, country summary, labels requiring
review, and highest-impact unmatched anomalies. A CSV retains the complete
unmatched-anomaly list, a second CSV retains notification candidates, and a
JSON file retains selected parameters. Source event spreadsheets and full
intermediate Pareto fronts remain in their input and work locations.

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
  --algorithm ensemble \
  --input /tmp/ensemble-pareto-front.json \
  --output /tmp/selected-ensemble-parameters.json
```

For Chebyshev, omit `--algorithm ensemble`:

```bash
cargo run --release -- \
  select \
  --input /path/to/pareto-front.json \
  --output /tmp/selected-parameters.json
```

For labeled-selection experiments, `--left-position N` selects the one-based
`N`th unique solution from the low-anomaly end of the front instead of fitting
and selecting its knee. Include `--algorithm ensemble` when the input is an
ensemble Pareto front. For example:

```bash
cargo run --release -- \
  select \
  --input /path/to/pareto-front.json \
  --output /tmp/left-3-parameters.json \
  --left-position 3
```

Selection writes a named parameter object accepted directly by `detect` for
the selected algorithm.
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
python3 -m compileall -q scripts
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

Standalone Isolation Forest and LOF can be checked with the same script by
adding `--algorithm isolation-forest` or `--algorithm lof` and supplying the
corresponding four-element parameter file. On the current Russia export,
Isolation Forest matches Python's anomaly count, interval dates, peaks, initial
scores, and residuals. Four impact values differ slightly (maximum absolute
difference approximately `0.025`) because the bounded optimizer can choose
different points on a flat, stepwise Isolation Forest score interval.

The Python validation helper also contains a direct implementation of the
three-member reference ensemble. For the illustrative ensemble parameter object
above, Rust and Python match all 35 Russia anomaly intervals and numeric metadata
with an absolute tolerance of `0.03`:

```bash
.venv/bin/python scripts/compare_outputs.py \
  --algorithm ensemble \
  --absolute-tolerance 0.03 \
  --input /path/to/RU-dashboard-export.json \
  --parameters /path/to/ensemble-parameters.json \
  --rust-binary target/release/cenalert-detector
```

### All-country parity

`scripts/batch_parity.py` runs the same Python/Rust comparison across every
country CSV in a paper series directory. It converts each CSV in a temporary
directory, runs countries in parallel, and writes a JSON report without
committing the source series or generated inputs:

```bash
.venv/bin/python scripts/batch_parity.py \
  --series-directory ../../cenalert-paper/series/012t0g \
  --chebyshev-parameters ../../cenalert-paper/parameters/chebyshev_selected \
  --rust-binary target/release/cenalert-detector \
  --output /tmp/cenalert-all-country-parity.json \
  --workers 6
```

The recorded run covers all 76 paper countries. Chebyshev uses each country's
selected paper parameters; Isolation Forest, LOF, and ensemble use the fixed
implementation-validation parameters shown above. These fixed parameters test
parity and are not proposed production selections.

| Algorithm | Country runs | Anomaly count and dates | Metadata within configured tolerance | Impact-only differences |
| --- | ---: | ---: | ---: | ---: |
| Chebyshev | 72 | 72 | 72 | 0 |
| Isolation Forest | 76 | 76 | 57 | 19 |
| LOF | 76 | 76 | 75 | 1 |
| Ensemble | 72 | 72 | 64 | 8 |

Chebyshev skips ER, KP, LR, and ME because their all-zero series have no
selected paper parameter file. Ensemble parity also skips these all-zero
series because the Python reference does not initialize a Croston forecast for
them; the standalone outlier detectors process and match them with zero
anomalies. Across all 296 executed country/algorithm
runs, Rust and Python have identical anomaly counts, start dates, end dates,
and peak dates. The 28 remaining report mismatches are impact values only:
Isolation Forest's stepwise score can make the two bounded optimizers choose
different, equally scoring points in a flat threshold interval. Initial score
and residual still match.

The cross-country run exposed two sparse-state edge cases that the original RU
and SS checks did not cover: simultaneous anomaly-ending conditions and
StatsForecast's treatment of NaNs introduced by sparse-window rescaling. Both
are covered by regression tests.

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
hotspot. The optimized window maintains sorted values and caches unchanged
moments. Croston SBA reconstructs a dense series only on the sparse path to
preserve StatsForecast edge-case behavior. The outputs remain byte-for-byte
equal to the initial Rust implementation for the recorded Russia and South
Sudan exports.

| In-process benchmark | Initial Rust | Optimized Rust | Improvement |
| --- | ---: | ---: | ---: |
| Russia, one parameter set | 6.83 ms/evaluation | 2.44 ms/evaluation | 2.80× |
| South Sudan, one parameter set | 0.76 ms/evaluation | 0.25 ms/evaluation | 3.10× |

A single-threaded 5,000-evaluation sweep over the Russia series, cycling through
73 paper parameter sets, completed in 9.00 seconds. This benchmark measures the
detection/evaluation core rather than the NSGA-II optimizer itself.
