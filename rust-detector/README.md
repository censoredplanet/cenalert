# CenAlert detector and publisher

## Build

Install Rust using the version in `rust-toolchain.toml`, and Python 3.10 or newer.
Linux builds also require a C compiler and libclang development libraries
(`build-essential libclang-dev` on Debian/Ubuntu); the Docker build installs them.
From this directory:

```sh
cargo build --release --locked
python3 -m venv .venv-runtime
.venv-runtime/bin/python -m pip install -r requirements-runtime.txt
```

`target/release/cenalert-detector --help` lists the detection, tuning, selection,
filtering, cross-country grouping, and series reconstruction commands.

## Daily updates

Prepare a labels JSON array and one saved parameter JSON file per country
(`<CC>.json`). The default algorithm is `ensemble`. Keep these inputs and
published data outside the repository.

To publish existing series, supply `<CC>.json`, `<CC>.csv`, or
`<CC>_stitched.csv` files. CSVs contain `date,value`; JSON rows contain
`date,rate,country,topic`.

```sh
.venv-runtime/bin/python scripts/run_daily.py \
  --series-directory /absolute/path/to/series \
  --parameters-directory /absolute/path/to/parameters \
  --labels /absolute/path/to/labels.json \
  --countries RU ID \
  --through-date YYYY-MM-DD \
  --output-root /absolute/path/to/daily
```

For parameters selected with a fixed normalization cutoff, also pass
`--reference-end YYYY-MM-DD` on every update and include that history in the
input. Otherwise normalization uses the full series maximum.

To collect from Google Trends, replace `--series-directory` with `--download`
and supply a key file. Initial collection starts in January 2011 unless
`--start-month YYYY-MM` is given. Collection uses 45 samples per country.

```sh
.venv-runtime/bin/python scripts/run_daily.py \
  --download --api-key-file /private/path/trends-key.txt \
  --parameters-directory /absolute/path/to/parameters \
  --labels /absolute/path/to/labels.json \
  --countries RU ID \
  --output-root /absolute/path/to/daily
```

For subsequent downloads, add
`--previous-stitched /absolute/path/to/daily/current/stitched` to reuse history.
This requires both normal and coarse stitched CSVs produced from raw samples;
a dashboard JSON export alone does not provide coarse history.
To replay archived samples, use `--samples-directory /path/to/batch`, containing
`sample0/<CC>/`, `sample1/<CC>/`, etc., instead of `--download`.

`--through-date` defaults to yesterday in UTC. Missing dates or stale inputs
stop publication. A failed run leaves the previous snapshot active and keeps
partial files in a `failed-*` directory for inspection. Concurrent publishers
using the same output directory are rejected. `--plan` validates paths and JSON
without executing the pipeline. `--plots` adds charts to notification previews.

Each successful run publishes:

```text
current -> run-<timestamp>/
  cenalert.sqlite
  stitched/
  detection/
  preview/
  parameters/
  labels.json
  run.json
```

Point Docker Compose's `CENALERT_DATA_ROOT` at the output root. Schedule the
publisher separately after collection is available. Keep the API key outside
Git; `GOOGLE_TRENDS_API_KEY` is also supported.

## Parameter tuning

Tuning is run explicitly, not as part of daily updates. For one country:

```sh
target/release/cenalert-detector tune \
  --algorithm ensemble --input /path/to/CC.json \
  --labels /path/to/labels.json --proximity-days 3 \
  --evaluations 5000 --population-size 100 \
  --output /path/to/front.json
target/release/cenalert-detector select \
  --algorithm ensemble --input /path/to/front.json \
  --output /path/to/parameters/CC.json
```

The input series determines the tuning period. Selection uses the Pareto knee
with a leftmost fallback. `detect --help`, `tune --help`, and `select --help`
describe parameter formats and options. Labels use country or global scope;
global date matching is capped at one day, and country matching defaults to
three days in the publisher.

## Tests

```sh
cargo test --locked
.venv-runtime/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

The Python tests cover snapshots, API filters, notification previews, label
matching, and daily publication. They do not download live data.
