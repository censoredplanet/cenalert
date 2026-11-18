# CenAlert dashboard

The dashboard reads the daily SQLite snapshot through a same-origin, read-only
API. Country lists, time series, and event details are fetched at runtime.
Structured event fields include title, description, cause, justification,
affected services, and sources. Events are raw detector intervals; a temporal
label match is exposed separately as `labelMatched` and `matchDistanceDays`.

## Local preview

From the repository root:

```sh
cd dashboard
npm ci
OBSERVABLE_TELEMETRY_DISABLE=true npm run build
cd ..
python3 rust-detector/scripts/serve_snapshot.py \
  --database /path/to/daily/current/cenalert.sqlite \
  --static-directory dashboard/dist --port 8080
```

Open http://127.0.0.1:8080/?range=all. The default view shows the last 60 days;
use All time when previewing historical snapshots.

## Containers

To populate a dashboard from saved label-constrained ensemble results, replay
the knee-selected parameters and check every event against the saved detections:

```sh
python3 rust-detector/scripts/snapshot_from_results.py \
  --results /absolute/path/to/results \
  --output /absolute/path/to/dashboard-data/current/cenalert.sqlite
```

The command uses the archived series, labels, detector binary, and training
normalization cutoff. It verifies the replay before publishing the snapshot.

Set the data root containing `current/cenalert.sqlite`, then start the API
and dashboard. For daily publication, `current` is a symlink to a run directory.

```sh
export CENALERT_DATA_ROOT=/absolute/path/to/daily
docker compose up --build -d
```

Open http://127.0.0.1:8080. The API uses Gunicorn; nginx serves the dashboard
and proxies `/api/`. Data is mounted read-only. Mount the entire daily output
root so atomic updates of `current` remain visible. The publisher remains a
separate process, using `rust-detector/scripts/run_daily.py`.

The port binds to localhost. Place the dashboard behind an HTTPS reverse proxy
for deployment.

## API

- `GET /api/health`: schema version and snapshot creation time.
- `GET /api/countries`: country codes, algorithms, topics, and observation bounds.
- `GET /api/series?country=RU`: daily observations in the detector's scale.
- `GET /api/events?country=RU`: raw anomaly intervals and matched event fields.

Series and events accept `start` and `end` (YYYY-MM-DD), `limit` (1–20000),
and `offset`. Events use interval overlap for date filtering. Countries use
uppercase ISO codes. Each request reads the current snapshot.

For events, omit `country` to query all countries. Optional event filters are:

- `labelMatched=true` or `false`: anomalies with or without a matching label.
- `category`, `cause`, `justification`: exact field values, ignoring ASCII case.
- `affectedServices`: an exact member of the service list, ignoring ASCII case.

Filters are combined with AND and applied before pagination. For example,
`/api/events?category=filtering&affectedServices=Facebook&labelMatched=true`
returns label-matched filtering anomalies involving Facebook across countries.
`/api/events?labelMatched=false&start=2025-01-01` returns unmatched anomalies
overlapping that period. An event here is a detected anomaly interval with its
matched label metadata; labels without a detection are not returned.

```sh
python3 -m unittest discover -s rust-detector/scripts -p test_api.py
```

## Source

Adapted from the CenAlert page in [censoredplanet/dashboard](https://github.com/censoredplanet/dashboard),
revision `3c84ce76c75fa368e35791b05ce5893fdf5df95c`.
