# CenAlert

CenAlert detects spikes in Google Trends search interest and presents them with
documented event labels. A daily publisher collects and stitches samples, runs
the Rust detector with saved parameters, and publishes a SQLite snapshot. The
API and dashboard serve that snapshot.

## Start the dashboard

Install Docker Compose and prepare a snapshot using the
[daily pipeline](rust-detector/README.md#daily-updates).
Set the absolute directory containing `current/cenalert.sqlite`:

```sh
export CENALERT_DATA_ROOT=/absolute/path/to/daily
export CENALERT_PORT=8080
docker compose up --build -d --wait
```

Open http://127.0.0.1:8080. Use **All time** when viewing historical data.
Check the API at http://127.0.0.1:8080/api/health.

The data directory is mounted read-only. Each successful daily update replaces
the `current` symlink; the API reads the new snapshot without restarting.
The publisher runs separately from the web containers. Containers restart
automatically; place the localhost-bound dashboard behind an HTTPS reverse
proxy for public deployment.

```sh
docker compose logs --tail=100 api dashboard
docker compose down
```

## Components

- `rust-detector/`: detection, parameter tuning, filtering, and daily publication.
- `src/`: Google Trends collection and time-series stitching used by the publisher.
- `dashboard/`: web frontend, nginx configuration, and API documentation.
- `compose.yaml`: API and dashboard services.

## Configuration and maintenance

Keep collected data, labels, saved parameters, and credentials outside the
repository. Each country needs a `<CC>.json` parameter file. Updates use these
parameters; tuning is a separate operation.

- [Build, tune, and run daily updates](rust-detector/README.md)
- [Frontend development and API filters](dashboard/README.md)

The API exposes raw anomaly intervals and matching label metadata. The daily
publisher produces local notification previews; it does not deliver messages.
