FROM rust:1.97.1-bookworm AS build
WORKDIR /build
COPY rust-detector/Cargo.toml rust-detector/Cargo.lock rust-detector/rust-toolchain.toml ./
COPY rust-detector/src ./src
RUN cargo build --release --locked

FROM python:3.13-slim-bookworm
COPY rust-detector/requirements-runtime.txt /app/requirements-runtime.txt
RUN pip install --no-cache-dir -r /app/requirements-runtime.txt \
    && useradd --uid 10001 --create-home cenalert
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app/rust-detector
COPY --from=build /build/target/release/cenalert-detector /usr/local/bin/cenalert-detector
COPY rust-detector/scripts/build_snapshot.py rust-detector/scripts/export_snapshot.py rust-detector/scripts/preview_notifications.py rust-detector/scripts/batch_ensemble_tuning.py rust-detector/scripts/run_daily.py ./scripts/
COPY src/stitching_job.py /app/src/
COPY src/lib/stitching.py src/lib/google_trends_utils.py /app/src/lib/
USER cenalert
CMD ["cenalert-detector", "--help"]
