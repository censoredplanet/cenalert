#!/bin/bash
set -euo pipefail
source bin/config
# Dummy events file (since it's not used)
DUMMY_EVENTS="dummy_events.csv"
touch $DUMMY_EVENTS


# Algorithm constant
ALGORITHM="chebyshev"

# Path to your Python script
SCRIPT_PATH="src/run_cenalert.py"

# Top-level where your term subdirectories live
TERMS_ROOT=$BASE_ROOT

for term_dir in "$TERMS_ROOT"/*; do
  # Strip trailing slash and leading path
  term=$(basename "${term_dir%/}")
  echo "=== Processing term: $term ==="

  # Within each term, these must exist:
PARAMS_DIR="$SCALED_PARAM/$term"
TIME_SERIES_DIR="$STITCHED_ROOT/$term"
OUTPUT_BASE="$RESULTS/$term"

# Skip anything that doesn’t look like a proper term folder
missing=0

if [[ ! -d "$PARAMS_DIR" ]]; then
  echo "  › Missing: $PARAMS_DIR" >&2
  missing=1
fi

if [[ ! -d "$TIME_SERIES_DIR" ]]; then
  echo "  › Missing: $TIME_SERIES_DIR" >&2
  missing=1
fi

if [[ $missing -eq 1 ]]; then
  echo "  › Skipping ${term_dir}, due to missing required subdir(s)." >&2
  continue
fi

mkdir -p "$OUTPUT_BASE"

for param_file in "$PARAMS_DIR"/*; do
  country_code=$(basename "$param_file")
  country_dir=$OUTPUT_BASE"/$country_code"
  mkdir -p "$country_dir"
  time_series_file="${TIME_SERIES_DIR}/${country_code}_stitched.csv"

  if [[ ! -f "$time_series_file" ]]; then
    echo "  ⚠️  No time series for ${country_code}, skipping."
    continue
  fi

python "$SCRIPT_PATH" \
  --path "$time_series_file" \
  --events "$DUMMY_EVENTS" \
  --algorithm "$ALGORITHM" \
  --parameters "$param_file"\
  --output "$country_dir"

echo "=== Done term: $term ==="$'\n'
done

echo "All terms processed!"
done