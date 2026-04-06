#!/bin/bash

source bin/config

BASE_DIR=$BASE_ROOT  # This is your data_root
DATE_SUFFIX=$(date +"%Y-%m-%d")
GET_GT_SCRIPT="src/get_gt_data.py"
COUNTRY_CODES_FILE="country_codes.csv"  # Must exist in current dir

START_MONTH_OVERRIDE=""
CURRENT_MONTH=$(date +"%Y-%m")
DAILY_UPDATE=""  # Default to true

# Define topics (or load dynamically)
TOPICS=("vpn")

# ----------------------------
# Functions
# ----------------------------

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --start_month)
        START_MONTH_OVERRIDE="$2"
        DAILY_UPDATE="--no_daily_update"  # Override flag
        shift 2
        ;;
      *)
        echo "❌ Unknown argument: $1"
        exit 1
        ;;
    esac
  done
}

get_start_month() {
  if [[ -n "$START_MONTH_OVERRIDE" ]]; then
    echo "$START_MONTH_OVERRIDE"
  else
    python3 -c 'from datetime import datetime; from dateutil.relativedelta import relativedelta; print((datetime.today() - relativedelta(days=1, months=10)).strftime("%Y-%m"))'
  fi
}

check_required_files() {
  if [[ ! -f "$COUNTRY_CODES_FILE" ]]; then
    echo "❌ Missing required file: $COUNTRY_CODES_FILE"
    exit 1
  fi
}

run_download() {
  local topic="$1"
  local sample_dir="$2"
  local start_month="$3"
  local end_month="$4"

  echo "🟢 Downloading topic '$topic' → $sample_dir"
  echo "   Start: $start_month → End: $end_month"
  echo "   Daily update: $DAILY_UPDATE"

  python3 "$GET_GT_SCRIPT" \
    --country_code "$COUNTRY_CODES_FILE" \
    --data_output_existing "$sample_dir" \
    --log_level "INFO" \
    --sliding_window_size 8 \
    --sliding_window_overlap 7 \
    --start_month "$start_month" \
    --topic "$topic" \
    $DAILY_UPDATE
}

main() {
  parse_args "$@"
  check_required_files

  START_MONTH=$(get_start_month)
  echo "[INFO] Using start month: $START_MONTH"

  for topic in "${TOPICS[@]}"; do
    OUTPUT_DIR="$BASE_DIR/$topic/output_$DATE_SUFFIX"
    mkdir -p "$OUTPUT_DIR"

    echo "📂 Processing topic: $topic"
    for i in $(seq 0 44); do
      SAMPLE_DIR="$OUTPUT_DIR/sample$i"
      mkdir -p "$SAMPLE_DIR"
      run_download "$topic" "$SAMPLE_DIR" "$START_MONTH" "$CURRENT_MONTH"
    done
  done
}

main "$@"
