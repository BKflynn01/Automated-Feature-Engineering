#!/usr/bin/env bash
set -euo pipefail

# Example paths; update before running.
INPUT_CSV="./data/bank.csv"
SAMPLES_DIR="./logs/samples"
OUTPUT_CSV="./outputs/bank_ga_features.csv"
METADATA_CSV="./outputs/bank_ga_features.meta.csv"

# If your label is always last in the source CSV, keep --use_last_column_as_label.
python -m ga_optimizer.main \
  --input_csv "${INPUT_CSV}" \
  --samples_dir "${SAMPLES_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --metadata_csv "${METADATA_CSV}" \
  --k_per_island 2 \
  --use_last_column_as_label
