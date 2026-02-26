#!/usr/bin/env bash
set -euo pipefail

# Set this once per run.
PROBLEM_NAME="btc"

# Candidate JSON roots (supports logs root, split root, or samples folder).
SAMPLES_DIR="./logs"

INPUT_CSV="./data/${PROBLEM_NAME}.csv"
OUTPUT_DIR="./ga_optimizer/data/${PROBLEM_NAME}"
OUTPUT_CSV="${OUTPUT_DIR}/features.csv"
METADATA_CSV="${OUTPUT_DIR}/features.meta.csv"

mkdir -p "${OUTPUT_DIR}"

# If your label is always last in the source CSV, keep --use_last_column_as_label.
python -m ga_optimizer.main \
  --input_csv "${INPUT_CSV}" \
  --samples_dir "${SAMPLES_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --metadata_csv "${METADATA_CSV}" \
  --k_per_island 2 \
  --use_last_column_as_label
