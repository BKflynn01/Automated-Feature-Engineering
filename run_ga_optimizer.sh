#!/usr/bin/env bash
set -euo pipefail

# Set this once per run.
PROBLEM_NAME="${PROBLEM_NAME:-btc-classification}"

# Candidate JSON root + optional in-root glob.
# Keep SAMPLES_DIR at ./logs and filter to only the split family you want.
SAMPLES_DIR="${SAMPLES_DIR:-./logs}"
SOURCE_FILE_GLOB="${SOURCE_FILE_GLOB:-btc_gpt3.5_split_*/samples/*.json}"
K_PER_ISLAND="${K_PER_ISLAND:-2}"
SEP="${SEP:-,}"

INPUT_CSV="./data/${PROBLEM_NAME}.csv"
OUTPUT_DIR="./ga_optimizer/data/${PROBLEM_NAME}"
OUTPUT_CSV="${OUTPUT_DIR}/features.csv"
METADATA_CSV="${OUTPUT_DIR}/features.meta.csv"
GA_OUTPUT_DIR="${GA_OUTPUT_DIR:-${OUTPUT_DIR}}"
GA_CONFIG_PATH="${GA_CONFIG_PATH:-}"
GA_PRESET="${GA_PRESET:-normal}"

mkdir -p "${OUTPUT_DIR}"

echo "GA preprocessing config:"
echo "  INPUT_CSV=${INPUT_CSV}"
echo "  SAMPLES_DIR=${SAMPLES_DIR}"
echo "  SOURCE_FILE_GLOB=${SOURCE_FILE_GLOB}"
echo "  K_PER_ISLAND=${K_PER_ISLAND}"
echo "  SEP=${SEP}"
echo "  OUTPUT_CSV=${OUTPUT_CSV}"
echo "  METADATA_CSV=${METADATA_CSV}"

# If your label is always last in the source CSV, keep --use_last_column_as_label.
python -m ga_optimizer.main \
  --input_csv "${INPUT_CSV}" \
  --samples_dir "${SAMPLES_DIR}" \
  --source_file_glob "${SOURCE_FILE_GLOB}" \
  --output_csv "${OUTPUT_CSV}" \
  --metadata_csv "${METADATA_CSV}" \
  --k_per_island "${K_PER_ISLAND}" \
  --sep "${SEP}" \
  --use_last_column_as_label

# GA stage (feature subset selection over the merged feature matrix).
echo "GA selection config:"
echo "  GA_CONFIG_PATH=${GA_CONFIG_PATH:-<none>}"
echo "  GA_PRESET=${GA_PRESET}"
echo "  GA_OUTPUT_DIR=${GA_OUTPUT_DIR}"

if [[ -z "${GA_CONFIG_PATH}" ]]; then
  echo "ERROR: GA_CONFIG_PATH is required." >&2
  exit 1
fi

GA_ARGS=(
  --input_csv "${OUTPUT_CSV}"
  --dataset_name "${PROBLEM_NAME}"
  --output_dir "${GA_OUTPUT_DIR}"
  --sep "${SEP}"
  --config "${GA_CONFIG_PATH}"
  --preset "${GA_PRESET}"
)
if [[ -n "${LABEL_COLUMN:-}" ]]; then
  GA_ARGS+=(--label_column "${LABEL_COLUMN}")
fi
if [[ -n "${TASK:-}" ]]; then
  GA_ARGS+=(--task "${TASK}")
fi
if [[ -n "${SCORING:-}" ]]; then
  GA_ARGS+=(--scoring "${SCORING}")
fi
if [[ -n "${N_GENERATIONS:-}" ]]; then
  GA_ARGS+=(--n_generations "${N_GENERATIONS}")
fi
if [[ -n "${POPULATION_SIZE:-}" ]]; then
  GA_ARGS+=(--population_size "${POPULATION_SIZE}")
fi
if [[ -n "${CX_PROB:-}" ]]; then
  GA_ARGS+=(--cx_prob "${CX_PROB}")
fi
if [[ -n "${MUT_PROB:-}" ]]; then
  GA_ARGS+=(--mut_prob "${MUT_PROB}")
fi
if [[ -n "${TOURNAMENT_SIZE:-}" ]]; then
  GA_ARGS+=(--tournament_size "${TOURNAMENT_SIZE}")
fi
if [[ -n "${CV_FOLDS:-}" ]]; then
  GA_ARGS+=(--cv_folds "${CV_FOLDS}")
fi
if [[ -n "${N_ESTIMATORS:-}" ]]; then
  GA_ARGS+=(--n_estimators "${N_ESTIMATORS}")
fi
if [[ -n "${RANDOM_STATE:-}" ]]; then
  GA_ARGS+=(--random_state "${RANDOM_STATE}")
fi
if [[ -n "${TIME_SERIES_TRAIN_WINDOW:-}" ]]; then
  GA_ARGS+=(--time_series_train_window "${TIME_SERIES_TRAIN_WINDOW}")
fi
if [[ -n "${TIME_SERIES_TEST_WINDOW:-}" ]]; then
  GA_ARGS+=(--time_series_test_window "${TIME_SERIES_TEST_WINDOW}")
fi
if [[ -n "${TIME_SERIES_STEP_WINDOW:-}" ]]; then
  GA_ARGS+=(--time_series_step_window "${TIME_SERIES_STEP_WINDOW}")
fi
if [[ -n "${TIME_SERIES_GAP:-}" ]]; then
  GA_ARGS+=(--time_series_gap "${TIME_SERIES_GAP}")
fi
if [[ -n "${MAX_TIME_SERIES_SPLITS:-}" ]]; then
  GA_ARGS+=(--max_time_series_splits "${MAX_TIME_SERIES_SPLITS}")
fi
if [[ -n "${N_JOBS:-}" ]]; then
  GA_ARGS+=(--n_jobs "${N_JOBS}")
fi
if [[ -n "${EARLY_STOP_PATIENCE:-}" ]]; then
  GA_ARGS+=(--early_stop_patience "${EARLY_STOP_PATIENCE}")
fi
if [[ -n "${EARLY_STOP_MIN_DELTA:-}" ]]; then
  GA_ARGS+=(--early_stop_min_delta "${EARLY_STOP_MIN_DELTA}")
fi

# Explicit time-series switch if requested (1=true, 0=false).
if [[ "${IS_TIME_SERIES:-}" == "1" ]]; then
  GA_ARGS+=(--is_time_series)
elif [[ "${IS_TIME_SERIES:-}" == "0" ]]; then
  GA_ARGS+=(--no_is_time_series)
fi

python -m ga_optimizer.ga_main "${GA_ARGS[@]}"
