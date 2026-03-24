#!/usr/bin/env bash
set -euo pipefail

# Optional local secret loading.
# Keep real keys in .env.local (gitignored)
#   API_KEY=...
#   GEMINI_API_KEY=...
#   GEMINI_API_EVALUATOR=...
#   WANDB_API_KEY=...
if [[ -f ".env.local" ]]; then
  set -a
  source ".env.local"
  set +a
fi

# Enable this check by running with USE_API=true, e.g.:
#   USE_API=true bash run_llmfe.sh
if [[ "${USE_API:-false}" == "true" ]]; then
  : "${API_KEY:?Missing API_KEY (OpenAI key)}"
  : "${GEMINI_API_KEY:?Missing GEMINI_API_KEY}"
  : "${GEMINI_API_EVALUATOR:?Missing GEMINI_API_EVALUATOR}"
fi

################ LLM-FE with API ################
#### Classification Datasets ####
WANDB_GROUP="${WANDB_GROUP:-btc_llmfe}"
python main.py --use_api TRUE --api_model "gpt-4o-mini" --problem_name btc-classification --spec_path ./specs/specification_btc-classification.txt --log_path ./logs/btc_gpt3.5 --is_time_series --time_series_train_window 365 --time_series_test_window 1 --time_series_step_window 3 --outer_splits 10 --wandb_project llmfe-feature-engineering-btc --wandb_group "$WANDB_GROUP" --wandb_run_name "llmfe_btc_classification" --run_type llmfe
## balance-scale ## 
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name balance-scale --spec_path ./specs/specification_balance-scale.txt --log_path ./logs/balance-scale_gpt3.5


 
