"""
Perform Feature Engineering
"""

import json

# Imports
import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold

import wandb
from cv_utils import resolve_sliding_window_params
from llmfe.classification_aggregate import aggregate_sample_metrics, log_aggregate_run
from llmfe.logging import SplitLogger
from utils import is_categorical

# Arguments
parser = ArgumentParser()
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--use_api", type=bool, default=False)
parser.add_argument("--api_model", type=str, default="gpt-3.5-turbo")
parser.add_argument("--spec_path", type=str)
parser.add_argument("--log_path", type=str, default="./logs/oscillator1")
parser.add_argument("--problem_name", type=str, default="oscillator1")
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--is_time_series", action="store_true")
parser.add_argument("--time_series_train_window", type=int, default=None)
parser.add_argument("--time_series_test_window", type=int, default=None)
parser.add_argument("--time_series_step_window", type=int, default=None)
parser.add_argument("--time_series_gap", type=int, default=0)
parser.add_argument("--outer_splits", type=int, default=1)
parser.add_argument("--wandb_project", type=str, default="llmfe-feature-engineering-btc")
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--run_type", type=str, default="llmfe")
args = parser.parse_args()


if __name__ == "__main__":
    # Define the maximum number of iterations
    global_max_sample_num = 40
    eval_splits = 4
    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()

    problem_name = args.problem_name
    label_encoder = preprocessing.LabelEncoder()
    regression_problems = {
        "btc",
        "btc_regression",
        "forest-fires",
        "housing",
        "insurance",
        "bike",
        "wine",
        "crab",
    }
    default_time_series_problems = {
        "btc",
        "btc_regression",
        "btc-classification",
        "btc_classification",
    }
    is_regression = problem_name in regression_problems
    is_time_series = args.is_time_series or (problem_name in default_time_series_problems)
    if is_regression:
        print("Regression Accepted")

    # Load data observations
    file_name = f"./data/{problem_name}.csv"
    df = pd.read_csv(file_name)

    target_attr = df.columns[-1]
    # Alteration for data cat
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data = {}
    try:
        with open(meta_data_name, "r") as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load metadata from {meta_data_name}: {e}")
    # End alteration for data cat

    attribute_names = df.columns[:-1].tolist()
    is_cat_map = {name: is_categorical(df[name], meta_data, name) for name in attribute_names}
    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    label_list = np.unique(y).tolist()

    X = X.drop(target_attr, axis=1)
    date_column_name = "date"  # ensure date column is not included
    if date_column_name in X.columns:
        X = X.drop(columns=[date_column_name])

    for col in X.columns:
        if X[col].dtype == "string":
            X[col] = label_encoder.fit_transform(X[col])

    # Handle missing values
    X = X.fillna(0)
    if not is_regression:
        y = label_encoder.fit_transform(y)
    else:
        y = y

    # Load metadata
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data = {}
    try:
        with open(meta_data_name, "r") as f:
            filed_meta_data = json.load(f)
        if isinstance(filed_meta_data, dict) and (
            "continuous_features" in filed_meta_data or "categorical_features" in filed_meta_data
        ):
            for feature in filed_meta_data.get("continuous_features", []):
                name = feature.get("name")
                if not name:
                    continue
                meta_data[name] = {
                    "type": feature.get("type", "continuous"),
                    "description": feature.get("description", ""),
                    "context": feature.get("context", ""),
                }
            for feature in filed_meta_data.get("categorical_features", []):
                name = feature.get("name")
                if not name:
                    continue
                meta_data[name] = {
                    "type": feature.get("type", "categorical"),
                    "description": feature.get("description", ""),
                    "context": feature.get("context", ""),
                }
        else:
            meta_data = filed_meta_data if isinstance(filed_meta_data, dict) else {}
    except Exception as e:
        print(f"Warning: failed to load metadata from {meta_data_name}: {e}")

    if args.outer_splits < 1:
        raise ValueError("--outer_splits must be >= 1")

    outer_jobs = []
    if args.outer_splits == 1:
        # Single outer run: preserve current behavior where spec owns splitting.
        resolved_train_window = None
        resolved_test_window = None
        resolved_step_window = None
        if is_time_series:
            resolved_train_window, resolved_test_window, resolved_step_window = (
                resolve_sliding_window_params(
                    n_samples=len(X.index),
                    n_splits=eval_splits,
                    train_window=args.time_series_train_window,
                    test_window=args.time_series_test_window,
                    step_window=args.time_series_step_window,
                    gap=args.time_series_gap,
                )
            )
            print(
                "Single-run mode. Inner evaluation uses SlidingWindowSplit("
                f"train={resolved_train_window}, test={resolved_test_window}, "
                f"step={resolved_step_window}, gap={args.time_series_gap})"
            )
        else:
            print("Single-run mode. Inner evaluation CV is spec-defined.")

        outer_jobs.append(
            {
                "split_id": 1,
                "X_fold": X,
                "y_fold": y,
                "inner_train_window": resolved_train_window,
                "inner_test_window": resolved_test_window,
                "inner_step_window": resolved_step_window,
                "log_path": args.log_path,
            }
        )
    else:
        # Multi outer split mode for per-split W&B visibility.
        if is_time_series:
            inner_train_window, inner_test_window, inner_step_window = (
                resolve_sliding_window_params(
                    n_samples=len(X.index),
                    n_splits=eval_splits,
                    train_window=args.time_series_train_window,
                    test_window=args.time_series_test_window,
                    step_window=args.time_series_step_window,
                    gap=args.time_series_gap,
                )
            )
            print(
                "Outer split mode (time series). "
                "Partitioning inner rolling windows by temporal offsets: "
                f"base(train={inner_train_window}, test={inner_test_window}, "
                f"step={inner_step_window}, gap={args.time_series_gap}), "
                f"outer_splits={args.outer_splits}"
            )
            min_required = inner_train_window + args.time_series_gap + inner_test_window
            partitioned_step = max(1, inner_step_window * args.outer_splits)
            for split_id in range(1, args.outer_splits + 1):
                offset = split_id - 1
                X_train_fold = X.iloc[offset:]
                y_train_fold = y[offset:]
                if len(X_train_fold.index) < min_required:
                    print(
                        f"Skipping outer split {split_id}: "
                        f"not enough rows after offset={offset} (have {len(X_train_fold.index)}, "
                        f"need >= {min_required})."
                    )
                    continue
                print(
                    f"Outer split {split_id}: offset={offset}, "
                    f"inner(train={inner_train_window}, test={inner_test_window}, "
                    f"step={partitioned_step}, gap={args.time_series_gap})"
                )
                outer_jobs.append(
                    {
                        "split_id": split_id,
                        "X_fold": X_train_fold,
                        "y_fold": y_train_fold,
                        "inner_train_window": inner_train_window,
                        "inner_test_window": inner_test_window,
                        "inner_step_window": partitioned_step,
                        "log_path": f"{args.log_path}_split_{split_id}",
                    }
                )
        else:
            cv_splitter = (
                KFold(n_splits=args.outer_splits, shuffle=True, random_state=42)
                if is_regression
                else StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=42)
            )
            print(f"Outer split mode. {cv_splitter}")
            for split_id, (train_idx, _test_idx) in enumerate(cv_splitter.split(X, y), start=1):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y[train_idx]
                outer_jobs.append(
                    {
                        "split_id": split_id,
                        "X_fold": X_train_fold,
                        "y_fold": y_train_fold,
                        "inner_train_window": None,
                        "inner_test_window": None,
                        "inner_step_window": None,
                        "log_path": f"{args.log_path}_split_{split_id}",
                    }
                )

    if not outer_jobs:
        raise RuntimeError("No valid outer split jobs were produced. Check split settings.")

    # Load config and parameters
    from llmfe import config as config_lib
    from llmfe import evaluator, pipeline, sampler

    class_config = config_lib.ClassConfig(
        llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox
    )
    run_config = config_lib.Config(
        use_api=args.use_api,
        api_model=args.api_model,
    )

    review_processes = []
    classification_metrics_sink = []
    current_is_cat = [is_cat_map.get(col, False) for col in X.columns]
    for job in outer_jobs:
        split_id = job["split_id"]
        data_dict = {
            "inputs": job["X_fold"],
            "outputs": job["y_fold"],
            "is_cat": current_is_cat,
            "is_regression": is_regression,
            "is_time_series": is_time_series,
            "time_series_train_window": job["inner_train_window"],
            "time_series_test_window": job["inner_test_window"],
            "time_series_step_window": job["inner_step_window"],
            "time_series_gap": args.time_series_gap,
        }
        dataset = {"data": data_dict}
        split_logger = SplitLogger(run_id=str(args.run_id), split_idx=split_id)
        current_log_file = split_logger.log_file_path
        print(f"--- Split {split_id}: Logging reviews to {current_log_file} ---")
        split_run_name = (
            f"{args.wandb_run_name}_split_{split_id}"
            if args.wandb_run_name
            else f"{job['log_path']}"
        )

        wandb_run_id = pipeline.main(
            specification=specification,
            inputs=dataset,
            config=run_config,
            meta_data=meta_data,
            max_sample_nums=global_max_sample_num,
            class_config=class_config,
            log_dir=job["log_path"],
            logger=split_logger,
            split_id=split_id,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            run_name=split_run_name,
            run_type=args.run_type,
            classification_metrics_sink=classification_metrics_sink,
        )
        split_logger.close()

        api_key = os.environ.get("GEMINI_API_EVALUATOR")
        review_model = "gemini-2.5-flash"
        if api_key and wandb_run_id:
            print(f"Triggering Gemini Review for Split {split_id} / Run ID: {wandb_run_id}")
            review_process = subprocess.Popen(
                [
                    "python",
                    "analysis/program_review.py",
                    "--input_file",
                    current_log_file,
                    "--api_key",
                    api_key,
                    "--wandb_run_id",
                    wandb_run_id,
                    "--model",
                    review_model,
                    "--metadata_path",
                    meta_data_name,
                ]
            )
            review_processes.append(review_process)
        else:
            print(f"Split {split_id}: skipping review due to missing API Key or WandB Run ID.")

    if classification_metrics_sink:
        aggregate_rows = aggregate_sample_metrics(classification_metrics_sink)
    else:
        aggregate_rows = []
    if aggregate_rows:
        aggregate_run_name = (
            f"{args.wandb_run_name}_aggregate"
            if args.wandb_run_name
            else f"{args.log_path}_aggregate"
        )
        aggregate_config = {
            "problem_name": problem_name,
            "outer_splits": args.outer_splits,
            "is_time_series": is_time_series,
            "time_series_train_window": args.time_series_train_window,
            "time_series_test_window": args.time_series_test_window,
            "time_series_step_window": args.time_series_step_window,
            "time_series_gap": args.time_series_gap,
        }
        aggregate_run_id = log_aggregate_run(
            wandb_api=wandb,
            project=args.wandb_project,
            group=args.wandb_group,
            run_name=aggregate_run_name,
            run_config=aggregate_config,
            rows=aggregate_rows,
            run_type="llmfe_classification_aggregate",
        )
        print(f"Aggregate comparison run complete. W&B run id: {aggregate_run_id}")
    elif (not is_regression) and problem_name in {
        "btc-classification",
        "btc_classification",
    }:
        print("Warning: no per-sample classification metrics were collected for aggregate logging.")

    print("Training loop finished. Waiting for all review uploads to complete...")
    for review_process in review_processes:
        review_process.wait()
    print("All processes finished. Exiting.")
