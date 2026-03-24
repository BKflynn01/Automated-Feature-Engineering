from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ga_optimizer.config import (
    DEFAULT_GA_CONFIG,
    GAPreset,
    DatasetRunConfig,
    get_ga_preset_values,
    load_dataset_run_config,
)
from ga_optimizer.ga.runner import run_ga


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GA feature selection over merged features produced by ga_optimizer.main."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to dataset YAML config (required).",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_GA_CONFIG.ga_presets.default_preset.value,
        choices=tuple(p.value for p in GAPreset),
        help="Universal GA preset to apply to the loaded dataset config.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name used for dataset-local output directory",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Optional override for merged feature CSV path",
    )
    parser.add_argument(
        "--label_column",
        default=None,
        help="Optional override for label column",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=("classification", "regression"),
        help="Optional override for task type",
    )
    parser.add_argument(
        "--scoring",
        default=None,
        help=(
            "Optional override for scoring metric. Defaults by task: "
            "classification -> accuracy, "
            "regression -> neg_root_mean_squared_error, "
            "time-series regression -> neg_normalized_root_mean_squared_error"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional override for output directory. Default: ga_optimizer/data/<dataset_name>",
    )
    ts_group = parser.add_mutually_exclusive_group()
    ts_group.add_argument(
        "--is_time_series",
        dest="is_time_series",
        action="store_true",
        help="Enable leakage-safe sliding-window CV for time-series data",
    )
    ts_group.add_argument(
        "--no_is_time_series",
        dest="is_time_series",
        action="store_false",
        help="Disable time-series CV and use non-time-series split strategy",
    )
    parser.set_defaults(is_time_series=None)
    parser.add_argument(
        "--time_series_train_window",
        type=int,
        default=None,
        help="Optional override for sliding-window train size (rows)",
    )
    parser.add_argument(
        "--time_series_test_window",
        type=int,
        default=None,
        help="Optional override for sliding-window test size (rows)",
    )
    parser.add_argument(
        "--time_series_step_window",
        type=int,
        default=None,
        help="Optional override for sliding-window step size (rows)",
    )
    parser.add_argument(
        "--time_series_gap",
        type=int,
        default=None,
        help="Optional override for gap size (rows) between train and test windows",
    )
    parser.add_argument(
        "--max_time_series_splits",
        type=int,
        default=None,
        help="Optional cap for number of time-series CV splits (most recent windows retained).",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=None,
        help="Optional override for GA generations",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=None,
        help="Optional override for GA population size",
    )
    parser.add_argument(
        "--cx_prob",
        type=float,
        default=None,
        help="Optional override for crossover probability",
    )
    parser.add_argument(
        "--mut_prob",
        type=float,
        default=None,
        help="Optional override for mutation probability",
    )
    parser.add_argument(
        "--tournament_size",
        type=int,
        default=None,
        help="Optional override for tournament selection size",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=None,
        help="Optional override for CV fold count",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Optional override for XGBoost estimator count",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Optional override for parallel evaluation worker count",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Optional early stopping patience in generations",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=None,
        help="Optional minimum score improvement required to reset early stopping counter",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Optional override for random seed",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Optional override for CSV separator",
    )
    return parser.parse_args()


def _resolve_scoring(
    task: str,
    is_time_series: bool,
    requested_scoring: str | None,
) -> str:
    if requested_scoring is not None:
        return requested_scoring
    if task == "classification":
        return DEFAULT_GA_CONFIG.runner.default_classification_scoring
    if is_time_series:
        return DEFAULT_GA_CONFIG.runner.default_time_series_regression_scoring
    return DEFAULT_GA_CONFIG.runner.default_regression_scoring


def _build_default_output_dir(dataset_name: str) -> Path:
    return Path("ga_optimizer") / "data" / dataset_name


def _apply_cli_overrides(resolved: dict[str, Any], args: argparse.Namespace) -> None:
    override_fields = (
        "dataset_name",
        "input_csv",
        "label_column",
        "task",
        "scoring",
        "time_series_train_window",
        "time_series_test_window",
        "time_series_step_window",
        "time_series_gap",
        "max_time_series_splits",
        "n_generations",
        "population_size",
        "cx_prob",
        "mut_prob",
        "tournament_size",
        "cv_folds",
        "n_estimators",
        "n_jobs",
        "early_stop_patience",
        "early_stop_min_delta",
        "random_state",
        "sep",
    )
    for field_name in override_fields:
        field_value = getattr(args, field_name)
        if field_value is not None:
            resolved[field_name] = field_value

    if args.is_time_series is not None:
        resolved["is_time_series"] = args.is_time_series

    resolved["scoring"] = _resolve_scoring(
        task=str(resolved["task"]),
        is_time_series=bool(resolved["is_time_series"]),
        requested_scoring=resolved.get("scoring"),
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (
            Path(str(resolved["output_dir"]))
            if resolved.get("output_dir")
            else _build_default_output_dir(str(resolved["dataset_name"]))
        )
    )
    resolved["output_dir"] = str(output_dir)


def _dataset_settings(dataset_cfg: DatasetRunConfig) -> dict[str, Any]:
    return {
        "dataset_name": dataset_cfg.dataset_name,
        "input_csv": dataset_cfg.input_csv,
        "label_column": dataset_cfg.label_column,
        "task": dataset_cfg.task,
        "scoring": dataset_cfg.scoring,
        "is_time_series": dataset_cfg.is_time_series,
        "time_series_train_window": dataset_cfg.time_series_train_window,
        "time_series_test_window": dataset_cfg.time_series_test_window,
        "time_series_step_window": dataset_cfg.time_series_step_window,
        "time_series_gap": dataset_cfg.time_series_gap,
        "max_time_series_splits": DEFAULT_GA_CONFIG.evaluator.max_time_series_splits,
        "n_generations": DEFAULT_GA_CONFIG.runner.n_generations,
        "population_size": DEFAULT_GA_CONFIG.toolbox.population_size,
        "cx_prob": DEFAULT_GA_CONFIG.toolbox.cx_prob,
        "mut_prob": DEFAULT_GA_CONFIG.toolbox.mut_prob,
        "tournament_size": DEFAULT_GA_CONFIG.toolbox.tournament_size,
        "cv_folds": DEFAULT_GA_CONFIG.evaluator.cv_folds,
        "n_estimators": DEFAULT_GA_CONFIG.xgbrf.n_estimators,
        "n_jobs": DEFAULT_GA_CONFIG.runner.n_jobs,
        "early_stop_patience": DEFAULT_GA_CONFIG.runner.early_stop_patience,
        "early_stop_min_delta": DEFAULT_GA_CONFIG.runner.early_stop_min_delta,
        "random_state": dataset_cfg.random_state,
        "sep": dataset_cfg.sep,
        "output_dir": dataset_cfg.output_dir,
    }


def resolve_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.config is None:
        raise ValueError("Missing required --config. Legacy --profile mode has been removed.")

    dataset_cfg = load_dataset_run_config(args.config)
    settings = _dataset_settings(dataset_cfg)
    preset = get_ga_preset_values(args.preset)
    settings.update(
        {
            "preset": preset.preset.value,
            "n_generations": preset.n_generations,
            "population_size": preset.population_size,
            "cv_folds": preset.cv_folds,
            "n_estimators": preset.n_estimators,
            "cx_prob": preset.cx_prob,
            "mut_prob": preset.mut_prob,
            "tournament_size": preset.tournament_size,
            "max_time_series_splits": preset.max_time_series_splits,
            "early_stop_patience": preset.early_stop_patience,
            "early_stop_min_delta": preset.early_stop_min_delta,
        }
    )
    _apply_cli_overrides(settings, args)
    return settings


def main() -> None:
    args = parse_args()
    settings = resolve_run_settings(args)

    input_path = Path(settings["input_csv"])
    if not input_path.exists():
        raise ValueError(f"input_csv does not exist: {input_path}")

    print("Resolved GA run settings:")
    print(json.dumps(settings, indent=2, sort_keys=True))
    
    best_features, _best_score = run_ga(
        csv_path=str(input_path),
        label_column=str(settings["label_column"]),
        task=str(settings["task"]),  # type: ignore
        scoring=str(settings["scoring"]),
        output_dir=str(settings["output_dir"]),
        is_time_series=bool(settings["is_time_series"]),
        time_series_train_window=settings["time_series_train_window"],
        time_series_test_window=settings["time_series_test_window"],
        time_series_step_window=settings["time_series_step_window"],
        time_series_gap=int(settings["time_series_gap"]),
        max_time_series_splits=settings["max_time_series_splits"],
        n_generations=int(settings["n_generations"]),
        population_size=int(settings["population_size"]),
        cx_prob=float(settings["cx_prob"]),
        mut_prob=float(settings["mut_prob"]),
        tournament_size=int(settings["tournament_size"]),
        cv_folds=int(settings["cv_folds"]),
        n_estimators=settings["n_estimators"],
        n_jobs=int(settings["n_jobs"]),
        early_stop_patience=settings["early_stop_patience"],
        early_stop_min_delta=float(settings["early_stop_min_delta"]),
        random_state=int(settings["random_state"]),
        sep=str(settings["sep"]),
    )

    print("Final selected features:")
    for feature in best_features:
        print(feature)


if __name__ == "__main__":
    main()
