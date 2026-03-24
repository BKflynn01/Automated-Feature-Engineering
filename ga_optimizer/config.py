from __future__ import annotations

from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import yaml


@dataclass(frozen=True)
class MainConfig:
    default_sep: str = ","
    default_k_per_island: int = 2
    default_exclude_original: bool = False
    default_use_last_column_as_label: bool = False


@dataclass(frozen=True)
class CandidateLoadConfig:
    json_glob_patterns: tuple[str, ...] = (
        "*.json",
        "samples/*.json",
        "*_split_*/samples/*.json",
    )


@dataclass(frozen=True)
class SelectionConfig:
    default_top_k_per_island: int = 2


@dataclass(frozen=True)
class DedupConfig:
    rounding_decimals: int = 8
    semantic_sample_fraction: float = 0.25
    semantic_min_rows: int = 1
    semantic_hash_nan_fill: float = 0.0
    semantic_hash_posinf_fill: float = 0.0
    semantic_hash_neginf_fill: float = 0.0


@dataclass(frozen=True)
class ExecutionConfig:
    preferred_function_names: tuple[str, ...] = (
        "modify_features_v2",
        "modify_features",
    )


@dataclass(frozen=True)
class OutputConfig:
    dedup_report_filename_template: str = "dedup_report_{dataset_name}.txt"
    manifest_filename_template: str = "top_{k_per_island}_samples.csv"
    default_dataset_name: str = "dataset"
    default_data_dir_name: str = "data"


@dataclass(frozen=True)
class GADataConfig:
    default_csv_sep: str = ","
    selected_gene_value: int = 1


@dataclass(frozen=True)
class XGBRFConfig:
    n_estimators: int | None = None
    subsample: float | None = None
    colsample_bynode: float | None = None
    random_state: int = 42
    verbosity: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class EvaluatorConfig:
    cv_folds: int = 4
    time_series_train_window: int | None = None
    time_series_test_window: int | None = None
    time_series_step_window: int | None = None
    time_series_gap: int = 0
    max_time_series_splits: int | None = None


@dataclass(frozen=True)
class ToolboxConfig:
    fitness_weights: tuple[float, ...] = (1.0,)
    attr_bool_min: int = 0
    attr_bool_max: int = 1
    population_size: int = 50
    dynamic_population_min: int = 8
    dynamic_population_max: int = 64
    dynamic_population_ratio: float = 1.0
    cx_prob: float = 0.5
    mut_prob: float = 0.2
    tournament_size: int = 3


@dataclass(frozen=True)
class RunnerConfig:
    n_generations: int = 40
    verbose: bool = True
    n_jobs: int = 1
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
    best_features_filename: str = "ga_best_features.txt"
    best_features_csv_filename: str = "ga_best_features.csv"
    best_feature_dataset_filename: str = "ga_best_feature_dataset.csv"
    generation_trace_filename: str = "ga_generation_trace.csv"
    logbook_filename: str = "ga_logbook.csv"
    run_manifest_filename: str = "ga_run_manifest.json"
    default_classification_scoring: str = "accuracy"
    default_regression_scoring: str = "neg_root_mean_squared_error"
    default_time_series_regression_scoring: str = "neg_normalized_root_mean_squared_error"


@dataclass(frozen=True)
class GARunProfile:
    name: str
    dataset_name: str
    input_csv: str
    label_column: str
    task: Literal["classification", "regression"]
    scoring: str | None = None
    is_time_series: bool = False
    time_series_train_window: int | None = None
    time_series_test_window: int | None = None
    time_series_step_window: int | None = None
    time_series_gap: int = 0
    n_generations: int = 40
    population_size: int = 50
    cx_prob: float = 0.5
    mut_prob: float = 0.2
    tournament_size: int = 3
    cv_folds: int = 4
    n_estimators: int | None = None
    random_state: int = 42
    sep: str = ","


class GAPreset(str, Enum):
    QUICK = "quick"
    NORMAL = "normal"
    EXTENDED = "extended"


@dataclass(frozen=True)
class GAPresetValues:
    preset: GAPreset
    n_generations: int
    population_size: int
    cv_folds: int
    n_estimators: int | None
    cx_prob: float
    mut_prob: float
    tournament_size: int
    max_time_series_splits: int | None
    early_stop_patience: int | None
    early_stop_min_delta: float


@dataclass(frozen=True)
class GAPresetConfig:
    default_preset: GAPreset = GAPreset.NORMAL
    presets: tuple[GAPresetValues, ...] = field(
        default_factory=lambda: (
            GAPresetValues(
                preset=GAPreset.QUICK,
                n_generations=2,
                population_size=6,
                cv_folds=4,
                n_estimators=None,
                cx_prob=0.5,
                mut_prob=0.2,
                tournament_size=3,
                max_time_series_splits=None,
                early_stop_patience=1,
                early_stop_min_delta=0.0,
            ),
            GAPresetValues(
                preset=GAPreset.NORMAL,
                n_generations=15,
                population_size=0,
                cv_folds=4,
                n_estimators=None,
                cx_prob=0.5,
                mut_prob=0.2,
                tournament_size=3,
                max_time_series_splits=None,
                early_stop_patience=5,
                early_stop_min_delta=0.0,
            ),
            GAPresetValues(
                preset=GAPreset.EXTENDED,
                n_generations=40,
                population_size=0,
                cv_folds=4,
                n_estimators=None,
                cx_prob=0.5,
                mut_prob=0.2,
                tournament_size=3,
                max_time_series_splits=None,
                early_stop_patience=10,
                early_stop_min_delta=0.0,
            ),
        )
    )


@dataclass(frozen=True)
class DatasetRunConfig:
    dataset_name: str
    input_csv: str
    label_column: str
    task: Literal["classification", "regression"]
    scoring: str | None = None
    is_time_series: bool = False
    time_series_train_window: int | None = None
    time_series_test_window: int | None = None
    time_series_step_window: int | None = None
    time_series_gap: int = 0
    random_state: int = 42
    sep: str = ","
    output_dir: str | None = None


@dataclass(frozen=True)
class GAProfileConfig:
    default_profile: str = "btc_classification"
    profiles: tuple[GARunProfile, ...] = field(
        default_factory=lambda: (
            GARunProfile(
                name="btc_classification",
                dataset_name="btc-classification",
                input_csv="./ga_optimizer/data/btc-classification/features.csv",
                label_column="price_direction",
                task="classification",
                scoring="accuracy",
                is_time_series=True,
                time_series_train_window=365,
                time_series_test_window=1,
                time_series_step_window=3,
                time_series_gap=0,
                n_generations=40,
                population_size=50,
                cx_prob=0.5,
                mut_prob=0.2,
                tournament_size=3,
                cv_folds=4,
                n_estimators=None,
                random_state=42,
                sep=",",
            ),
            GARunProfile(
                name="btc_classification_quick",
                dataset_name="btc-classification",
                input_csv="./ga_optimizer/data/btc-classification/features.csv",
                label_column="price_direction",
                task="classification",
                scoring="accuracy",
                is_time_series=True,
                time_series_train_window=365,
                time_series_test_window=1,
                time_series_step_window=3,
                time_series_gap=0,
                n_generations=4,
                population_size=0,
                cx_prob=0.5,
                mut_prob=0.2,
                tournament_size=3,
                cv_folds=4,
                n_estimators=50,
                random_state=42,
                sep=",",
            ),
        )
    )


@dataclass(frozen=True)
class GAOptimizerConfig:
    main: MainConfig = field(default_factory=MainConfig)
    candidate_load: CandidateLoadConfig = field(default_factory=CandidateLoadConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ga_data: GADataConfig = field(default_factory=GADataConfig)
    xgbrf: XGBRFConfig = field(default_factory=XGBRFConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    toolbox: ToolboxConfig = field(default_factory=ToolboxConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    ga_presets: GAPresetConfig = field(default_factory=GAPresetConfig)
    ga_profiles: GAProfileConfig = field(default_factory=GAProfileConfig)


DEFAULT_GA_CONFIG = GAOptimizerConfig()


def get_ga_preset_values(
    preset: str | GAPreset,
    config: GAOptimizerConfig = DEFAULT_GA_CONFIG,
) -> GAPresetValues:
    preset_value = GAPreset(str(preset))
    for preset_entry in config.ga_presets.presets:
        if preset_entry.preset == preset_value:
            return preset_entry
    available = ", ".join(sorted(p.value for p in GAPreset))
    raise ValueError(f"Unknown GA preset '{preset_value}'. Available presets: {available}")


def get_ga_profile(
    profile_name: str,
    config: GAOptimizerConfig = DEFAULT_GA_CONFIG,
) -> GARunProfile:
    for profile in config.ga_profiles.profiles:
        if profile.name == profile_name:
            return profile
    available = ", ".join(sorted(profile.name for profile in config.ga_profiles.profiles))
    raise ValueError(f"Unknown GA profile '{profile_name}'. Available profiles: {available}")


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid dataset config: '{key}' must be a non-empty string")
    return value


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid dataset config: '{key}' must be a string when provided")
    return value


def _optional_bool(data: dict[str, object], key: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Invalid dataset config: '{key}' must be a boolean when provided")
    return value


def _optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid dataset config: '{key}' must be an integer when provided")
    return int(value)


def _optional_non_negative_int(data: dict[str, object], key: str) -> int | None:
    value = _optional_int(data, key)
    if value is not None and value < 0:
        raise ValueError(f"Invalid dataset config: '{key}' must be >= 0")
    return value


def load_dataset_run_config(path: str | Path) -> DatasetRunConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Dataset config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise ValueError(f"Unable to read dataset config file: {config_path}") from exc

    if yaml is not None:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in dataset config: {config_path}") from exc
    else:
        raw = _parse_simple_yaml_mapping(text, config_path=config_path)

    if not isinstance(raw, dict):
        raise ValueError("Invalid dataset config: root YAML node must be a mapping/object")
    data = dict(raw)

    allowed_keys = {
        "dataset_name",
        "input_csv",
        "label_column",
        "task",
        "scoring",
        "is_time_series",
        "time_series_train_window",
        "time_series_test_window",
        "time_series_step_window",
        "time_series_gap",
        "random_state",
        "sep",
        "output_dir",
    }
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Invalid dataset config: unknown keys: {joined}")

    dataset_name = _require_str(data, "dataset_name")
    input_csv = _require_str(data, "input_csv")
    label_column = _require_str(data, "label_column")
    task_raw = _require_str(data, "task")
    if task_raw not in ("classification", "regression"):
        raise ValueError(
            "Invalid dataset config: 'task' must be either 'classification' or 'regression'"
        )
    task: Literal["classification", "regression"] = task_raw  # type: ignore[assignment]

    scoring = _optional_str(data, "scoring")
    is_time_series = _optional_bool(data, "is_time_series")
    time_series_train_window = _optional_int(data, "time_series_train_window")
    time_series_test_window = _optional_int(data, "time_series_test_window")
    time_series_step_window = _optional_int(data, "time_series_step_window")
    time_series_gap = _optional_non_negative_int(data, "time_series_gap")
    random_state = _optional_int(data, "random_state")
    sep = _optional_str(data, "sep")
    output_dir = _optional_str(data, "output_dir")

    if not Path(input_csv).is_absolute():
        input_csv = str((config_path.parent / input_csv).resolve())
    if output_dir is not None and not Path(output_dir).is_absolute():
        output_dir = str((config_path.parent / output_dir).resolve())

    return DatasetRunConfig(
        dataset_name=dataset_name,
        input_csv=input_csv,
        label_column=label_column,
        task=task,
        scoring=scoring,
        is_time_series=is_time_series if is_time_series is not None else False,
        time_series_train_window=time_series_train_window,
        time_series_test_window=time_series_test_window,
        time_series_step_window=time_series_step_window,
        time_series_gap=(
            time_series_gap
            if time_series_gap is not None
            else DEFAULT_GA_CONFIG.evaluator.time_series_gap
        ),
        random_state=(
            random_state if random_state is not None else DEFAULT_GA_CONFIG.xgbrf.random_state
        ),
        sep=sep if sep is not None else DEFAULT_GA_CONFIG.ga_data.default_csv_sep,
        output_dir=output_dir,
    )


def _parse_simple_yaml_mapping(text: str, *, config_path: Path) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid YAML in dataset config: {config_path}")
        key_part, value_part = line.split(":", 1)
        key = key_part.strip()
        value_text = value_part.strip()
        if not key:
            raise ValueError(f"Invalid YAML in dataset config: {config_path}")
        parsed[key] = _parse_simple_yaml_scalar(value_text)
    return parsed


def _parse_simple_yaml_scalar(value_text: str) -> object:
    if value_text in {"", "null", "Null", "NULL", "~"}:
        return None
    if value_text in {"true", "True", "TRUE"}:
        return True
    if value_text in {"false", "False", "FALSE"}:
        return False
    if len(value_text) >= 2 and value_text[0] == value_text[-1] and value_text[0] in {"'", '"'}:
        return value_text[1:-1]
    try:
        return int(value_text)
    except ValueError:
        return value_text
