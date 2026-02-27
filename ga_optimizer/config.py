from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MainConfig:
    default_sep: str = ","
    default_k_per_island: int = 2
    default_exclude_original: bool = False
    default_use_last_column_as_label: bool = False


@dataclass(frozen=True)
class CandidateLoadConfig:
    json_glob_patterns: tuple[str, ...] = ("*.json", "samples/*.json", "*_split_*/samples/*.json")


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
    preferred_function_names: tuple[str, ...] = ("modify_features_v2", "modify_features")


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
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bynode: float = 0.8
    random_state: int = 42
    verbosity: int = 0


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


DEFAULT_GA_CONFIG = GAOptimizerConfig()
