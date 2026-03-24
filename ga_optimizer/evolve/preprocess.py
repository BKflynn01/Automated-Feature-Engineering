from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Callable, Dict, List, Optional, Set, TextIO, Tuple, cast

import numpy as np
import pandas as pd

from ga_optimizer.config import DEFAULT_GA_CONFIG, GAOptimizerConfig


@dataclass
class FeatureCandidate:
    island_id: int
    score: float
    function_code: str
    sample_order: int
    source_file: str


@dataclass
class ExecutedCandidate:
    candidate: FeatureCandidate
    output_columns: Tuple[str, ...]
    semantic_hash: str
    output_df: Optional[pd.DataFrame] = None


def load_candidates(
    samples_dir: str,
    config: Optional[GAOptimizerConfig] = None,
    source_file_glob: Optional[str] = None,
) -> List[FeatureCandidate]:
    """Load feature candidates from JSON files."""
    if not os.path.exists(samples_dir):
        raise ValueError(f"Sample directory {samples_dir} does not exist")

    resolved_config = config or DEFAULT_GA_CONFIG
    patterns = resolved_config.candidate_load.json_glob_patterns
    files = sorted(
        {fpath for pattern in patterns for fpath in glob.glob(os.path.join(samples_dir, pattern))}
    )
    if source_file_glob:
        files = [
            fpath
            for fpath in files
            if fnmatch(os.path.relpath(fpath, samples_dir), source_file_glob)
        ]
    if not files:
        if source_file_glob:
            raise ValueError(
                f"No JSON files found in {samples_dir} matching source_file_glob='{source_file_glob}'"
            )
        raise ValueError(f"No JSON files found in {samples_dir}")

    candidates: List[FeatureCandidate] = []
    failed_reasons: Counter[str] = Counter()
    failure_examples: Dict[str, List[str]] = defaultdict(list)

    def record_failure(reason: str, rel_path: str) -> None:
        failed_reasons[reason] += 1
        if len(failure_examples[reason]) < 3:
            failure_examples[reason].append(rel_path)

    for fp in files:
        rel = os.path.relpath(fp, samples_dir)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            record_failure("json_decode_error", rel)
            continue
        except OSError:
            record_failure("file_read_error", rel)
            continue

        if not isinstance(data, dict):
            record_failure("invalid_json_root_type", rel)
            continue

        function_code = data.get("function_code", data.get("function"))
        if function_code is None:
            record_failure("missing_function_or_function_code", rel)
            continue

        raw_island_id = data.get("island_id")
        raw_score = data.get("score")
        raw_sample_order = data.get("sample_order")

        if raw_island_id is None:
            record_failure("null_or_missing_island_id", rel)
            continue

        if raw_score is None:
            record_failure("null_or_missing_score", rel)
            continue

        if raw_sample_order is None:
            record_failure("null_or_missing_sample_order", rel)
            continue

        try:
            candidate = FeatureCandidate(
                island_id=int(raw_island_id),
                score=float(raw_score),
                function_code=str(function_code),
                sample_order=int(raw_sample_order),
                source_file=os.path.relpath(fp, samples_dir),
            )
            candidates.append(candidate)
        except (KeyError, ValueError, TypeError):
            record_failure("type_conversion_error", rel)

    if not candidates:
        raise ValueError(f"No valid candidates found in {samples_dir}")
    failed_count = sum(failed_reasons.values())
    print(
        f"Candidate load summary: total={len(files)}, valid={len(candidates)}, failed={failed_count}"
    )
    if failed_count > 0:
        print("Candidate load failure reasons:")
        for reason, count in failed_reasons.most_common():
            examples = ", ".join(failure_examples[reason]) if failure_examples[reason] else "n/a"
            print(f"  - {reason}: {count} (examples: {examples})")
    return candidates


def select_top_k_per_island(
    candidates: List[FeatureCandidate],
    k: Optional[int] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> List[FeatureCandidate]:
    """Select top-k candidates per island by descending score."""
    resolved_config = config or DEFAULT_GA_CONFIG
    effective_k = k if k is not None else resolved_config.selection.default_top_k_per_island
    if effective_k <= 0:
        raise ValueError("k must be > 0")

    by_island: Dict[int, List[FeatureCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_island[candidate.island_id].append(candidate)

    selected: List[FeatureCandidate] = []
    for island_candidates in by_island.values():
        sorted_candidates = sorted(
            island_candidates,
            key=lambda c: (c.score, -c.sample_order),
            reverse=True,
        )
        selected.extend(sorted_candidates[:effective_k])
    return selected


def _candidate_id(candidate: FeatureCandidate) -> str:
    return f"is{candidate.island_id}_s{candidate.sample_order}::{candidate.source_file}"


def _is_better_candidate(candidate: FeatureCandidate, current: FeatureCandidate) -> bool:
    if candidate.score > current.score:
        return True
    if candidate.score == current.score and candidate.sample_order < current.sample_order:
        return True
    return False


def _sample_output_for_semantic_hash(
    df_out: pd.DataFrame, config: Optional[GAOptimizerConfig] = None
) -> pd.DataFrame:
    resolved_config = config or DEFAULT_GA_CONFIG
    n_rows = len(df_out.index)
    if n_rows == 0:
        return df_out.copy()

    k = max(
        resolved_config.dedup.semantic_min_rows,
        int(np.ceil(n_rows * resolved_config.dedup.semantic_sample_fraction)),
    )
    if 2 * k >= n_rows:
        return df_out.copy()

    head = df_out.head(k)
    tail = df_out.tail(k)
    return pd.concat([head, tail], axis=0)


def _build_semantic_hash(
    df_out: pd.DataFrame,
    rounding_decimals: Optional[int] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> str:
    resolved_config = config or DEFAULT_GA_CONFIG
    effective_rounding = (
        rounding_decimals
        if rounding_decimals is not None
        else resolved_config.dedup.rounding_decimals
    )
    sampled = _sample_output_for_semantic_hash(df_out, config=resolved_config)
    coerced = sampled.apply(pd.to_numeric, errors="coerce")
    arr = coerced.to_numpy(dtype=float, copy=False)
    arr = np.nan_to_num(
        arr,
        nan=resolved_config.dedup.semantic_hash_nan_fill,
        posinf=resolved_config.dedup.semantic_hash_posinf_fill,
        neginf=resolved_config.dedup.semantic_hash_neginf_fill,
    )
    arr = np.round(arr, decimals=effective_rounding)
    payload = (
        f"shape={df_out.shape[0]}x{df_out.shape[1]}|sample={arr.shape[0]}x{arr.shape[1]}|".encode(
            "utf-8"
        )
        + arr.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _extract_generated_feature_frame(
    df_out: pd.DataFrame,
    base_column_names: Set[str],
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    col_names = [str(c) for c in df_out.columns]
    label_name = str(label_column) if label_column is not None else None
    keep_mask = [
        col_name not in base_column_names and (label_name is None or col_name != label_name)
        for col_name in col_names
    ]
    generated_df = df_out.loc[:, keep_mask].copy()
    generated_df.columns = [col_name for col_name, keep in zip(col_names, keep_mask) if keep]
    return generated_df


def _build_feature_semantic_hash(
    series: pd.Series,
    rounding_decimals: Optional[int] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> str:
    return _build_semantic_hash(
        series.to_frame(name="feature"),
        rounding_decimals=rounding_decimals,
        config=config,
    )


def _candidate_key(candidate: FeatureCandidate) -> Tuple[int, int, str]:
    return (candidate.island_id, candidate.sample_order, candidate.source_file)


def _feature_name(item: ExecutedCandidate) -> str:
    if not item.output_columns:
        return ""
    return item.output_columns[0]


def _feature_id(item: ExecutedCandidate) -> str:
    return f"{_candidate_id(item.candidate)}::{_feature_name(item)}"


def _append_report_line(report_file: TextIO, line: str) -> None:
    report_file.write(line + "\n")


def _format_output_template(template: str, context: Dict[str, object], template_name: str) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Invalid {template_name}: missing placeholder '{missing_key}'") from exc


def _derive_execution_input(df: pd.DataFrame, label_column: Optional[str]) -> pd.DataFrame:
    if label_column is not None and label_column in df.columns:
        return df.drop(columns=[label_column])
    return df


def execute_candidates_for_dedup(
    candidates: List[FeatureCandidate],
    df_input: pd.DataFrame,
    report_file: TextIO,
    rounding_decimals: Optional[int] = None,
    label_column: Optional[str] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> Tuple[List[ExecutedCandidate], int]:
    resolved_config = config or DEFAULT_GA_CONFIG
    executed: List[ExecutedCandidate] = []
    dropped_count = 0
    base_column_names = {str(c) for c in df_input.columns}
    for candidate in candidates:
        cid = _candidate_id(candidate)
        try:
            df_out = _execute_candidate(candidate, df_input, config=resolved_config)
            generated_df = _extract_generated_feature_frame(
                df_out=df_out,
                base_column_names=base_column_names,
                label_column=label_column,
            )
            for col, series in generated_df.items():
                executed.append(
                    ExecutedCandidate(
                        candidate=candidate,
                        output_columns=(str(col),),
                        semantic_hash=_build_feature_semantic_hash(
                            series,
                            rounding_decimals=rounding_decimals,
                            config=resolved_config,
                        ),
                        output_df=generated_df,
                    )
                )
            _append_report_line(
                report_file,
                (
                    f"stage=execute action=kept candidate={cid} "
                    f"generated_features={len(generated_df.columns)} reason=execution_success"
                ),
            )
        except Exception as exc:
            dropped_count += 1
            _append_report_line(
                report_file,
                (
                    f"stage=execute action=dropped candidate={cid} "
                    f"reason=execution_failed error={str(exc).replace(chr(10), ' ')}"
                ),
            )
    return executed, dropped_count


def dedup_stage_by_output_columns(
    executed: List[ExecutedCandidate], report_file: TextIO
) -> Tuple[List[ExecutedCandidate], int]:
    best_by_columns: Dict[Tuple[str, ...], ExecutedCandidate] = {}
    dropped_count = 0
    for item in executed:
        current = best_by_columns.get(item.output_columns)
        fid = _feature_id(item)
        if current is None:
            best_by_columns[item.output_columns] = item
            _append_report_line(
                report_file,
                f"stage=dedup_columns action=kept feature={fid} reason=unique_feature_name",
            )
            continue

        current_id = _feature_id(current)
        if _is_better_candidate(item.candidate, current.candidate):
            dropped_count += 1
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_columns action=dropped feature={current_id} "
                    f"compared_to={fid} reason=lower_score_same_feature_name"
                ),
            )
            best_by_columns[item.output_columns] = item
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_columns action=kept feature={fid} "
                    f"reason=higher_score_same_feature_name"
                ),
            )
            continue

        dropped_count += 1
        _append_report_line(
            report_file,
            (
                f"stage=dedup_columns action=dropped feature={fid} "
                f"compared_to={current_id} reason=lower_or_equal_score_same_feature_name"
            ),
        )

    survivors = list(best_by_columns.values())
    return survivors, dropped_count


def dedup_stage_by_semantic_hash(
    executed: List[ExecutedCandidate], report_file: TextIO
) -> Tuple[List[ExecutedCandidate], int]:
    best_by_hash: Dict[str, ExecutedCandidate] = {}
    dropped_count = 0
    for item in executed:
        current = best_by_hash.get(item.semantic_hash)
        fid = _feature_id(item)
        if current is None:
            best_by_hash[item.semantic_hash] = item
            _append_report_line(
                report_file,
                f"stage=dedup_semantic_hash action=kept feature={fid} reason=unique_feature_output_signature",
            )
            continue

        current_id = _feature_id(current)
        if _is_better_candidate(item.candidate, current.candidate):
            dropped_count += 1
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_semantic_hash action=dropped feature={current_id} "
                    f"compared_to={fid} reason=lower_score_same_feature_output_signature"
                ),
            )
            best_by_hash[item.semantic_hash] = item
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_semantic_hash action=kept feature={fid} "
                    "reason=higher_score_same_feature_output_signature"
                ),
            )
            continue

        dropped_count += 1
        _append_report_line(
            report_file,
            (
                f"stage=dedup_semantic_hash action=dropped feature={fid} "
                f"compared_to={current_id} reason=lower_or_equal_score_same_feature_output_signature"
            ),
        )

    survivors = sorted(
        best_by_hash.values(),
        key=lambda i: (
            i.candidate.island_id,
            -i.candidate.score,
            i.candidate.sample_order,
            _feature_name(i),
        ),
    )
    return survivors, dropped_count


def deduplicate_candidates_multistage(
    candidates: List[FeatureCandidate],
    execution_input: pd.DataFrame,
    report_path: str,
    rounding_decimals: Optional[int] = None,
    label_column: Optional[str] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> Tuple[List[ExecutedCandidate], Dict[str, int]]:
    resolved_config = config or DEFAULT_GA_CONFIG
    effective_rounding = (
        rounding_decimals
        if rounding_decimals is not None
        else resolved_config.dedup.rounding_decimals
    )
    summary = {
        "total_input_candidates": len(candidates),
        "total_input": 0,
        "dropped_execute": 0,
        "dropped_stage_1_columns": 0,
        "dropped_stage_2_semantic_hash": 0,
        "total_survivors": 0,
    }

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        _append_report_line(
            report_file,
            "=== Multi-Stage Feature Deduplication Report ===",
        )
        _append_report_line(
            report_file,
            (
                f"config rounding_decimals={effective_rounding} "
                f"semantic_window_policy={resolved_config.dedup.semantic_sample_fraction:.2f}"
                "_head_and_tail "
                "column_signature_scope=feature_level_generated_only "
                f"label_column={label_column}"
            ),
        )

        executed, dropped_execute = execute_candidates_for_dedup(
            candidates,
            df_input=execution_input,
            report_file=report_file,
            rounding_decimals=effective_rounding,
            label_column=label_column,
            config=resolved_config,
        )
        summary["total_input"] = len(executed)
        summary["dropped_execute"] = dropped_execute

        stage1, dropped_stage1 = dedup_stage_by_output_columns(executed, report_file=report_file)
        summary["dropped_stage_1_columns"] = dropped_stage1

        stage2, dropped_stage2 = dedup_stage_by_semantic_hash(stage1, report_file=report_file)
        summary["dropped_stage_2_semantic_hash"] = dropped_stage2

        survivors = stage2
        summary["total_survivors"] = len(survivors)
        summary["total_survivor_candidates"] = len({_candidate_key(i.candidate) for i in survivors})

        _append_report_line(report_file, "")
        _append_report_line(report_file, "=== Summary ===")
        _append_report_line(
            report_file, f"total_input_candidates={summary['total_input_candidates']}"
        )
        _append_report_line(report_file, f"total_input={summary['total_input']}")
        _append_report_line(report_file, f"dropped_execute={summary['dropped_execute']}")
        _append_report_line(
            report_file, f"dropped_stage_1_columns={summary['dropped_stage_1_columns']}"
        )
        _append_report_line(
            report_file,
            f"dropped_stage_2_semantic_hash={summary['dropped_stage_2_semantic_hash']}",
        )
        _append_report_line(report_file, f"total_survivors={summary['total_survivors']}")
        _append_report_line(
            report_file,
            f"total_survivor_candidates={summary['total_survivor_candidates']}",
        )

    print("Deduplication stage summary:")
    print(f"  input_candidates={summary['total_input_candidates']}")
    print(f"  input_features={summary['total_input']}")
    print(f"  dropped_execute={summary['dropped_execute']}")
    print(f"  dropped_stage_1={summary['dropped_stage_1_columns']}")
    print(f"  dropped_stage_2={summary['dropped_stage_2_semantic_hash']}")
    print(f"  survivor_features={summary['total_survivors']}")
    print(f"Dedup report path: {report_path}")
    return survivors, summary


def _execute_candidate(
    candidate: FeatureCandidate,
    df_input: pd.DataFrame,
    config: Optional[GAOptimizerConfig] = None,
) -> pd.DataFrame:
    resolved_config = config or DEFAULT_GA_CONFIG
    namespace: Dict[str, object] = {"pd": pd}
    # Candidate code execution is required by the optimizer pipeline.
    exec(candidate.function_code, namespace)  # nosec B102

    function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    for function_name in resolved_config.execution.preferred_function_names:
        candidate_function = namespace.get(function_name)
        if callable(candidate_function):
            function = cast(Callable[[pd.DataFrame], pd.DataFrame], candidate_function)
            break
    if function is None:
        raise ValueError("Candidate must define modify_features_v2 or modify_features")

    df_out = function(df_input)
    if not isinstance(df_out, pd.DataFrame):
        raise TypeError("Feature function must return a pandas DataFrame")
    if not df_out.index.equals(df_input.index):
        raise ValueError("Returned dataframe index must match the input index")
    return df_out


class FeatureExtractionPipeline:
    def __init__(
        self,
        samples_dir: str,
        k_per_island: Optional[int] = None,
        label_column: Optional[str] = None,
        include_original: bool = True,
        data_dir: Optional[str] = None,
        config: Optional[GAOptimizerConfig] = None,
        source_file_glob: Optional[str] = None,
    ) -> None:
        self.config = config or DEFAULT_GA_CONFIG
        self.samples_dir = samples_dir
        self.k_per_island = (
            k_per_island
            if k_per_island is not None
            else self.config.selection.default_top_k_per_island
        )
        self.label_column = label_column
        self.include_original = include_original
        self.data_dir = data_dir
        self.source_file_glob = source_file_glob

    def run(
        self, df: pd.DataFrame, dataset_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        resolved_label_column = self.label_column
        execution_input = _derive_execution_input(df, resolved_label_column)

        candidates = load_candidates(
            self.samples_dir,
            config=self.config,
            source_file_glob=self.source_file_glob,
        )
        islands = sorted({c.island_id for c in candidates})
        print(f"Loaded candidates: {len(candidates)}")
        print(f"Islands found: {len(islands)} ({', '.join(str(i) for i in islands)})")
        selected = select_top_k_per_island(candidates, k=self.k_per_island, config=self.config)
        print(f"Selected after top-k per island (k={self.k_per_island}): {len(selected)}")
        resolved_dataset_name = dataset_name or self.config.output.default_dataset_name
        data_root_dir = self.data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            self.config.output.default_data_dir_name,
        )
        data_dir = os.path.join(data_root_dir, resolved_dataset_name)
        report_name = _format_output_template(
            self.config.output.dedup_report_filename_template,
            {"dataset_name": resolved_dataset_name, "k_per_island": self.k_per_island},
            template_name="dedup_report_filename_template",
        )
        report_path = os.path.join(data_dir, report_name)
        deduped_features, _ = deduplicate_candidates_multistage(
            selected,
            execution_input=execution_input,
            report_path=report_path,
            rounding_decimals=self.config.dedup.rounding_decimals,
            label_column=resolved_label_column,
            config=self.config,
        )
        manifest_name = _format_output_template(
            self.config.output.manifest_filename_template,
            {"dataset_name": resolved_dataset_name, "k_per_island": self.k_per_island},
            template_name="manifest_filename_template",
        )
        filtered_manifest_path = os.path.join(data_dir, manifest_name)
        pd.DataFrame(
            [
                {
                    "island_id": item.candidate.island_id,
                    "sample_order": item.candidate.sample_order,
                    "score": item.candidate.score,
                    "source_file": item.candidate.source_file,
                    "feature_name": _feature_name(item),
                    "feature_hash": item.semantic_hash,
                }
                for item in deduped_features
            ]
        ).to_csv(filtered_manifest_path, index=False)
        print(f"Filtered feature manifest: {filtered_manifest_path}")
        return self.build_full_dataframe(
            df=df,
            selected_features=deduped_features,
            label_column=resolved_label_column,
            include_original=self.include_original,
            execution_input=execution_input,
            config=self.config,
        )

    @staticmethod
    def build_full_dataframe(
        df: pd.DataFrame,
        candidates: Optional[List[FeatureCandidate]] = None,
        selected_features: Optional[List[ExecutedCandidate]] = None,
        label_column: Optional[str] = None,
        include_original: bool = True,
        execution_input: Optional[pd.DataFrame] = None,
        config: Optional[GAOptimizerConfig] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        resolved_config = config or DEFAULT_GA_CONFIG
        generated_parts: List[pd.DataFrame] = []
        metadata_rows: List[Dict[str, object]] = []
        runtime_input = (
            execution_input
            if execution_input is not None
            else _derive_execution_input(df, label_column)
        )
        base_cols = {str(c) for c in runtime_input.columns}
        if label_column is not None:
            base_cols.discard(str(label_column))

        if (candidates is None) == (selected_features is None):
            raise ValueError("Exactly one of candidates or selected_features must be provided")

        ordered_candidates: List[FeatureCandidate] = []
        allowed_features_by_candidate: Dict[Tuple[int, int, str], Set[str]] = {}
        cached_output_by_candidate: Dict[Tuple[int, int, str], pd.DataFrame] = {}
        if selected_features is not None:
            for item in selected_features:
                key = _candidate_key(item.candidate)
                if key not in allowed_features_by_candidate:
                    allowed_features_by_candidate[key] = set()
                    ordered_candidates.append(item.candidate)
                allowed_features_by_candidate[key].add(_feature_name(item))
                if item.output_df is not None and key not in cached_output_by_candidate:
                    cached_output_by_candidate[key] = item.output_df
        if candidates is not None:
            ordered_candidates = candidates

        for candidate in ordered_candidates:
            try:
                key = _candidate_key(candidate)

                cached_out = cached_output_by_candidate.get(key)
                out = (
                    cached_out
                    if cached_out is not None
                    else _execute_candidate(candidate, runtime_input, config=resolved_config)
                )
                out = _extract_generated_feature_frame(
                    df_out=out,
                    base_column_names=base_cols,
                    label_column=label_column,
                )
                if selected_features is not None:
                    allowed = allowed_features_by_candidate.get(key, set())
                    allowed_mask = [c in allowed for c in out.columns]
                    out = out.loc[:, allowed_mask]
                generated_parts.append(out)
                metadata_rows.append(
                    {
                        "island_id": candidate.island_id,
                        "sample_order": candidate.sample_order,
                        "score": candidate.score,
                        "source_file": candidate.source_file,
                        "generated_columns": list(out.columns),
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as exc:
                metadata_rows.append(
                    {
                        "island_id": candidate.island_id,
                        "sample_order": candidate.sample_order,
                        "score": candidate.score,
                        "source_file": candidate.source_file,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        generated_df = (
            pd.concat(generated_parts, axis=1) if generated_parts else pd.DataFrame(index=df.index)
        )

        label_series = None
        if label_column and label_column in df.columns:
            label_series = df[label_column].copy()

        parts: List[pd.DataFrame] = []
        if include_original:
            if label_series is not None:
                parts.append(df.drop(columns=[label_column]))
            else:
                parts.append(df.copy())
        parts.append(generated_df)

        final_df = pd.concat(parts, axis=1)
        if label_series is not None:
            final_df[label_column] = label_series.values

        metadata_df = pd.DataFrame(metadata_rows)
        return final_df, metadata_df
