from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureCandidate:
    island_id: int
    score: float
    function_code: str
    sample_order: int
    source_file: str

    def __hash__(self) -> int:
        return hash(self.function_code)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FeatureCandidate) and self.function_code == other.function_code


@dataclass
class ExecutedCandidate:
    candidate: FeatureCandidate
    output_columns: Tuple[str, ...]
    semantic_hash: str


def load_candidates(samples_dir: str) -> List[FeatureCandidate]:
    """Load feature candidates from JSON files."""
    if not os.path.exists(samples_dir):
        raise ValueError(f"Sample directory {samples_dir} does not exist")

    files = sorted(
        set(
            glob.glob(os.path.join(samples_dir, "*.json"))
            + glob.glob(os.path.join(samples_dir, "samples", "*.json"))
            + glob.glob(os.path.join(samples_dir, "*_split_*", "samples", "*.json"))
        )
    )
    if not files:
        raise ValueError(f"No JSON files found in {samples_dir}")

    candidates: List[FeatureCandidate] = []
    failed_count = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            function_code = data.get("function_code", data.get("function"))
            if function_code is None:
                failed_count += 1
                continue

            candidate = FeatureCandidate(
                island_id=int(data["island_id"]),
                score=float(data["score"]),
                function_code=str(function_code),
                sample_order=int(data["sample_order"]),
                source_file=os.path.relpath(fp, samples_dir),
            )
            candidates.append(candidate)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            failed_count += 1

    if not candidates:
        raise ValueError(f"No valid candidates found in {samples_dir}")
    if failed_count > 0:
        print(f"Warning: {failed_count} files failed to load and were skipped")
    return candidates


def select_top_k_per_island(
    candidates: List[FeatureCandidate], k: int = 2
) -> List[FeatureCandidate]:
    """Select top-k candidates per island by descending score."""
    if k <= 0:
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
        selected.extend(sorted_candidates[:k])
    return selected


def deduplicate_candidates(candidates: List[FeatureCandidate]) -> List[FeatureCandidate]:
    """Deduplicate by exact function body, keeping the highest-scoring entry."""
    best_by_code: Dict[str, FeatureCandidate] = {}
    for candidate in candidates:
        existing = best_by_code.get(candidate.function_code)
        if existing is None:
            best_by_code[candidate.function_code] = candidate
            continue
        if candidate.score > existing.score:
            best_by_code[candidate.function_code] = candidate
            continue
        if candidate.score == existing.score and candidate.sample_order < existing.sample_order:
            best_by_code[candidate.function_code] = candidate
    return sorted(best_by_code.values(), key=lambda c: (c.island_id, -c.score, c.sample_order))


def _candidate_id(candidate: FeatureCandidate) -> str:
    return f"is{candidate.island_id}_s{candidate.sample_order}::{candidate.source_file}"


def _is_better_candidate(candidate: FeatureCandidate, current: FeatureCandidate) -> bool:
    if candidate.score > current.score:
        return True
    if candidate.score == current.score and candidate.sample_order < current.sample_order:
        return True
    return False


def _sample_output_for_semantic_hash(df_out: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df_out.index)
    if n_rows == 0:
        return df_out.copy()

    if n_rows >= 40:
        k = 20
    else:
        k = max(1, int(n_rows * 0.25))

    head = df_out.head(k)
    tail = df_out.tail(k)
    return pd.concat([head, tail], axis=0)


def _build_semantic_hash(df_out: pd.DataFrame, rounding_decimals: int = 8) -> str:
    sampled = _sample_output_for_semantic_hash(df_out)
    normalized = (
        sampled.replace([np.inf, -np.inf], 0)
        .apply(lambda col: pd.to_numeric(col, errors="coerce"), axis=0)
        .fillna(0)
        .astype(float)
        .round(rounding_decimals)
    )
    payload = (
        f"shape={df_out.shape[0]}x{df_out.shape[1]}|sample={normalized.shape[0]}x{normalized.shape[1]}|".encode(
            "utf-8"
        )
        + normalized.to_numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _append_report_line(report_file: TextIO, line: str) -> None:
    report_file.write(line + "\n")


def execute_candidates_for_dedup(
    candidates: List[FeatureCandidate],
    df_input: pd.DataFrame,
    report_file: TextIO,
    rounding_decimals: int = 8,
) -> Tuple[List[ExecutedCandidate], int]:
    executed: List[ExecutedCandidate] = []
    dropped_count = 0
    for candidate in candidates:
        cid = _candidate_id(candidate)
        try:
            df_out = _execute_candidate(candidate, df_input)
            executed.append(
                ExecutedCandidate(
                    candidate=candidate,
                    output_columns=tuple(str(c) for c in df_out.columns.tolist()),
                    semantic_hash=_build_semantic_hash(df_out, rounding_decimals=rounding_decimals),
                )
            )
            _append_report_line(
                report_file,
                f"stage=execute action=kept candidate={cid} reason=execution_success",
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
        cid = _candidate_id(item.candidate)
        if current is None:
            best_by_columns[item.output_columns] = item
            _append_report_line(
                report_file,
                f"stage=dedup_columns action=kept candidate={cid} reason=unique_ordered_column_signature",
            )
            continue

        current_id = _candidate_id(current.candidate)
        if _is_better_candidate(item.candidate, current.candidate):
            dropped_count += 1
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_columns action=dropped candidate={current_id} "
                    f"compared_to={cid} reason=lower_score_same_ordered_column_signature"
                ),
            )
            best_by_columns[item.output_columns] = item
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_columns action=kept candidate={cid} "
                    f"reason=higher_score_same_ordered_column_signature"
                ),
            )
            continue

        dropped_count += 1
        _append_report_line(
            report_file,
            (
                f"stage=dedup_columns action=dropped candidate={cid} "
                f"compared_to={current_id} reason=lower_or_equal_score_same_ordered_column_signature"
            ),
        )

    survivors = sorted(
        best_by_columns.values(),
        key=lambda i: (i.candidate.island_id, -i.candidate.score, i.candidate.sample_order),
    )
    return survivors, dropped_count


def dedup_stage_by_exact_code(
    executed: List[ExecutedCandidate], report_file: TextIO
) -> Tuple[List[ExecutedCandidate], int]:
    best_by_code: Dict[str, ExecutedCandidate] = {}
    dropped_count = 0
    for item in executed:
        code = item.candidate.function_code
        current = best_by_code.get(code)
        cid = _candidate_id(item.candidate)
        if current is None:
            best_by_code[code] = item
            _append_report_line(
                report_file,
                f"stage=dedup_exact_code action=kept candidate={cid} reason=unique_function_string",
            )
            continue

        current_id = _candidate_id(current.candidate)
        if _is_better_candidate(item.candidate, current.candidate):
            dropped_count += 1
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_exact_code action=dropped candidate={current_id} "
                    f"compared_to={cid} reason=lower_score_same_function_string"
                ),
            )
            best_by_code[code] = item
            _append_report_line(
                report_file,
                f"stage=dedup_exact_code action=kept candidate={cid} reason=higher_score_same_function_string",
            )
            continue

        dropped_count += 1
        _append_report_line(
            report_file,
            (
                f"stage=dedup_exact_code action=dropped candidate={cid} "
                f"compared_to={current_id} reason=lower_or_equal_score_same_function_string"
            ),
        )

    survivors = sorted(
        best_by_code.values(),
        key=lambda i: (i.candidate.island_id, -i.candidate.score, i.candidate.sample_order),
    )
    return survivors, dropped_count


def dedup_stage_by_semantic_hash(
    executed: List[ExecutedCandidate], report_file: TextIO
) -> Tuple[List[ExecutedCandidate], int]:
    best_by_hash: Dict[str, ExecutedCandidate] = {}
    dropped_count = 0
    for item in executed:
        current = best_by_hash.get(item.semantic_hash)
        cid = _candidate_id(item.candidate)
        if current is None:
            best_by_hash[item.semantic_hash] = item
            _append_report_line(
                report_file,
                f"stage=dedup_semantic_hash action=kept candidate={cid} reason=unique_output_signature",
            )
            continue

        current_id = _candidate_id(current.candidate)
        if _is_better_candidate(item.candidate, current.candidate):
            dropped_count += 1
            _append_report_line(
                report_file,
                (
                    f"stage=dedup_semantic_hash action=dropped candidate={current_id} "
                    f"compared_to={cid} reason=lower_score_same_output_signature"
                ),
            )
            best_by_hash[item.semantic_hash] = item
            _append_report_line(
                report_file,
                f"stage=dedup_semantic_hash action=kept candidate={cid} reason=higher_score_same_output_signature",
            )
            continue

        dropped_count += 1
        _append_report_line(
            report_file,
            (
                f"stage=dedup_semantic_hash action=dropped candidate={cid} "
                f"compared_to={current_id} reason=lower_or_equal_score_same_output_signature"
            ),
        )

    survivors = sorted(
        best_by_hash.values(),
        key=lambda i: (i.candidate.island_id, -i.candidate.score, i.candidate.sample_order),
    )
    return survivors, dropped_count


def deduplicate_candidates_multistage(
    candidates: List[FeatureCandidate],
    df_input: pd.DataFrame,
    report_path: str,
    rounding_decimals: int = 8,
) -> Tuple[List[FeatureCandidate], Dict[str, int]]:
    summary = {
        "total_input": len(candidates),
        "dropped_execute": 0,
        "dropped_stage_1_columns": 0,
        "dropped_stage_2_exact_code": 0,
        "dropped_stage_3_semantic_hash": 0,
        "total_survivors": 0,
    }

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        _append_report_line(
            report_file,
            "=== Multi-Stage Candidate Deduplication Report ===",
        )
        _append_report_line(
            report_file,
            f"config rounding_decimals={rounding_decimals} semantic_window_policy=25pct_or_20_head_tail",
        )

        executed, dropped_execute = execute_candidates_for_dedup(
            candidates,
            df_input=df_input,
            report_file=report_file,
            rounding_decimals=rounding_decimals,
        )
        summary["dropped_execute"] = dropped_execute

        stage1, dropped_stage1 = dedup_stage_by_output_columns(executed, report_file=report_file)
        summary["dropped_stage_1_columns"] = dropped_stage1

        stage2, dropped_stage2 = dedup_stage_by_exact_code(stage1, report_file=report_file)
        summary["dropped_stage_2_exact_code"] = dropped_stage2

        stage3, dropped_stage3 = dedup_stage_by_semantic_hash(stage2, report_file=report_file)
        summary["dropped_stage_3_semantic_hash"] = dropped_stage3

        survivors = [item.candidate for item in stage3]
        summary["total_survivors"] = len(survivors)

        _append_report_line(report_file, "")
        _append_report_line(report_file, "=== Summary ===")
        _append_report_line(report_file, f"total_input={summary['total_input']}")
        _append_report_line(report_file, f"dropped_execute={summary['dropped_execute']}")
        _append_report_line(
            report_file, f"dropped_stage_1_columns={summary['dropped_stage_1_columns']}"
        )
        _append_report_line(
            report_file, f"dropped_stage_2_exact_code={summary['dropped_stage_2_exact_code']}"
        )
        _append_report_line(
            report_file, f"dropped_stage_3_semantic_hash={summary['dropped_stage_3_semantic_hash']}"
        )
        _append_report_line(report_file, f"total_survivors={summary['total_survivors']}")

    print(
        "Dedup summary:"
        f" input={summary['total_input']},"
        f" dropped_execute={summary['dropped_execute']},"
        f" dropped_stage_1={summary['dropped_stage_1_columns']},"
        f" dropped_stage_2={summary['dropped_stage_2_exact_code']},"
        f" dropped_stage_3={summary['dropped_stage_3_semantic_hash']},"
        f" survivors={summary['total_survivors']}"
    )
    print(f"Dedup report path: {report_path}")
    return survivors, summary


def _execute_candidate(candidate: FeatureCandidate, df_input: pd.DataFrame) -> pd.DataFrame:
    namespace: Dict[str, object] = {"pd": pd}
    exec(candidate.function_code, namespace)

    function = namespace.get("modify_features_v2")
    if not callable(function):
        function = namespace.get("modify_features")
    if not callable(function):
        raise ValueError("Candidate must define modify_features_v2 or modify_features")

    df_out = function(df_input.copy())
    if not isinstance(df_out, pd.DataFrame):
        raise TypeError("Feature function must return a pandas DataFrame")
    if not df_out.index.equals(df_input.index):
        raise ValueError("Returned dataframe index must match the input index")
    return df_out


class FeatureExtractionPipeline:
    def __init__(
        self,
        samples_dir: str,
        k_per_island: int = 2,
        label_column: Optional[str] = None,
        include_original: bool = True,
        data_dir: Optional[str] = None,
    ) -> None:
        self.samples_dir = samples_dir
        self.k_per_island = k_per_island
        self.label_column = label_column
        self.include_original = include_original
        self.data_dir = data_dir

    def run(
        self, df: pd.DataFrame, dataset_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        candidates = load_candidates(self.samples_dir)
        selected = select_top_k_per_island(candidates, k=self.k_per_island)
        resolved_dataset_name = dataset_name or "dataset"
        data_dir = self.data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        report_path = os.path.join(data_dir, f"dedup_report_{resolved_dataset_name}.txt")
        deduped, _ = deduplicate_candidates_multistage(
            selected,
            df_input=df,
            report_path=report_path,
            rounding_decimals=8,
        )
        filtered_manifest_path = os.path.join(data_dir, f"filtered_{resolved_dataset_name}.csv")
        os.makedirs(data_dir, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "island_id": c.island_id,
                    "sample_order": c.sample_order,
                    "score": c.score,
                    "source_file": c.source_file,
                }
                for c in deduped
            ]
        ).to_csv(filtered_manifest_path, index=False)
        print(f"Filtered candidate manifest: {filtered_manifest_path}")
        return self.build_full_dataframe(
            df=df,
            candidates=deduped,
            label_column=self.label_column,
            include_original=self.include_original,
        )

    @staticmethod
    def build_full_dataframe(
        df: pd.DataFrame,
        candidates: List[FeatureCandidate],
        label_column: Optional[str] = None,
        include_original: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        generated_parts: List[pd.DataFrame] = []
        metadata_rows: List[Dict[str, object]] = []

        for candidate in candidates:
            prefix = f"is{candidate.island_id}_s{candidate.sample_order}"
            try:
                out = _execute_candidate(candidate, df)
                out = out.copy()
                out.columns = [f"{prefix}_{col}" for col in out.columns]
                generated_parts.append(out)
                metadata_rows.append(
                    {
                        "island_id": candidate.island_id,
                        "sample_order": candidate.sample_order,
                        "score": candidate.score,
                        "source_file": candidate.source_file,
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

        if label_series is not None:
            parts.append(label_series.to_frame(name=label_column))

        final_df = pd.concat(parts, axis=1)
        if label_series is not None and final_df.columns[-1] != label_column:
            label_values = final_df.pop(label_column)
            final_df[label_column] = label_values

        metadata_df = pd.DataFrame(metadata_rows)
        return final_df, metadata_df
