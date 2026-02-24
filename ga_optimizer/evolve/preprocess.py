from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def load_candidates(samples_dir: str) -> List[FeatureCandidate]:
    """Load feature candidates from JSON files."""
    if not os.path.exists(samples_dir):
        raise ValueError(f"Sample directory {samples_dir} does not exist")

    files = sorted(glob.glob(os.path.join(samples_dir, "*.json")))
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
                source_file=os.path.basename(fp),
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
    ) -> None:
        self.samples_dir = samples_dir
        self.k_per_island = k_per_island
        self.label_column = label_column
        self.include_original = include_original

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        candidates = load_candidates(self.samples_dir)
        selected = select_top_k_per_island(candidates, k=self.k_per_island)
        deduped = deduplicate_candidates(selected)
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
