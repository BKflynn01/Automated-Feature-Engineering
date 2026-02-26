from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ga_optimizer.config import DEFAULT_GA_CONFIG


def load_ga_input(
    csv_path: str,
    label_column: str,
    sep: str = DEFAULT_GA_CONFIG.ga_data.default_csv_sep,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(csv_path, sep=sep)

    if label_column not in df.columns:
        raise ValueError(f"label column '{label_column}' not found in input columns")

    y = df[label_column].copy()
    X = df.drop(columns=[label_column]).copy()
    feature_names = X.columns.tolist()

    if not feature_names:
        raise ValueError("CSV has no feature columns after removing label column")

    return X, y, feature_names


def chromosome_length(feature_names: List[str]) -> int:
    return len(feature_names)


def decode_chromosome(
    chromosome: List[int],
    feature_names: List[str],
) -> List[str]:
    if len(chromosome) != len(feature_names):
        raise ValueError("chromosome length must match feature_names length")

    return [
        name
        for gene, name in zip(chromosome, feature_names)
        if gene == DEFAULT_GA_CONFIG.ga_data.selected_gene_value
    ]
