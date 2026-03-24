#!/usr/bin/env python3
"""Create a binary-classification variant of a dataset.

How it works:
1. Reads `data/{DATASET_NAME}.csv`.
2. Converts the last column to binary labels:
   - 1 if value > 0 (price up)
   - 0 otherwise (price down or unchanged)
   - Renames that output label column to `price_direction`
3. Writes `data/{DATASET_NAME}_classification.csv`.
4. Copies metadata from `data/{DATASET_NAME}-metadata.json` to
   `data/{DATASET_NAME}-classification-metadata.json`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

# Set this to the dataset stem in the data folder (without .csv).
# Example: "btc" -> reads data/btc.csv
DATASET_NAME = "btc"


def _parse_target_as_numeric(series: pd.Series) -> pd.Series:
    """Parse target values into numeric form, supporting simple percent strings."""
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False).str.removesuffix("%")
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.isna().any():
        bad_count = int(numeric.isna().sum())
        raise ValueError(f"Could not parse {bad_count} value(s) in the last column as numeric.")
    return numeric


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    input_csv = data_dir / f"{DATASET_NAME}.csv"
    input_metadata = data_dir / f"{DATASET_NAME}-metadata.json"
    output_csv = data_dir / f"{DATASET_NAME}_classification.csv"
    output_metadata = data_dir / f"{DATASET_NAME}-classification-metadata.json"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")
    if not input_metadata.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_metadata}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_csv}")

    target_col = df.columns[-1]
    target_numeric = _parse_target_as_numeric(df[target_col])
    df[target_col] = (target_numeric > 0).astype(int)
    df = df.rename(columns={target_col: "price_direction"})

    df.to_csv(output_csv, index=False)
    shutil.copy2(input_metadata, output_metadata)

    print(f"Input CSV: {input_csv}")
    print(f"Target column converted: {target_col} -> price_direction")
    print(f"Wrote classification CSV: {output_csv}")
    print(f"Copied metadata to: {output_metadata}")


if __name__ == "__main__":
    main()
