from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ga_optimizer.config import DEFAULT_GA_CONFIG
from ga_optimizer.evolve.preprocess import FeatureExtractionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GA optimizer preprocessing by selecting top-k candidates per island, "
            "executing them, and merging generated features with the original dataset."
        )
    )
    parser.add_argument("--input_csv", required=True, help="Path to input CSV dataset")
    parser.add_argument(
        "--samples_dir", required=True, help="Directory containing candidate JSON files"
    )
    parser.add_argument(
        "--source_file_glob",
        default=None,
        help=(
            "Optional relative glob filter applied within samples_dir, e.g. "
            "'btc_gpt3.5_split_*/samples/*.json'"
        ),
    )
    parser.add_argument("--output_csv", required=True, help="Path to write merged output CSV")
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Optional path to write candidate execution metadata CSV",
    )
    parser.add_argument(
        "--label_column",
        default=None,
        help=(
            "Label column name. If omitted and --use_last_column_as_label is enabled, "
            "the last column in input_csv is treated as label and forced to remain last."
        ),
    )
    parser.add_argument(
        "--use_last_column_as_label",
        action="store_true",
        default=DEFAULT_GA_CONFIG.main.default_use_last_column_as_label,
        help="Treat the last input column as label when --label_column is not set",
    )
    parser.add_argument(
        "--k_per_island",
        type=int,
        default=DEFAULT_GA_CONFIG.main.default_k_per_island,
        help="Top-k candidates per island",
    )
    parser.add_argument(
        "--exclude_original",
        action="store_true",
        default=DEFAULT_GA_CONFIG.main.default_exclude_original,
        help="If set, output contains generated features (and label if available), not original features",
    )
    parser.add_argument(
        "--sep",
        default=DEFAULT_GA_CONFIG.main.default_sep,
        help="CSV delimiter for input/output (default: ',')",
    )
    return parser.parse_args()


def resolve_label_column(
    df: pd.DataFrame, label_column: str | None, use_last_column: bool
) -> str | None:
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"label column '{label_column}' not found in input columns")
        return label_column
    if use_last_column:
        return str(df.columns[-1])
    return None


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    samples_path = Path(args.samples_dir)
    output_path = Path(args.output_csv)
    metadata_path = (
        Path(args.metadata_csv) if args.metadata_csv else output_path.with_suffix(".meta.csv")
    )

    if not input_path.exists():
        raise ValueError(f"input_csv does not exist: {input_path}")
    if not samples_path.exists():
        raise ValueError(f"samples_dir does not exist: {samples_path}")

    df = pd.read_csv(input_path, sep=args.sep)
    label_column = resolve_label_column(
        df,
        label_column=args.label_column,
        use_last_column=args.use_last_column_as_label,
    )

    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_path),
        k_per_island=args.k_per_island,
        label_column=label_column,
        include_original=not args.exclude_original,
        source_file_glob=args.source_file_glob,
    )

    final_df, metadata_df = pipeline.run(df, dataset_name=input_path.stem)

    for parent in {output_path.parent, metadata_path.parent}:
        parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(output_path, index=False, sep=args.sep)
    metadata_df.to_csv(metadata_path, index=False)

    print(f"Input rows/cols    : {df.shape[0]}/{df.shape[1]}")
    print(f"Output rows/cols   : {final_df.shape[0]}/{final_df.shape[1]}")
    print(f"Label column       : {label_column if label_column else 'None'}")
    print(f"Top-k per island   : {args.k_per_island}")
    print(f"Include original   : {not args.exclude_original}")
    print(f"Output CSV         : {output_path}")
    print(f"Metadata CSV       : {metadata_path}")


if __name__ == "__main__":
    main()
