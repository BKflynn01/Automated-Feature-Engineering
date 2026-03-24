import argparse

import pandas as pd
import pytest

import ga_optimizer.main as ga_main
from ga_optimizer.config import DEFAULT_GA_CONFIG


def test_resolve_label_column_explicit():
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    resolved = ga_main.resolve_label_column(
        df,
        label_column="target",
        use_last_column=False,
    )
    assert resolved == "target"


def test_resolve_label_column_uses_last_column():
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    resolved = ga_main.resolve_label_column(
        df,
        label_column=None,
        use_last_column=True,
    )
    assert resolved == "y"


def test_resolve_label_column_returns_none_when_last_column_inference_disabled():
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    resolved = ga_main.resolve_label_column(
        df,
        label_column=None,
        use_last_column=False,
    )
    assert resolved is None


def test_resolve_label_column_raises_for_missing_column():
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(ValueError, match="label column 'target' not found"):
        ga_main.resolve_label_column(df, label_column="target", use_last_column=False)


def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--input_csv",
            "in.csv",
            "--samples_dir",
            "samples",
            "--output_csv",
            "out.csv",
            "--source_file_glob",
            "btc_gpt3.5_split_*/samples/*.json",
            "--use_last_column_as_label",
            "--k_per_island",
            "3",
            "--sep",
            ";",
        ],
    )
    args = ga_main.parse_args()
    assert args.input_csv == "in.csv"
    assert args.samples_dir == "samples"
    assert args.output_csv == "out.csv"
    assert args.source_file_glob == "btc_gpt3.5_split_*/samples/*.json"
    assert args.use_last_column_as_label is True
    assert args.k_per_island == 3
    assert args.sep == ";"


def test_parse_args_defaults_follow_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--input_csv",
            "in.csv",
            "--samples_dir",
            "samples",
            "--output_csv",
            "out.csv",
        ],
    )
    args = ga_main.parse_args()
    assert args.k_per_island == DEFAULT_GA_CONFIG.main.default_k_per_island
    assert args.sep == DEFAULT_GA_CONFIG.main.default_sep
    assert args.source_file_glob is None


def test_main_writes_outputs_and_uses_pipeline(monkeypatch, tmp_path):
    input_csv = tmp_path / "input.csv"
    sample_dir = tmp_path / "samples"
    output_csv = tmp_path / "output.csv"
    sample_dir.mkdir()

    pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_csv(input_csv, index=False)

    captured = {}

    class DummyPipeline:
        def __init__(
            self,
            samples_dir,
            k_per_island,
            label_column,
            include_original,
            source_file_glob=None,
        ):
            captured["samples_dir"] = samples_dir
            captured["k_per_island"] = k_per_island
            captured["label_column"] = label_column
            captured["include_original"] = include_original
            captured["source_file_glob"] = source_file_glob

        def run(self, df, dataset_name=None):
            captured["run_shape"] = df.shape
            captured["dataset_name"] = dataset_name
            final_df = pd.DataFrame({"x": [1, 2], "f": [3, 4], "target": [0, 1]})
            metadata_df = pd.DataFrame([{"status": "success"}])
            return final_df, metadata_df

    args = argparse.Namespace(
        input_csv=str(input_csv),
        samples_dir=str(sample_dir),
        output_csv=str(output_csv),
        metadata_csv=None,
        source_file_glob="btc_gpt3.5_split_*/samples/*.json",
        label_column=None,
        use_last_column_as_label=True,
        k_per_island=2,
        exclude_original=False,
        sep=",",
    )

    monkeypatch.setattr(ga_main, "parse_args", lambda: args)
    monkeypatch.setattr(ga_main, "FeatureExtractionPipeline", DummyPipeline)

    ga_main.main()

    expected_metadata = output_csv.with_suffix(".meta.csv")
    assert output_csv.exists()
    assert expected_metadata.exists()

    out_df = pd.read_csv(output_csv)
    meta_df = pd.read_csv(expected_metadata)

    assert out_df.columns.tolist() == ["x", "f", "target"]
    assert meta_df.loc[0, "status"] == "success"
    assert captured["samples_dir"] == str(sample_dir)
    assert captured["k_per_island"] == 2
    assert captured["label_column"] == "target"
    assert captured["include_original"] is True
    assert captured["source_file_glob"] == "btc_gpt3.5_split_*/samples/*.json"
    assert captured["run_shape"] == (2, 2)
    assert captured["dataset_name"] == "input"


def test_main_raises_when_input_csv_missing(monkeypatch, tmp_path):
    missing_input = tmp_path / "missing.csv"
    sample_dir = tmp_path / "samples"
    output_csv = tmp_path / "output.csv"
    sample_dir.mkdir()

    args = argparse.Namespace(
        input_csv=str(missing_input),
        samples_dir=str(sample_dir),
        output_csv=str(output_csv),
        metadata_csv=None,
        source_file_glob=None,
        label_column=None,
        use_last_column_as_label=True,
        k_per_island=2,
        exclude_original=False,
        sep=",",
    )
    monkeypatch.setattr(ga_main, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="input_csv does not exist"):
        ga_main.main()


def test_main_raises_when_samples_dir_missing(monkeypatch, tmp_path):
    input_csv = tmp_path / "input.csv"
    missing_samples = tmp_path / "missing_samples"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_csv(input_csv, index=False)

    args = argparse.Namespace(
        input_csv=str(input_csv),
        samples_dir=str(missing_samples),
        output_csv=str(output_csv),
        metadata_csv=None,
        source_file_glob=None,
        label_column=None,
        use_last_column_as_label=True,
        k_per_island=2,
        exclude_original=False,
        sep=",",
    )
    monkeypatch.setattr(ga_main, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="samples_dir does not exist"):
        ga_main.main()


def test_main_respects_explicit_metadata_sep_label_and_exclude_original(monkeypatch, tmp_path):
    input_csv = tmp_path / "input.csv"
    sample_dir = tmp_path / "samples"
    output_csv = tmp_path / "nested" / "output.csv"
    metadata_csv = tmp_path / "meta" / "custom_metadata.csv"
    sample_dir.mkdir()

    pd.DataFrame({"x": [1, 2], "label": [0, 1]}).to_csv(input_csv, index=False, sep=";")

    captured = {}

    class DummyPipeline:
        def __init__(
            self,
            samples_dir,
            k_per_island,
            label_column,
            include_original,
            source_file_glob=None,
        ):
            captured["samples_dir"] = samples_dir
            captured["k_per_island"] = k_per_island
            captured["label_column"] = label_column
            captured["include_original"] = include_original
            captured["source_file_glob"] = source_file_glob

        def run(self, df, dataset_name=None):
            captured["run_shape"] = df.shape
            captured["dataset_name"] = dataset_name
            final_df = pd.DataFrame({"generated_feature": [5, 6], "label": [0, 1]})
            metadata_df = pd.DataFrame([{"status": "success"}])
            return final_df, metadata_df

    args = argparse.Namespace(
        input_csv=str(input_csv),
        samples_dir=str(sample_dir),
        output_csv=str(output_csv),
        metadata_csv=str(metadata_csv),
        source_file_glob="btc_gpt3.5_split_*/samples/*.json",
        label_column="label",
        use_last_column_as_label=False,
        k_per_island=3,
        exclude_original=True,
        sep=";",
    )
    monkeypatch.setattr(ga_main, "parse_args", lambda: args)
    monkeypatch.setattr(ga_main, "FeatureExtractionPipeline", DummyPipeline)

    ga_main.main()

    assert output_csv.exists()
    assert metadata_csv.exists()

    out_df = pd.read_csv(output_csv, sep=";")
    meta_df = pd.read_csv(metadata_csv)

    assert out_df.columns.tolist() == ["generated_feature", "label"]
    assert meta_df.loc[0, "status"] == "success"
    assert captured["samples_dir"] == str(sample_dir)
    assert captured["k_per_island"] == 3
    assert captured["label_column"] == "label"
    assert captured["include_original"] is False
    assert captured["source_file_glob"] == "btc_gpt3.5_split_*/samples/*.json"
    assert captured["run_shape"] == (2, 2)
    assert captured["dataset_name"] == "input"
