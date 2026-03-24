import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ga_optimizer.config import DEFAULT_GA_CONFIG, SelectionConfig
from ga_optimizer.evolve.preprocess import FeatureExtractionPipeline


@pytest.mark.parametrize(
    "data_path",
    [
        "data/btc.csv",
    ],
)
def test_dataset_pipeline_integration(tmp_path, data_path):
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / data_path
    df = pd.read_csv(dataset_path, nrows=200)

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    candidate_payload = {
        "island_id": 1,
        "score": 0.95,
        "sample_order": 1,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['feature_sum_first_two'] = df.iloc[:, 1] + df.iloc[:, 2]\n"
            "    return out\n"
        ),
    }

    with open(samples_dir / "sample_1.json", "w", encoding="utf-8") as f:
        json.dump(candidate_payload, f)

    label_column = df.columns[-1]
    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_dir),
        k_per_island=1,
        label_column=label_column,
        include_original=True,
    )

    out_df, metadata_df = pipeline.run(df)
    metadata_debug = metadata_df.to_string(index=False)
    assert out_df.shape[0] == df.shape[0]
    assert out_df.columns[-1] == label_column
    assert "feature_sum_first_two" in out_df.columns, metadata_debug
    assert len(metadata_df) == 1, metadata_debug
    assert metadata_df.iloc[0]["status"] == "success", metadata_debug


def test_pipeline_integration_uses_config_default_top_k_when_not_provided(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 6.0, 7.0], "target": [0, 1, 0]})
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    candidate_payload_1 = {
        "island_id": 1,
        "score": 0.95,
        "sample_order": 1,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['f1'] = df['x'] + df['y']\n"
            "    return out\n"
        ),
    }
    candidate_payload_2 = {
        "island_id": 1,
        "score": 0.50,
        "sample_order": 2,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['f2'] = df['x'] - df['y']\n"
            "    return out\n"
        ),
    }

    with open(samples_dir / "sample_1.json", "w", encoding="utf-8") as f:
        json.dump(candidate_payload_1, f)
    with open(samples_dir / "sample_2.json", "w", encoding="utf-8") as f:
        json.dump(candidate_payload_2, f)

    config = replace(DEFAULT_GA_CONFIG, selection=SelectionConfig(default_top_k_per_island=1))
    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_dir),
        label_column="target",
        include_original=True,
        config=config,
    )

    out_df, metadata_df = pipeline.run(df, dataset_name="demo_top_k_default")

    assert "f1" in out_df.columns
    assert "f2" not in out_df.columns
    assert len(metadata_df) == 1
