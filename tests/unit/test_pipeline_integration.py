import json
from pathlib import Path

import pandas as pd
import pytest

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
    assert any(col.startswith("is1_s1_") for col in out_df.columns), metadata_debug
    assert len(metadata_df) == 1, metadata_debug
    assert metadata_df.iloc[0]["status"] == "success", metadata_debug
