import json

import pandas as pd
import pytest

from ga_optimizer.evolve.preprocess import (
    FeatureCandidate,
    FeatureExtractionPipeline,
    deduplicate_candidates,
    load_candidates,
    select_top_k_per_island,
)


@pytest.fixture
def sample_dir(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    return samples


def _write_sample(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_load_candidates_supports_function_and_function_code(sample_dir):
    _write_sample(
        sample_dir / "a.json",
        {
            "island_id": 1,
            "score": 0.4,
            "function": "def modify_features(df):\n    return df[['x']]",
            "sample_order": 0,
        },
    )
    _write_sample(
        sample_dir / "b.json",
        {
            "island_id": 1,
            "score": 0.5,
            "function_code": "def modify_features(df):\n    return df[['x']]",
            "sample_order": 1,
        },
    )

    candidates = load_candidates(str(sample_dir))
    assert len(candidates) == 2
    assert all(isinstance(c, FeatureCandidate) for c in candidates)


def test_select_top_k_per_island():
    candidates = [
        FeatureCandidate(1, 0.10, "a", 1, "a.json"),
        FeatureCandidate(1, 0.90, "b", 2, "b.json"),
        FeatureCandidate(1, 0.80, "c", 3, "c.json"),
        FeatureCandidate(2, 0.70, "d", 1, "d.json"),
        FeatureCandidate(2, 0.20, "e", 2, "e.json"),
    ]
    selected = select_top_k_per_island(candidates, k=2)
    by_island = {}
    for c in selected:
        by_island.setdefault(c.island_id, []).append(c.score)
    assert sorted(by_island[1], reverse=True) == [0.9, 0.8]
    assert sorted(by_island[2], reverse=True) == [0.7, 0.2]


def test_deduplicate_candidates_keeps_highest_score():
    candidates = [
        FeatureCandidate(1, 0.2, "def modify_features(df):\n    return df", 2, "a.json"),
        FeatureCandidate(2, 0.9, "def modify_features(df):\n    return df", 1, "b.json"),
    ]
    deduped = deduplicate_candidates(candidates)
    assert len(deduped) == 1
    assert deduped[0].score == 0.9
    assert deduped[0].island_id == 2


def test_build_full_dataframe_keeps_label_last():
    df = pd.DataFrame({"x": [1, 2], "y": [10, 20], "target": [0, 1]})
    candidates = [
        FeatureCandidate(
            island_id=1,
            score=0.9,
            function_code=(
                "def modify_features(df):\n"
                "    out = pd.DataFrame(index=df.index)\n"
                "    out['x_plus_y'] = df['x'] + df['y']\n"
                "    return out\n"
            ),
            sample_order=11,
            source_file="s11.json",
        )
    ]

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        candidates=candidates,
        label_column="target",
        include_original=True,
    )

    assert out_df.columns[-1] == "target"
    assert "target" in out_df.columns
    assert any(col.startswith("is1_s11_") for col in out_df.columns)
    assert len(meta) == 1
    assert meta.iloc[0]["status"] == "success"


def test_build_full_dataframe_records_failure():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    candidates = [
        FeatureCandidate(
            island_id=1,
            score=0.3,
            function_code="def modify_features(df):\n    return 123\n",
            sample_order=1,
            source_file="bad.json",
        )
    ]

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        candidates=candidates,
        label_column="target",
        include_original=True,
    )

    assert out_df.columns.tolist() == ["x", "target"]
    assert len(meta) == 1
    assert meta.iloc[0]["status"] == "failed"
    assert "DataFrame" in meta.iloc[0]["error"]
