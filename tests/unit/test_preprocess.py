import json
from dataclasses import replace

import pandas as pd
import pytest

from ga_optimizer.config import (
    CandidateLoadConfig,
    DEFAULT_GA_CONFIG,
    DedupConfig,
    ExecutionConfig,
    OutputConfig,
    SelectionConfig,
)
from ga_optimizer.evolve.preprocess import (
    ExecutedCandidate,
    FeatureCandidate,
    FeatureExtractionPipeline,
    dedup_stage_by_output_columns,
    dedup_stage_by_semantic_hash,
    deduplicate_candidates_multistage,
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


def test_load_candidates_supports_problem_root_with_multiple_splits(tmp_path):
    problem_root = tmp_path / "problem"
    split1_samples = problem_root / "problem_split_1" / "samples"
    split2_samples = problem_root / "problem_split_2" / "samples"
    split1_samples.mkdir(parents=True)
    split2_samples.mkdir(parents=True)

    _write_sample(
        split1_samples / "a.json",
        {
            "island_id": 1,
            "score": 0.4,
            "function_code": "def modify_features(df):\n    return df[['x']]",
            "sample_order": 1,
        },
    )
    _write_sample(
        split2_samples / "b.json",
        {
            "island_id": 2,
            "score": 0.5,
            "function_code": "def modify_features(df):\n    return df[['x']]",
            "sample_order": 2,
        },
    )

    candidates = load_candidates(str(problem_root))
    assert len(candidates) == 2
    source_files = {c.source_file for c in candidates}
    assert "problem_split_1/samples/a.json" in source_files
    assert "problem_split_2/samples/b.json" in source_files


def test_load_candidates_supports_split_root_directory(tmp_path):
    split_root = tmp_path / "btc_gpt_4o_mini_split_2"
    samples_dir = split_root / "samples"
    samples_dir.mkdir(parents=True)
    _write_sample(
        samples_dir / "samples_16.json",
        {
            "island_id": 1,
            "score": 0.77,
            "function_code": "def modify_features(df):\n    return df[['x']]",
            "sample_order": 16,
        },
    )

    candidates = load_candidates(str(split_root))
    assert len(candidates) == 1
    assert candidates[0].source_file == "samples/samples_16.json"


def test_load_candidates_honors_configured_json_glob_patterns(tmp_path):
    root = tmp_path / "candidate_root"
    custom_dir = root / "custom"
    custom_dir.mkdir(parents=True)
    _write_sample(
        custom_dir / "c.json",
        {
            "island_id": 1,
            "score": 0.77,
            "function_code": "def modify_features(df):\n    return df[['x']]",
            "sample_order": 3,
        },
    )

    config = replace(
        DEFAULT_GA_CONFIG,
        candidate_load=CandidateLoadConfig(json_glob_patterns=("custom/*.json",)),
    )
    candidates = load_candidates(str(root), config=config)

    assert len(candidates) == 1
    assert candidates[0].source_file == "custom/c.json"


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


def test_select_top_k_per_island_uses_config_default_when_k_not_provided():
    candidates = [
        FeatureCandidate(1, 0.10, "a", 1, "a.json"),
        FeatureCandidate(1, 0.90, "b", 2, "b.json"),
        FeatureCandidate(2, 0.70, "d", 1, "d.json"),
        FeatureCandidate(2, 0.20, "e", 2, "e.json"),
    ]
    config = replace(DEFAULT_GA_CONFIG, selection=SelectionConfig(default_top_k_per_island=1))

    selected = select_top_k_per_island(candidates, config=config)

    assert len(selected) == 2
    assert {(c.island_id, c.score) for c in selected} == {(1, 0.9), (2, 0.7)}


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
    assert "x_plus_y" in out_df.columns
    assert len(meta) == 1
    assert meta.iloc[0]["status"] == "success"
    assert meta.iloc[0]["generated_columns"] == ["x_plus_y"]


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


def test_build_full_dataframe_compiles_and_runs_payload_function():
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.5, 101.2, 103.0, 104.1],
            "high": [101.0, 102.0, 103.0, 102.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.5, 100.0, 102.0, 103.0],
            "volume": [1000, 1100, 1050, 1200, 1300, 1250],
            "fng_value": [35, 40, 45, 30, 25, 50],
        }
    )

    function_code = (
        "def modify_features(df_input) -> pd.DataFrame:\n"
        '    """\n'
        "    Thought 1: Thought 1: Capture intraday panic (High/Low) and liquidity (Price*Vol) using stationary log-transforms to normalize data across all price regimes.\n"
        '    """\n'
        "    import pandas as pd\n"
        "    import numpy as np\n"
        '    """\n'
        "    This program captures market psychology and supply/demand imbalances through stationary features, enhancing predictive power for next-day Bitcoin price movements.\n"
        '    """\n'
        "    import numpy as np\n"
        "    import pandas as pd\n"
        "    \n"
        "    df_output = df_input.copy()\n"
        "    \n"
        "    # 1. Log Returns (Stationary Price Change)\n"
        "    df_output['log_return'] = np.log(df_output['close'] / df_output['close'].shift(1)).fillna(0)\n"
        "\n"
        "    # 2. Distance from 50-day Moving Average (%)\n"
        "    df_output['distance_from_50ma'] = (df_output['close'] - df_output['close'].rolling(window=50).mean()) / df_output['close'].rolling(window=50).mean()\n"
        "\n"
        "    # 3. Bollinger Band Width (Volatility Regime)\n"
        "    rolling_std = df_output['close'].rolling(window=20).std()\n"
        "    rolling_mean = df_output['close'].rolling(window=20).mean()\n"
        "    df_output['bollinger_band_width'] = (rolling_mean + 2 * rolling_std - (rolling_mean - 2 * rolling_std)) / rolling_mean\n"
        "\n"
        "    # 4. Average True Range (ATR) as a Volatility Proxy\n"
        "    high_low = df_output['high'] - df_output['low']\n"
        "    high_prev_close = np.abs(df_output['high'] - df_output['close'].shift(1))\n"
        "    low_prev_close = np.abs(df_output['low'] - df_output['close'].shift(1))\n"
        "    true_range = pd.DataFrame({'high_low': high_low, 'high_prev_close': high_prev_close, 'low_prev_close': low_prev_close}).max(axis=1)\n"
        "    df_output['atr'] = true_range.rolling(window=14).mean()\n"
        "\n"
        "    # 5. Volume-Weighted Price Change\n"
        "    df_output['volume_weighted_change'] = (df_output['close'] * df_output['volume']) / df_output['volume'].rolling(window=14).sum()\n"
        "\n"
        "    # 6. Z-Score of Fear & Greed Index\n"
        "    df_output['fng_zscore'] = (df_output['fng_value'] - df_output['fng_value'].rolling(window=14).mean()) / df_output['fng_value'].rolling(window=14).std()\n"
        "\n"
        "    # 7. Volume Confirmation (Price * Volume)\n"
        "    df_output['price_volume'] = df_output['close'] * df_output['volume']\n"
        "\n"
        "    # 8. Support/Resistance Proxies (Rolling Max/Min)\n"
        "    df_output['rolling_max'] = df_output['close'].rolling(window=20).max()\n"
        "    df_output['rolling_min'] = df_output['close'].rolling(window=20).min()\n"
        "\n"
        "    # 9. Overbought/Oversold Indicator (RSI)\n"
        "    delta = df_output['close'].diff()\n"
        "    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()\n"
        "    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()\n"
        "    rs = gain / loss\n"
        "    df_output['rsi'] = 100 - (100 / (1 + rs))\n"
        "\n"
        "    # 10. Sentiment Divergence: Trigger when sentiment is extreme but price is stable\n"
        "    df_output['sentiment_divergence'] = ((df_output['fng_value'] < 25) & (df_output['close'] > df_output['close'].rolling(window=5).mean())).fillna(False).astype(int)\n"
        "\n"
        "    # 11. Price Momentum: 5-day price change\n"
        "    df_output['momentum_5d'] = (df_output['close'] - df_output['close'].shift(5)) / df_output['close'].shift(5)\n"
        "\n"
        "    # 12. Price Change vs Volume Change: Ratio of log returns to log volume change\n"
        "    df_output['price_volume_ratio'] = df_output['log_return'] / np.log(df_output['volume'] + 1).fillna(0)\n"
        "\n"
        "    # Fill NaNs for the new features\n"
        "    df_output.fillna(0, inplace=True)\n"
        "\n"
        "    return df_output\n"
    )

    candidates = [
        FeatureCandidate(
            island_id=1,
            score=-1.1609591118086777,
            function_code=function_code,
            sample_order=16,
            source_file="payload.json",
        )
    ]

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        candidates=candidates,
        include_original=False,
    )

    assert len(meta) == 1
    assert meta.iloc[0]["status"] == "success"
    assert "log_return" in out_df.columns
    assert "rsi" in out_df.columns
    assert out_df.shape[0] == df.shape[0]


def test_dedup_stage_by_output_columns_keeps_highest_score(tmp_path):
    report_path = tmp_path / "stage1_report.txt"
    c1 = FeatureCandidate(1, 0.7, "a", 1, "a.json")
    c2 = FeatureCandidate(1, 0.9, "b", 2, "b.json")
    items = [
        ExecutedCandidate(candidate=c1, output_columns=("x", "y"), semantic_hash="h1"),
        ExecutedCandidate(candidate=c2, output_columns=("x", "y"), semantic_hash="h2"),
    ]

    with open(report_path, "w", encoding="utf-8") as report_file:
        survivors, dropped = dedup_stage_by_output_columns(items, report_file)

    assert dropped == 1
    assert len(survivors) == 1
    assert survivors[0].candidate.sample_order == 2


def test_dedup_stage_by_semantic_hash_keeps_highest_score(tmp_path):
    report_path = tmp_path / "stage3_report.txt"
    c1 = FeatureCandidate(1, 0.6, "a", 1, "a.json")
    c2 = FeatureCandidate(1, 0.95, "b", 2, "b.json")
    items = [
        ExecutedCandidate(candidate=c1, output_columns=("x",), semantic_hash="same"),
        ExecutedCandidate(candidate=c2, output_columns=("y",), semantic_hash="same"),
    ]

    with open(report_path, "w", encoding="utf-8") as report_file:
        survivors, dropped = dedup_stage_by_semantic_hash(items, report_file)

    assert dropped == 1
    assert len(survivors) == 1
    assert survivors[0].candidate.sample_order == 2


def test_deduplicate_candidates_multistage_reports_all_stages(tmp_path, capsys):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    report_path = tmp_path / "dedup_report.txt"

    base = (
        "def modify_features(df):\n"
        "    out = pd.DataFrame(index=df.index)\n"
        "    out['same_name'] = df['x'] + df['y']\n"
        "    return out\n"
    )
    # Different function body, same output column name signature => dropped in stage 1.
    stage1_clash = (
        "def modify_features(df):\n"
        "    out = pd.DataFrame(index=df.index)\n"
        "    out['same_name'] = (df['x'] * 2.0) + (df['y'] * 0.0)\n"
        "    return out\n"
    )
    # Name changed, so this gets past stage 1; same computation as base => stage 3 drop.
    semantic_same_1 = (
        "def modify_features(df):\n"
        "    out = pd.DataFrame(index=df.index)\n"
        "    out['different_name_1'] = (df['x'] + df['y'])\n"
        "    return out\n"
    )
    # Name changed + floating representation tweak, same values at 8 decimals => stage 3 drop.
    semantic_same_2 = (
        "def modify_features(df):\n"
        "    out = pd.DataFrame(index=df.index)\n"
        "    out['different_name_2'] = (df['x'] + df['y']) + 1e-12\n"
        "    return out\n"
    )
    # Invalid candidate to verify execution-stage drop and reporting.
    execution_fail = "def modify_features(df):\n    return 123\n"

    candidates = [
        FeatureCandidate(1, 0.90, base, 1, "base.json"),
        FeatureCandidate(1, 0.80, stage1_clash, 2, "stage1.json"),
        FeatureCandidate(1, 0.70, semantic_same_1, 3, "semantic1.json"),
        FeatureCandidate(1, 0.60, semantic_same_2, 4, "semantic2.json"),
        FeatureCandidate(1, 0.50, execution_fail, 5, "bad.json"),
    ]

    survivors, summary = deduplicate_candidates_multistage(
        candidates=candidates,
        execution_input=df,
        report_path=str(report_path),
        rounding_decimals=8,
    )

    captured = capsys.readouterr()
    report_text = report_path.read_text(encoding="utf-8")

    assert summary["dropped_execute"] == 1
    assert summary["dropped_stage_1_columns"] == 1
    assert summary["dropped_stage_2_semantic_hash"] == 2
    assert summary["total_survivors"] == 1
    assert len(survivors) == 1
    assert survivors[0].candidate.source_file == "base.json"
    assert "Deduplication stage summary:" in captured.out
    assert "Dedup report path:" in captured.out
    assert "stage=execute action=dropped" in report_text
    assert "stage=dedup_columns action=dropped" in report_text
    assert "stage=dedup_semantic_hash action=dropped" in report_text


def test_deduplicate_multistage_stage1_uses_generated_columns_only(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    report_path = tmp_path / "dedup_report_generated_only.txt"

    c1 = FeatureCandidate(
        island_id=1,
        score=0.9,
        function_code=(
            "def modify_features(df):\n"
            "    out = df.copy()\n"
            "    out['feat'] = out['x'] + out['y']\n"
            "    return out\n"
        ),
        sample_order=1,
        source_file="a.json",
    )
    c2 = FeatureCandidate(
        island_id=2,
        score=0.8,
        function_code=(
            "def modify_features(df):\n"
            "    out = df.copy()\n"
            "    out['feat'] = out['x'] + out['y']\n"
            "    out = out[['y', 'x', 'feat']]\n"
            "    return out\n"
        ),
        sample_order=2,
        source_file="b.json",
    )

    survivors, summary = deduplicate_candidates_multistage(
        candidates=[c1, c2],
        execution_input=df,
        report_path=str(report_path),
        rounding_decimals=8,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert len(survivors) == 1
    assert summary["dropped_stage_1_columns"] == 1
    assert "column_signature_scope=feature_level_generated_only" in report_text


def test_pipeline_run_writes_filtered_manifest_and_report(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    payload = {
        "island_id": 1,
        "score": 0.9,
        "sample_order": 1,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['f'] = df['x'] + 1\n"
            "    return out\n"
        ),
    }
    with open(samples_dir / "a.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_dir),
        k_per_island=1,
        label_column="target",
        include_original=True,
        data_dir=str(tmp_path / "data"),
    )

    out_df, meta = pipeline.run(df, dataset_name="demo")

    data_dir = tmp_path / "data" / "demo"
    report_path = data_dir / "dedup_report_demo.txt"
    filtered_path = data_dir / "top_1_samples.csv"

    assert "target" == out_df.columns[-1]
    assert len(meta) == 1
    assert meta.iloc[0]["status"] == "success"
    assert report_path.exists()
    assert filtered_path.exists()


def test_build_full_dataframe_short_names_collision_and_new_cols_only():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    candidates = [
        FeatureCandidate(
            island_id=1,
            score=0.9,
            function_code=(
                "def modify_features(df):\n"
                "    out = pd.DataFrame(index=df.index)\n"
                "    out['x'] = df['x'] * 10\n"
                "    out['feat'] = df['x'] + 1\n"
                "    return out\n"
            ),
            sample_order=11,
            source_file="s11.json",
        ),
        FeatureCandidate(
            island_id=2,
            score=0.8,
            function_code=(
                "def modify_features(df):\n"
                "    out = pd.DataFrame(index=df.index)\n"
                "    out['feat'] = df['x'] + 2\n"
                "    return out\n"
            ),
            sample_order=12,
            source_file="s12.json",
        ),
    ]

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        candidates=candidates,
        label_column="target",
        include_original=True,
    )

    assert "x" in out_df.columns
    assert "feat" in out_df.columns
    assert (out_df.columns == "feat").sum() == 2
    assert out_df.columns.tolist() == ["x", "feat", "feat", "target"]

    assert len(meta) == 2
    assert meta.iloc[0]["generated_columns"] == ["feat"]
    assert meta.iloc[1]["generated_columns"] == ["feat"]


def test_build_full_dataframe_excludes_label_from_generated_and_keeps_single_label_last():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    candidates = [
        FeatureCandidate(
            island_id=1,
            score=0.9,
            function_code=(
                "def modify_features(df):\n"
                "    out = df.copy()\n"
                "    out['feat'] = df['x'] + 10\n"
                "    return out\n"
            ),
            sample_order=1,
            source_file="s1.json",
        )
    ]

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        candidates=candidates,
        label_column="target",
        include_original=True,
    )

    assert out_df.columns.tolist() == ["x", "feat", "target"]
    assert (out_df.columns == "target").sum() == 1
    assert out_df.columns[-1] == "target"
    assert meta.iloc[0]["status"] == "success"


def test_build_full_dataframe_uses_cached_selected_feature_outputs(monkeypatch):
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    candidate = FeatureCandidate(
        island_id=1,
        score=0.9,
        function_code=(
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['feat'] = df['x'] + 100\n"
            "    return out\n"
        ),
        sample_order=1,
        source_file="cached.json",
    )
    cached_out = pd.DataFrame({"feat": [10, 20]}, index=df.index)
    selected_features = [
        ExecutedCandidate(
            candidate=candidate,
            output_columns=("feat",),
            semantic_hash="cached_hash",
            output_df=cached_out,
        )
    ]

    def _should_not_execute(*_args, **_kwargs):
        raise AssertionError("_execute_candidate should not be called when cached output_df is available")

    monkeypatch.setattr("ga_optimizer.evolve.preprocess._execute_candidate", _should_not_execute)

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        selected_features=selected_features,
        label_column="target",
        include_original=True,
        execution_input=df.drop(columns=["target"]),
    )

    assert out_df.columns.tolist() == ["x", "feat", "target"]
    assert out_df["feat"].tolist() == [10, 20]
    assert meta.iloc[0]["status"] == "success"


def test_build_full_dataframe_requires_exactly_one_input_mode():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    candidate = FeatureCandidate(
        island_id=1,
        score=0.9,
        function_code=(
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['feat'] = df['x'] + 1\n"
            "    return out\n"
        ),
        sample_order=1,
        source_file="s1.json",
    )
    selected_features = [
        ExecutedCandidate(
            candidate=candidate,
            output_columns=("feat",),
            semantic_hash="h1",
            output_df=pd.DataFrame({"feat": [2, 3]}, index=df.index),
        )
    ]

    with pytest.raises(
        ValueError, match="Exactly one of candidates or selected_features must be provided"
    ):
        FeatureExtractionPipeline.build_full_dataframe(
            df=df,
            label_column="target",
            include_original=True,
        )

    with pytest.raises(
        ValueError, match="Exactly one of candidates or selected_features must be provided"
    ):
        FeatureExtractionPipeline.build_full_dataframe(
            df=df,
            candidates=[candidate],
            selected_features=selected_features,
            label_column="target",
            include_original=True,
        )


def test_pipeline_run_without_label_column_does_not_infer_last_column(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    payload = {
        "island_id": 1,
        "score": 0.9,
        "sample_order": 1,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['f'] = df['x'] + 1\n"
            "    return out\n"
        ),
    }
    with open(samples_dir / "a.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_dir),
        k_per_island=1,
        label_column=None,
        include_original=True,
        data_dir=str(tmp_path / "data"),
    )

    out_df, _meta = pipeline.run(df, dataset_name="demo_no_label")
    assert out_df.columns.tolist() == ["x", "target", "f"]


def test_deduplicate_candidates_multistage_uses_configured_rounding(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    report_path = tmp_path / "dedup_rounding_report.txt"
    config = replace(DEFAULT_GA_CONFIG, dedup=DedupConfig(rounding_decimals=3))

    candidates = [
        FeatureCandidate(
            island_id=1,
            score=0.9,
            function_code=(
                "def modify_features(df):\n"
                "    out = pd.DataFrame(index=df.index)\n"
                "    out['feat_a'] = df['x'] + 0.12344\n"
                "    return out\n"
            ),
            sample_order=1,
            source_file="a.json",
        ),
        FeatureCandidate(
            island_id=1,
            score=0.8,
            function_code=(
                "def modify_features(df):\n"
                "    out = pd.DataFrame(index=df.index)\n"
                "    out['feat_b'] = df['x'] + 0.12346\n"
                "    return out\n"
            ),
            sample_order=2,
            source_file="b.json",
        ),
    ]

    survivors, summary = deduplicate_candidates_multistage(
        candidates=candidates,
        execution_input=df,
        report_path=str(report_path),
        config=config,
    )

    assert len(survivors) == 1
    assert summary["dropped_stage_2_semantic_hash"] == 1


def test_pipeline_run_uses_configured_output_templates(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    payload = {
        "island_id": 1,
        "score": 0.9,
        "sample_order": 1,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['f'] = df['x'] + 1\n"
            "    return out\n"
        ),
    }
    with open(samples_dir / "a.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    config = replace(
        DEFAULT_GA_CONFIG,
        output=OutputConfig(
            dedup_report_filename_template="report_{dataset_name}_{k_per_island}.txt",
            manifest_filename_template="manifest_{dataset_name}_{k_per_island}.csv",
            default_dataset_name="dataset",
            default_data_dir_name="data",
        ),
    )
    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_dir),
        k_per_island=1,
        label_column="target",
        include_original=True,
        data_dir=str(tmp_path / "data"),
        config=config,
    )

    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    pipeline.run(df, dataset_name="demo")

    data_dir = tmp_path / "data" / "demo"
    assert (data_dir / "report_demo_1.txt").exists()
    assert (data_dir / "manifest_demo_1.csv").exists()


def test_pipeline_run_raises_for_invalid_output_template(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    payload = {
        "island_id": 1,
        "score": 0.9,
        "sample_order": 1,
        "function_code": (
            "def modify_features(df):\n"
            "    out = pd.DataFrame(index=df.index)\n"
            "    out['f'] = df['x'] + 1\n"
            "    return out\n"
        ),
    }
    with open(samples_dir / "a.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    config = replace(
        DEFAULT_GA_CONFIG,
        output=OutputConfig(
            dedup_report_filename_template="report_{missing}.txt",
            manifest_filename_template="manifest_{k_per_island}.csv",
            default_dataset_name="dataset",
            default_data_dir_name="data",
        ),
    )
    pipeline = FeatureExtractionPipeline(
        samples_dir=str(samples_dir),
        k_per_island=1,
        label_column="target",
        include_original=True,
        data_dir=str(tmp_path / "data"),
        config=config,
    )
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})

    with pytest.raises(
        ValueError, match="Invalid dedup_report_filename_template: missing placeholder 'missing'"
    ):
        pipeline.run(df, dataset_name="demo")


def test_build_full_dataframe_uses_configured_function_names():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    candidates = [
        FeatureCandidate(
            island_id=1,
            score=0.9,
            function_code=(
                "def alt_modify(df):\n"
                "    out = pd.DataFrame(index=df.index)\n"
                "    out['feat'] = df['x'] + 1\n"
                "    return out\n"
            ),
            sample_order=1,
            source_file="alt.json",
        )
    ]
    config = replace(DEFAULT_GA_CONFIG, execution=ExecutionConfig(preferred_function_names=("alt_modify",)))

    out_df, meta = FeatureExtractionPipeline.build_full_dataframe(
        df=df,
        candidates=candidates,
        label_column="target",
        include_original=True,
        config=config,
    )

    assert out_df.columns.tolist() == ["x", "feat", "target"]
    assert meta.iloc[0]["status"] == "success"
