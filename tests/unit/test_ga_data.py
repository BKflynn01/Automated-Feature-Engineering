from dataclasses import replace

import pandas as pd
import pytest

from ga_optimizer.config import DEFAULT_GA_CONFIG, GADataConfig
from ga_optimizer.ga.data import chromosome_length, decode_chromosome, load_ga_input


def test_load_ga_input_returns_shapes_and_ordered_feature_names(tmp_path):
    csv_path = tmp_path / "merged.csv"
    source = pd.DataFrame(
        {
            "feat_a": [1, 2, 3],
            "feat_b": [10, 20, 30],
            "target": [0, 1, 0],
        }
    )
    source.to_csv(csv_path, index=False)

    X, y, feature_names = load_ga_input(str(csv_path), label_column="target")

    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert feature_names == ["feat_a", "feat_b"]
    assert X.columns.tolist() == feature_names
    assert y.name == "target"


def test_load_ga_input_raises_when_label_missing(tmp_path):
    csv_path = tmp_path / "merged.csv"
    pd.DataFrame({"feat_a": [1, 2], "target": [0, 1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="label column 'missing' not found"):
        load_ga_input(str(csv_path), label_column="missing")


def test_load_ga_input_raises_when_no_features_after_label_drop(tmp_path):
    csv_path = tmp_path / "merged.csv"
    pd.DataFrame({"target": [0, 1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="CSV has no feature columns after removing label column"):
        load_ga_input(str(csv_path), label_column="target")


def test_chromosome_length():
    assert chromosome_length(["a", "b", "c"]) == 3


def test_decode_chromosome_maps_binary_genes_to_feature_names():
    selected = decode_chromosome([1, 0, 1, 0], ["f1", "f2", "f3", "f4"])
    assert selected == ["f1", "f3"]


def test_decode_chromosome_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="chromosome length must match feature_names length"):
        decode_chromosome([1, 0], ["f1"])


def test_load_ga_input_default_sep_matches_shared_config():
    assert load_ga_input.__defaults__[0] == DEFAULT_GA_CONFIG.ga_data.default_csv_sep


def test_decode_chromosome_uses_configured_selected_gene_value(monkeypatch):
    monkeypatch.setattr(
        "ga_optimizer.ga.data.DEFAULT_GA_CONFIG",
        replace(DEFAULT_GA_CONFIG, ga_data=GADataConfig(selected_gene_value=2)),
    )

    selected = decode_chromosome([1, 2, 0, 2], ["f1", "f2", "f3", "f4"])
    assert selected == ["f2", "f4"]
