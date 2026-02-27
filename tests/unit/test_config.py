from dataclasses import FrozenInstanceError

import pytest

from ga_optimizer.config import DEFAULT_GA_CONFIG


def test_default_ga_config_values_match_expected_baseline():
    assert DEFAULT_GA_CONFIG.main.default_sep == ","
    assert DEFAULT_GA_CONFIG.main.default_k_per_island == 2
    assert DEFAULT_GA_CONFIG.selection.default_top_k_per_island == 2
    assert DEFAULT_GA_CONFIG.dedup.rounding_decimals == 8
    assert DEFAULT_GA_CONFIG.execution.preferred_function_names == (
        "modify_features_v2",
        "modify_features",
    )
    assert DEFAULT_GA_CONFIG.output.dedup_report_filename_template == "dedup_report_{dataset_name}.txt"
    assert DEFAULT_GA_CONFIG.output.manifest_filename_template == "top_{k_per_island}_samples.csv"
    assert DEFAULT_GA_CONFIG.ga_data.default_csv_sep == ","
    assert DEFAULT_GA_CONFIG.ga_data.selected_gene_value == 1
    assert DEFAULT_GA_CONFIG.xgbrf.n_estimators == 100
    assert DEFAULT_GA_CONFIG.xgbrf.subsample == 0.8
    assert DEFAULT_GA_CONFIG.xgbrf.colsample_bynode == 0.8
    assert DEFAULT_GA_CONFIG.xgbrf.random_state == 42
    assert DEFAULT_GA_CONFIG.xgbrf.verbosity == 0


def test_default_ga_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_GA_CONFIG.main.default_k_per_island = 9
