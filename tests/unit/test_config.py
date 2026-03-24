from dataclasses import FrozenInstanceError

import pytest

from ga_optimizer.config import DEFAULT_GA_CONFIG, GAPreset, get_ga_preset_values


def test_default_ga_config_values_match_expected_baseline():
    assert DEFAULT_GA_CONFIG.main.default_sep == ","
    assert DEFAULT_GA_CONFIG.main.default_k_per_island == 2
    assert DEFAULT_GA_CONFIG.selection.default_top_k_per_island == 2
    assert DEFAULT_GA_CONFIG.dedup.rounding_decimals == 8
    assert DEFAULT_GA_CONFIG.execution.preferred_function_names == (
        "modify_features_v2",
        "modify_features",
    )
    assert (
        DEFAULT_GA_CONFIG.output.dedup_report_filename_template
        == "dedup_report_{dataset_name}.txt"
    )
    assert (
        DEFAULT_GA_CONFIG.output.manifest_filename_template
        == "top_{k_per_island}_samples.csv"
    )
    assert DEFAULT_GA_CONFIG.ga_data.default_csv_sep == ","
    assert DEFAULT_GA_CONFIG.ga_data.selected_gene_value == 1
    assert DEFAULT_GA_CONFIG.xgbrf.n_estimators is None
    assert DEFAULT_GA_CONFIG.xgbrf.subsample is None
    assert DEFAULT_GA_CONFIG.xgbrf.colsample_bynode is None
    assert DEFAULT_GA_CONFIG.xgbrf.random_state == 42
    assert DEFAULT_GA_CONFIG.xgbrf.verbosity == 0
    assert DEFAULT_GA_CONFIG.xgbrf.device == "auto"
    assert DEFAULT_GA_CONFIG.evaluator.cv_folds == 4
    assert DEFAULT_GA_CONFIG.evaluator.time_series_train_window is None
    assert DEFAULT_GA_CONFIG.evaluator.time_series_test_window is None
    assert DEFAULT_GA_CONFIG.evaluator.time_series_step_window is None
    assert DEFAULT_GA_CONFIG.evaluator.time_series_gap == 0
    assert DEFAULT_GA_CONFIG.evaluator.max_time_series_splits is None
    assert DEFAULT_GA_CONFIG.toolbox.fitness_weights == (1.0,)
    assert DEFAULT_GA_CONFIG.toolbox.attr_bool_min == 0
    assert DEFAULT_GA_CONFIG.toolbox.attr_bool_max == 1
    assert DEFAULT_GA_CONFIG.toolbox.population_size == 50
    assert DEFAULT_GA_CONFIG.toolbox.dynamic_population_min == 8
    assert DEFAULT_GA_CONFIG.toolbox.dynamic_population_max == 64
    assert DEFAULT_GA_CONFIG.toolbox.dynamic_population_ratio == 1.0
    assert DEFAULT_GA_CONFIG.toolbox.cx_prob == 0.5
    assert DEFAULT_GA_CONFIG.toolbox.mut_prob == 0.2
    assert DEFAULT_GA_CONFIG.toolbox.tournament_size == 3
    assert DEFAULT_GA_CONFIG.runner.n_generations == 40
    assert DEFAULT_GA_CONFIG.runner.verbose is True
    assert DEFAULT_GA_CONFIG.runner.n_jobs == 1
    assert DEFAULT_GA_CONFIG.runner.early_stop_patience is None
    assert DEFAULT_GA_CONFIG.runner.early_stop_min_delta == 0.0
    assert DEFAULT_GA_CONFIG.runner.best_features_filename == "ga_best_features.txt"
    assert DEFAULT_GA_CONFIG.runner.best_features_csv_filename == "ga_best_features.csv"
    assert (
        DEFAULT_GA_CONFIG.runner.best_feature_dataset_filename
        == "ga_best_feature_dataset.csv"
    )
    assert (
        DEFAULT_GA_CONFIG.runner.generation_trace_filename == "ga_generation_trace.csv"
    )
    assert DEFAULT_GA_CONFIG.runner.logbook_filename == "ga_logbook.csv"
    assert DEFAULT_GA_CONFIG.runner.run_manifest_filename == "ga_run_manifest.json"
    assert DEFAULT_GA_CONFIG.runner.default_classification_scoring == "accuracy"
    assert (
        DEFAULT_GA_CONFIG.runner.default_regression_scoring
        == "neg_root_mean_squared_error"
    )
    assert (
        DEFAULT_GA_CONFIG.runner.default_time_series_regression_scoring
        == "neg_normalized_root_mean_squared_error"
    )
    assert DEFAULT_GA_CONFIG.ga_profiles.default_profile == "btc_classification"
    assert len(DEFAULT_GA_CONFIG.ga_profiles.profiles) >= 2
    assert {profile.name for profile in DEFAULT_GA_CONFIG.ga_profiles.profiles} >= {
        "btc_classification",
        "btc_classification_quick",
    }
    assert DEFAULT_GA_CONFIG.ga_presets.default_preset == GAPreset.NORMAL

    quick = get_ga_preset_values("quick")
    normal = get_ga_preset_values("normal")
    extended = get_ga_preset_values("extended")

    assert quick.n_generations == 2
    assert quick.cv_folds == 4
    assert quick.n_estimators is None
    assert quick.population_size == 6
    assert quick.max_time_series_splits is None
    assert quick.early_stop_patience == 1

    assert normal.n_generations == 15
    assert normal.cv_folds == 4
    assert normal.n_estimators is None
    assert normal.population_size == 0
    assert normal.max_time_series_splits is None
    assert normal.early_stop_patience == 5

    assert extended.n_generations == 40
    assert extended.cv_folds == 4
    assert extended.n_estimators is None
    assert extended.population_size == 0
    assert extended.max_time_series_splits is None
    assert extended.early_stop_patience == 10


def test_default_ga_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_GA_CONFIG.main.default_k_per_island = 9
