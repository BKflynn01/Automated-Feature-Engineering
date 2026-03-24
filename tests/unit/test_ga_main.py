import argparse

import pytest

import ga_optimizer.ga_main as ga_main
from ga_optimizer.config import DEFAULT_GA_CONFIG


def _build_args(**overrides):
    args = {
        "config": None,
        "preset": DEFAULT_GA_CONFIG.ga_presets.default_preset.value,
        "profile": None,
        "dataset_name": None,
        "input_csv": None,
        "label_column": None,
        "task": None,
        "scoring": None,
        "output_dir": None,
        "is_time_series": None,
        "time_series_train_window": None,
        "time_series_test_window": None,
        "time_series_step_window": None,
        "time_series_gap": None,
        "max_time_series_splits": None,
        "n_generations": None,
        "population_size": None,
        "cx_prob": None,
        "mut_prob": None,
        "tournament_size": None,
        "cv_folds": None,
        "n_estimators": None,
        "n_jobs": None,
        "early_stop_patience": None,
        "early_stop_min_delta": None,
        "random_state": None,
        "sep": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_parse_args_defaults_to_normal_preset(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    args = ga_main.parse_args()
    assert args.config is None
    assert args.profile is None
    assert args.preset == DEFAULT_GA_CONFIG.ga_presets.default_preset.value


def test_resolve_run_settings_rejects_profile_with_config():
    args = _build_args(config="cfg.yaml", profile="btc_classification")
    with pytest.raises(ValueError, match="Cannot use both --profile and --config"):
        ga_main.resolve_run_settings(args)


def test_main_raises_for_missing_input_csv(monkeypatch, tmp_path):
    args = _build_args(input_csv=str(tmp_path / "missing.csv"))
    monkeypatch.setattr(ga_main, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="input_csv does not exist"):
        ga_main.main()


def test_main_calls_run_ga_and_prints_features(monkeypatch, tmp_path, capsys):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("f1,f2,target\n1,2,0\n", encoding="utf-8")
    args = _build_args(
        input_csv=str(input_csv),
        dataset_name="my-dataset",
        label_column="target",
        task="classification",
        scoring="accuracy",
    )
    captured = {}

    def _fake_run_ga(**kwargs):
        captured.update(kwargs)
        return ["f1"], 0.9

    monkeypatch.setattr(ga_main, "parse_args", lambda: args)
    monkeypatch.setattr(ga_main, "run_ga", _fake_run_ga)

    with pytest.warns(DeprecationWarning):
        ga_main.main()

    out = capsys.readouterr().out
    assert "Resolved GA run settings:" in out
    assert "Final selected features:" in out
    assert "f1" in out
    assert captured["csv_path"] == str(input_csv)
    assert captured["label_column"] == "target"
    assert captured["output_dir"].endswith("ga_optimizer/data/my-dataset")


def test_resolve_run_settings_uses_legacy_profile_defaults():
    args = _build_args()
    with pytest.warns(DeprecationWarning):
        settings = ga_main.resolve_run_settings(args)
    assert settings["profile"] == DEFAULT_GA_CONFIG.ga_profiles.default_profile
    assert settings["dataset_name"] == "btc-classification"
    assert settings["scoring"] == "accuracy"
    assert settings["output_dir"].endswith("ga_optimizer/data/btc-classification")


def test_resolve_run_settings_uses_dataset_yaml_then_preset_then_cli(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: demo",
                "input_csv: ./features.csv",
                "label_column: target",
                "task: regression",
                "is_time_series: true",
                "time_series_train_window: 20",
                "time_series_test_window: 5",
                "time_series_step_window: 5",
                "time_series_gap: 1",
            ]
        ),
        encoding="utf-8",
    )
    args = _build_args(
        config=str(config_path),
        preset="quick",
        cv_folds=9,
        random_state=7,
    )

    settings = ga_main.resolve_run_settings(args)
    assert settings["profile"] is None
    assert settings["preset"] == "quick"
    assert settings["dataset_name"] == "demo"
    assert settings["n_generations"] == 2
    assert settings["population_size"] == 6
    assert settings["n_estimators"] is None
    assert settings["max_time_series_splits"] is None
    assert settings["early_stop_patience"] == 1
    assert settings["cv_folds"] == 9
    assert settings["random_state"] == 7
    assert settings["input_csv"] == str((tmp_path / "features.csv").resolve())
    assert settings["output_dir"].endswith("ga_optimizer/data/demo")
