from __future__ import annotations

from pathlib import Path

import pytest

from ga_optimizer.config import load_dataset_run_config


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_dataset_run_config_minimal_valid(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
dataset_name: ds
input_csv: ./features.csv
label_column: target
task: classification
""".strip(),
    )

    cfg = load_dataset_run_config(cfg_path)

    assert cfg.dataset_name == "ds"
    assert cfg.label_column == "target"
    assert cfg.task == "classification"
    assert cfg.is_time_series is False
    assert cfg.time_series_gap == 0
    assert cfg.sep == ","
    assert cfg.input_csv == str((tmp_path / "features.csv").resolve())


def test_load_dataset_run_config_rejects_unknown_keys(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
dataset_name: ds
input_csv: ./features.csv
label_column: target
task: classification
unknown_field: nope
""".strip(),
    )

    with pytest.raises(ValueError, match="unknown keys"):
        load_dataset_run_config(cfg_path)


def test_load_dataset_run_config_requires_task(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
dataset_name: ds
input_csv: ./features.csv
label_column: target
""".strip(),
    )

    with pytest.raises(ValueError, match="'task' must be a non-empty string"):
        load_dataset_run_config(cfg_path)


def test_load_dataset_run_config_rejects_invalid_task(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
dataset_name: ds
input_csv: ./features.csv
label_column: target
task: clustering
""".strip(),
    )

    with pytest.raises(ValueError, match="must be either 'classification' or 'regression'"):
        load_dataset_run_config(cfg_path)


def test_load_dataset_run_config_validates_numeric_fields(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
dataset_name: ds
input_csv: ./features.csv
label_column: target
task: classification
time_series_gap: bad
""".strip(),
    )

    with pytest.raises(ValueError, match="time_series_gap"):
        load_dataset_run_config(cfg_path)
