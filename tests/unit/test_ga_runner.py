from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

import ga_optimizer.ga.runner as ga_runner
from ga_optimizer.ga.runner import run_ga
from ga_optimizer.ga.toolbox import build_toolbox


def _build_classification_csv(csv_path: Path) -> None:
    X_np, y_np = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    df["target"] = y_np
    df.to_csv(csv_path, index=False)


def test_run_ga_writes_expected_outputs_and_returns_best_subset(tmp_path):
    csv_path = tmp_path / "ga_input.csv"
    output_dir = tmp_path / "ga_output"
    _build_classification_csv(csv_path)

    best_features, best_score = run_ga(
        csv_path=str(csv_path),
        label_column="target",
        task="classification",
        scoring="accuracy",
        output_dir=str(output_dir),
        n_generations=3,
        population_size=8,
        cx_prob=0.5,
        mut_prob=0.2,
        tournament_size=3,
        cv_folds=2,
        n_estimators=12,
        random_state=42,
        sep=",",
    )

    best_features_path = output_dir / "ga_best_features.txt"
    best_features_csv_path = output_dir / "ga_best_features.csv"
    best_feature_dataset_path = output_dir / "ga_best_feature_dataset.csv"
    generation_trace_path = output_dir / "ga_generation_trace.csv"
    logbook_path = output_dir / "ga_logbook.csv"
    run_manifest_path = output_dir / "ga_run_manifest.json"

    assert isinstance(best_score, float)
    assert len(best_features) >= 1
    assert best_features_path.exists()
    assert best_features_csv_path.exists()
    assert best_feature_dataset_path.exists()
    assert generation_trace_path.exists()
    assert logbook_path.exists()
    assert run_manifest_path.exists()

    file_best_features = [
        line.strip()
        for line in best_features_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert file_best_features == best_features

    best_features_df = pd.read_csv(best_features_csv_path)
    assert best_features_df.columns.tolist() == ["feature_name"]
    assert best_features_df["feature_name"].tolist() == best_features

    reduced_df = pd.read_csv(best_feature_dataset_path)
    assert reduced_df.columns[-1] == "target"
    assert reduced_df.shape[0] == 90
    assert reduced_df.columns[:-1].tolist() == best_features

    trace_df = pd.read_csv(generation_trace_path)
    assert trace_df.columns.tolist() == [
        "generation",
        "best_going_in",
        "generation_best",
        "best_so_far",
        "avg",
        "std",
        "min",
        "max",
    ]
    assert trace_df["generation"].iloc[0] == 0
    assert trace_df["generation"].iloc[-1] == 3
    assert trace_df["best_so_far"].is_monotonic_increasing

    logbook_df = pd.read_csv(logbook_path)
    for column in ["gen", "avg", "std", "min", "max"]:
        assert column in logbook_df.columns
    assert logbook_df["gen"].min() == 0
    assert logbook_df["gen"].max() == 3


def test_run_ga_is_reproducible_with_fixed_random_state(tmp_path):
    csv_path = tmp_path / "ga_input.csv"
    output_dir_1 = tmp_path / "ga_output_1"
    output_dir_2 = tmp_path / "ga_output_2"
    _build_classification_csv(csv_path)

    best_1, score_1 = run_ga(
        csv_path=str(csv_path),
        label_column="target",
        task="classification",
        scoring="accuracy",
        output_dir=str(output_dir_1),
        n_generations=2,
        population_size=8,
        cx_prob=0.5,
        mut_prob=0.2,
        tournament_size=3,
        cv_folds=2,
        n_estimators=10,
        random_state=99,
        sep=",",
    )
    best_2, score_2 = run_ga(
        csv_path=str(csv_path),
        label_column="target",
        task="classification",
        scoring="accuracy",
        output_dir=str(output_dir_2),
        n_generations=2,
        population_size=8,
        cx_prob=0.5,
        mut_prob=0.2,
        tournament_size=3,
        cv_folds=2,
        n_estimators=10,
        random_state=99,
        sep=",",
    )

    file_1 = (output_dir_1 / "ga_best_features.txt").read_text(encoding="utf-8")
    file_2 = (output_dir_2 / "ga_best_features.txt").read_text(encoding="utf-8")
    trace_1 = (output_dir_1 / "ga_generation_trace.csv").read_text(encoding="utf-8")
    trace_2 = (output_dir_2 / "ga_generation_trace.csv").read_text(encoding="utf-8")

    assert file_1 == file_2
    assert trace_1 == trace_2
    assert best_1 == best_2
    assert score_1 == score_2


def test_run_ga_dynamic_population_size_is_derived_from_feature_count(tmp_path):
    csv_path = tmp_path / "ga_input.csv"
    output_dir = tmp_path / "ga_output_dynamic"
    _build_classification_csv(csv_path)

    _best_features, _best_score = run_ga(
        csv_path=str(csv_path),
        label_column="target",
        task="classification",
        scoring="accuracy",
        output_dir=str(output_dir),
        n_generations=1,
        population_size=0,
        cx_prob=0.5,
        mut_prob=0.2,
        tournament_size=3,
        cv_folds=2,
        n_estimators=10,
        random_state=42,
        sep=",",
    )

    trace_df = pd.read_csv(output_dir / "ga_generation_trace.csv")
    manifest = pd.read_json(output_dir / "ga_run_manifest.json", typ="series")

    # Input has 6 feature columns; with min-bound this resolves to 8.
    assert manifest["requested_population_size"] == 0
    assert manifest["population_size"] == 8
    assert trace_df.shape[0] == 2  # gen 0 and gen 1


def test_evaluate_invalid_individuals_uses_cache_prepass():
    calls = {"count": 0}

    def _evaluate(selected_features):
        calls["count"] += 1
        return float(len(selected_features))

    toolbox = build_toolbox(
        n_genes=4,
        evaluate_fn=_evaluate,
        feature_names=["f1", "f2", "f3", "f4"],
    )
    ind_a = toolbox.individual()
    ind_b = toolbox.individual()
    ind_a[:] = [1, 0, 1, 0]
    ind_b[:] = [1, 0, 1, 0]  # duplicate chromosome
    population = [ind_a, ind_b]
    cache: dict[tuple[int, ...], float] = {}

    evaluated = ga_runner._evaluate_invalid_individuals(
        toolbox, population, cache, verbose=False
    )

    assert evaluated == 2
    assert calls["count"] == 1
    assert cache[(1, 0, 1, 0)] == 2.0


def test_run_ga_validates_n_jobs(tmp_path):
    csv_path = tmp_path / "ga_input.csv"
    _build_classification_csv(csv_path)

    with pytest.raises(ValueError, match="n_jobs must be > 0"):
        run_ga(
            csv_path=str(csv_path),
            label_column="target",
            task="classification",
            scoring="accuracy",
            output_dir=str(tmp_path / "out"),
            n_generations=1,
            population_size=6,
            cv_folds=2,
            n_estimators=5,
            n_jobs=0,
        )


def test_run_ga_early_stopping_reduces_effective_generations(tmp_path):
    csv_path = tmp_path / "ga_input.csv"
    output_dir = tmp_path / "ga_output_early_stop"
    _build_classification_csv(csv_path)

    run_ga(
        csv_path=str(csv_path),
        label_column="target",
        task="classification",
        scoring="accuracy",
        output_dir=str(output_dir),
        n_generations=8,
        population_size=8,
        cx_prob=0.5,
        mut_prob=0.2,
        tournament_size=3,
        cv_folds=2,
        n_estimators=8,
        early_stop_patience=1,
        early_stop_min_delta=10.0,
        random_state=42,
        sep=",",
    )

    trace_df = pd.read_csv(output_dir / "ga_generation_trace.csv")
    assert trace_df["generation"].max() < 8


def test_run_ga_uses_pool_map_when_n_jobs_gt_1(tmp_path, monkeypatch):
    csv_path = tmp_path / "ga_input.csv"
    output_dir = tmp_path / "ga_output_n_jobs"
    _build_classification_csv(csv_path)

    class FakePool:
        def __init__(self, processes: int):
            self.processes = processes
            self.closed = False
            self.joined = False
            self.mapped = False

        def map(self, fn, values):
            self.mapped = True
            return [fn(v) for v in values]

        def close(self):
            self.closed = True

        def join(self):
            self.joined = True

    created = {"pool": None}

    def _make_pool(processes: int):
        pool = FakePool(processes)
        created["pool"] = pool
        return pool

    monkeypatch.setattr(ga_runner.mp, "Pool", _make_pool)

    run_ga(
        csv_path=str(csv_path),
        label_column="target",
        task="classification",
        scoring="accuracy",
        output_dir=str(output_dir),
        n_generations=1,
        population_size=8,
        cv_folds=2,
        n_estimators=5,
        n_jobs=2,
        random_state=42,
    )

    assert created["pool"] is not None
    assert created["pool"].processes == 2
    assert created["pool"].mapped is True
    assert created["pool"].closed is True
    assert created["pool"].joined is True
