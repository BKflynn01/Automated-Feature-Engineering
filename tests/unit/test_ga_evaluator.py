import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

import ga_optimizer.ga.evaluator as ga_evaluator
from ga_optimizer.config import DEFAULT_GA_CONFIG, XGBRFConfig
from ga_optimizer.ga.evaluator import build_xgbrf_evaluator


def test_evaluator_returns_zero_for_empty_feature_subset():
    X_np, y_np = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=3,
        n_estimators=20,
        random_state=42,
    )

    assert evaluator([]) == 0.0


def test_evaluator_returns_reasonable_classification_score_for_all_features():
    X_np, y_np = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        random_state=7,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    feature_names = X.columns.tolist()

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=3,
        n_estimators=25,
        random_state=123,
    )
    score = evaluator(feature_names)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_evaluator_returns_reasonable_regression_score_for_all_features():
    X_np, y_np = make_regression(
        n_samples=240,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=11,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    feature_names = X.columns.tolist()

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="regression",
        scoring="r2",
        cv_folds=3,
        n_estimators=40,
        random_state=99,
    )
    score = evaluator(feature_names)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 1.0
    assert score > 0.0


def test_evaluator_is_picklable_for_parallel_use():
    X_np, y_np = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        random_state=1,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=3,
        n_estimators=15,
        random_state=1,
    )

    blob = pickle.dumps(evaluator)
    restored = pickle.loads(blob)
    score = restored(X.columns.tolist())

    assert isinstance(score, float)


def test_evaluator_can_be_called_multiple_times_without_reloading_data():
    X_np, y_np = make_classification(
        n_samples=90,
        n_features=7,
        n_informative=5,
        random_state=13,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    feature_names = X.columns.tolist()

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=3,
        n_estimators=20,
        random_state=13,
    )

    first = evaluator(feature_names)
    second = evaluator(feature_names)

    assert isinstance(first, float)
    assert isinstance(second, float)
    assert np.isfinite(first)
    assert np.isfinite(second)


def test_evaluator_uses_xgboost_params_from_config():
    X_np, y_np = make_classification(
        n_samples=70,
        n_features=6,
        n_informative=4,
        random_state=21,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    config = replace(
        DEFAULT_GA_CONFIG,
        xgbrf=XGBRFConfig(
            n_estimators=37,
            random_state=77,
            verbosity=0,
        ),
    )

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=3,
        config=config,
    )

    assert evaluator.n_estimators == 37
    assert evaluator.random_state == 77


def test_evaluator_supports_categorical_features():
    X = pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "c", "a", "b", "c", "a", "b", "c"], dtype="string"),
            "num": [1.0, 2.2, 1.8, 1.5, 2.6, 2.1, 1.3, 2.4, 1.9],
        }
    )
    y = pd.Series([0, 1, 0, 0, 1, 1, 0, 1, 0])

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=3,
        random_state=42,
    )
    score = evaluator(["cat", "num"])

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_evaluator_supports_sliding_window_time_series_scoring():
    n = 60
    X = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, n),
            "x2": np.linspace(0.0, 10.0, n),
        }
    )
    y = pd.Series(np.linspace(0.0, 2.0, n) + 0.1 * np.sin(np.arange(n)))

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="regression",
        scoring="neg_normalized_root_mean_squared_error",
        cv_folds=4,
        is_time_series=True,
        time_series_train_window=24,
        time_series_test_window=6,
        time_series_step_window=6,
        random_state=42,
    )
    score = evaluator(["x1", "x2"])

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_evaluator_caps_time_series_splits_to_most_recent():
    n = 80
    X = pd.DataFrame({"x1": np.arange(n), "x2": np.arange(n) * 2})
    y = pd.Series(np.arange(n) * 0.1)

    uncapped = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="regression",
        scoring="neg_root_mean_squared_error",
        cv_folds=4,
        is_time_series=True,
        time_series_train_window=20,
        time_series_test_window=5,
        time_series_step_window=5,
        time_series_gap=0,
        max_time_series_splits=None,
        random_state=42,
    )
    capped = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="regression",
        scoring="neg_root_mean_squared_error",
        cv_folds=4,
        is_time_series=True,
        time_series_train_window=20,
        time_series_test_window=5,
        time_series_step_window=5,
        time_series_gap=0,
        max_time_series_splits=3,
        random_state=42,
    )

    uncapped_splits = uncapped._iter_splits(X, y.to_numpy())
    capped_splits = capped._iter_splits(X, y.to_numpy())
    assert len(uncapped_splits) > 3
    assert len(capped_splits) == 3
    expected_recent = uncapped_splits[-3:]
    for (train_a, test_a), (train_b, test_b) in zip(capped_splits, expected_recent):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)


def test_evaluator_rejects_non_positive_time_series_split_cap():
    X = pd.DataFrame({"x1": np.arange(40), "x2": np.arange(40) * 2})
    y = pd.Series(np.arange(40) * 0.1)

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="regression",
        scoring="neg_root_mean_squared_error",
        cv_folds=3,
        is_time_series=True,
        time_series_train_window=20,
        time_series_test_window=5,
        time_series_step_window=5,
        time_series_gap=0,
        max_time_series_splits=0,
        random_state=42,
    )

    with pytest.raises(ValueError, match="max_time_series_splits must be > 0"):
        evaluator._iter_splits(X, y.to_numpy())


def test_evaluator_auto_device_uses_cuda_when_available(monkeypatch):
    X_np, y_np = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        random_state=8,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    config = replace(
        DEFAULT_GA_CONFIG,
        xgbrf=XGBRFConfig(
            n_estimators=10,
            random_state=42,
            verbosity=0,
            device="auto",
        ),
    )
    monkeypatch.setattr(ga_evaluator, "_gpu_available_for_xgboost", lambda: True)

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=2,
        config=config,
    )

    assert evaluator.device == "cuda"


def test_evaluator_auto_device_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    X_np, y_np = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        random_state=9,
    )
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])
    y = pd.Series(y_np)
    config = replace(
        DEFAULT_GA_CONFIG,
        xgbrf=XGBRFConfig(
            n_estimators=10,
            random_state=42,
            verbosity=0,
            device="auto",
        ),
    )
    monkeypatch.setattr(ga_evaluator, "_gpu_available_for_xgboost", lambda: False)

    evaluator = build_xgbrf_evaluator(
        X=X,
        y=y,
        task="classification",
        scoring="accuracy",
        cv_folds=2,
        config=config,
    )

    assert evaluator.device == "cpu"
