import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

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
            subsample=0.65,
            colsample_bynode=0.55,
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
    assert evaluator.subsample == 0.65
    assert evaluator.colsample_bynode == 0.55
    assert evaluator.random_state == 77
