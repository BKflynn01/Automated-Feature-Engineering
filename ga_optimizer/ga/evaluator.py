from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRFClassifier, XGBRFRegressor

from ga_optimizer.config import DEFAULT_GA_CONFIG, GAOptimizerConfig


@dataclass
class _XGBRFEvaluator:
    X: pd.DataFrame
    y: pd.Series
    task: Literal["classification", "regression"]
    scoring: str
    cv_folds: int
    n_estimators: int
    subsample: float
    colsample_bynode: float
    random_state: int
    verbosity: int

    def _build_estimator(self) -> XGBRFClassifier | XGBRFRegressor:
        common_params = {
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bynode": self.colsample_bynode,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
        }
        if self.task == "classification":
            return XGBRFClassifier(**common_params)
        if self.task == "regression":
            return XGBRFRegressor(**common_params)
        raise ValueError(f"Unsupported task: {self.task}")

    def __call__(self, selected_features: List[str]) -> float:
        if not selected_features:
            return 0.0

        X_selected = self.X.loc[:, selected_features]
        model = self._build_estimator()
        scores = cross_val_score(
            model,
            X_selected,
            self.y,
            cv=self.cv_folds,
            scoring=self.scoring,
        )
        return float(np.mean(scores))


def build_xgbrf_evaluator(
    X: pd.DataFrame,
    y: pd.Series,
    task: Literal["classification", "regression"],
    scoring: str,
    cv_folds: int = 3,
    n_estimators: Optional[int] = None,
    subsample: Optional[float] = None,
    colsample_bynode: Optional[float] = None,
    random_state: Optional[int] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> Callable[[List[str]], float]:
    resolved_config = config or DEFAULT_GA_CONFIG
    model_config = resolved_config.xgbrf
    return _XGBRFEvaluator(
        X=X,
        y=y,
        task=task,
        scoring=scoring,
        cv_folds=cv_folds,
        n_estimators=n_estimators if n_estimators is not None else model_config.n_estimators,
        subsample=subsample if subsample is not None else model_config.subsample,
        colsample_bynode=(
            colsample_bynode if colsample_bynode is not None else model_config.colsample_bynode
        ),
        random_state=random_state if random_state is not None else model_config.random_state,
        verbosity=model_config.verbosity,
    )
