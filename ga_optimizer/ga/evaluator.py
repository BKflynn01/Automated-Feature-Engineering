from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, List, Literal, Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, get_scorer, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from cv_utils import iter_sliding_window_splits, resolve_sliding_window_params
from ga_optimizer.config import DEFAULT_GA_CONFIG, GAOptimizerConfig
from preprocessing import preprocess_datasets

"""Cross-validated XGBoost evaluator used by the GA feature-selection loop.

The evaluator prefers GPU execution when configured, but prediction is designed
to degrade gracefully when CUDA/CuPy/runtime compatibility issues occur.
"""


@lru_cache(maxsize=1)
def _gpu_available_for_xgboost() -> bool:
    """Return True only if XGBoost is CUDA-enabled and a tiny CUDA fit succeeds.

    This performs a conservative runtime probe. Any failure (missing CUDA build,
    unavailable device, driver mismatch, etc.) returns False so callers can
    avoid hard failures and choose a CPU-safe path.
    """
    try:
        build_info = xgb.build_info()
        if not bool(build_info.get("USE_CUDA", False)):
            return False
    except Exception:
        return False

    try:
        X_probe = np.asarray([[0.0], [1.0]], dtype=float)
        y_probe = np.asarray([0, 1], dtype=int)
        probe_model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            random_state=0,
            verbosity=0,
            device="cuda",
            tree_method="hist",
        )
        probe_model.fit(X_probe, y_probe)
        return True
    except Exception:
        return False


def _resolve_xgboost_device(requested_device: str) -> str:
    """Resolve configured device value into the concrete device string.

    - ``"auto"`` performs a runtime CUDA probe and returns ``"cuda"`` when
      available, otherwise ``"cpu"``.
    - Any explicit value (for example ``"cuda"``) is preserved, which lets the
      caller force GPU usage.
    """
    normalized = str(requested_device).strip().lower()
    if normalized == "auto":
        return "cuda" if _gpu_available_for_xgboost() else "cpu"
    return normalized


def _to_ndarray(value: object) -> npt.NDArray[Any]:
    """Convert an array-like prediction result to a NumPy ndarray.

    This normalizes outputs from third-party libraries whose type stubs may be
    too broad for static type checking, while preserving the evaluator's
    contract of returning a NumPy array.
    """
    return cast(npt.NDArray[Any], np.asarray(cast(Any, value)))


@dataclass
class _XGBRFEvaluator:
    """Callable evaluator that scores a feature subset via cross-validation."""

    X: pd.DataFrame
    y: pd.Series
    task: Literal["classification", "regression"]
    scoring: str
    cv_folds: int
    is_time_series: bool
    time_series_train_window: Optional[int]
    time_series_test_window: Optional[int]
    time_series_step_window: Optional[int]
    time_series_gap: int
    max_time_series_splits: Optional[int]
    n_estimators: Optional[int]
    random_state: int
    verbosity: int
    device: str

    def _build_estimator(self) -> xgb.XGBClassifier | xgb.XGBRegressor:
        """Create an XGBoost estimator configured for the current task/device."""
        common_params: dict[str, object] = {
            "random_state": self.random_state,
        }
        # Match LLM-FE defaults; only add explicit device when using CUDA.
        if self.device == "cuda":
            common_params["device"] = "cuda"
        if self.n_estimators is not None:
            common_params["n_estimators"] = self.n_estimators

        if self.task == "classification":
            return xgb.XGBClassifier(**common_params)
        if self.task == "regression":
            return xgb.XGBRegressor(**common_params)
        raise ValueError(f"Unsupported task: {self.task}")

    def _iter_splits(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build train/test indices for time-series or standard K-fold CV."""
        if self.is_time_series:
            train_window, test_window, step_window = resolve_sliding_window_params(
                n_samples=len(X.index),
                n_splits=self.cv_folds,
                train_window=self.time_series_train_window,
                test_window=self.time_series_test_window,
                step_window=self.time_series_step_window,
                gap=self.time_series_gap,
            )
            splits = list(
                iter_sliding_window_splits(
                    n_samples=len(X.index),
                    train_window=train_window,
                    test_window=test_window,
                    step_window=step_window,
                    gap=self.time_series_gap,
                )
            )
            if self.max_time_series_splits is not None:
                if self.max_time_series_splits <= 0:
                    raise ValueError("max_time_series_splits must be > 0 when provided")
                if len(splits) > self.max_time_series_splits:
                    # Keep the most recent windows to reflect the latest market regime.
                    splits = splits[-self.max_time_series_splits :]
            return splits

        if self.task == "classification":
            splitter = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            return list(splitter.split(X, y))

        splitter = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        return list(splitter.split(X, y))

    def _score_fold(
        self,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute fold score with task-specific metrics and normalization rules."""
        if self.task == "classification":
            if self.scoring == "accuracy":
                return float(accuracy_score(y_test, y_pred))
            scorer = get_scorer(self.scoring)
            return float(scorer(model, X_test, y_test))

        if self.scoring == "neg_root_mean_squared_error":
            return -float(root_mean_squared_error(y_test, y_pred))
        if self.scoring == "neg_normalized_root_mean_squared_error":
            rmse = float(root_mean_squared_error(y_test, y_pred))
            scale = float(np.std(y_train))
            if scale <= 0.0:
                return -rmse
            return -rmse / scale
        scorer = get_scorer(self.scoring)
        return float(scorer(model, X_test, y_test))

    def _predict_with_model(
        self,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        X_test: object,
    ) -> npt.NDArray[Any]:
        """Use the sklearn wrapper predict path."""
        return _to_ndarray(model.predict(X_test))

    def _predict_with_cupy(
        self,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        X_test: object,
    ) -> npt.NDArray[Any]:
        """Use CuPy-backed input to keep CUDA inference on device when possible."""
        import cupy as cp

        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            pred_input = cp.asarray(X_test.to_numpy())
        else:
            pred_input = cp.asarray(X_test)
        y_pred = model.predict(pred_input)
        if isinstance(y_pred, cp.ndarray):
            return _to_ndarray(cp.asnumpy(y_pred))
        return _to_ndarray(y_pred)

    def _build_prediction_matrix(self, X_test: object) -> xgb.DMatrix:
        """Construct a DMatrix for booster-level prediction."""
        feature_names: list[str] | None
        matrix_data: npt.NDArray[Any]
        if isinstance(X_test, pd.DataFrame):
            feature_names = [str(col) for col in X_test.columns]
            matrix_data = _to_ndarray(X_test.to_numpy())
        elif isinstance(X_test, pd.Series):
            feature_names = [str(X_test.name)] if X_test.name is not None else None
            matrix_data = _to_ndarray(X_test.to_numpy().reshape(-1, 1))
        else:
            feature_names = None
            matrix_data = _to_ndarray(X_test)
        return xgb.DMatrix(matrix_data, feature_names=feature_names)

    def _decode_class_predictions(
        self,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        raw_pred: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Map booster outputs to class labels when classifier classes are available."""
        if raw_pred.ndim == 2:
            class_indices = np.argmax(raw_pred, axis=1).astype(int)
        else:
            class_indices = (raw_pred >= 0.5).astype(int)

        classes = getattr(model, "classes_", None)
        if classes is None:
            return _to_ndarray(class_indices)

        classes_array = _to_ndarray(classes)
        if classes_array.ndim != 1 or class_indices.size == 0:
            return _to_ndarray(class_indices)
        if int(np.max(class_indices)) >= classes_array.size:
            return _to_ndarray(class_indices)
        return _to_ndarray(classes_array[class_indices])

    def _predict_with_booster(
        self,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        X_test: object,
    ) -> npt.NDArray[Any]:
        """Use Booster.predict on an explicit DMatrix to avoid wrapper/device issues."""
        matrix = self._build_prediction_matrix(X_test)
        raw_pred = _to_ndarray(model.get_booster().predict(matrix))
        if self.task == "regression":
            return raw_pred
        return self._decode_class_predictions(model, raw_pred)

    def _predict(
        self,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        X_test: pd.DataFrame,
    ) -> npt.NDArray[Any]:
        """Run prediction using the clearest viable path for the current device.

        CUDA evaluation prefers GPU-backed inference first, then booster-level
        prediction, and finally the sklearn wrapper path. CPU evaluation uses
        the wrapper path directly.
        """
        if self.device != "cuda":
            return self._predict_with_model(model, X_test)

        for strategy in (self._predict_with_cupy, self._predict_with_booster):
            try:
                return strategy(model, X_test)
            except Exception:
                continue
        return self._predict_with_model(model, X_test)

    def __call__(self, selected_features: List[str]) -> float:
        """Evaluate one feature subset and return mean CV score.

        This method:
        - selects and encodes features/target as needed,
        - builds fold splits,
        - preprocesses train/test fold frames,
        - trains XGBoost and scores each fold,
        - returns the mean fold score.
        """
        if not selected_features:
            return 0.0

        label_encoder = LabelEncoder()
        X_selected = self.X.loc[:, selected_features].copy().convert_dtypes()
        y_values = self.y.to_numpy(copy=True)
        if self.task == "classification":
            y_values = label_encoder.fit_transform(y_values)

        for col in X_selected.columns:
            if str(X_selected[col].dtype) == "string":
                X_selected[col] = label_encoder.fit_transform(X_selected[col])

        splits = self._iter_splits(X_selected, y_values)
        if not splits:
            raise ValueError("No valid train/test splits were generated for evaluation")

        scores: list[float] = []
        for train_idx, test_idx in splits:
            X_train = X_selected.iloc[train_idx]
            X_test = X_selected.iloc[test_idx]
            y_train = y_values[train_idx]
            y_test = y_values[test_idx]

            X_train_new, X_test_new = preprocess_datasets(X_train, X_test, None)  # type: ignore[misc]
            model = self._build_estimator()
            model.fit(X_train_new, y_train)
            y_pred = self._predict(model, X_test_new)
            score = self._score_fold(model, X_test_new, y_train, y_test, y_pred)
            scores.append(score)

        return float(np.mean(scores))


def build_xgbrf_evaluator(
    X: pd.DataFrame,
    y: pd.Series,
    task: Literal["classification", "regression"],
    scoring: str,
    cv_folds: int = DEFAULT_GA_CONFIG.evaluator.cv_folds,
    is_time_series: bool = False,
    time_series_train_window: Optional[
        int
    ] = DEFAULT_GA_CONFIG.evaluator.time_series_train_window,
    time_series_test_window: Optional[
        int
    ] = DEFAULT_GA_CONFIG.evaluator.time_series_test_window,
    time_series_step_window: Optional[
        int
    ] = DEFAULT_GA_CONFIG.evaluator.time_series_step_window,
    time_series_gap: int = DEFAULT_GA_CONFIG.evaluator.time_series_gap,
    max_time_series_splits: Optional[
        int
    ] = DEFAULT_GA_CONFIG.evaluator.max_time_series_splits,
    n_estimators: Optional[int] = None,
    random_state: Optional[int] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> Callable[[List[str]], float]:
    """Factory for the GA evaluator with resolved configuration defaults.

    Device selection is resolved once here. ``device='auto'`` probes runtime
    CUDA availability and chooses CPU when unavailable to avoid crashing.
    """
    resolved_config = config or DEFAULT_GA_CONFIG
    model_config = resolved_config.xgbrf
    resolved_device = _resolve_xgboost_device(model_config.device)
    return _XGBRFEvaluator(
        X=X,
        y=y,
        task=task,
        scoring=scoring,
        cv_folds=cv_folds,
        is_time_series=is_time_series,
        time_series_train_window=time_series_train_window,
        time_series_test_window=time_series_test_window,
        time_series_step_window=time_series_step_window,
        time_series_gap=time_series_gap,
        max_time_series_splits=max_time_series_splits,
        n_estimators=(
            n_estimators if n_estimators is not None else model_config.n_estimators
        ),
        random_state=(
            random_state if random_state is not None else model_config.random_state
        ),
        verbosity=model_config.verbosity,
        device=resolved_device,
    )
