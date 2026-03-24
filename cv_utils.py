from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


def resolve_sliding_window_params(
    n_samples: int,
    n_splits: int,
    train_window: Optional[int] = None,
    test_window: Optional[int] = None,
    step_window: Optional[int] = None,
    gap: int = 0,
) -> Tuple[int, int, int]:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_splits <= 0:
        raise ValueError("n_splits must be > 0")
    if gap < 0:
        raise ValueError("gap must be >= 0")

    resolved_test = (
        test_window if test_window is not None else max(1, n_samples // (n_splits + 1))
    )
    resolved_train = (
        train_window
        if train_window is not None
        else max(1, n_samples - (resolved_test * n_splits) - gap)
    )
    resolved_step = step_window if step_window is not None else resolved_test

    if resolved_train <= 0:
        raise ValueError("train_window must be > 0")
    if resolved_test <= 0:
        raise ValueError("test_window must be > 0")
    if resolved_step <= 0:
        raise ValueError("step_window must be > 0")

    if resolved_train + gap + resolved_test > n_samples:
        raise ValueError(
            "Invalid time-series windows: train_window + gap + test_window exceeds sample count"
        )

    return resolved_train, resolved_test, resolved_step


def iter_sliding_window_splits(
    n_samples: int,
    train_window: int,
    test_window: int,
    step_window: int,
    gap: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if train_window <= 0:
        raise ValueError("train_window must be > 0")
    if test_window <= 0:
        raise ValueError("test_window must be > 0")
    if step_window <= 0:
        raise ValueError("step_window must be > 0")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if train_window + gap + test_window > n_samples:
        raise ValueError(
            "Invalid time-series windows: train_window + gap + test_window exceeds sample count"
        )

    start = 0
    max_train_start = n_samples - train_window - gap - test_window
    while start <= max_train_start:
        train_start = start
        train_end = train_start + train_window
        test_start = train_end + gap
        test_end = test_start + test_window

        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        yield train_idx, test_idx

        start += step_window
