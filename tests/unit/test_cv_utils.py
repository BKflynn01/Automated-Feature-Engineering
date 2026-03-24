import numpy as np

from cv_utils import iter_sliding_window_splits, resolve_sliding_window_params


def test_resolve_sliding_window_params_infers_balanced_defaults():
    train_window, test_window, step_window = resolve_sliding_window_params(
        n_samples=100,
        n_splits=4,
    )

    assert train_window == 20
    assert test_window == 20
    assert step_window == 20


def test_iter_sliding_window_splits_returns_expected_windows():
    splits = list(
        iter_sliding_window_splits(
            n_samples=30,
            train_window=10,
            test_window=5,
            step_window=5,
            gap=0,
        )
    )

    assert len(splits) == 4
    assert np.array_equal(splits[0][0], np.arange(0, 10))
    assert np.array_equal(splits[0][1], np.arange(10, 15))
    assert np.array_equal(splits[-1][0], np.arange(15, 25))
    assert np.array_equal(splits[-1][1], np.arange(25, 30))


def test_iter_sliding_window_splits_supports_gap():
    splits = list(
        iter_sliding_window_splits(
            n_samples=25,
            train_window=8,
            test_window=4,
            step_window=4,
            gap=2,
        )
    )

    assert len(splits) == 3
    first_train, first_test = splits[0]
    assert np.array_equal(first_train, np.arange(0, 8))
    assert np.array_equal(first_test, np.arange(10, 14))
