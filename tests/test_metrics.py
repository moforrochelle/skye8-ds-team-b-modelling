import numpy as np
import pandas as pd
import pytest

from claims_fraud.metrics import precision_at_k


def test_precision_at_k_perfect() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])

    assert precision_at_k(y_true, y_prob, k=2) == 1.0


def test_precision_at_k_zero() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])

    assert precision_at_k(y_true, y_prob, k=2) == 0.0


def test_precision_at_k_partial() -> None:
    y_true = np.array([1, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.1])

    # Top 3 probabilities: 0.9 (y=1), 0.8 (y=0), 0.7 (y=1) -> 2 true positives out of 3
    assert pytest.approx(precision_at_k(y_true, y_prob, k=3)) == 2 / 3


def test_precision_at_k_pandas_input() -> None:
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = pd.Series([0.8, 0.9, 0.1, 0.2])

    # Top 2 probabilities: index 1 (y=0), index 0 (y=1) -> 1 true positive out of 2
    assert precision_at_k(y_true.to_numpy(), y_prob.to_numpy(), k=2) == 0.5
