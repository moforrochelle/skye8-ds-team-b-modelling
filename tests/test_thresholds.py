from typing import Any

import numpy as np
import pandas as pd
import pytest

from claims_fraud.thresholds import compute_economic_loss, find_optimal_threshold


@pytest.fixture
def sample_data() -> dict[str, Any]:
    return {
        "y_true": np.array([0, 1, 0, 1]),
        "y_prob": np.array([0.1, 0.9, 0.8, 0.4]),
        "claim_amounts": np.array([10000, 50000, 20000, 80000]),
        "cost_fp": 25000,
    }


def test_compute_economic_loss(sample_data: dict[str, Any]) -> None:
    data = sample_data
    loss = compute_economic_loss(**data, threshold=0.5)

    assert loss["false_positives"] == 1
    assert loss["false_negatives"] == 1
    assert loss["total_cost"] == 105000


def test_compute_economic_loss2(sample_data: dict[str, Any]) -> None:
    data = sample_data.copy()
    data.pop("claim_amounts")
    loss = compute_economic_loss(**data, threshold=0.5, cost_fn_fixed=30000)

    assert loss["total_fn_cost"] == 30000
    assert loss["total_cost"] == 55000


def test_compute_economic_loss3(sample_data: dict[str, Any]) -> None:
    data = sample_data.copy()
    data.pop("claim_amounts")

    with pytest.raises(ValueError):
        compute_economic_loss(**data, threshold=0.5)


def test_compute_economic_loss4(sample_data: dict[str, Any]) -> None:
    data = sample_data
    loss = compute_economic_loss(**data, threshold=0.0)

    assert loss["false_positives"] == 2
    assert loss["false_negatives"] == 0

    loss = compute_economic_loss(**data, threshold=1.0)

    assert loss["false_positives"] == 0
    assert loss["false_negatives"] == 2


def test_find_optimal_threshold(sample_data: dict[str, Any]) -> None:
    data = sample_data
    thresholds = find_optimal_threshold(**data, n_thresholds=5)

    assert isinstance(thresholds, pd.DataFrame)
    assert thresholds.iloc[0]["total_cost"] < thresholds.iloc[-1]["total_cost"]
