from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claims_fraud.models import (
    PrevalenceBaseline,
    get_baseline_models,
    get_gradient_boosters,
    load_model,
    save_model,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng()
    return pd.DataFrame(
        {
            "X1": rng.integers(low=1, high=100, size=10),
            "X2": rng.integers(low=1, high=100, size=10),
            "y": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        }
    )


@pytest.fixture  # type: ignore[untyped-decorator]
def sample_prevalence(sample_data: pd.DataFrame) -> PrevalenceBaseline:
    df = sample_data

    X = df[["X1", "X2"]]
    y = df["y"]

    prevalence = PrevalenceBaseline()
    return prevalence.fit(X, y)


def test_get_baseline_model() -> None:
    baseline_model = get_baseline_models()

    assert "logistic_regression" in baseline_model
    assert "decision_tree" in baseline_model
    assert "prevalence" in baseline_model


def test_get_gradient_boosters() -> None:
    gradient_boosters = get_gradient_boosters()

    assert "random_forest" in gradient_boosters
    assert "lightgbm" in gradient_boosters
    assert "xgboost" in gradient_boosters


def test_prevalence_baseline(
    sample_prevalence: PrevalenceBaseline, sample_data: pd.DataFrame
) -> None:
    df = sample_data
    X = df[["X1", "X2"]]
    prevalence = sample_prevalence

    y_pred = prevalence.predict_proba(X)

    assert y_pred.shape == (10, 2)
    assert y_pred[0, 1] == 0.2


def test_save_load_model(
    tmp_path: Path, sample_prevalence: PrevalenceBaseline, sample_data: pd.DataFrame
) -> None:
    df = sample_data
    X = df[["X1", "X2"]]

    path = tmp_path / "model.joblib"

    model = sample_prevalence

    save_model(model, path)

    assert path.exists() == True

    loaded_model = load_model(path)

    origin_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)

    assert np.array_equal(loaded_preds, origin_preds)
