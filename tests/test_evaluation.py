import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from claims_fraud.evaluation import (
    evaluate_cross_validation,
    evaluate_temporal_holdout,
)


def make_test_data() -> tuple[pd.DataFrame, pd.Series]:
    features, target = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        weights=[0.8, 0.2],
        random_state=42,
    )
    return pd.DataFrame(features), pd.Series(target)


def test_random_cross_validation_returns_five_folds() -> None:
    x, y = make_test_data()
    model = LogisticRegression(max_iter=1_000)

    results = evaluate_cross_validation(model, x, y, scheme="random")

    assert len(results) == 5
    assert set(results["scheme"]) == {"random"}
    assert results["pr_auc"].between(0, 1).all()


def test_grouped_cross_validation_returns_five_folds() -> None:
    x, y = make_test_data()
    groups = pd.Series(np.repeat(np.arange(20), 10))
    model = LogisticRegression(max_iter=1_000)

    results = evaluate_cross_validation(
        model,
        x,
        y,
        scheme="grouped",
        groups=groups,
    )

    assert len(results) == 5
    assert set(results["scheme"]) == {"grouped"}
    assert results["pr_auc"].between(0, 1).all()


def test_temporal_holdout_uses_only_earlier_claims_for_training() -> None:
    x, y = make_test_data()
    reported_at = pd.Series(pd.date_range("2024-01-01", periods=len(x), freq="D"))
    model = LogisticRegression(max_iter=1_000)

    results = evaluate_temporal_holdout(
        estimator=model,
        x=x,
        y=y,
        reported_at=reported_at,
        holdout_start=pd.Timestamp("2024-05-30"),
    )

    row = results.iloc[0]
    assert len(results) == 1
    assert row["scheme"] == "temporal"
    assert row["n_train"] == 150
    assert row["n_test"] == 50
    assert 0 <= row["pr_auc"] <= 1
