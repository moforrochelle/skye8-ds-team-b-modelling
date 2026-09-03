"""Reusable model-evaluation utilities for the claims-fraud project."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

Scheme = Literal["random", "grouped"]


def _score_split(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Fit on one training split and return metrics for its test split."""
    model = clone(estimator)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision_at_0_5": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_at_0_5": float(recall_score(y_test, predictions)),
        "prevalence": float(y_test.mean()),
    }


def evaluate_cross_validation(
    estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    scheme: Scheme,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate one classifier with the approved random or grouped CV scheme."""
    if scheme == "random":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(x, y)
    elif scheme == "grouped":
        if groups is None:
            raise ValueError("groups are required for grouped cross-validation.")

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(x, y, groups)
    else:
        raise ValueError(f"Unknown validation scheme: {scheme}")

    rows: list[dict[str, float | int | str]] = []

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        metrics = _score_split(
            estimator=estimator,
            x_train=x.iloc[train_index],
            y_train=y.iloc[train_index],
            x_test=x.iloc[test_index],
            y_test=y.iloc[test_index],
        )
        rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_temporal_holdout(
    estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    reported_at: pd.Series,
    holdout_start: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate a classifier on claims reported on or after a time cutoff."""
    if not x.index.equals(y.index) or not x.index.equals(reported_at.index):
        raise ValueError("x, y, and reported_at must have matching indexes.")

    if not pd.api.types.is_datetime64_any_dtype(reported_at):
        raise TypeError("reported_at must be parsed datetime data.")

    if reported_at.isna().any():
        raise ValueError("reported_at must not contain missing dates.")

    cutoff = pd.Timestamp(holdout_start)
    train_mask = reported_at < cutoff
    test_mask = reported_at >= cutoff

    if not train_mask.any() or not test_mask.any():
        raise ValueError("The temporal split must contain training and test claims.")

    metrics = _score_split(
        estimator=estimator,
        x_train=x.loc[train_mask],
        y_train=y.loc[train_mask],
        x_test=x.loc[test_mask],
        y_test=y.loc[test_mask],
    )

    return pd.DataFrame(
        [
            {
                "scheme": "temporal",
                "fold": 1,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **metrics,
            }
        ]
    )
