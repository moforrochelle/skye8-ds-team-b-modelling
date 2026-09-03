from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class PrevalenceBaseline(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """
    Dummy baseline classifier that predicts constant class probabilities based on training prevalence
    """

    def __init__(self) -> None:
        self.dummy = DummyClassifier(strategy="prior")

    def fit(self, X: Any, y: Any) -> PrevalenceBaseline:
        self.dummy.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.dummy.predict_proba(X)

    def predict(self, X: Any) -> Any:
        return self.dummy.predict(X)


def get_baseline_models() -> dict[str, BaseEstimator]:
    """
    Return initialized baseline estimators
    """
    return {
        "prevalence": PrevalenceBaseline(),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            random_state=42,
        ),
    }


def get_gradient_boosters(scale_pos_weight: float = 32.0) -> dict[str, BaseEstimator]:
    """
    Return initialized gradient boosting and ensemble models
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        ),
    }


def load_model(filepath: str | Path) -> BaseEstimator:
    """
    Load a saved model
    """
    return joblib.load(filepath)


def save_model(model: BaseEstimator, filepath: str | Path) -> None:
    """
    Save a fitted model
    """
    joblib.dump(model, filepath)
