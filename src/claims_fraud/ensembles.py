from typing import Any

import numpy as np
import pandas as pd


class WeightedEnsemble:
    """
    Combines predictions of multiple models by assigning weights to each model's predictions.
    Validates model counts and normalizes weights to sum to 1.0.
    """

    def __init__(self, models: list[Any], weights: list[float] | None = None) -> None:
        if not models:
            raise ValueError("At least one model must be provided to WeightedEnsemble.")

        self.models = models

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) does not match "
                    f"number of models ({len(models)})."
                )

            weight_sum = float(np.sum(weights))
            if np.isclose(weight_sum, 0.0):
                raise ValueError("Sum of ensemble weights cannot be zero.")

            # Normalize weights to ensure valid probability combination
            self.weights = [float(w) / weight_sum for w in weights]

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        pred_probs = []

        for model, weight in zip(self.models, self.weights):
            probs = model.predict_proba(X)[:, 1]
            pred_probs.append(probs * weight)

        np_pred_probs = np.array(pred_probs)
        weighted_preds = np.sum(np_pred_probs, axis=0)

        return np.column_stack([1 - weighted_preds, weighted_preds])

    def predict(
        self, X: np.ndarray | pd.DataFrame, threshold: float = 0.5
    ) -> np.ndarray:
        pred_probs = self.predict_proba(X)
        weighted_preds = pred_probs[:, 1]

        return (weighted_preds >= threshold).astype(int)
