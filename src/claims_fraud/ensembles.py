from typing import Any

import numpy as np
import pandas as pd


class WeightedEnsemble:
    """
    Used to combine the predictions of multiple models by assigning weights to each model's predictions.
    The final prediction is based on the weighted average of the individual model predictions.
    """

    def __init__(self, models: list[Any], weights: list[float] | None = None) -> None:
        self.models = models
        self.weights = (
            weights if weights is not None else [1 / len(models)] * len(models)
        )

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:

        pred_probs = []

        for model, weight in zip(self.models, self.weights):
            pred_probs.append(model.predict_proba(X)[:, 1] * weight)

        np_pred_probs = np.array(pred_probs)

        weighted_preds = np.sum(np_pred_probs, axis=0)
        return np.array([1 - weighted_preds, weighted_preds]).T

    def predict(
        self, X: np.ndarray | pd.DataFrame, threshold: float = 0.5
    ) -> np.ndarray:
        pred_probs = self.predict_proba(X)
        weighted_preds = pred_probs[:, 1]

        return (weighted_preds >= threshold).astype(int)
