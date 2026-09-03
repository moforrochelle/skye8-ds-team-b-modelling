from typing import Any

import numpy as np


def precision_at_k(
    y_true: np.ndarray[Any, Any], y_prob: np.ndarray[Any, Any], k: int
) -> float:
    """
    Calculate precision at top k ranked predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    order = np.argsort(y_prob)[::-1]

    top_k = y_true[order][:k]

    return float(np.sum(top_k) / k)


def recall_at_k(
    y_true: np.ndarray[Any, Any], y_prob: np.ndarray[Any, Any], k: int
) -> float:
    """
    Calculate recall at top k ranked predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0

    order = np.argsort(y_prob)[::-1]
    top_k = y_true[order][:k]

    return float(np.sum(top_k) / total_positives)


def fraud_value_captured_at_k(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    claim_amounts: np.ndarray[Any, Any],
    k: int,
) -> tuple[float, float]:
    """
    Calculate the total XAF value of fraud captured in top k predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    claim_amounts = np.asarray(claim_amounts)

    order = np.argsort(y_prob)[::-1]
    top_k_indices = order[:k]

    # Mask for true fraudulent claims in top k
    fraud_in_top_k = y_true[top_k_indices] == 1
    captured_value = float(np.sum(claim_amounts[top_k_indices][fraud_in_top_k]))

    total_fraud_value = float(np.sum(claim_amounts[y_true == 1]))
    prop_captured = captured_value / total_fraud_value if total_fraud_value > 0 else 0.0

    return captured_value, prop_captured
