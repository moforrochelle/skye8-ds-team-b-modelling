from typing import Any

import numpy as np


def precision_at_k(
    y_true: np.ndarray[Any, Any], y_prob: np.ndarray[Any, Any], k: int
) -> float:
    order = np.argsort(y_prob)[::-1]

    top_k = y_true[order][:k]

    return float(np.sum(top_k) / k)
