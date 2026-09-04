from typing import Any

import numpy as np
import pandas as pd


def compute_economic_loss(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    threshold: float,
    cost_fp: float,
    claim_amounts: np.ndarray[Any, Any] | None = None,
    cost_fn_fixed: float | None = None,
) -> dict[str, float]:
    """
    Calculate total financial loss in XAF for a specific decision threshold.
    Supports either a fixed FN cost or dynamic FN cost based on actual claim amounts.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    preds = (y_prob >= threshold).astype(int)

    fp_mask = (preds == 1) & (y_true == 0)
    fn_mask = (preds == 0) & (y_true == 1)

    fp_count = int(np.sum(fp_mask))
    fn_count = int(np.sum(fn_mask))

    total_fp_cost = fp_count * cost_fp

    if claim_amounts is not None:
        claim_amounts = np.asarray(claim_amounts)
        total_fn_cost = float(np.sum(claim_amounts[fn_mask]))
    elif cost_fn_fixed is not None:
        total_fn_cost = fn_count * cost_fn_fixed
    else:
        raise ValueError("Either claim_amounts or cost_fn_fixed must be provided.")

    total_cost = total_fp_cost + total_fn_cost

    return {
        "threshold": float(threshold),
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "total_fp_cost": total_fp_cost,
        "total_fn_cost": total_fn_cost,
        "total_cost": total_cost,
    }


def find_optimal_threshold(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    cost_fp: float,
    claim_amounts: np.ndarray[Any, Any] | None = None,
    cost_fn_fixed: float | None = None,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """
    Evaluate financial loss across a grid of thresholds to locate the cost-minimizing operating point.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = []

    for thresh in thresholds:
        loss_dict = compute_economic_loss(
            y_true=y_true,
            y_prob=y_prob,
            threshold=thresh,
            cost_fp=cost_fp,
            claim_amounts=claim_amounts,
            cost_fn_fixed=cost_fn_fixed,
        )
        results.append(loss_dict)

    df_results = pd.DataFrame(results)
    return df_results.sort_values(by="total_cost", ascending=True).reset_index(
        drop=True
    )
