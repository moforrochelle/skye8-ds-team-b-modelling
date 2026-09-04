from typing import Any

import numpy as np
import pandas as pd


def get_top_k_claims(
    df: pd.DataFrame, y_probs: np.ndarray[Any, Any], k: int
) -> pd.DataFrame:
    """
    Extract top k highest risk claims based on predicted probabilities.
    """
    analysis_df = df.copy()
    analysis_df["predicted_prob"] = y_probs
    return analysis_df.nlargest(k, "predicted_prob")


def calculate_captured_value_at_k(
    df_top_k: pd.DataFrame,
    target_col: str = "fraud_flag",
    amount_col: str = "claim_amount_xaf",
) -> dict[str, float]:
    """
    Calculate precision and total financial value captured in XAF for top k claims.
    """
    analysis_df = df_top_k.copy()
    if len(analysis_df) == 0:
        return {
            "precision_at_k": 0.0,
            "captured_value_at_k": 0.0,
            "total_claims_flagged": 0,
        }

    precision_at_k = float(analysis_df[target_col].sum() / len(analysis_df))
    captured_value_at_k = float(
        analysis_df[analysis_df[target_col] == 1][amount_col].sum()
    )
    total_claims_flagged = int(analysis_df.shape[0])

    return {
        "precision_at_k": precision_at_k,
        "captured_value_at_k": captured_value_at_k,
        "total_claims_flagged": total_claims_flagged,
    }


def extract_case_study_samples(
    df: pd.DataFrame,
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    k: int,
    n_samples: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract 20 false positive and 20 false negative case studies for qualitative review.
    """
    analysis_df = df.copy()
    analysis_df["y_true"] = y_true
    analysis_df["y_prob"] = y_prob

    df_top_k = get_top_k_claims(analysis_df, y_prob, k)
    df_false_positive = df_top_k[df_top_k["y_true"] == 0]

    outside_top_k = analysis_df.drop(index=df_top_k.index)
    df_false_negative = outside_top_k[outside_top_k["y_true"] == 1]

    fp_sample = df_false_positive.sample(
        n=min(n_samples, len(df_false_positive)), random_state=random_state
    )
    fn_sample = df_false_negative.sample(
        n=min(n_samples, len(df_false_negative)), random_state=random_state
    )

    return fp_sample, fn_sample


def segment_error_rates(
    df: pd.DataFrame,
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    group_col: str,
) -> pd.DataFrame:
    """
    Break down false positive and false negative error rates across categorical slices.
    """
    analysis_df = df.copy()
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = y_pred

    analysis_df["is_fp"] = (analysis_df["y_pred"] == 1) & (analysis_df["y_true"] == 0)
    analysis_df["is_fn"] = (analysis_df["y_pred"] == 0) & (analysis_df["y_true"] == 1)

    summary = (
        analysis_df.groupby(group_col)
        .agg(
            total_claims=("y_true", "count"),
            false_positives=("is_fp", "sum"),
            false_negatives=("is_fn", "sum"),
        )
        .reset_index()
    )

    summary["error_rate"] = (
        summary["false_positives"] + summary["false_negatives"]
    ) / summary["total_claims"]

    return summary.sort_values(by="error_rate", ascending=False)
