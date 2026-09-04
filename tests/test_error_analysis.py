import numpy as np
import pandas as pd
import pytest

from claims_fraud.error_analysis import (
    calculate_captured_value_at_k,
    extract_case_study_samples,
    get_top_k_claims,
    segment_error_rates,
)


@pytest.fixture
def sample_claims_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": [f"C{i}" for i in range(1, 11)],
            "region": [
                "Centre",
                "Littoral",
                "Centre",
                "Littoral",
                "West",
                "Centre",
                "Littoral",
                "West",
                "Centre",
                "Littoral",
            ],
            "fraud_flag": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "claim_amount_xaf": [
                100000,
                50000,
                200000,
                150000,
                80000,
                300000,
                120000,
                90000,
                250000,
                60000,
            ],
        }
    )


def test_get_top_k_claims(sample_claims_df: pd.DataFrame) -> None:
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.95, 0.4, 0.05, 0.85, 0.15])
    top_3 = get_top_k_claims(sample_claims_df, y_probs, k=3)

    assert len(top_3) == 3
    assert list(top_3["claim_id"]) == ["C6", "C1", "C9"]


def test_calculate_captured_value_at_k(sample_claims_df: pd.DataFrame) -> None:
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.95, 0.4, 0.05, 0.85, 0.15])
    top_k_df = get_top_k_claims(sample_claims_df, y_probs, k=3)

    metrics = calculate_captured_value_at_k(
        top_k_df, target_col="fraud_flag", amount_col="claim_amount_xaf"
    )

    assert metrics["precision_at_k"] == 1.0
    assert metrics["captured_value_at_k"] == 650000.0
    assert metrics["total_claims_flagged"] == 3


def test_extract_case_study_samples(sample_claims_df: pd.DataFrame) -> None:
    y_true = sample_claims_df["fraud_flag"].to_numpy()
    y_probs = np.array([0.1, 0.95, 0.8, 0.85, 0.2, 0.15, 0.3, 0.1, 0.9, 0.05])

    fp_df, fn_df = extract_case_study_samples(
        sample_claims_df, y_true, y_probs, threshold=0.4, n_samples=2, random_state=42
    )

    assert len(fp_df) <= 2
    assert len(fn_df) <= 2
    assert (fp_df["y_true"] == 0).all()
    assert (fn_df["y_true"] == 1).all()


def test_segment_error_rates(sample_claims_df: pd.DataFrame) -> None:
    y_true = sample_claims_df["fraud_flag"].to_numpy()
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])

    summary = segment_error_rates(sample_claims_df, y_true, y_pred, group_col="region")

    assert "region" in summary.columns
    assert "error_rate" in summary.columns
    assert len(summary) == 3
