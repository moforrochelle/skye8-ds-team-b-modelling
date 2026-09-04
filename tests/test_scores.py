from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from claims_fraud.models import save_model
from claims_fraud.scores import main


class DummyClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """
    Lightweight dummy classifier for testing batch scoring inference behavior.
    """

    def fit(self, X: Any, y: Any = None) -> DummyClassifier:
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        # Always return a fixed probability of 0.7 for the positive fraud class
        probs = np.zeros((n_samples, 2))
        probs[:, 1] = 0.7
        return probs


@pytest.fixture
def mock_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Fixture to create temporary input CSV files and saved model for testing.
    """
    input_csv = tmp_path / "input_claims.csv"
    model_file = tmp_path / "dummy_model.joblib"
    output_csv = tmp_path / "output_scored.csv"

    # Create mock input data including post-assessment leakage columns to test dropping
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-029155", "CLM-029159"],
            "policy_id": ["PO910902", "PO910908"],
            "adjuster_id": ["AD-013", "AD-015"],
            "garage_id": ["GR-061", "GR-065"],
            "feature_col_1": [10.5, 20.1],
            "investigation_opened": [0, 1],  # Leakage column
            "amount_paid_xaf": [25000, 75000],  # Leakage column
        }
    )
    df.to_csv(input_csv, index=False)

    # Train and serialize the dummy model
    model = DummyClassifier()
    save_model(model, model_file)

    return input_csv, model_file, output_csv


def test_score_execution_success(
    mock_environment: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test a successful end-to-end run of the batch scoring script.
    """
    input_csv, model_file, output_csv = mock_environment

    # Simulate command-line terminal arguments
    test_args = [
        "score.py",
        "--input",
        str(input_csv),
        "--model",
        str(model_file),
        "--output",
        str(output_csv),
        "--threshold",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Execute main scoring script
    main()

    # Assertions
    assert output_csv.exists()
    result_df = pd.read_csv(output_csv)

    # Verify output columns match requirements
    assert list(result_df.columns) == [
        "claim_id",
        "policy_id",
        "garage_id",
        "adjuster_id",
        "fraud_prob",
        "fraud_flag",
    ]
    assert len(result_df) == 2
    assert result_df.loc[0, "fraud_prob"] == 0.7
    assert result_df.loc[0, "fraud_flag"] == 1  # 0.7 >= 0.5 threshold


def test_score_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that the script exits gracefully with error code 1 if the input file is missing.
    """
    non_existent_input = tmp_path / "non_existent.csv"
    model_file = tmp_path / "model.joblib"
    output_csv = tmp_path / "output.csv"

    # Save a valid dummy model so it fails on input file check rather than model check
    joblib.dump(DummyClassifier(), model_file)

    test_args = [
        "score.py",
        "--input",
        str(non_existent_input),
        "--model",
        str(model_file),
        "--output",
        str(output_csv),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
