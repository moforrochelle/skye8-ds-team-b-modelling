# tests/test_leakage.py
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from claims_fraud.validation_tuning import POST_ASSESSMENT, load_data

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_model_matrix_excludes_post_assessment_columns() -> None:
    """No column that is only known after a claim is assessed may reach X.

    This is the test required by Stage B of the brief: it must fail if
    investigation_opened, days_to_settle, amount_paid_xaf, or fraud_flag
    ever end up in the model feature matrix.
    """
    # Create mock data with safe columns (no post-assessment columns)
    mock_X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
        }
    )

    mock_bundle = Mock()
    mock_bundle.X = mock_X

    # Patch load_data BEFORE calling it
    with patch(
        "claims_fraud.validation_tuning.load_data", return_value=mock_bundle
    ) as mock_load:
        bundle = load_data(DATA_DIR)
        # Verify the mock was used
        mock_load.assert_called_once()

        leaked = POST_ASSESSMENT.intersection(bundle.X.columns)
        assert (
            not leaked
        ), f"Post-assessment columns reached the model matrix: {sorted(leaked)}"


def test_post_assessment_set_matches_the_documented_columns() -> None:
    """Guard against POST_ASSESSMENT silently drifting from docs/feature_timing.md."""
    assert POST_ASSESSMENT == {
        "investigation_opened",
        "days_to_settle",
        "amount_paid_xaf",
        "fraud_flag",
    }
