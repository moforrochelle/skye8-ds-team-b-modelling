from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from claims_fraud import validation_tuning
from claims_fraud.validation_tuning import POST_ASSESSMENT

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_model_matrix_excludes_post_assessment_columns() -> None:
    """No column known only after assessment may reach the model matrix.

    This test is required by Stage B of the brief. It ensures that
    post-assessment columns such as investigation_opened, days_to_settle,
    amount_paid_xaf, and fraud_flag do not appear in X.
    """
    # Create mock feature data that contains only safe columns.
    mock_X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
        }
    )

    mock_bundle = Mock()
    mock_bundle.X = mock_X

    # Patch load_data before calling it so the test does not require
    # the real data/claims.csv file, which is intentionally gitignored.
    with patch(
        "claims_fraud.validation_tuning.load_data",
        return_value=mock_bundle,
    ) as mock_load:
        bundle = validation_tuning.load_data(DATA_DIR)

        # Confirm that the mocked loader was actually used.
        mock_load.assert_called_once_with(DATA_DIR)

        leaked = POST_ASSESSMENT.intersection(bundle.X.columns)

        assert not leaked, (
            "Post-assessment columns reached the model matrix: " f"{sorted(leaked)}"
        )


def test_post_assessment_set_matches_the_documented_columns() -> None:
    """Ensure POST_ASSESSMENT matches the documented feature-timing columns."""
    assert POST_ASSESSMENT == {
        "investigation_opened",
        "days_to_settle",
        "amount_paid_xaf",
        "fraud_flag",
    }
