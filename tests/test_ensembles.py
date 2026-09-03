import numpy as np
import pytest

from claims_fraud.ensembles import WeightedEnsemble


class MockModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([[0.8, 0.2], [0.4, 0.6]])


@pytest.fixture
def mock_model() -> MockModel:
    return MockModel()


def test_equal_default_weights(mock_model: MockModel) -> None:
    ensemble = WeightedEnsemble(models=[mock_model, mock_model])
    assert ensemble.weights == [0.5, 0.5]


def test_custom_weights(mock_model: MockModel) -> None:
    ensemble = WeightedEnsemble(models=[mock_model, mock_model], weights=[0.7, 0.3])
    assert ensemble.weights == [0.7, 0.3]


def test_predict_proba_shape(mock_model: MockModel) -> None:
    ensemble = WeightedEnsemble(models=[mock_model, mock_model, mock_model])

    X = np.array([[1, 2], [3, 4]])
    pred_probs = ensemble.predict_proba(X)
    preds = ensemble.predict(X)

    assert pred_probs.shape == (2, 2)
    assert preds.shape == (2,)
    assert np.all((preds == 0) | (preds == 1))


def test_custom_threshold(mock_model: MockModel) -> None:
    ensemble = WeightedEnsemble(models=[mock_model, mock_model, mock_model])

    X = np.array([[1, 2], [3, 4]])
    preds1 = ensemble.predict(X, threshold=0.2)
    preds2 = ensemble.predict(X, threshold=0.8)

    assert np.sum(preds1) >= np.sum(preds2)
