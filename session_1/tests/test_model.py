# tests/test_model.py
import pytest
from unittest.mock import MagicMock
from src.model import RideDurationModel


# ── Basic test ─────────────────────────────────────
def test_predict_returns_float():
    model = RideDurationModel()
    model._model = MagicMock()
    model._model.predict.return_value = [23.5]
    result = model.predict([5.0, 1])
    assert isinstance(result, float)
    assert result == 23.5


# ── Parameterized tests ────────────────────────────
@pytest.mark.parametrize(
    "distance,pax,expected",
    [
        (1.0, 1, 5.2),
        (10.0, 2, 24.8),
        (0.5, 4, 3.1),
    ],
)
def test_predict_multiple_inputs(distance, pax, expected):
    model = RideDurationModel()
    model._model = MagicMock()
    model._model.predict.return_value = [expected]
    assert model.predict([distance, pax]) == expected


# ── Fixture for shared setup ───────────────────────
@pytest.fixture
def mock_model():
    m = RideDurationModel()
    m._model = MagicMock()
    m._model.predict.return_value = [15.0]
    return m


def test_threshold_clipping(mock_model):
    mock_model.threshold = 10.0
    result = mock_model.predict([100.0, 1])
    assert result == 10.0  # clipped to threshold
