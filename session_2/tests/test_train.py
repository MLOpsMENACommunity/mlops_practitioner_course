# tests/test_train.py
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.train import evaluate, generate_data, split_data, train_model


# ── Data generation ────────────────────────────────
def test_generate_data_shapes():
    X, y = generate_data(n_samples=50, seed=0)
    assert X.shape == (50, 2)
    assert y.shape == (50,)


def test_generate_data_is_reproducible():
    X1, y1 = generate_data(n_samples=100, seed=7)
    X2, y2 = generate_data(n_samples=100, seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


# ── Split ──────────────────────────────────────────
def test_split_data_sizes():
    X, y = generate_data(n_samples=100, seed=1)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=1)
    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(y_train) == 80
    assert len(y_val) == 20


# ── Training ───────────────────────────────────────
@pytest.fixture
def trained():
    X, y = generate_data(n_samples=500, seed=42)
    X_train, X_val, y_train, y_val = split_data(X, y, seed=42)
    model = train_model(X_train, y_train)
    return model, X_val, y_val


def test_train_model_returns_random_forest(trained):
    model, _, _ = trained
    assert isinstance(model, RandomForestRegressor)


def test_train_model_respects_custom_params():
    X, y = generate_data(n_samples=200, seed=3)
    model = train_model(X, y, {"n_estimators": 10, "max_depth": 3, "random_state": 3})
    assert model.n_estimators == 10
    assert model.max_depth == 3


# ── Evaluation ─────────────────────────────────────
def test_evaluate_returns_expected_keys(trained):
    model, X_val, y_val = trained
    metrics = evaluate(model, X_val, y_val)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_model_learns_the_signal(trained):
    # The synthetic relation is near-linear, so a fit model should score well.
    model, X_val, y_val = trained
    metrics = evaluate(model, X_val, y_val)
    assert metrics["r2"] > 0.9
