"""Training logic for the ride-duration model (session 2, MLflow example).

This module is intentionally free of any MLflow calls so it can be unit-tested
in isolation. The MLflow experiment tracking lives in ``mflow_example.py``,
which imports and orchestrates these functions.

The data is *synthetic*: it reproduces the same heuristic used by session 1's
``RideDurationModel`` (distance ÷ average speed + per-passenger overhead) plus a
little Gaussian noise, so the model can recover the underlying relation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config

#: Assumed average speed in km per minute (~30 km/h) — matches session 1.
AVG_SPEED_KM_PER_MIN = 0.5
#: Extra minutes added per passenger (boarding overhead) — matches session 1.
PASSENGER_OVERHEAD_MIN = 0.5

#: Default RandomForest hyperparameters logged to MLflow.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "random_state": 42,
}

#: Feature columns / target column used by the DVC pipeline (see prepare.py).
FEATURES = ["distance_km", "passengers"]
TARGET = "duration_min"


def generate_data(
    n_samples: int = 2_000,
    noise: float = 1.0,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic ``(X, y)`` ride-duration dataset.

    Each row of ``X`` is ``[distance_km, passengers]``; ``y`` is the trip
    duration in minutes derived from the session-1 heuristic plus noise.

    Args:
        n_samples: Number of rows to generate.
        noise: Standard deviation of the Gaussian noise added to durations.
        seed: RNG seed for reproducibility (``None`` for nondeterministic).

    Returns:
        ``(X, y)`` where ``X`` has shape ``(n_samples, 2)`` and ``y`` has
        shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.5, 30.0, size=n_samples)  # km
    passengers = rng.integers(1, 5, size=n_samples)  # 1..4

    duration = (
        distance / AVG_SPEED_KM_PER_MIN
        + passengers * PASSENGER_OVERHEAD_MIN
        + rng.normal(0.0, noise, size=n_samples)
    )

    X = np.column_stack([distance, passengers]).astype(float)
    y = duration.astype(float)
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(X, y)`` into train/validation partitions (thin wrapper)."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[dict[str, Any]] = None,
) -> RandomForestRegressor:
    """Fit and return a :class:`RandomForestRegressor` on the training data.

    Args:
        X_train: Training features, shape ``(n, 2)``.
        y_train: Training targets, shape ``(n,)``.
        params: RandomForest hyperparameters. Falls back to
            :data:`DEFAULT_PARAMS` when omitted.
    """
    model = RandomForestRegressor(**(params or DEFAULT_PARAMS))
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: RandomForestRegressor, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Score ``model`` on ``(X, y)`` and return regression metrics.

    Returns:
        A dict with ``rmse``, ``mae`` and ``r2`` (all floats).
    """
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    """DVC stage ``train`` — fit the model on the train split and save it.

    Reads ``data/processed/train.parquet`` and the ``model`` hyperparameters
    from ``config/config.yaml``, then writes ``models/rf_model.pkl``.
    """

    cfg = load_config()
    df = pd.read_parquet("data/processed/train.parquet")
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    model = train_model(X, y, cfg["model"])

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    print(
        "train: fitted RandomForest("
        + json.dumps(cfg["model"])
        + f") on {len(df)} rows → models/rf_model.pkl"
    )


if __name__ == "__main__":
    main()
