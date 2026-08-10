"""DVC stage ``evaluate`` — score the trained model on the validation split.

Loads ``models/rf_model.pkl`` and ``data/processed/val.parquet``, computes the
regression metrics, and writes them to ``metrics/scores.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.train import FEATURES, TARGET, evaluate


def main() -> None:
    df = pd.read_parquet("data/processed/val.parquet")
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    model = joblib.load("models/rf_model.pkl")
    metrics = evaluate(model, X, y)

    out_dir = Path("metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"evaluate: {metrics} → metrics/scores.json")


if __name__ == "__main__":
    main()
