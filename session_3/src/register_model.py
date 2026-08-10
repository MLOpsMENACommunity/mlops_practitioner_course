"""Train the ride-duration model and register it in the MLflow Model Registry.

``src/train.py`` is deliberately MLflow-free so its functions stay unit-testable;
this module is the tracking/registry orchestrator that imports them — the same
split session 2 uses for ``mlflow_example.py``.

The batch-scoring job (``src/batch_scoring.py``) loads the model by alias, so the
registry must hold a version tagged ``@production``. MLflow 3 removed the old
Production/Staging *stages*, so an alias is the supported mechanism.

Run (cwd = repo root, MLflow server on :5000):
    python src/register_model.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from src.config import load_config
from src.train import FEATURES, TARGET, evaluate, split_data, train_model

#: Registry name the batch-scoring job resolves (models:/<name>@production).
MODEL_NAME = "RideDurationModel"
#: Alias batch scoring loads. Replaces the removed "Production" stage.
PRODUCTION_ALIAS = "production"

#: Tracking server. Override with MLFLOW_TRACKING_URI to point at a remote host.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main() -> None:
    """Fit the model, log it to MLflow, and move ``@production`` onto it."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ride-duration")

    cfg = load_config()
    df = pd.read_parquet("data/processed/train.parquet")
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_train, X_val, y_train, y_val = split_data(
        X, y, test_size=cfg["data"]["test_size"], seed=cfg["data"]["seed"]
    )

    with mlflow.start_run() as run:
        model = train_model(X_train, y_train, cfg["model"])
        metrics = evaluate(model, X_val, y_val)

        mlflow.log_params(cfg["model"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train[:5],
            registered_model_name=MODEL_NAME,
        )
        print(f"register: run {run.info.run_id} metrics={metrics}")

    # The alias must point at the version we just logged, not merely the highest
    # one — a concurrent run could have registered a newer version meanwhile.
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = max(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version.version)
    print(
        f"register: {MODEL_NAME} v{version.version} "
        f"-> alias @{PRODUCTION_ALIAS} (models:/{MODEL_NAME}@{PRODUCTION_ALIAS})"
    )


if __name__ == "__main__":
    main()
