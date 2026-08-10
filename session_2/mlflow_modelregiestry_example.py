"""Load and manage a registered model with MlflowClient (MLflow 3.x).

This is the consumer side of ``mflow_example.py``: that script *registers* the
``RideDurationModel``; this one *lists*, *promotes*, and *loads* it for use.


Prereqs: the tracking server is up (``docker compose up -d mlflow``) and you've
run ``python mflow_example.py`` at least once to register a version.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "RideDurationModel"

# The MlflowClient URI only governs calls made through `client`. The *fluent*
# APIs (mlflow.sklearn.load_model, etc.) read the GLOBAL tracking URI instead —
# without this line they silently fall back to a local empty store and you get
# "Registered Model ... not found". So set both.
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# ── 1. List all registered versions ─────────────────────────────────────────
# In 3.x, inspect `aliases` (a list) instead of the dead `current_stage` field.
print(f"Versions of '{MODEL_NAME}':")
for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    # search results don't populate aliases, so fetch the version to see them.
    full = client.get_model_version(MODEL_NAME, v.version)
    print(f"  v{v.version}  aliases={list(full.aliases)}  run_id={v.run_id}")

# ── 2. Promote a version by moving an alias ──────────────────────────────────
# This is the modern equivalent of "transition to Production". An alias is just
# a label pointing at a version; re-assigning it is how you promote/roll back.
# Point `@production` at whatever `@champion` currently is.
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
client.set_registered_model_alias(MODEL_NAME, "production", champion.version)
print(f"\nSet @production -> v{champion.version} (was @champion)")

# ── 3. Load a model from the registry ────────────────────────────────────────
# Three URI forms you can load — pick by how stable you need the reference:
#   models:/Name@alias    → follows the alias (recommended for serving)
#   models:/Name/<version> → pins one exact, immutable version
# (The old models:/Name/<stage> form no longer works in 3.x.)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
print(f"\nLoaded '{MODEL_NAME}@production' -> {type(model).__name__}")

# Load a specific version instead (uncomment to try):
# model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion.version}")

# ── 4. Use the loaded model ──────────────────────────────────────────────────
# Features are [distance_km, passengers] — same schema the model was trained on.
sample = np.array([[10.0, 2], [3.5, 1], [25.0, 4]])
preds = model.predict(sample)
for (dist, pax), pred in zip(sample, preds):
    print(f"  {dist:>5} km, {int(pax)} passenger(s) -> {pred:.1f} min")
