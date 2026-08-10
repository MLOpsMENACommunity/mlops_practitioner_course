"""Load a registered model whose artifacts live in a GCS bucket (MLflow 3.x).

This is the GCS counterpart to ``mlflow_s3_registry_example.py``. The single
most important lesson here: **the client code is IDENTICAL** to the local-volume
version. Because the server runs with ``--serve-artifacts`` (see
``docker-compose.gcs.yml``), it PROXIES GCS for you — the client just talks to
``http://localhost:5000`` and never needs GCP credentials or
``google-cloud-storage``. Whether ``model.skops`` sits in a Docker volume or a
GCS bucket is entirely the server's concern; ``load_model("models:/Name@champion")``
looks the same either way.

Prereqs:
    1. Start the GCS-backed server:
         cp .env.gcs.example .env     # set bucket + key path
         # drop your service-account key at ./gcp-key.json (gitignored)
         docker compose -f docker-compose.gcs.yml up -d
    2. Register at least one version by running:  python mlflow_example.py
       (its code is unchanged too — it logs to the same http://localhost:5000).

Then:
    python mlflow_gcs_registry_example.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

# Same as always: the client only knows the SERVER address. It has no idea (and
# doesn't care) that the server stores files in GCS.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# export MLFLOW_TRACKING_URI=https://mlflow.your-company.internal   # or the VM's http://<ip>:5000


MODEL_NAME = "RideDurationModel"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# ── 1. List registered versions (registry = server's DB, unchanged by GCS) ────
print(f"Versions of '{MODEL_NAME}':")
for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    full = client.get_model_version(MODEL_NAME, v.version)
    print(f"  v{v.version}  aliases={list(full.aliases)}  run_id={v.run_id}")

# ── 2. See WHERE the artifacts physically live ────────────────────────────────
# The run's artifact_uri shows the storage location. Under --serve-artifacts it
# reads back as a `mlflow-artifacts:/…` proxy URI (the server resolves it to
# gs://<bucket>/mlflow/… internally). That indirection is exactly why the client
# needs no GCS access — it fetches bytes THROUGH the server, not from GCS directly.
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
run = client.get_run(champion.run_id)
print(f"\n@champion is v{champion.version}")
print(f"  run artifact_uri : {run.info.artifact_uri}")
print("  (server resolves that proxy URI to your gs://<bucket>/mlflow/… path)")

# ── 3. Load the model — the punchline: THIS CODE DID NOT CHANGE ───────────────
# Identical to the local-volume example. The bytes are streamed GCS → server →
# client over HTTP and deserialized into a RandomForestRegressor in memory here.
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
print(
    f"\nLoaded '{MODEL_NAME}@champion' from GCS-backed server -> {type(model).__name__}"
)

# ── 4. Use it ─────────────────────────────────────────────────────────────────
sample = np.array([[10.0, 2], [3.5, 1], [25.0, 4]])  # [distance_km, passengers]
for (dist, pax), pred in zip(sample, model.predict(sample)):
    print(f"  {dist:>5} km, {int(pax)} passenger(s) -> {pred:.1f} min")


# ── 5. (Optional) Prove the files are really in GCS, by reading the bucket ────
# This is the ONLY part that needs GCP credentials + google-cloud-storage on the
# CLIENT — and it's here purely to demonstrate the physical objects. Normal
# MLflow usage above never touches GCS directly. Runs only if you set the two
# env vars (bucket + a key path in GOOGLE_APPLICATION_CREDENTIALS).
bucket = os.getenv("MLFLOW_GCS_BUCKET")
if bucket and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    try:
        from google.cloud import storage

        gcs = storage.Client()
        blobs = gcs.list_blobs(bucket, prefix="mlflow/", max_results=10)
        print(f"\nObjects physically in gs://{bucket}/mlflow/ (first 10):")
        for blob in blobs:
            print(f"  {blob.name}  ({blob.size} bytes)")
    except Exception as exc:  # noqa: BLE001 - demo only, don't fail the script
        print(f"\n[skipped direct GCS listing: {exc}]")
else:
    print(
        "\n(Set MLFLOW_GCS_BUCKET + GOOGLE_APPLICATION_CREDENTIALS to also list "
        "the raw GCS objects — optional, not needed for load_model above.)"
    )
