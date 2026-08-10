"""Load a registered model whose artifacts live in an S3 bucket (MLflow 3.x).

This is the S3 counterpart to ``mlflow_modelregiestry_example.py``. The single
most important lesson here: **the client code is IDENTICAL** to the local-volume
version. Because the server runs with ``--serve-artifacts`` (see
``docker-compose.s3.yml``), it PROXIES S3 for you — the client just talks to
``http://localhost:5000`` and never needs AWS credentials or ``boto3``. Whether
``model.skops`` sits in a Docker volume or an S3 bucket is entirely the server's
concern; ``load_model("models:/Name@champion")`` looks the same either way.

Prereqs:
    1. Start the S3-backed server:
         cp .env.s3.example .env      # fill in bucket + AWS keys
         docker compose -f docker-compose.s3.yml up -d
    2. Register at least one version by running:  python mflow_example.py
       (its code is unchanged too — it logs to the same http://localhost:5000).

Then:
    python mlflow_s3_registry_example.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

# Same as always: the client only knows the SERVER address. It has no idea (and
# doesn't care) that the server stores files in S3.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "RideDurationModel"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# ── 1. List registered versions (registry = server's DB, unchanged by S3) ─────
print(f"Versions of '{MODEL_NAME}':")
for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    full = client.get_model_version(MODEL_NAME, v.version)
    print(f"  v{v.version}  aliases={list(full.aliases)}  run_id={v.run_id}")

# ── 2. See WHERE the artifacts physically live ────────────────────────────────
# The run's artifact_uri shows the storage location. Under --serve-artifacts it
# reads back as a `mlflow-artifacts:/…` proxy URI (the server resolves it to
# s3://<bucket>/mlflow/… internally). That indirection is exactly why the client
# needs no S3 access — it fetches bytes THROUGH the server, not from S3 directly.
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
run = client.get_run(champion.run_id)
print(f"\n@champion is v{champion.version}")
print(f"  run artifact_uri : {run.info.artifact_uri}")
print("  (server resolves that proxy URI to your s3://<bucket>/mlflow/… path)")

# ── 3. Load the model — the punchline: THIS CODE DID NOT CHANGE ───────────────
# Identical to the local-volume example. The bytes are streamed S3 → server →
# client over HTTP and deserialized into a RandomForestRegressor in memory here.
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
print(
    f"\nLoaded '{MODEL_NAME}@champion' from S3-backed server -> {type(model).__name__}"
)

# ── 4. Use it ─────────────────────────────────────────────────────────────────
sample = np.array([[10.0, 2], [3.5, 1], [25.0, 4]])  # [distance_km, passengers]
for (dist, pax), pred in zip(sample, model.predict(sample)):
    print(f"  {dist:>5} km, {int(pax)} passenger(s) -> {pred:.1f} min")


# ── 5. (Optional) Prove the files are really in S3, by reading the bucket ─────
# This is the ONLY part that needs AWS credentials + boto3 on the CLIENT — and
# it's here purely to demonstrate the physical objects. Normal MLflow usage
# above never touches S3 directly. Runs only if you set the two env vars.
bucket = os.getenv("MLFLOW_S3_BUCKET")
if bucket and os.getenv("AWS_ACCESS_KEY_ID"):
    try:
        import boto3

        s3 = boto3.client(
            "s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL") or None
        )
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="mlflow/", MaxKeys=10)
        print(f"\nObjects physically in s3://{bucket}/mlflow/ (first 10):")
        for obj in resp.get("Contents", []):
            print(f"  {obj['Key']}  ({obj['Size']} bytes)")
    except Exception as exc:  # noqa: BLE001 - demo only, don't fail the script
        print(f"\n[skipped direct S3 listing: {exc}]")
else:
    print(
        "\n(Set MLFLOW_S3_BUCKET + AWS_ACCESS_KEY_ID to also list the raw S3 "
        "objects — optional, not needed for load_model above.)"
    )
