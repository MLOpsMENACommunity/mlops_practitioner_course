# src/batch_score.py
import io
from datetime import date

import mlflow
import mlflow.sklearn
import pandas as pd
from google.cloud import storage

from src.register_model import MODEL_NAME, PRODUCTION_ALIAS, TRACKING_URI
from src.train import FEATURES


def load_production_model():
    """Load the model carrying the ``@production`` alias — no hardcoded paths.

    MLflow 3 removed the Production/Staging *stages*, so the old
    ``models:/<name>/Production`` URI no longer resolves; aliases replace them.
    Run ``python src/register_model.py`` to create the alias.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")


def batch_score(bucket_name: str, input_blob: str, output_blob: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # ── 1. Read input data from GCS ────────────────────
    data = bucket.blob(input_blob).download_as_bytes()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"Scoring {len(df):,} rows …")

    # ── 2. Load model & score ──────────────────────────
    model = load_production_model()
    # FEATURES is the single source of truth (src/train.py) — the model was fit
    # on exactly these columns, so selecting them by hand risks a silent skew.
    df["pred"] = model.predict(df[FEATURES].to_numpy())
    df["run_date"] = date.today().isoformat()

    # ── 3. Write predictions back to GCS ──────────────
    out_buffer = io.BytesIO()
    df.to_parquet(out_buffer, index=False)
    bucket.blob(output_blob).upload_from_string(out_buffer.getvalue())
    print(f"Predictions written to gs://{bucket_name}/{output_blob}")


if __name__ == "__main__":
    batch_score(
        bucket_name="mlops-session2-iti",
        input_blob="scoring/input/2026-07-14.parquet",
        output_blob=f"scoring/output/{date.today()}.parquet",
    )
