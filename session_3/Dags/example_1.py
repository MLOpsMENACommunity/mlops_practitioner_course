# dags/retrain_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── DAG definition ─────────────────────────────────────
dag = DAG(
    dag_id="ride_duration_retrain",
    schedule="@weekly",  # every Monday 00:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,  # don't backfill missed runs
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "owner": "ml-team",
    },
)


# ── Tasks (Python callables) ───────────────────────────
def extract(**ctx):
    """Pull new rides data from GCS into local data/raw/."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket("mlops-session2-iti")
    bucket.blob("rides/latest.parquet").download_to_filename("data/raw/rides.parquet")


def train(**ctx):
    """Train model; push run ID to XCom for downstream tasks."""
    import mlflow
    import subprocess

    with mlflow.start_run() as run:
        # Run as a module, not a script path: train.py does `from src.config
        # import ...`, which needs /opt/airflow (the cwd) on sys.path — running
        # `python src/train.py` puts src/ there instead and the import fails.
        subprocess.run(["python", "-m", "src.train"], check=True)
        ctx["ti"].xcom_push(key="run_id", value=run.info.run_id)


def evaluate(**ctx):
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train")
    # compare MAE against current Production model
    ...


# ── Wire tasks into the DAG ────────────────────────────
t_extract = PythonOperator(task_id="extract", python_callable=extract, dag=dag)
t_train = PythonOperator(task_id="train", python_callable=train, dag=dag)
t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate, dag=dag)

t_extract >> t_train >> t_evaluate  # dependency chain
