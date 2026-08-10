# Extend the official Airflow 3.3.0 image with the libraries our DAGs import
# at task-run time. Building a custom image (instead of _PIP_ADDITIONAL_REQUIREMENTS)
# means deps are baked in once, not reinstalled on every container start.
ARG AIRFLOW_VERSION=3.3.0
FROM apache/airflow:${AIRFLOW_VERSION}

# Install as the unprivileged `airflow` user (pip must not run as root here).
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
