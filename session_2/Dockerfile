# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Inference image for the Ride Duration model (session 2).
#
# Multi-stage build:
#   • builder  — installs the project + serving deps into an isolated venv
#   • runtime  — a slim image carrying only the venv and the app (no model)
#
# Build context is `session_2/` (see .github/workflows/ci_cd.yml → context).
#
# The image ships CODE ONLY — the model is NOT baked in, so one image serves any
# model version. Provide a model at run time by mounting it at MODEL_PATH:
# Build:  docker build -t ride-api .
# Run:    docker run -p 8000:8000 \
#           -v "$(pwd)/models/rf_model.pkl:/app/models/rf_model.pkl:ro" ride-api
#         → POST http://localhost:8000/predict  {"distance_km": 10, "passengers": 2}
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps into an isolated venv we copy into the final stage (keeps the
# runtime image free of pip caches and build cruft).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy only what pip needs to resolve/install first, so this layer stays cached
# as long as dependencies don't change. `src` is the installable package;
# README.md is referenced by pyproject's `readme` field.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the project + the `serve` extra (fastapi + uvicorn). The core deps
# (scikit-learn, numpy, joblib) are what unpickle and run the model.
RUN pip install ".[serve]"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/rf_model.pkl

# Bring over the ready-to-use virtualenv (includes the installed `src` package).
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The inference entry point only — the model is NOT baked in. serve.py loads the
# model from MODEL_PATH at startup, so provide one at run time by mounting a file
# there (e.g. the latest @champion pulled from the MLflow/S3 model registry). One
# image then serves any model version, and retraining never rebuilds it.
COPY serve.py ./

# Run as a non-root user for safety.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Container-level liveness check hitting the same endpoint CI uses. Uses stdlib
# only (no curl in the slim image).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Bind to 0.0.0.0 so the app is reachable from outside the container.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
