# syntax=docker/dockerfile:1

# ── Stage 1: builder ──────────────────────────────────
# Install dependencies into an isolated virtualenv so the final image
# carries only the runtime artifacts (no build tools, no pip cache).
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create the venv that we'll copy into the final stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy only what pip needs to resolve/install the project. This keeps the
# Docker layer cache warm as long as dependencies don't change.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the project (core deps: fastapi, uvicorn, pydantic, litestar) into
# the venv. The 'onnx' extra is intentionally excluded — the API doesn't need it.
RUN pip install .

# ── Stage 2: runtime ──────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Bring over the ready-to-use virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy the application entry point. `src` is already installed in the venv.
COPY fastapi_example.py ./

# Run as a non-root user for safety.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Bind to 0.0.0.0 so the app is reachable from outside the container.
CMD ["uvicorn", "fastapi_example:app", "--host", "0.0.0.0", "--port", "8000"]
