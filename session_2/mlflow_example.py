"""MLflow tracking example — compare MANY models, not just track one.

This script trains **13 runs across 4 model families** on the same
ride-duration data and logs every one of them to the same MLflow experiment:

    • Linear models   (LinearRegression, Ridge with different alphas)
    • Random Forest   (different tree counts / depths)
    • XGBoost         (different depths / learning rates)
    • MLP             (different architectures / learning rates)

This is the moment MLflow earns its keep: one run is just a logbook entry,
but 13 runs in one experiment become a *leaderboard*. In the UI you can sort
every run by `mae`, filter by `tags.model_family`, and see at a glance which
family + hyperparameters win — and whether the expensive models are worth it.

Start the MLflow tracking server first (it ships in this folder's
``docker-compose.yml``):

    docker compose up -d mlflow      #  → UI at http://localhost:5000

Then run this script:

    python mlflow_example.py

Open http://localhost:5000 → experiment ``ride-duration-model``:
  1. Click the ``mae`` column header to sort — instant leaderboard.
  2. Tick several runs → "Compare" → parallel-coordinates + scatter plots.
  3. Search bar: ``tags.model_family = 'xgboost'`` to isolate one family.

macOS note: XGBoost needs the OpenMP runtime → ``brew install libomp``.
"""

from __future__ import annotations

import os
import time

# Use a non-interactive matplotlib backend: we only save figures to disk / hand
# them to MLflow, we never pop up a window. This MUST be set before pyplot is
# imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from src.train import FEATURES, evaluate, generate_data, split_data

# Point at the tracking server started via docker compose. Override with the
# MLFLOW_TRACKING_URI env var (e.g. to log to a remote server).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Name under which the WINNING model lives in the Model Registry (a separate
# concept from the experiment: an experiment groups *runs*, the registry
# versions *models* promoted out of those runs).
REGISTERED_MODEL_NAME = "RideDurationModel"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("ride-duration-model")

# ── Data ───────────────────────────────────────────────────────────────────
# Synthetic ride data, split into train / validation partitions. Every run
# below sees the SAME split — otherwise the leaderboard wouldn't be fair.
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)


# ── The model zoo ──────────────────────────────────────────────────────────
# One entry per run: (run_name, family, params, build_fn). The params dict is
# exactly what gets logged with `mlflow.log_params`, so what you see in the UI
# is what the model was actually built with.
#
# Note the MLP is wrapped in a Pipeline with a StandardScaler: neural nets
# need standardized inputs (distance_km spans 0.5–30, passengers 1–4), while
# tree models are scale-invariant and don't care.


def make_linear(params):
    if params["alpha"] == 0.0:
        return LinearRegression()
    return Ridge(alpha=params["alpha"])


def make_forest(params):
    return RandomForestRegressor(**params)


def make_xgb(params):
    return XGBRegressor(**params, random_state=42, verbosity=0)


def make_mlp(params):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**params, max_iter=500, random_state=42)),
        ]
    )


MODEL_ZOO = [
    # family: linear — alpha is the L2 regularisation strength (0 = plain OLS)
    ("linear-ols", "linear", {"alpha": 0.0}, make_linear),
    ("linear-ridge-a1", "linear", {"alpha": 1.0}, make_linear),
    ("linear-ridge-a100", "linear", {"alpha": 100.0}, make_linear),
    # family: random_forest — more/deeper trees = more capacity
    (
        "rf-small",
        "random_forest",
        {"n_estimators": 50, "max_depth": 4, "random_state": 42},
        make_forest,
    ),
    (
        "rf-baseline",
        "random_forest",
        {"n_estimators": 100, "max_depth": 6, "random_state": 42},
        make_forest,
    ),
    (
        "rf-big",
        "random_forest",
        {"n_estimators": 300, "max_depth": 10, "random_state": 42},
        make_forest,
    ),
    # family: xgboost — depth × learning-rate trade-off
    (
        "xgb-shallow-fast",
        "xgboost",
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.3},
        make_xgb,
    ),
    (
        "xgb-baseline",
        "xgboost",
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        make_xgb,
    ),
    (
        "xgb-deep",
        "xgboost",
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        make_xgb,
    ),
    (
        "xgb-slow-learner",
        "xgboost",
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03},
        make_xgb,
    ),
    # family: mlp — width/depth of the network and its learning rate
    (
        "mlp-tiny",
        "mlp",
        {"hidden_layer_sizes": (16,), "learning_rate_init": 0.01},
        make_mlp,
    ),
    (
        "mlp-wide",
        "mlp",
        {"hidden_layer_sizes": (64, 32), "learning_rate_init": 0.001},
        make_mlp,
    ),
    (
        "mlp-deep",
        "mlp",
        {"hidden_layer_sizes": (64, 64, 32), "learning_rate_init": 0.001},
        make_mlp,
    ),
]


# ── Family-specific diagnostics ────────────────────────────────────────────
# Different model families expose different introspection, and MLflow happily
# stores whichever artifacts each run produces — runs in one experiment do NOT
# need identical outputs:
#   linear        → coefficients        (what does each feature contribute?)
#   trees / xgb   → feature importances (which feature drives the splits?)
#   mlp           → per-epoch loss curve as STEP metrics (did it converge?)


def log_family_diagnostics(model, family: str) -> None:
    if family == "linear":
        coefs = model.coef_
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(FEATURES, coefs)
        ax.set_ylabel("coefficient")
        ax.set_title("Linear model coefficients")
        mlflow.log_figure(fig, "plots/coefficients.png")
        plt.close(fig)

    elif family in ("random_forest", "xgboost"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([FEATURES[i] for i in order], importances[order])
        ax.set_ylabel("importance")
        ax.set_title("Feature importances")
        mlflow.log_figure(fig, "plots/feature_importances.png")
        plt.close(fig)

    elif family == "mlp":
        # A metric logged repeatedly with an increasing `step` becomes a
        # *series*, which MLflow renders as an interactive line chart — here
        # the network's training loss per epoch.
        for epoch, loss in enumerate(model.named_steps["mlp"].loss_curve_):
            mlflow.log_metric("train_loss", loss, step=epoch)


def log_fit_quality(model) -> None:
    """Predicted-vs-actual scatter — logged for EVERY run so you can flip
    between runs in the artifact browser and compare fits visually."""
    preds = model.predict(X_val)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_val, preds, s=8, alpha=0.4)
    lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect")  # y = x reference
    ax.set_xlabel("actual duration (min)")
    ax.set_ylabel("predicted duration (min)")
    ax.set_title("Predicted vs. actual")
    ax.legend()
    mlflow.log_figure(fig, "plots/predicted_vs_actual.png")
    plt.close(fig)


# ── Train every candidate, one MLflow run each ─────────────────────────────
results = []  # (mae, run_name, family, run_id, metrics) per candidate

for run_name, family, params, build in MODEL_ZOO:
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Config first: params become sortable/filterable columns in the
        #    experiment table — the backbone of run comparison.
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "model_family": family,  # → search: tags.model_family = 'mlp'
                "dataset": "synthetic-v1",
                "stage": "experiment",
            }
        )

        # 2. Train — and time it. Wall-clock cost is a metric too: a model
        #    that's 1% better but 50× slower may not be worth it.
        t0 = time.perf_counter()
        model = build(params)
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0

        # 3. One shared metric set across ALL runs (rmse/mae/r2). Identical
        #    names are what make cross-run sorting meaningful.
        metrics = evaluate(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_seconds", train_seconds)

        # 4. Family-specific extras (see above) + a fit plot for every run.
        log_family_diagnostics(model, family)
        log_fit_quality(model)

        # 5. Log the fitted model as a run artifact. We do NOT register it
        #    yet — registration is reserved for the winner, below. MLflow has
        #    one *flavor* per framework, so each family logs with its own —
        #    but every flavor can be loaded uniformly via `mlflow.pyfunc`.
        if family == "xgboost":
            mlflow.xgboost.log_model(model, name="model", input_example=X_train[:5])
        else:
            # MLflow ≥3 serializes sklearn models with `skops`, which audits
            # every class in the pickle and rejects ones it doesn't know.
            # The MLP's Adam optimizer state is such a class → trust it
            # explicitly (safer than switching back to raw pickle).
            trusted = (
                ["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
                if family == "mlp"
                else None
            )
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train[:5],
                skops_trusted_types=trusted,
            )

        results.append((metrics["mae"], run_name, family, run.info.run_id, metrics))
        print(
            f"  {run_name:<20} [{family:<13}] "
            f"MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}  "
            f"({train_seconds:.2f}s)"
        )


# ── Leaderboard ────────────────────────────────────────────────────────────
# The same view you get in the UI by clicking the `mae` column header.
results.sort(key=lambda r: r[0])

print("\n=== Leaderboard (val MAE, lower is better) ===")
for rank, (mae, run_name, family, _, metrics) in enumerate(results, start=1):
    print(
        f"  {rank:>2}. {run_name:<20} [{family:<13}] "
        f"MAE={mae:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}"
    )

best_mae, best_name, best_family, best_run_id, best_metrics = results[0]

# Spoiler: the linear models usually win here. The synthetic data IS linear
# (distance/speed + passenger overhead), so extra capacity buys nothing —
# exactly the kind of conclusion a run-comparison table makes obvious.


# ── Register ONLY the winner ───────────────────────────────────────────────
# Every run logged its model as an artifact, but only the champion gets
# promoted into the Model Registry. `runs:/<run_id>/model` points at the
# artifact we logged inside the winning run.
registered = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=REGISTERED_MODEL_NAME,
)
version = registered.version

# Enrich the new version — these calls target the *registry*, not the run.
client = MlflowClient()
client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=version,
    description=(
        f"Best of {len(MODEL_ZOO)} candidates ({best_name}, family={best_family}) "
        f"on synthetic-v1 data. Validation MAE={best_mae:.3f}, "
        f"R2={best_metrics['r2']:.3f}."
    ),
)
client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "validated", "true")
client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "source_run", best_run_id)
client.set_model_version_tag(
    REGISTERED_MODEL_NAME, version, "model_family", best_family
)

# An *alias* is a moving pointer to a specific version (aliases replaced the
# old "stages" like Staging/Production in MLflow 2.9+). Downstream code loads
# `models:/RideDurationModel@champion` and always gets the blessed version —
# without hard-coding a version number.
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\nWinner: {best_name} ({best_family})  |  MAE: {best_mae:.3f}")
print(f"Registered '{REGISTERED_MODEL_NAME}' version {version} (alias: @champion)")
print(f"\nOpen the MLflow UI: {TRACKING_URI}")
print("  • Sort the leaderboard:   experiment table → click the 'mae' column")
print("  • Compare runs:           tick runs → Compare → parallel coordinates")
print("  • Filter one family:      search bar → tags.model_family = 'xgboost'")
print("  • MLP convergence:        any mlp run → Model metrics → train_loss")
print(f"  • Registered model:       Models → {REGISTERED_MODEL_NAME}")
print(f"Load the blessed model later with: models:/{REGISTERED_MODEL_NAME}@champion")
