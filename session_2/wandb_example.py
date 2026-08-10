"""Weights & Biases tracking example — the W&B counterpart to ``mflow_example.py``.

Same synthetic ride-duration model as the MLflow example (reused from
``src.train``), but tracked with W&B. It mirrors the MLflow script feature for
feature so you can compare the two tools side by side:

    MLflow                          →  Weights & Biases
    ─────────────────────────────────────────────────────────────
    mlflow.log_params               →  wandb.config
    mlflow.set_tags                 →  wandb.init(tags=..., notes=...)
    mlflow.log_metrics              →  wandb.log({...})
    log_metric(name, v, step=...)   →  wandb.log({...}, step=...)   (line charts)
    mlflow.log_figure               →  wandb.log({name: wandb.Image(fig)})
    log_model + registered_model    →  wandb.Artifact + aliases ("champion")
    client.set_..._alias/tag        →  artifact aliases + metadata

Authentication — pick one before running:

    wandb login                     # log to wandb.ai (needs a free account + API key)
    export WANDB_MODE=offline       # no account/network: writes runs to ./wandb/

Then run:

    python wandb_example.py

With ``WANDB_MODE=offline`` everything below runs locally EXCEPT the sweep
(sweeps are orchestrated by the W&B server and need a login). Push saved
offline runs later with:  ``wandb sync wandb/offline-run-*``
"""

from __future__ import annotations

# Use a non-interactive matplotlib backend: we only hand figures to W&B, we
# never pop up a window. This MUST be set before pyplot is imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import joblib
import wandb

from src.train import (
    FEATURES,  # ["distance_km", "passengers"]
    evaluate,
    generate_data,
    split_data,
    train_model,
)

PROJECT = "ride-duration"
#: Name of the model artifact — the W&B analog of MLflow's registered model.
MODEL_ARTIFACT = "ride-duration-model"

# ── Data ─────────────────────────────────────────────────────────────────────
# Synthetic ride data, split into train / validation partitions (module-level
# so both the baseline run and the sweep's train_fn can reuse it).
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)


# ── Baseline run ──────────────────────────────────────────────────────────────
# `config` is the run's hyperparameter record (≈ MLflow log_params); `tags` and
# `notes` are free-form metadata (≈ MLflow set_tags), searchable in the UI.
params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
run = wandb.init(
    project=PROJECT,
    name="rf-baseline",
    config=params,
    tags=["random_forest", "synthetic-v1", "experiment"],
    notes="RandomForest ride-duration baseline on synthetic-v1 data.",
)

# ── 1. Train + scalar metrics ─────────────────────────────────────────────────
model = train_model(X_train, y_train, params)
metrics = evaluate(model, X_val, y_val)  # -> {"rmse", "mae", "r2"}
wandb.log({**metrics, "val_size": len(y_val)})

# ── 2. Line chart, technique A: STEP metrics (a learning curve) ────────────────
# A metric logged repeatedly with an increasing `step` becomes a *series* that
# W&B plots as an interactive line chart. Here we retrain the forest with a
# growing number of trees and record validation error at each size — answering
# "do more trees keep helping, or have we plateaued?" at a glance.
for n_estimators in (10, 25, 50, 100, 200, 400):
    lc_params = {**params, "n_estimators": n_estimators}
    lc_model = train_model(X_train, y_train, lc_params)
    lc_metrics = evaluate(lc_model, X_val, y_val)
    # `step` is the x-axis position (here: the tree count). Prefixing with
    # "lc/" groups these into their own "lc" section/panel in the W&B workspace.
    wandb.log(
        {
            "lc/val_rmse": lc_metrics["rmse"],
            "lc/val_mae": lc_metrics["mae"],
            "lc/val_r2": lc_metrics["r2"],
        },
        step=n_estimators,
    )

# ── 3. Diagnostics, technique B: FIGURE artifacts (PNGs) ───────────────────────
# `wandb.Image(fig)` uploads a matplotlib Figure as an image that renders inline
# in the run's Media panel — great for diagnostics that aren't a single number.
preds = model.predict(X_val)
residuals = y_val - preds

# (a) Predicted vs. actual — points hug the diagonal when the model fits.
fig_pva, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_val, preds, s=8, alpha=0.4)
lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
ax.plot(lims, lims, "r--", linewidth=1, label="perfect")  # y = x reference
ax.set_xlabel("actual duration (min)")
ax.set_ylabel("predicted duration (min)")
ax.set_title("Predicted vs. actual")
ax.legend()

# (b) Residual histogram — should be roughly centred on 0 and bell-shaped.
fig_res, ax = plt.subplots(figsize=(6, 4))
ax.hist(residuals, bins=40)
ax.axvline(0, color="r", linestyle="--", linewidth=1)
ax.set_xlabel("residual (actual − predicted, min)")
ax.set_ylabel("count")
ax.set_title("Residual distribution")

# (c) Feature importances — which inputs drive the prediction.
importances = model.feature_importances_
order = np.argsort(importances)[::-1]
fig_imp, ax = plt.subplots(figsize=(6, 4))
ax.bar([FEATURES[i] for i in order], importances[order])
ax.set_ylabel("importance")
ax.set_title("Feature importances")

wandb.log(
    {
        "plots/predicted_vs_actual": wandb.Image(fig_pva),
        "plots/residuals": wandb.Image(fig_res),
        "plots/feature_importances": wandb.Image(fig_imp),
    }
)
for fig in (fig_pva, fig_res, fig_imp):
    plt.close(fig)  # free the figures so we don't leak memory across runs

# ── 4. A W&B Table — the raw prediction sample behind the plots ───────────────
# Tables are sortable/filterable in the UI and can be queried across runs.
table = wandb.Table(columns=["distance_km", "passengers", "actual", "predicted"])
for row, actual, pred in zip(X_val[:100], y_val[:100], preds[:100]):
    table.add_data(float(row[0]), int(row[1]), float(actual), float(pred))
wandb.log({"val_predictions": table})

# ── 5. Log AND "register" the model ────────────────────────────────────────────
# W&B has no separate registry API like MLflow; instead you log the model as a
# versioned *artifact* and attach aliases. An alias like "champion" is a moving
# pointer to a specific version — downstream code fetches
# `run.use_artifact("ride-duration-model:champion")` and always gets whichever
# version we've blessed, without hard-coding a version number. `metadata` +
# `description` carry the same governance info MLflow puts on a model version.
joblib.dump(model, "model.pkl")
artifact = wandb.Artifact(
    MODEL_ARTIFACT,
    type="model",
    description=(
        "RandomForest ride-duration regressor trained on synthetic-v1 data. "
        f"Validation MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}."
    ),
    metadata={**params, **metrics, "validated": True},
)
artifact.add_file("model.pkl")
run.log_artifact(artifact, aliases=["champion", "synthetic-v1"])

run.finish()

print(f"Baseline run logged  |  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}")
print("  • Learning curve:  run → 'lc' panels (lc/val_* series)")
print("  • Diagnostics:     run → Media panel (plots/*)")
print(f"  • Model artifact:  {MODEL_ARTIFACT}:champion")
print(f"  Load it later with: run.use_artifact('{MODEL_ARTIFACT}:champion')")


# ── 6. Hyperparameter sweep (Bayesian search) ─────────────────────────────────
def train_fn() -> None:
    """One sweep trial: W&B injects the sampled params via ``wandb.config``."""
    with wandb.init() as sweep_run:
        cfg = dict(sweep_run.config)
        cfg.setdefault("random_state", 42)
        sweep_model = train_model(X_train, y_train, cfg)
        sweep_metrics = evaluate(sweep_model, X_val, y_val)
        wandb.log(sweep_metrics)  # logs "mae" (the sweep's optimisation target)


sweep_config = {
    "method": "bayes",
    "metric": {"name": "mae", "goal": "minimize"},
    "parameters": {
        "n_estimators": {"values": [50, 100, 200]},
        "max_depth": {"min": 3, "max": 10},
    },
}

# Sweeps are orchestrated by the W&B *server*, so they require a login and do
# NOT work in offline mode. Skip gracefully when we can't reach the backend so
# the baseline run above still completes on its own.
if wandb.setup().settings._offline:
    print(
        "Skipping sweep: WANDB_MODE=offline. Run `wandb login` and re-run "
        "without WANDB_MODE=offline to execute the Bayesian sweep."
    )
else:
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=train_fn, count=20)  # run 20 trials
    print(f"Sweep {sweep_id} complete.")
