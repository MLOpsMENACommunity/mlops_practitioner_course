"""MLflow end-to-end on ONE model and ONE dataset — a tour of every feature.

``mlflow_example.py`` (next door) trains 13 models to show why *comparing* runs
matters. This script does the opposite: **one** RandomForest on **one**
synthetic ride-duration dataset, but it exercises essentially everything MLflow
gives you around that single run, in the order you'd actually meet it:

    0. `mlflow.doctor()`            — sanity-check the client/server setup
    1. Experiment                   — create with tags, set it, tag it
    2. Autologging                  — `mlflow.sklearn.autolog()`
    3. Dataset lineage              — `mlflow.data` + `mlflow.log_input`
    4. The run                      — name, tags, description, system metrics
    5. Nested runs                  — one child run per cross-validation fold
    6. Metrics                      — scalars *and* a stepped metric series
    7. Artifacts                    — text, dict, figure, image, table, file, dir
    8. The model                    — signature, input example, `log_model`
    9. Evaluation                   — `mlflow.models.evaluate` + a custom metric
   10. Reading it back              — get_run, metric history, search_runs
   11. Model Registry               — versions, descriptions, tags, aliases
   12. Loading                      — by alias, by version, by flavor

Start the tracking server first (it ships in this folder's compose file — the
Model Registry needs a *database* backend, which the plain file store can't
provide):

    docker compose up -d mlflow      #  → UI at http://localhost:5000

Then:

    python mlflow_simple_example.py

Everything below runs against that server; override it with the
``MLFLOW_TRACKING_URI`` env var to log somewhere else.
"""

from __future__ import annotations

import os
import platform
import tempfile
import time
from pathlib import Path

# Non-interactive matplotlib backend: figures are handed straight to MLflow,
# never shown in a window. MUST be set before pyplot is imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.data
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import convert_input_example_to_serving_input, infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from src.train import FEATURES, TARGET, evaluate, generate_data, split_data

# ── 0. Configuration & client ──────────────────────────────────────────────
# Three URIs matter in MLflow, and they are NOT the same thing:
#   • tracking URI  — where runs (params/metrics/artifacts) are recorded
#   • registry URI  — where registered models live (defaults to the tracking
#                     URI; split them when the registry is a separate service)
#   • artifact URI  — per-run storage for files (set by the server; ours is
#                     proxied through it, see docker-compose.yml)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "mlflow-simple-example"
REGISTERED_MODEL_NAME = "RideDurationSimple"

#: The single set of hyperparameters this script trains with.
PARAMS = {
    "n_estimators": 120,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "random_state": 42,
}

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

# `MlflowClient` is the *low-level* API: everything the fluent `mlflow.*`
# functions do to the "current" run, the client can do to any run/model by id.
# Most registry operations are client-only — `mlflow.register_model` is the
# only fluent one.
client = MlflowClient()

# `mlflow.doctor()` prints the resolved URIs, versions and dependency status —
# the first thing to run when tracking "silently does nothing".
print("=== mlflow.doctor() ===")
mlflow.doctor()

# System metrics (CPU/RAM/disk, GPU if present) are sampled in a background
# thread and logged as normal stepped metrics. Needs `psutil`. The defaults
# sample every 10s — far too slow for a script that finishes in seconds.
mlflow.enable_system_metrics_logging()
mlflow.set_system_metrics_sampling_interval(1)
mlflow.set_system_metrics_samples_before_logging(1)


# ── 1. The experiment ──────────────────────────────────────────────────────
# An *experiment* is the folder that groups runs. Creating it explicitly (as
# opposed to letting `set_experiment` do it) is what lets you attach tags at
# creation time. `mlflow.note.content` is the magic tag MLflow renders as the
# experiment's description box in the UI.
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(
        EXPERIMENT_NAME,
        tags={
            "project": "ride-duration",
            "owner": "mlops-course",
            "mlflow.note.content": (
                "One model, one dataset — a tour of every MLflow feature."
            ),
        },
    )
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)
# Tags can be added later too, either in bulk (fluent) or one at a time.
mlflow.set_experiment_tags({"course_session": "2", "tracker": "mlflow"})
client.set_experiment_tag(experiment_id, "data_version", "synthetic-v1")


# ── 2. Autologging ─────────────────────────────────────────────────────────
# One call and MLflow patches `fit()` for the whole library: hyperparameters,
# training metrics, the model, an input example and a signature all get logged
# without a single explicit call. We keep `log_models=False` so that the ONE
# model artifact in this run is the one we log by hand below (with our own
# signature and registry name) — otherwise you get two copies.
#
# Flavors with autolog support include sklearn, xgboost, lightgbm, pytorch,
# keras, transformers, spark, statsmodels…; `mlflow.autolog()` enables all of
# them at once.
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False,
    log_datasets=False,
    log_post_training_metrics=True,  # metrics from post-fit `model.score(...)`
    extra_tags={"autologged": "true"},
    silent=False,
)


# ── 3. The data (+ lineage) ────────────────────────────────────────────────
# Same synthetic generator the rest of session 2 uses, wrapped in DataFrames so
# the logged model gets *named* columns in its signature (nicer for serving).
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)

train_df = pd.DataFrame(X_train, columns=FEATURES).assign(**{TARGET: y_train})
val_df = pd.DataFrame(X_val, columns=FEATURES).assign(**{TARGET: y_val})
X_train_df, X_val_df = train_df[FEATURES], val_df[FEATURES]

# `mlflow.data` turns a DataFrame into a *dataset object*: MLflow records its
# schema, its profile (row/column counts) and a content **digest** — a hash you
# can compare across runs to prove two runs really saw the same data.
train_dataset = mlflow.data.from_pandas(
    train_df, targets=TARGET, name="ride-duration-synthetic-v1-train"
)
val_dataset = mlflow.data.from_pandas(
    val_df, targets=TARGET, name="ride-duration-synthetic-v1-val"
)


# ── 4. The run ─────────────────────────────────────────────────────────────
# `start_run` as a context manager: the run is ended (and marked FINISHED, or
# FAILED on an exception) automatically. Everything logged inside lands here.
with mlflow.start_run(
    run_name="rf-full-tour",
    tags={"stage": "demo", "model_family": "random_forest"},
    description="Single RandomForest run that exercises the whole MLflow API.",
    log_system_metrics=True,
) as run:
    run_id = run.info.run_id
    print(f"\n=== run {run_id} ===")

    # Dataset lineage: `context` is free-form, but "training"/"validation"/
    # "test" are the conventional values the UI groups by.
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(val_dataset, context="validation")

    # -- Params: the singular and bulk forms, plus tags -----------------------
    # Params are IMMUTABLE: logging the same key with a different value inside
    # one run is an error. That is deliberate — a run describes one config.
    mlflow.log_params(PARAMS)
    mlflow.log_param("cv_folds", 3)
    mlflow.set_tag("git_branch", os.getenv("GIT_BRANCH", "local"))
    mlflow.set_tags({"python": platform.python_version(), "os": platform.system()})
    # Tags, unlike params, ARE mutable — set them freely as the run progresses.

    # -- 5. Nested runs: one child per CV fold -------------------------------
    # `nested=True` hangs a run under the active one. The UI shows them as a
    # collapsible tree, so a sweep / CV / multi-step job stays one logical unit.
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_maes: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_train_df), start=1):
        with mlflow.start_run(run_name=f"cv-fold-{fold}", nested=True):
            fold_model = RandomForestRegressor(**PARAMS)  # autolog logs PARAMS
            fold_model.fit(X_train_df.iloc[tr_idx], y_train[tr_idx])
            fold_metrics = evaluate(
                fold_model, X_train_df.iloc[va_idx], y_train[va_idx]
            )
            mlflow.log_param("fold", fold)
            mlflow.log_metrics(fold_metrics)
            fold_maes.append(fold_metrics["mae"])
            print(f"  fold {fold}: MAE={fold_metrics['mae']:.3f}")

    # Aggregate the folds back onto the PARENT run.
    mlflow.log_metrics(
        {
            "cv_mae_mean": float(np.mean(fold_maes)),
            "cv_mae_std": float(np.std(fold_maes)),
        }
    )

    # -- 6. Train the real model + metrics -----------------------------------
    t0 = time.perf_counter()
    model = RandomForestRegressor(**PARAMS)
    model.fit(X_train_df, y_train)  # ← autolog captures params/metrics here
    train_seconds = time.perf_counter() - t0

    metrics = evaluate(model, X_val_df, y_val)
    mlflow.log_metrics(metrics)  # bulk
    mlflow.log_metric("train_seconds", train_seconds)  # single

    # A metric logged repeatedly with a rising `step` becomes a SERIES, which
    # the UI draws as a line chart. Here: validation MAE as the forest grows,
    # computed by averaging the first k trees — no refitting needed.
    tree_preds = np.stack([t.predict(X_val_df.to_numpy()) for t in model.estimators_])
    running_mean = np.cumsum(tree_preds, axis=0) / np.arange(
        1, len(tree_preds) + 1
    ).reshape(-1, 1)
    for k in range(4, len(tree_preds), 5):
        mlflow.log_metric(
            "val_mae_vs_trees", mean_absolute_error(y_val, running_mean[k]), step=k + 1
        )

    # -- 7. Artifacts: every logging helper MLflow ships ---------------------
    # Artifacts are arbitrary files attached to the run. There is a helper for
    # each common shape, so you rarely have to touch the filesystem yourself.

    # (a) log_text — any string, rendered inline in the UI for .md/.txt/.json
    mlflow.log_text(
        f"# Run report\n\n"
        f"- rows (train/val): {len(train_df)}/{len(val_df)}\n"
        f"- validation MAE: {metrics['mae']:.4f}\n"
        f"- validation R2: {metrics['r2']:.4f}\n",
        "reports/summary.md",
    )

    # (b) log_dict — a dict serialized by extension (.json here, .yaml works)
    mlflow.log_dict(
        {"params": PARAMS, "features": FEATURES, "target": TARGET},
        "config/run_config.json",
    )

    # (c) log_figure — a matplotlib figure, no temp file dance
    preds = model.predict(X_val_df)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_val, preds, s=8, alpha=0.4)
    lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
    ax.set_xlabel("actual duration (min)")
    ax.set_ylabel("predicted duration (min)")
    ax.set_title("Predicted vs. actual")
    ax.legend()
    mlflow.log_figure(fig, "plots/predicted_vs_actual.png")
    plt.close(fig)

    # (d) log_image — a raw numpy array / PIL image. Here: the model's response
    #     surface over the (distance, passengers) grid, as a grayscale image.
    grid_distance = np.linspace(0.5, 30.0, 120)
    grid_passengers = np.arange(1, 5)
    surface = np.array(
        [
            model.predict(
                pd.DataFrame(
                    {"distance_km": grid_distance, "passengers": np.full(120, p)}
                )
            )
            for p in grid_passengers
        ]
    )
    scaled = (surface - surface.min()) / (surface.max() - surface.min())
    mlflow.log_image((scaled * 255).astype(np.uint8), "plots/response_surface.png")

    # (e) log_table — a DataFrame stored as a queryable MLflow table
    mlflow.log_table(
        pd.DataFrame(
            {
                "distance_km": X_val_df["distance_km"][:50],
                "passengers": X_val_df["passengers"][:50],
                "actual": y_val[:50],
                "predicted": preds[:50],
                "abs_error": np.abs(y_val[:50] - preds[:50]),
            }
        ),
        "tables/validation_predictions.json",
    )

    # (f) log_artifact / log_artifacts — a single file, or a whole directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        val_df.head(100).to_csv(tmp_path / "validation_sample.csv", index=False)
        mlflow.log_artifact(str(tmp_path / "validation_sample.csv"), "data")

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "feature_names.txt").write_text("\n".join(FEATURES))
        (bundle / "notes.txt").write_text("Everything in this dir is uploaded.")
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")

    # -- 8. The model itself -------------------------------------------------
    # A *signature* is the model's typed contract (input columns + output). It
    # is what makes serving reject a malformed request instead of silently
    # predicting nonsense, so always log one — `infer_signature` derives it.
    signature = infer_signature(X_val_df, preds)
    input_example = X_val_df.head(3)

    # In MLflow 3 a logged model is its OWN entity (`models:/m-<id>`) linked to
    # the run, not just a folder of run artifacts — that's why it shows up under
    # the run's "Models" tab and can be referenced without the run id.
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",  # the model's name within the run
        signature=signature,
        input_example=input_example,  # shown in the UI + used by validators
        registered_model_name=REGISTERED_MODEL_NAME,  # ← registers in one step
        code_paths=["src"],  # project code shipped with the model
        extra_pip_requirements=["pandas"],  # appended to the inferred env
        metadata={"dataset": "synthetic-v1", "task": "regression"},
    )
    print(f"  logged model: {model_info.model_uri}")

    # Sanity-check the model exactly the way a serving endpoint will call it:
    # `convert_input_example_to_serving_input` builds the JSON body a request
    # would carry, and `mlflow.models.predict` scores it through the real
    # pyfunc scoring path — catching schema and environment mistakes *before*
    # deployment.
    # (`env_manager="local"` reuses this interpreter; the default builds a
    # fresh virtualenv from the model's requirements, which is the stricter
    # check and what you want in CI.)
    serving_payload = convert_input_example_to_serving_input(input_example)
    mlflow.log_text(serving_payload, "serving/example_request.json")
    print("  scoring the example request through the serving path:")
    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data=input_example,
        env_manager="local",
    )
    print()  # the scoring subprocess prints its JSON without a trailing newline

    # -- 9. mlflow.models.evaluate -------------------------------------------
    # One call: runs the model over a dataset, computes the whole built-in
    # metric suite for the model type, and logs metrics *and* diagnostic
    # artifacts to the active run. `extra_metrics` bolts on your own.
    # (Top-level `mlflow.evaluate` is the same function, deprecated in 3.0;
    # `mlflow.genai.evaluate` is its LLM-flavoured sibling.)
    def _within_two_minutes(predictions, targets, metrics=None):
        """Share of predictions within 2 minutes of the truth."""
        errors = np.abs(np.asarray(predictions) - np.asarray(targets))
        return float(np.mean(errors <= 2.0))

    eval_result = mlflow.models.evaluate(
        model=model_info.model_uri,
        data=val_df,
        targets=TARGET,
        model_type="regressor",
        evaluators="default",
        extra_metrics=[
            mlflow.metrics.make_metric(
                eval_fn=_within_two_minutes,
                greater_is_better=True,
                name="pct_within_2min",
            )
        ],
        # SHAP explainability is on by default and needs the `shap` package;
        # turn it off to keep this script's dependency list short.
        evaluator_config={"log_model_explainability": False},
    )
    print("  mlflow.models.evaluate metrics:")
    for key, value in sorted(eval_result.metrics.items()):
        print(f"    {key:<32} {value:.4f}")

    artifact_uri = mlflow.get_artifact_uri()
    print(f"  artifact uri: {artifact_uri}")

# The run is closed here. `last_active_run()` is the handle to what just ended.
finished = mlflow.last_active_run()
print(f"\nrun {finished.info.run_name} finished with status {finished.info.status}")


# ── 10. Reading everything back ────────────────────────────────────────────
# Anything the UI shows, the API can return — tracking is a queryable database,
# not a write-only log. This is what makes CI checks and reports possible.
logged_run = client.get_run(run_id)
print("\n=== read back ===")
print(f"  params logged:  {len(logged_run.data.params)}")
print(f"  metrics logged: {len(logged_run.data.metrics)}")
print(f"  tags logged:    {len(logged_run.data.tags)}")
print(f"  datasets:       {[d.dataset.name for d in logged_run.inputs.dataset_inputs]}")

# A stepped metric comes back as its full history, not just the last value.
history = client.get_metric_history(run_id, "val_mae_vs_trees")
print(f"  val_mae_vs_trees history: {len(history)} points")

# Artifacts can be listed and downloaded without touching the UI.
print(f"  artifact tree: {[f.path for f in client.list_artifacts(run_id)]}")
with tempfile.TemporaryDirectory() as tmp:
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="reports/summary.md", dst_path=tmp
    )
    print(f"  downloaded {Path(local).name}: {Path(local).read_text().splitlines()[0]}")

# Logged tables come back as a DataFrame, ready for analysis.
table = mlflow.load_table("tables/validation_predictions.json", run_ids=[run_id])
print(f"  table reloaded: {table.shape[0]} rows × {table.shape[1]} cols")

# `search_runs` returns a DataFrame — filter with the same SQL-ish syntax the
# UI search bar uses (`metrics.mae < 1`, `tags.stage = 'demo'`, `params.…`).
runs_df = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.stage = 'demo'",
    order_by=["metrics.mae ASC"],
    max_results=5,
)
print(f"  search_runs → {len(runs_df)} matching run(s)")

# The client can also write to a *finished* run — useful for a later review
# step (a CI job stamping its verdict onto yesterday's training run).
client.set_tag(run_id, "reviewed_by", "mlops-course")
client.log_metric(run_id, "post_hoc_check", 1.0)


# ── 11. The Model Registry ─────────────────────────────────────────────────
# Runs are history; the registry is the *catalogue of deployable models*.
# `log_model(registered_model_name=...)` already created version N — here we
# curate it. (`mlflow.register_model("runs:/<id>/model", name)` is the explicit
# alternative when you decide to promote a run only later.)
model_version = client.search_model_versions(
    f"name = '{REGISTERED_MODEL_NAME}'", order_by=["version_number DESC"], max_results=1
)[0].version

client.update_registered_model(
    REGISTERED_MODEL_NAME,
    description="Ride-duration regressor (RandomForest) from the MLflow tour script.",
)
client.set_registered_model_tag(REGISTERED_MODEL_NAME, "task", "regression")

client.update_model_version(
    REGISTERED_MODEL_NAME,
    model_version,
    description=(
        f"Trained on synthetic-v1 ({len(train_df)} rows). "
        f"Validation MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}."
    ),
)
client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "validated", "true")
client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "source_run", run_id)

# An ALIAS is a movable pointer to a version. Aliases replaced the old
# Staging/Production *stages* (deprecated in MLflow 2.9, gone in 3.x): they are
# arbitrary names, so `@champion`, `@challenger`, `@eu-prod` all work, and
# deployment code references `models:/<name>@champion` instead of a version
# number — promoting a new model is then a one-line alias move.
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", model_version)
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", model_version)

by_alias = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
print("\n=== registry ===")
print(f"  {REGISTERED_MODEL_NAME}@champion → version {by_alias.version}")
print(f"  tags on the version: {by_alias.tags}")
print(
    "  registered models on this server: "
    f"{[m.name for m in client.search_registered_models(max_results=10)]}"
)


# ── 12. Loading the model back ─────────────────────────────────────────────
# Three ways in, all pointing at the same bytes:
#   • by alias   — models:/<name>@champion  (what production code should use)
#   • by version — models:/<name>/<version> (pin an exact release)
#   • by run     — runs:/<run_id>/model     (the artifact, pre-registry)
champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
print("\n=== load back ===")
print(f"  pyfunc @champion predictions: {np.round(champion.predict(input_example), 2)}")

# `pyfunc` is the universal flavor — every framework loads through it with the
# same `.predict()`. Loading the *native* flavor instead hands back the real
# sklearn estimator, with everything sklearn-specific still on it.
native = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{model_version}")
print(f"  native flavor: {type(native).__name__}, {native.n_estimators} trees")
importances = dict(zip(FEATURES, np.round(native.feature_importances_, 3)))
print(f"  feature importances: {importances}")

# The model's own metadata travels with it: signature, flavors, requirements.
loaded_info = mlflow.models.get_model_info(f"models:/{REGISTERED_MODEL_NAME}@champion")
print(f"  flavors: {list(loaded_info.flavors)}")
print(f"  signature inputs: {loaded_info.signature.inputs}")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n=== done ===")
print(f"  validation MAE {metrics['mae']:.3f} | R2 {metrics['r2']:.3f}")
print(f"  run:      {TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}")
print(f"  model:    {REGISTERED_MODEL_NAME} v{model_version} (@champion, @challenger)")
print("\nIn the UI, look for:")
print("  • Overview      → params, tags, the dataset card and its digest")
print("  • Model metrics → val_mae_vs_trees as a curve, system metrics alongside")
print("  • Artifacts     → reports/, config/, plots/, tables/, data/, bundle/")
print("  • Models (tab)  → the logged model, its signature, env and metadata")
print("  • the run tree  → three nested cv-fold-* runs under rf-full-tour")
print(f"  • Models (nav)  → {REGISTERED_MODEL_NAME}, its aliases and version tags")
print("\nServe the registered model straight from the registry with:")
print(f"  mlflow models serve -m 'models:/{REGISTERED_MODEL_NAME}@champion' -p 5001")

# Not covered here (deliberately — they need more than one model or a GenAI
# workload): run comparison across families (see mlflow_example.py), MLflow
# Projects (`MLproject` files), `mlflow.pyfunc.PythonModel` custom wrappers,
# hyperparameter sweeps, and LLM tracing / prompt registry.
