# Ride Duration — Experiment Tracking (Session 2)

Session 2 of the MLOps course. It trains a small **ride-duration** regression
model on synthetic data and tracks the run — parameters, metrics, and the fitted
model — with two experiment trackers so you can compare them side by side:
[MLflow](https://mlflow.org/) and [Weights & Biases](https://wandb.ai/). The
model reuses the same distance/passenger relationship from session 1.

Around the training core it also wires up the surrounding MLOps toolchain: a DVC
pipeline, a FastAPI serving image, Ruff + pre-commit code quality, a GitHub
Actions CI/CD workflow, and Terraform for the cloud storage bucket.

**This README is organised in the order you'd actually work through the stack:**

1. [MLflow](#1-mlflow--experiment-tracking--model-registry) — what it is, its important features, track & register the model
2. [Weights & Biases](#2-weights--biases-alternative-tracker) — the alternative tracker, and what **sweeps** are
3. [DVC](#3-dvc--data-versioning--the-important-commands) — version the data, all the important commands
4. [`dvc.yaml`](#4-dvcyaml--a-small-automated-ml-lifecycle) — a small automated ML lifecycle
5. [Terraform](#5-terraform--infrastructure-as-code) — why IaC, all the commands, provision the bucket
6. [CI, CD & CT](#6-what-are-ci-cd--ct) — the concepts, and why Continuous Training is ML-specific
7. [GitHub Actions](#7-github-actions--the-repos-cicd-pipeline) — the sections of a workflow, annotated from `ci_cd.yml`
8. [Integration testing](#8-integration-testing) — what it is and how ours is built

Two supporting sections sit where they're needed:
[Code quality](#code-quality-ruff-black--pre-commit) (Ruff, Black, pre-commit —
after the pipeline sections) and the
[serving image](#serving-the-model-inference-api) (just before CI/CD, since
that's what CI builds and pushes).

## Project structure

```
session_2/
├── mlflow_example.py                 # trains 13 models (linear/RF/XGBoost/MLP), compares runs, registers the winner
├── mlflow_modelregiestry_example.py  # load/promote a registered model via MlflowClient (aliases)
├── mlflow_s3_registry_example.py     # same, against an S3-backed tracking server
├── mlflow_gcs_registry_example.py    # same, against a GCS-backed tracking server
├── wandb_example.py                  # Weights & Biases counterpart to mlflow_example.py
├── serve.py               # FastAPI inference service (/health, /predict)
├── Dockerfile             # multi-stage, code-only image (model mounted at runtime)
├── .dockerignore          # trims the Docker build context
├── pyproject.toml         # Project metadata + Python dependencies
├── docker-compose.yml     # MLflow tracking server, local volume artifacts (from session 1)
├── docker-compose.s3.yml  # MLflow tracking server, artifacts in an S3 bucket
├── docker-compose.gcs.yml # MLflow tracking server, artifacts in a GCS bucket
├── .env.s3.example        # template env for docker-compose.s3.yml (copy to .env)
├── .env.gcs.example       # template env for docker-compose.gcs.yml (copy to .env)
├── dvc.yaml               # DVC pipeline definition (prepare → train → evaluate)
├── dvc.lock               # DVC lock file (recorded stage input/output hashes)
├── .dvc/
│   └── config             # DVC remotes: local (default) + gcs + s3
├── config/
│   └── config.yaml        # pipeline parameters (data + model)
├── data/
│   └── raw/
│       ├── rides.csv      # raw synthetic dataset (pipeline input)
│       └── rides.csv.dvc  # DVC pointer for the raw dataset
├── src/
│   ├── __init__.py
│   ├── train.py           # data gen, train, evaluate + DVC `train` stage
│   ├── prepare.py         # DVC `prepare` stage (raw CSV → train/val parquet)
│   ├── evaluate.py        # DVC `evaluate` stage (model → metrics/scores.json)
│   └── config.py          # YAML config loader
├── tests/
│   ├── test_train.py      # pytest suite for src/train.py
│   └── test_integration.py# end-to-end training/eval integration test
├── infra/                 # Terraform IaC (GCP GCS bucket; AWS S3/ECR kept commented)
│   ├── main.tf            # providers, GCS bucket, outputs
│   ├── variables.tf       # input variables (project_name, gcp_project, gcp_region)
│   └── commands.md        # Terraform command cheatsheet
├── .pre-commit-config.yaml
└── README.md

# The CI/CD workflow lives at the REPO ROOT (one level up), not in session_2/:
mlops_sessions/
├── .github/workflows/
│   └── ci_cd.yml          # lint + test, then build & push the Docker image
└── .pre-commit-config.yaml
```

> `src/train.py` deliberately contains **no tracking calls** so it can be unit
> tested in isolation. It holds the shared model core (`generate_data`,
> `split_data`, `train_model`, `evaluate`) reused by every tracking script; all
> experiment tracking lives in `mlflow_example.py` / `wandb_example.py`.

## Requirements

- Python 3.10+
- Docker (for the MLflow tracking server)
- Terraform 1.x + the `gcloud` CLI (for the `infra/` GCS bucket)
- AWS CLI — optional, only if you point DVC at an S3 remote
- GitHub CLI (`gh`) — optional, for CI/PR workflows

## Installing the toolchain

The commands below target **macOS (Homebrew)**, with notes for Linux/Windows.

### Homebrew (macOS package manager)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Terraform

```bash
# macOS
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Linux (Debian/Ubuntu)
wget -O- https://apt.releases.hashicorp.com/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Windows
choco install terraform        # or: winget install HashiCorp.Terraform

terraform version              # verify
```

### Google Cloud SDK (`gcloud`)

```bash
brew install --cask google-cloud-sdk    # macOS
# Linux:   curl https://sdk.cloud.google.com | bash && exec -l $SHELL
# Windows: winget install Google.CloudSDK

gcloud version                          # verify
```

### AWS CLI (optional — only for an S3 DVC remote)

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# Windows
winget install Amazon.AWSCLI

aws --version                 # verify
aws configure                 # set credentials + default region
```

### Docker

Install **Docker Desktop** (macOS/Windows) from
<https://www.docker.com/products/docker-desktop/>, or Docker Engine on Linux:

```bash
# macOS
brew install --cask docker

# Linux (Debian/Ubuntu)
curl -fsSL https://get.docker.com | sh

docker --version              # verify
```

### GitHub CLI — optional

```bash
brew install gh               # macOS
sudo apt install gh           # Linux (Debian/Ubuntu)
winget install GitHub.cli     # Windows

gh --version
gh auth login                 # authenticate
```

### `act` — run GitHub Actions locally (optional)

Runs workflows from `.github/workflows/` on your machine (requires Docker):

```bash
brew install act              # macOS
# Linux: curl -fsSL https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

act --version
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"            # core deps + pytest + ruff + pre-commit
```

---

## 1. MLflow — experiment tracking & model registry

### What is MLflow?

[MLflow](https://mlflow.org/) is an open-source platform for managing the ML
lifecycle. Without it, experiments end up as hyperparameters in a spreadsheet
and `model_final_v3.pkl` on someone's laptop; with it, every training run is
recorded in one queryable place. Its important features:

- **Experiment tracking** — log the parameters, metrics (including per-step
  curves like a training loss), tags, and artifacts (plots, files) of every
  run, then sort/filter/compare runs in the UI.
- **Model Registry** — a named, versioned store for blessed models
  (`RideDurationModel` v1, v2, …) with moving **aliases** like `@champion`, so
  "which version is production" is decoupled from code.
- **Model packaging ("flavors")** — a standard format (`mlflow.sklearn`,
  `mlflow.xgboost`, …) so any logged model can be reloaded or served the same
  way regardless of the library that trained it.
- **Tracking server + UI** — a central HTTP server backed by a database
  (run/registry metadata) and an artifact store (model files, plots). With
  `--serve-artifacts` it also proxies the artifact storage, so clients never
  need cloud credentials.

**Run the code for this section** (from `session_2/`):

```bash
docker compose up -d mlflow               # 1. start the tracking server → http://localhost:5000
python mlflow_example.py                  # 2. train 13 candidates, register the winner
python mlflow_modelregiestry_example.py   # 3. consume + promote the registered model
```

Each step is explained below.

### Start the tracking server

The MLflow service is carried over from session 1's `docker-compose.yml`:

```bash
docker compose up -d mlflow
```

- MLflow UI: **http://localhost:5000**
- Backend store: SQLite (`mlflow.db`)
- Artifacts: served by the tracking server and persisted in the `mlflow-data`
  volume

Tear down with `docker compose down` (add `-v` to also drop the volume).

#### Getting inside the container

`docker compose up -d mlflow` creates a container named `session_2-mlflow-1`:

```bash
docker compose ps                  # is it up? (add -a to also list it when stopped)
docker compose logs -f mlflow      # follow the server log — every artifact PUT/GET shows up

# Interactive shell inside the container
docker compose exec mlflow bash
docker exec -it session_2-mlflow-1 bash    # equivalent, addressed by container name

# One-off commands, no interactive shell needed
docker compose exec mlflow ls /artifacts
docker exec session_2-mlflow-1 find /artifacts -type d -name plots
```

Inside, the layout is `/artifacts/<experiment_id>/<run_id>/artifacts/plots/*.png`.
That `plots/` prefix comes from `mlflow.log_figure(fig, "plots/…")` — it is a path
*inside the run*, not a folder in this repo, which is why `find . -name plots`
finds nothing here.

Two things worth knowing about this compose file:

- The SQLite backend store is `/mlflow.db` in the **container's own filesystem**,
  not in the `mlflow-data` volume. Any `docker compose down` removes the
  container and the run metadata with it, while the artifact bytes survive in the
  volume — orphaned. Only the artifacts are actually persistent here.
- Port 5000 is contested (a pip-installed `mlflow server`, macOS AirPlay
  Receiver). If it is taken: `MLFLOW_PORT=5001 docker compose up -d mlflow`.

#### Copying the artifacts onto the host

From the volume — works even when the container is stopped:

```bash
docker run --rm -v session_2_mlflow-data:/a -v "$PWD":/out alpine \
  cp -r /a /out/mlflow-artifacts
```

From the running container:

```bash
docker compose cp mlflow:/artifacts ./mlflow-artifacts
```

Either way you get `./mlflow-artifacts/<experiment_id>/<run_id>/artifacts/plots/*.png`
on the host. It is regenerable output — add `mlflow-artifacts/` to `.gitignore`.

To pull one run's figures instead of the whole store, go through the API:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.artifacts.download_artifacts(run_id="<run_id>", artifact_path="plots", dst_path=".")
```

#### Storing artifacts in S3 instead of a local volume

[`docker-compose.s3.yml`](docker-compose.s3.yml) runs the same server but points
`--artifacts-destination` at an S3 bucket instead of the local Docker volume.
Thanks to `--serve-artifacts`, the server **proxies** S3 for you — client code and
`load_model()` calls stay byte-for-byte identical and never need AWS credentials
(only the server does).

```bash
cp .env.s3.example .env                       # fill in bucket + AWS keys
docker compose -f docker-compose.s3.yml up -d # UI still at http://localhost:5000
```

The registry metadata (versions, `@champion`) still lives in a database — here
SQLite bind-mounted to `mlflow-s3.db` on the host so it survives `down` (swap for
Postgres in production; a commented service shows how). Register a version with
`python mlflow_example.py` as usual, then load it back with
[`mlflow_s3_registry_example.py`](mlflow_s3_registry_example.py) — whose code is
unchanged from the local example, which is the whole point.

> The real `.env` holds secrets and is git-ignored; only `.env.s3.example` is
> committed. Set `MLFLOW_S3_ENDPOINT_URL` to target MinIO / R2 / LocalStack for a
> fully local, no-AWS demo. Leave it **empty** for real AWS — the SDK derives the
> endpoint from the region, and the `or None` guard in the example turns the empty
> string into "use the default AWS endpoint."

#### Storing artifacts in GCS instead of a local volume

[`docker-compose.gcs.yml`](docker-compose.gcs.yml) is the Google Cloud Storage twin
of the S3 setup: it points `--artifacts-destination` at
`gs://mlops-session2-iti/mlflow` instead of the local volume. The only real
differences from S3 are the storage SDK (`google-cloud-storage`) and how the server
authenticates — a **service-account key JSON** mounted into the container rather than
env-var access keys. `--serve-artifacts` still proxies GCS, so client code and
`load_model()` calls stay byte-for-byte identical and never need GCP credentials
(only the server does).

```bash
# 1. Drop a service-account key (roles/storage.objectAdmin on the bucket) at:
#      ./gcp-key.json          # gitignored — never committed
# 2. Configure and start:
cp .env.gcs.example .env                        # set bucket + key path
docker compose -f docker-compose.gcs.yml up -d  # UI still at http://localhost:5000
```

The registry metadata (versions, `@champion`) still lives in a database — here
SQLite kept in the bind-mounted `mlflow-gcs-data/` directory so it survives `down`
(swap for Postgres in production; a commented service shows how). Register a version with
`python mlflow_example.py` as usual, then load it back with
[`mlflow_gcs_registry_example.py`](mlflow_gcs_registry_example.py) — whose code is
unchanged from the local example, which is the whole point.

> The service-account key and the real `.env` are git-ignored (`gcp-key.json`,
> `.env`); only `.env.gcs.example` is committed.

### Running MLflow with pip instead of Docker

MLflow is an ordinary Python package, and `pip install -e ".[dev]"` from
[Setup](#setup) already installed it (`mlflow>=2.12`), so the `mlflow` CLI is
on your PATH and Docker is optional — confirm with `mlflow --version`.

The decision that actually matters is not pip-vs-Docker, it is **which backend
store** the server writes run metadata to:

| Option | Command | Run metadata | Artifacts | Note |
|---|---|---|---|---|
| **A. SQLite server** (recommended) | `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` | `./mlflow.db` | `./mlartifacts/`, written by the server | Same behaviour as the compose service |
| **B. SQLite, client-written artifacts** | `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --no-serve-artifacts --host 127.0.0.1 --port 5000` | `./mlflow.db` | `./mlartifacts/`, written by the **client** | Only valid when client and server share a disk |
| **C. No server at all** | *(no process)* `MLFLOW_ALLOW_FILE_STORE=true` + `mlflow.set_tracking_uri("file:./mlruns")` | `./mlruns/` | `./mlruns/…/artifacts/` | Deprecated store — see below |
| **D. UI over an existing store** | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` | reads A/B's `mlflow.db` | — | Browser only, no new runs |

**Option A is the one to use.** Two defaults do the work that
`docker-compose.yml` spells out by hand: `--serve-artifacts` is already on in
MLflow 3.x, and `--artifacts-destination` already defaults to `./mlartifacts`.
So the short command above is equivalent to the container's, and
`mlflow_example.py` needs no change — it already defaults to
`http://localhost:5000`. Keeping artifacts *proxied through* the server is also
what lets the same script later point at S3/GCS with no client-side credentials.

Option B turns the proxy off, so the client writes artifact files itself. Fine
locally; broken the moment the server moves to another host or a bucket.

Option C runs no process at all, but **the plain file store is deprecated in
MLflow 3.x** and now hard-fails:

```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
maintenance mode and will not receive further updates. Please migrate to a
database backend (e.g., 'sqlite:///mlflow.db') ...
```

`--backend-store-uri` still *defaults* to `./mlruns`, so a bare `mlflow server`
or `mlflow ui` walks straight into that error. Either pass
`sqlite:///mlflow.db` (options A/D) or opt out with `MLFLOW_ALLOW_FILE_STORE=true`.

Trade-offs against the Docker route:

- **pip is lighter** — no daemon, no image pull, and `mlflow.db` plus
  `./mlartifacts/` sit right in the working directory, so you can open a plot
  PNG without copying anything out of a volume first.
- **pip does not pin the server version.** Compose pins
  `ghcr.io/mlflow/mlflow:v3.14.0`; pip gives whatever `mlflow>=2.12` resolved to
  in your venv, so two people can silently run different servers.
- **The server is tied to your shell** — it dies with the terminal unless you
  background it (`nohup mlflow server … > mlflow.log 2>&1 &`), whereas
  `docker compose up -d` survives and is one `down` away from gone.
- **Client and server share one venv,** so bumping a training dependency also
  upgrades the server. Docker keeps the two apart.
- **First start is slow** (tens of seconds): MLflow imports a lot, then forks
  worker processes that import it again. It is not hung — wait for
  `Listening at: http://127.0.0.1:5000`, or poll `curl -s localhost:5000/health`.

Tear it down by stopping the process (`Ctrl-C`, or `pkill -f "mlflow server"`),
then delete `mlflow.db` and `mlartifacts/` for a clean slate.

### Run the training script

```bash
python mlflow_example.py
```

> **macOS:** XGBoost needs the OpenMP runtime — `brew install libomp` once.

[`mlflow_example.py`](mlflow_example.py) is deliberately *not* a "train one model
and log it" demo. It trains **13 candidate models across 4 families** on one
shared dataset, logs each as its own MLflow run in a single experiment, ranks
them, and promotes only the winner into the Model Registry. One run is a
logbook entry; 13 runs in one experiment are a **leaderboard** — which is the
point at which experiment tracking starts paying for itself.

#### Step 0 — Connect to the server and choose an experiment

```python
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "RideDurationModel"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("ride-duration-model")
```

- `set_tracking_uri` decides **where** runs go. Everything after this line is
  an HTTP call to the tracking server — nothing is written to the local
  filesystem. Export `MLFLOW_TRACKING_URI` to log to a remote/shared server
  instead, with no code change.
- `set_experiment` picks the **namespace for runs**, creating it on first use.
  An *experiment* groups runs; the *registry* (`RideDurationModel`) versions the
  models promoted out of them — two distinct concepts that the script keeps
  deliberately separate.
- The script also forces matplotlib's non-interactive `Agg` backend *before*
  importing `pyplot`, because it only ever hands figures to MLflow and must
  never try to open a GUI window (which would hang in Docker/CI).

#### Step 1 — Build one dataset, shared by every candidate

```python
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)
```

Both helpers come from [`src/train.py`](src/train.py), which contains **zero
MLflow calls** so it stays unit-testable in isolation. The data is synthetic and
reproduces session 1's heuristic:

```
duration_min = distance_km / 0.5  +  passengers * 0.5  +  N(0, 1)
```

with `distance_km ~ U(0.5, 30)` and `passengers ~ {1,2,3,4}` — features
`["distance_km", "passengers"]`, target `duration_min`, giving a 1 600 / 400
train/validation split.

The split happens **once, outside the loop**, with a fixed seed. Every candidate
is scored on the exact same 400 validation rows — without that, the leaderboard
would be comparing models *and* their luck of the draw, and the ranking would be
meaningless.

#### Step 2 — Declare the model zoo

`MODEL_ZOO` is a list of `(run_name, family, params, build_fn)` tuples. Keeping
the candidates as **data** rather than 13 copy-pasted blocks means the `params`
dict passed to `mlflow.log_params()` is the *same object* used to build the
estimator — so what the UI shows is provably what was trained.

| Run name | Family | Hyperparameters | Why it's in the zoo |
| --- | --- | --- | --- |
| `linear-ols` | `linear` | `alpha=0.0` | Plain least squares — the baseline |
| `linear-ridge-a1` | `linear` | `alpha=1.0` | Mild L2 regularisation |
| `linear-ridge-a100` | `linear` | `alpha=100.0` | Heavy L2 — shows over-regularising hurts |
| `rf-small` | `random_forest` | `n_estimators=50, max_depth=4` | Under-powered forest |
| `rf-baseline` | `random_forest` | `n_estimators=100, max_depth=6` | Sensible default |
| `rf-big` | `random_forest` | `n_estimators=300, max_depth=10` | More capacity, more train time |
| `xgb-shallow-fast` | `xgboost` | `n_estimators=100, max_depth=2, lr=0.3` | Few shallow trees, big steps |
| `xgb-baseline` | `xgboost` | `n_estimators=200, max_depth=3, lr=0.1` | Typical starting point |
| `xgb-deep` | `xgboost` | `n_estimators=200, max_depth=6, lr=0.1` | Deeper trees |
| `xgb-slow-learner` | `xgboost` | `n_estimators=400, max_depth=3, lr=0.03` | Many small steps |
| `mlp-tiny` | `mlp` | `hidden=(16,), lr=0.01` | Minimal network |
| `mlp-wide` | `mlp` | `hidden=(64, 32), lr=0.001` | Wider, slower learning |
| `mlp-deep` | `mlp` | `hidden=(64, 64, 32), lr=0.001` | Deepest network |

Two modelling details are encoded in the builders:

- `make_linear` returns `LinearRegression()` when `alpha == 0.0` and
  `Ridge(alpha=…)` otherwise — so "no regularisation" is expressible as a
  *logged parameter value* rather than a separate code path.
- `make_mlp` wraps `MLPRegressor` in a `Pipeline` with a `StandardScaler`.
  Neural nets need standardised inputs (`distance_km` spans 0.5–30 while
  `passengers` spans 1–4), whereas tree models are scale-invariant and need no
  such treatment. Scaling *inside* the pipeline also means the scaler is
  serialised with the model, so inference can't forget to apply it.

#### Step 3 — One MLflow run per candidate

```python
for run_name, family, params, build in MODEL_ZOO:
    with mlflow.start_run(run_name=run_name) as run:
        ...
```

`start_run` as a context manager guarantees the run is closed with the right
terminal status even if training raises. Inside each run, in order:

**a. Log the configuration first.**

```python
mlflow.log_params(params)
mlflow.set_tags({
    "model_family": family,     # → search: tags.model_family = 'mlp'
    "dataset": "synthetic-v1",
    "stage": "experiment",
})
```

Params become sortable/filterable columns in the run table. Tags are the
*grouping* dimension: `model_family` powers the search bar, `dataset` records
which data generation this ranking is valid for, and `stage` marks these runs as
exploratory (as opposed to a release candidate).

**b. Train, and time it.**

```python
t0 = time.perf_counter()
model = build(params)
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - t0
```

Wall-clock cost is treated as a first-class metric. A model that is 1 % more
accurate but 50× slower to train is often the wrong choice, and you can only see
that trade-off if training time sits in the same table as accuracy.

**c. Log the shared metric set.**

```python
metrics = evaluate(model, X_val, y_val)   # {"rmse", "mae", "r2"}
mlflow.log_metrics(metrics)
mlflow.log_metric("train_seconds", train_seconds)
```

Every run logs the **same metric names**, computed on the **same validation
split**, by the same `evaluate()` function. That consistency is what makes
clicking a column header in the UI a legitimate ranking rather than a
coincidence.

**d. Log artifacts** — family-specific diagnostics plus a fit plot for every
run (detailed in step 4 below).

**e. Log the fitted model as a run artifact.**

```python
if family == "xgboost":
    mlflow.xgboost.log_model(model, name="model", input_example=X_train[:5])
else:
    trusted = (["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
               if family == "mlp" else None)
    mlflow.sklearn.log_model(model, name="model", input_example=X_train[:5],
                             skops_trusted_types=trusted)
```

Three things are happening here:

- **Flavors.** MLflow serialises each framework with its own flavor
  (`mlflow.sklearn`, `mlflow.xgboost`). The *write* path differs per library,
  but every flavor can be read back uniformly through `mlflow.pyfunc` — which is
  what makes a single serving container able to host any of them.
- **`input_example=X_train[:5]`** ships five real rows with the model. MLflow
  infers the **signature** (input shape/dtypes) from it, so a later
  `load_model()` can validate incoming payloads, and the UI can show a
  ready-made example request.
- **`skops_trusted_types`.** MLflow ≥ 3 serialises sklearn models with
  [`skops`](https://skops.readthedocs.io/), which audits every class in the
  payload and refuses ones it doesn't recognise. The MLP carries Adam optimizer
  state, so that one class is trusted explicitly — narrower and safer than
  falling back to raw pickle.

Note that **nothing is registered inside the loop.** All 13 models are logged as
run artifacts; promotion to the registry is a separate, deliberate decision made
after the results are in.

#### Step 4 — Family-specific diagnostics

Runs in one experiment do **not** have to produce identical artifacts. MLflow
stores whatever each run logs, so each family logs the introspection it actually
supports:

| Family | Logged by `log_family_diagnostics()` | Question it answers |
| --- | --- | --- |
| `linear` | `plots/coefficients.png` — bar chart of `model.coef_` | What does each feature contribute? |
| `random_forest`, `xgboost` | `plots/feature_importances.png` — sorted `feature_importances_` | Which feature drives the splits? |
| `mlp` | `train_loss` logged per epoch from `loss_curve_` | Did the network actually converge? |

The MLP case is the interesting one:

```python
for epoch, loss in enumerate(model.named_steps["mlp"].loss_curve_):
    mlflow.log_metric("train_loss", loss, step=epoch)
```

Logging the *same metric name* repeatedly with an increasing `step` turns it
into a **series** instead of a scalar, which MLflow renders as an interactive
line chart. This is the same mechanism you'd use to stream a live training curve
epoch by epoch.

On top of that, `log_fit_quality()` runs for **every** candidate and logs
`plots/predicted_vs_actual.png` — validation predictions scattered against the
truth with a red `y = x` reference line. Because the filename is identical
across runs, you can flip between runs in the artifact browser and compare fits
visually at the same path.

#### Step 5 — Rank the candidates

```python
results.sort(key=lambda r: r[0])            # r[0] is val MAE, lower is better
best_mae, best_name, best_family, best_run_id, best_metrics = results[0]
```

The script keeps its own `results` list and prints the leaderboard to the
terminal — the same ordering you get in the UI by clicking the `mae` column.
Sorting in-process is what lets the *next* step run automatically: the winner's
`run_id` is needed to build the model URI.

#### Step 6 — Register only the winner

```python
registered = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=REGISTERED_MODEL_NAME,
)
version = registered.version
```

`runs:/<run_id>/model` points at the artifact logged inside the winning run —
the model is *promoted*, not retrained, so the registered bytes are exactly the
ones that produced the winning score. Each call creates a new **version**
(v1, v2, v3…) under the name `RideDurationModel`.

The new version is then annotated through `MlflowClient` — these calls target the
**registry**, not the run:

```python
client.update_model_version(name=…, version=version, description=…)  # why it won
client.set_model_version_tag(…, "validated",    "true")
client.set_model_version_tag(…, "source_run",   best_run_id)         # traceability
client.set_model_version_tag(…, "model_family", best_family)
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
```

The `source_run` tag closes the loop: from a production model version you can
jump straight back to the run that produced it, with its params, metrics, and
plots intact.

The **alias** is the key idea. `@champion` is a moving pointer to a specific
version — aliases replaced the old `Staging`/`Production` stages in MLflow 2.9+.
Downstream code loads `models:/RideDurationModel@champion` and always gets the
blessed version, so "which version is production" becomes a registry setting
rather than a hard-coded number in application code.

#### Step 7 — Print where to look

The script closes by printing the winner, the new version number, and a short
list of things to try in the UI, ending with the URI a consumer would use:
`models:/RideDurationModel@champion`.

#### What ends up in MLflow

| MLflow object | Written by this script |
| --- | --- |
| Experiment | `ride-duration-model` |
| Runs | 13 — one per zoo entry, named `linear-ols`, `rf-big`, … |
| Params | Each candidate's hyperparameters (`alpha`, `n_estimators`, `max_depth`, `learning_rate`, `hidden_layer_sizes`, …) |
| Tags | `model_family`, `dataset=synthetic-v1`, `stage=experiment` |
| Metrics | `rmse`, `mae`, `r2`, `train_seconds` on every run; `train_loss` as a per-epoch series on MLP runs |
| Artifacts | `model/` (the serialised model + signature) and `plots/*.png` on every run |
| Registered model | `RideDurationModel` — one new version per script execution |
| Alias | `@champion` → the version just registered |

Because a new version is created on every execution, re-running the script is
how you get v2, v3, … — and `@champion` moves with it.

#### Expected output

```
=== Leaderboard (val MAE, lower is better) ===
   1. linear-ols           [linear       ] MAE=0.790  RMSE=0.986  R2=0.995
   2. linear-ridge-a1      [linear       ] MAE=0.790  RMSE=0.986  R2=0.995
   ...
  13. rf-small             [random_forest] MAE=1.526  RMSE=1.916  R2=0.981

Winner: linear-ols (linear)  |  MAE: 0.790
Registered 'RideDurationModel' version 1 (alias: @champion)
```

(Spoiler: the linear models usually win — the synthetic data *is* linear, so
the extra capacity of XGBoost/MLP buys nothing. That "fancier isn't better
here" conclusion is exactly what a run-comparison table makes obvious.)

#### Explore the results in the UI

Open **http://localhost:5000** → experiment `ride-duration-model`, then:

- **Sort** the run table by the `mae` column → instant leaderboard.
- **Tick several runs → Compare** → parallel-coordinates & scatter charts of
  hyperparameters vs. metrics.
- **Filter** with the search bar: `tags.model_family = 'xgboost'`.
- Open any `mlp-*` run → *Model metrics* → the `train_loss` convergence curve.
- Open any run → *Artifacts* → `plots/predicted_vs_actual.png`.
- **Models** → `RideDurationModel` → the registered version, its description,
  tags, and the `@champion` alias.

> The script reads the tracking URI from the `MLFLOW_TRACKING_URI` env var,
> defaulting to `http://localhost:5000`. Point it at a remote server by
> exporting that variable.

### Consume the registered model

Once a version exists, [`mlflow_modelregiestry_example.py`](mlflow_modelregiestry_example.py)
is the consumer side: it lists the registered versions, promotes one by moving
an alias (`@champion` → `@production`), loads it via
`models:/RideDurationModel@production`, and runs a few predictions.

```bash
python mlflow_modelregiestry_example.py
```

**How a teammate loads your model.** The registry lives in the *server's*
database, so anyone who can reach the same tracking server can load a model by
alias or version — with **no cloud credentials** (the server proxies the
artifacts):

```bash
export MLFLOW_TRACKING_URI=https://mlflow.your-company.internal   # the shared server
```

```python
import mlflow

# by alias (recommended — "which version is prod" is decoupled from code):
model = mlflow.sklearn.load_model("models:/RideDurationModel@champion")

# or by explicit version:
model = mlflow.sklearn.load_model("models:/RideDurationModel/3")
```

> A model registered against `http://localhost:5000` is only reachable on *your*
> machine. To share it, register against a shared server (GCP), or expose your
> local server on `0.0.0.0` / a tunnel. See
> [`mlflow_gcs_registry_example.py`](mlflow_gcs_registry_example.py) for the
> GCS-backed consumer.

---

## 2. Weights & Biases (alternative tracker)

[Weights & Biases](https://wandb.ai/) (W&B) is a hosted experiment-tracking
platform: it does the same job as MLflow's tracking + registry (runs, config,
metrics, artifacts, dashboards) but as a managed cloud service with richer
collaborative dashboards. Its headline extra is **Sweeps** — built-in
hyperparameter optimisation.

**What is a sweep?** A sweep is a W&B-orchestrated hyperparameter search. You
declare a search space, a strategy (`grid`, `random`, or `bayes`), and the
metric to optimise; the W&B server then hands out trial configurations one at
a time, and one or more **agents** running on your machine execute a training
function for each trial. Every trial is recorded as its own run, so the sweep
dashboard shows which hyperparameters actually matter
(parallel-coordinates and importance plots).

**Run the code for this section** (from `session_2/`):

```bash
pip install -e ".[wandb]"     # wandb + matplotlib
wandb login                   # required for the sweep (free account + API key)
python wandb_example.py       # baseline run, then a 20-trial Bayesian sweep
```

No account? `export WANDB_MODE=offline` runs everything locally **except the
sweep** — sweeps are orchestrated by the W&B server, so the script detects
offline mode and skips it with a message. The script prints the run and sweep
URLs so you can watch trials arrive live.

[`wandb_example.py`](wandb_example.py) mirrors the single-model (RandomForest)
version of `mlflow_example.py` **feature for feature** against
[W&B](https://wandb.ai/), so you can see how the two trackers map onto each
other:

| MLflow                              | Weights & Biases                                |
|-------------------------------------|-------------------------------------------------|
| `mlflow.log_params(...)`            | `wandb.init(config=...)`                         |
| `mlflow.set_tags(...)`              | `wandb.init(tags=..., notes=...)`                |
| `mlflow.log_metrics(...)`           | `wandb.log({...})`                               |
| `log_metric(name, v, step=...)`     | `wandb.log({...}, step=...)` (line charts)       |
| `mlflow.log_figure(fig, name)`      | `wandb.log({name: wandb.Image(fig)})`            |
| `log_model` + `registered_model`    | `wandb.Artifact(...)` + aliases (`"champion"`)   |
| `client.set_..._alias` / tags       | artifact aliases + `metadata`                    |

### Install & authenticate

```bash
pip install -e ".[wandb]"          # wandb + matplotlib

# Pick an auth mode first:
wandb login                        # log to wandb.ai (free account + API key), or
export WANDB_MODE=offline          # no account/network → runs written to ./wandb/

python wandb_example.py
```

### What the script logs (and how)

Everything happens inside a run opened with `wandb.init(...)` and closed with
`run.finish()`. The pieces:

- **Config & metadata** — hyperparameters go into `config`, free-form labels into
  `tags`/`notes`; all are searchable in the UI:
  ```python
  run = wandb.init(
      project="ride-duration", name="rf-baseline",
      config={"n_estimators": 100, "max_depth": 6},
      tags=["random_forest", "synthetic-v1"], notes="baseline on synthetic-v1",
  )
  ```
- **Scalar metrics** — one `wandb.log({...})` call records the run's headline
  numbers:
  ```python
  wandb.log({"rmse": rmse, "mae": mae, "r2": r2, "val_size": len(y_val)})
  ```
- **Learning curve (stepped metrics)** — logging the *same* key repeatedly with an
  increasing `step` makes W&B plot it as an interactive line chart. Prefixing the
  keys (`lc/…`) groups them into their own panel:
  ```python
  for n in (10, 25, 50, 100, 200, 400):
      m = evaluate(train_model(X_train, y_train, {**params, "n_estimators": n}), X_val, y_val)
      wandb.log({"lc/val_rmse": m["rmse"], "lc/val_mae": m["mae"]}, step=n)
  ```
- **Diagnostic figures** — hand a matplotlib `Figure` to `wandb.Image(...)` and it
  renders inline in the run's Media panel (predicted-vs-actual, residual
  histogram, feature importances):
  ```python
  wandb.log({"plots/residuals": wandb.Image(fig_res)})
  ```
- **A W&B Table** — sortable/filterable raw rows behind the plots, queryable across
  runs:
  ```python
  table = wandb.Table(columns=["distance_km", "passengers", "actual", "predicted"])
  table.add_data(10.0, 2, 21.3, 21.1)
  wandb.log({"val_predictions": table})
  ```
- **Model as a versioned artifact ("register")** — W&B has no separate registry
  API; you log the model as an `Artifact` and attach a moving `champion` alias.
  Downstream code fetches `run.use_artifact("ride-duration-model:champion")` and
  always gets the blessed version without hard-coding a number:
  ```python
  art = wandb.Artifact("ride-duration-model", type="model",
                       metadata={**params, **metrics, "validated": True})
  art.add_file("model.pkl")
  run.log_artifact(art, aliases=["champion", "synthetic-v1"])
  ```
- **Hyperparameter sweep (Bayesian)** — declare a search space and let a W&B agent
  drive trials, minimising `mae`:
  ```python
  sweep_id = wandb.sweep({"method": "bayes",
                          "metric": {"name": "mae", "goal": "minimize"},
                          "parameters": {"n_estimators": {"values": [50, 100, 200]},
                                         "max_depth": {"min": 3, "max": 10}}},
                         project="ride-duration")
  wandb.agent(sweep_id, function=train_fn, count=20)   # 20 trials
  ```

> In offline mode everything runs locally **except the sweep** (sweeps are
> orchestrated by the W&B server and need a login); the script skips it
> gracefully. Push saved offline runs later with `wandb sync wandb/offline-run-*`.

---

## 3. DVC — data versioning & the important commands

MLflow/W&B version **experiments and models**; [DVC](https://dvc.org/) versions the
**data (and any large artifact) that feeds them**. It keeps a tiny text pointer
(`*.dvc`) in git while the real bytes live in a remote (S3, GCS, a local disk…),
so `git checkout <tag> && dvc checkout` reproduces the *exact* data a commit was
built on.

> **This repo already has DVC initialised** with a GCS remote (see
> [`.dvc/config`](.dvc/config): `gs://mlops-session2-iti/dvc-store`). The
> `git init && dvc init` step below is only for bootstrapping a *fresh* repo.
> The `s3://…` URLs are the generic pattern — swap in your own S3 or GCS bucket.

**Run the code for this section** (from `session_2/` — the repo is already
DVC-initialised, so this is all you need to get the data):

```bash
pip install -e ".[dvc]"       # DVC + remote support
dvc pull                      # fetch the tracked data/model matching this commit
dvc status                    # check workspace ↔ cache ↔ remote sync
```

### The important commands at a glance

| Command | What it does |
|---------|--------------|
| `dvc init` | Initialise DVC in a git repo (creates `.dvc/`) |
| `dvc add <file>` | Track a large file: writes a tiny `.dvc` pointer, git-ignores the bytes |
| `dvc remote add -d <name> <url>` | Register a remote (S3/GCS/local disk); `-d` makes it the default |
| `dvc remote modify <name> <key> <value>` | Configure a remote (region, `credentialpath`, …) |
| `dvc remote list` | Show configured remotes and their URLs |
| `dvc push` / `dvc push -r <remote>` | Upload cached data to the default (or a specific) remote |
| `dvc pull` | Download the data matching the checked-out git commit |
| `dvc checkout` | Sync files on disk to the current `.dvc` / `dvc.lock` pointers |
| `dvc status` | Show what's out of sync between workspace, cache, and remote |
| `dvc diff <rev_a> <rev_b>` | Compare tracked data between two commits/tags |
| `dvc repro` | Run the `dvc.yaml` pipeline — only stale stages ([section 4](#4-dvcyaml--a-small-automated-ml-lifecycle)) |
| `dvc dag` | Visualise the pipeline stage graph |
| `dvc metrics show` | Print the metrics files declared in `dvc.yaml` |

The rest of this section walks through each command in context, in the order
you'd actually use them.

### Step 1 — Initialise DVC in a repo

```bash
# ── Step 1: Initialize DVC in your repo ──────────────
git init && dvc init
git add .dvc/ && git commit -m "init: add dvc"
```

`dvc init` creates the `.dvc/` directory (config + cache metadata). Commit it so
collaborators inherit the same DVC setup.

### Step 2 — Track a large data file

```bash
# ── Step 2: Track a large data file ───────────────────
dvc add data/raw/rides.csv
# Creates: data/raw/rides.csv.dvc  (tiny pointer file holding the content hash)
# Adds:    data/raw/rides.csv to .gitignore automatically (git never sees the bytes)
git add data/raw/rides.csv.dvc data/raw/.gitignore
git commit -m "data: track rides.csv with DVC"
```

Git now tracks only the small `.dvc` pointer; the actual CSV is stored in DVC's
local cache and (after `dvc push`) in the remote.

> A file cannot be tracked by both Git and DVC. If `dvc add` reports the file is
> "already tracked by SCM", stop Git tracking it first:
> `git rm -r --cached data/raw/rides.csv`.

### Step 3 — Define a remote

Remotes are defined in [`.dvc/config`](.dvc/config). `-d` marks a remote as the
**default** target for `dvc push`/`dvc pull`.

```bash
# ── Step 3a: an S3 bucket (generic pattern) ───────────
dvc remote add -d s3remote s3://mlops-dvc-artifacts-ride-duration/data
dvc remote modify s3remote region us-east-1
git add .dvc/config && git commit -m "dvc: add S3 remote"

# ── Step 3b: this repo's actual remote is GCS ─────────
dvc remote add -d gcs gs://mlops-session2-iti/dvc-store
dvc remote modify gcs credentialpath ../gcp-key.json   # SA key, gitignored
git add .dvc/config && git commit -m "dvc: add GCS remote"
```

S3 access keys are secrets, so keep them out of the committed config — put them in
the git-ignored `.dvc/config.local`:

```bash
dvc remote modify --local s3 access_key_id     YOUR_KEY
dvc remote modify --local s3 secret_access_key YOUR_SECRET
```

### Step 4 — Push data to the remote

```bash
# ── Step 4: Push data to the remote ───────────────────
dvc push                      # upload all tracked data to the default remote
dvc push -r gcs               # …or target a specific remote
dvc push data/processed/      # …or push only a specific folder
git push                      # push the .dvc pointers to GitHub
```

Always pair `dvc push` (bytes → remote) with `git push` (pointers → GitHub) so
the two stay in sync.

### Step 5 — Pull data on another machine (or in CI)

```bash
# ── Step 5: On any other machine — pull the data ──────
git clone https://github.com/you/ride-duration
cd ride-duration
dvc pull                      # downloads the data matching the checked-out commit
# → you now have byte-identical data to what was committed
```

### Versioning: tie data to a git tag

Because the data hash lives in a git-tracked pointer, a **git tag captures a data
version too** — checking out the tag later restores that exact dataset.

```bash
# v1.0: 500k rows of training data
git tag -a v1.0 -m "baseline model — 500k rows"
git push origin v1.0

# Update dataset to 1M rows
dvc add data/raw/rides.csv           # new content hash written into the .dvc file
git add data/raw/rides.csv.dvc
git commit -m "data: expand to 1M rows"
git tag -a v2.0 -m "retrained — 1M rows"
git push origin v2.0 && dvc push
```

### Switch between data versions instantly

```bash
git checkout v1.0     # code + .dvc pointers go back to v1
dvc checkout          # data files on disk are swapped to match v1

git checkout v2.0
dvc checkout          # data files swapped to match v2
```

`git checkout` moves the *pointers*; `dvc checkout` reconciles the *files* on
disk to whatever those pointers reference (pulling from cache/remote as needed).

### Multiple remotes for different environments

```bash
dvc remote add local_cache /mnt/fast-ssd/dvc-cache
dvc remote add s3prod      s3://prod-bucket/dvc
dvc remote add s3staging   s3://staging-bucket/dvc

# Push / pull to a specific remote with -r
dvc push -r s3staging     # push to staging only
dvc push -r s3prod        # push to production
dvc pull -r local_cache   # fast local pull during dev
```

### Inspect what DVC is tracking

```bash
dvc status                # what's out of sync between workspace, cache, and remote?
dvc diff v1.0 v2.0        # compare tracked data between two git tags
dvc remote list           # show configured remotes and their URLs
```

### Pushing data & model artifacts to GCS (end-to-end)

This sends **tracked data** (via DVC) and **registered-model artifacts** (via the
MLflow server) to `gs://mlops-session2-iti`, which has two prefixes:

```
gs://mlops-session2-iti/
├── dvc-store/   ← DVC data lands here   (dvc push -r gcs)
└── mlflow/      ← MLflow artifacts here (the tracking server writes these)
```

The client/training code never changes and never needs GCP credentials — only the
DVC CLI and the MLflow **server** authenticate to the bucket, both reading the same
service-account key.

**Step 1 — Get a service-account key** with `roles/storage.objectAdmin` on the
bucket, saved as `./gcp-key.json` (git-ignored). Either via **Cloud Console** (IAM
& Admin → Service Accounts → Keys → Add key → JSON) or the CLI:

```bash
gcloud iam service-accounts keys create ./gcp-key.json \
  --iam-account=YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com
```

**Step 2 — Install the GCS extra & push data:**

```bash
pip install -e ".[gcs]"           # adds dvc[gs]
dvc push -r gcs                   # uploads data → gs://…/dvc-store/
dvc pull -r gcs                   # (later, on another machine) pulls it back
dvc remote default gcs            # make gcs the default so a bare `dvc push` targets it
```

**Step 3 — Push model artifacts via the GCS-backed MLflow server:**

```bash
cp .env.gcs.example .env                        # bucket + key path (defaults fit)
docker compose -f docker-compose.gcs.yml up -d  # UI at http://localhost:5000
python mlflow_example.py                         # trains + registers → gs://…/mlflow/
```

**Step 4 — Verify it landed:**

```bash
python mlflow_gcs_registry_example.py            # loads models:/…@champion back
gcloud storage ls -r gs://mlops-session2-iti/dvc-store/
gcloud storage ls -r gs://mlops-session2-iti/mlflow/
```

> **Never commit** `gcp-key.json` or the real `.env` — both are git-ignored. Only
> `.env.gcs.example` and the `credentialpath` *pointer* in `.dvc/config` are
> committed (a path is not a secret; the key file it points to is). As an
> alternative to a key file, authenticate with
> `gcloud auth application-default login` and drop `credentialpath`.

---

## 4. `dvc.yaml` — a small automated ML lifecycle

Tracking files with `dvc add` is only half of DVC. [`dvc.yaml`](dvc.yaml) is
the other half: it turns the loose scripts in `src/` into a small **automated,
reproducible ML lifecycle** — a pipeline DVC can run, cache, and re-run
end-to-end with one command. Ours has three stages:

| Stage      | Command                  | Inputs                                     | Outputs                              |
|------------|--------------------------|--------------------------------------------|--------------------------------------|
| `prepare`  | `python src/prepare.py`  | `data/raw/rides.csv`, `config/config.yaml` | `data/processed/{train,val}.parquet` |
| `train`    | `python src/train.py`    | `train.parquet`, `model.*` params          | `models/rf_model.pkl`                |
| `evaluate` | `python src/evaluate.py` | `rf_model.pkl`, `val.parquet`              | `metrics/scores.json`                |

### Anatomy of a stage

Every stage in the file follows the same shape — here is our `train` stage:

```yaml
train:
  cmd: python src/train.py      # how to run the stage
  deps:                         # inputs — content-hashed by DVC
    - src/train.py
    - src/config.py
    - data/processed/train.parquet
  params:                       # individual config values, tracked separately
    - config/config.yaml:
        - model.n_estimators
        - model.max_depth
  outs:                         # outputs — cached, hashed, git-ignored
    - models/rf_model.pkl
```

- **`cmd`** — the shell command DVC runs for the stage.
- **`deps`** — files the stage reads. If any hash changes, the stage is stale.
- **`params`** — fine-grained deps on *values inside* a config file: editing
  `model.n_estimators` in [`config/config.yaml`](config/config.yaml)
  invalidates `train`, but editing an unrelated key does not.
- **`outs` / `metrics`** — files the stage produces. `outs` are cached and
  pushed like `dvc add`-ed data; `metrics` (our `metrics/scores.json`, with
  `cache: false`) stay in git so `dvc metrics show` can read them.
- **Chaining** — `evaluate` lists `models/rf_model.pkl` (an `out` of `train`)
  as a `dep`, which is what wires the stages into the
  `prepare → train → evaluate` DAG.

Every run records the exact input/output hashes into [`dvc.lock`](dvc.lock) —
commit it, so anyone who checks out the repo can tell whether the pipeline is
up to date and reproduce it byte-for-byte.

### Why this is "a small automated lifecycle"

`dvc repro` re-runs **only what changed**. Touch nothing → everything is a
cache hit. Change a training hyperparameter → `prepare` is skipped, `train`
and `evaluate` re-run. Swap in a new `data/raw/rides.csv` → all three re-run.
That is the data→train→evaluate loop of an ML lifecycle, automated by
dependency tracking instead of a human remembering which scripts to re-run —
and it's the building block CT (Continuous Training,
[section 6](#6-what-are-ci-cd--ct)) automates further.

**Run the code for this section** (from `session_2/`):

```bash
dvc repro              # run prepare → train → evaluate (only stale stages)
dvc status             # which stages are stale, and why
dvc dag                # visualise the stage graph
dvc metrics show       # print metrics/scores.json

# See the automation kick in:
#   edit config/config.yaml (e.g. model.n_estimators: 300), then:
dvc repro              # prepare is a cache hit; only train + evaluate re-run
```

---

## Code quality (Ruff, Black & pre-commit)

### Ruff — linter + formatter

[Ruff](https://docs.astral.sh/ruff/) is an extremely fast Python **linter and
formatter** (written in Rust) that replaces the older Flake8 + **Black** + isort
stack with a single tool. It runs as two separate concerns:

```bash
ruff check src/ tests/           # the LINTER
ruff format --check src/ tests/  # the FORMATTER (verify-only)
```

**The main commands (the ones you'll use every day):**

```bash
ruff check .          # lint the whole project (report issues)
ruff check --fix .    # lint AND auto-fix everything it safely can
ruff format .         # format every file in place
ruff format --check . # verify formatting only, don't write (what CI runs)
```

| Command | Does |
|---------|------|
| `ruff check .` | Report lint problems across the project |
| `ruff check --fix .` | Report **and auto-fix** the safe ones |
| `ruff format .` | Reformat all files to the canonical style |
| `ruff format --check .` | Fail if anything isn't already formatted (CI mode) |

> Everyday loop before a commit: **`ruff check --fix . && ruff format .`** — fix
> lint issues, then normalise layout. (Both also run automatically via
> [pre-commit](#pre-commit-hooks).) The exhaustive flag list is below.

**What is "linting"?** Linting is static analysis — a tool reads your source code
*without running it* and flags likely bugs, suspicious constructs, and style-guide
violations. It catches problems early, before they reach review or production.

The two commands look similar but check completely different things:

| | `ruff check` | `ruff format --check` |
|---|--------------|-----------------------|
| Role | **Linter** | **Formatter** (verify-only) |
| Concerned with | *What the code does* | *How the code is arranged* |
| Catches | Unused imports/variables, undefined names, mutable default args, unreachable code, PEP 8 violations, common bugs | Whitespace, indentation, quote style, line length, trailing commas not matching the canonical style |
| In CI, fails when | Code has lint errors | Code isn't already formatted |
| Fix it with | `ruff check --fix` | `ruff format` (drops `--check` → rewrites files) |

`--check` matters: `ruff format` on its own **rewrites** your files, but
`ruff format --check` **changes nothing** — it only verifies and exits non-zero if
not formatted. That's why CI uses `--check`: it enforces formatting without
modifying the checkout.

**Most important Ruff commands:**

```bash
# ── Linting (code quality) ────────────────────────────────────────────
ruff check src/ tests/           # report lint issues (what CI runs)
ruff check .                     # lint the whole project
ruff check --fix .               # auto-fix everything it safely can
ruff check --fix --unsafe-fixes .# also apply fixes that may change behaviour
ruff check --watch .             # re-lint continuously as you edit
ruff check --select I --fix .    # fix only a rule group (I = import sorting)
ruff check --statistics .        # count violations grouped by rule
ruff check --add-noqa .          # insert `# noqa` comments on existing issues

# ── Formatting (layout) ───────────────────────────────────────────────
ruff format src/ tests/          # reformat files in place
ruff format .                    # format the whole project
ruff format --check .            # verify only, don't write (what CI runs)
ruff format --diff .             # show the changes it *would* make

# ── Housekeeping ──────────────────────────────────────────────────────
ruff --version                   # print the installed version
ruff rule F401                   # explain a specific rule (F401 = unused import)
ruff clean                       # clear Ruff's cache
```

Typical local loop before a commit: **`ruff check --fix .`** then
**`ruff format .`** — the first fixes lint issues, the second normalises layout.
Suppress a single unavoidable warning inline with `# noqa: <RULE>`
(e.g. `import os  # noqa: F401`).

### A note on Black

This repo uses **`ruff format`, which is a drop-in replacement for Black** — same
formatting style, same defaults, far faster, and it shares config with the linter.
So there is no separate Black step. If you specifically prefer the Black tool, it's
compatible and you can run it standalone:

```bash
pip install black
black src/ tests/                # reformat in place (equivalent to `ruff format`)
black --check --diff src/ tests/ # verify only + show the diff (like CI)
```

Pick one formatter, not both — running Black and `ruff format` together just does
the same work twice.

### Pre-commit hooks

The repo uses [pre-commit](https://pre-commit.com/) to run automated checks before
a commit (and, for the test suite, before a push). The config
([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) lives at the git root, so it
covers every session. If a hook fails or modifies a file, the commit is aborted so
you can review, re-stage, and try again — bad code never reaches history.

**Enable it (once per clone):**

```bash
pip install pre-commit
pre-commit install                        # hooks run on `git commit`
pre-commit install --hook-type pre-push   # also run the pre-push hooks
pre-commit run --all-files                # run everything now, across the repo
```

**What the hooks do.** There are two speed tiers: fast static checks run on every
commit, while the slow test suite runs only on push — so per-commit feedback stays
fast but broken tests can't reach the remote.

*Ruff — linter + formatter (Rust-fast; replaces flake8 + Black + isort)*

| Hook            | Purpose                                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------------------|
| `ruff (--fix)`  | Flags unused imports, bad style, likely bugs; auto-fixes what it safely can. Commit aborts if it changed files, so you review + re-stage. |
| `ruff-format`   | Reflows code to a consistent style (indentation, quotes, line length).                                    |

*Generic hygiene hooks (from the pre-commit project)*

| Hook                                    | Purpose                                                                        |
|-----------------------------------------|--------------------------------------------------------------------------------|
| `trailing-whitespace`                   | Strip trailing spaces at line ends.                                            |
| `end-of-file-fixer`                     | Ensure files end with exactly one newline.                                     |
| `mixed-line-ending (--fix=lf)`          | Normalise CRLF/LF so Windows/mac edits agree.                                  |
| `check-yaml` / `check-toml` / `check-json` | Validate config files parse — fail early, not at runtime.                   |
| `check-merge-conflict`                  | Block commits that still contain `<<<<<<<` markers.                            |
| `check-added-large-files (--maxkb=500)` | Stop big blobs entering git — the DVC guardrail: data belongs in DVC, not git. |
| `check-ast`                             | Confirm every `.py` file parses (valid syntax).                                |
| `debug-statements`                      | Catch leftover `breakpoint()` / `pdb.set_trace()`.                             |
| `check-executables-have-shebangs`       | An executable (`+x`) file must have a shebang.                                 |
| `detect-private-key`                    | Block accidentally committing an SSH/PEM key.                                  |

*Local pytest hook (runs at the pre-push stage only)*

Actually executes the session_2 test suite (`cd session_2 && pytest`) to verify
behaviour. Because it's slower, it runs on `git push` — not on every commit — and
always runs the whole suite. Requires the dev deps: `pip install -e ".[dev]"`.

> Because that hook shells out to the `pytest` on your `PATH`, activate the venv
> (`source .venv/bin/activate`) before `git push`, or the push fails with
> `pytest: command not found`.

**Maintenance.** Bump the pinned hook versions anytime with `pre-commit autoupdate`.

### Running tests directly

```bash
pip install -e ".[dev]"
pytest
```

The suite exercises `src/train.py` (data generation, split, training, evaluation)
plus an end-to-end integration test — it does **not** require a running MLflow
server. CI enforces a 50% coverage gate.

---

## 5. Terraform — infrastructure as code

[Terraform](https://developer.hashicorp.com/terraform) is an
**Infrastructure-as-Code (IaC)** tool: you *declare* the cloud resources you
want in `.tf` files, and Terraform figures out what to create, change, or
destroy to make reality match. Why that's useful:

- **Declarative & reviewable** — the bucket exists because a git-committed,
  code-reviewed file says so, not because someone clicked through a console.
- **Plan before apply** — `terraform plan` shows exactly what would change
  *before* anything touches the cloud; no surprises.
- **State tracking** — `terraform.tfstate` records what Terraform manages, so
  `apply` is idempotent (safe to re-run) and `destroy` tears down everything
  it created, nothing more.
- **Reproducible environments** — the same files recreate identical infra in
  a new project/region/account in minutes.
- **One workflow, any cloud** — the same commands drive the GCP bucket here
  and the commented-out AWS S3/ECR resources.

The `infra/` folder provisions the cloud storage this project depends on. It
currently creates a **GCS bucket** (the AWS S3/ECR resources are kept in
[`main.tf`](infra/main.tf) but commented out after the migration to GCP):

- **GCS bucket** (GCP) — remote storage for MLflow + DVC artifacts
  (`mlops-artifacts-<project_name>`, uniform access + object versioning).

```
infra/
├── main.tf          # providers (google; aws commented), GCS bucket, outputs
├── variables.tf     # input variables (project_name, gcp_project, gcp_region)
├── terraform.tfvars # real values (set gcp_project to your project ID)
└── commands.md      # Terraform command cheatsheet
```

Requires the [Terraform CLI](https://developer.hashicorp.com/terraform/install)
and the `gcloud` CLI authenticated with Application Default Credentials.

### Set up the `gcloud` CLI (one-time)

The `google` provider authenticates with **Application Default Credentials
(ADC)**, which `gcloud` writes to disk:

```bash
# 1. Log in and pick the active project
gcloud init                             # interactive: login + choose project
# or non-interactively:
gcloud auth login                       # browser login (your user identity)
gcloud config set project YOUR_PROJECT_ID

# 2. Write Application Default Credentials for Terraform / SDKs to use
gcloud auth application-default login    # what `terraform apply` reads

# 3. Enable the API Terraform needs (once per project)
gcloud services enable storage.googleapis.com
```

> `gcloud auth login` authenticates the **CLI** (for `gcloud storage ls`, etc.);
> `gcloud auth application-default login` authenticates **libraries and tools**
> like Terraform. You typically run both. In CI, use a service-account key
> (`GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json`) instead of interactive login.

### All the Terraform commands (the standard workflow)

**Run the code for this section** — this *is* how you run `infra/`; the same
sequence is kept as a cheatsheet in [`infra/commands.md`](infra/commands.md):

```bash
cd infra

# 1. Initialise — download providers, set up the backend (run once per checkout)
terraform init

# 2. Format — auto-format all .tf files
terraform fmt

# 3. Validate — check syntax/config without calling the cloud provider
terraform validate

# 4. Plan — preview what will change (read-only)
terraform plan -out=tfplan

# 5. Apply — create/update resources (applies the saved plan exactly)
terraform apply tfplan

# 6. Show outputs (e.g. the created bucket name)
terraform output gcs_bucket_name

# 7. Destroy — tear everything down (be careful!)
terraform destroy
```

A few more you'll reach for day to day:

```bash
terraform apply               # plan + apply in one step (prompts for approval)
terraform show                # human-readable dump of the current state
terraform state list          # list every resource Terraform manages
terraform output              # print all outputs
terraform plan -destroy       # preview what `destroy` would delete
```

> Bucket names are **globally unique** across all of GCS — if
> `mlops-artifacts-<project_name>` is taken, change `project_name` (or the name
> prefix in `main.tf`). `force_destroy = false`, so `terraform destroy` won't
> delete a non-empty bucket; empty it first (or flip the flag) to tear it down.

---

## Serving the model (inference API)

[`serve.py`](serve.py) is a small FastAPI app that loads the fitted model and
exposes two endpoints:

| Method & path  | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `GET  /health` | Liveness probe (used by the Docker `HEALTHCHECK` and CI)   |
| `POST /predict`| Predict duration from `{"distance_km": …, "passengers": …}`|

The model file is chosen by the `MODEL_PATH` env var (default
`models/rf_model.pkl`, produced by the DVC `train` stage). Run it locally:

```bash
pip install -e ".[serve]"                    # fastapi + uvicorn
uvicorn serve:app --port 8000
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km": 10, "passengers": 2}'   # -> {"duration_min": 21.08}
```

### The container ships **code only** — mount the model at run time

The Docker image deliberately does **not** bake in a model. `serve.py` loads the
model from `MODEL_PATH` at startup, so the image carries only the app + its
dependencies and you supply the model as a **read-only bind mount** when you run
it. This is the standard production pattern — the model artifact is decoupled from
the code artifact:

- **One image serves any model version.** Roll a new model out (or roll back) by
  mounting a different file — no rebuild, no redeploy of the image.
- **Retraining never rebuilds the image.** Training and serving evolve on separate
  cadences.
- **Smaller, immutable images** whose contents don't depend on which model is
  current.

Because there's no baked-in model, running the image **without** a mounted model
fails fast at startup (`serve.py` can't load `MODEL_PATH`) — that's intended.

```bash
docker build -t ride-api .

# Mount the model file at MODEL_PATH (default /app/models/rf_model.pkl), read-only
docker run -p 8000:8000 \
  -v "$(pwd)/models/rf_model.pkl:/app/models/rf_model.pkl:ro" \
  ride-api
```

**Where does the mounted model come from?** Pull the latest blessed model from
your **model registry** first, then mount whatever you fetched. For example, with
the MLflow registry from this session (`@champion` alias):

```bash
# Fetch the current champion from the registry into ./models/rf_model.pkl
python -c "import mlflow, joblib; \
  m = mlflow.sklearn.load_model('models:/RideDurationModel@champion'); \
  joblib.dump(m, 'models/rf_model.pkl')"

# Point MODEL_PATH somewhere else if you prefer, and mount to match:
docker run -p 8000:8000 \
  -e MODEL_PATH=/models/champion.pkl \
  -v "$(pwd)/models/rf_model.pkl:/models/champion.pkl:ro" \
  ride-api
```

In a real deployment the same idea applies: an init container / entrypoint runs
`dvc pull` or an `mlflow`/registry download to place the blessed model on a volume,
and the serving container mounts it via `MODEL_PATH`.

### Multi-stage build — why the `builder` stage matters

The [`Dockerfile`](Dockerfile) uses a **two-stage build** that separates *how the
image is built* from *what actually ships*:

| Stage     | Role                                                                          |
|-----------|-------------------------------------------------------------------------------|
| `builder` | Installs the project + `serve` extra into an isolated venv at `/opt/venv`      |
| `runtime` | Copies **only** that finished venv (`COPY --from=builder /opt/venv /opt/venv`) plus `serve.py` |

The builder stage does all the messy work — running `pip install`, downloading
wheels, compiling — inside a throwaway image. The runtime stage then copies over
just the ready-to-use virtualenv and nothing else. Think **kitchen vs. plate**: you
cook in the kitchen (builder) and carry only the finished meal (`/opt/venv`) to the
plate (runtime) the user actually gets. The payoff: a smaller, cleaner final image;
a smaller attack surface (no pip cache or build cruft, plus a non-root `appuser`);
better layer caching; and reproducible, decoupled builds.

---

## 6. What are CI, CD & CT?

Three levels of automation, each building on the previous one:

- **CI — Continuous Integration.** Every push and pull request automatically
  runs the checks (lint, format, tests) on a clean machine. Broken code is
  caught in minutes, before a human reviews it and long before it ships. In
  this repo: the `lint-and-test` job.
- **CD — Continuous Delivery/Deployment.** When code reaches the main branch,
  the deployable artifact is built, verified, and published automatically — no
  "works on my machine" release steps. In this repo: the `build-and-push` job,
  which builds the Docker image, smoke-tests it against the real model, and
  pushes it to Docker Hub.
- **CT — Continuous Training.** The ML-specific third leg: **retraining the
  model automatically** when its inputs change — new data, new features, new
  hyperparameters — instead of waiting for a human to remember to do it.

**Why CT matters for ML.** A normal application only misbehaves when its
*code* changes — which CI already guards. A model also degrades when the
*world* changes: user behaviour shifts, prices inflate, new categories appear
(**data drift**), and the relationship the model learned goes stale (**concept
drift**) — all while the code stays green and every test passes. So the
training loop itself must be a versioned, triggerable pipeline, not a person
with a notebook: retrain on fresh data, evaluate against the current champion,
and only promote if the metrics hold up.

How the three map onto this repo:

| Concept | In this repo | How you run it |
|---------|--------------|----------------|
| CI  | `lint-and-test` job in `ci_cd.yml` | push / open a PR ([section 7](#7-github-actions--the-repos-cicd-pipeline)) |
| CD  | `build-and-push` job in `ci_cd.yml` | merge to `main` |
| CT  | the DVC pipeline ([section 4](#4-dvcyaml--a-small-automated-ml-lifecycle)) | `dvc repro` after data/params change |

CT here is still hand-triggered — but because the whole lifecycle is already
scripted (`dvc repro` retrains and re-evaluates whenever
`data/raw/rides.csv` or the tracked params change), promoting it to *true* CT
is just adding a trigger: e.g. a scheduled GitHub Actions job that runs
`dvc pull && dvc repro && dvc push` and opens a PR with the new `dvc.lock`
and metrics.

---

## 7. GitHub Actions — the repo's CI/CD pipeline

[GitHub Actions](https://docs.github.com/actions) is GitHub's built-in
automation service: you commit a YAML **workflow** under `.github/workflows/`,
and GitHub runs its **jobs** on hosted runner VMs whenever the **events** you
subscribed to happen (a push, a PR, a schedule, …).

Our pipeline is defined in **`.github/workflows/ci_cd.yml`** at the
**repository root** (`mlops_sessions/`), not inside `session_2/` — GitHub only
discovers workflows at the repo root. Every `run:` step is scoped to
`session_2/` via `defaults.run.working-directory`.

| Job              | Trigger                                | What it does                                          |
|------------------|----------------------------------------|-------------------------------------------------------|
| `lint-and-test`  | push to `main`/`develop`, PR to `main` | Ruff lint/format check, pytest + 50% coverage         |
| `build-and-push` | push to `main` only                    | Build the **code-only** image and push to Docker Hub  |

### The sections of a workflow file (annotated from `ci_cd.yml`)

Every workflow is assembled from the same blocks. Here is each one as it
actually appears in ours:

| Section | In `ci_cd.yml` | Role |
|---------|----------------|------|
| `name:` | `CI / CD Pipeline` | Display name in the Actions tab |
| `on:` | `push: branches: [main, develop]` + `pull_request: branches: [main]` | The **events** that trigger the workflow |
| `env:` | `DOCKERHUB_IMAGE: ${{ secrets.DOCKERHUB_USERNAME }}/ride-api` | Workflow-wide variables (may read secrets) |
| `defaults:` | `run: working-directory: session_2` | Defaults for all steps — here, every `run:` executes inside `session_2/` |
| `jobs:` | `lint-and-test`, `build-and-push` | Independent units, each on a fresh runner (`runs-on: ubuntu-latest`) |
| `strategy.matrix:` | `python-version: ["3.10", "3.11", "3.12"]` | Fans one job definition out into parallel variants |
| `steps:` | checkout → setup-python → cache → install → ruff → pytest → … | Sequential actions within a job |
| `uses:` steps | `actions/checkout@v7`, `docker/build-push-action@v7`, … | Reusable **marketplace actions** (inputs via `with:`) |
| `run:` steps | `ruff check src/ tests/`, `pytest --cov=src …` | Plain shell commands |
| `needs:` / `if:` | `needs: lint-and-test`, `if: github.ref == 'refs/heads/main'` | Job ordering and gating — build only after tests pass, only on `main` |
| `${{ secrets.* }}` | `DOCKERHUB_TOKEN`, `GCP_SA_KEY`, … | Encrypted credentials injected at runtime (next subsection) |

Two gotchas the file itself documents: `defaults.run.working-directory` only
affects `run:` steps, so paths handed to `uses:` actions are written
repo-root-relative (`session_2/coverage.xml`); and `build-and-push` pulls the
DVC-pinned model from GCS (not the MLflow `@champion` alias, whose registry DB
is local) to smoke-test the image before pushing it.

### 1. Required secrets

Add these as **repository secrets** — repo → **Settings** → **Secrets and
variables** → **Actions** → *New repository secret*. They live on GitHub and are
never committed. (This is the **repo's** Settings tab, not your personal account
settings.)

| Secret               | Needed for            | Where to get it                                                                 |
|----------------------|-----------------------|---------------------------------------------------------------------------------|
| `DOCKERHUB_USERNAME` | `build-and-push`      | Your Docker Hub username (also becomes the image namespace `<user>/ride-api`)   |
| `DOCKERHUB_TOKEN`    | `build-and-push`      | hub.docker.com → Account settings → **Personal access tokens** → *Read & Write* |
| `GCP_SA_KEY`         | `build-and-push`      | Full service-account **key JSON** (`roles/storage.objectAdmin` on the GCS bucket) — the same key used locally as `gcp-key.json` |
| `CODECOV_TOKEN`      | `lint-and-test` (opt) | app.codecov.io → your repo → Settings → upload token                            |

Notes:

- `DOCKERHUB_TOKEN` is a Docker Hub **access token** (looks like `dckr_pat_…`),
  **not** your password. Scope it *Read & Write* so CI can push.
- `GCP_SA_KEY` is the **entire contents** of the service-account JSON (not a path).
  `build-and-push` writes it to `gcp-key.json` and runs `dvc pull -r gcs` to fetch
  the model from `gs://mlops-session2-iti/dvc-store`, then smoke-tests the image
  against it before pushing. Without it, that job fails at the model-pull step.
- `CODECOV_TOKEN` is optional — the upload step has `fail_ci_if_error: false`, so
  CI stays green without it. `lint-and-test` needs **no** secrets to run.

Set them from the terminal with the GitHub CLI (must be authenticated as the repo
owner — check with `gh auth status`):

```bash
gh secret set DOCKERHUB_USERNAME --body "your-dockerhub-username"
gh secret set DOCKERHUB_TOKEN      # prompts, paste the token (hidden)
gh secret set GCP_SA_KEY < gcp-key.json   # feed the whole key JSON from the file
gh secret set CODECOV_TOKEN        # optional
gh secret list                     # verify
```

### 2. Build & verify the Docker image locally (before triggering CI)

The `build-and-push` job just automates these steps — running them by hand first
confirms the image builds, serves correctly, and that your Docker Hub token works,
before you rely on CI.

```bash
cd session_2

# 1. Build the code-only image (Dockerfile + app live in session_2/; no model)
docker build -t ride-api:local .

# 2. Run it WITH the model mounted, then smoke-test the endpoints CI health-checks.
#    (You need a models/rf_model.pkl first — `dvc pull`, run the DVC train stage,
#     or fetch @champion from the registry as shown above.)
docker run -d --name ride-api -p 8000:8000 \
  -v "$(pwd)/models/rf_model.pkl:/app/models/rf_model.pkl:ro" ride-api:local
curl localhost:8000/health                        # {"status":"ok"}
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km": 10, "passengers": 2}'        # {"duration_min": 21.08}
docker rm -f ride-api

# 3. (Optional) push by hand to confirm the token/permissions work
docker login -u <your-dockerhub-username>          # paste the dckr_pat_… token as the password
docker tag ride-api:local <your-dockerhub-username>/ride-api:test
docker push <your-dockerhub-username>/ride-api:test
```

In CI the same build runs via `docker/build-push-action`, tagging every image two
ways and reusing cached layers (`cache-from: …:latest`):

| Tag                       | Example                      | Purpose                        |
|---------------------------|------------------------------|--------------------------------|
| `…/ride-api:<short-sha>`  | `ayanasser/ride-api:a1b2c3d` | Immutable, one per commit      |
| `…/ride-api:latest`       | `ayanasser/ride-api:latest`  | Always the newest `main` build |

### 3. Trigger the pipeline

The workflow fires automatically on the events in its `on:` block — you trigger it
with ordinary git pushes / PRs, there is no button to click:

| Event                    | How you cause it                        | Jobs that run                      |
|--------------------------|-----------------------------------------|------------------------------------|
| **PR targeting `main`**  | open a PR from your branch              | `lint-and-test` only               |
| **Push to `develop`**    | `git push origin develop`               | `lint-and-test` only               |
| **Push to `main`**       | `git push origin main` / merge a PR     | `lint-and-test` → `build-and-push` |

Typical flow — branch → PR (tests) → merge (tests + image build & push):

```bash
# 1. Work on a branch
git checkout -b my-change
git add -A && git commit -m "describe the change"
git push -u origin my-change

# 2. Open a PR to main  → runs lint-and-test
gh pr create --base main --fill          # or use the link git prints on push

# 3. Merge it           → push to main re-runs tests, then build-and-push
gh pr merge --squash --delete-branch
```

> `build-and-push` is gated on `if: github.ref == 'refs/heads/main'`, so the image
> is published **only after merge to `main`**, never from a PR — unreviewed code
> never reaches your registry.

To enable manual runs, add `workflow_dispatch:` under `on:` in `ci_cd.yml`:

```yaml
on:
  workflow_dispatch:        # adds a "Run workflow" button in the Actions tab
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

### 4. Watch the results

```bash
gh run list                  # recent runs + pass/fail status
gh run watch                 # live-tail the in-progress run
gh run view --log-failed     # show logs for only the failed steps
```

Or open the repo's **Actions** tab in the browser. After a successful `main` build,
the image appears on Docker Hub under `<your-username>/ride-api`.

### 5. Dry-run the workflow locally with `act` (optional)

From the **repo root** (where `act` finds `.github/workflows/`):

```bash
cd ..                                    # into mlops_sessions/
act -l                                   # list detected jobs
act pull_request -j lint-and-test        # run the test job in Docker
```

---

## 8. Integration testing

**What is it?** A **unit test** checks one function in isolation
([`tests/test_train.py`](tests/test_train.py) calls `train_model()` directly
and inspects the return value). An **integration test** checks that several
pieces work correctly *together*, across their real boundaries. Ours exercises
the whole serving path: HTTP request → FastAPI routing → Pydantic validation →
`model.predict()` → JSON response — the contract CI's smoke test and real
clients depend on.

**How ours is built.** [`tests/test_integration.py`](tests/test_integration.py)
drives the real app in [`serve.py`](serve.py) through FastAPI's `TestClient`
(real request/response cycle, no network socket). The one trick: `serve.py`
loads the model at import time, so the test patches `joblib.load` to return a
**stub model** (`predict()` always returns `15.5`) *before* importing `serve`.
That makes the test **hermetic** — no DVC pull, no trained `.pkl`, no MLflow
server needed — and pins down what it verifies: the HTTP contract, not the
model's numerical accuracy.

What it asserts:

- `GET /health` returns `200 {"status": "ok"}` — the probe Docker and CI use.
- `POST /predict` happy path returns `200` with `duration_min` as a float.
- The response schema is exactly `{"duration_min"}` — no extra keys leak out.
- Seven parametrised invalid payloads (negative/zero distance, 0 or 7
  passengers, missing fields, empty body) each return **422** — the Pydantic
  `Field` constraints in `serve.py` actually reject bad input.

**How to build one — the recipe used here:**

1. **Pick the boundary** you want to guarantee (here: the HTTP API surface).
2. **Stub external dependencies at the seam** — patch the one call that
   touches the outside world (`joblib.load`), not your own logic.
3. **Drive the real entrypoint** (`TestClient(serve.app)`), so routing,
   validation, and serialisation all actually execute.
4. **Assert the contract, not internals** — status codes, response shape,
   error behaviour; things a client would notice breaking.
5. **Parametrise the edge cases** so every validation rule is pinned by a test.

**Run the code for this section** (from `session_2/`):

```bash
pip install -e ".[dev]"               # pytest + fastapi/httpx (via the serve extra)
pytest tests/test_integration.py -v   # just the integration tests
pytest -v                             # full suite — what CI runs (with coverage)
```
