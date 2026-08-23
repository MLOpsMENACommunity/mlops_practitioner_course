# Ride Duration API

A minimal [FastAPI](https://fastapi.tiangolo.com/) service that predicts ride
duration (in minutes) from a trip's distance and passenger count. Built as an
MLOps course example.

## 🚫 What NOT to do: the "everything notebook"

[`bad_notebook_example.ipynb`](bad_notebook_example.ipynb) is a **deliberately
bad** notebook that crams exploration, preprocessing, training, evaluation and
"tests" into one file — the classic starting point of most ML projects, and the
reason most of them never reach production. Every problem below is present in
the notebook (marked with ❌ comments). The rest of this project is the
*fixed* version of the same workflow.

### Bad example of development in a notebook

1. Everything in one notebook (can't reuse/schedule/deploy)
2. Unpinned `!pip install` — irreproducible environment
3. Hardcoded absolute paths for data and model
4. Hardcoded secrets
5. No random seeds — nobody can reproduce today's model
6. Hidden state: out-of-order cells + the non-idempotent miles→km cell
7. Silent failure — missing file quietly trains on random data
8. Magic numbers with no config
9. No experiment tracking — metrics in prints and comments, hyperparameter history lost by hand-editing (→ MLflow + git)
10. Training/serving skew — copy-pasted, divergent preprocessing
11. Model "versioned" by filename on a Desktop, fitted scaler never saved (→ registry)
12. Tests as manual cells, never in CI (`assert True`, `except: pass`)
13. No entry point a scheduler or server can call

### From notebook to production — how this repo fixes it

| Notebook problem | Production fix in this repo |
|---|---|
| Everything in one `.ipynb` | Logic extracted into importable modules: [`src/model.py`](src/model.py) |
| `!pip install` with no versions | Declared, versioned dependencies in [`pyproject.toml`](pyproject.toml) (`pip install -e .`) |
| Manual cell-based "tests" | Real `pytest` unit tests in [`tests/test_model.py`](tests/test_model.py), runnable in CI |
| No entry point | A served API: [`fastapi_example.py`](fastapi_example.py) / [`litestar_example.py`](litestar_example.py) with `/predict` + `/health` |
| Hardcoded paths & "works on my machine" | Containerized with the [`Dockerfile`](Dockerfile); model path injected via the `MODEL_PATH` env var in [`docker-compose.yml`](docker-compose.yml) |
| Metrics in print statements & comments | MLflow tracking server in the compose stack |
| Pickle on a Desktop | Models mounted from a versioned [`models/`](models/) directory (e.g. `models/v1/`) |

The general recipe: **extract** pure functions (load → preprocess → train →
evaluate) out of the notebook into modules, **parameterize** every path and
magic number (config/env vars/CLI args), **pin** the environment, **test** the
functions with pytest, **track** experiments instead of printing them, and keep
notebooks only for what they're good at — exploration and reporting.

## Project structure

```
session_1/
├── bad_notebook_example.ipynb  # ⚠️ intentional ANTI-example (see section above)
├── fastapi_example.py    # FastAPI app + entry point
├── litestar_example.py   # Litestar app (same model, DI-based)
├── msgpack_example.py    # MessagePack (de)serialization example
├── pytorch_to_onnx.py    # Export a PyTorch model to ONNX + validate
├── pyproject.toml        # Project metadata + dependencies (core + extras)
├── Dockerfile            # Multi-stage image for the FastAPI app
├── .dockerignore         # Files excluded from the Docker build context
├── docker-compose.yml    # API + MLflow stack
├── models/               # Local model mount (read-only in compose)
├── src/
│   ├── __init__.py
│   └── model.py          # RideDurationModel (delegates to a swappable estimator)
├── tests/
│   └── test_model.py     # pytest unit tests for the model
├── docs/                 # Notes / supporting docs
└── README.md
```

> **Note:** `RideDurationModel` delegates prediction to an internal estimator
> (`self._model`), which defaults to a simple heuristic (distance ÷ average
> speed + per-passenger overhead). Swap in a real trained model by assigning any
> object with a `predict(features) -> [value]` method.

## Requirements

- Python 3.10+

## Setup

Create a virtual environment and install the project (dependencies are
declared in `pyproject.toml`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .                   # installs deps from pyproject.toml (editable)
```

The `-e` flag installs the project in *editable* mode, so changes to the source
are picked up without reinstalling. Drop `-e` for a plain install (`pip install .`).

## uv & `pyproject.toml`

### What is uv?

[**uv**](https://docs.astral.sh/uv/) is a fast, all-in-one Python project
manager (from Astral, the makers of Ruff). It replaces the whole toolchain of
`pip` + `venv` + `pip-tools` with one binary: it creates projects, manages the
virtualenv for you, resolves and installs dependencies (10–100× faster than
pip), and — the MLOps-relevant part — maintains a **lockfile** (`uv.lock`) that
pins every transitive dependency, so two installs a month apart produce the
*identical* environment.

### Starting a project with `uv init`

```bash
uv init ride-duration-api        # or `uv init .` inside an existing folder
cd ride-duration-api
```

This scaffolds the project skeleton:

```
ride-duration-api/
├── pyproject.toml   # project metadata + dependencies (starts almost empty)
├── .python-version  # pins the Python version for this project
├── README.md
└── main.py          # hello-world entry point
```

From then on you never edit the dependency list by hand and never activate a
venv manually — uv keeps `pyproject.toml`, `uv.lock`, and the hidden `.venv/`
in sync on every command.

### The commands you'll actually use

| Command | What it does |
|---|---|
| `uv init` | Create a new project (pyproject.toml, .python-version, …) |
| `uv add fastapi` | Add a dependency: writes it to `[project.dependencies]`, resolves, updates `uv.lock`, installs into `.venv` — one command, all four steps |
| `uv add --dev pytest` | Add a dev-only tool (test runners, linters) to the dev group, kept out of production installs |
| `uv add "torch>=2.2" --optional onnx` | Add to an optional extra (like our `[project.optional-dependencies]` groups) |
| `uv remove fastapi` | The reverse: drop it from pyproject.toml, re-lock, uninstall |
| `uv sync` | Make `.venv` match the lockfile exactly — the "clone → working env" command for teammates and CI |
| `uv lock` | Re-resolve and rewrite `uv.lock` without installing |
| `uv run python train.py` / `uv run pytest` | Run a command inside the project env **without activating anything** (auto-syncs first) |
| `uv python install 3.12` | Download/manage Python interpreters themselves — no pyenv needed |
| `uv pip install …` | Escape hatch: pip-compatible interface for one-off installs |

The daily workflow is just three of them:

```bash
uv add <package>     # when you need a new library
uv run pytest        # to run anything in the project env
uv sync              # after pulling changes from a teammate
```

Compare that with the notebook's `!pip install pandas sklearn ...` — no record
of what was installed, no versions, no lockfile, different result every month.
`uv add` + `uv.lock` is the fix for bad-notebook problem #2.

### ⚠️ Add the *package* name, not the *import* name

The name you `import` in Python and the name of the package on PyPI are **not
always the same**. Dependencies in `pyproject.toml` (and `uv add`) always use
the **PyPI package name**:

| Import name (in code) | Actual package (what you `uv add`) |
|---|---|
| `import cv2` | `opencv-python` |
| `import sklearn` | `scikit-learn` |
| `import PIL` | `pillow` |
| `import yaml` | `PyYAML` |
| `import bs4` | `beautifulsoup4` |

```bash
uv add opencv-python scikit-learn pillow PyYAML beautifulsoup4   # ✅
uv add cv2 sklearn PIL yaml bs4                                  # ❌ wrong/deprecated packages
```

Getting this wrong ranges from a `No solution found` error (`cv2`, `PIL`) to
something worse: `sklearn` *is* an installable package on PyPI — a deprecated
stub whose only job is to tell you to install `scikit-learn`. When in doubt,
check the package's page on [pypi.org](https://pypi.org) — the `pip install` /
`uv add` name is at the top.

### Anatomy of `pyproject.toml`

`pyproject.toml` is Python's standard, single configuration file for a project.
Walking through [this project's file](pyproject.toml) section by section:

```toml
[project]                          # ── WHO/WHAT: the project's identity ──
name = "ride-duration-api"         # package name (what `pip install` would be called)
version = "0.1.0"                  # bump on release
description = "..."
readme = "README.md"
requires-python = ">=3.10"         # Python floor — installs fail fast on 3.9
dependencies = [                   # ── runtime deps: what the app NEEDS to run ──
    "fastapi>=0.138",              # PyPI package names (see table above!)
    "uvicorn>=0.49",
    "pydantic>=2.13",
    "litestar>=2.24",
]

[project.optional-dependencies]    # ── opt-in extras: `pip install -e ".[onnx]"` ──
onnx = ["torch>=2.2", ...]         # heavy deps only pytorch_to_onnx.py needs
dev = ["pytest>=8.0"]              # tooling for developers, not for production
msgpack = ["msgpack>=1.0"]         # dep only msgpack_example.py needs
redis = ["redis>=5.0"]             # only the optional cache half of that demo

[build-system]                     # ── HOW to build/install the project ──
requires = ["setuptools>=61"]      # the build tool itself
build-backend = "setuptools.build_meta"

[tool.setuptools]                  # ── [tool.*]: per-tool config, namespaced ──
packages = ["src"]                 # which folders become importable on install

[tool.pytest.ini_options]          # pytest reads its config from here too —
testpaths = ["tests"]              # one file instead of pytest.ini/setup.cfg/...
```

The mental model:

- **`[project]`** — metadata + *what the app needs* (runtime dependencies).
- **`[project.optional-dependencies]`** — extras users opt into
  (`.[onnx]`, `.[dev]`); keeps the core install slim.
- **`[build-system]`** — *how* to turn the folder into an installable package.
- **`[tool.*]`** — every tool's config lives in its own namespaced table
  (`[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.uv]`…), replacing the
  old pile of `pytest.ini`, `.flake8`, `setup.cfg` files.

> **Note:** uv-managed projects express dev tooling as a `[dependency-groups]`
> table (the newer standard, PEP 735) instead of a `dev` extra — `uv add --dev`
> writes there. This project uses the classic `dev` extra so plain
> `pip install -e ".[dev]"` works too; both patterns are common and do the
> same job.

One thing `pyproject.toml` does **not** do by itself: pin exact versions.
`fastapi>=0.138` says "at least 0.138" — installs at different times can pick
different versions. That's the lockfile's job (`uv.lock`), which records the
exact version *and hash* of every package (including dependencies of
dependencies) that a resolve produced. Commit it, and `uv sync` reproduces the
environment byte-for-byte anywhere.

## How `pyproject.toml` grew, stage by stage

This project started with a flat `requirements.txt` and migrated to
`pyproject.toml`, then grew as each new capability was added. Below is the exact
edit made to the file at every stage — a small tour of how to shape a
`pyproject.toml` around a project's needs.

### Stage 0 — the starting point: `requirements.txt`

Originally dependencies lived in a plain text file:

```text
fastapi
uvicorn
pydantic
```

No versions, no metadata, no way to declare optional/dev dependencies or build
the project as an installable package.

### Stage 1 — migrate to `pyproject.toml` (core dependencies)

We replaced `requirements.txt` with a `pyproject.toml` that declares project
metadata, a Python floor, pinned lower bounds, and a build backend. `litestar`
was added here because [`litestar_example.py`](litestar_example.py) needs it:

```toml
[project]
name = "ride-duration-api"
version = "0.1.0"
description = "A minimal FastAPI/Litestar service that predicts ride duration from trip distance and passenger count."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.138",
    "uvicorn>=0.49",
    "pydantic>=2.13",
    "litestar>=2.24",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["src"]      # makes `src` importable after `pip install`
```

Install everything (editable) with a single command:

```bash
pip install -e .
```

### Stage 2 — add an optional `onnx` extra (PyTorch → ONNX export)

[`pytorch_to_onnx.py`](pytorch_to_onnx.py) needs `torch` and `onnx`, which are
large and irrelevant to serving the API. Instead of bloating the core
dependencies, we added an **optional-dependency group** so they install only on
demand. (`onnxscript` was added after we found torch's exporter requires it.)

```toml
[project.optional-dependencies]
# Heavy deps used only by pytorch_to_onnx.py (ONNX export). Install with:
#   pip install -e ".[onnx]"
onnx = [
    "torch>=2.2",
    "onnx>=1.16",
    "onnxscript>=0.2",
]
```

Install the core project **plus** the ONNX tooling:

```bash
pip install -e ".[onnx]"
```

### Stage 3 — add a `dev` extra + pytest config (tests)

To run [`tests/`](tests/), we added a second optional group for test tooling and
a `[tool.pytest.ini_options]` block so `pytest` knows where the tests live:

```toml
[project.optional-dependencies]
# ... onnx group from Stage 2 ...
# Test/dev tooling. Install with:
#   pip install -e ".[dev]"
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Install the core project **plus** the dev tooling, then run the suite:

```bash
pip install -e ".[dev]"
pytest
```

### The result

The final [`pyproject.toml`](pyproject.toml) cleanly separates concerns: a small
core install for running the API, and opt-in extras for the heavy ONNX tooling
and the test suite. Combine extras when you need several at once:

```bash
pip install -e ".[onnx,dev]"
```

## Running the server

This repo ships two equivalent implementations of the same API — one with
FastAPI, one with [Litestar](https://litestar.dev/). Pick either.

### FastAPI (port 8000)

**As a Python script:**

```bash
python fastapi_example.py
```

**With uvicorn (adds auto-reload for development):**

```bash
uvicorn fastapi_example:app --host 127.0.0.1 --port 8000 --reload
```

The server starts at **http://127.0.0.1:8000**. Stop it with `Ctrl+C`.

### Litestar (port 8001)

**As a Python script:**

```bash
python litestar_example.py
```

**With the Litestar CLI (adds auto-reload for development):**

```bash
litestar --app litestar_example:app run --host 127.0.0.1 --port 8001 --reload
```

The server starts at **http://127.0.0.1:8001**. Stop it with `Ctrl+C`.

> Both apps expose the same `/health` and `/predict` endpoints — just swap the
> port (`8000` → `8001`) in the examples below.

## API endpoints

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Health check                         |
| POST   | `/predict`  | Predict ride duration                |

### `POST /predict`

**Request body:**

| Field         | Type  | Required | Default | Constraints   | Description          |
|---------------|-------|----------|---------|---------------|----------------------|
| `distance_km` | float | yes      | —       | `> 0`         | Trip distance (km)   |
| `passengers`  | int   | no       | `1`     | `1 … 8`       | Number of passengers |

The constraints are declared with pydantic `Field(...)`, so a request that
violates them never reaches the model — the framework rejects it with a `422`
and a body describing which field failed.

**Example:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 10, "passengers": 2}'
```

**Response:**

```json
{"duration_min": 21.0}
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "healthy"}
```

## Interactive documentation

Both frameworks auto-generate OpenAPI docs once the server is running.

**FastAPI** (port 8000):

- **Swagger UI:** http://127.0.0.1:8000/docs — try requests in the browser
- **ReDoc:** http://127.0.0.1:8000/redoc — clean reference view
- **OpenAPI schema (JSON):** http://127.0.0.1:8000/openapi.json

**Litestar** (port 8001):

- **Swagger UI:** http://127.0.0.1:8001/schema/swagger — try requests in the browser
- **ReDoc:** http://127.0.0.1:8001/schema/redoc — clean reference view
- **OpenAPI schema (JSON):** http://127.0.0.1:8001/schema/openapi.json

## MessagePack serialization example

[`msgpack_example.py`](msgpack_example.py) is a standalone demo (not part of the
API) that compares JSON vs [MessagePack](https://msgpack.org/) — a compact
binary serialization format — and shows caching feature vectors in Redis.

It needs one package that is **not** part of the core API dependencies, so it
lives in its own opt-in extra:

```bash
pip install -e ".[msgpack]"
```

```bash
python msgpack_example.py
```

> Run it with the **project venv's** interpreter (`source .venv/bin/activate`
> first, or `.venv/bin/python msgpack_example.py`). A `ModuleNotFoundError: No
> module named 'msgpack'` almost always means a different Python — a system or
> pyenv one — picked up the script.

The first half prints the byte sizes of the same payload as JSON vs MessagePack
(62 vs 53 bytes — MessagePack is ~15% smaller here; the gap widens on larger,
number-heavy payloads). The second half is illustrative: it writes/reads a
feature vector to Redis using MessagePack, and the `model.predict(...)` call is
a placeholder to show how cached features feed a prediction.

That second half is entirely optional. It needs the `redis` package **and** a
running server on the default port:

```bash
pip install -e ".[msgpack,redis]"
docker run -d -p 6379:6379 redis:7-alpine   # or a local `redis-server`
```

Missing either one, the script prints a short notice and finishes cleanly — the
serialization comparison above stands on its own and never touches Redis.

## Exporting to ONNX

### What is ONNX?

[**ONNX**](https://onnx.ai/) (Open Neural Network Exchange) is an open,
framework-neutral **file format for ML models**. The idea: train in whatever
framework you like (PyTorch, TensorFlow, scikit-learn via `skl2onnx`), export
once to `.onnx`, then run it anywhere with an ONNX **runtime** — a C++ server,
ONNX Runtime in Python, a phone, the browser, specialized accelerators — with
no PyTorch installed on the serving side. For MLOps that decouples the
*training* stack from the *serving* stack: the heavyweight framework stays in
the training environment, and production ships a small, portable artifact.

### What actually gets converted?

An `.onnx` file is a [protobuf](https://protobuf.dev/) containing a full,
self-contained description of the model's *forward pass* — three things:

| Part | What it is | In this project's export |
|---|---|---|
| **The graph** | Your model's forward computation, flattened into a static **directed acyclic graph** of nodes — not your Python code, just the math it performs. Python control flow, classes and method structure are *gone*; the exporter **traces** the forward pass and records the operations it saw | one `Gemm` node (matrix multiply + bias) |
| **The operators (ops)** | Each graph node is one operator from ONNX's **standardized catalog** (~190 ops: `Conv`, `MatMul`, `Relu`, `Softmax`, …), versioned by **opset** number. The opset is a contract: any runtime supporting opset 18 can execute any graph that only uses opset-18 ops | `Gemm` from the standard opset (the 17→18 warning in the output is the opset being auto-bumped) |
| **The weights (initializers)** | Every learned parameter — layer weights, biases, batch-norm statistics — stored as raw tensors baked into the file. This is why an `.onnx` file is self-contained: graph + weights travel together | the linear layer's `weight` (initialized to `[[2.0, 0.5]]`) and `bias` |

Plus metadata: named, typed **inputs/outputs** with shapes (that's the
`Inputs: ['features'] / Outputs: ['duration']` printed by the script), the
opset version, and producer info.

What is **not** converted: the training logic (loss, optimizer, gradients),
your Python classes, and any preprocessing outside the model's `forward()` —
the same "scaler left behind" trap as pickling (see the bad notebook): if
scaling isn't part of the exported graph, the serving side must reproduce it.

### ⚠️ The catch: new models may need ops ONNX doesn't have

The operator catalog is ONNX's strength *and* its weakness. Because every
graph node must map to a standardized op, a model is only exportable if
**every operation in its forward pass has an ONNX equivalent**. Established
architectures (CNNs, ResNets, standard transformers) export cleanly — but
cutting-edge or custom models often use operations the spec hasn't caught up
with yet (novel attention variants, custom CUDA kernels, exotic
scatter/gather patterns). Then the export fails with the infamous:

```
UnsupportedOperatorError: Exporting the operator 'aten::my_fancy_op' to ONNX
opset version 18 is not supported
```

The standard escalation path, roughly cheapest-first:

1. **Try a newer opset** (`opset_version=...`) — the op may have been added
   to a later spec revision.
2. **Rewrite the op as a composition of supported ops** — e.g. approximate a
   custom activation with standard `Sigmoid`/`Mul`/`Add` nodes. Cheap when
   possible, but changes the graph and sometimes the numerics.
3. **Register a custom translation** — teach the exporter how to translate
   your PyTorch op (a *custom symbolic function* / ONNX-script translation in
   PyTorch's exporter registry). PyTorch has an [official tutorial dedicated
   to exactly this](https://docs.pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html).
4. **Implement the op in the runtime yourself** — write the kernel as an ONNX
   Runtime *custom op* (C++/Python) or use its `com.microsoft` contrib-op
   domain. Full control, but you now maintain custom serving code — much of
   the "portable artifact" benefit is gone.

The practical MLOps takeaway: **ONNX exportability is a deployment
constraint to check *before* committing to an architecture**, not an
afterthought. A 30-second `torch.onnx.export` smoke test on a prototype model
(exactly what [`pytorch_to_onnx.py`](pytorch_to_onnx.py) does) tells you
whether your serving plan works before you've spent weeks training.

### Exporting this project's model

[`pytorch_to_onnx.py`](pytorch_to_onnx.py) exports a small PyTorch model
(`RideDurationTorchModel` — a single linear layer initialized to match the
`RideDurationModel` heuristic) to the portable [ONNX](https://onnx.ai/) format
and validates the result.

These deps are heavy and unrelated to serving the API, so they live in an
optional `onnx` extra rather than the core dependencies. Install them with:

```bash
pip install -e ".[onnx]"
```

Run the export:

```bash
python pytorch_to_onnx.py
```

This writes **`model.onnx`** to the project root, runs `onnx.checker` to verify
the graph is valid, and prints the model's inputs/outputs:

```
Inputs:  ['features']
Outputs: ['duration']
```

> A couple of non-fatal warnings are expected (opset auto-bumped 17→18,
> `torchvision not installed`) — they don't affect the exported model.

## Docker

### What is Docker?

**Docker** packages an application together with *everything it needs to run* —
Python interpreter, libraries, system packages, environment — into a single
portable unit that behaves identically on your laptop, a teammate's machine,
CI, and a production server. For MLOps this is the answer to *"works on my
machine"*: the environment itself becomes a versioned, shippable artifact, just
like the code and the model.

The four core concepts:

| Concept | What it is | Analogy |
|---|---|---|
| **Dockerfile** | A text file of instructions (`FROM`, `COPY`, `RUN`, `CMD`) describing how to build the environment | The *recipe* |
| **Image** | The built, immutable snapshot produced from a Dockerfile — layers of filesystem + metadata, identified by a tag like `ride-duration-api:session1` | The *frozen meal* — cook once, reheat anywhere |
| **Container** | A *running instance* of an image: an isolated process with its own filesystem, network and resources. You can run many containers from one image; changes inside die with the container | The *served plate* — one recipe, many servings |
| **Registry** | A server that stores and distributes images, so a machine that has never seen your code can `docker pull` and run it | The *supermarket* for images |

The lifecycle ties them together:

```
Dockerfile ──docker build──▶ Image ──docker run──▶ Container
                              │
                              └──docker push/pull──▶ Registry
```

The best-known registry is **[Docker Hub](https://hub.docker.com/)** — the
default when you `docker pull python:3.12-slim` (that base image comes from
there). Alternatives include GitHub Container Registry (`ghcr.io` — where the
MLflow image in our compose file lives), AWS ECR, and Google Artifact Registry.
This repo's CI pipeline pushes its verified image to Docker Hub on every merge
to `main`, which is how a built image travels from CI to any server that wants
to run it.

### Building this project's image

The [`Dockerfile`](Dockerfile) builds the **FastAPI** app as a slim,
production-style image using a **multi-stage build**: a `builder` stage installs
the dependencies into an isolated virtualenv, and a minimal `runtime` stage
copies only that venv plus the app code onto `python:3.12-slim`. The container
runs as a non-root user and serves on port **8000** (bound to `0.0.0.0`).

**Build the image:**

```bash
docker build -t ride-duration-api:session1 .
```

**Run the container:**

```bash
docker run --rm -p 8000:8000 ride-duration-api:session1
```

The API is then available at **http://127.0.0.1:8000** (same endpoints and docs
as above). Stop it with `Ctrl+C`.

## Docker Compose

[`docker-compose.yml`](docker-compose.yml) runs the API alongside an
[MLflow](https://mlflow.org/) tracking server, wiring together a small local
MLOps stack:

| Service  | Image / build      | Port   | Purpose                              |
|----------|--------------------|--------|--------------------------------------|
| `api`    | built from `Dockerfile` | `8000` | Ride Duration API                    |
| `mlflow` | `ghcr.io/mlflow/mlflow` | `5000` | Experiment tracking + artifact store |

The `api` service mounts a local [`models/`](models/) directory read-only at
`/models` (via the `MODEL_PATH` env var) and waits for `mlflow` to start.
MLflow persists its artifacts in the named `mlflow-data` volume.

**Start the stack:**

```bash
docker compose up --build
```

- API: **http://127.0.0.1:8000**
- MLflow UI: **http://127.0.0.1:5000**

**Run in the background / tear down:**

```bash
docker compose up -d --build   # detached
docker compose down            # stop and remove containers
docker compose down -v         # also remove the mlflow-data volume
```

> Drop trained model files into `models/v1/` (e.g. `model.pkl`) to make them
> available inside the container at `/models/v1/`.

## Testing

### What is unit testing?

A **unit test** checks one small piece of code (a "unit" — typically a single
function or method) in isolation: *given this input, does it return this
output?* Unit tests are fast (milliseconds), need no network/database/GPU, and
pinpoint exactly which piece broke. They sit at the bottom of the classic
**testing pyramid**:

```
        ▲  e2e tests         — few: the whole system, slow, broad
       ▲▲  integration tests — some: pieces working together (e.g. API + model)
      ▲▲▲  unit tests        — many: single functions, fast, precise
```

Contrast this with the "tests" in the [bad notebook](bad_notebook_example.ipynb)
(`assert True` in a cell, run by hand, gone on kernel restart): real tests are
**code that lives in the repo**, run by a test runner, and executed
*automatically* on every push by CI — so a broken change is caught before it
merges, not after it ships. In ML projects unit tests are what let you refactor
preprocessing or swap a model implementation with confidence that the contract
(`predict([distance, passengers]) -> minutes`) still holds.

**pytest** is Python's de-facto test runner: it auto-discovers files named
`test_*.py`, treats every `test_*` function as a test case, and gives you plain
`assert` statements, **fixtures** (reusable setup, like a pre-configured model)
and **parametrize** (run one test body over many input/expected pairs). All
three appear in [`tests/test_model.py`](tests/test_model.py).

### Running the tests

Unit tests for the model live in [`tests/`](tests/) and use `pytest`. Install
the `dev` extra, then run them:

```bash
pip install -e ".[dev]"
pytest            # run everything
pytest -v         # verbose: one line per test case
pytest -k thresh  # run only tests matching a keyword
```

Expected output: `5 passed`. The tests mock the model's internal estimator
(`RideDurationModel._model`) and exercise the prediction path plus threshold
clipping — mocking the estimator keeps the tests focused on
`RideDurationModel`'s own logic (delegation + clipping) rather than the math of
whichever model happens to be plugged in.

## MLOps best practices — covered here, and what's next

What this session already demonstrates, and the practices that would naturally
come next *at this session's scope* (a single model behind an API — no
pipelines or orchestration yet).

**✅ Covered in this session**

- Code as importable modules ([`src/model.py`](src/model.py)) instead of a notebook
- Declared, versioned dependencies ([`pyproject.toml`](pyproject.toml)) with optional extras
- Unit tests with pytest ([`tests/`](tests/))
- A served model behind a documented API (FastAPI/Litestar + OpenAPI)
- Request validation at the edge (pydantic `Field` constraints on `/predict`)
- Reproducible, non-root, multi-stage container build ([`Dockerfile`](Dockerfile))
- Local stack wiring with compose (API + MLflow) and a read-only model mount
- Portable model format (ONNX export)

**🔜 Next best practices at this scope**

1. **Actually load the model from `MODEL_PATH`** — compose sets it and mounts
   `models/`, but the app currently always uses the built-in heuristic. Load
   the pickle at startup (FastAPI *lifespan*), fall back to the heuristic with
   a loud log line.
2. **Response validation too** — the request schema is constrained, but
   nothing checks the *output* (a negative or `NaN` duration would ship
   happily). Add bounds to `PredictResponse` and decide what a violation
   means: `500`, or a clipped value plus a warning log.
3. **Structured logging** — read `LOG_LEVEL`, log every prediction (inputs,
   output, latency, model version). Today the app never logs anything.
4. **API tests** — the current tests only cover the model class; add
   `TestClient` tests for `/predict` (happy path + validation errors) and
   `/health`.
5. **A real health check** — `/health` should verify the model is loaded;
   add a `HEALTHCHECK` to the Dockerfile / compose so orchestrators can act
   on it. Add a `/model-info` endpoint (version, source, loaded-at).
6. **A lockfile** — `>=` bounds reproduce the *project*, not the *environment*;
   `uv lock` or `pip-compile` pins every transitive dependency so two image
   builds a month apart are identical.
7. **Run this session's tests in CI** — the repo's GitHub Actions workflow
   currently tests `session_2/` only; a paths-filtered job should run
   `session_1/` tests on every push too.
8. **Model + preprocessing saved together** — whatever transforms the features
   (scaler, encoder) must ship with the model (one pipeline artifact), or you
   get the training/serving skew shown in the bad notebook.

Later sessions pick up the bigger machinery: experiment tracking (session 2),
data/model versioning with DVC, CI/CD to a registry, and serving/scaling
(session 3).
