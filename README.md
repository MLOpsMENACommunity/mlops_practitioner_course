# mlops_sessions_iti

MLOps course work, organized one folder per session.

## Structure

```
mlops_sessions/
├── session_1/        # Ride Duration API (FastAPI + Litestar) & PyTorch→ONNX export
└── ...               # future sessions
```

Each session is self-contained — see the `README.md` inside each session
folder for setup and usage.

## Pre-commit hooks

The repo uses [pre-commit](https://pre-commit.com) to run automated checks
before a commit (and, for the test suite, before a push). The config
(`.pre-commit-config.yaml`) lives at the **git root**, so it covers every
session. If a hook fails or modifies a file, the commit is aborted so you can
review, re-stage, and try again — bad code never reaches history.

### Enable it (once per clone)

```bash
pip install pre-commit
pre-commit install                        # hooks run on `git commit`
pre-commit install --hook-type pre-push   # also run the pre-push hooks
pre-commit run --all-files                # run everything now, across the repo
```

### What the hooks do

There are two speed tiers: fast static checks run on **every commit**, while
the slow test suite runs only on **push** — so per-commit feedback stays fast
but broken tests can't reach the remote.

**Ruff — linter + formatter** (Rust-fast; replaces flake8 + Black + isort)

| Hook | Purpose |
|------|---------|
| `ruff` (`--fix`) | Flags unused imports, bad style, likely bugs; auto-fixes what it safely can. Commit aborts if it changed files, so you review + re-stage. |
| `ruff-format` | Reflows code to a consistent style (indentation, quotes, line length). |

**Generic hygiene hooks** (from the pre-commit project)

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Strip trailing spaces at line ends. |
| `end-of-file-fixer` | Ensure files end with exactly one newline. |
| `mixed-line-ending` (`--fix=lf`) | Normalise CRLF/LF so Windows/mac edits agree. |
| `check-yaml` / `check-toml` / `check-json` | Validate config files parse — fail early, not at runtime. |
| `check-merge-conflict` | Block commits that still contain `<<<<<<<` markers. |
| `check-added-large-files` (`--maxkb=500`) | Stop big blobs entering git — the DVC guardrail: data belongs in DVC, not git. |
| `check-ast` | Confirm every `.py` file parses (valid syntax). |
| `debug-statements` | Catch leftover `breakpoint()` / `pdb.set_trace()`. |
| `check-executables-have-shebangs` | An executable (+x) file must have a shebang. |
| `detect-private-key` | Block accidentally committing an SSH/PEM key. |

**Local pytest hook** (runs at the `pre-push` stage only)

Actually executes the `session_2` test suite (`cd session_2 && pytest`) to
verify behaviour. Because it's slower, it runs on `git push` — not on every
commit — and always runs the whole suite. Requires the dev deps:
`pip install -e ".[dev]"`.

### Maintenance

Bump the pinned hook versions anytime with `pre-commit autoupdate`.
