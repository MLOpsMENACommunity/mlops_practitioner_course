"""Seed local training data so `src/train.py` can run without the GCS extract.

`train.py:main()` reads ``data/processed/train.parquet``. In session 2 that file
is produced by a DVC prepare stage; here we regenerate an equivalent parquet from
the same synthetic generator used by ``train.generate_data`` so the Airflow
``train`` task is runnable on a laptop with no cloud dependencies.

Run inside the Airflow container (cwd = /opt/airflow):
    python src/prepare.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.train import FEATURES, TARGET, generate_data


def main() -> None:
    """Write ``data/processed/train.parquet`` from the synthetic generator."""
    cfg = load_config()
    data_cfg = cfg.get("data", {})

    X, y = generate_data(
        n_samples=data_cfg.get("n_samples", 2000),
        noise=data_cfg.get("noise", 1.0),
        seed=data_cfg.get("seed", 42),
    )

    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = y

    out = Path("data/processed/train.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"prepare: wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
