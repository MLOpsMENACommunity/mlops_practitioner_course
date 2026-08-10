"""DVC stage ``prepare`` — split the raw rides CSV into train/val parquet.

Reads ``data/raw/rides.csv`` and writes ``data/processed/{train,val}.parquet``.
Run standalone (``python src/prepare.py``) or via ``dvc repro config/dvc.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config


def main() -> None:
    cfg = load_config()
    data_cfg = cfg["data"]

    df = pd.read_csv(data_cfg["raw_path"])
    train_df, val_df = train_test_split(
        df,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["seed"],
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)

    print(f"prepare: {len(train_df)} train / {len(val_df)} val rows → {out_dir}/")


if __name__ == "__main__":
    main()
