# tests/test_prepare.py
import os

import pandas as pd

from src import prepare


def test_prepare_main_writes_train_val_parquet(tmp_path, monkeypatch):
    # prepare.main() resolves paths relative to the cwd, so run it inside a
    # throwaway project tree seeded with a config + raw CSV.
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)

    (tmp_path / "config" / "config.yaml").write_text(
        "data:\n  raw_path: data/raw/rides.csv\n  test_size: 0.2\n  seed: 42\n"
    )
    raw = pd.DataFrame(
        {
            "distance_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "passengers": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            "duration_min": [5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0],
        }
    )
    raw.to_csv(tmp_path / "data" / "raw" / "rides.csv", index=False)

    monkeypatch.chdir(tmp_path)
    prepare.main()

    train = pd.read_parquet(tmp_path / "data" / "processed" / "train.parquet")
    val = pd.read_parquet(tmp_path / "data" / "processed" / "val.parquet")

    # 80/20 split of 10 rows, and no rows lost or duplicated.
    assert len(train) == 8
    assert len(val) == 2
    assert len(train) + len(val) == len(raw)
    assert list(train.columns) == list(raw.columns)


def test_prepare_main_is_runnable_repeatedly(tmp_path, monkeypatch):
    # A second run must overwrite cleanly (mkdir exist_ok=True path).
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "config" / "config.yaml").write_text(
        "data:\n  raw_path: data/raw/rides.csv\n  test_size: 0.5\n  seed: 0\n"
    )
    pd.DataFrame(
        {"distance_km": [1.0, 2.0], "passengers": [1, 2], "duration_min": [5.0, 8.0]}
    ).to_csv(tmp_path / "data" / "raw" / "rides.csv", index=False)

    monkeypatch.chdir(tmp_path)
    prepare.main()
    prepare.main()  # should not raise

    assert os.path.exists(tmp_path / "data" / "processed" / "train.parquet")
