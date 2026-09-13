"""Optional: convert CCPD (real Chinese parking-lot plates, MIT licence) to this session's format.

CCPD is not downloaded by `make all` — it is several GB, hosted on Google Drive and
BaiduYun, and every image is a real plate. Fetch it yourself from the links in
https://github.com/detectRecog/CCPD, then:

    python -m src.datasets.ccpd --root /path/to/CCPD2019 --out data/ccpd

CCPD encodes the annotation in the filename, dash-separated:
    area - tilt - bbox(x1&y1_x2&y2) - four corners - plate character indices - brightness - blur

The first plate character is a Chinese province. This session's recognizer only
knows 0-9 A-Z, so the province is dropped and the remaining six characters are kept —
a limitation to state, not hide, in any result measured on CCPD.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Alphabet tables from the CCPD README (the province table is not needed: it is dropped).
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
             "X", "Y", "Z", "O"]  # fmt: skip
ADS = [*ALPHABETS[:-1], "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

# CCPD's own test subsets, mapped onto this session's condition tags (approximate by design).
CONDITION = {"ccpd_base": "day", "ccpd_db": "night", "ccpd_weather": "rain", "ccpd_blur": "motion_blur",
             "ccpd_challenge": "low_contrast", "ccpd_fn": "day", "ccpd_rotate": "day", "ccpd_tilt": "day"}  # fmt: skip


def parse(filename: str) -> dict:
    """One CCPD filename -> {"box": [x0, y0, x1, y1], "text": "A12345"} (province dropped)."""
    fields = Path(filename).stem.split("-")
    (x0, y0), (x1, y1) = (map(int, p.split("&")) for p in fields[2].split("_"))
    idx = [int(i) for i in fields[4].split("_")]
    return {"box": [x0, y0, x1, y1], "text": ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])}


def convert(root: Path, out: Path, per_subset: int) -> None:
    (out / "val").mkdir(parents=True, exist_ok=True)
    with open(out / "val.jsonl", "w") as ann:
        n = 0
        for subset, condition in CONDITION.items():
            for image in sorted((root / subset).glob("*.jpg"))[:per_subset]:
                name = f"{n:06d}.jpg"
                shutil.copy(image, out / "val" / name)
                ann.write(json.dumps({"file": name, "condition": condition, "plates": [parse(image.name)]}) + "\n")
                n += 1
    print(f"wrote {n} CCPD frames to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/ccpd"))
    parser.add_argument("--per-subset", type=int, default=100)
    args = parser.parse_args()
    convert(args.root, args.out, args.per_subset)


if __name__ == "__main__":
    main()
