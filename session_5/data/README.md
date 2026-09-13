# Data — provenance and licences

The session runs on **synthetic data by default**, so `make all` needs no download and
contains no real licence plate. A real dataset can be swapped in, and this page is the
record of which ones may be, and why the others may not.

## Default: the synthetic generator

```bash
make data                         # ANPR_PROFILE=quick by default -> data/synthetic/quick/
```

[`src/datasets/synthetic.py`](../src/datasets/synthetic.py) renders 1280x720 JPEG road scenes:

- **Scene:** sky, buildings, a road with lane markings, and 0-3 cars. Each car has one plate.
- **Plates:** white or yellow boards in five text formats (`LLL DDDD`, `LL DDDD`, ...), rotated ±6°, 70-260 px wide.
- **Distractors:**
  - coloured signs with plate-like text;
  - light bordered boards reading `TAXI`, `EXIT 24` and similar, so the detector must learn more than "light rectangle with dark letters".
- **Condition tag on every frame:** `day`, `night`, `rain`, `motion_blur` or `low_contrast`. The quantization guides slice accuracy by this tag and build calibration sets from it.
- **Font:** Pillow's bundled **Aileron Regular**, licensed **CC0** (no rights reserved), per the [Pillow ImageFont docs](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html) and [dotcolon](https://dotcolon.net/fonts/aileron). No font file is downloaded or committed.
- **Determinism:** each frame is seeded by `(SEED, split, index)`, so the set is identical however many worker processes generate it.
- **Integrity:** `manifest.json` stores a sha256 over annotations and JPEG bytes per split. The benchmark verifies the validation split before scoring and records the hash in every result row.

| Split | Purpose | quick profile | full profile |
|---|---|---|---|
| `train` | training and fine-tuning | 1600 frames | 12000 frames |
| `val` | every reported accuracy number | 240 frames | 1000 frames |
| `calib` | calibration only, uniform over conditions — never used for scoring | 640 frames | 2560 frames |

Each split also has an `*_ocr.npz` cache of grayscale plate crops, with extra margin around each crop. The recognizer's training jitters that margin, because deployed crops come from *detected* boxes.

## Real datasets considered

Checked on 2026-09-13 against each project's own licence page.

| Dataset | Licence / terms | Used here? | Why |
|---|---|---|---|
| **CCPD** ([github.com/detectRecog/CCPD](https://github.com/detectRecog/CCPD)) | MIT (README + LICENSE) | **Optional**, via [`src/datasets/ccpd.py`](../src/datasets/ccpd.py) | Permissive. Its DB, Blur, Rotate, Tilt and Challenge subsets map onto our conditions. Download it yourself; never commit or mirror images — they are real plates, which is personal data. |
| UFPR-ALPR ([licence agreement](https://github.com/raysonlaroca/ufpr-alpr-dataset/blob/master/license-agreement.md)) | Academic, non-commercial; request from a university address | No | No redistribution and no commercial use: incompatible with a repo students reuse at work |
| RodoSol-ALPR ([repo](https://github.com/raysonlaroca/rodosol-alpr-dataset)) | Academic, non-commercial; signed agreement | No | Same |
| OpenALPR benchmarks ([repo](https://github.com/openalpr/benchmarks)) | AGPL-3.0 | No | Copyleft obligations for anyone who ships derived work |
| EALPR, Egyptian plates ([repo](https://github.com/ahmedramadan96/EALPR)) | No licence file | No | No licence means all rights reserved. No permissively licensed Arabic-plate dataset was found. |

### Using CCPD

```bash
# 1. Download CCPD2019 from the links in the CCPD README and extract it.
# 2. Convert 100 frames per subset into this session's format:
.venv/bin/python -m src.datasets.ccpd --root /path/to/CCPD2019 --out data/ccpd --per-subset 100
```

The first character of a Chinese plate is a province, which this recognizer's `0-9 A-Z`
alphabet cannot represent, so the converter drops it. Say so next to any number measured on CCPD.

## Pretrained weights

**None.** Every model trains from scratch on the data above:

- **torchvision** code is BSD-3-Clause. Its [model docs](https://docs.pytorch.org/vision/stable/models.html) warn that pretrained weights "may have their own licenses or terms and conditions derived from the dataset used for training". We use the backbone *classes* with `weights=None`.
- **Ultralytics YOLO** is AGPL-3.0, and its [licence page](https://www.ultralytics.com/license) states this covers trained models. A company product built on it inherits copyleft obligations unless it buys the enterprise licence. This session's detector is written in [`src/models/detector.py`](../src/models/detector.py) instead.
