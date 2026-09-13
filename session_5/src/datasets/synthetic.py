"""Synthetic roadside frames with licence plates: the zero-download dataset.

Why synthetic is the default: the real ALPR datasets are academic-only
(UFPR-ALPR, RodoSol-ALPR), copyleft (OpenALPR benchmarks) or unlicensed (EALPR),
and every real plate is personal data. See data/README.md for the licences.

Every frame is tagged with the condition it was rendered under. That tag is what
lets the quantization guides slice accuracy by condition and build a calibration
set that covers night, rain, motion blur and low contrast — or deliberately
doesn't.

    python -m src.datasets.synthetic            # uses ANPR_PROFILE (default quick)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config

SPLIT_IDS = {"train": 0, "val": 1, "calib": 2}
TRAIN_WEIGHTS = {"day": 0.40, "night": 0.15, "rain": 0.15, "motion_blur": 0.15, "low_contrast": 0.15}
PLATE_FORMATS = ("LLL DDDD", "LL DDDD", "DDD LLL", "L DDDDD", "LL DD LL")
LETTERS, DIGITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"
CROP_CACHE_H, CROP_CACHE_W = 64, 256  # OCR crops are cached larger, jittered at train time
CROP_MARGIN = 0.25  # context kept around each cached crop so the loader can jitter


@lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled Aileron (CC0) — no font file to download or license.
    return ImageFont.load_default(size=size)


def _rng(split: str, idx: int) -> np.random.Generator:
    # Seeded per (split, index), so a frame is identical no matter how many
    # worker processes generated the set or in what order.
    return np.random.default_rng([config.SEED, SPLIT_IDS[split], idx])


def _plate_text(rng: np.random.Generator) -> str:
    fmt = PLATE_FORMATS[rng.integers(len(PLATE_FORMATS))]
    pick = {"L": LETTERS, "D": DIGITS}
    return "".join(c if c == " " else pick[c][rng.integers(len(pick[c]))] for c in fmt)


def _render_plate(rng: np.random.Generator, text: str, condition: str, max_w: int) -> Image.Image:
    font = _font(64)
    x0, y0, x1, y1 = font.getbbox(text)
    w, h = x1 - x0 + 40, 90
    if condition == "low_contrast":
        base = int(rng.integers(140, 175))
        bg, fg = (base,) * 3, (base - int(rng.integers(28, 45)),) * 3
    else:
        bg = (242, 242, 238) if rng.random() < 0.7 else (236, 200, 45)
        fg = (int(rng.integers(10, 50)),) * 3
    plate = Image.new("RGBA", (w, h), bg + (255,))
    draw = ImageDraw.Draw(plate)
    draw.rectangle([2, 2, w - 3, h - 3], outline=fg + (255,), width=4)
    draw.text((20 - x0, (h - (y1 - y0)) // 2 - y0), text, font=font, fill=fg + (255,))
    target_w = int(rng.integers(min(70, max_w - 1), min(260, max_w)))  # down to 70 px: a car further away
    plate = plate.resize((target_w, max(18, int(target_w * h / w))), Image.BILINEAR)
    return plate.rotate(float(rng.uniform(-6, 6)), resample=Image.BICUBIC, expand=True)


def _background(rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (config.FRAME_W, config.FRAME_H))
    draw = ImageDraw.Draw(img)
    horizon = int(config.FRAME_H * rng.uniform(0.3, 0.42))
    sky = rng.integers(120, 230, size=3)
    for y in range(horizon):
        shade = tuple(int(c * (0.75 + 0.25 * y / horizon)) for c in sky)
        draw.line([(0, y), (config.FRAME_W, y)], fill=shade)
    for _ in range(int(rng.integers(3, 9))):  # buildings: large flat distractors
        x, bw = int(rng.integers(0, config.FRAME_W)), int(rng.integers(80, 300))
        top = int(rng.integers(horizon // 3, horizon))
        draw.rectangle([x, top, x + bw, horizon], fill=tuple(int(v) for v in rng.integers(40, 200, 3)))
    road = int(rng.integers(70, 115))
    draw.rectangle([0, horizon, config.FRAME_W, config.FRAME_H], fill=(road, road, road + 4))
    vx = config.FRAME_W // 2 + int(rng.integers(-150, 150))
    for lane_x in (-0.6, 0.0, 0.6, 1.2):  # dashed lane markings toward a vanishing point
        bx = int(config.FRAME_W * (0.5 + lane_x))

        def point(s: float, bx: int = bx) -> tuple[float, float]:
            return vx + (bx - vx) * s, horizon + (config.FRAME_H - horizon) * s

        for t in np.arange(0.05, 1.0, 0.12):
            draw.line([point(t), point(t + 0.05)], fill=(230, 230, 210), width=max(1, int(8 * t)))
    for _ in range(int(rng.integers(0, 3))):  # text signs: letters that are NOT plates
        sx, sy = int(rng.integers(0, config.FRAME_W - 260)), int(rng.integers(20, horizon))
        colour = tuple(int(v) for v in rng.integers(0, 160, 3))
        draw.rectangle([sx, sy, sx + 240, sy + 70], fill=colour)
        draw.text((sx + 12, sy + 10), _plate_text(rng), font=_font(44), fill=(250, 250, 250))
    for _ in range(int(rng.integers(0, 3))):  # plate-LIKE distractors: light board, dark border, dark text
        words = ("TAXI", "STOP", "EXIT 24", "BUS 12", "PARK", "KM 40", "NO ENTRY")
        text, size = words[rng.integers(len(words))], int(rng.integers(18, 40))
        bw, bh = int(_font(size).getlength(text)) + 16, int(size * 1.4)
        sx, sy = int(rng.integers(0, config.FRAME_W - bw)), int(rng.integers(horizon // 2, config.FRAME_H - bh))
        draw.rectangle([sx, sy, sx + bw, sy + bh], fill=(238, 238, 232), outline=(30, 30, 30), width=3)
        draw.text((sx + 8, sy + bh // 6), text, font=_font(size), fill=(25, 25, 25))
    return img


def _add_car(rng, frame: Image.Image, cx: int, bottom: int, plate: Image.Image, slot_w: int) -> list[int]:
    draw = ImageDraw.Draw(frame)
    # Capped to the car's own slot: a car overlapping its neighbour hides part of
    # a plate whose label still spells the full text — an impossible OCR target.
    body_w = min(int(plate.width * rng.uniform(3.0, 4.0)), int(slot_w * 0.96))
    body_h = int(body_w * rng.uniform(0.5, 0.65))
    colour = tuple(int(v) for v in rng.integers(20, 235, 3))
    x0, y0 = cx - body_w // 2, bottom - body_h
    draw.rounded_rectangle([x0, y0, x0 + body_w, bottom], radius=body_w // 10, fill=colour)
    glass = tuple(int(c * 0.35) for c in colour)
    draw.rectangle([x0 + body_w // 6, y0 + body_h // 10, x0 + 5 * body_w // 6, y0 + body_h // 3], fill=glass)
    px = cx - plate.width // 2
    py = bottom - int(body_h * 0.18) - plate.height
    frame.paste(plate, (px, py), plate)
    ax0, ay0, ax1, ay1 = plate.getchannel("A").getbbox()  # tight box of the rotated plate
    return [px + ax0, py + ay0, px + ax1, py + ay1]


def _apply_condition(rng, img: np.ndarray, condition: str, boxes: list[list[int]]) -> np.ndarray:
    out = img.astype(np.float32)
    if condition == "night":
        lit = out.copy()
        out *= rng.uniform(0.15, 0.3)
        for x0, y0, x1, y1 in boxes:  # plates are lamp-lit, so darker but readable
            out[y0:y1, x0:x1] = lit[y0:y1, x0:x1] * rng.uniform(0.45, 0.65)
        out += rng.normal(0, 10, out.shape)
    elif condition == "rain":
        pil = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil, "RGBA")
        for _ in range(400):
            x, y = rng.integers(0, config.FRAME_W), rng.integers(0, config.FRAME_H)
            draw.line([(x, y), (x + 6, y + 26)], fill=(210, 210, 225, 90), width=1)
        out = np.asarray(pil, dtype=np.float32) * 0.8 + out.mean() * 0.2
    elif condition == "motion_blur":
        k = int(rng.integers(9, 31))  # horizontal box blur, via a cumulative sum
        c = np.cumsum(np.pad(out, ((0, 0), (k, 0), (0, 0)), mode="edge"), axis=1)
        out = (c[:, k:] - c[:, :-k]) / k
    elif condition == "low_contrast":
        out = out * 0.65 + 255 * 0.35 * rng.uniform(0.5, 0.8)  # haze
    else:
        out *= rng.uniform(0.9, 1.1)
    return out.clip(0, 255).astype(np.uint8)


def _ocr_crop(frame: np.ndarray, box: list[int]) -> tuple[np.ndarray, list[float]]:
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    cx1, cy1 = min(config.FRAME_W, int(x1 + mx)), min(config.FRAME_H, int(y1 + my))
    crop = Image.fromarray(frame[cy0:cy1, cx0:cx1]).convert("L")
    inner = [(x0 - cx0) / (cx1 - cx0), (y0 - cy0) / (cy1 - cy0), (x1 - cx0) / (cx1 - cx0), (y1 - cy0) / (cy1 - cy0)]
    return np.asarray(crop.resize((CROP_CACHE_W, CROP_CACHE_H), Image.BILINEAR)), inner


def render_frame(split: str, idx: int) -> dict:
    """Render one frame; return its JPEG bytes, annotation and cached OCR crops."""
    rng = _rng(split, idx)
    if split == "calib":  # uniform, so every condition has calibration candidates
        condition = config.CONDITIONS[idx % len(config.CONDITIONS)]
    else:
        condition = str(rng.choice(list(TRAIN_WEIGHTS), p=list(TRAIN_WEIGHTS.values())))
    frame = _background(rng)
    n_cars = 0 if rng.random() < 0.08 else int(rng.integers(1, config.MAX_PLATES))
    slot_w = config.FRAME_W // max(n_cars, 1)
    plates = []
    for i in range(n_cars):
        text = _plate_text(rng)
        plate = _render_plate(rng, text, condition, max_w=int(slot_w / 3.3))
        cx = slot_w * i + slot_w // 2
        bottom = int(rng.integers(int(config.FRAME_H * 0.62), config.FRAME_H - 5))
        box = _add_car(rng, frame, cx, bottom, plate, slot_w)
        plates.append({"box": box, "text": text.replace(" ", "")})
    pixels = _apply_condition(rng, np.asarray(frame), condition, [p["box"] for p in plates])
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=int(rng.integers(75, 93)))
    crops = [_ocr_crop(pixels, p["box"]) for p in plates]
    return {"file": f"{idx:06d}.jpg", "condition": condition, "plates": plates, "jpeg": buf.getvalue(), "crops": crops}


def _render(args: tuple[str, int]) -> dict:
    return render_frame(*args)


def generate(split: str, n: int, workers: int = 8) -> str:
    """Write one split to disk and return its content hash."""
    root = config.data_dir()
    (root / split).mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    crops, inner, texts, conds = [], [], [], []
    with Pool(workers) as pool, open(root / f"{split}.jsonl", "w") as ann:
        for rec in pool.imap(_render, [(split, i) for i in range(n)], chunksize=8):
            (root / split / rec["file"]).write_bytes(rec["jpeg"])
            line = json.dumps({k: rec[k] for k in ("file", "condition", "plates")})
            ann.write(line + "\n")
            digest.update(line.encode())
            digest.update(rec["jpeg"])
            for (crop, box), plate in zip(rec["crops"], rec["plates"]):
                crops.append(crop)
                inner.append(box)
                texts.append(plate["text"])
                conds.append(rec["condition"])
    np.savez_compressed(
        root / f"{split}_ocr.npz",
        crops=np.stack(crops) if crops else np.zeros((0, CROP_CACHE_H, CROP_CACHE_W), np.uint8),
        inner=np.asarray(inner, np.float32).reshape(-1, 4),
        texts=np.asarray(texts),
        conditions=np.asarray(conds),
    )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prof = config.profile()
    sizes = {"train": prof.n_train, "val": prof.n_val, "calib": prof.n_calib * len(config.CONDITIONS)}
    manifest = {"profile": prof.name, "seed": config.SEED, "splits": {}}
    for split, n in sizes.items():
        manifest["splits"][split] = {"n": n, "sha256": generate(split, n, args.workers)}
        print(f"  {split:<6} {n:>6} frames  sha256 {manifest['splits'][split]['sha256'][:12]}")
    (config.data_dir() / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {config.data_dir()}")


if __name__ == "__main__":
    main()
