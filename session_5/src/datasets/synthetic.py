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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import config

SPLIT_IDS = {"train": 0, "val": 1, "calib": 2}
TRAIN_WEIGHTS = {"day": 0.40, "night": 0.15, "rain": 0.15, "motion_blur": 0.15, "low_contrast": 0.15}
PLATE_FORMATS = ("LLL DDDD", "LL DDDD", "DDD LLL", "L DDDDD", "LL DD LL")
LETTERS, DIGITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"
PLATE_W_MIN, PLATE_W_MAX = 44, 250  # plate width in frame pixels; at the low end the characters are ~5 px wide
OCCLUDED_FRACTION = 0.25  # plates with a tow bar or bike rack across their lower edge
DECAL_FRACTION = 0.4  # cars carrying a plate-styled dealer tag or sticker somewhere else on the body
DEALER_FRACTION = 0.15  # cars with a dealer insert in the plate holder instead of a plate
DEALER_WORDS = ("AUTO", "CARS", "MOTORS", "DRIVE", "SALES", "RENT")
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


def _dealer_text(rng: np.random.Generator) -> str:
    """Dealer name + number, or a phone number: plate-like at a distance, not a plate format up close."""
    if rng.random() < 0.5:
        return f"{DEALER_WORDS[rng.integers(len(DEALER_WORDS))]} {rng.integers(10, 100)}"
    return f"0{rng.integers(10, 100)} {rng.integers(1000, 10000)}"


def _plate_width(rng: np.random.Generator, max_w: int) -> int:
    """Log-uniform between PLATE_W_MIN and PLATE_W_MAX: far-away, small plates are the common case, as on a real road."""
    hi = max(PLATE_W_MIN + 1, min(PLATE_W_MAX, max_w))
    return int(np.exp(rng.uniform(np.log(PLATE_W_MIN), np.log(hi))))


def _render_plate(rng: np.random.Generator, text: str, condition: str, target_w: int, border: bool = True) -> Image.Image:
    font = _font(64)
    x0, y0, x1, y1 = font.getbbox(text)
    w, h = x1 - x0 + 40, 90
    if condition == "low_contrast":
        base = int(rng.integers(140, 175))
        bg, fg = (base,) * 3, (base - int(rng.integers(24, 40)),) * 3
    else:
        bg = (242, 242, 238) if rng.random() < 0.7 else (236, 200, 45)
        fg = (int(rng.integers(10, 50)),) * 3
    plate = Image.new("RGBA", (w, h), bg + (255,))
    draw = ImageDraw.Draw(plate)
    if border:
        draw.rectangle([2, 2, w - 3, h - 3], outline=fg + (255,), width=4)
    draw.text((20 - x0, (h - (y1 - y0)) // 2 - y0), text, font=font, fill=fg + (255,))
    plate = plate.resize((target_w, max(12, int(target_w * h / w))), Image.BILINEAR)
    return plate.rotate(float(rng.uniform(-7, 7)), resample=Image.BICUBIC, expand=True)


def _background(rng: np.random.Generator) -> tuple[Image.Image, int]:
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
    return img, horizon


def _lookalikes(rng: np.random.Generator, frame: Image.Image, horizon: int) -> None:
    """Boards that look like plates but are not on a car.

    Colour, border and text format alone cannot separate these from a plate, so the
    detector has to use context: a plate sits low on a car body, not on a facade.
    """
    for _ in range(int(rng.integers(1, 5))):
        kind, w = rng.random(), int(rng.integers(50, 170))
        if kind < 0.4:  # real plate styling, mounted on a building above the road
            board = _render_plate(rng, _plate_text(rng), "day", w)
            y = int(rng.integers(0, max(1, horizon - board.height)))
        elif kind < 0.7:  # borderless plate-format board at car height: a dealer or parking-bay tag
            board = _render_plate(rng, _plate_text(rng), "day", w, border=False)
            y = int(rng.integers(horizon, config.FRAME_H - board.height))
        else:  # street-name sign: plate proportions and text format, white on blue or green
            text, font = _plate_text(rng), _font(64)
            tx0, ty0, tx1, ty1 = font.getbbox(text)
            colour = (20, 70, 160) if rng.random() < 0.5 else (20, 110, 60)
            board = Image.new("RGBA", (tx1 - tx0 + 40, 90), colour + (255,))
            ImageDraw.Draw(board).text((20 - tx0, (90 - (ty1 - ty0)) // 2 - ty0), text, font=font, fill=(245, 245, 245, 255))
            board = board.resize((w, max(12, w * 90 // board.width)), Image.BILINEAR)
            y = int(rng.integers(horizon // 2, config.FRAME_H - board.height))
        x = int(rng.integers(0, config.FRAME_W - board.width))
        frame.paste(board, (x, y), board)


def _add_car(rng, frame: Image.Image, cx: int, bottom: int, plate: Image.Image, slot_w: int) -> tuple[list[int], list[int]]:
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
    return [px + ax0, py + ay0, px + ax1, py + ay1], [x0, y0, x0 + body_w, bottom]


def _decal(rng: np.random.Generator, frame: Image.Image, car: list[int], plate_box: list[int], condition: str) -> None:
    """A plate-styled tag ON the car — a dealer plate in the rear window, a bumper sticker.

    The hardest negative: same colours, same font, same text format, on a car. What
    separates it from the plate is where it sits and, often, the missing border.
    """
    x0, y0, x1, y1 = car
    pw = plate_box[2] - plate_box[0]
    tag = _render_plate(rng, _plate_text(rng), condition, max(24, int(pw * rng.uniform(0.55, 0.95))),
                        border=bool(rng.random() < 0.5))  # fmt: skip
    body_w, body_h = x1 - x0, y1 - y0
    if rng.random() < 0.5:  # rear window
        tx = int(rng.integers(x0 + body_w // 6, max(x0 + body_w // 6 + 1, x0 + 5 * body_w // 6 - tag.width)))
        ty = y0 + body_h // 10 + int(rng.integers(0, max(1, body_h * 7 // 30 - tag.height)))
    else:  # bumper corner, beside the plate
        left = rng.random() < 0.5
        tx = x0 + int(body_w * 0.06) if left else x1 - int(body_w * 0.06) - tag.width
        ty = int(rng.integers(y0 + body_h // 3, max(y0 + body_h // 3 + 1, y1 - tag.height - 2)))
    px0, py0, px1, py1 = plate_box
    if tx < px1 + 4 and px0 - 4 < tx + tag.width and ty < py1 + 4 and py0 - 4 < ty + tag.height:
        return  # never on top of the real plate
    frame.paste(tag, (tx, ty), tag)


def _occlude(rng: np.random.Generator, frame: Image.Image, box: list[int]) -> None:
    """A tow bar or bike rack across the lower edge of the plate.

    It covers the border band under the characters, never the characters themselves:
    the box label still spans the whole plate, and the text label stays readable.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    left = x0 + int(w * rng.uniform(-0.3, 0.3))
    right = left + int(w * rng.uniform(0.5, 1.3))
    top = y1 - int(h * rng.uniform(0.12, 0.26))
    shade = int(rng.integers(15, 70))
    ImageDraw.Draw(frame).rectangle([left, top, right, y1 + max(2, h // 3)], fill=(shade, shade, shade))


def _poles(rng: np.random.Generator, frame: Image.Image, horizon: int, plates: list[list[int]]) -> None:
    """Foreground posts that cut across cars — but never across a plate, whose text must stay readable."""
    draw = ImageDraw.Draw(frame)
    for _ in range(int(rng.integers(0, 4))):
        x, w = int(rng.integers(0, config.FRAME_W)), int(rng.integers(6, 18))
        if any(x - 8 <= b[2] and b[0] <= x + w + 8 for b in plates):
            continue
        shade = int(rng.integers(30, 120))
        draw.rectangle([x, int(rng.integers(0, horizon)), x + w, config.FRAME_H], fill=(shade, shade, shade + 6))


def _hblur(region: np.ndarray, k: int) -> np.ndarray:
    """Centred horizontal box blur: an object moving sideways during the exposure, box labels stay aligned."""
    pad = np.pad(region, ((0, 0), (k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    c = np.concatenate([np.zeros_like(pad[:, :1]), np.cumsum(pad, axis=1)], axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def _motion_blur(rng: np.random.Generator, img: np.ndarray, cars: list[tuple[list[int], list[int], int]]) -> np.ndarray:
    """Each car smears by its own amount; the road and buildings stay sharp.

    The blur length is a fraction of that plate's CHARACTER width. A blur longer than
    a character erases the text while the label still spells it — which is what the
    previous frame-wide 9-31 px blur did to most plates.
    """
    out = img.astype(np.float32)
    for plate_box, (cx0, cy0, cx1, cy1), n_chars in cars:
        char_w = (plate_box[2] - plate_box[0]) / (n_chars + 1.5)
        k = max(3, int(round(char_w * rng.uniform(0.25, 0.6))))
        x0, x1 = max(0, cx0 - k), min(config.FRAME_W, cx1 + k)
        y0, y1 = max(0, cy0), min(config.FRAME_H, cy1)
        out[y0:y1, x0:x1] = _hblur(out[y0:y1, x0:x1], k)
    return out


def _glow(out: np.ndarray, cx: float, cy: float, sigma: float, peak: float) -> None:
    """Add a Gaussian bloom in place, touching only the pixels within 3 sigma."""
    r = int(3 * sigma)
    x0, x1 = max(0, int(cx) - r), min(config.FRAME_W, int(cx) + r)
    y0, y1 = max(0, int(cy) - r), min(config.FRAME_H, int(cy) + r)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] += (peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))[..., None]


def _apply_condition(rng, img: np.ndarray, condition: str, boxes: list[list[int]], cars: list[list[int]]) -> np.ndarray:
    out = img.astype(np.float32)
    if condition == "night":
        lit = out.copy()
        out *= rng.uniform(0.06, 0.18)
        for x0, y0, x1, y1 in boxes:
            # Retroreflective plates under IR or headlights read brighter than anything in a
            # daytime frame; plates outside the beam are barely above the dark scene.
            gain = rng.uniform(1.1, 1.6) if rng.random() < 0.5 else rng.uniform(0.3, 0.5)
            out[y0:y1, x0:x1] = lit[y0:y1, x0:x1] * gain
        for cx0, cy0, cx1, cy1 in cars:  # lamp glare at both corners of every car
            w, h = cx1 - cx0, cy1 - cy0
            for lx in (cx0 + 0.12 * w, cx1 - 0.12 * w):
                _glow(out, lx, cy1 - 0.3 * h, w * rng.uniform(0.03, 0.06), rng.uniform(180, 320))
        out += rng.normal(0, rng.uniform(12, 26), out.shape)  # high-gain sensor noise
    elif condition == "rain":
        pil = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil, "RGBA")
        for _ in range(int(rng.integers(700, 1400))):
            x, y, length = rng.integers(0, config.FRAME_W), rng.integers(0, config.FRAME_H), int(rng.integers(18, 40))
            draw.line([(x, y), (x + length // 4, y + length)], fill=(210, 210, 225, int(rng.integers(60, 130))), width=1)
        pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.4, 1.0))))  # water on the lens
        out = np.asarray(pil, dtype=np.float32) * 0.7 + out.mean() * 0.3
    elif condition == "low_contrast":
        out = out * 0.6 + 255 * 0.4 * rng.uniform(0.5, 0.85)  # haze
    else:
        out *= rng.uniform(0.85, 1.15)
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
    frame, horizon = _background(rng)
    _lookalikes(rng, frame, horizon)
    n_cars = 0 if rng.random() < 0.08 else int(rng.integers(1, config.MAX_PLATES))
    slot_w = config.FRAME_W // max(n_cars, 1)
    plates, cars = [], []
    for i in range(n_cars):
        # A dealer insert sits in the plate holder, styled like a plate. It is NOT labelled: up close its
        # text format gives it away, far away it cannot be told apart — as for a real ANPR camera.
        dealer = rng.random() < DEALER_FRACTION
        text = _dealer_text(rng) if dealer else _plate_text(rng)
        width = _plate_width(rng, int(slot_w / 3.3))
        plate = _render_plate(rng, text, condition, width)
        cx = slot_w * i + slot_w // 2
        # Perspective: a small plate belongs to a car further up the road, nearer the horizon.
        near = (width - PLATE_W_MIN) / (PLATE_W_MAX - PLATE_W_MIN)
        bottom = int(horizon + (config.FRAME_H - 5 - horizon) * np.clip(0.45 + 0.55 * near + rng.uniform(-0.08, 0.08), 0.35, 1.0))
        box, car = _add_car(rng, frame, cx, bottom, plate, slot_w)
        cars.append((box, car, len(text)))
        if dealer:
            continue
        if rng.random() < DECAL_FRACTION:
            _decal(rng, frame, car, box, condition)
        if rng.random() < OCCLUDED_FRACTION:
            _occlude(rng, frame, box)
        plates.append({"box": box, "text": text.replace(" ", "")})
    _poles(rng, frame, horizon, [c[0] for c in cars])
    pixels = np.asarray(frame)
    if condition == "motion_blur":
        pixels = _motion_blur(rng, pixels, cars)
    # Every plate holder is lit at night, inserts included: lighting must not reveal which is which.
    pixels = _apply_condition(rng, pixels, condition, [c[0] for c in cars], [c[1] for c in cars])
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=int(rng.integers(62, 90)))
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
