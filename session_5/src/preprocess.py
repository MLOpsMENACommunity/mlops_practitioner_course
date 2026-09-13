"""Everything between the camera's JPEG and the model's tensor — and back to plate crops.

Deliberately torch-free (numpy + Pillow): the ONNX Runtime, OpenVINO and TFLite
pipelines import this too, and importing torch just to resize an image would add
hundreds of MB to the peak-memory number of a stage that never runs a torch op.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src import config


@dataclass(frozen=True)
class Letterbox:
    """How a frame was fitted into the detector input, so boxes can be mapped back."""

    scale: float
    pad_x: int
    pad_y: int


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> HxWx3 uint8 RGB."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# --- snippet:letterbox ---
def letterbox(frame: np.ndarray, fast: bool = False) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio, pad to DET_H x DET_W with grey (114)."""
    h, w = frame.shape[:2]
    scale = min(config.DET_W / w, config.DET_H / h)
    nw, nh = round(w * scale), round(h * scale)
    img = Image.fromarray(frame)
    if fast and w % nw == 0 and h % nh == 0:
        img = img.reduce(w // nw)  # exact integer factor: a box average, no resampling kernel
    else:
        img = img.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((config.DET_H, config.DET_W, 3), 114, np.uint8)
    px, py = (config.DET_W - nw) // 2, (config.DET_H - nh) // 2
    canvas[py : py + nh, px : px + nw] = np.asarray(img)
    return canvas, Letterbox(scale, px, py)


# --- end-snippet ---


def to_nchw(images: list[np.ndarray]) -> np.ndarray:
    """uint8 HWC images -> float32 NCHW in [0, 1]. No mean/std: keeps INT8 ranges simple."""
    return np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def crop_plates(frame: np.ndarray, dets: np.ndarray, lb: Letterbox, margin: float = 0.08) -> np.ndarray:
    """Cut each detected plate out of the FULL-RESOLUTION frame -> [N,1,32,128] float32.

    Cropping from the original frame rather than the downscaled detector input is
    why a small plate is still readable: the detector saw it at half resolution,
    the recognizer does not have to.
    """
    crops = []
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in dets[:, :4]:
        x0, x1 = (x0 - lb.pad_x) / lb.scale, (x1 - lb.pad_x) / lb.scale
        y0, y1 = (y0 - lb.pad_y) / lb.scale, (y1 - lb.pad_y) / lb.scale
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        box = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx) + 1), min(h, int(y1 + my) + 1))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            box = (0, 0, 4, 4)  # degenerate detection: keep the batch shape, it will read as ""
        crop = Image.fromarray(frame[box[1] : box[3], box[0] : box[2]]).convert("L")
        crops.append(np.asarray(crop.resize((config.OCR_W, config.OCR_H), Image.BILINEAR)))
    if not crops:
        return np.zeros((0, 1, config.OCR_H, config.OCR_W), np.float32)
    return np.stack(crops)[:, None].astype(np.float32) / 255.0
