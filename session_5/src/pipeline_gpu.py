"""The same pipeline with decode and resize moved onto the GPU (CUDA only).

On a GPU box the CPU-side JPEG decode and resize can cost more than the model.
torchvision's decode_jpeg(device="cuda") runs nvJPEG on a whole batch at once, and
the letterbox becomes an interpolate + pad on tensors that are already on the device.
Crops still go back to the CPU, and that copy is counted in the crop phase.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg

from src import config
from src.pipeline import Pipeline
from src.preprocess import Letterbox, crop_plates


class GpuPipeline(Pipeline):
    def decode(self, jpegs: list[bytes]) -> list[torch.Tensor]:
        data = [torch.frombuffer(bytearray(j), dtype=torch.uint8) for j in jpegs]
        return decode_jpeg(data, device="cuda")  # batched nvJPEG: one call for the whole batch

    def preprocess(self, frames: list[torch.Tensor]) -> tuple[torch.Tensor, list[Letterbox]]:
        h, w = frames[0].shape[1:]
        scale = min(config.DET_W / w, config.DET_H / h)
        nw, nh = round(w * scale), round(h * scale)
        px, py = (config.DET_W - nw) // 2, (config.DET_H - nh) // 2
        x = F.interpolate(torch.stack(frames).float(), size=(nh, nw), mode="bilinear", align_corners=False)
        x = F.pad(x, (px, config.DET_W - nw - px, py, config.DET_H - nh - py), value=114.0) / 255.0
        return x, [Letterbox(scale, px, py)] * len(frames)

    def crop(self, frames, dets, lbs) -> np.ndarray:
        crops = [crop_plates(f.permute(1, 2, 0).cpu().numpy(), d, lb) for f, d, lb in zip(frames, dets, lbs)]
        return np.concatenate(crops) if crops else np.zeros((0, 1, config.OCR_H, config.OCR_W), np.float32)
