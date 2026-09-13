"""torch Datasets for training. Inference never imports this file (see preprocess.py)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config
from src.datasets.splits import gt_crop, load_split
from src.postprocess import encode_text
from src.preprocess import decode_jpeg, letterbox, to_nchw


class DetectionDataset(Dataset):
    """Frame -> (letterboxed tensor [3,384,640], plate boxes [G,4] in the same pixel space)."""

    def __init__(self, split: str, augment: bool = False) -> None:
        self.samples, self.augment = load_split(split), augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[i]
        canvas, lb = letterbox(decode_jpeg(sample.path.read_bytes()))
        x = torch.from_numpy(to_nchw([canvas])[0])
        if self.augment:  # photometric only: geometry would have to move the boxes too
            x = ((x - 0.5) * np.random.uniform(0.7, 1.3) + 0.5 + np.random.uniform(-0.1, 0.1)).clamp(0, 1)
        offset = torch.tensor([lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y], dtype=torch.float32)
        boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4) * lb.scale + offset
        return x, boxes


def detection_collate(batch: list) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class OCRDataset(Dataset):
    """Cached plate crop -> ([1,32,128] tensor, target indices), with crop-margin jitter.

    The jitter is the point: at inference the crop comes from a *detected* box, which
    is never exactly the ground truth. A recognizer trained only on perfect crops
    reads perfect crops.
    """

    def __init__(self, split: str, augment: bool = False) -> None:
        cache = np.load(config.data_dir() / f"{split}_ocr.npz")
        self.crops, self.inner = cache["crops"], cache["inner"]
        self.texts, self.conditions = cache["texts"], cache["conditions"]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.texts)

    def crop(self, i: int) -> np.ndarray:
        margins = np.random.uniform(0.0, 0.16, 4) if self.augment else np.full(4, 0.08)
        return gt_crop(self.crops[i], self.inner[i], margins)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.crop(i)
        if self.augment:
            x = (x - x.mean()) * np.random.uniform(0.6, 1.4) + x.mean() + np.random.uniform(-0.15, 0.15)
            x = np.clip(x + np.random.normal(0, 0.03, x.shape), 0, 1).astype(np.float32)
        return torch.from_numpy(x)[None], torch.tensor(encode_text(str(self.texts[i])))


def ocr_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = [b[1] for b in batch]
    return torch.stack([b[0] for b in batch]), torch.cat(targets), torch.tensor([len(t) for t in targets])
