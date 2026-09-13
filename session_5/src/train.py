"""Training loops for both models. Stages reuse them; the CLI trains the baselines.

    python -m src.train detector --arch resnet34 --name detector_baseline
    python -m src.train ocr --arch crnn --name ocr_baseline

`extra_loss` is how pruning, QAT and distillation change what is optimized without
copying the loop: each stage passes a function that adds its own term.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src import config
from src.datasets.torch_data import DetectionDataset, OCRDataset, detection_collate, ocr_collate
from src.losses import detector_loss, ocr_loss
from src.models import io
from src.postprocess import ctc_decode, greedy_nms


def train_device() -> torch.device:
    """cuda > mps > cpu, overridable with ANPR_TRAIN_DEVICE. Training only: benchmarks pick their own."""
    if name := os.environ.get("ANPR_TRAIN_DEVICE"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _loader(dataset, batch: int, shuffle: bool, collate) -> DataLoader:
    workers = min(6, os.cpu_count() or 1)
    return DataLoader(dataset, batch, shuffle=shuffle, num_workers=workers, collate_fn=collate, persistent_workers=workers > 0)


def _schedule(opt: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, total_steps // 20)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))
    )


def fit_detector(
    model: nn.Module,
    epochs: int,
    lr: float = 2e-3,
    extra_loss: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    tag: str = "detector",
) -> nn.Module:
    """Train a PlateDetector. `extra_loss(x, features, obj, box)` adds a stage-specific term."""
    dev = train_device()
    model.to(dev).train()
    loader = _loader(DetectionDataset("train", augment=True), 16, True, detection_collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = _schedule(opt, epochs * len(loader))
    for epoch in range(epochs):
        start, total = time.perf_counter(), 0.0
        for x, gts in loader:
            x = x.to(dev)
            feats, obj, box = model.forward_with_features(x)
            loss = detector_loss(model, obj, box, gts)
            if extra_loss is not None:
                loss = loss + extra_loss(x, feats, obj, box)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  [{tag}] epoch {epoch + 1}/{epochs}  loss {total / len(loader):.4f}  {time.perf_counter() - start:.0f}s", flush=True)
    return model.eval()


def fit_ocr(
    model: nn.Module,
    epochs: int,
    lr: float = 2e-3,
    extra_loss: Callable[[Tensor, Tensor], Tensor] | None = None,
    tag: str = "ocr",
) -> nn.Module:
    """Train a recognizer with CTC. `extra_loss(x, logits)` adds a stage-specific term."""
    dev = train_device()
    model.to(dev).train()
    # Batch 32, not 128: CTC sits on a "predict all blanks" plateau for the first
    # ~800 optimizer steps, and at 128 a small dataset gives too few steps to leave it.
    loader = _loader(OCRDataset("train", augment=True), 32, True, ocr_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = _schedule(opt, epochs * len(loader))
    for epoch in range(epochs):
        start, total = time.perf_counter(), 0.0
        for x, targets, lengths in loader:
            x = x.to(dev)
            logits = model(x)
            loss = ocr_loss(logits, targets, lengths)
            if extra_loss is not None:
                loss = loss + extra_loss(x, logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # CTC + LSTM: the occasional huge gradient
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  [{tag}] epoch {epoch + 1}/{epochs}  loss {total / len(loader):.4f}  {time.perf_counter() - start:.0f}s", flush=True)
    return model.eval()


@torch.no_grad()
def quick_map(model: nn.Module, split: str = "val") -> float:
    """Training-time sanity check only. Reported accuracy comes from src/benchmark.py."""
    from src.metrics import FrameResult, average_precision

    dev = next(model.parameters()).device
    model.eval()
    frames = []
    for x, gts in _loader(DetectionDataset(split), 16, False, detection_collate):
        scores, boxes = model.decode(*model(x.to(dev)))
        for s, b, g in zip(scores.cpu().numpy(), boxes.cpu().numpy(), gts):
            frames.append(FrameResult("", g.numpy(), [], greedy_nms(s, b)))
    return average_precision(frames)


@torch.no_grad()
def quick_ocr_accuracy(model: nn.Module, split: str = "val") -> float:
    """Exact match on ground-truth crops: the recognizer alone, detector errors excluded."""
    dev = next(model.parameters()).device
    model.eval()
    data = OCRDataset(split)
    hits = 0
    for start in range(0, len(data), 256):
        idx = range(start, min(start + 256, len(data)))
        x = torch.from_numpy(np.stack([data.crop(i) for i in idx]))[:, None].to(dev)
        hits += sum(p == str(data.texts[i]) for p, i in zip(ctc_decode(model(x).cpu().numpy()), idx))
    return hits / max(len(data), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline model.")
    parser.add_argument("kind", choices=["detector", "ocr"])
    parser.add_argument("--arch", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    torch.manual_seed(config.SEED)
    prof = config.profile()
    print(f"training {args.name} ({args.arch}) on {train_device()}, profile={prof.name}")
    start = time.perf_counter()
    if args.kind == "detector":
        model = fit_detector(io.build("detector", args.arch), args.epochs or prof.det_epochs, tag=args.name)
        score = {"val_map50_quick": round(quick_map(model), 4)}
    else:
        model = fit_ocr(io.build("ocr", args.arch), args.epochs or prof.ocr_epochs, tag=args.name)
        score = {"val_gt_crop_exact_match": round(quick_ocr_accuracy(model), 4)}
    minutes = round((time.perf_counter() - start) / 60, 1)
    path = io.save(model, args.name, args.kind, args.arch, train_device=str(train_device()), train_minutes=minutes, **score)
    print(f"saved {path}  {score}  ({minutes} min)")


if __name__ == "__main__":
    main()
