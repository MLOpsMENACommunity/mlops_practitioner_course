"""Checkpoints: a state_dict for known architectures, the whole module for sliced ones.

A physically pruned network no longer matches its constructor — layer widths
changed — so its weights cannot be loaded back into `PlateDetector("resnet34")`.
Those are saved as whole modules and loaded with `weights_only=False`, which is
safe only because we wrote the file ourselves. Never do that with a download.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src import config
from src.models.detector import PlateDetector
from src.models.ocr import ARCHS


def build(kind: str, arch: str) -> nn.Module:
    return PlateDetector(arch) if kind == "detector" else ARCHS[arch]()


def path_for(name: str) -> Path:
    return config.artifacts_dir() / f"{name}.pt"


def save(model: nn.Module, name: str, kind: str, arch: str, whole: bool = False, **meta: object) -> Path:
    payload: dict = {"kind": kind, "arch": arch, "meta": meta}
    if whole:
        payload["module"] = model.cpu().eval()
    else:
        payload["state_dict"] = model.state_dict()
    path = path_for(name)
    torch.save(payload, path)
    return path


def load(name: str, device: str = "cpu") -> nn.Module:
    """Load a checkpoint by name, in eval mode — dropout off, BatchNorm on running stats."""
    path = path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run: make train")
    payload = torch.load(path, map_location=device, weights_only=False)
    if "module" in payload:
        return payload["module"].to(device).eval()
    model = build(payload["kind"], payload["arch"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def meta(name: str) -> dict:
    return torch.load(path_for(name), map_location="cpu", weights_only=False)["meta"]
