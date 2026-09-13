"""Two claims the guides make in prose, checked in code."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src import cost
from src.stages.s07_distillation import ocr_kd_loss


def test_temperature_cancels_exactly_for_mse() -> None:
    """guides/09: mse(s/T, t/T) * T^2 == mse(s, t) — a temperature on box regression is a no-op."""
    s, t = torch.randn(64, 4), torch.randn(64, 4)
    for temperature in (1.0, 2.0, 4.0, 10.0):
        scaled = F.mse_loss(s / temperature, t / temperature) * temperature**2
        assert torch.allclose(scaled, F.mse_loss(s, t), rtol=1e-5)


def test_temperature_changes_the_kd_target_when_there_is_a_softmax() -> None:
    """...whereas on per-step character logits, T genuinely reshapes the target distribution."""
    student, teacher = torch.randn(8, 32, 37), torch.randn(8, 32, 37) * 5
    assert not math.isclose(ocr_kd_loss(student, teacher, 1.0).item(), ocr_kd_loss(student, teacher, 4.0).item(), rel_tol=1e-3)
    assert ocr_kd_loss(teacher, teacher, 4.0).item() < 1e-5  # identical logits: nothing left to distill


def test_cost_per_million_formula() -> None:
    inputs = cost.CostInputs(rps=100.0, p95_ms=20.0, usd_per_hour=3.6, batch_size=4, sla_ms=30.0)
    assert math.isclose(cost.cost_per_million(inputs), 3.6 / (100 * 3600) * 1e6)


def test_cost_is_none_when_the_operating_point_breaks_the_sla() -> None:
    assert cost.cost_per_million(cost.CostInputs(rps=500.0, p95_ms=45.0, usd_per_hour=1.0, batch_size=32, sla_ms=30.0)) is None


def test_usable_point_ignores_faster_batches_that_miss_the_sla() -> None:
    curve = [{"batch": 1, "fps": 40, "p95_ms": 25}, {"batch": 8, "fps": 120, "p95_ms": 80}]
    assert cost.usable_point(curve, sla_ms=30)["batch"] == 1


def test_fleet_rounds_instances_up() -> None:
    assert cost.fleet(cameras=10, camera_fps=5, capacity_fps=12, usd_per_hour=1.0)["instances"] == 5
