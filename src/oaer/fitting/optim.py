"""Optimiser and schedule.

A linear warm-up followed by a cosine decay to a floor, both counted in optimizer
steps rather than epochs so the schedule is independent of the batch size. The
schedule is a declared engineering default: the article reports the model's
read-outs, not its schedule.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ScheduleConfig:
    """Warm-up length, total steps and the decay floor."""

    warmup_steps: int = 50
    total_steps: int = 3000
    floor: float = 0.02

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("the warm-up length cannot be negative")
        if self.total_steps < 1:
            raise ValueError("the schedule needs at least one step")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("the decay floor must lie in the unit interval")


def learning_rate_at(step: int, base: float, config: ScheduleConfig) -> float:
    """The learning rate at one optimizer step."""

    if step < 0:
        raise ValueError("a step index cannot be negative")
    if config.warmup_steps and step < config.warmup_steps:
        return base * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (config.floor + (1.0 - config.floor) * cosine)


class WarmupCosine:
    """A step-indexed schedule object that a loop can advance."""

    def __init__(
        self, optimiser: torch.optim.Optimizer, base: float, config: ScheduleConfig
    ) -> None:
        self.optimiser = optimiser
        self.base = float(base)
        self.config = config
        self.step = 0
        self._apply()

    def _apply(self) -> None:
        rate = learning_rate_at(self.step, self.base, self.config)
        for group in self.optimiser.param_groups:
            group["lr"] = rate

    def advance(self) -> float:
        self.step += 1
        self._apply()
        return self.current()

    def current(self) -> float:
        return learning_rate_at(self.step, self.base, self.config)


def build_optimiser(
    parameters: Iterable[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW over the parameters that require a gradient."""

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("no parameter requires a gradient")
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def backpropagate(loss: Tensor) -> None:
    """Run the backward pass through the autograd engine.

    ``Tensor.backward`` carries no annotation in the shipped stubs, so the engine
    entry point is called directly and the pass stays typed.
    """

    torch.autograd.backward(loss)


def clip_gradients(model: nn.Module, maximum: float) -> float:
    """Clip the global gradient norm and return the norm before clipping."""

    if maximum <= 0.0:
        raise ValueError("the clipping threshold must be positive")
    norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), maximum)
    return float(norm)


def parameter_groups(model: nn.Module) -> dict[str, int]:
    """Count the trainable and frozen parameters, for the audit block."""

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    return {"trainable": int(trainable), "frozen": int(frozen)}
