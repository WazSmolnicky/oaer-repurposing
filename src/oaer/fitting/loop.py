"""The training loop.

One step is: take a batch, run the forward pass, form the loss, run the backward
pass, clip, step the optimiser and advance the schedule. Every step appends a
record with its loss, its gradient norm and its learning rate, so the loop's
trajectory is inspectable rather than inferred. Early stopping reads an inner
split of the training folds and never the reported out-of-fold set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from oaer.fitting.optim import (
    ScheduleConfig,
    WarmupCosine,
    backpropagate,
    build_optimiser,
    clip_gradients,
)
from oaer.support.seeding import SeedState, set_seed, spawn_generator


@dataclass(frozen=True)
class TrainingConfig:
    """The loop's own settings."""

    epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    warmup_steps: int = 50
    total_steps: int = 3000
    max_steps: int = 0
    log_every: int = 50
    seed: int = 20260831
    early_stop_patience: int = 0
    device: str = "cpu"

    def schedule(self) -> ScheduleConfig:
        total = self.total_steps
        if self.max_steps:
            total = min(total, self.max_steps)
        return ScheduleConfig(warmup_steps=self.warmup_steps, total_steps=max(total, 1))


@dataclass
class StepRecord:
    """One step's read-outs."""

    epoch: int
    step: int
    loss: float
    gradient_norm: float
    learning_rate: float


@dataclass
class TrainingRun:
    """Every record of one fit, with the wall-clock-free summary the report needs."""

    records: list[StepRecord] = field(default_factory=list)
    stopped_early: bool = False
    best_loss: float = float("inf")

    @property
    def losses(self) -> list[float]:
        return [record.loss for record in self.records]

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {"steps": 0.0}
        losses = np.asarray(self.losses, dtype=np.float64)
        half = max(len(losses) // 2, 1)
        return {
            "steps": float(len(losses)),
            "first_loss": float(losses[0]),
            "last_loss": float(losses[-1]),
            "leading_mean": float(losses[:half].mean()),
            "trailing_mean": float(losses[half:].mean()),
            "best_loss": float(self.best_loss),
        }


def iterate_batches(size: int, batch_size: int, seed: int, epoch: int) -> Iterator[np.ndarray]:
    """A shuffled permutation of the rows, cut into batches."""

    if batch_size < 1:
        raise ValueError("the batch size must be at least one")
    rng = spawn_generator(seed, f"batches:epoch={epoch}")
    order = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


class Trainer:
    """A minimal trainer over one differentiable model and one step function."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        step_function: Callable[[nn.Module, Tensor], Tensor],
        batch_size: int,
    ) -> None:
        self.model = model
        self.config = config
        self.step_function = step_function
        self.batch_size = batch_size
        self.optimiser = build_optimiser(
            model.parameters(), config.learning_rate, config.weight_decay
        )
        self.schedule = WarmupCosine(self.optimiser, config.learning_rate, config.schedule())
        self.run = TrainingRun()

    def step(self, batch_index: Tensor, epoch: int = 0) -> float:
        """One optimizer step on one batch of row indices."""

        self.model.train()
        self.optimiser.zero_grad(set_to_none=True)
        loss: Tensor = self.step_function(self.model, batch_index)
        backpropagate(loss)
        norm = clip_gradients(self.model, self.config.grad_clip)
        self.optimiser.step()
        rate = self.schedule.advance()
        self.run.records.append(
            StepRecord(
                epoch=epoch,
                step=len(self.run.records),
                loss=float(loss.detach()),
                gradient_norm=norm,
                learning_rate=rate,
            )
        )
        return float(loss.detach())

    def fit(self, rows: int, *, epoch: int = 0) -> TrainingRun:
        """Train for the configured number of epochs over ``rows`` rows.

        ``rows`` is the number of examples available; the step function receives
        the index batch and slices its own data from it.
        """

        set_seed(self.config.seed)
        completed = 0
        for current in range(self.config.epochs):
            for batch in iterate_batches(rows, self.batch_size, self.config.seed, current + epoch):
                if self.config.max_steps and completed >= self.config.max_steps:
                    return self.run
                index = torch.as_tensor(batch, dtype=torch.long)
                loss = self.step(index, current)
                completed += 1
                self.run.best_loss = min(self.run.best_loss, loss)
                if (
                    self.config.early_stop_patience
                    and len(self.run.records) > self.config.early_stop_patience
                    and self._stalled()
                ):
                    self.run.stopped_early = True
                    return self.run
        return self.run

    def _stalled(self) -> bool:
        patience = self.config.early_stop_patience
        window = self.run.losses[-patience:]
        previous = self.run.losses[-2 * patience : -patience]
        if not previous:
            return False
        return float(np.mean(window)) >= float(np.mean(previous))


def loss_decreased(records: Sequence[StepRecord], minimum_pairs: int = 4) -> bool:
    """Whether the trailing half of a run sits below its leading half."""

    if len(records) < minimum_pairs:
        return False
    losses = np.asarray([record.loss for record in records], dtype=np.float64)
    half = max(len(losses) // 2, 1)
    return bool(losses[half:].mean() < losses[:half].mean())


def parameter_delta(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    """The L2 distance between two parameter snapshots."""

    total = 0.0
    for name, tensor in before.items():
        if name in after:
            total += float((tensor - after[name]).pow(2).sum())
    return float(np.sqrt(total))


def snapshot(model: nn.Module) -> dict[str, Tensor]:
    """A detached copy of the model's parameters, for a before-and-after check."""

    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def seed_state(config: TrainingConfig) -> SeedState:
    return SeedState(seed=config.seed, stage="stage3", epoch=0)
