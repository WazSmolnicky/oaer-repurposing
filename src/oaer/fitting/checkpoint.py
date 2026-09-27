"""Checkpointing: atomic writes, a seed that travels with the weights, a hash check.

A checkpoint carries the stage, the epoch, the step, the seed and the resolved
config beside the weights and the optimiser state, so a resumed run restores the
seed it was trained with rather than the seed of the process that resumed it. The
write goes to a neighbouring temporary file and is moved into place, so a crash
never leaves a half-written checkpoint where a whole one used to be.

The digest is taken over the parameter payload rather than over the container
bytes: ``torch.save`` embeds container metadata, so a file hash of a ``.pt``
changes on every write while the tensors inside are identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class CheckpointPayload:
    """Everything a resumed run needs, beside the weights themselves."""

    stage: str
    epoch: int
    step: int
    seed: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "step": self.step,
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "config": dict(self.config),
        }


def parameter_digest(model: nn.Module) -> str:
    """A SHA-256 over the parameter names, shapes and raw bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().to(torch.float64).cpu().numpy().tobytes())
    return digest.hexdigest()


def save_state(
    path: str | Path,
    payload: CheckpointPayload,
    *,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
) -> Path:
    """Write a checkpoint atomically and return the path it landed on."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "payload": payload.as_dict(),
        "model": model.state_dict(),
        "digest": parameter_digest(model),
    }
    if optimiser is not None:
        state["optimiser"] = optimiser.state_dict()
    descriptor, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=".ckpt-", suffix=".part"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state, handle)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return target


def load_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> CheckpointPayload:
    """Load a checkpoint and, optionally, restore the seed it carries."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no checkpoint at {source}")
    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "payload" not in state or "model" not in state:
        raise ValueError(f"{source} does not hold a checkpoint written by this package")
    model.load_state_dict(state["model"])
    if optimiser is not None and "optimiser" in state:
        optimiser.load_state_dict(state["optimiser"])
    payload = CheckpointPayload(**state["payload"])
    if restore_rng:
        from oaer.support.seeding import set_seed

        set_seed(payload.seed)
    return payload


def verify_digest(path: str | Path, model: nn.Module) -> bool:
    """Whether the model's parameters still match the checkpoint's digest."""

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    reference = str(state.get("digest", ""))
    return reference == parameter_digest(model)


def torch_state_equal(left: nn.Module, right: nn.Module) -> bool:
    """Whether two modules carry identical parameter payloads."""

    left_state = left.state_dict()
    right_state = right.state_dict()
    if set(left_state) != set(right_state):
        return False
    return all(torch.equal(left_state[name], right_state[name]) for name in left_state)


def checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    """A JSON-safe description of a checkpoint, for the report."""

    source = Path(path)
    state = torch.load(source, map_location="cpu", weights_only=False)
    payload = state.get("payload", {})
    return {
        "path": source.name,
        "bytes": source.stat().st_size,
        "digest": state.get("digest", ""),
        "stage": payload.get("stage", ""),
        "step": payload.get("step", 0),
        "seed": payload.get("seed", 0),
    }


def summarise_checkpoint(path: str | Path) -> str:
    return json.dumps(checkpoint_manifest(path), sort_keys=True)
