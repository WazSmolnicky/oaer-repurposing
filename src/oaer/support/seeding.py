"""Seed management.

One call sets every random stream the pipeline can reach, and the seed travels with
every checkpoint so a resumed run continues the stream rather than restarting it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """The seed of one run, carried next to a checkpoint."""

    seed: int
    stage: str
    epoch: int = 0

    def derive(self, purpose: str) -> int:
        """A deterministic child seed for one reproducible sub-stream."""

        return derive_seed(self.seed, f"{self.stage}:{self.epoch}:{purpose}")


def derive_seed(seed: int, purpose: str) -> int:
    """A stable 63-bit child seed from a parent seed and a purpose string."""

    digest = 1469598103934665603
    for chunk in f"{seed}:{purpose}".encode():
        digest ^= chunk
        digest = (digest * 1099511628211) & ((1 << 64) - 1)
    return digest & ((1 << 63) - 1)


def spawn_generator(seed: int, purpose: str) -> np.random.Generator:
    """A random stream on a child purpose, so one purpose cannot shift another."""

    return np.random.default_rng(derive_seed(seed, purpose))


def set_seed(seed: int) -> None:
    """Seed the standard library, NumPy and torch, and pin the hash seed."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator(seed: int) -> np.random.Generator:
    """A NumPy generator isolated from the global stream."""

    return np.random.default_rng(seed)
