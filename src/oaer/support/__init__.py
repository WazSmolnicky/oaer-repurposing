"""Shared configuration, serialisation, numerics and logging helpers.

Ref: Sec. 2.6 (design of the pipeline), Sec. 2.7 (metric definitions).
"""

from __future__ import annotations

from oaer.support.config import ProtocolConfig, load_protocol, merge_overrides
from oaer.support.io import read_json, read_text, write_json, write_text
from oaer.support.logging import configure_logging, get_logger
from oaer.support.numerics import (
    Interval,
    logit,
    normal_quantile,
    one_sided_upper,
    safe_divide,
    sigmoid,
    weighted_mean,
)
from oaer.support.seeding import SeedState, set_seed, spawn_generator
from oaer.support.version import RELEASE_SLUG

__all__ = [
    "Interval",
    "RELEASE_SLUG",
    "ProtocolConfig",
    "SeedState",
    "configure_logging",
    "spawn_generator",
    "get_logger",
    "load_protocol",
    "logit",
    "merge_overrides",
    "normal_quantile",
    "one_sided_upper",
    "read_json",
    "read_text",
    "safe_divide",
    "set_seed",
    "sigmoid",
    "weighted_mean",
    "write_json",
    "write_text",
]
