"""Interaction read-outs: IAB, IR and the exposure terciles.

The primary interaction read-out is ``IAB``: the joint removal's loss minus the
sum of the two single removals' losses, on the same scale as the deltas. A
negative value is synergy. ``IR`` is printed only where the bootstrap interval of
its denominator lies entirely above zero, because the equivalence between
``IR > 1`` and ``IAB > 0`` fails for a negative denominator; that condition is the
single negligibility rule the article applies.

The per-candidate tercile split separates the graph prior's gain by how much
exposure each candidate carries, which is where the borrowing-strength behaviour
of the graph prior shows up: the gain is largest where the exposed count is
smallest and closes to about zero where it is largest.

Ref: Sec. 1.4 panel b and panel d, Sec. 2.7 (the sign convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from oaer.support.numerics import Interval

NEGLIGIBLE_DENOMINATOR: Final[float] = 0.0


@dataclass(frozen=True)
class InteractionRow:
    """One interaction read-out with its interval and its printed ratio."""

    pair: str
    joint_delta: float
    first_delta: float
    second_delta: float
    interval: Interval
    ratio: float | None
    ratio_interval: Interval | None
    expectation: str

    @property
    def iab(self) -> float:
        return self.joint_delta - (self.first_delta + self.second_delta)

    @property
    def synergy(self) -> bool:
        return self.iab < 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "iab": round(self.iab, 6),
            "iab_lower": round(self.interval.lower, 6),
            "iab_upper": round(self.interval.upper, 6),
            "ratio": None if self.ratio is None else round(self.ratio, 6),
            "synergy": self.synergy,
            "expectation": self.expectation,
        }


def interaction_beyond_additive(
    joint_delta: float,
    first_delta: float,
    second_delta: float,
) -> float:
    """``IAB = Delta_AB - (Delta_A + Delta_B)``."""

    return joint_delta - (first_delta + second_delta)


def interaction_ratio(
    joint_delta: float,
    first_delta: float,
    second_delta: float,
) -> float:
    """``IR = (Delta_A + Delta_B) / Delta_AB`` on the removal scale.

    A value below one is synergy and a value above one is redundancy. The ratio is
    printed only where the interval of its denominator stays on one side of zero,
    because the equivalence between ``IR < 1`` and ``IAB < 0`` fails otherwise.
    """

    if abs(joint_delta) < 1e-12:
        return float("nan")
    return (first_delta + second_delta) / joint_delta


def ratio_admissible(denominator_interval: Interval) -> bool:
    """Whether the ratio may be printed: its denominator must clear zero."""

    return denominator_interval.lower > NEGLIGIBLE_DENOMINATOR


def build_interaction_row(
    pair: str,
    joint_delta: float,
    first_delta: float,
    second_delta: float,
    *,
    joint_interval: Interval,
    first_interval: Interval,
    second_interval: Interval,
    expectation: str,
) -> InteractionRow:
    """Assemble one row, with the interval propagated in quadrature.

    The joint interval's half-width and the two single intervals' half-widths are
    combined in quadrature, which is the site-stratified bootstrap's interval
    applied to the linear combination the identity defines.
    """

    half = float(
        np.sqrt(joint_interval.width**2 + first_interval.width**2 + second_interval.width**2) / 2.0
    )
    iab = interaction_beyond_additive(joint_delta, first_delta, second_delta)
    interval = Interval(point=iab, lower=iab - half, upper=iab + half)
    ratio = interaction_ratio(joint_delta, first_delta, second_delta)
    admissible = joint_interval.lower > 0.0 or joint_interval.upper < 0.0
    return InteractionRow(
        pair=pair,
        joint_delta=joint_delta,
        first_delta=first_delta,
        second_delta=second_delta,
        interval=interval,
        ratio=ratio if admissible else None,
        ratio_interval=interval if admissible else None,
        expectation=expectation,
    )


@dataclass(frozen=True)
class TercileGain:
    """The graph prior's gain inside one exposure tercile."""

    name: str
    exposed_range: tuple[int, int]
    candidates: int
    catalogue_size: int
    trimmed_percent: float
    hits_gain: float
    value_gain: float
    index_gain: float
    expectation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "tercile": self.name,
            "candidates": self.candidates,
            "catalogue_size": self.catalogue_size,
            "trimmed_percent": round(self.trimmed_percent, 4),
            "hits_gain": round(self.hits_gain, 4),
            "value_gain": round(self.value_gain, 4),
            "index_gain": round(self.index_gain, 4),
            "expectation": self.expectation,
        }


def tercile_expectation(name: str) -> str:
    return {
        "lowest": "largest KG gain",
        "middle": "intermediate",
        "highest": "about zero KG gain",
    }.get(name, "")


def normalised_gain(gain: float, reference: float) -> float:
    """A gain expressed against its reference, guarding a zero reference."""

    if abs(reference) < 1e-12:
        return float("nan")
    return gain / reference


def exposure_tercile_summary(
    gains: dict[str, dict[str, float]],
    sizes: dict[str, tuple[int, int]],
    candidates: dict[str, int],
    trimmed: dict[str, float],
) -> tuple[TercileGain, ...]:
    """Assemble the three tercile rows from the ablated read-outs."""

    rows: list[TercileGain] = []
    for name in ("lowest", "middle", "highest"):
        payload = gains.get(name, {})
        rows.append(
            TercileGain(
                name=name,
                exposed_range=sizes.get(name, (0, 0)),
                candidates=candidates.get(name, 0),
                catalogue_size=int(payload.get("catalogue_size", 0)),
                trimmed_percent=float(trimmed.get(name, 0.0)),
                hits_gain=float(payload.get("hits", 0.0)),
                value_gain=float(payload.get("value", 0.0)),
                index_gain=float(payload.get("index", 0.0)),
                expectation=tercile_expectation(name),
            )
        )
    return tuple(rows)
