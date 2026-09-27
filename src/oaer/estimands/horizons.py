"""Restricted-time horizons and the integration grid.

The horizons are set from the real-world median progression-free survival of the
disease at each line rather than from trial medians, and the overall-survival
horizon is set from the observed overall survival from treatment start. The
integration grid is dense enough that the martingale integral of Eq. (2) lands on
the observed censoring times without a step being missed.

Ref: Sec. 2.5 (horizons determined before the analysis), Sec. 2.3 (t*_L).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.support.types import LINE_HORIZON_MONTHS, Decision, LineOfTherapy

HORIZON_GRID: Final[Mapping[str, float]] = dict(LINE_HORIZON_MONTHS)

# The overall-survival horizon, in months, is the observed overall survival from
# treatment start rather than the progression-free median.
OVERALL_SURVIVAL_HORIZON_MONTHS: Final[float] = 30.4

# The landmark read-out, in months.
LANDMARK_MONTHS: Final[float] = 12.0

GRID_POINTS: Final[int] = 241


def line_horizon(line: LineOfTherapy) -> float:
    """The restricted-time horizon of one line of therapy."""

    return HORIZON_GRID[line.value]


def real_world_medians() -> Mapping[str, float]:
    """The horizons as they are declared, keyed by line."""

    return dict(HORIZON_GRID)


def restricted_time(decision: Decision) -> float:
    """``Y~_i = min(Y_i, t*_L)``."""

    return min(decision.follow_up_months, line_horizon(decision.state.line))


def integration_grid(horizon: float, points: int = GRID_POINTS) -> npt.NDArray[np.float64]:
    """A uniform grid that always contains both ends of ``[0, t*]``."""

    if horizon <= 0.0:
        raise ValueError("a horizon must be positive")
    if points < 3:
        raise ValueError("the integration grid needs at least three points")
    return np.linspace(0.0, horizon, points)


def observed_step_times(decisions: Sequence[Decision]) -> npt.NDArray[np.float64]:
    """Every distinct observed restricted time in a batch, ascending."""

    values = sorted({round(min(d.follow_up_months, d.horizon_months), 9) for d in decisions})
    return np.asarray(values, dtype=np.float64)


def median_survival(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
) -> float:
    """The Kaplan-Meier median of a sample, censoring-aware."""

    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=bool)
    if time.size != event.size:
        raise ValueError("times and events must have the same length")
    if time.size == 0:
        return float("nan")
    survival = 1.0
    for point in np.unique(time[event]):
        at_risk = int((time >= point).sum())
        deaths = int(((time == point) & event).sum())
        if at_risk == 0:
            continue
        survival *= 1.0 - deaths / at_risk
        if survival <= 0.5:
            return float(point)
    return float("nan")


def horizon_ratios() -> Mapping[str, float]:
    """Each line's horizon as a share of the first-line horizon."""

    first = HORIZON_GRID[LineOfTherapy.FIRST.value]
    return {line: value / first for line, value in HORIZON_GRID.items()}
