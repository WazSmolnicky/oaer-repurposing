"""Sensitivity read-outs: the E-value, the risk-ratio scale and the negative control.

Each emulation report carries four things beside its estimate: the E-value of the
risk-ratio contrast, the negative-control-outcome estimate, the trimmed fraction
and the number of sites clearing the read-out. The E-value is computed on the
risk-ratio scale rather than on a hazard ratio, because it is the ratio of the
cumulative incidences inside the horizon under the same pair of survival curves;
a hazard ratio from two different sets of subjects is not that quantity.

Ref: Sec. 2.5 (E-value computed on the risk-ratio scale), Sec. 2.7 (sensitivity),
Table 1 panel d.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.estimands.horizons import line_horizon
from oaer.support.types import Decision

NEGATIVE_CONTROL_HORIZON_FRACTION: Final[float] = 0.25


@dataclass(frozen=True)
class EValue:
    """The E-value of a risk-ratio contrast and of the interval's null limit."""

    point: float
    interval_limit: float
    risk_ratio: float
    direction_upward: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "point": round(self.point, 4),
            "interval_limit": round(self.interval_limit, 4),
            "risk_ratio": round(self.risk_ratio, 6),
            "direction_upward": self.direction_upward,
        }


def _e_value_from_ratio(ratio: float) -> float:
    """``E = RR + sqrt(RR (RR - 1))`` on the null-away side of the ratio."""

    if ratio <= 0.0:
        return float("nan")
    shifted = ratio if ratio >= 1.0 else 1.0 / ratio
    if shifted < 1.0:
        return 1.0
    return float(shifted + math.sqrt(shifted * (shifted - 1.0)))


def e_value(risk_ratio: float, lower: float, upper: float) -> EValue:
    """The E-value of a point ratio and of the interval limit nearest the null."""

    point = _e_value_from_ratio(risk_ratio)
    upward = risk_ratio >= 1.0
    if upward:
        limit = lower if np.isfinite(lower) and lower > 0.0 else float("nan")
        if not np.isfinite(limit):
            limit_e = 1.0
        elif limit <= 1.0:
            # An interval that already contains the null has an E-value of one at
            # its limit, by the definition of the limiting value.
            limit_e = 1.0
        else:
            limit_e = _e_value_from_ratio(limit)
    else:
        limit = upper
        if not np.isfinite(limit) or limit >= 1.0:
            limit_e = 1.0
        elif limit <= 0.0:
            limit_e = float("inf")
        else:
            limit_e = _e_value_from_ratio(limit)
    return EValue(
        point=point, interval_limit=limit_e, risk_ratio=risk_ratio, direction_upward=upward
    )


def cumulative_incidence(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    horizon: float,
) -> float:
    """``1 - S(horizon)`` from a Kaplan-Meier fit, censoring-aware."""

    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=bool)
    if time.size == 0:
        return float("nan")
    survival = 1.0
    for point in np.unique(time[event]):
        if point > horizon:
            break
        at_risk = int((time >= point).sum())
        deaths = int(((time == point) & event).sum())
        if at_risk > 0:
            survival *= 1.0 - deaths / at_risk
    return float(1.0 - survival)


def risk_ratio_at_horizon(
    decisions: Sequence[Decision],
    candidate: str,
    *,
    horizon: float | None = None,
) -> tuple[float, float, float]:
    """The risk ratio between the exposed and unexposed groups, with a Wald interval.

    The horizon defaults to the shortest line-specific horizon present, so a
    mixed-line batch is compared over a window every line actually spans.
    """

    if not decisions:
        return (float("nan"), float("nan"), float("nan"))
    window = horizon if horizon is not None else _shortest_horizon(decisions)
    exposed = [d for d in decisions if d.exposure.get(candidate, False)]
    control = [d for d in decisions if not d.exposure.get(candidate, False)]
    if not exposed or not control:
        return (float("nan"), float("nan"), float("nan"))
    exposed_risk = cumulative_incidence(
        [min(d.follow_up_months, window) for d in exposed], [d.event for d in exposed], window
    )
    control_risk = cumulative_incidence(
        [min(d.follow_up_months, window) for d in control], [d.event for d in control], window
    )
    if control_risk <= 0.0:
        return (float("nan"), float("nan"), float("nan"))
    ratio = exposed_risk / control_risk
    error = math.sqrt(
        max(1.0 - exposed_risk, 1e-9) / max(len(exposed) * exposed_risk, 1e-9)
        + max(1.0 - control_risk, 1e-9) / max(len(control) * control_risk, 1e-9)
    )
    log_ratio = math.log(ratio) if ratio > 0.0 else float("nan")
    lower = math.exp(log_ratio - 1.96 * error) if np.isfinite(log_ratio) else float("nan")
    upper = math.exp(log_ratio + 1.96 * error) if np.isfinite(log_ratio) else float("nan")
    return (ratio, lower, upper)


def _shortest_horizon(decisions: Sequence[Decision]) -> float:
    return min(line_horizon(decision.state.line) for decision in decisions)


def negative_control_estimate(
    decisions: Sequence[Decision],
    candidate: str,
    *,
    horizon_fraction: float = NEGATIVE_CONTROL_HORIZON_FRACTION,
) -> tuple[float, float, float]:
    """The risk ratio on a pre-exposure window that the candidate cannot affect.

    The window ends a fixed fraction into the follow-up, before the add-on is
    expected to act, so a value near one is the consistency read-out the article
    reports beside the estimate.
    """

    if not decisions:
        return (float("nan"), float("nan"), float("nan"))
    window = _shortest_horizon(decisions) * horizon_fraction
    return risk_ratio_at_horizon(decisions, candidate, horizon=window)


def screened_candidates(
    risk_ratios: dict[str, float],
    *,
    tolerance: float = 0.15,
) -> tuple[str, ...]:
    """Candidates whose negative-control read-out stays close to one."""

    return tuple(
        name
        for name, value in risk_ratios.items()
        if np.isfinite(value) and abs(value - 1.0) <= tolerance
    )


def e_values_for(
    ratios: dict[str, tuple[float, float, float]],
) -> dict[str, EValue]:
    return {name: e_value(point, lower, upper) for name, (point, lower, upper) in ratios.items()}


def pooled_e_value(values: Sequence[EValue]) -> tuple[float, float]:
    """The point and interval E-values of a pooled read-out."""

    points = [value.point for value in values if np.isfinite(value.point)]
    limits = [value.interval_limit for value in values if np.isfinite(value.interval_limit)]
    if not points:
        return (float("nan"), float("nan"))
    return (float(np.mean(points)), float(np.mean(limits)))


@dataclass(frozen=True)
class LandmarkContrast:
    """The contrast at a landmark month, with the count each side rests on."""

    candidate: str
    month: float
    exposed_risk: float
    control_risk: float
    exposed_n: int
    control_n: int

    @property
    def difference(self) -> float:
        return self.control_risk - self.exposed_risk


def landmark_contrast(
    decisions: Sequence[Decision],
    candidate: str,
    month: float,
) -> LandmarkContrast:
    """Cumulative incidence on both sides at one landmark month."""

    exposed = [d for d in decisions if d.exposure.get(candidate, False)]
    control = [d for d in decisions if not d.exposure.get(candidate, False)]
    return LandmarkContrast(
        candidate=candidate,
        month=month,
        exposed_risk=cumulative_incidence(
            [min(d.follow_up_months, month) for d in exposed],
            [d.event for d in exposed],
            month,
        ),
        control_risk=cumulative_incidence(
            [min(d.follow_up_months, month) for d in control],
            [d.event for d in control],
            month,
        ),
        exposed_n=len(exposed),
        control_n=len(control),
    )
