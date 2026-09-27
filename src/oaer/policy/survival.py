"""Survival read-outs: the Cox hazard ratio, the restricted mean and its contrast.

The primary contrasts are reported as adjusted hazard ratios beside a weighted
restricted-mean policy contrast, and the article fixes which quantity is which: a
contrast between arms is a restricted-mean difference and is never called a hazard
ratio, because the restricted mean is collapsible and the hazard ratio is not.
The hazard ratio is fitted by Breslow's partial likelihood with a ridge penalty,
which is what makes it finite when a stratum is small.

Ref: Sec. 2.7 (endpoints and metrics), Table 1 panel b, Lemma 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.support.numerics import normal_quantile, step_integral

RIDGE: Final[float] = 0.01
MAX_ITERATIONS: Final[int] = 60
TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class HazardRatio:
    """An adjusted hazard ratio with its Wald interval."""

    estimate: float
    standard_error: float
    coefficient: float
    events: int
    subjects: int

    @property
    def interval(self) -> tuple[float, float]:
        half = normal_quantile(0.975) * self.standard_error
        return (float(np.exp(self.coefficient - half)), float(np.exp(self.coefficient + half)))

    def as_dict(self) -> dict[str, float | int]:
        lower, upper = self.interval
        return {
            "hazard_ratio": round(self.estimate, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "events": self.events,
            "subjects": self.subjects,
        }


@dataclass(frozen=True)
class RestrictedMean:
    """A restricted mean with the horizon it was read at."""

    value: float
    horizon: float
    standard_error: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {"value": round(self.value, 6), "horizon": round(self.horizon, 6)}


def kaplan_meier(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    grid: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A weighted Kaplan-Meier curve, evaluated on a grid.

    The weights are the clone or inverse-probability weights, so an unweighted fit
    is the special case of all-ones.
    """

    time = np.asarray(times, dtype=np.float64).ravel()
    event = np.asarray(events, dtype=bool).ravel()
    mass = np.ones_like(time) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if time.size != event.size or time.size != mass.size:
        raise ValueError("times, events and weights must have the same length")
    points = (
        np.linspace(0.0, float(time.max()), 241)
        if grid is None
        else np.asarray(grid, dtype=np.float64).ravel()
    )
    survival = np.ones_like(points)
    if time.size == 0:
        return points, survival
    curve = 1.0
    event_times = np.unique(time[event])
    processed = 0
    for position, point in enumerate(points):
        while processed < event_times.size and event_times[processed] <= point:
            at_time = float(event_times[processed])
            at_risk = float(mass[time >= at_time].sum())
            deaths = float(mass[(time == at_time) & event].sum())
            if at_risk > 0.0:
                curve *= 1.0 - deaths / at_risk
            processed += 1
        survival[position] = curve
    return points, np.clip(survival, 0.0, 1.0)


def restricted_mean(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    horizon: float,
    weights: npt.ArrayLike | None = None,
) -> RestrictedMean:
    """``RMST(t*) = int_0^{t*} S(u) du`` from a weighted Kaplan-Meier curve.

    The integration knots are the curve's own jump times together with the two
    ends, so the integral is exact for the step function: no censoring means the
    value is the sample mean of the capped times, to machine precision.
    """

    time = np.asarray(times, dtype=np.float64).ravel()
    event = np.asarray(events, dtype=bool).ravel()
    knots = np.unique(np.concatenate(([0.0], time[event], [float(horizon)])))
    knots = knots[knots <= float(horizon)]
    if knots.size < 2:
        knots = np.asarray([0.0, float(horizon)], dtype=np.float64)
    points, survival = kaplan_meier(times, events, weights, knots)
    mass = (
        np.ones_like(np.asarray(times, dtype=np.float64))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    return RestrictedMean(
        value=step_integral(survival, points),
        horizon=horizon,
        standard_error=restricted_mean_error(times, events, mass, horizon),
    )


def cox_hazard_ratio(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    covariate: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    ridge: float = RIDGE,
) -> HazardRatio:
    """A single-covariate Cox hazard ratio by Breslow's partial likelihood.

    The covariate is standardised before the fit so the ridge penalty is on the
    same scale for every read-out, and the reported ratio is rescaled back to the
    covariate's own unit.
    """

    time = np.asarray(times, dtype=np.float64).ravel()
    event = np.asarray(events, dtype=bool).ravel()
    raw = np.asarray(covariate, dtype=np.float64).ravel()
    mass = np.ones_like(time) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if not (time.size == event.size == raw.size == mass.size):
        raise ValueError("times, events, covariates and weights must share a length")
    events_total = int(event.sum())
    if events_total == 0 or time.size < 3:
        return HazardRatio(1.0, float("nan"), 0.0, events_total, int(time.size))
    scale = float(raw.std())
    if scale < 1e-12:
        return HazardRatio(1.0, float("nan"), 0.0, events_total, int(time.size))
    standardised = (raw - float(raw.mean())) / scale
    order = np.argsort(time, kind="stable")
    time = time[order]
    event = event[order]
    standardised = standardised[order]
    mass = mass[order]
    coefficient = 0.0
    information = 0.0
    for _ in range(MAX_ITERATIONS):
        risk = mass * np.exp(np.clip(standardised * coefficient, -30.0, 30.0))
        gradient = 0.0
        information = 0.0
        for index in np.nonzero(event)[0]:
            at_risk = time >= time[index]
            total = float(risk[at_risk].sum())
            if total <= 0.0:
                continue
            first = float((risk[at_risk] * standardised[at_risk]).sum()) / total
            second = float((risk[at_risk] * standardised[at_risk] ** 2).sum()) / total
            gradient += standardised[index] - first
            information += second - first**2
        gradient -= ridge * coefficient
        information += ridge
        if information <= 0.0:
            break
        step = gradient / information
        coefficient += step
        if abs(step) < TOLERANCE:
            break
    error_standardised = float(np.sqrt(1.0 / information)) if information > 0.0 else float("nan")
    return HazardRatio(
        estimate=float(np.exp(coefficient / scale)),
        standard_error=error_standardised / scale,
        coefficient=coefficient / scale,
        events=events_total,
        subjects=int(time.size),
    )


def weighted_restricted_mean_contrast(
    treated_times: npt.ArrayLike,
    treated_events: npt.ArrayLike,
    treated_weights: npt.ArrayLike,
    control_times: npt.ArrayLike,
    control_events: npt.ArrayLike,
    control_weights: npt.ArrayLike,
    horizon: float,
) -> tuple[float, float, float]:
    """The censoring-weighted restricted-mean contrast between two clone arms.

    Returns the contrast and its two limits, where the standard error is the
    root-sum-square of the two arms' own restricted-mean errors.
    """

    treated = restricted_mean(treated_times, treated_events, horizon, treated_weights)
    control = restricted_mean(control_times, control_events, horizon, control_weights)
    difference = treated.value - control.value
    error = float(np.sqrt(treated.standard_error**2 + control.standard_error**2))
    if not np.isfinite(error):
        return (difference, float("nan"), float("nan"))
    half = normal_quantile(0.975) * error
    return (difference, difference - half, difference + half)


def restricted_mean_error(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    weights: npt.ArrayLike,
    horizon: float,
) -> float:
    """A Greenwood-type standard error for a weighted restricted mean.

    The variance is accumulated as ``S(u)^2 * dLambda(u) / Y(u)`` over the event
    times, integrated across the horizon and rescaled by the grid width.
    """

    time = np.asarray(times, dtype=np.float64).ravel()
    event = np.asarray(events, dtype=bool).ravel()
    mass = np.asarray(weights, dtype=np.float64).ravel()
    if time.size == 0:
        return float("nan")
    variance = 0.0
    survival = 1.0
    for point in np.unique(time[event]):
        at_risk = float(mass[time >= point].sum())
        deaths = float(mass[(time == point) & event].sum())
        if at_risk <= 0.0:
            continue
        survival *= 1.0 - deaths / at_risk
        if point <= horizon:
            variance += survival**2 * deaths / at_risk**2
    return float(np.sqrt(max(variance, 0.0)) * horizon)


def landmark_survival(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    month: float,
    weights: npt.ArrayLike | None = None,
) -> float:
    """The Kaplan-Meier survival at a landmark month."""

    grid = np.linspace(0.0, month, 121)
    points, survival = kaplan_meier(times, events, weights, grid)
    return float(survival[-1]) if points.size else float("nan")


def events_by_arm(
    exposure: npt.ArrayLike,
    events: npt.ArrayLike,
) -> dict[str, int]:
    """Event counts on both sides of an exposure indicator."""

    indicator = np.asarray(exposure, dtype=np.float64).ravel() >= 0.5
    event = np.asarray(events, dtype=bool).ravel()
    return {
        "exposed_events": int(event[indicator].sum()),
        "control_events": int(event[~indicator].sum()),
        "exposed_n": int(indicator.sum()),
        "control_n": int((~indicator).sum()),
    }


def standardised_difference(
    values: npt.ArrayLike,
    groups: npt.ArrayLike,
) -> float:
    """The standardised mean difference between two groups of a covariate."""

    array = np.asarray(values, dtype=np.float64).ravel()
    indicator = np.asarray(groups, dtype=np.float64).ravel() >= 0.5
    if indicator.sum() == 0 or (~indicator).sum() == 0:
        return float("nan")
    left, right = array[indicator], array[~indicator]
    pooled = float(np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0))
    if pooled < 1e-12:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def balance_table(
    covariates: dict[str, npt.ArrayLike],
    groups: npt.ArrayLike,
) -> dict[str, float]:
    """The standardised difference of every covariate, keyed by name."""

    return {
        name: standardised_difference(values, groups) for name, values in sorted(covariates.items())
    }


def site_heterogeneity(
    estimates: Sequence[float],
    errors: Sequence[float],
) -> tuple[float, float]:
    """Cochran's ``Q`` and the ``I^2`` share derived from it.

    With three sites ``Q`` carries two degrees of freedom, so ``I^2`` is read for
    direction rather than magnitude, as the table caption states.
    """

    values = np.asarray(estimates, dtype=np.float64)
    sigma = np.asarray(errors, dtype=np.float64)
    if values.size < 2:
        return (float("nan"), 0.0)
    valid = np.isfinite(values) & np.isfinite(sigma) & (sigma > 0.0)
    values, sigma = values[valid], sigma[valid]
    if values.size < 2:
        return (float("nan"), 0.0)
    weights = 1.0 / sigma**2
    fixed = float((weights * values).sum() / weights.sum())
    q = float((weights * (values - fixed) ** 2).sum())
    degrees = values.size - 1
    i_squared = max(0.0, (q - degrees) / q) * 100.0 if q > 0.0 else 0.0
    return (q, i_squared)
