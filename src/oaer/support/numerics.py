"""Numeric primitives used across the estimator and the read-outs.

The inverse normal CDF is implemented here rather than taken from a library so
that a conformal quantile and a Wald limit are computable without a SciPy import
on the inference path. Its accuracy is checked against a closed-form value.

Ref: Sec. 2.4 (Eq. (1)), Sec. 2.7 (intervals of the reported contrasts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

# Acklam's rational approximation, with the two refinement steps that bring the
# relative error below 1.15e-9 over the whole open unit interval.
_A: Final[tuple[float, ...]] = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B: Final[tuple[float, ...]] = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C: Final[tuple[float, ...]] = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D: Final[tuple[float, ...]] = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW: Final[float] = 0.02425
_P_HIGH: Final[float] = 1.0 - _P_LOW


@dataclass(frozen=True)
class Interval:
    """A point estimate carried with the two limits of its interval."""

    point: float
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def excludes(self, value: float) -> bool:
        return value < self.lower or value > self.upper


def sigmoid(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def logit(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(values / (1.0 - values))


def safe_divide(
    numerator: npt.ArrayLike,
    denominator: npt.ArrayLike,
    *,
    fill: float = 0.0,
) -> npt.NDArray[np.float64]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(top, bottom).shape, fill, dtype=np.float64)
    np.divide(top, bottom, out=out, where=np.abs(bottom) > 1e-12)
    return out


def weighted_mean(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    fill: float = 0.0,
) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    total = float(w.sum())
    if abs(total) < 1e-12:
        return fill
    return float(np.dot(v, w) / total)


def normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF by the Acklam rational approximation."""

    if not 0.0 < probability < 1.0:
        raise ValueError("the probability must lie strictly inside the unit interval")
    if probability < _P_LOW:
        q = math.sqrt(-2.0 * math.log(probability))
        numerator = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        denominator = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return numerator / denominator
    if probability > _P_HIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        numerator = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        denominator = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return -numerator / denominator
    q = probability - 0.5
    r = q * q
    numerator = ((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]
    denominator = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    return q * numerator / denominator


def one_sided_upper(point: float, standard_error: float, level: float = 0.95) -> float:
    """The upper limit of a one-sided normal interval at ``level`` coverage."""

    return point + normal_quantile(level) * standard_error


def two_sided(point: float, standard_error: float, level: float = 0.95) -> Interval:
    """A symmetric Wald interval whose half-width uses the two-sided quantile."""

    half = normal_quantile(0.5 + level / 2.0) * standard_error
    return Interval(point=point, lower=point - half, upper=point + half)


def finite_difference_step(scale: float) -> float:
    """A step that keeps a central difference inside double precision."""

    return max(abs(scale), 1.0) * 1e-6


def softmax(values: npt.ArrayLike, axis: int = -1) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    normalised: npt.NDArray[np.float64] = np.asarray(
        exponentials / np.sum(exponentials, axis=axis, keepdims=True), dtype=np.float64
    )
    return normalised


def log_softmax(values: npt.ArrayLike, axis: int = -1) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    shifted_log: npt.NDArray[np.float64] = np.asarray(
        shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)), dtype=np.float64
    )
    return shifted_log


def step_integral(values: npt.ArrayLike, points: npt.ArrayLike) -> float:
    """The exact integral of a right-continuous step function on its own knots.

    A survival curve is constant between its jump times, so the integral is the
    sum of each interval's width times the value at its left end. A trapezoid rule
    would instead add half of every drop, which is why a restricted mean built on
    one does not reduce to the sample mean of the capped times when nothing is
    censored.
    """

    ordinates = np.asarray(values, dtype=np.float64).ravel()
    abscissae = np.asarray(points, dtype=np.float64).ravel()
    if ordinates.size != abscissae.size:
        raise ValueError("ordinates and abscissae must have the same length")
    if ordinates.size < 2:
        return 0.0
    widths = np.diff(abscissae)
    return float(np.sum(widths * ordinates[:-1]))
