"""The weighted split-conformal rule of Eq. (1).

The gate defers a decision when the eligible set is empty or when no candidate's
lower conformal limit clears the pre-set margin:

    D_eps(s) = empty   or   max_{d in D_eps(s)} tau^alpha_lower(s, d) <= delta.

The interval is a weighted split-conformal interval: the calibration set is the
most recent retrospective window of Sites A and B, and each calibration point
carries a covariate-only weight that transports it to the target site and period.
The lower limit is the ``alpha``-quantile of the weighted nonconformity scores,
which is what controls miscoverage under the covariate shift the weights encode.

Ref: Sec. 2.4 (Eq. (1)), Algorithm 3 lines 9-11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CovariateWeights:
    """The transport weights of one calibration set."""

    values: npt.NDArray[np.float64]
    clip: float = 10.0

    @property
    def total(self) -> float:
        return float(self.values.sum())

    def normalised(self) -> npt.NDArray[np.float64]:
        total = self.total
        if total <= 0.0:
            return np.full(self.values.shape, 1.0 / max(self.values.size, 1), dtype=np.float64)
        return self.values / total

    def effective_size(self) -> float:
        if self.total <= 0.0:
            return 0.0
        return float(self.values.sum() ** 2 / float(np.square(self.values).sum()))


def covariate_weights(
    target: npt.ArrayLike,
    source: npt.ArrayLike,
    *,
    clip: float = 10.0,
) -> CovariateWeights:
    """Transport weights by the ratio of two one-dimensional covariate densities.

    The ratio is taken between the target period's covariate histogram and the
    calibration period's, then clipped, which is the covariate-only weighting the
    article describes. It carries no outcome.
    """

    target_values = np.asarray(target, dtype=np.float64).ravel()
    source_values = np.asarray(source, dtype=np.float64).ravel()
    if target_values.size == 0 or source_values.size == 0:
        return CovariateWeights(np.ones(max(source_values.size, 1), dtype=np.float64), clip)
    edges = np.histogram_bin_edges(np.concatenate([target_values, source_values]), bins=10)
    source_density, _ = np.histogram(source_values, bins=edges, density=False)
    target_density, _ = np.histogram(target_values, bins=edges, density=False)
    source_density = np.maximum(source_density, 1).astype(np.float64)
    target_density = target_density.astype(np.float64)
    ratio = target_density / source_density
    ratio = np.maximum(ratio, 1.0 / max(clip, 1.0))
    index = np.clip(np.digitize(source_values, edges[1:-1]), 0, ratio.size - 1)
    return CovariateWeights(np.clip(ratio[index], 1.0 / clip, clip), clip)


def weighted_quantile(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    quantile: float,
) -> float:
    """The weighted ``quantile`` of a sample.

    The weighted empirical CDF is inverted at ``quantile``; ties are broken
    towards the smaller value, which is the conservative side for a lower limit.
    """

    sample = np.asarray(values, dtype=np.float64).ravel()
    mass = np.asarray(weights, dtype=np.float64).ravel()
    if sample.size == 0:
        return float("nan")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("a quantile must lie in the unit interval")
    if sample.size != mass.size:
        raise ValueError("the sample and the weights must have the same length")
    order = np.argsort(sample, kind="stable")
    ordered_values = sample[order]
    ordered_mass = mass[order]
    total = float(ordered_mass.sum())
    if total <= 0.0:
        return float(np.quantile(ordered_values, quantile))
    cumulative = np.cumsum(ordered_mass) / total
    position = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(ordered_values[min(position, ordered_values.size - 1)])


@dataclass(frozen=True)
class ConformalCalibration:
    """A frozen calibration set with its transport weights."""

    scores: npt.NDArray[np.float64]
    weights: CovariateWeights
    alpha: float

    @property
    def size(self) -> int:
        return int(self.scores.size)

    def quantile(self) -> float:
        if self.scores.size == 0:
            return float("inf")
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / self.scores.size))
        return weighted_quantile(self.scores, self.weights.values, level)


class WeightedConformal:
    """The gate's conformal component, frozen on a calibration set."""

    def __init__(self, calibration: ConformalCalibration) -> None:
        self.calibration = calibration
        self._quantile = calibration.quantile()

    @property
    def quantile(self) -> float:
        return self._quantile

    def lower_limit(self, point: float) -> float:
        """The lower limit of the interval around a point score."""

        return point - self._quantile

    def upper_limit(self, point: float) -> float:
        return point + self._quantile

    def covers(self, point: float, truth: float) -> bool:
        return self.lower_limit(point) <= truth <= self.upper_limit(point)


def conformal_lower_limit(
    point: float,
    calibration: ConformalCalibration,
) -> float:
    """The lower limit of the weighted conformal interval at one point."""

    return WeightedConformal(calibration).lower_limit(point)


def miscoverage_rate(
    points: npt.ArrayLike,
    truths: npt.ArrayLike,
    calibration: ConformalCalibration,
) -> float:
    """The empirical miscoverage of a calibration set on a fresh sample."""

    values = np.asarray(points, dtype=np.float64).ravel()
    truths = np.asarray(truths, dtype=np.float64).ravel()
    if values.size != truths.size or values.size == 0:
        return float("nan")
    interval = WeightedConformal(calibration)
    covered = np.asarray(
        [
            interval.covers(float(point), float(truth))
            for point, truth in zip(values, truths, strict=True)
        ],
        dtype=bool,
    )
    return float(1.0 - covered.mean())


DEFAULT_ALPHA: Final[float] = 0.1
