"""Resampling: the site-stratified bootstrap and the paired seed test.

Every interval the article prints is a site-stratified bootstrap interval: the
resampling is done inside each site and the replicates are pooled, so a site that
contributes few decisions cannot be drowned by one that contributes many. The
``P vs OAER`` column is a paired test across the ten fixed seeds, which tests
optimization noise rather than sampling uncertainty and therefore licenses no
population claim.

Ref: Table 2 note (ten fixed seeds, paired test), Table 3 (site-stratified
bootstrap), Sec. 1.4 (IAB carries the same bootstrap interval).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from oaer.support.numerics import Interval, normal_quantile
from oaer.support.seeding import spawn_generator


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its bootstrap interval and the draws behind it."""

    point: float
    interval: Interval
    draws: npt.NDArray[np.float64]

    @property
    def standard_error(self) -> float:
        return float(self.draws.std(ddof=1)) if self.draws.size > 1 else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "point": round(self.point, 6),
            "lower": round(self.interval.lower, 6),
            "upper": round(self.interval.upper, 6),
            "standard_error": round(self.standard_error, 6),
        }


def site_indices(sites: Sequence[str]) -> dict[str, npt.NDArray[np.int64]]:
    """The row indices of each site, in a stable order."""

    grouped: dict[str, list[int]] = {}
    for index, site in enumerate(sites):
        grouped.setdefault(site, []).append(index)
    return {site: np.asarray(indices, dtype=np.int64) for site, indices in sorted(grouped.items())}


def site_stratified_bootstrap(
    statistic: Callable[[npt.NDArray[np.int64]], float],
    sites: Sequence[str],
    *,
    resamples: int = 400,
    confidence: float = 0.95,
    seed: int = 20260831,
) -> BootstrapResult:
    """Resample inside each site, pool the replicates and take a percentile interval.

    ``statistic`` receives a row-index array and returns the read-out on that
    resample, so the resampling is independent of what is being measured. The
    point estimate is the statistic on the identity index set.
    """

    if resamples < 1:
        raise ValueError("a bootstrap needs at least one resample")
    identity = np.arange(len(sites), dtype=np.int64)
    point = float(statistic(identity))
    grouped = site_indices(sites)
    stream = spawn_generator(seed, f"bootstrap:{resamples}")
    draws = np.zeros(resamples, dtype=np.float64)
    for replicate in range(resamples):
        pieces: list[npt.NDArray[np.int64]] = []
        for indices in grouped.values():
            drawn = stream.integers(0, indices.size, size=indices.size)
            pieces.append(indices[drawn])
        draws[replicate] = float(statistic(np.concatenate(pieces)))
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1.0 - alpha))
    return BootstrapResult(point=point, interval=Interval(point, lower, upper), draws=draws)


def paired_seed_test(
    candidate: Sequence[float],
    reference: Sequence[float],
) -> float:
    """A paired two-sided ``t`` test across seeds, as the article's P column is.

    The test is on the seed differences, so it measures optimization noise and not
    sampling uncertainty; it is reported with that reading attached.
    """

    left = np.asarray(candidate, dtype=np.float64).ravel()
    right = np.asarray(reference, dtype=np.float64).ravel()
    if left.size != right.size:
        raise ValueError("the paired test needs the same number of seeds on both sides")
    if left.size < 2:
        return float("nan")
    difference = left - right
    error = float(difference.std(ddof=1) / np.sqrt(difference.size))
    if error <= 0.0:
        return 0.0 if abs(float(difference.mean())) < 1e-12 else 1e-12
    statistic = float(difference.mean()) / error
    return float(2.0 * (1.0 - _normal_cdf(abs(statistic))))


def one_sample_seed_test(values: Sequence[float], constant: float) -> float:
    """A one-sample test of the seed values against a constant.

    The deterministic published formulas carry no across-seed dispersion, so their
    comparison column is this test rather than the paired one.
    """

    return paired_seed_test(values, [constant] * len(values))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + _erf(value / float(np.sqrt(2.0))))


def _erf(value: float) -> float:
    """Abramowitz and Stegun 7.1.26, accurate to about 1.5e-7."""

    sign = -1.0 if value < 0 else 1.0
    x = abs(value)
    t = 1.0 / (1.0 + 0.3275911 * x)
    polynomial = (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t
    return sign * (1.0 - polynomial * float(np.exp(-x * x)))


def bootstrap_contrast(
    statistic: Callable[[npt.NDArray[np.int64]], float],
    sites: Sequence[str],
    *,
    resamples: int = 400,
    confidence: float = 0.95,
    seed: int = 20260831,
) -> Interval:
    """A bootstrap interval whose point estimate is the statistic's own value."""

    result = site_stratified_bootstrap(
        statistic, sites, resamples=resamples, confidence=confidence, seed=seed
    )
    return result.interval


def delta_with_interval(
    full: float,
    ablated: float,
    *,
    full_interval: Interval,
    ablated_interval: Interval,
) -> tuple[float, Interval]:
    """A removal delta with its interval, propagated in quadrature."""

    delta = ablated - full
    half = float(np.sqrt(full_interval.width**2 + ablated_interval.width**2) / 2.0)
    return (delta, Interval(point=delta, lower=delta - half, upper=delta + half))


def interval_half_width(points: npt.ArrayLike, confidence: float = 0.95) -> float:
    """A normal half-width from a set of draws."""

    values = np.asarray(points, dtype=np.float64).ravel()
    if values.size < 2:
        return float("nan")
    return normal_quantile(0.5 + confidence / 2.0) * float(
        values.std(ddof=1) / np.sqrt(values.size)
    )


def pooled_site_interval(
    per_site: Mapping[str, tuple[float, float]],
    confidence: float = 0.95,
) -> Interval:
    """Pool per-site point estimates and errors by inverse-variance weighting."""

    if not per_site:
        return Interval(float("nan"), float("nan"), float("nan"))
    weights: list[float] = []
    values: list[float] = []
    for point, error in per_site.values():
        if not np.isfinite(point) or not np.isfinite(error) or error <= 0.0:
            continue
        weights.append(1.0 / error**2)
        values.append(point)
    if not weights:
        return Interval(float("nan"), float("nan"), float("nan"))
    mass = np.asarray(weights, dtype=np.float64)
    estimate = float(np.dot(mass, np.asarray(values, dtype=np.float64)) / mass.sum())
    error = float(np.sqrt(1.0 / mass.sum()))
    half = normal_quantile(0.5 + confidence / 2.0) * error
    return Interval(point=estimate, lower=estimate - half, upper=estimate + half)
