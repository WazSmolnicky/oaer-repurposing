"""Ranking read-outs: Hits@k, nDCG@k and the top-``k`` overlap.

The catalogue ranking is read inside the eligible set ``D_eps(s)`` only, since a
candidate the gate rejected is not ranked at all. Relevance is the sign-verified
benefit: a candidate counts as a hit when its effect is positive, and the graded
gain for nDCG is the doubly robust effect itself, so a strongly positive candidate
contributes more than a marginal one.

Ref: Sec. 1.3 (Hits@20, nDCG@20), Sec. 1.4 (Top-5 Jaccard).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def hits_at_k(relevance: npt.ArrayLike, k: int) -> float:
    """The share of the leading ``k`` entries whose relevance is positive."""

    values = np.asarray(relevance, dtype=np.float64).ravel()
    if values.size == 0 or k < 1:
        return 0.0
    take = min(k, values.size)
    return float((values[:take] > 0.0).mean())


def discount(k: int) -> float:
    return float(np.log2(k + 1.0))


def ndcg_at_k(relevance: npt.ArrayLike, k: int) -> float:
    """Normalised discounted cumulative gain over the leading ``k`` entries.

    Gains are the positive part of the values, so a harmful candidate contributes
    no gain rather than a negative one, and the ideal ordering is the values
    sorted from largest to smallest.
    """

    values = np.asarray(relevance, dtype=np.float64).ravel()
    if values.size == 0 or k < 1:
        return 0.0
    take = min(k, values.size)
    gains = np.maximum(values[:take], 0.0)
    discounts = np.log2(np.arange(2, take + 2))
    actual = float((gains / discounts).sum())
    ideal_gains = np.sort(np.maximum(values, 0.0))[::-1][:take]
    ideal = float((ideal_gains / discounts).sum())
    if ideal <= 0.0:
        return 0.0
    return actual / ideal


def precision_at_k(relevance: npt.ArrayLike, k: int) -> float:
    values = np.asarray(relevance, dtype=np.float64).ravel()
    if values.size == 0 or k < 1:
        return 0.0
    take = min(k, values.size)
    return float((values[:take] > 0.0).sum() / take)


def recall_at_k(relevance: npt.ArrayLike, k: int) -> float:
    values = np.asarray(relevance, dtype=np.float64).ravel()
    positives = int((values > 0.0).sum())
    if positives == 0:
        return 0.0
    take = min(k, values.size)
    return float((values[:take] > 0.0).sum() / positives)


def average_precision(relevance: npt.ArrayLike) -> float:
    values = np.asarray(relevance, dtype=np.float64).ravel()
    positives = int((values > 0.0).sum())
    if positives == 0:
        return 0.0
    hits = (values > 0.0).astype(np.float64)
    cumulative = np.cumsum(hits)
    positions = np.arange(1, values.size + 1)
    precision = cumulative / positions
    return float((precision * hits).sum() / positives)


def top_k_set(scores: npt.ArrayLike, identifiers: Sequence[str], k: int) -> tuple[str, ...]:
    """The leading ``k`` identifiers by decreasing score, ties by identifier."""

    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size != len(identifiers):
        raise ValueError("every score needs an identifier")
    if k < 1:
        raise ValueError("the rank depth must be at least one")
    order = sorted(range(values.size), key=lambda index: (-values[index], identifiers[index]))
    return tuple(identifiers[index] for index in order[:k])


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    """The overlap of two recommended sets."""

    first, second = set(left), set(right)
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def mean_jaccard_across_sets(sets: Sequence[Sequence[str]]) -> float:
    """The mean pairwise Jaccard overlap of a collection of top-``k`` sets."""

    if len(sets) < 2:
        return float("nan")
    values: list[float] = []
    for first in range(len(sets)):
        for second in range(first + 1, len(sets)):
            values.append(jaccard(sets[first], sets[second]))
    return float(np.mean(values))


def seed_dispersion(values: npt.ArrayLike) -> float:
    """The across-seed standard deviation of a read-out."""

    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1))


def hits_table(
    rankings: dict[str, npt.ArrayLike],
    depth: int,
) -> dict[str, float]:
    """Hits@``depth`` per stratum, keyed by the stratum name."""

    return {name: hits_at_k(values, depth) for name, values in sorted(rankings.items())}


def ndcg_table(
    rankings: dict[str, npt.ArrayLike],
    depth: int,
) -> dict[str, float]:
    return {name: ndcg_at_k(values, depth) for name, values in sorted(rankings.items())}


def recall_table(
    rankings: dict[str, npt.ArrayLike],
    depths: Sequence[int],
) -> dict[str, dict[int, float]]:
    return {
        name: {depth: recall_at_k(values, depth) for depth in depths}
        for name, values in sorted(rankings.items())
    }
