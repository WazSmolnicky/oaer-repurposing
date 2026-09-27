"""The top-``k`` policy value and its doubly robust estimator.

For a ranking rule whose recommended set at state ``s`` is ``T_k^pi(s)``,

    V_k(pi) = E[ min{k, |T_k^pi(S)|}^-1 * sum_{d in T_k^pi(S)} Gamma_d ],

which is Theorem 1(b). The deployed read-out is the doubly robust top-three value
``V3``: the same functional evaluated on the cross-fitted pseudo-outcomes of the
candidates the rule puts in its top three. A policy value can be computed from
any ranking, which is why the comparator families that emit no effect estimate are
still scored in the same currency.

Ref: Theorem 1(b), Sec. 1.3 (V3 as the reported policy value), Lemma 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from oaer.support.numerics import normal_quantile


@dataclass(frozen=True)
class PolicyValue:
    """A policy value with its standard error and the depth it was read at."""

    depth: int
    value: float
    standard_error: float
    decisions: int
    covered: int

    @property
    def interval(self) -> tuple[float, float]:
        half = normal_quantile(0.975) * self.standard_error
        return (self.value - half, self.value + half)

    def as_dict(self) -> dict[str, float | int]:
        lower, upper = self.interval
        return {
            "depth": self.depth,
            "value": round(self.value, 6),
            "standard_error": round(self.standard_error, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "decisions": self.decisions,
            "covered": self.covered,
        }


def policy_value_k(
    pseudo_outcomes: npt.ArrayLike,
    depth: int,
) -> float:
    """Theorem 1(b) on a single state: the mean pseudo-outcome of the top-``k`` set.

    ``pseudo_outcomes`` is the vector of eligible candidates' values in the order
    the rule recommends them, so the expression is the mean of the leading
    ``min(k, |T|)`` entries.
    """

    values = np.asarray(pseudo_outcomes, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    take = min(depth, values.size)
    if take < 1:
        raise ValueError("the rank depth must be at least one")
    return float(values[:take].mean())


def doubly_robust_top_three_value(
    table: Mapping[str, npt.ArrayLike],
    rankings: Mapping[str, Sequence[str]],
    rows: Mapping[str, int],
    depth: int = 3,
) -> PolicyValue:
    """``V3`` over a population.

    ``table`` maps a candidate to its per-decision pseudo-outcome column,
    ``rankings`` maps a decision to the rule's ordered candidate list, and
    ``rows`` maps a decision to its row in that column. A decision whose eligible
    set is empty contributes a zero value and is counted as uncovered, because
    the rule recommends nothing there.
    """

    per_decision: list[float] = []
    covered = 0
    for decision_id, ordered in rankings.items():
        row = rows[decision_id]
        values = [
            float(np.asarray(table[candidate], dtype=np.float64)[row])
            for candidate in ordered
            if candidate in table
        ]
        if not values:
            per_decision.append(0.0)
            continue
        covered += 1
        per_decision.append(policy_value_k(values, depth))
    array = np.asarray(per_decision, dtype=np.float64)
    if array.size == 0:
        return PolicyValue(depth, float("nan"), float("nan"), 0, 0)
    error = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
    return PolicyValue(
        depth=depth,
        value=float(array.mean()),
        standard_error=error,
        decisions=int(array.size),
        covered=covered,
    )


def table_columns(
    values: npt.NDArray[np.float64],
    candidates: Sequence[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """Index a pseudo-outcome matrix by candidate."""

    if values.shape[1] != len(candidates):
        raise ValueError("the matrix width must match the candidate list")
    return {candidate: values[:, index] for index, candidate in enumerate(candidates)}


def top_three_rankings(
    values: npt.NDArray[np.float64],
    candidates: Sequence[str],
    eligible: npt.NDArray[np.bool_],
    decision_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Per decision, the ordered eligible candidates carrying a positive value."""

    rankings: dict[str, tuple[str, ...]] = {}
    for row, decision_id in enumerate(decision_ids):
        names = [
            candidates[index]
            for index in range(len(candidates))
            if eligible[row, index] and values[row, index] > 0.0
        ]
        names.sort(key=lambda name: (-values[row, candidates.index(name)], name))
        rankings[decision_id] = tuple(names)
    return rankings


def policy_contrast(
    agreement: PolicyValue,
    no_initiation: PolicyValue,
) -> dict[str, float]:
    """The weighted restricted-mean policy contrast of the two strategies."""

    difference = agreement.value - no_initiation.value
    variance = agreement.standard_error**2 + no_initiation.standard_error**2
    error = float(np.sqrt(variance)) if variance > 0.0 else float("nan")
    half = normal_quantile(0.975) * error if np.isfinite(error) else float("nan")
    return {
        "contrast": difference,
        "standard_error": error,
        "lower": difference - half,
        "upper": difference + half,
    }


def selection_regret(
    values: npt.NDArray[np.float64],
    eligible: npt.NDArray[np.bool_],
    depth: int = 3,
) -> float:
    """The mean gap between the rule's top-``k`` mean and the attainable best.

    The attainable best is the mean of the ``k`` largest eligible values at that
    decision, which the rule would reach if its scores were exact. It bounds the
    regret Proposition 3 decomposes into a curation term and an estimation term.
    """

    gaps: list[float] = []
    for row in range(values.shape[0]):
        selected = values[row][eligible[row]]
        if selected.size == 0:
            continue
        take = min(depth, selected.size)
        order = np.sort(selected)[::-1]
        gaps.append(float(order[:take].mean() - selected[:take].mean()))
    if not gaps:
        return 0.0
    return float(np.mean(gaps))


def orbit_regret_floor(
    orbit_sizes: Sequence[int],
    spread_within_orbit: Sequence[float],
    depth: int,
) -> float:
    """The regret floor a relational encoder cannot avoid.

    When the encoder reads only structure, candidates in one orbit receive one
    score, so the regret cannot fall below the mean within-orbit spread of the
    values inside the top-``k`` boundary. The floor is a function of the orbit
    partition alone and does not shrink with the cohort, which is the point of
    Proposition 3.
    """

    if len(orbit_sizes) != len(spread_within_orbit):
        raise ValueError("every orbit needs its size and its spread")
    if depth < 1:
        raise ValueError("the rank depth must be at least one")
    total = sum(orbit_sizes)
    if total == 0:
        return 0.0
    weighted = sum(
        size * spread for size, spread in zip(orbit_sizes, spread_within_orbit, strict=True)
    )
    return float(weighted / total)
