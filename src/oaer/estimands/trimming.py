"""The identification gate's first stage: the eligible set ``D_eps(s)``.

A candidate is outcome-identified for a decision when its estimated propensity
lies inside ``[eps, 1 - eps]`` and its exposed count reaches ``nmin``. Anything
else is carried in the not-outcome-identified block with its reason -- trimming,
or exposure below the minimum -- rather than dropped silently, because the block
is one of the four outputs the frozen model produces.

Ref: Sec. 2.3 (D_eps), Sec. 2.4 (Eq. (1) and the D_eps = empty case),
Algorithm 2 lines 8-9, Algorithm 3 lines 2-7.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.support.types import Decision

MINIMUM_EXPOSURE_DEFAULT: Final[int] = 40
TRIMMING_EPSILON_DEFAULT: Final[float] = 0.05


class IneligibleReason(str, Enum):
    """Why a candidate failed to enter the eligible set."""

    TRIMMING = "trimming"
    EXPOSURE_BELOW_MINIMUM = "exposure below nmin"


@dataclass(frozen=True)
class EligibilityRule:
    """The trimming rule the gate applies, with its two thresholds."""

    epsilon: float = TRIMMING_EPSILON_DEFAULT
    minimum_exposed: int = MINIMUM_EXPOSURE_DEFAULT

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("the trimming fraction must lie strictly inside (0, 0.5)")
        if self.minimum_exposed < 1:
            raise ValueError("the minimum exposed count must be at least one")

    def admit(self, propensity: float, exposed_count: int) -> IneligibleReason | None:
        """``None`` when the candidate is identified, else the blocking reason."""

        if exposed_count < self.minimum_exposed:
            return IneligibleReason.EXPOSURE_BELOW_MINIMUM
        if propensity < self.epsilon or propensity > 1.0 - self.epsilon:
            return IneligibleReason.TRIMMING
        return None

    def admissible_grid(self) -> tuple[float, float]:
        return (self.epsilon, 1.0 - self.epsilon)


@dataclass(frozen=True)
class TrimmingOutcome:
    """The eligible set of one decision, and the block that did not enter it."""

    decision_id: str
    eligible: tuple[str, ...]
    blocked: tuple[tuple[str, IneligibleReason], ...]

    @property
    def size(self) -> int:
        return len(self.eligible)

    @property
    def is_empty(self) -> bool:
        return not self.eligible

    def reason_counts(self) -> Mapping[IneligibleReason, int]:
        counts = dict.fromkeys(IneligibleReason, 0)
        for _, reason in self.blocked:
            counts[reason] += 1
        return counts


def eligible_set(
    candidate_pairs: Sequence[tuple[str, float, int]],
    rule: EligibilityRule | None = None,
) -> TrimmingOutcome:
    """Partition ``(candidate, propensity, exposed count)`` triples by the rule."""

    active = rule if rule is not None else EligibilityRule()
    eligible: list[str] = []
    blocked: list[tuple[str, IneligibleReason]] = []
    for candidate, propensity, exposed in candidate_pairs:
        verdict = active.admit(propensity, exposed)
        if verdict is None:
            eligible.append(candidate)
        else:
            blocked.append((candidate, verdict))
    return TrimmingOutcome(
        decision_id="",
        eligible=tuple(eligible),
        blocked=tuple(blocked),
    )


def apply_trimming(
    decision: Decision,
    propensities: Mapping[str, float],
    exposed_counts: Mapping[str, int],
    candidates: Sequence[str],
    rule: EligibilityRule | None = None,
) -> TrimmingOutcome:
    """The eligible set of one decision under the rule."""

    triples = [
        (candidate, float(propensities.get(candidate, 0.0)), int(exposed_counts.get(candidate, 0)))
        for candidate in candidates
    ]
    partitioned = eligible_set(triples, rule)
    return TrimmingOutcome(
        decision_id=decision.decision_id,
        eligible=partitioned.eligible,
        blocked=partitioned.blocked,
    )


def propensity_margins(
    propensities: npt.ArrayLike,
    rule: EligibilityRule | None = None,
) -> npt.NDArray[np.float64]:
    """How far each propensity sits inside the admissible interval.

    A negative value means the propensity is trimmed; the margin is used by the
    sensitivity sweep rather than by the gate itself.
    """

    active = rule if rule is not None else EligibilityRule()
    values = np.asarray(propensities, dtype=np.float64)
    lower = values - active.epsilon
    upper = (1.0 - active.epsilon) - values
    return np.minimum(lower, upper)


def trimming_rate(
    propensities: npt.ArrayLike,
    rule: EligibilityRule | None = None,
) -> float:
    """The share of propensities the rule trims."""

    margins = propensity_margins(propensities, rule)
    if margins.size == 0:
        return 0.0
    return float((margins < 0.0).mean())


def eligible_sizes(
    propensities: Mapping[str, float],
    exposed_counts: Mapping[str, int],
    rule: EligibilityRule | None = None,
) -> int:
    """How many candidates one decision keeps under the rule."""

    active = rule if rule is not None else EligibilityRule()
    return int(
        sum(
            1
            for candidate, propensity in propensities.items()
            if active.admit(float(propensity), int(exposed_counts.get(candidate, 0))) is None
        )
    )


def block_catalogue(
    trimmed: Sequence[TrimmingOutcome],
) -> Mapping[IneligibleReason, int]:
    """Reason totals across decisions, which the not-identified block reports."""

    totals = dict.fromkeys(IneligibleReason, 0)
    for outcome in trimmed:
        for reason, count in outcome.reason_counts().items():
            totals[reason] += count
    return totals


def trimming_sweep(
    epsilon: float,
    propensities: npt.ArrayLike,
) -> float:
    """The trimmed fraction at one candidate value of ``eps``."""

    return trimming_rate(propensities, EligibilityRule(epsilon=epsilon))
