"""Stage 1 of the gate: the eligible set and the propensity range it reports.

Identification is a property of the candidate at a decision, not of the
candidate: the same agent can be identified at one site and trimmed at another,
which is why the gate is evaluated per decision and the block of candidates it
rejects is reported with a reason rather than dropped.

Ref: Sec. 2.3 (``D_eps``), Sec. 2.4 (Eq. (1)), Algorithm 3 lines 2-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from oaer.estimands.trimming import EligibilityRule, IneligibleReason


@dataclass(frozen=True)
class IdentificationGate:
    """The gate's first stage over one decision."""

    rule: EligibilityRule
    eligible: tuple[str, ...]
    blocked: tuple[tuple[str, IneligibleReason], ...]
    propensities: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.eligible)

    @property
    def is_empty(self) -> bool:
        return not self.eligible

    def is_eligible(self, candidate: str) -> bool:
        return candidate in self.eligible

    def reason_for(self, candidate: str) -> IneligibleReason | None:
        for name, reason in self.blocked:
            if name == candidate:
                return reason
        return None


def identified_set(
    propensities: dict[str, float],
    exposed_counts: dict[str, int],
    rule: EligibilityRule | None = None,
) -> IdentificationGate:
    """Partition the catalogue into the eligible set and the blocked block."""

    active = rule if rule is not None else EligibilityRule()
    eligible: list[str] = []
    blocked: list[tuple[str, IneligibleReason]] = []
    for candidate in sorted(propensities):
        reason = active.admit(float(propensities[candidate]), int(exposed_counts.get(candidate, 0)))
        if reason is None:
            eligible.append(candidate)
        else:
            blocked.append((candidate, reason))
    return IdentificationGate(
        rule=active,
        eligible=tuple(eligible),
        blocked=tuple(blocked),
        propensities={name: float(propensities[name]) for name in propensities},
    )


def propensity_range(values: npt.ArrayLike) -> tuple[float, float]:
    """The range of propensities the gate reports beside a decision."""

    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return (float("nan"), float("nan"))
    return (float(array.min()), float(array.max()))


def trimmed_fraction(propensities: npt.ArrayLike, rule: EligibilityRule | None = None) -> float:
    """The share of propensities the rule rejects."""

    active = rule if rule is not None else EligibilityRule()
    array = np.asarray(propensities, dtype=np.float64).ravel()
    if array.size == 0:
        return 0.0
    outside = (array < active.epsilon) | (array > 1.0 - active.epsilon)
    return float(outside.mean())


def gate_diagnostics(gate: IdentificationGate) -> dict[str, float]:
    """Counters for the audit block."""

    counts = dict.fromkeys(IneligibleReason, 0)
    for _, reason in gate.blocked:
        counts[reason] += 1
    return {
        "eligible": float(gate.size),
        "blocked": float(len(gate.blocked)),
        "trimming": float(counts[IneligibleReason.TRIMMING]),
        "exposure_below_minimum": float(counts[IneligibleReason.EXPOSURE_BELOW_MINIMUM]),
    }
