"""The gate's four outputs: ranking, harm block, not-identified block and the flag.

For each decision the frozen model produces a ranking of the eligible candidates
with their intervals and the path evidence behind them, a harm block of
candidates whose lower conformal limit is negative, the not-outcome-identified
block with a per-candidate reason, and an act-or-defer signal. Deferral happens
when the eligible set is empty or when no candidate's lower limit clears the
margin, and the flag carries the reason it fired.

Ref: Sec. 2.4 (outputs; Eq. (1)), Sec. 2.7 (deferral rule), Algorithm 3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from oaer.estimands.trimming import IneligibleReason
from oaer.gate.conformal import WeightedConformal


class DeferralReason(str, Enum):
    """Why the act-or-defer flag fired."""

    NONE = "no deferral"
    EMPTY_ELIGIBLE_SET = "eligible set empty"
    MARGIN_NOT_CLEARED = "margin not cleared"


@dataclass(frozen=True)
class PathEvidence:
    """The evidence attached to one ranked candidate.

    The path is the motif sequence the encoder walked to place the candidate for
    this state, and the two counts are the exposed decisions behind the score and
    the trimmed fraction of the candidate's emulation.
    """

    candidate: str
    path: tuple[str, ...]
    exposed_count: int
    trimmed_fraction: float
    propensity: float
    mechanism: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateDecision:
    """One decision's gate output."""

    decision_id: str
    ranking: tuple[str, ...]
    scores: Mapping[str, float]
    lower_limits: Mapping[str, float]
    harm: tuple[str, ...]
    not_identified: tuple[tuple[str, IneligibleReason], ...]
    act: bool
    reason: DeferralReason
    evidence: Mapping[str, PathEvidence] = field(default_factory=dict)

    @property
    def top_three(self) -> tuple[str, ...]:
        return self.ranking[:3]

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "ranking": list(self.ranking),
            "harm": list(self.harm),
            "not_identified": [[name, reason.value] for name, reason in self.not_identified],
            "act": self.act,
            "reason": self.reason.value,
        }


@dataclass(frozen=True)
class GateOutcome:
    """Every decision's gate output, with the counters the read-outs need."""

    decisions: tuple[GateDecision, ...]

    @property
    def deferral_rate(self) -> float:
        if not self.decisions:
            return 0.0
        return float(np.mean([not decision.act for decision in self.decisions]))

    @property
    def harm_rate(self) -> float:
        if not self.decisions:
            return 0.0
        return float(np.mean([bool(decision.harm) for decision in self.decisions]))

    def reason_counts(self) -> Mapping[str, int]:
        counts = {reason.value: 0 for reason in DeferralReason}
        for decision in self.decisions:
            counts[decision.reason.value] += 1
        return counts

    def eligible_sizes(self) -> np.ndarray:
        return np.asarray(
            [len(decision.ranking) + len(decision.harm) for decision in self.decisions]
        )

    def acted(self) -> tuple[GateDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.act)


def harm_block(
    scores: Mapping[str, float],
    conformal: WeightedConformal,
) -> tuple[str, ...]:
    """Candidates whose lower conformal limit is strictly negative."""

    return tuple(
        candidate
        for candidate, score in sorted(scores.items())
        if conformal.lower_limit(score) < 0.0
    )


def apply_gate(
    decision_id: str,
    scores: Mapping[str, float],
    eligible: Sequence[str],
    not_identified: Sequence[tuple[str, IneligibleReason]],
    conformal: WeightedConformal,
    *,
    margin: float = 0.0,
    rank_depth: int = 3,
    evidence: Mapping[str, PathEvidence] | None = None,
) -> GateDecision:
    """Apply Algorithm 3's ordering, sign condition, harm block and flag."""

    identified = {name: float(scores[name]) for name in eligible if name in scores}
    limits = {name: conformal.lower_limit(value) for name, value in identified.items()}
    harm = harm_block(identified, conformal)
    ordered = sorted(
        (name for name in identified if name not in harm and identified[name] > 0.0),
        key=lambda name: (-identified[name], name),
    )
    if not identified:
        act, reason = False, DeferralReason.EMPTY_ELIGIBLE_SET
    elif not ordered:
        act, reason = False, DeferralReason.MARGIN_NOT_CLEARED
    else:
        best = max(limits[name] for name in ordered[: max(rank_depth, 1)])
        cleared = best > margin
        act = bool(cleared)
        reason = DeferralReason.NONE if act else DeferralReason.MARGIN_NOT_CLEARED
    return GateDecision(
        decision_id=decision_id,
        ranking=tuple(ordered),
        scores=identified,
        lower_limits=limits,
        harm=harm,
        not_identified=tuple(not_identified),
        act=act,
        reason=reason,
        evidence=dict(evidence or {}),
    )


def pathway_readout(
    outcomes: Sequence[GateDecision],
    started: Mapping[str, Sequence[str]],
) -> Mapping[str, dict[str, object]]:
    """The recommendation-to-action-to-outcome pathway of Table 1 panel e.

    Each decision falls in one pathway by where the agents actually started sit
    relative to the ranking: the top-ranked identified candidate, another top-three
    candidate, an identified candidate outside the top three, the backbone only,
    a de-prioritised agent co-prescribed, or no agent started.
    """

    buckets: dict[str, list[str]] = {
        "top_ranked_started": [],
        "another_top3_started": [],
        "outside_top3_started": [],
        "backbone_only": [],
        "de_prioritised_coprescribed": [],
        "none_started": [],
    }
    agreements: list[str] = []
    discordant: list[str] = []
    for decision in outcomes:
        agents = set(started.get(decision.decision_id, ()))
        top_three = set(decision.top_three)
        if agents and decision.ranking and decision.ranking[0] in agents:
            buckets["top_ranked_started"].append(decision.decision_id)
        elif agents & top_three:
            buckets["another_top3_started"].append(decision.decision_id)
        elif agents & set(decision.ranking):
            buckets["outside_top3_started"].append(decision.decision_id)
        elif agents & set(decision.harm):
            buckets["de_prioritised_coprescribed"].append(decision.decision_id)
        elif agents:
            buckets["backbone_only"].append(decision.decision_id)
        else:
            buckets["none_started"].append(decision.decision_id)
        # The supporting contrast splits started agents by whether the start
        # agrees with the top three or departs from it.
        if agents & top_three:
            agreements.append(decision.decision_id)
        elif agents:
            discordant.append(decision.decision_id)
    total = max(len(outcomes), 1)
    payload: dict[str, dict[str, object]] = {
        pathway: {"n": len(members), "share": len(members) / total}
        for pathway, members in buckets.items()
    }
    payload["top_agreement"] = {"n": len(agreements), "share": len(agreements) / total}
    payload["discordant_initiation"] = {"n": len(discordant), "share": len(discordant) / total}
    return payload
