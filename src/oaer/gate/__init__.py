"""The identification gate, the weighted conformal rule and the harm block.

The gate is the third construct. It first bounds the set of candidates the score
is allowed to rank -- the eligible set ``D_eps(s)`` -- then attaches a weighted
split-conformal interval to each eligible candidate, forms the harm block from the
candidates whose lower limit is negative, and returns an act-or-defer flag. The
conformal calibration set is the most recent retrospective window of Sites A and
B only, with covariate-only weights that transport it to the target site and
period, so a held-out site never calibrates its own gate.
"""

from __future__ import annotations

from oaer.gate.conformal import (
    ConformalCalibration,
    WeightedConformal,
    conformal_lower_limit,
    covariate_weights,
    weighted_quantile,
)
from oaer.gate.harm import (
    DeferralReason,
    GateDecision,
    GateOutcome,
    PathEvidence,
    apply_gate,
    harm_block,
)
from oaer.gate.identification import (
    IdentificationGate,
    identified_set,
    propensity_range,
)

__all__ = [
    "ConformalCalibration",
    "DeferralReason",
    "GateDecision",
    "GateOutcome",
    "IdentificationGate",
    "PathEvidence",
    "WeightedConformal",
    "apply_gate",
    "conformal_lower_limit",
    "covariate_weights",
    "harm_block",
    "identified_set",
    "propensity_range",
    "weighted_quantile",
]
