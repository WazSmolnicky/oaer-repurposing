"""Estimands: target-trial emulation, nuisance estimation and the pseudo-outcome.

The training target is an out-of-fold doubly robust pseudo-outcome. This package
builds it: the clone-censor-weight design that places time zero, the line-specific
restricted-time horizons, the nuisance fits (propensity, arm-specific outcome
regressions, censoring survival, the conditional restricted-mean function and the
censoring martingale), Eq. (2) itself, the trimming rule that defines the
eligible set, and the sensitivity read-outs each emulation reports beside its
estimate.
"""

from __future__ import annotations

from oaer.estimands.cloning import (
    Clone,
    CloneCensorWeight,
    censor_and_weight,
    clone_decision,
    weighted_person_time,
)
from oaer.estimands.horizons import (
    HORIZON_GRID,
    integration_grid,
    line_horizon,
    real_world_medians,
    restricted_time,
)
from oaer.estimands.nuisance import (
    CensoringSurvival,
    LogisticPropensity,
    NuisanceFit,
    OutcomeRegression,
    RidgeOutcomeModel,
    fit_nuisances,
    martingale_increment,
    restricted_mean_function,
)
from oaer.estimands.pseudo_outcome import (
    DoublyRobustEstimate,
    OutcomeTable,
    build_outcome_table,
    doubly_robust_pseudo_outcome,
)
from oaer.estimands.sensitivity import (
    EValue,
    e_value,
    negative_control_estimate,
    risk_ratio_at_horizon,
)
from oaer.estimands.trimming import (
    EligibilityRule,
    IneligibleReason,
    TrimmingOutcome,
    apply_trimming,
    eligible_set,
)

__all__ = [
    "HORIZON_GRID",
    "CensoringSurvival",
    "Clone",
    "CloneCensorWeight",
    "DoublyRobustEstimate",
    "EValue",
    "EligibilityRule",
    "IneligibleReason",
    "LogisticPropensity",
    "NuisanceFit",
    "OutcomeRegression",
    "OutcomeTable",
    "RidgeOutcomeModel",
    "TrimmingOutcome",
    "apply_trimming",
    "build_outcome_table",
    "censor_and_weight",
    "clone_decision",
    "doubly_robust_pseudo_outcome",
    "e_value",
    "eligible_set",
    "fit_nuisances",
    "integration_grid",
    "line_horizon",
    "martingale_increment",
    "negative_control_estimate",
    "real_world_medians",
    "restricted_mean_function",
    "restricted_time",
    "risk_ratio_at_horizon",
    "weighted_person_time",
]
