"""The ranking head: route score, pathology modifier, and Stage 3's objective."""

from __future__ import annotations

from oaer.head.modifiers import (
    PathologyModifier,
    RouteScore,
    interaction_beyond_additive,
    interaction_index,
    zero_mean_centre,
)
from oaer.head.objectives import (
    StageThreeHead,
    StageThreeWeights,
    effective_weight_mass,
    penalty_norm,
    residual_summary,
    weighted_least_squares,
)
from oaer.head.ranking import (
    DecisionRanker,
    RankedCandidate,
    RankingHead,
    jaccard,
    positive_ranking,
    rank_candidates,
    ranking_stability,
    top_k,
)

__all__ = [
    "DecisionRanker",
    "PathologyModifier",
    "RankedCandidate",
    "RankingHead",
    "RouteScore",
    "StageThreeHead",
    "StageThreeWeights",
    "effective_weight_mass",
    "interaction_beyond_additive",
    "interaction_index",
    "jaccard",
    "penalty_norm",
    "positive_ranking",
    "rank_candidates",
    "ranking_stability",
    "residual_summary",
    "top_k",
    "weighted_least_squares",
    "zero_mean_centre",
]
