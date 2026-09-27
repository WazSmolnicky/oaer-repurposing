"""The ranking head ``tau_phi`` and the score it produces.

``tau_phi(h_theta(s, d)) = g_phi(h) + b_phi(slide)`` after the zero-mean
condition has been applied to the modifier over ``D_eps(s)``. The head is common
across candidates -- Theorem 1(c) needs one head that is efficient for every rank
depth at once -- so the candidate enters only through the representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from oaer.head.modifiers import PathologyModifier, RouteScore, zero_mean_centre


class RankingHead(nn.Module):
    """Route score plus centred pathology modifier."""

    def __init__(
        self, width: int, slide_dim: int, route_hidden: int = 32, modifier_hidden: int = 16
    ) -> None:
        super().__init__()
        self.route = RouteScore(width, route_hidden)
        self.modifier = PathologyModifier(slide_dim, modifier_hidden)

    def forward(
        self,
        representation: Tensor,
        slide: Tensor,
        eligible: Tensor | None = None,
        slide_available: Tensor | None = None,
    ) -> Tensor:
        route = self.route(representation)
        modifier = self.modifier(slide, slide_available)
        combined: Tensor = route + zero_mean_centre(modifier, eligible)
        return combined

    def components(
        self,
        representation: Tensor,
        slide: Tensor,
        eligible: Tensor | None = None,
        slide_available: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The two pieces separately, for the component ablation."""

        route = self.route(representation)
        modifier = zero_mean_centre(self.modifier(slide, slide_available), eligible)
        return route, modifier


class DecisionRanker(nn.Module):
    """Score every candidate at every decision from the state alone.

    The head is shared across candidates, as Theorem 1(c) needs; a candidate
    enters through its own embedding, so the same parameters score the whole
    catalogue and a candidate the head never saw fitted still receives a score.
    """

    def __init__(
        self,
        state_dim: int,
        candidate_count: int,
        candidate_dim: int = 8,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if state_dim < 1 or candidate_count < 1:
            raise ValueError("the ranker needs a state width and at least one candidate")
        self.state_dim = state_dim
        self.candidate_count = candidate_count
        self.candidate_dim = candidate_dim
        self.patient = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
        )
        self.candidate = nn.Embedding(candidate_count, candidate_dim)
        self.interaction = nn.Sequential(
            nn.Linear(hidden + candidate_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        first = self.patient[0]
        if isinstance(first, nn.Linear):
            nn.init.xavier_uniform_(first.weight)
            nn.init.zeros_(first.bias)
        nn.init.normal_(self.candidate.weight, mean=0.0, std=0.05)
        for module in self.interaction:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: Tensor) -> Tensor:
        """A ``(rows, candidates)`` score matrix."""

        rows = state.shape[0]
        patient = self.patient(state)
        candidate = self.candidate.weight.unsqueeze(0).expand(rows, -1, -1)
        joined = torch.cat(
            [patient.unsqueeze(1).expand(-1, self.candidate_count, -1), candidate], dim=-1
        )
        scores: Tensor = self.interaction(joined).squeeze(-1)
        return scores

    def score_rows(self, state: Tensor, rows: Tensor) -> Tensor:
        return self.forward(state[rows])


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate with its score and the pieces behind it."""

    identifier: str
    score: float
    route: float
    modifier: float
    eligible: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "candidate": self.identifier,
            "score": round(self.score, 6),
            "route": round(self.route, 6),
            "modifier": round(self.modifier, 6),
            "eligible": self.eligible,
        }


def rank_candidates(
    identifiers: list[str],
    scores: Tensor,
    routes: Tensor,
    modifiers: Tensor,
    eligible: Tensor,
) -> list[RankedCandidate]:
    """Order the eligible candidates by decreasing score, ties by identifier."""

    ranked: list[RankedCandidate] = []
    for index, identifier in enumerate(identifiers):
        if float(eligible[index]) <= 0.5:
            continue
        ranked.append(
            RankedCandidate(
                identifier=identifier,
                score=float(scores[index]),
                route=float(routes[index]),
                modifier=float(modifiers[index]),
                eligible=True,
            )
        )
    ranked.sort(key=lambda candidate: (-candidate.score, candidate.identifier))
    return ranked


def top_k(ranked: list[RankedCandidate], k: int) -> list[RankedCandidate]:
    """The leading ``k`` entries of a ranking."""

    if k < 1:
        raise ValueError("the rank depth must be at least one")
    return ranked[:k]


def positive_ranking(ranked: list[RankedCandidate]) -> list[RankedCandidate]:
    """The ranking restricted by the sign condition ``tau_d(s) > 0``."""

    return [candidate for candidate in ranked if candidate.score > 0.0]


def jaccard(left: list[str], right: list[str]) -> float:
    """The overlap of two candidate sets as a Jaccard index."""

    first, second = set(left), set(right)
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def ranking_stability(left: list[str], right: list[str], k: int) -> float:
    """Top-``k`` overlap between two rankings, as a share of ``k``."""

    if k < 1:
        raise ValueError("the rank depth must be at least one")
    return len(set(left[:k]) & set(right[:k])) / k
