"""The ranking head: the route score, the centred modifier and Eq. (5)'s weights.

Ref: Sec. 2.4 (the head's two routes and the zero-mean condition), Sec. 2.6 Eq. (5).
"""

from __future__ import annotations

import math

import pytest
import torch

from oaer.head.modifiers import (
    PathologyModifier,
    RouteScore,
    interaction_beyond_additive,
    interaction_index,
    zero_mean_centre,
)
from oaer.head.objectives import StageThreeHead, StageThreeWeights, weighted_least_squares
from oaer.head.ranking import (
    DecisionRanker,
    RankedCandidate,
    jaccard,
    positive_ranking,
    rank_candidates,
    ranking_stability,
    top_k,
)


class TestZeroMeanCondition:
    def test_centring_over_every_row_leaves_a_zero_mean(self) -> None:
        values = torch.as_tensor([1.0, 2.0, 3.0, 4.0])
        centred = zero_mean_centre(values)
        assert float(centred.mean()) == pytest.approx(0.0, abs=1e-6)

    def test_centring_over_the_eligible_set_leaves_it_at_zero(self) -> None:
        values = torch.as_tensor([5.0, 1.0, 2.0, 9.0])
        eligible = torch.as_tensor([0.0, 1.0, 1.0, 0.0])
        centred = zero_mean_centre(values, eligible)
        assert float(centred[1:3].mean()) == pytest.approx(0.0, abs=1e-6)

    def test_an_ineligible_row_carries_no_modifier(self) -> None:
        values = torch.as_tensor([5.0, 1.0, 2.0, 9.0])
        eligible = torch.as_tensor([0.0, 1.0, 1.0, 0.0])
        centred = zero_mean_centre(values, eligible)
        assert float(centred[0]) == pytest.approx(0.0)
        assert float(centred[3]) == pytest.approx(0.0)

    def test_an_empty_eligible_set_yields_zero(self) -> None:
        values = torch.as_tensor([1.0, 2.0])
        centred = zero_mean_centre(values, torch.zeros(2))
        assert torch.allclose(centred, torch.zeros(2))

    def test_an_empty_tensor_centres_to_itself(self) -> None:
        assert zero_mean_centre(torch.zeros(0)).numel() == 0


class TestModifierRoutes:
    def test_the_route_score_scores_one_value_per_row(self) -> None:
        route = RouteScore(width=6, hidden=4)
        assert tuple(route(torch.zeros(5, 6)).shape) == (5,)

    def test_the_modifier_distinguishes_a_missing_slide(self) -> None:
        modifier = PathologyModifier(slide_dim=4, hidden=3)
        slide = torch.ones(2, 4)
        available = torch.as_tensor([1.0, 0.0])
        with_route = modifier(slide)
        gated = modifier(slide, available)
        assert not torch.allclose(with_route, gated)
        bias = float(modifier.present)
        assert float(gated[1]) == pytest.approx(bias)

    def test_a_narrow_width_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RouteScore(width=0)

    def test_the_interaction_index_is_the_two_singles_over_the_joint(self) -> None:
        assert interaction_index(-17.7, -8.4, -7.8) == pytest.approx(0.9153, abs=1e-3)

    def test_the_index_is_undefined_for_a_null_joint_removal(self) -> None:
        assert math.isnan(interaction_index(0.0, -1.0, -1.0))

    def test_the_additive_interaction_is_signed_for_synergy(self) -> None:
        assert interaction_beyond_additive(-17.7, -8.4, -7.8) < 0.0


class TestStageThreeWeights:
    def test_candidate_balancing_gives_each_candidate_equal_mass(self) -> None:
        eligible = torch.as_tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = StageThreeWeights(balanced=True).build(eligible)
        per_candidate = weights.sum(dim=0)
        assert float(per_candidate[0]) == pytest.approx(float(per_candidate[1]))

    def test_the_unbalanced_weights_are_the_eligible_mask(self) -> None:
        eligible = torch.as_tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = StageThreeWeights(balanced=False).build(eligible)
        assert torch.allclose(weights, eligible)

    def test_neither_form_weights_a_cell_outside_the_eligible_set(self) -> None:
        eligible = torch.as_tensor([[1.0, 0.0], [0.0, 1.0]])
        for balanced in (True, False):
            weights = StageThreeWeights(balanced=balanced).build(eligible)
            assert float(weights[eligible == 0.0].sum()) == pytest.approx(0.0)

    def test_an_empty_eligible_set_carries_no_weight(self) -> None:
        weights = StageThreeWeights(balanced=True).build(torch.zeros(3, 2))
        assert float(weights.sum()) == pytest.approx(0.0)


class TestWeightedLeastSquares:
    def test_a_perfect_prediction_leaves_only_the_penalty(self) -> None:
        target = torch.as_tensor([[1.0, 2.0]])
        weights = torch.ones_like(target)
        parameter = torch.nn.Parameter(torch.zeros(3))
        loss = weighted_least_squares(target, target, weights, [parameter], penalty=0.0)
        assert float(loss) == pytest.approx(0.0)

    def test_the_penalty_grows_with_the_parameter_norm(self) -> None:
        target = torch.zeros(1, 1)
        weights = torch.ones_like(target)
        small = torch.nn.Parameter(torch.tensor([0.1]))
        large = torch.nn.Parameter(torch.tensor([10.0]))
        assert float(weighted_least_squares(target, target, weights, [large], penalty=1.0)) > float(
            weighted_least_squares(target, target, weights, [small], penalty=1.0)
        )

    def test_a_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_least_squares(torch.zeros(2), torch.zeros(3), torch.zeros(2), [], penalty=0.0)

    def test_the_head_loss_falls_under_its_own_gradient(self) -> None:
        torch.manual_seed(11)
        head = DecisionRanker(state_dim=6, candidate_count=4, candidate_dim=3, hidden=8)
        module = StageThreeHead(head, penalty=0.0)
        state = torch.randn(12, 6)
        target = torch.randn(12, 4)
        weights = torch.ones_like(target)
        optimizer = torch.optim.AdamW(head.parameters(), lr=5e-2)
        first = 0.0
        for step in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = module.loss(head(state), target, weights)
            if step == 0:
                first = float(loss.detach())
            loss.backward()
            optimizer.step()
        assert float(loss.detach()) < first


class TestDecisionRanker:
    def test_the_score_matrix_is_rows_by_candidates(self) -> None:
        ranker = DecisionRanker(state_dim=5, candidate_count=7, candidate_dim=2, hidden=4)
        assert tuple(ranker(torch.zeros(3, 5)).shape) == (3, 7)

    def test_the_head_is_shared_so_every_candidate_is_scored(self) -> None:
        ranker = DecisionRanker(state_dim=4, candidate_count=6, candidate_dim=2, hidden=4)
        scores = ranker(torch.randn(2, 4))
        assert int(torch.isfinite(scores).sum()) == 12

    def test_scoring_a_row_subset_matches_the_full_matrix(self) -> None:
        ranker = DecisionRanker(state_dim=4, candidate_count=3, candidate_dim=2, hidden=4)
        state = torch.randn(5, 4)
        rows = torch.as_tensor([1, 3])
        assert torch.allclose(ranker.score_rows(state, rows), ranker(state)[rows], atol=1e-6)

    def test_a_candidate_count_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DecisionRanker(state_dim=4, candidate_count=0)


class TestRankingUtilities:
    def _candidates(self) -> list[RankedCandidate]:
        return [
            RankedCandidate("a", 1.2, 0.9, 0.3, True),
            RankedCandidate("b", 0.4, 0.4, 0.0, True),
            RankedCandidate("c", -0.2, -0.2, 0.0, True),
            RankedCandidate("d", -0.3, -0.3, 0.0, False),
        ]

    def test_rank_candidates_orders_by_decreasing_score(self) -> None:
        ranked = rank_candidates(
            ["a", "b", "c", "d"],
            torch.as_tensor([1.2, 0.4, -0.2, 0.9]),
            torch.as_tensor([0.9, 0.4, -0.2, 0.9]),
            torch.as_tensor([0.3, 0.0, 0.0, 0.0]),
            torch.as_tensor([1.0, 1.0, 1.0, 0.0]),
        )
        assert [candidate.identifier for candidate in ranked] == ["a", "b", "c"]

    def test_ties_break_on_the_identifier(self) -> None:
        ranked = rank_candidates(
            ["b", "a"],
            torch.as_tensor([1.0, 1.0]),
            torch.as_tensor([1.0, 1.0]),
            torch.as_tensor([0.0, 0.0]),
            torch.as_tensor([1.0, 1.0]),
        )
        assert [candidate.identifier for candidate in ranked] == ["a", "b"]

    def test_the_positive_ranking_applies_the_sign_condition(self) -> None:
        assert [c.identifier for c in positive_ranking(self._candidates())] == ["a", "b"]

    def test_top_k_takes_the_leading_entries(self) -> None:
        assert [c.identifier for c in top_k(self._candidates(), 2)] == ["a", "b"]

    def test_a_rank_depth_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            top_k(self._candidates(), 0)

    def test_the_jaccard_of_disjoint_sets_is_zero(self) -> None:
        assert jaccard(["a"], ["b"]) == 0.0

    def test_ranking_stability_counts_the_shared_leading_entries(self) -> None:
        assert ranking_stability(["a", "b", "c"], ["a", "c", "b"], 2) == pytest.approx(0.5)

    def test_stability_at_a_depth_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ranking_stability(["a"], ["a"], 0)


class TestRankedCandidateSerialisation:
    def test_the_payload_carries_every_route(self) -> None:
        payload = RankedCandidate("a", 1.2345678, 0.5, 0.25, True).as_dict()
        assert payload["candidate"] == "a"
        assert payload["score"] == pytest.approx(1.234568)
        assert payload["eligible"] is True
