"""Discrimination, calibration, ranking and resampling read-outs.

Every expected value here is built by hand rather than by calling the function
under test, so a passing test is a statement about the metric and not about the
implementation agreeing with itself.

Ref: Sec. 1.3 (C-index, ECE, twelve-month AUROC), Sec. 1.4 (IAB, IR, Top-5),
Table 2 note (the paired seed test).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oaer.readings.discrimination import (
    brier_score,
    calibration_slope,
    concordance_index,
    expected_calibration_error,
    net_benefit,
    net_reclassification_improvement,
    time_dependent_auroc,
)
from oaer.readings.interference import (
    interaction_beyond_additive,
    interaction_ratio,
    normalised_gain,
    ratio_admissible,
)
from oaer.readings.ranking import (
    average_precision,
    hits_at_k,
    jaccard,
    mean_jaccard_across_sets,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    seed_dispersion,
    top_k_set,
)
from oaer.readings.resampling import (
    interval_half_width,
    one_sample_seed_test,
    paired_seed_test,
    site_indices,
    site_stratified_bootstrap,
)
from oaer.support.numerics import Interval


class TestRankingReadouts:
    def test_hits_counts_only_the_positive_leading_entries(self) -> None:
        relevance = np.asarray([3.0, 0.0, 2.0, -1.0, 1.0])
        assert hits_at_k(relevance, 3) == pytest.approx(2.0 / 3.0)

    def test_hits_does_not_read_past_the_depth(self) -> None:
        relevance = np.asarray([0.0, 0.0, 5.0])
        assert hits_at_k(relevance, 2) == 0.0

    def test_hits_on_an_empty_ranking_is_zero(self) -> None:
        assert hits_at_k([], 5) == 0.0

    def test_ndcg_of_a_perfect_ordering_is_one(self) -> None:
        relevance = np.asarray([3.0, 2.0, 1.0])
        assert ndcg_at_k(relevance, 3) == pytest.approx(1.0)

    def test_ndcg_uses_the_positive_part_as_gain(self) -> None:
        relevance = np.asarray([-5.0, 1.0])
        expected = (1.0 / math.log2(3.0)) / (1.0 / math.log2(2.0))
        assert ndcg_at_k(relevance, 2) == pytest.approx(expected)

    def test_ndcg_without_any_positive_gain_is_zero(self) -> None:
        assert ndcg_at_k(np.asarray([-1.0, -2.0]), 2) == 0.0

    def test_precision_and_recall_agree_on_a_full_positive_set(self) -> None:
        relevance = np.asarray([1.0, 2.0])
        assert precision_at_k(relevance, 2) == pytest.approx(1.0)
        assert recall_at_k(relevance, 2) == pytest.approx(1.0)

    def test_recall_reads_the_whole_catalogue_for_its_denominator(self) -> None:
        relevance = np.asarray([1.0, 1.0, -1.0])
        assert recall_at_k(relevance, 1) == pytest.approx(0.5)

    def test_average_precision_of_a_perfect_ordering_is_one(self) -> None:
        assert average_precision(np.asarray([1.0, 1.0])) == pytest.approx(1.0)

    def test_average_precision_of_a_reversed_ordering_is_penalised(self) -> None:
        assert average_precision(np.asarray([-1.0, 1.0])) == pytest.approx(0.5)

    def test_the_top_k_set_sorts_by_score_then_name(self) -> None:
        assert top_k_set([1.0, 1.0, 0.0], ["b", "a", "c"], 2) == ("a", "b")

    def test_a_mismatched_score_and_identifier_list_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            top_k_set([1.0], ["a", "b"], 1)

    def test_the_jaccard_of_disjoint_sets_is_zero(self) -> None:
        assert jaccard(["a"], ["b"]) == 0.0

    def test_two_empty_sets_count_as_identical(self) -> None:
        assert jaccard([], []) == 1.0

    def test_the_mean_overlap_of_one_set_is_undefined(self) -> None:
        assert math.isnan(mean_jaccard_across_sets([("a",)]))

    def test_the_mean_overlap_of_two_sets_is_their_own_overlap(self) -> None:
        value = mean_jaccard_across_sets([("a", "b"), ("a", "c")])
        assert value == pytest.approx(1.0 / 3.0)

    def test_seed_dispersion_needs_two_seeds(self) -> None:
        assert seed_dispersion([1.0]) == 0.0
        assert seed_dispersion([1.0, 3.0]) == pytest.approx(math.sqrt(2.0))


class TestDiscrimination:
    def test_the_concordance_of_a_perfect_ordering_is_one(self) -> None:
        result = concordance_index([1.0, 2.0, 3.0], [True, True, True], [3.0, 2.0, 1.0])
        assert result.value == pytest.approx(1.0)
        assert result.comparable == 3

    def test_a_censored_record_contributes_no_comparable_pair(self) -> None:
        result = concordance_index([1.0, 2.0], [False, True], [1.0, 0.0])
        assert result.comparable == 0
        assert math.isnan(result.value)

    def test_a_tie_is_half_a_concordance(self) -> None:
        result = concordance_index([1.0, 2.0], [True, True], [1.0, 1.0])
        assert result.value == pytest.approx(0.5)
        assert result.tied == 1

    def test_the_auroc_of_a_separated_split_is_one(self) -> None:
        value = time_dependent_auroc([3.0, 6.0, 20.0], [True, True, False], [0.9, 0.8, 0.1], 12.0)
        assert value == pytest.approx(1.0)

    def test_a_record_censored_before_the_horizon_is_on_neither_side(self) -> None:
        times = np.asarray([3.0, 6.0, 20.0])
        events = np.asarray([True, False, False])
        scores = np.asarray([0.1, 0.99, 0.5])
        horizon = 12.0
        value = time_dependent_auroc(times, events, scores, horizon)
        # The record censored at month six is neither a case nor a control, so the
        # single comparable pair is the event by the horizon against the record
        # still at risk at it. Were the censored record counted, the value would
        # be one half rather than zero.
        assert value == pytest.approx(0.0)

    def test_the_calibration_error_of_a_calibrated_bin_is_zero(self) -> None:
        value = expected_calibration_error([0.5, 0.5], [1.0, 0.0], bins=2)
        assert value == pytest.approx(0.0)

    def test_the_calibration_error_counts_bin_mass(self) -> None:
        value = expected_calibration_error([0.1, 0.9], [1.0, 1.0], bins=2)
        assert value == pytest.approx(0.5)

    def test_the_brier_score_of_a_perfect_forecast_is_zero(self) -> None:
        assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_the_brier_score_of_a_confident_miss_is_one(self) -> None:
        assert brier_score([1.0], [0.0]) == pytest.approx(1.0)

    def test_the_calibration_slope_of_a_calibrated_set_is_near_one(self) -> None:
        stream = np.random.default_rng(3)
        probabilities = stream.uniform(0.05, 0.95, 4000)
        outcomes = (stream.uniform(size=4000) < probabilities).astype(float)
        assert abs(calibration_slope(probabilities, outcomes) - 1.0) < 0.15

    def test_net_benefit_of_a_useless_rule_is_zero(self) -> None:
        values = net_benefit([0.5, 0.5], [1.0, 0.0], (0.5,))
        assert values[0.5] == pytest.approx(0.0)

    def test_a_threshold_at_the_boundary_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            net_benefit([0.5], [1.0], (1.0,))

    def test_the_reclassification_improvement_is_zero_for_an_identical_rule(self) -> None:
        probabilities = [0.2, 0.8, 0.4, 0.6]
        outcomes = [0.0, 1.0, 0.0, 1.0]
        value = net_reclassification_improvement(probabilities, probabilities, outcomes, 0.5)
        assert value == pytest.approx(0.0)


class TestInterference:
    def test_the_additive_interaction_is_the_joint_less_the_two_singles(self) -> None:
        assert interaction_beyond_additive(-17.7, -8.4, -7.8) == pytest.approx(-1.5, abs=1e-9)

    def test_the_ratio_is_the_two_singles_over_the_joint(self) -> None:
        assert interaction_ratio(-17.7, -8.4, -7.8) == pytest.approx(0.9153, abs=1e-3)

    def test_a_null_joint_removal_has_no_ratio(self) -> None:
        assert math.isnan(interaction_ratio(0.0, -1.0, -1.0))

    def test_a_denominator_interval_clearing_zero_admits_the_ratio(self) -> None:
        assert ratio_admissible(Interval(point=1.0, lower=0.5, upper=1.5))

    def test_a_denominator_interval_straddling_zero_does_not(self) -> None:
        assert not ratio_admissible(Interval(point=0.0, lower=-0.5, upper=0.5))

    def test_a_normalised_gain_is_the_gain_over_its_reference(self) -> None:
        assert normalised_gain(3.2, 1.6) == pytest.approx(2.0)


class TestResampling:
    def test_the_bootstrap_reproduces_itself_under_one_seed(self) -> None:
        values = np.arange(8, dtype=np.float64)
        sites = ["A", "A", "B", "B", "C", "C", "C", "C"]

        def statistic(index: np.ndarray) -> float:
            return float(values[index].mean())

        first = site_stratified_bootstrap(statistic, sites, resamples=32, seed=7)
        second = site_stratified_bootstrap(statistic, sites, resamples=32, seed=7)
        assert np.array_equal(first.draws, second.draws)

    def test_the_bootstrap_point_estimate_is_the_sample_value(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        sites = ["A", "B", "A", "B"]
        result = site_stratified_bootstrap(
            lambda index: float(values[index].mean()), sites, resamples=16, seed=9
        )
        assert result.point == pytest.approx(2.5)

    def test_the_site_index_groups_preserve_every_record(self) -> None:
        grouped = site_indices(["A", "B", "A", "C"])
        assert sorted(int(index) for members in grouped.values() for index in members) == [
            0,
            1,
            2,
            3,
        ]

    def test_a_bootstrap_needs_at_least_one_resample(self) -> None:
        with pytest.raises(ValueError):
            site_stratified_bootstrap(lambda index: 0.0, ["A"], resamples=0)

    def test_the_paired_test_needs_two_seeds(self) -> None:
        assert math.isnan(paired_seed_test([1.0], [2.0]))

    def test_a_constant_shift_is_detected_by_the_paired_test(self) -> None:
        candidate = [2.0, 2.4, 2.2, 2.6, 2.1]
        reference = [1.0, 1.4, 1.2, 1.6, 1.1]
        assert paired_seed_test(candidate, reference) < 0.01

    def test_an_identical_pair_is_not_separated(self) -> None:
        values = [1.0, 1.3, 0.9, 1.1]
        assert paired_seed_test(values, values) == pytest.approx(0.0, abs=1e-9)

    def test_the_one_sample_test_compares_against_a_constant(self) -> None:
        assert one_sample_seed_test([1.0, 1.0, 1.0], 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_the_half_width_shrinks_with_a_tighter_spread(self) -> None:
        wide = interval_half_width([0.0, 10.0, -10.0])
        narrow = interval_half_width([1.0, 1.1, 0.9])
        assert wide > narrow

    def test_a_single_point_has_no_width(self) -> None:
        assert math.isnan(interval_half_width([1.0]))
