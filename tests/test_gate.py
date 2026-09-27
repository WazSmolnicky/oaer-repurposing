"""The identification gate: the eligible set, the conformal rule and the four outputs.

Ref: Sec. 2.3 (D_eps), Sec. 2.4 (Eq. (1) and the outputs), Algorithm 3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oaer.estimands.trimming import EligibilityRule, IneligibleReason
from oaer.gate.conformal import (
    ConformalCalibration,
    CovariateWeights,
    WeightedConformal,
    covariate_weights,
    miscoverage_rate,
    weighted_quantile,
)
from oaer.gate.harm import (
    DeferralReason,
    PathEvidence,
    apply_gate,
    harm_block,
    pathway_readout,
)
from oaer.gate.identification import (
    gate_diagnostics,
    identified_set,
    propensity_range,
    trimmed_fraction,
)


def _unit_interval(alpha: float = 0.2, size: int = 200) -> WeightedConformal:
    scores = np.linspace(0.0, 1.0, size)
    return WeightedConformal(
        ConformalCalibration(
            scores=scores,
            weights=CovariateWeights(np.ones(size, dtype=np.float64)),
            alpha=alpha,
        )
    )


def _flat_interval() -> WeightedConformal:
    """A calibration set of exact hits, so the interval is a point."""

    size = 50
    return WeightedConformal(
        ConformalCalibration(
            scores=np.zeros(size, dtype=np.float64),
            weights=CovariateWeights(np.ones(size, dtype=np.float64)),
            alpha=0.2,
        )
    )


class TestEligibleSet:
    def test_a_propensity_inside_the_window_is_admitted(self) -> None:
        gate = identified_set(
            {"a": 0.5}, {"a": 100}, EligibilityRule(epsilon=0.05, minimum_exposed=10)
        )
        assert gate.eligible == ("a",)
        assert gate.blocked == ()

    def test_a_trimmed_propensity_is_blocked_with_its_reason(self) -> None:
        rule = EligibilityRule(epsilon=0.05, minimum_exposed=10)
        gate = identified_set({"a": 0.01, "b": 0.5}, {"a": 100, "b": 100}, rule)
        assert gate.eligible == ("b",)
        assert gate.reason_for("a") is IneligibleReason.TRIMMING

    def test_a_thin_exposure_is_blocked_before_the_trimming_reason(self) -> None:
        rule = EligibilityRule(epsilon=0.05, minimum_exposed=10)
        gate = identified_set({"a": 0.5}, {"a": 3}, rule)
        assert gate.reason_for("a") is IneligibleReason.EXPOSURE_BELOW_MINIMUM

    def test_the_gate_is_ordered_by_candidate_name(self) -> None:
        gate = identified_set({"z": 0.5, "a": 0.5}, {"z": 50, "a": 50})
        assert gate.eligible == ("a", "z")

    def test_an_empty_gate_reports_itself(self) -> None:
        gate = identified_set({"a": 0.001}, {"a": 0})
        assert gate.is_empty and gate.size == 0

    def test_the_diagnostics_carry_a_count_per_reason(self) -> None:
        rule = EligibilityRule(epsilon=0.05, minimum_exposed=10)
        gate = identified_set({"a": 0.01, "b": 0.5, "c": 0.5}, {"a": 90, "b": 3, "c": 90}, rule)
        counts = gate_diagnostics(gate)
        assert counts["eligible"] == 1.0
        assert counts["trimming"] == 1.0


class TestPropensityReadouts:
    def test_the_range_is_the_minimum_and_maximum(self) -> None:
        assert propensity_range([0.2, 0.7, 0.4]) == (0.2, 0.7)

    def test_an_empty_range_is_undefined(self) -> None:
        low, high = propensity_range([])
        assert math.isnan(low) and math.isnan(high)

    def test_the_trimmed_fraction_counts_both_tails(self) -> None:
        rule = EligibilityRule(epsilon=0.1, minimum_exposed=1)
        assert trimmed_fraction([0.05, 0.5, 0.95, 0.5], rule) == pytest.approx(0.5)


class TestConformalRule:
    def test_a_weighted_quantile_inverts_the_cumulative_mass(self) -> None:
        value = weighted_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], 0.25)
        assert value == pytest.approx(1.0)

    def test_the_uniform_weighted_median_is_the_sample_median(self) -> None:
        sample = np.arange(101, dtype=np.float64)
        assert weighted_quantile(sample, np.ones(101), 0.5) == pytest.approx(50.0)

    def test_a_quantile_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_quantile([1.0], [1.0], 1.5)

    def test_a_mismatched_sample_and_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_quantile([1.0, 2.0], [1.0], 0.5)

    def test_the_calibration_quantile_is_monotone_in_the_miscoverage(self) -> None:
        tight = WeightedConformal(
            ConformalCalibration(np.linspace(0.0, 2.0, 500), CovariateWeights(np.ones(500)), 0.1)
        )
        loose = WeightedConformal(
            ConformalCalibration(np.linspace(0.0, 2.0, 500), CovariateWeights(np.ones(500)), 0.4)
        )
        assert tight.quantile > loose.quantile

    def test_the_interval_brackets_the_point(self) -> None:
        interval = _unit_interval()
        assert interval.lower_limit(0.5) < 0.5 < interval.upper_limit(0.5)

    def test_the_coverage_of_a_fresh_sample_clears_the_nominal_level(self) -> None:
        stream = np.random.default_rng(5)
        calibration = ConformalCalibration(
            scores=np.abs(stream.normal(0.0, 1.0, 4000)),
            weights=CovariateWeights(np.ones(4000)),
            alpha=0.1,
        )
        evaluation = stream.normal(0.0, 1.0, 4000)
        rate = miscoverage_rate(np.zeros_like(evaluation), evaluation, calibration)
        assert rate <= 0.1 + 0.02

    def test_the_covariate_weights_are_clipped(self) -> None:
        target = np.zeros(50)
        source = np.concatenate([np.zeros(25), np.ones(25)])
        weights = covariate_weights(target, source, clip=3.0)
        assert weights.values.min() >= 1.0 / 3.0 - 1e-12
        assert weights.values.max() <= 3.0 + 1e-12

    def test_the_effective_size_never_exceeds_the_sample(self) -> None:
        weights = CovariateWeights(np.asarray([1.0, 1.0, 1.0, 1.0]))
        assert weights.effective_size() == pytest.approx(4.0)


class TestGateOutputs:
    def test_the_harm_block_holds_the_negative_lower_limits(self) -> None:
        interval = _unit_interval(alpha=0.0, size=3)
        harm = harm_block({"a": 0.05, "b": 1.5}, interval)
        assert harm == ("a",)

    def test_the_ordering_carries_the_sign_condition(self) -> None:
        decision = apply_gate(
            "d0",
            {"a": 0.9, "b": 1.4, "c": -0.2, "d": 0.3},
            ("a", "b", "c", "d"),
            (),
            _flat_interval(),
            rank_depth=3,
        )
        assert decision.ranking == ("b", "a", "d")

    def test_an_empty_eligible_set_defers(self) -> None:
        decision = apply_gate("d0", {}, (), (), _flat_interval())
        assert not decision.act
        assert decision.reason is DeferralReason.EMPTY_ELIGIBLE_SET

    def test_a_bound_not_cleared_defers_with_its_own_reason(self) -> None:
        interval = _unit_interval(alpha=0.2, size=200)
        decision = apply_gate("d0", {"a": 0.01}, ("a",), (), interval, margin=0.5)
        assert not decision.act
        assert decision.reason is DeferralReason.MARGIN_NOT_CLEARED

    def test_a_cleared_bound_acts(self) -> None:
        decision = apply_gate("d0", {"a": 1.0}, ("a",), (), _unit_interval(), margin=0.0)
        assert decision.act
        assert decision.reason is DeferralReason.NONE

    def test_the_evidence_is_attached_to_the_decision(self) -> None:
        evidence = {
            "a": PathEvidence(
                candidate="a",
                path=("has_target", "affects_pathway"),
                exposed_count=120,
                trimmed_fraction=0.03,
                propensity=0.41,
            )
        }
        decision = apply_gate("d0", {"a": 1.0}, ("a",), (), _flat_interval(), evidence=evidence)
        assert decision.evidence["a"].exposed_count == 120
        assert decision.evidence["a"].path[0] == "has_target"

    def test_the_top_three_is_the_leading_slice(self) -> None:
        decision = apply_gate(
            "d0",
            {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5},
            ("a", "b", "c", "d"),
            (),
            _flat_interval(),
        )
        assert decision.top_three == ("a", "b", "c")

    def test_the_blocked_block_carries_its_reason(self) -> None:
        decision = apply_gate(
            "d0",
            {"a": 1.0},
            ("a",),
            (("b", IneligibleReason.TRIMMING),),
            _flat_interval(),
        )
        assert decision.not_identified == (("b", IneligibleReason.TRIMMING),)
        assert decision.as_dict()["not_identified"] == [["b", "trimming"]]


class TestPathwayReadout:
    def _decision(self, ranking: tuple[str, ...], harm: tuple[str, ...] = ()) -> object:
        return apply_gate(
            "d0",
            {name: float(len(ranking) - index) for index, name in enumerate(ranking)},
            ranking,
            (),
            _flat_interval(),
            rank_depth=3,
        )

    def test_a_started_top_candidate_is_counted_as_such(self) -> None:
        decision = self._decision(("a", "b", "c", "d"))
        payload = pathway_readout([decision], {"d0": ("a",)})
        assert payload["top_ranked_started"]["n"] == 1

    def test_starting_nothing_is_its_own_bucket(self) -> None:
        decision = self._decision(("a", "b"))
        payload = pathway_readout([decision], {"d0": ()})
        assert payload["none_started"]["n"] == 1
        assert payload["top_agreement"]["n"] == 0

    def test_a_start_outside_the_top_three_is_discordant(self) -> None:
        decision = self._decision(("a", "b", "c", "d"))
        payload = pathway_readout([decision], {"d0": ("d",)})
        assert payload["outside_top3_started"]["n"] == 1
        assert payload["discordant_initiation"]["n"] == 1

    def test_the_shares_sum_to_one_over_the_buckets(self) -> None:
        decisions = [self._decision(("a", "b", "c", "d")), self._decision(("a", "b"))]
        payload = pathway_readout(decisions, {"d0": ("a",), "d1": ()})
        shares = sum(
            value["share"]
            for key, value in payload.items()
            if key not in {"top_agreement", "discordant_initiation"}
        )
        assert shares == pytest.approx(1.0)
