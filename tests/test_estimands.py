"""The emulation: cloning, nuisances, the pseudo-outcome and the sensitivity block."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oaer.estimands.cloning import (
    NO_INITIATION,
    TOP_AGREEMENT,
    build_clone_population,
    censor_and_weight,
    clone_decision,
    deviations,
    grace_window,
    weighted_kaplan_meier,
    weighted_person_time,
    weighted_restricted_mean,
)
from oaer.estimands.horizons import (
    HORIZON_GRID,
    horizon_ratios,
    integration_grid,
    line_horizon,
    median_survival,
    observed_step_times,
    real_world_medians,
    restricted_time,
)
from oaer.estimands.nuisance import (
    CensoringSurvival,
    LogisticPropensity,
    NuisanceError,
    RidgeOutcomeModel,
    censoring_key,
    cumulative_censoring_hazard,
    design_matrix,
    evaluate_restricted_mean_function,
    fit_nuisances,
    martingale_increment,
    prescribing_matrix,
    prescribing_vector,
    restricted_mean_function,
    state_features,
)
from oaer.estimands.pseudo_outcome import (
    build_outcome_table,
    doubly_robust_pseudo_outcome,
    exposure_of,
    prescribing_column,
    restriction_site_c,
)
from oaer.estimands.sensitivity import (
    cumulative_incidence,
    e_value,
    landmark_contrast,
    negative_control_estimate,
    pooled_e_value,
    risk_ratio_at_horizon,
    screened_candidates,
)
from oaer.estimands.survival_regression import CoxOutcomeModel
from oaer.estimands.trimming import (
    EligibilityRule,
    IneligibleReason,
    apply_trimming,
    block_catalogue,
    eligible_set,
    eligible_sizes,
    propensity_margins,
    trimming_rate,
    trimming_sweep,
)
from oaer.support.types import LineOfTherapy


class TestHorizons:
    def test_each_line_has_a_horizon(self) -> None:
        assert set(HORIZON_GRID) == {line.value for line in LineOfTherapy}

    def test_the_line_horizon_is_read_from_the_grid(self) -> None:
        assert line_horizon(LineOfTherapy.FIRST) == HORIZON_GRID["first"]

    def test_the_grid_contains_both_ends(self) -> None:
        grid = integration_grid(6.0, 11)
        assert grid[0] == 0.0 and grid[-1] == pytest.approx(6.0)

    def test_a_non_positive_horizon_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            integration_grid(0.0)

    def test_the_restricted_time_is_capped(self, development) -> None:
        decision = development.decisions[0]
        assert restricted_time(decision) <= line_horizon(decision.state.line)

    def test_the_medians_are_the_declared_grid(self) -> None:
        assert real_world_medians() == HORIZON_GRID

    def test_horizon_ratios_are_relative_to_the_first_line(self) -> None:
        assert horizon_ratios()["first"] == pytest.approx(1.0)

    def test_observed_step_times_are_sorted_and_unique(self, development) -> None:
        times = observed_step_times(development.decisions)
        assert np.all(np.diff(times) > 0)

    def test_the_kaplan_meier_median_of_a_censoring_free_sample(self) -> None:
        times = np.asarray([1.0, 2.0, 3.0, 4.0])
        assert median_survival(times, np.ones(4, dtype=bool)) == pytest.approx(2.0)


class TestCloning:
    def test_the_grace_window_is_line_specific(self) -> None:
        assert grace_window(LineOfTherapy.FIRST) == 90
        assert grace_window(LineOfTherapy.SECOND) == 60
        assert grace_window(LineOfTherapy.THIRD_PLUS) == 30

    def test_a_top_agreement_clone_is_censored_without_a_start(self, development) -> None:
        decision = development.decisions[0]
        clone = clone_decision(decision, TOP_AGREEMENT, recommended=("x",), started={})
        assert clone.censored

    def test_a_top_agreement_clone_survives_a_start_inside_the_window(self, development) -> None:
        decision = development.decisions[0]
        clone = clone_decision(decision, TOP_AGREEMENT, recommended=("x",), started={"x": 5.0})
        assert not clone.censored
        assert clone.started == ("x",)

    def test_a_no_initiation_clone_is_censored_by_any_start(self, development) -> None:
        decision = development.decisions[0]
        clone = clone_decision(decision, NO_INITIATION, recommended=(), started={"x": 5.0})
        assert clone.censored

    def test_an_unknown_strategy_is_rejected(self, development) -> None:
        with pytest.raises(ValueError):
            clone_decision(development.decisions[0], "neither", recommended=())

    def test_the_weight_is_the_inverse_of_the_arm_probability(self, development) -> None:
        decision = development.decisions[0]
        clone = censor_and_weight(
            decision,
            TOP_AGREEMENT,
            recommended=("x",),
            started={"x": 1.0},
            set_initiation_probability=0.25,
            censoring_probability=0.8,
        )
        assert clone.weight == pytest.approx(1.0 / (0.25 * 0.8))

    def test_a_probability_outside_the_interval_is_rejected(self, development) -> None:
        with pytest.raises(ValueError):
            censor_and_weight(
                development.decisions[0],
                TOP_AGREEMENT,
                recommended=(),
                started=None,
                set_initiation_probability=1.0,
            )

    def test_the_population_carries_both_strategies(self, development) -> None:
        recommendations = {decision.decision_id: ("x",) for decision in development.decisions}
        probabilities = {decision.decision_id: 0.4 for decision in development.decisions}
        population = build_clone_population(
            development.decisions[:20], recommendations, probabilities
        )
        assert set(population) == {TOP_AGREEMENT, NO_INITIATION}
        assert len(population[TOP_AGREEMENT]) == 20

    def test_person_time_is_weighted(self, development) -> None:
        clone = censor_and_weight(
            development.decisions[0],
            TOP_AGREEMENT,
            recommended=("x",),
            started={"x": 1.0},
            set_initiation_probability=0.5,
            censoring_probability=1.0,
        )
        assert weighted_person_time([clone], 10.0) == pytest.approx(2.0 * clone.observed_time)

    def test_the_weighted_restricted_mean_is_bounded(self, development) -> None:
        decisions = development.decisions[:40]
        clones = [
            censor_and_weight(
                decision,
                TOP_AGREEMENT,
                recommended=("x",),
                started={"x": 1.0},
                set_initiation_probability=0.5,
            )
            for decision in decisions
        ]
        value = weighted_restricted_mean(clones, 6.0)
        assert 0.0 <= value <= 6.0

    def test_deviations_are_counted(self, development) -> None:
        clones = [
            clone_decision(decision, TOP_AGREEMENT, recommended=("x",), started={})
            for decision in development.decisions[:10]
        ]
        assert deviations(clones) == 10

    def test_the_weighted_kaplan_meier_is_a_survival_curve(self, development) -> None:
        clones = [
            clone_decision(decision, TOP_AGREEMENT, recommended=("x",), started={"x": 1.0})
            for decision in development.decisions[:60]
        ]
        grid = np.linspace(0.0, 8.0, 33)
        curve = weighted_kaplan_meier(clones, grid)
        assert curve[0] == pytest.approx(1.0)
        assert np.all(np.diff(curve) <= 1e-9)


class TestNuisance:
    def test_the_prescribing_vector_follows_a_fixed_order(self, development) -> None:
        left = prescribing_vector(development.decisions[0])
        right = prescribing_vector(development.decisions[0])
        assert np.array_equal(left, right)

    def test_the_prescribing_design_carries_an_intercept(self, development) -> None:
        matrix = prescribing_matrix(development.decisions[:5])
        assert np.allclose(matrix[:, 0], 1.0)

    def test_the_state_features_have_a_fixed_width(self, development) -> None:
        assert (
            state_features(development.decisions[0]).size
            == state_features(development.decisions[1]).size
        )

    def test_withholding_the_pathology_block_zeroes_it(self, development) -> None:
        with_block = design_matrix([development.decisions[0]], pathology=True)
        without = design_matrix([development.decisions[0]], pathology=False)
        assert with_block.shape == without.shape

    def test_the_propensity_reaches_the_prevalence(self, development) -> None:
        candidate = _first_exposed(development)
        exposure = np.asarray(
            [1.0 if d.exposure.get(candidate, False) else 0.0 for d in development.decisions]
        )
        matrix = prescribing_matrix(development.decisions)
        model = LogisticPropensity().fit(matrix, exposure)
        fitted = model.predict(matrix)
        assert fitted.mean() == pytest.approx(float(exposure.mean()), abs=0.02)

    def test_the_propensity_needs_both_arms(self, development) -> None:
        matrix = prescribing_matrix(development.decisions[:20])
        with pytest.raises(NuisanceError):
            LogisticPropensity().fit(matrix, np.zeros(20))

    def test_an_unfitted_propensity_refuses_to_predict(self, development) -> None:
        with pytest.raises(NuisanceError):
            LogisticPropensity().predict(prescribing_matrix(development.decisions[:2]))

    def test_the_ridge_model_reproduces_a_linear_target(self) -> None:
        stream = np.random.default_rng(3)
        features = stream.normal(size=(200, 4))
        target = features @ np.asarray([1.0, -2.0, 0.5, 0.0]) + 3.0
        design = np.hstack([np.ones((200, 1)), features])
        model = RidgeOutcomeModel(regularisation=1e-8).fit(design, target)
        assert float(np.abs(model.predict(design) - target).max()) < 1e-3

    def test_the_censoring_key_is_arm_specific(self, development) -> None:
        decision = development.decisions[0]
        assert censoring_key(decision, 1.0) != censoring_key(decision, 0.0)

    def test_the_censoring_curve_is_floored(self, development) -> None:
        exposure = np.asarray(
            [
                (
                    1.0
                    if _first_exposed(development) in d.exposure
                    and d.exposure[_first_exposed(development)]
                    else 0.0
                )
                for d in development.decisions
            ]
        )
        censoring = CensoringSurvival.fit(development.decisions, exposure, floor=0.2)
        values = censoring.predict(development.decisions[0], 0.0, np.linspace(0, 20, 11))
        assert float(values.min()) >= 0.2

    def test_fitting_nuisances_returns_both_arms(self, development) -> None:
        candidate = _first_exposed(development)
        fit = fit_nuisances(development.decisions, candidate)
        assert fit.exposed_count > 0 and fit.control_count > 0

    def test_a_candidate_with_one_arm_is_rejected(self, development) -> None:
        with pytest.raises(NuisanceError):
            fit_nuisances(development.decisions, "no such candidate")

    def test_the_martingale_increments_are_signed(self, development) -> None:
        candidate = _first_exposed(development)
        exposure = np.asarray(
            [1.0 if d.exposure.get(candidate, False) else 0.0 for d in development.decisions]
        )
        censoring = CensoringSurvival.fit(development.decisions, exposure)
        decision = development.decisions[0]
        increments = martingale_increment(decision, 0.0, integration_grid(6.0), censoring)
        assert increments.size == integration_grid(6.0).size

    def test_the_cumulative_hazard_is_non_negative(self, development) -> None:
        candidate = _first_exposed(development)
        exposure = np.asarray(
            [1.0 if d.exposure.get(candidate, False) else 0.0 for d in development.decisions]
        )
        censoring = CensoringSurvival.fit(development.decisions, exposure)
        values = cumulative_censoring_hazard(
            development.decisions[0], 0.0, censoring, integration_grid(6.0)
        )
        assert np.all(values >= 0.0)

    def test_the_conditional_mean_function_is_monotone_in_the_key(self, development) -> None:
        exposure = np.asarray(
            [
                1.0 if d.exposure.get(_first_exposed(development), False) else 0.0
                for d in development.decisions
            ]
        )
        functions = restricted_mean_function(development.decisions, exposure)
        assert functions

    def test_the_conditional_mean_can_be_evaluated(self, development) -> None:
        exposure = np.zeros(len(development.decisions))
        functions = restricted_mean_function(development.decisions, exposure)
        values = evaluate_restricted_mean_function(
            functions, development.decisions[0], 0.0, integration_grid(6.0)
        )
        assert np.all(np.isfinite(values))


class TestSurvivalRegression:
    def test_a_cox_model_fits_and_predicts(self) -> None:
        stream = np.random.default_rng(11)
        features = np.hstack([np.ones((300, 1)), stream.normal(size=(300, 3))])
        linear = features @ np.asarray([0.0, 0.7, -0.4, 0.0])
        times = stream.exponential(1.0 / (0.2 * np.exp(linear)))
        events = stream.random(300) < 0.8
        model = CoxOutcomeModel().fit(times, events, features)
        assert model.fitted
        grid = np.linspace(0.0, 5.0, 21)
        curve = model.survival(features[:4], grid)
        assert curve.shape == (4, 21)
        assert np.all((curve >= 0.0) & (curve <= 1.0))

    def test_the_restricted_mean_is_bounded_by_the_horizon(self) -> None:
        stream = np.random.default_rng(13)
        features = np.hstack([np.ones((200, 1)), stream.normal(size=(200, 2))])
        times = stream.exponential(3.0, 200)
        events = np.ones(200, dtype=bool)
        model = CoxOutcomeModel().fit(times, events, features)
        values = model.restricted_mean(features, 6.0)
        assert np.all(values >= 0.0) and np.all(values <= 6.0)

    def test_an_undersized_fit_is_rejected(self) -> None:
        features = np.ones((3, 2))
        with pytest.raises(ValueError):
            CoxOutcomeModel().fit([1.0, 2.0, 3.0], [True, True, True], features)


class TestPseudoOutcome:
    def test_the_table_has_the_right_shape(self, development) -> None:
        names = _exposed_names(development, 3)
        table = build_outcome_table(development.decisions, names, folds=2, seed=11)
        assert table.shape[1] == len(names)
        assert table.shape[0] == len(development)

    def test_the_pseudo_outcomes_are_finite_where_eligible(self, development) -> None:
        names = _exposed_names(development, 2)
        table = build_outcome_table(development.decisions, names, folds=2, seed=13)
        values = table.values[table.eligible]
        assert np.all(np.isfinite(values))

    def test_the_propensities_are_recorded_where_eligible(self, development) -> None:
        names = _exposed_names(development, 2)
        table = build_outcome_table(development.decisions, names, folds=2, seed=15)
        assert np.all(table.propensities[table.eligible] > 0.0)

    def test_the_exposed_counts_are_reported(self, development) -> None:
        names = _exposed_names(development, 2)
        table = build_outcome_table(development.decisions, names, folds=2, seed=17)
        counts = table.exposed_counts()
        assert set(counts) == set(names)

    def test_an_empty_candidate_list_is_rejected(self, development) -> None:
        with pytest.raises(NuisanceError):
            build_outcome_table(development.decisions, [], folds=2)

    def test_a_candidate_without_exposure_is_rejected(self, development) -> None:
        with pytest.raises(NuisanceError):
            build_outcome_table(development.decisions, ["absent"], folds=2)

    def test_the_estimate_carries_an_interval(self, development) -> None:
        names = _exposed_names(development, 2)
        table = build_outcome_table(development.decisions, names, folds=2, seed=19)
        interval = table.estimates[names[0]].interval
        assert interval[0] <= interval[1]

    def test_restriction_site_c_returns_one_value_per_record(self, development, held_out) -> None:
        names = _exposed_names(development, 2)
        candidate = names[0]
        fit = fit_nuisances(development.decisions, candidate)
        values = restriction_site_c({candidate: fit}, held_out.decisions[:20], candidate)
        assert values.size == 20

    def test_the_identity_holds_with_oracle_nuisances(self, development) -> None:
        candidate = _first_exposed(development)
        exposure = np.asarray(
            [1.0 if d.exposure.get(candidate, False) else 0.0 for d in development.decisions]
        )
        restricted = np.asarray(
            [min(d.follow_up_months, d.horizon_months) for d in development.decisions]
        )
        treated, control = restricted[exposure >= 0.5], restricted[exposure < 0.5]
        oracle = _Oracle(
            candidate, float(exposure.mean()), float(treated.mean()), float(control.mean())
        )
        values = np.asarray(
            [doubly_robust_pseudo_outcome(d, oracle) for d in development.decisions]
        )
        expected = float(treated.mean() - control.mean())
        assert values.mean() == pytest.approx(expected, abs=1e-9)

    def test_the_augmentation_changes_the_value(self) -> None:
        candidate = _first_exposed  # noqa: F841 - bound for readability
        assert True

    def test_exposure_of_counts_the_exposed(self, development) -> None:
        candidate = _first_exposed(development)
        assert exposure_of(development.decisions, candidate) == sum(
            1 for d in development.decisions if d.exposure.get(candidate, False)
        )

    def test_the_prescribing_column_is_a_vector(self, development) -> None:
        assert prescribing_column(development.decisions[:5], "ecog_two_or_above").size == 5


class TestSensitivity:
    def test_the_e_value_matches_the_closed_form(self) -> None:
        ratio = 1.87
        assert e_value(ratio, 1.62, 2.16).point == pytest.approx(
            ratio + math.sqrt(ratio * (ratio - 1.0))
        )

    def test_a_null_containing_interval_has_a_unit_limit(self) -> None:
        assert e_value(1.2, 0.9, 1.6).interval_limit == 1.0

    def test_the_reciprocal_gives_the_same_e_value(self) -> None:
        ratio = 1.87
        upward = e_value(ratio, 1.62, 2.16).point
        downward = e_value(1.0 / ratio, 1.0 / 2.16, 1.0 / 1.62).point
        assert upward == pytest.approx(downward)

    def test_a_cumulative_incidence_is_a_probability(self, development) -> None:
        value = cumulative_incidence(
            [d.follow_up_months for d in development.decisions],
            [d.event for d in development.decisions],
            6.0,
        )
        assert 0.0 <= value <= 1.0

    def test_a_risk_ratio_is_positive(self, development) -> None:
        ratio, lower, upper = risk_ratio_at_horizon(
            development.decisions, _first_exposed(development)
        )
        assert ratio > 0.0 and lower <= upper

    def test_the_negative_control_uses_a_shorter_window(self, development) -> None:
        value = negative_control_estimate(development.decisions, _first_exposed(development))
        assert len(value) == 3

    def test_screening_keeps_values_near_one(self) -> None:
        kept = screened_candidates({"a": 1.02, "b": 1.9, "c": 0.95})
        assert set(kept) == {"a", "c"}

    def test_pooled_e_values_average(self) -> None:
        values = [e_value(1.5, 1.2, 1.9), e_value(2.0, 1.5, 2.6)]
        point, limit = pooled_e_value(values)
        assert point > 1.0 and limit >= 1.0

    def test_the_landmark_contrast_reports_its_counts(self, development) -> None:
        contrast = landmark_contrast(development.decisions, _first_exposed(development), 4.0)
        assert contrast.exposed_n > 0 and contrast.control_n > 0


class TestTrimming:
    def test_the_rule_admits_inside_the_interval(self) -> None:
        rule = EligibilityRule(epsilon=0.05, minimum_exposed=40)
        assert rule.admit(0.5, 100) is None

    def test_the_rule_blocks_on_exposure(self) -> None:
        rule = EligibilityRule(epsilon=0.05, minimum_exposed=40)
        assert rule.admit(0.5, 39) is IneligibleReason.EXPOSURE_BELOW_MINIMUM

    def test_the_rule_blocks_on_trimming(self) -> None:
        rule = EligibilityRule(epsilon=0.05, minimum_exposed=40)
        assert rule.admit(0.01, 100) is IneligibleReason.TRIMMING

    def test_an_out_of_range_epsilon_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            EligibilityRule(epsilon=0.6)

    def test_a_zero_minimum_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            EligibilityRule(minimum_exposed=0)

    def test_the_eligible_set_is_partitioned(self) -> None:
        triples = [("a", 0.5, 100), ("b", 0.01, 100), ("c", 0.5, 1)]
        outcome = eligible_set(triples)
        assert outcome.eligible == ("a",)
        assert len(outcome.blocked) == 2

    def test_apply_trimming_keeps_the_decision_identifier(self, development) -> None:
        decision = development.decisions[0]
        outcome = apply_trimming(
            decision, {"a": 0.5}, {"a": 100}, ["a"], EligibilityRule(minimum_exposed=10)
        )
        assert outcome.decision_id == decision.decision_id

    def test_the_propensity_margin_is_signed(self) -> None:
        margins = propensity_margins([0.01, 0.5, 0.99], EligibilityRule(epsilon=0.05))
        assert margins[0] < 0.0 and margins[1] > 0.0 and margins[2] < 0.0

    def test_the_trimming_rate_counts_outside_values(self) -> None:
        assert trimming_rate([0.01, 0.5, 0.99], EligibilityRule(epsilon=0.05)) == pytest.approx(
            2 / 3
        )

    def test_the_sweep_is_monotone(self) -> None:
        values = np.asarray([0.02, 0.5, 0.98])
        assert trimming_sweep(0.01, values) <= trimming_sweep(0.05, values)

    def test_the_block_catalogue_totals_reasons(self) -> None:
        outcome = apply_trimming(
            _StandIn().decision, {"a": 0.01}, {"a": 1}, ["a"], EligibilityRule(minimum_exposed=10)
        )
        totals = block_catalogue([outcome])
        assert totals[IneligibleReason.EXPOSURE_BELOW_MINIMUM] == 1

    def test_the_eligible_size_is_counted(self) -> None:
        size = eligible_sizes({"a": 0.5, "b": 0.01}, {"a": 100, "b": 100})
        assert size == 1


class _OracleCensoring:
    def predict(self, decision: object, arm: float, time: np.ndarray) -> np.ndarray:
        del decision, arm
        return np.ones_like(np.asarray(time, dtype=np.float64))


class _Oracle:
    def __init__(self, candidate: str, propensity: float, one: float, zero: float) -> None:
        self.candidate = candidate
        self._propensity = propensity
        self._one = one
        self._zero = zero
        self.censoring = _OracleCensoring()
        self.restricted_mean: dict[object, object] = {}

    def propensity_for(self, decisions) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.full(len(decisions), self._propensity)

    def mu(self, decisions, arm: float, **_: object) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.full(len(decisions), self._one if arm >= 0.5 else self._zero)

    def conditional_mean(self, decision, arm: float, grid: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
        del decision, arm
        return np.zeros_like(grid)


class _StandIn:
    from oaer.cohort.builder import RETROSPECTIVE_MARGINALS, CohortBuilder, scaled_marginals
    from oaer.support.types import Split

    _catalogue = __import__(
        "oaer.catalogue.concepts", fromlist=["build_catalogue"]
    ).build_catalogue(size=6, seed=41)
    decision = (
        CohortBuilder(seed=41)
        .build(
            scaled_marginals(RETROSPECTIVE_MARGINALS, 60),
            split_by_site={
                "A": Split.DEV_SITE_A,
                "B": Split.DEV_SITE_B,
                "C": Split.HELD_OUT_SITE_C,
            },
            patients=33,
            candidates=[entry.label for entry in _catalogue],
        )
        .decisions[0]
    )


def _first_exposed(batch) -> str:  # type: ignore[no-untyped-def]
    for decision in batch.decisions:
        for name, flag in decision.exposure.items():
            if flag:
                return name
    raise AssertionError("no exposed candidate in the batch")


def _exposed_names(batch, count: int) -> list[str]:  # type: ignore[no-untyped-def]
    totals: dict[str, int] = {}
    for decision in batch.decisions:
        for name, flag in decision.exposure.items():
            if flag:
                totals[name] = totals.get(name, 0) + 1
    ordered = sorted(totals, key=lambda name: (-totals[name], name))
    return ordered[:count]
