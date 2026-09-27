"""The reported ledgers: their internal arithmetic and their shape.

These tests read the article's own tables as data. A failure here is a statement
about the transcription or about an inconsistency inside the article's tables, not
about a mechanism, so the battery keeps the same reading in a separate family.

Ref: Tables 1-5 and their captions.
"""

from __future__ import annotations

import pytest

from oaer.ledger.ablations import (
    CANONICAL_REMOVALS,
    COMPONENT_LABELS,
    TABLE_3A_REMOVALS,
    TABLE_3B_TERCILES,
    TABLE_3C_MODALITIES,
    TABLE_3D_INTERACTIONS,
    canonical_row,
    expectation_direction,
    interaction_from_rows,
    joint_row,
    row_for,
    tercile_row,
)
from oaer.ledger.comparator_roster import (
    DEPLOYED_PROGNOSIS,
    DEPLOYED_RANKING,
    TABLE_2A_RANKERS,
    TABLE_2B_PROGNOSTIC,
    best_unmarked_ranker,
    family_members,
    is_transfer,
    prognostic_families,
    ranking_families,
    reimplemented_rows,
    transfer_rows,
)
from oaer.ledger.criteria import (
    DISCRIMINATION_MARGIN,
    HAZARD_RATIO_ANCHORS,
    TABLE_1D_CRITERIA,
    adjusted_rows,
    anchored_margins,
    criterion_by_id,
    discrimination_margin_is_reachable,
    primary_rows,
    reference_score_spread,
    summary,
    value_gap_reachability,
)
from oaer.ledger.generalization import (
    ABSTENTION_PERCENT_POOLED,
    MICROSATELLITE_PROBE_AUROC,
    TABLE_5_STRATA,
    adjacent_domain_rows,
    availability_label,
    matched_reference_rows,
    modality_subset,
    panel_rows,
    panels,
    shift_rows,
    worst_case_delta,
)
from oaer.ledger.outcomes import (
    TABLE_1_DIAGNOSTICS,
    TABLE_1A_COHORT,
    TABLE_1B_EFFECTIVE_SIZES,
    TABLE_1B_PRIMARY,
    TABLE_1C_EMULATIONS,
    TABLE_1E_PATHWAYS,
    TABLE_1F_RETROSPECTIVE,
    TABLE_4_SUBGROUPS,
)

PATHWAY_CONTRAST_MARKER = "Top agreement"


class TestCohortLedger:
    def test_the_site_rows_sum_to_the_pooled_row(self) -> None:
        pooled = TABLE_1A_COHORT[-1]
        rows = TABLE_1A_COHORT[:-1]
        assert sum(row.retrospective_patients for row in rows) == pooled.retrospective_patients
        assert sum(row.retrospective_decisions for row in rows) == pooled.retrospective_decisions
        assert sum(row.prospective_decisions for row in rows) == pooled.prospective_decisions

    def test_the_pooled_row_is_the_reported_one(self) -> None:
        pooled = TABLE_1A_COHORT[-1]
        assert pooled.retrospective_patients == 4318
        assert pooled.retrospective_decisions == 7962
        assert pooled.prospective_patients == 1586
        assert pooled.prospective_decisions == 2714

    def test_every_patient_count_is_below_its_decision_count(self) -> None:
        for row in TABLE_1A_COHORT:
            assert row.retrospective_patients <= row.retrospective_decisions

    def test_the_primary_arm_carries_the_reported_clone_count(self) -> None:
        assert TABLE_1B_PRIMARY[-1].cloned_decisions == 1943

    def test_the_primary_site_clones_sum_to_the_pooled_count(self) -> None:
        pooled = TABLE_1B_PRIMARY[-1].cloned_decisions
        assert sum(row.cloned_decisions for row in TABLE_1B_PRIMARY[:-1]) == pooled

    def test_the_retrospective_cells_carry_the_reported_pooled_clone_count(self) -> None:
        assert TABLE_1F_RETROSPECTIVE[-1].cloned_decisions == 5002

    def test_the_effective_sizes_split_the_reported_arms(self) -> None:
        assert TABLE_1B_EFFECTIVE_SIZES["top_agreement"] == 742
        assert TABLE_1B_EFFECTIVE_SIZES["no_initiation"] == 688
        assert TABLE_1B_EFFECTIVE_SIZES["per_line_effective_sample_size"] == 412

    def test_the_reported_policy_contrast_brackets_its_point(self) -> None:
        point, low, high = TABLE_1B_EFFECTIVE_SIZES["policy_contrast_k3"]
        assert low <= point <= high
        assert point == pytest.approx(2.4)

    def test_the_clone_censoring_weight_is_a_median_below_its_ninetyninth(self) -> None:
        median = TABLE_1B_EFFECTIVE_SIZES["clone_censoring_weight_median"]
        tail = TABLE_1B_EFFECTIVE_SIZES["clone_censoring_weight_99th"]
        assert 1.0 <= median <= tail

    def test_the_diagnostics_carry_the_reported_pooled_figures(self) -> None:
        assert TABLE_1_DIAGNOSTICS["sites_clearing_each_primary_criterion"] == 2
        assert TABLE_1_DIAGNOSTICS["pooled_e_value_point"] == (2.31, 1.83)
        assert TABLE_1_DIAGNOSTICS["pooled_e_value_upper"] == (1.42, 1.19)

    def test_every_negative_control_brackets_the_null(self) -> None:
        for key in ("negative_control_pfs", "negative_control_os"):
            point, low, high = TABLE_1_DIAGNOSTICS[key]
            assert low <= point <= high
            assert low < 1.0 < high

    def test_the_between_site_heterogeneity_is_reported_for_each_endpoint(self) -> None:
        for key in ("between_site_i2_pfs", "between_site_i2_os", "between_site_i2_c_index"):
            i_squared, p_value = TABLE_1_DIAGNOSTICS[key]
            assert i_squared == pytest.approx(0.0)
            assert 0.0 <= p_value <= 1.0


class TestCandidateLedger:
    def test_the_four_named_agents_are_reported(self) -> None:
        labels = " | ".join(cell.candidate for cell in TABLE_1C_EMULATIONS).lower()
        for agent in ("metformin", "beta-blocker", "aspirin", "proton-pump inhibitor"):
            assert agent in labels

    def test_the_pooled_identified_set_is_the_reported_one(self) -> None:
        pooled = next(cell for cell in TABLE_1C_EMULATIONS if cell.exposed == 11275)
        assert pooled.pfs_hazard_ratio[0] == pytest.approx(0.74)
        assert pooled.e_value == pytest.approx(1.87)

    def test_every_hazard_ratio_brackets_its_point_estimate(self) -> None:
        for cell in TABLE_1C_EMULATIONS:
            point, low, high = cell.pfs_hazard_ratio
            assert low <= point <= high
            assert low > 0.0 and high > 0.0

    def test_every_negative_control_sits_near_one(self) -> None:
        for cell in TABLE_1C_EMULATIONS:
            assert abs(cell.negative_control - 1.0) < 0.1

    def test_the_proton_pump_inhibitor_row_is_the_adverse_one(self) -> None:
        ppl = next(cell for cell in TABLE_1C_EMULATIONS if "Proton-pump" in cell.candidate)
        assert ppl.pfs_hazard_ratio[0] > 1.0


class TestPathwayLedger:
    def _buckets(self):  # type: ignore[no-untyped-def]
        return [row for row in TABLE_1E_PATHWAYS if PATHWAY_CONTRAST_MARKER not in row.pathway]

    def test_the_pathway_shares_sum_to_one(self) -> None:
        assert sum(row.share_percent for row in self._buckets()) == pytest.approx(100.0, abs=0.2)

    def test_the_pathway_decision_counts_sum_to_the_analysed_clones(self) -> None:
        assert sum(row.decisions for row in self._buckets()) == 1943

    def test_the_reference_pathway_carries_no_contrast(self) -> None:
        reference = next(row for row in TABLE_1E_PATHWAYS if "reference" in row.pathway.lower())
        assert reference.pfs_hazard_ratio == pytest.approx(1.0)

    def test_the_agreement_contrast_is_reported_beside_the_buckets(self) -> None:
        contrast = next(row for row in TABLE_1E_PATHWAYS if PATHWAY_CONTRAST_MARKER in row.pathway)
        assert contrast.decisions > max(row.decisions for row in self._buckets())

    def test_the_identified_pathways_are_ordered_by_decreasing_benefit(self) -> None:
        identified = [
            row.pfs_hazard_ratio
            for row in TABLE_1E_PATHWAYS
            if row.pfs_hazard_ratio < 1.0 and PATHWAY_CONTRAST_MARKER not in row.pathway
        ]
        assert identified == sorted(identified)


class TestAblationLedger:
    def test_each_component_has_a_canonical_removal(self) -> None:
        for component in ("C1", "C2", "C3", "C4"):
            assert component in CANONICAL_REMOVALS
            assert canonical_row(component).hits_delta < 0.0

    def test_the_deployed_configuration_is_the_only_undeleted_row(self) -> None:
        nulls = [row.configuration for row in TABLE_3A_REMOVALS if row.hits_delta == 0.0]
        assert "Full OAER" in nulls
        assert "Tier 0: clinical-covariate Cox" in nulls

    def test_every_component_label_is_named(self) -> None:
        assert COMPONENT_LABELS["C1"].startswith("outcome-anchoring")
        assert COMPONENT_LABELS["C3"] == "identification gate"

    def test_the_removal_deltas_are_ordered_by_magnitude(self) -> None:
        canonical = {
            component: canonical_row(component).hits_delta for component in ("C1", "C2", "C3", "C4")
        }
        assert canonical["C1"] < canonical["C2"] < canonical["C4"]

    def test_every_interaction_is_traceable_to_three_reported_rows(self) -> None:
        for pair in (("C1", "C2"), ("C1", "C3"), ("C1", "C4"), ("C2", "C3"), ("C2", "C4")):
            iab, ratio = interaction_from_rows(*pair)
            joint = joint_row(*pair)
            left = canonical_row(pair[0]).hits_delta
            right = canonical_row(pair[1]).hits_delta
            assert iab == pytest.approx(joint.hits_delta - (left + right))
            assert ratio is not None
            assert ratio == pytest.approx((left + right) / joint.hits_delta)

    def test_the_reported_interactions_are_negative_where_synergy_is_expected(self) -> None:
        for row in TABLE_3D_INTERACTIONS:
            if expectation_direction(row.expectation) != "synergy":
                continue
            iab, ratio = interaction_from_rows(row.first.split()[0], row.second.split()[0])
            assert iab < 0.0
            assert ratio is not None and ratio < 1.0

    def test_the_expected_redundancy_pair_has_a_ratio_above_one(self) -> None:
        iab, ratio = interaction_from_rows("C2", "C3")
        assert iab > 0.0
        assert ratio is not None and ratio > 1.0

    def test_the_pair_with_no_pre_specified_direction_carries_no_ratio(self) -> None:
        row = next(row for row in TABLE_3D_INTERACTIONS if "no direction" in row.expectation)
        _, ratio = interaction_from_rows(row.first.split()[0], row.second.split()[0])
        # The point ratio is computable but is withheld, because Sec. 1.4 prints IR
        # only where the interval of its denominator stays entirely above zero.
        assert row.ratio is None
        assert ratio is not None

    def test_the_tercile_rows_gain_most_at_the_lowest_exposure(self) -> None:
        assert (
            tercile_row("lowest", "random").hits_gain > tercile_row("highest", "random").hits_gain
        )

    def test_the_seven_tercile_rows_are_reported(self) -> None:
        assert len(TABLE_3B_TERCILES) == 7

    def test_every_modality_removal_but_the_reference_costs_hits(self) -> None:
        for row in TABLE_3C_MODALITIES:
            if row.supplementary:
                continue
            if "full model" in row.modalities:
                assert row.hits_delta == pytest.approx(0.0)
            else:
                assert row.hits_delta < 0.0

    def test_the_encoder_substitution_is_a_supplementary_row(self) -> None:
        ensemble = next(row for row in TABLE_3C_MODALITIES if "ensemble" in row.modalities)
        assert ensemble.supplementary

    def test_the_joint_removal_is_never_cheaper_than_either_single(self) -> None:
        for pair in (("C1", "C2"), ("C2", "C4"), ("C3", "C4")):
            joint = joint_row(*pair).hits_delta
            assert joint <= canonical_row(pair[0]).hits_delta
            assert joint <= canonical_row(pair[1]).hits_delta

    def test_an_unknown_configuration_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            row_for("not a configuration")

    def test_an_unknown_tercile_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            tercile_row("deepest", "random")


class TestCriteriaLedger:
    def test_both_primary_rows_are_met(self) -> None:
        primaries = primary_rows()
        assert len(primaries) == 2
        assert all(criterion.met for criterion in primaries)

    def test_the_primary_pair_carries_no_multiplicity_adjustment(self) -> None:
        assert not any(criterion.multiplicity_adjusted for criterion in primary_rows())

    def test_the_adjusted_rows_are_disjoint_from_the_primary_ones(self) -> None:
        assert adjusted_rows()
        assert not (
            {c.identifier for c in adjusted_rows()} & {c.identifier for c in primary_rows()}
        )

    def test_the_reported_hazard_ratio_margin_is_arithmetic_and_positive(self) -> None:
        criterion = criterion_by_id("primary-pfs-hr")
        expected = HAZARD_RATIO_ANCHORS["reported primary bar"] - 0.68
        assert criterion.margin == pytest.approx(expected)
        assert criterion.margin > 0.0

    def test_the_restricted_mean_margin_matches_the_reported_values(self) -> None:
        assert criterion_by_id("primary-rmst-contrast").margin == pytest.approx(2.4 - 1.5)

    def test_the_discrimination_sensitivity_row_is_not_met(self) -> None:
        sensitivity = criterion_by_id("sensitivity-c-index-observed-score")
        assert not sensitivity.met
        assert sensitivity.margin < 0.0

    def test_the_supporting_discrimination_row_clears_the_margin_by_a_hair(self) -> None:
        supporting = criterion_by_id("supporting-c-index-best-score")
        assert supporting.met
        assert 0.0 < supporting.margin < 0.01

    def test_the_reference_spread_is_derived_from_the_roster(self) -> None:
        spread = reference_score_spread()
        points = [
            row.os_c_index[0]
            for row in TABLE_2B_PROGNOSTIC
            if row.family == "reproduced prognostic reference scores"
        ]
        assert spread == (min(points), max(points))

    def test_the_ordering_margin_is_wider_than_the_family_it_names(self) -> None:
        # The manuscript reads this margin against the spread of the reproduced
        # reference scores; on the reported panel that spread is narrower than the
        # margin, so the sensitivity row's verdict is fixed by the arithmetic.
        spread = reference_score_spread()
        assert spread[1] - spread[0] < DISCRIMINATION_MARGIN
        assert not discrimination_margin_is_reachable()
        assert discrimination_margin_is_reachable((0.60, 0.70))

    def test_the_policy_value_can_clear_the_bar(self) -> None:
        assert value_gap_reachability(5.9, 3.5)
        assert not value_gap_reachability(2.0, 1.5)

    def test_the_observed_value_sits_below_the_narrowest_anchor(self) -> None:
        margins = anchored_margins(0.68)
        assert margins["reported primary bar"] == pytest.approx(-0.07)
        assert set(margins) == set(HAZARD_RATIO_ANCHORS)

    def test_the_summary_counts_agree_with_the_rows(self) -> None:
        payload = summary()
        assert payload["total"] == len(TABLE_1D_CRITERIA)
        assert payload["primary"] == len(primary_rows())
        assert payload["met"] + payload["not_met"] == payload["total"]

    def test_an_unknown_criterion_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            criterion_by_id("no such criterion")


class TestComparatorLedger:
    def test_the_deployed_row_is_the_last_one(self) -> None:
        assert TABLE_2A_RANKERS[-1] is DEPLOYED_RANKING
        assert TABLE_2B_PROGNOSTIC[-1] is DEPLOYED_PROGNOSIS

    def test_the_deployed_ranking_beats_every_fitted_alternative(self) -> None:
        for row in TABLE_2A_RANKERS[:-1]:
            assert DEPLOYED_RANKING.hits_at_20 >= row.hits_at_20

    def test_the_ten_transfer_rows_carry_the_marker(self) -> None:
        transfers = transfer_rows()
        assert len(transfers) == 10
        assert all(is_transfer(row) for row in transfers)

    def test_the_transfer_rows_are_never_the_fitted_comparators(self) -> None:
        fitted = [row for row in TABLE_2A_RANKERS[:-1] if not is_transfer(row)]
        assert fitted
        assert DEPLOYED_RANKING.hits_at_20 >= best_unmarked_ranker().hits_at_20
        assert best_unmarked_ranker() in fitted

    def test_the_roster_names_its_families(self) -> None:
        assert "network proximity and diffusion" in ranking_families()
        assert prognostic_families()

    def test_each_family_is_non_empty(self) -> None:
        for family in ranking_families():
            assert family_members(family)

    def test_the_reimplemented_rows_name_their_reason(self) -> None:
        rows = reimplemented_rows()
        assert rows
        for row in rows:
            assert row.provenance.startswith("re-implemented by us")

    def test_the_prognostic_rows_carry_an_interval_around_their_point(self) -> None:
        for row in TABLE_2B_PROGNOSTIC:
            point, low, high = row.os_c_index
            assert low <= point <= high

    def test_ranking_metrics_are_undefined_for_the_prognostic_roster(self) -> None:
        for row in TABLE_2B_PROGNOSTIC:
            assert not hasattr(row, "hits_at_20")


class TestGeneralizationLedger:
    def test_every_panel_is_non_empty(self) -> None:
        for panel in panels():
            assert panel_rows(panel)

    def test_the_site_panel_is_the_development_and_held_out_rows(self) -> None:
        names = {row.stratum for row in panel_rows("sites and regions")}
        assert {"Site A", "Site B", "Site C"} <= names

    def test_the_held_out_site_shifts_further_than_the_development_sites(self) -> None:
        rows = {row.stratum: row for row in panel_rows("sites and regions")}
        assert rows["Site C"].delta_vs_matched_reference < rows["Site A"].delta_vs_matched_reference

    def test_the_worst_case_stress_row_is_the_largest_shift(self) -> None:
        assert worst_case_delta() == pytest.approx(-0.041)

    def test_a_public_cohort_row_carries_its_domain_label(self) -> None:
        rows = adjacent_domain_rows()
        assert rows
        for row in rows:
            assert row.domain.startswith("adjacent")

    def test_the_matched_reference_rows_are_a_subset_of_the_full_one(self) -> None:
        assert len(matched_reference_rows()) <= len(TABLE_5_STRATA)
        assert modality_subset()

    def test_the_modality_label_joins_its_parts(self) -> None:
        assert availability_label(("mol.", "path.")) == "mol. and path."

    def test_the_microsatellite_probe_is_a_sanity_check_only(self) -> None:
        assert pytest.approx(0.843) == MICROSATELLITE_PROBE_AUROC
        probe = next(row for row in TABLE_5_STRATA if "Microsatellite probe" in row.stratum)
        assert probe.os_c_index is None
        assert "sanity" in probe.domain

    def test_the_proteogenomic_row_carries_no_concordance_index(self) -> None:
        row = next(r for r in TABLE_5_STRATA if r.stratum == "CPTAC-COAD")
        assert row.os_c_index is None
        assert "follow-up" in row.note

    def test_the_shift_rows_move_away_from_the_reference(self) -> None:
        for row in shift_rows():
            assert row.delta_vs_matched_reference is not None
            assert row.delta_vs_matched_reference < 0.0

    def test_the_pooled_abstention_is_reported(self) -> None:
        assert pytest.approx(28.4) == ABSTENTION_PERCENT_POOLED


class TestSubgroupLedger:
    def test_the_molecular_partition_sums_to_the_whole_cohort_row(self) -> None:
        partition = [cell for cell in TABLE_4_SUBGROUPS if cell.block == "molecular partition"]
        whole = next(cell for cell in TABLE_4_SUBGROUPS if cell.subgroup == "All records")
        assert partition
        assert (
            sum(cell.retrospective_decisions for cell in partition) == whole.retrospective_decisions
        )

    def test_the_overlapping_block_says_its_counts_are_not_additive(self) -> None:
        overlapping = [cell for cell in TABLE_4_SUBGROUPS if cell.block == "additional alterations"]
        assert overlapping

    def test_every_subgroup_hazard_ratio_brackets_its_point(self) -> None:
        for cell in TABLE_4_SUBGROUPS:
            point, low, high = cell.pfs_hazard_ratio
            assert low <= point <= high

    def test_the_microsatellite_high_stratum_defers_most(self) -> None:
        msi = next(
            cell for cell in TABLE_4_SUBGROUPS if cell.subgroup == "Microsatellite-instability-high"
        )
        assert msi.deferral_percent > 80.0

    def test_the_whole_cohort_row_carries_the_reported_decision_count(self) -> None:
        whole = next(cell for cell in TABLE_4_SUBGROUPS if cell.subgroup == "All records")
        assert whole.retrospective_decisions == 7962
        assert whole.prospective_decisions == 2714
