"""The cohort schema, the builder, the fold construction and the slide route."""

from __future__ import annotations

import numpy as np
import pytest

from oaer.cohort.builder import (
    NAMED_EXPOSURE_PREVALENCE,
    PROSPECTIVE_MARGINALS,
    RETROSPECTIVE_MARGINALS,
    RETROSPECTIVE_MARGINALS as REPORTED,
    CohortBuilder,
    build_cohort,
    interleave,
    marginal_counts,
    marginal_table,
    modality_census,
    reported_hazard_ratio,
    reported_restricted_mean,
    scaled_marginals,
)
from oaer.cohort.pathology import (
    MORPHOLOGY_KEYS,
    SlideEmbedder,
    centre_modifier,
    embed_slide,
    embedding_norm,
    morphological_lexicon_score,
    slide_signature,
    source_indicator,
    tile_occupancy,
)
from oaer.cohort.schema import (
    DECISION_COLUMNS,
    MOLECULAR_PANEL_COLUMNS,
    CohortSchemaError,
    RecordSchema,
    panel_from_record,
    validate_record,
)
from oaer.cohort.splits import (
    DEVELOPMENT_SPLITS,
    cross_fitted_folds,
    first_decision_per_patient,
    fold_assignment,
    fold_size_spread,
    inner_early_stop_split,
    outer_site_split,
    partition_closes_on,
    site_counts,
)
from oaer.support.types import ModalityAvailability, MolecularStratum


class TestMarginals:
    def test_every_reported_retrospective_variable_matches(self) -> None:
        names = [entry.label for entry in _catalogue()]
        batch, _ = build_cohort(seed=21, candidates=names)
        for variable, expected in RETROSPECTIVE_MARGINALS.items():
            realised = marginal_counts(batch, variable)
            assert realised == dict(expected), variable

    def test_every_reported_prospective_variable_matches(self) -> None:
        names = [entry.label for entry in _catalogue()]
        _, batch = build_cohort(seed=21, candidates=names)
        for variable, expected in PROSPECTIVE_MARGINALS.items():
            assert marginal_counts(batch, variable) == dict(expected)

    def test_the_reported_totals_are_the_pooled_decision_counts(self) -> None:
        assert sum(RETROSPECTIVE_MARGINALS["site"].values()) == 7962
        assert sum(PROSPECTIVE_MARGINALS["site"].values()) == 2714

    def test_interleaving_keeps_each_count_exactly(self) -> None:
        counts = {"a": 3, "b": 5, "c": 2}
        spread = interleave(counts)
        assert len(spread) == 10
        assert {label: spread.count(label) for label in counts} == counts

    def test_interleaving_spreads_rather_than_blocks(self) -> None:
        spread = interleave({"a": 5, "b": 5})
        assert spread[0] != spread[1] or spread[1] != spread[2]

    def test_an_empty_marginal_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            interleave({})

    def test_scaling_preserves_the_total_and_the_labels(self) -> None:
        scaled = scaled_marginals(REPORTED, 226)
        assert all(sum(counts.values()) == 226 for counts in scaled.values())
        assert all(set(counts) == set(REPORTED[variable]) for variable, counts in scaled.items())

    def test_a_scaled_arm_lands_on_the_scaled_table(self) -> None:
        names = [entry.label for entry in _catalogue()]
        batch, _ = build_cohort(seed=23, candidates=names, total=226)
        assert marginal_counts(batch, "site") == scaled_marginals(REPORTED, 226)["site"]

    def test_the_marginal_table_carries_its_total(self) -> None:
        assert marginal_table({"a": 2, "b": 3})["_total"] == 5

    def test_a_negative_marginal_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            interleave({"a": -1})


class TestStratum:
    def test_the_partition_is_exhaustive_and_disjoint(self) -> None:
        names = [entry.label for entry in _catalogue()]
        batch, _ = build_cohort(seed=25, candidates=names)
        grouped = batch.by_stratum()
        assert sum(len(members) for members in grouped.values()) == len(batch)
        for stratum, members in grouped.items():
            assert all(d.state.molecular.stratum is stratum for d in members)

    def test_an_unassayed_marker_puts_a_record_in_the_residual_block(self) -> None:
        panel = panel_from_record(
            {
                "ras_mutant": False,
                "braf_v600e": False,
                "msi_high": False,
                "pi3k_altered": False,
                "erbb2_amplified": None,
                "kras_g12c": False,
                "assay_panel": "narrow",
                "microsatellite_assay": "pcr",
            }
        )
        assert panel.stratum is MolecularStratum.OTHER
        assert not panel.fully_typed

    def test_microsatellite_high_takes_precedence(self) -> None:
        panel = panel_from_record(
            {
                "ras_mutant": False,
                "braf_v600e": True,
                "msi_high": True,
                "pi3k_altered": False,
                "erbb2_amplified": True,
                "kras_g12c": False,
                "assay_panel": "broad",
                "microsatellite_assay": "pcr",
            }
        )
        assert panel.stratum is MolecularStratum.MSI_HIGH

    def test_a_fully_typed_wild_type_record_lands_in_the_wild_type_stratum(self) -> None:
        panel = panel_from_record(
            {
                "ras_mutant": False,
                "braf_v600e": False,
                "msi_high": False,
                "pi3k_altered": False,
                "erbb2_amplified": False,
                "kras_g12c": False,
                "assay_panel": "broad",
                "microsatellite_assay": "pcr",
            }
        )
        assert panel.stratum is MolecularStratum.RAS_BRAF_WILD_TYPE_MSS


class TestExposure:
    def test_the_named_prevalences_follow_the_reported_counts(self) -> None:
        assert NAMED_EXPOSURE_PREVALENCE["metformin"] == pytest.approx(1286 / 7962)
        assert NAMED_EXPOSURE_PREVALENCE["omeprazole"] == pytest.approx(1704 / 7962)

    def test_realised_prevalence_sits_near_the_reported_one(self) -> None:
        names = [entry.label for entry in _catalogue()]
        batch, _ = build_cohort(seed=27, candidates=names)
        for name, prevalence in NAMED_EXPOSURE_PREVALENCE.items():
            realised = sum(
                1 for decision in batch.decisions if decision.exposure.get(name, False)
            ) / len(batch)
            tolerance = 6.0 * (prevalence * (1 - prevalence) / len(batch)) ** 0.5
            assert abs(realised - prevalence) <= tolerance + 1e-9

    def test_the_builder_is_deterministic(self) -> None:
        names = [entry.label for entry in _catalogue()]
        left, _ = build_cohort(seed=29, candidates=names, total=200)
        right, _ = build_cohort(seed=29, candidates=names, total=200)
        assert [d.exposure for d in left.decisions] == [d.exposure for d in right.decisions]


class TestClosedForm:
    def test_the_reported_restricted_mean_matches_a_sample(self) -> None:
        stream = np.random.default_rng(5)
        eta, horizon, shape, scale = -0.25, 8.0, 1.18, 6.85
        uniform = np.clip(stream.random(120000), 1e-12, 1 - 1e-12)
        times = scale * ((-np.log(uniform)) / np.exp(eta)) ** (1 / shape)
        sample = float(np.minimum(times, horizon).mean())
        closed = reported_restricted_mean(eta, horizon, shape=shape, scale=scale)
        error = float(np.std(np.minimum(times, horizon), ddof=1) / np.sqrt(times.size))
        assert abs(sample - closed) <= 4.0 * error

    def test_the_hazard_ratio_is_the_exponential_of_the_effect(self) -> None:
        assert reported_hazard_ratio(np.log(0.71)) == pytest.approx(0.71)

    def test_builder_exposure_census_is_recorded(self) -> None:
        builder = CohortBuilder(seed=31)
        names = [entry.label for entry in _catalogue()]
        builder.build(
            scaled_marginals(REPORTED, 200),
            split_by_site={"A": _dev_a(), "B": _dev_b(), "C": _held()},
            patients=108,
            candidates=names,
        )
        assert set(builder.exposure_census) == set(names)


class TestSplits:
    def test_folds_assign_every_patient(self, development) -> None:
        structure = fold_assignment(development, 4, seed=11)
        assert set(structure.assignment) == {
            decision.patient_id for decision in development.decisions
        }

    def test_fold_sizes_are_balanced(self, development) -> None:
        structure = fold_assignment(development, 4, seed=11)
        assert fold_size_spread(structure) <= 6

    def test_cross_fitting_partitions_the_records(self, development) -> None:
        structure, partitions = cross_fitted_folds(development, 3, seed=13)
        for train, test in partitions:
            assert partition_closes_on(test, len(test))
            assert len(train) + len(test) == len(development)
        assert len(structure.assignment) > 0

    def test_site_split_separates_the_held_out_site(self, cohort) -> None:
        development, held = outer_site_split(cohort[0])
        assert all(decision.split in DEVELOPMENT_SPLITS for decision in development.decisions)
        assert all(decision.site == "C" for decision in held.decisions)

    def test_the_inner_split_is_non_empty(self, development) -> None:
        early, late = inner_early_stop_split(development, seed=17)
        assert len(early) > 0 and len(late) > 0

    def test_the_first_decision_per_patient_is_kept_once(self, cohort) -> None:
        primary = first_decision_per_patient(cohort[0])
        patients = [decision.patient_id for decision in primary.decisions]
        assert len(patients) == len(set(patients))

    def test_site_counts_sum_to_the_batch(self, cohort) -> None:
        assert sum(site_counts(cohort[0]).values()) == len(cohort[0])

    def test_a_single_fold_is_rejected(self, development) -> None:
        with pytest.raises(ValueError):
            fold_assignment(development, 1)


class TestSchema:
    def test_the_decision_contract_carries_every_layer(self) -> None:
        for column in ("ras_mutant", "slide_source", "line_of_therapy", "follow_up_months"):
            assert column in DECISION_COLUMNS

    def test_the_molecular_contract_is_a_subset_of_the_decision_contract(self) -> None:
        assert set(MOLECULAR_PANEL_COLUMNS) <= set(DECISION_COLUMNS)

    def test_a_valid_record_passes(self) -> None:
        validate_record(_valid_record())

    def test_a_missing_column_is_reported(self) -> None:
        record = _valid_record()
        del record["site"]
        assert "site" in RecordSchema().missing(record)

    def test_an_out_of_domain_site_is_rejected(self) -> None:
        record = _valid_record()
        record["site"] = "D"
        with pytest.raises(CohortSchemaError):
            validate_record(record)

    def test_a_microsatellite_high_ras_mutant_record_is_rejected(self) -> None:
        record = _valid_record()
        record["ras_mutant"] = True
        record["msi_high"] = True
        with pytest.raises(CohortSchemaError):
            validate_record(record)

    def test_a_non_positive_follow_up_is_rejected(self) -> None:
        record = _valid_record()
        record["follow_up_months"] = 0.0
        with pytest.raises(CohortSchemaError):
            validate_record(record)

    def test_an_unassayed_marker_must_be_absent_rather_than_false(self) -> None:
        record = _valid_record()
        record["erbb2_amplified"] = "not a boolean"
        with pytest.raises(CohortSchemaError):
            validate_record(record)

    def test_modality_census_is_additive(self, cohort) -> None:
        census = modality_census(cohort[0])
        assert sum(census.values()) == len(cohort[0])
        assert census["all_three"] == sum(
            1
            for d in cohort[0].decisions
            if d.modalities_available is ModalityAvailability.ALL_THREE
        )


class TestPathology:
    def test_the_lexicon_keys_are_declared(self) -> None:
        assert len(MORPHOLOGY_KEYS) == 4

    def test_the_embedder_round_trips_the_morphology_block(self) -> None:
        embedder = SlideEmbedder(dim=16, seed=3)
        features = {key: 0.4 + 0.1 * index for index, key in enumerate(MORPHOLOGY_KEYS)}
        vector = embedder.encode(features, purity=0.6)
        recovered = embedder.decode(vector)
        assert recovered.size == len(MORPHOLOGY_KEYS)

    def test_the_embedding_has_the_configured_width(self, development) -> None:
        profile = development.decisions[0].state.pathology
        assert len(embed_slide(profile, dim=16)) == 16

    def test_the_signature_is_a_scalar(self, development) -> None:
        embedding = embed_slide(development.decisions[0].state.pathology)
        assert isinstance(slide_signature(embedding), float)

    def test_centring_removes_the_eligible_mean(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 100.0])
        mask = np.asarray([True, True, True, False])
        centred = centre_modifier(values, mask)
        assert centred[:3].mean() == pytest.approx(0.0)

    def test_centring_without_a_mask_uses_every_row(self) -> None:
        centred = centre_modifier(np.asarray([1.0, 3.0]))
        assert centred.mean() == pytest.approx(0.0)

    def test_the_tile_occupancy_is_bounded(self, development) -> None:
        assert 0.0 <= tile_occupancy(development.decisions[0].state.pathology) <= 1.0

    def test_the_source_indicator_is_a_pair(self, development) -> None:
        pair = source_indicator(development.decisions[0].state.pathology.slide_source)
        assert len(pair) == 2 and sum(pair) <= 1.0

    def test_the_lexicon_score_is_finite(self, development) -> None:
        assert np.isfinite(morphological_lexicon_score(development.decisions[0].state.pathology))

    def test_the_embedding_norm_is_non_negative(self, development) -> None:
        assert embedding_norm(embed_slide(development.decisions[0].state.pathology)) >= 0.0


def _catalogue():  # type: ignore[no-untyped-def]
    from oaer.catalogue.concepts import build_catalogue

    return build_catalogue(size=12, seed=11)


def _dev_a():  # type: ignore[no-untyped-def]
    from oaer.support.types import Split

    return Split.DEV_SITE_A


def _dev_b():  # type: ignore[no-untyped-def]
    from oaer.support.types import Split

    return Split.DEV_SITE_B


def _held():  # type: ignore[no-untyped-def]
    from oaer.support.types import Split

    return Split.HELD_OUT_SITE_C


def _valid_record() -> dict[str, object]:
    return {
        "decision_id": "d0",
        "patient_id": "p0",
        "site": "A",
        "line_of_therapy": "first",
        "age_years": 61.0,
        "female": False,
        "ecog": 0,
        "sidedness": "left_sided_including_rectum",
        "metastatic_pattern": "liver_limited",
        "presentation": "synchronous",
        "prior_lines": 0,
        "organ_sites": 1,
        "albumin_g_per_l": 39.0,
        "ldh_ratio_to_upper_limit": 1.1,
        "cea_ng_per_ml": 12.0,
        "comorbidity_count": 2,
        "ras_mutant": True,
        "braf_v600e": False,
        "msi_high": False,
        "pi3k_altered": False,
        "erbb2_amplified": False,
        "kras_g12c": False,
        "assay_panel": "broad",
        "microsatellite_assay": "pcr",
        "slide_source": "primary_resection",
        "tile_count": 9000.0,
        "tumour_purity": 0.6,
        "backbone_regimen": "folfox",
        "follow_up_months": 6.0,
        "event": True,
        "objective_response": True,
        "modality_availability": "all_three",
    }
