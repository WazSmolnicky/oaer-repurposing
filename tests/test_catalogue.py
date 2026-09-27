"""The candidate catalogue and the typed knowledge graph."""

from __future__ import annotations

import numpy as np
import pytest

from oaer.catalogue.concepts import (
    IndicationRecord,
    admitted_by_indications,
    build_catalogue,
    catalogue_by_identifier,
    classify_indication,
    degree_histogram,
    degree_terciles,
    named_candidates,
    reprocessing_priority,
)
from oaer.catalogue.graph import (
    ANCHOR_RELATION,
    RELATION_INDEX,
    AnchorType,
    NodeKind,
    anchors_of,
    attach_patient_node,
    non_anchors,
    relation_counts,
)
from oaer.catalogue.motifs import (
    MOTIF_SET,
    Motif,
    enumerate_motifs,
    motif_bank,
    motif_frequencies,
    motif_laplacian,
    motif_operator,
    motif_signature,
    motif_spectrum,
    orbit_partition,
)
from oaer.catalogue.sampling import sample_masked_edges
from oaer.catalogue.sources import (
    AUXILIARY_SOURCES,
    CANDIDATE_SOURCES,
    PUBLIC_COHORTS,
    candidate_source_envelope,
)


class TestIndications:
    def test_oncology_terms_are_recognised(self) -> None:
        assert classify_indication("metastatic colorectal carcinoma")
        assert classify_indication("acute lymphoblastic leukaemia")

    def test_non_oncology_labels_are_not(self) -> None:
        assert not classify_indication("type 2 diabetes mellitus")
        assert not classify_indication("gastro-oesophageal reflux disease")

    def test_hyphens_and_case_do_not_change_the_verdict(self) -> None:
        assert classify_indication("NON-SMALL-CELL LUNG CARCINOMA") == classify_indication(
            "non small cell lung carcinoma"
        )

    def test_admission_needs_one_non_oncological_record(self) -> None:
        records = [
            IndicationRecord("colon carcinoma", "ChEMBL", True),
            IndicationRecord("hypertension", "Open Targets Platform", False),
        ]
        assert admitted_by_indications(records)

    def test_an_oncology_only_label_is_rejected(self) -> None:
        records = [IndicationRecord("colon carcinoma", "ChEMBL", True)]
        assert not admitted_by_indications(records)

    def test_an_empty_label_set_is_rejected(self) -> None:
        assert not admitted_by_indications([])


class TestCatalogue:
    def test_the_directory_closes_at_the_requested_size(self) -> None:
        assert len(build_catalogue(size=17, seed=3)) == 17

    def test_the_named_agents_are_present(self) -> None:
        catalogue = build_catalogue(size=12, seed=5)
        labels = set(named_candidates(catalogue))
        assert labels == {"metformin", "bisoprolol", "aspirin", "omeprazole"}

    def test_every_admitted_agent_carries_a_non_oncological_indication(self) -> None:
        for entry in build_catalogue(size=40, seed=7):
            assert entry.admitted
            assert any(not classify_indication(label) for label in entry.approved_indications)

    def test_the_build_is_deterministic_under_a_seed(self) -> None:
        left = build_catalogue(size=20, seed=9)
        right = build_catalogue(size=20, seed=9)
        assert [entry.label for entry in left] == [entry.label for entry in right]

    def test_named_only_returns_the_four(self) -> None:
        assert len(build_catalogue(size=99, seed=1, named_only=True)) == 4

    def test_the_identifier_index_covers_the_directory(self) -> None:
        catalogue = build_catalogue(size=14, seed=2)
        index = catalogue_by_identifier(catalogue)
        assert len(index) == len(catalogue)

    def test_the_prospective_floor_is_applied(self) -> None:
        entry = build_catalogue(size=12, seed=4)[0]
        assert reprocessing_priority(entry, floor=150.0) == pytest.approx(
            entry.knowledge_graph_degree * 10.0 - 150.0
        )

    def test_the_tercile_split_covers_the_directory_once(self) -> None:
        catalogue = build_catalogue(size=15, seed=6)
        terciles = degree_terciles(catalogue)
        union = set().union(*terciles.values())
        assert len(union) == len(catalogue)

    def test_the_degree_histogram_counts_every_entry(self) -> None:
        catalogue = build_catalogue(size=11, seed=8)
        assert sum(degree_histogram(catalogue).values()) == len(catalogue)


class TestGraph:
    def test_the_seven_relations_are_declared(self) -> None:
        assert len(RELATION_INDEX) == 7

    def test_the_graph_is_built_with_typed_edges(self, graph) -> None:
        counts = relation_counts(graph)
        assert sum(counts.values()) == len(graph.edges)
        assert counts["has_target"] > 0

    def test_the_node_order_is_stable(self, graph) -> None:
        assert graph.index() == {node: position for position, node in enumerate(graph.nodes)}

    def test_adding_a_duplicate_edge_is_refused(self, graph) -> None:
        edge = graph.edges[0]
        assert not graph.add_edge(edge.head, edge.relation, edge.tail)

    def test_an_unknown_relation_is_rejected(self, graph) -> None:
        with pytest.raises(ValueError):
            graph.add_edge(graph.edges[0].head, "not_a_relation", graph.edges[0].tail)

    def test_an_edge_needs_both_endpoints(self) -> None:
        from oaer.catalogue.graph import KnowledgeGraph, Node

        fresh = KnowledgeGraph()
        fresh.add_node(Node("a", NodeKind.TARGET, "a"))
        with pytest.raises(KeyError):
            fresh.add_edge("a", "has_target", "missing")

    def test_the_arrays_match_the_edge_list(self, graph) -> None:
        head, relation, tail = graph.arrays()
        assert head.size == relation.size == tail.size == len(graph.edges)

    def test_patient_anchors_round_trip(self, graph, development) -> None:
        decision = development.decisions[0]
        attach_patient_node(graph, decision.patient_id, decision.state)
        attachment = anchors_of(graph, decision.patient_id)
        assert attachment[AnchorType.MOLECULAR_ALTERATION]
        assert attachment[AnchorType.DISEASE]

    def test_non_anchors_are_drawn_from_the_same_kind(self, graph) -> None:
        pool = non_anchors(graph, AnchorType.PATHWAY, 3)
        for node in pool:
            assert graph.nodes[node].kind is NodeKind.PATHWAY

    def test_every_anchor_type_has_a_relation(self) -> None:
        assert set(ANCHOR_RELATION) == set(AnchorType)

    def test_the_census_is_additive(self, graph) -> None:
        census = graph.census()
        assert census["nodes"] == sum(census[f"nodes_{kind.value}"] for kind in NodeKind)

    def test_degree_matches_the_edge_list(self, graph) -> None:
        node = graph.edges[0].head
        assert graph.out_degree(node) == sum(1 for edge in graph.edges if edge.head == node)


class TestMotifs:
    def test_the_motif_set_is_declared(self) -> None:
        assert len(MOTIF_SET) >= 8
        assert all(motif.order >= 1 for motif in MOTIF_SET)

    def test_a_single_relation_operator_is_the_adjacency(self, graph) -> None:
        operator = motif_operator(graph, ("has_target",))
        assert operator.max() <= 1.0
        assert operator.sum() == sum(1 for edge in graph.edges if edge.relation == "has_target")

    def test_the_laplacian_has_a_zero_diagonal_row_sum_property(self, graph) -> None:
        laplacian = motif_laplacian(graph, ("has_target",))
        assert laplacian.shape[0] == laplacian.shape[1]
        assert np.allclose(np.diag(laplacian), 1.0)

    def test_the_spectrum_is_non_negative(self, graph) -> None:
        spectrum = motif_spectrum(graph, ("has_target", "pathway_membership"))
        assert np.all(spectrum >= -1e-9)

    def test_the_signature_has_one_entry_per_motif(self, graph) -> None:
        signature = motif_signature(graph, graph.nodes_of(NodeKind.TARGET)[0])
        assert signature.size == len(MOTIF_SET)
        assert np.all(signature >= 0.0)

    def test_the_bank_has_one_row_per_node(self, graph) -> None:
        assert motif_bank(graph).shape == (len(graph.nodes), len(MOTIF_SET))

    def test_the_frequencies_are_a_distribution(self, graph) -> None:
        frequencies = motif_frequencies(graph)
        assert sum(frequencies.values()) == pytest.approx(1.0)

    def test_the_enumeration_is_a_subset_of_the_realised_patterns(self, graph) -> None:
        enumerated = enumerate_motifs(graph, 2)
        for motif in enumerated:
            assert motif.order == 2
            assert np.count_nonzero(motif_operator(graph, motif)) > 0

    def test_an_orbit_partition_covers_a_neighbourhood(self, graph) -> None:
        identifier = graph.edges[0].head
        partition = orbit_partition(graph, identifier, Motif(("has_target",), 1))
        assert all(len(orbit) >= 1 for orbit in partition)

    def test_a_zero_order_motif_is_rejected(self, graph) -> None:
        with pytest.raises(ValueError):
            enumerate_motifs(graph, 0)


class TestSampling:
    def test_masked_edges_draw_negatives_off_the_positive_tail(self, graph) -> None:
        batch = sample_masked_edges(graph, 6, negatives=3, seed=11)
        for row in batch.as_rows():
            head, _, tail, negatives = row
            assert tail not in negatives
            assert head not in negatives

    def test_the_batch_is_deterministic_under_a_seed(self, graph) -> None:
        left = sample_masked_edges(graph, 5, negatives=2, seed=3)
        right = sample_masked_edges(graph, 5, negatives=2, seed=3)
        assert np.array_equal(left.head, right.head)
        assert np.array_equal(left.negative, right.negative)

    def test_a_zero_count_is_rejected(self, graph) -> None:
        with pytest.raises(ValueError):
            sample_masked_edges(graph, 0, negatives=2, seed=1)

    def test_an_unknown_relation_is_reported(self) -> None:
        from oaer.catalogue.sampling import malformed_relations

        assert malformed_relations(["has_target", "nope"]) == ("nope",)


class TestSources:
    def test_every_resource_carries_a_licence_and_a_probe(self) -> None:
        for source in AUXILIARY_SOURCES:
            assert source.licence
            assert source.url.startswith("https://")
            assert source.probe

    def test_the_candidate_sources_are_in_the_resource_card(self) -> None:
        names = set(candidate_source_envelope())
        assert names == set(CANDIDATE_SOURCES)

    def test_every_public_cohort_carries_its_modalities(self) -> None:
        for cohort in PUBLIC_COHORTS:
            assert cohort.modalities
            assert cohort.access

    def test_the_pathology_only_supplementary_row_is_marked(self) -> None:
        probe = next(
            cohort for cohort in PUBLIC_COHORTS if cohort.name.startswith("Microsatellite probe")
        )
        assert probe.c_index is None
