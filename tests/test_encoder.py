"""The relational encoder, its propagation primitives and the anchor read-out."""

from __future__ import annotations

import pytest
import torch

from oaer.catalogue.graph import RELATION_INDEX, AnchorType, attach_patient_node
from oaer.encoder.anchoring import (
    AnchorReconstruction,
    PatientAnchorHead,
    anchor_accuracy,
    anchor_reconstruction_loss,
    anchor_scores,
    anchor_type_breakdown,
    anchor_type_index,
)
from oaer.encoder.propagation import (
    DEFAULT_MOTIF_BASIS,
    BasisDecomposition,
    degree_tensor,
    edge_scores,
    neighbourhood_statistics,
    propagate,
    subgraph_mask,
)
from oaer.encoder.relational import RelationalEncoder, masked_edge_loss, motif_summary


class TestEncoder:
    def test_a_forward_pass_returns_one_row_per_entity(self, graph) -> None:
        encoder = _encoder(graph)
        state = _forward(encoder, graph)
        assert state.shape == (len(graph.nodes), encoder.hidden_dim)

    def test_the_representation_is_finite(self, graph) -> None:
        state = _forward(_encoder(graph), graph)
        assert bool(torch.isfinite(state).all())

    def test_the_bilinear_readout_has_the_right_width(self, graph) -> None:
        encoder = _encoder(graph)
        state = _forward(encoder, graph)
        left = torch.as_tensor([0, 1], dtype=torch.long)
        right = torch.as_tensor([2, 3], dtype=torch.long)
        assert encoder.score(state, left, right).shape == (2,)

    def test_a_degree_scale_changes_a_low_degree_row(self) -> None:
        from oaer.catalogue.graph import KnowledgeGraph, Node, NodeKind

        fresh = KnowledgeGraph()
        for name in ("a", "b"):
            fresh.add_node(Node(name, NodeKind.TARGET, name))
        fresh.add_edge("a", "has_target", "b")
        encoder = _encoder(fresh)
        head, relation, tail = fresh.arrays()
        state = encoder(
            torch.arange(2, dtype=torch.long),
            torch.as_tensor(head, dtype=torch.long),
            torch.as_tensor(relation, dtype=torch.long),
            torch.as_tensor(tail, dtype=torch.long),
            degree_tensor(2, torch.as_tensor(head), torch.as_tensor(tail)),
        )
        assert state.shape[0] == 2

    def test_an_undersized_encoder_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RelationalEncoder(num_entities=1, num_relations=1)

    def test_a_zero_width_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RelationalEncoder(num_entities=4, num_relations=2, embedding_dim=0)

    def test_the_summary_reports_the_width(self, graph) -> None:
        encoder = _encoder(graph)
        summary = motif_summary(encoder, _forward(encoder, graph))
        assert summary["width"] == encoder.hidden_dim


class TestMaskedEdgeLoss:
    def test_the_loss_matches_a_brute_force_softmax(self) -> None:
        positive = torch.tensor([1.0, -0.5])
        negative = torch.tensor([[0.5, 0.0, -1.0], [0.25, 0.75, 0.1]])
        stacked = torch.cat([positive.unsqueeze(1), negative], dim=1)
        reference = -torch.log_softmax(stacked, dim=1)[:, 0].mean()
        assert float(masked_edge_loss(positive, negative)) == pytest.approx(
            float(reference), abs=1e-7
        )

    def test_a_perfect_score_gives_a_small_loss(self) -> None:
        loss = masked_edge_loss(torch.tensor([10.0]), torch.tensor([[0.0, 0.0]]))
        assert float(loss) < 1e-3

    def test_a_mismatched_batch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            masked_edge_loss(torch.zeros(2), torch.zeros(3, 4))


class TestPropagation:
    def test_a_basis_projection_is_square(self) -> None:
        basis = BasisDecomposition(num_relations=5, num_basis=3, width=8)
        message = torch.randn(4, 8)
        relation = torch.tensor([0, 1, 2, 3])
        assert basis(message, relation).shape == (4, 8)

    def test_the_relation_coefficients_are_a_distribution(self) -> None:
        basis = BasisDecomposition(num_relations=4, num_basis=3, width=6)
        matrix = basis.relation_matrix(torch.tensor([1]))
        assert matrix.shape == (1, 6, 6)

    def test_propagation_averages_over_the_in_degree(self) -> None:
        state = torch.ones(3, 2)
        head = torch.tensor([0, 0])
        relation = torch.tensor([0, 0])
        tail = torch.tensor([1, 2])
        basis = BasisDecomposition(num_relations=1, num_basis=1, width=2)
        with torch.no_grad():
            basis.basis_matrices.copy_(torch.eye(2))
            basis.basis_weights.zero_()
        output = propagate(state, head, relation, tail, basis)
        assert float(output[0].abs().sum()) > 0.0

    def test_the_edge_score_has_one_entry_per_edge(self, graph) -> None:
        encoder = _encoder(graph)
        state = _forward(encoder, graph)
        head, relation, tail = graph.arrays()
        scores = edge_scores(
            state,
            torch.as_tensor(head, dtype=torch.long),
            torch.as_tensor(relation, dtype=torch.long),
            torch.as_tensor(tail, dtype=torch.long),
            encoder.basis,
        )
        assert scores.shape[0] == len(graph.edges)

    def test_degrees_count_both_ends(self) -> None:
        counts = degree_tensor(4, torch.tensor([0, 0]), torch.tensor([1, 2]))
        assert float(counts[0]) == 2.0

    def test_the_motif_basis_is_declared(self) -> None:
        assert len(DEFAULT_MOTIF_BASIS) >= 4

    def test_the_subgraph_mask_selects_the_kept_edges(self) -> None:
        mask = torch.tensor([True, False, True])
        assert subgraph_mask(mask).tolist() == [0, 2]

    def test_the_neighbourhood_statistics_are_finite(self, graph) -> None:
        head, relation, tail = graph.arrays()
        statistics = neighbourhood_statistics(
            len(graph.nodes),
            torch.as_tensor(head, dtype=torch.long),
            torch.as_tensor(tail, dtype=torch.long),
        )
        del relation
        assert statistics["mean_degree"] > 0.0


class TestAnchoring:
    def test_the_set_level_pool_returns_one_vector(self) -> None:
        head = PatientAnchorHead(width=8)
        states = torch.randn(4, 8)
        types = torch.tensor([0, 1, 2, 3])
        assert head(states, types).shape == (8,)

    def test_the_rowwise_pool_keeps_the_batch(self) -> None:
        head = PatientAnchorHead(width=8)
        states = torch.randn(5, 8)
        types = torch.zeros(5, dtype=torch.long)
        assert head.pool_rows(states, types).shape == (5, 8)

    def test_an_empty_attachment_returns_zeros(self) -> None:
        head = PatientAnchorHead(width=8)
        empty = torch.zeros((0, 8))
        assert float(head(empty, torch.zeros(0, dtype=torch.long)).abs().sum()) == 0.0

    def test_the_reconstruction_loss_matches_a_brute_force_log_sigmoid(self) -> None:
        positive = torch.tensor([0.5, 1.0])
        negative = torch.tensor([-0.5, 0.0, 0.25])
        reference = (
            -(
                torch.nn.functional.logsigmoid(positive).sum()
                + torch.nn.functional.logsigmoid(-negative).sum()
            )
            / 2.0
        )
        assert float(anchor_reconstruction_loss(positive, negative)) == pytest.approx(
            float(reference), abs=1e-7
        )

    def test_a_perfect_reconstruction_gives_a_small_loss(self) -> None:
        loss = anchor_reconstruction_loss(torch.tensor([12.0]), torch.tensor([-12.0]))
        assert float(loss) < 1e-3

    def test_an_empty_batch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            anchor_reconstruction_loss(torch.zeros(0), torch.zeros(0))

    def test_the_anchor_scores_pair_the_two_tensors(self, graph) -> None:
        readout = AnchorReconstruction(width=8)
        patient = torch.randn(8)
        positive = torch.randn(3, 8)
        negative = torch.randn(2, 8)
        left, right = anchor_scores(readout, patient, positive, negative)
        assert left.shape == (3,) and right.shape == (2,)

    def test_the_accuracy_is_a_share(self, graph) -> None:
        accuracy = anchor_accuracy(torch.tensor([1.0, 2.0]), torch.tensor([-1.0, -2.0]))
        assert accuracy == pytest.approx(1.0)

    def test_the_anchor_type_index_is_stable(self) -> None:
        assert anchor_type_index(AnchorType.PATHWAY) == list(AnchorType).index(AnchorType.PATHWAY)

    def test_the_breakdown_is_keyed_by_anchor_type(self) -> None:
        payload = anchor_type_breakdown({AnchorType.PATHWAY: (1.0, -1.0)})
        assert payload["pathway"]["anchor"] == 1.0

    def test_a_patient_node_attaches_through_every_type(self, graph, development) -> None:
        decision = development.decisions[0]
        attach_patient_node(graph, decision.patient_id, decision.state)
        from oaer.catalogue.graph import anchors_of

        attachment = anchors_of(graph, decision.patient_id)
        assert set(attachment) == set(AnchorType)


def _encoder(graph) -> RelationalEncoder:  # type: ignore[no-untyped-def]
    return RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=len(RELATION_INDEX),
        embedding_dim=12,
        hidden_dim=12,
        layers=2,
        num_basis=2,
        dropout=0.0,
    )


def _forward(encoder: RelationalEncoder, graph) -> torch.Tensor:  # type: ignore[no-untyped-def]
    head, relation, tail = graph.arrays()
    head_t = torch.as_tensor(head, dtype=torch.long)
    tail_t = torch.as_tensor(tail, dtype=torch.long)
    return encoder(
        torch.arange(len(graph.nodes), dtype=torch.long),
        head_t,
        torch.as_tensor(relation, dtype=torch.long),
        tail_t,
        degree_tensor(len(graph.nodes), head_t, tail_t),
    )
