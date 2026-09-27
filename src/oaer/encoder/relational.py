"""Relational encoder: the graph-side representation and its two pretraining stages.

The encoder is a higher-order motif relational encoder. Its operators are the
motif Laplacians of a fixed motif set rather than single relations, and its
relation scoring is a basis decomposition: one learned projection per basis,
mixed by a relation-specific weight, which is what lets a relation the encoder
never trained on still be scored. Node features are retained, because the
foundation-model family the article names carries no attributes of its own and a
patient anchor has to be readable.

Stage 1 hides relation edges and scores the true tail against negative tails;
Stage 2 hides a patient node's typed anchors and reconstructs them against
non-anchors of the same type. Neither stage sees an outcome.

Ref: Sec. 2.4 (relational encoder), Sec. 2.6 (Stage 1, Stage 2), Eq. (3)-(4).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from oaer.encoder.anchoring import (
    AnchorReconstruction,
    PatientAnchorHead,
    anchor_reconstruction_loss,
    anchor_scores,
)
from oaer.encoder.propagation import (
    BasisDecomposition,
    MotifBasis,
    edge_scores,
    propagate,
)

__all__ = [
    "AnchorReconstruction",
    "BasisDecomposition",
    "MotifBasis",
    "PatientAnchorHead",
    "RelationalEncoder",
    "anchor_reconstruction_loss",
    "anchor_scores",
    "edge_scores",
    "masked_edge_loss",
    "propagate",
]


class RelationalEncoder(nn.Module):
    """Entity embeddings propagated along a learned motif basis.

    The forward pass returns one representation per node. Scoring a candidate
    against a state is then a bilinear read-out between the candidate's node
    representation and the patient node's, which is what the ranking head
    consumes.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        layers: int = 2,
        num_basis: int = 4,
        dropout: float = 0.1,
        degree_scale: float = 8.0,
    ) -> None:
        super().__init__()
        if num_entities < 2 or num_relations < 1:
            raise ValueError("the encoder needs at least two entities and one relation")
        if embedding_dim < 1 or hidden_dim < 1 or layers < 1 or num_basis < 1:
            raise ValueError("every dimensional argument must be at least one")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.degree_scale = float(degree_scale)
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.node_scale = nn.Embedding(num_entities, 1)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.basis = BasisDecomposition(num_relations, num_basis, hidden_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.readout = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.zeros_(self.node_scale.weight)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def node_features(self, entity: Tensor, degree: Tensor) -> Tensor:
        """The initial node state: the learned embedding scaled by log-degree.

        The scale is what carries the borrowing-strength behaviour of a graph
        prior: a low-degree node borrows more from its neighbourhood than a
        high-degree one does.
        """

        scale = torch.tanh(self.node_scale(entity).squeeze(-1))
        weight = torch.ones_like(scale) + scale * degree.log1p() / self.degree_scale
        state: Tensor = self.entity(entity) * weight.unsqueeze(-1)
        return state

    def forward(
        self,
        entity: Tensor,
        head: Tensor,
        relation: Tensor,
        tail: Tensor,
        degree: Tensor,
    ) -> Tensor:
        state = self.input_projection(self.node_features(entity, degree))
        for index in range(self.layers):
            message = propagate(state, head, relation, tail, self.basis)
            state = self.layer_norms[index](state + self.dropout(message))
            state = self.activation(state)
        readout: Tensor = self.readout(state)
        return readout

    def representation(self, state: Tensor, index: Tensor) -> Tensor:
        return state[index]

    def score(self, state: Tensor, candidate: Tensor, patient: Tensor) -> Tensor:
        """A bilinear read-out between a candidate node and a patient node."""

        left = state[candidate]
        right = state[patient]
        score: Tensor = (left * right).sum(dim=-1) / (self.hidden_dim**0.5)
        return score


def masked_edge_loss(
    scores_positive: Tensor,
    scores_negative: Tensor,
) -> Tensor:
    """Eq. (3): the softmax-over-negatives masked-edge objective.

    ``scores_negative`` carries one row of ``|V-(u, rho)|`` alternatives per
    masked edge, so the log-softmax is taken along that axis.
    """

    if scores_positive.dim() != 1 or scores_negative.dim() != 2:
        raise ValueError("the positive scores must be a vector and the negatives a matrix")
    if scores_negative.shape[0] != scores_positive.shape[0]:
        raise ValueError("every masked edge needs its own negative row")
    stacked = torch.cat([scores_positive.unsqueeze(1), scores_negative], dim=1)
    log_probability = torch.log_softmax(stacked, dim=1)[:, 0]
    return -log_probability.mean()


def motif_summary(encoder: RelationalEncoder, state: Tensor) -> dict[str, float]:
    """A read-only summary of one encoded state, used by the audit."""

    with torch.no_grad():
        return {
            "rows": float(state.shape[0]),
            "width": float(state.shape[1]),
            "mean_norm": float(state.norm(dim=-1).mean()),
            "max_norm": float(state.norm(dim=-1).max()),
            "relation_bases": float(encoder.basis.num_basis),
        }
