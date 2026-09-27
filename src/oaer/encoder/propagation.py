"""Message-passing primitives and the basis decomposition.

A relation operator is a learned mixture over a small set of basis matrices, so
the number of parameters grows with the basis count rather than with the relation
count and a relation can be scored from its basis coordinates alone. Propagation
is a scatter-add of transformed messages, with the relation weight coming from
the decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MotifBasis:
    """One element of the motif basis, with the order it was built at."""

    name: str
    order: int
    basis_index: int


DEFAULT_MOTIF_BASIS: Final[tuple[MotifBasis, ...]] = (
    MotifBasis("single_relation", 1, 0),
    MotifBasis("typed_path_2", 2, 1),
    MotifBasis("typed_path_3", 3, 2),
    MotifBasis("same_relation_square", 2, 3),
)


class BasisDecomposition(nn.Module):
    """One projection per basis, mixed by a relation-specific weight vector."""

    def __init__(self, num_relations: int, num_basis: int, width: int) -> None:
        super().__init__()
        if num_relations < 1 or num_basis < 1 or width < 1:
            raise ValueError("the decomposition dimensions must be at least one")
        self.num_relations = num_relations
        self.num_basis = num_basis
        self.width = width
        self.basis_weights = nn.Parameter(torch.empty(num_relations, num_basis))
        self.basis_matrices = nn.Parameter(torch.empty(num_basis, width, width))
        self.layernorm = nn.LayerNorm(width, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.basis_weights, mean=0.0, std=0.1)
        for index in range(self.num_basis):
            nn.init.orthogonal_(self.basis_matrices[index])

    def relation_matrix(self, relation: Tensor) -> Tensor:
        """The relation-specific operator, as a weighted sum of the bases."""

        coefficients = torch.softmax(self.basis_weights[relation], dim=-1)
        return torch.einsum("rb,bij->rij", coefficients, self.basis_matrices)

    def forward(self, message: Tensor, relation: Tensor) -> Tensor:
        matrix = self.relation_matrix(relation)
        return torch.bmm(matrix, message.unsqueeze(-1)).squeeze(-1)


def propagate(
    state: Tensor,
    head: Tensor,
    relation: Tensor,
    tail: Tensor,
    basis: BasisDecomposition | None = None,
) -> Tensor:
    """Scatter transformed messages from tail to head along typed edges.

    The head's own state is carried through the output so a node with no incoming
    edge keeps its representation rather than collapsing to zero.
    """

    width = state.shape[1]
    if basis is None:
        basis = BasisDecomposition(
            int(relation.max().item()) + 1 if relation.numel() else 1, 1, width
        )
        basis.to(state.device)
    transformed = basis(state[tail], relation)
    output = torch.zeros_like(state)
    index = head.unsqueeze(-1).expand_as(transformed)
    output.scatter_add_(0, index, transformed)
    counts = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
    counts.scatter_add_(0, head, torch.ones_like(head, dtype=state.dtype))
    normaliser = counts.clamp(min=1.0).unsqueeze(-1)
    return output / normaliser.sqrt()


def edge_scores(
    state: Tensor,
    head: Tensor,
    relation: Tensor,
    tail: Tensor,
    basis: BasisDecomposition,
) -> Tensor:
    """Score each edge by matching the transformed head state against the tail."""

    transformed = basis(state[head], relation)
    edge_score: Tensor = (transformed * state[tail]).sum(dim=-1) / (state.shape[1] ** 0.5)
    return edge_score


def degree_tensor(num_entities: int, head: Tensor, tail: Tensor) -> Tensor:
    """Undirected degree per entity, as a float tensor."""

    counts = torch.zeros(num_entities, dtype=torch.float32)
    counts.scatter_add_(0, head, torch.ones_like(head, dtype=torch.float32))
    counts.scatter_add_(0, tail, torch.ones_like(tail, dtype=torch.float32))
    return counts


def subgraph_mask(keep: Tensor) -> Tensor:
    """The retained edge indices of a subgraph selection."""

    selected: Tensor = (
        torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if bool(keep.any())
        else torch.empty(0, dtype=torch.long)
    )
    return selected


def neighbourhood_statistics(
    num_entities: int,
    head: Tensor,
    tail: Tensor,
) -> dict[str, float]:
    """Degree statistics used by the borrowing-strength read-out."""

    degrees = degree_tensor(num_entities, head, tail)
    return {
        "mean_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "isolated": float((degrees == 0).sum()),
    }
