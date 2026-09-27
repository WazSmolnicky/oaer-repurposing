"""Higher-order motif algebra for the relational encoder.

A motif is an ordered pattern of relations. The encoder reads the graph through
the operators of a small motif set rather than through single relations, which is
what makes the representation higher order: an order-2 motif is a typed path of
two relations, order-3 a typed path of three. The motif signature of a node is
the count of walks that realise each pattern from it, and the motif Laplacian is
the normalised operator the encoder propagates along.

Ref: Sec. 2.4 (relational encoder), Eq. (3)-(4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.catalogue.graph import RELATION_TYPES, KnowledgeGraph


@dataclass(frozen=True)
class Motif:
    """One ordered relation pattern."""

    relations: tuple[str, ...]
    order: int

    @property
    def name(self) -> str:
        return ">".join(self.relations)


# The motif set the encoder reads, in a fixed order so the basis is reproducible.
MOTIF_SET: Final[tuple[Motif, ...]] = (
    Motif(("has_target",), 1),
    Motif(("has_indication",), 1),
    Motif(("affects_pathway",), 1),
    Motif(("associated_with_disease",), 1),
    Motif(("anchor_of_type",), 1),
    Motif(("has_target", "pathway_membership"), 2),
    Motif(("has_target", "has_target"), 2),
    Motif(("affects_pathway", "co_occurs_with"), 2),
    Motif(("associated_with_disease", "has_indication"), 2),
    Motif(("has_target", "pathway_membership", "co_occurs_with"), 3),
    Motif(("has_target", "has_target", "pathway_membership"), 3),
    Motif(("affects_pathway", "co_occurs_with", "pathway_membership"), 3),
)


def enumerate_motifs(
    graph: KnowledgeGraph,
    order: int,
    *,
    limit: int | None = None,
) -> tuple[Motif, ...]:
    """Every relation pattern of ``order`` that occurs at least once.

    Patterns are visited in lexicographic relation order so the enumeration does
    not depend on edge insertion order.
    """

    if order < 1:
        raise ValueError("a motif order must be at least one")
    present = _present_relations(graph)
    patterns: list[Motif] = []
    for combination in product(sorted(present), repeat=order):
        motif = Motif(tuple(combination), order)
        if np.count_nonzero(motif_operator(graph, motif)) > 0:
            patterns.append(motif)
        if limit is not None and len(patterns) >= limit:
            break
    return tuple(patterns)


def _present_relations(graph: KnowledgeGraph) -> set[str]:
    return {edge.relation for edge in graph.edges}


def relation_operator(graph: KnowledgeGraph, relation: str) -> npt.NDArray[np.float64]:
    """The 0/1 adjacency of one relation, over the graph's node order."""

    if relation not in RELATION_TYPES:
        raise KeyError(f"{relation!r} is not one of the seven typed relations")
    lookup = graph.index()
    size = len(lookup)
    operator = np.zeros((size, size), dtype=np.float64)
    for edge in graph.edges:
        if edge.relation == relation:
            operator[lookup[edge.head], lookup[edge.tail]] = 1.0
    return operator


def motif_operator(graph: KnowledgeGraph, motif: Motif | Sequence[str]) -> npt.NDArray[np.float64]:
    """The operator of a motif, as the product of its relation operators.

    The product is left-to-right on the pattern, so a walk of the motif from
    ``u`` lands on the row of the final relation's target.
    """

    relations = motif.relations if isinstance(motif, Motif) else tuple(motif)
    if not relations:
        raise ValueError("a motif must carry at least one relation")
    operator = relation_operator(graph, relations[0])
    for relation in relations[1:]:
        operator = operator @ relation_operator(graph, relation)
    return operator


def motif_laplacian(graph: KnowledgeGraph, motif: Motif | Sequence[str]) -> npt.NDArray[np.float64]:
    """The symmetric normalised motif Laplacian ``I - D^-1/2 A D^-1/2``."""

    adjacency = motif_operator(graph, motif)
    degree = adjacency.sum(axis=1)
    inverse_root = np.zeros_like(degree)
    np.divide(1.0, np.sqrt(degree), out=inverse_root, where=degree > 0.0)
    normalised = (inverse_root[:, None] * adjacency) * inverse_root[None, :]
    laplacian: npt.NDArray[np.float64] = np.asarray(
        np.eye(adjacency.shape[0], dtype=np.float64) - normalised, dtype=np.float64
    )
    return laplacian


def motif_signature(graph: KnowledgeGraph, node: str) -> npt.NDArray[np.float64]:
    """The walk counts of every motif of ``MOTIF_SET`` starting at ``node``."""

    position = graph.index()[node]
    counts = np.zeros(len(MOTIF_SET), dtype=np.float64)
    for index, motif in enumerate(MOTIF_SET):
        operator = motif_operator(graph, motif)
        counts[index] = float(operator[position].sum())
    return counts


def motif_bank(graph: KnowledgeGraph) -> npt.NDArray[np.float64]:
    """Stack the motif signatures of every node into a dense bank."""

    order = graph.index()
    bank = np.zeros((len(order), len(MOTIF_SET)), dtype=np.float64)
    for node, position in order.items():
        bank[position] = motif_signature(graph, node)
    return bank


def motif_frequencies(graph: KnowledgeGraph) -> Mapping[str, float]:
    """The share of motif-realising walks each pattern carries."""

    totals = np.zeros(len(MOTIF_SET), dtype=np.float64)
    for index, motif in enumerate(MOTIF_SET):
        totals[index] = float(motif_operator(graph, motif).sum())
    grand = float(totals.sum())
    if grand <= 0.0:
        return {motif.name: 0.0 for motif in MOTIF_SET}
    return {motif.name: float(totals[index] / grand) for index, motif in enumerate(MOTIF_SET)}


def motif_spectrum(graph: KnowledgeGraph, motif: Motif | Sequence[str]) -> npt.NDArray[np.float64]:
    """The eigenvalues of a motif Laplacian, ascending, for the basis read-out."""

    spectrum: npt.NDArray[np.float64] = np.asarray(
        np.linalg.eigvalsh(motif_laplacian(graph, motif)), dtype=np.float64
    )
    return spectrum


def orbit_partition(
    graph: KnowledgeGraph, identifier: str, motif: Motif | Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """The orbit partition a motif induces on a candidate's neighbourhood.

    Two neighbours share an orbit when their motif signatures onto the candidate
    agree. The partition is the object Proposition 3 reads: a relational encoder
    that sees only structure has the same score across an orbit, so the regret
    floor is set by the partition rather than by the cohort size.
    """

    position = graph.index()
    operator = motif_operator(graph, motif)
    anchors = operator[position[identifier]]
    buckets: dict[tuple[float, ...], list[str]] = {}
    for node, value in zip(graph.nodes, anchors, strict=True):
        if value <= 0.0:
            continue
        bucket = buckets.setdefault((float(value),), [])
        bucket.append(node)
    return tuple(tuple(sorted(nodes)) for _, nodes in sorted(buckets.items()))
