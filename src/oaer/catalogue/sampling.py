"""Corruption sampling for the two graph-side pretraining stages.

Stage 1 hides relation edges of the knowledge graph and scores the true tail
against a set of negative tails. Stage 2 hides a collection of a patient node's
typed anchors and reconstructs them against non-anchors of the same type. Both
samplers are deterministic under a seed and never read an outcome.

Ref: Sec. 2.6 (Stage 1, Stage 2), Eq. (3)-(4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from oaer.catalogue.graph import (
    ANCHOR_RELATION,
    RELATION_INDEX,
    RELATION_TYPES,
    AnchorType,
    KnowledgeGraph,
    NodeKind,
    anchors_of,
    non_anchors,
)
from oaer.support.seeding import spawn_generator


@dataclass(frozen=True)
class MaskedEdgeBatch:
    """Positive masked edges and, per edge, a set of negative tails."""

    head: npt.NDArray[np.int64]
    relation: npt.NDArray[np.int64]
    tail: npt.NDArray[np.int64]
    negative: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.head.shape[0])

    def as_rows(self) -> list[tuple[int, int, int, tuple[int, ...]]]:
        return [
            (
                int(self.head[index]),
                int(self.relation[index]),
                int(self.tail[index]),
                tuple(int(value) for value in self.negative[index]),
            )
            for index in range(len(self))
        ]


@dataclass(frozen=True)
class AnchorCorruption:
    """One masked patient anchor and the non-anchors it is reconstructed against."""

    patient: str
    anchor_type: AnchorType
    relation: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]


def sample_masked_edges(
    graph: KnowledgeGraph,
    count: int,
    *,
    negatives: int = 12,
    seed: int = 20260831,
    relations: Sequence[str] | None = None,
) -> MaskedEdgeBatch:
    """Mask ``count`` relation edges and draw negative tails for each.

    A negative tail is an entity that does not already sit at that relation from
    that head, which is the ``V-(u, rho)`` set of Eq. (3).
    """

    if count <= 0:
        raise ValueError("a masked-edge batch must carry at least one edge")
    if negatives <= 0:
        raise ValueError("at least one negative tail is required per masked edge")
    allowed = set(relations) if relations is not None else None
    eligible = [
        edge
        for edge in graph.edges
        if (allowed is None or edge.relation in allowed)
        and graph.nodes[edge.tail].kind is not NodeKind.PATIENT
        and graph.nodes[edge.head].kind is not NodeKind.PATIENT
    ]
    if not eligible:
        raise ValueError("the graph carries no relation edge to mask")
    stream = spawn_generator(seed, "masked-edges")
    lookup = graph.index()
    node_order = list(graph.nodes)
    chosen = stream.choice(
        len(eligible), size=min(count, len(eligible)), replace=False, shuffle=False
    )
    heads: list[int] = []
    relation_ids: list[int] = []
    tails: list[int] = []
    negative_rows: list[tuple[int, ...]] = []
    for index in np.atleast_1d(chosen):
        edge = eligible[int(index)]
        occupied = set(graph.neighbours(edge.head, edge.relation))
        candidates = [
            node
            for node in node_order
            if node not in occupied
            and node != edge.head
            and graph.nodes[node].kind is not NodeKind.PATIENT
        ]
        if len(candidates) < negatives:
            continue
        drawn = stream.choice(len(candidates), size=negatives, replace=False, shuffle=False)
        heads.append(lookup[edge.head])
        relation_ids.append(RELATION_INDEX[edge.relation])
        tails.append(lookup[edge.tail])
        negative_rows.append(tuple(lookup[candidates[int(value)]] for value in drawn))
    if not heads:
        raise ValueError("no masked edge could be drawn with the requested negative count")
    return MaskedEdgeBatch(
        head=np.asarray(heads, dtype=np.int64),
        relation=np.asarray(relation_ids, dtype=np.int64),
        tail=np.asarray(tails, dtype=np.int64),
        negative=np.asarray(negative_rows, dtype=np.int64),
    )


def corrupt_anchors(
    graph: KnowledgeGraph,
    patients: Sequence[str],
    *,
    negatives: int = 5,
    seed: int = 20260831,
    anchor_types: Sequence[AnchorType] | None = None,
) -> tuple[AnchorCorruption, ...]:
    """Mask each patient's anchors of the named types against same-type non-anchors."""

    if negatives <= 0:
        raise ValueError("at least one non-anchor is required per masked anchor")
    kinds = tuple(anchor_types) if anchor_types is not None else tuple(AnchorType)
    stream = spawn_generator(seed, "anchor-corruption")
    records: list[AnchorCorruption] = []
    for patient in patients:
        attachment = anchors_of(graph, patient)
        for anchor_type in kinds:
            positive = attachment.get(anchor_type, ())
            if not positive:
                continue
            choices = [
                node
                for node in non_anchors(graph, anchor_type, len(graph.nodes))
                if node not in positive
            ]
            if len(choices) < negatives:
                continue
            drawn = stream.choice(len(choices), size=negatives, replace=False, shuffle=False)
            records.append(
                AnchorCorruption(
                    patient=patient,
                    anchor_type=anchor_type,
                    relation=ANCHOR_RELATION[anchor_type],
                    positive=positive,
                    negative=tuple(choices[int(value)] for value in drawn),
                )
            )
    return tuple(records)


def malformed_relations(relations: Sequence[str]) -> tuple[str, ...]:
    """Relations in ``relations`` that are not one of the seven typed relations."""

    return tuple(relation for relation in relations if relation not in RELATION_TYPES)
