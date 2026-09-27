"""The typed knowledge graph.

Nodes are candidates, targets, pathways, diseases and record concepts. Edges are
typed by relation, and the anchor types are the four the patient node attaches
through. The graph is a plain adjacency structure with a fixed node order, so the
encoder can read it as index arrays and every census is reproducible.

Ref: Sec. 2.2 (knowledge-graph construction), Sec. 2.4 (relational encoder).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.support.seeding import spawn_generator
from oaer.support.types import Candidate, DecisionState


class NodeKind(str, Enum):
    CANDIDATE = "candidate"
    TARGET = "target"
    PATHWAY = "pathway"
    DISEASE = "disease"
    RECORD_CONCEPT = "record_concept"
    PATIENT = "patient"


class AnchorType(str, Enum):
    """The four typed anchors a patient node attaches through."""

    MOLECULAR_ALTERATION = "molecular_alteration"
    PATHWAY = "pathway"
    DISEASE = "disease"
    RECORD_CONCEPT = "record_concept"


# Seven relations, matching ``catalogue.relations``. The first three are the
# candidate side, the last four are the anchor side the patient node attaches to.
RELATION_TYPES: Final[tuple[str, ...]] = (
    "has_target",
    "has_indication",
    "affects_pathway",
    "associated_with_disease",
    "pathway_membership",
    "anchor_of_type",
    "co_occurs_with",
)

RELATION_INDEX: Final[Mapping[str, int]] = {
    name: position for position, name in enumerate(RELATION_TYPES)
}

ANCHOR_RELATION: Final[Mapping[AnchorType, str]] = {
    AnchorType.MOLECULAR_ALTERATION: "anchor_of_type",
    AnchorType.PATHWAY: "affects_pathway",
    AnchorType.DISEASE: "associated_with_disease",
    AnchorType.RECORD_CONCEPT: "anchor_of_type",
}

CONCEPT_LAYER: Final[tuple[tuple[str, NodeKind, str], ...]] = (
    ("TG:PTGS2", NodeKind.TARGET, "PTGS2"),
    ("TG:PTGS1", NodeKind.TARGET, "PTGS1"),
    ("TG:HMGCR", NodeKind.TARGET, "HMGCR"),
    ("TG:ACE", NodeKind.TARGET, "ACE"),
    ("TG:AGTR1", NodeKind.TARGET, "AGTR1"),
    ("TG:SLC12A3", NodeKind.TARGET, "SLC12A3"),
    ("TG:PPARG", NodeKind.TARGET, "PPARG"),
    ("TG:INSR", NodeKind.TARGET, "INSR"),
    ("TG:ADRB1", NodeKind.TARGET, "ADRB1"),
    ("TG:ADRB2", NodeKind.TARGET, "ADRB2"),
    ("TG:CACNA1C", NodeKind.TARGET, "CACNA1C"),
    ("TG:P2RY12", NodeKind.TARGET, "P2RY12"),
    ("TG:ATP4A", NodeKind.TARGET, "ATP4A"),
    ("TG:ATP4B", NodeKind.TARGET, "ATP4B"),
    ("TG:HRH2", NodeKind.TARGET, "HRH2"),
    ("TG:DPP4", NodeKind.TARGET, "DPP4"),
    ("TG:GLP1R", NodeKind.TARGET, "GLP1R"),
    ("TG:NFKB1", NodeKind.TARGET, "NFKB1"),
    ("TG:MTOR", NodeKind.TARGET, "MTOR"),
    ("TG:PIK3CA", NodeKind.TARGET, "PIK3CA"),
    ("PW:00", NodeKind.PATHWAY, "cyclo-oxygenase-2 signalling"),
    ("PW:01", NodeKind.PATHWAY, "PI3K-pathway"),
    ("PW:02", NodeKind.PATHWAY, "AMPK signalling"),
    ("PW:03", NodeKind.PATHWAY, "renin-angiotensin system"),
    ("PW:04", NodeKind.PATHWAY, "adrenergic signalling"),
    ("PW:05", NodeKind.PATHWAY, "purinergic signalling"),
    ("PW:06", NodeKind.PATHWAY, "gastric acid secretion"),
    ("PW:07", NodeKind.PATHWAY, "insulin signalling"),
    ("DS:00", NodeKind.DISEASE, "colorectal adenocarcinoma"),
    ("DS:01", NodeKind.DISEASE, "type 2 diabetes mellitus"),
    ("DS:02", NodeKind.DISEASE, "hypertension"),
    ("DS:03", NodeKind.DISEASE, "gastro-oesophageal reflux disease"),
    ("DS:04", NodeKind.DISEASE, "microsatellite instability"),
    ("RC:00", NodeKind.RECORD_CONCEPT, "PI3K-altered pathway"),
    ("RC:01", NodeKind.RECORD_CONCEPT, "RAS wild-type"),
    ("RC:02", NodeKind.RECORD_CONCEPT, "BRAF V600E"),
    ("RC:03", NodeKind.RECORD_CONCEPT, "mismatch-repair deficient"),
    ("RC:04", NodeKind.RECORD_CONCEPT, "liver-limited metastases"),
    ("RC:05", NodeKind.RECORD_CONCEPT, "right-sided primary"),
)

TARGET_LOOKUP: Final[Mapping[str, str]] = {
    label: identifier for identifier, kind, label in CONCEPT_LAYER if kind is NodeKind.TARGET
}
PATHWAY_IDS: Final[tuple[str, ...]] = tuple(
    identifier for identifier, kind, _ in CONCEPT_LAYER if kind is NodeKind.PATHWAY
)
DISEASE_LOOKUP: Final[Mapping[str, str]] = {
    label: identifier for identifier, kind, label in CONCEPT_LAYER if kind is NodeKind.DISEASE
}


@dataclass(frozen=True)
class Node:
    identifier: str
    kind: NodeKind
    label: str
    features: tuple[float, ...] = ()


@dataclass(frozen=True)
class Edge:
    head: str
    relation: str
    tail: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


@dataclass
class KnowledgeGraph:
    """A typed graph with a fixed node order and a relation-indexed adjacency."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _out: dict[tuple[str, str], list[str]] = field(default_factory=dict, repr=False)
    _in: dict[tuple[str, str], list[str]] = field(default_factory=dict, repr=False)
    _keys: set[tuple[str, str, str]] = field(default_factory=set, repr=False)
    _order: list[str] = field(default_factory=list, repr=False)

    def add_node(self, node: Node) -> None:
        if node.identifier in self.nodes:
            raise ValueError(f"node {node.identifier!r} is already in the graph")
        self.nodes[node.identifier] = node
        self._order.append(node.identifier)

    def has_node(self, identifier: str) -> bool:
        return identifier in self.nodes

    def has_edge(self, head: str, relation: str, tail: str) -> bool:
        return (head, relation, tail) in self._keys

    def add_edge(self, head: str, relation: str, tail: str) -> bool:
        """Add a typed edge; a duplicate is ignored and reported as such."""

        if relation not in RELATION_INDEX:
            raise ValueError(f"relation {relation!r} is not one of the seven typed relations")
        if head not in self.nodes or tail not in self.nodes:
            raise KeyError("both endpoints must be present before an edge is added")
        key = (head, relation, tail)
        if key in self._keys:
            return False
        self._keys.add(key)
        self.edges.append(Edge(head, relation, tail))
        self._out.setdefault((head, relation), []).append(tail)
        self._in.setdefault((tail, relation), []).append(head)
        return True

    def neighbours(self, node: str, relation: str) -> Sequence[str]:
        return tuple(self._out.get((node, relation), ()))

    def predecessors(self, node: str, relation: str) -> Sequence[str]:
        return tuple(self._in.get((node, relation), ()))

    def out_degree(self, node: str, relation: str | None = None) -> int:
        if relation is not None:
            return len(self._out.get((node, relation), ()))
        return sum(len(targets) for (head, _), targets in self._out.items() if head == node)

    def in_degree(self, node: str, relation: str | None = None) -> int:
        if relation is not None:
            return len(self._in.get((node, relation), ()))
        return sum(len(heads) for (tail, _), heads in self._in.items() if tail == node)

    def total_degree(self, node: str) -> int:
        return self.out_degree(node) + self.in_degree(node)

    def nodes_of(self, kind: NodeKind) -> tuple[str, ...]:
        return tuple(
            identifier for identifier in self._order if self.nodes[identifier].kind is kind
        )

    def index(self) -> dict[str, int]:
        return {identifier: position for position, identifier in enumerate(self._order)}

    def arrays(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Head, relation and tail index arrays in insertion order."""

        lookup = self.index()
        head = np.asarray([lookup[edge.head] for edge in self.edges], dtype=np.int64)
        relation = np.asarray(
            [RELATION_INDEX[edge.relation] for edge in self.edges], dtype=np.int64
        )
        tail = np.asarray([lookup[edge.tail] for edge in self.edges], dtype=np.int64)
        return head, relation, tail

    def census(self) -> dict[str, int]:
        counts: dict[str, int] = {"edges": len(self.edges), "nodes": len(self.nodes)}
        for kind in NodeKind:
            counts[f"nodes_{kind.value}"] = len(self.nodes_of(kind))
        for relation in RELATION_TYPES:
            counts[f"edges_{relation}"] = sum(1 for edge in self.edges if edge.relation == relation)
        return counts

    def degree_vector(self, identifiers: Iterable[str]) -> npt.NDArray[np.float64]:
        return np.asarray(
            [float(self.total_degree(identifier)) for identifier in identifiers],
            dtype=np.float64,
        )


def build_knowledge_graph(
    catalogue: Sequence[Candidate],
    *,
    seed: int = 20260831,
) -> KnowledgeGraph:
    """Build the typed graph over the candidate directory and the concept layer."""

    graph = KnowledgeGraph()
    for identifier, kind, label in CONCEPT_LAYER:
        graph.add_node(Node(identifier, kind, label))
    stream = spawn_generator(seed, "graph")
    colorectal = DISEASE_LOOKUP["colorectal adenocarcinoma"]
    for entry in catalogue:
        node_id = f"DR:{entry.identifier}"
        degree = float(entry.knowledge_graph_degree)
        graph.add_node(
            Node(
                node_id,
                NodeKind.CANDIDATE,
                entry.label,
                (degree, float(len(entry.approved_indications)), 1.0 if entry.admitted else 0.0),
            )
        )
        for target in entry.targets:
            resolved = TARGET_LOOKUP.get(target)
            if resolved is not None:
                graph.add_edge(node_id, "has_target", resolved)
        for indication in entry.approved_indications:
            disease = DISEASE_LOOKUP.get(indication)
            if disease is not None:
                graph.add_edge(node_id, "has_indication", disease)
        graph.add_edge(
            node_id,
            "affects_pathway",
            PATHWAY_IDS[int(stream.integers(0, len(PATHWAY_IDS)))],
        )
        graph.add_edge(node_id, "associated_with_disease", colorectal)
    _add_concept_edges(graph, seed=seed)
    return graph


def _add_concept_edges(graph: KnowledgeGraph, *, seed: int) -> None:
    """Target-pathway membership and the co-occurrence edges of the concept layer."""

    stream = spawn_generator(seed, "concept-edges")
    for target in graph.nodes_of(NodeKind.TARGET):
        for pathway in graph.nodes_of(NodeKind.PATHWAY):
            if float(stream.random()) < 0.18:
                graph.add_edge(target, "pathway_membership", pathway)
    concepts = graph.nodes_of(NodeKind.RECORD_CONCEPT)
    for position, first in enumerate(concepts):
        for second in concepts[position + 1 :]:
            if float(stream.random()) < 0.25:
                graph.add_edge(first, "co_occurs_with", second)


def anchor_nodes(state: DecisionState) -> Mapping[AnchorType, tuple[str, ...]]:
    """The typed anchors one decision state attaches to the graph through."""

    molecular = state.molecular
    alteration: list[str] = []
    if molecular.pi3k_altered:
        alteration.append("RC:00")
    if not molecular.ras_mutant and not molecular.braf_v600e:
        alteration.append("RC:01")
    if molecular.braf_v600e:
        alteration.append("RC:02")
    if molecular.msi_high:
        alteration.append("RC:03")
    if not alteration:
        alteration.append("RC:01")
    clinical = state.clinical
    record: list[str] = []
    if clinical.pattern.value == "liver_limited":
        record.append("RC:04")
    if clinical.sidedness.value.startswith("right"):
        record.append("RC:05")
    if not record:
        record.append("RC:04")
    pathway = ["PW:01"] if molecular.pi3k_altered else ["PW:02"]
    disease = ["DS:00"]
    if molecular.msi_high:
        disease.append("DS:04")
    return {
        AnchorType.MOLECULAR_ALTERATION: tuple(alteration),
        AnchorType.PATHWAY: tuple(pathway),
        AnchorType.DISEASE: tuple(disease),
        AnchorType.RECORD_CONCEPT: tuple(record),
    }


def attach_patient_node(graph: KnowledgeGraph, patient_id: str, state: DecisionState) -> str:
    """Add (or reuse) a patient node and attach its typed anchors.

    The attachment is the object Stage 2 reconstructs: a masked anchor is one of
    these edges, and a corrupted anchor is a substitution drawn from the same
    anchor type's non-anchors.
    """

    node_id = f"PT:{patient_id}"
    if not graph.has_node(node_id):
        graph.add_node(
            Node(
                node_id,
                NodeKind.PATIENT,
                patient_id,
                tuple(state.clinical.reference_vector),
            )
        )
    for anchor_type, anchors in anchor_nodes(state).items():
        relation = ANCHOR_RELATION[anchor_type]
        for anchor in anchors:
            graph.add_edge(node_id, relation, anchor)
    return node_id


def anchors_of(graph: KnowledgeGraph, patient_id: str) -> Mapping[AnchorType, tuple[str, ...]]:
    """Read back the typed anchors a patient node carries."""

    node_id = f"PT:{patient_id}"
    collected: dict[AnchorType, list[str]] = {kind: [] for kind in AnchorType}
    for anchor_type, relation in ANCHOR_RELATION.items():
        for tail in graph.neighbours(node_id, relation):
            if _kind_matches(anchor_type, graph.nodes[tail].kind):
                collected[anchor_type].append(tail)
    return {kind: tuple(values) for kind, values in collected.items()}


def _kind_matches(anchor_type: AnchorType, kind: NodeKind) -> bool:
    if anchor_type is AnchorType.PATHWAY:
        return kind is NodeKind.PATHWAY
    if anchor_type is AnchorType.DISEASE:
        return kind is NodeKind.DISEASE
    return kind is NodeKind.RECORD_CONCEPT


def non_anchors(graph: KnowledgeGraph, anchor_type: AnchorType, count: int) -> tuple[str, ...]:
    """Candidate substitutions of one anchor type, drawn in node order."""

    if anchor_type is AnchorType.PATHWAY:
        pool = graph.nodes_of(NodeKind.PATHWAY)
    elif anchor_type is AnchorType.DISEASE:
        pool = graph.nodes_of(NodeKind.DISEASE)
    else:
        pool = graph.nodes_of(NodeKind.RECORD_CONCEPT)
    return tuple(pool[:count])


def relation_counts(graph: KnowledgeGraph) -> Mapping[str, int]:
    counts: dict[str, int] = dict.fromkeys(RELATION_TYPES, 0)
    for edge in graph.edges:
        counts[edge.relation] += 1
    return counts
