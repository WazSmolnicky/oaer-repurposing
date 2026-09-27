"""Catalogue: the candidate directory and the typed knowledge graph.

The catalogue admits an agent when at least one of its approved indications is
non-oncological, so the oncology-only agents are excluded by construction. The
typed graph carries the four anchor types the patient node attaches through --
molecular alteration, pathway membership, disease and record concept -- and the
two relational encoders read it.
"""

from __future__ import annotations

from oaer.catalogue.concepts import (
    CANDIDATE_CLASSES,
    CatalogueError,
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
    RELATION_TYPES,
    AnchorType,
    Edge,
    KnowledgeGraph,
    Node,
    build_knowledge_graph,
)
from oaer.catalogue.motifs import (
    Motif,
    enumerate_motifs,
    motif_laplacian,
    motif_signature,
)
from oaer.catalogue.sampling import (
    AnchorCorruption,
    MaskedEdgeBatch,
    corrupt_anchors,
    sample_masked_edges,
)
from oaer.catalogue.sources import (
    AUXILIARY_SOURCES,
    CANDIDATE_SOURCES,
    PUBLIC_COHORTS,
    AuxiliarySource,
    candidate_source_envelope,
)

__all__ = [
    "AUXILIARY_SOURCES",
    "CANDIDATE_CLASSES",
    "CANDIDATE_SOURCES",
    "PUBLIC_COHORTS",
    "RELATION_TYPES",
    "AnchorCorruption",
    "AnchorType",
    "AuxiliarySource",
    "CatalogueError",
    "Edge",
    "IndicationRecord",
    "KnowledgeGraph",
    "MaskedEdgeBatch",
    "Motif",
    "Node",
    "PublicCohort",
    "admitted_by_indications",
    "build_catalogue",
    "build_knowledge_graph",
    "candidate_source_envelope",
    "catalogue_by_identifier",
    "classify_indication",
    "corrupt_anchors",
    "degree_histogram",
    "degree_terciles",
    "enumerate_motifs",
    "motif_laplacian",
    "motif_signature",
    "named_candidates",
    "orbit_partition",
    "reprocessing_priority",
    "sample_masked_edges",
]
