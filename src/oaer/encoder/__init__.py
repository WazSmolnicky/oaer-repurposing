"""The relational encoder, its pretraining stages and the patient anchor route."""

from __future__ import annotations

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
    MotifBasis,
    degree_tensor,
    edge_scores,
    neighbourhood_statistics,
    propagate,
)
from oaer.encoder.relational import RelationalEncoder, masked_edge_loss, motif_summary

__all__ = [
    "DEFAULT_MOTIF_BASIS",
    "AnchorReconstruction",
    "BasisDecomposition",
    "MotifBasis",
    "PatientAnchorHead",
    "RelationalEncoder",
    "anchor_accuracy",
    "anchor_reconstruction_loss",
    "anchor_scores",
    "anchor_type_breakdown",
    "anchor_type_index",
    "degree_tensor",
    "edge_scores",
    "masked_edge_loss",
    "motif_summary",
    "neighbourhood_statistics",
    "propagate",
]
