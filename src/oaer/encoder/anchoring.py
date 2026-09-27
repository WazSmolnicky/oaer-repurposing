"""Stage 2: the patient anchor attachment and its reconstruction objective.

A patient node attaches to the graph through four typed anchors -- molecular
alteration, pathway, disease and record concept. Stage 2 hides an incomplete
collection of those anchors and reconstructs them against non-anchors of the same
type, which is what teaches the representation to place a patient the encoder has
never seen. The head that reads the attachment is the patient anchor head, and
its output is the ``h_theta(s, d)`` the ranking head consumes.

Ref: Sec. 2.4 (patient anchor, zero-mean modifier), Sec. 2.6 (Stage 2), Eq. (4).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from oaer.catalogue.graph import ANCHOR_RELATION, AnchorType


class PatientAnchorHead(nn.Module):
    """Combine a patient's typed anchor representations into one patient vector."""

    def __init__(self, width: int, anchor_types: int = 4) -> None:
        super().__init__()
        if anchor_types < 1:
            raise ValueError("at least one anchor type is required")
        self.width = width
        self.anchor_types = anchor_types
        self.type_embedding = nn.Embedding(anchor_types, width)
        self.attention = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.norm = nn.LayerNorm(width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, anchor_states: Tensor, type_index: Tensor) -> Tensor:
        """Attention-pool the anchors of one patient into a single vector."""

        if anchor_states.dim() != 2 or type_index.dim() != 1:
            raise ValueError("anchor states must be a matrix and type indices a vector")
        if anchor_states.shape[0] != type_index.shape[0]:
            raise ValueError("every anchor state needs its type index")
        if anchor_states.shape[0] == 0:
            return torch.zeros(self.width, device=anchor_states.device)
        enriched = anchor_states + self.type_embedding(type_index)
        logits = self.attention(enriched).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        pooled: Tensor = self.norm((weights.unsqueeze(-1) * enriched).sum(dim=0))
        return pooled

    def pool_rows(self, anchor_states: Tensor, type_index: Tensor) -> Tensor:
        """Combine one anchor per row, keeping the batch.

        The set-level forward pools a whole attachment into one vector. Stage 2
        needs one patient vector per masked edge, so this variant keeps the row
        axis and shares the same parameters rather than introducing a second head.
        """

        if anchor_states.dim() != 2 or type_index.dim() != 1:
            raise ValueError("anchor states must be a matrix and type indices a vector")
        if anchor_states.shape[0] != type_index.shape[0]:
            raise ValueError("every anchor state needs its type index")
        pooled: Tensor = self.norm(anchor_states + self.type_embedding(type_index))
        return pooled


class AnchorReconstruction(nn.Module):
    """The Stage 2 read-out: a score for a masked anchor against its non-anchors."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, patient: Tensor, anchor: Tensor) -> Tensor:
        projected = self.projection(anchor)
        score: Tensor = (patient * projected).sum(dim=-1) + self.bias
        return score


def anchor_type_index(anchor_type: AnchorType) -> int:
    return list(AnchorType).index(anchor_type)


def relation_for(anchor_type: AnchorType) -> str:
    return ANCHOR_RELATION[anchor_type]


def anchor_scores(
    head: AnchorReconstruction,
    patient: Tensor,
    positive: Tensor,
    negative: Tensor,
) -> tuple[Tensor, Tensor]:
    """Score one masked anchor and its non-anchor set.

    ``patient`` is a single vector, ``positive`` a matrix of the patient's own
    anchors and ``negative`` a matrix of non-anchors of the same types.
    """

    if positive.dim() != 2 or negative.dim() != 2:
        raise ValueError("anchor tensors must be matrices")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("the anchors and the non-anchors must share a width")
    return head(patient, positive), head(patient, negative)


def anchor_reconstruction_loss(
    positive_scores: Tensor,
    negative_scores: Tensor,
) -> Tensor:
    """Eq. (4): the log-sigmoid objective over anchors and non-anchors.

    The two sums are the article's: every masked anchor contributes
    ``log sigma(s)`` and every non-anchor contributes ``log(1 - sigma(s))``.
    """

    if positive_scores.dim() != 1 or negative_scores.dim() != 1:
        raise ValueError("the score tensors must be vectors")
    if positive_scores.numel() == 0:
        raise ValueError("a reconstruction batch must carry at least one anchor")
    positive_term = torch.nn.functional.logsigmoid(positive_scores).sum()
    if negative_scores.numel():
        negative_term = torch.nn.functional.logsigmoid(-negative_scores).sum()
    else:
        negative_term = torch.zeros((), device=positive_scores.device)
    return -(positive_term + negative_term) / positive_scores.numel()


def anchor_accuracy(positive_scores: Tensor, negative_scores: Tensor) -> float:
    """The share of anchors scored above the median non-anchor score."""

    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return float("nan")
    threshold = float(negative_scores.median())
    return float((positive_scores > threshold).float().mean())


def anchor_type_breakdown(
    scores: dict[AnchorType, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Per-type mean anchor and non-anchor scores, for the audit block."""

    return {
        anchor_type.value: {"anchor": anchor, "non_anchor": non_anchor}
        for anchor_type, (anchor, non_anchor) in scores.items()
    }
