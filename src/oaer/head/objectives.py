"""Stage 3's objective: weighted least squares on the out-of-fold pseudo-outcomes.

    L(theta, phi) = sum_i sum_{d in D_eps(s_i)} w_{i,d}
                    ( tau_phi(h_theta(s_i, d)) - Gamma_i,d )^2
                    + lambda ( ||theta||^2 + ||phi||^2 )

The weights default to one. The article's ``w_{i,d}`` is the candidate-balancing
weight: without it the loss is dominated by the candidates whose exposed count is
largest, and the sensitivity row that removes it reports the sparse fall that
follows.

Ref: Sec. 2.6 Eq. (5), Sec. 1.4 panel a (the no-candidate-balanced-weights row).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StageThreeWeights:
    """The candidate-balancing weights of Stage 3.

    The balanced form gives every candidate the same total mass, so a candidate
    with a large exposed count cannot dominate. The unbalanced form leaves each
    eligible cell at unit weight, which is the sensitivity row that removes the
    balancing; a cell outside ``D_eps`` carries no valid pseudo-outcome and keeps
    zero weight under either form.
    """

    balanced: bool = True
    floor: float = 1.0

    def build(self, eligible: Tensor) -> Tensor:
        """One weight per eligible cell, normalised so the total mass is fixed."""

        mask = (eligible > 0.0).to(torch.float32)
        if float(mask.sum()) <= 0.0:
            return mask
        if not self.balanced:
            return mask
        counts = mask.sum(dim=0, keepdim=True).clamp(min=self.floor)
        weights = mask / counts
        return weights * (mask.sum() / weights.sum())


def weighted_least_squares(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    parameters: Sequence[Tensor],
    penalty: float,
) -> Tensor:
    """Eq. (5) with an explicit penalty over the parameter list."""

    if prediction.shape != target.shape or prediction.shape != weights.shape:
        raise ValueError("prediction, target and weights must share a shape")
    residual = prediction - target
    data_term = (weights * residual * residual).sum() / weights.sum().clamp(min=1.0)
    penalty_term = torch.zeros((), device=prediction.device)
    for parameter in parameters:
        penalty_term = penalty_term + parameter.pow(2).sum()
    return data_term + penalty * penalty_term


def penalty_norm(parameters: list[Tensor]) -> float:
    """The squared parameter norm, for the read-out."""

    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().pow(2).sum())
    return total


class StageThreeHead(nn.Module):
    """The trainable Stage 3 wrapper over a ranking head.

    Only the head's parameters are optimised here; the encoder is frozen after the
    graph pretraining stages, because Stage 3 is the only stage that reads an
    outcome and the article states that no outcome reaches the encoder.
    """

    def __init__(self, head: nn.Module, penalty: float = 1.0e-4) -> None:
        super().__init__()
        self.head = head
        self.penalty = penalty

    def loss(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Tensor,
    ) -> Tensor:
        parameters = [parameter for parameter in self.head.parameters() if parameter.requires_grad]
        return weighted_least_squares(prediction, target, weights, parameters, self.penalty)


def residual_summary(prediction: Tensor, target: Tensor, weights: Tensor) -> dict[str, float]:
    """Residual statistics for the audit block."""

    residual = (prediction - target).detach()
    if residual.numel() == 0:
        return {"mean": float("nan"), "rmse": float("nan"), "weighted_rmse": float("nan")}
    weighted = (weights * residual * residual).sum() / weights.sum().clamp(min=1.0)
    return {
        "mean": float(residual.mean()),
        "rmse": float(torch.sqrt(residual.pow(2).mean())),
        "weighted_rmse": float(torch.sqrt(weighted)),
    }


def effective_weight_mass(weights: Tensor) -> float:
    """The effective sample size of a weight matrix, by Kish's definition."""

    total = float(weights.sum())
    squares = float(weights.pow(2).sum())
    if squares <= 0.0:
        return 0.0
    return total * total / squares
