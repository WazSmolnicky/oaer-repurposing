"""The pathology benefit-modifier route.

The ranking head is split in two: a route score that carries what the graph says
about the candidate's mechanism, and a benefit modifier that carries what the
slide says. The modifier is a special kind of variable -- it acts on the effect
through the informative predictor without being connectable to the outcome on its
own -- so it is centred on the eligible set before it is allowed to move the
ranking, which is the zero-mean condition of Sec. 2.4. Centring is over ``D_eps``
rather than over the catalogue, because an ineligible candidate is not part of the
set the score is read on.

Ref: Sec. 2.4 (ranking head and outcome anchoring), Sec. 1.4 panel d.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RouteScore(nn.Module):
    """The route score ``g_phi`` over the patient-candidate representation."""

    def __init__(self, width: int, hidden: int = 32) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("the route score needs positive widths")
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, representation: Tensor) -> Tensor:
        score: Tensor = self.network(representation).squeeze(-1) * self.scale
        return score


class PathologyModifier(nn.Module):
    """The benefit modifier ``b_phi`` read off the fixed slide embedding."""

    def __init__(self, slide_dim: int, hidden: int = 16) -> None:
        super().__init__()
        if slide_dim < 1 or hidden < 1:
            raise ValueError("the modifier needs positive widths")
        self.encoder = nn.Sequential(
            nn.Linear(slide_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.present = nn.Parameter(torch.tensor(0.0))
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, slide: Tensor, available: Tensor | None = None) -> Tensor:
        """The raw modifier, before the zero-mean condition is applied."""

        raw: Tensor = self.encoder(slide).squeeze(-1)
        if available is None:
            return raw
        # A decision whose slide route is absent carries no morphology signal, so
        # its raw modifier is held at the route's own bias rather than at zero.
        held: Tensor = torch.where(available > 0.5, raw, torch.zeros_like(raw) + self.present)
        return held


def zero_mean_centre(values: Tensor, eligible: Tensor | None = None) -> Tensor:
    """Centre a modifier on the eligible set; a row outside it carries none.

    ``eligible`` is a 0/1 vector over the catalogue, so the mean is taken over
    ``D_eps(s)`` alone and a candidate outside the set is left at zero rather than
    at a value the condition does not define. With no mask the mean is taken over
    the rows supplied.
    """

    if eligible is None:
        if values.numel() == 0:
            return values
        return values - values.mean()
    mask = eligible > 0.5
    if not bool(mask.any()):
        return torch.zeros_like(values)
    selected = values[mask]
    centred = torch.zeros_like(values)
    centred = centred.index_copy(
        0, torch.nonzero(mask, as_tuple=False).squeeze(-1), selected - selected.mean()
    )
    return centred


def interaction_index(delta_ab: float, delta_a: float, delta_b: float) -> float:
    """The interaction ratio ``IR`` of Sec. 1.4 panel d.

    ``IR = (delta_a + delta_b) / delta_ab`` on the removal scale, so a value below
    one is synergy -- the joint removal costs more than the two single removals
    together -- and a value above one is redundancy. The ratio is printed only
    where the interval of its denominator is entirely on one side of zero.
    """

    denominator = delta_ab
    if abs(denominator) < 1e-12:
        return float("nan")
    return (delta_a + delta_b) / denominator


def interaction_beyond_additive(
    delta_ab: float,
    delta_a: float,
    delta_b: float,
) -> float:
    """``Delta_AB - (Delta_A + Delta_B)`` on the same scale as the deltas.

    A negative value is synergy, which is the sign convention the article fixes:
    the pre-specified expectation is that outcome anchoring and the graph prior
    are synergistic, that is ``IAB < 0`` together with ``IR < 1``.
    """

    return delta_ab - (delta_a + delta_b)
