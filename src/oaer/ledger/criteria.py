"""The pre-specified criteria of Table 1 panel d, with their margins.

Two primary rows, two supporting rows, two secondary rows and one descriptive
row. The two primary rows carry no multiplicity adjustment because the criterion
is their intersection and both must be met; the supporting and secondary rows
carry the false-discovery-rate correction. The discrimination criterion is judged
on the lower confidence limit of the paired difference, as the Methods fix, so a
row can have a point estimate above the bar and still be reported as not met.

Each row also records whether its threshold is arithmetically reachable at all on
the reported cohort, because a margin a design cannot clear is a different
statement from a margin the cohort did not clear.

Ref: Table 1 panel d, Sec. 2.7 (criteria), Sec. 1 (E-value and margin notes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Criterion:
    """One pre-specified criterion with its threshold and reported read-out."""

    identifier: str
    kind: str
    threshold: str
    observed: str
    met: bool
    margin: float
    anchor: str
    multiplicity_adjusted: bool
    reachable: bool
    note: str = ""


TABLE_1D_CRITERIA: Final[tuple[Criterion, ...]] = (
    Criterion(
        identifier="primary-pfs-hr",
        kind="primary",
        threshold="prospective PFS hazard ratio <= 0.75 with the upper 95% limit below 1",
        observed="0.68 (0.55-0.84)",
        met=True,
        margin=0.07,
        anchor="ESMO magnitude-of-clinical-benefit scale",
        multiplicity_adjusted=False,
        reachable=True,
        note="margin is how far below the bar the reported value sits",
    ),
    Criterion(
        identifier="primary-rmst-contrast",
        kind="primary",
        threshold="weighted restricted-mean policy contrast >= 1.5 months with the lower 95% limit above 0",
        observed="2.4 (0.9-3.9)",
        met=True,
        margin=0.9,
        anchor="ESMO magnitude-of-clinical-benefit scale anchor of 1.5 months",
        multiplicity_adjusted=False,
        reachable=True,
    ),
    Criterion(
        identifier="supporting-c-index-best-score",
        kind="supporting",
        threshold="prospective OS C-index against the best score, lower 95% limit of the difference >= +0.05",
        observed="0.746 against 0.658; difference 0.088 (0.051-0.125)",
        met=True,
        margin=0.001,
        anchor="best reproduced prognostic reference score",
        multiplicity_adjusted=True,
        reachable=True,
        note="the lower limit clears the margin by 0.001",
    ),
    Criterion(
        identifier="sensitivity-c-index-observed-score",
        kind="sensitivity",
        threshold="same margin against the best prospectively observed score",
        observed="0.746 against 0.671; difference 0.075 (0.038-0.112)",
        met=False,
        margin=-0.012,
        anchor="best prospectively observed score",
        multiplicity_adjusted=True,
        reachable=True,
        note="the lower limit sits 0.012 below the margin, so the row is not met",
    ),
    Criterion(
        identifier="secondary-calibration",
        kind="secondary",
        threshold="calibration slope 0.85-1.15 and expected calibration error <= 0.05",
        observed="slope 0.97, expected calibration error 0.028",
        met=True,
        margin=0.022,
        anchor="none",
        multiplicity_adjusted=True,
        reachable=True,
    ),
    Criterion(
        identifier="secondary-net-benefit",
        kind="secondary",
        threshold="net benefit above treat-all and treat-none across the 10-40% threshold range",
        observed="+0.031 to +0.064",
        met=True,
        margin=0.031,
        anchor="decision curve analysis",
        multiplicity_adjusted=True,
        reachable=True,
    ),
    Criterion(
        identifier="descriptive-nri",
        kind="descriptive",
        threshold="net reclassification improvement with its interval, read only beside calibration",
        observed="0.081 (0.024-0.138)",
        met=False,
        margin=0.0,
        anchor="reclassification literature",
        multiplicity_adjusted=False,
        reachable=True,
        note="a descriptive read-out carries no met-or-not verdict; the field records the absence",
    ),
    Criterion(
        identifier="supporting-top-agreement-discordant",
        kind="supporting",
        threshold="top agreement against discordant initiation, the active-comparator counterpart",
        observed="1.14 (0.97-1.34)",
        met=False,
        margin=-0.17,
        anchor="active-comparator counterpart",
        multiplicity_adjusted=True,
        reachable=True,
    ),
)

# The comparators the primary hazard-ratio bar is read against. None of them
# prices an association between ranking strata, which is why the observed value is
# reported against each as that framework states it rather than as a single bar.
HAZARD_RATIO_ANCHORS: Final[dict[str, float]] = {
    "reported primary bar": 0.75,
    "ESMO magnitude-of-clinical-benefit scale": 0.65,
    "ASCO clinically meaningful outcome": 0.67,
}

DISCRIMINATION_MARGIN: Final[float] = 0.05
REPRODUCED_REFERENCE_FAMILY: Final[str] = "reproduced prognostic reference scores"


def reference_score_spread() -> tuple[float, float]:
    """The spread of the reproduced reference family's point estimates.

    Sec. 1 states the discrimination margin is read against the published spread
    of reproduced metastatic colorectal prognostic scores, which is the family the
    roster records under that name. The spread is derived from those rows rather
    than stated as a free constant, so the margin's reachability is decided by the
    rows the article reports and not by a number carried beside them.
    """

    from oaer.ledger.comparator_roster import TABLE_2B_PROGNOSTIC

    points = [
        row.os_c_index[0]
        for row in TABLE_2B_PROGNOSTIC
        if row.family == REPRODUCED_REFERENCE_FAMILY
    ]
    if not points:
        raise ValueError("the roster carries no reproduced reference score")
    return (min(points), max(points))


def criterion_by_id(identifier: str) -> Criterion:
    for criterion in TABLE_1D_CRITERIA:
        if criterion.identifier == identifier:
            return criterion
    raise KeyError(f"{identifier!r} is not a pre-specified criterion")


def primary_rows() -> tuple[Criterion, ...]:
    return tuple(c for c in TABLE_1D_CRITERIA if c.kind == "primary")


def adjusted_rows() -> tuple[Criterion, ...]:
    return tuple(c for c in TABLE_1D_CRITERIA if c.multiplicity_adjusted)


def summary() -> dict[str, object]:
    primaries = primary_rows()
    return {
        "total": len(TABLE_1D_CRITERIA),
        "primary": len(primaries),
        "primary_all_met": all(c.met for c in primaries),
        "met": sum(1 for c in TABLE_1D_CRITERIA if c.met),
        "not_met": sum(1 for c in TABLE_1D_CRITERIA if not c.met),
        "multiplicity_adjusted": len(adjusted_rows()),
        "reachable": sum(1 for c in TABLE_1D_CRITERIA if c.reachable),
    }


def anchored_margins(observed: float) -> dict[str, float]:
    """The observed value read against each hazard-ratio anchor."""

    return {name: observed - bar for name, bar in HAZARD_RATIO_ANCHORS.items()}


def discrimination_margin_is_reachable(
    spread: tuple[float, float] | None = None,
    margin: float = DISCRIMINATION_MARGIN,
) -> bool:
    """Whether a paired difference of ``margin`` can exist inside the spread.

    The margin is an ordering margin against the published spread of reproduced
    scores. It is reachable when the spread is at least as wide as the margin,
    because a difference larger than the whole spread cannot be realised by any
    member of that family. The spread defaults to the one the roster's own
    reproduced-reference rows carry.
    """

    window = reference_score_spread() if spread is None else spread
    return (window[1] - window[0]) >= margin


def value_gap_reachability(
    policy_value: float,
    no_initiation_value: float,
    bar: float = 1.5,
) -> bool:
    """Whether the reported policy value can clear the restricted-mean bar."""

    return (policy_value - no_initiation_value) >= bar
