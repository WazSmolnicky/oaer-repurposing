"""The ablation ledger of Table 3, as data.

Three removal families. ``C1`` is the outcome-anchoring doubly robust objective,
``C2`` the graph-pretrained patient-anchored representation (whole, and as its
three sub-parts ``C2a`` random initialisation, ``C2b`` no graph representation and
``C2c`` no patient anchor), ``C3`` the identification gate and ``C4`` the
pathology benefit-modifier route. Every joint row removes the whole of ``C2``
rather than one sub-part, because the pairwise interactions are read against the
component as it is defined.

Each row carries its deltas in three currencies, the top-five Jaccard overlap and
the pre-specified expectation that was fixed in Methods before the prospective
window opened. The ``IAB`` and ``IR`` of panel d are computed from three rows of
panel a -- the two canonical single removals and the joint removal of the same
pair -- which is what makes each of them arithmetically traceable.

Ref: Table 3 panels a-d, Sec. 1.4, Sec. 2.7 (sign convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class AblationRow:
    """One configuration's removal deltas and the expectation it is read against."""

    configuration: str
    hits_delta: float
    hits_interval: tuple[float, float]
    hits_seed_sd: float
    value_delta: float
    value_interval: tuple[float, float]
    value_seed_sd: float
    index_delta: float
    index_interval: tuple[float, float]
    index_seed_sd: float
    top_five_jaccard: float | None
    expectation: str


TABLE_3A_REMOVALS: Final[tuple[AblationRow, ...]] = (
    AblationRow(
        "Full OAER",
        0.0,
        (0.0, 0.0),
        0.0,
        0.42,
        (0.0, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        0.0,
        0.42,
        "reference",
    ),
    AblationRow(
        "Tier 0: random-input control",
        -32.9,
        (-35.4, -30.4),
        0.61,
        -2.7,
        (-3.1, -2.3),
        0.11,
        -0.171,
        (-0.188, -0.154),
        0.009,
        0.06,
        "at chance",
    ),
    AblationRow(
        "Tier 0: trivial pooled-effect ranking",
        -24.1,
        (-26.8, -21.4),
        0.72,
        -1.9,
        (-2.3, -1.5),
        0.13,
        -0.128,
        (-0.145, -0.111),
        0.010,
        0.11,
        "invariant",
    ),
    AblationRow(
        "Tier 0: prescribing-prevalence ranking",
        -19.6,
        (-22.1, -17.1),
        0.68,
        -1.6,
        (-2.0, -1.2),
        0.12,
        -0.104,
        (-0.120, -0.088),
        0.010,
        0.19,
        "popularity floor",
    ),
    AblationRow(
        "Tier 0: clinical-covariate Cox",
        0.0,
        (0.0, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        0.0,
        None,
        "prognostic floor",
    ),
    AblationRow(
        "C1 removed, curated-label target",
        -8.4,
        (-11.2, -5.6),
        0.94,
        -1.4,
        (-2.0, -0.8),
        0.16,
        -0.061,
        (-0.079, -0.043),
        0.012,
        0.71,
        "largest loss",
    ),
    AblationRow(
        "C1 removed, population average-effect target",
        -6.9,
        (-9.4, -4.4),
        0.88,
        -1.1,
        (-1.7, -0.5),
        0.15,
        -0.052,
        (-0.069, -0.035),
        0.011,
        0.78,
        "invariant optimum",
    ),
    AblationRow(
        "C1 removed, trial-outcome-reward target",
        -5.3,
        (-7.6, -3.0),
        0.83,
        -0.9,
        (-1.4, -0.4),
        0.14,
        -0.044,
        (-0.060, -0.028),
        0.011,
        0.74,
        "invariant optimum",
    ),
    AblationRow(
        "C2 removed, whole component",
        -7.8,
        (-10.1, -5.5),
        0.86,
        -1.2,
        (-1.7, -0.7),
        0.14,
        -0.058,
        (-0.074, -0.042),
        0.011,
        None,
        "second largest",
    ),
    AblationRow(
        "C2a removed, randomly initialized encoder",
        -6.4,
        (-8.9, -3.9),
        0.91,
        -1.0,
        (-1.5, -0.5),
        0.15,
        -0.049,
        (-0.066, -0.032),
        0.012,
        None,
        "second largest",
    ),
    AblationRow(
        "C2b removed, no graph representation",
        -6.9,
        (-9.4, -4.4),
        0.89,
        -1.1,
        (-1.6, -0.6),
        0.15,
        -0.052,
        (-0.069, -0.035),
        0.011,
        None,
        "second largest",
    ),
    AblationRow(
        "C2c removed, no patient anchor",
        -8.1,
        (-10.6, -5.6),
        0.93,
        -1.3,
        (-1.9, -0.7),
        0.16,
        -0.057,
        (-0.075, -0.039),
        0.012,
        None,
        "as costly as C1",
    ),
    AblationRow(
        "C3 removed, no identification gate",
        -3.4,
        (-5.2, -1.6),
        0.77,
        -1.3,
        (-1.9, -0.7),
        0.13,
        -0.026,
        (-0.040, -0.012),
        0.010,
        None,
        "contrast up, value down",
    ),
    AblationRow(
        "C3 removed, propensity on the embedding",
        -2.9,
        (-4.6, -1.2),
        0.81,
        -1.6,
        (-2.2, -1.0),
        0.14,
        -0.021,
        (-0.035, -0.007),
        0.011,
        None,
        "validity loss",
    ),
    AblationRow(
        "C4 removed, no pathology route",
        -3.0,
        (-4.5, -1.5),
        0.74,
        -0.5,
        (-0.9, -0.1),
        0.10,
        -0.019,
        (-0.032, -0.006),
        0.009,
        None,
        "smallest; about zero live",
    ),
    AblationRow(
        "C4 removed, no zero-mean constraint",
        -1.8,
        (-3.3, -0.3),
        0.79,
        -0.3,
        (-0.7, 0.1),
        0.10,
        0.009,
        (-0.004, 0.022),
        0.009,
        None,
        "C may rise",
    ),
    AblationRow(
        "C1 and C2 removed jointly",
        -17.7,
        (-20.6, -14.8),
        0.99,
        -3.1,
        (-3.8, -2.4),
        0.18,
        -0.128,
        (-0.147, -0.109),
        0.013,
        None,
        "joint, feeds panel d",
    ),
    AblationRow(
        "C1 and C3 removed jointly",
        -13.1,
        (-15.9, -10.3),
        1.00,
        -2.9,
        (-3.6, -2.2),
        0.19,
        -0.095,
        (-0.114, -0.076),
        0.013,
        None,
        "joint, feeds panel d",
    ),
    AblationRow(
        "C1 and C4 removed jointly",
        -12.6,
        (-15.2, -10.0),
        0.93,
        -2.1,
        (-2.7, -1.5),
        0.17,
        -0.090,
        (-0.108, -0.072),
        0.012,
        None,
        "joint, feeds panel d",
    ),
    AblationRow(
        "C2 and C3 removed jointly",
        -10.6,
        (-13.5, -7.7),
        1.02,
        -2.3,
        (-2.9, -1.7),
        0.18,
        -0.078,
        (-0.096, -0.060),
        0.013,
        None,
        "joint, feeds panel d",
    ),
    AblationRow(
        "C2 and C4 removed jointly",
        -10.9,
        (-13.5, -8.3),
        0.95,
        -1.5,
        (-2.1, -0.9),
        0.16,
        -0.081,
        (-0.098, -0.064),
        0.012,
        None,
        "joint, feeds panel d",
    ),
    AblationRow(
        "C3 and C4 removed jointly",
        -6.6,
        (-8.7, -4.5),
        0.83,
        -1.8,
        (-2.4, -1.2),
        0.14,
        -0.047,
        (-0.063, -0.031),
        0.011,
        None,
        "joint, feeds panel d",
    ),
    AblationRow(
        "All four components removed, backbone only",
        -22.6,
        (-25.7, -19.5),
        1.04,
        -4.8,
        (-5.6, -4.0),
        0.19,
        -0.160,
        (-0.180, -0.140),
        0.014,
        None,
        "assembly floor",
    ),
    AblationRow(
        "Sensitivity: no candidate-balanced weights",
        -2.7,
        (-4.4, -1.0),
        0.87,
        -0.6,
        (-1.1, -0.1),
        0.13,
        -0.017,
        (-0.031, -0.003),
        0.010,
        None,
        "sparse fall",
    ),
    AblationRow(
        "Full OAER at the comparator's budget",
        -0.4,
        (-1.5, 0.7),
        0.55,
        -0.1,
        (-0.4, 0.2),
        0.09,
        -0.006,
        (-0.016, 0.004),
        0.008,
        0.45,
        "equal-budget row",
    ),
)

ROW_INDEX: Final[dict[str, int]] = {
    row.configuration: index for index, row in enumerate(TABLE_3A_REMOVALS)
}


@dataclass(frozen=True)
class PairInteraction:
    """One row of Table 3 panel d."""

    pair: str
    first: str
    second: str
    joint: str
    iab: float
    iab_interval: tuple[float, float]
    ratio: float | None
    expectation: str


TABLE_3D_INTERACTIONS: Final[tuple[PairInteraction, ...]] = (
    PairInteraction(
        "C1 x C2",
        "C1 removed, curated-label target",
        "C2 removed, whole component",
        "C1 and C2 removed jointly",
        -1.5,
        (-2.6, -0.4),
        0.92,
        "synergy, IR < 1; primary claim",
    ),
    PairInteraction(
        "C1 x C3",
        "C1 removed, curated-label target",
        "C3 removed, no identification gate",
        "C1 and C3 removed jointly",
        -1.3,
        (-2.4, -0.2),
        0.91,
        "synergy, IR < 1",
    ),
    PairInteraction(
        "C1 x C4",
        "C1 removed, curated-label target",
        "C4 removed, no pathology route",
        "C1 and C4 removed jointly",
        -1.2,
        (-2.3, -0.1),
        0.91,
        "synergy, IR < 1 if the share exceeds zero",
    ),
    PairInteraction(
        "C2 x C4",
        "C2 removed, whole component",
        "C4 removed, no pathology route",
        "C2 and C4 removed jointly",
        -0.1,
        (-1.0, 0.8),
        0.99,
        "IR about 1, or weak synergy",
    ),
    PairInteraction(
        "C2 x C3",
        "C2 removed, whole component",
        "C3 removed, no identification gate",
        "C2 and C3 removed jointly",
        0.6,
        (-0.2, 1.4),
        1.06,
        "redundancy, IR > 1",
    ),
    PairInteraction(
        "C3 x C4",
        "C3 removed, no identification gate",
        "C4 removed, no pathology route",
        "C3 and C4 removed jointly",
        -0.2,
        (-1.1, 0.7),
        None,
        "no direction pre-specified",
    ),
)


@dataclass(frozen=True)
class TercileRow:
    """One row of Table 3 panel b."""

    configuration: str
    hits_gain: float
    hits_interval: tuple[float, float]
    hits_seed_sd: float
    value_gain: float
    value_interval: tuple[float, float]
    value_seed_sd: float
    index_gain: float
    index_interval: tuple[float, float]
    index_seed_sd: float
    expectation: str


TABLE_3B_TERCILES: Final[tuple[TercileRow, ...]] = (
    TercileRow(
        "Lowest tercile, KG-pretrained minus random",
        3.2,
        (1.4, 5.0),
        0.58,
        0.7,
        (0.2, 1.2),
        0.10,
        0.021,
        (0.010, 0.032),
        0.008,
        "largest KG gain",
    ),
    TercileRow(
        "Lowest tercile, KG-pretrained minus no graph",
        2.4,
        (0.8, 4.0),
        0.61,
        0.5,
        (0.1, 0.9),
        0.11,
        0.016,
        (0.006, 0.026),
        0.008,
        "largest KG gain",
    ),
    TercileRow(
        "Middle tercile, KG-pretrained minus random",
        1.6,
        (0.2, 3.0),
        0.49,
        0.3,
        (-0.1, 0.7),
        0.09,
        0.010,
        (0.001, 0.019),
        0.007,
        "intermediate",
    ),
    TercileRow(
        "Middle tercile, KG-pretrained minus no graph",
        1.2,
        (0.0, 2.4),
        0.52,
        0.2,
        (-0.1, 0.5),
        0.09,
        0.008,
        (0.000, 0.016),
        0.007,
        "intermediate",
    ),
    TercileRow(
        "Highest tercile, KG-pretrained minus random",
        0.4,
        (-0.7, 1.5),
        0.41,
        0.1,
        (-0.2, 0.4),
        0.08,
        0.003,
        (-0.004, 0.010),
        0.006,
        "about zero KG gain",
    ),
    TercileRow(
        "Highest tercile, KG-pretrained minus no graph",
        0.3,
        (-0.8, 1.4),
        0.43,
        0.0,
        (-0.3, 0.3),
        0.08,
        0.002,
        (-0.005, 0.009),
        0.006,
        "about zero KG gain",
    ),
    TercileRow(
        "Highest tercile, per-drug DR-learner contrast",
        -0.2,
        (-1.2, 0.8),
        0.46,
        -0.1,
        (-0.4, 0.2),
        0.08,
        -0.004,
        (-0.011, 0.003),
        0.007,
        "parity",
    ),
)

TERCILE_SIZES: Final[tuple[tuple[str, int, float], ...]] = (
    ("lowest", 1104, 6.8),
    ("middle", 3318, 2.1),
    ("highest", 5210, 0.7),
)


@dataclass(frozen=True)
class ModalityRow:
    """One row of Table 3 panel c."""

    modalities: str
    hits_delta: float
    hits_interval: tuple[float, float]
    hits_seed_sd: float
    value_delta: float
    value_interval: tuple[float, float]
    value_seed_sd: float
    index_delta: float
    index_interval: tuple[float, float]
    index_seed_sd: float
    supplementary: bool = False


TABLE_3C_MODALITIES: Final[tuple[ModalityRow, ...]] = (
    ModalityRow(
        "Molecular only",
        -9.6,
        (-12.3, -6.9),
        0.88,
        -1.7,
        (-2.3, -1.1),
        0.15,
        -0.069,
        (-0.086, -0.052),
        0.011,
    ),
    ModalityRow(
        "Pathology only",
        -3.1,
        (-4.6, -1.6),
        0.72,
        -0.6,
        (-1.0, -0.2),
        0.10,
        -0.022,
        (-0.035, -0.009),
        0.009,
    ),
    ModalityRow(
        "Clinical only",
        -11.4,
        (-14.1, -8.7),
        0.84,
        -1.9,
        (-2.5, -1.3),
        0.14,
        -0.081,
        (-0.098, -0.064),
        0.011,
    ),
    ModalityRow(
        "Molecular and pathology",
        -18.9,
        (-21.7, -16.1),
        0.96,
        -3.1,
        (-3.8, -2.4),
        0.17,
        -0.134,
        (-0.153, -0.115),
        0.013,
    ),
    ModalityRow(
        "Molecular and clinical",
        -22.6,
        (-25.7, -19.5),
        1.02,
        -3.9,
        (-4.7, -3.1),
        0.18,
        -0.160,
        (-0.180, -0.140),
        0.013,
    ),
    ModalityRow(
        "Pathology and clinical, no molecular",
        -16.5,
        (-19.2, -13.8),
        0.90,
        -2.9,
        (-3.6, -2.2),
        0.17,
        -0.117,
        (-0.135, -0.099),
        0.012,
    ),
    ModalityRow(
        "All three, full model", 0.0, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0), 0.0
    ),
    ModalityRow(
        "All three, slide-encoder ensemble",
        0.4,
        (-0.7, 1.5),
        0.44,
        0.1,
        (-0.2, 0.4),
        0.08,
        0.003,
        (-0.004, 0.010),
        0.006,
        supplementary=True,
    ),
)

CANONICAL_REMOVALS: Final[dict[str, str]] = {
    "C1": "C1 removed, curated-label target",
    "C2": "C2 removed, whole component",
    "C3": "C3 removed, no identification gate",
    "C4": "C4 removed, no pathology route",
}

COMPONENT_LABELS: Final[dict[str, str]] = {
    "C1": "outcome-anchoring doubly robust objective",
    "C2": "graph-pretrained patient-anchored representation",
    "C3": "identification gate",
    "C4": "pathology benefit-modifier route",
}


def row_for(configuration: str) -> AblationRow:
    return TABLE_3A_REMOVALS[ROW_INDEX[configuration]]


def canonical_row(component: str) -> AblationRow:
    return row_for(CANONICAL_REMOVALS[component])


JOINT_REMOVALS: Final[dict[tuple[str, str], str]] = {
    ("C1", "C2"): "C1 and C2 removed jointly",
    ("C1", "C3"): "C1 and C3 removed jointly",
    ("C1", "C4"): "C1 and C4 removed jointly",
    ("C2", "C3"): "C2 and C3 removed jointly",
    ("C2", "C4"): "C2 and C4 removed jointly",
    ("C3", "C4"): "C3 and C4 removed jointly",
}


def joint_row(first: str, second: str) -> AblationRow:
    """The joint-removal row for one pair, which the panel-d interaction reads."""

    key = (first, second) if (first, second) in JOINT_REMOVALS else (second, first)
    if key not in JOINT_REMOVALS:
        raise KeyError(f"no joint removal row for {first} and {second}")
    return row_for(JOINT_REMOVALS[key])


def interaction_from_rows(first: str, second: str) -> tuple[float, float | None]:
    """``IAB`` and ``IR`` computed from the three panel-a rows the pair names."""

    left = canonical_row(first)
    right = canonical_row(second)
    joint = joint_row(first, second)
    iab = joint.hits_delta - (left.hits_delta + right.hits_delta)
    if abs(joint.hits_delta) < 1e-12:
        return (iab, None)
    return (iab, (left.hits_delta + right.hits_delta) / joint.hits_delta)


def expectation_direction(text: str) -> str:
    """The direction a row's expectation names, for the pre-specification check."""

    lowered = text.lower()
    if "synergy" in lowered:
        return "synergy"
    if "redundancy" in lowered:
        return "redundancy"
    if "no direction" in lowered:
        return "none"
    if "largest" in lowered or "smallest" in lowered or "second" in lowered:
        return "magnitude"
    return "none"


TERCILE_LOOKUP: Final[dict[tuple[str, str], int]] = {
    ("lowest", "random"): 0,
    ("lowest", "no graph"): 1,
    ("middle", "random"): 2,
    ("middle", "no graph"): 3,
    ("highest", "random"): 4,
    ("highest", "no graph"): 5,
    ("highest", "per-drug DR-learner"): 6,
}


def tercile_row(name: str, contrast: str) -> TercileRow:
    """One panel-b row, keyed by the tercile and the contrast it is read against."""

    if (name, contrast) not in TERCILE_LOOKUP:
        raise KeyError(f"no tercile row for {name!r} against {contrast!r}")
    return TABLE_3B_TERCILES[TERCILE_LOOKUP[(name, contrast)]]
