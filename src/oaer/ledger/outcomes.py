"""The article's reported outcome values, as data.

Every number here is a value the article prints. Nothing in this module computes
anything: it is the comparison ledger the reports read against, so that a printed
read-out can be placed beside the reported one without either being retyped. The
cohort those values came from is not redistributed, so no read-out in this package
is claimed to reproduce them; they are carried for comparison and every report
that prints them says so.

Ref: Table 1 panels a-f, Table 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class CohortCell:
    """One site's patient and decision counts in each arm."""

    site: str
    region: str
    retrospective_patients: int
    retrospective_decisions: int
    prospective_patients: int
    prospective_decisions: int
    follow_up_months: float
    molecular_modal_percent: float
    events: int


TABLE_1A_COHORT: Final[tuple[CohortCell, ...]] = (
    CohortCell("A", "I", 1847, 3412, 692, 1187, 32.6, 89.2, 2908),
    CohortCell("B", "I", 1592, 2938, 574, 983, 30.8, 86.1, 2401),
    CohortCell("C", "II (held out)", 879, 1612, 320, 544, 28.9, 84.7, 1286),
    CohortCell("pooled", "I and II", 4318, 7962, 1586, 2714, 30.4, 87.4, 6595),
)

WINDOW_RETROSPECTIVE: Final[str] = "2018-01-01 to 2023-06-30"
WINDOW_POOLED: Final[str] = "2018-01-01 to 2025-08-31"


@dataclass(frozen=True)
class PrimaryArmCell:
    """One site's row of Table 1 panel b."""

    site: str
    cloned_decisions: int
    weighted_events: int
    rmst_pfs_months: float
    pfs_hazard_ratio: tuple[float, float, float]
    rmst_os_months: float
    os_hazard_ratio: tuple[float, float, float]
    c_index: tuple[float, float]


TABLE_1B_PRIMARY: Final[tuple[PrimaryArmCell, ...]] = (
    PrimaryArmCell(
        "A", 851, 1061, 6.7, (0.61, 0.49, 0.76), 11.6, (0.73, 0.55, 0.97), (0.764, 0.671)
    ),
    PrimaryArmCell(
        "B", 693, 930, 6.3, (0.71, 0.56, 0.90), 11.1, (0.81, 0.61, 1.08), (0.741, 0.652)
    ),
    PrimaryArmCell(
        "C", 399, 511, 6.0, (0.78, 0.58, 1.05), 10.8, (0.86, 0.62, 1.19), (0.722, 0.641)
    ),
    PrimaryArmCell(
        "pooled", 1943, 2502, 6.4, (0.68, 0.55, 0.84), 11.3, (0.79, 0.63, 0.99), (0.746, 0.658)
    ),
)

TABLE_1B_EFFECTIVE_SIZES: Final[dict[str, object]] = {
    "top_agreement": 742,
    "no_initiation": 688,
    "policy_value_k1": (5.9, 5.2, 6.6),
    "policy_contrast_k3": (2.4, 0.9, 3.9),
    "follow_up_ended_before_grace_percent": 6.8,
    "clone_censoring_weight_median": 1.04,
    "clone_censoring_weight_99th": 2.61,
    "per_line_effective_sample_size": 412,
}


@dataclass(frozen=True)
class CandidateEmulation:
    """One row of Table 1 panel c."""

    candidate: str
    active_comparator: str
    exposed: int
    events: int
    pfs_hazard_ratio: tuple[float, float, float]
    os_hazard_ratio: tuple[float, float, float]
    e_value: float
    negative_control: float
    trimmed_percent: tuple[float, float] | None


TABLE_1C_EMULATIONS: Final[tuple[CandidateEmulation, ...]] = (
    CandidateEmulation(
        "Metformin, biguanide",
        "DPP-4 inhibitor",
        1286,
        412,
        (0.71, 0.62, 0.81),
        (0.83, 0.71, 0.97),
        1.94,
        1.01,
        (4.2, 4.2),
    ),
    CandidateEmulation(
        "Beta-blocker, hypertension only",
        "calcium-channel blocker",
        962,
        338,
        (0.76, 0.65, 0.89),
        (0.79, 0.67, 0.93),
        1.72,
        0.99,
        (3.6, 3.6),
    ),
    CandidateEmulation(
        "Aspirin, low-dose antiplatelet",
        "clopidogrel",
        534,
        176,
        (0.68, 0.55, 0.84),
        (0.86, 0.70, 1.06),
        2.06,
        1.03,
        (6.1, 6.1),
    ),
    CandidateEmulation(
        "Proton-pump inhibitor",
        "H2-receptor antagonist",
        1704,
        589,
        (1.18, 1.06, 1.31),
        (1.12, 1.01, 1.24),
        1.52,
        1.00,
        (2.4, 2.4),
    ),
    CandidateEmulation(
        "Prevalent-new-user sensitivity, chronic-medication subset",
        "not applicable",
        2118,
        741,
        (0.74, 0.66, 0.83),
        (0.85, 0.74, 0.98),
        1.81,
        1.02,
        (5.3, 5.3),
    ),
    CandidateEmulation(
        "Prescription-only sensitivity, aspirin and PPI",
        "not applicable",
        881,
        297,
        (0.69, 0.56, 0.85),
        (0.87, 0.71, 1.07),
        2.01,
        1.01,
        (5.7, 5.7),
    ),
    CandidateEmulation(
        "Further identified candidates",
        "per candidate",
        5908,
        1944,
        (0.77, 0.73, 0.81),
        (0.84, 0.79, 0.89),
        1.59,
        1.00,
        (7.6, 7.6),
    ),
    CandidateEmulation(
        "Pooled, identified set",
        "per candidate",
        11275,
        4497,
        (0.74, 0.69, 0.79),
        (0.82, 0.77, 0.88),
        1.87,
        1.01,
        (0.9, 7.6),
    ),
)


@dataclass(frozen=True)
class PathwayRow:
    """One row of Table 1 panel e."""

    pathway: str
    decisions: int
    share_percent: float
    pfs_events: int
    rmst_pfs_months: float
    pfs_hazard_ratio: float
    rmst_os_months: float
    objective_response_percent: float


TABLE_1E_PATHWAYS: Final[tuple[PathwayRow, ...]] = (
    PathwayRow("Top-ranked identified candidate started", 438, 22.5, 171, 7.3, 0.62, 12.4, 38.6),
    PathwayRow("Another top-3 identified candidate started", 296, 15.2, 124, 6.8, 0.71, 11.8, 35.2),
    PathwayRow("Identified candidate outside the top 3", 341, 17.5, 154, 6.1, 0.91, 11.1, 32.4),
    PathwayRow("No candidate, backbone only (reference)", 512, 26.4, 238, 5.6, 1.0, 10.4, 30.1),
    PathwayRow("De-prioritised agent co-prescribed", 168, 8.6, 79, 5.2, 1.13, 9.9, 28.7),
    PathwayRow(
        "No agent started: set empty or margin not cleared", 188, 9.7, 81, 5.4, 1.07, 10.2, 29.4
    ),
    PathwayRow(
        "Top agreement against discordant initiation", 734, 37.8, 295, 7.1, 0.74, 12.1, 37.1
    ),
)


@dataclass(frozen=True)
class RetrospectiveCell:
    """One row of Table 1 panel f."""

    site: str
    cloned_decisions: int
    weighted_events: int
    rmst_pfs_months: float
    pfs_hazard_ratio: tuple[float, float, float]
    rmst_os_months: float
    os_hazard_ratio: tuple[float, float, float]
    c_index: tuple[float, float]


TABLE_1F_RETROSPECTIVE: Final[tuple[RetrospectiveCell, ...]] = (
    RetrospectiveCell(
        "A", 2214, 4187, 6.5, (0.76, 0.69, 0.84), 11.0, (0.84, 0.75, 0.94), (0.734, 0.652)
    ),
    RetrospectiveCell(
        "B", 1806, 3391, 6.3, (0.79, 0.71, 0.88), 10.8, (0.86, 0.76, 0.97), (0.719, 0.641)
    ),
    RetrospectiveCell(
        "C, scored by the frozen A+B model",
        982,
        1904,
        6.0,
        (0.83, 0.73, 0.94),
        10.5,
        (0.89, 0.77, 1.03),
        (0.701, 0.629),
    ),
    RetrospectiveCell(
        "pooled", 5002, 9482, 6.3, (0.78, 0.72, 0.85), 10.8, (0.86, 0.79, 0.94), (0.724, 0.646)
    ),
)

TABLE_1_DIAGNOSTICS: Final[dict[str, object]] = {
    "sites_clearing_each_primary_criterion": 2,
    "pooled_e_value_point": (2.31, 1.83),
    "pooled_e_value_upper": (1.42, 1.19),
    "negative_control_pfs": (1.02, 0.94, 1.11),
    "negative_control_os": (0.98, 0.89, 1.08),
    "between_site_i2_pfs": (0.0, 0.62),
    "between_site_i2_os": (0.0, 0.58),
    "between_site_i2_c_index": (0.0, 0.71),
}


@dataclass(frozen=True)
class SubgroupCell:
    """One row of Table 4."""

    subgroup: str
    block: str
    retrospective_decisions: int
    prospective_decisions: int
    os_c_index: float
    pfs_hazard_ratio: tuple[float, float, float]
    deferral_percent: float


TABLE_4_SUBGROUPS: Final[tuple[SubgroupCell, ...]] = (
    SubgroupCell("All records", "whole cohort", 7962, 2714, 0.746, (0.68, 0.55, 0.84), 28.4),
    SubgroupCell("Right-sided", "sidedness", 3186, 1124, 0.732, (0.72, 0.56, 0.93), 30.1),
    SubgroupCell(
        "Left-sided, including rectum", "sidedness", 4776, 1590, 0.752, (0.66, 0.52, 0.84), 27.2
    ),
    SubgroupCell(
        "RAS-mutant, microsatellite stable",
        "molecular partition",
        3512,
        1196,
        0.738,
        (0.71, 0.56, 0.90),
        29.6,
    ),
    SubgroupCell(
        "RAS/BRAF wild-type, microsatellite stable",
        "molecular partition",
        1902,
        654,
        0.761,
        (0.64, 0.48, 0.85),
        24.8,
    ),
    SubgroupCell("BRAF V600E", "molecular partition", 862, 298, 0.724, (0.76, 0.55, 1.05), 33.4),
    SubgroupCell(
        "Microsatellite-instability-high",
        "molecular partition",
        486,
        168,
        0.788,
        (0.83, 0.56, 1.23),
        81.6,
    ),
    SubgroupCell(
        "Other, including records not fully typed",
        "molecular partition",
        1200,
        398,
        0.729,
        (0.74, 0.57, 0.96),
        31.7,
    ),
    SubgroupCell(
        "BRAF V600E and microsatellite-instability-high",
        "additional alterations",
        168,
        58,
        0.771,
        (0.81, 0.51, 1.29),
        78.4,
    ),
    SubgroupCell(
        "PI3K-pathway-altered, microsatellite stable",
        "additional alterations",
        1208,
        412,
        0.749,
        (0.63, 0.48, 0.83),
        26.1,
    ),
    SubgroupCell(
        "ERBB2-amplified, retrospective arm only",
        "additional alterations",
        246,
        0,
        0.741,
        (0.69, 0.46, 1.03),
        22.4,
    ),
    SubgroupCell(
        "KRAS G12C, retrospective arm only",
        "additional alterations",
        182,
        0,
        0.736,
        (0.72, 0.47, 1.10),
        23.8,
    ),
    SubgroupCell("First line", "line of therapy", 2682, 918, 0.758, (0.66, 0.50, 0.87), 25.3),
    SubgroupCell("Second line", "line of therapy", 2914, 996, 0.741, (0.70, 0.55, 0.89), 28.9),
    SubgroupCell(
        "Third line or later", "line of therapy", 2366, 800, 0.726, (0.73, 0.57, 0.94), 31.6
    ),
    SubgroupCell(
        "Liver-limited", "metastatic pattern", 3104, 1058, 0.751, (0.67, 0.52, 0.86), 26.4
    ),
    SubgroupCell("Lung-only", "metastatic pattern", 1286, 438, 0.762, (0.64, 0.47, 0.87), 25.1),
    SubgroupCell("Multi-organ", "metastatic pattern", 3572, 1218, 0.733, (0.72, 0.57, 0.91), 30.8),
    SubgroupCell("Synchronous presentation", "timing", 4618, 1574, 0.741, (0.69, 0.55, 0.87), 28.1),
    SubgroupCell(
        "Metachronous presentation", "timing", 3344, 1140, 0.748, (0.67, 0.52, 0.86), 28.8
    ),
    SubgroupCell(
        "All three modalities recorded",
        "modality availability",
        6958,
        2372,
        0.749,
        (0.67, 0.54, 0.83),
        27.6,
    ),
    SubgroupCell(
        "At least one modality missing",
        "modality availability",
        1004,
        342,
        0.706,
        (0.79, 0.58, 1.08),
        34.2,
    ),
    SubgroupCell(
        "Age under 65 years", "equity strata", 3514, 1198, 0.744, (0.68, 0.53, 0.87), 27.9
    ),
    SubgroupCell("Age 65-74 years", "equity strata", 2948, 1004, 0.749, (0.67, 0.52, 0.86), 28.1),
    SubgroupCell(
        "Age 75 years or older", "equity strata", 1500, 512, 0.738, (0.72, 0.54, 0.96), 30.1
    ),
    SubgroupCell("Female", "equity strata", 3586, 1222, 0.751, (0.66, 0.51, 0.85), 27.4),
    SubgroupCell("Male", "equity strata", 4376, 1492, 0.742, (0.70, 0.55, 0.89), 29.2),
    SubgroupCell("ECOG 0-1", "equity strata", 6214, 2118, 0.752, (0.66, 0.53, 0.82), 26.2),
    SubgroupCell("ECOG 2 or above", "equity strata", 1748, 596, 0.719, (0.79, 0.60, 1.04), 36.1),
)


def cohort_total(arm: str) -> int:
    """The pooled decision count of one arm, as Table 1 panel a prints it."""

    pooled = TABLE_1A_COHORT[-1]
    if arm == "retrospective":
        return pooled.retrospective_decisions
    if arm == "prospective":
        return pooled.prospective_decisions
    raise KeyError(f"{arm!r} is not one of the two arms")


def pathway_by_name(name: str) -> PathwayRow | None:
    for row in TABLE_1E_PATHWAYS:
        if row.pathway.startswith(name):
            return row
    return None


def subgroup_blocks() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for cell in TABLE_4_SUBGROUPS:
        grouped.setdefault(cell.block, []).append(cell.subgroup)
    return {block: tuple(names) for block, names in grouped.items()}
