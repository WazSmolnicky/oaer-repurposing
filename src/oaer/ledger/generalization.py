"""The generalisation ledger of Table 5, as data.

Every row is a change from a matched internal reference: development performance
recomputed with exactly the modalities that row's cohort carries withheld, so a
public-cohort row measures domain shift conditional on the modalities available
rather than domain shift confounded with a modality ablation. The site rows are a
consistency read-out and never per-site significance tests; with three sites their
Cochran statistic has two degrees of freedom, so the heterogeneity share is read
for direction rather than magnitude. A public cohort that is predominantly
non-metastatic is an adjacent-domain check and never evidence about metastatic
disease.

Ref: Table 5 panels a-e and its caption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

MOLECULAR: Final[str] = "mol."
PATHOLOGY: Final[str] = "path."
RECORD: Final[str] = "rec."


@dataclass(frozen=True)
class GeneralizationRow:
    """One stratum of Table 5."""

    stratum: str
    panel: str
    domain: str
    modalities: tuple[str, ...]
    records: int
    os_c_index: float | None
    hits_at_20: float | None
    delta_vs_matched_reference: float | None
    abstention_percent: float | None
    note: str = ""


TABLE_5_STRATA: Final[tuple[GeneralizationRow, ...]] = (
    GeneralizationRow(
        "Site A",
        "sites and regions",
        "development",
        (MOLECULAR, PATHOLOGY, RECORD),
        3412,
        0.741,
        40.2,
        -0.003,
        26.8,
    ),
    GeneralizationRow(
        "Site B",
        "sites and regions",
        "development",
        (MOLECULAR, PATHOLOGY, RECORD),
        2938,
        0.736,
        39.6,
        -0.006,
        25.1,
    ),
    GeneralizationRow(
        "Site C",
        "sites and regions",
        "held out from training",
        (MOLECULAR, PATHOLOGY, RECORD),
        1612,
        0.722,
        37.9,
        -0.019,
        29.3,
    ),
    GeneralizationRow(
        "Region I",
        "sites and regions",
        "pooled",
        (MOLECULAR, PATHOLOGY, RECORD),
        6350,
        0.739,
        40.0,
        -0.004,
        26.1,
    ),
    GeneralizationRow(
        "Region II",
        "sites and regions",
        "pooled",
        (MOLECULAR, PATHOLOGY, RECORD),
        1612,
        0.722,
        37.9,
        -0.019,
        29.3,
    ),
    GeneralizationRow(
        "Scanner vendor 1",
        "acquisition strata",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        3882,
        0.740,
        39.9,
        -0.005,
        26.4,
    ),
    GeneralizationRow(
        "Scanner vendor 2",
        "acquisition strata",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        2904,
        0.735,
        39.1,
        -0.008,
        27.2,
    ),
    GeneralizationRow(
        "Scanner vendor 3",
        "acquisition strata",
        "shifted",
        (MOLECULAR, PATHOLOGY, RECORD),
        1176,
        0.726,
        38.2,
        -0.013,
        28.6,
    ),
    GeneralizationRow(
        "Narrow sequencing panel",
        "acquisition strata",
        "shifted",
        (MOLECULAR, PATHOLOGY, RECORD),
        1208,
        0.729,
        38.6,
        -0.011,
        28.1,
    ),
    GeneralizationRow(
        "Intermediate sequencing panel",
        "acquisition strata",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        4022,
        0.739,
        39.8,
        -0.005,
        26.3,
    ),
    GeneralizationRow(
        "Broad sequencing panel",
        "acquisition strata",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        2732,
        0.742,
        40.3,
        -0.003,
        25.7,
    ),
    GeneralizationRow(
        "Microsatellite status by PCR",
        "acquisition strata",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        5614,
        0.740,
        40.0,
        -0.004,
        26.0,
    ),
    GeneralizationRow(
        "Microsatellite status by IHC",
        "acquisition strata",
        "shifted",
        (MOLECULAR, PATHOLOGY, RECORD),
        2348,
        0.731,
        38.9,
        -0.011,
        27.9,
    ),
    GeneralizationRow(
        "Worst-case input perturbation",
        "acquisition strata",
        "stress test",
        (MOLECULAR, PATHOLOGY, RECORD),
        7962,
        0.712,
        35.4,
        -0.041,
        31.2,
    ),
    GeneralizationRow(
        "Development window",
        "temporal separation",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        5204,
        0.740,
        40.1,
        -0.002,
        26.0,
    ),
    GeneralizationRow(
        "Prospective window, post-freeze",
        "temporal separation",
        "temporally shifted",
        (MOLECULAR, PATHOLOGY, RECORD),
        2714,
        0.728,
        38.4,
        -0.016,
        28.4,
    ),
    GeneralizationRow(
        "TCGA-COADREAD",
        "independent public cohorts",
        "adjacent, non-metastatic",
        (MOLECULAR, PATHOLOGY),
        296,
        0.681,
        None,
        -0.021,
        None,
        "no candidate exposure is recorded, so no ranking or abstention read-out exists",
    ),
    GeneralizationRow(
        "CPTAC-COAD",
        "independent public cohorts",
        "adjacent, proteogenomic",
        (MOLECULAR,),
        110,
        None,
        None,
        None,
        None,
        "the open-tier release carries no follow-up fields, so no concordance index is computed",
    ),
    GeneralizationRow(
        "MSK-IMPACT colorectal",
        "independent public cohorts",
        "metastatic, clinicogenomic",
        (MOLECULAR, RECORD),
        1134,
        0.668,
        None,
        -0.027,
        None,
    ),
    GeneralizationRow(
        "MSK-CHORD colorectal",
        "independent public cohorts",
        "metastatic, real-world",
        (MOLECULAR, RECORD),
        2316,
        0.672,
        None,
        -0.024,
        None,
    ),
    GeneralizationRow(
        "SurGen, 426 survival cases",
        "independent public cohorts",
        "adjacent, predominantly non-metastatic",
        (PATHOLOGY,),
        426,
        0.681,
        None,
        -0.031,
        None,
    ),
    GeneralizationRow(
        "SurGen, metastatic subset of the 426",
        "independent public cohorts",
        "metastatic, pathology only",
        (PATHOLOGY,),
        168,
        0.669,
        None,
        -0.034,
        None,
    ),
    GeneralizationRow(
        "GSE39582",
        "independent public cohorts",
        "adjacent, transcriptomic",
        (MOLECULAR,),
        585,
        0.652,
        None,
        -0.038,
        None,
    ),
    GeneralizationRow(
        "Microsatellite probe, TCGA-CRC-DX",
        "independent public cohorts",
        "sanity probe only",
        (PATHOLOGY,),
        4112,
        None,
        None,
        None,
        None,
        "label AUROC 0.843 against chance; no survival or ranking read-out",
    ),
    GeneralizationRow(
        "Primary resection",
        "slide source",
        "in-domain",
        (MOLECULAR, PATHOLOGY, RECORD),
        4506,
        0.738,
        39.4,
        -0.006,
        26.9,
    ),
    GeneralizationRow(
        "Primary biopsy",
        "slide source",
        "shifted",
        (MOLECULAR, PATHOLOGY, RECORD),
        2118,
        0.724,
        37.6,
        -0.013,
        28.8,
    ),
    GeneralizationRow(
        "Metastasis",
        "slide source",
        "shifted",
        (MOLECULAR, PATHOLOGY, RECORD),
        1338,
        0.719,
        36.8,
        -0.016,
        29.6,
    ),
)

HETEROGENEITY: Final[dict[str, tuple[float, float]]] = {
    "os_c_index": (0.0, 0.71),
    "hits_at_20": (0.0, 0.48),
}

SITE_HETEROGENEITY_TABLE_1B: Final[dict[str, tuple[float, float]]] = {
    "pfs_hazard_ratio": (0.0, 0.62),
    "os_hazard_ratio": (0.0, 0.58),
    "c_index": (0.0, 0.71),
}

MICROSATELLITE_PROBE_AUROC: Final[float] = 0.843
ABSTENTION_PERCENT_POOLED: Final[float] = 28.4
ACTED_RESTRICTED_MEAN_PFS: Final[float] = 6.1
DEFERRED_RESTRICTED_MEAN_PFS: Final[float] = 5.4
PUBLIC_COHORT_AGREEMENT: Final[float] = 0.681

PAIP2020_EXCLUSION: Final[str] = (
    "PAIP2020 is deliberately absent: its download sits behind a click-through "
    "research-use gate, so it is never used as an evidence source."
)


def panel_rows(panel: str) -> tuple[GeneralizationRow, ...]:
    return tuple(row for row in TABLE_5_STRATA if row.panel == panel)


def panels() -> tuple[str, ...]:
    seen: list[str] = []
    for row in TABLE_5_STRATA:
        if row.panel not in seen:
            seen.append(row.panel)
    return tuple(seen)


def modality_subset() -> tuple[str, ...]:
    """The three-modality set every development row carries."""

    return (MOLECULAR, PATHOLOGY, RECORD)


def availability_label(modalities: tuple[str, ...]) -> str:
    """A readable label for a row's modality set."""

    return " and ".join(modalities)


def matched_reference_rows() -> tuple[GeneralizationRow, ...]:
    """The rows a public cohort is compared against: development, same modalities."""

    return tuple(row for row in TABLE_5_STRATA if row.domain == "development")


def shift_rows() -> tuple[GeneralizationRow, ...]:
    """The rows the article labels as shifted rather than in-domain."""

    return tuple(
        row
        for row in TABLE_5_STRATA
        if row.domain in {"shifted", "temporally shifted", "stress test"}
    )


def adjacent_domain_rows() -> tuple[GeneralizationRow, ...]:
    """The rows that are adjacent-domain checks rather than metastatic evidence."""

    return tuple(row for row in TABLE_5_STRATA if row.domain.startswith("adjacent"))


def worst_case_delta() -> float:
    values = [
        row.delta_vs_matched_reference
        for row in TABLE_5_STRATA
        if row.delta_vs_matched_reference is not None
    ]
    return min(values) if values else float("nan")


def site_rows() -> tuple[GeneralizationRow, ...]:
    return tuple(row for row in TABLE_5_STRATA if row.stratum in {"Site A", "Site B", "Site C"})
