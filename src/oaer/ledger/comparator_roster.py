"""The comparison roster of Table 2, as data.

Two panels. Panel A is the candidate ranking inside the eligible set and is read
in four currencies: Hits@20, nDCG@20, the doubly robust top-three policy value in
months, and the agreement stratum's hazard ratio. Panel B is overall-survival
discrimination on the same records and is read as a concordance index, a
progression-free concordance, an expected calibration error and a twelve-month
AUROC. The two panels carry different columns rather than empty cells because the
metrics are undefined in opposite directions: a ranking metric is undefined for a
prognostic model and a concordance index for a ranker that emits no risk.

Every row also records how it was produced: a transfer row is a disease-level
model scored on this cohort without cohort-specific fitting, an unmarked row is
fitted on these records, and a re-implemented row had no code released. The
headline comparison is restricted to the unmarked rows.

Ref: Table 2 panels A and B and its provenance key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

UNMARKED: Final[str] = "fitted on these records"
AUTHORS_CODE: Final[str] = "authors' code run by us"
REIMPLEMENTED: Final[str] = "re-implemented by us, no code released"
STANDARD: Final[str] = "standard implementation run by us"
CONTROL: Final[str] = "control defined in Methods"

# The ten ``z`` rows: a disease-level model scored on this cohort without any
# cohort-specific fitting. They are exactly the three families of Sec. 1.3 whose
# members all carry the transfer mark, so the mark is recorded once per family
# rather than repeated on every row.
TRANSFER_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "network proximity and diffusion",
        "knowledge-graph link prediction",
        "inductive knowledge-graph foundation models",
    }
)


@dataclass(frozen=True)
class RankerRow:
    """One row of Table 2 panel A."""

    family: str
    method: str
    hits_at_20: float
    ndcg_at_20: float
    policy_value_months: float
    agreement: float
    pfs_hazard_ratio: float | None
    seed_standard_deviation: float | None
    p_vs_oaer: float | None
    provenance: str


TABLE_2A_RANKERS: Final[tuple[RankerRow, ...]] = (
    RankerRow(
        "network proximity and diffusion",
        "Network proximity",
        17.2,
        0.281,
        4.9,
        0.98,
        None,
        0.031,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "network proximity and diffusion",
        "Metapath ranker",
        18.6,
        0.294,
        5.2,
        0.96,
        None,
        0.028,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "network proximity and diffusion",
        "DREAMwalk",
        19.4,
        0.302,
        5.1,
        0.97,
        None,
        0.024,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "knowledge-graph link prediction",
        "Classical KG scorers, best of six",
        20.1,
        0.311,
        5.3,
        0.95,
        None,
        0.022,
        None,
        STANDARD,
    ),
    RankerRow(
        "knowledge-graph link prediction",
        "TxGNN",
        21.3,
        0.324,
        5.6,
        0.96,
        None,
        0.019,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "knowledge-graph link prediction",
        "CellAwareGNN",
        22.0,
        0.331,
        5.4,
        0.94,
        None,
        0.017,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "knowledge-graph link prediction",
        "BioPathNet",
        22.8,
        0.339,
        5.7,
        0.93,
        None,
        0.015,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "inductive knowledge-graph foundation models",
        "ULTRA, fine-tuned",
        23.5,
        0.346,
        5.6,
        0.93,
        None,
        0.014,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "inductive knowledge-graph foundation models",
        "MOTIF, fine-tuned",
        24.2,
        0.353,
        5.7,
        0.92,
        None,
        0.012,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "inductive knowledge-graph foundation models",
        "TRIX, fine-tuned",
        24.9,
        0.361,
        5.8,
        0.92,
        None,
        0.011,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "patient-conditioned rankers",
        "Patient-scalar relation modulation",
        26.4,
        0.378,
        5.9,
        0.91,
        1.34,
        0.009,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "patient-conditioned rankers",
        "Patient-specific link loss",
        27.1,
        0.386,
        6.0,
        0.90,
        1.21,
        0.008,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "patient-conditioned rankers",
        "Patient-specific ranking loss",
        27.8,
        0.393,
        6.1,
        0.89,
        1.18,
        0.007,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "New-user active-comparator PS + Cox",
        28.6,
        0.402,
        6.2,
        0.88,
        1.05,
        0.006,
        None,
        STANDARD,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "Causal survival forest, T- and X-learner",
        29.4,
        0.412,
        6.3,
        0.87,
        1.29,
        0.005,
        None,
        STANDARD,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "Per-drug DR-learner, no graph prior",
        30.1,
        0.421,
        6.4,
        0.86,
        1.11,
        0.004,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "CaML-shape meta-learned estimator",
        31.8,
        0.437,
        6.5,
        0.85,
        1.16,
        0.004,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "KG-TREAT-shape factual heads",
        30.9,
        0.429,
        6.4,
        0.86,
        1.09,
        0.005,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "AACE-shape DR-learner on embeddings",
        31.2,
        0.432,
        6.4,
        0.86,
        1.13,
        0.005,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "CauReL-class counterfactual estimator",
        29.9,
        0.418,
        6.3,
        0.87,
        1.24,
        0.006,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "STEDR-class subgroup emulation",
        29.2,
        0.409,
        6.2,
        0.88,
        1.19,
        0.006,
        None,
        REIMPLEMENTED,
    ),
    RankerRow(
        "causal machine-learning effect estimators",
        "High-throughput emulation screen",
        28.1,
        0.396,
        6.1,
        0.90,
        0.98,
        0.008,
        None,
        AUTHORS_CODE,
    ),
    RankerRow(
        "controls",
        "Random input / pooled-effect / prescribing-prevalence ranking",
        8.6,
        0.147,
        4.6,
        1.01,
        0.74,
        None,
        0.001,
        CONTROL,
    ),
    RankerRow(
        "same target",
        "OAER head on TxGNN, MOTIF or ULTRA",
        39.4,
        0.581,
        7.1,
        0.74,
        1.07,
        None,
        0.041,
        AUTHORS_CODE,
    ),
    RankerRow("deployed", "OAER", 41.7, 0.612, 7.4, 0.71, 1.12, None, None, UNMARKED),
)


@dataclass(frozen=True)
class PrognosticRow:
    """One row of Table 2 panel B."""

    family: str
    method: str
    os_c_index: tuple[float, float, float]
    pfs_c_index: float
    expected_calibration_error: float
    auroc_12_month: float
    seed_standard_deviation: float | None
    p_vs_oaer: float | None
    provenance: str


TABLE_2B_PROGNOSTIC: Final[tuple[PrognosticRow, ...]] = (
    PrognosticRow(
        "multimodal pathology-omics-clinical models",
        "SurvPath",
        (0.682, 0.651, 0.713),
        0.634,
        0.041,
        0.744,
        0.021,
        0.006,
        AUTHORS_CODE,
    ),
    PrognosticRow(
        "multimodal pathology-omics-clinical models",
        "PORPOISE",
        (0.671, 0.639, 0.703),
        0.622,
        0.046,
        0.731,
        0.024,
        0.007,
        AUTHORS_CODE,
    ),
    PrognosticRow(
        "multimodal pathology-omics-clinical models",
        "ABMIL and TransMIL, slides only",
        (0.648, 0.615, 0.681),
        0.597,
        0.053,
        0.702,
        0.029,
        0.004,
        STANDARD,
    ),
    PrognosticRow(
        "multimodal pathology-omics-clinical models",
        "HONeYBEE-style integration",
        (0.688, 0.657, 0.719),
        0.641,
        0.039,
        0.751,
        0.019,
        0.005,
        AUTHORS_CODE,
    ),
    PrognosticRow(
        "electronic-record foundation-model encoders",
        "Language-model record encoder",
        (0.694, 0.664, 0.724),
        0.648,
        0.036,
        0.758,
        0.017,
        0.005,
        AUTHORS_CODE,
    ),
    PrognosticRow(
        "electronic-record foundation-model encoders",
        "Generative patient-timeline model",
        (0.679, 0.648, 0.710),
        0.631,
        0.043,
        0.741,
        0.022,
        0.006,
        AUTHORS_CODE,
    ),
    PrognosticRow(
        "electronic-record foundation-model encoders",
        "Count-based features with gradient boosting",
        (0.662, 0.630, 0.694),
        0.611,
        0.049,
        0.719,
        0.016,
        0.004,
        STANDARD,
    ),
    PrognosticRow(
        "reproduced prognostic reference scores",
        "Koehne classification, 2002",
        (0.618, 0.585, 0.651),
        0.571,
        0.061,
        0.664,
        None,
        0.001,
        REIMPLEMENTED,
    ),
    PrognosticRow(
        "reproduced prognostic reference scores",
        "GERCOR simplified model, 2011",
        (0.634, 0.601, 0.667),
        0.588,
        0.057,
        0.681,
        None,
        0.001,
        REIMPLEMENTED,
    ),
    PrognosticRow(
        "reproduced prognostic reference scores",
        "Real-life multicenter Cox model, 2020",
        (0.658, 0.625, 0.691),
        0.612,
        0.052,
        0.706,
        None,
        0.001,
        REIMPLEMENTED,
    ),
    PrognosticRow(
        "reproduced prognostic reference scores",
        "FOLFIRI-aflibercept nomogram, 2019",
        (0.649, 0.616, 0.682),
        0.603,
        0.054,
        0.697,
        None,
        0.001,
        REIMPLEMENTED,
    ),
    PrognosticRow(
        "reproduced prognostic reference scores",
        "Fruquintinib nomogram, 2024, third line or later only",
        (0.641, 0.607, 0.675),
        0.594,
        0.058,
        0.688,
        None,
        0.001,
        REIMPLEMENTED,
    ),
    PrognosticRow(
        "deployed", "OAER", (0.746, 0.718, 0.774), 0.703, 0.028, 0.812, 0.014, None, UNMARKED
    ),
)

DEPLOYED_RANKING: Final[RankerRow] = TABLE_2A_RANKERS[-1]
DEPLOYED_PROGNOSIS: Final[PrognosticRow] = TABLE_2B_PROGNOSTIC[-1]


def family_members(family: str) -> tuple[RankerRow, ...]:
    return tuple(row for row in TABLE_2A_RANKERS if row.family == family)


def is_transfer(row: RankerRow) -> bool:
    """Whether a row carries the ``z`` transfer mark."""

    return row.family in TRANSFER_FAMILIES


def best_unmarked_ranker() -> RankerRow:
    """The strongest comparator row that was fitted on these records.

    Sec. 1.3 restricts its headline comparison to the rows that are not transfer
    rows, and reports the ten ``z`` rows as a reference instead. The deployed row
    is the model under comparison, so it is not its own comparator.
    """

    candidates = [row for row in TABLE_2A_RANKERS[:-1] if not is_transfer(row)]
    if not candidates:
        raise ValueError("the roster carries no comparator row fitted on these records")
    return max(candidates, key=lambda row: row.hits_at_20)


def transfer_rows() -> tuple[RankerRow, ...]:
    """The ten ``z`` rows, which carry no cohort-specific fitting."""

    return tuple(row for row in TABLE_2A_RANKERS if is_transfer(row))


def reimplemented_rows() -> tuple[RankerRow, ...]:
    return tuple(row for row in TABLE_2A_RANKERS if row.provenance == REIMPLEMENTED)


def ranking_families() -> tuple[str, ...]:
    seen: list[str] = []
    for row in TABLE_2A_RANKERS:
        if row.family not in seen:
            seen.append(row.family)
    return tuple(seen)


def prognostic_families() -> tuple[str, ...]:
    seen: list[str] = []
    for row in TABLE_2B_PROGNOSTIC:
        if row.family not in seen:
            seen.append(row.family)
    return tuple(seen)
