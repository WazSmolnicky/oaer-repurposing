"""The state, decision and candidate schemas shared by every module.

One decision unit is a patient-line. Its state is the quadruple (molecular,
pathology, clinical, line) that Sec. 2.3 writes as ``(Mi, Pi, Ci, Li)``; the
prescribing set ``W~_{d,i}`` is the covariate block the propensity model reads,
and the catalogue ``D`` holds the candidate agents with ``|D| = K``.

Ref: Sec. 2.3 (problem formulation), Sec. 2.5 (time zero, grace period).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

# The line-specific grace window, in days, inside which a clone must start one of
# its assigned strategy's agents or be censored as a deviation.
GRACE_DAYS: Final[Mapping[str, int]] = {"first": 90, "second": 60, "third_plus": 30}

# Line-specific restricted-time horizons on the month scale, set from the
# real-world median progression-free survival of the disease rather than from
# trial medians.
LINE_HORIZON_MONTHS: Final[Mapping[str, float]] = {"first": 8.0, "second": 6.5, "third_plus": 5.5}

DAYS_PER_MONTH: Final[float] = 30.4375

# The three sites and the region each sits in. Site C is held out from training.
SITE_REGION: Final[Mapping[str, str]] = {"A": "I", "B": "I", "C": "II"}


class MolecularStratum(str, Enum):
    """The exhaustive and mutually exclusive molecular partition of Sec. 1.5."""

    RAS_MUTANT_MSS = "ras_mutant_mss"
    RAS_BRAF_WILD_TYPE_MSS = "ras_braf_wild_type_mss"
    BRAF_V600E = "braf_v600e"
    MSI_HIGH = "msi_high"
    OTHER = "other_including_not_fully_typed"


class LineOfTherapy(str, Enum):
    FIRST = "first"
    SECOND = "second"
    THIRD_PLUS = "third_plus"

    @property
    def grace_days(self) -> int:
        return GRACE_DAYS[self.value]

    @property
    def horizon_months(self) -> float:
        return LINE_HORIZON_MONTHS[self.value]


class Sidedness(str, Enum):
    RIGHT = "right_sided"
    LEFT = "left_sided_including_rectum"


class MetastaticPattern(str, Enum):
    LIVER_LIMITED = "liver_limited"
    LUNG_ONLY = "lung_only"
    MULTI_ORGAN = "multi_organ"


class Presentation(str, Enum):
    SYNCHRONOUS = "synchronous"
    METACHRONOUS = "metachronous"


class OperatorStatus(int, Enum):
    """Eastern Cooperative Oncology Group performance status, collapsed at two."""

    ZERO_ONE = 0
    TWO_OR_ABOVE = 1


class ModalityAvailability(str, Enum):
    ALL_THREE = "all_three"
    PARTIAL = "at_least_one_missing"


class Domain(str, Enum):
    """How far a stratum sits from the development distribution.

    The label is explicit because a predominantly non-metastatic public cohort is
    an adjacent-domain check and never evidence about metastatic disease.
    """

    DEVELOPMENT = "development"
    IN_DOMAIN = "in-domain"
    SHIFTED = "shifted"
    STRESS = "stress test"
    ADJACENT = "adjacent, non-metastatic"
    METASTATIC = "metastatic"
    SANITY = "sanity probe only"


class SlideSource(str, Enum):
    PRIMARY_RESECTION = "primary_resection"
    PRIMARY_BIOPSY = "primary_biopsy"
    METASTASIS = "metastasis"


class Split(str, Enum):
    DEV_SITE_A = "dev_site_a"
    DEV_SITE_B = "dev_site_b"
    HELD_OUT_SITE_C = "held_out_site_c"
    PROSPECTIVE = "prospective"


@dataclass(frozen=True)
class MolecularPanel:
    """The typed molecular layer of one decision.

    ``typed`` distinguishes an assayed wild type from an unassayed alteration: a
    record with no result for ERBB2 or KRAS G12C is recorded as unassayed and is
    not folded into the wild-type stratum, which is why the partition is
    exhaustive over typed records and carries a residual block for the rest.
    """

    ras_mutant: bool
    braf_v600e: bool
    msi_high: bool
    pi3k_altered: bool
    erbb2_amplified: bool | None
    kras_g12c: bool | None
    assay_panel: str
    microsatellite_assay: str

    @property
    def stratum(self) -> MolecularStratum:
        """The molecular stratum, in the partition's own precedence order.

        The microsatellite-high and BRAF strata are read first because they are
        assayed independently of the extended panel; the residual block then takes
        every record that is not fully typed, so an unassayed ERBB2 or KRAS G12C
        record is never counted as wild type.
        """

        if self.msi_high:
            return MolecularStratum.MSI_HIGH
        if self.braf_v600e:
            return MolecularStratum.BRAF_V600E
        if not self.fully_typed:
            return MolecularStratum.OTHER
        if self.ras_mutant:
            return MolecularStratum.RAS_MUTANT_MSS
        return MolecularStratum.RAS_BRAF_WILD_TYPE_MSS

    @property
    def fully_typed(self) -> bool:
        return self.erbb2_amplified is not None and self.kras_g12c is not None


@dataclass(frozen=True)
class PathologyProfile:
    """The fixed slide embedding and the morphological lexicon of one decision."""

    embedding: tuple[float, ...]
    tile_count: int
    slide_source: SlideSource
    tumour_purity: float
    morphological_features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ClinicalRecord:
    """The health timeline of one decision, on the month scale."""

    age_years: float
    female: bool
    ecog: OperatorStatus
    sidedness: Sidedness
    pattern: MetastaticPattern
    presentation: Presentation
    line: LineOfTherapy
    prior_lines: int
    organ_sites: int
    albumin_g_per_l: float
    ldh_ratio_to_upper_limit: float
    cea_ng_per_ml: float
    comorbidity_count: int

    @property
    def reference_vector(self) -> tuple[float, ...]:
        """The clinical-covariate block of the prognostic reference score."""

        return (
            float(self.ecog.value),
            1.0 if self.sidedness is Sidedness.RIGHT else 0.0,
            float(self.line is LineOfTherapy.THIRD_PLUS),
            float(self.organ_sites),
            self.ldh_ratio_to_upper_limit,
            self.albumin_g_per_l / 40.0,
            math.log1p(self.cea_ng_per_ml),
            float(self.comorbidity_count),
            (self.age_years - 66.0) / 10.0,
            float(self.prior_lines),
        )


@dataclass(frozen=True)
class DecisionState:
    """``(Mi, Pi, Ci, Li)`` -- everything the score conditions on."""

    molecular: MolecularPanel
    pathology: PathologyProfile
    clinical: ClinicalRecord

    @property
    def line(self) -> LineOfTherapy:
        return self.clinical.line

    @property
    def horizon_months(self) -> float:
        return self.clinical.line.horizon_months


@dataclass(frozen=True)
class Candidate:
    """One entry of the candidate catalogue ``D``.

    ``approved_non_oncology`` is the catalogue's admission test: an agent enters
    when at least one of its approved indications is non-oncological. Agents
    approved for oncology applications only are excluded.
    """

    identifier: str
    label: str
    drug_class: str
    active_comparator: str
    approved_indications: tuple[str, ...]
    targets: tuple[str, ...]
    knowledge_graph_degree: int
    oncology_only: bool

    @property
    def admitted(self) -> bool:
        return not self.oncology_only and bool(self.approved_indications)


@dataclass(frozen=True)
class Decision:
    """One patient-line decision with its observed follow-up.

    ``exposure`` maps a candidate identifier to the indicator that the candidate
    was newly initiated inside the line's grace window after time zero. ``follow_up``
    is the observed restricted time and ``event`` the indicator that progression or
    death was seen before the horizon.
    """

    decision_id: str
    patient_id: str
    site: str
    split: Split
    state: DecisionState
    prescribing_set: Mapping[str, float]
    exposure: Mapping[str, bool]
    follow_up_months: float
    event: bool
    backbone_regimen: str
    objective_response: bool
    modalities_available: ModalityAvailability = ModalityAvailability.ALL_THREE

    @property
    def region(self) -> str:
        return SITE_REGION[self.site]

    @property
    def horizon_months(self) -> float:
        return self.state.horizon_months

    @property
    def restricted_time(self) -> float:
        return min(self.follow_up_months, self.horizon_months)


@dataclass(frozen=True)
class DecisionBatch:
    """A homogeneous collection of decisions that shares an evaluation pass."""

    decisions: tuple[Decision, ...]

    def __len__(self) -> int:
        return len(self.decisions)

    def by_site(self) -> dict[str, list[Decision]]:
        grouped: dict[str, list[Decision]] = {}
        for decision in self.decisions:
            grouped.setdefault(decision.site, []).append(decision)
        return grouped

    def by_line(self) -> dict[str, list[Decision]]:
        grouped: dict[str, list[Decision]] = {}
        for decision in self.decisions:
            grouped.setdefault(decision.state.line.value, []).append(decision)
        return grouped

    def by_stratum(self) -> dict[MolecularStratum, list[Decision]]:
        grouped: dict[MolecularStratum, list[Decision]] = {}
        for decision in self.decisions:
            grouped.setdefault(decision.state.molecular.stratum, []).append(decision)
        return grouped

    def exposures_for(self, candidate: str) -> Sequence[bool]:
        return tuple(bool(decision.exposure.get(candidate, False)) for decision in self.decisions)
