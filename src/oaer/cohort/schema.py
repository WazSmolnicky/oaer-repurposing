"""The record schema the cohort declares.

The article states the minimum record data it needs -- the molecular layer, the
pathology layer, the structured timeline and the line of therapy -- and states
that a record with an unassayed alteration is recorded as unassayed rather than
folded into the wild-type stratum. This module is that contract: the columns each
layer must supply, their domains, and the validation a record has to pass before
it enters the pipeline.

Ref: Sec. 2.1 (minimum record data), Sec. 2.3 (the state quadruple).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from oaer.support.types import (
    LineOfTherapy,
    MetastaticPattern,
    ModalityAvailability,
    MolecularPanel,
    OperatorStatus,
    Presentation,
    Sidedness,
    SlideSource,
)

DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "decision_id",
    "patient_id",
    "site",
    "line_of_therapy",
    "age_years",
    "female",
    "ecog",
    "sidedness",
    "metastatic_pattern",
    "presentation",
    "prior_lines",
    "organ_sites",
    "albumin_g_per_l",
    "ldh_ratio_to_upper_limit",
    "cea_ng_per_ml",
    "comorbidity_count",
    "ras_mutant",
    "braf_v600e",
    "msi_high",
    "pi3k_altered",
    "erbb2_amplified",
    "kras_g12c",
    "assay_panel",
    "microsatellite_assay",
    "slide_source",
    "tile_count",
    "tumour_purity",
    "backbone_regimen",
    "follow_up_months",
    "event",
    "objective_response",
    "modality_availability",
)

MOLECULAR_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "ras_mutant",
    "braf_v600e",
    "msi_high",
    "pi3k_altered",
    "erbb2_amplified",
    "kras_g12c",
    "assay_panel",
    "microsatellite_assay",
)

ASSAY_PANELS: Final[tuple[str, ...]] = ("narrow", "intermediate", "broad")
MICROSATELLITE_ASSAYS: Final[tuple[str, ...]] = ("pcr", "ihc")
BACKBONE_REGIMENS: Final[tuple[str, ...]] = (
    "folfox",
    "folfiri",
    "folfoxiri",
    "capox",
    "trifluridine-tipiracil",
    "regorafenib",
)

SITE_DOMAIN: Final[tuple[str, ...]] = ("A", "B", "C")

_NUMERIC_BOUNDS: Final[Mapping[str, tuple[float, float]]] = {
    "age_years": (18.0, 105.0),
    "prior_lines": (0.0, 8.0),
    "organ_sites": (1.0, 6.0),
    "albumin_g_per_l": (15.0, 55.0),
    "ldh_ratio_to_upper_limit": (0.2, 8.0),
    "cea_ng_per_ml": (0.0, 5000.0),
    "comorbidity_count": (0.0, 12.0),
    "tile_count": (1.0, 200000.0),
    "tumour_purity": (0.0, 1.0),
    "follow_up_months": (0.0, 120.0),
}


class CohortSchemaError(ValueError):
    """Raised when a record violates the declared schema."""


@dataclass(frozen=True)
class RecordSchema:
    """The schema as a value, so a check can assert on it rather than on prose."""

    columns: tuple[str, ...] = DECISION_COLUMNS
    molecular_columns: tuple[str, ...] = MOLECULAR_PANEL_COLUMNS
    sites: tuple[str, ...] = SITE_DOMAIN
    lines: tuple[str, ...] = tuple(line.value for line in LineOfTherapy)
    slide_sources: tuple[str, ...] = tuple(source.value for source in SlideSource)
    patterns: tuple[str, ...] = tuple(pattern.value for pattern in MetastaticPattern)
    presentations: tuple[str, ...] = tuple(item.value for item in Presentation)
    sidedness: tuple[str, ...] = tuple(item.value for item in Sidedness)
    availability: tuple[str, ...] = tuple(item.value for item in ModalityAvailability)
    assay_panels: tuple[str, ...] = ASSAY_PANELS
    microsatellite_assays: tuple[str, ...] = MICROSATELLITE_ASSAYS
    backbone_regimens: tuple[str, ...] = BACKBONE_REGIMENS

    def required(self) -> tuple[str, ...]:
        return self.columns

    def missing(self, record: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(column for column in self.columns if column not in record)


def validate_record(record: Mapping[str, object], schema: RecordSchema | None = None) -> None:
    """Raise when a record breaks the declared contract."""

    contract = schema if schema is not None else RecordSchema()
    absent = contract.missing(record)
    if absent:
        raise CohortSchemaError(f"record is missing the column(s) {absent}")
    if record["site"] not in contract.sites:
        raise CohortSchemaError(f"site {record['site']!r} is outside the declared domain")
    if record["line_of_therapy"] not in contract.lines:
        raise CohortSchemaError(f"line {record['line_of_therapy']!r} is not declared")
    if record["slide_source"] not in contract.slide_sources:
        raise CohortSchemaError(f"slide source {record['slide_source']!r} is not declared")
    if record["metastatic_pattern"] not in contract.patterns:
        raise CohortSchemaError("metastatic pattern is outside the declared domain")
    if record["presentation"] not in contract.presentations:
        raise CohortSchemaError("presentation is outside the declared domain")
    if record["sidedness"] not in contract.sidedness:
        raise CohortSchemaError("sidedness is outside the declared domain")
    if record["modality_availability"] not in contract.availability:
        raise CohortSchemaError("modality availability is outside the declared domain")
    if record["assay_panel"] not in contract.assay_panels:
        raise CohortSchemaError("assay panel is outside the declared domain")
    if record["microsatellite_assay"] not in contract.microsatellite_assays:
        raise CohortSchemaError("microsatellite assay is outside the declared domain")
    if record["backbone_regimen"] not in contract.backbone_regimens:
        raise CohortSchemaError("backbone regimen is outside the declared domain")
    if bool(record["msi_high"]) and bool(record["ras_mutant"]):
        # The partition is mutually exclusive: the microsatellite-high stratum is
        # carried ahead of the RAS stratum, so a record cannot be in both.
        raise CohortSchemaError("the microsatellite-high stratum is not also RAS-mutant")
    for column, (low, high) in _NUMERIC_BOUNDS.items():
        value = float(record[column])  # type: ignore[arg-type]
        if not low <= value <= high:
            raise CohortSchemaError(f"{column}={value} is outside [{low}, {high}]")
    for column in ("erbb2_amplified", "kras_g12c"):
        flag = record[column]
        if flag is not None and not isinstance(flag, bool):
            raise CohortSchemaError(f"{column} must be a boolean or unassayed (None)")
    if record["event"] not in (True, False) and record["event"] not in (0, 1):
        raise CohortSchemaError("the event indicator must be boolean")
    if float(record["follow_up_months"]) <= 0.0:  # type: ignore[arg-type]
        raise CohortSchemaError("a decision unit must carry positive follow-up")


def panel_from_record(record: Mapping[str, object]) -> MolecularPanel:
    return MolecularPanel(
        ras_mutant=bool(record["ras_mutant"]),
        braf_v600e=bool(record["braf_v600e"]),
        msi_high=bool(record["msi_high"]),
        pi3k_altered=bool(record["pi3k_altered"]),
        erbb2_amplified=_optional_bool(record["erbb2_amplified"]),
        kras_g12c=_optional_bool(record["kras_g12c"]),
        assay_panel=str(record["assay_panel"]),
        microsatellite_assay=str(record["microsatellite_assay"]),
    )


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def modality_availability(record: Mapping[str, object]) -> ModalityAvailability:
    missing = (
        record.get("slide_source") is None
        or record.get("assay_panel") is None
        or float(record.get("albumin_g_per_l", 0.0)) <= 0.0  # type: ignore[arg-type]
    )
    return ModalityAvailability.PARTIAL if missing else ModalityAvailability.ALL_THREE


def assert_columns(records: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """The columns every record in a table carries, or raise on a mismatch."""

    if not records:
        return ()
    first = set(records[0])
    for record in records[1:]:
        if set(record) != first:
            raise CohortSchemaError("the table does not carry a single column set")
    return tuple(sorted(first))


def ecog_from_value(value: float) -> OperatorStatus:
    return OperatorStatus.ZERO_ONE if value < 2.0 else OperatorStatus.TWO_OR_ABOVE
