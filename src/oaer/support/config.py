"""Protocol configuration.

Each experiment is one YAML file under ``protocols/``. Values the article reports
are carried verbatim and carry a ``# Ref:`` line at the point of declaration;
values the article does not report are declared engineering defaults and are
labelled as such. The loader is intentionally small: a nested mapping is walked
and coerced into the section dataclasses below, and unknown keys raise rather
than being dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class CohortConfig:
    """Cohort construction.

    The site fields are decision counts, because the reported marginals the
    builder holds are marginals over decisions; the patient totals are carried
    separately for the census.

    Ref: Table 1 panel a (site rows), Sec. 1.2 (prospective window).
    """

    site_a_decisions: int = 3412
    site_b_decisions: int = 2938
    site_c_decisions: int = 1612
    site_a_prospective_decisions: int = 1187
    site_b_prospective_decisions: int = 983
    site_c_prospective_decisions: int = 544
    retrospective_patients: int = 4318
    prospective_patients: int = 1586
    decisions_per_patient: float = 1.84
    candidate_count: int = 60
    exposed_candidates: int = 0
    profile_scale: float = 1.0


@dataclass(frozen=True)
class CatalogueConfig:
    """Candidate catalogue and knowledge-graph construction.

    Ref: Sec. 2.1 (the minimum record data), Sec. 2.2 (catalogue scope).
    """

    non_oncology_indications: int = 612
    oncology_only_excluded: int = 96
    relations: int = 7
    anchor_types: int = 4
    motif_orders: tuple[int, ...] = (2, 3)
    negative_samples: int = 12
    anchor_corruptions: int = 5


@dataclass(frozen=True)
class EmulationConfig:
    """Target-trial emulation and the cross-fitted pseudo-outcome.

    Ref: Algorithm 2; Methods Sec. 2.5.
    """

    trimming_epsilon: float = 0.05
    minimum_exposed: int = 40
    folds: int = 5
    censoring_floor: float = 0.05
    new_user_lookback_days: int = 365
    regularisation: float = 1.0
    horizons_months: tuple[float, float, float] = (8.0, 6.5, 5.5)
    placebo_candidates: int = 3


@dataclass(frozen=True)
class EncoderConfig:
    """The graph-pretrained patient-anchored representation (C2).

    Ref: Sec. 2.4 (relational encoder), Eq. (3)-(4).
    """

    embedding_dim: int = 64
    hidden_dim: int = 64
    layers: int = 2
    num_basis: int = 4
    dropout: float = 0.1
    graph_degree_scale: float = 8.0
    degree_transfer_gain: float = 1.0


@dataclass(frozen=True)
class HeadConfig:
    """The ranking head: route score plus pathology benefit modifier.

    Ref: Sec. 2.4 (ranking head and outcome anchoring), Eq. (5).
    """

    route_dim: int = 32
    modifier_dim: int = 16
    pathology_only_dim: int = 12
    l2_penalty: float = 1.0e-4
    outcome_weight_scale: float = 1.0


@dataclass(frozen=True)
class GateConfig:
    """The identification gate and the weighted conformal act-or-defer rule.

    Ref: Sec. 2.4, Eq. (1); Algorithm 3.
    """

    miscoverage_alpha: float = 0.1
    margin_delta: float = 0.0
    rank_depth_k: int = 3
    calibration_window_months: int = 12
    transport_clip: float = 10.0


@dataclass(frozen=True)
class FittingConfig:
    """Training schedule.

    The article reports the model's ranking and discrimination read-outs, not its
    schedule, so every field here is a declared engineering default.
    """

    stage1_epochs: int = 4
    stage2_epochs: int = 4
    stage3_epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 50
    total_steps: int = 3000
    grad_clip: float = 1.0
    precision: str = "fp32"
    device: str = "cpu"
    early_stop_patience: int = 8


@dataclass(frozen=True)
class ReadingConfig:
    """Read-out settings, including the fixed seed panel.

    Ref: Table 2 note (ten fixed seeds), Sec. 2.7 (five patient-level folds).
    """

    seeds: tuple[int, ...] = (11, 23, 37, 41, 59, 67, 73, 89, 97, 103)
    bootstrap_resamples: int = 400
    bootstrap_confidence: float = 0.95
    hits_at: int = 20
    ndcg_at: int = 20
    jaccard_top: int = 5
    landmark_months: float = 12.0


@dataclass(frozen=True)
class ComputeConfig:
    """The reported deployment footprint.

    The article gives the model's frozen-decision latency and nothing about the
    hardware that produced it, so the accelerator fields stay unset and the
    latency fields carry the reported measurement.

    Ref: Sec. 2 Discussion (ranking latency, 11 minutes; interquartile range 7-18).
    """

    accelerator: str = "not reported by the manuscript"
    peak_memory_gb: float | None = None
    wall_clock_per_decision_minutes: float = 11.0
    wall_clock_interquartile_low: float = 7.0
    wall_clock_interquartile_high: float = 18.0
    frozen_throughput_percent: float = 96.4
    notes: str = "no hardware figure is stated in the manuscript"


@dataclass(frozen=True)
class ProtocolConfig:
    """One protocol: every section, plus the run-level settings."""

    name: str
    seed: int
    site_holdout: str
    cohort: CohortConfig
    catalogue: CatalogueConfig
    emulation: EmulationConfig
    encoder: EncoderConfig
    head: HeadConfig
    gate: GateConfig
    fitting: FittingConfig
    readings: ReadingConfig
    compute: ComputeConfig
    raw: Mapping[str, Any]


_SECTION_TYPES: Mapping[str, type[Any]] = {
    "cohort": CohortConfig,
    "catalogue": CatalogueConfig,
    "emulation": EmulationConfig,
    "encoder": EncoderConfig,
    "head": HeadConfig,
    "gate": GateConfig,
    "fitting": FittingConfig,
    "readings": ReadingConfig,
    "compute": ComputeConfig,
}


def _coerce(section: type[T], payload: Mapping[str, Any]) -> T:
    known = {field.name for field in fields(section)}  # type: ignore[arg-type]
    unknown = sorted(set(payload) - known)
    if unknown:
        raise KeyError(f"{section.__name__} does not carry the key(s) {unknown}")
    values: dict[str, Any] = {}
    for field_info in fields(section):  # type: ignore[arg-type]
        if field_info.name not in payload:
            continue
        raw = payload[field_info.name]
        if field_info.type in {"tuple[float, ...]", "tuple[int, ...]"} and isinstance(raw, list):
            values[field_info.name] = tuple(raw)
        else:
            values[field_info.name] = raw
    return section(**values)


def load_protocol(path: str | Path) -> ProtocolConfig:
    """Read one protocol file into its section dataclasses."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} does not hold a mapping at the top level")
    return protocol_from_mapping(payload, name=payload.get("name", source.stem))


def protocol_from_mapping(payload: Mapping[str, Any], *, name: str) -> ProtocolConfig:
    sections: dict[str, Any] = {}
    for key, section in _SECTION_TYPES.items():
        body = payload.get(key, {})
        if not isinstance(body, Mapping):
            raise TypeError(f"section {key!r} must be a mapping")
        sections[key] = _coerce(section, body)
    return ProtocolConfig(
        name=str(name),
        seed=int(payload.get("seed", 20260831)),
        site_holdout=str(payload.get("site_holdout", "C")),
        raw=dict(payload),
        **sections,
    )


def merge_overrides(payload: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` CLI overrides onto a raw protocol mapping."""

    merged: dict[str, Any] = yaml.safe_load(yaml.safe_dump(dict(payload)))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} is not of the form key=value")
        dotted, literal = override.split("=", 1)
        parts = dotted.split(".")
        cursor: dict[str, Any] = merged
        for part in parts[:-1]:
            node = cursor.get(part)
            if not isinstance(node, dict):
                node = {}
                cursor[part] = node
            cursor = node
        cursor[parts[-1]] = yaml.safe_load(literal)
    return merged


def as_mapping(config: ProtocolConfig) -> dict[str, Any]:
    """Render the resolved configuration back to a plain mapping."""

    out: dict[str, Any] = {
        "name": config.name,
        "seed": config.seed,
        "site_holdout": config.site_holdout,
    }
    for key in _SECTION_TYPES:
        section = getattr(config, key)
        out[key] = {field.name: getattr(section, field.name) for field in fields(section)}
    return out


def is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)
