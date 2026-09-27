"""Plain-text rendering of a study and of the verification battery.

Every report this package writes is plain text under ``reports/``. The release
carries exactly one Markdown file, its README, so nothing here may introduce a
second one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from oaer.ledger.ablations import (
    CANONICAL_REMOVALS,
    COMPONENT_LABELS,
    TABLE_3A_REMOVALS,
    TABLE_3B_TERCILES,
    TABLE_3C_MODALITIES,
    TABLE_3D_INTERACTIONS,
)
from oaer.ledger.comparator_roster import TABLE_2A_RANKERS, TABLE_2B_PROGNOSTIC
from oaer.ledger.criteria import TABLE_1D_CRITERIA
from oaer.ledger.generalization import TABLE_5_STRATA
from oaer.ledger.outcomes import (
    TABLE_1A_COHORT,
    TABLE_1B_PRIMARY,
    TABLE_1C_EMULATIONS,
    TABLE_1E_PATHWAYS,
    TABLE_1F_RETROSPECTIVE,
    TABLE_4_SUBGROUPS,
)
from oaer.study import StudyResult
from oaer.validation.reporting import (
    VerificationEntry,
    code_status,
    count_by_status,
    live_status,
    manuscript_status,
    overall_status,
)

WIDTH = 78
STATUS_ORDER = ("PASS", "PARTIAL", "FAIL", "NOT_RUN", "BLOCKED")


def rule(character: str = "-") -> str:
    return character * WIDTH


def header(title: str) -> str:
    return f"{title}\n{rule('=')}"


def kv_block(pairs: Iterable[tuple[str, object]]) -> str:
    return "\n".join(f"{str(name):<38} {value}" for name, value in pairs)


def table_block(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[format_cell(cell) for cell in row] for row in rows]
    widths = [len(str(name)) for name in headers]
    for row in cells:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))
    head = "  ".join(str(name).ljust(widths[index]) for index, name in enumerate(headers))
    body = ["  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in cells]
    return "\n".join([head, rule(), *body])


def format_cell(cell: object) -> str:
    if isinstance(cell, float):
        return "not run" if cell != cell else f"{cell:.4f}"
    if cell is None:
        return "-"
    return str(cell)


def _reason_counts(gate: Mapping[str, object]) -> Mapping[str, object]:
    """The per-reason deferral counts out of a gate payload."""

    counts = gate.get("gate_reason_counts", {})
    return counts if isinstance(counts, Mapping) else {}


def study_report(result: StudyResult) -> str:
    """The study's own read-outs, each beside the article's reported value."""

    lines = [
        header(f"Protocol run: {result.protocol}"),
        "Every value below was produced by this repository on the arm it builds. That arm",
        "is schema-compatible with the reported one and holds its",
        "reported one-variable marginals; it is not the reported cohort, so no value below",
        "was measured on the article's arm. The article's own numbers are",
        "printed in the ledger block at the end of this report, for comparison only.",
        "",
        header("Cohort"),
        kv_block(
            [
                ("retrospective decisions", result.cohort_census["retrospective_decisions"]),
                ("prospective decisions", result.cohort_census["prospective_decisions"]),
                ("development decisions", result.cohort_census["development_decisions"]),
                ("held-out decisions", result.cohort_census["held_out_decisions"]),
                ("catalogue size", result.catalogue_size),
            ]
        ),
        "",
        table_block(
            ["site", "decisions"],
            [[site, count] for site, count in sorted(result.site_census.items())],
        ),
        "",
        table_block(
            ["reported marginal", "holds at the configured scale"],
            [[name, str(value).lower()] for name, value in sorted(result.marginal_match.items())],
        ),
        "",
        header("Graph"),
        table_block(
            ["quantity", "value"],
            [[name, value] for name, value in sorted(result.graph_census.items())],
        ),
        "",
        header("Pretraining stages"),
        table_block(
            ["stage", "steps", "leading mean", "trailing mean", "parameters"],
            [
                [
                    name,
                    payload.get("steps", 0),
                    payload.get("leading_mean", float("nan")),
                    payload.get("trailing_mean", float("nan")),
                    payload.get("parameters", 0),
                ]
                for name, payload in sorted(result.stage_reports.items())
            ],
        ),
        "",
        header("Per-candidate emulation"),
        table_block(
            ["candidate", "tau (months)", "se", "exposed", "eligible"],
            [
                [
                    name,
                    payload["tau"],
                    payload["standard_error"],
                    payload["exposed"],
                    payload["eligible_decisions"],
                ]
                for name, payload in result.emulation.items()
            ],
        ),
        "",
        header("Ranking read-outs"),
        kv_block(
            [
                ("Hits at the configured depth", f"{result.policy_value['hits_at_depth']:.4f}"),
                ("nDCG at the configured depth", f"{result.policy_value['ndcg_at_depth']:.4f}"),
                (
                    "top-five overlap across strata",
                    f"{result.policy_value['top_five_jaccard']:.4f}",
                ),
                ("decisions scored", int(result.policy_value["decisions"])),
            ]
        ),
        "",
        header("Gate"),
        kv_block(
            [
                ("deferral rate", f"{result.gate['gate_deferral_rate']:.4f}"),
                ("decisions carrying a harm block", f"{result.gate['gate_harm_rate']:.4f}"),
                ("mean trimmed fraction", f"{result.readouts['trimmed_fraction_mean']:.4f}"),
                ("mean eligible set size", f"{result.readouts['eligible_per_candidate']:.2f}"),
            ]
        ),
        "",
        table_block(
            ["deferral reason", "decisions"],
            [[name, count] for name, count in sorted(_reason_counts(result.gate).items())],
        ),
        "",
        header("Reproduced prognostic reference score"),
        kv_block(
            [
                ("development concordance", f"{result.reference_score['development_c_index']:.4f}"),
                ("held-out-site concordance", f"{result.reference_score['held_out_c_index']:.4f}"),
            ]
        ),
        "",
        header("Stage 3 trajectory"),
        kv_block(
            [
                (name, f"{value:.6f}" if isinstance(value, float) else value)
                for name, value in sorted(result.training_records.items())
            ]
        ),
        "",
        header("Article ledger, for comparison only"),
        "These are the article's reported values on its own cohort. They were not",
        "produced by this run.",
        "",
        table_block(
            ["primary criterion", "reported"],
            [
                [
                    criterion.identifier,
                    f"met {criterion.met} margin {criterion.margin:+.3f}",
                ]
                for criterion in TABLE_1D_CRITERIA
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def ledger_report() -> str:
    """Every reported table the release carries, rendered for side-by-side reading."""

    lines = [
        header("Reported cohort, Table 1 panel a"),
        table_block(
            ["site", "pat. retro.", "dec. retro.", "pat. pro.", "dec. pro.", "follow-up", "events"],
            [
                [
                    cell.site,
                    cell.retrospective_patients,
                    cell.retrospective_decisions,
                    cell.prospective_patients,
                    cell.prospective_decisions,
                    cell.follow_up_months,
                    cell.events,
                ]
                for cell in TABLE_1A_COHORT
            ],
        ),
        "",
        header("Reported prospective arm, Table 1 panel b"),
        table_block(
            ["site", "cloned", "weighted events", "RMST PFS", "PFS HR", "C-index"],
            [
                [
                    cell.site,
                    cell.cloned_decisions,
                    cell.weighted_events,
                    cell.rmst_pfs_months,
                    f"{cell.pfs_hazard_ratio[0]:.2f} ({cell.pfs_hazard_ratio[1]:.2f}-{cell.pfs_hazard_ratio[2]:.2f})",
                    f"{cell.c_index[0]:.3f}/{cell.c_index[1]:.3f}",
                ]
                for cell in TABLE_1B_PRIMARY
            ],
        ),
        "",
        header("Reported per-candidate emulations, Table 1 panel c"),
        table_block(
            ["candidate", "comparator", "exposed", "PFS HR", "E-value", "negative control"],
            [
                [
                    cell.candidate,
                    cell.active_comparator,
                    cell.exposed,
                    f"{cell.pfs_hazard_ratio[0]:.2f} ({cell.pfs_hazard_ratio[1]:.2f}-{cell.pfs_hazard_ratio[2]:.2f})",
                    cell.e_value,
                    cell.negative_control,
                ]
                for cell in TABLE_1C_EMULATIONS
            ],
        ),
        "",
        header("Reported recommendation pathway, Table 1 panel e"),
        table_block(
            ["pathway", "decisions", "share %", "RMST PFS", "PFS HR"],
            [
                [
                    cell.pathway,
                    cell.decisions,
                    cell.share_percent,
                    cell.rmst_pfs_months,
                    cell.pfs_hazard_ratio,
                ]
                for cell in TABLE_1E_PATHWAYS
            ],
        ),
        "",
        header("Reported retrospective strata, Table 1 panel f"),
        table_block(
            ["site", "cloned", "RMST PFS", "PFS HR", "C-index"],
            [
                [
                    cell.site,
                    cell.cloned_decisions,
                    cell.rmst_pfs_months,
                    f"{cell.pfs_hazard_ratio[0]:.2f} ({cell.pfs_hazard_ratio[1]:.2f}-{cell.pfs_hazard_ratio[2]:.2f})",
                    f"{cell.c_index[0]:.3f}/{cell.c_index[1]:.3f}",
                ]
                for cell in TABLE_1F_RETROSPECTIVE
            ],
        ),
        "",
        header("Reported comparison roster, Table 2 panel A"),
        table_block(
            ["family", "method", "Hits@20", "nDCG@20", "V3", "PFS HR", "provenance"],
            [
                [
                    row.family,
                    row.method,
                    row.hits_at_20,
                    row.ndcg_at_20,
                    row.policy_value_months,
                    row.pfs_hazard_ratio,
                    row.provenance,
                ]
                for row in TABLE_2A_RANKERS
            ],
        ),
        "",
        header("Reported comparison roster, Table 2 panel B"),
        table_block(
            ["method", "OS C-index", "PFS C-index", "ECE", "AUROC 12 mo"],
            [
                [
                    row.method,
                    f"{row.os_c_index[0]:.3f} ({row.os_c_index[1]:.3f}-{row.os_c_index[2]:.3f})",
                    row.pfs_c_index,
                    row.expected_calibration_error,
                    row.auroc_12_month,
                ]
                for row in TABLE_2B_PROGNOSTIC
            ],
        ),
        "",
        header("Reported component removals, Table 3 panel a"),
        table_block(
            ["configuration", "dHits", "dV3", "dC-index", "expectation"],
            [
                [
                    row.configuration,
                    row.hits_delta,
                    row.value_delta,
                    row.index_delta,
                    row.expectation,
                ]
                for row in TABLE_3A_REMOVALS
            ],
        ),
        "",
        header("Reported interaction rows, Table 3 panel d"),
        table_block(
            ["pair", "IAB", "interval", "IR", "expectation"],
            [
                [
                    row.pair,
                    row.iab,
                    f"({row.iab_interval[0]:.1f}, {row.iab_interval[1]:.1f})",
                    row.ratio,
                    row.expectation,
                ]
                for row in TABLE_3D_INTERACTIONS
            ],
        ),
        "",
        header("Reported exposure terciles, Table 3 panel b"),
        table_block(
            ["configuration", "dHits", "dV3", "dC-index", "expectation"],
            [
                [row.configuration, row.hits_gain, row.value_gain, row.index_gain, row.expectation]
                for row in TABLE_3B_TERCILES
            ],
        ),
        "",
        header("Reported modality contributions, Table 3 panel c"),
        table_block(
            ["modalities", "dHits", "dV3", "dC-index"],
            [
                [row.modalities, row.hits_delta, row.value_delta, row.index_delta]
                for row in TABLE_3C_MODALITIES
            ],
        ),
        "",
        header("Component labels"),
        kv_block(
            [(component, COMPONENT_LABELS[component]) for component in sorted(COMPONENT_LABELS)]
        ),
        "",
        kv_block(
            [
                (f"canonical removal of {key}", value)
                for key, value in sorted(CANONICAL_REMOVALS.items())
            ]
        ),
        "",
        header("Reported subgroup read-outs, Table 4"),
        table_block(
            ["block", "subgroup", "dec. retro.", "OS C-index", "PFS HR", "deferral %"],
            [
                [
                    cell.block,
                    cell.subgroup,
                    cell.retrospective_decisions,
                    cell.os_c_index,
                    f"{cell.pfs_hazard_ratio[0]:.2f} ({cell.pfs_hazard_ratio[1]:.2f}-{cell.pfs_hazard_ratio[2]:.2f})",
                    cell.deferral_percent,
                ]
                for cell in TABLE_4_SUBGROUPS
            ],
        ),
        "",
        header("Reported generalisation strata, Table 5"),
        table_block(
            ["stratum", "domain", "modalities", "N", "OS C-index", "delta"],
            [
                [
                    row.stratum,
                    row.domain,
                    "+".join(row.modalities),
                    row.records,
                    row.os_c_index,
                    row.delta_vs_matched_reference,
                ]
                for row in TABLE_5_STRATA
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def verification_report(entries: Sequence[VerificationEntry]) -> str:
    """The battery's entries, with the two verdicts kept apart."""

    counts = count_by_status(entries)
    lines = [
        header("Verification battery"),
        kv_block(
            [
                ("release verdict", overall_status(entries)),
                ("code checks", code_status(entries)),
                ("manuscript-table checks", manuscript_status(entries)),
                ("resource-probe checks", live_status(entries)),
                *[(f"count {status}", counts.get(status, 0)) for status in STATUS_ORDER],
            ]
        ),
        "",
        table_block(
            ["check", "family", "status", "evidence"],
            [[entry.name, entry.family, entry.status, entry.evidence] for entry in entries],
        ),
    ]
    return "\n".join(lines) + "\n"
