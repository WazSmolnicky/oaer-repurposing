"""Per-candidate target-trial emulation: the cross-fitted pseudo-outcome table.

Ref: Algorithm 2, Eq. (2), Sec. 2.5.
"""

from __future__ import annotations

import numpy as np

from oaer.console.common import configure_and_parse, load_config, report_path
from oaer.estimands.sensitivity import e_value, negative_control_estimate, risk_ratio_at_horizon
from oaer.readings.resampling import pooled_site_interval
from oaer.report import header, kv_block, table_block
from oaer.study import assemble_study, emulation_rule, run_emulation
from oaer.support.io import write_text
from oaer.support.logging import get_logger
from oaer.validation.manifest import manifest_root

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    invocation = configure_and_parse("Emulate one candidate's target trial", argv)
    config = load_config(invocation)
    root = manifest_root(invocation.protocol)
    inputs = assemble_study(config)
    names = [entry.label for entry in inputs.catalogue][: config.cohort.candidate_count]
    LOGGER.info("cross-fitting Eq. (2) over %d candidates", len(names))
    table = run_emulation(config, inputs.development, names)
    rule = emulation_rule(config)

    rows: list[list[object]] = []
    per_site: dict[str, dict[str, tuple[float, float]]] = {}
    for name in table.candidates:
        estimate = table.estimates[name]
        rows.append(
            [
                name,
                estimate.tau,
                estimate.standard_error,
                estimate.exposed,
                estimate.control,
                estimate.eligible_decisions,
                table.trimmed_fraction(name),
            ]
        )
        for site in sorted({decision.site for decision in table.decisions}):
            index = [position for position, d in enumerate(table.decisions) if d.site == site]
            values = table.values[index, table.candidates.index(name)]
            mask = table.eligible[index, table.candidates.index(name)]
            if mask.sum() < 3:
                continue
            selected = values[mask]
            per_site.setdefault(name, {})[site] = (
                float(selected.mean()),
                float(selected.std(ddof=1) / np.sqrt(selected.size)),
            )

    reason_counts: dict[str, int] = {}
    for reasons in table.reasons.values():
        for reason, count in reasons.items():
            reason_counts[reason.value] = reason_counts.get(reason.value, 0) + count

    sensitivity: list[list[object]] = []
    for name in table.candidates[:8]:
        ratio, lower, upper = risk_ratio_at_horizon(list(table.decisions), name)
        control, control_lower, control_upper = negative_control_estimate(
            list(table.decisions), name
        )
        value = e_value(ratio, lower, upper)
        sensitivity.append(
            [
                name,
                ratio,
                lower,
                upper,
                value.point,
                control,
                control_lower,
                control_upper,
            ]
        )

    lines = [
        header(f"Emulation table: {config.name}"),
        "Every value below was produced on the arm this release builds. That arm holds the",
        "one-variable marginals and is not the reported cohort, so these are this",
        "repository's estimates and not reproductions of the article's.",
        "",
        header("Frozen rule"),
        kv_block(
            [
                ("trimming fraction", rule.epsilon),
                ("minimum exposed count", rule.minimum_exposed),
                ("folds", config.emulation.folds),
                ("censoring floor", config.emulation.censoring_floor),
                ("augmentation", "off (see claim_to_code.json)"),
            ]
        ),
        "",
        header("Candidate estimates"),
        table_block(
            ["candidate", "tau", "se", "exposed", "control", "eligible", "trimmed"],
            rows,
        ),
        "",
        header("Not-outcome-identified block"),
        table_block(
            ["reason", "cells"],
            [[reason, count] for reason, count in sorted(reason_counts.items())],
        ),
        "",
        header("Sensitivity read-outs"),
        table_block(
            [
                "candidate",
                "risk ratio",
                "RR lower",
                "RR upper",
                "E-value",
                "negative control",
                "nc lower",
                "nc upper",
            ],
            sensitivity,
        ),
        "",
        header("Per-site pooled estimates"),
        table_block(
            ["candidate", "pooled", "lower", "upper"],
            [
                [
                    name,
                    interval.point,
                    interval.lower,
                    interval.upper,
                ]
                for name, interval in sorted(
                    (name, pooled_site_interval(sites)) for name, sites in per_site.items()
                )
            ],
        ),
        "",
    ]
    target = report_path(invocation, f"emulate_{config.name}.txt", root)
    write_text("\n".join(lines), target)
    if invocation.emit:
        LOGGER.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
