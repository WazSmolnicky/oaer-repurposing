"""The whole protocol: pretraining, emulation, the head, the gate and the read-outs.

Ref: Algorithm 1, Sec. 2.6-2.7.
"""

from __future__ import annotations

from oaer.console.common import configure_and_parse, load_config, report_path
from oaer.report import study_report
from oaer.study import run_study
from oaer.support.io import write_text
from oaer.support.logging import get_logger
from oaer.validation.manifest import manifest_root

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    invocation = configure_and_parse("Run the full outcome-anchored effect ranking protocol", argv)
    config = load_config(invocation)
    root = manifest_root(invocation.protocol)
    LOGGER.info("running protocol %s at seed %d", config.name, config.seed)
    result = run_study(config)
    target = report_path(invocation, f"study_{config.name}.txt", root)
    write_text(study_report(result), target)
    LOGGER.info(
        "protocol %s finished: %d development decisions, deferral rate %.4f",
        config.name,
        result.cohort_census["development_decisions"],
        result.gate["gate_deferral_rate"],
    )
    if invocation.emit:
        LOGGER.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
