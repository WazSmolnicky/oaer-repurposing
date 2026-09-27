"""The article's reported tables, rendered for side-by-side reading.

Every table the release carries is printed as transcribed, so a reviewer can read
the reported values beside a run's own read-outs without opening the JSON. Nothing
here measures anything: the run's own report is ``oaer-rank``.

Ref: Tables 1-5 and their captions.
"""

from __future__ import annotations

from oaer.console.common import configure_and_parse, load_config, report_path
from oaer.report import ledger_report
from oaer.support.io import write_text
from oaer.support.logging import get_logger
from oaer.validation.manifest import manifest_root

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    invocation = configure_and_parse("Render the reported tables", argv)
    config = load_config(invocation)
    root = manifest_root(invocation.protocol)
    target = report_path(invocation, f"ledger_{config.name}.txt", root)
    write_text(ledger_report(), target)
    if invocation.emit:
        LOGGER.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
