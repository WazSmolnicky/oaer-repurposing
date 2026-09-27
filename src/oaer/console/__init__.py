"""Console entry points.

Each command loads one protocol file, applies ``key=value`` overrides, runs its
stage and writes a plain-text report under ``reports/``. The verification command
is the only one that writes JSON, and it writes the integrity manifest last so the
manifest digests the finished tree.

The command modules are reached through their ``python -m`` paths and their
``oaer-*`` console scripts, so nothing here imports them eagerly: importing a
command to run it as a module would otherwise load it twice.
"""

from __future__ import annotations

from oaer.console.common import (
    DEFAULT_PROTOCOL,
    ParsedInvocation,
    configure_and_parse,
    load_config,
    report_path,
)

__all__ = [
    "DEFAULT_PROTOCOL",
    "ParsedInvocation",
    "configure_and_parse",
    "load_config",
    "report_path",
]
