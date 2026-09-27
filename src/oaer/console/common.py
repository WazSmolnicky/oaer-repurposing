"""Shared console helpers: argument parsing, overrides and report paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from oaer.support.config import ProtocolConfig, merge_overrides, protocol_from_mapping
from oaer.support.logging import configure_logging

DEFAULT_PROTOCOL = "protocols/main.yaml"
REPORTS_DIRECTORY = "reports"


@dataclass(frozen=True)
class ParsedInvocation:
    """One command line, resolved."""

    protocol: Path
    overrides: tuple[str, ...]
    report: Path | None
    emit: bool
    live: bool = False
    no_study: bool = False


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--protocol", "--config", dest="protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a dotted key=value override applied to the protocol before it is resolved",
    )
    parser.add_argument("--report", default="")
    parser.add_argument("--print", dest="emit", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "also probe the auxiliary resources over the network; off by default so a "
            "run's artefacts stay a pure function of this tree"
        ),
    )
    parser.add_argument(
        "--no-study",
        action="store_true",
        help="skip the end-to-end protocol run inside the verification battery",
    )
    return parser


def configure_and_parse(description: str, argv: list[str] | None) -> ParsedInvocation:
    configure_logging()
    args = build_parser(description).parse_args(argv)
    return ParsedInvocation(
        protocol=Path(args.protocol),
        overrides=tuple(args.overrides),
        report=Path(args.report) if args.report else None,
        emit=bool(args.emit),
        live=bool(args.live),
        no_study=bool(args.no_study),
    )


def load_config(invocation: ParsedInvocation) -> ProtocolConfig:
    """Read the protocol and apply any ``key=value`` overrides."""

    payload = yaml.safe_load(invocation.protocol.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{invocation.protocol} does not hold a mapping")
    if invocation.overrides:
        payload = merge_overrides(payload, invocation.overrides)
    return protocol_from_mapping(payload, name=str(payload.get("name", invocation.protocol.stem)))


def report_path(invocation: ParsedInvocation, default_name: str, root: Path) -> Path:
    """Where a command's plain-text report lands."""

    if invocation.report is not None:
        return invocation.report
    return root / REPORTS_DIRECTORY / default_name
