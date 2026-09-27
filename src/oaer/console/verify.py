"""The verification driver.

Runs the battery, writes the verification report, renders its plain-text summary
and then writes the integrity manifest, in that order, so the manifest digests a
finished tree that already carries the report and the summary. Writing the
manifest before the summary would leave one file outside the manifest, and
patching the manifest afterwards would make its own digest stale.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from oaer.console.common import configure_and_parse, load_config
from oaer.report import verification_report
from oaer.support.io import write_json, write_text
from oaer.support.logging import get_logger
from oaer.support.seeding import set_seed
from oaer.validation.battery import (
    check_manifest_freshness,
    check_report_is_complete,
    expected_report_minimum,
    run_battery,
)
from oaer.validation.manifest import manifest_root, verify_manifest, write_integrity_manifest
from oaer.validation.reporting import VerificationEntry, family_summary

LOGGER = get_logger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANCHOR = "protocols/main.yaml"


def build_report(
    entries: Sequence[VerificationEntry],
    claim_map: dict[str, object],
    extras: dict[str, object],
) -> dict[str, object]:
    """The verification report payload, with the two verdicts kept apart."""

    payload: dict[str, object] = {
        "scope": (
            "the data path, the cross-fitted estimand, the gate, the read-outs, the "
            "training step and the live public resources"
        ),
        "checks": [entry.as_dict() for entry in entries],
        "claim_count": len(claim_map.get("claims", [])),  # type: ignore[arg-type]
    }
    payload.update(family_summary(entries))
    payload.update(extras)
    return payload


def main(argv: list[str] | None = None) -> int:
    invocation = configure_and_parse("Run the verification battery", argv)
    config = load_config(invocation)
    root = manifest_root(invocation.protocol)
    claim_path = root / "claim_to_code.json"
    claim_map: dict[str, object] = (
        json.loads(claim_path.read_text(encoding="utf-8"))
        if claim_path.is_file()
        else {"claims": []}
    )
    set_seed(config.seed)
    LOGGER.info("running the verification battery at %s", root.name)
    include_live = not invocation.no_live
    include_study = not invocation.no_study
    entries = run_battery(claim_map, root, include_live=include_live, include_study=include_study)
    # The last two checks read artefacts this driver writes, so they run here: the
    # shipped report's completeness is judged against the report already on disk,
    # and the manifest's freshness against the manifest already on disk. A first
    # run therefore records NOT_RUN for both, which is the honest reading of
    # "there was nothing to read yet"; a second run, with the tree unchanged,
    # carries the two verdicts and is the one to ship.
    minimum = expected_report_minimum(include_live=include_live, include_study=include_study)
    if len(entries) < minimum:
        LOGGER.error("the battery produced %d entries, short of %d", len(entries), minimum)
    entries = [
        *entries,
        check_manifest_freshness(root, root / "integrity_manifest.json"),
        check_report_is_complete(root / "verification_report.json", minimum),
    ]

    extras: dict[str, object] = {
        "protocol": config.name,
        "seed": config.seed,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "privacy": (
            "no absolute path, credential, personal identifier, account name or model "
            "provenance marker is carried by this tree"
        ),
        "not_reproduced": [
            "every cohort-level column of the article, whose arm is private and is not redistributed",
            "the reported checkpoint of the deployed model, which is not released",
            "the auxiliary public cohorts, which are not redistributed and are read at their source",
        ],
        "reproduced_scope": (
            "the mechanisms, the reported one-variable marginals on the arm this release builds, the "
            "arithmetic of the article's own ablation and criteria tables, and the closed "
            "form of the builder's hazard"
        ),
    }
    payload = build_report(entries, claim_map, extras)
    report_path = root / "verification_report.json"
    write_json(payload, report_path)

    summary_text = verification_report(entries)
    summary_path = root / "reports" / "verification.txt"
    write_text(summary_text, summary_path)

    manifest = write_integrity_manifest(root, root / "integrity_manifest.json")
    fresh, drift = verify_manifest(root, root / "integrity_manifest.json")
    LOGGER.info(
        "verification verdict %s; code %s; manuscript tables %s; resource probes %s; "
        "manifest covers %d files",
        payload["overall"],
        payload["code_status"],
        payload["manuscript_status"],
        payload["live_status"],
        len(manifest["files"]),  # type: ignore[arg-type]
    )
    if not fresh:
        LOGGER.error("the manifest does not digest the live tree: %s", list(drift)[:6])
    if invocation.emit:
        sys.stdout.write(summary_text)
    return 0 if payload.get("overall") != "FAIL" and fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
