"""Verification entries and their roll-up.

A check is reported as exactly one of PASS, FAIL, PARTIAL, NOT_RUN or BLOCKED,
and the release-level verdict is derived from them rather than asserted. PARTIAL
exists for a check whose direction was confirmed while its magnitude was not, so
that a partially confirmed identity is not rounded up to a pass or down to a
failure.

Ref: the release's own verification contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

STATUS_ORDER: Final[tuple[str, ...]] = ("PASS", "PARTIAL", "FAIL", "NOT_RUN", "BLOCKED")

FAILING_STATUSES: Final[frozenset[str]] = frozenset({"FAIL"})
UNRESOLVED_STATUSES: Final[frozenset[str]] = frozenset({"NOT_RUN", "BLOCKED", "PARTIAL"})


@dataclass(frozen=True)
class VerificationEntry:
    """One check with its verdict and the evidence it rests on."""

    name: str
    status: str
    evidence: str
    family: str = "code"

    def __post_init__(self) -> None:
        if self.status not in STATUS_ORDER:
            raise ValueError(f"{self.status!r} is not one of {STATUS_ORDER}")

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "family": self.family,
            "evidence": self.evidence,
        }


def count_by_status(entries: Sequence[VerificationEntry]) -> dict[str, int]:
    counts = dict.fromkeys(STATUS_ORDER, 0)
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return counts


def count_by_family(entries: Sequence[VerificationEntry]) -> dict[str, dict[str, int]]:
    families: dict[str, list[VerificationEntry]] = {}
    for entry in entries:
        families.setdefault(entry.family, []).append(entry)
    return {family: count_by_status(members) for family, members in sorted(families.items())}


def overall_status(entries: Sequence[VerificationEntry]) -> str:
    """Roll the entries up into the release-level verdict.

    A family of ``manuscript`` entries is reported beside the code family rather
    than merged into it, so a discrepancy inside the article's own tables is never
    read as a broken release.
    """

    if not entries:
        return "UNVERIFIED"
    statuses = {entry.status for entry in entries}
    if statuses & FAILING_STATUSES:
        return "FAIL"
    if statuses <= {"PASS"}:
        return "VERIFIED"
    return "PARTIALLY_VERIFIED"


def code_status(entries: Sequence[VerificationEntry]) -> str:
    """The verdict restricted to the checks that exercise this tree.

    The network probes are excluded as well as the manuscript checks: a resource
    this machine cannot reach says nothing about the code, and a verdict that read
    a blocked probe as a code failure would be wrong in the direction that matters.
    """

    return overall_status([entry for entry in entries if entry.family == "code"])


def manuscript_status(entries: Sequence[VerificationEntry]) -> str:
    """The verdict restricted to the checks that read the article's own tables."""

    subset = [entry for entry in entries if entry.family == "manuscript"]
    return overall_status(subset) if subset else "NOT_RUN"


def live_status(entries: Sequence[VerificationEntry]) -> str:
    """The verdict restricted to the resource probes, which this tree does not govern."""

    subset = [entry for entry in entries if entry.family == "live"]
    return overall_status(subset) if subset else "NOT_RUN"


def outstanding(entries: Sequence[VerificationEntry]) -> tuple[str, ...]:
    return tuple(entry.name for entry in entries if entry.status in UNRESOLVED_STATUSES)


def family_summary(entries: Sequence[VerificationEntry]) -> Mapping[str, object]:
    return {
        "overall": overall_status(entries),
        "code_status": code_status(entries),
        "manuscript_status": manuscript_status(entries),
        "live_status": live_status(entries),
        "counts": count_by_status(entries),
        "by_family": count_by_family(entries),
        "outstanding": outstanding(entries),
    }


def summarise(entries: Iterable[VerificationEntry]) -> str:
    """A one-line rendering, for a report footer or a log line."""

    collected = list(entries)
    counts = count_by_status(collected)
    parts = ", ".join(f"{status} {counts[status]}" for status in STATUS_ORDER if counts[status])
    return f"{overall_status(collected)} ({parts})"
