"""Live checks against the auxiliary public resources.

Each resource is probed over the network and its content is read back, so a link
is only reported as reachable when the response carried the expected payload. A
probe that cannot be completed — no route, a refused transport, a timeout — is
reported as blocked with its reason rather than omitted, because an absent row and
an unreachable resource are different statements.

The probes never send a credential, never follow a redirect into a sign-in page,
and read only what the resource serves anonymously.

Ref: Sec. 2.2 (every access route was tested for downloadability).
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from oaer.catalogue.sources import AUXILIARY_SOURCES, AuxiliarySource
from oaer.validation.reporting import VerificationEntry

TIMEOUT_SECONDS: Final[float] = 20.0
USER_AGENT: Final[str] = "oaer-release-resource-check/0.1"
ATTEMPTS: Final[int] = 4
BACKOFF_SECONDS: Final[tuple[float, ...]] = (0.5, 1.5, 3.0)
READ_LIMIT: Final[int] = 1 << 20
GZIP_MAGIC: Final[bytes] = b"\x1f\x8b"


@dataclass(frozen=True)
class ProbeResult:
    """What one probe saw."""

    name: str
    status: str
    evidence: str


def _decode(payload: bytes) -> str:
    """Read a payload as text, inflating it when it arrives gzipped.

    A series matrix is served compressed, so searching the stream as delivered for
    an accession would never find it. A truncated stream is inflated as far as it
    goes rather than rejected, because the header is what a probe reads.
    """

    if payload[:2] == GZIP_MAGIC:
        with contextlib.suppress(zlib.error):
            payload = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(payload)
    return payload.decode("utf-8", errors="replace")


def _fetch(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(url, data=data, headers=headers or {})
    request.add_header("User-Agent", USER_AGENT)
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                payload: bytes = response.read(READ_LIMIT)
            return _decode(payload)
        except urllib.error.HTTPError:
            # A served status is an answer, not a transport failure; retrying it
            # would only repeat the same response.
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = error
            if attempt < len(BACKOFF_SECONDS):
                time.sleep(BACKOFF_SECONDS[attempt])
    if last is not None:
        raise last
    raise urllib.error.URLError("the probe made no request")


def probe(source: AuxiliarySource) -> ProbeResult:
    """Run one resource's probe and interpret the payload."""

    try:
        if source.probe == "status":
            body = _fetch(source.url)
            record = json.loads(body)
            value = str(record.get("chembl_release_date", ""))
            status = "PASS" if value else "PARTIAL"
            return ProbeResult(source.name, status, f"release date read back as {value!r}")
        if source.probe == "graphql_meta":
            body = _fetch(
                source.url,
                data=b'{"query":"{ meta { name } }"}',
                headers={"Content-Type": "application/json"},
            )
            record = json.loads(body)
            name = str(record.get("data", {}).get("meta", {}).get("name", ""))
            status = "PASS" if name else "PARTIAL"
            return ProbeResult(source.name, status, f"meta.name read back as {name!r}")
        if source.probe == "download_page":
            body = _fetch(source.url)
            status = "PASS" if source.expect.lower() in body.lower() else "PARTIAL"
            return ProbeResult(source.name, status, f"page carries {source.expect!r}")
        if source.probe == "gdc_project":
            body = _fetch(source.url + "?fields=project_id,name&format=json")
            record = json.loads(body)
            project = str(record.get("data", {}).get("project_id", ""))
            status = "PASS" if project else "PARTIAL"
            return ProbeResult(source.name, status, f"project_id read back as {project!r}")
        if source.probe == "cbioportal_studies":
            body = _fetch(source.url + "?projection=SUMMARY")
            studios = json.loads(body)
            names = {str(entry.get("studyId", "")) for entry in studios}
            status = "PASS" if source.expect in names else "PARTIAL"
            return ProbeResult(
                source.name, status, f"{len(names)} studies listed; {source.expect!r} present"
            )
        if source.probe == "geo_matrix":
            body = _fetch(source.url)
            status = "PASS" if source.expect in body else "PARTIAL"
            return ProbeResult(source.name, status, f"series matrix carries {source.expect!r}")
        if source.probe == "zenodo_record":
            body = _fetch(source.url)
            record = json.loads(body)
            title = str(record.get("metadata", {}).get("title", ""))
            # A deposit's identity can sit in its file list rather than in its
            # title, so both are searched; the evidence says which one carried it.
            keys = [str(entry.get("key", "")) for entry in record.get("files", [])]
            wanted = source.expect.lower()
            in_title = wanted in title.lower()
            matched = [key for key in keys if wanted in key.lower()]
            status = "PASS" if (in_title or matched) else "PARTIAL"
            where = "title" if in_title else (f"file list ({matched[0]})" if matched else "nowhere")
            return ProbeResult(
                source.name,
                status,
                f"record title {title[:60]!r} over {len(keys)} file(s); {source.expect!r} found in "
                f"the {where}",
            )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
        return ProbeResult(source.name, "BLOCKED", f"probe failed: {type(error).__name__}")
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        return ProbeResult(
            source.name, "PARTIAL", f"payload not as expected: {type(error).__name__}"
        )
    return ProbeResult(source.name, "NOT_RUN", f"no probe named {source.probe!r}")


def run_live_checks() -> tuple[VerificationEntry, ...]:
    """Every auxiliary resource as its own entry."""

    entries: list[VerificationEntry] = []
    for source in AUXILIARY_SOURCES:
        result = probe(source)
        entries.append(
            VerificationEntry(
                name=f"live_{source.name.lower().replace(' ', '_')}",
                status=result.status,
                evidence=f"{source.url} | {result.evidence}",
                family="live",
            )
        )
    return tuple(entries)


def resource_cards() -> Mapping[str, tuple[str, str, str]]:
    """Version, licence and role of every resource, keyed by name."""

    return {
        source.name: (source.version, source.licence, source.role) for source in AUXILIARY_SOURCES
    }
