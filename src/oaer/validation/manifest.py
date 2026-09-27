"""The integrity manifest.

Every tracked file below the release root is digested with SHA-256, in path
order. Container metadata is excluded by construction: what is hashed is the file
on disk, and the manifest excludes only itself. The root is found by walking up
from a configuration file until a ``pyproject.toml`` appears, so the manifest is
independent of how deep the anchor sits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from oaer.support.io import write_json
from oaer.support.version import RELEASE_SLUG

EXCLUDED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "node_modules"}
)
EXCLUDED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo", ".part")
MANIFEST_EXCLUDE: Final[frozenset[str]] = frozenset({"integrity_manifest.json"})


def manifest_root(anchor: Path) -> Path:
    """The release root, found by walking up from an anchor file."""

    resolved = anchor.resolve()
    for candidate in [resolved.parent, *resolved.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    if len(resolved.parents) >= 2:
        return resolved.parents[2]
    return resolved.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(path: Path) -> bool:
    if set(path.parts) & EXCLUDED_DIRECTORIES:
        return True
    if path.name in MANIFEST_EXCLUDE:
        return True
    return path.suffix in EXCLUDED_SUFFIXES


def source_manifest(root: Path) -> list[dict[str, object]]:
    """One row per tracked file, in path order."""

    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _excluded(path):
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_integrity_manifest(root: Path, output: Path) -> dict[str, object]:
    """Write the manifest and return the payload it wrote.

    The manifest is written last, after every other artefact, so it digests the
    finished tree; it excludes only itself.
    """

    payload: dict[str, object] = {
        "algorithm": "SHA-256",
        "root": RELEASE_SLUG,
        "files": source_manifest(root),
    }
    write_json(payload, output)
    return payload


def verify_manifest(root: Path, manifest: Path) -> tuple[bool, tuple[str, ...]]:
    """Re-hash the live tree and diff it against a written manifest."""

    if not manifest.is_file():
        return (False, (str(manifest),))
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    expected: dict[str, str] = {
        str(row["path"]): str(row["sha256"]) for row in recorded.get("files", [])
    }
    live: dict[str, str] = {str(row["path"]): str(row["sha256"]) for row in source_manifest(root)}
    drift = tuple(
        sorted(
            set(expected) ^ set(live)
            | {path for path in set(expected) & set(live) if expected[path] != live[path]}
        )
    )
    return (not drift, drift)


def manifest_digest(rows: Sequence[dict[str, object]]) -> str:
    """A single digest over the manifest rows, for a second-level check."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(str(row["sha256"]).encode("utf-8"))
    return digest.hexdigest()


def manifest_summary(manifest: Path) -> dict[str, object]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("files", [])
    return {
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "digest": manifest_digest(rows),
    }
