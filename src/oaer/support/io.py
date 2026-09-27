"""Plain and JSON serialisation with atomic replacement.

Every artefact this package writes is either JSON or plain text; no Markdown is
produced, because the release carries exactly one Markdown file and it is the
README. Writes go through a temporary neighbour so a partial file never lands.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: IO[str]
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        # mkstemp hands back 0600; artefacts must be world readable like the rest
        # of the tree, so the mode is set before the replacement becomes visible.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_text(text: str, path: str | Path) -> Path:
    target = Path(path)
    if not text.endswith("\n"):
        text = text + "\n"
    _atomic_write(target, text)
    return target


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(payload: Any, path: str | Path, *, indent: int = 2) -> Path:
    target = Path(path)
    _atomic_write(target, json.dumps(payload, indent=indent, sort_keys=True, default=_encode))
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _encode(value: Any) -> Any:
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys, for side-by-side comparison."""

    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        composed = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{composed}."))
        else:
            flat[composed] = value
    return flat


def unflatten(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Invert ``flatten``."""

    root: dict[str, Any] = {}
    for key, value in mapping.items():
        parts = key.split(".")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def table_to_text(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a fixed-width table for the plain-text reports."""

    cells = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(name)) for name in headers]
    for row in cells:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))
    head = "  ".join(str(name).ljust(widths[index]) for index, name in enumerate(headers))
    separator = "-" * len(head)
    body = ["  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in cells]
    return "\n".join([head, separator, *body])
