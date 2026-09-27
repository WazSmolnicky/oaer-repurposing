"""Shared fixtures.

Every fixture builds from the package's own published entry points rather than
from private helpers, so a test that passes is evidence about the public surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from oaer.catalogue.concepts import build_catalogue
from oaer.catalogue.graph import KnowledgeGraph, build_knowledge_graph
from oaer.cohort.builder import build_cohort
from oaer.cohort.splits import outer_site_split
from oaer.support.config import load_protocol
from oaer.support.types import DecisionBatch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PROTOCOL = REPOSITORY_ROOT / "protocols" / "_smoke.yaml"


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def smoke_config():  # type: ignore[no-untyped-def]
    return load_protocol(SMOKE_PROTOCOL)


@pytest.fixture(scope="session")
def catalogue():  # type: ignore[no-untyped-def]
    return build_catalogue(size=12, seed=11)


@pytest.fixture(scope="session")
def graph(catalogue) -> KnowledgeGraph:  # type: ignore[no-untyped-def]
    return build_knowledge_graph(catalogue, seed=11)


@pytest.fixture(scope="session")
def cohort() -> tuple[DecisionBatch, DecisionBatch]:
    names = [entry.label for entry in build_catalogue(size=12, seed=13)]
    return build_cohort(seed=13, candidates=names, total=320)


@pytest.fixture(scope="session")
def development(cohort) -> DecisionBatch:  # type: ignore[no-untyped-def]
    return outer_site_split(cohort[0])[0]


@pytest.fixture(scope="session")
def held_out(cohort) -> DecisionBatch:  # type: ignore[no-untyped-def]
    return outer_site_split(cohort[0])[1]


@pytest.fixture
def temporary_root(tmp_path: Path) -> Iterator[Path]:
    """An empty directory that looks enough like a release root to write into."""

    (tmp_path / "protocols").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "probe"\n', encoding="utf-8")
    yield tmp_path
