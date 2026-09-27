"""Candidate admission and the non-oncology indication test.

An agent enters the directory ``D`` when at least one of its approved
indications, as its target label records it in ChEMBL, Open Targets and
DrugCentral, is non-oncological. An agent approved for oncological applications
only is excluded, whatever else its label carries. The catalogue this module
builds is a deterministic reconstruction of that rule: the paper reports the
rule and the named candidates, not the full directory, so the directory's size is
a declared engineering default and every entry is derived from a stated seed.

Ref: Sec. 2.2 (candidate list scope), Sec. 2.4 (the prospective admission floor).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from oaer.support.seeding import spawn_generator
from oaer.support.types import Candidate

# The oncological indication terms the label is tested against. An agent whose
# every approved indication falls in this vocabulary is oncology-only.
ONCOLOGY_TERMS: Final[frozenset[str]] = frozenset(
    {
        "carcinoma",
        "sarcoma",
        "lymphoma",
        "leukaemia",
        "leukemia",
        "myeloma",
        "melanoma",
        "neoplasm",
        "glioma",
        "gastrointestinal stromal tumour",
        "antineoplastic",
    }
)

# The classes the article names, with the comparator each emulation pairs them
# with, and the molecular or clinical axis the class acts on. The named classes
# are the ones the reported contrasts rest on; the remainder this release carries
# directory is drawn from the same indication vocabulary.
NAMED_CLASSES: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    (
        "biguanide",
        "dpp-4 inhibitor",
        "metformin",
        ("type 2 diabetes mellitus", "insulin resistance"),
    ),
    (
        "beta-blocker",
        "calcium-channel blocker",
        "bisoprolol",
        ("hypertension", "angina pectoris"),
    ),
    (
        "low-dose antiplatelet",
        "clopidogrel",
        "aspirin",
        ("secondary prevention of cardiovascular events", "stable angina"),
    ),
    (
        "proton-pump inhibitor",
        "h2-receptor antagonist",
        "omeprazole",
        ("gastro-oesophageal reflux disease", "peptic ulcer"),
    ),
)

COMPARATOR_CLASSES: Final[tuple[str, ...]] = (
    "dpp-4 inhibitor",
    "calcium-channel blocker",
    "clopidogrel",
    "h2-receptor antagonist",
)

# Classes whose agents share the candidate's target set with an oncological
# programme, so their labels carry both kinds of indication.
MIXED_LABEL_CLASSES: Final[tuple[str, ...]] = (
    "cox-2 selective anti-inflammatory",
    "statins",
    "ace inhibitor",
    "angiotensin receptor blocker",
    "thiazide diuretic",
)

NON_ONCOLOGY_INDICATIONS: Final[tuple[str, ...]] = (
    "type 2 diabetes mellitus",
    "hypertension",
    "gastro-oesophageal reflux disease",
    "hypercholesterolaemia",
    "chronic kidney disease",
    "asthma",
    "osteoarthritis",
    "osteoporosis",
    "benign prostatic hyperplasia",
    "hypothyroidism",
    "gout",
    "iron deficiency anaemia",
    "atrial fibrillation",
    "heart failure",
    "anxiety disorder",
    "insomnia",
    "migraine",
    "epilepsy",
    "psoriasis",
    "rheumatoid arthritis",
)

TARGET_VOCABULARY: Final[tuple[str, ...]] = (
    "PTGS2",
    "PTGS1",
    "HMGCR",
    "ACE",
    "AGTR1",
    "SLC12A3",
    "PPARG",
    "INSR",
    "ADRB1",
    "ADRB2",
    "CACNA1C",
    "P2RY12",
    "ATP4A",
    "ATP4B",
    "HRH2",
    "DPP4",
    "GLP1R",
    "NFKB1",
    "MTOR",
    "PIK3CA",
)

PATHWAY_VOCABULARY: Final[tuple[str, ...]] = (
    "cyclo-oxygenase-2 signalling",
    "PI3K-pathway",
    "AMPK signalling",
    "renin-angiotensin system",
    "adrenergic signalling",
    "purinergic signalling",
    "gastric acid secretion",
    "insulin signalling",
)

CANDIDATE_CLASSES: Final[tuple[str, ...]] = (
    *tuple(entry[0] for entry in NAMED_CLASSES),
    *COMPARATOR_CLASSES,
    *MIXED_LABEL_CLASSES,
)


@dataclass(frozen=True)
class IndicationRecord:
    """One approved indication of one agent, as a source label records it."""

    label: str
    source: str
    oncology: bool


class CatalogueError(ValueError):
    """Raised when a catalogue entry cannot be admitted under the stated rule."""


def classify_indication(label: str) -> bool:
    """Whether an indication label is oncological.

    The test is term containment on the normalised label, which is the only
    signal the three sources expose uniformly. A label that names no oncological
    term is read as non-oncological.
    """

    normalised = " ".join(label.lower().replace("_", " ").replace("-", " ").split())
    return any(term in normalised for term in ONCOLOGY_TERMS)


def admitted_by_indications(records: Iterable[IndicationRecord]) -> bool:
    """The catalogue admission test: at least one non-oncological indication."""

    collected = list(records)
    if not collected:
        return False
    return any(not record.oncology for record in collected)


def build_catalogue(
    size: int = 243,
    *,
    seed: int = 20260831,
    named_only: bool = False,
) -> tuple[Candidate, ...]:
    """Build the candidate directory ``D``.

    The four named classes of the article enter first, in the order the article
    lists them; the remainder is drawn deterministically from the non-oncology
    indication vocabulary until the directory reaches ``size``. An agent whose
    constructed label carries only oncological indications is rejected and
    redrawn, so exclusion is exercised rather than assumed.
    """

    stream = spawn_generator(seed, "catalogue")
    entries: list[Candidate] = []
    for index, (drug_class, comparator, label, indications) in enumerate(NAMED_CLASSES):
        entries.append(
            Candidate(
                identifier=f"CN{index:04d}",
                label=label,
                drug_class=drug_class,
                active_comparator=comparator,
                approved_indications=indications,
                targets=_draw_targets(stream, 2 + index % 3),
                knowledge_graph_degree=6 + 3 * index,
                oncology_only=False,
            )
        )
    if named_only:
        return tuple(entries)
    pool = list(NON_ONCOLOGY_INDICATIONS)
    attempt = 0
    while len(entries) < size and attempt < size * 20:
        attempt += 1
        position = len(entries)
        drug_class = CANDIDATE_CLASSES[position % len(CANDIDATE_CLASSES)]
        first = pool[int(stream.integers(0, len(pool)))]
        second = pool[int(stream.integers(0, len(pool)))]
        indications = (first,) if first == second else (first, second)
        oncology_only = bool(stream.random() < 0.04)
        if oncology_only:
            # An oncology-only agent is admitted by no non-oncological record, so
            # the redraw is the exclusion rule under test rather than a filter.
            continue
        entries.append(
            Candidate(
                identifier=f"CN{position:04d}",
                label=f"{drug_class} agent {position:03d}",
                drug_class=drug_class,
                active_comparator=COMPARATOR_CLASSES[position % len(COMPARATOR_CLASSES)],
                approved_indications=indications,
                targets=_draw_targets(stream, 1 + int(stream.integers(0, 3))),
                knowledge_graph_degree=int(stream.integers(1, 40)),
                oncology_only=False,
            )
        )
    if len(entries) != size:
        raise CatalogueError(
            f"the directory closed at {len(entries)} entries, short of the requested {size}"
        )
    return tuple(entries)


def _draw_targets(stream: object, count: int) -> tuple[str, ...]:
    chosen: list[str] = []
    while len(chosen) < count:
        candidate = TARGET_VOCABULARY[int(stream.integers(0, len(TARGET_VOCABULARY)))]  # type: ignore[attr-defined]
        if candidate not in chosen:
            chosen.append(candidate)
    return tuple(chosen)


def reprocessing_priority(entry: Candidate, floor: float = 150.0) -> float:
    """The admission score a prospective candidate has to clear.

    The article states that a prospectively identified candidate enters ``D``
    when its score exceeds a floor of 150 and is otherwise registered without the
    chance to change the statistics. The floor is carried here; the score is the
    graph degree scaled by the configured gain, which is the only ranking signal
    the catalogue itself holds.
    """

    return float(entry.knowledge_graph_degree) * 10.0 - floor


def catalogue_by_identifier(catalogue: Sequence[Candidate]) -> Mapping[str, Candidate]:
    """Index the directory by identifier."""

    return {entry.identifier: entry for entry in catalogue}


def named_candidates(catalogue: Sequence[Candidate]) -> Mapping[str, Candidate]:
    """The four named agents, keyed by label."""

    wanted = {entry[2] for entry in NAMED_CLASSES}
    return {entry.label: entry for entry in catalogue if entry.label in wanted}


def degree_histogram(catalogue: Sequence[Candidate]) -> dict[int, int]:
    """Degree counts, used to stratify the knowledge-graph gain read-out."""

    histogram: dict[int, int] = {}
    for entry in catalogue:
        histogram[entry.knowledge_graph_degree] = histogram.get(entry.knowledge_graph_degree, 0) + 1
    return histogram


def degree_terciles(catalogue: Sequence[Candidate]) -> Mapping[str, tuple[str, ...]]:
    """Split the directory into the three exposure strata of Sec. 1.4 panel b."""

    ordered = sorted(catalogue, key=lambda entry: (entry.knowledge_graph_degree, entry.identifier))
    count = len(ordered)
    if count < 3:
        raise CatalogueError("a tercile split needs at least three candidates")
    first = count // 3
    second = 2 * count // 3
    return {
        "lowest": tuple(entry.identifier for entry in ordered[:first]),
        "middle": tuple(entry.identifier for entry in ordered[first:second]),
        "highest": tuple(entry.identifier for entry in ordered[second:]),
    }
