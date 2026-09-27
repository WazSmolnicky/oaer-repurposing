"""Cross-fitting folds and the site hold-out.

The folds are patient-level and stratified by site within Sites A and B. Site C
never enters a fitting fold: it is scored once per frozen configuration by
nuisances fitted on A and B alone. Model selection happens on an inner split of
the training folds, never on the reported out-of-fold set.

Ref: Sec. 1.2 (Site C forms no training pseudo-outcome), Sec. 2.6 (Algorithm 1,
line 3 and line 9), Sec. 2.7 (five patient-level folds).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from oaer.support.seeding import spawn_generator
from oaer.support.types import Decision, DecisionBatch, Split

DEVELOPMENT_SPLITS: Final[tuple[Split, ...]] = (Split.DEV_SITE_A, Split.DEV_SITE_B)
HELD_OUT_SPLIT: Final[Split] = Split.HELD_OUT_SITE_C


@dataclass(frozen=True)
class SiteStratifiedFolds:
    """Patient-level folds, stratified by site within the development splits."""

    folds: int
    assignment: Mapping[str, int]

    def mask(self, batch: DecisionBatch, fold: int) -> np.ndarray:
        """Whether each decision belongs to the held-in part of ``fold``."""

        return np.asarray(
            [self.assignment[decision.patient_id] != fold for decision in batch.decisions],
            dtype=bool,
        )

    def fold_mask(self, batch: DecisionBatch, fold: int) -> np.ndarray:
        return ~self.mask(batch, fold)

    def sizes(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for fold in self.assignment.values():
            counts[fold] = counts.get(fold, 0) + 1
        return counts


def outer_site_split(batch: DecisionBatch) -> tuple[DecisionBatch, DecisionBatch]:
    """Development records (Sites A and B) and the held-out site's records."""

    development = tuple(d for d in batch.decisions if d.split in DEVELOPMENT_SPLITS)
    held_out = tuple(d for d in batch.decisions if d.split is HELD_OUT_SPLIT)
    return DecisionBatch(development), DecisionBatch(held_out)


def fold_assignment(
    batch: DecisionBatch,
    folds: int = 5,
    *,
    seed: int = 20260831,
    splits: Sequence[Split] | None = None,
) -> SiteStratifiedFolds:
    """Assign every patient to one fold, balanced within each site."""

    if folds < 2:
        raise ValueError("cross-fitting needs at least two folds")
    allowed = set(splits) if splits is not None else set(DEVELOPMENT_SPLITS) | {HELD_OUT_SPLIT}
    stream = spawn_generator(seed, f"folds:{folds}")
    by_site: dict[str, set[str]] = {}
    for decision in batch.decisions:
        if decision.split not in allowed:
            continue
        by_site.setdefault(decision.site, set()).add(decision.patient_id)
    assignment: dict[str, int] = {}
    for site in sorted(by_site):
        patients = sorted(by_site[site])
        order = stream.permutation(len(patients))
        for position, index in enumerate(order):
            assignment[patients[int(index)]] = position % folds
    missing = sorted(
        {
            decision.patient_id
            for decision in batch.decisions
            if decision.patient_id not in assignment
        }
    )
    if missing:
        raise KeyError(f"{len(missing)} patient(s) were left without a fold")
    return SiteStratifiedFolds(folds=folds, assignment=assignment)


def cross_fitted_folds(
    batch: DecisionBatch,
    folds: int = 5,
    *,
    seed: int = 20260831,
) -> tuple[SiteStratifiedFolds, tuple[tuple[DecisionBatch, DecisionBatch], ...]]:
    """The fold object plus, per fold, the held-in and held-out partitions."""

    structure = fold_assignment(batch, folds, seed=seed)
    partitions: list[tuple[DecisionBatch, DecisionBatch]] = []
    for fold in range(folds):
        train = tuple(
            decision
            for decision in batch.decisions
            if structure.assignment[decision.patient_id] != fold
        )
        test = tuple(
            decision
            for decision in batch.decisions
            if structure.assignment[decision.patient_id] == fold
        )
        partitions.append((DecisionBatch(train), DecisionBatch(test)))
    return structure, tuple(partitions)


def inner_early_stop_split(
    train: DecisionBatch,
    *,
    fraction: float = 0.25,
    seed: int = 20260831,
) -> tuple[DecisionBatch, DecisionBatch]:
    """An inner split of the training folds, for early stopping only."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("the inner-split fraction must lie strictly between zero and one")
    stream = spawn_generator(seed, "inner-split")
    patients = sorted({decision.patient_id for decision in train.decisions})
    order = stream.permutation(len(patients))
    cut = max(1, int(len(patients) * fraction))
    early = {patients[int(index)] for index in order[:cut]}
    first = tuple(d for d in train.decisions if d.patient_id in early)
    second = tuple(d for d in train.decisions if d.patient_id not in early)
    if not first or not second:
        raise ValueError("the inner split left one side empty")
    return DecisionBatch(first), DecisionBatch(second)


def partition_closes_on(batch: DecisionBatch, expected: int) -> bool:
    """Whether a partition accounts for every decision exactly once."""

    return (
        len(batch.decisions) == expected
        and len({decision.decision_id for decision in batch.decisions}) == expected
    )


def site_counts(batch: DecisionBatch) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in batch.decisions:
        counts[decision.site] = counts.get(decision.site, 0) + 1
    return counts


def patient_counts(batch: DecisionBatch) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in batch.decisions:
        counts[decision.patient_id] = counts.get(decision.patient_id, 0) + 1
    return counts


def fold_size_spread(structure: SiteStratifiedFolds) -> int:
    sizes = sorted(structure.sizes().values())
    return sizes[-1] - sizes[0]


def first_decision_per_patient(batch: DecisionBatch) -> DecisionBatch:
    """The primary analysis population: the first decision of each patient."""

    seen: set[str] = set()
    chosen: list[Decision] = []
    for decision in batch.decisions:
        if decision.patient_id in seen:
            continue
        seen.add(decision.patient_id)
        chosen.append(decision)
    return DecisionBatch(tuple(chosen))
