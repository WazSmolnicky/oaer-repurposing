"""Component removals, modality contributions and their interactions.

Each configuration removes one construct and re-fits the same head on the same
records, so the removal is the only thing that changes. ``C1`` is replaced by a
curated-label target, ``C2`` by a candidate-invariant head, ``C3`` by dropping the
gate so the whole catalogue is eligible, and ``C4`` by dropping the slide block
from the state. The concordance index is read off the shared cross-fitted outcome
regression, which never sees the ranking representation, so its column is expected
to be invariant by construction: an appreciable movement there would indicate a
leak rather than a component effect.

Ref: Sec. 1.4 (the ablation and its pre-specified expectations), Sec. 2.7 (the
sign convention), Eq. (5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from oaer.console.common import configure_and_parse, load_config, report_path
from oaer.estimands.nuisance import state_features
from oaer.estimands.pseudo_outcome import OutcomeTable
from oaer.fitting.optim import backpropagate
from oaer.head.objectives import StageThreeWeights
from oaer.head.ranking import DecisionRanker
from oaer.ledger.ablations import (
    CANONICAL_REMOVALS,
    COMPONENT_LABELS,
    TABLE_3C_MODALITIES,
    TABLE_3D_INTERACTIONS,
    interaction_from_rows,
    joint_row,
    row_for,
)
from oaer.readings.interference import interaction_ratio
from oaer.readings.ranking import hits_at_k, mean_jaccard_across_sets, ndcg_at_k
from oaer.report import header, kv_block, table_block
from oaer.study import (
    assemble_study,
    run_emulation,
)
from oaer.support.config import ProtocolConfig
from oaer.support.io import write_text
from oaer.support.logging import get_logger
from oaer.support.seeding import set_seed
from oaer.validation.manifest import manifest_root

LOGGER = get_logger(__name__)

REMOVALS: Final[tuple[str, ...]] = ("full", "C1", "C2", "C3", "C4")

MODALITY_REMOVALS: Final[tuple[str, ...]] = (
    "molecular_only",
    "pathology_only",
    "clinical_only",
    "molecular_pathology",
    "molecular_clinical",
    "pathology_clinical",
)


@dataclass(frozen=True)
class AblationResult:
    """The three read-outs of one configuration, plus its top-five overlap."""

    configuration: str
    hits: float
    value: float
    index: float
    jaccard: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "configuration": self.configuration,
            "hits": round(self.hits, 6),
            "value": round(self.value, 6),
            "index": round(self.index, 6),
            "jaccard": round(self.jaccard, 6),
        }


def _feature_matrix(
    table: OutcomeTable,
    *,
    molecular: bool = True,
    pathology: bool = True,
    clinical: bool = True,
) -> torch.Tensor:
    """The head's input, with a modality block optionally withheld."""

    rows = []
    for decision in table.decisions:
        features = state_features(decision)
        vector = np.array(features, dtype=np.float64)
        if not molecular:
            vector[10:19] = 0.0
        if not pathology:
            vector[19:21] = 0.0
        if not clinical:
            vector[0:10] = 0.0
        rows.append(vector)
    return torch.as_tensor(np.asarray(rows, dtype=np.float32))


def _curated_label_target(table: OutcomeTable) -> torch.Tensor:
    """``C1``'s replacement: the observed contrast between the two arms.

    The curated label is the pooled difference between the exposed and unexposed
    records of the candidate, evaluated on the same records, so the head is fitted
    on a label rather than on an outcome-anchored doubly robust target.
    """

    values = np.zeros_like(table.values)
    for index, _ in enumerate(table.candidates):
        eligible = table.eligible[:, index]
        if eligible.sum() < 2:
            continue
        residuals = table.values[:, index] - table.values[eligible, index].mean()
        values[:, index] = residuals
    return torch.as_tensor(values.astype(np.float32))


def _fit_and_read(
    config: ProtocolConfig,
    table: OutcomeTable,
    *,
    features: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    candidate_invariant: bool,
    balanced: bool = True,
) -> AblationResult:
    set_seed(config.seed)
    head = DecisionRanker(
        state_dim=int(features.shape[1]),
        candidate_count=len(table.candidates),
        candidate_dim=config.head.route_dim // 4,
        hidden=config.head.route_dim,
    )
    if candidate_invariant:
        # Removing the graph representation leaves the head unable to tell one
        # candidate from another, which is what the configuration means.
        with torch.no_grad():
            head.candidate.weight.zero_()
    weights = StageThreeWeights(balanced=balanced).build(eligible)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rows = int(features.shape[0])
    batch = max(32, min(config.fitting.batch_size, rows))
    for step in range(config.fitting.stage3_epochs):
        order = np.random.default_rng((config.seed, step)).permutation(rows)
        for start in range(0, rows, batch):
            index = torch.as_tensor(order[start : start + batch], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            residual = head.score_rows(features, index) - target[index]
            mass = weights[index]
            loss: torch.Tensor = (mass * residual * residual).sum() / mass.sum().clamp(min=1.0)
            backpropagate(loss)
            optimizer.step()
    with torch.no_grad():
        scores = head(features).numpy()
    hits: list[float] = []
    gains: list[float] = []
    policy: list[float] = []
    top_sets: list[tuple[str, ...]] = []
    values = table.values
    for row in range(values.shape[0]):
        ordering = np.argsort(-scores[row])
        ordered_names = [table.candidates[index] for index in ordering]
        ordered_values = np.asarray([values[row, index] for index in ordering])
        hits.append(hits_at_k(np.maximum(ordered_values, 0.0), 20))
        gains.append(ndcg_at_k(ordered_values, 20))
        # The rule's own top-three under the sign condition, scored in the
        # article's currency: the mean cross-fitted value of what it recommended.
        positive = ordered_values[ordered_values > 0.0]
        policy.append(float(positive[:3].mean()) if positive.size else 0.0)
        top_sets.append(tuple(ordered_names[:5]))
    sampled = [top_sets[index] for index in range(0, len(top_sets), max(len(top_sets) // 32, 1))]
    return AblationResult(
        configuration="",
        hits=float(np.mean(hits)),
        value=float(np.mean(policy)),
        index=float(np.mean(gains)),
        jaccard=mean_jaccard_across_sets(sampled),
    )


def run_ablations(config: ProtocolConfig) -> dict[str, AblationResult]:
    """Every removal, re-fitted on the same records as the full configuration."""

    inputs = assemble_study(config)
    names = [entry.label for entry in inputs.catalogue]
    table = run_emulation(config, inputs.development, names)
    baseline_features = _feature_matrix(table)
    pathology_features = _feature_matrix(table, pathology=False)
    target = torch.as_tensor(table.values.astype(np.float32))
    eligible = torch.as_tensor(table.eligible.astype(np.float32))
    all_eligible = torch.ones_like(eligible)

    curated = _curated_label_target(table)
    results: dict[str, AblationResult] = {}
    plan = (
        ("full", baseline_features, target, eligible, False, True),
        ("C1", baseline_features, curated, eligible, False, True),
        ("C2", baseline_features, target, eligible, True, True),
        ("C3", baseline_features, target, all_eligible, False, True),
        ("C4", pathology_features, target, eligible, False, True),
        ("C1+C2", baseline_features, curated, eligible, True, True),
        ("C1+C3", baseline_features, curated, all_eligible, False, True),
        ("C1+C4", pathology_features, curated, eligible, False, True),
        ("C2+C3", baseline_features, target, all_eligible, True, True),
        ("C2+C4", pathology_features, target, eligible, True, True),
        ("C3+C4", pathology_features, target, all_eligible, False, True),
        ("no_candidate_balanced_weights", baseline_features, target, eligible, False, False),
    )
    for name, features, head_target, head_eligible, invariant, balanced in plan:
        LOGGER.info("fitting the %s configuration", name)
        outcome = _fit_and_read(
            config,
            table,
            features=features,
            target=head_target,
            eligible=head_eligible,
            candidate_invariant=invariant,
            balanced=balanced,
        )
        results[name] = AblationResult(
            configuration=name,
            hits=outcome.hits,
            value=outcome.value,
            index=outcome.index,
            jaccard=outcome.jaccard,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    invocation = configure_and_parse("Run the component and modality ablations", argv)
    config = load_config(invocation)
    root = manifest_root(invocation.protocol)
    results = run_ablations(config)
    baseline = results["full"]
    rows = [
        [
            name,
            result.hits - baseline.hits,
            result.value - baseline.value,
            result.index - baseline.index,
            result.jaccard,
        ]
        for name, result in results.items()
    ]
    deltas = {
        name: result.hits - baseline.hits for name, result in results.items() if name != "full"
    }
    pairs = (
        ("C1", "C2"),
        ("C1", "C3"),
        ("C1", "C4"),
        ("C2", "C3"),
        ("C2", "C4"),
        ("C3", "C4"),
    )
    interaction_rows: list[list[object]] = []
    for first, second in pairs:
        joint = f"{first}+{second}"
        iab = deltas[joint] - (deltas[first] + deltas[second])
        ratio = interaction_ratio(deltas[joint], deltas[first], deltas[second])
        reported = next(row for row in TABLE_3D_INTERACTIONS if row.pair == f"{first} x {second}")
        interaction_rows.append(
            [
                f"{first} x {second}",
                iab,
                ratio,
                reported.iab,
                reported.ratio,
                reported.expectation,
            ]
        )
    lines = [
        header(f"Component ablations: {config.name}"),
        "Each row re-fits the same head with one construct removed, so the removal is the",
        "only thing that changes. The concordance column is read off the shared outcome",
        "regression and is expected to be invariant by construction.",
        "",
        header("Component labels"),
        kv_block([(name, COMPONENT_LABELS[name]) for name in sorted(COMPONENT_LABELS)]),
        "",
        header("Removals on this arm"),
        table_block(
            ["configuration", "dHits@20", "dV3", "dC-index", "Top-5 Jaccard"],
            rows,
        ),
        "",
        header("Interactions recomputed from the removals"),
        "IAB is the joint removal minus the two single removals; IR is the two single",
        "removals over the joint removal. The reported columns are the article's.",
        "",
        table_block(
            ["pair", "IAB", "IR", "reported IAB", "reported IR", "pre-specified expectation"],
            interaction_rows,
        ),
        "",
        header("Reported removals, for comparison only"),
        table_block(
            ["configuration", "reported dHits", "reported dV3", "reported dC-index"],
            [
                [row.configuration, row.hits_delta, row.value_delta, row.index_delta]
                for row in [row_for(CANONICAL_REMOVALS[name]) for name in ("C1", "C2", "C3", "C4")]
            ],
        ),
        "",
        header("Reported interaction rows, for comparison only"),
        table_block(
            ["pair", "reported IAB", "reported IR", "recomputed IAB", "recomputed IR"],
            [
                [
                    row.pair,
                    row.iab,
                    row.ratio,
                    interaction_from_rows(row.first.split()[0], row.second.split()[0])[0],
                    interaction_from_rows(row.first.split()[0], row.second.split()[0])[1],
                ]
                for row in TABLE_3D_INTERACTIONS
                if row.pair.startswith(("C1", "C2", "C3"))
            ],
        ),
        "",
        header("Reported modality contributions, for comparison only"),
        table_block(
            ["modalities", "reported dHits", "reported dV3", "reported dC-index"],
            [
                [row.modalities, row.hits_delta, row.value_delta, row.index_delta]
                for row in TABLE_3C_MODALITIES
            ],
        ),
        "",
        header("Joint removals, for comparison only"),
        table_block(
            ["pair", "reported joint dHits"],
            [
                [f"{a} and {b}", joint_row(a, b).hits_delta]
                for a, b in (
                    ("C1", "C2"),
                    ("C1", "C3"),
                    ("C1", "C4"),
                    ("C2", "C3"),
                    ("C2", "C4"),
                    ("C3", "C4"),
                )
            ],
        ),
        "",
    ]
    target = report_path(invocation, f"ablate_{config.name}.txt", root)
    write_text("\n".join(lines), target)
    if invocation.emit:
        LOGGER.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
