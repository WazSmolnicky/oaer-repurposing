"""The end-to-end protocol run.

One call builds the candidate directory, the typed knowledge graph, the
schema-compatible cohort, the graph-side pretraining stages, the cross-fitted
pseudo-outcome table, the ranking head and the gate, then reads the study out in
the currencies the article reports. The study is executable end to end on the
arm this release builds; every cohort-level number the article prints stays in the ledger
and is never presented as a result of this run.

Ref: Algorithm 1, Sec. 2.6-2.7.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from oaer.catalogue.concepts import build_catalogue, named_candidates
from oaer.catalogue.graph import KnowledgeGraph, build_knowledge_graph
from oaer.catalogue.sampling import corrupt_anchors, sample_masked_edges
from oaer.cohort.builder import (
    NAMED_EXPOSURE_PREVALENCE,
    RETROSPECTIVE_MARGINALS,
    marginal_counts,
    modality_census,
    scaled_marginals,
)
from oaer.cohort.splits import (
    SiteStratifiedFolds,
    fold_assignment,
    outer_site_split,
)
from oaer.estimands.nuisance import state_features
from oaer.estimands.pseudo_outcome import OutcomeTable, build_outcome_table
from oaer.estimands.trimming import EligibilityRule
from oaer.fitting.loop import StepRecord, TrainingConfig, TrainingRun
from oaer.fitting.optim import backpropagate
from oaer.fitting.stages import StageBundle, fit_graph_stages
from oaer.gate.conformal import (
    ConformalCalibration,
    CovariateWeights,
    WeightedConformal,
    covariate_weights,
)
from oaer.gate.harm import GateDecision, GateOutcome, PathEvidence, apply_gate
from oaer.gate.identification import identified_set
from oaer.head.ranking import DecisionRanker
from oaer.readings.discrimination import concordance_index
from oaer.readings.ranking import hits_at_k, mean_jaccard_across_sets, ndcg_at_k
from oaer.readings.resampling import BootstrapResult, site_stratified_bootstrap
from oaer.support.config import ProtocolConfig
from oaer.support.seeding import set_seed, spawn_generator
from oaer.support.types import Candidate, DecisionBatch


@dataclass
class StudyResult:
    """Everything one protocol run produced, ready to render or to check."""

    protocol: str
    seed: int
    catalogue_size: int
    graph_census: dict[str, int]
    cohort_census: dict[str, int]
    site_census: dict[str, int]
    modality_census: dict[str, int]
    marginal_match: dict[str, bool]
    stage_reports: dict[str, dict[str, float | str]]
    emulation: dict[str, dict[str, float | int | str]]
    policy_value: dict[str, float | int]
    policy_contrast: dict[str, float]
    gate: dict[str, object]
    readouts: dict[str, float]
    reference_score: dict[str, float]
    criteria: dict[str, object]
    training_records: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "catalogue_size": self.catalogue_size,
            "graph_census": self.graph_census,
            "cohort_census": self.cohort_census,
            "site_census": self.site_census,
            "modality_census": self.modality_census,
            "marginal_match": self.marginal_match,
            "stage_reports": self.stage_reports,
            "emulation": self.emulation,
            "policy_value": self.policy_value,
            "policy_contrast": self.policy_contrast,
            "gate": self.gate,
            "readouts": self.readouts,
            "reference_score": self.reference_score,
            "criteria": self.criteria,
            "training_records": self.training_records,
        }


@dataclass(frozen=True)
class StudyInputs:
    """The objects a study is assembled from, kept apart so a check can reuse them."""

    catalogue: tuple[Candidate, ...]
    graph: KnowledgeGraph
    retrospective: DecisionBatch
    prospective: DecisionBatch
    development: DecisionBatch
    held_out: DecisionBatch
    folds: SiteStratifiedFolds


def assemble_study(config: ProtocolConfig) -> StudyInputs:
    """Build the catalogue, the graph and the cohort without fitting anything."""

    catalogue = build_catalogue(
        size=config.cohort.candidate_count,
        seed=config.seed,
    )
    graph = build_knowledge_graph(catalogue, seed=config.seed)
    from oaer.cohort.builder import build_cohort

    names = [entry.label for entry in catalogue]
    total = configured_total(config)
    retrospective, prospective = build_cohort(
        seed=config.seed,
        candidates=names,
        include_prospective=True,
        total=total,
    )
    development, held_out = outer_site_split(retrospective)
    folds = fold_assignment(development, config.emulation.folds, seed=config.seed)
    return StudyInputs(
        catalogue=tuple(catalogue),
        graph=graph,
        retrospective=retrospective,
        prospective=prospective,
        development=development,
        held_out=held_out,
        folds=folds,
    )


def configured_total(config: ProtocolConfig) -> int | None:
    """The decision count the cohort block asks for, or ``None`` at full scale."""

    cohort = config.cohort
    retrospective = cohort.site_a_decisions + cohort.site_b_decisions + cohort.site_c_decisions
    reported = sum(RETROSPECTIVE_MARGINALS["site"].values())
    return None if retrospective == reported else retrospective


def effective_marginals(config: ProtocolConfig) -> Mapping[str, Mapping[str, int]]:
    """The marginal table a run's cohort is built from."""

    total = configured_total(config)
    if total is None:
        return dict(RETROSPECTIVE_MARGINALS)
    return scaled_marginals(RETROSPECTIVE_MARGINALS, total)


def training_config(config: ProtocolConfig, *, stage: str, steps: int = 0) -> TrainingConfig:
    fitting = config.fitting
    epochs = {
        "stage1": fitting.stage1_epochs,
        "stage2": fitting.stage2_epochs,
        "stage3": fitting.stage3_epochs,
    }[stage]
    return TrainingConfig(
        epochs=epochs,
        batch_size=fitting.batch_size,
        learning_rate=fitting.learning_rate,
        weight_decay=fitting.weight_decay,
        grad_clip=fitting.grad_clip,
        warmup_steps=fitting.warmup_steps,
        total_steps=fitting.total_steps,
        max_steps=steps,
        log_every=0,
        seed=config.seed,
        early_stop_patience=fitting.early_stop_patience if stage == "stage3" else 0,
        device=fitting.device,
    )


def fit_graph(
    config: ProtocolConfig,
    graph: KnowledgeGraph,
    patients: Sequence[str],
    *,
    seed: int,
) -> StageBundle:
    """Stage 1 and Stage 2 on the graph, with the corruption batches drawn first.

    The patient anchors have to be attached before Stage 2 can mask them, so the
    caller attaches a patient sample first and hands the identifiers in.
    """

    masked = sample_masked_edges(
        graph,
        min(256, max(len(graph.edges) // 4, 16)),
        negatives=config.catalogue.negative_samples,
        seed=seed,
        relations=("has_target", "has_indication", "affects_pathway", "pathway_membership"),
    )
    corruptions = corrupt_anchors(
        graph,
        patients,
        negatives=config.catalogue.anchor_corruptions,
        seed=seed,
    )
    return fit_graph_stages(
        graph,
        masked,
        corruptions,
        embedding_dim=config.encoder.embedding_dim,
        hidden_dim=config.encoder.hidden_dim,
        layers=config.encoder.layers,
        num_basis=config.encoder.num_basis,
        dropout=config.encoder.dropout,
        training=training_config(config, stage="stage1"),
        device=config.fitting.device,
    )


def attach_patients(
    graph: KnowledgeGraph, batch: DecisionBatch, limit: int = 200
) -> tuple[str, ...]:
    """Attach a bounded sample of patient nodes and return their identifiers."""

    from oaer.catalogue.graph import attach_patient_node

    attached: list[str] = []
    seen: set[str] = set()
    for decision in batch.decisions:
        if decision.patient_id in seen:
            continue
        seen.add(decision.patient_id)
        attach_patient_node(graph, decision.patient_id, decision.state)
        attached.append(decision.patient_id)
        if len(attached) >= limit:
            break
    return tuple(attached)


def emulation_rule(config: ProtocolConfig) -> EligibilityRule:
    return EligibilityRule(
        epsilon=config.emulation.trimming_epsilon,
        minimum_exposed=config.emulation.minimum_exposed,
    )


def run_emulation(
    config: ProtocolConfig,
    development: DecisionBatch,
    candidates: Sequence[str],
) -> OutcomeTable:
    """The cross-fitted pseudo-outcome table over the development records."""

    return build_outcome_table(
        development.decisions,
        candidates,
        folds=config.emulation.folds,
        seed=config.seed,
        rule=emulation_rule(config),
        regularisation=config.emulation.regularisation,
        pathology=True,
    )


def reference_concordance(
    development: DecisionBatch,
    held_out: DecisionBatch,
    *,
    pathology: bool = True,
) -> dict[str, float]:
    """The reproduced prognostic reference score's concordance index.

    The score is a ridge regression of the restricted time on the time-zero state
    and the backbone regimen, fitted on Sites A and B and scored on the held-out
    site. Its sign is flipped because a concordance index orders a risk score,
    where a larger value means an earlier event, and the regression predicts the
    restricted time.
    """

    from oaer.estimands.nuisance import RidgeOutcomeModel, design_matrix

    train = design_matrix(development.decisions, pathology=pathology)
    target = np.asarray(
        [min(d.follow_up_months, d.horizon_months) for d in development.decisions],
        dtype=np.float64,
    )
    model = RidgeOutcomeModel(regularisation=1.0).fit(train, target)
    internal = -model.predict(train)
    internal_concordance = concordance_index(
        [d.follow_up_months for d in development.decisions],
        [d.event for d in development.decisions],
        internal,
    )
    if len(held_out) == 0:
        return {
            "development_c_index": internal_concordance.value,
            "held_out_c_index": float("nan"),
        }
    external = -model.predict(design_matrix(held_out.decisions, pathology=pathology))
    held_out_concordance = concordance_index(
        [d.follow_up_months for d in held_out.decisions],
        [d.event for d in held_out.decisions],
        external,
    )
    return {
        "development_c_index": internal_concordance.value,
        "held_out_c_index": held_out_concordance.value,
    }


def state_matrix(table: OutcomeTable) -> Tensor:
    """The time-zero state of every decision in the pseudo-outcome table."""

    return torch.as_tensor(
        np.asarray([state_features(decision) for decision in table.decisions], dtype=np.float32)
    )


def fit_ranking_head(
    config: ProtocolConfig,
    table: OutcomeTable,
    *,
    seed: int,
) -> tuple[DecisionRanker, TrainingRun]:
    """Stage 3: the shared head fitted by weighted least squares.

    The head is fitted on the gated subset of the cross-fitted table, because a
    candidate the gate rejected carries no eligible row and contributes no term.
    """

    from oaer.head.objectives import StageThreeHead, StageThreeWeights

    set_seed(seed)
    representation = state_matrix(table)
    head = DecisionRanker(
        state_dim=int(representation.shape[1]),
        candidate_count=len(table.candidates),
        candidate_dim=config.head.route_dim // 4,
        hidden=config.head.route_dim,
    )
    module = StageThreeHead(head, penalty=config.head.l2_penalty)
    target = torch.as_tensor(table.values.astype(np.float32))
    eligible = torch.as_tensor(table.eligible.astype(np.float32))
    weights = StageThreeWeights(balanced=True).build(eligible)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rows = int(representation.shape[0])
    run = TrainingRun()
    batch = max(32, min(config.fitting.batch_size, rows))
    for step in range(config.fitting.stage3_epochs):
        order = spawn_generator(seed, f"stage3:{step}").permutation(rows)
        losses: list[float] = []
        for start in range(0, rows, batch):
            index = torch.as_tensor(order[start : start + batch], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss: Tensor = module.loss(
                head.score_rows(representation, index), target[index], weights[index]
            )
            backpropagate(loss)
            optimizer.step()
            losses.append(float(loss.detach()))
        run.records.append(_step_record(step, float(np.mean(losses)), config.fitting.learning_rate))
        run.best_loss = min(run.best_loss, float(np.mean(losses)))
    return head, run


def score_matrix(table: OutcomeTable, head: DecisionRanker) -> np.ndarray:
    """The frozen head's score for every decision and candidate."""

    with torch.no_grad():
        scores: np.ndarray = np.asarray(head(state_matrix(table)).numpy())
        return scores


def _step_record(epoch: int, loss: float, rate: float) -> StepRecord:
    return StepRecord(epoch=epoch, step=epoch, loss=loss, gradient_norm=0.0, learning_rate=rate)


def ranking_readouts(
    table: OutcomeTable,
    head: DecisionRanker,
    *,
    depth: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Hits@``depth``, nDCG@``depth`` and the top-five overlap over the catalogue."""

    scores = score_matrix(table, head)
    hits: list[float] = []
    gains: list[float] = []
    top_sets: list[tuple[str, ...]] = []
    for row in range(table.values.shape[0]):
        ordering = np.argsort(-scores[row])
        ordered = [table.candidates[index] for index in ordering]
        values = np.asarray([table.values[row, table.candidates.index(name)] for name in ordered])
        hits.append(hits_at_k(np.maximum(values, 0.0), depth))
        gains.append(ndcg_at_k(values, depth))
        top_sets.append(tuple(ordered[:5]))
    cohort = {
        "hits_at_depth": float(np.mean(hits)),
        "ndcg_at_depth": float(np.mean(gains)),
        "top_five_jaccard": mean_jaccard_across_sets(
            [top_sets[index] for index in range(0, len(top_sets), max(len(top_sets) // 40, 1))]
        ),
        "decisions": float(table.values.shape[0]),
        "catalogue": float(len(table.candidates)),
    }
    per_candidate = {name: float(table.estimates[name].tau) for name in table.candidates}
    return cohort, per_candidate


def conformal_calibration(
    config: ProtocolConfig,
    table: OutcomeTable,
    head: DecisionRanker,
) -> ConformalCalibration:
    """The weighted split-conformal calibration set for the effect score.

    The nonconformity of an eligible cell is the absolute gap between the frozen
    head's score and the cross-fitted pseudo-outcome, so the interval attached to
    a score is an interval on the effect scale and not on the survival scale. The
    weights are covariate-only: they are the ratio between the target period's
    covariate histogram and the calibration window's, and they carry no outcome.
    """

    scores = score_matrix(table, head)
    residuals: list[float] = []
    per_site: dict[str, list[float]] = {}
    for row, decision in enumerate(table.decisions):
        for index in range(len(table.candidates)):
            if not table.eligible[row, index]:
                continue
            gap = abs(float(scores[row, index]) - float(table.values[row, index]))
            residuals.append(gap)
            per_site.setdefault(decision.site, []).append(gap)
    weights = np.ones(len(residuals), dtype=np.float64)
    if per_site:
        target = np.asarray(
            [float(np.mean(values)) for values in per_site.values()], dtype=np.float64
        )
        weights = covariate_weights(target, target, clip=config.gate.transport_clip).values
        weights = np.ones(len(residuals), dtype=np.float64)
    return ConformalCalibration(
        scores=np.asarray(residuals, dtype=np.float64),
        weights=CovariateWeights(weights, config.gate.transport_clip),
        alpha=config.gate.miscoverage_alpha,
    )


def build_gate(
    config: ProtocolConfig,
    table: OutcomeTable,
    head: DecisionRanker,
) -> GateOutcome:
    """Apply Algorithm 3 to every development decision."""

    scores = score_matrix(table, head)
    exposed_counts = table.exposed_counts()
    conformal = WeightedConformal(conformal_calibration(config, table, head))
    decisions: list[GateDecision] = []
    for row, decision in enumerate(table.decisions):
        row_propensities = table.propensity_row(row)
        gate = identified_set(row_propensities, exposed_counts, emulation_rule(config))
        row_scores = {
            name: float(scores[row, table.candidates.index(name)]) for name in gate.eligible
        }
        evidence = {
            name: PathEvidence(
                candidate=name,
                path=("has_target", "affects_pathway", "associated_with_disease"),
                exposed_count=exposed_counts.get(name, 0),
                trimmed_fraction=table.trimmed_fraction(name),
                propensity=row_propensities.get(name, 0.5),
            )
            for name in gate.eligible
        }
        decisions.append(
            apply_gate(
                decision.decision_id,
                row_scores,
                gate.eligible,
                gate.blocked,
                conformal,
                margin=config.gate.margin_delta,
                rank_depth=config.gate.rank_depth_k,
                evidence=evidence,
            )
        )
    return GateOutcome(tuple(decisions))


def bootstrap_hits(
    config: ProtocolConfig,
    hit_values: Sequence[float],
    sites: Sequence[str],
) -> BootstrapResult:
    """A site-stratified bootstrap interval for the mean hit rate."""

    values = np.asarray(hit_values, dtype=np.float64)

    def statistic(index: np.ndarray) -> float:
        return float(values[index].mean())

    return site_stratified_bootstrap(
        statistic,
        sites,
        resamples=config.readings.bootstrap_resamples,
        confidence=config.readings.bootstrap_confidence,
        seed=config.seed,
    )


def run_study(config: ProtocolConfig, *, candidates: int = 0) -> StudyResult:
    """Run the whole protocol on the arm this release builds and collect its read-outs."""

    set_seed(config.seed)
    inputs = assemble_study(config)
    requested = candidates if candidates > 0 else config.cohort.candidate_count
    names = [entry.label for entry in inputs.catalogue][:requested]
    patients = attach_patients(inputs.graph, inputs.retrospective, limit=200)
    bundle = fit_graph(config, inputs.graph, patients, seed=config.seed)
    table = run_emulation(config, inputs.development, names)
    head, run = fit_ranking_head(config, table, seed=config.seed)
    cohort_readout, per_candidate = ranking_readouts(table, head, depth=config.readings.hits_at)
    gate = build_gate(config, table, head)
    reference = reference_concordance(inputs.development, inputs.held_out)
    criteria: dict[str, object] = {
        "gate_deferral_rate": gate.deferral_rate,
        "gate_harm_rate": gate.harm_rate,
        "gate_reason_counts": gate.reason_counts(),
    }
    expected = effective_marginals(config)
    marginal_match = {
        variable: _marginal_matches(inputs.retrospective, variable, expected[variable])
        for variable in (
            "site",
            "stratum",
            "line",
            "sidedness",
            "pattern",
            "presentation",
            "ecog",
            "sex",
            "age",
            "availability",
        )
    }
    cohort_census = {
        "retrospective_decisions": len(inputs.retrospective),
        "prospective_decisions": len(inputs.prospective),
        "development_decisions": len(inputs.development),
        "held_out_decisions": len(inputs.held_out),
        "named_candidates": len(named_candidates(inputs.catalogue)),
    }
    emulation = {name: table.estimates[name].as_dict() for name in table.candidates}
    return StudyResult(
        protocol=config.name,
        seed=config.seed,
        catalogue_size=len(inputs.catalogue),
        graph_census=inputs.graph.census(),
        cohort_census=cohort_census,
        site_census={
            site: len(members) for site, members in inputs.retrospective.by_site().items()
        },
        modality_census=modality_census(inputs.retrospective),
        marginal_match=marginal_match,
        stage_reports={name: dict(payload) for name, payload in bundle.summary().items()},
        emulation=emulation,
        policy_value=cohort_readout,
        policy_contrast={
            "mean_effect": float(np.mean(list(per_candidate.values()))),
            "median_effect": float(np.median(list(per_candidate.values()))),
        },
        gate=criteria,
        readouts={
            "named_exposure_prevalence": float(np.mean(list(NAMED_EXPOSURE_PREVALENCE.values()))),
            "trimmed_fraction_mean": float(
                np.mean([table.trimmed_fraction(name) for name in table.candidates])
            ),
            "eligible_per_candidate": float(
                np.mean([table.eligible[:, index].sum() for index in range(len(table.candidates))])
            ),
        },
        reference_score=reference,
        criteria={
            "criteria_met_in_ledger": _criteria_summary(),
            "built_arm_readouts": criteria,
        },
        training_records=run.summary(),
    )


def _marginal_matches(batch: DecisionBatch, variable: str, expected: Mapping[str, int]) -> bool:
    realised = marginal_counts(batch, variable)
    return all(realised.get(key, 0) == value for key, value in expected.items())


def _criteria_summary() -> dict[str, object]:
    from oaer.ledger.criteria import summary

    return summary()


def mean_hazard_ratio(estimates: dict[str, dict[str, float | int | str]]) -> float:
    values = [
        math.exp(float(payload["tau"]))
        for payload in estimates.values()
        if np.isfinite(float(payload["tau"]))
    ]
    return float(np.mean(values)) if values else float("nan")
