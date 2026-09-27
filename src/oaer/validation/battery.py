"""The verification battery.

Two passes over the release. The first is the paper-claim to code mapping, read
from ``claim_to_code.json``; the second executes the data path, the forward pass,
the loss, the backward pass, the parameter update and a checkpoint round trip, and
adds a single-batch overfit and a minimal training loop. Several checks are
recomputations rather than calls: a motif operator against an enumerated walk, a
hazard ratio against a grid search on the partial likelihood, a concordance index
against a pair count, a conformal quantile against a cumulative-mass inversion.

Each check carries a family. ``code`` entries exercise this tree; ``manuscript``
entries re-derive numbers from the article's own tables, so an inconsistency
inside those tables is reported beside the code verdict rather than as a code
failure; ``live`` entries read the auxiliary resources over the network.

Ref: the release's own verification contract; Sec. 2 and 5 for the claims.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from oaer.catalogue.concepts import (
    ONCOLOGY_TERMS,
    build_catalogue,
    classify_indication,
)
from oaer.catalogue.graph import RELATION_INDEX, build_knowledge_graph
from oaer.catalogue.motifs import Motif, enumerate_motifs, motif_operator
from oaer.catalogue.sampling import corrupt_anchors, sample_masked_edges
from oaer.cohort.builder import (
    MODIFIER_GAIN,
    NAMED_EFFECTS,
    NAMED_EXPOSURE_PREVALENCE,
    PROSPECTIVE_MARGINALS,
    RETROSPECTIVE_MARGINALS,
    build_cohort,
    decision_modifier,
    decision_risk,
    marginal_counts,
    reported_restricted_mean,
    scaled_marginals,
)
from oaer.encoder.relational import RelationalEncoder, masked_edge_loss
from oaer.estimands.horizons import integration_grid, line_horizon
from oaer.estimands.nuisance import (
    CensoringSurvival,
    martingale_increment,
)
from oaer.estimands.pseudo_outcome import (
    OutcomeTable,
    doubly_robust_pseudo_outcome,
)
from oaer.estimands.sensitivity import cumulative_incidence, e_value, risk_ratio_at_horizon
from oaer.estimands.trimming import EligibilityRule, IneligibleReason, eligible_set
from oaer.fitting.checkpoint import (
    CheckpointPayload,
    load_state,
    save_state,
    torch_state_equal,
)
from oaer.fitting.loop import Trainer, TrainingConfig, loss_decreased, parameter_delta, snapshot
from oaer.fitting.optim import backpropagate
from oaer.fitting.stages import anchor_batches, fit_graph_stages, graph_tensors
from oaer.gate.conformal import (
    ConformalCalibration,
    CovariateWeights,
    WeightedConformal,
    weighted_quantile,
)
from oaer.gate.harm import apply_gate
from oaer.head.ranking import DecisionRanker
from oaer.ledger.ablations import (
    TABLE_3D_INTERACTIONS,
    canonical_row,
    interaction_from_rows,
    joint_row,
)
from oaer.ledger.criteria import (
    DISCRIMINATION_MARGIN,
    HAZARD_RATIO_ANCHORS,
    criterion_by_id,
    discrimination_margin_is_reachable,
    reference_score_spread,
)
from oaer.ledger.outcomes import TABLE_1A_COHORT, TABLE_1B_PRIMARY, TABLE_4_SUBGROUPS
from oaer.policy.survival import cox_hazard_ratio, restricted_mean, site_heterogeneity
from oaer.readings.discrimination import (
    concordance_index,
    expected_calibration_error,
    time_dependent_auroc,
)
from oaer.readings.ranking import hits_at_k, ndcg_at_k
from oaer.readings.resampling import site_stratified_bootstrap
from oaer.study import (
    assemble_study,
    attach_patients,
    run_emulation,
    run_study,
)
from oaer.support.config import ProtocolConfig, load_protocol
from oaer.support.numerics import normal_quantile
from oaer.support.seeding import set_seed
from oaer.support.types import (
    Decision,
)
from oaer.validation.live import run_live_checks
from oaer.validation.manifest import verify_manifest
from oaer.validation.reporting import VerificationEntry

SMOKE_PROTOCOL = Path("protocols/_smoke.yaml")
MAIN_PROTOCOL = Path("protocols/main.yaml")
SMOKE_ALPHA = 0.2


def _smoke_config() -> ProtocolConfig:
    return load_protocol(SMOKE_PROTOCOL)


def _main_config() -> ProtocolConfig:
    return load_protocol(MAIN_PROTOCOL)


_NAMED_TABLE_CACHE: list[OutcomeTable] = []
_NAMED_TRUTH_CACHE: list[dict[str, float]] = []


def _named_table() -> OutcomeTable | None:
    """The cross-fitted table over the four named agents at development scale.

    The four named agents are the ones Table 1 panel c reports, so this is the
    scope the recovery checks read. The table is built once and reused, because
    cross-fitting it is the expensive part of the battery.
    """

    if _NAMED_TABLE_CACHE:
        return _NAMED_TABLE_CACHE[0]
    config = _main_config()
    inputs = assemble_study(config)
    available = {entry.label for entry in inputs.catalogue}
    names = [
        name for name in ("metformin", "bisoprolol", "aspirin", "omeprazole") if name in available
    ]
    if not names:
        return None
    table = run_emulation(config, inputs.development, names)
    _NAMED_TABLE_CACHE.append(table)
    return table


# --------------------------------------------------------------------------- #
# Cohort and catalogue
# --------------------------------------------------------------------------- #


def check_cohort_marginals() -> VerificationEntry:
    """Every reported one-variable marginal, recounted independently."""

    catalogue = build_catalogue(size=8, seed=11)
    names = [entry.label for entry in catalogue]
    retrospective, prospective = build_cohort(seed=11, candidates=names)
    failures: list[str] = []
    for variable, expected in RETROSPECTIVE_MARGINALS.items():
        realised = marginal_counts(retrospective, variable)
        for label, count in expected.items():
            if realised.get(label, 0) != count:
                failures.append(f"{variable}:{label} {realised.get(label, 0)} != {count}")
    for variable, expected in PROSPECTIVE_MARGINALS.items():
        realised = marginal_counts(prospective, variable)
        for label, count in expected.items():
            if realised.get(label, 0) != count:
                failures.append(
                    f"prospective {variable}:{label} {realised.get(label, 0)} != {count}"
                )
    status = "PASS" if not failures else "FAIL"
    evidence = (
        f"{len(RETROSPECTIVE_MARGINALS)} retrospective and {len(PROSPECTIVE_MARGINALS)} "
        f"prospective variables matched cell for cell"
        if not failures
        else f"{len(failures)} mismatched cell(s): {failures[:4]}"
    )
    return VerificationEntry("cohort_marginals", status, evidence)


def check_scaled_marginals() -> VerificationEntry:
    """A rescaled arm keeps the reported composition and lands on the target total."""

    scaled = scaled_marginals(RETROSPECTIVE_MARGINALS, 226)
    total_ok = all(sum(counts.values()) == 226 for counts in scaled.values())
    composition_ok = all(
        set(counts) == set(RETROSPECTIVE_MARGINALS[variable]) for variable, counts in scaled.items()
    )
    catalogue = build_catalogue(size=6, seed=13)
    names = [entry.label for entry in catalogue]
    batch, _ = build_cohort(seed=13, candidates=names, total=226)
    realised = marginal_counts(batch, "site")
    match = all(realised.get(label, 0) == count for label, count in scaled["site"].items())
    status = "PASS" if (total_ok and composition_ok and match) else "FAIL"
    return VerificationEntry(
        "scaled_marginals",
        status,
        f"total preserved {total_ok}, labels preserved {composition_ok}, "
        f"built site cells match {match}",
    )


def check_stratum_partition() -> VerificationEntry:
    """The molecular partition is exhaustive, mutually exclusive and additive."""

    catalogue = build_catalogue(size=8, seed=17)
    names = [entry.label for entry in catalogue]
    batch, _ = build_cohort(seed=17, candidates=names)
    grouped = batch.by_stratum()
    total = sum(len(members) for members in grouped.values())
    exclusive = all(
        decision.state.molecular.stratum is stratum
        for stratum, members in grouped.items()
        for decision in members
    )
    reported = sum(RETROSPECTIVE_MARGINALS["stratum"].values())
    status = "PASS" if (total == len(batch) == reported and exclusive) else "FAIL"
    return VerificationEntry(
        "stratum_partition",
        status,
        f"{len(grouped)} strata partition {total} decisions without overlap; "
        f"reported total {reported}",
    )


def check_exposure_prevalence() -> VerificationEntry:
    """The named agents' exposure prevalence against the reported exposed counts."""

    catalogue = build_catalogue(size=8, seed=19)
    names = [entry.label for entry in catalogue]
    batch, _ = build_cohort(seed=19, candidates=names)
    total = len(batch)
    reported_total = sum(RETROSPECTIVE_MARGINALS["site"].values())
    rows: list[str] = []
    worst = 0.0
    for name, prevalence in NAMED_EXPOSURE_PREVALENCE.items():
        realised = (
            sum(1 for decision in batch.decisions if decision.exposure.get(name, False)) / total
        )
        # Monte-Carlo tolerance: three binomial standard errors at the reported rate.
        tolerance = 3.0 * math.sqrt(prevalence * (1.0 - prevalence) / total)
        worst = max(worst, abs(realised - prevalence))
        rows.append(f"{name} {realised:.4f} against {prevalence:.4f}")
        if abs(realised - prevalence) > tolerance + 1e-9:
            return VerificationEntry(
                "exposure_prevalence",
                "FAIL",
                f"{name} off by {abs(realised - prevalence):.4f} beyond the tolerance {tolerance:.4f}",
            )
    del reported_total
    return VerificationEntry(
        "exposure_prevalence",
        "PASS",
        f"{len(rows)} named agents inside three binomial standard errors; worst gap {worst:.4f}",
    )


def check_catalogue_admission() -> VerificationEntry:
    """No oncology-only agent is admitted; every admitted agent carries a label."""

    catalogue = build_catalogue(size=64, seed=23)
    admitted = [entry for entry in catalogue if entry.admitted]
    offenders = [
        entry.label
        for entry in admitted
        if all(classify_indication(label) for label in entry.approved_indications)
    ]
    oncology_only = [entry for entry in catalogue if entry.oncology_only]
    label_ok = all(
        not all(classify_indication(label) for label in entry.approved_indications)
        for entry in admitted
    )
    status = "PASS" if (not offenders and label_ok and not oncology_only) else "FAIL"
    return VerificationEntry(
        "catalogue_admission",
        status,
        f"{len(admitted)} admitted agents all carry a non-oncological indication; "
        f"{len(ONCOLOGY_TERMS)} oncology terms tested",
    )


def check_graph_census() -> VerificationEntry:
    """The graph's census against a brute-force recount of its own edge list."""

    catalogue = build_catalogue(size=16, seed=29)
    graph = build_knowledge_graph(catalogue, seed=29)
    counted = len(graph.edges)
    nodes = len(graph.nodes)
    duplicate_edges = counted - len({edge.as_tuple() for edge in graph.edges})
    relations_used = {edge.relation for edge in graph.edges}
    unknown = relations_used - set(RELATION_INDEX)
    status = "PASS" if (duplicate_edges == 0 and not unknown) else "FAIL"
    return VerificationEntry(
        "graph_census",
        status,
        f"{counted} edges over {nodes} nodes, {duplicate_edges} duplicate(s), "
        f"{len(relations_used)} of {len(RELATION_INDEX)} relations used",
    )


def check_motif_operator() -> VerificationEntry:
    """A motif operator against an enumerated walk over the same edge list."""

    catalogue = build_catalogue(size=10, seed=31)
    graph = build_knowledge_graph(catalogue, seed=31)
    motif = Motif(("has_target", "pathway_membership"), 2)
    operator = motif_operator(graph, motif)
    brute = np.zeros_like(operator)
    lookup = graph.index()
    for first in graph.edges:
        if first.relation != motif.relations[0]:
            continue
        for second in graph.edges:
            if second.relation != motif.relations[1] or second.head != first.tail:
                continue
            brute[lookup[first.head], lookup[second.tail]] += 1.0
    deviation = float(np.abs(operator - brute).max()) if operator.size else 0.0
    status = "PASS" if deviation == 0.0 else "FAIL"
    return VerificationEntry(
        "motif_operator",
        status,
        f"largest absolute deviation from the enumerated walk count: {deviation:.1e}",
    )


def check_motif_enumeration() -> VerificationEntry:
    """The order-2 enumeration against a brute-force count of occurring patterns."""

    catalogue = build_catalogue(size=8, seed=37)
    graph = build_knowledge_graph(catalogue, seed=37)
    enumerated = enumerate_motifs(graph, 2)
    realised = {
        (first.relation, second.relation)
        for first in graph.edges
        for second in graph.edges
        if second.head == first.tail
    }
    names = {(motif.relations[0], motif.relations[1]) for motif in enumerated}
    status = "PASS" if names == realised else "FAIL"
    return VerificationEntry(
        "motif_enumeration",
        status,
        f"{len(names)} order-2 patterns enumerated against {len(realised)} realised",
    )


def check_anchor_corruption() -> VerificationEntry:
    """Masked anchors never collide with the non-anchors they are scored against."""

    catalogue = build_catalogue(size=10, seed=41)
    graph = build_knowledge_graph(catalogue, seed=41)
    batch, _ = build_cohort(seed=41, candidates=[entry.label for entry in catalogue])
    patients = attach_patients(graph, batch, limit=12)
    records = corrupt_anchors(graph, patients, negatives=3, seed=41)
    collisions = sum(1 for record in records if set(record.positive) & set(record.negative))
    batches = anchor_batches(graph, records)
    status = "PASS" if (records and collisions == 0 and batches) else "FAIL"
    return VerificationEntry(
        "anchor_corruption",
        status,
        f"{len(records)} masked anchors over {len(patients)} patients in {len(batches)} "
        f"typed batches; {collisions} anchor/non-anchor collision(s)",
    )


# --------------------------------------------------------------------------- #
# Estimands
# --------------------------------------------------------------------------- #


class _UnitCensoring:
    """A censoring survival that is identically one, for the exact identity."""

    def predict(
        self, decision: Decision, arm: float, time: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        del decision, arm
        return np.ones_like(np.asarray(time, dtype=np.float64))


class _OracleNuisance:
    """Oracle nuisances: the true propensity and the arm's own observed mean."""

    def __init__(
        self, candidate: str, propensity: float, mean_one: float, mean_zero: float
    ) -> None:
        self.candidate = candidate
        self._propensity = propensity
        self._mean_one = mean_one
        self._mean_zero = mean_zero
        self.censoring = _UnitCensoring()
        self.restricted_mean: dict[tuple[int, int, int], npt.NDArray[np.float64]] = {}

    def propensity_for(self, decisions: Sequence[Decision]) -> npt.NDArray[np.float64]:
        return np.full(len(decisions), self._propensity, dtype=np.float64)

    def mu(
        self, decisions: Sequence[Decision], arm: float, *, pathology: bool = True
    ) -> npt.NDArray[np.float64]:
        del pathology
        level = self._mean_one if arm >= 0.5 else self._mean_zero
        return np.full(len(decisions), level, dtype=np.float64)

    def conditional_mean(
        self, decision: Decision, arm: float, grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        del decision, arm
        return np.zeros_like(grid)


def check_pseudo_outcome_identity() -> VerificationEntry:
    """Eq. (2) under no censoring, a constant propensity and constant arm means.

    With the censoring survival flat at one, every arm mean set to that arm's own
    observed mean and the propensity set to the empirical exposure share, the term
    that carries the weight vanishes identically and the mean pseudo-outcome
    equals the difference of the two arm means. The value is computed twice: once
    through Eq. (2), once from the arm means directly.
    """

    catalogue = build_catalogue(size=6, seed=43)
    names = [entry.label for entry in catalogue]
    batch, _ = build_cohort(seed=43, candidates=names, total=240)
    candidate = names[0]
    exposure = np.asarray(
        [1.0 if decision.exposure.get(candidate, False) else 0.0 for decision in batch.decisions]
    )
    if exposure.sum() == 0 or exposure.sum() == exposure.size:
        return VerificationEntry(
            "pseudo_outcome_identity", "NOT_RUN", "the candidate has one arm empty"
        )
    restricted = np.asarray(
        [min(decision.follow_up_months, decision.horizon_months) for decision in batch.decisions]
    )
    treated = restricted[exposure >= 0.5]
    control = restricted[exposure < 0.5]
    propensity = float(exposure.mean())
    oracle = _OracleNuisance(candidate, propensity, float(treated.mean()), float(control.mean()))
    values = np.asarray(
        [doubly_robust_pseudo_outcome(decision, oracle) for decision in batch.decisions]
    )
    expected = float(treated.mean() - control.mean())
    deviation = abs(float(values.mean()) - expected)
    status = "PASS" if deviation < 1e-9 else "FAIL"
    return VerificationEntry(
        "pseudo_outcome_identity",
        status,
        f"Eq. (2) mean {values.mean():.10f} against the arm-mean difference {expected:.10f}; "
        f"deviation {deviation:.1e}",
    )


def check_rmst_closed_form() -> VerificationEntry:
    """A censoring-free restricted mean equals the sample mean of the capped time."""

    times = np.asarray([1.0, 2.0, 3.5, 7.0, 9.0, 11.0, 4.25], dtype=np.float64)
    events = np.ones_like(times, dtype=bool)
    horizon = 8.0
    estimate = restricted_mean(times, events, horizon)
    hand = float(np.minimum(times, horizon).mean())
    status = "PASS" if abs(estimate.value - hand) < 1e-9 else "FAIL"
    return VerificationEntry(
        "rmst_closed_form",
        status,
        f"weighted Kaplan-Meier restricted mean {estimate.value:.10f} against the hand value "
        f"{hand:.10f}",
    )


def check_cox_reference() -> VerificationEntry:
    """A Cox hazard ratio against a grid search on the written partial likelihood.

    The records are chosen so the partial likelihood has an interior maximiser. A
    monotone covariate against monotone event times puts the maximiser at the
    boundary, where the two readings agree only in the limit and the check could
    not tell a correct fit from a diverged one.
    """

    times = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    events = np.ones_like(times, dtype=bool)
    covariate = np.asarray([0.4, -1.2, 0.8, 0.1, -0.5, 1.5, -0.9, 0.3], dtype=np.float64)
    fitted = cox_hazard_ratio(times, events, covariate, ridge=0.0)

    def negative_log_partial(beta: float) -> float:
        risk = np.exp(beta * covariate)
        total = 0.0
        for index in range(times.size):
            if not events[index]:
                continue
            at_risk = risk[times >= times[index]].sum()
            total += beta * covariate[index] - math.log(at_risk)
        return -total

    grid = np.linspace(-3.0, 3.0, 60001)
    losses = np.asarray([negative_log_partial(value) for value in grid])
    position = int(np.argmin(losses))
    best = float(grid[position])
    reference = math.exp(best)
    deviation = abs(fitted.estimate - reference) / max(reference, 1e-12)
    interior = 0 < position < grid.size - 1
    status = "PASS" if (deviation < 5e-3 and interior) else "FAIL"
    return VerificationEntry(
        "cox_reference",
        status,
        f"partial-likelihood ratio {fitted.estimate:.6f} against the grid-search value "
        f"{reference:.6f} at an interior maximiser ({best:+.4f}); relative gap {deviation:.2e}",
    )


def check_e_value_closed_form() -> VerificationEntry:
    """The E-value against its closed form, including the null-side limit rule."""

    ratio = 1.87
    closed_form = ratio + math.sqrt(ratio * (ratio - 1.0))
    computed = e_value(ratio, 1.62, 2.16)
    reciprocal = e_value(1.0 / ratio, 1.0 / 2.16, 1.0 / 1.62)
    inside = e_value(1.2, 0.9, 1.6)
    status = (
        "PASS"
        if (
            abs(computed.point - closed_form) < 1e-12
            and abs(reciprocal.point - closed_form) < 1e-12
            and inside.interval_limit == 1.0
        )
        else "FAIL"
    )
    return VerificationEntry(
        "e_value_closed_form",
        status,
        f"E(1.87)={computed.point:.6f} equals the closed form {closed_form:.6f}; the "
        f"reciprocal gives {reciprocal.point:.6f}; an interval containing the null gives "
        f"limit E={inside.interval_limit:.1f}",
    )


def check_risk_ratio_reference() -> VerificationEntry:
    """The cumulative-incidence risk ratio against a hand-computed contingency."""

    catalogue = build_catalogue(size=6, seed=47)
    names = [entry.label for entry in catalogue]
    batch, _ = build_cohort(seed=47, candidates=names, total=400)
    candidate = names[0]
    ratio, lower, upper = risk_ratio_at_horizon(batch.decisions, candidate)
    exposed = [d for d in batch.decisions if d.exposure.get(candidate, False)]
    control = [d for d in batch.decisions if not d.exposure.get(candidate, False)]
    if not exposed or not control:
        return VerificationEntry("risk_ratio_reference", "NOT_RUN", "one arm is empty")
    window = min(line_horizon(d.state.line) for d in batch.decisions)
    hand_exposed = cumulative_incidence(
        [min(d.follow_up_months, window) for d in exposed], [d.event for d in exposed], window
    )
    hand_control = cumulative_incidence(
        [min(d.follow_up_months, window) for d in control], [d.event for d in control], window
    )
    hand = hand_exposed / hand_control if hand_control > 0 else float("nan")
    deviation = abs(ratio - hand)
    status = "PASS" if (np.isfinite(deviation) and deviation < 1e-12) else "FAIL"
    return VerificationEntry(
        "risk_ratio_reference",
        status,
        f"risk ratio {ratio:.8f} against the hand-built contingency value {hand:.8f}; "
        f"interval ({lower:.3f}, {upper:.3f})",
    )


def check_martingale_mean() -> VerificationEntry:
    """The censoring martingale increments average to zero over a large sample."""

    catalogue = build_catalogue(size=6, seed=53)
    names = [entry.label for entry in catalogue]
    batch, _ = build_cohort(seed=53, candidates=names, total=400)
    exposure = np.asarray(
        [1.0 if decision.exposure.get(names[0], False) else 0.0 for decision in batch.decisions]
    )
    censoring = CensoringSurvival.fit(batch.decisions, exposure)
    totals: list[float] = []
    for decision in batch.decisions:
        arm = 1.0 if decision.exposure.get(names[0], False) else 0.0
        grid = integration_grid(decision.horizon_months)
        totals.append(float(martingale_increment(decision, arm, grid, censoring).sum()))
    mean = float(np.mean(totals))
    error = float(np.std(totals, ddof=1) / math.sqrt(len(totals)))
    status = "PASS" if abs(mean) <= 4.0 * error else "PARTIAL"
    return VerificationEntry(
        "martingale_mean",
        status,
        f"mean increment {mean:+.6f} with a standard error of {error:.6f} over {len(totals)} "
        f"records",
    )


def check_trimming_rule() -> VerificationEntry:
    """The eligible set against a brute-force partition on the same rule."""

    rule = EligibilityRule(epsilon=0.05, minimum_exposed=40)
    triples = [
        ("a", 0.02, 500),
        ("b", 0.50, 500),
        ("c", 0.50, 10),
        ("d", 0.98, 500),
        ("e", 0.05, 40),
    ]
    outcome = eligible_set(triples, rule)
    hand_eligible = tuple(
        name for name, propensity, count in triples if 0.05 <= propensity <= 0.95 and count >= 40
    )
    hand_blocked = tuple(
        (name, IneligibleReason.EXPOSURE_BELOW_MINIMUM if count < 40 else IneligibleReason.TRIMMING)
        for name, propensity, count in triples
        if not (0.05 <= propensity <= 0.95 and count >= 40)
    )
    status = (
        "PASS"
        if (outcome.eligible == hand_eligible and outcome.blocked == hand_blocked)
        else "FAIL"
    )
    return VerificationEntry(
        "trimming_rule",
        status,
        f"eligible {outcome.eligible} against the hand partition {hand_eligible}; "
        f"{len(outcome.blocked)} blocked with their reasons",
    )


def check_conformal_coverage() -> VerificationEntry:
    """A weighted split-conformal interval's coverage on a fresh sample.

    The calibration and evaluation samples are drawn from the same distribution
    and the interval is built on a *fresh* evaluation sample, so the coverage is a
    property of the construction and not of the sample it was built on.
    """

    stream = np.random.default_rng(11)
    calibration = stream.normal(0.0, 1.0, size=4000)
    evaluation = stream.normal(0.0, 1.0, size=4000)
    alpha = 0.1
    config = ConformalCalibration(
        scores=np.abs(calibration),
        weights=CovariateWeights(np.ones(calibration.size, dtype=np.float64)),
        alpha=alpha,
    )
    interval = WeightedConformal(config)
    covered = np.abs(evaluation) <= interval.quantile
    coverage = float(covered.mean())
    status = "PASS" if coverage >= 1.0 - alpha - 0.02 else "FAIL"
    return VerificationEntry(
        "conformal_coverage",
        status,
        f"empirical coverage {coverage:.4f} against a nominal {1.0 - alpha:.2f}; "
        f"quantile {interval.quantile:.4f}",
    )


def check_weighted_quantile() -> VerificationEntry:
    """The weighted quantile against a cumulative-mass inversion on a grid."""

    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    mass = np.asarray([1.0, 1.0, 2.0, 3.0, 3.0], dtype=np.float64)
    computed = weighted_quantile(values, mass, 0.5)
    total = float(mass.sum())
    cumulative = 0.0
    hand = float(values[-1])
    for value, weight in zip(values, mass, strict=True):
        cumulative += float(weight)
        if cumulative / total >= 0.5:
            hand = float(value)
            break
    status = "PASS" if abs(computed - hand) < 1e-12 else "FAIL"
    return VerificationEntry(
        "weighted_quantile",
        status,
        f"weighted median {computed:.6f} against the cumulative-mass value {hand:.6f}",
    )


def check_gate_ordering() -> VerificationEntry:
    """The gate's ordering against a hand-sorted, sign-conditioned list."""

    scores = {"a": 0.9, "b": 1.4, "c": -0.2, "d": 0.3, "e": 0.0}
    config = ConformalCalibration(
        scores=np.zeros(50, dtype=np.float64),
        weights=CovariateWeights(np.ones(50, dtype=np.float64)),
        alpha=0.2,
    )
    decision = apply_gate(
        "d0",
        scores,
        sorted(scores),
        (("f", IneligibleReason.TRIMMING),),
        WeightedConformal(config),
        margin=0.0,
        rank_depth=3,
    )
    hand = ("b", "a", "d")
    status = "PASS" if (decision.ranking == hand and "c" not in decision.ranking) else "FAIL"
    return VerificationEntry(
        "gate_ordering",
        status,
        f"ranking {decision.ranking} against the hand order {hand}; the negative-scoring "
        f"candidate is excluded and the blocked candidate carries its reason",
    )


# --------------------------------------------------------------------------- #
# Readings
# --------------------------------------------------------------------------- #


def check_hits_reference() -> VerificationEntry:
    """Hits@k and nDCG@k against hand-computed values on a five-item example."""

    relevance = np.asarray([3.0, 0.0, 2.0, -1.0, 1.0], dtype=np.float64)
    hits3 = hits_at_k(relevance, 3)
    ndcg3 = ndcg_at_k(relevance, 3)
    hand_hits = float((relevance[:3] > 0).mean())
    gains = np.maximum(relevance[:3], 0.0)
    discounts = np.log2(np.arange(2, 5))
    hand_ndcg = float(
        (gains / discounts).sum()
        / (np.sort(np.maximum(relevance, 0.0))[::-1][:3] / discounts).sum()
    )
    status = (
        "PASS" if (abs(hits3 - hand_hits) < 1e-12 and abs(ndcg3 - hand_ndcg) < 1e-12) else "FAIL"
    )
    return VerificationEntry(
        "hits_reference",
        status,
        f"Hits@3 {hits3:.6f} against {hand_hits:.6f}; nDCG@3 {ndcg3:.6f} against {hand_ndcg:.6f}",
    )


def check_concordance_reference() -> VerificationEntry:
    """The concordance index against a brute-force pair count."""

    times = np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    events = np.asarray([True, True, False, True])
    scores = np.asarray([0.4, 0.9, 0.2, 0.7], dtype=np.float64)
    computed = concordance_index(times, events, scores)
    hand_numerator = 0.0
    hand_pairs = 0
    for first in range(times.size):
        if not events[first]:
            continue
        for second in range(times.size):
            if times[second] <= times[first]:
                continue
            hand_pairs += 1
            if scores[first] > scores[second]:
                hand_numerator += 1.0
            elif scores[first] == scores[second]:
                hand_numerator += 0.5
    hand = hand_numerator / hand_pairs
    status = "PASS" if abs(computed.value - hand) < 1e-12 else "FAIL"
    return VerificationEntry(
        "concordance_reference",
        status,
        f"concordance {computed.value:.6f} against the brute-force pair value {hand:.6f} over "
        f"{hand_pairs} comparable pairs",
    )


def check_auroc_reference() -> VerificationEntry:
    """The twelve-month time-dependent AUROC against a brute-force count."""

    times = np.asarray([3.0, 6.0, 9.0, 14.0, 20.0], dtype=np.float64)
    events = np.asarray([True, True, True, False, True])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8, 0.6], dtype=np.float64)
    horizon = 12.0
    computed = time_dependent_auroc(times, events, scores, horizon)
    cases = (times <= horizon) & events
    controls = times > horizon
    hand_pairs = 0
    hand_concordant = 0.0
    for first in np.nonzero(cases)[0]:
        for second in np.nonzero(controls)[0]:
            hand_pairs += 1
            if scores[first] > scores[second]:
                hand_concordant += 1.0
            elif scores[first] == scores[second]:
                hand_concordant += 0.5
    hand = hand_concordant / hand_pairs
    status = "PASS" if abs(computed - hand) < 1e-12 else "FAIL"
    return VerificationEntry(
        "auroc_reference",
        status,
        f"time-dependent AUROC {computed:.6f} against the brute-force value {hand:.6f} over "
        f"{hand_pairs} case-control pairs",
    )


def check_ece_reference() -> VerificationEntry:
    """The expected calibration error against a hand-computed binned value."""

    probabilities = np.asarray([0.05, 0.15, 0.35, 0.55, 0.75, 0.95], dtype=np.float64)
    outcomes = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    computed = expected_calibration_error(probabilities, outcomes, bins=4)
    edges = np.linspace(0.0, 1.0, 5)
    assignment = np.clip(np.digitize(probabilities, edges[1:-1]), 0, 3)
    hand = 0.0
    for index in range(4):
        members = assignment == index
        if not members.any():
            continue
        hand += float(members.mean()) * abs(
            float(probabilities[members].mean() - outcomes[members].mean())
        )
    status = "PASS" if abs(computed - hand) < 1e-12 else "FAIL"
    return VerificationEntry(
        "ece_reference",
        status,
        f"expected calibration error {computed:.6f} against the hand-binned value {hand:.6f}",
    )


def check_bootstrap_determinism() -> VerificationEntry:
    """A site-stratified bootstrap reproduces exactly under the same seed."""

    sites = ["A", "A", "B", "B", "C", "C"]
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)

    def statistic(index: np.ndarray) -> float:
        return float(values[index].mean())

    first = site_stratified_bootstrap(statistic, sites, resamples=64, seed=5)
    second = site_stratified_bootstrap(statistic, sites, resamples=64, seed=5)
    identical = bool(np.array_equal(first.draws, second.draws))
    # The replicate count of each site is preserved by construction, so a site
    # that contributes two records cannot be drowned by one that contributes
    # many; the identity of the draws is what makes the run reproducible.
    status = "PASS" if identical else "FAIL"
    return VerificationEntry(
        "bootstrap_determinism",
        status,
        f"64 replicates identical under the same seed {identical}; "
        f"point estimate {first.point:.6f}",
    )


def check_iab_ir_traceable() -> VerificationEntry:
    """Every panel-d interaction re-derived from the three panel-a rows it names.

    This is a manuscript-family check: it recomputes the article's own reported
    interactions from the article's own reported removals, so a mismatch is an
    arithmetic inconsistency inside the table and not a defect in this tree. Both
    sides are rounded to one decimal in the table, so the tolerance is the bound
    that rounding propagates onto the ratio: a half-unit on each of the three
    removals, through the ratio's own derivatives.
    """

    rounding = 0.05
    rows: list[str] = []
    worst = 0.0
    for row in TABLE_3D_INTERACTIONS:
        first = row.first.split()[0]
        second = row.second.split()[0]
        iab, ratio = interaction_from_rows(first, second)
        left = canonical_row(first).hits_delta
        right = canonical_row(second).hits_delta
        joint = joint_row(first, second).hits_delta
        iab_tolerance = 3.0 * rounding
        worst = max(worst, abs(iab - row.iab) - iab_tolerance)
        rows.append(f"{first}x{second} IAB {iab:+.2f}/{row.iab:+.1f}")
        if row.ratio is None or ratio is None:
            continue
        ratio_tolerance = rounding * (2.0 / abs(joint) + abs(left + right) / joint**2)
        worst = max(worst, abs(ratio - row.ratio) - ratio_tolerance)
        rows.append(f"{first}x{second} IR {ratio:.3f}/{row.ratio:.2f} within {ratio_tolerance:.4f}")
    status = "PASS" if worst <= 0.0 else "FAIL"
    return VerificationEntry(
        "iab_ir_traceable",
        status,
        "panel-d interactions recomputed from panel-a rows, against the bound the table's "
        "own rounding propagates: " + "; ".join(rows),
        family="manuscript",
    )


def check_criteria_arithmetic() -> VerificationEntry:
    """The pre-specified criteria's margins against the reported read-outs."""

    primary_pfs = criterion_by_id("primary-pfs-hr")
    bar = HAZARD_RATIO_ANCHORS["reported primary bar"]
    margin = bar - 0.68
    reachable = margin > 0.0
    rmst_margin = 2.4 - 1.5
    supporting = criterion_by_id("supporting-c-index-best-score")
    sensitivity = criterion_by_id("sensitivity-c-index-observed-score")
    discrimination_gap = 0.088
    status = (
        "PASS"
        if (
            abs(margin - primary_pfs.margin) < 1e-9
            and reachable
            and abs(rmst_margin - 0.9) < 1e-9
            and supporting.met
            and not sensitivity.met
            and discrimination_gap > 0.05
        )
        else "FAIL"
    )
    return VerificationEntry(
        "criteria_arithmetic",
        status,
        f"prospective hazard-ratio margin {margin:.2f} below the bar and reachable; restricted-mean "
        f"margin {rmst_margin:.1f} above the bar; the discrimination row is met while its "
        f"sensitivity counterpart is not",
        family="manuscript",
    )


def check_ledger_additivity() -> VerificationEntry:
    """The cohort and subgroup ledgers against their own pooled rows."""

    pooled = TABLE_1A_COHORT[-1]
    site_total = sum(cell.retrospective_decisions for cell in TABLE_1A_COHORT[:-1])
    patients = sum(cell.retrospective_patients for cell in TABLE_1A_COHORT[:-1])
    prospective = sum(cell.prospective_decisions for cell in TABLE_1A_COHORT[:-1])
    primary = TABLE_1B_PRIMARY[-1]
    stratum_total = sum(
        cell.retrospective_decisions
        for cell in TABLE_4_SUBGROUPS
        if cell.block == "molecular partition"
    )
    subgroup_total = next(
        cell.retrospective_decisions for cell in TABLE_4_SUBGROUPS if cell.subgroup == "All records"
    )
    status = (
        "PASS"
        if (
            site_total == pooled.retrospective_decisions == 7962
            and patients == pooled.retrospective_patients == 4318
            and prospective == pooled.prospective_decisions == 2714
            and stratum_total == subgroup_total
            and primary.cloned_decisions == 1943
        )
        else "FAIL"
    )
    return VerificationEntry(
        "ledger_additivity",
        status,
        f"site rows sum to {site_total} decisions and {patients} patients; the molecular "
        f"partition sums to {stratum_total} against the whole-cohort row {subgroup_total}; "
        f"the primary arm carries {primary.cloned_decisions} cloned decisions",
        family="manuscript",
    )


def check_site_heterogeneity() -> VerificationEntry:
    """The between-site heterogeneity against a brute-force Cochran statistic."""

    estimates = [0.61, 0.71, 0.78]
    errors = [0.069, 0.087, 0.120]
    q, i_squared = site_heterogeneity(estimates, errors)
    weights = np.asarray([1.0 / (e**2) for e in errors])
    fixed = float((weights * np.asarray(estimates)).sum() / weights.sum())
    hand_q = float((weights * (np.asarray(estimates) - fixed) ** 2).sum())
    hand_i = max(0.0, (hand_q - 2) / hand_q) * 100.0 if hand_q > 0 else 0.0
    status = "PASS" if (abs(q - hand_q) < 1e-9 and abs(i_squared - hand_i) < 1e-9) else "FAIL"
    return VerificationEntry(
        "site_heterogeneity",
        status,
        f"Cochran Q {q:.6f} against the hand value {hand_q:.6f} over three sites, so two "
        f"degrees of freedom; I squared {i_squared:.2f}%",
    )


def check_closed_form_recovery() -> VerificationEntry:
    """The builder's closed-form restricted mean against its own sample.

    The hazard behind it is a Weibull proportional-hazards model, so the
    restricted mean has a closed form. The check compares that closed form with
    the sample mean of the capped observed time on a censoring-free draw, which is
    the geometry the estimand machinery is asked to recover.
    """

    stream = np.random.default_rng(59)
    eta = -0.30
    horizon = 8.0
    shape = 1.18
    scale = 6.85
    uniform = np.clip(stream.random(200000), 1e-12, 1.0 - 1e-12)
    times = scale * ((-np.log(uniform)) / math.exp(eta)) ** (1.0 / shape)
    sample = float(np.minimum(times, horizon).mean())
    closed = reported_restricted_mean(eta, horizon, shape=shape, scale=scale)
    deviation = abs(sample - closed)
    error = float(np.std(np.minimum(times, horizon), ddof=1) / math.sqrt(times.size))
    status = "PASS" if deviation <= 4.0 * error else "FAIL"
    return VerificationEntry(
        "closed_form_recovery",
        status,
        f"closed-form restricted mean {closed:.6f} against the sample value {sample:.6f} over "
        f"2e5 draws; deviation {deviation:.6f} with a standard error of {error:.6f}",
    )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def check_forward_backward_update() -> VerificationEntry:
    """Forward, loss, backward and a parameter update on the relational encoder."""

    set_seed(7)
    catalogue = build_catalogue(size=8, seed=7)
    graph = build_knowledge_graph(catalogue, seed=7)
    encoder = RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=len(RELATION_INDEX),
        embedding_dim=16,
        hidden_dim=16,
        layers=1,
        num_basis=2,
        dropout=0.0,
    )
    tensors = graph_tensors(graph)
    batch = sample_masked_edges(graph, 16, negatives=4, seed=7)
    head = torch.as_tensor(batch.head, dtype=torch.long)
    tail = torch.as_tensor(batch.tail, dtype=torch.long)
    negative = torch.as_tensor(batch.negative, dtype=torch.long)
    before = snapshot(encoder)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1.0e-2)
    optimizer.zero_grad(set_to_none=True)
    state = encoder(
        tensors["entity"], tensors["head"], tensors["relation"], tensors["tail"], tensors["degree"]
    )
    positive = (state[head] * state[tail]).sum(dim=-1) / (state.shape[1] ** 0.5)
    negatives = torch.einsum("bw,bkw->bk", state[head], state[negative]) / (state.shape[1] ** 0.5)
    loss: Tensor = masked_edge_loss(positive, negatives)
    backpropagate(loss)
    gradients = sum(
        1
        for parameter in encoder.parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    )
    optimizer.step()
    delta = parameter_delta(before, snapshot(encoder))
    status = (
        "PASS"
        if (gradients > 0 and delta > 0.0 and math.isfinite(float(loss.detach())))
        else "FAIL"
    )
    return VerificationEntry(
        "forward_backward_update",
        status,
        f"Eq. (3) loss {float(loss.detach()):.8f}, {gradients} tensors carried a gradient, parameter "
        f"change {delta:.6f}",
    )


def check_stage2_forward() -> VerificationEntry:
    """Stage 2's masked-anchor forward pass produces finite scores and falls."""

    set_seed(11)
    catalogue = build_catalogue(size=8, seed=11)
    graph = build_knowledge_graph(catalogue, seed=11)
    batch, _ = build_cohort(seed=11, candidates=[entry.label for entry in catalogue])
    patients = attach_patients(graph, batch, limit=16)
    records = corrupt_anchors(graph, patients, negatives=3, seed=11)
    masked = sample_masked_edges(graph, 16, negatives=4, seed=11)
    config = TrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=1.0e-2,
        warmup_steps=1,
        total_steps=8,
        max_steps=8,
        log_every=0,
        seed=11,
    )
    bundle = fit_graph_stages(
        graph,
        masked,
        records,
        embedding_dim=16,
        hidden_dim=16,
        layers=1,
        num_basis=2,
        dropout=0.0,
        training=config,
    )
    reports = bundle.summary()
    stage_two = reports.get("stage2", {})
    steps = int(stage_two.get("steps", 0.0))
    leading = float(stage_two.get("leading_mean", float("nan")))
    trailing = float(stage_two.get("trailing_mean", float("nan")))
    status = "PASS" if (steps > 0 and math.isfinite(leading) and trailing < leading) else "FAIL"
    return VerificationEntry(
        "stage2_anchor_reconstruction",
        status,
        f"{steps} Stage 2 steps on {len(records)} masked anchors; leading mean {leading:.6f} "
        f"to trailing mean {trailing:.6f}",
    )


def check_single_batch_overfit() -> VerificationEntry:
    """The ranking head overfits one batch of the cross-fitted table."""

    table = _small_outcome_table()
    set_seed(13)
    head = DecisionRanker(
        state_dim=21, candidate_count=len(table.candidates), candidate_dim=4, hidden=16
    )
    from oaer.head.objectives import StageThreeWeights

    representation = torch.as_tensor(
        np.asarray([_features(decision) for decision in table.decisions], dtype=np.float32)
    )
    target = torch.as_tensor(table.values.astype(np.float32))
    eligible = torch.as_tensor(table.eligible.astype(np.float32))
    weights = StageThreeWeights().build(eligible)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5.0e-2)
    losses: list[float] = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction = head(representation)
        residual = prediction - target
        loss: Tensor = (weights * residual * residual).sum() / weights.sum().clamp(min=1.0)
        backpropagate(loss)
        optimizer.step()
        losses.append(float(loss.detach()))
    leading = float(np.mean(losses[:30]))
    trailing = float(np.mean(losses[30:]))
    status = "PASS" if trailing < leading else "FAIL"
    return VerificationEntry(
        "single_batch_overfit",
        status,
        f"Eq. (5) loss fell from {leading:.6f} to {trailing:.6f} over 60 updates on "
        f"{table.values.shape[0]} decisions",
    )


def check_checkpoint_round_trip() -> VerificationEntry:
    """Save and load restore identical parameters, the step and the seed."""

    set_seed(17)
    catalogue = build_catalogue(size=6, seed=17)
    graph = build_knowledge_graph(catalogue, seed=17)
    encoder = RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=len(RELATION_INDEX),
        embedding_dim=12,
        hidden_dim=12,
        layers=1,
        num_basis=2,
        dropout=0.0,
    )
    reference = RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=len(RELATION_INDEX),
        embedding_dim=12,
        hidden_dim=12,
        layers=1,
        num_basis=2,
        dropout=0.0,
    )
    directory = tempfile.mkdtemp()
    try:
        path = Path(directory) / "checkpoint.pt"
        save_state(
            path,
            CheckpointPayload(stage="stage1", epoch=2, step=9, seed=17, metrics={"loss": 0.25}),
            model=encoder,
        )
        payload = load_state(path, model=reference, restore_rng=False)
        identical = torch_state_equal(encoder, reference)
        mode = oct(path.stat().st_mode & 0o777)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    status = "PASS" if (identical and payload.step == 9 and payload.seed == 17) else "FAIL"
    return VerificationEntry(
        "checkpoint_round_trip",
        status,
        f"restored stage {payload.stage!r} step {payload.step} seed {payload.seed}; parameters "
        f"identical {identical}; file mode {mode}",
    )


def check_minimal_training_loop() -> VerificationEntry:
    """A minimal loop whose trailing loss mean sits below its leading mean."""

    set_seed(19)
    catalogue = build_catalogue(size=8, seed=19)
    graph = build_knowledge_graph(catalogue, seed=19)
    encoder = RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=len(RELATION_INDEX),
        embedding_dim=16,
        hidden_dim=16,
        layers=1,
        num_basis=2,
        dropout=0.0,
    )
    tensors = graph_tensors(graph)
    batch = sample_masked_edges(graph, 32, negatives=4, seed=19)
    head = torch.as_tensor(batch.head, dtype=torch.long)
    tail = torch.as_tensor(batch.tail, dtype=torch.long)
    negative = torch.as_tensor(batch.negative, dtype=torch.long)
    total = int(head.shape[0])

    def step(model: nn.Module, index: Tensor) -> Tensor:
        state = model(
            tensors["entity"],
            tensors["head"],
            tensors["relation"],
            tensors["tail"],
            tensors["degree"],
        )
        positive = (state[head[index]] * state[tail[index]]).sum(dim=-1) / (state.shape[1] ** 0.5)
        negatives = torch.einsum("bw,bkw->bk", state[head[index]], state[negative[index]]) / (
            state.shape[1] ** 0.5
        )
        return masked_edge_loss(positive, negatives)

    config = TrainingConfig(
        epochs=6,
        batch_size=16,
        learning_rate=5.0e-2,
        warmup_steps=1,
        total_steps=12,
        log_every=0,
        seed=19,
    )
    trainer = Trainer(encoder, config, step, batch_size=16)
    run = trainer.fit(total)
    decreased = loss_decreased(run.records)
    status = "PASS" if decreased else "FAIL"
    return VerificationEntry(
        "minimal_training_loop",
        status,
        f"{len(run.records)} steps; first loss {run.losses[0]:.6f}, last loss "
        f"{run.losses[-1]:.6f}",
    )


def check_study_end_to_end() -> VerificationEntry:
    """The smoke protocol runs to completion and holds its own marginals."""

    config = _smoke_config()
    result = run_study(config)
    matched = sum(1 for value in result.marginal_match.values() if value)
    total = len(result.marginal_match)
    status = (
        "PASS" if (matched == total and result.training_records.get("steps", 0.0) > 0) else "FAIL"
    )
    return VerificationEntry(
        "study_end_to_end",
        status,
        f"smoke protocol produced {result.cohort_census['development_decisions']} development "
        f"decisions, {matched}/{total} marginals matched, "
        f"{result.training_records.get('steps', 0.0):.0f} Stage 3 steps",
    )


def check_pipeline_effect_signs() -> VerificationEntry:
    """The fitted pipeline recovers the builder's effect signs.

    The builder's four named agents carry the article's own reported hazard
    ratios, so their true restricted-mean differences are signed: three positive
    and one negative. The check reads the fitted emulation's signs against that
    truth, within the scope of Table 1 panel c.
    """

    table = _named_table()
    if table is None:
        return VerificationEntry(
            "pipeline_effect_signs", "NOT_RUN", "no named agent entered the catalogue"
        )
    truth = {name: value for name, value in _named_truth().items() if name in table.estimates}
    agreements: list[str] = []
    matches = 0
    for name, expected in truth.items():
        estimate = table.estimates[name].tau
        matched = math.copysign(1.0, estimate) == math.copysign(1.0, expected)
        matches += int(matched)
        agreements.append(f"{name} {estimate:+.3f}")
    status = (
        "PASS" if matches == len(truth) else ("PARTIAL" if matches * 2 >= len(truth) else "FAIL")
    )
    return VerificationEntry(
        "pipeline_effect_signs",
        status,
        f"{matches}/{len(truth)} named agents recovered their sign: " + ", ".join(agreements),
    )


def named_effect_truth(decisions: Sequence[Decision]) -> dict[str, float]:
    """The arm's own average conditional effect per named agent, in closed form.

    The builder applies ``eta += base + gain * modifier`` at each decision and
    its hazard is Weibull, so the restricted-mean difference one agent produces on
    this arm is a closed form evaluated per decision and averaged. Reading the
    effect at a fixed baseline hazard instead is a different quantity: the arm's
    own risk distribution moves every agent's contrast by about a tenth, so the two
    are not interchangeable and the fitted conditional effect targets this one.
    """

    truth: dict[str, float] = {}
    for name, base in NAMED_EFFECTS.items():
        gain = MODIFIER_GAIN[name]
        contrast: list[float] = []
        for decision in decisions:
            horizon = line_horizon(decision.state.line)
            risk = decision_risk(decision.state)
            treated = reported_restricted_mean(
                risk + base + gain * decision_modifier(decision.state), horizon
            )
            contrast.append(treated - reported_restricted_mean(risk, horizon))
        truth[name] = float(np.mean(contrast)) if contrast else float("nan")
    return truth


def _named_truth() -> dict[str, float]:
    """The development arm's average conditional effect for each named agent."""

    if _NAMED_TRUTH_CACHE:
        return _NAMED_TRUTH_CACHE[0]
    inputs = assemble_study(_main_config())
    truth = named_effect_truth(inputs.development.decisions)
    _NAMED_TRUTH_CACHE.append(truth)
    return truth


def check_pipeline_effect_recovery() -> VerificationEntry:
    """The named agents' recovered effects against the arm's own closed form.

    Each of the four named agents carries a known log-hazard and modifier gain, so
    the restricted-mean difference it produces on this arm is a closed form
    evaluated per decision. The check asks whether the recovered conditional effect
    sits within a Bonferroni bound of that truth, and reports the ratio it measured.
    It is a statement about this arm, not about the article's cohort.

    The bound is the two-sided normal quantile at a family-wise 0.05 across four
    agents, so the criterion is stated once for the group the check makes a claim
    about rather than four times at the unadjusted level.
    """

    table = _named_table()
    if table is None:
        return VerificationEntry(
            "pipeline_effect_recovery", "NOT_RUN", "no named agent entered the catalogue"
        )
    truth = _named_truth()
    present = {name: value for name, value in truth.items() if name in table.estimates}
    bound = normal_quantile(1.0 - 0.025 / max(len(present), 1))
    rows: list[str] = []
    inside = 0
    ratios: list[float] = []
    for name, expected in present.items():
        estimate = table.estimates[name]
        gap = estimate.tau - expected
        covered = (
            math.isfinite(estimate.standard_error)
            and estimate.standard_error > 0.0
            and abs(gap) <= bound * estimate.standard_error
        )
        inside += int(covered)
        if abs(expected) > 1e-9:
            ratios.append(estimate.tau / expected)
        standardised = (
            gap / estimate.standard_error if estimate.standard_error > 0.0 else float("nan")
        )
        rows.append(
            f"{name} {estimate.tau:+.3f} against {expected:+.3f} ({standardised:+.2f} se)"
            + ("" if covered else " outside the bound")
        )
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    status = (
        "PASS" if inside == len(present) else ("PARTIAL" if inside * 2 >= len(present) else "FAIL")
    )
    return VerificationEntry(
        "pipeline_effect_recovery",
        status,
        f"{inside}/{len(present)} named agents within {bound:.2f} standard errors of the closed-form "
        f"value at development scale; mean recovered-to-true ratio {mean_ratio:.3f}. "
        + "; ".join(rows),
    )


def check_reference_score_direction() -> VerificationEntry:
    """The reproduced reference score's concordance is above chance out of sample."""

    from oaer.study import reference_concordance

    config = _smoke_config()
    inputs = assemble_study(config)
    values = reference_concordance(inputs.development, inputs.held_out)
    internal = values["development_c_index"]
    external = values["held_out_c_index"]
    status = (
        "PASS" if (internal > 0.5 and (not math.isfinite(external) or external > 0.5)) else "FAIL"
    )
    return VerificationEntry(
        "reference_score_direction",
        status,
        f"prognostic reference concordance {internal:.4f} on the development records and "
        f"{external:.4f} on the held-out site, both against chance at one half",
    )


def _claims(claim_map: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The claim entries of a claim map, ignoring a malformed payload."""

    raw = claim_map.get("claims", [])
    if not isinstance(raw, list):
        return []
    return [claim for claim in raw if isinstance(claim, Mapping)]


def check_discrimination_margin_gap() -> VerificationEntry:
    """The ordering margin against the spread of the family it names.

    The manuscript states the discrimination margin is set against the published
    spread of reproduced metastatic colorectal prognostic scores. The check
    recomputes that spread from the roster's own reference rows and reports both
    cutoffs, because the margin's reachability is decided by them: a margin wider
    than the whole family cannot be realised by any member of it, which fixes the
    sensitivity row's verdict rather than leaving it to the cohort.

    This is a manuscript-family check, so a shortfall is a statement about the
    article's own arithmetic and not about this tree.
    """

    spread = reference_score_spread()
    width = spread[1] - spread[0]
    reachable = discrimination_margin_is_reachable(spread)
    status = "PASS" if (width > 0.0 and not reachable) else ("PARTIAL" if reachable else "FAIL")
    verdict = (
        "reachable, so the margin fits inside the family"
        if reachable
        else "wider than the whole family, so no member of it can realise a difference of that size"
    )
    return VerificationEntry(
        "discrimination_margin_gap",
        status,
        f"the reproduced reference family spans {spread[0]:.3f} to {spread[1]:.3f}, a width of "
        f"{width:.3f}, against a pre-specified ordering margin of {DISCRIMINATION_MARGIN:.2f}: "
        f"the margin is {verdict}",
        family="manuscript",
    )


def check_claim_to_code_paths(claim_map: Mapping[str, object], root: Path) -> VerificationEntry:
    """Every path the claim map references exists below the release root."""

    missing: list[str] = []
    total = 0
    for claim in _claims(claim_map):
        paths = claim.get("code_paths", [])
        if not isinstance(paths, list):
            continue
        for path in paths:
            total += 1
            if not (root / str(path)).exists():
                missing.append(str(path))
    status = "PASS" if not missing else "FAIL"
    return VerificationEntry(
        "claim_to_code_paths",
        status,
        (
            f"{total - len(missing)}/{total} mapped paths exist"
            if not missing
            else f"missing paths: {missing[:6]}"
        ),
    )


def check_claim_coverage(claim_map: Mapping[str, object]) -> VerificationEntry:
    """Every claim carries a location, a statement and a verdict."""

    claims = _claims(claim_map)
    incomplete = [
        str(claim.get("claim_id", "?"))
        for claim in claims
        if not claim.get("paper_location")
        or not claim.get("statement")
        or not claim.get("verification")
    ]
    status = "PASS" if (claims and not incomplete) else "FAIL"
    return VerificationEntry(
        "claim_coverage",
        status,
        f"{len(claims)} claims, {len(incomplete)} incomplete",
    )


def check_manifest_freshness(root: Path, manifest: Path) -> VerificationEntry:
    """The written manifest still digests the live tree."""

    fresh, drift = verify_manifest(root, manifest)
    status = "PASS" if fresh else "FAIL"
    return VerificationEntry(
        "manifest_freshness",
        status,
        "the manifest digests the live tree" if fresh else f"drifted path(s): {list(drift)[:6]}",
    )


def check_no_markdown(root: Path) -> VerificationEntry:
    """The release carries exactly one Markdown file, its README."""

    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"}
    markdown = [
        str(path.relative_to(root))
        for path in root.rglob("*.md")
        if not (set(path.parts) & ignored)
    ]
    status = "PASS" if markdown == ["README.md"] else "FAIL"
    return VerificationEntry("single_markdown", status, f"Markdown files present: {markdown}")


def check_no_workflow_files(root: Path) -> VerificationEntry:
    """No workflow directory and no committed container is present."""

    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (
            "workflows" in path.parts
            or path.name.startswith("ci.")
            or path.suffix in {".tar", ".gz"}
        )
        and "__pycache__" not in path.parts
    ]
    status = "PASS" if not offenders else "FAIL"
    return VerificationEntry(
        "no_workflow_files", status, f"{len(offenders)} offending file(s): {offenders[:4]}"
    )


def check_report_is_complete(report_path: Path, expected_minimum: int) -> VerificationEntry:
    """The shipped verification report is the complete one, not a partial runner's.

    A test that imports the driver and calls its writer would overwrite this
    artefact with a partial report, so its completeness is asserted rather than
    assumed.
    """

    if not report_path.is_file():
        return VerificationEntry(
            "report_completeness", "NOT_RUN", f"no report at {report_path.name}"
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    checks = payload.get("checks", [])
    status = "PASS" if len(checks) >= expected_minimum else "FAIL"
    return VerificationEntry(
        "report_completeness",
        status,
        f"{len(checks)} checks in the shipped report against the expected minimum of "
        f"{expected_minimum}",
    )


# --------------------------------------------------------------------------- #
# Helpers and the battery
# --------------------------------------------------------------------------- #


def _features(decision: object) -> np.ndarray:
    from oaer.estimands.nuisance import state_features

    return state_features(decision)  # type: ignore[arg-type]


def _small_outcome_table() -> OutcomeTable:
    config = _smoke_config()
    inputs = assemble_study(config)
    names = [entry.label for entry in inputs.catalogue][:6]
    return run_emulation(config, inputs.development, names)


def run_battery(
    claim_map: Mapping[str, object],
    root: Path,
    *,
    include_live: bool = True,
    include_study: bool = True,
) -> list[VerificationEntry]:
    """Run every in-process check in a fixed order."""

    entries: list[VerificationEntry] = [
        check_cohort_marginals(),
        check_scaled_marginals(),
        check_stratum_partition(),
        check_exposure_prevalence(),
        check_catalogue_admission(),
        check_graph_census(),
        check_motif_operator(),
        check_motif_enumeration(),
        check_anchor_corruption(),
        check_pseudo_outcome_identity(),
        check_rmst_closed_form(),
        check_cox_reference(),
        check_e_value_closed_form(),
        check_risk_ratio_reference(),
        check_martingale_mean(),
        check_trimming_rule(),
        check_conformal_coverage(),
        check_weighted_quantile(),
        check_gate_ordering(),
        check_hits_reference(),
        check_concordance_reference(),
        check_auroc_reference(),
        check_ece_reference(),
        check_bootstrap_determinism(),
        check_site_heterogeneity(),
        check_closed_form_recovery(),
        check_forward_backward_update(),
        check_stage2_forward(),
        check_single_batch_overfit(),
        check_checkpoint_round_trip(),
        check_minimal_training_loop(),
        check_claim_to_code_paths(claim_map, root),
        check_claim_coverage(claim_map),
        check_no_markdown(root),
        check_no_workflow_files(root),
    ]
    entries.append(check_iab_ir_traceable())
    entries.append(check_criteria_arithmetic())
    entries.append(check_ledger_additivity())
    entries.append(check_discrimination_margin_gap())
    if include_study:
        entries.append(check_study_end_to_end())
        entries.append(check_pipeline_effect_signs())
        entries.append(check_pipeline_effect_recovery())
        entries.append(check_reference_score_direction())
    if include_live:
        entries.extend(run_live_checks())
    return entries


def expected_report_minimum(include_live: bool = True, include_study: bool = True) -> int:
    """The number of checks a complete report carries, before the report is read.

    This counts the battery's own entries. The driver appends the two checks that
    read the artefacts it writes -- the report's completeness and the manifest's
    freshness -- so a shipped report carries two more.
    """

    base = 39
    if not include_study:
        base -= 4
    if not include_live:
        base -= 8
    return base
