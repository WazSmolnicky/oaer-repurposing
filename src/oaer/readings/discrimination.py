"""Discrimination and calibration read-outs.

The concordance index is read off the shared cross-fitted outcome regression,
which takes the time-zero state and the backbone regimen and never the ranking
representation. That is why the concordance column of a component ablation is
expected to be invariant by construction: an appreciable movement there would
indicate a leak between the two, not that a component moved the prognosis. The
twelve-month time-dependent AUROC, the expected calibration error and the
decision-curve net benefit are read on the same records.

Ref: Sec. 1.3 (C-index, ECE, 12-month AUROC), Sec. 1.4 (the C-index column),
Table 1 panel d (net benefit, calibration slope).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from oaer.support.numerics import sigmoid


@dataclass(frozen=True)
class Concordance:
    """Harrell's concordance index with the pair counts behind it."""

    value: float
    comparable: int
    concordant: int
    tied: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "c_index": round(self.value, 6),
            "comparable_pairs": self.comparable,
            "concordant_pairs": self.concordant,
            "tied_pairs": self.tied,
        }


def concordance_index(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    scores: npt.ArrayLike,
) -> Concordance:
    """Harrell's C over all comparable pairs, with ties counted as half.

    A pair is comparable when the earlier of the two follow-ups is an event; a
    higher score on the shorter follow-up is concordant.
    """

    time = np.asarray(times, dtype=np.float64).ravel()
    event = np.asarray(events, dtype=bool).ravel()
    score = np.asarray(scores, dtype=np.float64).ravel()
    if not (time.size == event.size == score.size):
        raise ValueError("times, events and scores must share a length")
    comparable = 0
    concordant = 0.0
    tied = 0
    for first in range(time.size):
        if not event[first]:
            continue
        later = time > time[first]
        if not later.any():
            continue
        comparable += int(later.sum())
        difference = score[first] - score[later]
        concordant += float((difference > 0).sum()) + 0.5 * float((difference == 0).sum())
        tied += int((difference == 0).sum())
    value = concordant / comparable if comparable else float("nan")
    return Concordance(value=value, comparable=comparable, concordant=int(concordant), tied=tied)


def time_dependent_auroc(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    scores: npt.ArrayLike,
    horizon: float,
) -> float:
    """Cumulative/dynamic AUROC at a horizon.

    Cases are the records with an event by the horizon; controls are the records
    still at risk at the horizon. Records censored before the horizon are
    excluded from both sides because their status is unknown.
    """

    time = np.asarray(times, dtype=np.float64).ravel()
    event = np.asarray(events, dtype=bool).ravel()
    score = np.asarray(scores, dtype=np.float64).ravel()
    cases = (time <= horizon) & event
    controls = time > horizon
    if not cases.any() or not controls.any():
        return float("nan")
    case_scores = score[cases]
    control_scores = score[controls]
    comparisons = case_scores[:, None] - control_scores[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def expected_calibration_error(
    probabilities: npt.ArrayLike,
    outcomes: npt.ArrayLike,
    bins: int = 10,
) -> float:
    """The binned expected calibration error."""

    probability = np.clip(np.asarray(probabilities, dtype=np.float64).ravel(), 0.0, 1.0)
    outcome = np.asarray(outcomes, dtype=np.float64).ravel()
    if probability.size != outcome.size or probability.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for index in range(bins):
        members = assignment == index
        if not members.any():
            continue
        total += float(members.mean()) * abs(
            float(probability[members].mean() - outcome[members].mean())
        )
    return total


def calibration_slope(
    scores: npt.ArrayLike,
    outcomes: npt.ArrayLike,
    iterations: int = 40,
) -> float:
    """The slope of an outcome on the logit of the score, by Newton steps."""

    score = np.clip(np.asarray(scores, dtype=np.float64).ravel(), 1e-6, 1 - 1e-6)
    outcome = np.asarray(outcomes, dtype=np.float64).ravel()
    if score.size != outcome.size or score.size < 3:
        return float("nan")
    logit_score = np.log(score / (1.0 - score))
    design = np.column_stack([np.ones_like(logit_score), logit_score])
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(iterations):
        linear = design @ coefficients
        probability = sigmoid(linear)
        weight = probability * (1.0 - probability) + 1e-9
        gradient = design.T @ (outcome - probability)
        curvature = (design.T * weight) @ design
        step = np.linalg.solve(curvature, gradient)
        coefficients = coefficients + step
        if float(np.abs(step).max()) < 1e-10:
            break
    return float(coefficients[1])


def net_benefit(
    probabilities: npt.ArrayLike,
    outcomes: npt.ArrayLike,
    thresholds: Sequence[float],
) -> dict[float, float]:
    """Decision-curve net benefit against the treat-all and treat-none baselines."""

    probability = np.asarray(probabilities, dtype=np.float64).ravel()
    outcome = np.asarray(outcomes, dtype=np.float64).ravel()
    if probability.size != outcome.size or probability.size == 0:
        return {threshold: float("nan") for threshold in thresholds}
    prevalence = float(outcome.mean())
    values: dict[float, float] = {}
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError("a decision threshold must lie strictly inside the unit interval")
        selected = probability >= threshold
        if not selected.any():
            values[float(threshold)] = 0.0
            continue
        true_positive = float((selected & (outcome > 0.5)).mean())
        false_positive = float((selected & (outcome <= 0.5)).mean())
        values[float(threshold)] = true_positive - false_positive * threshold / (1.0 - threshold)
    del prevalence
    return values


def net_benefit_advantage(
    probabilities: npt.ArrayLike,
    outcomes: npt.ArrayLike,
    thresholds: Sequence[float],
) -> dict[float, float]:
    """Net benefit above the better of treat-all and treat-none at each threshold."""

    outcome = np.asarray(outcomes, dtype=np.float64).ravel()
    prevalence = float(outcome.mean()) if outcome.size else 0.0
    values = net_benefit(probabilities, outcomes, thresholds)
    return {
        threshold: value - max(0.0, prevalence - (1 - prevalence) * threshold / (1 - threshold))
        for threshold, value in values.items()
    }


def net_reclassification_improvement(
    reference: npt.ArrayLike,
    candidate: npt.ArrayLike,
    outcomes: npt.ArrayLike,
    threshold: float = 0.5,
) -> float:
    """The categorical net reclassification improvement at one threshold."""

    base = np.asarray(reference, dtype=np.float64).ravel() >= threshold
    new = np.asarray(candidate, dtype=np.float64).ravel() >= threshold
    outcome = np.asarray(outcomes, dtype=np.float64).ravel() > 0.5
    if base.size != new.size or base.size != outcome.size:
        return float("nan")
    events = outcome
    non_events = ~outcome
    up_events = float((new & ~base & events).sum())
    down_events = float((~new & base & events).sum())
    up_non_events = float((new & ~base & non_events).sum())
    down_non_events = float((~new & base & non_events).sum())
    event_total = max(int(events.sum()), 1)
    non_event_total = max(int(non_events.sum()), 1)
    return (up_events - down_events) / event_total - (
        up_non_events - down_non_events
    ) / non_event_total


def brier_score(probabilities: npt.ArrayLike, outcomes: npt.ArrayLike) -> float:
    probability = np.asarray(probabilities, dtype=np.float64).ravel()
    outcome = np.asarray(outcomes, dtype=np.float64).ravel()
    if probability.size != outcome.size or probability.size == 0:
        return float("nan")
    return float(np.mean((probability - outcome) ** 2))


def auprc(scores: npt.ArrayLike, outcomes: npt.ArrayLike) -> float:
    """The area under the precision-recall curve by the trapezoid rule."""

    score = np.asarray(scores, dtype=np.float64).ravel()
    outcome = np.asarray(outcomes, dtype=np.float64).ravel() > 0.5
    positives = int(outcome.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ordered = outcome[order]
    cumulative = np.cumsum(ordered)
    positions = np.arange(1, ordered.size + 1)
    precision = cumulative / positions
    recall = cumulative / positives
    return float(np.trapezoid(np.concatenate([[1.0], precision]), np.concatenate([[0.0], recall])))
