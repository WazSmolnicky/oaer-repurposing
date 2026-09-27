"""Clone-censor-weight design.

At time zero a decision unit is cloned into two strategies: the top-agreement
strategy, under which routine care starts one of the model's three highest-ranked
identified candidates inside the line's grace window, and the no-initiation
strategy, under which none is started. A clone whose observed exposure deviates
from its assigned strategy inside the window is censored at the deviation, and the
remaining clones carry the inverse probability of staying uncensored as a weight.
The two strategies are not a partition, so their counts do not sum to the number
of decisions.

The set-level initiation probability ``p_R(s) = Pr(A ∩ R(s) != empty | S = s)`` is
what the weight is built from; the no-initiation weight is ``1 - p_R(s)`` rather
than the reciprocal of ``p_R(s)``, which is why an unweighted comparison of the
agreement and non-agreement strata is not the policy contrast of Lemma 2.

Ref: Table 1 (clone-censor-weight design and its caption), Lemma 2, Sec. 2.7.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from oaer.support.numerics import step_integral
from oaer.support.types import Decision, LineOfTherapy

TOP_AGREEMENT: Final[str] = "top_agreement"
NO_INITIATION: Final[str] = "no_initiation"
STRATEGIES: Final[tuple[str, str]] = (TOP_AGREEMENT, NO_INITIATION)

# The grace window in days, per line, inside which an assigned start counts as
# carrying out the strategy.
GRACE_WINDOW_DAYS: Final[Mapping[str, int]] = {
    LineOfTherapy.FIRST.value: 90,
    LineOfTherapy.SECOND.value: 60,
    LineOfTherapy.THIRD_PLUS.value: 30,
}


@dataclass(frozen=True)
class Clone:
    """One decision unit under one strategy."""

    decision_id: str
    patient_id: str
    site: str
    strategy: str
    assigned: tuple[str, ...]
    started: tuple[str, ...]
    grace_days: int
    censored: bool
    censor_day: float
    weight: float
    follow_up_months: float
    event: bool

    @property
    def observed_time(self) -> float:
        return self.censor_day if self.censored else self.follow_up_months

    @property
    def observed_event(self) -> bool:
        return (not self.censored) and self.event


@dataclass(frozen=True)
class CloneCensorWeight:
    """The clone population of one strategy, with its weighted summaries."""

    strategy: str
    clones: tuple[Clone, ...]

    def __len__(self) -> int:
        return len(self.clones)

    @property
    def weighted_events(self) -> float:
        return float(sum(clone.weight for clone in self.clones if clone.observed_event))

    @property
    def censored_share(self) -> float:
        if not self.clones:
            return 0.0
        return float(np.mean([clone.censored for clone in self.clones]))

    def effective_sample_size(self) -> float:
        weights = np.asarray([clone.weight for clone in self.clones], dtype=np.float64)
        if weights.size == 0:
            return 0.0
        return float(weights.sum() ** 2 / float(np.square(weights).sum()))

    def weight_quantiles(self, quantiles: Sequence[float] = (0.5, 0.99)) -> dict[float, float]:
        weights = np.asarray([clone.weight for clone in self.clones], dtype=np.float64)
        if weights.size == 0:
            return {quantile: float("nan") for quantile in quantiles}
        return {quantile: float(np.quantile(weights, quantile)) for quantile in quantiles}

    def person_time(self, horizon_months: float) -> float:
        """Clone-weighted person-time at risk up to the horizon."""

        total = 0.0
        for clone in self.clones:
            total += clone.weight * min(clone.observed_time, horizon_months)
        return total


def grace_window(line: LineOfTherapy) -> int:
    return GRACE_WINDOW_DAYS[line.value]


def clone_decision(
    decision: Decision,
    strategy: str,
    *,
    recommended: Sequence[str],
    started: Mapping[str, float] | None = None,
) -> Clone:
    """Clone one decision into one strategy and censor it on deviation.

    ``recommended`` is the strategy's recommended set: for the top-agreement
    strategy it is the model's three highest-ranked identified candidates with a
    positive estimated effect, for the no-initiation strategy it is empty.
    ``started`` maps a candidate to the day, from time zero, at which it was
    started, and is what the deviation is read from.
    """

    if strategy not in STRATEGIES:
        raise ValueError(f"{strategy!r} is not one of the two strategies")
    window = grace_window(decision.state.line)
    observed = dict(started or {})
    inside = {name: day for name, day in observed.items() if day <= window}
    if strategy == TOP_AGREEMENT:
        hit = [name for name in recommended if name in inside]
        censored = not hit
        censor_day = 0.0 if hit else float(window)
        assigned = tuple(recommended)
        started_within = tuple(sorted(hit))
    else:
        censored = bool(inside)
        censor_day = float(min(inside.values())) if inside else 0.0
        assigned = ()
        started_within = tuple(sorted(inside))
    return Clone(
        decision_id=decision.decision_id,
        patient_id=decision.patient_id,
        site=decision.site,
        strategy=strategy,
        assigned=assigned,
        started=started_within,
        grace_days=window,
        censored=censored,
        censor_day=censor_day,
        weight=1.0,
        follow_up_months=decision.follow_up_months,
        event=decision.event,
    )


def initiation_probability(logit_value: float) -> float:
    """The set-level initiation probability from its log-odds."""

    return 1.0 / (1.0 + float(np.exp(-logit_value)))


def censor_and_weight(
    decision: Decision,
    strategy: str,
    *,
    recommended: Sequence[str],
    started: Mapping[str, float] | None,
    set_initiation_probability: float,
    censoring_probability: float = 1.0,
) -> Clone:
    """Clone a decision and attach the inverse-probability-of-censoring weight.

    The weight is ``1 / (p_R * G)`` under the agreement strategy and
    ``1 / ((1 - p_R) * G)`` under no initiation, which is the weighting Lemma 2
    states. A probability outside the open unit interval leaves the weight
    undefined and therefore marks the clone ineligible rather than inflating it.
    """

    if not 0.0 < set_initiation_probability < 1.0:
        raise ValueError("the set-level initiation probability must lie in (0, 1)")
    if not 0.0 < censoring_probability <= 1.0:
        raise ValueError("the censoring probability must lie in (0, 1]")
    base = clone_decision(decision, strategy, recommended=recommended, started=started)
    if strategy == TOP_AGREEMENT:
        weight = 1.0 / (set_initiation_probability * censoring_probability)
    else:
        weight = 1.0 / ((1.0 - set_initiation_probability) * censoring_probability)
    return Clone(
        decision_id=base.decision_id,
        patient_id=base.patient_id,
        site=base.site,
        strategy=base.strategy,
        assigned=base.assigned,
        started=base.started,
        grace_days=base.grace_days,
        censored=base.censored,
        censor_day=base.censor_day,
        weight=weight,
        follow_up_months=base.follow_up_months,
        event=base.event,
    )


def build_clone_population(
    decisions: Sequence[Decision],
    recommendations: Mapping[str, Sequence[str]],
    initiation_probabilities: Mapping[str, float],
    *,
    start_days: Mapping[str, Mapping[str, float]] | None = None,
) -> Mapping[str, CloneCensorWeight]:
    """Clone a whole batch into both strategies, keyed by strategy."""

    observed = start_days or {}
    collected: dict[str, list[Clone]] = {strategy: [] for strategy in STRATEGIES}
    for decision in decisions:
        recommended = recommendations.get(decision.decision_id, ())
        probability = float(initiation_probabilities.get(decision.decision_id, 0.5))
        for strategy in STRATEGIES:
            collected[strategy].append(
                censor_and_weight(
                    decision,
                    strategy,
                    recommended=recommended,
                    started=observed.get(decision.decision_id, {}),
                    set_initiation_probability=probability,
                )
            )
    return {
        strategy: CloneCensorWeight(strategy, tuple(clones))
        for strategy, clones in collected.items()
    }


def weighted_person_time(
    clones: Sequence[Clone],
    horizon_months: float,
) -> float:
    """The denominator of the primary analysis: clone-weighted person-time."""

    return float(sum(clone.weight * min(clone.observed_time, horizon_months) for clone in clones))


def deviations(clones: Sequence[Clone]) -> int:
    return int(sum(1 for clone in clones if clone.censored))


def weighted_kaplan_meier(
    clones: Sequence[Clone],
    grid: Sequence[float],
) -> np.ndarray:
    """The weighted Kaplan-Meier survival curve evaluated on a grid.

    The weights are the clone weights, so the curve is the one the inverse
    probability of staying uncensored defines rather than the unweighted one.
    """

    times = np.asarray([clone.observed_time for clone in clones], dtype=np.float64)
    events = np.asarray([clone.observed_event for clone in clones], dtype=bool)
    weights = np.asarray([clone.weight for clone in clones], dtype=np.float64)
    points = np.asarray(grid, dtype=np.float64)
    survival = np.ones_like(points)
    if times.size == 0:
        return survival
    event_times = np.unique(times[events])
    curve = 1.0
    processed = 0
    for position, point in enumerate(points):
        while processed < event_times.size and event_times[processed] <= point:
            at_time = float(event_times[processed])
            at_risk = float(weights[times >= at_time].sum())
            deaths = float(weights[(times == at_time) & events].sum())
            if at_risk > 0.0:
                curve *= 1.0 - deaths / at_risk
            processed += 1
        survival[position] = curve
    return survival


def weighted_restricted_mean(
    clones: Sequence[Clone],
    horizon_months: float,
) -> float:
    """The censoring-weighted restricted mean of one clone population.

    ``RMST(t*) = int_0^{t*} S_w(u) du`` with ``S_w`` the clone-weighted
    Kaplan-Meier curve, integrated exactly over the curve's own jump times, so the
    value is a function of the weights rather than of the clone count.
    """

    if not clones:
        return float("nan")
    event_times = [clone.observed_time for clone in clones if clone.observed_event]
    knots = np.unique(np.concatenate(([0.0], event_times, [horizon_months])))
    knots = knots[knots <= horizon_months]
    if knots.size < 2:
        knots = np.asarray([0.0, horizon_months], dtype=np.float64)
    survival = weighted_kaplan_meier(clones, knots.tolist())
    return step_integral(survival, knots)
