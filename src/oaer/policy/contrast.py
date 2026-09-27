"""Lemma 2: the weighted agreement identity and what an unweighted comparison is.

For a rule whose recommended set at state ``s`` is ``R(s) = T_k^pi(s)`` and the
set-level initiation probability ``p_R(s) = Pr(A ∩ R(s) != empty | S = s)``,

    V^obs(pi) = E[ 1{A ∩ R(S) != empty} Delta* Y~ / { p_R(S) G(Y~- | A, S) } ],

and the administered-policy value of a stochastic rule that picks a favoured agent
from the observed within-set choice probabilities equals that expression. The
identity is what makes the prospective arm's read-out a policy contrast rather
than a stratum comparison: an unweighted comparison of the agreement and
non-agreement strata equals the policy contrast only when ``p_R`` is constant
across the prescribing set, and it is not, and even then a hazard ratio is not the
policy contrast.

Ref: Lemma 2, Sec. 2.7 (the no-initiation weight and the primary contrast).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from oaer.support.types import Decision


@dataclass(frozen=True)
class AgreementIdentity:
    """The two sides of Lemma 2's identity, as the audit reads them."""

    weighted_value: float
    within_set_value: float
    unweighted_value: float
    constant_propensity_value: float
    subjects: int
    agreement_share: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "weighted_value": round(self.weighted_value, 6),
            "within_set_value": round(self.within_set_value, 6),
            "unweighted_value": round(self.unweighted_value, 6),
            "constant_propensity_value": round(self.constant_propensity_value, 6),
            "subjects": self.subjects,
            "agreement_share": round(self.agreement_share, 6),
        }


def _observation(decision: Decision, horizon: float) -> tuple[float, float]:
    """The observed restricted time and the event indicator, capped at a horizon."""

    restricted = min(decision.follow_up_months, horizon)
    return (restricted, 1.0 if decision.event else 0.0)


def weighted_agreement_value(
    decisions: Sequence[Decision],
    recommended: Sequence[Sequence[str]],
    set_initiation: npt.ArrayLike,
    censoring_survival: npt.ArrayLike,
) -> float:
    """The left-hand side of Lemma 2's identity over a population.

    ``set_initiation`` is ``p_R(s)`` per decision and ``censoring_survival`` is
    ``G(Y~- | A, S)`` per decision, both evaluated outside this function so the
    identity can be checked against an independent construction.
    """

    probability = np.asarray(set_initiation, dtype=np.float64).ravel()
    survival = np.asarray(censoring_survival, dtype=np.float64).ravel()
    if len(decisions) != probability.size or len(decisions) != survival.size:
        raise ValueError("one initiation probability and one survival value per decision")
    total = 0.0
    for index, decision in enumerate(decisions):
        hit = any(decision.exposure.get(name, False) for name in recommended[index])
        if not hit:
            continue
        restricted, event = _observation(decision, decision.horizon_months)
        denominator = max(probability[index] * survival[index], 1e-9)
        total += event * restricted / denominator
    return float(total / max(len(decisions), 1))


def within_set_value(
    decisions: Sequence[Decision],
    recommended: Sequence[Sequence[str]],
    horizon: float,
) -> float:
    """A stochastic in-set rule's value under the observed choice probabilities.

    The rule picks a favoured agent from the agreement set with the observed
    within-set choice probabilities, so its value is the mean observed restricted
    time over the decisions whose rule fired.
    """

    total = 0.0
    fired = 0
    for index, decision in enumerate(decisions):
        started = [name for name in recommended[index] if decision.exposure.get(name, False)]
        if not started:
            continue
        restricted, _ = _observation(decision, horizon)
        total += restricted
        fired += 1
    return float(total / fired) if fired else float("nan")


def unweighted_value(
    decisions: Sequence[Decision],
    recommended: Sequence[Sequence[str]],
    horizon: float,
) -> float:
    """The naive stratum comparison: the mean restricted time of the agreement set."""

    total = 0.0
    fired = 0
    for index, decision in enumerate(decisions):
        hit = any(decision.exposure.get(name, False) for name in recommended[index])
        if not hit:
            continue
        restricted, _ = _observation(decision, horizon)
        total += restricted
        fired += 1
    return float(total / fired) if fired else float("nan")


def constant_propensity_value(
    decisions: Sequence[Decision],
    recommended: Sequence[Sequence[str]],
    horizon: float,
) -> float:
    """The weighted value with ``p_R`` replaced by its own mean.

    This is the quantity the unweighted comparison estimates, which is why it
    agrees with the weighted value only under a constant set-level initiation
    probability.
    """

    probabilities = np.asarray(
        [
            1.0 if any(decision.exposure.get(name, False) for name in recommended[index]) else 0.0
            for index, decision in enumerate(decisions)
        ],
        dtype=np.float64,
    )
    mean = float(probabilities.mean())
    if mean <= 0.0:
        return float("nan")
    total = 0.0
    for index, decision in enumerate(decisions):
        hit = any(decision.exposure.get(name, False) for name in recommended[index])
        if not hit:
            continue
        restricted, _ = _observation(decision, horizon)
        total += restricted
    return float(total / (mean * max(len(decisions), 1)))


def agreement_identity(
    decisions: Sequence[Decision],
    recommended: Sequence[Sequence[str]],
    set_initiation: npt.ArrayLike,
    censoring_survival: npt.ArrayLike,
) -> AgreementIdentity:
    """Assemble both sides of Lemma 2 and the two comparisons that are not it."""

    horizon = min(decision.horizon_months for decision in decisions) if decisions else 0.0
    fired = sum(
        1
        for index, decision in enumerate(decisions)
        if any(decision.exposure.get(name, False) for name in recommended[index])
    )
    return AgreementIdentity(
        weighted_value=weighted_agreement_value(
            decisions, recommended, set_initiation, censoring_survival
        ),
        within_set_value=within_set_value(decisions, recommended, horizon),
        unweighted_value=unweighted_value(decisions, recommended, horizon),
        constant_propensity_value=constant_propensity_value(decisions, recommended, horizon),
        subjects=len(decisions),
        agreement_share=fired / max(len(decisions), 1),
    )


def propensity_constant(probabilities: npt.ArrayLike, tolerance: float = 1e-9) -> bool:
    """Whether a set-level initiation probability is constant across the cohort."""

    values = np.asarray(probabilities, dtype=np.float64).ravel()
    if values.size == 0:
        return True
    return bool(float(values.max() - values.min()) <= tolerance)


def collapsibility_gap(
    hazard_ratio: float,
    restricted_mean_contrast: float,
) -> float:
    """The gap between a hazard ratio and a restricted-mean contrast.

    The restricted mean is collapsible and the hazard ratio is not, so the two
    numbers answer different questions and the gap is reported rather than closed.
    """

    return float(abs(hazard_ratio - 1.0) - abs(restricted_mean_contrast))
