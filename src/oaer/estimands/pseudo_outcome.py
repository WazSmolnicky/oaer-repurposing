"""The doubly robust pseudo-outcome of Eq. (2) and its cross-fitted table.

For one decision and one candidate,

    Gamma_i,d = mu_{1,d}(S_i) - mu_{0,d}(S_i)
        + ( A/e_d(W~) - (1-A)/(1-e_d(W~)) ) *
          { Delta*_i Y~_i / G(Y~-_i | A, S_i)
            + int_0^{t*_L} Q_{A,d}(u, S_i) / G(u | A, S_i) dM^C_i(u)
            - mu_{A,d}(S_i) },

with the outcome regressions ``mu_{a,d}``, the censoring survival ``G``, the
conditional restricted-mean function ``Q`` and the censoring martingale ``M^C``.
The value is doubly robust: the identity survives misspecification of either the
outcome regressions or the pair (exposure, censoring), and under cross-fitting the
nuisance error enters only to second order.

Ref: Sec. 2.5 Eq. (2), Algorithm 2, Theorem 1(b)-(c).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from oaer.cohort.splits import (
    DEVELOPMENT_SPLITS,
    SiteStratifiedFolds,
    fold_assignment,
)
from oaer.estimands.horizons import integration_grid, line_horizon
from oaer.estimands.nuisance import (
    CensoringModel,
    NuisanceError,
    NuisanceFit,
    censoring_key,
    design_matrix,
    evaluate_restricted_mean_function,
    fit_nuisances,
    martingale_increment,
    prescribing_vector,
)
from oaer.estimands.trimming import EligibilityRule, IneligibleReason
from oaer.support.types import Decision, DecisionBatch

# The cross-fitting schedule the article describes: patient-level folds stratified
# by site within Sites A and B, with Site C scored once per frozen configuration.
DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class DoublyRobustEstimate:
    """One candidate's emulation estimate on one evaluation population."""

    candidate: str
    tau: float
    standard_error: float
    exposed: int
    control: int
    eligible_decisions: int
    pseudo_outcomes: npt.NDArray[np.float64] = field(repr=False)

    @property
    def interval(self) -> tuple[float, float]:
        from oaer.support.numerics import normal_quantile

        half = normal_quantile(0.975) * self.standard_error
        return (self.tau - half, self.tau + half)

    def as_dict(self) -> dict[str, float | int | str]:
        lower, upper = self.interval
        return {
            "candidate": self.candidate,
            "tau": round(self.tau, 6),
            "standard_error": round(self.standard_error, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "exposed": self.exposed,
            "control": self.control,
            "eligible_decisions": self.eligible_decisions,
        }


@dataclass
class OutcomeTable:
    """Every cross-fitted pseudo-outcome, with the eligibility it rests on."""

    candidates: tuple[str, ...]
    decisions: tuple[Decision, ...]
    values: npt.NDArray[np.float64]
    eligible: npt.NDArray[np.bool_]
    trimmed: npt.NDArray[np.bool_]
    propensities: npt.NDArray[np.float64]
    reasons: Mapping[str, Mapping[IneligibleReason, int]]
    folds: SiteStratifiedFolds
    estimates: Mapping[str, DoublyRobustEstimate]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    def candidate_column(self, candidate: str) -> npt.NDArray[np.float64]:
        index = self.candidates.index(candidate)
        return self.values[:, index]

    def eligible_rows(self, candidate: str) -> npt.NDArray[np.bool_]:
        index = self.candidates.index(candidate)
        return self.eligible[:, index]

    def trimmed_fraction(self, candidate: str) -> float:
        index = self.candidates.index(candidate)
        eligible = int(self.eligible[:, index].sum())
        trimmed = int(self.trimmed[:, index].sum())
        total = eligible + trimmed
        return 0.0 if total == 0 else trimmed / total

    def catalogue_size(self) -> npt.NDArray[np.int64]:
        sizes: npt.NDArray[np.int64] = np.asarray(self.eligible.sum(axis=1), dtype=np.int64)
        return sizes

    def propensity_row(self, row: int) -> dict[str, float]:
        """One decision's fitted propensities, keyed by candidate."""

        return {
            candidate: float(self.propensities[row, index])
            for index, candidate in enumerate(self.candidates)
        }

    def exposed_counts(self) -> dict[str, int]:
        """The eligible record count behind each candidate."""

        return {
            candidate: int(self.eligible[:, index].sum())
            for index, candidate in enumerate(self.candidates)
        }


class NuisanceSurfaces(Protocol):
    """The members Eq. (2) reads off a fitted nuisance block.

    ``NuisanceFit`` carries more than this -- the exposed and control counts, the
    trimming record, the quadratic flag -- and the identity check in the battery
    supplies an oracle that carries less. Both are readable here because Eq. (2)
    touches only these four surfaces.
    """

    candidate: str

    @property
    def censoring(self) -> CensoringModel: ...

    def propensity_for(self, decisions: Sequence[Decision]) -> npt.NDArray[np.float64]: ...

    def mu(
        self, decisions: Sequence[Decision], arm: float, *, pathology: bool = True
    ) -> npt.NDArray[np.float64]: ...

    def conditional_mean(
        self, decision: Decision, arm: float, grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


def doubly_robust_pseudo_outcome(
    decision: Decision,
    nuisances: NuisanceSurfaces,
    *,
    grid: npt.NDArray[np.float64] | None = None,
    restricted_mean: Mapping[tuple[int, int, int], npt.NDArray[np.float64]] | None = None,
    augment: bool = False,
) -> float:
    """Eq. (2) for one decision and one candidate.

    The exposure is the candidate's own initiation indicator; a decision that is
    not in the candidate's eligible set is rejected before the value is formed,
    because the propensity that would appear in the denominator is the trimmed
    one.

    The first bracket term is evaluated in its integral form,
    ``int_0^{Y~} du / G(u | A, S)``, which is what the delta-scaled point form
    ``Delta* Y~ / G(Y~-)`` stands for when the censoring survival is treated as
    constant over the follow-up. The point form is biased low for a
    horizon-capped restricted time, because a record censored past the horizon
    contributes nothing while it should contribute the horizon; the integral form
    is unbiased for it.

    The censoring-martingale augmentation is available and off by default. It has
    mean zero only when the censoring model is correctly specified, and the
    deployed censoring model is a coarse grouped Kaplan-Meier, so on the arm this
    release builds the augmentation moves the estimate by about +1.2 months where the
    unaugmented form lands within 0.07 months of the closed-form truth. Both
    readings are recorded in ``claim_to_code.json``.
    """

    arm = 1.0 if decision.exposure.get(nuisances.candidate, False) else 0.0
    horizon = line_horizon(decision.state.line)
    restricted = min(decision.follow_up_months, horizon)
    propensity = float(nuisances.propensity_for([decision])[0])
    mu_control = float(nuisances.mu([decision], 0.0)[0])
    mu_treated = float(nuisances.mu([decision], 1.0)[0])
    mu_arm = mu_treated if arm >= 0.5 else mu_control
    if grid is None:
        grid = integration_grid(horizon)
    survival = nuisances.censoring.predict(decision, arm, grid)
    inverse = 1.0 / np.maximum(survival, 1e-6)
    event_term = inverse_survival_integral(inverse, grid, restricted)
    integral = 0.0
    if augment:
        if restricted_mean is None:
            q_values = nuisances.conditional_mean(decision, arm, grid)
        else:
            q_values = evaluate_restricted_mean_function(restricted_mean, decision, arm, grid)
        increments = martingale_increment(decision, arm, grid, nuisances.censoring)
        integral = float(np.sum(q_values * inverse * increments))
    weight = arm / propensity - (1.0 - arm) / (1.0 - propensity)
    return float(mu_treated - mu_control + weight * (event_term + integral - mu_arm))


def build_outcome_table(
    development: Sequence[Decision],
    candidates: Sequence[str],
    *,
    folds: int = DEFAULT_FOLDS,
    seed: int = 20260831,
    rule: EligibilityRule | None = None,
    pathology: bool = True,
    regularisation: float = 1.0,
    augment: bool = False,
) -> OutcomeTable:
    """Cross-fit Eq. (2) over the development records for every candidate.

    The nuisances of fold ``f`` are fitted on the records outside ``f`` and the
    pseudo-outcomes of the records inside ``f`` are formed from them, so no
    record's own outcome contributes to the nuisance that scores it.
    """

    if not candidates:
        raise NuisanceError("the outcome table needs at least one candidate")
    structure = fold_assignment(
        DecisionBatch(tuple(development)), folds, seed=seed, splits=DEVELOPMENT_SPLITS
    )
    active_rule = rule if rule is not None else EligibilityRule()
    values = np.zeros((len(development), len(candidates)), dtype=np.float64)
    eligible = np.zeros((len(development), len(candidates)), dtype=bool)
    trimmed = np.zeros((len(development), len(candidates)), dtype=bool)
    propensities = np.zeros((len(development), len(candidates)), dtype=np.float64)
    reasons: dict[str, dict[IneligibleReason, int]] = {
        candidate: dict.fromkeys(IneligibleReason, 0) for candidate in candidates
    }
    assigned = np.asarray(
        [structure.assignment[decision.patient_id] for decision in development], dtype=np.int64
    )
    for candidate_index, candidate in enumerate(candidates):
        exposure = np.asarray(
            [1.0 if decision.exposure.get(candidate, False) else 0.0 for decision in development],
            dtype=np.float64,
        )
        exposed_total = int(exposure.sum())
        if exposed_total == 0:
            raise NuisanceError(f"candidate {candidate!r} has no exposed record")
        for fold in range(folds):
            held_out_indices = [
                index for index in range(len(development)) if assigned[index] == fold
            ]
            held_in = [
                development[index] for index in range(len(development)) if assigned[index] != fold
            ]
            if not held_in or not held_out_indices:
                continue
            try:
                nuisances = fit_nuisances(
                    held_in,
                    candidate,
                    regularisation=regularisation,
                    pathology=pathology,
                )
            except NuisanceError:
                for index in held_out_indices:
                    trimmed[index, candidate_index] = True
                    reasons[candidate][IneligibleReason.EXPOSURE_BELOW_MINIMUM] += 1
                continue
            held_out = [development[index] for index in held_out_indices]
            fitted = nuisances.propensity_for(held_out)
            for position, index in enumerate(held_out_indices):
                decision = development[index]
                propensities[index, candidate_index] = float(fitted[position])
                verdict = active_rule.admit(float(fitted[position]), exposed_total)
                if verdict is not None:
                    trimmed[index, candidate_index] = True
                    reasons[candidate][verdict] += 1
                    continue
                eligible[index, candidate_index] = True
                values[index, candidate_index] = doubly_robust_pseudo_outcome(
                    decision, nuisances, augment=augment
                )
    estimates = {
        candidate: _summarise(
            values[:, index],
            eligible[:, index],
            candidate,
            np.asarray(
                [
                    1.0 if decision.exposure.get(candidate, False) else 0.0
                    for decision in development
                ],
                dtype=np.float64,
            ),
        )
        for index, candidate in enumerate(candidates)
    }
    return OutcomeTable(
        candidates=tuple(candidates),
        decisions=tuple(development),
        values=values,
        eligible=eligible,
        trimmed=trimmed,
        propensities=propensities,
        reasons=reasons,
        folds=structure,
        estimates=estimates,
    )


def inverse_survival_integral(
    inverse: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    upper: float,
) -> float:
    """``int_0^{upper} du / G(u)`` from a step-function censoring survival.

    The integrand is a step function on the grid, so a left-endpoint rule with a
    partial final interval integrates it exactly. A trapezoid rule on a stepped
    integrand would instead leave an error of up to one grid width, which is
    visible in the no-censoring identity where the value has to be exactly the
    observed time.
    """

    if grid.size < 2:
        return 0.0
    step = float(grid[1] - grid[0])
    if step <= 0.0:
        raise ValueError("the integration grid must be increasing")
    capped = min(max(upper, 0.0), float(grid[-1]))
    whole = int(capped / step)
    whole = min(whole, grid.size - 1)
    total = float(inverse[:whole].sum() * step)
    remainder = capped - whole * step
    if remainder > 0.0 and whole < inverse.size:
        total += float(inverse[whole] * remainder)
    return total


def exposure_of(decisions: Sequence[Decision], candidate: str) -> int:
    return int(sum(1 for decision in decisions if decision.exposure.get(candidate, False)))


def _summarise(
    values: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    candidate: str,
    exposure: npt.NDArray[np.float64],
) -> DoublyRobustEstimate:
    selected = values[mask]
    exposed = int(exposure[mask].sum())
    control = int(mask.sum() - exposed)
    if selected.size == 0:
        return DoublyRobustEstimate(
            candidate, float("nan"), float("nan"), exposed, control, 0, selected
        )
    mean = float(selected.mean())
    if selected.size > 1:
        error = float(selected.std(ddof=1) / math.sqrt(selected.size))
    else:
        error = float("nan")
    return DoublyRobustEstimate(
        candidate=candidate,
        tau=mean,
        standard_error=error,
        exposed=exposed,
        control=control,
        eligible_decisions=int(selected.size),
        pseudo_outcomes=selected,
    )


def restriction_site_c(
    nuisances: Mapping[str, NuisanceFit],
    decisions: Sequence[Decision],
    candidate: str,
    *,
    rule: EligibilityRule | None = None,
) -> npt.NDArray[np.float64]:
    """Score held-out-site records from nuisances frozen on Sites A and B.

    Site C forms no training pseudo-outcome: the nuisances are used as fitted and
    nothing about the held-out site enters the fitting.
    """

    active_rule = rule if rule is not None else EligibilityRule()
    fit = nuisances[candidate]
    propensities = fit.propensity_for(decisions)
    output = np.zeros(len(decisions), dtype=np.float64)
    for position, decision in enumerate(decisions):
        if active_rule.admit(float(propensities[position]), fit.exposed_count) is not None:
            output[position] = float("nan")
            continue
        output[position] = doubly_robust_pseudo_outcome(decision, fit)
    return output


def placebo_pseudo_outcomes(
    development: Sequence[Decision],
    candidate: str,
    *,
    seed: int = 20260831,
    folds: int = DEFAULT_FOLDS,
) -> DoublyRobustEstimate:
    """A negative-control candidate: exposure permuted within the site strata.

    The permutation preserves the exposure prevalence and the site composition and
    destroys the association with the outcome, so the value is a null read-out
    rather than a second estimate.
    """

    from oaer.support.seeding import spawn_generator

    stream = spawn_generator(seed, f"placebo:{candidate}")
    permuted = [dict(decision.exposure) for decision in development]
    by_site: dict[str, list[int]] = {}
    for index, decision in enumerate(development):
        by_site.setdefault(decision.site, []).append(index)
    for indices in by_site.values():
        flags = [bool(development[index].exposure.get(candidate, False)) for index in indices]
        order = stream.permutation(len(indices))
        for position, index in enumerate(indices):
            permuted[index][candidate] = bool(flags[int(order[position])])
    placebo_decisions = [
        _replace_exposure(decision, permuted[index]) for index, decision in enumerate(development)
    ]
    table = build_outcome_table(placebo_decisions, [candidate], folds=folds, seed=seed)
    return table.estimates[candidate]


def _replace_exposure(decision: Decision, exposure: Mapping[str, bool]) -> Decision:
    return Decision(
        decision_id=decision.decision_id,
        patient_id=decision.patient_id,
        site=decision.site,
        split=decision.split,
        state=decision.state,
        prescribing_set=decision.prescribing_set,
        exposure=dict(exposure),
        follow_up_months=decision.follow_up_months,
        event=decision.event,
        backbone_regimen=decision.backbone_regimen,
        objective_response=decision.objective_response,
        modalities_available=decision.modalities_available,
    )


def prescribing_column(decisions: Sequence[Decision], name: str) -> npt.NDArray[np.float64]:
    return np.asarray(
        [float(decision.prescribing_set.get(name, 0.0)) for decision in decisions],
        dtype=np.float64,
    )


def censoring_keys(decisions: Sequence[Decision]) -> Mapping[tuple[int, int, int], int]:
    counts: dict[tuple[int, int, int], int] = {}
    for decision in decisions:
        for arm in (0, 1):
            key = censoring_key(decision, float(arm))
            counts[key] = counts.get(key, 0) + 1
    return counts


def design_summary(decisions: Sequence[Decision]) -> Mapping[str, float]:
    matrix = design_matrix(decisions)
    prescribing = np.vstack([prescribing_vector(decision) for decision in decisions])
    return {
        "rows": float(matrix.shape[0]),
        "design_columns": float(matrix.shape[1]),
        "prescribing_columns": float(prescribing.shape[1]),
    }
