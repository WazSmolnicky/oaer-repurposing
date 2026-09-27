"""Nuisance estimation.

Four nuisances per candidate: the propensity ``e_d(W~)`` on the low-dimensional
prescribing set, the two arm-specific outcome regressions ``mu_{0,d}`` and
``mu_{1,d}`` for the restricted time at ``t*_L``, the censoring survival
``G(u | a, s)``, and the conditional restricted-mean function ``Q_{a,d}(u, s)``
that together with the censoring martingale completes Eq. (2). The propensity is
a regularised logistic regression and is deliberately fitted on the prescribing
set rather than on the multimodal embedding, which is what Corollary 4 asks for.
All four are fitted on the records outside the fold.

Ref: Sec. 2.5, Algorithm 2 lines 4-5 and line 12, Corollary 4.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from oaer.cohort.pathology import slide_signature
from oaer.estimands.horizons import line_horizon
from oaer.estimands.survival_regression import CoxOutcomeModel
from oaer.support.types import Decision

PRESCRIBING_SET_ORDER: Final[tuple[str, ...]] = (
    "age_years",
    "female",
    "ecog_two_or_above",
    "right_sided",
    "line_index",
    "organ_sites",
    "albumin_scaled",
    "ldh_scaled",
    "cea_log",
    "comorbidities",
    "pi3k_altered",
    "msi_high",
    "ras_mutant",
    "tumour_purity",
    "morphology_burden",
    "noise",
)

CENSORING_STRATA: Final[tuple[str, ...]] = ("ecog", "line_bucket")


class NuisanceError(ValueError):
    """Raised when a nuisance cannot be fitted on the records it is given."""


def prescribing_vector(decision: Decision) -> npt.NDArray[np.float64]:
    """The prescribing-set covariate vector in a fixed column order."""

    return np.asarray(
        [float(decision.prescribing_set.get(key, 0.0)) for key in PRESCRIBING_SET_ORDER],
        dtype=np.float64,
    )


def prescribing_matrix(decisions: Sequence[Decision]) -> npt.NDArray[np.float64]:
    """The propensity design: the prescribing set plus an intercept column.

    The intercept is what lets the fitted propensity reach the exposure
    prevalence; without it the model is anchored at one half and the inverse
    weights are drawn from the wrong scale entirely.
    """

    rows = np.vstack([prescribing_vector(decision) for decision in decisions])
    intercept = np.ones((rows.shape[0], 1), dtype=np.float64)
    return np.hstack([intercept, rows])


def state_features(decision: Decision, *, pathology: bool = True) -> npt.NDArray[np.float64]:
    """The state vector the outcome regression and the concordance index read.

    It carries the time-zero state and the backbone regimen and never the ranking
    representation, so a component removal cannot move a read-out that is taken
    off this regression.
    """

    clinical = decision.state.clinical
    molecular = decision.state.molecular
    base = list(clinical.reference_vector)
    base.extend(
        [
            1.0 if molecular.ras_mutant else 0.0,
            1.0 if molecular.braf_v600e else 0.0,
            1.0 if molecular.msi_high else 0.0,
            1.0 if molecular.pi3k_altered else 0.0,
            1.0 if molecular.erbb2_amplified else 0.0,
            1.0 if molecular.kras_g12c else 0.0,
            1.0 if molecular.fully_typed else 0.0,
            {"narrow": 0.0, "intermediate": 1.0, "broad": 2.0}[molecular.assay_panel],
            1.0 if molecular.microsatellite_assay == "ihc" else 0.0,
        ]
    )
    if pathology:
        base.append(
            slide_signature(decision.state.pathology.embedding)
            if decision.state.pathology.embedding
            else 0.0
        )
        base.append(decision.state.pathology.tumour_purity)
    else:
        base.extend([0.0, 0.0])
    return np.asarray(base, dtype=np.float64)


def design_matrix(
    decisions: Sequence[Decision],
    *,
    pathology: bool = True,
    quadratic: bool = False,
) -> npt.NDArray[np.float64]:
    """State features with an appended intercept column.

    ``quadratic`` appends the squares and pairwise products of the continuous
    covariates. The restricted mean is a nonlinear function of the linear
    predictor, so a purely linear outcome regression is misspecified by
    construction; the augmentation is what keeps the regression close enough for
    the doubly robust identity to bite.
    """

    rows = [state_features(decision, pathology=pathology) for decision in decisions]
    matrix = np.asarray(rows, dtype=np.float64)
    intercept = np.ones((matrix.shape[0], 1), dtype=np.float64)
    stacked = np.hstack([intercept, matrix])
    if not quadratic:
        return stacked
    continuous = matrix[:, CONTINUOUS_COLUMNS]
    squares = continuous**2
    products = np.einsum("ni,nj->nij", continuous, continuous)
    upper = products[:, *np.triu_indices(continuous.shape[1], k=1)]
    return np.hstack([stacked, squares, upper])


# The columns of the state vector that are on a continuous scale, so that squaring
# and interacting them is meaningful.
CONTINUOUS_COLUMNS: Final[tuple[int, ...]] = (3, 4, 5, 6, 7, 8, 9, 17, 19, 20)


def standardise(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Column means and scales, with the intercept column left alone."""

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    scales[0] = 1.0
    means[0] = 0.0
    return (matrix - means) / scales, means, scales


@dataclass
class LogisticPropensity:
    """L2-regularised logistic regression, fitted by Newton steps."""

    weights: npt.NDArray[np.float64] | None = None
    means: npt.NDArray[np.float64] | None = None
    scales: npt.NDArray[np.float64] | None = None
    regularisation: float = 1.0
    iterations: int = 40
    tolerance: float = 1e-8
    floor: float = 0.01
    ceiling: float = 0.99

    def fit(
        self, features: npt.NDArray[np.float64], outcome: npt.NDArray[np.float64]
    ) -> LogisticPropensity:
        if features.shape[0] != outcome.shape[0]:
            raise NuisanceError("the propensity design and the exposure vector disagree")
        if outcome.size == 0 or outcome.all() or not outcome.any():
            raise NuisanceError("the propensity needs both exposed and unexposed records")
        scaled, means, scales = standardise(features)
        weights = np.zeros(scaled.shape[1], dtype=np.float64)
        penalty = self.regularisation * np.eye(scaled.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        for _ in range(self.iterations):
            linear = scaled @ weights
            probability = 1.0 / (1.0 + np.exp(-linear))
            gradient = scaled.T @ (outcome - probability) - penalty @ weights
            curvature = -(scaled.T * (probability * (1.0 - probability))) @ scaled - penalty
            step = np.linalg.solve(curvature, -gradient)
            weights = weights + step
            if float(np.abs(step).max()) < self.tolerance:
                break
        self.weights = weights
        self.means = means
        self.scales = scales
        return self

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.weights is None or self.means is None or self.scales is None:
            raise NuisanceError("the propensity has not been fitted")
        scaled = (features - self.means) / self.scales
        probability = 1.0 / (1.0 + np.exp(-(scaled @ self.weights)))
        return np.clip(probability, self.floor, self.ceiling)


@dataclass
class RidgeOutcomeModel:
    """Ridge regression for the arm-specific restricted time."""

    coefficients: npt.NDArray[np.float64] | None = None
    means: npt.NDArray[np.float64] | None = None
    scales: npt.NDArray[np.float64] | None = None
    regularisation: float = 1.0

    def fit(
        self, features: npt.NDArray[np.float64], outcome: npt.NDArray[np.float64]
    ) -> RidgeOutcomeModel:
        if features.shape[0] != outcome.shape[0] or outcome.size == 0:
            raise NuisanceError("the outcome regression needs paired rows")
        scaled, means, scales = standardise(features)
        penalty = self.regularisation * np.eye(scaled.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        gram = scaled.T @ scaled + penalty
        coefficients = np.asarray(np.linalg.solve(gram, scaled.T @ outcome), dtype=np.float64)
        self.coefficients = coefficients
        self.means = means
        self.scales = scales
        return self

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.coefficients is None or self.means is None or self.scales is None:
            raise NuisanceError("the outcome regression has not been fitted")
        scaled = (features - self.means) / self.scales
        return scaled @ self.coefficients


@dataclass
class CensoringSurvival:
    """A grouped Kaplan-Meier estimate of ``G(u | a, s)``.

    The grouping is the exposure arm crossed with the performance status and the
    prior-line bucket, which is the covariate block the censoring model is allowed
    to read. A group with no event contributes the flat curve at one.
    """

    curves: Mapping[tuple[int, int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    floor: float = 0.05

    @classmethod
    def fit(
        cls,
        decisions: Sequence[Decision],
        exposure: npt.NDArray[np.float64],
        *,
        floor: float = 0.05,
    ) -> CensoringSurvival:
        if len(decisions) != exposure.size:
            raise NuisanceError("the censoring fit needs one exposure per decision")
        groups: dict[tuple[int, int, int], list[tuple[float, bool]]] = {}
        for decision, arm in zip(decisions, exposure, strict=True):
            key = censoring_key(decision, arm)
            horizon = line_horizon(decision.state.line)
            time = min(decision.follow_up_months, horizon)
            groups.setdefault(key, []).append((time, bool(decision.event)))
        curves: dict[
            tuple[int, int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = {}
        for key, records in groups.items():
            times = np.asarray([record[0] for record in records], dtype=np.float64)
            events = np.asarray([record[1] for record in records], dtype=bool)
            curves[key] = _km_curve(times, events, floor=floor)
        return cls(curves=curves, floor=floor)

    def predict(
        self,
        decision: Decision,
        arm: float,
        time: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """``G(u | a, s)`` on a grid, floored away from zero."""

        key = censoring_key(decision, arm)
        curve = self.curves.get(key)
        if curve is None:
            return np.full(np.asarray(time).shape, 1.0, dtype=np.float64)
        knots, values = curve
        queried = np.asarray(time, dtype=np.float64)
        evaluated = np.ones_like(queried)
        position = np.searchsorted(knots, queried, side="right") - 1
        positive = position >= 0
        evaluated[positive] = values[position[positive]]
        return np.clip(evaluated, self.floor, 1.0)


def censoring_key(decision: Decision, arm: float) -> tuple[int, int, int]:
    prior_bucket = min(2, decision.state.clinical.prior_lines)
    return (int(arm), int(decision.state.clinical.ecog.value), prior_bucket)


def _km_curve(
    times: npt.NDArray[np.float64],
    events: npt.NDArray[np.float64] | npt.NDArray[np.bool_],
    *,
    floor: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A right-continuous Kaplan-Meier curve of the censoring distribution."""

    unique = np.unique(times)
    knots = [0.0]
    values = [1.0]
    survival = 1.0
    indicator = np.asarray(events, dtype=bool)
    for point in unique:
        at_risk = int((times >= point).sum())
        removed = int(((times == point) & ~indicator).sum())
        if at_risk > 0 and removed > 0:
            survival *= 1.0 - removed / at_risk
        knots.append(float(point))
        values.append(max(survival, floor))
    return np.asarray(knots, dtype=np.float64), np.asarray(values, dtype=np.float64)


@dataclass
class OutcomeRegression:
    """The two arm-specific outcome regressions of one candidate.

    Each arm carries a Cox model, and the restricted mean is read off its fitted
    baseline cumulative hazard at the line-specific horizon. A linear regression
    of the capped time would be misspecified, because the cap puts a large share
    of the records at the horizon and the response is nonlinear in the linear
    predictor there.
    """

    control: CoxOutcomeModel
    treated: CoxOutcomeModel

    def predict(
        self,
        features: npt.NDArray[np.float64],
        arm: float,
        horizons: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        model = self.treated if arm >= 0.5 else self.control
        if horizons is None:
            raise NuisanceError("the restricted mean needs one horizon per row")
        grid_points = 41
        output = np.zeros(features.shape[0], dtype=np.float64)
        for horizon in np.unique(horizons):
            rows = horizons == horizon
            grid = np.linspace(0.0, float(horizon), grid_points)
            curve = model.survival(features[rows], grid)
            output[rows] = np.trapezoid(curve, grid, axis=1)
        return output


@dataclass
class NuisanceFit:
    """Everything one candidate's emulation needs before Eq. (2) can be evaluated."""

    candidate: str
    propensity: LogisticPropensity
    outcome: OutcomeRegression
    censoring: CensoringSurvival
    exposed_count: int
    control_count: int
    trimmed: float
    restricted_mean: Mapping[tuple[int, int, int], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    quadratic: bool = True

    def propensity_for(self, decisions: Sequence[Decision]) -> npt.NDArray[np.float64]:
        return self.propensity.predict(prescribing_matrix(decisions))

    def mu(
        self, decisions: Sequence[Decision], arm: float, *, pathology: bool = True
    ) -> npt.NDArray[np.float64]:
        matrix = design_matrix(decisions, pathology=pathology, quadratic=False)
        horizons = np.asarray([decision.horizon_months for decision in decisions], dtype=np.float64)
        return self.outcome.predict(matrix, arm, horizons)

    def conditional_mean(
        self, decision: Decision, arm: float, grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """``Q_{a,d}(u, s)`` read from the curve fitted on the training records.

        The curve is built from records the decision does not belong to, which is
        what makes it predictable for the martingale integral; an at-risk average
        that included the decision's own outcome would not be.
        """

        return evaluate_restricted_mean_function(self.restricted_mean, decision, arm, grid)


def fit_nuisances(
    decisions: Sequence[Decision],
    candidate: str,
    *,
    regularisation: float = 1.0,
    pathology: bool = True,
    quadratic: bool = True,
    censoring_floor: float = 0.05,
) -> NuisanceFit:
    """Fit the four nuisances of one candidate on the records supplied."""

    if not decisions:
        raise NuisanceError("no record was supplied to the nuisance fit")
    exposure = np.asarray(
        [1.0 if decision.exposure.get(candidate, False) else 0.0 for decision in decisions],
        dtype=np.float64,
    )
    restricted = np.asarray(
        [min(decision.follow_up_months, decision.horizon_months) for decision in decisions],
        dtype=np.float64,
    )
    design = design_matrix(decisions, pathology=pathology, quadratic=False)
    prescribing = prescribing_matrix(decisions)
    propensity = LogisticPropensity(regularisation=regularisation).fit(prescribing, exposure)
    treated_mask = exposure >= 0.5
    control_mask = ~treated_mask
    if treated_mask.sum() < 3 or control_mask.sum() < 3:
        raise NuisanceError(f"candidate {candidate!r} has too few records in one arm")
    event_flag = np.asarray([decision.event for decision in decisions], dtype=bool)
    try:
        outcome = OutcomeRegression(
            control=CoxOutcomeModel(ridge=regularisation).fit(
                restricted[control_mask], event_flag[control_mask], design[control_mask]
            ),
            treated=CoxOutcomeModel(ridge=regularisation).fit(
                restricted[treated_mask], event_flag[treated_mask], design[treated_mask]
            ),
        )
    except ValueError as error:
        raise NuisanceError(
            f"the outcome regression of {candidate!r} could not be fitted: {error}"
        ) from error
    censoring = CensoringSurvival.fit(decisions, exposure, floor=censoring_floor)
    return NuisanceFit(
        candidate=candidate,
        propensity=propensity,
        outcome=outcome,
        censoring=censoring,
        exposed_count=int(treated_mask.sum()),
        control_count=int(control_mask.sum()),
        trimmed=0.0,
        restricted_mean=restricted_mean_function(decisions, exposure),
        quadratic=quadratic,
    )


def restricted_mean_function(
    decisions: Sequence[Decision],
    exposure: npt.NDArray[np.float64],
) -> Mapping[tuple[int, int, int], npt.NDArray[np.float64]]:
    """``Q_{a,d}(u, s)``: the expected restricted time beyond ``u``.

    The estimate is an at-risk average of the observed restricted times above
    ``u`` inside the arm-and-covariate group, evaluated on the union of the
    observed times in that group. The arm is the candidate's own exposure, so the
    function is candidate-specific even though its covariate key is not.
    """

    if len(decisions) != exposure.size:
        raise NuisanceError("the conditional-mean fit needs one exposure per decision")
    grouped: dict[tuple[int, int, int], list[float]] = {}
    for decision, arm in zip(decisions, exposure, strict=True):
        restricted = min(decision.follow_up_months, decision.horizon_months)
        grouped.setdefault(censoring_key(decision, arm), []).append(restricted)
    functions: dict[tuple[int, int, int], npt.NDArray[np.float64]] = {}
    for key, records in grouped.items():
        times = np.asarray(records, dtype=np.float64)
        knots = np.unique(times)
        values = np.zeros_like(knots)
        for index, point in enumerate(knots):
            above = times >= point
            if above.any():
                values[index] = float(times[above].mean())
        functions[key] = np.vstack([knots, values])
    return functions


def evaluate_restricted_mean_function(
    functions: Mapping[tuple[int, int, int], npt.NDArray[np.float64]],
    decision: Decision,
    arm: float,
    grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Read ``Q_{a,d}(u, s)`` on a grid, held flat past the last observed time."""

    table = functions.get(censoring_key(decision, arm))
    if table is None:
        return np.zeros_like(grid)
    knots, values = table[0], table[1]
    position = np.searchsorted(knots, grid, side="right") - 1
    evaluated = np.zeros_like(grid)
    inside = position >= 0
    evaluated[inside] = values[position[inside]]
    tail = grid > knots[-1]
    if tail.any():
        evaluated[tail] = values[-1]
    return evaluated


class CensoringModel(Protocol):
    """The only member the martingale reads off a censoring model.

    ``CensoringSurvival`` carries the fitted curves and the floor; the identity
    check in the battery supplies a flat curve that carries neither, so the
    martingale is written against the surface rather than the class.
    """

    def predict(
        self, decision: Decision, arm: float, time: npt.ArrayLike
    ) -> npt.NDArray[np.float64]: ...


def martingale_increment(
    decision: Decision,
    arm: float,
    grid: npt.NDArray[np.float64],
    censoring: CensoringModel,
) -> npt.NDArray[np.float64]:
    """The increments of the censoring martingale ``dM^C(u)`` on a grid.

    ``dM^C(u) = dN^C(u) - 1{Y^- >= u} dLambda^C(u)``, with the censoring hazard
    increment taken from the fitted ``G`` so the two terms move together.
    """

    horizon = line_horizon(decision.state.line)
    observed = min(decision.follow_up_months, horizon)
    censored = not decision.event
    increments = np.zeros_like(grid)
    if grid.size < 2:
        return increments
    widths = np.diff(grid)
    knots = grid[:-1] + widths / 2.0
    left = censoring.predict(decision, arm, grid[:-1])
    right = censoring.predict(decision, arm, grid[1:])
    hazard = np.zeros_like(knots)
    positive = left > 0.0
    hazard[positive] = np.clip(1.0 - right[positive] / left[positive], 0.0, 1.0)
    at_risk = (observed >= knots).astype(np.float64)
    increments[:-1] = -at_risk * hazard
    if censored:
        position = int(np.searchsorted(grid, observed, side="right")) - 1
        if 0 <= position < increments.size:
            increments[position] += 1.0
    return increments


def cumulative_censoring_hazard(
    decision: Decision,
    arm: float,
    censoring: CensoringSurvival,
    grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """``-log G(u | a, s)`` on a grid, used by the Eq. (2) audit."""

    survival = censoring.predict(decision, arm, grid)
    return -np.log(np.clip(survival, 1e-9, 1.0))


def outcome_residual_scale(
    model: RidgeOutcomeModel, features: npt.NDArray[np.float64], outcome: npt.NDArray[np.float64]
) -> float:
    residual = outcome - model.predict(features)
    return float(math.sqrt(float(np.mean(np.square(residual)))))
