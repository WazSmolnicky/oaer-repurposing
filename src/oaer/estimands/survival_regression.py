"""A Cox outcome regression for the restricted mean.

The arm-specific outcome regression ``mu_{a,d}(s)`` is the expected restricted
time at the line's horizon. Fitting it as a linear regression of the capped time
on the state is misspecified by construction, because the cap makes the response
nonlinear in the linear predictor and a large share of the records sits at the
cap. This module fits a proportional-hazards model instead -- Breslow's partial
likelihood by Newton steps with a ridge penalty -- and reads the restricted mean
off the fitted baseline cumulative hazard, which is the natural functional for the
estimand.

The Newton step uses the reverse-cumulative-sum form of the risk-set averages, so
one iteration costs ``O(n p + U p^2)`` rather than ``O(U n p)``.

Ref: Sec. 2.5 (the outcome regression of Eq. (2)), Sec. 2.7 (the prognostic
reference score is a Cox model).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DEFAULT_RIDGE: float = 1.0
DEFAULT_ITERATIONS: int = 25
DEFAULT_TOLERANCE: float = 1e-7


class CoxOutcomeModel:
    """A ridge-penalised Cox model with its baseline cumulative hazard."""

    def __init__(
        self,
        ridge: float = DEFAULT_RIDGE,
        iterations: int = DEFAULT_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        self.ridge = float(ridge)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.coefficients: npt.NDArray[np.float64] | None = None
        self.means: npt.NDArray[np.float64] | None = None
        self.scales: npt.NDArray[np.float64] | None = None
        self.baseline_times: npt.NDArray[np.float64] | None = None
        self.baseline_hazard: npt.NDArray[np.float64] | None = None
        self.risk_level: float = 0.0

    @property
    def fitted(self) -> bool:
        return self.coefficients is not None

    def _standardise(
        self, features: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.means is None or self.scales is None:
            means = features.mean(axis=0)
            scales = features.std(axis=0)
            scales[scales < 1e-8] = 1.0
            if means.size:
                means[0] = 0.0
                scales[0] = 1.0
            return (features - means) / scales, means, scales
        return (features - self.means) / self.scales, self.means, self.scales

    def fit(
        self,
        times: npt.ArrayLike,
        events: npt.ArrayLike,
        features: npt.NDArray[np.float64],
    ) -> CoxOutcomeModel:
        """Fit by Newton steps on the penalised partial likelihood."""

        time = np.asarray(times, dtype=np.float64).ravel()
        event = np.asarray(events, dtype=bool).ravel()
        if time.size != event.size or time.size != features.shape[0]:
            raise ValueError("times, events and features must agree on the row count")
        if time.size < 4 or event.sum() < 2:
            raise ValueError("the Cox fit needs at least four records and two events")
        scaled, means, scales = self._standardise(features)
        self.means = means
        self.scales = scales
        order = np.argsort(time, kind="stable")
        ordered_time = time[order]
        ordered_event = event[order]
        ordered = scaled[order]
        rows, width = ordered.shape
        penalty = self.ridge * np.eye(width)
        penalty[0, 0] = 0.0
        coefficients = np.zeros(width)
        for _ in range(self.iterations):
            linear = np.clip(ordered @ coefficients, -30.0, 30.0)
            risk = np.exp(linear)
            reverse_risk = np.cumsum(risk[::-1])[::-1]
            reverse_first = np.cumsum((risk[:, None] * ordered)[::-1], axis=0)[::-1]
            gradient = np.zeros(width)
            hessian = np.zeros((width, width))
            event_rows = np.nonzero(ordered_event)[0]
            previous = -np.inf
            for row in event_rows:
                if ordered_time[row] == previous:
                    continue
                previous = float(ordered_time[row])
                total = reverse_risk[row]
                if total <= 0.0:
                    continue
                mean = reverse_first[row] / total
                gradient += ordered[row] - mean
                hessian -= (ordered[row][:, None] * ordered[row][None, :]) - mean[:, None] * mean[
                    None, :
                ]
            gradient -= penalty @ coefficients
            hessian -= penalty
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            coefficients = coefficients - step
            if float(np.abs(step).max()) < self.tolerance:
                break
        self.coefficients = coefficients
        self._estimate_baseline(ordered_time, ordered_event, ordered @ coefficients)
        return self

    def _estimate_baseline(
        self,
        time: npt.NDArray[np.float64],
        event: npt.NDArray[np.bool_],
        linear: npt.NDArray[np.float64],
    ) -> None:
        """Breslow's baseline cumulative hazard at the event times."""

        risk = np.exp(np.clip(linear, -30.0, 30.0))
        reverse_risk = np.cumsum(risk[::-1])[::-1]
        knots = [0.0]
        values = [0.0]
        cumulative = 0.0
        previous = -np.inf
        for row in np.nonzero(event)[0]:
            point = float(time[row])
            if point == previous:
                continue
            previous = point
            total = reverse_risk[row]
            if total > 0.0:
                cumulative += 1.0 / total
            knots.append(point)
            values.append(cumulative)
        self.baseline_times = np.asarray(knots, dtype=np.float64)
        self.baseline_hazard = np.asarray(values, dtype=np.float64)
        self.risk_level = float(np.mean(linear))

    def _design(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.means is None or self.scales is None:
            raise ValueError("the model has not been fitted")
        return (features - self.means) / self.scales

    def survival(
        self,
        features: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """``S(u | s)`` on a grid, one row per feature row."""

        if not self.fitted or self.baseline_times is None or self.baseline_hazard is None:
            raise ValueError("the model has not been fitted")
        if self.coefficients is None:
            raise ValueError("the model has not been fitted")
        scaled = self._design(features)
        linear = np.clip(scaled @ self.coefficients, -30.0, 30.0)
        position = np.searchsorted(self.baseline_times, grid, side="right") - 1
        position = np.clip(position, 0, self.baseline_hazard.size - 1)
        baseline = self.baseline_hazard[position]
        return np.exp(-baseline[None, :] * np.exp(linear)[:, None])

    def restricted_mean(
        self,
        features: npt.NDArray[np.float64],
        horizon: float,
        *,
        points: int = 121,
    ) -> npt.NDArray[np.float64]:
        """``E[min(T, horizon) | s] = int_0^horizon S(u | s) du``."""

        grid = np.linspace(0.0, horizon, points)
        curve = self.survival(features, grid)
        restricted: npt.NDArray[np.float64] = np.asarray(
            np.trapezoid(curve, grid, axis=1), dtype=np.float64
        )
        return restricted

    def linear_predictor(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.coefficients is None:
            raise ValueError("the model has not been fitted")
        return self._design(features) @ self.coefficients


def cox_restricted_mean(
    times: npt.ArrayLike,
    events: npt.ArrayLike,
    features: npt.NDArray[np.float64],
    horizon: float,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> tuple[CoxOutcomeModel, npt.NDArray[np.float64]]:
    """Fit a Cox model and return it with its own restricted-mean predictions."""

    model = CoxOutcomeModel(ridge=ridge).fit(times, events, features)
    return model, model.restricted_mean(features, horizon)
