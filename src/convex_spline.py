"""Constrained cubic splines for a single-expiry call-price curve."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.linalg import qr, solve_triangular
from scipy.optimize import linprog, minimize

from black import BlackPricer


FloatArray = NDArray[np.float64]


def _scale_constraints(
    matrix: FloatArray,
    bounds: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Rescale inequalities without changing their feasible set."""
    scales = np.linalg.norm(matrix, axis=1)

    if np.any(scales == 0.0):
        raise ValueError("A constraint has a zero coefficient row.")

    return matrix / scales[:, None], bounds / scales


@dataclass(frozen=True, slots=True)
class ConvexCallSpline:
    """A fitted call-price curve, restricted to its calibrated domain."""

    pricer: BlackPricer
    strike_origin: float
    strike_span: float
    knots: FloatArray = field(repr=False, compare=False)
    coefficients: FloatArray = field(repr=False, compare=False)
    minimum_multiplier: float
    band_cap: float
    smoothing_weight: float
    curvature_floor: float
    roughness: float
    objective: float
    iterations: int
    rms_half_spreads: float
    max_half_spreads: float
    outside_bands: int
    _spline: BSpline = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        knots = np.array(self.knots, dtype=float, copy=True)
        coefficients = np.array(
            self.coefficients,
            dtype=float,
            copy=True,
        )
        knots.setflags(write=False)
        coefficients.setflags(write=False)

        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "_spline",
            BSpline(
                knots,
                coefficients,
                3,
                extrapolate=False,
            ),
        )

    @property
    def strike_max(self) -> float:
        return self.strike_origin + self.strike_span

    @property
    def n_intervals(self) -> int:
        return len(self.coefficients) - 3

    def _evaluate(
        self,
        strike: ArrayLike,
        derivative_order: int,
    ) -> float | FloatArray:
        strikes = np.asarray(strike, dtype=float)

        if np.any(~np.isfinite(strikes)):
            raise ValueError("Strikes must be finite.")

        if np.any(
            (strikes < self.strike_origin)
            | (strikes > self.strike_max)
        ):
            raise ValueError(
                "Evaluation outside the calibrated strike domain "
                "is not supported."
            )

        x = (strikes - self.strike_origin) / self.strike_span
        values = np.asarray(
            self._spline(x, nu=derivative_order)
            / self.strike_span**derivative_order
        )

        return float(values) if values.ndim == 0 else values

    def price(self, strike: ArrayLike) -> float | FloatArray:
        return self._evaluate(strike, 0)

    def strike_slope(
        self,
        strike: ArrayLike,
    ) -> float | FloatArray:
        return self._evaluate(strike, 1)

    def strike_curvature(
        self,
        strike: ArrayLike,
    ) -> float | FloatArray:
        return self._evaluate(strike, 2)

    def shape_checks(self) -> dict:
        """Check the constraints at the points determining their extrema."""
        breakpoints = np.unique(self.knots)
        curvatures = (
            self._spline(breakpoints, nu=2)
            / self.strike_span**2
        )

        endpoint_strikes = np.array(
            [self.strike_origin, self.strike_max]
        )
        endpoint_prices = np.asarray(
            self.price(endpoint_strikes)
        )
        endpoint_slopes = np.asarray(
            self.strike_slope(endpoint_strikes)
        )

        intrinsic = self.pricer.discount_factor * np.maximum(
            self.pricer.forward - endpoint_strikes,
            0.0,
        )
        upper_bound = (
            self.pricer.discount_factor * self.pricer.forward
        )

        price_violation = max(
            0.0,
            float(np.max(intrinsic - endpoint_prices)),
            float(np.max(endpoint_prices - upper_bound)),
        )
        slope_violation = max(
            0.0,
            -self.pricer.discount_factor
            - float(endpoint_slopes[0]),
            float(endpoint_slopes[1]),
        )
        curvature_shortfall = max(
            0.0,
            self.curvature_floor - float(np.min(curvatures)),
        )

        return {
            "minimum_curvature": float(np.min(curvatures)),
            "left_slope": float(endpoint_slopes[0]),
            "right_slope": float(endpoint_slopes[1]),
            "price_bound_violation_points": price_violation,
            "slope_bound_violation": slope_violation,
            "curvature_floor_shortfall": curvature_shortfall,
            "passed": bool(
                price_violation <= 1e-7
                and slope_violation <= 1e-9
                and curvature_shortfall <= 1e-10
            ),
        }

    def metadata(self) -> dict:
        return {
            "format_version": 1,
            "pricer": {
                "forward": self.pricer.forward,
                "discount_factor": self.pricer.discount_factor,
                "maturity": self.pricer.maturity,
            },
            "strike_origin": self.strike_origin,
            "strike_span": self.strike_span,
            "minimum_multiplier": self.minimum_multiplier,
            "band_cap": self.band_cap,
            "smoothing_weight": self.smoothing_weight,
            "curvature_floor": self.curvature_floor,
            "roughness": self.roughness,
            "objective": self.objective,
            "iterations": self.iterations,
            "rms_half_spreads": self.rms_half_spreads,
            "max_half_spreads": self.max_half_spreads,
            "outside_bands": self.outside_bands,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)

        if path.suffix != ".npz":
            raise ValueError("Spline model files must use .npz.")

        np.savez_compressed(
            path,
            knots=self.knots,
            coefficients=self.coefficients,
            metadata=json.dumps(self.metadata()),
        )

    @classmethod
    def load(cls, path: str | Path) -> ConvexCallSpline:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(
                str(stored["metadata"].item())
            )

            if metadata.pop("format_version") != 1:
                raise ValueError("Unsupported spline model format.")

            pricer = BlackPricer(**metadata.pop("pricer"))

            return cls(
                pricer=pricer,
                knots=stored["knots"],
                coefficients=stored["coefficients"],
                **metadata,
            )


@dataclass(frozen=True, slots=True)
class ConvexSplineCalibrator:
    """Fit a smooth call curve under shape and quote-band constraints."""

    pricer: BlackPricer
    n_intervals: int = 30
    smoothing_weight: float = 1e-3
    band_slack: float = 1.5
    curvature_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.pricer.maturity <= 0.0:
            raise ValueError("Calibration requires positive maturity.")

        if (
            not isinstance(self.n_intervals, int)
            or self.n_intervals < 1
        ):
            raise ValueError("n_intervals must be a positive integer.")

        for name in ("smoothing_weight", "curvature_floor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if (
            not np.isfinite(self.band_slack)
            or self.band_slack < 0.0
        ):
            raise ValueError("band_slack must be finite and nonnegative.")

    def fit(
        self,
        strikes: ArrayLike,
        midpoints: ArrayLike,
        half_widths: ArrayLike,
    ) -> ConvexCallSpline:
        k, mid, half = [
            np.array(values, dtype=float, copy=True)
            for values in (strikes, midpoints, half_widths)
        ]

        if (
            k.ndim != 1
            or mid.shape != k.shape
            or half.shape != k.shape
            or len(k) < 4
        ):
            raise ValueError(
                "Provide matching one-dimensional arrays "
                "with at least four observations."
            )

        if not all(
            np.all(np.isfinite(values))
            for values in (k, mid, half)
        ):
            raise ValueError("Calibration inputs must be finite.")

        if np.any(k <= 0.0) or np.any(np.diff(k) <= 0.0):
            raise ValueError(
                "Strikes must be positive and strictly increasing."
            )

        if np.any(half <= 0.0):
            raise ValueError("Quote half-widths must be positive.")

        origin = float(k[0])
        span = float(k[-1] - k[0])
        x_quotes = (k - origin) / span
        breaks = np.linspace(0.0, 1.0, self.n_intervals + 1)

        knots = np.concatenate(
            [
                np.repeat(0.0, 4),
                breaks[1:-1],
                np.repeat(1.0, 4),
            ]
        )
        n_coefficients = len(knots) - 4

        basis = BSpline(
            knots,
            np.eye(n_coefficients),
            3,
            extrapolate=False,
        )
        quote_basis = basis(x_quotes)
        endpoint_values = basis([0.0, 1.0])
        endpoint_slopes = basis([0.0, 1.0], nu=1) / span
        knot_curvatures = basis(breaks, nu=2) / span**2

        discount = self.pricer.discount_factor
        forward = self.pricer.forward
        endpoint_intrinsic = discount * np.maximum(
            forward - k[[0, -1]],
            0.0,
        )

        # G @ coefficients <= g.
        shape_matrix = np.vstack(
            [
                -knot_curvatures,
                -endpoint_slopes[0],
                endpoint_slopes[1],
                -endpoint_values[0],
                -endpoint_values[1],
                endpoint_values[0],
            ]
        )
        shape_bounds = np.concatenate(
            [
                np.full(len(breaks), -self.curvature_floor),
                [
                    discount,
                    0.0,
                    -endpoint_intrinsic[0],
                    -endpoint_intrinsic[1],
                    discount * forward,
                ],
            ]
        )

        # LP variables are spline coefficients and a band multiplier.
        lp_matrix = np.vstack(
            [
                np.column_stack(
                    [
                        shape_matrix,
                        np.zeros(len(shape_bounds)),
                    ]
                ),
                np.column_stack([quote_basis, -half]),
                np.column_stack([-quote_basis, -half]),
            ]
        )
        lp_bounds = np.concatenate(
            [shape_bounds, mid, -mid]
        )
        lp_matrix, lp_bounds = _scale_constraints(
            lp_matrix,
            lp_bounds,
        )

        lp_objective = np.zeros(n_coefficients + 1)
        lp_objective[-1] = 1.0

        lp_result = linprog(
            lp_objective,
            A_ub=lp_matrix,
            b_ub=lp_bounds,
            bounds=[(None, None)] * n_coefficients
            + [(0.0, None)],
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )

        if not lp_result.success:
            raise RuntimeError(
                f"Spline feasibility LP failed: {lp_result.message}"
            )

        minimum_multiplier = max(
            0.0,
            float(lp_result.x[-1]),
        )
        band_cap = (
            minimum_multiplier + self.band_slack + 1e-6
        )

        # For u(x) = C(K(x)) / span, this gives
        # R = integral_0^1 [u'''(x)]^2 dx exactly.
        interval_midpoints = 0.5 * (
            breaks[:-1] + breaks[1:]
        )
        roughness_matrix = (
            np.sqrt(np.diff(breaks))[:, None]
            * basis(interval_midpoints, nu=3)
            / span
        )

        weighted_design = (
            quote_basis / half[:, None] / np.sqrt(len(k))
        )
        weighted_target = mid / half / np.sqrt(len(k))

        design = np.vstack(
            [
                weighted_design,
                np.sqrt(self.smoothing_weight)
                * roughness_matrix,
            ]
        )
        target = np.concatenate(
            [
                weighted_target,
                np.zeros(self.n_intervals),
            ]
        )

        if np.linalg.matrix_rank(design) < n_coefficients:
            raise ValueError("The fitting system is rank deficient.")

        q_matrix, r_matrix = qr(design, mode="economic")
        unconstrained = solve_triangular(
            r_matrix,
            q_matrix.T @ target,
        )

        fit_matrix = np.vstack(
            [shape_matrix, quote_basis, -quote_basis]
        )
        fit_bounds = np.concatenate(
            [
                shape_bounds,
                mid + band_cap * half,
                -mid + band_cap * half,
            ]
        )

        # coefficients = unconstrained + R^{-1} z.
        # The variable part of the objective is then 0.5 * z @ z.
        transformed_matrix = solve_triangular(
            r_matrix.T,
            fit_matrix.T,
            lower=True,
        ).T
        transformed_bounds = (
            fit_bounds - fit_matrix @ unconstrained
        )
        transformed_matrix, transformed_bounds = (
            _scale_constraints(
                transformed_matrix,
                transformed_bounds,
            )
        )

        initial_z = r_matrix @ (
            lp_result.x[:-1] - unconstrained
        )

        result = minimize(
            fun=lambda z: 0.5 * float(z @ z),
            x0=initial_z,
            jac=lambda z: z,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda z: (
                    transformed_bounds
                    - transformed_matrix @ z
                ),
                "jac": lambda z: -transformed_matrix,
            },
            options={
                "ftol": 1e-11,
                "maxiter": 500,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"Constrained spline fit failed: {result.message}"
            )

        coefficients = unconstrained + solve_triangular(
            r_matrix,
            result.x,
        )
        normalized_moves = (
            quote_basis @ coefficients - mid
        ) / half
        roughness = float(
            np.sum((roughness_matrix @ coefficients) ** 2)
        )
        rms_half_spreads = float(
            np.sqrt(np.mean(normalized_moves**2))
        )
        max_half_spreads = float(
            np.max(np.abs(normalized_moves))
        )

        fitted = ConvexCallSpline(
            pricer=self.pricer,
            strike_origin=origin,
            strike_span=span,
            knots=knots,
            coefficients=coefficients,
            minimum_multiplier=minimum_multiplier,
            band_cap=band_cap,
            smoothing_weight=self.smoothing_weight,
            curvature_floor=self.curvature_floor,
            roughness=roughness,
            objective=0.5 * (
                rms_half_spreads**2
                + self.smoothing_weight * roughness
            ),
            iterations=int(result.nit),
            rms_half_spreads=rms_half_spreads,
            max_half_spreads=max_half_spreads,
            outside_bands=int(
                np.count_nonzero(
                    np.abs(normalized_moves) > 1.0 + 1e-7
                )
            ),
        )

        if (
            not fitted.shape_checks()["passed"]
            or max_half_spreads > band_cap + 1e-6
        ):
            raise RuntimeError(
                "The fitted curve failed its constraint checks."
            )

        return fitted