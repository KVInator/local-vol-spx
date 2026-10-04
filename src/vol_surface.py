"""Call-price surface with shape-preserving maturity interpolation."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PPoly

from black import BlackPricer
from convex_spline import ConvexCallSpline
from implied_vol import ImpliedVolSolver


def _curve_value(curve, z, derivative=0):
    """Protect evaluations against endpoint roundoff after domain checks."""
    points = np.clip(z, curve.x[0], curve.x[-1])
    return np.asarray(
        curve(points, nu=derivative),
        dtype=float,
    )


def _normalized_curve(model):
    """Express a fitted call spline in z = K/F and c = C/(DF)."""
    polynomial = PPoly.from_spline(
        (model.knots, model.coefficients, 3),
        extrapolate=False,
    )

    forward = model.pricer.forward
    discount = model.pricer.discount_factor
    scale = forward / model.strike_span
    powers = np.arange(3, -1, -1)[:, None]

    coefficients = (
        polynomial.c
        * scale**powers
        / (discount * forward)
    )
    breakpoints = (
        model.strike_origin
        + model.strike_span * polynomial.x
    ) / forward

    curve = PPoly(
        coefficients,
        breakpoints,
        extrapolate=False,
    )
    curve.c.setflags(write=False)
    curve.x.setflags(write=False)
    return curve


def _calendar_minimum(earlier, later, lower, upper):
    """Find the minimum cubic price difference over the full domain."""
    interior_breaks = np.concatenate(
        [
            earlier.x[
                (earlier.x > lower) & (earlier.x < upper)
            ],
            later.x[
                (later.x > lower) & (later.x < upper)
            ],
        ]
    )
    breaks = np.unique(
        np.concatenate(([lower], interior_breaks, [upper]))
    )

    minimum = np.inf
    minimum_location = np.nan

    for left, right in zip(breaks[:-1], breaks[1:]):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)

        # Represent the cubic difference on a scaled interval [-1, 1].
        coefficients = np.array(
            [
                float(
                    _curve_value(later, midpoint, order)
                    - _curve_value(earlier, midpoint, order)
                )
                * half_width**order
                / (1, 1, 2, 6)[order]
                for order in range(4)
            ]
        )

        derivative_coefficients = np.array(
            [
                coefficients[1],
                2.0 * coefficients[2],
                3.0 * coefficients[3],
            ]
        )
        roots = np.polynomial.polynomial.polyroots(
            derivative_coefficients
        )

        candidates = [left, right]
        for root in roots:
            if abs(root.imag) <= 1e-10:
                position = float(root.real)
                if -1.0 <= position <= 1.0:
                    candidates.append(
                        midpoint + half_width * position
                    )

        points = np.asarray(candidates)
        differences = (
            _curve_value(later, points)
            - _curve_value(earlier, points)
        )
        index = int(np.argmin(differences))

        if differences[index] < minimum:
            minimum = float(differences[index])
            minimum_location = float(points[index])

    return {
        "intervals_checked": int(len(breaks) - 1),
        "minimum_normalized_call_change": minimum,
        "log_moneyness_at_minimum": float(
            np.log(minimum_location)
        ),
    }


@dataclass(frozen=True, eq=False)
class NormalizedCallSurface:
    """Interpolate normalized call prices between calibrated expiries.

    Forward prices and discount factors use log-linear interpolation.
    Strike and maturity extrapolation are excluded.

    Calendar checks cover every cubic interval, subject to floating-point
    precision and the stated normalized-price tolerance.
    """

    splines: tuple[ConvexCallSpline, ...]
    min_log_moneyness: float = -0.09
    max_log_moneyness: float = 0.09
    calendar_tolerance: float = 1e-12

    _times: np.ndarray = field(init=False, repr=False)
    _log_forwards: np.ndarray = field(init=False, repr=False)
    _log_discounts: np.ndarray = field(init=False, repr=False)
    _curves: tuple = field(init=False, repr=False)
    _checks: tuple = field(init=False, repr=False)

    def __post_init__(self):
        splines = tuple(self.splines)

        if len(splines) < 2:
            raise ValueError("At least two expiry splines are required.")

        if (
            not np.isfinite(self.min_log_moneyness)
            or not np.isfinite(self.max_log_moneyness)
            or self.min_log_moneyness >= self.max_log_moneyness
        ):
            raise ValueError("Invalid log-moneyness domain.")

        if (
            not np.isfinite(self.calendar_tolerance)
            or self.calendar_tolerance < 0.0
        ):
            raise ValueError("Invalid calendar tolerance.")

        times = np.array(
            [model.pricer.maturity for model in splines],
            dtype=float,
        )
        if (
            np.any(~np.isfinite(times))
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "Spline maturities must be positive and strictly increasing."
            )

        curves = []
        endpoint_tolerance = (
            32.0
            * np.finfo(float).eps
            * max(1.0, self.z_max)
        )

        for model in splines:
            if not model.shape_checks()["passed"]:
                raise ValueError("An input spline failed its shape checks.")

            forward = model.pricer.forward
            supported_lower = model.strike_origin / forward
            supported_upper = model.strike_max / forward

            if (
                self.z_min < supported_lower - endpoint_tolerance
                or self.z_max > supported_upper + endpoint_tolerance
            ):
                raise ValueError(
                    "Surface domain exceeds an input spline's support."
                )

            curves.append(_normalized_curve(model))

        checks = []
        for index in range(len(curves) - 1):
            check = _calendar_minimum(
                curves[index],
                curves[index + 1],
                self.z_min,
                self.z_max,
            )
            check.update(
                {
                    "earlier_days": float(365.0 * times[index]),
                    "later_days": float(365.0 * times[index + 1]),
                    "passed": bool(
                        check["minimum_normalized_call_change"]
                        >= -self.calendar_tolerance
                    ),
                }
            )
            checks.append(check)

            if not check["passed"]:
                raise ValueError(
                    "Calendar ordering failed between "
                    f"{365.0 * times[index]:.2f} and "
                    f"{365.0 * times[index + 1]:.2f} days: "
                    "minimum normalized call change "
                    f"{check['minimum_normalized_call_change']:.6e}."
                )

        log_forwards = np.log(
            [model.pricer.forward for model in splines]
        )
        log_discounts = np.log(
            [model.pricer.discount_factor for model in splines]
        )

        for array in (times, log_forwards, log_discounts):
            array.setflags(write=False)

        object.__setattr__(self, "splines", splines)
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_log_forwards", log_forwards)
        object.__setattr__(self, "_log_discounts", log_discounts)
        object.__setattr__(self, "_curves", tuple(curves))
        object.__setattr__(self, "_checks", tuple(checks))

    @property
    def z_min(self):
        return float(np.exp(self.min_log_moneyness))

    @property
    def z_max(self):
        return float(np.exp(self.max_log_moneyness))

    @property
    def maturities(self):
        return self._times.copy()

    @staticmethod
    def _restore(value):
        array = np.asarray(value, dtype=float)
        return float(array) if array.ndim == 0 else array

    def _validated_time(self, maturity):
        time = np.asarray(maturity, dtype=float)
        if (
            np.any(~np.isfinite(time))
            or np.any(time < self._times[0])
            or np.any(time > self._times[-1])
        ):
            raise ValueError("Requested maturity is outside surface support.")
        return time

    def _coordinates(self, normalized_strike, maturity):
        z, time = np.broadcast_arrays(
            np.asarray(normalized_strike, dtype=float),
            self._validated_time(maturity),
        )

        tolerance = (
            32.0 * np.finfo(float).eps * max(1.0, self.z_max)
        )
        if (
            np.any(~np.isfinite(z))
            or np.any(z < self.z_min - tolerance)
            or np.any(z > self.z_max + tolerance)
        ):
            raise ValueError(
                "Requested strike is outside the surface domain."
            )

        return np.clip(z, self.z_min, self.z_max), time

    def _brackets(self, time, side="right"):
        return np.clip(
            np.searchsorted(self._times, time, side=side) - 1,
            0,
            len(self._times) - 2,
        )

    def forward(self, maturity):
        time = self._validated_time(maturity)
        return self._restore(
            np.exp(np.interp(time, self._times, self._log_forwards))
        )

    def discount_factor(self, maturity):
        time = self._validated_time(maturity)
        return self._restore(
            np.exp(np.interp(time, self._times, self._log_discounts))
        )

    def normalized_call(self, normalized_strike, maturity, derivative=0):
        """Return c or its first/second derivative with respect to z."""
        if derivative not in (0, 1, 2):
            raise ValueError("Supported strike derivative orders are 0, 1, 2.")

        z, time = self._coordinates(normalized_strike, maturity)
        flat_z = z.ravel()
        flat_time = time.ravel()
        brackets = self._brackets(flat_time)
        output = np.empty(flat_z.size)

        for index in np.unique(brackets):
            selected = brackets == index
            weight = (
                flat_time[selected] - self._times[index]
            ) / (
                self._times[index + 1] - self._times[index]
            )

            earlier = _curve_value(
                self._curves[index],
                flat_z[selected],
                derivative,
            )
            later = _curve_value(
                self._curves[index + 1],
                flat_z[selected],
                derivative,
            )
            output[selected] = (
                (1.0 - weight) * earlier + weight * later
            )

        return self._restore(output.reshape(z.shape))

    def normalized_time_derivative(
        self,
        normalized_strike,
        maturity,
        side="right",
    ):
        """Return dc/dT at fixed z.

        At an internal expiry, side selects the left or right derivative.
        At either support endpoint, the available interior derivative is used.
        """
        if side not in ("left", "right"):
            raise ValueError("Derivative side must be 'left' or 'right'.")

        z, time = self._coordinates(normalized_strike, maturity)
        flat_z = z.ravel()
        brackets = self._brackets(time.ravel(), side=side)
        output = np.empty(flat_z.size)

        for index in np.unique(brackets):
            selected = brackets == index
            output[selected] = (
                _curve_value(
                    self._curves[index + 1],
                    flat_z[selected],
                )
                - _curve_value(
                    self._curves[index],
                    flat_z[selected],
                )
            ) / (
                self._times[index + 1] - self._times[index]
            )

        return self._restore(output.reshape(z.shape))

    def call_price(self, strike, maturity):
        forward = self.forward(maturity)
        discount = self.discount_factor(maturity)
        normalized = self.normalized_call(
            np.asarray(strike, dtype=float) / forward,
            maturity,
        )
        return self._restore(discount * forward * normalized)

    def strike_slope(self, strike, maturity):
        forward = self.forward(maturity)
        discount = self.discount_factor(maturity)
        slope = self.normalized_call(
            np.asarray(strike, dtype=float) / forward,
            maturity,
            derivative=1,
        )
        return self._restore(discount * slope)

    def strike_curvature(self, strike, maturity):
        forward = self.forward(maturity)
        discount = self.discount_factor(maturity)
        curvature = self.normalized_call(
            np.asarray(strike, dtype=float) / forward,
            maturity,
            derivative=2,
        )
        return self._restore(discount / forward * curvature)

    def implied_volatility(self, strike, maturity):
        strikes, times = np.broadcast_arrays(
            np.asarray(strike, dtype=float),
            self._validated_time(maturity),
        )
        prices = np.asarray(self.call_price(strikes, times))
        flat_strikes = strikes.ravel()
        flat_times = times.ravel()
        flat_prices = prices.ravel()
        output = np.empty(flat_strikes.size)

        # Reuse a pricer and solver for each distinct maturity.
        for time in np.unique(flat_times):
            solver = ImpliedVolSolver(
                BlackPricer(
                    forward=self.forward(time),
                    discount_factor=self.discount_factor(time),
                    maturity=float(time),
                )
            )
            for index in np.flatnonzero(flat_times == time):
                output[index] = solver.solve(
                    float(flat_prices[index]),
                    float(flat_strikes[index]),
                    "call",
                ).volatility

        return self._restore(output.reshape(strikes.shape))

    def total_variance(self, strike, maturity):
        volatility = self.implied_volatility(strike, maturity)
        return self._restore(
            np.asarray(volatility) ** 2 * np.asarray(maturity)
        )

    def calendar_checks(self):
        return [dict(check) for check in self._checks]

    def metadata(self):
        return {
            "format_version": 1,
            "model_type": "normalized_call_surface",
            "min_log_moneyness": float(self.min_log_moneyness),
            "max_log_moneyness": float(self.max_log_moneyness),
            "calendar_tolerance": float(self.calendar_tolerance),
            "maturity_years": self._times.tolist(),
            "forward_prices": np.exp(self._log_forwards).tolist(),
            "discount_factors": np.exp(self._log_discounts).tolist(),
            "maturity_interpolation": "linear normalized call prices",
            "carry_interpolation": "log-linear forward and discount factor",
            "maturity_derivatives": (
                "piecewise constant at fixed normalized strike; "
                "one-sided at expiry pillars"
            ),
            "calendar_check_scope": (
                "endpoints and stationary points on every cubic interval "
                "within the supported domain"
            ),
            "global_tail_extension": False,
            "calendar_checks": self.calendar_checks(),
        }

    @classmethod
    def load(cls, path):
        """Load a surface manifest and its referenced expiry models."""
        path = Path(path)
        specification = json.loads(path.read_text())

        if (
            specification["format_version"] != 1
            or specification["model_type"] != "normalized_call_surface"
        ):
            raise ValueError("Unsupported surface manifest.")

        splines = tuple(
            ConvexCallSpline.load(path.parent / filename)
            for filename in specification["slice_files"]
        )
        return cls(
            splines=splines,
            min_log_moneyness=specification["min_log_moneyness"],
            max_log_moneyness=specification["max_log_moneyness"],
            calendar_tolerance=specification["calendar_tolerance"],
        )