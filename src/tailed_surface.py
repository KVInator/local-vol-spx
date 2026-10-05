"""Interpolate calendar-ordered tail slices and extend them to time zero."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from black import BlackPricer
from convex_spline import ConvexCallSpline
from coupled_tails import CoupledTailSlice, check_coupled_calendar
from implied_vol import ImpliedVolSolver
from short_end import ShortEndSurface


@dataclass(frozen=True, eq=False)
class TailedCallSurface:
    """Linear normalized-call interpolation of globally extended slices.

    Tail formulas apply to all positive strikes. The explicit numerical
    domain limits evaluation and the PDE experiment; it is not market
    coverage. Calendar checks are recomputed from the loaded slices.
    """

    slices: tuple
    min_log_moneyness: float = -1.0
    max_log_moneyness: float = 1.0
    calendar_tolerance: float = 1e-12
    _times: np.ndarray = field(init=False, repr=False)
    _checks: tuple = field(init=False, repr=False)

    def __post_init__(self):
        slices = tuple(self.slices)
        times = np.array(
            [model.maturity for model in slices],
            dtype=float,
        )

        if (
            len(slices) < 2
            or not np.all(np.isfinite(times))
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "Provide at least two increasing positive maturities."
            )

        if not (
            np.isfinite(self.min_log_moneyness)
            and np.isfinite(self.max_log_moneyness)
            and self.min_log_moneyness < 0.0 < self.max_log_moneyness
            and np.isfinite(self.calendar_tolerance)
            and self.calendar_tolerance >= 0.0
        ):
            raise ValueError(
                "Invalid evaluation domain or calendar tolerance."
            )

        carry = np.array(
            [
                [
                    model.core.pricer.forward,
                    model.core.pricer.discount_factor,
                ]
                for model in slices
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(carry)) or np.any(carry <= 0.0):
            raise ValueError(
                "Forwards and discounts must be finite and positive."
            )

        checks = tuple(
            check_coupled_calendar(
                earlier,
                later,
                price_tolerance=self.calendar_tolerance,
            )
            for earlier, later in zip(slices[:-1], slices[1:])
        )

        if not all(check["passed"] for check in checks):
            raise ValueError(
                "Loaded tail slices failed continuous calendar checks."
            )

        times.setflags(write=False)

        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_checks", checks)

    @property
    def maturities(self):
        return self._times.copy()

    def calendar_checks(self):
        return tuple(dict(check) for check in self._checks)

    def _checked_time(self, time):
        time = float(time)

        if not np.isfinite(time):
            raise ValueError("Maturity must be finite.")

        nearest = int(np.argmin(np.abs(self._times - time)))
        tolerance = (
            32.0
            * np.finfo(float).eps
            * max(1.0, abs(time))
        )

        if abs(time - self._times[nearest]) <= tolerance:
            time = float(self._times[nearest])

        if not self._times[0] <= time <= self._times[-1]:
            raise ValueError("Maturity is outside the fitted range.")

        return time

    def _checked_times(self, time):
        """Validate arrays without discarding their broadcast dimensions."""

        times = np.asarray(time, dtype=float)

        if times.ndim == 0:
            return np.asarray(self._checked_time(times.item()))

        unique, inverse = np.unique(times, return_inverse=True)
        checked = np.array(
            [self._checked_time(value) for value in unique]
        )

        return checked[inverse].reshape(times.shape)

    def _evaluate_time_grid(
        self,
        normalized_strikes,
        time,
        evaluator,
    ):
        """Evaluate scalar-time kernels over broadcastable strike/time grids."""

        z = self._checked_strikes(normalized_strikes)
        times = self._checked_times(time)

        if times.ndim == 0:
            return evaluator(z, times.item())

        z, times = np.broadcast_arrays(z, times)
        unique = np.unique(times)

        if len(unique) == 1:
            return evaluator(z, float(unique[0]))

        values = np.empty(z.shape, dtype=float)

        for maturity in unique:
            mask = times == maturity
            values[mask] = evaluator(
                z[mask],
                float(maturity),
            )

        return values

    def _checked_strikes(self, normalized_strikes):
        z = np.asarray(normalized_strikes, dtype=float)

        if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
            raise ValueError(
                "Normalized strikes must be finite and positive."
            )

        y = np.log(z)

        if (
            np.any(y < self.min_log_moneyness - 1e-12)
            or np.any(y > self.max_log_moneyness + 1e-12)
        ):
            raise ValueError(
                "Strikes are outside the numerical evaluation domain."
            )

        return z

    def _interval(self, time, side="right"):
        if side not in ("left", "right"):
            raise ValueError(
                "Derivative side must be 'left' or 'right'."
            )

        time = self._checked_time(time)
        index = int(
            np.searchsorted(self._times, time, side=side)
        ) - 1

        index = min(
            max(index, 0),
            len(self.slices) - 2,
        )

        return index, time

    def _interpolate(self, z, time, evaluator):
        index, time = self._interval(time)
        left, right = self._times[index:index + 2]
        fraction = (time - left) / (right - left)

        if fraction == 0.0:
            return evaluator(self.slices[index], z)

        if fraction == 1.0:
            return evaluator(self.slices[index + 1], z)

        return (
            (1.0 - fraction)
            * evaluator(self.slices[index], z)
            + fraction
            * evaluator(self.slices[index + 1], z)
        )

    def _carry(self, time, name):
        times = self._checked_times(time)

        values = [
            getattr(model.core.pricer, name)
            for model in self.slices
        ]

        result = np.exp(
            np.interp(times, self._times, np.log(values))
        )

        return (
            float(result)
            if np.ndim(result) == 0
            else result
        )

    def forward(self, time):
        return self._carry(time, "forward")

    def discount_factor(self, time):
        return self._carry(time, "discount_factor")

    def normalized_call(
        self,
        normalized_strikes,
        time,
        derivative=0,
    ):
        if derivative not in (0, 1, 2):
            raise ValueError(
                "Supported strike derivative orders are 0, 1 and 2."
            )

        return self._evaluate_time_grid(
            normalized_strikes,
            time,
            lambda z, maturity: self._interpolate(
                z,
                maturity,
                lambda model, values: model.normalized_call(
                    values,
                    derivative,
                ),
            ),
        )

    @staticmethod
    def _slice_put(model, z, derivative):
        if derivative == 0:
            return model.normalized_put(z)

        values = np.array(
            model.normalized_call(z, derivative),
            copy=True,
        )

        if derivative == 1:
            values += 1.0

            # Evaluate tiny left-tail put slopes without
            # subtracting two numbers close to one.
            left = z < model.left.normalized_strike

            if np.any(left):
                anchor = model.left
                ratio = z[left] / anchor.normalized_strike
                denominator = (
                    1.0
                    + anchor.shape * (1.0 - ratio)
                )
                elasticity = (
                    anchor.power
                    + anchor.shape * ratio / denominator
                )

                values[left] = (
                    model.normalized_put(z[left])
                    * elasticity
                    / z[left]
                )

        return values

    def normalized_put(
        self,
        normalized_strikes,
        time,
        derivative=0,
    ):
        if derivative not in (0, 1, 2):
            raise ValueError(
                "Supported strike derivative orders are 0, 1 and 2."
            )

        return self._evaluate_time_grid(
            normalized_strikes,
            time,
            lambda z, maturity: self._interpolate(
                z,
                maturity,
                lambda model, values: self._slice_put(
                    model,
                    values,
                    derivative,
                ),
            ),
        )

    def normalized_time_derivative(
        self,
        normalized_strikes,
        time,
        side="right",
    ):
        if side not in ("left", "right"):
            raise ValueError(
                "Derivative side must be 'left' or 'right'."
            )

        return self._evaluate_time_grid(
            normalized_strikes,
            time,
            lambda z, maturity: self._time_derivative(
                z,
                maturity,
                side,
            ),
        )

    def _time_derivative(self, z, time, side):
        index, _ = self._interval(time, side)
        early, late = self.slices[index:index + 2]

        # The intrinsic term cancels. OTM prices avoid
        # cancellation in the wings.
        gap = np.where(
            z < 1.0,
            late.normalized_put(z)
            - early.normalized_put(z),
            late.normalized_call(z)
            - early.normalized_call(z),
        )

        if (
            not np.all(np.isfinite(gap))
            or np.any(gap < 0.0)
        ):
            raise ValueError(
                "A negative or nonfinite calendar increment was found."
            )

        return gap / (late.maturity - early.maturity)

    def call_price(self, strikes, time):
        forward = self.forward(time)

        return (
            self.discount_factor(time)
            * forward
            * self.normalized_call(
                np.asarray(strikes, dtype=float) / forward,
                time,
            )
        )

    def strike_slope(self, strikes, time):
        forward = self.forward(time)

        return (
            self.discount_factor(time)
            * self.normalized_call(
                np.asarray(strikes, dtype=float) / forward,
                time,
                derivative=1,
            )
        )

    def strike_curvature(self, strikes, time):
        forward = self.forward(time)

        return (
            self.discount_factor(time)
            / forward
            * self.normalized_call(
                np.asarray(strikes, dtype=float) / forward,
                time,
                derivative=2,
            )
        )

    def implied_volatility(self, strikes, time):
        times = self._checked_times(time)

        strikes, times = np.broadcast_arrays(
            np.asarray(strikes, dtype=float),
            times,
        )

        forward = self.forward(times)
        discount = self.discount_factor(times)
        z = self._checked_strikes(strikes / forward)

        puts = self.normalized_put(z, times)
        calls = self.normalized_call(z, times)

        result = np.empty_like(z)

        for maturity in np.unique(times):
            maturity = float(maturity)
            f = self.forward(maturity)
            d = self.discount_factor(maturity)

            solver = ImpliedVolSolver(
                BlackPricer(f, d, maturity)
            )

            mask = times == maturity

            selected_z = z[mask]
            selected_k = strikes[mask]
            selected_put = puts[mask]
            selected_call = calls[mask]

            solved = np.empty(
                selected_z.shape,
                dtype=float,
            )

            for index in range(len(solved)):
                kind = (
                    "put"
                    if selected_z[index] < 1.0
                    else "call"
                )

                native_price = (
                    selected_put[index]
                    if kind == "put"
                    else selected_call[index]
                )

                price = d * f * native_price

                solved[index] = solver.solve(
                    price=float(price),
                    strike=float(selected_k[index]),
                    kind=kind,
                ).volatility

            result[mask] = solved

        return result

    def metadata(self):
        return {
            "format_version": 1,
            "model_type": "tailed_normalized_call_surface",
            "maturity_years": self._times.tolist(),
            "min_log_moneyness": self.min_log_moneyness,
            "max_log_moneyness": self.max_log_moneyness,
            "domain_scope": (
                "Numerical evaluation bounds, including assumed tails."
            ),
            "maturity_interpolation": (
                "Linear normalized call prices."
            ),
            "carry_interpolation": (
                "Log-linear forward and discount factor."
            ),
            "global_tail_extension": True,
            "calendar_checks": self.calendar_checks(),
        }

    @classmethod
    def load(
        cls,
        path,
        min_log_moneyness=-1.0,
        max_log_moneyness=1.0,
    ):
        path = Path(path)

        specification = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            specification.get("format_version") != 1
            or specification.get("model_type")
            != "calendar_ordered_tail_slices"
        ):
            raise ValueError(
                "Unsupported coupled-tail specification."
            )

        slices = []

        for row in specification["slices"]:
            core = ConvexCallSpline.load(
                path.parent / row["core_file"]
            )

            if not np.isclose(
                core.pricer.maturity,
                row["maturity_years"],
                rtol=0.0,
                atol=1e-14,
            ):
                raise ValueError(
                    "Manifest maturity disagrees with the core model."
                )

            slices.append(
                CoupledTailSlice(
                    core=core,
                    left_log_moneyness=row["left_log_moneyness"],
                    right_log_moneyness=row["right_log_moneyness"],
                    left_power=row["left_power"],
                    right_power=row["right_power"],
                )
            )

        return cls(
            tuple(slices),
            min_log_moneyness,
            max_log_moneyness,
        )


@dataclass(frozen=True, eq=False)
class TailedShortEndSurface(ShortEndSurface):
    """Use the existing frozen first smile with stable density factors.

    For alpha=T/T1, g(alpha)=(1-alpha)*g0+alpha*g1
    +alpha*(1-alpha)*w_y**2/16. g1 is recovered directly from
    anchor call curvature. Neither prices nor local variance are clipped.
    """

    def _calculate_anchor(self, strike_key):
        z = np.asarray(strike_key, dtype=float)
        y = np.log(z)
        time = self.first_maturity
        forward = self.surface.forward(time)

        sigma = np.asarray(
            self.surface.implied_volatility(
                forward * z,
                time,
            )
        )

        w = sigma**2 * time

        if (
            not np.all(np.isfinite(w))
            or np.any(w <= 0.0)
        ):
            raise ValueError(
                "Anchor variance must be finite and positive."
            )

        root_w = np.sqrt(w)
        d1 = -y / root_w + 0.5 * root_w
        d2 = d1 - root_w

        phi = (
            np.exp(-0.5 * d1**2)
            / np.sqrt(2.0 * np.pi)
        )
        black_w = phi / (2.0 * root_w)

        if np.any(black_w <= 0.0):
            raise ValueError(
                "Anchor IV sensitivity underflowed."
            )

        observed_y = np.where(
            z < 1.0,
            z * self.surface.normalized_put(
                z,
                time,
                derivative=1,
            ),
            z * self.surface.normalized_call(
                z,
                time,
                derivative=1,
            ),
        )

        black_y = np.where(
            z < 1.0,
            z * ndtr(-d2),
            -z * ndtr(d2),
        )

        w_y = (observed_y - black_y) / black_w

        g0 = (
            1.0 - y * w_y / (2.0 * w)
        ) ** 2

        g1 = (
            self.surface.normalized_call(
                z,
                time,
                derivative=2,
            )
            * z**2
            * root_w
            / phi
        )

        # Preserve the parent class's cached-anchor interface.
        w_yy = 2.0 * (
            g1
            - g0
            + w_y**2 / (4.0 * w)
            + w_y**2 / 16.0
        )

        values = (w, w_y, w_yy)

        if any(
            not np.all(np.isfinite(value))
            for value in values
        ):
            raise ValueError(
                "Anchor derivatives must be finite."
            )

        for value in values:
            value.setflags(write=False)

        return values

    def _short_geometry(self, z, time):
        w, w_y, _ = self._anchor(z)

        root_w = np.sqrt(w)
        d1 = -np.log(z) / root_w + 0.5 * root_w

        phi = (
            np.exp(-0.5 * d1**2)
            / np.sqrt(2.0 * np.pi)
        )

        g0 = (
            1.0 - np.log(z) * w_y / (2.0 * w)
        ) ** 2

        g1 = (
            self.surface.normalized_call(
                z,
                self.first_maturity,
                derivative=2,
            )
            * z**2
            * root_w
            / phi
        )

        alpha = time / self.first_maturity

        factor = (
            (1.0 - alpha) * g0
            + alpha * g1
            + alpha * (1.0 - alpha) * w_y**2 / 16.0
        )

        if (
            not np.all(np.isfinite(factor))
            or np.any(factor <= 0.0)
        ):
            raise ValueError(
                "Short-end density factor is nonpositive or nonfinite."
            )

        return (
            w,
            alpha * w_y,
            alpha * w,
            factor,
        )

    def metadata(self):
        return {
            **super().metadata(),
            "model_type": (
                "tailed_first_smile_short_end_extension"
            ),
            "global_tail_extension": True,
            "density_factor_evaluation": (
                "Direct endpoint curvature and concave interpolation."
            ),
        }

    @classmethod
    def load(cls, path):
        path = Path(path)

        specification = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            specification.get("format_version") != 1
            or specification.get("model_type")
            != "tailed_first_smile_short_end_extension"
        ):
            raise ValueError(
                "Unsupported tailed short-end specification."
            )

        return cls(
            surface=TailedCallSurface.load(
                path.parent / specification["tail_slices_file"],
                specification["min_log_moneyness"],
                specification["max_log_moneyness"],
            ),
            spot=specification["spot"],
        )