"""Backward prices and fixed-model Greeks with positive-time coefficients."""

import numpy as np
from scipy.linalg import solve_banded

from ah_local_vol import AHLocalVariance
from andreasen_huge import AndreasenHugeSurface
from backward_pricer import BackwardPriceGrid


class AHBackwardPricer:
    """Midpoint Crank-Nicolson in calendar time, marching backwards.

    The generator is 0.5*a(exp(y), t)*(d_yy-d_y), with y=log(S/F(t)).
    Coefficients are evaluated strictly inside each calendar interval.
    Every pillar is a time break; early breaks are graded toward zero.
    Terminal Rannacher steps use two implicit half-steps. No coefficient
    is evaluated at zero and no positive-time coefficient is clipped.

    Boundary values use martingale payoff asymptotes. Spot Greeks keep
    the original physical local-volatility function and carry fixed.
    """

    def __init__(
        self, model, *, domain_width=0.5, space_intervals=4000,
        steps_per_day=32, early_time_power=2.0, state_shift=0.0,
        rannacher_steps=2,
    ):
        self.model = model
        self.variance = (
            AHLocalVariance(model)
            if isinstance(model, AndreasenHugeSurface) else model
        )

        settings = [
            domain_width, steps_per_day, early_time_power, state_shift,
        ]
        if (
            not np.all(np.isfinite(settings))
            or domain_width <= 0
            or steps_per_day <= 0
            or not 1 <= early_time_power <= 3
            or not isinstance(space_intervals, (int, np.integer))
            or space_intervals < 4
            or space_intervals % 2
            or not isinstance(rannacher_steps, (int, np.integer))
            or rannacher_steps < 0
        ):
            raise ValueError("Invalid spatial or temporal settings.")

        if isinstance(self.variance, AHLocalVariance):
            lower, upper = self.variance.y[[0, -1]]
        else:
            lower = model.min_log_moneyness
            upper = model.max_log_moneyness

        if (
            state_shift - domain_width < lower
            or state_shift + domain_width > upper
        ):
            raise ValueError(
                "Numerical domain exceeds the coefficient domain."
            )

        self.y = np.linspace(
            state_shift - domain_width,
            state_shift + domain_width,
            space_intervals + 1,
        )
        self.z = np.exp(self.y)
        self.h = 2 * domain_width / space_intervals

        if self.h >= 2:
            raise ValueError("Spatial spacing must be below two.")

        self.max_dt = 1 / (365 * steps_per_day)
        self.steps_per_day = float(steps_per_day)
        self.early_time_power = float(early_time_power)
        self.rannacher_steps = int(rannacher_steps)
        self.last_diagnostics = {}

    @classmethod
    def load(cls, path, **settings):
        return cls(AndreasenHugeSurface.load(path), **settings)

    def _calendar_mesh(self, valuation_time, expiry):
        pillars = np.asarray(self.model.maturities, float)
        positive = pillars[pillars > 0]

        milestones = [valuation_time, expiry]
        milestones.extend(
            pillars[
                (pillars > valuation_time) & (pillars < expiry)
            ]
        )

        if len(positive):
            first = float(positive[0])
            count = max(
                8,
                int(np.ceil(365 * first * self.steps_per_day)),
            )
            graded = (
                first
                * np.linspace(0, 1, count + 1)
                ** self.early_time_power
            )
            milestones.extend(
                graded[
                    (graded > valuation_time) & (graded < expiry)
                ]
            )

        milestones = np.unique(milestones)
        pieces = []

        for start, stop in zip(
            milestones[:-1], milestones[1:]
        ):
            count = max(
                1,
                int(np.ceil((stop - start) / self.max_dt)),
            )
            pieces.append(
                np.linspace(start, stop, count + 1)[:-1]
            )

        return np.r_[np.concatenate(pieces), expiry]

    def _advance(
        self, values, boundaries, old_time, new_time, theta,
    ):
        dt = old_time - new_time
        midpoint = new_time + 0.5 * dt

        if not midpoint > 0:
            raise ArithmeticError(
                "Coefficient evaluation requires positive time."
            )

        variance = np.broadcast_to(
            np.asarray(
                self.variance.normalized_variance(
                    self.z[1:-1], midpoint
                ),
                float,
            ),
            (len(self.z) - 2,),
        )

        if (
            not np.all(np.isfinite(variance))
            or np.any(variance < 0)
        ):
            raise ArithmeticError(
                "Invalid positive-time local variance."
            )

        a = 0.5 * variance
        lower = a * (
            1 / self.h**2 + 1 / (2 * self.h)
        )
        diagonal = -2 * a / self.h**2
        upper = a * (
            1 / self.h**2 - 1 / (2 * self.h)
        )

        rhs = values[1:-1].copy()

        if theta < 1:
            rhs += (1 - theta) * dt * (
                lower[:, None] * values[:-2]
                + diagonal[:, None] * values[1:-1]
                + upper[:, None] * values[2:]
            )

        rhs[0] += (
            theta * dt * lower[0] * boundaries[0]
        )
        rhs[-1] += (
            theta * dt * upper[-1] * boundaries[1]
        )

        bands = np.zeros((3, len(diagonal)))
        bands[0, 1:] = -theta * dt * upper[:-1]
        bands[1] = 1 - theta * dt * diagonal
        bands[2, :-1] = -theta * dt * lower[1:]

        updated = np.empty_like(values)
        updated[[0, -1]] = boundaries
        updated[1:-1] = solve_banded(
            (1, 1),
            bands,
            rhs,
            check_finite=False,
        )

        if not np.all(np.isfinite(updated)):
            raise ArithmeticError(
                "Backward solve produced nonfinite values."
            )

        self._coefficient_times.append(midpoint)
        return updated

    def solve_many(
        self, strikes, expiry, *,
        valuation_time=0.0, kind="call",
    ):
        strikes = np.atleast_1d(
            np.asarray(strikes, float)
        )
        expiry = float(expiry)
        valuation_time = float(valuation_time)

        if (
            strikes.ndim != 1
            or not len(strikes)
            or not np.all(np.isfinite(strikes))
            or np.any(strikes <= 0)
            or not np.all(
                np.isfinite([expiry, valuation_time])
            )
            or not 0 <= valuation_time <= expiry
            or kind not in ("call", "put")
        ):
            raise ValueError(
                "Invalid strikes, maturities or option kind."
            )

        forward = float(self.model.forward(expiry))
        fixing_forward = float(
            self.model.forward(valuation_time)
        )
        discount = float(
            self.model.discount_factor(expiry)
        )
        fixing_discount = float(
            self.model.discount_factor(valuation_time)
        )

        carry = [
            forward, fixing_forward,
            discount, fixing_discount,
        ]
        if (
            min(carry) <= 0
            or not np.all(np.isfinite(carry))
        ):
            raise ValueError(
                "Carry must be finite and positive."
            )

        k = strikes / forward

        if (
            np.any(k <= self.z[0])
            or np.any(k >= self.z[-1])
        ):
            raise ValueError(
                "Terminal strikes must lie inside the state domain."
            )

        signed = self.z[:, None] - k
        values = np.maximum(
            signed if kind == "call" else -signed,
            0,
        )
        boundaries = values[[0, -1]].copy()
        self._coefficient_times = []
        step_count = 0

        if expiry > valuation_time:
            mesh = self._calendar_mesh(
                valuation_time, expiry
            )

            for nominal, (old, new) in enumerate(
                zip(mesh[:0:-1], mesh[-2::-1])
            ):
                if nominal < self.rannacher_steps:
                    middle = 0.5 * (old + new)
                    values = self._advance(
                        values, boundaries,
                        old, middle, 1.0,
                    )
                    values = self._advance(
                        values, boundaries,
                        middle, new, 1.0,
                    )
                    step_count += 2
                else:
                    values = self._advance(
                        values, boundaries,
                        old, new, 0.5,
                    )
                    step_count += 1

        self.last_diagnostics = {
            "actual_steps": step_count,
            "minimum_coefficient_time_years": min(
                self._coefficient_times,
                default=None,
            ),
            "zero_time_coefficient_requested": False,
        }

        return tuple(
            BackwardPriceGrid(
                self.y,
                values[:, i],
                float(strike),
                expiry,
                valuation_time,
                kind,
                fixing_forward,
                discount / fixing_discount * forward,
                step_count,
            )
            for i, strike in enumerate(strikes)
        )

    def solve(
        self, strike, expiry, *,
        valuation_time=0.0, kind="call",
    ):
        return self.solve_many(
            [strike],
            expiry,
            valuation_time=valuation_time,
            kind=kind,
        )[0]