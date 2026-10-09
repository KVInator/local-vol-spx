"""Forward and backward grid solvers and spot Greeks."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from surface import AHLocalVariance
from surface import AndreasenHugeSurface

LocalVariance = Callable[[NDArray[np.float64], float, str], ArrayLike]
BoundaryValues = Callable[[float], tuple[float, float]]


@dataclass(frozen=True, eq=False)
class ForwardPDEResult:
    """Normalized call prices at the requested maturities."""

    log_moneyness: NDArray[np.float64]
    maturities: NDArray[np.float64]
    normalized_calls: NDArray[np.float64]
    time_steps: int

    def __post_init__(self) -> None:
        for name in ("log_moneyness", "maturities", "normalized_calls"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ForwardPDESolver:
    """
    Solve

        u_T = 0.5 * v(exp(y), T) * (u_yy - u_y),

    where u(y, T) is the normalized call price.

    The local-variance callback receives:
        normalized strikes, maturity, derivative side.

    The boundary callback supplies the left and right normalized
    call prices at a maturity.

    Time breaks align the grid with coefficient discontinuities.
    Initial Rannacher steps replace each nominal step with two
    implicit Euler half-steps.
    """

    min_log_moneyness: float
    max_log_moneyness: float
    n_space_intervals: int = 400
    max_time_step: float = 1.0 / 365.0
    theta: float = 0.5
    rannacher_steps: int = 2
    _log_grid: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _strike_grid: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _spacing: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        bounds = (self.min_log_moneyness, self.max_log_moneyness)
        if not np.all(np.isfinite(bounds)):
            raise ValueError("Log-moneyness bounds must be finite.")
        if self.min_log_moneyness >= self.max_log_moneyness:
            raise ValueError("Minimum log-moneyness must be below the maximum.")
        if (
            not isinstance(self.n_space_intervals, (int, np.integer))
            or self.n_space_intervals < 4
        ):
            raise ValueError("At least four spatial intervals are required.")
        if not np.isfinite(self.max_time_step) or self.max_time_step <= 0.0:
            raise ValueError("Maximum time step must be positive.")
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("Theta must lie between 0.5 and 1.")
        if (
            not isinstance(self.rannacher_steps, (int, np.integer))
            or self.rannacher_steps < 0
        ):
            raise ValueError("Rannacher steps must be a nonnegative integer.")
        spacing = (
            self.max_log_moneyness - self.min_log_moneyness
        ) / self.n_space_intervals
        if spacing >= 2.0:
            raise ValueError("Spatial spacing must be below 2 for this stencil.")
        log_grid = np.linspace(
            self.min_log_moneyness, self.max_log_moneyness, self.n_space_intervals + 1
        )
        strike_grid = np.exp(log_grid)
        if not np.all(np.isfinite(strike_grid)):
            raise ValueError("Normalized strikes must be finite.")
        log_grid.setflags(write=False)
        strike_grid.setflags(write=False)
        object.__setattr__(self, "_log_grid", log_grid)
        object.__setattr__(self, "_strike_grid", strike_grid)
        object.__setattr__(self, "_spacing", float(spacing))

    @property
    def log_moneyness(self) -> NDArray[np.float64]:
        return self._log_grid.copy()

    @property
    def normalized_strikes(self) -> NDArray[np.float64]:
        return self._strike_grid.copy()

    @staticmethod
    def _boundaries(
        boundary_values: BoundaryValues, maturity: float
    ) -> NDArray[np.float64]:
        values = np.asarray(boundary_values(maturity), dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("Boundary callback must return two finite prices.")
        return values

    def _operator(
        self, local_variance: LocalVariance, maturity: float, side: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        interior_strikes = self._strike_grid[1:-1]
        values = np.asarray(
            local_variance(interior_strikes, maturity, side), dtype=float
        )
        try:
            variance = np.broadcast_to(values, interior_strikes.shape)
        except ValueError as error:
            raise ValueError(
                "Local variance must be scalar or match the interior strike grid."
            ) from error
        if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("Local variance must be finite and nonnegative.")
        h = self._spacing
        diffusion = 0.5 * variance
        lower = diffusion * (1.0 / h**2 + 1.0 / (2.0 * h))
        diagonal = -2.0 * diffusion / h**2
        upper = diffusion * (1.0 / h**2 - 1.0 / (2.0 * h))
        return (lower, diagonal, upper)

    def _advance(
        self,
        prices: NDArray[np.float64],
        old_time: float,
        new_time: float,
        theta: float,
        local_variance: LocalVariance,
        boundary_values: BoundaryValues,
    ) -> NDArray[np.float64]:
        dt = new_time - old_time
        if theta == 1.0:
            rhs = prices[1:-1].copy()
        else:
            lower_old, diagonal_old, upper_old = self._operator(
                local_variance, old_time, "right"
            )
            rhs = (1.0 + (1.0 - theta) * dt * diagonal_old) * prices[1:-1] + (
                1.0 - theta
            ) * dt * (lower_old * prices[:-2] + upper_old * prices[2:])
        lower_new, diagonal_new, upper_new = self._operator(
            local_variance, new_time, "left"
        )
        boundaries = self._boundaries(boundary_values, new_time)
        rhs[0] += theta * dt * lower_new[0] * boundaries[0]
        rhs[-1] += theta * dt * upper_new[-1] * boundaries[1]
        interior_count = len(rhs)
        matrix = np.zeros((3, interior_count), dtype=float)
        matrix[0, 1:] = -theta * dt * upper_new[:-1]
        matrix[1, :] = 1.0 - theta * dt * diagonal_new
        matrix[2, :-1] = -theta * dt * lower_new[1:]
        updated = np.empty_like(prices)
        updated[0] = boundaries[0]
        updated[-1] = boundaries[1]
        updated[1:-1] = solve_banded((1, 1), matrix, rhs, check_finite=False)
        if not np.all(np.isfinite(updated)):
            raise RuntimeError("PDE solve produced nonfinite prices.")
        return updated

    def solve(
        self,
        maturities: ArrayLike,
        initial_prices: ArrayLike,
        local_variance: LocalVariance,
        boundary_values: BoundaryValues,
        *,
        time_breaks: ArrayLike = (),
    ) -> ForwardPDEResult:
        """
        Start from initial_prices at maturities[0].

        Return prices at every requested maturity. Internal time
        steps respect max_time_step and the supplied time breaks.
        """
        times = np.asarray(maturities, dtype=float)
        if (
            times.ndim != 1
            or len(times) < 2
            or (not np.all(np.isfinite(times)))
            or (times[0] < 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "Maturities must be finite, nonnegative and strictly increasing."
            )
        prices = np.array(initial_prices, dtype=float, copy=True)
        if prices.shape != self._strike_grid.shape or not np.all(np.isfinite(prices)):
            raise ValueError(
                "Initial prices must be finite and match the spatial grid."
            )
        initial_boundaries = self._boundaries(boundary_values, float(times[0]))
        if not np.allclose(prices[[0, -1]], initial_boundaries, rtol=0.0, atol=1e-12):
            raise ValueError("Initial prices and boundary prices disagree.")
        breaks = np.asarray(time_breaks, dtype=float)
        if (
            breaks.ndim != 1
            or not np.all(np.isfinite(breaks))
            or np.any(breaks < times[0])
            or np.any(breaks > times[-1])
        ):
            raise ValueError("Time breaks must lie inside the maturity range.")
        milestones = np.unique(np.concatenate((times, breaks)))
        saved = np.empty((len(times), len(self._strike_grid)), dtype=float)
        saved[0] = prices
        checkpoint = 1
        nominal_steps = 0
        actual_steps = 0
        for start, stop in zip(milestones[:-1], milestones[1:]):
            step_count = max(1, int(np.ceil((stop - start) / self.max_time_step)))
            edges = np.linspace(start, stop, step_count + 1)
            for old_time, new_time in zip(edges[:-1], edges[1:]):
                use_startup = nominal_steps < self.rannacher_steps and self.theta < 1.0
                if use_startup:
                    middle = 0.5 * (old_time + new_time)
                    prices = self._advance(
                        prices,
                        float(old_time),
                        float(middle),
                        1.0,
                        local_variance,
                        boundary_values,
                    )
                    prices = self._advance(
                        prices,
                        float(middle),
                        float(new_time),
                        1.0,
                        local_variance,
                        boundary_values,
                    )
                    actual_steps += 2
                else:
                    prices = self._advance(
                        prices,
                        float(old_time),
                        float(new_time),
                        self.theta,
                        local_variance,
                        boundary_values,
                    )
                    actual_steps += 1
                nominal_steps += 1
            if checkpoint < len(times) and stop == times[checkpoint]:
                saved[checkpoint] = prices
                checkpoint += 1
        return ForwardPDEResult(
            log_moneyness=self._log_grid,
            maturities=times,
            normalized_calls=saved,
            time_steps=actual_steps,
        )


@dataclass(frozen=True)
class VanillaGreeks:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True, eq=False)
class BackwardPriceGrid:
    """Conditional option values at one valuation time.

    The state is x=S/F(t), using the original calibrated forward curve.
    Values are E[(X_T-k)+ | X_t=x], where k=K/F(T), for calls.
    Multiply by D(T)/D(t)*F(T) to obtain currency/index-point prices.
    Spot derivatives hold the calibrated local-volatility function and
    deterministic carry fixed. They do not recalibrate the smile.
    """

    log_states: np.ndarray
    normalized_values: np.ndarray
    strike: float
    expiry: float
    valuation_time: float
    kind: str
    valuation_forward: float
    price_scale: float
    time_steps: int
    _curve: CubicSpline = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("log_states", "normalized_values"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "_curve",
            CubicSpline(self.log_states, self.normalized_values, extrapolate=False),
        )

    def _state(self, spots):
        spots = np.asarray(spots, dtype=float)
        if not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
            raise ValueError("Spots must be finite and positive.")
        y = np.log(spots / self.valuation_forward)
        if np.any(y < self.log_states[0]) or np.any(y > self.log_states[-1]):
            raise ValueError("Spot is outside the numerical state domain.")
        return (spots, y)

    def price(self, spots):
        spots, y = self._state(spots)
        if self.expiry == self.valuation_time:
            signed = spots - self.strike if self.kind == "call" else self.strike - spots
            return np.maximum(signed, 0.0)
        return self.price_scale * self._curve(y)

    def greeks(self, spots):
        spots, y = self._state(spots)
        if self.expiry == self.valuation_time:
            if np.any(spots == self.strike):
                raise ValueError("Expiry Greeks are undefined at the payoff kink.")
            delta = (spots > self.strike).astype(float)
            if self.kind == "put":
                delta -= 1.0
            return VanillaGreeks(self.price(spots), delta, np.zeros_like(spots))
        first = self._curve(y, 1)
        second = self._curve(y, 2)
        return VanillaGreeks(
            self.price(spots),
            self.price_scale * first / spots,
            self.price_scale * (second - first) / spots**2,
        )


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
        self,
        model,
        *,
        domain_width=0.5,
        space_intervals=4000,
        steps_per_day=32,
        early_time_power=2.0,
        state_shift=0.0,
        rannacher_steps=2,
    ):
        self.model = model
        self.variance = (
            AHLocalVariance(model) if isinstance(model, AndreasenHugeSurface) else model
        )
        settings = [domain_width, steps_per_day, early_time_power, state_shift]
        if (
            not np.all(np.isfinite(settings))
            or domain_width <= 0
            or steps_per_day <= 0
            or (not 1 <= early_time_power <= 3)
            or (not isinstance(space_intervals, (int, np.integer)))
            or (space_intervals < 4)
            or space_intervals % 2
            or (not isinstance(rannacher_steps, (int, np.integer)))
            or (rannacher_steps < 0)
        ):
            raise ValueError("Invalid spatial or temporal settings.")
        if isinstance(self.variance, AHLocalVariance):
            lower, upper = self.variance.y[[0, -1]]
        else:
            lower = model.min_log_moneyness
            upper = model.max_log_moneyness
        if state_shift - domain_width < lower or state_shift + domain_width > upper:
            raise ValueError("Numerical domain exceeds the coefficient domain.")
        self.y = np.linspace(
            state_shift - domain_width, state_shift + domain_width, space_intervals + 1
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
        milestones.extend(pillars[(pillars > valuation_time) & (pillars < expiry)])
        if len(positive):
            first = float(positive[0])
            count = max(8, int(np.ceil(365 * first * self.steps_per_day)))
            graded = first * np.linspace(0, 1, count + 1) ** self.early_time_power
            milestones.extend(graded[(graded > valuation_time) & (graded < expiry)])
        milestones = np.unique(milestones)
        pieces = []
        for start, stop in zip(milestones[:-1], milestones[1:]):
            count = max(1, int(np.ceil((stop - start) / self.max_dt)))
            pieces.append(np.linspace(start, stop, count + 1)[:-1])
        return np.r_[np.concatenate(pieces), expiry]

    def _advance(self, values, boundaries, old_time, new_time, theta):
        dt = old_time - new_time
        midpoint = new_time + 0.5 * dt
        if not midpoint > 0:
            raise ArithmeticError("Coefficient evaluation requires positive time.")
        variance = np.broadcast_to(
            np.asarray(
                self.variance.normalized_variance(self.z[1:-1], midpoint), float
            ),
            (len(self.z) - 2,),
        )
        if not np.all(np.isfinite(variance)) or np.any(variance < 0):
            raise ArithmeticError("Invalid positive-time local variance.")
        a = 0.5 * variance
        lower = a * (1 / self.h**2 + 1 / (2 * self.h))
        diagonal = -2 * a / self.h**2
        upper = a * (1 / self.h**2 - 1 / (2 * self.h))
        rhs = values[1:-1].copy()
        if theta < 1:
            rhs += (
                (1 - theta)
                * dt
                * (
                    lower[:, None] * values[:-2]
                    + diagonal[:, None] * values[1:-1]
                    + upper[:, None] * values[2:]
                )
            )
        rhs[0] += theta * dt * lower[0] * boundaries[0]
        rhs[-1] += theta * dt * upper[-1] * boundaries[1]
        bands = np.zeros((3, len(diagonal)))
        bands[0, 1:] = -theta * dt * upper[:-1]
        bands[1] = 1 - theta * dt * diagonal
        bands[2, :-1] = -theta * dt * lower[1:]
        updated = np.empty_like(values)
        updated[[0, -1]] = boundaries
        updated[1:-1] = solve_banded((1, 1), bands, rhs, check_finite=False)
        if not np.all(np.isfinite(updated)):
            raise ArithmeticError("Backward solve produced nonfinite values.")
        self._coefficient_times.append(midpoint)
        return updated

    def solve_many(self, strikes, expiry, *, valuation_time=0.0, kind="call"):
        strikes = np.atleast_1d(np.asarray(strikes, float))
        expiry = float(expiry)
        valuation_time = float(valuation_time)
        if (
            strikes.ndim != 1
            or not len(strikes)
            or (not np.all(np.isfinite(strikes)))
            or np.any(strikes <= 0)
            or (not np.all(np.isfinite([expiry, valuation_time])))
            or (not 0 <= valuation_time <= expiry)
            or (kind not in ("call", "put"))
        ):
            raise ValueError("Invalid strikes, maturities or option kind.")
        forward = float(self.model.forward(expiry))
        fixing_forward = float(self.model.forward(valuation_time))
        discount = float(self.model.discount_factor(expiry))
        fixing_discount = float(self.model.discount_factor(valuation_time))
        carry = [forward, fixing_forward, discount, fixing_discount]
        if min(carry) <= 0 or not np.all(np.isfinite(carry)):
            raise ValueError("Carry must be finite and positive.")
        k = strikes / forward
        if np.any(k <= self.z[0]) or np.any(k >= self.z[-1]):
            raise ValueError("Terminal strikes must lie inside the state domain.")
        signed = self.z[:, None] - k
        values = np.maximum(signed if kind == "call" else -signed, 0)
        boundaries = values[[0, -1]].copy()
        self._coefficient_times = []
        step_count = 0
        if expiry > valuation_time:
            mesh = self._calendar_mesh(valuation_time, expiry)
            for nominal, (old, new) in enumerate(zip(mesh[:0:-1], mesh[-2::-1])):
                if nominal < self.rannacher_steps:
                    middle = 0.5 * (old + new)
                    values = self._advance(values, boundaries, old, middle, 1.0)
                    values = self._advance(values, boundaries, middle, new, 1.0)
                    step_count += 2
                else:
                    values = self._advance(values, boundaries, old, new, 0.5)
                    step_count += 1
        self.last_diagnostics = {
            "actual_steps": step_count,
            "minimum_coefficient_time_years": min(
                self._coefficient_times, default=None
            ),
            "zero_time_coefficient_requested": False,
        }
        return tuple(
            (
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
        )

    def solve(self, strike, expiry, *, valuation_time=0.0, kind="call"):
        return self.solve_many(
            [strike], expiry, valuation_time=valuation_time, kind=kind
        )[0]
