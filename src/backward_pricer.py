"""Backward vanilla pricing with a fixed calibrated local-volatility function."""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from forward_pde import ForwardPDESolver


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
            self, "_curve",
            CubicSpline(self.log_states, self.normalized_values, extrapolate=False),
        )

    def _state(self, spots):
        spots = np.asarray(spots, dtype=float)
        if not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
            raise ValueError("Spots must be finite and positive.")
        y = np.log(spots / self.valuation_forward)
        if np.any(y < self.log_states[0]) or np.any(y > self.log_states[-1]):
            raise ValueError("Spot is outside the numerical state domain.")
        return spots, y

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


@dataclass(frozen=True, eq=False)
class BackwardVanillaPricer:
    """Solve the backward generator in normalized log-spot coordinates.

    v_t + 0.5*a(exp(y),t)*(v_yy-v_y)=0.

    With tau=T-t this has the existing forward solver's PDE form.
    Reversing time also reverses one-sided coefficient evaluations:
    tau-right means calendar-left; tau-left means calendar-right.
    Rannacher startup is therefore applied at the terminal payoff.

    Spatial boundaries use martingale asymptotes. Prices and Greeks near
    these artificial boundaries require domain-sensitivity checks.
    """

    surface: object
    min_log_state: float = -1.0
    max_log_state: float = 1.0
    n_space_intervals: int = 1600
    max_time_step: float = 1.0 / (365.0 * 32.0)
    theta: float = 0.5
    rannacher_steps: int = 2
    _solver: ForwardPDESolver = field(init=False, repr=False)

    def __post_init__(self):
        if (
            self.min_log_state < self.surface.min_log_moneyness
            or self.max_log_state > self.surface.max_log_moneyness
        ):
            raise ValueError("Numerical state domain exceeds the surface domain.")
        solver = ForwardPDESolver(
            self.min_log_state, self.max_log_state,
            n_space_intervals=self.n_space_intervals,
            max_time_step=self.max_time_step,
            theta=self.theta, rannacher_steps=self.rannacher_steps,
        )
        object.__setattr__(self, "_solver", solver)

    def solve(self, strike, expiry, *, valuation_time=0.0, kind="call"):
        strike, expiry, valuation_time = map(float, (strike, expiry, valuation_time))
        if (
            not np.all(np.isfinite([strike, expiry, valuation_time]))
            or strike <= 0.0 or valuation_time < 0.0 or expiry < valuation_time
        ):
            raise ValueError("Require positive strike and 0 <= valuation time <= expiry.")
        if kind not in ("call", "put"):
            raise ValueError("Option kind must be 'call' or 'put'.")
        expiry_forward = float(self.surface.forward(expiry))
        valuation_forward = float(self.surface.forward(valuation_time))
        expiry_discount = float(self.surface.discount_factor(expiry))
        valuation_discount = float(self.surface.discount_factor(valuation_time))
        if (
            not np.all(np.isfinite([
                expiry_forward, valuation_forward, expiry_discount, valuation_discount,
            ]))
            or min(expiry_forward, valuation_forward, expiry_discount, valuation_discount) <= 0.0
        ):
            raise ValueError("Forwards and discounts must be finite and positive.")
        x = self._solver.normalized_strikes
        k = strike / expiry_forward
        if not x[0] < k < x[-1]:
            raise ValueError("The terminal strike must lie inside the state domain.")
        payoff = np.maximum(x - k if kind == "call" else k - x, 0.0)
        horizon = expiry - valuation_time
        if horizon == 0.0:
            values, step_count = payoff, 0
        else:
            pillars = np.asarray(self.surface.maturities, dtype=float)
            inside = pillars[(pillars > valuation_time) & (pillars < expiry)]
            breaks = np.sort(expiry - inside)

            def variance(states, tau, side):
                time = expiry - tau
                if tau == 0.0:
                    time = expiry
                elif tau == horizon:
                    time = valuation_time
                if len(pillars):
                    nearest = int(np.argmin(np.abs(pillars - time)))
                    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(expiry))
                    if abs(time - pillars[nearest]) <= tolerance:
                        time = float(pillars[nearest])
                calendar_side = "left" if side == "right" else "right"
                return self.surface.normalized_variance(states, time, calendar_side)

            def boundaries(tau):
                return tuple(payoff[[0, -1]])

            result = self._solver.solve(
                [0.0, horizon], payoff, variance, boundaries,
                time_breaks=breaks,
            )
            values, step_count = result.normalized_calls[-1], result.time_steps
        return BackwardPriceGrid(
            log_states=self._solver.log_moneyness,
            normalized_values=values,
            strike=strike, expiry=expiry, valuation_time=valuation_time, kind=kind,
            valuation_forward=valuation_forward,
            price_scale=expiry_discount / valuation_discount * expiry_forward,
            time_steps=step_count,
        )