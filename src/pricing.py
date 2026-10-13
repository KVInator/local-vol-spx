"""Black pricing and implied-volatility inversion."""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import erf, log_ndtr, ndtr

OptionKind = Literal["call", "put"]
NumericResult = float | NDArray[np.float64]
INV_SQRT_TWO_PI = 1.0 / np.sqrt(2.0 * np.pi)
SQRT_EIGHT = np.sqrt(8.0)


@dataclass(frozen=True, slots=True)
class BlackPricer:
    """Price European options for one expiry.

    Parameters
    ----------
    forward
        Positive forward price.
    discount_factor
        Positive discount factor to expiry.
    maturity
        Nonnegative time to expiry, in years.

    Notes
    -----
    Strikes and prices use the same units as the forward.
    Volatility is annualised and expressed as a decimal.
    Scalar inputs return a float; array inputs follow NumPy broadcasting.
    """

    forward: float
    discount_factor: float
    maturity: float

    def __post_init__(self) -> None:
        for name in ("forward", "discount_factor", "maturity"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if name == "maturity":
                if value < 0.0:
                    raise ValueError("maturity must be nonnegative.")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)

    @staticmethod
    def _sign(kind: OptionKind) -> float:
        if kind == "call":
            return 1.0
        if kind == "put":
            return -1.0
        raise ValueError("kind must be 'call' or 'put'.")

    @staticmethod
    def _inputs(
        strike: ArrayLike, volatility: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        strikes = np.asarray(strike, dtype=float)
        volatilities = np.asarray(volatility, dtype=float)
        if np.any(~np.isfinite(strikes)) or np.any(strikes <= 0.0):
            raise ValueError("Strikes must be finite and positive.")
        if np.any(~np.isfinite(volatilities)) or np.any(volatilities < 0.0):
            raise ValueError("Volatilities must be finite and nonnegative.")
        strikes, volatilities = np.broadcast_arrays(strikes, volatilities)
        return (strikes, volatilities)

    @staticmethod
    def _result(value: ArrayLike) -> NumericResult:
        array = np.asarray(value, dtype=float)
        return float(array) if array.ndim == 0 else array

    @staticmethod
    def _log_ratio(
        numerator: NDArray[np.float64], denominator: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate log(numerator/denominator), including near equality."""
        result = np.log(numerator) - np.log(denominator)
        nearby = (numerator > 0.5 * denominator) & (denominator > 0.5 * numerator)
        result[nearby] = np.log1p(
            (numerator[nearby] - denominator[nearby]) / denominator[nearby]
        )
        return result

    def price_bounds(
        self, strike: ArrayLike, kind: OptionKind = "call"
    ) -> tuple[NumericResult, NumericResult]:
        """Return discounted intrinsic value and the price upper bound."""
        sign = self._sign(kind)
        strikes, _ = self._inputs(strike, 0.0)
        lower = self.discount_factor * np.maximum(sign * (self.forward - strikes), 0.0)
        upper = self.discount_factor * (
            np.full_like(strikes, self.forward) if kind == "call" else strikes
        )
        return (self._result(lower), self._result(upper))

    def price(
        self, strike: ArrayLike, volatility: ArrayLike, kind: OptionKind = "call"
    ) -> NumericResult:
        """Return the discounted option price.

        Zero volatility or zero maturity returns discounted intrinsic
        value. Both option types share the same time value.
        """
        sign = self._sign(kind)
        strikes, volatilities = self._inputs(strike, volatility)
        intrinsic = np.maximum(sign * (self.forward - strikes), 0.0)
        time_value = np.zeros_like(strikes)
        if self.maturity == 0.0:
            return self._result(self.discount_factor * intrinsic)
        positive = volatilities > 0.0
        if np.any(positive):
            active_strikes = strikes[positive]
            small = np.minimum(self.forward, active_strikes)
            large = np.maximum(self.forward, active_strikes)
            log_moneyness = self._log_ratio(large, small)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                std = volatilities[positive] * np.sqrt(self.maturity)
                d1 = -log_moneyness / std + 0.5 * std
                d2 = -log_moneyness / std - 0.5 * std
                log_a = log_ndtr(d1)
                log_b = log_ndtr(d2)
            otm_value = np.zeros_like(std)
            finite_a = np.isfinite(log_a)
            log_b_over_a = log_moneyness[finite_a] + log_b[finite_a] - log_a[finite_a]
            otm_value[finite_a] = (
                small[finite_a]
                * np.exp(log_a[finite_a])
                * -np.expm1(np.minimum(log_b_over_a, 0.0))
            )
            at_money = small == large
            otm_value[at_money] = small[at_money] * erf(std[at_money] / SQRT_EIGHT)
            time_value[positive] = otm_value
        return self._result(self.discount_factor * (intrinsic + time_value))

    def vega(self, strike: ArrayLike, volatility: ArrayLike) -> NumericResult:
        """Return dPrice/dVolatility, holding forward and discount fixed.

        Vega is per unit decimal volatility. Multiply by 0.01 for
        sensitivity to one volatility percentage point.

        At zero volatility, return the right-hand derivative:
        zero away from ATM and D*F*sqrt(T)/sqrt(2*pi) at ATM.
        At zero maturity, vega is zero.
        """
        strikes, volatilities = self._inputs(strike, volatility)
        values = np.zeros_like(strikes)
        if self.maturity == 0.0:
            return self._result(values)
        root_time = np.sqrt(self.maturity)
        positive = volatilities > 0.0
        if np.any(positive):
            active_strikes = strikes[positive]
            forwards = np.full_like(active_strikes, self.forward)
            log_moneyness = self._log_ratio(forwards, active_strikes)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                std = volatilities[positive] * root_time
                d1 = log_moneyness / std + 0.5 * std
                d1 = np.where((std == 0.0) & (active_strikes == self.forward), 0.0, d1)
                density = INV_SQRT_TWO_PI * np.exp(-0.5 * d1**2)
            values[positive] = self.discount_factor * self.forward * root_time * density
        zero_at_money = (volatilities == 0.0) & (strikes == self.forward)
        values[zero_at_money] = (
            self.discount_factor * self.forward * root_time * INV_SQRT_TWO_PI
        )
        return self._result(values)


IVStatus = Literal["solved", "at_lower_bound"]


@dataclass(frozen=True, slots=True)
class IVResult:
    """An IV estimate and its numerical diagnostics.

    price_error is repriced_price minus the supplied price.
    Vega and volatility sensitivity use decimal volatility units.
    price_evaluations includes bracketing, root finding and repricing.
    """

    volatility: float
    repriced_price: float
    price_error: float
    vega: float
    iterations: int
    price_evaluations: int
    bracket_upper: float
    status: IVStatus

    @property
    def price_to_vol_sensitivity(self) -> float:
        """Local volatility change per unit price change."""
        return 1.0 / self.vega if self.vega > 0.0 else np.inf


@dataclass(frozen=True, slots=True)
class ImpliedVolSolver:
    """Invert individual European option prices for one expiry.

    Tolerances
    ----------
    vol_tolerance
        Absolute root tolerance in decimal volatility.
    price_tolerance
        Maximum absolute repricing error, in option-price units.

    Boundary policy
    ---------------
    The exact lower price bound is reported as zero volatility with
    status 'at_lower_bound'. This is a boundary convention, rather
    than evidence that an observed market volatility is exactly zero.

    Prices below the lower bound or at/above the upper bound are
    rejected. Zero maturity is rejected because volatility cannot
    be identified from an expiry payoff.
    """

    pricer: BlackPricer
    initial_upper: float = 0.5
    max_volatility: float = 8.0
    vol_tolerance: float = 1e-12
    price_tolerance: float = 1e-09
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.pricer, BlackPricer):
            raise TypeError("pricer must be a BlackPricer.")
        for name in (
            "initial_upper",
            "max_volatility",
            "vol_tolerance",
            "price_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.initial_upper > self.max_volatility:
            raise ValueError("initial_upper cannot exceed max_volatility.")
        if (
            not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))

    @staticmethod
    def _scalar(value, name: str) -> float:
        array = np.asarray(value, dtype=float)
        if array.ndim != 0 or not np.isfinite(array):
            raise ValueError(f"{name} must be a finite scalar.")
        return float(array)

    def solve(self, price: float, strike: float, kind: OptionKind = "call") -> IVResult:
        """Recover IV for a single quote without modifying its price."""
        target = self._scalar(price, "price")
        strike = self._scalar(strike, "strike")
        if self.pricer.maturity == 0.0:
            raise ValueError("Implied volatility is not identifiable at zero maturity.")
        lower_price, upper_price = self.pricer.price_bounds(strike, kind)
        if target < lower_price:
            raise ValueError(
                f"Price {target:.12g} is below the lower bound {lower_price:.12g}."
            )
        if target >= upper_price:
            raise ValueError(
                f"Price {target:.12g} is at or above the upper bound {upper_price:.12g}; no finite IV is available."
            )
        evaluations = 0

        def objective(volatility: float) -> float:
            nonlocal evaluations
            evaluations += 1
            residual = float(self.pricer.price(strike, volatility, kind)) - target
            if not np.isfinite(residual):
                raise RuntimeError(
                    "A nonfinite pricing residual occurred during inversion."
                )
            return residual

        if target == lower_price:
            volatility = 0.0
            iterations = 0
            upper = 0.0
            status: IVStatus = "at_lower_bound"
        else:
            upper = self.initial_upper
            upper_residual = objective(upper)
            while upper_residual < 0.0 and upper < self.max_volatility:
                upper = min(2.0 * upper, self.max_volatility)
                upper_residual = objective(upper)
            if upper_residual < 0.0:
                raise RuntimeError(
                    f"Could not bracket IV within the configured max_volatility={self.max_volatility:.12g}."
                )
            if upper_residual == 0.0:
                volatility = upper
                iterations = 0
            else:
                volatility, information = brentq(
                    objective,
                    0.0,
                    upper,
                    xtol=self.vol_tolerance,
                    rtol=4.0 * np.finfo(float).eps,
                    maxiter=self.max_iterations,
                    full_output=True,
                    disp=False,
                )
                if not information.converged:
                    raise RuntimeError(
                        f"Brent's method did not converge: {information.flag}."
                    )
                iterations = int(information.iterations)
            status = "solved"
        repriced = float(self.pricer.price(strike, volatility, kind))
        evaluations += 1
        price_error = repriced - target
        if not np.isfinite(repriced) or abs(price_error) > self.price_tolerance:
            raise RuntimeError(
                f"The IV estimate failed the repricing check: error={price_error:.6e}, tolerance={self.price_tolerance:.6e}."
            )
        return IVResult(
            volatility=float(volatility),
            repriced_price=repriced,
            price_error=price_error,
            vega=float(self.pricer.vega(strike, volatility)),
            iterations=iterations,
            price_evaluations=evaluations,
            bracket_upper=float(upper),
            status=status,
        )


def black_time_value(y, w):
    """Normalized OTM price using logarithmic CDFs to reduce cancellation."""
    y, w = np.broadcast_arrays(np.asarray(y, float), np.asarray(w, float))
    result = np.full(y.shape, np.nan)
    zero = np.isfinite(y) & (w == 0)
    result[zero] = 0.0
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
    yy, ww = (y[valid], w[valid])
    s = np.sqrt(ww)
    d1, d2 = (-yy / s + s / 2, -yy / s - s / 2)
    call = yy >= 0
    a = np.where(call, log_ndtr(d1), yy + log_ndtr(-d2))
    b = np.where(call, yy + log_ndtr(d2), log_ndtr(-d1))
    result[valid] = np.exp(a) * -np.expm1(b - a)
    return result


def black_call(y, w):
    return np.maximum(1 - np.exp(y), 0) + black_time_value(y, w)


def invert_time_value(y, time_value):
    """Return w and explicit status. Invalid prices remain invalid, never clipped."""
    y, price = np.broadcast_arrays(np.asarray(y, float), np.asarray(time_value, float))
    w = np.full(y.shape, np.nan)
    status = np.full(y.shape, "nonfinite", dtype="<U32")
    finite = np.isfinite(y) & np.isfinite(price)
    status[finite & (price <= 0)] = "at_or_below_intrinsic"
    upper_price = np.minimum(1.0, np.exp(y))
    status[finite & (price >= upper_price)] = "at_or_above_upper_bound"
    valid = finite & (price > 0) & (price < upper_price)
    if not valid.any():
        return (w, status)
    yy, target = (y[valid], price[valid])
    upper = np.full(yy.shape, 0.04)
    for _ in range(20):
        small = black_time_value(yy, upper) < target
        if not small.any():
            break
        upper[small] *= 4
    bracketed = black_time_value(yy, upper) >= target
    lower = np.zeros_like(upper)
    for _ in range(64):
        middle = (upper + lower) / 2
        small = black_time_value(yy, middle) < target
        lower = np.where(small, middle, lower)
        upper = np.where(small, upper, middle)
    candidate = (upper + lower) / 2
    w[valid] = np.where(bracketed, candidate, np.nan)
    status[valid] = np.where(bracketed, "ready", "inversion_unresolved")
    return (w, status)


class BlackFixedIVBenchmark:
    """Observed-price IV; spot delta holds IV and F/S fixed. No clipping."""

    @staticmethod
    def price(forward, discount, strike, stddev, kind):
        if stddev == 0:
            sign = 1 if kind == "call" else -1
            return discount * max((forward - strike) * sign, 0)
        d1 = np.log(forward / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev
        if kind == "call":
            return discount * (forward * ndtr(d1) - strike * ndtr(d2))
        return discount * (strike * ndtr(-d2) - forward * ndtr(-d1))

    def estimate(self, spot, strike, time, forward, discount, mark, kind):
        values = [spot, strike, time, forward, discount, mark]
        if (
            not np.isfinite(values).all()
            or min(values[:5]) <= 0
            or kind not in {"call", "put"}
        ):
            raise ValueError("Invalid Black benchmark inputs")
        lower = self.price(forward, discount, strike, 0, kind)
        upper = discount * (forward if kind == "call" else strike)
        if not lower < mark < upper:
            return {
                "black_status": "no_finite_positive_iv",
                "black_lower_bound": lower,
                "black_upper_bound": upper,
            }
        std = brentq(
            lambda x: (self.price(forward, discount, strike, x, kind) - mark)
            / (discount * forward),
            0,
            32,
            xtol=1e-13,
            rtol=1e-13,
        )
        d1 = np.log(forward / strike) / std + std / 2
        ratio = discount * forward / spot
        delta = ratio * (ndtr(d1) - (kind == "put"))
        gamma = ratio * np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * spot * std)
        price = self.price(forward, discount, strike, std, kind)
        return {
            "black_status": "ready",
            "black_iv": std / np.sqrt(time),
            "black_price": price,
            "black_delta": delta,
            "black_gamma": gamma,
            "black_repricing_error": price - mark,
        }
