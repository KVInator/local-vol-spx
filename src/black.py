"""Forward-based Black pricing for European calls and puts."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf, log_ndtr


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
        strike: ArrayLike,
        volatility: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        strikes = np.asarray(strike, dtype=float)
        volatilities = np.asarray(volatility, dtype=float)

        if np.any(~np.isfinite(strikes)) or np.any(strikes <= 0.0):
            raise ValueError("Strikes must be finite and positive.")

        if (
            np.any(~np.isfinite(volatilities))
            or np.any(volatilities < 0.0)
        ):
            raise ValueError(
                "Volatilities must be finite and nonnegative."
            )

        strikes, volatilities = np.broadcast_arrays(
            strikes,
            volatilities,
        )
        return strikes, volatilities

    @staticmethod
    def _result(value: ArrayLike) -> NumericResult:
        array = np.asarray(value, dtype=float)
        return float(array) if array.ndim == 0 else array

    @staticmethod
    def _log_ratio(
        numerator: NDArray[np.float64],
        denominator: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate log(numerator/denominator), including near equality."""
        result = np.log(numerator) - np.log(denominator)

        nearby = (
            (numerator > 0.5 * denominator)
            & (denominator > 0.5 * numerator)
        )
        result[nearby] = np.log1p(
            (numerator[nearby] - denominator[nearby])
            / denominator[nearby]
        )
        return result

    def price_bounds(
        self,
        strike: ArrayLike,
        kind: OptionKind = "call",
    ) -> tuple[NumericResult, NumericResult]:
        """Return discounted intrinsic value and the price upper bound."""
        sign = self._sign(kind)
        strikes, _ = self._inputs(strike, 0.0)

        lower = self.discount_factor * np.maximum(
            sign * (self.forward - strikes),
            0.0,
        )
        upper = self.discount_factor * (
            np.full_like(strikes, self.forward)
            if kind == "call"
            else strikes
        )

        return self._result(lower), self._result(upper)

    def price(
        self,
        strike: ArrayLike,
        volatility: ArrayLike,
        kind: OptionKind = "call",
    ) -> NumericResult:
        """Return the discounted option price.

        Zero volatility or zero maturity returns discounted intrinsic
        value. Both option types share the same time value.
        """
        sign = self._sign(kind)
        strikes, volatilities = self._inputs(strike, volatility)

        intrinsic = np.maximum(
            sign * (self.forward - strikes),
            0.0,
        )
        time_value = np.zeros_like(strikes)

        if self.maturity == 0.0:
            return self._result(
                self.discount_factor * intrinsic
            )

        positive = volatilities > 0.0

        if np.any(positive):
            active_strikes = strikes[positive]
            small = np.minimum(self.forward, active_strikes)
            large = np.maximum(self.forward, active_strikes)
            log_moneyness = self._log_ratio(large, small)

            # Extremely small standard deviations can imply a time
            # value below floating-point resolution away from ATM.
            with np.errstate(
                over="ignore",
                divide="ignore",
                invalid="ignore",
            ):
                std = (
                    volatilities[positive]
                    * np.sqrt(self.maturity)
                )
                d1 = -log_moneyness / std + 0.5 * std
                d2 = -log_moneyness / std - 0.5 * std

                log_a = log_ndtr(d1)
                log_b = log_ndtr(d2)

            otm_value = np.zeros_like(std)
            finite_a = np.isfinite(log_a)

            log_b_over_a = (
                log_moneyness[finite_a]
                + log_b[finite_a]
                - log_a[finite_a]
            )

            # The ratio is at most one mathematically. Enforce that
            # limit against tiny positive floating-point errors.
            otm_value[finite_a] = (
                small[finite_a]
                * np.exp(log_a[finite_a])
                * -np.expm1(
                    np.minimum(log_b_over_a, 0.0)
                )
            )

            at_money = small == large
            otm_value[at_money] = (
                small[at_money]
                * erf(std[at_money] / SQRT_EIGHT)
            )

            time_value[positive] = otm_value

        return self._result(
            self.discount_factor * (intrinsic + time_value)
        )

    def vega(
        self,
        strike: ArrayLike,
        volatility: ArrayLike,
    ) -> NumericResult:
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
            forwards = np.full_like(
                active_strikes,
                self.forward,
            )
            log_moneyness = self._log_ratio(
                forwards,
                active_strikes,
            )

            with np.errstate(
                over="ignore",
                divide="ignore",
                invalid="ignore",
            ):
                std = volatilities[positive] * root_time
                d1 = log_moneyness / std + 0.5 * std

                # Covers an underflowed standard deviation at ATM.
                d1 = np.where(
                    (std == 0.0)
                    & (active_strikes == self.forward),
                    0.0,
                    d1,
                )
                density = (
                    INV_SQRT_TWO_PI
                    * np.exp(-0.5 * d1**2)
                )

            values[positive] = (
                self.discount_factor
                * self.forward
                * root_time
                * density
            )

        zero_at_money = (
            (volatilities == 0.0)
            & (strikes == self.forward)
        )
        values[zero_at_money] = (
            self.discount_factor
            * self.forward
            * root_time
            * INV_SQRT_TWO_PI
        )

        return self._result(values)