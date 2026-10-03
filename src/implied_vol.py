"""Scalar implied-volatility inversion using the public Black pricer."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from black import BlackPricer, OptionKind


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
    initial_upper: float = 0.50
    max_volatility: float = 8.0
    vol_tolerance: float = 1e-12
    price_tolerance: float = 1e-9
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
            raise ValueError(
                "initial_upper cannot exceed max_volatility."
            )

        if (
            not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")

        object.__setattr__(
            self,
            "max_iterations",
            int(self.max_iterations),
        )

    @staticmethod
    def _scalar(value, name: str) -> float:
        array = np.asarray(value, dtype=float)
        if array.ndim != 0 or not np.isfinite(array):
            raise ValueError(f"{name} must be a finite scalar.")
        return float(array)

    def solve(
        self,
        price: float,
        strike: float,
        kind: OptionKind = "call",
    ) -> IVResult:
        """Recover IV for a single quote without modifying its price."""
        target = self._scalar(price, "price")
        strike = self._scalar(strike, "strike")

        if self.pricer.maturity == 0.0:
            raise ValueError(
                "Implied volatility is not identifiable at zero maturity."
            )

        lower_price, upper_price = self.pricer.price_bounds(
            strike,
            kind,
        )

        if target < lower_price:
            raise ValueError(
                f"Price {target:.12g} is below the lower bound "
                f"{lower_price:.12g}."
            )

        if target >= upper_price:
            raise ValueError(
                f"Price {target:.12g} is at or above the upper bound "
                f"{upper_price:.12g}; no finite IV is available."
            )

        evaluations = 0

        def objective(volatility: float) -> float:
            nonlocal evaluations
            evaluations += 1

            residual = (
                float(self.pricer.price(strike, volatility, kind))
                - target
            )
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
                    "Could not bracket IV within the configured "
                    f"max_volatility={self.max_volatility:.12g}."
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
                        "Brent's method did not converge: "
                        f"{information.flag}."
                    )

                iterations = int(information.iterations)

            status = "solved"

        repriced = float(
            self.pricer.price(strike, volatility, kind)
        )
        evaluations += 1
        price_error = repriced - target

        if (
            not np.isfinite(repriced)
            or abs(price_error) > self.price_tolerance
        ):
            raise RuntimeError(
                "The IV estimate failed the repricing check: "
                f"error={price_error:.6e}, "
                f"tolerance={self.price_tolerance:.6e}."
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