"""Dupire local volatility from normalized call-price derivatives."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


ScalarOrArray = float | np.ndarray


class NormalizedCallDerivatives(Protocol):
    """Surface interface required by the Dupire calculation."""

    def forward(self, maturity: ArrayLike) -> ScalarOrArray:
        ...

    def normalized_call(
        self,
        normalized_strike: ArrayLike,
        maturity: ArrayLike,
        derivative: int = 0,
    ) -> ScalarOrArray:
        ...

    def normalized_time_derivative(
        self,
        normalized_strike: ArrayLike,
        maturity: ArrayLike,
        side: str = "right",
    ) -> ScalarOrArray:
        ...


@dataclass(frozen=True)
class DupireLocalVolatility:
    """Evaluate annualized local variance and local volatility.

    The source surface controls its supported strike and maturity domain.
    At internal expiry pillars, ``side`` selects the maturity derivative.
    """

    surface: NormalizedCallDerivatives

    @staticmethod
    def _restore(values: np.ndarray) -> ScalarOrArray:
        return float(values) if values.ndim == 0 else values

    def normalized_variance(
        self,
        normalized_strike: ArrayLike,
        maturity: ArrayLike,
        side: str = "right",
    ) -> ScalarOrArray:
        if side not in {"left", "right"}:
            raise ValueError("Side must be 'left' or 'right'.")

        z, t = np.broadcast_arrays(
            np.asarray(normalized_strike, dtype=float),
            np.asarray(maturity, dtype=float),
        )

        if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
            raise ValueError("Normalized strikes must be finite and positive.")

        if np.any(~np.isfinite(t)) or np.any(t <= 0.0):
            raise ValueError("Maturities must be finite and positive.")

        curvature = np.asarray(
            self.surface.normalized_call(z, t, derivative=2),
            dtype=float,
        )
        time_derivative = np.asarray(
            self.surface.normalized_time_derivative(z, t, side=side),
            dtype=float,
        )

        if np.any(~np.isfinite(curvature)) or np.any(curvature <= 0.0):
            raise ValueError(
                "Dupire requires finite, strictly positive strike curvature."
            )

        if np.any(~np.isfinite(time_derivative)):
            raise ValueError("Maturity derivatives must be finite.")

        if np.any(time_derivative < 0.0):
            raise ValueError(
                "Negative maturity derivatives imply negative local variance."
            )

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            variance = (
                2.0 * time_derivative
                / (z * z * curvature)
            )

        if np.any(~np.isfinite(variance)):
            raise ValueError("The derivative ratio produced nonfinite variance.")

        return self._restore(variance)

    def normalized_volatility(
        self,
        normalized_strike: ArrayLike,
        maturity: ArrayLike,
        side: str = "right",
    ) -> ScalarOrArray:
        variance = np.asarray(
            self.normalized_variance(normalized_strike, maturity, side),
            dtype=float,
        )
        return self._restore(np.sqrt(variance))

    def variance(
        self,
        strike: ArrayLike,
        maturity: ArrayLike,
        side: str = "right",
    ) -> ScalarOrArray:
        """Evaluate local variance at the underlying state represented by strike."""
        k, t = np.broadcast_arrays(
            np.asarray(strike, dtype=float),
            np.asarray(maturity, dtype=float),
        )
        forward = np.asarray(self.surface.forward(t), dtype=float)
        return self.normalized_variance(k / forward, t, side)

    def volatility(
        self,
        strike: ArrayLike,
        maturity: ArrayLike,
        side: str = "right",
    ) -> ScalarOrArray:
        """Return annualized volatility in decimal units."""
        variance = np.asarray(
            self.variance(strike, maturity, side),
            dtype=float,
        )
        return self._restore(np.sqrt(variance))