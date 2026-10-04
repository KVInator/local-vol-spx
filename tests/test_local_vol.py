"""Independent analytical benchmarks for Dupire extraction."""

import unittest
from unittest.mock import patch

import numpy as np

from local_vol import DupireLocalVolatility


class AnalyticalBlackSurface:
    """Analytical normalized Black derivatives for a variance term structure."""

    def __init__(self, total_variance, variance_rate):
        self.total_variance = total_variance
        self.variance_rate = variance_rate

    @staticmethod
    def _normal_density(value):
        return np.exp(-0.5 * value * value) / np.sqrt(2.0 * np.pi)

    def _terms(self, normalized_strike, maturity):
        z, t = np.broadcast_arrays(
            np.asarray(normalized_strike, dtype=float),
            np.asarray(maturity, dtype=float),
        )
        root_variance = np.sqrt(self.total_variance(t))
        d1 = -np.log(z) / root_variance + 0.5 * root_variance
        d2 = d1 - root_variance
        return z, t, root_variance, d1, d2

    def forward(self, maturity):
        return 100.0 * np.exp(0.03 * np.asarray(maturity, dtype=float))

    def normalized_call(
        self,
        normalized_strike,
        maturity,
        derivative=0,
    ):
        if derivative != 2:
            raise ValueError("This benchmark supplies the second derivative.")

        z, _, root_variance, _, d2 = self._terms(
            normalized_strike,
            maturity,
        )
        return self._normal_density(d2) / (z * root_variance)

    def normalized_time_derivative(
        self,
        normalized_strike,
        maturity,
        side="right",
    ):
        _, t, root_variance, d1, _ = self._terms(
            normalized_strike,
            maturity,
        )
        return (
            self._normal_density(d1)
            * self.variance_rate(t, side)
            / (2.0 * root_variance)
        )


class TestDupireLocalVolatility(unittest.TestCase):
    def test_recovers_constant_black_volatility(self):
        z = np.exp(np.array([-0.08, 0.0, 0.08]))[:, None]
        maturity = np.array([0.02, 0.10, 0.50])[None, :]

        for sigma in (0.10, 0.20, 0.50):
            with self.subTest(sigma=sigma):
                surface = AnalyticalBlackSurface(
                    total_variance=lambda t: sigma**2 * t,
                    variance_rate=lambda t, side: np.full_like(t, sigma**2),
                )
                model = DupireLocalVolatility(surface)

                np.testing.assert_allclose(
                    model.normalized_volatility(z, maturity),
                    sigma,
                    rtol=1e-12,
                    atol=1e-14,
                )

    def test_recovers_instantaneous_variance_term_structure(self):
        surface = AnalyticalBlackSurface(
            total_variance=lambda t: 0.04 * t + 0.15 * t**2,
            variance_rate=lambda t, side: 0.04 + 0.30 * t,
        )
        model = DupireLocalVolatility(surface)

        maturity = np.array([0.05, 0.20, 0.60])
        z = np.exp(np.array([-0.10, 0.0, 0.10]))
        strikes = surface.forward(maturity) * z

        np.testing.assert_allclose(
            model.variance(strikes, maturity),
            0.04 + 0.30 * maturity,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_one_sided_values_at_variance_rate_change(self):
        pillar = 0.10

        def total_variance(t):
            return (
                0.04 * np.minimum(t, pillar)
                + 0.09 * np.maximum(t - pillar, 0.0)
            )

        def variance_rate(t, side):
            at_pillar = 0.04 if side == "left" else 0.09
            return np.where(
                t < pillar,
                0.04,
                np.where(t > pillar, 0.09, at_pillar),
            )

        model = DupireLocalVolatility(
            AnalyticalBlackSurface(total_variance, variance_rate)
        )

        self.assertAlmostEqual(
            model.normalized_volatility(1.0, pillar, side="left"),
            0.20,
        )
        self.assertAlmostEqual(
            model.normalized_volatility(1.0, pillar, side="right"),
            0.30,
        )

    def test_zero_time_derivative_gives_zero_local_variance(self):
        surface = AnalyticalBlackSurface(
            total_variance=lambda t: 0.04 * t,
            variance_rate=lambda t, side: np.zeros_like(t),
        )

        self.assertEqual(
            DupireLocalVolatility(surface).normalized_variance(1.0, 0.10),
            0.0,
        )

    def test_invalid_derivatives_are_rejected(self):
        surface = AnalyticalBlackSurface(
            total_variance=lambda t: 0.04 * t,
            variance_rate=lambda t, side: np.full_like(t, 0.04),
        )
        model = DupireLocalVolatility(surface)

        for curvature in (0.0, -1.0, np.nan):
            with self.subTest(curvature=curvature):
                with patch.object(
                    surface,
                    "normalized_call",
                    return_value=curvature,
                ):
                    with self.assertRaises(ValueError):
                        model.normalized_variance(1.0, 0.10)

        with patch.object(
            surface,
            "normalized_time_derivative",
            return_value=-0.01,
        ):
            with self.assertRaises(ValueError):
                model.normalized_variance(1.0, 0.10)


if __name__ == "__main__":
    unittest.main()