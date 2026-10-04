"""Analytical and independent-curve checks for the short-end extension."""

import unittest

import numpy as np
from scipy.special import ndtr

from black import BlackPricer
from implied_vol import ImpliedVolSolver
from short_end import ShortEndSurface


class AnalyticalBlackSurface:
    """Constant-volatility benchmark with deterministic carry."""

    maturities = np.array([0.1, 0.25])
    min_log_moneyness = -0.25
    max_log_moneyness = 0.25
    sigma = 0.2

    def forward(self, time):
        return float(100.0 * np.exp(0.03 * time))

    def discount_factor(self, time):
        return float(np.exp(-0.04 * time))

    def implied_volatility(self, strikes, time):
        return np.full_like(
            np.asarray(strikes, dtype=float),
            self.sigma,
        )

    def normalized_call(self, z, time, derivative=0):
        z = np.asarray(z, dtype=float)
        root_w = self.sigma * np.sqrt(time)
        d1 = -np.log(z) / root_w + 0.5 * root_w
        d2 = d1 - root_w

        if derivative == 0:
            return ndtr(d1) - z * ndtr(d2)

        if derivative == 1:
            return -ndtr(d2)

        return (
            np.exp(-0.5 * d2**2)
            / np.sqrt(2.0 * np.pi)
            / (z * root_w)
        )

    def normalized_time_derivative(self, z, time, side="right"):
        z = np.asarray(z, dtype=float)
        root_w = self.sigma * np.sqrt(time)
        d1 = -np.log(z) / root_w + 0.5 * root_w

        return (
            np.exp(-0.5 * d1**2)
            / np.sqrt(2.0 * np.pi)
            * self.sigma
            / (2.0 * np.sqrt(time))
        )


class QuadraticAnchorSurface(AnalyticalBlackSurface):
    """An independent polynomial anchor for testing smile derivatives."""

    min_log_moneyness = -0.02
    max_log_moneyness = 0.02

    def normalized_call(self, z, time, derivative=0):
        z = np.asarray(z, dtype=float)
        displacement = z - 1.0

        if derivative == 0:
            return 0.02 - 0.5 * displacement + 3.0 * displacement**2

        if derivative == 1:
            return -0.5 + 6.0 * displacement

        return np.full_like(z, 6.0)

    def implied_volatility(self, strikes, time):
        z = np.asarray(strikes, dtype=float) / self.forward(time)
        prices = self.normalized_call(z, time)
        solver = ImpliedVolSolver(BlackPricer(1.0, 1.0, time))

        values = [
            solver.solve(
                price=float(price),
                strike=float(strike),
                kind="call",
            ).volatility
            for strike, price in zip(z.ravel(), prices.ravel())
        ]

        return np.asarray(values).reshape(z.shape)


class TestShortEndSurface(unittest.TestCase):
    def test_matches_black_prices_carry_and_local_variance(self):
        extension = ShortEndSurface(
            AnalyticalBlackSurface(),
            spot=100.0,
        )
        z = np.array([0.95, 1.0, 1.05])

        for time in (0.01, 0.05, 0.1, 0.2):
            forward = 100.0 * np.exp(0.03 * time)
            discount = np.exp(-0.04 * time)
            strikes = forward * z
            expected = BlackPricer(
                forward, discount, time
            ).price(strikes, 0.2, "call")

            self.assertAlmostEqual(extension.forward(time), forward)
            self.assertAlmostEqual(
                extension.discount_factor(time), discount
            )
            np.testing.assert_allclose(
                extension.call_price(strikes, time),
                expected,
                rtol=1e-11,
                atol=1e-11,
            )
            np.testing.assert_allclose(
                extension.normalized_variance(z, time),
                0.2**2,
                rtol=1e-11,
                atol=1e-12,
            )

    def test_zero_maturity_payoff_and_local_variance_limit(self):
        extension = ShortEndSurface(
            AnalyticalBlackSurface(),
            spot=100.0,
        )
        strikes = np.array([90.0, 100.0, 110.0])

        self.assertEqual(extension.forward(0.0), 100.0)
        self.assertEqual(extension.discount_factor(0.0), 1.0)
        np.testing.assert_array_equal(
            extension.call_price(strikes, 0.0),
            np.array([10.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            extension.normalized_variance(strikes / 100.0, 0.0),
            0.2**2,
            atol=1e-12,
            rtol=1e-11,
        )

    def test_prices_and_strike_derivatives_join_the_anchor(self):
        base = QuadraticAnchorSurface()
        extension = ShortEndSurface(base, spot=100.0)
        z = np.array([0.99, 1.0, 1.01])
        just_before = np.nextafter(extension.first_maturity, 0.0)

        for derivative in (0, 1, 2):
            np.testing.assert_allclose(
                extension.normalized_call(
                    z, just_before, derivative=derivative
                ),
                base.normalized_call(
                    z, extension.first_maturity, derivative=derivative
                ),
                rtol=1e-9,
                atol=1e-10,
            )

    def test_skew_derivatives_and_dupire_variance(self):
        extension = ShortEndSurface(
            QuadraticAnchorSurface(),
            spot=100.0,
        )
        z = np.array([0.99, 1.0, 1.01])
        time = 0.04
        strike_bump = 1e-4
        time_bump = 1e-6

        center = extension.normalized_call(z, time)
        above = extension.normalized_call(z + strike_bump, time)
        below = extension.normalized_call(z - strike_bump, time)

        slope_fd = (above - below) / (2.0 * strike_bump)
        curvature_fd = (
            above - 2.0 * center + below
        ) / strike_bump**2
        maturity_fd = (
            extension.normalized_call(z, time + time_bump)
            - extension.normalized_call(z, time - time_bump)
        ) / (2.0 * time_bump)

        np.testing.assert_allclose(
            extension.normalized_call(z, time, derivative=1),
            slope_fd,
            rtol=0.0,
            atol=5e-7,
        )
        np.testing.assert_allclose(
            extension.normalized_call(z, time, derivative=2),
            curvature_fd,
            rtol=0.0,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            extension.normalized_time_derivative(z, time),
            maturity_fd,
            rtol=0.0,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            extension.normalized_variance(z, time),
            2.0 * maturity_fd / (z**2 * curvature_fd),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_rejects_unsupported_evaluations(self):
        extension = ShortEndSurface(
            AnalyticalBlackSurface(),
            spot=100.0,
        )

        with self.assertRaises(ValueError):
            extension.normalized_call(1.5, 0.05)

        with self.assertRaises(ValueError):
            extension.normalized_call(1.0, 0.3)

        with self.assertRaises(ValueError):
            extension.normalized_call(1.0, 0.0, derivative=2)

        with self.assertRaises(ValueError):
            extension.implied_volatility(100.0, 0.0)


if __name__ == "__main__":
    unittest.main()