"""Independent checks for power-tail construction."""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from black import BlackPricer
from strike_tails import PowerTailSlice, check_tail_calendar


class AnalyticalBlackCore:
    """Analytical reference curve with known density."""

    def __init__(self, maturity, volatility=0.2):
        self.pricer = BlackPricer(100.0, 0.98, maturity)
        self.volatility = volatility
        self.strike_origin = 100.0 * np.exp(-0.5)
        self.strike_max = 100.0 * np.exp(0.5)

    def price(self, strikes):
        return self.pricer.price(
            strikes, self.volatility, "call"
        )

    def _d2(self, strikes):
        z = np.asarray(strikes, dtype=float) / self.pricer.forward
        width = self.volatility * np.sqrt(self.pricer.maturity)
        return -np.log(z) / width - 0.5 * width

    def strike_slope(self, strikes):
        return -self.pricer.discount_factor * ndtr(
            self._d2(strikes)
        )

    def strike_curvature(self, strikes):
        strikes = np.asarray(strikes, dtype=float)
        width = self.volatility * np.sqrt(self.pricer.maturity)
        density = np.exp(-0.5 * self._d2(strikes)**2) / np.sqrt(
            2.0 * np.pi
        )
        return self.pricer.discount_factor * density / (
            strikes * width
        )


class TestStrikeTails(unittest.TestCase):
    def setUp(self):
        self.core = AnalyticalBlackCore(0.1)
        self.model = PowerTailSlice(self.core, -0.08, 0.08)

    def test_preserves_core_and_matches_join_prices_and_slopes(self):
        z = np.exp([-0.06, 0.0, 0.06])
        normalization = (
            self.core.pricer.forward
            * self.core.pricer.discount_factor
        )
        expected = self.core.price(
            self.core.pricer.forward * z
        ) / normalization

        np.testing.assert_allclose(
            self.model.normalized_call(z),
            expected,
            rtol=0.0,
            atol=1e-14,
        )

        for anchor in (
            self.model.left.normalized_strike,
            self.model.right.normalized_strike,
        ):
            nearby = np.array(
                [
                    np.nextafter(anchor, 0.0),
                    anchor,
                    np.nextafter(anchor, np.inf),
                ]
            )

            for derivative in (0, 1):
                values = self.model.normalized_call(
                    nearby, derivative=derivative
                )
                np.testing.assert_allclose(
                    values,
                    np.full(3, values[1]),
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_scalar_and_vector_queries_agree_in_both_tails_and_core(self):
        strikes = np.exp(np.array([-0.15, 0.0, 0.15]))
        vector_puts = self.model.normalized_put(strikes)

        for index, strike in enumerate(strikes):
            for scalar in (float(strike), np.asarray(strike)):
                scalar_put = self.model.normalized_put(scalar)

                self.assertEqual(np.asarray(scalar_put).shape, ())
                np.testing.assert_allclose(
                    scalar_put,
                    vector_puts[index],
                    rtol=0.0,
                    atol=1e-14,
                )

                for derivative in (0, 1, 2):
                    scalar_call = self.model.normalized_call(
                        scalar, derivative=derivative
                    )
                    vector_call = self.model.normalized_call(
                        strikes, derivative=derivative
                    )

                    self.assertEqual(
                        np.asarray(scalar_call).shape, ()
                    )
                    np.testing.assert_allclose(
                        scalar_call,
                        vector_call[index],
                        rtol=0.0,
                        atol=1e-14,
                    )

                np.testing.assert_allclose(
                    self.model.normalized_call(scalar) - scalar_put,
                    1.0 - strike,
                    rtol=0.0,
                    atol=1e-14,
                )

    def test_density_has_unit_mass_and_unit_normalized_mean(self):
        a = self.model.left.normalized_strike
        b = self.model.right.normalized_strike

        def density(z):
            return float(
                self.model.normalized_call(z, derivative=2)
            )

        intervals = ((0.0, a), (a, b), (b, np.inf))

        mass = sum(
            quad(density, left, right, epsabs=1e-9)[0]
            for left, right in intervals
        )
        mean = sum(
            quad(
                lambda z: z * density(z),
                left,
                right,
                epsabs=1e-9,
            )[0]
            for left, right in intervals
        )

        self.assertAlmostEqual(mass, 1.0, places=8)
        self.assertAlmostEqual(mean, 1.0, places=8)

    def test_tail_derivatives_match_finite_differences(self):
        z = np.exp(np.array([-0.15, 0.15]))
        bump = 1e-5

        base = self.model.normalized_call(z)
        up = self.model.normalized_call(z + bump)
        down = self.model.normalized_call(z - bump)

        slope = (up - down) / (2.0 * bump)
        curvature = (up - 2.0 * base + down) / bump**2

        np.testing.assert_allclose(
            slope,
            self.model.normalized_call(z, derivative=1),
            rtol=3e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            curvature,
            self.model.normalized_call(z, derivative=2),
            rtol=3e-5,
            atol=3e-6,
        )

    def test_calendar_order_for_increasing_black_variance(self):
        # Different joins exercise scalar queries in mixed core/tail regions.
        later = PowerTailSlice(
            AnalyticalBlackCore(0.2), -0.09, 0.09
        )
        checks = check_tail_calendar(self.model, later)

        self.assertTrue(checks["left_tail_ordered"])
        self.assertTrue(checks["right_tail_ordered"])
        self.assertEqual(checks["middle_status"], "passed")
        self.assertTrue(checks["passed"])

    def test_detects_decreasing_variance_calendar_failure(self):
        later = PowerTailSlice(
            AnalyticalBlackCore(0.2, volatility=0.1),
            -0.08,
            0.08,
        )
        checks = check_tail_calendar(self.model, later)

        self.assertFalse(checks["left_tail_ordered"])
        self.assertFalse(checks["right_tail_ordered"])
        self.assertFalse(checks["passed"])

    def test_rejects_unsupported_joins_and_invalid_strikes(self):
        with self.assertRaises(ValueError):
            PowerTailSlice(self.core, -0.6, 0.08)

        with self.assertRaises(ValueError):
            PowerTailSlice(self.core, 0.01, 0.08)

        with self.assertRaises(ValueError):
            self.model.normalized_call([0.0, 1.0])

        with self.assertRaises(ValueError):
            self.model.normalized_call([np.nan])

        with self.assertRaises(ValueError):
            self.model.normalized_call([1.0], derivative=3)


if __name__ == "__main__":
    unittest.main()