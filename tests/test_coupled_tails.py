"""Independent checks of coupled tail prices and distributions."""

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from coupled_tails import (
    CoupledTailBuilder,
    CoupledTailSlice,
    check_coupled_calendar,
)


class AnalyticalBlackCore:
    def __init__(self, maturity=0.1, volatility=0.2):
        self.pricer = SimpleNamespace(
            forward=100.0,
            discount_factor=0.98,
            maturity=maturity,
        )
        self.volatility = volatility
        self.strike_origin = 100.0 * np.exp(-0.4)
        self.strike_max = 100.0 * np.exp(0.4)

    def _terms(self, strikes):
        strikes = np.asarray(strikes, dtype=float)
        scale = self.volatility * np.sqrt(self.pricer.maturity)
        d1 = (
            np.log(self.pricer.forward / strikes) / scale
            + 0.5 * scale
        )
        return strikes, scale, d1, d1 - scale

    def price(self, strikes):
        strikes, _, d1, d2 = self._terms(strikes)
        return self.pricer.discount_factor * (
            self.pricer.forward * ndtr(d1) - strikes * ndtr(d2)
        )

    def strike_slope(self, strikes):
        _, _, _, d2 = self._terms(strikes)
        return -self.pricer.discount_factor * ndtr(d2)

    def strike_curvature(self, strikes):
        strikes, scale, _, d2 = self._terms(strikes)
        density = np.exp(-0.5 * d2**2) / np.sqrt(2.0 * np.pi)
        return self.pricer.discount_factor * density / (strikes * scale)


class TestCoupledTails(unittest.TestCase):
    def setUp(self):
        self.core = AnalyticalBlackCore()
        self.model = CoupledTailSlice(self.core, -0.09, 0.09)

    def test_scalar_vector_and_preserved_core(self):
        z = np.exp(np.array([-0.3, 0.0, 0.3]))
        calls = self.model.normalized_call(z)
        puts = self.model.normalized_put(z)

        np.testing.assert_allclose(
            calls - puts, 1.0 - z, rtol=0.0, atol=2e-15
        )

        for index, strike in enumerate(z):
            for derivative in (0, 1, 2):
                scalar = self.model.normalized_call(
                    float(strike), derivative
                )
                vector = self.model.normalized_call(
                    z, derivative
                )[index]
                self.assertEqual(np.asarray(scalar).shape, ())
                self.assertAlmostEqual(float(scalar), float(vector), places=14)

            self.assertAlmostEqual(
                float(self.model.normalized_put(float(strike))),
                float(puts[index]),
                places=14,
            )

        interior = np.exp(np.linspace(-0.08, 0.08, 17))
        expected = self.core.price(100.0 * interior) / 98.0
        np.testing.assert_allclose(
            self.model.normalized_call(interior),
            expected,
            rtol=0.0,
            atol=2e-15,
        )

    def test_price_and_slope_join_continuity(self):
        for anchor in (self.model.left, self.model.right):
            z = anchor.normalized_strike
            neighbours = np.array(
                [np.nextafter(z, 0.0), z, np.nextafter(z, np.inf)]
            )

            for derivative in (0, 1):
                values = self.model.normalized_call(
                    neighbours, derivative
                )
                np.testing.assert_allclose(
                    values,
                    np.full(3, values[1]),
                    rtol=0.0,
                    atol=3e-14,
                )

    def test_tail_derivatives_and_density_moments(self):
        for z in np.exp(np.array([-0.25, 0.25])):
            h = 1e-5
            values = self.model.normalized_call(
                np.array([z - h, z, z + h])
            )
            fd_slope = (values[2] - values[0]) / (2.0 * h)
            fd_curvature = (
                values[2] - 2.0 * values[1] + values[0]
            ) / h**2

            np.testing.assert_allclose(
                fd_slope,
                self.model.normalized_call(z, 1),
                rtol=3e-6,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                fd_curvature,
                self.model.normalized_call(z, 2),
                rtol=3e-5,
                atol=1e-7,
            )

        a = self.model.left.normalized_strike
        b = self.model.right.normalized_strike

        def moment(power):
            return sum(
                quad(
                    lambda z: z**power
                    * float(self.model.normalized_call(z, 2)),
                    lower,
                    upper,
                    epsabs=1e-10,
                )[0]
                for lower, upper in ((0.0, a), (a, b), (b, np.inf))
            )

        self.assertAlmostEqual(moment(0), 1.0, places=8)
        self.assertAlmostEqual(moment(1), 1.0, places=8)
        second_moment = moment(2)
        self.assertTrue(np.isfinite(second_moment))
        self.assertGreater(second_moment, 1.0)

    def test_calendar_order_and_failed_surface(self):
        later = CoupledTailSlice(
            AnalyticalBlackCore(maturity=0.2), -0.09, 0.09
        )
        self.assertTrue(
            check_coupled_calendar(self.model, later)["passed"]
        )

        decreasing_variance = CoupledTailSlice(
            AnalyticalBlackCore(maturity=0.2, volatility=0.1),
            -0.09,
            0.09,
        )
        self.assertFalse(
            check_coupled_calendar(
                self.model, decreasing_variance
            )["passed"]
        )

    def test_joint_builder_preserves_admissible_joins(self):
        cores = tuple(
            AnalyticalBlackCore(maturity=time)
            for time in (0.1, 0.2, 0.3)
        )
        models, checks = CoupledTailBuilder().construct(cores)

        self.assertEqual(len(models), 3)
        self.assertTrue(all(check["passed"] for check in checks))
        for model in models:
            self.assertAlmostEqual(model.left_log_moneyness, -0.09)
            self.assertAlmostEqual(model.right_log_moneyness, 0.09)


if __name__ == "__main__":
    unittest.main()