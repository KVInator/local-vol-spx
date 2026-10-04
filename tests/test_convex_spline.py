"""Tests for constrained single-expiry call-price splines."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from black import BlackPricer
from convex_spline import ConvexCallSpline, ConvexSplineCalibrator


class TestConvexSpline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pricer = BlackPricer(
            forward=100.0,
            discount_factor=0.97,
            maturity=0.5,
        )
        cls.strikes = np.linspace(80.0, 120.0, 17)
        offset = cls.strikes - 100.0

        cls.midpoints = (
            10.0 - 0.4 * offset + 0.005 * offset**2
        )
        cls.half_widths = np.full(len(cls.strikes), 0.05)

        cls.calibrator = ConvexSplineCalibrator(
            pricer=cls.pricer,
            n_intervals=6,
        )
        cls.fitted = cls.calibrator.fit(
            cls.strikes,
            cls.midpoints,
            cls.half_widths,
        )

    def test_recovers_independent_quadratic_curve(self):
        grid = np.linspace(80.0, 120.0, 101)
        offset = grid - 100.0
        expected = (
            10.0 - 0.4 * offset + 0.005 * offset**2
        )

        np.testing.assert_allclose(
            self.fitted.price(grid),
            expected,
            rtol=0.0,
            atol=2e-7,
        )
        self.assertLess(
            self.fitted.minimum_multiplier,
            1e-6,
        )
        self.assertTrue(
            self.fitted.shape_checks()["passed"]
        )

    def test_analytical_strike_derivatives(self):
        grid = np.linspace(82.0, 118.0, 25)

        np.testing.assert_allclose(
            self.fitted.strike_slope(grid),
            -0.4 + 0.01 * (grid - 100.0),
            rtol=0.0,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            self.fitted.strike_curvature(grid),
            np.full(len(grid), 0.01),
            rtol=0.0,
            atol=2e-7,
        )

        h = 0.02
        finite_difference = (
            self.fitted.price(grid + h)
            - 2.0 * self.fitted.price(grid)
            + self.fitted.price(grid - h)
        ) / h**2

        np.testing.assert_allclose(
            finite_difference,
            self.fitted.strike_curvature(grid),
            rtol=0.0,
            atol=1e-8,
        )

    def test_inconsistent_quotes_require_band_relaxation(self):
        midpoints = self.midpoints.copy()
        midpoints[8] += 0.5
        original = midpoints.copy()

        fitted = self.calibrator.fit(
            self.strikes,
            midpoints,
            self.half_widths,
        )

        self.assertGreater(
            fitted.minimum_multiplier,
            1.0,
        )
        self.assertLessEqual(
            fitted.max_half_spreads,
            fitted.band_cap + 1e-6,
        )
        self.assertTrue(fitted.shape_checks()["passed"])
        np.testing.assert_array_equal(midpoints, original)

    def test_evaluation_respects_calibrated_domain(self):
        self.assertIsInstance(
            self.fitted.price(100.0),
            float,
        )

        for strike in (79.0, 121.0, np.nan):
            with self.subTest(strike=strike):
                with self.assertRaises(ValueError):
                    self.fitted.price(strike)

    def test_saved_model_preserves_prices_and_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.npz"
            self.fitted.save(path)
            restored = ConvexCallSpline.load(path)

        grid = np.linspace(80.0, 120.0, 51)

        for method in (
            "price",
            "strike_slope",
            "strike_curvature",
        ):
            np.testing.assert_array_equal(
                getattr(restored, method)(grid),
                getattr(self.fitted, method)(grid),
            )

        self.assertEqual(
            restored.metadata(),
            self.fitted.metadata(),
        )

    def test_invalid_calibration_inputs_are_rejected(self):
        duplicate_strikes = self.strikes.copy()
        duplicate_strikes[4] = duplicate_strikes[3]

        invalid_midpoints = self.midpoints.copy()
        invalid_midpoints[4] = np.nan

        invalid_widths = self.half_widths.copy()
        invalid_widths[4] = 0.0

        cases = [
            (
                duplicate_strikes,
                self.midpoints,
                self.half_widths,
            ),
            (
                self.strikes,
                invalid_midpoints,
                self.half_widths,
            ),
            (
                self.strikes,
                self.midpoints,
                invalid_widths,
            ),
        ]

        for strikes, midpoints, half_widths in cases:
            with self.subTest():
                with self.assertRaises(ValueError):
                    self.calibrator.fit(
                        strikes,
                        midpoints,
                        half_widths,
                    )


if __name__ == "__main__":
    unittest.main()