import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from black import BlackPricer
from convex_spline import ConvexSplineCalibrator
from vol_surface import NormalizedCallSurface


class TestNormalizedCallSurface(unittest.TestCase):
    @staticmethod
    def make_slice(maturity, normalized_price):
        forward = 100.0 * np.exp(0.03 * maturity)
        discount = np.exp(-0.04 * maturity)
        z = np.linspace(0.8, 1.2, 25)

        pricer = BlackPricer(
            forward=forward,
            discount_factor=discount,
            maturity=maturity,
        )
        return ConvexSplineCalibrator(
            pricer=pricer,
            n_intervals=8,
            smoothing_weight=1e-3,
        ).fit(
            forward * z,
            discount * forward * normalized_price(z),
            np.full(z.size, 0.01 * forward),
        )

    @staticmethod
    def expected_normalized_price(z, maturity):
        return (
            0.12
            + 0.30 * maturity
            - 0.50 * (z - 1.0)
            + 0.20 * (z - 1.0) ** 2
        )

    @classmethod
    def setUpClass(cls):
        cls.splines = tuple(
            cls.make_slice(
                time,
                lambda z, time=time: cls.expected_normalized_price(
                    z, time
                ),
            )
            for time in (0.1, 0.2, 0.3)
        )
        cls.surface = NormalizedCallSurface(
            cls.splines,
            min_log_moneyness=-0.18,
            max_log_moneyness=0.18,
        )

    def test_reproduces_expiry_models(self):
        z = np.linspace(0.85, 1.15, 19)

        for model in self.splines:
            time = model.pricer.maturity
            strikes = model.pricer.forward * z

            np.testing.assert_allclose(
                self.surface.call_price(strikes, time),
                model.price(strikes),
                rtol=0.0,
                atol=1e-9,
            )

    def test_intermediate_prices_and_strike_derivatives(self):
        z = np.array([0.88, 1.0, 1.12])
        times = np.array([0.15, 0.17, 0.25])
        forward = 100.0 * np.exp(0.03 * times)
        discount = np.exp(-0.04 * times)
        strikes = forward * z

        np.testing.assert_allclose(
            self.surface.forward(times),
            forward,
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            self.surface.discount_factor(times),
            discount,
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            self.surface.call_price(strikes, times),
            discount
            * forward
            * self.expected_normalized_price(z, times),
            rtol=0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            self.surface.strike_slope(strikes, times),
            discount * (-0.50 + 0.40 * (z - 1.0)),
            rtol=0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            self.surface.strike_curvature(strikes, times),
            discount / forward * 0.40,
            rtol=0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            self.surface.normalized_time_derivative(z, times),
            0.30,
            rtol=0.0,
            atol=1e-8,
        )

    def test_implied_volatility_reprices_intermediate_calls(self):
        time = 0.17
        z = np.array([0.9, 1.0, 1.1])
        forward = 100.0 * np.exp(0.03 * time)
        discount = np.exp(-0.04 * time)
        strikes = forward * z

        volatility = self.surface.implied_volatility(strikes, time)
        pricer = BlackPricer(forward, discount, time)

        np.testing.assert_allclose(
            pricer.price(strikes, volatility, "call"),
            discount
            * forward
            * self.expected_normalized_price(z, time),
            rtol=0.0,
            atol=1e-8,
        )

    def test_detects_calendar_crossing_between_sample_points(self):
        def base(z):
            return 0.15 - 0.50 * (z - 1.0) + 0.20 * (z - 1.0) ** 2

        def later(z):
            return base(z) + 0.20 * (z - 1.0377) ** 2 - 8e-7

        endpoint_samples = np.exp([-0.18, 0.18])
        self.assertTrue(
            np.all(later(endpoint_samples) > base(endpoint_samples))
        )

        models = (
            self.make_slice(0.1, base),
            self.make_slice(0.2, later),
        )

        with self.assertRaisesRegex(ValueError, "Calendar ordering"):
            NormalizedCallSurface(
                models,
                min_log_moneyness=-0.18,
                max_log_moneyness=0.18,
            )

    def test_one_sided_maturity_derivatives(self):
        later_model = self.make_slice(
            0.3,
            lambda z: self.expected_normalized_price(z, 0.3) + 0.04,
        )
        surface = NormalizedCallSurface(
            (*self.splines[:2], later_model),
            min_log_moneyness=-0.18,
            max_log_moneyness=0.18,
        )

        self.assertAlmostEqual(
            surface.normalized_time_derivative(1.0, 0.2, side="left"),
            0.30,
            places=7,
        )
        self.assertAlmostEqual(
            surface.normalized_time_derivative(1.0, 0.2, side="right"),
            0.70,
            places=7,
        )
        self.assertAlmostEqual(
            surface.normalized_time_derivative(1.0, 0.3),
            0.70,
            places=7,
        )

    def test_rejects_maturity_and_strike_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "maturity"):
            self.surface.forward(0.05)

        with self.assertRaisesRegex(ValueError, "maturity"):
            self.surface.discount_factor(0.31)

        with self.assertRaisesRegex(ValueError, "domain"):
            self.surface.normalized_call(1.5, 0.15)

        with self.assertRaisesRegex(ValueError, "domain"):
            self.surface.call_price(500.0, 0.15)

    def test_saved_manifest_preserves_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            filenames = []

            for index, model in enumerate(self.splines):
                filename = f"slice_{index}.npz"
                model.save(directory / filename)
                filenames.append(filename)

            specification = self.surface.metadata()
            specification["slice_files"] = filenames
            manifest = directory / "call_surface.json"
            manifest.write_text(json.dumps(specification))

            restored = NormalizedCallSurface.load(manifest)
            strikes = np.array([90.0, 100.0, 110.0])

            np.testing.assert_allclose(
                restored.call_price(strikes, 0.17),
                self.surface.call_price(strikes, 0.17),
                rtol=0.0,
                atol=1e-10,
            )
            self.assertTrue(
                all(check["passed"] for check in restored.calendar_checks())
            )


if __name__ == "__main__":
    unittest.main()