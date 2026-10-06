"""Independent numerical and financial checks for AH interpolation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from andreasen_huge import (
    AHGrid,
    AndreasenHugeCalibrator,
    AndreasenHugeSurface,
)
from black import BlackPricer


class TestAndreasenHuge(unittest.TestCase):
    def surface(self, intervals=800, times=(0.2, 0.4), varying=False):
        nodes = np.array([-0.2, 0.0, 0.2])
        vol = [0.24, 0.18, 0.16] if varying else [0.2, 0.2, 0.2]
        return AndreasenHugeSurface(
            AHGrid(1.0, intervals), 100.0, times,
            [100.0] * len(times), [0.98] * len(times),
            [nodes] * len(times), [np.log(vol)] * len(times)
        )

    def test_constant_proxy_matches_analytical_resolvent(self):
        z = np.exp(np.linspace(-0.2, 0.2, 81))
        time, variance = 0.2, 0.2 ** 2
        discriminant = np.sqrt(1.0 + 8.0 / (time * variance))
        left = (1.0 + discriminant) / 2.0
        right = (1.0 - discriminant) / 2.0
        exact = np.where(
            z < 1.0,
            1.0 - z + z ** left / discriminant,
            z ** right / discriminant
        )
        errors = [
            np.max(abs(self.surface(n).normalized_call(z, time) - exact))
            for n in (400, 800, 1600)
        ]
        self.assertLess(errors[-1], 2e-6)
        self.assertGreater(errors[0] / errors[1], 3.0)
        self.assertGreater(errors[1] / errors[2], 3.0)

    def test_time_derivative_matches_independent_price_bump(self):
        model = self.surface(varying=True)
        time, bump = 0.3, 1e-6
        finite_difference = (
            model.node_state(time + bump)["calls"][1:-1]
            - model.node_state(time - bump)["calls"][1:-1]
        ) / (2.0 * bump)
        np.testing.assert_allclose(
            model.node_state(time)["time_derivative"],
            finite_difference,
            atol=2e-9,
            rtol=2e-6
        )

    def test_price_shape_and_calendar_order(self):
        model = self.surface(varying=True)
        previous = np.maximum(1.0 - model.grid.z, 0.0)
        for time in (0.01, 0.1, 0.2, 0.3, 0.4):
            state = model.node_state(time)
            calls = state["calls"]
            slopes = np.diff(calls) / np.diff(model.grid.z)
            self.assertGreaterEqual(np.min(calls - previous), -1e-12)
            self.assertLessEqual(np.max(slopes), 1e-10)
            self.assertGreaterEqual(np.min(slopes), -1.0 - 1e-10)
            self.assertGreaterEqual(np.min(np.diff(slopes)), -1e-10)
            mask = abs(np.log(model.grid.z[1:-1])) < 0.2
            np.testing.assert_allclose(
                model.grid.curvature(calls)[mask],
                state["curvature"][mask],
                atol=1e-8,
                rtol=1e-7
            )
            previous = calls

    def test_dupire_variance_is_not_the_proxy(self):
        model = self.surface()
        state = model.node_state(0.15)
        mask = abs(np.log(model.grid.z[1:-1])) < 0.1
        self.assertTrue(np.all(state["local_variance"][mask] > 0.0))
        self.assertGreater(
            np.max(abs(
                state["local_variance"][mask]
                - state["proxy_variance"][mask]
            )),
            0.001
        )

    def test_recovers_independent_black_prices(self):
        times = np.array([0.2, 0.4])
        strikes = np.linspace(85.0, 115.0, 41)
        quotes = [
            np.column_stack([
                strikes,
                BlackPricer(100.0, 0.98, t).price(strikes, 0.2, "call"),
                np.full(len(strikes), 0.02)
            ])
            for t in times
        ]
        model, reports = AndreasenHugeCalibrator(
            AHGrid(1.0, 1000),
            control_points=31,
            smoothing=0.01
        ).calibrate(
            100.0, times, [100.0] * 2, [0.98] * 2, quotes
        )
        for time, data in zip(times, quotes):
            self.assertLess(
                np.max(abs(
                    model.call_price(data[:, 0], time) - data[:, 1]
                )),
                0.01
            )
        self.assertEqual(len(reports), 2)

    def test_pillar_prices_join_and_derivative_sides_match_bumps(self):
        model = self.surface(varying=True)
        time, bump = 0.2, 2e-7
        left = model.node_state(time, "left")
        right = model.node_state(time, "right")
        np.testing.assert_allclose(
            left["calls"], right["calls"], atol=1e-14
        )
        for side, sign in (("left", -1), ("right", 1)):
            bumped = model.node_state(time + sign * bump)["calls"]
            twice = model.node_state(time + sign * 2.0 * bump)["calls"]
            difference = (
                -3.0 * right["calls"] + 4.0 * bumped - twice
            )[1:-1] / (sign * 2.0 * bump)
            np.testing.assert_allclose(
                model.node_state(time, side)["time_derivative"],
                difference,
                atol=3e-9,
                rtol=2e-6
            )

    def test_saved_model_preserves_prices_and_node_derivatives(self):
        model = self.surface(varying=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = AndreasenHugeSurface.load(path)
            for key, values in model.node_state(0.3).items():
                np.testing.assert_allclose(
                    restored.node_state(0.3)[key],
                    values,
                    rtol=1e-13,
                    atol=1e-13,
                    equal_nan=True
                )

    def test_zero_payoff_and_invalid_evaluation(self):
        model = self.surface()
        np.testing.assert_allclose(
            model.call_price([90.0, 100.0, 110.0], 0.0),
            [10.0, 0.0, 0.0],
            rtol=0.0,
            atol=2e-14
        )
        self.assertTrue(
            np.isnan(model.node_state(0.0)["local_variance"]).all()
        )
        for time in (-0.1, 0.5, np.nan):
            with self.assertRaises(ValueError):
                model.node_state(time)
        with self.assertRaises(ValueError):
            model.normalized_call(10.0, 0.1)

    def test_quote_validation_rejects_bad_spreads_and_duplicates(self):
        calibrator = AndreasenHugeCalibrator(AHGrid(1.0, 200))
        for data in (
            [[90, 11, 0], [100, 3, 0.1], [110, 1, 0.1]],
            [[90, 11, 0.1], [90, 10, 0.1], [110, 1, 0.1]],
        ):
            with self.assertRaises(ValueError):
                calibrator.calibrate(
                    100.0, [0.2], [100.0], [0.98], [data]
                )


if __name__ == "__main__":
    unittest.main()