import tempfile
import unittest
from pathlib import Path
import numpy as np
from surface import AHGrid, AndreasenHugeCalibrator, AndreasenHugeSurface
from pricing import BlackPricer


class TestAndreasenHuge(unittest.TestCase):

    def surface(self, intervals=800, times=(0.2, 0.4), varying=False):
        nodes = np.array([-0.2, 0.0, 0.2])
        vol = [0.24, 0.18, 0.16] if varying else [0.2, 0.2, 0.2]
        return AndreasenHugeSurface(
            AHGrid(1.0, intervals),
            100.0,
            times,
            [100.0] * len(times),
            [0.98] * len(times),
            [nodes] * len(times),
            [np.log(vol)] * len(times),
        )

    def test_constant_proxy_matches_analytical_resolvent(self):
        z = np.exp(np.linspace(-0.2, 0.2, 81))
        time, variance = (0.2, 0.2**2)
        discriminant = np.sqrt(1.0 + 8.0 / (time * variance))
        left = (1.0 + discriminant) / 2.0
        right = (1.0 - discriminant) / 2.0
        exact = np.where(
            z < 1.0, 1.0 - z + z**left / discriminant, z**right / discriminant
        )
        errors = [
            np.max(abs(self.surface(n).normalized_call(z, time) - exact))
            for n in (400, 800, 1600)
        ]
        self.assertLess(errors[-1], 2e-06)
        self.assertGreater(errors[0] / errors[1], 3.0)
        self.assertGreater(errors[1] / errors[2], 3.0)

    def test_time_derivative_matches_independent_price_bump(self):
        model = self.surface(varying=True)
        time, bump = (0.3, 1e-06)
        finite_difference = (
            model.node_state(time + bump)["calls"][1:-1]
            - model.node_state(time - bump)["calls"][1:-1]
        ) / (2.0 * bump)
        np.testing.assert_allclose(
            model.node_state(time)["time_derivative"],
            finite_difference,
            atol=2e-09,
            rtol=2e-06,
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
                atol=1e-08,
                rtol=1e-07,
            )
            previous = calls

    def test_dupire_variance_is_not_the_proxy(self):
        model = self.surface()
        state = model.node_state(0.15)
        mask = abs(np.log(model.grid.z[1:-1])) < 0.1
        self.assertTrue(np.all(state["local_variance"][mask] > 0.0))
        self.assertGreater(
            np.max(abs(state["local_variance"][mask] - state["proxy_variance"][mask])),
            0.001,
        )

    def test_recovers_independent_black_prices(self):
        times = np.array([0.2, 0.4])
        strikes = np.linspace(85.0, 115.0, 41)
        quotes = [
            np.column_stack(
                [
                    strikes,
                    BlackPricer(100.0, 0.98, t).price(strikes, 0.2, "call"),
                    np.full(len(strikes), 0.02),
                ]
            )
            for t in times
        ]
        model, reports = AndreasenHugeCalibrator(
            AHGrid(1.0, 1000), control_points=31, smoothing=0.01
        ).calibrate(100.0, times, [100.0] * 2, [0.98] * 2, quotes)
        for time, data in zip(times, quotes):
            self.assertLess(
                np.max(abs(model.call_price(data[:, 0], time) - data[:, 1])), 0.01
            )
        self.assertEqual(len(reports), 2)

    def test_pillar_prices_join_and_derivative_sides_match_bumps(self):
        model = self.surface(varying=True)
        time, bump = (0.2, 2e-07)
        left = model.node_state(time, "left")
        right = model.node_state(time, "right")
        np.testing.assert_allclose(left["calls"], right["calls"], atol=1e-14)
        for side, sign in (("left", -1), ("right", 1)):
            bumped = model.node_state(time + sign * bump)["calls"]
            twice = model.node_state(time + sign * 2.0 * bump)["calls"]
            difference = (-3.0 * right["calls"] + 4.0 * bumped - twice)[1:-1] / (
                sign * 2.0 * bump
            )
            np.testing.assert_allclose(
                model.node_state(time, side)["time_derivative"],
                difference,
                atol=3e-09,
                rtol=2e-06,
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
                    equal_nan=True,
                )

    def test_zero_payoff_and_invalid_evaluation(self):
        model = self.surface()
        np.testing.assert_allclose(
            model.call_price([90.0, 100.0, 110.0], 0.0),
            [10.0, 0.0, 0.0],
            rtol=0.0,
            atol=2e-14,
        )
        self.assertTrue(np.isnan(model.node_state(0.0)["local_variance"]).all())
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
                calibrator.calibrate(100.0, [0.2], [100.0], [0.98], [data])


from scipy.linalg import solve_banded
from surface import AHLocalVariance, log_positive_solve
from surface import AHGrid, AndreasenHugeSurface
from pde import ForwardPDESolver


def ah_local_vol_make_surface(intervals=800):
    return AndreasenHugeSurface(
        AHGrid(1, intervals),
        100,
        [0.1, 0.2],
        [100, 100],
        [1, 1],
        [np.array([-0.2, 0, 0.2])] * 2,
        [np.log([0.24, 0.18, 0.16]), np.log([0.3, 0.22, 0.19])],
    )


class TestAHLocalVariance(unittest.TestCase):

    def test_log_solve_matches_independent_banded_solve(self):
        rng = np.random.default_rng(42)
        grid = AHGrid(1, 400)
        q = np.exp(rng.uniform(-8, 2, len(grid.z) - 2))
        rhs = np.exp(rng.uniform(-30, 10, len(q)))
        rhs[::7] = 0
        log_rhs = np.full(rhs.shape, -np.inf)
        log_rhs[rhs > 0] = np.log(rhs[rhs > 0])
        for density in (False, True):
            bands = grid.matrix(q, 0.05, density)
            expected = solve_banded((1, 1), bands, rhs)
            np.testing.assert_allclose(
                np.exp(log_positive_solve(bands, log_rhs)),
                expected,
                rtol=2e-10,
                atol=1e-10,
            )

    def test_log_solve_recovers_underflowed_toeplitz_green_function(self):
        n = 2001
        bands = np.zeros((3, n))
        bands[1] = 3
        bands[0, 1:] = -1
        bands[2, :-1] = -1
        rhs = np.zeros(n)
        rhs[n // 2] = 1
        self.assertEqual(solve_banded((1, 1), bands, rhs)[0], 0)
        log_rhs = np.full(n, -np.inf)
        log_rhs[n // 2] = 0
        actual = log_positive_solve(bands, log_rhs)
        alpha = np.arccosh(1.5)

        def log_sinh(x):
            return x + np.log(-np.expm1(-2 * x)) - np.log(2)

        i = np.arange(1, n + 1)
        j = n // 2 + 1
        expected = (
            log_sinh(np.minimum(i, j) * alpha)
            + log_sinh((n + 1 - np.maximum(i, j)) * alpha)
            - log_sinh(alpha)
            - log_sinh((n + 1) * alpha)
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)

    def test_variance_matches_resolved_nodes_and_both_pillar_sides(self):
        model = ah_local_vol_make_surface()
        adapter = AHLocalVariance(model)
        mask = abs(adapter.y) < 0.3
        requests = (
            (0.005, "right"),
            (0.1, "left"),
            (0.1, "right"),
            (0.15, "right"),
            (0.2, "left"),
        )
        for time, side in requests:
            expected = model.node_state(time, side)["local_variance"][mask]
            actual = adapter(model.grid.z[1:-1][mask], time, side)
            np.testing.assert_allclose(actual, expected, rtol=2e-09, atol=1e-10)

    def test_positive_early_wing_variance_when_density_underflows(self):
        model = ah_local_vol_make_surface(2000)
        time = 1e-08
        self.assertTrue(np.any(model.node_state(time)["curvature"] == 0))
        values = AHLocalVariance(model)(np.exp([-0.8, 0, 0.8]), time)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values > 0))

    def test_rejects_zero_time_invalid_side_and_extrapolation(self):
        adapter = AHLocalVariance(ah_local_vol_make_surface())
        requests = (
            (1, 0, "right"),
            (1, 0.3, "right"),
            (1, [0.1], "right"),
            (1, 0.1, "bad"),
            (0, 0.1, "right"),
            (np.exp(1), 0.1, "right"),
        )
        for z, time, side in requests:
            with self.assertRaises(ValueError):
                adapter(z, time, side)

    def test_pde_from_payoff_never_requests_zero_time_variance(self):
        errors = []
        for count in (400, 800, 1600):
            model = ah_local_vol_make_surface(count)
            solver = ForwardPDESolver(
                -0.5,
                0.5,
                n_space_intervals=count,
                max_time_step=0.0002,
                rannacher_steps=2,
            )
            z = solver.normalized_strikes
            result = solver.solve(
                [0, 0.05, 0.1],
                np.maximum(1 - z, 0),
                AHLocalVariance(model),
                lambda t: tuple(model.normalized_call(z[[0, -1]], t)),
            )
            mask = abs(result.log_moneyness) < 0.2
            error = result.normalized_calls[-1] - model.normalized_call(z, 0.1)
            errors.append(np.max(abs(error[mask])))
        self.assertLess(errors[-1], 1e-05)
        self.assertGreater(errors[0] / errors[1], 2.5)
        self.assertGreater(errors[1] / errors[2], 2.5)


from dataclasses import replace
import pandas as pd
from calibration import DailyAHCalibrator, DailyAHSettings


def daily_calibration_fixture():
    times = np.array([14, 35]) / 365
    forwards = 100 * np.exp(0.01 * times)
    discounts = np.exp(-0.05 * times)
    source = AndreasenHugeSurface(
        AHGrid(0.5, 200),
        100,
        times,
        forwards,
        discounts,
        [np.array([-0.1, 0.1])] * 2,
        [np.log([0.2, 0.2])] * 2,
    )
    carry, quotes = ([], [])
    for t, f, d, expiry in zip(
        times, forwards, discounts, ["2023-09-15", "2023-10-06"]
    ):
        meta = {
            "quote_date": "2023-09-01",
            "root": "UNKNOWN",
            "expire_date": expiry,
            "case": "rate_5pct",
            "forward": f,
            "discount_factor": d,
            "carry_ready": True,
            "quote_timestamp_utc": "2023-09-01T20:00:00Z",
            "assumed_fixing_utc": expiry + "T20:00:00Z",
        }
        carry.append(
            {
                **meta,
                "spot": 100,
                "maturity_years": t,
                "annual_rate": 0.05,
                "parity_bands_incompatible": False,
                "fitted_parity_outside_bands": False,
            }
        )
        strikes = np.linspace(96, 104, 9)
        prices = source.call_price(strikes, t)
        for k, price in zip(strikes, prices):
            quotes.append(
                {
                    **meta,
                    "strike": k,
                    "underlying_last": 100,
                    "assumed_maturity_years": t,
                    "source_kind": "call",
                    "source_bid": price - 0.005,
                    "source_ask": price + 0.005,
                    "call_bid": price - 0.005,
                    "call_ask": price + 0.005,
                    "call_mid": price,
                    "call_half_width": 0.005,
                    "preferred_side_missing": k < f,
                    "band_disjoint_from_call_bounds": False,
                    "midpoint_outside_call_bounds": False,
                }
            )
    return (pd.DataFrame(quotes), pd.DataFrame(carry))


class DailyAHTests(unittest.TestCase):

    def setUp(self):
        self.q, self.c = daily_calibration_fixture()
        self.runner = DailyAHCalibrator(
            DailyAHSettings(intervals=200, width=0.5, control_points=5, smoothing=0.1)
        )

    def test_synthetic_prices_and_input_preservation(self):
        original = self.q.copy(deep=True)
        model, tables = self.runner.calibrate(self.q, self.c)
        self.assertEqual(len(model.maturities), 2)
        self.assertLess(
            tables["quote_residuals"].price_residual_points.abs().max(), 1e-05
        )
        self.assertEqual(len(tables["shape_checks"]), 4)
        self.assertEqual(len(tables["conditioning"]), 3)
        pd.testing.assert_frame_equal(self.q, original)

    def test_disjoint_band_retained_without_clipping(self):
        row = self.q.iloc[0]
        lower = row.discount_factor * (row.forward - row.strike)
        self.q.loc[0, ["source_bid", "call_bid"]] = lower - 0.03
        self.q.loc[0, ["source_ask", "call_ask"]] = lower - 0.02
        self.q.loc[0, "call_mid"] = lower - 0.025
        self.q.loc[
            0, ["band_disjoint_from_call_bounds", "midpoint_outside_call_bounds"]
        ] = True
        _, tables = self.runner.calibrate(self.q, self.c)
        fitted = tables["quote_residuals"].iloc[0]
        self.assertAlmostEqual(fitted.call_mid, lower - 0.025)
        self.assertTrue(fitted.outside_original_band)
        self.assertEqual(
            tables["expiry_summary"].band_disjoint_from_call_bounds.sum(), 1
        )

    def test_duplicate_strike_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.runner.calibrate(pd.concat([self.q, self.q.iloc[:1]]), self.c)

    def test_duplicate_carry_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.runner.calibrate(self.q, pd.concat([self.c, self.c.iloc[:1]]))

    def test_unavailable_expiry_not_silently_dropped(self):
        self.c.loc[0, "carry_ready"] = False
        with self.assertRaisesRegex(ValueError, "Unavailable carry"):
            self.runner.calibrate(self.q, self.c)

    def test_mixed_dates_rejected(self):
        future = self.q.iloc[:1].copy()
        future["quote_date"] = "2023-09-05"
        with self.assertRaisesRegex(ValueError, "one date"):
            self.runner.calibrate(pd.concat([self.q, future]), self.c)

    def test_forward_mismatch_rejected(self):
        self.q.loc[0, "forward"] += 1
        with self.assertRaisesRegex(ValueError, "Quote/carry mismatch"):
            self.runner.calibrate(self.q, self.c)

    def test_changed_conversion_rejected(self):
        self.q.loc[0, "call_mid"] += 0.01
        with self.assertRaisesRegex(ValueError, "parity conversion"):
            self.runner.calibrate(self.q, self.c)

    def test_utc_maturity_mismatch_rejected(self):
        self.c["assumed_fixing_utc"] = "2023-10-07T20:00:00Z"
        with self.assertRaisesRegex(ValueError, "mismatch|UTC maturities"):
            self.runner.calibrate(self.q, self.c)

    def test_assumed_rate_mismatch_rejected(self):
        self.c["annual_rate"] = 0.07
        with self.assertRaisesRegex(ValueError, "assumed rate"):
            self.runner.calibrate(self.q, self.c)

    def test_stale_price_bound_flag_rejected(self):
        self.q.loc[0, "band_disjoint_from_call_bounds"] = True
        with self.assertRaisesRegex(ValueError, "call-bound flags"):
            self.runner.calibrate(self.q, self.c)

    def test_invalid_settings_rejected(self):
        for field, value in [
            ("smoothing", -1),
            ("control_points", 2),
            ("max_evaluations", 0),
        ]:
            with self.assertRaises(ValueError):
                replace(DailyAHSettings(), **{field: value})
