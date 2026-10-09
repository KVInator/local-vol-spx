import unittest
import numpy as np
from scipy.special import ndtr
from pde import ForwardPDESolver


def forward_pde_normalized_black(strikes, integrated_variance):
    """Black call with forward and discount factor equal to one."""
    strikes = np.asarray(strikes, dtype=float)
    variance = float(integrated_variance)
    if variance == 0.0:
        return np.maximum(1.0 - strikes, 0.0)
    standard_deviation = np.sqrt(variance)
    d1 = (-np.log(strikes) + 0.5 * variance) / standard_deviation
    d2 = d1 - standard_deviation
    return ndtr(d1) - strikes * ndtr(d2)


def forward_pde_solve_analytical_case(
    solver, times, integrated_variance, local_variance, time_breaks=()
):
    strikes = solver.normalized_strikes
    endpoint_strikes = strikes[[0, -1]]

    def boundaries(maturity):
        values = forward_pde_normalized_black(
            endpoint_strikes, integrated_variance(maturity)
        )
        return (float(values[0]), float(values[1]))

    return solver.solve(
        maturities=times,
        initial_prices=forward_pde_normalized_black(
            strikes, integrated_variance(times[0])
        ),
        local_variance=local_variance,
        boundary_values=boundaries,
        time_breaks=time_breaks,
    )


class TestForwardPDE(unittest.TestCase):

    def test_constant_volatility_and_spatial_convergence(self):
        errors = []
        for intervals in (100, 200, 400):
            solver = ForwardPDESolver(
                min_log_moneyness=-1.0,
                max_log_moneyness=1.0,
                n_space_intervals=intervals,
                max_time_step=0.0005,
            )
            result = forward_pde_solve_analytical_case(
                solver=solver,
                times=[0.0, 0.5],
                integrated_variance=lambda t: 0.04 * t,
                local_variance=lambda z, t, side: 0.04,
            )
            expected = forward_pde_normalized_black(
                solver.normalized_strikes, 0.04 * 0.5
            )
            central = np.abs(result.log_moneyness) <= 0.25
            errors.append(
                float(
                    np.max(
                        np.abs(result.normalized_calls[-1, central] - expected[central])
                    )
                )
            )
        self.assertLess(errors[1], 0.6 * errors[0])
        self.assertLess(errors[2], 0.6 * errors[1])
        self.assertLess(errors[-1], 5e-05)

    def test_time_varying_variance_recovers_analytical_prices(self):
        solver = ForwardPDESolver(
            min_log_moneyness=-1.0,
            max_log_moneyness=1.0,
            n_space_intervals=400,
            max_time_step=0.001,
        )

        def integrated_variance(t):
            return 0.04 * t + 0.15 * t**2

        result = forward_pde_solve_analytical_case(
            solver=solver,
            times=[0.0, 0.2, 0.5],
            integrated_variance=integrated_variance,
            local_variance=lambda z, t, side: 0.04 + 0.3 * t,
        )
        central = np.abs(result.log_moneyness) <= 0.25
        for index, maturity in enumerate(result.maturities[1:], 1):
            expected = forward_pde_normalized_black(
                solver.normalized_strikes, integrated_variance(maturity)
            )
            np.testing.assert_allclose(
                result.normalized_calls[index, central],
                expected[central],
                rtol=0.0,
                atol=8e-05,
            )

    def test_variance_jump_uses_both_sides_at_time_break(self):
        switch = 0.2
        sides_seen = []
        solver = ForwardPDESolver(
            min_log_moneyness=-1.0,
            max_log_moneyness=1.0,
            n_space_intervals=400,
            max_time_step=0.007,
        )

        def integrated_variance(t):
            return 0.04 * min(t, switch) + 0.09 * max(t - switch, 0.0)

        def local_variance(z, t, side):
            if t == switch:
                sides_seen.append(side)
                return 0.04 if side == "left" else 0.09
            return 0.04 if t < switch else 0.09

        result = forward_pde_solve_analytical_case(
            solver=solver,
            times=[0.0, 0.5],
            integrated_variance=integrated_variance,
            local_variance=local_variance,
            time_breaks=[switch],
        )
        expected = forward_pde_normalized_black(
            solver.normalized_strikes, integrated_variance(0.5)
        )
        central = np.abs(result.log_moneyness) <= 0.25
        np.testing.assert_allclose(
            result.normalized_calls[-1, central],
            expected[central],
            rtol=0.0,
            atol=0.0001,
        )
        self.assertIn("left", sides_seen)
        self.assertIn("right", sides_seen)

    def test_time_refinement_with_smooth_initial_prices(self):
        start = 0.1
        stop = 0.5

        def run(nominal_steps):
            solver = ForwardPDESolver(
                min_log_moneyness=-1.0,
                max_log_moneyness=1.0,
                n_space_intervals=400,
                max_time_step=(stop - start) / nominal_steps,
                rannacher_steps=0,
            )
            return forward_pde_solve_analytical_case(
                solver=solver,
                times=[start, stop],
                integrated_variance=lambda t: 0.04 * t,
                local_variance=lambda z, t, side: 0.04,
            )

        reference = run(1024)
        central = np.abs(reference.log_moneyness) <= 0.25
        errors = []
        for nominal_steps in (8, 16, 32):
            result = run(nominal_steps)
            errors.append(
                float(
                    np.max(
                        np.abs(
                            result.normalized_calls[-1, central]
                            - reference.normalized_calls[-1, central]
                        )
                    )
                )
            )
        self.assertLess(errors[1], 0.6 * errors[0])
        self.assertLess(errors[2], 0.6 * errors[1])

    def test_zero_variance_preserves_initial_payoff(self):
        solver = ForwardPDESolver(
            min_log_moneyness=-1.0,
            max_log_moneyness=1.0,
            n_space_intervals=40,
            max_time_step=0.03,
        )
        result = forward_pde_solve_analytical_case(
            solver=solver,
            times=[0.0, 0.1, 0.2],
            integrated_variance=lambda t: 0.0,
            local_variance=lambda z, t, side: 0.0,
        )
        initial = np.maximum(1.0 - solver.normalized_strikes, 0.0)
        np.testing.assert_array_equal(
            result.normalized_calls, np.repeat(initial[None, :], 3, axis=0)
        )


from dataclasses import dataclass
from scipy.linalg import expm
from pde import AHBackwardPricer
from surface import AHGrid, AndreasenHugeSurface


@dataclass
class BlackSurface:
    spot: float = 100.0
    sigma: float = 0.2
    carry: float = 0.03
    rate: float = 0.05
    min_log_moneyness: float = -1.0
    max_log_moneyness: float = 1.0

    @property
    def maturities(self):
        return np.array([0.1, 0.25, 0.5])

    def forward(self, time):
        if not 0 <= time <= 0.5:
            raise ValueError("Unsupported maturity.")
        return self.spot * np.exp(self.carry * time)

    def discount_factor(self, time):
        return np.exp(-self.rate * time)

    def normalized_variance(self, states, time, side="right"):
        if time <= 0:
            raise AssertionError("Zero-time variance was requested.")
        return np.full_like(states, self.sigma**2)


def ah_backward_pricer_black_values(
    surface, spots, strike, expiry, valuation_time=0, variance=None
):
    spots = np.asarray(spots, float)
    horizon = expiry - valuation_time
    total = surface.sigma**2 * horizon if variance is None else variance
    root = np.sqrt(total)
    forward = spots * np.exp(surface.carry * horizon)
    discount = np.exp(-surface.rate * horizon)
    d1 = np.log(forward / strike) / root + 0.5 * root
    factor = np.exp((surface.carry - surface.rate) * horizon)
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    return (
        discount * (forward * ndtr(d1) - strike * ndtr(d1 - root)),
        factor * ndtr(d1),
        factor * phi / (spots * root),
    )


class TestAHBackwardPricer(unittest.TestCase):

    def test_black_prices_and_greeks_improve_with_spatial_refinement(self):
        surface = BlackSurface()
        spots = np.array([95, 100, 105])
        strike = surface.forward(0.2)
        expected = np.array(
            ah_backward_pricer_black_values(surface, spots, strike, 0.2)
        )
        errors = []
        for n in (200, 400, 800):
            result = AHBackwardPricer(
                surface, space_intervals=n, steps_per_day=32
            ).solve(strike, 0.2)
            actual = result.greeks(spots)
            errors.append(
                np.max(
                    abs(
                        np.array([actual.price, actual.delta, actual.gamma]) - expected
                    ),
                    axis=1,
                )
            )
        errors = np.asarray(errors)
        self.assertTrue(np.all(errors[1:] < errors[:-1]))
        self.assertTrue(np.all(errors[-1] < [0.002, 0.0001, 2e-05]))

    def test_batched_calls_puts_and_nonzero_valuation_time(self):
        surface = BlackSurface()
        pricer = AHBackwardPricer(surface, space_intervals=800)
        strikes = [95, 100, 105]
        spots = np.array([97, 100, 103])
        calls = pricer.solve_many(strikes, 0.2, valuation_time=0.03)
        puts = pricer.solve_many(strikes, 0.2, valuation_time=0.03, kind="put")
        for strike, call, put in zip(strikes, calls, puts):
            expected = ah_backward_pricer_black_values(
                surface, spots, strike, 0.2, 0.03
            )
            actual = call.greeks(spots)
            for value, reference, tolerance in zip(
                [actual.price, actual.delta, actual.gamma],
                expected,
                [0.002, 0.0001, 2e-05],
            ):
                np.testing.assert_allclose(value, reference, rtol=0, atol=tolerance)
            parity = np.exp(-surface.rate * 0.17) * (
                spots * np.exp(surface.carry * 0.17) - strike
            )
            np.testing.assert_allclose(
                call.price(spots) - put.price(spots), parity, atol=3e-05
            )
        single = pricer.solve(100, 0.2, valuation_time=0.03)
        np.testing.assert_array_equal(
            single.normalized_values, calls[1].normalized_values
        )
        self.assertGreater(pricer.last_diagnostics["minimum_coefficient_time_years"], 0)

    def test_time_order_against_independent_matrix_exponentials(self):
        surface = BlackSurface(carry=0, rate=0)

        def variance(states, time, side="right"):
            sign = 1 if time < 0.25 else -1
            return 0.04 * (1 + sign * 0.5 * np.tanh(4 * np.log(states)))

        surface.normalized_variance = variance
        pricer = AHBackwardPricer(surface, space_intervals=60, steps_per_day=16)
        result = pricer.solve(100, 0.5)
        matrices = []
        for time in (0.1, 0.4):
            matrix = np.zeros((len(pricer.z), len(pricer.z)))
            a = 0.5 * variance(pricer.z[1:-1], time)
            rows = np.arange(1, len(pricer.z) - 1)
            matrix[rows, rows - 1] = a * (1 / pricer.h**2 + 1 / (2 * pricer.h))
            matrix[rows, rows] = -2 * a / pricer.h**2
            matrix[rows, rows + 1] = a * (1 / pricer.h**2 - 1 / (2 * pricer.h))
            matrices.append(expm(0.25 * matrix))
        payoff = np.maximum(pricer.z - 1, 0)
        expected = matrices[0] @ (matrices[1] @ payoff)
        wrong = matrices[1] @ (matrices[0] @ payoff)
        self.assertGreater(np.max(abs(expected - wrong)), 1e-05)
        np.testing.assert_allclose(
            result.normalized_values, expected, rtol=0, atol=5e-07
        )

    def test_integrable_singular_variance_with_analytical_total_variance(self):
        surface = BlackSurface(carry=0, rate=0)
        sampled = []

        def variance(states, time, side="right"):
            sampled.append(time)
            self.assertGreater(time, 0)
            return np.full_like(states, 0.04 / np.sqrt(time))

        surface.normalized_variance = variance
        expected = ah_backward_pricer_black_values(
            surface, 100, 100, 0.1, variance=0.08 * np.sqrt(0.1)
        )[0]
        errors = []
        for steps in (8, 16, 32):
            result = AHBackwardPricer(
                surface, space_intervals=1200, steps_per_day=steps, early_time_power=3
            ).solve(100, 0.1)
            errors.append(abs(float(result.price(100)) - expected))
        self.assertTrue(np.all(np.diff(errors) < 0))
        self.assertLess(errors[-1], 0.01)
        self.assertGreater(min(sampled), 0)

    def test_ah_prices_improve_against_original_spot_target(self):
        errors = []
        for n in (400, 800, 1600):
            model = AndreasenHugeSurface(
                AHGrid(1, n),
                100,
                [0.03, 0.1],
                [100, 100],
                [1, 1],
                [np.array([-0.2, 0, 0.2])] * 2,
                [np.log([0.24, 0.18, 0.16]), np.log([0.3, 0.22, 0.19])],
            )
            result = AHBackwardPricer(model, space_intervals=n, steps_per_day=32).solve(
                100, 0.1
            )
            errors.append(
                abs(float(result.price(100)) - float(model.call_price(100, 0.1)))
            )
        self.assertTrue(np.all(np.diff(errors) < 0))
        self.assertLess(errors[-1], 0.01)

    def test_spot_bumps_match_analytical_black_greeks(self):
        surface = BlackSurface()
        result = AHBackwardPricer(surface, space_intervals=800).solve(100, 0.2)
        _, delta, gamma = ah_backward_pricer_black_values(surface, 100, 100, 0.2)
        low, centre, high = result.price([99.9, 100, 100.1])
        self.assertAlmostEqual(float((high - low) / 0.2), float(delta), delta=0.0001)
        self.assertAlmostEqual(
            float((high - 2 * centre + low) / 0.1**2), float(gamma), delta=2e-05
        )

    def test_expiry_payoff_invalid_settings_and_no_extrapolation(self):
        surface = BlackSurface()
        pricer = AHBackwardPricer(surface, space_intervals=100)
        result = pricer.solve(100, 0)
        np.testing.assert_array_equal(result.price([90, 100, 110]), [0, 0, 10])
        self.assertEqual(result.time_steps, 0)
        with self.assertRaises(ValueError):
            result.greeks(100)
        for settings in (
            {"domain_width": 2},
            {"space_intervals": 101},
            {"steps_per_day": 0},
            {"early_time_power": 4},
        ):
            with self.assertRaises(ValueError):
                AHBackwardPricer(surface, **settings)
        for strikes, expiry, kwargs in (
            ([0], 0.1, {}),
            ([[100]], 0.1, {}),
            ([100], -1, {}),
            ([100], 0.1, {"valuation_time": 0.2}),
        ):
            with self.assertRaises(ValueError):
                pricer.solve_many(strikes, expiry, **kwargs)
        with self.assertRaises(ValueError):
            result.price(300)


from dataclasses import replace
import pandas as pd
from surface import AHShortEndVariance
from validation import DailyAHValidator, DailyValidationSettings, quote_metrics


def numerical_validation_model_fixture():
    times = np.array([1, 2]) / 365
    return AndreasenHugeSurface(
        AHGrid(0.5, 200),
        100,
        times,
        np.full(2, 100.0),
        np.ones(2),
        [np.array([-0.1, 0.1])] * 2,
        [np.log([0.2, 0.2])] * 2,
    )


class DailyValidationTests(unittest.TestCase):

    def setUp(self):
        self.model = numerical_validation_model_fixture()

    def test_zero_radius_preserves_positive_time_coefficient(self):
        variant = AHShortEndVariance(self.model, 0)
        z = np.exp(np.array([-0.03, 0, 0.03]))
        for t in [0.25 / 365, 1 / 365, 1.5 / 365]:
            np.testing.assert_array_equal(
                variant.normalized_variance(z, t),
                variant.base.normalized_variance(z, t),
            )

    def test_no_change_at_or_after_first_pillar(self):
        variant = AHShortEndVariance(self.model, 0.01)
        for t, side in [(1 / 365, "left"), (1 / 365, "right"), (1.5 / 365, "right")]:
            np.testing.assert_array_equal(
                variant.normalized_variance([0.98, 1, 1.02], t, side),
                variant.base.normalized_variance([0.98, 1, 1.02], t, side),
            )

    def test_early_smoothing_positive_and_changes_coefficient(self):
        variant = AHShortEndVariance(self.model, 0.01)
        z = variant.native_z
        raw = variant.base.normalized_variance(z, 0.25 / 365)
        smooth = variant.normalized_variance(z, 0.25 / 365)
        self.assertTrue(np.isfinite(smooth).all())
        self.assertTrue((smooth > 0).all())
        self.assertGreater(np.max(abs(smooth - raw)), 1e-06)

    def test_zero_time_and_extrapolation_rejected(self):
        variant = AHShortEndVariance(self.model, 0.01)
        with self.assertRaises(ValueError):
            variant.normalized_variance([1], 0)
        with self.assertRaises(ValueError):
            variant.normalized_variance([10], 0.25 / 365)

    def test_underresolved_native_radius_rejected(self):
        for radius in [-1, np.nan, 0.001]:
            with self.assertRaises(ValueError):
                AHShortEndVariance(self.model, radius)

    def test_wide_domain_preserves_log_spacing(self):
        s = DailyValidationSettings()
        fine, wide = (s.cases()[2], s.cases()[-1])
        self.assertAlmostEqual(2 * fine[1] / fine[2], 2 * wide[1] / wide[2])

    def test_invalid_study_settings_rejected(self):
        for key, value in [
            ("radius", 0),
            ("fine_intervals", 12000),
            ("wide_width", 0.75),
            ("steps_per_day", 0),
        ]:
            with self.assertRaises(ValueError):
                replace(DailyValidationSettings(), **{key: value})

    def test_quote_metrics_use_original_bands(self):
        q = pd.DataFrame(
            {
                "call_mid": [1.0, 2.0],
                "call_half_width": [0.1, 0.2],
                "call_bid": [0.9, 1.8],
                "call_ask": [1.1, 2.2],
            }
        )
        result = quote_metrics(np.array([1.0, 2.4]), q)
        self.assertEqual(result["outside_original_bands"], 1)
        self.assertAlmostEqual(result["rms_half_spreads"], np.sqrt(2))

    def test_end_to_end_study_without_refit(self):
        c = pd.DataFrame(
            {
                "quote_date": ["2023-09-01"] * 2,
                "root": ["UNKNOWN"] * 2,
                "expire_date": ["2023-09-02", "2023-09-03"],
            }
        )
        groups = []
        for expiry, t in zip(c.expire_date, self.model.maturities):
            strikes = np.array([99, 100, 101])
            prices = self.model.call_price(strikes, float(t))
            groups.append(
                pd.DataFrame(
                    {
                        "expire_date": expiry,
                        "strike": strikes,
                        "call_mid": prices,
                        "call_half_width": 0.01,
                        "call_bid": prices - 0.01,
                        "call_ask": prices + 0.01,
                    }
                )
            )
        q = pd.concat(groups, ignore_index=True)
        original = q.copy(deep=True)
        settings = DailyValidationSettings(
            radius=0.01,
            width=0.1,
            wide_width=0.15,
            coarse_intervals=100,
            fine_intervals=200,
            steps_per_day=4,
        )
        result = DailyAHValidator(settings).run(
            self.model, q, c, progress=lambda text: None
        )
        self.assertEqual(len(result["quote_fit"]), 3)
        self.assertEqual(len(result["sensitivity"]), 8)
        self.assertEqual(len(result["quote_prices"]), 2 * len(q))
        self.assertEqual(
            set(result["forward_shapes"].scope), {"full_domain", "report_window"}
        )
        self.assertFalse(result["spot_greeks"].zero_time_coefficient_requested.any())
        self.assertTrue(
            (result["spot_greeks"].minimum_coefficient_time_years > 0).all()
        )
        self.assertLess(
            result["forward_backward"].forward_minus_backward_points.abs().max(), 0.05
        )
        pd.testing.assert_frame_equal(q, original)
