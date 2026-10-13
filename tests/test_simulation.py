from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.stats import chi2, poisson
from simulation import (
    KnownModelSettings,
    BlackScholesModel,
    SquareRootCEVModel,
    KnownModelPDE,
    funded_hedge,
    simulate_paths,
    estimate,
    slope_bootstrap,
    quadratic_error_proxy,
    entry_implied_volatility,
    representative_ledger,
)


class KnownModelHedgingTests(unittest.TestCase):

    def setUp(self):
        self.settings = KnownModelSettings(
            paths=64,
            frequencies=(2, 4, 8),
            pde_intervals=256,
            pde_steps_per_day=8,
            bootstrap_replicates=20,
        )
        self.models = (
            BlackScholesModel(100.0, 0.05, 0.2),
            SquareRootCEVModel(100.0, 0.05, 0.2),
        )

    def test_rejects_non_nested_schedules(self):
        for frequencies in ((2, 3, 8), (2, 4, 4), (4, 2, 8), (2, 4), (True, 2, 4)):
            with self.subTest(frequencies=frequencies), self.assertRaises(ValueError):
                replace(self.settings, frequencies=frequencies)

    def test_rejects_invalid_synthetic_settings(self):
        for value in (
            {"paths": 1},
            {"sigma": 0},
            {"rate": np.nan},
            {"hedge_fee_bps": -1},
            {"strike_ratios": (1.0, 1.0)},
            {"pde_intervals": 15},
            {"maturity_days": 0},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.settings, **value)

    def test_loads_declared_settings_without_running(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"paths": 100, "frequencies": [2, 4, 8]}))
            settings = KnownModelSettings.from_json(path)
            self.assertEqual(settings.paths, 100)
            self.assertEqual(settings.frequencies, (2, 4, 8))
            self.assertFalse(settings.plan()["historical_inputs_used"])
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_cev_price_matches_independent_poisson_chi_square_tail_sum(self):
        model = SquareRootCEVModel(100.0, 0.03, 0.8)
        tau, strike, spot = (0.5, 105.0, 95.0)
        c = model.scale(tau)
        lam = spot * np.exp(model.rate * tau) / c
        upper = int(poisson.ppf(1 - 1e-14, lam / 2))
        counts = np.arange(1, upper + 1)
        z = strike / c
        expectation = np.sum(
            poisson.pmf(counts, lam / 2)
            * (
                c * 2 * counts * chi2.sf(z, 2 * counts + 2)
                - strike * chi2.sf(z, 2 * counts)
            )
        )
        independent = np.exp(-model.rate * tau) * expectation
        self.assertAlmostEqual(
            float(model.greeks(spot, strike, tau)[0]), independent, places=10
        )

    def test_analytical_delta_and_gamma_against_price_and_delta_bumps(self):
        for model in self.models:
            for tau in (0.01, 0.2):
                for kind in ("call", "put"):
                    s = np.array([90.0, 100.0, 110.0])
                    h = 0.002
                    price, delta, gamma = model.greeks(s, 100.0, tau, kind)
                    plus, minus = (
                        model.greeks(s + h, 100.0, tau, kind),
                        model.greeks(s - h, 100.0, tau, kind),
                    )
                    np.testing.assert_allclose(
                        delta, (plus[0] - minus[0]) / (2 * h), atol=5e-08
                    )
                    np.testing.assert_allclose(
                        gamma, (plus[1] - minus[1]) / (2 * h), atol=2e-08
                    )

    def test_put_call_price_delta_gamma_parity(self):
        for model in self.models:
            spot = np.array([0.0, 80.0, 100.0, 125.0])
            call, put = (
                model.greeks(spot, 100.0, 0.3),
                model.greeks(spot, 100.0, 0.3, "put"),
            )
            np.testing.assert_allclose(
                call[0] - put[0], spot - 100 * np.exp(-0.05 * 0.3), atol=3e-11
            )
            np.testing.assert_allclose(call[1] - put[1], 1.0, atol=1e-15)
            np.testing.assert_allclose(call[2], put[2], atol=1e-15)

    def test_cev_zero_rate_limit(self):
        zero = SquareRootCEVModel(100.0, 0.0, 0.2)
        near = SquareRootCEVModel(100.0, 1e-10, 0.2)
        np.testing.assert_allclose(
            zero.greeks(np.array([90.0, 100.0, 110.0]), 100.0, 0.2),
            near.greeks(np.array([90.0, 100.0, 110.0]), 100.0, 0.2),
            atol=3e-09,
        )

    def test_cev_fixed_reference_coefficient_under_bump(self):
        model = self.models[1]
        self.assertEqual(model.k, 2.0)
        self.assertAlmostEqual(float(model.diffusion_squared(90.0)), 360.0)
        self.assertNotAlmostEqual(
            float(model.greeks(90.0, 100.0, 0.2)[1]),
            float(SquareRootCEVModel(90.0, 0.05, 0.2).greeks(90.0, 100.0, 0.2)[1]),
            places=4,
        )

    def test_exact_cev_zero_atom_and_absorption(self):
        model = SquareRootCEVModel(1.0, 0.0, 2.0)
        states = np.ones(30000)
        dt = 1.0
        result = model.step(states, dt, np.random.default_rng(999))
        expected = np.exp(-0.5 / model.scale(dt))
        fraction = np.mean(result == 0)
        self.assertLess(
            abs(fraction - expected),
            6 * np.sqrt(expected * (1 - expected) / len(states)),
        )
        np.testing.assert_array_equal(
            model.step(np.zeros(100), dt, np.random.default_rng(1)), np.zeros(100)
        )

    def test_exact_transitions_match_theoretical_conditional_moments(self):
        for model in self.models:
            rng = np.random.default_rng(912)
            count = 60000
            dt = 0.2
            observed = (
                model.step(np.full(count, 100.0), dt, rng) * np.exp(-model.rate * dt)
                - 100.0
            )
            variance, fourth_variance = model.increment_moments(100.0, 0.0, dt)
            self.assertLess(abs(observed.mean()), 6 * np.sqrt(variance / count))
            self.assertLess(
                abs(np.mean(observed**2) - variance),
                6 * np.sqrt(fourth_variance / count),
            )

    def test_same_seed_reproduces_nested_paths(self):
        for model in self.models:
            t, x = simulate_paths(model, self.settings, np.random.default_rng(17))
            t2, x2 = simulate_paths(model, self.settings, np.random.default_rng(17))
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(t, t2)
            self.assertEqual(x[:, ::4].shape, (64, 3))

    def test_transition_rejects_negative_cev_state(self):
        with self.assertRaises(ValueError):
            self.models[1].step(np.array([-1.0]), 0.1, np.random.default_rng(1))

    def test_pde_price_and_delta_approach_independent_analytical_model(self):
        settings = replace(self.settings, pde_intervals=800, pde_steps_per_day=32)
        spots = np.array([95.0, 100.0, 105.0])
        for model in self.models:
            coarse = KnownModelPDE(model, 100.0, settings, intervals=400)
            fine = KnownModelPDE(model, 100.0, settings)
            for index in (0, len(fine.times) // 2, len(fine.times) - 2):
                exact = model.greeks(spots, 100.0, settings.horizon - fine.times[index])
                a, b = (coarse.greeks(index, spots), fine.greeks(index, spots))
                self.assertLess(np.max(np.abs(b[0] - exact[0])), 0.005)
                self.assertLess(np.max(np.abs(b[1] - exact[1])), 0.0003)
                self.assertLess(
                    np.max(np.abs(b[0] - exact[0])), np.max(np.abs(a[0] - exact[0]))
                )

    def test_pde_restores_calendar_time_forward_scale_for_positive_and_negative_rates(
        self,
    ):
        settings = replace(
            self.settings,
            frequencies=(64, 128, 256),
            pde_intervals=1600,
            pde_steps_per_day=32,
        )
        spots = np.array([95.0, 100.0, 105.0])
        for model_type in (BlackScholesModel, SquareRootCEVModel):
            for rate in (0.05, -0.02):
                model = model_type(100.0, rate, 0.2)
                grid = KnownModelPDE(model, 100.0, settings)
                for index in (128, 255):
                    tau = settings.horizon - grid.times[index]
                    call = grid.greeks(index, spots)
                    put = grid.greeks(index, spots, "put")
                    exact = model.greeks(spots, 100.0, tau)
                    np.testing.assert_allclose(call[0], exact[0], atol=0.001, rtol=0)
                    np.testing.assert_allclose(call[1], exact[1], atol=0.0005, rtol=0)
                    np.testing.assert_allclose(
                        call[0] - put[0], spots - 100 * np.exp(-rate * tau), atol=1e-12
                    )
                    np.testing.assert_allclose(call[1] - put[1], 1.0, atol=1e-12)

    def test_pde_no_outside_domain_extrapolation(self):
        model = self.models[0]
        grid = KnownModelPDE(model, 100.0, self.settings)
        self.assertTrue(np.isnan(grid.greeks(0, 10000.0)[1]))
        self.assertEqual(float(grid.greeks(0, 0.0)[0]), 0.0)

    def test_true_initial_price_matches_entry_iv_control(self):
        for model in self.models:
            for strike in (95.0, 100.0, 105.0):
                iv, price = entry_implied_volatility(model, strike, 0.2)
                equivalent = BlackScholesModel(model.spot, model.rate, iv)
                self.assertAlmostEqual(
                    float(equivalent.greeks(100.0, strike, 0.2)[0]), price, places=10
                )

    def test_manual_cash_path(self):
        result = funded_hedge(
            np.array([[100.0, 110.0, 105.0]]),
            np.array([0.0, 0.1, 0.2]),
            np.array([[0.5, 0.25]]),
            5.0,
            100.0,
            "call",
            0.0,
        )
        self.assertAlmostEqual(result["net_pnl"][0], 3.75)

    def test_linear_claim_replication_at_any_frequency(self):
        s = np.array([[100.0, 110.0, 107.0, 115.0], [100.0, 98.0, 99.0, 94.0]])
        times = np.array([0.0, 0.03, 0.07, 0.2])
        premium = 100 - 50 * np.exp(-0.05 * 0.2)
        result = funded_hedge(s, times, np.ones((2, 3)), premium, 50.0, "call", 0.05)
        np.testing.assert_allclose(result["net_pnl"], 0.0, atol=5e-13)

    def test_funded_fee_identity_with_negative_rate(self):
        s = np.array([[100.0, 110.0, 105.0], [100.0, 95.0, 90.0]])
        times = np.array([0.0, 0.1, 0.2])
        delta = np.array([[0.5, 0.25], [0.5, 0.6]])
        free = funded_hedge(s, times, delta, 5.0, 100.0, "call", -0.01)
        fee = funded_hedge(s, times, delta, 5.0, 100.0, "call", -0.01, 2.0)
        np.testing.assert_allclose(
            fee["net_pnl"] - free["net_pnl"], -fee["funded_costs"], atol=3e-13
        )
        self.assertTrue(np.all(fee["direct_costs"] > 0))

    def test_existing_scalar_ledger_matches_batch_cash(self):
        for model in self.models:
            times, spots = simulate_paths(
                model, self.settings, np.random.default_rng(18)
            )
            for kind in ("call", "put"):
                delta = np.array(
                    [
                        float(model.greeks(s, 100.0, times[-1] - t, kind)[1])
                        for s, t in zip(spots[0, :-1], times[:-1])
                    ]
                )
                premium = float(model.greeks(100.0, 100.0, times[-1], kind)[0])
                expected = funded_hedge(
                    spots[:1], times, delta[None, :], premium, 100.0, kind, 0.05, 1.0
                )["net_pnl"][0]
                ledger, row = representative_ledger(
                    model,
                    100.0,
                    kind,
                    spots[0],
                    times,
                    delta,
                    premium,
                    0.05,
                    1.0,
                    expected,
                    0,
                )
                self.assertEqual(row["status"], "verified")
                self.assertLess(abs(row["difference"]), 1e-10)
                self.assertEqual(ledger.iloc[-1].option_position, 0.0)

    def test_failed_pde_decision_is_retained_without_fallback(self):
        spots = np.array([[100.0, 105.0, 110.0], [100.0, 95.0, 90.0]])
        delta = np.array([[0.5, np.nan], [0.5, 0.3]])
        result = funded_hedge(
            spots, np.array([0.0, 0.1, 0.2]), delta, 5.0, 100.0, "call", 0.05
        )
        self.assertFalse(result["valid"][0])
        self.assertTrue(np.isnan(result["net_pnl"][0]))
        self.assertTrue(result["valid"][1])

    def test_call_put_hedge_error_identity(self):
        model = self.models[1]
        times, spots = simulate_paths(model, self.settings, np.random.default_rng(6))
        delta = np.column_stack(
            [
                model.delta_gamma(spots[:, i], 100.0, times[-1] - t)[0]
                for i, t in enumerate(times[:-1])
            ]
        )
        premium = float(model.greeks(100.0, 100.0, times[-1])[0])
        call = funded_hedge(spots, times, delta, premium, 100.0, "call", 0.05)
        put = funded_hedge(
            spots,
            times,
            delta - 1.0,
            premium - 100.0 + 100.0 * np.exp(-0.05 * times[-1]),
            100.0,
            "put",
            0.05,
        )
        np.testing.assert_allclose(call["net_pnl"], put["net_pnl"], atol=4e-12)

    def test_quadratic_proxy_exact_variance_for_one_interval(self):
        model = self.models[0]
        settings = replace(self.settings, paths=40000)
        rng = np.random.default_rng(22)
        next_spot = model.step(np.full(settings.paths, 100.0), 0.1, rng)
        spots = np.column_stack([np.full(settings.paths, 100.0), next_spot])
        gamma = np.full((settings.paths, 1), 0.02)
        proxy, qv = quadratic_error_proxy(model, spots, np.array([0.0, 0.1]), gamma)
        difference = estimate(proxy * proxy - qv)
        self.assertLess(abs(difference["mean"]), 6 * difference["se"])

    def test_paired_path_bootstrap_preserves_frequency_scaling(self):
        errors = np.arange(1.0, 21.0)[:, None] / np.sqrt(
            np.array([16.0, 32.0, 64.0, 128.0])
        )
        row = slope_bootstrap(errors, [16, 32, 64, 128], 30, np.random.default_rng(1))
        self.assertAlmostEqual(row["slope"], -0.5, places=14)
        self.assertAlmostEqual(row["slope_ci_low"], -0.5, places=14)
        self.assertAlmostEqual(row["slope_ci_high"], -0.5, places=14)

    def test_estimate_uses_independent_sample_count(self):
        row = estimate([1.0, 2.0, np.nan, 3.0])
        self.assertEqual(row["samples"], 3)
        self.assertEqual(row["mean"], 2.0)
        self.assertAlmostEqual(row["se"], 1 / np.sqrt(3))
