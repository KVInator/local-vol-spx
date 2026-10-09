import unittest
import numpy as np
from pricing import BlackPricer


class TestBlackPricer(unittest.TestCase):

    def test_existing_benchmark(self):
        spot = 100.0
        maturity = 0.5
        rate = 0.05
        dividend_yield = 0.02
        pricer = BlackPricer(
            forward=spot * np.exp((rate - dividend_yield) * maturity),
            discount_factor=np.exp(-rate * maturity),
            maturity=maturity,
        )
        np.testing.assert_allclose(
            pricer.price(100.0, 0.2, "call"), 6.307635, rtol=0.0, atol=1e-06
        )
        np.testing.assert_allclose(
            pricer.price(100.0, 0.2, "put"), 4.833643, rtol=0.0, atol=1e-06
        )
        np.testing.assert_allclose(
            pricer.vega(100.0, 0.2), 27.495794412, rtol=0.0, atol=1e-09
        )

    def test_at_money_values(self):
        pricer = BlackPricer(forward=100.0, discount_factor=1.0, maturity=1.0)
        for kind in ("call", "put"):
            np.testing.assert_allclose(
                pricer.price(100.0, 0.2, kind), 7.965567455405804, rtol=0.0, atol=1e-12
            )
            tiny_volatility = 1e-12
            expected = 100.0 * tiny_volatility / np.sqrt(2.0 * np.pi)
            np.testing.assert_allclose(
                pricer.price(100.0, tiny_volatility, kind),
                expected,
                rtol=1e-12,
                atol=0.0,
            )

    def test_vector_prices_bounds_and_parity(self):
        pricer = BlackPricer(forward=101.5, discount_factor=0.975, maturity=0.5)
        strikes = np.array([60.0, 80.0, 100.0, 120.0, 140.0])[None, :]
        volatilities = np.array([0.0, 0.1, 0.2, 0.6])[:, None]
        calls = pricer.price(strikes, volatilities, "call")
        puts = pricer.price(strikes, volatilities, "put")
        self.assertEqual(calls.shape, (4, 5))
        self.assertEqual(puts.shape, (4, 5))
        parity = np.broadcast_to(
            pricer.discount_factor * (pricer.forward - strikes), calls.shape
        )
        np.testing.assert_allclose(calls - puts, parity, rtol=0.0, atol=2e-14)
        for kind, prices in (("call", calls), ("put", puts)):
            lower, upper = pricer.price_bounds(strikes, kind)
            self.assertTrue(np.all(prices >= lower - 1e-12))
            self.assertTrue(np.all(prices <= upper + 1e-12))

    def test_zero_volatility_and_expiry(self):
        strikes = np.array([80.0, 100.0, 120.0])
        pricer = BlackPricer(forward=100.0, discount_factor=0.97, maturity=0.25)
        np.testing.assert_allclose(
            pricer.price(strikes, 0.0, "call"),
            0.97 * np.array([20.0, 0.0, 0.0]),
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            pricer.price(strikes, 0.0, "put"),
            0.97 * np.array([0.0, 0.0, 20.0]),
            rtol=0.0,
            atol=1e-14,
        )
        expected_vega = np.array([0.0, 0.97 * 100.0 * 0.5 / np.sqrt(2.0 * np.pi), 0.0])
        np.testing.assert_allclose(
            pricer.vega(strikes, 0.0), expected_vega, rtol=0.0, atol=1e-13
        )
        expired = BlackPricer(forward=100.0, discount_factor=1.0, maturity=0.0)
        np.testing.assert_array_equal(
            expired.price(strikes, 0.2, "call"), [20.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(expired.vega(strikes, 0.2), np.zeros(3))

    def test_vega_matches_price_derivative(self):
        pricer = BlackPricer(
            forward=100.0 * np.exp(0.03 * 0.5),
            discount_factor=np.exp(-0.05 * 0.5),
            maturity=0.5,
        )
        strikes = np.array([80.0, 100.0, 120.0])
        volatility = 0.2
        bump = 1e-05
        for kind in ("call", "put"):
            finite_difference = (
                pricer.price(strikes, volatility + bump, kind)
                - pricer.price(strikes, volatility - bump, kind)
            ) / (2.0 * bump)
            np.testing.assert_allclose(
                finite_difference,
                pricer.vega(strikes, volatility),
                rtol=0.0,
                atol=1e-06,
            )


from pricing import ImpliedVolSolver


class TestImpliedVolSolver(unittest.TestCase):

    def setUp(self):
        self.pricer = BlackPricer(
            forward=100.0 * np.exp(0.03 * 0.5),
            discount_factor=np.exp(-0.05 * 0.5),
            maturity=0.5,
        )
        self.solver = ImpliedVolSolver(self.pricer)

    def test_known_at_money_price(self):
        pricer = BlackPricer(forward=100.0, discount_factor=1.0, maturity=1.0)
        solver = ImpliedVolSolver(pricer)
        for kind in ("call", "put"):
            with self.subTest(kind=kind):
                result = solver.solve(price=7.965567455405804, strike=100.0, kind=kind)
                np.testing.assert_allclose(result.volatility, 0.2, rtol=0.0, atol=1e-12)

    def test_volatility_recovery_across_quotes(self):
        for kind in ("call", "put"):
            for strike in (80.0, 100.0, 120.0):
                for volatility in (0.1, 0.2, 0.6):
                    with self.subTest(kind=kind, strike=strike, volatility=volatility):
                        price = self.pricer.price(strike, volatility, kind)
                        result = self.solver.solve(price, strike, kind)
                        self.assertEqual(result.status, "solved")
                        np.testing.assert_allclose(
                            result.volatility, volatility, rtol=0.0, atol=1e-09
                        )
                        self.assertLessEqual(
                            abs(result.price_error), self.solver.price_tolerance
                        )

    def test_bracket_expansion(self):
        solver = ImpliedVolSolver(self.pricer, initial_upper=0.1, max_volatility=3.0)
        price = self.pricer.price(100.0, 1.2, "call")
        result = solver.solve(price, 100.0, "call")
        np.testing.assert_allclose(result.volatility, 1.2, rtol=0.0, atol=1e-10)
        self.assertGreater(result.bracket_upper, solver.initial_upper)

    def test_root_at_bracket_endpoint(self):
        solver = ImpliedVolSolver(self.pricer, initial_upper=0.2)
        price = self.pricer.price(100.0, 0.2, "call")
        result = solver.solve(price, 100.0, "call")
        self.assertEqual(result.volatility, 0.2)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.status, "solved")

    def test_financial_boundaries(self):
        for kind, strike in (("call", 80.0), ("put", 120.0)):
            with self.subTest(kind=kind):
                lower, upper = self.pricer.price_bounds(strike, kind)
                result = self.solver.solve(lower, strike, kind)
                self.assertEqual(result.volatility, 0.0)
                self.assertEqual(result.status, "at_lower_bound")
                self.assertEqual(result.price_error, 0.0)
                for invalid_price in (lower - 0.01, upper, upper + 0.01):
                    with self.assertRaises(ValueError):
                        self.solver.solve(invalid_price, strike, kind)
        expired = ImpliedVolSolver(
            BlackPricer(forward=100.0, discount_factor=1.0, maturity=0.0)
        )
        with self.assertRaises(ValueError):
            expired.solve(0.0, 100.0, "call")

    def test_configured_search_limit(self):
        solver = ImpliedVolSolver(self.pricer, initial_upper=0.05, max_volatility=0.1)
        price = self.pricer.price(100.0, 0.2, "call")
        with self.assertRaisesRegex(RuntimeError, "bracket"):
            solver.solve(price, 100.0, "call")

    def test_quote_sensitivity(self):
        volatility = 0.2
        price = self.pricer.price(150.0, volatility, "call")
        baseline = self.solver.solve(price, 150.0, "call")
        price_bump = 1e-05
        bumped = self.solver.solve(price + price_bump, 150.0, "call")
        observed_sensitivity = (bumped.volatility - baseline.volatility) / price_bump
        np.testing.assert_allclose(
            observed_sensitivity,
            baseline.price_to_vol_sensitivity,
            rtol=0.001,
            atol=0.0,
        )
        atm_price = self.pricer.price(100.0, volatility, "call")
        near_atm = self.solver.solve(atm_price, 100.0, "call")
        self.assertGreater(
            baseline.price_to_vol_sensitivity, near_atm.price_to_vol_sensitivity
        )
