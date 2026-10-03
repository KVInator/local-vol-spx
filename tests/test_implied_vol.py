"""Checks for IV recovery, financial boundaries and conditioning."""

import unittest

import numpy as np

from black import BlackPricer
from implied_vol import ImpliedVolSolver


class TestImpliedVolSolver(unittest.TestCase):
    def setUp(self):
        self.pricer = BlackPricer(
            forward=100.0 * np.exp(0.03 * 0.5),
            discount_factor=np.exp(-0.05 * 0.5),
            maturity=0.5,
        )
        self.solver = ImpliedVolSolver(self.pricer)

    def test_known_at_money_price(self):
        pricer = BlackPricer(
            forward=100.0,
            discount_factor=1.0,
            maturity=1.0,
        )
        solver = ImpliedVolSolver(pricer)

        for kind in ("call", "put"):
            with self.subTest(kind=kind):
                result = solver.solve(
                    price=7.965567455405804,
                    strike=100.0,
                    kind=kind,
                )
                np.testing.assert_allclose(
                    result.volatility,
                    0.20,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_volatility_recovery_across_quotes(self):
        for kind in ("call", "put"):
            for strike in (80.0, 100.0, 120.0):
                for volatility in (0.10, 0.20, 0.60):
                    with self.subTest(
                        kind=kind,
                        strike=strike,
                        volatility=volatility,
                    ):
                        price = self.pricer.price(
                            strike,
                            volatility,
                            kind,
                        )
                        result = self.solver.solve(
                            price,
                            strike,
                            kind,
                        )

                        self.assertEqual(result.status, "solved")
                        np.testing.assert_allclose(
                            result.volatility,
                            volatility,
                            rtol=0.0,
                            atol=1e-9,
                        )
                        self.assertLessEqual(
                            abs(result.price_error),
                            self.solver.price_tolerance,
                        )

    def test_bracket_expansion(self):
        solver = ImpliedVolSolver(
            self.pricer,
            initial_upper=0.10,
            max_volatility=3.0,
        )
        price = self.pricer.price(100.0, 1.20, "call")
        result = solver.solve(price, 100.0, "call")

        np.testing.assert_allclose(
            result.volatility,
            1.20,
            rtol=0.0,
            atol=1e-10,
        )
        self.assertGreater(
            result.bracket_upper,
            solver.initial_upper,
        )

    def test_root_at_bracket_endpoint(self):
        solver = ImpliedVolSolver(
            self.pricer,
            initial_upper=0.20,
        )
        price = self.pricer.price(100.0, 0.20, "call")
        result = solver.solve(price, 100.0, "call")

        self.assertEqual(result.volatility, 0.20)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.status, "solved")

    def test_financial_boundaries(self):
        for kind, strike in (("call", 80.0), ("put", 120.0)):
            with self.subTest(kind=kind):
                lower, upper = self.pricer.price_bounds(
                    strike,
                    kind,
                )

                result = self.solver.solve(lower, strike, kind)
                self.assertEqual(result.volatility, 0.0)
                self.assertEqual(result.status, "at_lower_bound")
                self.assertEqual(result.price_error, 0.0)

                for invalid_price in (
                    lower - 0.01,
                    upper,
                    upper + 0.01,
                ):
                    with self.assertRaises(ValueError):
                        self.solver.solve(
                            invalid_price,
                            strike,
                            kind,
                        )

        expired = ImpliedVolSolver(
            BlackPricer(
                forward=100.0,
                discount_factor=1.0,
                maturity=0.0,
            )
        )
        with self.assertRaises(ValueError):
            expired.solve(0.0, 100.0, "call")

    def test_configured_search_limit(self):
        solver = ImpliedVolSolver(
            self.pricer,
            initial_upper=0.05,
            max_volatility=0.10,
        )
        price = self.pricer.price(100.0, 0.20, "call")

        with self.assertRaisesRegex(RuntimeError, "bracket"):
            solver.solve(price, 100.0, "call")

    def test_quote_sensitivity(self):
        volatility = 0.20
        price = self.pricer.price(150.0, volatility, "call")
        baseline = self.solver.solve(price, 150.0, "call")

        price_bump = 1e-5
        bumped = self.solver.solve(
            price + price_bump,
            150.0,
            "call",
        )
        observed_sensitivity = (
            bumped.volatility - baseline.volatility
        ) / price_bump

        np.testing.assert_allclose(
            observed_sensitivity,
            baseline.price_to_vol_sensitivity,
            rtol=1e-3,
            atol=0.0,
        )

        atm_price = self.pricer.price(100.0, volatility, "call")
        near_atm = self.solver.solve(
            atm_price,
            100.0,
            "call",
        )
        self.assertGreater(
            baseline.price_to_vol_sensitivity,
            near_atm.price_to_vol_sensitivity,
        )


if __name__ == "__main__":
    unittest.main()