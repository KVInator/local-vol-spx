"""Mathematical checks for the public Black pricing module."""

import unittest

import numpy as np

from black import BlackPricer


class TestBlackPricer(unittest.TestCase):
    def test_existing_benchmark(self):
        spot = 100.0
        maturity = 0.5
        rate = 0.05
        dividend_yield = 0.02

        pricer = BlackPricer(
            forward=spot * np.exp(
                (rate - dividend_yield) * maturity
            ),
            discount_factor=np.exp(-rate * maturity),
            maturity=maturity,
        )

        # Prices were recorded to six decimal places.
        np.testing.assert_allclose(
            pricer.price(100.0, 0.20, "call"),
            6.307635,
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            pricer.price(100.0, 0.20, "put"),
            4.833643,
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            pricer.vega(100.0, 0.20),
            27.4957944120,
            rtol=0.0,
            atol=1e-9,
        )

    def test_at_money_values(self):
        pricer = BlackPricer(
            forward=100.0,
            discount_factor=1.0,
            maturity=1.0,
        )

        for kind in ("call", "put"):
            np.testing.assert_allclose(
                pricer.price(100.0, 0.20, kind),
                7.965567455405804,
                rtol=0.0,
                atol=1e-12,
            )

            # At very small sigma, price is approximately
            # F*sigma*sqrt(T)/sqrt(2*pi).
            tiny_volatility = 1e-12
            expected = (
                100.0
                * tiny_volatility
                / np.sqrt(2.0 * np.pi)
            )
            np.testing.assert_allclose(
                pricer.price(100.0, tiny_volatility, kind),
                expected,
                rtol=1e-12,
                atol=0.0,
            )

    def test_vector_prices_bounds_and_parity(self):
        pricer = BlackPricer(
            forward=101.5,
            discount_factor=0.975,
            maturity=0.5,
        )
        strikes = np.array(
            [60.0, 80.0, 100.0, 120.0, 140.0]
        )[None, :]
        volatilities = np.array(
            [0.0, 0.10, 0.20, 0.60]
        )[:, None]

        calls = pricer.price(strikes, volatilities, "call")
        puts = pricer.price(strikes, volatilities, "put")

        self.assertEqual(calls.shape, (4, 5))
        self.assertEqual(puts.shape, (4, 5))

        parity = np.broadcast_to(
            pricer.discount_factor
            * (pricer.forward - strikes),
            calls.shape,
        )
        np.testing.assert_allclose(
            calls - puts,
            parity,
            rtol=0.0,
            atol=2e-14,
        )

        for kind, prices in (("call", calls), ("put", puts)):
            lower, upper = pricer.price_bounds(strikes, kind)
            self.assertTrue(
                np.all(prices >= lower - 1e-12)
            )
            self.assertTrue(
                np.all(prices <= upper + 1e-12)
            )

    def test_zero_volatility_and_expiry(self):
        strikes = np.array([80.0, 100.0, 120.0])
        pricer = BlackPricer(
            forward=100.0,
            discount_factor=0.97,
            maturity=0.25,
        )

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

        expected_vega = np.array(
            [
                0.0,
                0.97 * 100.0 * 0.5 / np.sqrt(2.0 * np.pi),
                0.0,
            ]
        )
        np.testing.assert_allclose(
            pricer.vega(strikes, 0.0),
            expected_vega,
            rtol=0.0,
            atol=1e-13,
        )

        expired = BlackPricer(
            forward=100.0,
            discount_factor=1.0,
            maturity=0.0,
        )
        np.testing.assert_array_equal(
            expired.price(strikes, 0.20, "call"),
            [20.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            expired.vega(strikes, 0.20),
            np.zeros(3),
        )

    def test_vega_matches_price_derivative(self):
        pricer = BlackPricer(
            forward=100.0 * np.exp(0.03 * 0.5),
            discount_factor=np.exp(-0.05 * 0.5),
            maturity=0.5,
        )
        strikes = np.array([80.0, 100.0, 120.0])
        volatility = 0.20
        bump = 1e-5

        for kind in ("call", "put"):
            finite_difference = (
                pricer.price(
                    strikes,
                    volatility + bump,
                    kind,
                )
                - pricer.price(
                    strikes,
                    volatility - bump,
                    kind,
                )
            ) / (2.0 * bump)

            np.testing.assert_allclose(
                finite_difference,
                pricer.vega(strikes, volatility),
                rtol=0.0,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()