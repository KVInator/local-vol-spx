"""Analytical and independent checks of positive-time backward pricing."""

import unittest
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr

from ah_backward_pricer import AHBackwardPricer
from andreasen_huge import AHGrid, AndreasenHugeSurface


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

    def normalized_variance(
        self, states, time, side="right",
    ):
        if time <= 0:
            raise AssertionError(
                "Zero-time variance was requested."
            )
        return np.full_like(states, self.sigma**2)


def black_values(
    surface, spots, strike, expiry,
    valuation_time=0, variance=None,
):
    spots = np.asarray(spots, float)
    horizon = expiry - valuation_time
    total = (
        surface.sigma**2 * horizon
        if variance is None else variance
    )
    root = np.sqrt(total)
    forward = spots * np.exp(
        surface.carry * horizon
    )
    discount = np.exp(
        -surface.rate * horizon
    )
    d1 = (
        np.log(forward / strike) / root
        + 0.5 * root
    )
    factor = np.exp(
        (surface.carry - surface.rate) * horizon
    )
    phi = (
        np.exp(-0.5 * d1**2)
        / np.sqrt(2 * np.pi)
    )

    return (
        discount * (
            forward * ndtr(d1)
            - strike * ndtr(d1 - root)
        ),
        factor * ndtr(d1),
        factor * phi / (spots * root),
    )


class TestAHBackwardPricer(unittest.TestCase):
    def test_black_prices_and_greeks_improve_with_spatial_refinement(self):
        surface = BlackSurface()
        spots = np.array([95, 100, 105])
        strike = surface.forward(0.2)
        expected = np.array(
            black_values(
                surface, spots, strike, 0.2
            )
        )
        errors = []

        for n in (200, 400, 800):
            result = AHBackwardPricer(
                surface,
                space_intervals=n,
                steps_per_day=32,
            ).solve(strike, 0.2)
            actual = result.greeks(spots)
            errors.append(
                np.max(
                    abs(
                        np.array([
                            actual.price,
                            actual.delta,
                            actual.gamma,
                        ]) - expected
                    ),
                    axis=1,
                )
            )

        errors = np.asarray(errors)
        self.assertTrue(
            np.all(errors[1:] < errors[:-1])
        )
        self.assertTrue(
            np.all(
                errors[-1] < [0.002, 1e-4, 2e-5]
            )
        )

    def test_batched_calls_puts_and_nonzero_valuation_time(self):
        surface = BlackSurface()
        pricer = AHBackwardPricer(
            surface, space_intervals=800
        )
        strikes = [95, 100, 105]
        spots = np.array([97, 100, 103])

        calls = pricer.solve_many(
            strikes, 0.2, valuation_time=0.03
        )
        puts = pricer.solve_many(
            strikes, 0.2,
            valuation_time=0.03, kind="put",
        )

        for strike, call, put in zip(
            strikes, calls, puts
        ):
            expected = black_values(
                surface, spots, strike, 0.2, 0.03
            )
            actual = call.greeks(spots)

            for value, reference, tolerance in zip(
                [
                    actual.price,
                    actual.delta,
                    actual.gamma,
                ],
                expected,
                [0.002, 1e-4, 2e-5],
            ):
                np.testing.assert_allclose(
                    value, reference,
                    rtol=0, atol=tolerance,
                )

            parity = (
                np.exp(-surface.rate * 0.17)
                * (
                    spots
                    * np.exp(surface.carry * 0.17)
                    - strike
                )
            )
            np.testing.assert_allclose(
                call.price(spots) - put.price(spots),
                parity,
                atol=3e-5,
            )

        single = pricer.solve(
            100, 0.2, valuation_time=0.03
        )
        np.testing.assert_array_equal(
            single.normalized_values,
            calls[1].normalized_values,
        )
        self.assertGreater(
            pricer.last_diagnostics[
                "minimum_coefficient_time_years"
            ],
            0,
        )

    def test_time_order_against_independent_matrix_exponentials(self):
        surface = BlackSurface(
            carry=0, rate=0
        )

        def variance(states, time, side="right"):
            sign = 1 if time < 0.25 else -1
            return 0.04 * (
                1 + sign * 0.5
                * np.tanh(4 * np.log(states))
            )

        surface.normalized_variance = variance
        pricer = AHBackwardPricer(
            surface,
            space_intervals=60,
            steps_per_day=16,
        )
        result = pricer.solve(100, 0.5)
        matrices = []

        for time in (0.1, 0.4):
            matrix = np.zeros(
                (len(pricer.z), len(pricer.z))
            )
            a = 0.5 * variance(
                pricer.z[1:-1], time
            )
            rows = np.arange(
                1, len(pricer.z) - 1
            )
            matrix[rows, rows - 1] = a * (
                1 / pricer.h**2
                + 1 / (2 * pricer.h)
            )
            matrix[rows, rows] = (
                -2 * a / pricer.h**2
            )
            matrix[rows, rows + 1] = a * (
                1 / pricer.h**2
                - 1 / (2 * pricer.h)
            )
            matrices.append(
                expm(0.25 * matrix)
            )

        payoff = np.maximum(
            pricer.z - 1, 0
        )
        expected = (
            matrices[0]
            @ (matrices[1] @ payoff)
        )
        wrong = (
            matrices[1]
            @ (matrices[0] @ payoff)
        )

        self.assertGreater(
            np.max(abs(expected - wrong)),
            1e-5,
        )
        np.testing.assert_allclose(
            result.normalized_values,
            expected,
            rtol=0,
            atol=5e-7,
        )

    def test_integrable_singular_variance_with_analytical_total_variance(self):
        surface = BlackSurface(
            carry=0, rate=0
        )
        sampled = []

        def variance(states, time, side="right"):
            sampled.append(time)
            self.assertGreater(time, 0)
            return np.full_like(
                states, 0.04 / np.sqrt(time)
            )

        surface.normalized_variance = variance
        expected = black_values(
            surface, 100, 100, 0.1,
            variance=0.08 * np.sqrt(0.1),
        )[0]
        errors = []

        for steps in (8, 16, 32):
            result = AHBackwardPricer(
                surface,
                space_intervals=1200,
                steps_per_day=steps,
                early_time_power=3,
            ).solve(100, 0.1)
            errors.append(
                abs(
                    float(result.price(100))
                    - expected
                )
            )

        self.assertTrue(
            np.all(np.diff(errors) < 0)
        )
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
                [
                    np.log([0.24, 0.18, 0.16]),
                    np.log([0.3, 0.22, 0.19]),
                ],
            )
            result = AHBackwardPricer(
                model,
                space_intervals=n,
                steps_per_day=32,
            ).solve(100, 0.1)
            errors.append(
                abs(
                    float(result.price(100))
                    - float(
                        model.call_price(100, 0.1)
                    )
                )
            )

        self.assertTrue(
            np.all(np.diff(errors) < 0)
        )
        self.assertLess(errors[-1], 0.01)

    def test_spot_bumps_match_analytical_black_greeks(self):
        surface = BlackSurface()
        result = AHBackwardPricer(
            surface, space_intervals=800
        ).solve(100, 0.2)
        _, delta, gamma = black_values(
            surface, 100, 100, 0.2
        )
        low, centre, high = result.price(
            [99.9, 100, 100.1]
        )

        self.assertAlmostEqual(
            float((high - low) / 0.2),
            float(delta),
            delta=1e-4,
        )
        self.assertAlmostEqual(
            float(
                (high - 2 * centre + low)
                / 0.1**2
            ),
            float(gamma),
            delta=2e-5,
        )

    def test_expiry_payoff_invalid_settings_and_no_extrapolation(self):
        surface = BlackSurface()
        pricer = AHBackwardPricer(
            surface, space_intervals=100
        )
        result = pricer.solve(100, 0)

        np.testing.assert_array_equal(
            result.price([90, 100, 110]),
            [0, 0, 10],
        )
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
                AHBackwardPricer(
                    surface, **settings
                )

        for strikes, expiry, kwargs in (
            ([0], 0.1, {}),
            ([[100]], 0.1, {}),
            ([100], -1, {}),
            ([100], 0.1, {"valuation_time": 0.2}),
        ):
            with self.assertRaises(ValueError):
                pricer.solve_many(
                    strikes, expiry, **kwargs
                )

        with self.assertRaises(ValueError):
            result.price(300)


if __name__ == "__main__":
    unittest.main()