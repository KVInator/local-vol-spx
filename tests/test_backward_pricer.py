"""Analytical checks for prices, spot Greeks and reverse-time coefficients."""

import unittest
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr

from backward_pricer import BackwardVanillaPricer
from black import BlackPricer


@dataclass
class BlackSurface:
    spot: float = 100.0
    rate: float = 0.05
    carry: float = 0.03
    sigma: float = 0.2
    min_log_moneyness: float = -1.0
    max_log_moneyness: float = 1.0

    @property
    def maturities(self):
        return np.array([0.0, 0.25, 0.5, 1.0])

    def forward(self, time):
        if not 0.0 <= time <= 1.0:
            raise ValueError("Unsupported maturity.")
        return self.spot * np.exp(self.carry * time)

    def discount_factor(self, time):
        return np.exp(-self.rate * time)

    def normalized_variance(self, states, time, side="right"):
        return np.full_like(states, self.sigma**2)


def analytical(surface, spots, strike, expiry, valuation_time=0.0, kind="call"):
    spots = np.asarray(spots, dtype=float)
    horizon = expiry - valuation_time
    forward = spots * np.exp(surface.carry * horizon)
    discount = np.exp(-surface.rate * horizon)
    root_w = surface.sigma * np.sqrt(horizon)
    d1 = np.log(forward / strike) / root_w + 0.5 * root_w
    d2 = d1 - root_w
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    factor = np.exp((surface.carry - surface.rate) * horizon)
    price = discount * (forward * ndtr(d1) - strike * ndtr(d2))
    delta = factor * ndtr(d1)
    if kind == "put":
        price -= discount * (forward - strike)
        delta -= factor
    return price, delta, factor * phi / (spots * root_w)


class TestBackwardPricer(unittest.TestCase):
    def setUp(self):
        self.surface = BlackSurface()

    def pricer(self, intervals=800, time_step=0.001):
        return BackwardVanillaPricer(
            self.surface, n_space_intervals=intervals,
            max_time_step=time_step,
        )

    def test_call_put_prices_and_greeks_against_black(self):
        spots = np.array([90.0, 100.0, 110.0])
        for kind in ("call", "put"):
            result = self.pricer().solve(100.0, 0.5, kind=kind)
            actual = result.greeks(spots)
            price, delta, gamma = analytical(self.surface, spots, 100.0, 0.5, kind=kind)
            np.testing.assert_allclose(actual.price, price, rtol=0.0, atol=0.003)
            np.testing.assert_allclose(actual.delta, delta, rtol=0.0, atol=0.0002)
            np.testing.assert_allclose(actual.gamma, gamma, rtol=0.0, atol=0.00005)
            np.testing.assert_allclose(
                result.price(100.0),
                BlackPricer(self.surface.forward(0.5), self.surface.discount_factor(0.5), 0.5).price(100.0, 0.2, kind),
                atol=0.003,
            )

    def test_nonzero_valuation_time_and_carry_scaling(self):
        spots = np.array([95.0, 103.0, 112.0])
        result = self.pricer().solve(105.0, 0.5, valuation_time=0.1)
        actual = result.greeks(spots)
        expected = analytical(self.surface, spots, 105.0, 0.5, 0.1)
        for value, reference, tolerance in zip(
            (actual.price, actual.delta, actual.gamma), expected, (0.003, 0.0002, 0.00005),
        ):
            np.testing.assert_allclose(value, reference, rtol=0.0, atol=tolerance)

    def test_reverse_time_sides_and_internal_break(self):
        calls = []

        def variance(states, time, side):
            calls.append((time, side))
            value = 0.01 if time < 0.25 or (time == 0.25 and side == "left") else 0.09
            return np.full_like(states, value)

        self.surface.normalized_variance = variance
        result = BackwardVanillaPricer(
            self.surface, n_space_intervals=800,
            max_time_step=0.05, rannacher_steps=0,
        ).solve(100.0, 0.5)
        self.assertIn((0.25, "left"), calls)
        self.assertIn((0.25, "right"), calls)
        self.assertIn((0.5, "left"), calls)
        self.assertIn((0.0, "right"), calls)
        self.assertNotIn((0.0, "left"), calls)
        reference = BlackPricer(
            self.surface.forward(0.5), self.surface.discount_factor(0.5), 0.5,
        ).price(100.0, np.sqrt(0.05), "call")
        self.assertAlmostEqual(float(result.price(100.0)), float(reference), delta=0.02)

    def test_price_delta_and_gamma_improve_with_space_refinement(self):
        strike = self.surface.forward(0.5)
        reference = np.array(analytical(self.surface, 100.0, strike, 0.5))
        errors = []
        for intervals in (200, 400, 800):
            value = self.pricer(intervals, 0.00025).solve(strike, 0.5).greeks(100.0)
            errors.append(np.abs(np.array([value.price, value.delta, value.gamma]) - reference))
        errors = np.asarray(errors)
        self.assertTrue(np.all(errors[1:] < errors[:-1]))
        self.assertTrue(np.all(errors[-1] < 0.4 * errors[-2]))

    def test_spot_bumps_reprice_the_same_fixed_model(self):
        result = self.pricer().solve(100.0, 0.5)
        greeks = result.greeks(100.0)
        for bump in (0.5, 0.25, 0.125):
            lower, centre, upper = result.price(np.array([100.0 - bump, 100.0, 100.0 + bump]))
            delta = (upper - lower) / (2.0 * bump)
            gamma = (upper - 2.0 * centre + lower) / bump**2
            self.assertAlmostEqual(float(delta), float(greeks.delta), delta=0.0001)
            self.assertAlmostEqual(float(gamma), float(greeks.gamma), delta=0.00002)

    def test_state_dependent_variance_uses_backward_operator_order(self):
        def variance(states, time, side):
            early = time < 0.25 or (time == 0.25 and side == "left")
            sign = 1.0 if early else -1.0
            return 0.04 * (1.0 + sign * 0.5 * np.tanh(4.0 * np.log(states)))

        self.surface.normalized_variance = variance
        pricer = BackwardVanillaPricer(
            self.surface, -0.5, 0.5, 80, max_time_step=0.0002,
        )
        result = pricer.solve(100.0, 0.5)
        x = np.exp(result.log_states)
        h = result.log_states[1] - result.log_states[0]
        matrices = []
        for time in (0.1, 0.4):
            matrix = np.zeros((len(x), len(x)))
            a = 0.5 * variance(x[1:-1], time, "right")
            rows = np.arange(1, len(x) - 1)
            matrix[rows, rows - 1] = a * (1.0 / h**2 + 1.0 / (2.0 * h))
            matrix[rows, rows] = -2.0 * a / h**2
            matrix[rows, rows + 1] = a * (1.0 / h**2 - 1.0 / (2.0 * h))
            matrices.append(expm(0.25 * matrix))
        payoff = np.maximum(x - 100.0 / self.surface.forward(0.5), 0.0)
        expected = matrices[0] @ (matrices[1] @ payoff)
        wrong_order = matrices[1] @ (matrices[0] @ payoff)
        self.assertGreater(float(np.max(np.abs(expected - wrong_order))), 1e-5)
        np.testing.assert_allclose(result.normalized_values, expected, rtol=0.0, atol=2e-7)

    def test_expiry_payoff_and_kink(self):
        for kind in ("call", "put"):
            result = self.pricer().solve(100.0, 0.0, kind=kind)
            expected = [0.0, 0.0, 10.0] if kind == "call" else [10.0, 0.0, 0.0]
            np.testing.assert_array_equal(result.price([90.0, 100.0, 110.0]), expected)
            self.assertEqual(result.time_steps, 0)
            np.testing.assert_array_equal(result.greeks([90.0, 110.0]).gamma, [0.0, 0.0])
            with self.assertRaises(ValueError):
                result.greeks(100.0)

    def test_invalid_inputs_and_no_state_extrapolation(self):
        pricer = self.pricer()
        for arguments in ((0.0, 0.5), (100.0, -0.1), (100.0, np.nan)):
            with self.assertRaises(ValueError):
                pricer.solve(*arguments)
        with self.assertRaises(ValueError):
            pricer.solve(100.0, 0.1, valuation_time=0.2)
        with self.assertRaises(ValueError):
            pricer.solve(100.0, 0.5, kind="invalid")
        with self.assertRaises(ValueError):
            BackwardVanillaPricer(self.surface, min_log_state=-1.1)
        result = pricer.solve(100.0, 0.5)
        for spot in (0.0, np.nan, 400.0):
            with self.assertRaises(ValueError):
                result.price(spot)
        self.assertEqual(np.asarray(result.price(100.0)).shape, ())
        self.assertEqual(result.price(np.ones((2, 3)) * 100.0).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()