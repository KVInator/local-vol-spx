"""Analytical benchmarks for the normalized forward PDE."""

import unittest

import numpy as np
from scipy.special import ndtr

from forward_pde import ForwardPDESolver


def normalized_black(strikes, integrated_variance):
    """Black call with forward and discount factor equal to one."""
    strikes = np.asarray(strikes, dtype=float)
    variance = float(integrated_variance)

    if variance == 0.0:
        return np.maximum(1.0 - strikes, 0.0)

    standard_deviation = np.sqrt(variance)
    d1 = (
        -np.log(strikes) + 0.5 * variance
    ) / standard_deviation
    d2 = d1 - standard_deviation

    return ndtr(d1) - strikes * ndtr(d2)


def solve_analytical_case(
    solver,
    times,
    integrated_variance,
    local_variance,
    time_breaks=(),
):
    strikes = solver.normalized_strikes
    endpoint_strikes = strikes[[0, -1]]

    def boundaries(maturity):
        values = normalized_black(
            endpoint_strikes,
            integrated_variance(maturity),
        )
        return float(values[0]), float(values[1])

    return solver.solve(
        maturities=times,
        initial_prices=normalized_black(
            strikes,
            integrated_variance(times[0]),
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

            result = solve_analytical_case(
                solver=solver,
                times=[0.0, 0.5],
                integrated_variance=lambda t: 0.04 * t,
                local_variance=lambda z, t, side: 0.04,
            )

            expected = normalized_black(
                solver.normalized_strikes,
                0.04 * 0.5,
            )
            central = np.abs(result.log_moneyness) <= 0.25

            errors.append(float(np.max(np.abs(
                result.normalized_calls[-1, central]
                - expected[central]
            ))))

        self.assertLess(errors[1], 0.6 * errors[0])
        self.assertLess(errors[2], 0.6 * errors[1])
        self.assertLess(errors[-1], 5e-5)

    def test_time_varying_variance_recovers_analytical_prices(self):
        solver = ForwardPDESolver(
            min_log_moneyness=-1.0,
            max_log_moneyness=1.0,
            n_space_intervals=400,
            max_time_step=0.001,
        )

        def integrated_variance(t):
            return 0.04 * t + 0.15 * t**2

        result = solve_analytical_case(
            solver=solver,
            times=[0.0, 0.2, 0.5],
            integrated_variance=integrated_variance,
            local_variance=lambda z, t, side: 0.04 + 0.3 * t,
        )

        central = np.abs(result.log_moneyness) <= 0.25

        for index, maturity in enumerate(result.maturities[1:], 1):
            expected = normalized_black(
                solver.normalized_strikes,
                integrated_variance(maturity),
            )

            np.testing.assert_allclose(
                result.normalized_calls[index, central],
                expected[central],
                rtol=0.0,
                atol=8e-5,
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
            return (
                0.04 * min(t, switch)
                + 0.09 * max(t - switch, 0.0)
            )

        def local_variance(z, t, side):
            if t == switch:
                sides_seen.append(side)
                return 0.04 if side == "left" else 0.09

            return 0.04 if t < switch else 0.09

        result = solve_analytical_case(
            solver=solver,
            times=[0.0, 0.5],
            integrated_variance=integrated_variance,
            local_variance=local_variance,
            time_breaks=[switch],
        )

        expected = normalized_black(
            solver.normalized_strikes,
            integrated_variance(0.5),
        )
        central = np.abs(result.log_moneyness) <= 0.25

        np.testing.assert_allclose(
            result.normalized_calls[-1, central],
            expected[central],
            rtol=0.0,
            atol=1e-4,
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

            return solve_analytical_case(
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

            errors.append(float(np.max(np.abs(
                result.normalized_calls[-1, central]
                - reference.normalized_calls[-1, central]
            ))))

        self.assertLess(errors[1], 0.6 * errors[0])
        self.assertLess(errors[2], 0.6 * errors[1])

    def test_zero_variance_preserves_initial_payoff(self):
        solver = ForwardPDESolver(
            min_log_moneyness=-1.0,
            max_log_moneyness=1.0,
            n_space_intervals=40,
            max_time_step=0.03,
        )

        result = solve_analytical_case(
            solver=solver,
            times=[0.0, 0.1, 0.2],
            integrated_variance=lambda t: 0.0,
            local_variance=lambda z, t, side: 0.0,
        )

        initial = np.maximum(
            1.0 - solver.normalized_strikes,
            0.0,
        )

        np.testing.assert_array_equal(
            result.normalized_calls,
            np.repeat(initial[None, :], 3, axis=0),
        )


if __name__ == "__main__":
    unittest.main()