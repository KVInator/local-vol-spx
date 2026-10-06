"""Independent checks of logarithmic AH density and variance evaluation."""

import unittest

import numpy as np
from scipy.linalg import solve_banded

from ah_local_vol import AHLocalVariance, log_positive_solve
from andreasen_huge import AHGrid, AndreasenHugeSurface
from forward_pde import ForwardPDESolver


def make_surface(intervals=800):
    return AndreasenHugeSurface(
        AHGrid(1, intervals),
        100,
        [0.1, 0.2],
        [100, 100],
        [1, 1],
        [np.array([-0.2, 0, 0.2])] * 2,
        [
            np.log([0.24, 0.18, 0.16]),
            np.log([0.3, 0.22, 0.19]),
        ],
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
        model = make_surface()
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
            np.testing.assert_allclose(
                actual, expected, rtol=2e-9, atol=1e-10
            )

    def test_positive_early_wing_variance_when_density_underflows(self):
        model = make_surface(2000)
        time = 1e-8
        self.assertTrue(np.any(model.node_state(time)["curvature"] == 0))
        values = AHLocalVariance(model)(np.exp([-0.8, 0, 0.8]), time)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values > 0))

    def test_rejects_zero_time_invalid_side_and_extrapolation(self):
        adapter = AHLocalVariance(make_surface())
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
            model = make_surface(count)
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
            error = (
                result.normalized_calls[-1]
                - model.normalized_call(z, 0.1)
            )
            errors.append(np.max(abs(error[mask])))

        self.assertLess(errors[-1], 1e-5)
        self.assertGreater(errors[0] / errors[1], 2.5)
        self.assertGreater(errors[1] / errors[2], 2.5)


if __name__ == "__main__":
    unittest.main()