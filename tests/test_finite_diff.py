import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lv_project.config import default_target_maturities_years
from lv_project.finite_diff import (
    first_derivative, first_derivative_1d, second_derivative, second_derivative_1d,
)


class FiniteDifferenceTests(unittest.TestCase):
    def test_first_derivative_quadratic_on_unequal_neighbors(self):
        x = np.array([0., 1., 3.])
        np.testing.assert_allclose(first_derivative_1d(x, x*x), 2*x, atol=1e-13)

    def test_polynomial_exactness_on_uniform_and_actual_maturity_grids(self):
        grids = [np.linspace(-1, 1, 17), default_target_maturities_years(),
                 np.array([0., .08, .19, .5, .9, 1.4])]
        for x in grids:
            with self.subTest(grid=x):
                np.testing.assert_allclose(first_derivative_1d(x, 3*x*x - 2*x + 7),
                                           6*x - 2, atol=2e-12)
                np.testing.assert_allclose(second_derivative_1d(x, 3*x*x - 2*x + 7),
                                           6, atol=2e-10)

    def test_nonuniform_second_derivative_boundary_is_at_the_endpoint(self):
        x = np.array([0., .13, .4, .9, 1.4])
        np.testing.assert_allclose(second_derivative_1d(x, x**3)[[0, -1]],
                                   6*x[[0, -1]], atol=1e-12)

    def test_first_derivative_agrees_with_independent_numpy_operator(self):
        x = np.array([-.7, -.5, -.11, .07, .6, 1.1])
        f = np.exp(x) + np.sin(2*x)
        np.testing.assert_allclose(first_derivative_1d(x, f),
                                   np.gradient(f, x, edge_order=2), atol=2e-14)

    def test_second_order_refinement_for_smooth_grids_and_boundaries(self):
        for kind in ['uniform', 'smooth_nonuniform']:
            for operator in [first_derivative_1d, second_derivative_1d]:
                errors = []
                for n in [33, 65, 129]:
                    u = np.linspace(0, 1, n)
                    x = u if kind == 'uniform' else (u + u*u)/2
                    error = np.abs(operator(x, np.exp(x)) - np.exp(x))
                    errors.append([error[1:-1].max(), error[[0, -1]].max()])
                rates = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
                with self.subTest(grid=kind, derivative=operator.__name__):
                    self.assertGreater(rates.min(), 1.8)

    def test_abrupt_nonuniform_second_derivative_is_only_first_order(self):
        errors = []
        for n in [33, 65, 129]:
            steps = np.resize([1., 2.], n-1)
            x = np.r_[0., np.cumsum(steps)/steps.sum()]
            errors.append(np.max(np.abs(second_derivative_1d(x, x**3)[1:-1]
                                         - 6*x[1:-1])))
        np.testing.assert_allclose(np.log2(np.array(errors[:-1])/errors[1:]),
                                   1, atol=1e-8)

    def test_axis_application_and_nan_support_holes(self):
        x = default_target_maturities_years()
        values = np.stack([x*x, 2*x*x], axis=1)
        np.testing.assert_allclose(first_derivative(values, x, axis=0),
                                   np.stack([2*x, 4*x], axis=1), atol=1e-13)
        np.testing.assert_allclose(second_derivative(values.T, x, axis=-1),
                                   np.tile([[2.], [4.]], (1, len(x))), atol=1e-12)
        f = np.linspace(0, 1, 9)**2
        f[4] = np.nan
        for operator in [first_derivative_1d, second_derivative_1d]:
            derivative = operator(np.linspace(0, 1, 9), f)
            self.assertTrue(np.isnan(derivative[3:6]).all())
            self.assertTrue(np.isfinite(derivative[[0, 1, 7, 8]]).all())

    def test_invalid_grids_and_axes_are_rejected(self):
        for x in [[0, 1], [0, 1, 1], [0, np.nan, 2], [0, 1, np.inf]]:
            with self.subTest(grid=x), self.assertRaises(ValueError):
                first_derivative_1d(x, np.zeros(len(x)))
        with self.assertRaises(ValueError):
            first_derivative(np.ones((3, 2)), [0, 1, 2], 2)
        with self.assertRaises(ValueError):
            first_derivative(np.ones((3, 2)), [0, 1, 2], 1)


if __name__ == '__main__':
    unittest.main()
