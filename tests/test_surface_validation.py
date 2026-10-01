import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lv_project.black_scholes import vectorized_call_price
from lv_project.diagnostics import check_call_slice, reconstruct_call_prices
from lv_project.surface import (
    build_daily_surface, count_calendar_violations, interpolate_target_surface,
    repair_calendar_monotonicity, surface_to_long_frame,
)


def manufactured_quotes():
    date = pd.Timestamp('2023-12-22')
    rows = []
    for days in [14, 30, 90]:
        t = days/365
        for y in np.linspace(-.1, .1, 11):
            rows.append(dict(quote_date=date, expiry=date+pd.Timedelta(days=days),
                             underlying_spot=100., time_to_expiry=t,
                             strike=100*np.exp((.03-.01)*t+y), implied_vol=.2))
    return pd.DataFrame(rows)


class SurfaceValidationTests(unittest.TestCase):
    def test_flat_vol_surface_and_carry_are_preserved(self):
        y = np.linspace(-.08, .08, 17)
        t = np.array([14, 21, 30, 60, 90])/365
        result = build_daily_surface(manufactured_quotes(), '2023-12-22', y, t,
                                     flat_rate=.03, flat_dividend_yield=.01)
        np.testing.assert_allclose(result.target_implied_vol, .2, atol=1e-14)
        self.assertEqual(result.diagnostics['calendar_violations_after'], 0)
        self.assertEqual(result.diagnostics['edge_fill_count'], 0)
        np.testing.assert_allclose(result.surface_points.forward,
                                   100*np.exp(.02*result.surface_points.time_to_expiry))
        self.assertEqual(len(surface_to_long_frame(result)), len(t)*len(y))

    def test_unsupported_maturities_and_constant_edge_fills_are_explicit(self):
        y = np.array([-.2, 0, .2])
        t = np.array([7, 14, 30, 90, 120])/365
        result = build_daily_surface(manufactured_quotes(), '2023-12-22', y, t,
                                     flat_rate=.03, flat_dividend_yield=.01)
        self.assertTrue(np.isnan(result.target_total_variance[[0, -1]]).all())
        self.assertTrue(np.isfinite(result.target_total_variance[1:-1]).all())
        self.assertEqual(result.diagnostics['edge_fill_count'], 6)
        np.testing.assert_allclose(result.pillar_total_variance,
                                   np.tile(result.pillar_maturities[:, None]*.04, (1, 3)))

    def test_duplicate_strikes_and_sparse_expiries(self):
        quotes = manufactured_quotes()
        quotes = pd.concat([quotes, quotes.iloc[[0]]], ignore_index=True)
        result = build_daily_surface(quotes, '2023-12-22', np.array([-.05, 0, .05]),
                                     np.array([14, 30])/365,
                                     flat_rate=.03, flat_dividend_yield=.01)
        self.assertEqual(result.expiry_coverage.n_points.tolist(), [11, 11, 11])
        with self.assertRaisesRegex(ValueError, 'Fewer than two expiries'):
            build_daily_surface(quotes.iloc[:7], '2023-12-22', np.array([0]), np.array([.1]))

    def test_calendar_repair_is_upward_idempotent_and_interpolation_monotone(self):
        raw = np.array([[.04, .01], [.03, .02], [.05, .015]])
        fixed = repair_calendar_monotonicity(raw)
        self.assertEqual(count_calendar_violations(raw), 2)
        self.assertEqual(count_calendar_violations(fixed), 0)
        np.testing.assert_allclose(fixed-raw, [[0, 0], [.01, 0], [0, .005]])
        np.testing.assert_array_equal(repair_calendar_monotonicity(fixed), fixed)
        target = interpolate_target_surface([.1, .2, .4], fixed, [.05, .1, .15, .3, .4, .5])
        self.assertTrue(np.isnan(target[[0, -1]]).all())
        self.assertTrue((np.diff(target[1:-1], axis=0) >= 0).all())


class CallDiagnosticTests(unittest.TestCase):
    def test_forward_formula_agrees_with_spot_bs_for_nonzero_carry(self):
        t = np.array([.04, .2, .9])
        y = np.linspace(-.2, .2, 101)
        w = np.tile(t[:, None]*.2**2, (1, len(y)))
        k, c, f, d = reconstruct_call_prices(100, t, y, w, .04, .015)
        expected = vectorized_call_price(100, k, t[:, None], .04, .015, .2)
        np.testing.assert_allclose(c, expected, atol=4e-14)
        for i in range(len(t)):
            table = check_call_slice(k[i], c[i], f[i], d[i])
            self.assertFalse(table[['bounds_violation', 'monotonicity_violation',
                                    'vertical_spread_violation', 'butterfly_violation']].any().any())

    def test_price_checks_detect_known_violations_on_nonuniform_strikes(self):
        table = check_call_slice([80, 90, 105, 120], [25, 23, 15, 2], 100, 1)
        self.assertEqual(table.butterfly_violation.sum(), 2)
        self.assertTrue((table.butterfly_cost.iloc[1:-1] < 0).all())
        table = check_call_slice([80, 90, 105, 120], [19, 31, 0, 1], 100, 1)
        self.assertEqual(table.bounds_violation.sum(), 1)
        self.assertEqual(table.monotonicity_violation.sum(), 2)
        self.assertEqual(table.vertical_spread_violation.sum(), 1)

    def test_nan_holes_are_counted_as_unchecked_and_zero_variance_is_intrinsic(self):
        k, c, f, d = reconstruct_call_prices(100, [.2], [-.1, 0, .1],
                                            np.array([[0., np.nan, -.01]]))
        self.assertAlmostEqual(c[0, 0], 100-k[0, 0])
        self.assertTrue(np.isnan(c[0, 1:]).all())
        table = check_call_slice([80, 90, 100, 110, 120], [22, 14, np.nan, 4, 2], 100, 1)
        self.assertEqual(table.price_valid.sum(), 4)
        self.assertEqual(table.right_pair_valid.sum(), 2)
        self.assertEqual(table.butterfly_valid.sum(), 0)


if __name__ == '__main__':
    unittest.main()
