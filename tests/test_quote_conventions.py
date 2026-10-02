import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lv_project.black_scholes import call_price, put_price
from lv_project.quote_conventions import (
    prepare_price_pairs, estimate_parity, forward_option_price, invert_forward_iv,
)


def synthetic_pairs(forward=103., discount=.96, half_spread=.10):
    k = np.linspace(80, 120, 21)
    c = forward_option_price(forward, k, discount, .75, .25, 'call')
    p = forward_option_price(forward, k, discount, .75, .25, 'put')
    return pd.DataFrame(dict(strike=k, underlying_last=100., parity_mid=c-p,
                             parity_lower=c-p-2*half_spread, parity_upper=c-p+2*half_spread))


class ParityTests(unittest.TestCase):
    def test_recovers_known_forward_discount_with_nonzero_carry(self):
        for f, d in [(103., .96), (97., 1.02), (102.5, .91)]:
            with self.subTest(forward=f, discount=d):
                result = estimate_parity(synthetic_pairs(f, d))
                self.assertEqual(result['status'], 'identified')
                self.assertAlmostEqual(result['forward'], f, places=11)
                self.assertAlmostEqual(result['discount'], d, places=12)
                self.assertLess(result['residual_rmse'], 1e-12)
                self.assertLessEqual(result['forward_lower'], f)
                self.assertGreaterEqual(result['forward_upper'], f)
                self.assertLessEqual(result['discount_lower'], d)
                self.assertGreaterEqual(result['discount_upper'], d)

    def test_feasible_bid_ask_ranges_contain_truth_despite_midpoint_noise(self):
        quotes = synthetic_pairs()
        noise = .08*np.sin(np.arange(len(quotes)))
        quotes[['parity_mid', 'parity_lower', 'parity_upper']] += noise[:, None]
        result = estimate_parity(quotes)
        self.assertTrue(result['interval_feasible'])
        self.assertLess(result['forward_lower'], 103.)
        self.assertGreater(result['forward_upper'], 103.)
        self.assertLess(result['discount_lower'], .96)
        self.assertGreater(result['discount_upper'], .96)
        self.assertEqual(result['interval_coverage'], 1)

    def test_inconsistent_quotes_are_flagged_without_zero_carry_fallback(self):
        quotes = synthetic_pairs()
        quotes.loc[10, ['parity_mid', 'parity_lower', 'parity_upper']] += 3
        result = estimate_parity(quotes)
        self.assertEqual(result['status'], 'weak')
        self.assertFalse(result['interval_feasible'])
        self.assertGreater(result['minimum_extra_half_width'], .5)
        self.assertIn('inconsistent_bid_ask_intervals', result['reasons'])
        self.assertTrue(np.isnan(result['forward_lower']))
        self.assertNotEqual(result['forward'], 100)
        self.assertNotEqual(result['discount'], 1)

    def test_discount_cannot_be_identified_from_one_or_repeated_strikes(self):
        quotes = synthetic_pairs().iloc[:1]
        self.assertEqual(estimate_parity(quotes)['reasons'], 'insufficient_pairs')
        result = estimate_parity(pd.concat([quotes]*10, ignore_index=True))
        self.assertEqual(result['reasons'], 'unidentified_strike_slope')
        self.assertTrue(np.isnan(result['forward']))
        self.assertTrue(np.isnan(result['discount']))

    def test_narrow_span_and_broad_uncertainty_are_weak(self):
        quotes = synthetic_pairs()
        quotes.strike = np.linspace(99.99, 100.01, len(quotes))
        quotes.parity_mid = .96*(103-quotes.strike)
        quotes.parity_lower = quotes.parity_mid-.2
        quotes.parity_upper = quotes.parity_mid+.2
        result = estimate_parity(quotes)
        self.assertEqual(result['status'], 'weak')
        self.assertIn('narrow_strike_span', result['reasons'])
        self.assertIn('wide_discount_range', result['reasons'])

    def test_increasing_parity_has_no_positive_discount_solution(self):
        quotes = synthetic_pairs()
        quotes.parity_mid *= -1
        quotes.parity_lower = quotes.parity_mid-.01
        quotes.parity_upper = quotes.parity_mid+.01
        result = estimate_parity(quotes)
        self.assertLess(result['discount'], 0)
        self.assertTrue(np.isnan(result['forward']))
        self.assertIn('nonpositive_forward_or_discount', result['reasons'])


def raw_fixture():
    return pd.DataFrame([dict(quote_date=pd.Timestamp('2023-12-22'),
                             expire_date=pd.Timestamp('2024-01-19'),
                             quote_unixtime=1703278800, expire_unix=1705698000,
                             dte=28.,
                             strike=100., underlying_last=100., c_bid=4.9, c_ask=5.1,
                             p_bid=3.9, p_ask=4.1, c_iv=np.nan, p_iv=np.nan)])


class PairingTests(unittest.TestCase):
    def test_missing_vendor_iv_does_not_discard_prices(self):
        q = prepare_price_pairs(raw_fixture())
        self.assertTrue(q.pair_usable.iloc[0])
        self.assertEqual(q.exclusion_reasons.iloc[0], '')
        self.assertTrue(q.c_iv.isna().all())
        self.assertTrue(q.p_iv.isna().all())
        self.assertAlmostEqual(q.parity_lower.iloc[0], .8)
        self.assertAlmostEqual(q.parity_upper.iloc[0], 1.2)
        self.assertAlmostEqual(q.maturity.iloc[0], 28/365)

    def test_exact_duplicate_retains_one_and_conflicts_exclude_all(self):
        raw = raw_fixture()
        q = prepare_price_pairs(pd.concat([raw, raw], ignore_index=True))
        self.assertEqual(q.pair_usable.sum(), 1)
        self.assertEqual(q.exclusion_reasons.iloc[1], 'identical_duplicate')
        conflict = raw.copy()
        conflict.c_ask += .1
        q = prepare_price_pairs(pd.concat([raw, conflict], ignore_index=True))
        self.assertEqual(q.pair_usable.sum(), 0)
        self.assertTrue(q.exclusion_reasons.str.contains('conflicting_duplicate').all())

    def test_snapshots_and_contract_types_are_separate_groups(self):
        raw = pd.concat([raw_fixture()]*3, ignore_index=True)
        raw['contract_root'] = ['SPX', 'SPXW', 'SPXW']
        raw.loc[2, 'quote_unixtime'] += 60
        q = prepare_price_pairs(raw)
        self.assertEqual(q.pair_usable.sum(), 3)
        self.assertEqual(q.pair_group.nunique(), 3)

    def test_side_timestamp_mismatch_and_bad_quote_are_excluded(self):
        raw = pd.concat([raw_fixture()]*2, ignore_index=True)
        raw.loc[1, 'strike'] = 105
        raw['c_quote_unixtime'] = [1703278799, 1703278800]
        raw.loc[1, 'p_ask'] = 3.8
        q = prepare_price_pairs(raw)
        self.assertEqual(q.pair_usable.sum(), 0)
        self.assertIn('asynchronous_sides', q.exclusion_reasons.iloc[0])
        self.assertIn('p_price_liquidity', q.exclusion_reasons.iloc[1])
        self.assertTrue(q.c_price_usable.iloc[1])


class InversionTests(unittest.TestCase):
    def test_forward_price_matches_spot_bs_under_nonzero_carry(self):
        s, r, q, t, vol = 100., .06, .018, .7, .29
        f, d = s*np.exp((r-q)*t), np.exp(-r*t)
        for k in [65., 100., 150.]:
            for option, reference in [('call', call_price), ('put', put_price)]:
                price = forward_option_price(f, k, d, t, vol, option)
                self.assertAlmostEqual(price, reference(s, k, t, r, q, vol), places=12)
                iv = invert_forward_iv(price, f, k, d, t, option)
                self.assertEqual(iv.status, 'ok')
                self.assertAlmostEqual(iv.volatility, vol, places=11)
                self.assertLess(abs(iv.repricing_error), 1e-10)

    def test_call_put_parity_and_iv_roundtrips_across_maturities(self):
        for t in [.02, .25, 1.]:
            f, d, vol = 100*np.exp(.04*t), np.exp(-.06*t), .32
            for k in [90, 100, 110]:
                c = forward_option_price(f, k, d, t, vol, 'call')
                p = forward_option_price(f, k, d, t, vol, 'put')
                self.assertAlmostEqual(c-p, d*(f-k), places=12)
                for price, option in [(c, 'call'), (p, 'put')]:
                    iv = invert_forward_iv(price, f, k, d, t, option)
                    self.assertEqual(iv.status, 'ok')
                    self.assertAlmostEqual(iv.volatility, vol, places=10)

    def test_bounds_and_zero_infinite_limits_are_not_positive_iv_success(self):
        f, k, d, t = 105., 100., .94, .3
        for option, low, high in [('call', 4.7, 98.7), ('put', 0, 94.)]:
            for price, status in [(low-.01, 'below_lower_bound'), (high+.01, 'above_upper_bound'),
                                  (low, 'at_intrinsic_limit'), (high, 'at_infinite_volatility_limit')]:
                with self.subTest(option=option, status=status):
                    self.assertEqual(invert_forward_iv(price, f, k, d, t, option).status, status)
        price = forward_option_price(f, k, d, t, .3)
        self.assertEqual(invert_forward_iv(price, f, k, d, t, max_volatility=.1).status, 'volatility_cap_exceeded')
        self.assertEqual(invert_forward_iv(np.nan, f, k, d, t).status, 'nonfinite_input')
        self.assertEqual(invert_forward_iv(price, f, k, 0, t).status, 'invalid_convention')

    def test_bid_ask_iv_uncertainty_and_zero_variance_prices(self):
        f, k, d, t = 103., 100., .97, .2
        mid = forward_option_price(f, k, d, t, .2)
        bid = invert_forward_iv(mid-.1, f, k, d, t)
        ask = invert_forward_iv(mid+.1, f, k, d, t)
        self.assertLess(bid.volatility, .2)
        self.assertGreater(ask.volatility, .2)
        self.assertEqual(forward_option_price(f, k, d, t, 0), d*(f-k))
        self.assertEqual(forward_option_price(f, k, d, t, 0, 'put'), 0.)


class PreservationTests(unittest.TestCase):
    def test_runner_marks_weak_conventions_without_inverting_or_substituting(self):
        spec = importlib.util.spec_from_file_location('conventions_runner', ROOT/'scripts/validate_quote_conventions.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        q = prepare_price_pairs(raw_fixture())
        fits, _, _ = runner.fit_conventions(q, runner.metadata_table(q))
        options = runner.option_rows(q, fits, set())
        self.assertTrue(options.mid_iv_status.eq('weak_parity_estimate').all())
        self.assertTrue(options.forward.isna().all())
        self.assertTrue(options.discount.isna().all())
        self.assertTrue(options.vendor_parity_price.isna().all())
        self.assertEqual(runner.price_metrics(options)['comparable_vendor_quotes'], 0)

    def test_prior_validation_outputs_are_monitored_and_current_run_is_excluded(self):
        spec = importlib.util.spec_from_file_location('conventions_runner', ROOT/'scripts/validate_quote_conventions.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root/'outputs/diagnostics/validation/checkpoint'
            output = root/'outputs/diagnostics/validation/conventions'
            previous.mkdir(parents=True)
            output.mkdir(parents=True)
            (previous/'result.csv').write_text('baseline')
            (output/'generated.csv').write_text('new')
            before = runner.preservation_snapshot(root, output)
            self.assertEqual(len(before), 1)
            self.assertEqual(runner.verify_preservation(root, before, output)['changed_files'], [])
            (previous/'result.csv').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'Preservation failed'):
                runner.verify_preservation(root, before, output)


if __name__ == '__main__':
    unittest.main()
