"""Analytical smile, leakage, least-squares, coverage and provenance controls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from andreasen_huge import AHGrid, AndreasenHugeSurface
from optimal_hedging import (AHSmileHedges, SmileSettings, EmpiricalMVSettings,
    PastOnlyEmpiricalMV, black_greeks, smile_conventions, analytical_controls,
    compare_hedges, paired_gains, OptimalHedgeInputs, sha256)


def empirical_fixture(days=15):
    """Exact declared quadratic response, not historical marks or a BS path."""
    rows = []
    beta = np.array([-.12, .07, -.035])
    for i, stamp in enumerate(pd.date_range('2023-08-01T20:00:00Z', periods=days)):
        fixing = pd.Timestamp('2023-11-01T20:00:00Z')
        t = (fixing-stamp).total_seconds()/(365*86400)
        f, d = 100*np.exp(.01*t), np.exp(-.05*t)
        for kind in ('call', 'put'):
            for k in (93., 100., 107.):
                b = black_greeks(100, k, t, f, d, .25, kind)
                row = dict(entry_id=f'{i}-{kind}-{k}', contract_id=f'{kind}-{k}',
                    quote_date=stamp.date().isoformat(), quote_timestamp_utc=stamp,
                    assumed_fixing_utc=fixing, expire_date='2023-11-01', root='UNKNOWN', kind=kind,
                    strike=k, underlying_last=100., forward=f, discount_factor=d,
                    assumed_maturity_years=t, black_status='ready', black_iv=.25,
                    black_delta=b['delta'], ah_status='ready', ah_delta=b['delta'],
                    mid=b['price'], bid=b['price']-.01, ask=b['price']+.01,
                    end_status='matched', end_timestamp=stamp+pd.Timedelta(days=1),
                    end_date=(stamp+pd.Timedelta(days=1)).date().isoformat(),
                    end_spot=100+(.4 if i%2 else -.3),
                    calendar_days=t*365, entry_spot_y=np.log(k/100))
                actual_beta = beta if kind == 'call' else beta*np.array([.8, 1.2, .7])
                delta = b['delta']+PastOnlyEmpiricalMV.feature(row)@actual_beta
                row['end_mid'] = row['mid']+delta*(row['end_spot']-100)
                row['end_bid'], row['end_ask'] = row['end_mid']-.01, row['end_mid']+.01
                rows.append(row)
    return pd.DataFrame(rows), beta


def ah_fixture():
    t = np.array([14, 28, 45])/365
    model = AndreasenHugeSurface(AHGrid(1., 8000), 100., t,
        100*np.exp(.01*t), np.exp(-.05*t),
        [np.array([-.2, .2]) for _ in t], [np.log([.2, .2]) for _ in t])
    rows, quotes = [], []
    for time in t:
        expiry = f'T{time}'
        f, d = model.forward(time), model.discount_factor(time)
        for k in f*np.exp(np.linspace(-.08,.08,25)):
            quotes.append(dict(expire_date=expiry, strike=k))
        for kind in ('call', 'put'):
            for y in (-.02, 0., .02):
                k = f*np.exp(y)
                iv = float(model.implied_volatility(np.array([k]), time)[0])
                b = black_greeks(100., k, time, f, d, iv, kind)
                rows.append(dict(entry_id=f'{time}-{kind}-{y}', underlying_last=100., strike=k,
                    assumed_maturity_years=time, forward=f, discount_factor=d, kind=kind,
                    black_iv=iv, black_delta=b['delta'], black_status='ready', expire_date=expiry))
    return model, pd.DataFrame(rows), pd.DataFrame(quotes)


class SmileControls(unittest.TestCase):
    def test_known_smiles_match_direct_spot_bumps(self):
        controls = analytical_controls()
        self.assertLess(controls.bump_error_1e_4.max(), 1e-6)
        self.assertTrue((controls.bump_error_1e_4 < controls.bump_error_1e_3).all())

    def test_flat_smile_all_surface_deltas_are_black(self):
        g = smile_conventions(100, 101, .2, 102, .99, .25, 0., 'call')
        self.assertEqual(g['surface_sticky_strike_delta'], g['surface_sticky_delta_delta'])
        self.assertEqual(g['surface_sticky_strike_delta'], g['surface_hw_lv_delta'])

    def test_negative_skew_sticky_delta_and_hw_lv_have_opposite_signs(self):
        g = smile_conventions(100, 101, .2, 102, .99, .25, -.3, 'call')
        self.assertGreater(g['surface_sticky_delta_delta'], g['surface_sticky_strike_delta'])
        self.assertLess(g['surface_hw_lv_delta'], g['surface_sticky_strike_delta'])

    def test_call_put_surface_parity(self):
        call = smile_conventions(100, 101, .2, 102, .99, .25, -.3, 'call')
        put = smile_conventions(100, 101, .2, 102, .99, .25, -.3, 'put')
        for name in ('surface_sticky_strike_delta', 'surface_sticky_delta_delta', 'surface_hw_lv_delta'):
            self.assertAlmostEqual(call[name]-put[name], .99*102/100)
        self.assertEqual(call['surface_vega'], put['surface_vega'])

    def test_vega_is_per_decimal_volatility(self):
        args = (100, 101, .2, 102, .99)
        g = black_greeks(*args, .25, 'call')
        fd = (black_greeks(*args, .250001, 'call')['price']-
              black_greeks(*args, .249999, 'call')['price'])/.000002
        self.assertAlmostEqual(g['vega'], fd, places=7)

    def test_sticky_forward_delta_implicit_repricing(self):
        # Freeze sigma(q), q=N(d1), rather than merely relabelling a spot bump.
        spot, k, t, f, d = 100., 101., .2, 102., .99
        sigma, sigma_y = .25, -.3
        y0 = np.log(k/f)
        def q(y):
            vol = sigma+sigma_y*(y-y0)
            return ndtr(-y/(vol*np.sqrt(t))+vol*np.sqrt(t)/2)
        def price(s):
            ff = f*s/spot
            # Seek a quote-delta point on the frozen smile that prices this K.
            def equation(y):
                vol = sigma+sigma_y*(y-y0)
                return q(y)-ndtr(np.log(ff/k)/(vol*np.sqrt(t))+vol*np.sqrt(t)/2)
            y = brentq(equation, y0-.02, y0+.02)
            return black_greeks(s,k,t,ff,d,sigma+sigma_y*(y-y0),'call')['price']
        fd = (price(100.001)-price(99.999))/.002
        g = smile_conventions(spot,k,t,f,d,sigma,sigma_y,'call')
        self.assertAlmostEqual(fd,g['surface_sticky_delta_delta'],places=7)

    def test_noninvertible_delta_coordinate_is_detectable(self):
        g = smile_conventions(100, 130, .2, 102, .99, .25, 3., 'call')
        self.assertGreater(g['surface_delta_coordinate_slope'], 0)

    def test_saved_ah_smiles_pass_bump_and_width_controls(self):
        model, entries, quotes = ah_fixture()
        out = AHSmileHedges().calculate(entries, model, quotes)
        self.assertTrue(out.smile_status.eq('ready').all(), out.to_string())
        self.assertTrue(out.sticky_delta_status.eq('ready').all())
        self.assertTrue(out.lv_smile_status.eq('ready').all())
        self.assertLess(out.smile_chain_minus_bump.max(), SmileSettings().delta_tolerance)

    def test_smile_does_not_read_endpoint_marks(self):
        model, entries, quotes = ah_fixture()
        first = AHSmileHedges().calculate(entries, model, quotes)
        entries['end_mid'] = np.arange(len(entries))*1000
        entries['end_spot'] = 1.
        pd.testing.assert_frame_equal(first, AHSmileHedges().calculate(entries, model, quotes))

    def test_quote_support_failure_keeps_every_entry(self):
        model, entries, quotes = ah_fixture()
        out = AHSmileHedges().calculate(entries, model, quotes.loc[quotes.strike.gt(1000)])
        self.assertEqual(len(out),len(entries))
        self.assertTrue(out.smile_status.eq('outside_observed_stencil_support').all())

    def test_model_carry_mismatch_is_recorded(self):
        model, entries, quotes = ah_fixture()
        entries['forward'] *= 1.01
        out = AHSmileHedges().calculate(entries, model, quotes)
        self.assertTrue(out.smile_status.eq('evaluation_failed').all())


class EmpiricalControls(unittest.TestCase):
    def setUp(self):
        self.panel, self.beta = empirical_fixture()
        self.estimator = PastOnlyEmpiricalMV(EmpiricalMVSettings(window_dates=8, minimum_dates=3, minimum_rows=9))

    def test_quadratic_coefficients_are_recovered_separately(self):
        predictions, fits, members = self.estimator.predict(self.panel,self.panel)
        ready = fits.loc[fits.status.eq('ready')]
        self.assertGreater(len(ready),0)
        for row in ready.itertuples(index=False):
            expected = self.beta if row.kind == 'call' else self.beta*np.array([.8,1.2,.7])
            np.testing.assert_allclose([row.a,row.b,row.c],expected,rtol=1e-9,atol=1e-10)
        self.assertTrue(predictions.empirical_mv_status.eq('warmup').any())
        self.assertTrue(predictions.empirical_mv_status.eq('ready').any())

    def test_training_sse_never_exceeds_black_on_training_data(self):
        _, fits, _ = self.estimator.predict(self.panel,self.panel)
        fit = fits.loc[fits.status.eq('ready')]
        self.assertTrue((fit.training_mv_sse <= fit.training_black_sse+1e-12).all())

    def test_all_training_endpoints_are_strictly_before_prediction(self):
        _, fits, membership = self.estimator.predict(self.panel,self.panel)
        merged = membership.merge(fits[['fit_id','prediction_timestamp']],on='fit_id',validate='many_to_one')
        self.assertTrue((pd.to_datetime(merged.training_endpoint,utc=True) <
                         pd.to_datetime(merged.prediction_timestamp,utc=True)).all())

    def test_future_price_changes_do_not_change_earlier_predictions(self):
        entry = self.panel.loc[self.panel.quote_date.eq('2023-08-08')]
        first = self.estimator.predict(entry,self.panel)[0]
        changed = self.panel.copy()
        changed.loc[changed.quote_date.ge('2023-08-08'),'end_mid'] += 10000
        changed.loc[changed.quote_date.ge('2023-08-08'),'end_spot'] *= 10
        pd.testing.assert_frame_equal(first,self.estimator.predict(entry,changed)[0])

    def test_contemporaneous_endpoint_is_excluded(self):
        entry = self.panel.loc[self.panel.quote_date.eq('2023-08-08')]
        training = self.panel.copy()
        same_close = training.end_timestamp.eq(entry.quote_timestamp_utc.iloc[0])
        first = self.estimator.predict(entry,training)[0]
        training.loc[same_close,'end_mid'] += 5000
        pd.testing.assert_frame_equal(first,self.estimator.predict(entry,training)[0])

    def test_rolling_window_is_enforced(self):
        _, fits, _ = self.estimator.predict(self.panel,self.panel)
        self.assertLessEqual(fits.available_dates.max(),8)
        last = fits.loc[fits.quote_date.eq(self.panel.quote_date.max())]
        self.assertTrue(last.available_dates.eq(8).all())

    def test_put_marks_do_not_change_call_coefficients(self):
        entry = self.panel.loc[self.panel.quote_date.eq('2023-08-08') & self.panel.kind.eq('call')]
        first = self.estimator.predict(entry,self.panel)[0]
        changed = self.panel.copy()
        changed.loc[changed.kind.eq('put'),'end_mid'] += 10000
        pd.testing.assert_frame_equal(first,self.estimator.predict(entry,changed)[0])

    def test_rank_deficient_training_is_not_a_fallback_black_strategy(self):
        training = self.panel.loc[self.panel.strike.eq(100)].copy()
        # Fix all entry maturities/carry/Greeks at one snapshot to give one delta.
        for kind in ('call','put'):
            selected = training.kind.eq(kind)
            first = training.loc[selected].iloc[0]
            for col in ('assumed_maturity_years','forward','discount_factor','black_iv','black_delta'):
                training.loc[selected,col] = first[col]
        entry = self.panel.loc[self.panel.quote_date.eq('2023-08-15')]
        estimator = PastOnlyEmpiricalMV(EmpiricalMVSettings(15,3,6))
        prediction, fits, _ = estimator.predict(entry,training)
        self.assertTrue(fits.status.eq('rank_deficient').all())
        self.assertNotIn('empirical_mv_delta', prediction)

    def test_duplicate_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.estimator.predict(self.panel,pd.concat([self.panel,self.panel.iloc[:1]]))

    def test_naive_training_timestamps_are_rejected(self):
        training = self.panel.copy()
        training['end_timestamp'] = pd.to_datetime(training.end_timestamp).dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError,'timezone-aware'):
            self.estimator.predict(self.panel,training)

    def test_saved_black_delta_mismatch_is_rejected(self):
        training = self.panel.copy()
        training['black_delta'] += .05
        with self.assertRaisesRegex(ValueError,'disagree'):
            self.estimator.predict(self.panel,training)


class GainAndLedgerControls(unittest.TestCase):
    def test_gain_uses_identical_entries_and_pooled_sse(self):
        rows = []
        for strategy, ids, errors in [('black',[1,2,3],[2.,4.,100.]),('lv_smile',[1,2],[1.,2.])]:
            for i,e in zip(ids,errors):
                rows.append(dict(scenario='mid', strategy=strategy, entry_id=str(i),
                    quote_date='2023-08-01', kind='call', calendar_days=21,
                    entry_spot_y=0.,black_entry_delta=.5,net_pnl=e,raw_mark_error=-e))
        out = paired_gains(pd.DataFrame(rows))
        overall = out.loc[out.dimension.eq('overall')].iloc[0]
        self.assertEqual(overall.comparisons,2)
        self.assertAlmostEqual(overall.gain,.75)
        self.assertAlmostEqual(overall.black_sse,20.)

    def test_gain_can_be_negative_and_is_not_clipped(self):
        row = dict(scenario='mid',entry_id='1',quote_date='2023-08-01',kind='put',
            calendar_days=21,entry_spot_y=0.,black_entry_delta=-.5)
        out = paired_gains(pd.DataFrame([dict(**row,strategy='black',net_pnl=1.,raw_mark_error=1.),
                                       dict(**row,strategy='lv_smile',net_pnl=2.,raw_mark_error=2.)]))
        self.assertEqual(out.loc[out.dimension.eq('overall'),'gain'].iloc[0],-3.)

    def test_zero_black_sse_is_explicit(self):
        row = dict(scenario='mid',entry_id='1',quote_date='2023-08-01',kind='put',
            calendar_days=21,entry_spot_y=0.,black_entry_delta=-.5)
        out = paired_gains(pd.DataFrame([dict(**row,strategy='black',net_pnl=0.,raw_mark_error=0.),
                                       dict(**row,strategy='lv_smile',net_pnl=1.,raw_mark_error=1.)]))
        self.assertTrue(out.gain.isna().all())
        self.assertTrue(out.gain_status.eq('zero_black_sse').all())

    def test_strategy_failures_remain_in_coverage_and_ledger_reconciles(self):
        panel, _ = empirical_fixture(2)
        panel['smile_status'] = panel['sticky_delta_status'] = panel['lv_smile_status'] = 'ready'
        panel['surface_sticky_strike_delta'] = panel['surface_sticky_delta_delta'] = panel.black_delta
        panel['lv_smile_delta'] = panel.black_delta-.1
        panel['empirical_mv_status'] = 'warmup'
        trials, coverage, gains = compare_hedges(panel)
        self.assertEqual(len(coverage),len(panel)*6*3)
        self.assertTrue(coverage.loc[coverage.strategy.eq('empirical_mv'),'status'].eq('warmup').all())
        self.assertLess(trials.max_reconciliation.max(),1e-10)
        self.assertLess(abs(trials.closed_form_error).max(),1e-10)
        self.assertNotIn('empirical_mv',set(trials.strategy))

    def test_not_matched_endpoints_do_not_remove_entries(self):
        panel, _ = empirical_fixture(1)
        panel['end_status'] = 'sample_end'
        trials, coverage, _ = compare_hedges(panel)
        self.assertTrue(trials.empty)
        self.assertEqual(len(coverage),len(panel)*6*3)
        self.assertTrue(coverage.status.eq('sample_end').all())


class InputProvenanceControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        for name in ('carry','calibration','panel'):
            (self.base/name).mkdir()
        pd.DataFrame([dict(quote_date='2023-08-01', root='UNKNOWN', expire_date='2023-11-01')]).to_csv(
            self.base/'carry'/'calibration_quotes.csv', index=False)
        pd.DataFrame([dict(quote_date='2023-08-01', root='UNKNOWN')]).to_csv(
            self.base/'carry'/'primary_carry.csv', index=False)
        self.write_audit('carry')
        pd.DataFrame([dict(quote_date='2023-08-01',root='UNKNOWN',status='failed',
                           model_file='',model_sha256='')]).to_csv(self.base/'calibration'/'model_manifest.csv',index=False)
        self.write_audit('calibration', {str(p):sha256(p) for p in (self.base/'carry').glob('*.csv')})
        empirical_fixture(1)[0].to_csv(self.base/'panel'/'delta_panel.csv', index=False)
        pins = {str(p):sha256(p) for p in (self.base/'calibration').iterdir() if p.is_file()}
        pins[str(self.base/'carry'/'primary_carry.csv')] = sha256(self.base/'carry'/'primary_carry.csv')
        self.write_audit('panel', pins)
        self.index = self.base/'run_index.json'
        self.index.write_text(json.dumps(dict(months={'2023-08':{name:str(self.base/name)
                            for name in ('carry','calibration','panel')}})))

    def tearDown(self):
        self.temporary.cleanup()

    def write_audit(self,stage,inputs=None):
        folder = self.base/stage
        (folder/'audit.json').write_text(json.dumps(dict(status='completed',input_sha256=inputs or {},
            output_sha256={p.name:sha256(p) for p in folder.iterdir() if p.name!='audit.json'})))

    def test_pins_and_failed_model_records_load_without_inventing_a_model(self):
        inputs = OptimalHedgeInputs(self.index,'2023-08')
        self.assertEqual(len(inputs.panel),6)
        self.assertEqual(inputs.models,{})
        inputs.verify_unchanged()

    def test_tampered_panel_is_rejected(self):
        path = self.base/'panel'/'delta_panel.csv'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'pin'):
            OptimalHedgeInputs(self.index,'2023-08')

    def test_unrelated_panel_calibration_is_rejected(self):
        path = self.base/'panel'/'audit.json'
        audit = json.loads(path.read_text())
        audit['input_sha256'] = {}
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'pin this calibration'):
            OptimalHedgeInputs(self.index,'2023-08')

    def test_inputs_changed_after_loading_are_detected(self):
        inputs = OptimalHedgeInputs(self.index,'2023-08')
        path = self.base/'carry'/'primary_carry.csv'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            inputs.verify_unchanged()

    def test_output_cannot_overlap_saved_inputs(self):
        inputs = OptimalHedgeInputs(self.index,'2023-08')
        with self.assertRaisesRegex(ValueError,'separate'):
            inputs.safe_output(self.base/'calibration'/'new')

    def test_incomplete_panel_is_rejected(self):
        path = self.base/'panel'/'audit.json'
        audit = json.loads(path.read_text()); audit['status']='running'
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'not complete'):
            OptimalHedgeInputs(self.index,'2023-08')


if __name__ == '__main__':
    unittest.main()
