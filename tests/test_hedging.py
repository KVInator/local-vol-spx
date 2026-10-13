from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pricing import BlackFixedIVBenchmark
from hedging import DailyDeltas
from selection import HedgeSettings, ContractSelector


def hedge_decisions_fixture():
    dates = ["2023-09-01", "2023-09-05", "2023-09-06"]
    fixing = pd.Timestamp("2023-09-29T20:00:00Z")
    quotes, carry = ([], [])
    black = BlackFixedIVBenchmark()
    for date in dates:
        stamp = pd.Timestamp(date + "T20:00:00Z")
        time = (fixing - stamp).total_seconds() / (365 * 86400)
        forward = 100 * np.exp(0.05 * time)
        discount = np.exp(-0.05 * time)
        carry.append(
            dict(
                quote_date=date,
                root="UNKNOWN",
                expire_date="2023-09-29",
                case="rate_5pct",
                carry_ready=True,
                spot=100.0,
                maturity_years=time,
                forward=forward,
                discount_factor=discount,
                annual_rate=0.05,
                quote_timestamp_utc=stamp,
                assumed_fixing_utc=fixing,
            )
        )
        for strike in [99.0, 100.0, 101.0]:
            for kind in ["call", "put"]:
                mark = black.price(forward, discount, strike, 0.2 * np.sqrt(time), kind)
                quotes.append(
                    dict(
                        quote_date=date,
                        root="UNKNOWN",
                        expire_date="2023-09-29",
                        strike=strike,
                        kind=kind,
                        quote_timestamp_utc=stamp,
                        assumed_fixing_utc=fixing,
                        assumed_maturity_years=time,
                        underlying_last=100.0,
                        bid=mark - 0.01,
                        ask=mark + 0.01,
                        mid=mark,
                        entry_research_candidate=True,
                    )
                )
    timeline = pd.DataFrame(
        dict(
            quote_date=dates,
            reference_session=True,
            session_policy="provisional_standard_session",
        )
    )
    return (pd.DataFrame(quotes), pd.DataFrame(carry), timeline)


class DailyHedgeTests(unittest.TestCase):

    def setUp(self):
        self.q, self.c, self.p = hedge_decisions_fixture()
        self.s = HedgeSettings(target_days=(21,), target_spot_y=(0.0,))
        self.selector = ContractSelector(self.s)

    def entries(self, date="2023-09-01"):
        return self.selector.select(self.q, self.p, [date])[0]

    def test_selection_is_current_date_only(self):
        original = self.entries()
        changed = self.q.loc[self.q.quote_date.eq("2023-09-01")].copy()
        result = self.selector.select(changed, self.p, ["2023-09-01"])[0]
        pd.testing.assert_frame_equal(original, result)

    def test_future_carry_does_not_change_entry_deltas(self):
        entries = self.entries()
        runner = DailyDeltas(self.s)
        base = runner.calculate(entries, self.c)
        changed = self.c.copy()
        changed.loc[changed.quote_date.ne("2023-09-01"), "forward"] = 900
        pd.testing.assert_frame_equal(base, runner.calculate(entries, changed))

    def test_same_contract_is_selected_once(self):
        entries, buckets = ContractSelector().select(self.q, self.p, ["2023-09-01"])
        self.assertEqual(len(entries), 6)
        self.assertEqual(len(buckets), 18)
        self.assertEqual(buckets.status.eq("duplicate_bucket").sum(), 12)
        self.assertFalse(entries.entry_id.duplicated().any())

    def test_call_and_put_have_distinct_identity(self):
        entries = self.entries()
        self.assertEqual(entries.strike.nunique(), 1)
        self.assertEqual(entries.contract_id.nunique(), 2)

    def test_missing_exit_does_not_remove_entry(self):
        entries = self.entries()
        next_quotes = self.q.loc[self.q.quote_date.ne("2023-09-05")]
        pairs = self.selector.match_next(entries, next_quotes, self.q.iloc[:0], self.p)
        self.assertEqual(len(pairs), len(entries))
        self.assertTrue(pairs.end_status.eq("next_quote_unavailable_in_pilot").all())
        self.assertTrue(pairs.end_date.eq("2023-09-05").all())

    def test_missing_session_is_not_bridged(self):
        policy = self.p.copy()
        policy.loc[policy.quote_date.eq("2023-09-05"), "session_policy"] = (
            "missing_observation"
        )
        pairs = self.selector.match_next(
            self.entries(), self.q, self.q.iloc[:0], policy
        )
        self.assertTrue(pairs.end_status.eq("next_session_excluded").all())
        self.assertTrue(pairs.end_date.eq("2023-09-05").all())

    def test_endpoint_option_kind_is_preserved(self):
        quotes = self.q.loc[
            ~(self.q.quote_date.eq("2023-09-05") & self.q.kind.eq("put"))
        ]
        pairs = self.selector.match_next(
            self.entries(), quotes, self.q.iloc[:0], self.p
        )
        joined = self.entries()[["entry_id", "kind"]].merge(pairs, on="entry_id")
        self.assertEqual(
            joined.loc[joined.kind.eq("call"), "end_status"].iloc[0], "matched"
        )
        self.assertEqual(
            joined.loc[joined.kind.eq("put"), "end_status"].iloc[0],
            "next_quote_unavailable_in_pilot",
        )

    def test_zero_bid_endpoint_is_retained(self):
        key = (
            self.q.quote_date.eq("2023-09-05")
            & self.q.kind.eq("call")
            & self.q.strike.eq(100)
        )
        bound = self.q.loc[key].copy()
        bound["bid"] = 0.0
        bound["ask"] = 0.08
        bound["mid"] = 0.04
        pairs = self.selector.match_next(
            self.entries(), self.q.loc[~key], bound, self.p
        )
        row = pairs.loc[pairs.end_quote_source.eq("zero_bid_bounds")].iloc[0]
        self.assertEqual(row.end_bid, 0)
        self.assertEqual(row.end_mid, 0.04)

    def test_sample_end_retains_entries(self):
        entries = self.entries("2023-09-06")
        pairs = self.selector.match_next(entries, self.q, self.q.iloc[:0], self.p)
        self.assertEqual(len(entries), len(pairs))
        self.assertTrue(pairs.end_status.eq("sample_end").all())

    def test_iv_and_delta_recover_analytical_black(self):
        black = BlackFixedIVBenchmark()
        for kind in ["call", "put"]:
            for y in [-0.02, 0.0, 0.02]:
                spot = 100
                forward = 102
                discount = 0.98
                time = 0.1
                sigma = 0.2
                strike = forward * np.exp(y)
                mark = black.price(
                    forward, discount, strike, sigma * np.sqrt(time), kind
                )
                result = black.estimate(
                    spot, strike, time, forward, discount, mark, kind
                )
                self.assertAlmostEqual(result["black_iv"], sigma, places=10)
                h = 0.001
                hi = black.price(
                    forward * (spot + h) / spot,
                    discount,
                    strike,
                    sigma * np.sqrt(time),
                    kind,
                )
                lo = black.price(
                    forward * (spot - h) / spot,
                    discount,
                    strike,
                    sigma * np.sqrt(time),
                    kind,
                )
                self.assertAlmostEqual(
                    result["black_delta"], (hi - lo) / (2 * h), places=8
                )

    def test_invalid_iv_target_is_flagged_without_clipping(self):
        result = BlackFixedIVBenchmark().estimate(100, 90, 0.1, 100, 1, 9, "call")
        self.assertEqual(result["black_status"], "no_finite_positive_iv")
        self.assertNotIn("black_delta", result)

    def test_model_unavailability_preserves_black_and_entries(self):
        result = DailyDeltas(self.s).calculate(self.entries(), self.c)
        self.assertTrue(result.black_status.eq("ready").all())
        self.assertTrue(result.ah_status.eq("model_unavailable").all())
        self.assertEqual(len(result), 2)

    def test_put_ah_greeks_use_exact_parity(self):
        entries = self.entries()
        item = self.c.iloc[0]
        model = SimpleNamespace(
            spot=100,
            maturities=np.array([item.maturity_years]),
            forwards=np.array([item.forward]),
            discounts=np.array([item.discount_factor]),
        )
        value = SimpleNamespace(
            price=entries.loc[entries.kind.eq("call"), "mid"].iloc[0],
            delta=0.55,
            gamma=0.02,
        )
        solver = SimpleNamespace(
            last_diagnostics={"actual_steps": 8},
            solve_many=lambda strikes, time: [
                SimpleNamespace(greeks=lambda spot: value) for k in strikes
            ],
        )
        with patch("hedging.AHShortEndVariance"), patch(
            "hedging.AHBackwardPricer", return_value=solver
        ):
            result = DailyDeltas(self.s).calculate(
                entries, self.c, model, progress=lambda text: None
            )
        call = result.loc[result.kind.eq("call")].iloc[0]
        put = result.loc[result.kind.eq("put")].iloc[0]
        self.assertAlmostEqual(
            call.ah_delta - put.ah_delta, item.discount_factor * item.forward / 100
        )
        self.assertAlmostEqual(
            call.ah_price - put.ah_price, item.discount_factor * (item.forward - 100)
        )
        self.assertEqual(call.ah_gamma, put.ah_gamma)

    def test_rejects_duplicate_quotes_and_changed_maturity(self):
        with self.assertRaises(ValueError):
            self.entries_from(pd.concat([self.q, self.q.iloc[:1]]))
        changed = self.q.copy()
        changed.loc[0, "assumed_maturity_years"] += 1 / 365
        with self.assertRaises(ValueError):
            self.entries_from(changed)

    def test_missing_current_carry_keeps_entry_status(self):
        absent = self.c.loc[self.c.quote_date.ne("2023-09-01")]
        result = DailyDeltas(self.s).calculate(self.entries(), absent)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.black_status.eq("carry_unavailable").all())

    def test_wrong_current_model_is_rejected(self):
        item = self.c.iloc[0]
        model = SimpleNamespace(
            spot=101,
            maturities=np.array([item.maturity_years]),
            forwards=np.array([item.forward]),
            discounts=np.array([item.discount_factor]),
        )
        with self.assertRaises(ValueError):
            DailyDeltas(self.s).calculate(self.entries(), self.c, model)

    def test_failed_ah_solve_keeps_black_and_entries(self):
        item = self.c.iloc[0]
        model = SimpleNamespace(
            spot=100,
            maturities=np.array([item.maturity_years]),
            forwards=np.array([item.forward]),
            discounts=np.array([item.discount_factor]),
        )

        def fail(strikes, time):
            raise ArithmeticError("control failure")

        solver = SimpleNamespace(solve_many=fail)
        with patch("hedging.AHShortEndVariance"), patch(
            "hedging.AHBackwardPricer", return_value=solver
        ):
            result = DailyDeltas(self.s).calculate(
                self.entries(), self.c, model, progress=lambda text: None
            )
        self.assertEqual(len(result), 2)
        self.assertTrue(result.black_status.eq("ready").all())
        self.assertTrue(result.ah_status.eq("solve_failed").all())
        self.assertTrue(result.ah_message.str.contains("control failure").all())

    def entries_from(self, quotes):
        return self.selector.select(quotes, self.p, ["2023-09-01"])[0]


from scipy.optimize import brentq
from scipy.special import ndtr
from surface import AHGrid, AndreasenHugeSurface
from hedging import (
    AHSmileHedges,
    SmileSettings,
    EmpiricalMVSettings,
    PastOnlyEmpiricalMV,
    black_greeks,
    smile_conventions,
    analytical_controls,
)
from fixtures import empirical_fixture


def hedge_strategies_ah_fixture():
    t = np.array([14, 28, 45]) / 365
    model = AndreasenHugeSurface(
        AHGrid(1.0, 8000),
        100.0,
        t,
        100 * np.exp(0.01 * t),
        np.exp(-0.05 * t),
        [np.array([-0.2, 0.2]) for _ in t],
        [np.log([0.2, 0.2]) for _ in t],
    )
    rows, quotes = ([], [])
    for time in t:
        expiry = f"T{time}"
        f, d = (model.forward(time), model.discount_factor(time))
        for k in f * np.exp(np.linspace(-0.08, 0.08, 25)):
            quotes.append(dict(expire_date=expiry, strike=k))
        for kind in ("call", "put"):
            for y in (-0.02, 0.0, 0.02):
                k = f * np.exp(y)
                iv = float(model.implied_volatility(np.array([k]), time)[0])
                b = black_greeks(100.0, k, time, f, d, iv, kind)
                rows.append(
                    dict(
                        entry_id=f"{time}-{kind}-{y}",
                        underlying_last=100.0,
                        strike=k,
                        assumed_maturity_years=time,
                        forward=f,
                        discount_factor=d,
                        kind=kind,
                        black_iv=iv,
                        black_delta=b["delta"],
                        black_status="ready",
                        expire_date=expiry,
                    )
                )
    return (model, pd.DataFrame(rows), pd.DataFrame(quotes))


class SmileControls(unittest.TestCase):

    def test_known_smiles_match_direct_spot_bumps(self):
        controls = analytical_controls()
        self.assertLess(controls.bump_error_1e_4.max(), 1e-06)
        self.assertTrue((controls.bump_error_1e_4 < controls.bump_error_1e_3).all())

    def test_flat_smile_all_surface_deltas_are_black(self):
        g = smile_conventions(100, 101, 0.2, 102, 0.99, 0.25, 0.0, "call")
        self.assertEqual(
            g["surface_sticky_strike_delta"], g["surface_sticky_delta_delta"]
        )
        self.assertEqual(g["surface_sticky_strike_delta"], g["surface_hw_lv_delta"])

    def test_negative_skew_sticky_delta_and_hw_lv_have_opposite_signs(self):
        g = smile_conventions(100, 101, 0.2, 102, 0.99, 0.25, -0.3, "call")
        self.assertGreater(
            g["surface_sticky_delta_delta"], g["surface_sticky_strike_delta"]
        )
        self.assertLess(g["surface_hw_lv_delta"], g["surface_sticky_strike_delta"])

    def test_call_put_surface_parity(self):
        call = smile_conventions(100, 101, 0.2, 102, 0.99, 0.25, -0.3, "call")
        put = smile_conventions(100, 101, 0.2, 102, 0.99, 0.25, -0.3, "put")
        for name in (
            "surface_sticky_strike_delta",
            "surface_sticky_delta_delta",
            "surface_hw_lv_delta",
        ):
            self.assertAlmostEqual(call[name] - put[name], 0.99 * 102 / 100)
        self.assertEqual(call["surface_vega"], put["surface_vega"])

    def test_vega_is_per_decimal_volatility(self):
        args = (100, 101, 0.2, 102, 0.99)
        g = black_greeks(*args, 0.25, "call")
        fd = (
            black_greeks(*args, 0.250001, "call")["price"]
            - black_greeks(*args, 0.249999, "call")["price"]
        ) / 2e-06
        self.assertAlmostEqual(g["vega"], fd, places=7)

    def test_sticky_forward_delta_implicit_repricing(self):
        spot, k, t, f, d = (100.0, 101.0, 0.2, 102.0, 0.99)
        sigma, sigma_y = (0.25, -0.3)
        y0 = np.log(k / f)

        def q(y):
            vol = sigma + sigma_y * (y - y0)
            return ndtr(-y / (vol * np.sqrt(t)) + vol * np.sqrt(t) / 2)

        def price(s):
            ff = f * s / spot

            def equation(y):
                vol = sigma + sigma_y * (y - y0)
                return q(y) - ndtr(
                    np.log(ff / k) / (vol * np.sqrt(t)) + vol * np.sqrt(t) / 2
                )

            y = brentq(equation, y0 - 0.02, y0 + 0.02)
            return black_greeks(s, k, t, ff, d, sigma + sigma_y * (y - y0), "call")[
                "price"
            ]

        fd = (price(100.001) - price(99.999)) / 0.002
        g = smile_conventions(spot, k, t, f, d, sigma, sigma_y, "call")
        self.assertAlmostEqual(fd, g["surface_sticky_delta_delta"], places=7)

    def test_noninvertible_delta_coordinate_is_detectable(self):
        g = smile_conventions(100, 130, 0.2, 102, 0.99, 0.25, 3.0, "call")
        self.assertGreater(g["surface_delta_coordinate_slope"], 0)

    def test_saved_ah_smiles_pass_bump_and_width_controls(self):
        model, entries, quotes = hedge_strategies_ah_fixture()
        out = AHSmileHedges().calculate(entries, model, quotes)
        self.assertTrue(out.smile_status.eq("ready").all(), out.to_string())
        self.assertTrue(out.sticky_delta_status.eq("ready").all())
        self.assertTrue(out.lv_smile_status.eq("ready").all())
        self.assertLess(
            out.smile_chain_minus_bump.max(), SmileSettings().delta_tolerance
        )

    def test_smile_does_not_read_endpoint_marks(self):
        model, entries, quotes = hedge_strategies_ah_fixture()
        first = AHSmileHedges().calculate(entries, model, quotes)
        entries["end_mid"] = np.arange(len(entries)) * 1000
        entries["end_spot"] = 1.0
        pd.testing.assert_frame_equal(
            first, AHSmileHedges().calculate(entries, model, quotes)
        )

    def test_quote_support_failure_keeps_every_entry(self):
        model, entries, quotes = hedge_strategies_ah_fixture()
        out = AHSmileHedges().calculate(
            entries, model, quotes.loc[quotes.strike.gt(1000)]
        )
        self.assertEqual(len(out), len(entries))
        self.assertTrue(out.smile_status.eq("outside_observed_stencil_support").all())

    def test_model_carry_mismatch_is_recorded(self):
        model, entries, quotes = hedge_strategies_ah_fixture()
        entries["forward"] *= 1.01
        out = AHSmileHedges().calculate(entries, model, quotes)
        self.assertTrue(out.smile_status.eq("evaluation_failed").all())


class EmpiricalControls(unittest.TestCase):

    def setUp(self):
        self.panel, self.beta = empirical_fixture()
        self.estimator = PastOnlyEmpiricalMV(
            EmpiricalMVSettings(window_dates=8, minimum_dates=3, minimum_rows=9)
        )

    def test_quadratic_coefficients_are_recovered_separately(self):
        predictions, fits, members = self.estimator.predict(self.panel, self.panel)
        ready = fits.loc[fits.status.eq("ready")]
        self.assertGreater(len(ready), 0)
        for row in ready.itertuples(index=False):
            expected = (
                self.beta
                if row.kind == "call"
                else self.beta * np.array([0.8, 1.2, 0.7])
            )
            np.testing.assert_allclose(
                [row.a, row.b, row.c], expected, rtol=1e-09, atol=1e-10
            )
        self.assertTrue(predictions.empirical_mv_status.eq("warmup").any())
        self.assertTrue(predictions.empirical_mv_status.eq("ready").any())

    def test_training_sse_never_exceeds_black_on_training_data(self):
        _, fits, _ = self.estimator.predict(self.panel, self.panel)
        fit = fits.loc[fits.status.eq("ready")]
        self.assertTrue((fit.training_mv_sse <= fit.training_black_sse + 1e-12).all())

    def test_all_training_endpoints_are_strictly_before_prediction(self):
        _, fits, membership = self.estimator.predict(self.panel, self.panel)
        merged = membership.merge(
            fits[["fit_id", "prediction_timestamp"]],
            on="fit_id",
            validate="many_to_one",
        )
        self.assertTrue(
            (
                pd.to_datetime(merged.training_endpoint, utc=True)
                < pd.to_datetime(merged.prediction_timestamp, utc=True)
            ).all()
        )

    def test_future_price_changes_do_not_change_earlier_predictions(self):
        entry = self.panel.loc[self.panel.quote_date.eq("2023-08-08")]
        first = self.estimator.predict(entry, self.panel)[0]
        changed = self.panel.copy()
        changed.loc[changed.quote_date.ge("2023-08-08"), "end_mid"] += 10000
        changed.loc[changed.quote_date.ge("2023-08-08"), "end_spot"] *= 10
        pd.testing.assert_frame_equal(first, self.estimator.predict(entry, changed)[0])

    def test_contemporaneous_endpoint_is_excluded(self):
        entry = self.panel.loc[self.panel.quote_date.eq("2023-08-08")]
        training = self.panel.copy()
        same_close = training.end_timestamp.eq(entry.quote_timestamp_utc.iloc[0])
        first = self.estimator.predict(entry, training)[0]
        training.loc[same_close, "end_mid"] += 5000
        pd.testing.assert_frame_equal(first, self.estimator.predict(entry, training)[0])

    def test_rolling_window_is_enforced(self):
        _, fits, _ = self.estimator.predict(self.panel, self.panel)
        self.assertLessEqual(fits.available_dates.max(), 8)
        last = fits.loc[fits.quote_date.eq(self.panel.quote_date.max())]
        self.assertTrue(last.available_dates.eq(8).all())

    def test_put_marks_do_not_change_call_coefficients(self):
        entry = self.panel.loc[
            self.panel.quote_date.eq("2023-08-08") & self.panel.kind.eq("call")
        ]
        first = self.estimator.predict(entry, self.panel)[0]
        changed = self.panel.copy()
        changed.loc[changed.kind.eq("put"), "end_mid"] += 10000
        pd.testing.assert_frame_equal(first, self.estimator.predict(entry, changed)[0])

    def test_rank_deficient_training_is_not_a_fallback_black_strategy(self):
        training = self.panel.loc[self.panel.strike.eq(100)].copy()
        for kind in ("call", "put"):
            selected = training.kind.eq(kind)
            first = training.loc[selected].iloc[0]
            for col in (
                "assumed_maturity_years",
                "forward",
                "discount_factor",
                "black_iv",
                "black_delta",
            ):
                training.loc[selected, col] = first[col]
        entry = self.panel.loc[self.panel.quote_date.eq("2023-08-15")]
        estimator = PastOnlyEmpiricalMV(EmpiricalMVSettings(15, 3, 6))
        prediction, fits, _ = estimator.predict(entry, training)
        self.assertTrue(fits.status.eq("rank_deficient").all())
        self.assertNotIn("empirical_mv_delta", prediction)

    def test_duplicate_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.estimator.predict(
                self.panel, pd.concat([self.panel, self.panel.iloc[:1]])
            )

    def test_naive_training_timestamps_are_rejected(self):
        training = self.panel.copy()
        training["end_timestamp"] = pd.to_datetime(
            training.end_timestamp
        ).dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.estimator.predict(self.panel, training)

    def test_saved_black_delta_mismatch_is_rejected(self):
        training = self.panel.copy()
        training["black_delta"] += 0.05
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.estimator.predict(self.panel, training)
