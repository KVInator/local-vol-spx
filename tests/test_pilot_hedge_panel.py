from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pilot_hedge_panel import (
    BlackFixedIVBenchmark,
    DailyHedgeDeltas,
    HedgePanelSettings,
    PilotContractSelector,
)


def fixture():
    dates = [
        "2023-09-01",
        "2023-09-05",
        "2023-09-06",
    ]
    fixing = pd.Timestamp("2023-09-29T20:00:00Z")
    quotes, carry = [], []
    black = BlackFixedIVBenchmark()

    for date in dates:
        stamp = pd.Timestamp(date + "T20:00:00Z")
        time = (
            fixing - stamp
        ).total_seconds() / (365 * 86400)

        forward = 100 * np.exp(0.05 * time)
        discount = np.exp(-0.05 * time)

        carry.append(dict(
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
        ))

        for strike in [99.0, 100.0, 101.0]:
            for kind in ["call", "put"]:
                mark = black.price(
                    forward,
                    discount,
                    strike,
                    0.2 * np.sqrt(time),
                    kind,
                )

                quotes.append(dict(
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
                ))

    timeline = pd.DataFrame(dict(
        quote_date=dates,
        reference_session=True,
        session_policy="provisional_standard_session",
    ))

    return pd.DataFrame(quotes), pd.DataFrame(carry), timeline


class PilotHedgePanelTests(unittest.TestCase):
    def setUp(self):
        self.q, self.c, self.p = fixture()
        self.s = HedgePanelSettings(
            target_days=(21,),
            target_spot_y=(0.0,),
        )
        self.selector = PilotContractSelector(self.s)

    def entries(self, date="2023-09-01"):
        return self.selector.select(
            self.q, self.p, [date]
        )[0]

    def test_selection_is_current_date_only(self):
        original = self.entries()
        changed = self.q.loc[
            self.q.quote_date.eq("2023-09-01")
        ].copy()

        result = self.selector.select(
            changed, self.p, ["2023-09-01"]
        )[0]

        pd.testing.assert_frame_equal(original, result)

    def test_future_carry_does_not_change_entry_deltas(self):
        entries = self.entries()
        runner = DailyHedgeDeltas(self.s)
        base = runner.calculate(entries, self.c)

        changed = self.c.copy()
        changed.loc[
            changed.quote_date.ne("2023-09-01"), "forward"
        ] = 900

        pd.testing.assert_frame_equal(
            base, runner.calculate(entries, changed)
        )

    def test_same_contract_is_selected_once(self):
        entries, buckets = PilotContractSelector().select(
            self.q, self.p, ["2023-09-01"]
        )

        self.assertEqual(len(entries), 6)
        self.assertEqual(len(buckets), 18)
        self.assertEqual(
            buckets.status.eq("duplicate_bucket").sum(), 12
        )
        self.assertFalse(entries.entry_id.duplicated().any())

    def test_call_and_put_have_distinct_identity(self):
        entries = self.entries()
        self.assertEqual(entries.strike.nunique(), 1)
        self.assertEqual(entries.contract_id.nunique(), 2)

    def test_missing_exit_does_not_remove_entry(self):
        entries = self.entries()
        next_quotes = self.q.loc[
            self.q.quote_date.ne("2023-09-05")
        ]

        pairs = self.selector.match_next(
            entries, next_quotes, self.q.iloc[:0], self.p
        )

        self.assertEqual(len(pairs), len(entries))
        self.assertTrue(
            pairs.end_status.eq(
                "next_quote_unavailable_in_pilot"
            ).all()
        )
        self.assertTrue(
            pairs.end_date.eq("2023-09-05").all()
        )

    def test_missing_session_is_not_bridged(self):
        policy = self.p.copy()
        policy.loc[
            policy.quote_date.eq("2023-09-05"),
            "session_policy",
        ] = "missing_observation"

        pairs = self.selector.match_next(
            self.entries(), self.q, self.q.iloc[:0], policy
        )

        self.assertTrue(
            pairs.end_status.eq("next_session_excluded").all()
        )
        self.assertTrue(
            pairs.end_date.eq("2023-09-05").all()
        )

    def test_endpoint_option_kind_is_preserved(self):
        quotes = self.q.loc[~(
            self.q.quote_date.eq("2023-09-05")
            & self.q.kind.eq("put")
        )]

        pairs = self.selector.match_next(
            self.entries(), quotes, self.q.iloc[:0], self.p
        )
        joined = self.entries()[
            ["entry_id", "kind"]
        ].merge(pairs, on="entry_id")

        self.assertEqual(
            joined.loc[
                joined.kind.eq("call"), "end_status"
            ].iloc[0],
            "matched",
        )
        self.assertEqual(
            joined.loc[
                joined.kind.eq("put"), "end_status"
            ].iloc[0],
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
            self.entries(),
            self.q.loc[~key],
            bound,
            self.p,
        )

        row = pairs.loc[
            pairs.end_quote_source.eq("zero_bid_bounds")
        ].iloc[0]

        self.assertEqual(row.end_bid, 0)
        self.assertEqual(row.end_mid, 0.04)

    def test_sample_end_retains_entries(self):
        entries = self.entries("2023-09-06")
        pairs = self.selector.match_next(
            entries, self.q, self.q.iloc[:0], self.p
        )

        self.assertEqual(len(entries), len(pairs))
        self.assertTrue(
            pairs.end_status.eq("sample_end").all()
        )

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
                    forward, discount, strike,
                    sigma * np.sqrt(time), kind,
                )
                result = black.estimate(
                    spot, strike, time,
                    forward, discount, mark, kind,
                )

                self.assertAlmostEqual(
                    result["black_iv"], sigma, places=10
                )

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
                    result["black_delta"],
                    (hi - lo) / (2 * h),
                    places=8,
                )

    def test_invalid_iv_target_is_flagged_without_clipping(self):
        result = BlackFixedIVBenchmark().estimate(
            100, 90, 0.1, 100, 1, 9, "call"
        )

        self.assertEqual(
            result["black_status"],
            "no_finite_positive_iv",
        )
        self.assertNotIn("black_delta", result)

    def test_model_unavailability_preserves_black_and_entries(self):
        result = DailyHedgeDeltas(self.s).calculate(
            self.entries(), self.c
        )

        self.assertTrue(result.black_status.eq("ready").all())
        self.assertTrue(
            result.ah_status.eq("model_unavailable").all()
        )
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
            price=entries.loc[
                entries.kind.eq("call"), "mid"
            ].iloc[0],
            delta=0.55,
            gamma=0.02,
        )
        solver = SimpleNamespace(
            last_diagnostics={"actual_steps": 8},
            solve_many=lambda strikes, time: [
                SimpleNamespace(greeks=lambda spot: value)
                for k in strikes
            ],
        )

        with (
            patch("pilot_hedge_panel.AHShortEndVariance"),
            patch(
                "pilot_hedge_panel.AHBackwardPricer",
                return_value=solver,
            ),
        ):
            result = DailyHedgeDeltas(self.s).calculate(
                entries, self.c, model,
                progress=lambda text: None,
            )

        call = result.loc[result.kind.eq("call")].iloc[0]
        put = result.loc[result.kind.eq("put")].iloc[0]

        self.assertAlmostEqual(
            call.ah_delta - put.ah_delta,
            item.discount_factor * item.forward / 100,
        )
        self.assertAlmostEqual(
            call.ah_price - put.ah_price,
            item.discount_factor * (item.forward - 100),
        )
        self.assertEqual(call.ah_gamma, put.ah_gamma)

    def test_rejects_duplicate_quotes_and_changed_maturity(self):
        with self.assertRaises(ValueError):
            self.entries_from(
                pd.concat([self.q, self.q.iloc[:1]])
            )

        changed = self.q.copy()
        changed.loc[0, "assumed_maturity_years"] += 1 / 365

        with self.assertRaises(ValueError):
            self.entries_from(changed)

    def test_missing_current_carry_keeps_entry_status(self):
        absent = self.c.loc[
            self.c.quote_date.ne("2023-09-01")
        ]
        result = DailyHedgeDeltas(self.s).calculate(
            self.entries(), absent
        )

        self.assertEqual(len(result), 2)
        self.assertTrue(
            result.black_status.eq("carry_unavailable").all()
        )

    def test_wrong_current_model_is_rejected(self):
        item = self.c.iloc[0]
        model = SimpleNamespace(
            spot=101,
            maturities=np.array([item.maturity_years]),
            forwards=np.array([item.forward]),
            discounts=np.array([item.discount_factor]),
        )

        with self.assertRaises(ValueError):
            DailyHedgeDeltas(self.s).calculate(
                self.entries(), self.c, model
            )

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

        with (
            patch("pilot_hedge_panel.AHShortEndVariance"),
            patch(
                "pilot_hedge_panel.AHBackwardPricer",
                return_value=solver,
            ),
        ):
            result = DailyHedgeDeltas(self.s).calculate(
                self.entries(), self.c, model,
                progress=lambda text: None,
            )

        self.assertEqual(len(result), 2)
        self.assertTrue(result.black_status.eq("ready").all())
        self.assertTrue(
            result.ah_status.eq("solve_failed").all()
        )
        self.assertTrue(
            result.ah_message.str.contains("control failure").all()
        )

    def entries_from(self, quotes):
        return self.selector.select(
            quotes, self.p, ["2023-09-01"]
        )[0]


if __name__ == "__main__":
    unittest.main()