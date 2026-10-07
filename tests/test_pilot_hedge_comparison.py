import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hedge_ledger import SelfFinancingHedgeLedger
from pilot_hedge_comparison import (
    PilotComparisonSettings,
    PilotHedgeComparison,
    summarize,
)


def panel():
    return pd.DataFrame([{
        "entry_id": "entry1",
        "contract_id": "contract1",
        "quote_date": "2023-09-01",
        "end_date": "2023-09-05",
        "root": "UNKNOWN",
        "expire_date": "2023-09-22",
        "strike": 100.0,
        "kind": "call",
        "calendar_days": 21,
        "entry_spot_y": 0.0,
        "quote_timestamp_utc": "2023-09-01T20:00:00Z",
        "end_timestamp": "2023-09-05T20:00:00Z",
        "assumed_fixing_utc": "2023-09-22T20:00:00Z",
        "underlying_last": 100.0,
        "end_spot": 110.0,
        "mid": 5.0,
        "bid": 4.8,
        "ask": 5.2,
        "end_mid": 8.0,
        "end_bid": 7.7,
        "end_ask": 8.3,
        "end_status": "matched",
        "black_status": "ready",
        "ah_status": "ready",
        "black_delta": 0.5,
        "ah_delta": 0.4,
    }])


class PilotHedgeComparisonTests(unittest.TestCase):
    def compare(self, data=None, **settings):
        return PilotHedgeComparison(
            PilotComparisonSettings(**settings)
        ).run(panel() if data is None else data)

    def test_manual_short_call_and_paired_errors(self):
        result = self.compare(lend_rate=0, borrow_rate=0)
        mid = result.trials.loc[
            result.trials.scenario.eq("mid")
        ].set_index("strategy")

        self.assertAlmostEqual(mid.loc["unhedged", "net_pnl"], -3)
        self.assertAlmostEqual(mid.loc["black", "net_pnl"], 2)
        self.assertAlmostEqual(mid.loc["ah", "net_pnl"], 1)

        pair = result.paired.loc[
            result.paired.scenario.eq("mid")
        ].iloc[0]
        self.assertAlmostEqual(pair.abs_error_improvement, 1)
        self.assertAlmostEqual(pair.squared_error_improvement, 3)

    def test_observed_spreads_and_fee_funding(self):
        result = self.compare(
            lend_rate=0.03, borrow_rate=0.08, hedge_fee_bps=2
        )
        values = result.trials.loc[
            result.trials.strategy.eq("black")
        ].set_index("scenario")

        expected = (
            (4.8 - 50 - 0.01) * np.exp(0.08 * 4 / 365)
            + 55 - 8.3 - 0.011
        )
        self.assertAlmostEqual(
            values.loc["option_spread_fee", "net_pnl"], expected
        )
        self.assertAlmostEqual(
            values.loc["option_spread_fee", "direct_costs"], 0.521
        )

        rows = result.ledgers.loc[
            result.ledgers.scenario.eq("option_spread_fee")
            & result.ledgers.strategy.eq("black")
        ]
        np.testing.assert_allclose(rows.option_fill, [4.8, 8.3])

    def test_put_has_negative_hedge_and_lends_cash(self):
        data = panel()
        data["kind"] = "put"
        data["black_delta"] = -0.5
        data["ah_delta"] = -0.4

        values = self.compare(data).trials
        black = values.loc[
            values.scenario.eq("mid") & values.strategy.eq("black")
        ].iloc[0]

        self.assertAlmostEqual(
            black.net_pnl,
            55 * np.exp(0.05 * 4 / 365) - 55 - 8,
        )
        self.assertGreater(black.total_interest, 0)

    def test_elapsed_time_includes_weekend_and_dst(self):
        data = panel()
        data["quote_date"] = "2023-11-03"
        data["end_date"] = "2023-11-06"
        data["expire_date"] = "2023-11-24"
        data["quote_timestamp_utc"] = "2023-11-03T20:00:00Z"
        data["end_timestamp"] = "2023-11-06T21:00:00Z"
        data["assumed_fixing_utc"] = "2023-11-24T21:00:00Z"

        values = self.compare(data).trials
        unhedged = values.loc[
            values.scenario.eq("mid") & values.strategy.eq("unhedged")
        ].iloc[0]

        self.assertAlmostEqual(
            unhedged.net_pnl,
            5 * np.exp(0.05 * 73 / (365 * 24)) - 8,
        )

    def test_sample_end_and_unready_entries_remain_in_coverage(self):
        data = pd.concat([
            panel(),
            panel().assign(
                entry_id="last",
                end_status="sample_end",
                end_date=None,
                end_mid=np.nan,
            ),
            panel().assign(
                entry_id="failed",
                ah_status="solve_failed",
            ),
        ])
        result = self.compare(data)

        self.assertEqual(len(result.trials), 9)
        self.assertEqual(len(result.coverage), 9)
        self.assertEqual(
            result.coverage.status.value_counts().to_dict(),
            {
                "compared": 3,
                "sample_end": 3,
                "ah_not_ready": 3,
            },
        )

    def test_no_endpoint_delta_is_used(self):
        first = self.compare()
        data = panel().assign(
            end_black_delta=1e8,
            end_ah_delta=-1e8,
        )
        second = self.compare(data)

        pd.testing.assert_frame_equal(first.trials, second.trials)
        closing = second.ledgers.loc[
            second.ledgers.event.eq("liquidation"), "delta"
        ]
        self.assertTrue(closing.eq(0).all())

    def test_model_prices_do_not_replace_observed_marks(self):
        first = self.compare()
        second = self.compare(
            panel().assign(
                black_price=999,
                ah_price=888,
                end_model_price=777,
            )
        )
        pd.testing.assert_frame_equal(first.trials, second.trials)

    def test_zero_bid_endpoint_is_permitted(self):
        data = panel().assign(
            end_bid=0.0,
            end_ask=0.2,
            end_mid=0.1,
        )
        result = self.compare(data)
        self.assertTrue(result.coverage.status.eq("compared").all())

    def test_rejects_corrupted_ready_data(self):
        changes = [
            {"black_delta": np.nan},
            {"bid": 0},
            {"end_mid": 9},
            {"end_ask": 7},
            {"end_spot": 0},
            {"kind": "other"},
            {"end_timestamp": "2023-09-05T20:00:00"},
            {"end_timestamp": "2023-09-22T20:00:00Z"},
            {"end_date": "2023-09-06"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare(panel().assign(**change))

    def test_duplicate_entries_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(pd.concat([panel(), panel()]))

    def test_accounting_failure_does_not_leave_partial_comparison(self):
        original = SelfFinancingHedgeLedger.run

        def fail_black(engine, observations):
            if observations.delta.iloc[0] == 0.5:
                raise ArithmeticError("injected failure")
            return original(engine, observations)

        with patch.object(
            SelfFinancingHedgeLedger, "run", fail_black
        ):
            result = self.compare()

        self.assertTrue(result.trials.empty)
        self.assertTrue(result.ledgers.empty)
        self.assertTrue(
            result.coverage.status.eq("accounting_failed").all()
        )

    def test_rms_is_zero_centred_and_date_count_is_not_row_count(self):
        result = self.compare(lend_rate=0, borrow_rate=0)
        trial = result.trials.loc[
            result.trials.scenario.eq("mid")
            & result.trials.strategy.eq("ah")
        ]
        rows = pd.concat([
            trial.assign(entry_id="a", net_pnl=-3),
            trial.assign(entry_id="b", net_pnl=1),
        ])
        summary = summarize(
            rows, ["scenario", "strategy"]
        ).iloc[0]

        self.assertAlmostEqual(summary.rms_net_pnl, np.sqrt(5))
        self.assertAlmostEqual(summary.mean_abs_net_pnl, 2)
        self.assertEqual(summary.entry_dates, 1)

    def test_inputs_are_unchanged(self):
        data = panel()
        before = data.copy(deep=True)
        self.compare(data)
        pd.testing.assert_frame_equal(data, before)

    def test_invalid_settings_are_rejected(self):
        for change in (
            {"hedge_fee_bps": 0},
            {"hedge_fee_bps": -1},
            {"lend_rate": np.nan},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                PilotComparisonSettings(**change)


if __name__ == "__main__":
    unittest.main()