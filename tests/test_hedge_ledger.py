import unittest

import numpy as np
import pandas as pd

from hedge_ledger import HedgeLedgerSettings, SelfFinancingHedgeLedger


def manual_path():
    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2023-09-01", periods=3, tz="UTC"
        ),
        "spot": [100.0, 110.0, 105.0],
        "option_mark": [5.0, 8.0, 6.0],
        "delta": [0.5, 0.25, 0.9],
        "contract_id": "research_call",
    })


class HedgeLedgerTests(unittest.TestCase):
    def run_path(self, data=None, **settings):
        return SelfFinancingHedgeLedger(
            HedgeLedgerSettings(**settings)
        ).run(manual_path() if data is None else data)

    def test_linear_claim_is_exactly_hedged(self):
        data = manual_path()
        data["option_mark"] = data["spot"]
        data["delta"] = 1.0
        result = self.run_path(data)

        np.testing.assert_allclose(
            result.ledger["equity"], 0, atol=1e-12
        )
        self.assertEqual(result.summary["final_cash"], 0)

    def test_cash_claim_matches_lending(self):
        data = manual_path()
        times = np.arange(3) / 365
        data["option_mark"] = 10 * np.exp(0.05 * times)
        data["delta"] = 0.0
        result = self.run_path(data)

        self.assertAlmostEqual(
            result.summary["net_pnl"], 0, places=12
        )
        self.assertGreater(result.summary["total_interest"], 0)

    def test_deterministic_itm_call_matches_borrowing(self):
        days = np.array([0, 7, 30])
        data = pd.DataFrame({
            "timestamp": (
                pd.Timestamp("2023-09-01", tz="UTC")
                + pd.to_timedelta(days, unit="D")
            ),
            "spot": 100 * np.exp(0.05 * days / 365),
            "option_mark": (
                100 * np.exp(0.05 * days / 365)
                - 90 * np.exp(-0.05 * (30 - days) / 365)
            ),
            "delta": 1.0,
        })
        result = self.run_path(data)

        self.assertAlmostEqual(
            result.summary["net_pnl"], 0, places=11
        )
        self.assertLess(result.summary["total_interest"], 0)

    def test_rebalance_uses_previous_hedge(self):
        result = self.run_path(lend_rate=0, borrow_rate=0)

        np.testing.assert_allclose(
            result.ledger["cash"], [-45, -17.5, 2.75]
        )
        np.testing.assert_allclose(
            result.ledger["hedge_pnl"], [0, 5, -1.25]
        )
        np.testing.assert_allclose(
            result.ledger["option_pnl"], [0, -3, 2]
        )
        self.assertAlmostEqual(result.summary["net_pnl"], 2.75)
        self.assertEqual(result.summary["final_option_position"], 0)
        self.assertEqual(result.summary["final_hedge_position"], 0)

    def test_entry_rebalance_and_exit_fees(self):
        result = self.run_path(
            lend_rate=0,
            borrow_rate=0,
            hedge_fee_bps=10,
            hedge_fee_per_trade=0.01,
            option_fee_per_contract=0.01,
        )
        self.assertAlmostEqual(
            result.summary["direct_costs"], 0.15375
        )
        self.assertAlmostEqual(
            result.summary["net_pnl"], 2.59625
        )
        self.assertAlmostEqual(result.summary["gross_pnl"], 2.75)

    def test_bid_ask_uses_correct_side_without_double_charging(self):
        data = manual_path()
        data["delta"] = 0.5
        data["option_bid"] = [4.8, 7.8, 5.9]
        data["option_ask"] = [5.2, 8.2, 6.1]
        data["hedge_bid"] = data["spot"] - 0.1
        data["hedge_ask"] = data["spot"] + 0.1

        result = self.run_path(
            data,
            execution="bid_ask",
            lend_rate=0,
            borrow_rate=0,
        )
        self.assertAlmostEqual(result.summary["gross_pnl"], 1.5)
        self.assertAlmostEqual(result.summary["direct_costs"], 0.4)
        self.assertAlmostEqual(result.summary["final_cash"], 1.1)
        self.assertEqual(result.ledger.loc[0, "option_fill"], 4.8)
        self.assertEqual(result.ledger.loc[2, "option_fill"], 6.1)

    def test_long_and_short_are_symmetric_without_costs(self):
        data = manual_path()
        data["dividend_per_unit"] = [0, 2, 1]

        short = self.run_path(data)
        long = self.run_path(data, option_quantity=1)

        for name in (
            "cash", "equity", "option_position",
            "hedge_position", "interest", "net_pnl",
        ):
            np.testing.assert_allclose(
                long.ledger[name],
                -short.ledger[name],
                atol=1e-12,
            )

    def test_multipliers_preserve_monetary_exposure(self):
        base = self.run_path(lend_rate=0, borrow_rate=0)
        scaled = self.run_path(
            option_multiplier=100,
            hedge_multiplier=50,
            lend_rate=0,
            borrow_rate=0,
        )

        np.testing.assert_allclose(
            scaled.ledger["hedge_position"],
            2 * base.ledger["hedge_position"],
        )
        self.assertAlmostEqual(
            scaled.summary["net_pnl"],
            100 * base.summary["net_pnl"],
        )

    def test_dividends_apply_to_position_held_before_rebalance(self):
        data = manual_path()
        data["dividend_per_unit"] = [0, 2, 1]
        result = self.run_path(data, lend_rate=0, borrow_rate=0)

        np.testing.assert_allclose(
            result.ledger["dividend_income"], [0, 1, 0.25]
        )
        self.assertAlmostEqual(result.summary["net_pnl"], 4)

    def test_funding_rate_switches_with_opening_cash_sign(self):
        data = manual_path()
        data["delta"] = [0.5, 0, 99]

        result = self.run_path(
            data, lend_rate=0.02, borrow_rate=0.1
        )

        intermediate_cash = -45 * np.exp(0.1 / 365) + 55
        expected = intermediate_cash * np.exp(0.02 / 365) - 6

        self.assertAlmostEqual(
            result.summary["final_cash"], expected, places=12
        )
        np.testing.assert_allclose(
            result.ledger["funding_rate_used"].iloc[1:],
            [0.1, 0.02],
        )

    def test_elapsed_time_covers_weekend_and_dst(self):
        data = pd.DataFrame({
            "timestamp": [
                "2023-11-03T16:00:00-04:00",
                "2023-11-06T16:00:00-05:00",
            ],
            "spot": [100, 100],
            "option_mark": [0, 0],
            "delta": [0, 0],
        })
        result = self.run_path(
            data, initial_capital=100, lend_rate=0.1
        )
        expected = 100 * np.expm1(0.1 * 73 / (365 * 24))

        self.assertAlmostEqual(
            result.summary["net_pnl"], expected, places=12
        )

    def test_open_book_is_marked_without_exit_costs(self):
        result = self.run_path(
            liquidate_final=False, lend_rate=0, borrow_rate=0
        )
        self.assertEqual(result.summary["final_option_position"], -1)
        self.assertEqual(result.summary["final_hedge_position"], 0.9)
        self.assertAlmostEqual(result.summary["net_pnl"], 2.75)
        self.assertNotEqual(
            result.summary["final_cash"],
            result.summary["final_equity"],
        )

    def test_future_values_do_not_change_earlier_ledger(self):
        data = manual_path()
        base = self.run_path(data, liquidate_final=False)

        data.loc[2, ["spot", "option_mark", "delta"]] = [
            150, 40, 0.7
        ]
        changed = self.run_path(data, liquidate_final=False)

        pd.testing.assert_frame_equal(
            base.ledger.iloc[:2], changed.ledger.iloc[:2]
        )

    def test_inputs_are_not_modified(self):
        data = manual_path()
        original = data.copy(deep=True)
        self.run_path(data)
        pd.testing.assert_frame_equal(data, original)

    def test_rejects_invalid_times_and_changed_contract(self):
        modes = (
            "naive", "unordered", "duplicate",
            "missing", "roll", "single",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                data = manual_path()

                if mode == "naive":
                    data["timestamp"] = (
                        data["timestamp"].dt.tz_localize(None)
                    )
                elif mode == "unordered":
                    data = data.iloc[::-1]
                elif mode == "duplicate":
                    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
                elif mode == "missing":
                    data.loc[1, "timestamp"] = pd.NaT
                elif mode == "roll":
                    data.loc[1, "contract_id"] = "another_call"
                else:
                    data = data.iloc[:1]

                with self.assertRaises(ValueError):
                    self.run_path(data)

    def test_rejects_invalid_prices_quotes_and_dividends(self):
        modes = (
            "nan", "negative_spot", "partial", "crossed",
            "outside", "entry_dividend", "missing_quotes",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                data = manual_path()
                settings = {}

                if mode == "nan":
                    data.loc[1, "delta"] = np.nan
                elif mode == "negative_spot":
                    data.loc[1, "spot"] = -1
                elif mode == "partial":
                    data["option_bid"] = 1
                elif mode in {"crossed", "outside"}:
                    data["option_bid"] = data["option_mark"] - 0.1
                    data["option_ask"] = data["option_mark"] + 0.1
                    data.loc[1, "option_bid"] = (
                        9 if mode == "crossed" else 8.01
                    )
                elif mode == "entry_dividend":
                    data["dividend_per_unit"] = [1, 0, 0]
                else:
                    settings["execution"] = "bid_ask"

                with self.assertRaises(ValueError):
                    self.run_path(data, **settings)

    def test_rejects_invalid_settings(self):
        invalid = (
            {"option_quantity": 0},
            {"option_multiplier": 0},
            {"hedge_fee_bps": -1},
            {"lend_rate": np.nan},
            {"execution": "last"},
            {"liquidate_final": "yes"},
        )
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    HedgeLedgerSettings(**settings)

    def test_costs_have_their_correct_funding_effect(self):
        data = manual_path()
        rate = 0.07

        base = self.run_path(
            data, lend_rate=rate, borrow_rate=rate
        )
        charged = self.run_path(
            data,
            lend_rate=rate,
            borrow_rate=rate,
            hedge_fee_bps=10,
            option_fee_per_contract=0.02,
        )

        fees = np.array([
            0.05 + 0.02,
            0.0275,
            0.02625 + 0.02,
        ])
        expected_difference = -np.sum(
            fees * np.exp(rate * np.array([2, 1, 0]) / 365)
        )

        self.assertAlmostEqual(
            charged.summary["net_pnl"] - base.summary["net_pnl"],
            expected_difference,
            places=12,
        )
        self.assertLess(
            charged.summary["max_reconciliation"], 1e-12
        )


if __name__ == "__main__":
    unittest.main()