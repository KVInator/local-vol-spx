import unittest

import numpy as np
import pandas as pd

from hedging_pilot import HedgingPilot, PilotSettings


def policy(start="2023-08-01", end="2024-01-31"):
    # Synthetic scheduled calendar for unit tests.
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({
        "quote_date": dates,
        "reference_session": True,
        "early_cash_close": False,
        "session_policy": "provisional_standard_session",
    })


def quote(day="2023-09-01", expiry="2023-09-29", strike=100.0, **changes):
    stamp = pd.Timestamp(f"{day} 16:00", tz="America/New_York")
    row = {
        "quote_unixtime": stamp.timestamp(),
        "quote_readtime": f"{day} 16:00",
        "quote_date": day,
        "expire_date": expiry,
        "underlying_last": 100.0,
        "strike": strike,
        "c_bid": 5.0,
        "c_ask": 5.2,
        "p_bid": 4.0,
        "p_ask": 4.2,
    }
    row.update(changes)
    return row


class TestHedgingPilot(unittest.TestCase):
    def prepare(self, rows, p=None, settings=None):
        return HedgingPilot(
            policy() if p is None else p, settings
        ).prepare(pd.DataFrame(rows))

    def test_monthly_aliases_are_conservatively_excluded(self):
        rows = [
            quote(expiry=d)
            for d in (
                "2023-09-14", "2023-09-15",
                "2023-09-16", "2023-09-29",
            )
        ]
        tables, audit = self.prepare(rows)
        self.assertEqual(len(tables["pilot_quotes"]), 2)
        self.assertTrue(
            tables["pilot_quotes"]["expire_date"]
            .eq(pd.Timestamp("2023-09-29")).all()
        )
        self.assertIn(
            "2023-09-14", audit["monthly_date_aliases_excluded"]
        )
        self.assertFalse(audit["contract_identity_verified"])

    def test_holiday_adjusted_monthly_dates(self):
        p = policy("2022-03-01", "2022-07-01")
        mask = p["quote_date"].eq(pd.Timestamp("2022-04-15"))
        p.loc[mask, "reference_session"] = False
        p.loc[mask, "session_policy"] = "nonreference_date"
        s = PilotSettings(start="2022-04-01", end="2022-04-01")
        model = HedgingPilot(p, s)
        expected = pd.to_datetime([
            "2022-04-13", "2022-04-14",
            "2022-04-15", "2022-04-16",
        ])
        self.assertTrue(expected.isin(model.monthly_aliases).all())
        tables, _ = model.prepare(pd.DataFrame([
            quote(day="2022-04-01", expiry="2022-04-14"),
            quote(day="2022-04-01", expiry="2022-04-29"),
        ]))
        self.assertEqual(len(tables["pilot_quotes"]), 2)

    def test_future_observation_availability_does_not_select_entries(self):
        baseline, _ = self.prepare([quote()])
        p = policy()
        mask = p["quote_date"].eq(pd.Timestamp("2023-09-29"))
        p.loc[mask, "session_policy"] = "missing_observation"
        missing, _ = self.prepare([quote()], p)
        pd.testing.assert_frame_equal(
            baseline["pilot_quotes"], missing["pilot_quotes"]
        )
        self.assertTrue(
            missing["pilot_quotes"]["entry_research_candidate"].all()
        )

    def test_quote_quality_is_side_specific_and_zero_bids_are_bounds(self):
        rows = [
            quote(strike=99, c_bid=6.0, c_ask=5.0),
            quote(strike=100, c_bid=0.0, c_ask=0.2),
            quote(strike=101, c_bid=1.0, c_ask=1.0),
        ]
        tables, _ = self.prepare(rows)
        self.assertEqual(
            tables["pilot_quotes"]["kind"].tolist(),
            ["put", "put", "put"],
        )
        self.assertEqual(
            tables["zero_bid_bounds"]["kind"].tolist(), ["call"]
        )
        self.assertFalse(
            tables["zero_bid_bounds"]["entry_research_candidate"].any()
        )

    def test_nullable_string_input_with_missing_side(self):
        raw = pd.DataFrame([
            quote(c_bid=None), quote(strike=101),
        ]).astype("string")
        tables, _ = HedgingPilot(policy()).prepare(raw)
        quotes = tables["pilot_quotes"]
        self.assertEqual(len(quotes), 3)
        self.assertEqual(
            quotes.query("kind == 'put'")["strike"].tolist(), [100, 101]
        )

    def test_duplicate_rows_are_all_quarantined(self):
        tables, audit = self.prepare([
            quote(), quote(c_ask=6), quote(strike=102),
        ])
        self.assertEqual(audit["duplicate_rows_quarantined"], 2)
        self.assertTrue(tables["pilot_quotes"]["strike"].eq(102).all())

    def test_calendar_and_current_session_exclusions(self):
        p = policy()
        mask = p["quote_date"].eq(pd.Timestamp("2023-09-05"))
        p.loc[mask, "session_policy"] = "missing_observation"
        mask = p["quote_date"].eq(pd.Timestamp("2023-10-06"))
        p.loc[mask, "early_cash_close"] = True

        tables, _ = self.prepare([
            quote(),
            quote(day="2023-09-05"),
            quote(expiry="2023-10-06"),
        ], p)
        self.assertEqual(len(tables["pilot_quotes"]), 2)
        statuses = tables["quote_selection_summary"]["quote_status"].tolist()
        self.assertIn("session_excluded", statuses)
        self.assertIn("expiry_calendar_excluded", statuses)
        row = (
            tables["session_timeline"].set_index("quote_date")
            .loc["2023-09-05"]
        )
        self.assertEqual(row["session_policy"], "missing_observation")

    def test_elapsed_utc_maturity_handles_dst(self):
        s = PilotSettings(start="2023-11-03", end="2023-11-03")
        tables, _ = self.prepare([
            quote(day="2023-11-03", expiry="2023-11-10"),
        ], settings=s)
        maturity = tables["pilot_quotes"]["assumed_maturity_years"].iloc[0]
        self.assertAlmostEqual(maturity, 169 / (365 * 24))

    def test_wide_calibration_coverage_does_not_force_entry_selection(self):
        tables, _ = self.prepare([
            quote(strike=100 * np.exp(0.06)),
        ])
        self.assertEqual(len(tables["pilot_quotes"]), 2)
        self.assertFalse(
            tables["pilot_quotes"]["entry_research_candidate"].any()
        )
        self.assertTrue(tables["pilot_quotes"]["root"].eq("UNKNOWN").all())

    def test_policy_dates_and_horizon_are_checked(self):
        p = policy()
        with self.assertRaisesRegex(ValueError, "unique"):
            HedgingPilot(pd.concat([p, p.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "horizon"):
            HedgingPilot(p.loc[p["quote_date"] < "2023-11-01"])
        p["reference_session"] = p["reference_session"].astype(object)
        p.loc[0, "reference_session"] = "unknown"
        with self.assertRaisesRegex(ValueError, "flags"):
            HedgingPilot(p)


if __name__ == "__main__":
    unittest.main()