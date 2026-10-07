import unittest

import pandas as pd

from hedging_data_audit import AuditSettings, HedgingDataAudit


def quote(
    day="2023-09-01",
    strike=100.0,
    expiry="2023-10-06",
    spot=100.0,
):
    stamp = pd.Timestamp(
        f"{day} 16:00", tz="America/New_York"
    )
    discount, forward = 0.995, 100.5
    cp = discount * (forward - strike)
    put_mid = 20.0

    return {
        "quote_unixtime": stamp.timestamp(),
        "quote_readtime": f"{day} 16:00",
        "quote_date": day,
        "expire_date": expiry,
        "underlying_last": spot,
        "strike": strike,
        "c_bid": put_mid + cp - 0.1,
        "c_ask": put_mid + cp + 0.1,
        "p_bid": put_mid - 0.1,
        "p_ask": put_mid + 0.1,
    }


class TestHedgingDataAudit(unittest.TestCase):
    def run_audit(self, rows, **settings):
        return HedgingDataAudit(
            AuditSettings(**settings)
        ).run(pd.DataFrame(rows))

    def test_exact_parity_and_bracket_headers(self):
        raw = pd.DataFrame(
            [quote(strike=k) for k in (98, 99, 100, 101, 102)]
        )
        raw.columns = [
            f" [{c.upper()}] " for c in raw.columns
        ]
        result = HedgingDataAudit().run(raw)
        parity = result["parity_summary"].iloc[0]

        self.assertEqual(
            parity["status"], "diagnostic_fit"
        )
        self.assertAlmostEqual(
            parity["fitted_discount"], 0.995, places=10
        )
        self.assertAlmostEqual(
            parity["fitted_forward"], 100.5, places=10
        )
        self.assertEqual(parity["outside_parity_bands"], 0)
        self.assertTrue(
            result["quote_panel"]["identity_status"]
            .eq("provisional_no_root")
            .all()
        )
        self.assertTrue(
            result["expiry_metadata"]["model_fixing_timestamp_utc"]
            .eq("")
            .all()
        )

    def test_duplicate_quarantine_does_not_choose_first(self):
        first, duplicate = quote(), quote()
        duplicate["c_ask"] += 1

        result = self.run_audit(
            [first, duplicate, quote("2023-09-05")]
        )
        self.assertEqual(
            result["row_audit"]["duplicate_close_key"].sum(), 2
        )
        self.assertFalse(
            result["row_audit"].iloc[:2]["c_mark_usable"].any()
        )
        self.assertEqual(len(result["quote_panel"]), 2)

    def test_root_separates_otherwise_identical_keys(self):
        rows = [
            dict(quote(), option_root=root)
            for root in ("SPX", "SPXW")
        ]
        result = self.run_audit(
            rows, root_column="option_root"
        )
        self.assertFalse(
            result["row_audit"]["duplicate_close_key"].any()
        )
        self.assertEqual(len(result["contract_continuity"]), 4)
        self.assertTrue(
            result["expiry_metadata"]["settlement_status"]
            .eq("unverified")
            .all()
        )

    def test_crossed_locked_zero_and_missing_quotes(self):
        rows = [
            quote(strike=k) for k in (98, 99, 100, 101)
        ]
        rows[0]["c_bid"], rows[0]["c_ask"] = 2, 1
        rows[1]["c_bid"], rows[1]["c_ask"] = 1, 1
        rows[2]["c_bid"], rows[2]["c_ask"] = 0, 0.1
        rows[3]["c_bid"] = "broken"

        result = self.run_audit(rows)
        calls = (
            result["quote_panel"]
            .query("kind == 'call'")
            .set_index("strike")
        )
        self.assertTrue(calls.loc[98, "bad_quote"])
        self.assertFalse(calls.loc[99, "mark_usable"])
        self.assertTrue(calls.loc[100, "mark_usable"])
        self.assertFalse(calls.loc[100, "two_sided_quote"])
        self.assertTrue(calls.loc[101, "bad_quote"])

    def test_timestamp_mismatch_and_bad_epoch(self):
        bad_read = quote(strike=99)
        bad_epoch = quote(strike=100)
        bad_read["quote_readtime"] = "2023-09-01 15:59"
        bad_epoch["quote_unixtime"] = "not a number"

        result = self.run_audit(
            [bad_read, bad_epoch, quote(strike=101)]
        )
        self.assertTrue(
            result["row_audit"].iloc[0]["timestamp_mismatch"]
        )
        self.assertTrue(
            result["row_audit"].iloc[1]["bad_timestamp"]
        )
        self.assertFalse(
            result["row_audit"].iloc[:2]["c_mark_usable"].any()
        )

    def test_missing_close_stays_in_continuity_calendar(self):
        off_close = quote("2023-09-05")
        off_close["quote_unixtime"] += 60
        off_close["quote_readtime"] = "2023-09-05 16:01"

        result = self.run_audit(
            [quote(), off_close, quote("2023-09-06")]
        )
        self.assertTrue(
            result["contract_continuity"][
                "missing_between_first_and_last"
            ].eq(1).all()
        )

        first = result["adjacent_observations"]
        first = first[
            first["quote_date"].eq(pd.Timestamp("2023-09-01"))
        ]
        self.assertTrue(
            first["status"]
            .eq("missing_or_ambiguous_next")
            .all()
        )
        self.assertTrue(
            first["next_date"]
            .eq(pd.Timestamp("2023-09-05"))
            .all()
        )

    def test_expiry_is_not_missing_and_no_strike_substitution(self):
        rows = [
            quote(expiry="2023-09-05"),
            quote(strike=99),
            quote("2023-09-05", strike=98),
        ]
        result = self.run_audit(rows)

        first = result["adjacent_observations"]
        first = first[
            first["quote_date"].eq(pd.Timestamp("2023-09-01"))
        ]
        expiring = first[
            first["expire_date"].eq(pd.Timestamp("2023-09-05"))
        ]
        self.assertTrue(
            expiring["status"]
            .eq("expiry_before_or_on_next_date")
            .all()
        )

        alive = first[first["strike"].eq(99)]
        self.assertTrue(
            alive["status"]
            .eq("missing_or_ambiguous_next")
            .all()
        )
        self.assertTrue(alive["next_mid"].isna().all())

    def test_scope_exit_and_vendor_time_are_explicit(self):
        first = quote()
        second = quote("2023-09-05", spot=90)
        first["expire_unix"] = pd.Timestamp(
            "2023-10-06 13:30", tz="UTC"
        ).timestamp()

        result = self.run_audit([first, second])
        pairs = result["adjacent_observations"]
        pairs = pairs[
            pairs["quote_date"].eq(pd.Timestamp("2023-09-01"))
        ]
        self.assertTrue(
            pairs["status"].eq("out_of_scope_next").all()
        )
        self.assertTrue(
            result["expiry_metadata"]["settlement_status"]
            .eq("unverified")
            .all()
        )
        self.assertTrue(
            result["contract_continuity"]["usable_observations"]
            .eq(2)
            .all()
        )

    def test_inconsistent_spot_quarantines_snapshot(self):
        result = self.run_audit(
            [quote(strike=99), quote(strike=100, spot=101)]
        )
        self.assertTrue(
            result["row_audit"]["spot_inconsistent"].all()
        )
        self.assertFalse(
            result["row_audit"]["c_mark_usable"].any()
        )

    def test_missing_schema_fails_before_outputs(self):
        raw = pd.DataFrame([quote()]).drop(columns="p_ask")
        with self.assertRaisesRegex(
            ValueError, "Missing raw columns"
        ):
            HedgingDataAudit().run(raw)


if __name__ == "__main__":
    unittest.main()