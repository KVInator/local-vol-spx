from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from historical_data_audit import (
    HistoricalAuditSettings,
    HistoricalDataAudit,
)


def quote(
    day,
    clock="16:00",
    strike=100,
    spot=100,
    expiry="2023-10-06",
):
    stamp = pd.Timestamp(
        f"{day} {clock}", tz="America/New_York"
    )
    return {
        "quote_unixtime": stamp.timestamp(),
        "quote_readtime": f"{day} {clock}",
        "quote_date": day,
        "expire_date": expiry,
        "underlying_last": spot,
        "strike": strike,
        "c_bid": 1.0,
        "c_ask": 1.2,
        "p_bid": 1.0,
        "p_ask": 1.2,
    }


class TestHistoricalDataAudit(unittest.TestCase):
    def run_case(
        self,
        monthly,
        start="2023-08-31",
        end="2023-09-06",
        **kwargs,
    ):
        with (
            tempfile.TemporaryDirectory() as directory,
            redirect_stdout(StringIO()),
        ):
            root = Path(directory)
            files = []
            for month, rows in monthly.items():
                path = root / f"spx_eod_{month}.txt"
                pd.DataFrame(rows).to_csv(path, index=False)
                files.append(path)
            settings = HistoricalAuditSettings(
                start=start, end=end, **kwargs
            )
            return HistoricalDataAudit(settings).run(
                files, root / "outputs"
            )

    def test_cross_month_pair_and_weekend_holiday_are_not_gaps(self):
        tables, _ = self.run_case({
            "202308": [quote("2023-08-31")],
            "202309": [
                quote("2023-09-01"),
                quote("2023-09-05"),
                quote("2023-09-06"),
            ],
        })
        adjacent = tables["adjacent_summary"]
        cross = adjacent.loc[adjacent["cross_month"]]
        self.assertEqual(cross["contracts"].sum(), 2)
        self.assertTrue(cross["status"].eq("matched").all())
        self.assertEqual(
            adjacent.loc[
                adjacent["status"].eq("sample_end"),
                "contracts",
            ].sum(),
            2,
        )
        self.assertTrue(tables["session_anomalies"].empty)

    def test_missing_session_is_not_skipped(self):
        tables, _ = self.run_case({
            "202308": [quote("2023-08-31")],
            "202309": [
                quote("2023-09-05"),
                quote("2023-09-06"),
            ],
        })
        a = tables["adjacent_summary"]
        first = a.loc[
            a["quote_date"].eq(pd.Timestamp("2023-08-31"))
        ]
        self.assertTrue(
            first["next_date"]
            .eq(pd.Timestamp("2023-09-01"))
            .all()
        )
        self.assertTrue(
            first["status"].eq("missing_session_observation").all()
        )

    def test_early_close_clock_is_reported_without_relabeling(self):
        tables, _ = self.run_case(
            {
                "202307": [
                    quote("2023-07-03"),
                    quote("2023-07-04"),
                ]
            },
            start="2023-07-03",
            end="2023-07-05",
        )
        d = tables["daily_summary"].set_index("quote_date")
        early = d.loc[pd.Timestamp("2023-07-03")]
        self.assertTrue(early["early_cash_close"])
        self.assertEqual(early["clocks_ny"], "16:00:00")
        self.assertEqual(early["after_cash_close_rows"], 1)
        self.assertEqual(
            d.loc[pd.Timestamp("2023-07-04"), "status"],
            "observed_nonreference_date",
        )

    def test_single_holiday_window_can_be_audited(self):
        tables, _ = self.run_case(
            {"202307": [quote("2023-07-04")]},
            start="2023-07-04",
            end="2023-07-04",
        )
        self.assertEqual(
            tables["daily_summary"].iloc[0]["status"],
            "observed_nonreference_date",
        )
        self.assertTrue(tables["adjacent_summary"].empty)

        tables, _ = self.run_case(
            {"202307": [quote("2023-07-05")]},
            start="2023-07-04",
            end="2023-07-04",
        )
        self.assertTrue(tables["daily_summary"].empty)

    def test_missing_schema_is_recorded_as_failed_not_success(self):
        bad = quote("2023-09-01")
        del bad["p_ask"]
        tables, audit = self.run_case(
            {"202309": [bad]},
            start="2023-09-01",
            end="2023-09-06",
        )
        self.assertEqual(
            audit["files_by_status"], {"file_failed": 1}
        )
        self.assertIn(
            "p_ask",
            tables["file_summary"].iloc[0]["missing_columns"],
        )
        self.assertTrue(
            tables["daily_summary"]["status"]
            .eq("file_failed")
            .all()
        )

    def test_duplicate_next_key_is_quarantined(self):
        q = quote("2023-09-01")
        tables, _ = self.run_case(
            {
                "202308": [quote("2023-08-31")],
                "202309": [q, q],
            },
            end="2023-09-01",
        )
        a = tables["adjacent_summary"]
        self.assertTrue(
            a["status"].eq("missing_or_ambiguous_next").all()
        )
        self.assertEqual(
            tables["daily_summary"].iloc[-1]["duplicate_key_rows"],
            2,
        )

    def test_scope_exit_uses_wider_next_day_lookup(self):
        tables, _ = self.run_case(
            {
                "202308": [quote("2023-08-31")],
                "202309": [quote("2023-09-01", spot=90)],
            },
            end="2023-09-01",
        )
        self.assertTrue(
            tables["adjacent_summary"]["status"]
            .eq("out_of_scope_next")
            .all()
        )

    def test_expiry_boundary_is_not_missing_data(self):
        tables, _ = self.run_case(
            {
                "202308": [
                    quote("2023-08-31", expiry="2023-09-01")
                ],
                "202309": [quote("2023-09-01")],
            },
            end="2023-09-01",
        )
        a = tables["adjacent_summary"]
        first = a.loc[
            a["quote_date"].eq(pd.Timestamp("2023-08-31"))
        ]
        self.assertTrue(
            first["status"]
            .eq("expiry_before_or_on_next_session")
            .all()
        )

    def test_multiple_clocks_do_not_select_latest_snapshot(self):
        tables, _ = self.run_case(
            {
                "202309": [
                    quote("2023-09-01", "15:59"),
                    quote("2023-09-01", strike=101),
                ]
            },
            start="2023-09-01",
            end="2023-09-05",
        )
        first = tables["daily_summary"].iloc[0]
        self.assertEqual(
            first["snapshot_status"],
            "multiple_or_missing_timestamps",
        )
        self.assertEqual(first["usable_c_marks"], 0)
        self.assertTrue(tables["adjacent_summary"].empty)

    def test_dst_and_readtime_disagreement(self):
        winter = quote("2023-01-03")
        summer = quote("2023-07-03", "13:00")
        bad = quote("2023-07-05")
        bad["quote_readtime"] = "2023-07-05 15:00"
        tables, _ = self.run_case(
            {
                "202301": [winter],
                "202307": [summer, bad],
            },
            start="2023-01-03",
            end="2023-07-05",
        )
        d = tables["daily_summary"].set_index("quote_date")
        self.assertEqual(
            d.loc[pd.Timestamp("2023-01-03"), "timestamp_mismatch"],
            0,
        )
        self.assertEqual(
            d.loc[pd.Timestamp("2023-07-03"), "at_cash_close_rows"],
            1,
        )
        self.assertEqual(
            d.loc[pd.Timestamp("2023-07-05"), "timestamp_mismatch"],
            1,
        )
        self.assertEqual(
            d.loc[pd.Timestamp("2023-07-05"), "usable_c_marks"],
            0,
        )

    def test_wrong_file_month_and_bad_quotes_are_counted(self):
        wrong = quote("2023-08-31")
        crossed = quote("2023-09-01")
        crossed["c_bid"], crossed["c_ask"] = 2, 1
        crossed["p_bid"] = 0
        tables, _ = self.run_case(
            {"202309": [wrong, crossed]},
            start="2023-09-01",
            end="2023-09-05",
        )
        info = tables["file_summary"].iloc[0]
        self.assertEqual(info["file_month_mismatch"], 1)
        self.assertEqual(info["c_bad_quote"], 1)
        self.assertEqual(info["p_zero_bid"], 1)

    def test_roots_separate_keys_but_do_not_certify_fixings(self):
        rows = [
            dict(quote("2023-09-01"), option_root=r)
            for r in ("SPX", "SPXW")
        ]
        tables, audit = self.run_case(
            {"202309": rows},
            start="2023-09-01",
            end="2023-09-05",
            root_column="option_root",
        )
        self.assertEqual(
            tables["daily_summary"].iloc[0]["duplicate_key_rows"],
            0,
        )
        self.assertFalse(audit["contract_identity_certified"])
        self.assertFalse(audit["fixings_certified"])


if __name__ == "__main__":
    unittest.main()