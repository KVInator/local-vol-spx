"""Checks for vendor parsing and snapshot integrity."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from market_data import SPXQuoteFile


FIELDS = (
    "QUOTE_UNIXTIME",
    "QUOTE_READTIME",
    "QUOTE_DATE",
    "UNDERLYING_LAST",
    "EXPIRE_DATE",
    "EXPIRE_UNIX",
    "DTE",
    "STRIKE",
    "C_BID",
    "C_ASK",
    "C_IV",
    "P_BID",
    "P_ASK",
    "P_IV",
)


class TestMarketData(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "quotes.txt"

        self.rows = [
            self._row(
                STRIKE=99.0,
                C_BID=2.0,
                C_ASK=2.2,
                P_BID=0.0,
                P_ASK=0.2,
                P_IV="",
            ),
            self._row(
                STRIKE=100.0,
                C_BID=1.3,
                C_ASK=1.1,
            ),
            self._row(
                STRIKE=101.0,
                C_BID="",
                C_ASK=0.8,
                C_IV="",
            ),
            self._row(
                QUOTE_UNIXTIME=1693944000,
                QUOTE_READTIME="2023-09-05 16:00",
                QUOTE_DATE="2023-09-05",
                DTE=31.0,
            ),
        ]
        self._write()

    @staticmethod
    def _row(**updates):
        row = {
            "QUOTE_UNIXTIME": 1693598400,
            "QUOTE_READTIME": "2023-09-01 16:00",
            "QUOTE_DATE": "2023-09-01",
            "UNDERLYING_LAST": 100.0,
            "EXPIRE_DATE": "2023-10-06",
            "EXPIRE_UNIX": 1696622400,
            "DTE": 35.0,
            "STRIKE": 100.0,
            "C_BID": 1.0,
            "C_ASK": 1.2,
            "C_IV": 0.20,
            "P_BID": 0.5,
            "P_ASK": 0.7,
            "P_IV": 0.21,
        }
        row.update(updates)
        return row

    def _write(self):
        header = ", ".join(f"[{name}]" for name in FIELDS)
        records = [
            ", ".join(str(row[name]) for name in FIELDS)
            for row in self.rows
        ]
        self.path.write_text(
            "\n".join([header, *records]) + "\n",
            encoding="utf-8-sig",
        )

    def test_selection_and_timestamps(self):
        reader = SPXQuoteFile(self.path)
        snapshot = reader.snapshot("2023-09-01", "2023-10-06")

        self.assertEqual(reader.rows, 4)
        self.assertEqual(len(reader.observation_dates), 2)
        self.assertEqual(len(snapshot.quotes), 3)
        self.assertEqual(snapshot.spot, 100.0)
        self.assertEqual(
            snapshot.quote_timestamp_utc,
            pd.Timestamp("2023-09-01T20:00:00Z"),
        )
        self.assertEqual(
            snapshot.quote_timestamp_utc
            .tz_convert("America/New_York")
            .hour,
            16,
        )
        self.assertEqual(
            snapshot.quotes["strike"].tolist(),
            [99.0, 100.0, 101.0],
        )
        self.assertIn(
            pd.Timestamp("2023-10-06"),
            reader.available_expiries("2023-09-01"),
        )

    def test_problem_quotes_are_preserved_and_reported(self):
        snapshot = SPXQuoteFile(self.path).snapshot(
            "2023-09-01",
            "2023-10-06",
        )
        report = snapshot.quote_diagnostics()

        self.assertEqual(report.loc["missing_bid_or_ask", "call"], 1)
        self.assertEqual(report.loc["nonfinite_bid_or_ask", "call"], 1)
        self.assertEqual(report.loc["crossed_quote", "call"], 1)
        self.assertEqual(report.loc["zero_bid", "put"], 1)
        self.assertEqual(report.loc["missing_vendor_iv", "call"], 1)
        self.assertEqual(report.loc["missing_vendor_iv", "put"], 1)

        quotes = snapshot.quotes
        self.assertEqual(len(quotes), 3)
        self.assertEqual(quotes.loc[0, "p_bid"], 0.0)
        self.assertTrue(pd.isna(quotes.loc[2, "c_bid"]))

    def test_duplicates_are_retained(self):
        self.rows.append(self.rows[0].copy())
        self._write()

        snapshot = SPXQuoteFile(self.path).snapshot(
            "2023-09-01",
            "2023-10-06",
        )

        self.assertEqual(len(snapshot.quotes), 4)
        self.assertEqual(
            snapshot.row_diagnostics()["duplicate_key_rows"],
            2,
        )

    def test_multiple_times_require_explicit_selection(self):
        later = self.rows[0].copy()
        later["QUOTE_UNIXTIME"] += 3600
        later["QUOTE_READTIME"] = "2023-09-01 17:00"
        self.rows.append(later)
        self._write()

        reader = SPXQuoteFile(self.path)

        with self.assertRaisesRegex(ValueError, "Multiple"):
            reader.snapshot("2023-09-01", "2023-10-06")

        snapshot = reader.snapshot(
            "2023-09-01",
            "2023-10-06",
            quote_timestamp_utc="2023-09-01T21:00:00Z",
        )
        self.assertEqual(len(snapshot.quotes), 1)
        self.assertEqual(snapshot.quote_timestamp_utc.hour, 21)

    def test_copies_and_derived_prices_preserve_source_values(self):
        snapshot = SPXQuoteFile(self.path).snapshot(
            "2023-09-01",
            "2023-10-06",
        )

        copied = snapshot.quotes
        copied.loc[0, "c_bid"] = 999.0
        self.assertEqual(snapshot.quotes.loc[0, "c_bid"], 2.0)

        derived = snapshot.with_midpoints()
        self.assertAlmostEqual(derived.loc[0, "c_mid"], 2.1)
        self.assertAlmostEqual(derived.loc[1, "c_spread"], -0.2)
        self.assertNotIn("c_mid", snapshot.quotes.columns)


if __name__ == "__main__":
    unittest.main()