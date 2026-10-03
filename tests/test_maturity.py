import unittest

import pandas as pd

from maturity import ExpiryConvention


class TestMaturity(unittest.TestCase):
    @staticmethod
    def convention(
        expiry_date: str,
        settlement_time_ny: str = "16:00",
    ) -> ExpiryConvention:
        return ExpiryConvention(
            expiry_date=expiry_date,
            settlement_time_ny=settlement_time_ny,
            settlement_kind="PM",
            verification_status="inferred",
            evidence="Explicit convention for the test case.",
        )

    def test_current_market_slice(self):
        convention = self.convention("2023-10-06")
        result = convention.maturity("2023-09-01T20:00:00Z")

        self.assertEqual(
            result.settlement_timestamp_utc,
            pd.Timestamp("2023-10-06T20:00:00Z"),
        )
        self.assertEqual(result.elapsed_days, 35.0)
        self.assertAlmostEqual(result.year_fraction, 35.0 / 365.0)

    def test_timezone_equivalence_and_fractional_day(self):
        convention = self.convention("2023-10-06")

        utc_result = convention.maturity(
            "2023-10-05T20:30:00Z"
        )
        ny_result = convention.maturity(
            "2023-10-05T16:30:00-04:00"
        )

        self.assertEqual(
            utc_result.quote_timestamp_utc,
            ny_result.quote_timestamp_utc,
        )
        self.assertEqual(
            utc_result.year_fraction,
            ny_result.year_fraction,
        )
        self.assertAlmostEqual(
            utc_result.elapsed_days, 23.5 / 24.0
        )

    def test_daylight_saving_changes(self):
        cases = [
            (
                "2023-11-10",
                "2023-11-03T16:00:00-04:00",
                7.0 + 1.0 / 24.0,
            ),
            (
                "2024-03-15",
                "2024-03-08T16:00:00-05:00",
                7.0 - 1.0 / 24.0,
            ),
        ]

        for expiry, quote, expected_days in cases:
            with self.subTest(expiry=expiry):
                result = self.convention(expiry).maturity(quote)

                self.assertAlmostEqual(
                    result.elapsed_days, expected_days
                )
                self.assertAlmostEqual(
                    result.year_fraction,
                    expected_days / 365.0,
                )

    def test_leap_day_with_fixed_365_denominator(self):
        result = self.convention("2024-03-01").maturity(
            "2024-02-28T16:00:00-05:00"
        )

        self.assertEqual(result.elapsed_days, 2.0)
        self.assertAlmostEqual(result.year_fraction, 2.0 / 365.0)

    def test_explicit_early_close_and_expired_quotes(self):
        convention = self.convention(
            "2023-11-24",
            settlement_time_ny="13:00",
        )

        result = convention.maturity(
            "2023-11-24T12:00:00-05:00"
        )
        self.assertAlmostEqual(result.elapsed_days, 1.0 / 24.0)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            convention.maturity("2023-11-24 12:00")

        for quote in (
            "2023-11-24T13:00:00-05:00",
            "2023-11-24T13:01:00-05:00",
        ):
            with self.subTest(quote=quote):
                with self.assertRaisesRegex(
                    ValueError, "must be after"
                ):
                    convention.maturity(quote)


if __name__ == "__main__":
    unittest.main()