"""Integration tests for the single-expiry calibration workflow."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from black import BlackPricer
from calibration_pipeline import ExpirySliceCalibrator
from discount_curve import TreasuryYieldProxy
from market_data import SPXQuoteFile
from maturity import ExpiryConvention


class TestCalibrationPipeline(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

        self.quote_date = date(2023, 9, 1)
        self.expiry_date = date(2023, 10, 6)
        quote_time = pd.Timestamp("2023-09-01T20:00:00Z")
        fixing_time = pd.Timestamp("2023-10-06T20:00:00Z")

        self.proxy = TreasuryYieldProxy(
            as_of=self.quote_date,
            tenor_days=[30.0, 60.0],
            bey_rates=[0.0551, 0.0555],
            source_url="https://home.treasury.gov",
        )
        maturity = 35.0 / 365.0
        discount = self.proxy.discount_factor(maturity)
        rate = self.proxy.continuous_rate(maturity)

        self.forward = 100.0 * np.exp(
            (rate - 0.02) * maturity
        )
        pricer = BlackPricer(
            forward=self.forward,
            discount_factor=discount,
            maturity=maturity,
        )

        strikes = np.linspace(85.0, 115.0, 61)
        calls = pricer.price(strikes, 0.20, "call")
        puts = pricer.price(strikes, 0.20, "put")

        rows = []
        for strike, call, put in zip(strikes, calls, puts):
            rows.append(
                {
                    "[QUOTE_UNIXTIME]": int(quote_time.timestamp()),
                    "[QUOTE_READTIME]": "2023-09-01 16:00",
                    "[QUOTE_DATE]": "2023-09-01",
                    "[UNDERLYING_LAST]": 100.0,
                    "[EXPIRE_DATE]": "2023-10-06",
                    "[EXPIRE_UNIX]": int(fixing_time.timestamp()),
                    "[DTE]": 35.0,
                    "[STRIKE]": strike,
                    "[C_BID]": max(float(call) - 0.005, 0.0),
                    "[C_ASK]": float(call) + 0.005,
                    "[C_IV]": 0.20,
                    "[P_BID]": max(float(put) - 0.005, 0.0),
                    "[P_ASK]": float(put) + 0.005,
                    "[P_IV]": 0.20,
                }
            )

        path = self.directory / "synthetic_quotes.txt"
        pd.DataFrame(rows).to_csv(path, index=False)

        self.snapshot = SPXQuoteFile(path).snapshot(
            self.quote_date,
            self.expiry_date,
            quote_timestamp_utc=quote_time,
        )
        self.convention = ExpiryConvention(
            expiry_date=self.expiry_date,
            settlement_time_ny="16:00:00",
            settlement_kind="PM",
            verification_status="verified",
            evidence="Synthetic fixture with an explicitly specified clock.",
        )

    def test_recovers_known_inputs_through_full_pipeline(self):
        before = self.snapshot.quotes

        result = ExpirySliceCalibrator(
            discount_proxy=self.proxy,
            n_intervals=10,
        ).calibrate(self.snapshot, self.convention)

        self.assertAlmostEqual(
            result.spline.pricer.forward,
            self.forward,
            places=9,
        )
        self.assertAlmostEqual(
            result.spline.pricer.maturity,
            35.0 / 365.0,
            places=14,
        )
        np.testing.assert_allclose(
            result.quotes["iv_mid"].to_numpy(),
            0.20,
            rtol=0.0,
            atol=1e-9,
        )
        self.assertTrue(
            result.spline.shape_checks()["passed"]
        )
        self.assertLessEqual(
            result.spline.max_half_spreads,
            result.spline.band_cap + 1e-6,
        )
        pd.testing.assert_frame_equal(
            before,
            self.snapshot.quotes,
        )

    def test_rejects_mismatched_discount_reference_date(self):
        wrong_date_proxy = TreasuryYieldProxy(
            as_of=date(2023, 9, 2),
            tenor_days=[30.0, 60.0],
            bey_rates=[0.0551, 0.0555],
            source_url="https://home.treasury.gov",
        )

        with self.assertRaisesRegex(ValueError, "date"):
            ExpirySliceCalibrator(
                discount_proxy=wrong_date_proxy
            ).calibrate(self.snapshot, self.convention)


if __name__ == "__main__":
    unittest.main()