import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from andreasen_huge import AHGrid, AndreasenHugeSurface
from daily_ah import DailyAHCalibrator, DailyAHSettings


def fixture():
    times = np.array([14, 35]) / 365
    forwards = 100 * np.exp(0.01 * times)
    discounts = np.exp(-0.05 * times)

    source = AndreasenHugeSurface(
        AHGrid(0.5, 200),
        100,
        times,
        forwards,
        discounts,
        [np.array([-0.10, 0.10])] * 2,
        [np.log([0.20, 0.20])] * 2,
    )
    carry, quotes = [], []

    for t, f, d, expiry in zip(
        times, forwards, discounts,
        ["2023-09-15", "2023-10-06"],
    ):
        meta = {
            "quote_date": "2023-09-01",
            "root": "UNKNOWN",
            "expire_date": expiry,
            "case": "rate_5pct",
            "forward": f,
            "discount_factor": d,
            "carry_ready": True,
            "quote_timestamp_utc": "2023-09-01T20:00:00Z",
            "assumed_fixing_utc": expiry + "T20:00:00Z",
        }
        carry.append({
            **meta,
            "spot": 100,
            "maturity_years": t,
            "annual_rate": 0.05,
            "parity_bands_incompatible": False,
            "fitted_parity_outside_bands": False,
        })

        strikes = np.linspace(96, 104, 9)
        prices = source.call_price(strikes, t)
        for k, price in zip(strikes, prices):
            quotes.append({
                **meta,
                "strike": k,
                "underlying_last": 100,
                "assumed_maturity_years": t,
                "source_kind": "call",
                "source_bid": price - 0.005,
                "source_ask": price + 0.005,
                "call_bid": price - 0.005,
                "call_ask": price + 0.005,
                "call_mid": price,
                "call_half_width": 0.005,
                "preferred_side_missing": k < f,
                "band_disjoint_from_call_bounds": False,
                "midpoint_outside_call_bounds": False,
            })

    return pd.DataFrame(quotes), pd.DataFrame(carry)


class DailyAHTests(unittest.TestCase):
    def setUp(self):
        self.q, self.c = fixture()
        self.runner = DailyAHCalibrator(DailyAHSettings(
            intervals=200,
            width=0.5,
            control_points=5,
            smoothing=0.1,
        ))

    def test_synthetic_prices_and_input_preservation(self):
        original = self.q.copy(deep=True)
        model, tables = self.runner.calibrate(self.q, self.c)
        self.assertEqual(len(model.maturities), 2)
        self.assertLess(
            tables["quote_residuals"]
            .price_residual_points.abs().max(),
            1e-5,
        )
        self.assertEqual(len(tables["shape_checks"]), 4)
        self.assertEqual(len(tables["conditioning"]), 3)
        pd.testing.assert_frame_equal(self.q, original)

    def test_disjoint_band_retained_without_clipping(self):
        row = self.q.iloc[0]
        lower = row.discount_factor * (row.forward - row.strike)
        self.q.loc[
            0, ["source_bid", "call_bid"]
        ] = lower - 0.03
        self.q.loc[
            0, ["source_ask", "call_ask"]
        ] = lower - 0.02
        self.q.loc[0, "call_mid"] = lower - 0.025
        self.q.loc[
            0,
            [
                "band_disjoint_from_call_bounds",
                "midpoint_outside_call_bounds",
            ],
        ] = True

        _, tables = self.runner.calibrate(self.q, self.c)
        fitted = tables["quote_residuals"].iloc[0]
        self.assertAlmostEqual(fitted.call_mid, lower - 0.025)
        self.assertTrue(fitted.outside_original_band)
        self.assertEqual(
            tables["expiry_summary"]
            .band_disjoint_from_call_bounds.sum(),
            1,
        )

    def test_duplicate_strike_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.runner.calibrate(
                pd.concat([self.q, self.q.iloc[:1]]), self.c
            )

    def test_duplicate_carry_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.runner.calibrate(
                self.q, pd.concat([self.c, self.c.iloc[:1]])
            )

    def test_unavailable_expiry_not_silently_dropped(self):
        self.c.loc[0, "carry_ready"] = False
        with self.assertRaisesRegex(ValueError, "Unavailable carry"):
            self.runner.calibrate(self.q, self.c)

    def test_mixed_dates_rejected(self):
        future = self.q.iloc[:1].copy()
        future["quote_date"] = "2023-09-05"
        with self.assertRaisesRegex(ValueError, "one date"):
            self.runner.calibrate(
                pd.concat([self.q, future]), self.c
            )

    def test_forward_mismatch_rejected(self):
        self.q.loc[0, "forward"] += 1
        with self.assertRaisesRegex(
            ValueError, "Quote/carry mismatch"
        ):
            self.runner.calibrate(self.q, self.c)

    def test_changed_conversion_rejected(self):
        self.q.loc[0, "call_mid"] += 0.01
        with self.assertRaisesRegex(
            ValueError, "parity conversion"
        ):
            self.runner.calibrate(self.q, self.c)

    def test_utc_maturity_mismatch_rejected(self):
        self.c["assumed_fixing_utc"] = "2023-10-07T20:00:00Z"
        with self.assertRaisesRegex(
            ValueError, "mismatch|UTC maturities"
        ):
            self.runner.calibrate(self.q, self.c)

    def test_assumed_rate_mismatch_rejected(self):
        self.c["annual_rate"] = 0.07
        with self.assertRaisesRegex(ValueError, "assumed rate"):
            self.runner.calibrate(self.q, self.c)

    def test_stale_price_bound_flag_rejected(self):
        self.q.loc[0, "band_disjoint_from_call_bounds"] = True
        with self.assertRaisesRegex(
            ValueError, "call-bound flags"
        ):
            self.runner.calibrate(self.q, self.c)

    def test_invalid_settings_rejected(self):
        for field, value in [
            ("smoothing", -1),
            ("control_points", 2),
            ("max_evaluations", 0),
        ]:
            with self.assertRaises(ValueError):
                replace(DailyAHSettings(), **{field: value})


if __name__ == "__main__":
    unittest.main()