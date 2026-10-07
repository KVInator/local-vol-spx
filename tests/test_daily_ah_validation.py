import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from andreasen_huge import AHGrid, AndreasenHugeSurface
from daily_ah_validation import (
    AHShortEndVariance, DailyAHValidator, DailyValidationSettings, quote_metrics
)


def model_fixture():
    times = np.array([1, 2]) / 365
    return AndreasenHugeSurface(
        AHGrid(0.5, 200), 100, times,
        np.full(2, 100.0), np.ones(2),
        [np.array([-0.1, 0.1])] * 2,
        [np.log([0.2, 0.2])] * 2,
    )


class DailyValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = model_fixture()

    def test_zero_radius_preserves_positive_time_coefficient(self):
        variant = AHShortEndVariance(self.model, 0)
        z = np.exp(np.array([-0.03, 0, 0.03]))
        for t in [0.25/365, 1/365, 1.5/365]:
            np.testing.assert_array_equal(
                variant.normalized_variance(z, t),
                variant.base.normalized_variance(z, t),
            )

    def test_no_change_at_or_after_first_pillar(self):
        variant = AHShortEndVariance(self.model, 0.01)
        for t, side in [
            (1/365, "left"), (1/365, "right"), (1.5/365, "right")
        ]:
            np.testing.assert_array_equal(
                variant.normalized_variance([0.98, 1, 1.02], t, side),
                variant.base.normalized_variance([0.98, 1, 1.02], t, side),
            )

    def test_early_smoothing_positive_and_changes_coefficient(self):
        variant = AHShortEndVariance(self.model, 0.01)
        z = variant.native_z
        raw = variant.base.normalized_variance(z, 0.25/365)
        smooth = variant.normalized_variance(z, 0.25/365)
        self.assertTrue(np.isfinite(smooth).all())
        self.assertTrue((smooth > 0).all())
        self.assertGreater(np.max(abs(smooth-raw)), 1e-6)

    def test_zero_time_and_extrapolation_rejected(self):
        variant = AHShortEndVariance(self.model, 0.01)
        with self.assertRaises(ValueError):
            variant.normalized_variance([1], 0)
        with self.assertRaises(ValueError):
            variant.normalized_variance([10], 0.25/365)

    def test_underresolved_native_radius_rejected(self):
        for radius in [-1, np.nan, 0.001]:
            with self.assertRaises(ValueError):
                AHShortEndVariance(self.model, radius)

    def test_wide_domain_preserves_log_spacing(self):
        s = DailyValidationSettings()
        fine, wide = s.cases()[2], s.cases()[-1]
        self.assertAlmostEqual(2*fine[1]/fine[2], 2*wide[1]/wide[2])

    def test_invalid_study_settings_rejected(self):
        for key, value in [
            ("radius", 0), ("fine_intervals", 12000),
            ("wide_width", 0.75), ("steps_per_day", 0),
        ]:
            with self.assertRaises(ValueError):
                replace(DailyValidationSettings(), **{key: value})

    def test_quote_metrics_use_original_bands(self):
        q = pd.DataFrame({
            "call_mid": [1.0, 2.0], "call_half_width": [0.1, 0.2],
            "call_bid": [0.9, 1.8], "call_ask": [1.1, 2.2],
        })
        result = quote_metrics(np.array([1.0, 2.4]), q)
        self.assertEqual(result["outside_original_bands"], 1)
        self.assertAlmostEqual(result["rms_half_spreads"], np.sqrt(2))

    def test_end_to_end_study_without_refit(self):
        c = pd.DataFrame({
            "quote_date": ["2023-09-01"]*2,
            "root": ["UNKNOWN"]*2,
            "expire_date": ["2023-09-02", "2023-09-03"],
        })
        groups = []
        for expiry, t in zip(c.expire_date, self.model.maturities):
            strikes = np.array([99, 100, 101])
            prices = self.model.call_price(strikes, float(t))
            groups.append(pd.DataFrame({
                "expire_date": expiry, "strike": strikes,
                "call_mid": prices, "call_half_width": 0.01,
                "call_bid": prices-0.01, "call_ask": prices+0.01,
            }))
        q = pd.concat(groups, ignore_index=True)
        original = q.copy(deep=True)
        settings = DailyValidationSettings(
            radius=0.01, width=0.1, wide_width=0.15,
            coarse_intervals=100, fine_intervals=200, steps_per_day=4,
        )
        result = DailyAHValidator(settings).run(
            self.model, q, c, progress=lambda text: None
        )
        self.assertEqual(len(result["quote_fit"]), 3)
        self.assertEqual(len(result["sensitivity"]), 8)
        self.assertEqual(len(result["quote_prices"]), 2*len(q))
        self.assertEqual(
            set(result["forward_shapes"].scope),
            {"full_domain", "report_window"},
        )
        self.assertFalse(
            result["spot_greeks"].zero_time_coefficient_requested.any()
        )
        self.assertTrue(
            (result["spot_greeks"].minimum_coefficient_time_years > 0).all()
        )
        self.assertLess(
            result["forward_backward"].forward_minus_backward_points.abs().max(),
            0.05,
        )
        pd.testing.assert_frame_equal(q, original)


if __name__ == "__main__":
    unittest.main()
