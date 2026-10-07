import unittest

import numpy as np
import pandas as pd

from pilot_carry import CarrySettings, PilotCarryEstimator


def panel(
    discount=None,
    forward=100.35,
    day="2023-09-01",
    spread_half=0.02,
):
    timestamp = pd.Timestamp(
        f"{day} 16:00", tz="America/New_York"
    ).tz_convert("UTC")
    fixing = pd.Timestamp(
        "2023-09-29 16:00", tz="America/New_York"
    ).tz_convert("UTC")
    maturity = (
        fixing - timestamp
    ).total_seconds() / (365 * 86400)
    discount = (
        np.exp(-0.05 * maturity)
        if discount is None else discount
    )
    rows = []
    for strike in np.linspace(97.5, 102.5, 21):
        for kind, mid in [
            ("call", 5 + discount * (forward - strike)),
            ("put", 5),
        ]:
            rows.append({
                "quote_date": day,
                "root": "UNKNOWN",
                "expire_date": "2023-09-29",
                "strike": strike,
                "kind": kind,
                "bid": mid - spread_half,
                "ask": mid + spread_half,
                "underlying_last": 100.0,
                "quote_timestamp_utc": timestamp,
                "assumed_fixing_utc": fixing,
                "assumed_maturity_years": maturity,
                "quote_status": "research_calibration_quote",
            })
    return pd.DataFrame(rows)


class TestPilotCarry(unittest.TestCase):
    def run_panel(self, q):
        return PilotCarryEstimator(
            CarrySettings(windows=(0.03,))
        ).run(q)

    def test_exact_forward_and_discount_recovery(self):
        q = panel()
        tables, audit = self.run_panel(q)
        row = tables["carry_estimates"].iloc[0]
        expected = np.exp(
            -0.05 * q["assumed_maturity_years"].iloc[0]
        )
        self.assertEqual(row["status"], "fitted")
        self.assertAlmostEqual(
            row["discount_factor"], expected, places=10
        )
        self.assertAlmostEqual(row["forward"], 100.35, places=10)
        self.assertAlmostEqual(
            row["implied_zero_rate_pct"], 5, places=8
        )
        self.assertTrue(row["original_bands_feasible"])
        self.assertLess(row["minimum_band_multiplier"], 1e-6)
        self.assertFalse(audit["daily_discount_curve_verified"])

    def test_known_fixed_rate_scenario_recovers_forward(self):
        tables, _ = self.run_panel(panel())
        row = tables["fixed_rate_sensitivity"].query(
            "annual_rate_pct == 5"
        ).iloc[0]
        self.assertAlmostEqual(row["forward"], 100.35, places=10)
        self.assertLess(row["rms_parity_half_widths"], 1e-8)

    def test_unpaired_side_remains_out_of_parity_fit(self):
        q = panel()
        q = q.loc[
            ~(q["strike"].eq(100) & q["kind"].eq("put"))
        ]
        tables, audit = self.run_panel(q)
        self.assertEqual(audit["matched_pairs"], 20)
        self.assertEqual(audit["unpaired_quote_sides"], 1)
        self.assertAlmostEqual(
            tables["carry_estimates"].iloc[0]["forward"],
            100.35,
            places=10,
        )

    def test_duplicate_sides_are_not_averaged(self):
        q = panel()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_panel(pd.concat([q, q.iloc[:1]]))

    def test_pairs_never_cross_quote_dates(self):
        q = pd.concat([
            panel().query("kind == 'call'"),
            panel(day="2023-09-05").query("kind == 'put'"),
        ])
        tables, audit = self.run_panel(q)
        self.assertEqual(audit["matched_pairs"], 0)
        self.assertTrue(
            tables["carry_estimates"]["status"]
            .eq("insufficient_pairs").all()
        )

    def test_negative_implied_rates_are_not_clipped(self):
        tables, _ = self.run_panel(panel(discount=1.003))
        row = tables["carry_estimates"].iloc[0]
        self.assertAlmostEqual(
            row["discount_factor"], 1.003, places=10
        )
        self.assertLess(row["implied_zero_rate_pct"], 0)
        self.assertEqual(row["status"], "fitted")

    def test_discount_identification_worsens_with_wider_bands(self):
        narrow, _ = self.run_panel(panel(spread_half=0.02))
        wide, _ = self.run_panel(panel(spread_half=0.20))
        a = narrow["carry_estimates"].iloc[0]
        b = wide["carry_estimates"].iloc[0]
        self.assertGreater(
            b["discount_upper"] - b["discount_lower"],
            5 * (a["discount_upper"] - a["discount_lower"]),
        )

    def test_incompatible_parity_bands_are_reported(self):
        q = panel()
        mask = q["strike"].eq(100) & q["kind"].eq("call")
        q.loc[mask, ["bid", "ask"]] += 0.5
        tables, _ = self.run_panel(q)
        row = tables["carry_estimates"].iloc[0]
        self.assertFalse(row["original_bands_feasible"])
        self.assertGreater(row["minimum_band_multiplier"], 1)
        self.assertEqual(
            row["band_bounds_status"], "original_bands_infeasible"
        )
        self.assertTrue(pd.isna(row["rate_band_width_pp"]))
        self.assertLess(abs(row["forward"] - 100.35), 0.03)

    def test_invalid_discount_is_not_promoted(self):
        q = panel()
        mask = q["kind"].eq("call")
        mid = 5 + 0.5 * (q.loc[mask, "strike"] - 100)
        q.loc[mask, "bid"] = mid - 0.02
        q.loc[mask, "ask"] = mid + 0.02
        tables, _ = self.run_panel(q)
        self.assertEqual(
            tables["carry_estimates"].iloc[0]["status"],
            "invalid_discount_or_forward",
        )

    def test_narrow_window_is_not_silently_widened(self):
        tables, _ = PilotCarryEstimator(
            CarrySettings(windows=(0.001, 0.03))
        ).run(panel())
        fits = tables["carry_estimates"].set_index("window")
        self.assertEqual(
            fits.loc[0.001, "status"], "insufficient_pairs"
        )
        self.assertEqual(fits.loc[0.03, "status"], "fitted")
        self.assertTrue(pd.isna(
            tables["window_sensitivity"].iloc[0][
                "forward_window_range_points"
            ]
        ))

    def test_maturity_and_snapshot_conflicts_fail(self):
        q = panel()
        q.loc[0, "assumed_maturity_years"] += 0.01
        with self.assertRaisesRegex(ValueError, "maturity"):
            self.run_panel(q)

        q = panel()
        q.loc[0, "underlying_last"] = 101
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.run_panel(q)


if __name__ == "__main__":
    unittest.main()