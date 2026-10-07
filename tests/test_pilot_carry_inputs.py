import unittest

import numpy as np
import pandas as pd

from pilot_carry_inputs import CarryInputSettings, PilotCarryInputs


def panel(
    day="2023-09-01",
    expiry="2023-09-29",
    forward=100.35,
):
    stamp = pd.Timestamp(
        f"{day} 16:00",
        tz="America/New_York",
    ).tz_convert("UTC")

    fixing = pd.Timestamp(
        f"{expiry} 16:00",
        tz="America/New_York",
    ).tz_convert("UTC")

    time = (fixing - stamp).total_seconds() / (365 * 86400)
    discount = np.exp(-0.05 * time)
    rows = []

    for strike in np.linspace(97.5, 102.5, 21):
        for kind, mid in [
            ("call", 5 + discount * (forward - strike)),
            ("put", 5),
        ]:
            rows.append(
                dict(
                    quote_date=day,
                    root="UNKNOWN",
                    expire_date=expiry,
                    strike=strike,
                    kind=kind,
                    bid=mid - 0.02,
                    ask=mid + 0.02,
                    underlying_last=100.0,
                    quote_timestamp_utc=stamp,
                    assumed_fixing_utc=fixing,
                    assumed_maturity_years=time,
                    quote_status="research_calibration_quote",
                )
            )

    return pd.DataFrame(rows)


class TestPilotCarryInputs(unittest.TestCase):
    def test_discount_units_and_conditional_forward_refits(self):
        q = panel()
        tables, _ = PilotCarryInputs().prepare(q)
        time = q.assumed_maturity_years.iloc[0]

        for _, row in tables["carry_inputs"].iterrows():
            expected_d = np.exp(-row.annual_rate * time)

            expected_f = (
                100
                + np.exp(-0.05 * time) / expected_d * 0.35
            )

            self.assertAlmostEqual(
                row.discount_factor,
                expected_d,
                places=12,
            )

            self.assertAlmostEqual(
                row.forward,
                expected_f,
                places=10,
            )

        self.assertEqual(
            tables["primary_carry"].case.iloc[0],
            "rate_5pct",
        )

    def test_otm_selection_and_put_conversion(self):
        tables, _ = PilotCarryInputs().prepare(panel())
        q = tables["calibration_quotes"]

        self.assertEqual(len(q), 21)
        self.assertFalse(q.duplicated(PilotCarryInputs.KEY).any())

        self.assertTrue(
            q.loc[q.strike.lt(q.forward), "source_kind"]
            .eq("put")
            .all()
        )

        self.assertTrue(
            q.loc[q.strike.ge(q.forward), "source_kind"]
            .eq("call")
            .all()
        )

        adjustment = np.where(
            q.source_kind.eq("put"),
            q.discount_factor * (q.forward - q.strike),
            0,
        )

        np.testing.assert_allclose(
            q.call_bid,
            q.source_bid + adjustment,
        )

        np.testing.assert_allclose(
            q.call_half_width,
            (q.source_ask - q.source_bid) / 2,
        )

    def test_missing_preferred_side_uses_flagged_fallback(self):
        q = panel()
        q = q.loc[
            ~(q.kind.eq("put") & q.strike.eq(99))
        ]

        tables, _ = PilotCarryInputs().prepare(q)
        selected = tables["calibration_quotes"]
        row = selected.loc[selected.strike.eq(99)].iloc[0]

        self.assertEqual(row.source_kind, "call")
        self.assertTrue(row.preferred_side_missing)
        self.assertEqual(len(selected), 21)
        self.assertEqual(selected.strike.min(), 97.5)
        self.assertEqual(selected.strike.max(), 102.5)

    def test_incompatible_parity_is_retained(self):
        q = panel()

        q.loc[
            q.kind.eq("call") & q.strike.eq(100),
            ["bid", "ask"],
        ] += 0.5

        tables, _ = PilotCarryInputs().prepare(q)
        row = tables["primary_carry"].iloc[0]

        self.assertTrue(row.carry_ready)
        self.assertTrue(row.parity_bands_incompatible)

        self.assertEqual(
            row.carry_status,
            "ready_with_parity_incompatibility",
        )

        self.assertEqual(
            len(tables["calibration_quotes"]),
            21,
        )

    def test_insufficient_pairs_preserve_group_without_forward(self):
        q = panel().loc[lambda x: x.strike.le(98.5)]

        tables, _ = PilotCarryInputs().prepare(q)
        carry = tables["carry_inputs"]

        self.assertEqual(len(carry), 3)
        self.assertFalse(carry.carry_ready.any())
        self.assertTrue(carry.forward.isna().all())
        self.assertTrue(carry.discount_factor.gt(0).all())

        self.assertTrue(
            tables["calibration_quotes"].call_mid.isna().all()
        )

    def test_pairs_cannot_cross_dates(self):
        q = pd.concat(
            [
                panel().query("kind == 'call'"),
                panel(day="2023-09-05").query("kind == 'put'"),
            ],
            ignore_index=True,
        )

        tables, audit = PilotCarryInputs().prepare(q)

        self.assertEqual(audit["matched_pairs"], 0)
        self.assertEqual(len(tables["primary_carry"]), 2)
        self.assertFalse(
            tables["primary_carry"].carry_ready.any()
        )

    def test_later_quotes_cannot_change_earlier_inputs(self):
        original = panel()
        first, _ = PilotCarryInputs().prepare(original)

        augmented = pd.concat(
            [
                original,
                panel(day="2023-09-05", forward=101),
            ],
            ignore_index=True,
        )

        augmented["next_mid"] = 999999.0
        second, _ = PilotCarryInputs().prepare(augmented)

        for name in ["primary_carry", "calibration_quotes"]:
            earlier = second[name].loc[
                second[name].quote_date.eq(
                    pd.Timestamp("2023-09-01")
                )
            ]

            pd.testing.assert_frame_equal(
                first[name],
                earlier.reset_index(drop=True),
            )

    def test_daily_snapshot_must_be_consistent_across_expiries(self):
        second = panel(expiry="2023-10-06")
        second["underlying_last"] = 100.5

        with self.assertRaisesRegex(
            ValueError,
            "daily snapshot",
        ):
            PilotCarryInputs().prepare(
                pd.concat(
                    [panel(), second],
                    ignore_index=True,
                )
            )

    def test_actual_utc_maturity_survives_daylight_saving_change(self):
        q = panel(
            day="2023-11-02",
            expiry="2023-11-09",
        )

        tables, _ = PilotCarryInputs().prepare(q)
        row = tables["primary_carry"].iloc[0]

        self.assertAlmostEqual(
            row.maturity_years,
            169 / (365 * 24),
            places=12,
        )

        self.assertAlmostEqual(
            row.discount_factor,
            np.exp(-0.05 * 169 / (365 * 24)),
            places=12,
        )

    def test_call_bound_breach_is_reported_without_clipping(self):
        q = panel()

        q.loc[
            q.kind.eq("call") & q.strike.eq(102.5),
            ["bid", "ask"],
        ] = [105, 106]

        tables, _ = PilotCarryInputs().prepare(q)

        row = tables["calibration_quotes"].loc[
            lambda x: x.strike.eq(102.5)
        ].iloc[0]

        self.assertTrue(row.band_disjoint_from_call_bounds)
        self.assertEqual(row.call_bid, 105)
        self.assertEqual(row.call_ask, 106)

    def test_raw_quotes_and_assumption_labels_remain_intact(self):
        q = panel()
        before = q.copy(deep=True)

        _, audit = PilotCarryInputs().prepare(q)

        pd.testing.assert_frame_equal(q, before)
        self.assertFalse(audit["funding_curve_verified"])
        self.assertFalse(audit["backtest_performed"])
        self.assertFalse(audit["prices_clipped"])

    def test_primary_rate_must_be_present(self):
        with self.assertRaises(ValueError):
            CarryInputSettings(
                primary_rate=0.05,
                rates=(0.03, 0.07),
            )


if __name__ == "__main__":
    unittest.main()