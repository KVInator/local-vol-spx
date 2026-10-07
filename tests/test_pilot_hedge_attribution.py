import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pilot_hedge_attribution import (
    PilotHedgeAttribution, read_comparison, sha256,
)


def example(two_dates=True, costs=True):
    trials = []
    dates = [("2023-09-01", "2023-09-05", 90.0, 35.0)]

    if two_dates:
        dates.append(("2023-09-05", "2023-09-06", -10.0, -6.0))

    cases = (
        ["mid", "option_spread", "option_spread_fee"]
        if costs else ["mid"]
    )

    for i, (date, end, move, mark_change) in enumerate(dates):
        for case in cases:
            for strategy, delta in (
                ("ah", 0.4), ("black", 0.5), ("unhedged", 0.0)
            ):
                cost = 0 if case == "mid" else 0.5
                if case == "option_spread_fee":
                    cost += abs(delta) * 0.1

                interest = (
                    0 if case == "mid" else -abs(delta) * 0.01
                )
                option = -mark_change

                trials.append({
                    "scenario": case,
                    "entry_id": f"e{i}",
                    "strategy": strategy,
                    "contract_id": "c1",
                    "quote_date": date,
                    "end_date": end,
                    "root": "UNKNOWN",
                    "expire_date": "2023-09-29",
                    "strike": 100.0,
                    "kind": "call",
                    "calendar_days": 28 - i * 4,
                    "entry_spot_y": 0.0,
                    "entry_delta": delta,
                    "delta_difference": -0.1,
                    "spot_change": move,
                    "option_mark_change": mark_change,
                    "total_option_pnl": option,
                    "total_hedge_pnl": delta * move,
                    "total_interest": interest,
                    "total_dividend_income": 0.0,
                    "direct_costs": cost,
                    "net_pnl": (
                        option + delta * move + interest - cost
                    ),
                    "observations": 2,
                    "liquidated": True,
                    "initial_capital": 0.0,
                    "final_option_position": 0.0,
                    "final_hedge_position": 0.0,
                })

    t = pd.DataFrame(trials)
    p = t.pivot(
        index=["scenario", "entry_id"],
        columns="strategy",
        values="net_pnl",
    ).reset_index()

    p = p.rename(columns={
        s: s + "_net_pnl"
        for s in ("ah", "black", "unhedged")
    })
    p["ah_minus_black_net_pnl"] = (
        p.ah_net_pnl - p.black_net_pnl
    )
    p["abs_error_improvement"] = (
        p.black_net_pnl.abs() - p.ah_net_pnl.abs()
    )
    p["squared_error_improvement"] = (
        p.black_net_pnl**2 - p.ah_net_pnl**2
    )

    c = t[["scenario", "entry_id"]].drop_duplicates().assign(
        status="compared"
    )
    return t, p, c


class AttributionTests(unittest.TestCase):
    def test_delta_exposure_has_correct_sign_and_units(self):
        out = PilotHedgeAttribution().run(*example())
        d = out["attribution"].query(
            "scenario == 'mid'"
        ).set_index("entry_id")

        self.assertAlmostEqual(d.loc["e0", "delta_exposure"], -9)
        self.assertAlmostEqual(d.loc["e1", "delta_exposure"], 1)
        self.assertAlmostEqual(d.loc["e0", "spot_return"], 0.9)
        self.assertEqual(d.loc["e0", "spot_direction"], "rising")
        self.assertEqual(d.loc["e1", "spot_direction"], "falling")
        np.testing.assert_allclose(
            d.identity_residual, 0, atol=1e-12
        )

    def test_costs_and_funding_reconcile_without_changing_deltas(self):
        out = PilotHedgeAttribution().run(
            *example()
        )["cost_shifts"]

        row = out.query(
            "scenario == 'option_spread_fee' and entry_id == 'e0'"
        ).iloc[0]

        self.assertAlmostEqual(row.ah_pnl_shift, -0.544)
        self.assertAlmostEqual(row.black_pnl_shift, -0.555)
        self.assertAlmostEqual(
            row.ah_pnl_shift,
            row.ah_funding_shift + row.ah_cost_effect,
        )

    def test_mse_decomposes_into_variance_and_squared_mean(self):
        row = PilotHedgeAttribution().run(
            *example()
        )["summary"].query("scenario == 'mid'").iloc[0]

        self.assertAlmostEqual(row.mse_improvement, 48)
        self.assertAlmostEqual(row.variance_improvement, 20)
        self.assertAlmostEqual(row.squared_mean_improvement, 28)
        self.assertAlmostEqual(row.mae_improvement, 4)
        self.assertAlmostEqual(row.ah_rms, np.sqrt(2.5))

    def test_date_contributions_sum_to_pooled_improvements(self):
        out = PilotHedgeAttribution().run(*example())
        daily = out["daily_attribution"].groupby("scenario")
        summary = out["summary"].set_index("scenario")

        np.testing.assert_allclose(
            daily.pooled_mse_contribution.sum(),
            summary.mse_improvement,
        )
        np.testing.assert_allclose(
            daily.pooled_mae_contribution.sum(),
            summary.mae_improvement,
        )

    def test_leave_one_date_out_can_reverse_the_result(self):
        loo = PilotHedgeAttribution().run(
            *example()
        )["leave_one_date_out"]

        mid = loo.query(
            "scenario == 'mid'"
        ).set_index("excluded_date")

        self.assertAlmostEqual(
            mid.loc["2023-09-01", "mae_improvement"], -1
        )
        self.assertAlmostEqual(
            mid.loc["2023-09-05", "mae_improvement"], 9
        )
        self.assertTrue(mid.comparisons.eq(1).all())

    def test_unequal_date_counts_are_weighted_and_entire_dates_are_omitted(self):
        t, p, c = example()
        frames = []

        for frame in (t, p, c):
            extra = frame.loc[
                frame.entry_id.eq("e0")
            ].assign(entry_id="extra")
            frames.append(
                pd.concat([frame, extra], ignore_index=True)
            )

        out = PilotHedgeAttribution().run(*frames)
        summary = out["summary"].query(
            "scenario == 'mid'"
        ).iloc[0]

        self.assertAlmostEqual(summary.mae_improvement, 17 / 3)
        self.assertAlmostEqual(summary.mse_improvement, 65)

        daily = out["daily_attribution"].query(
            "scenario == 'mid'"
        )
        self.assertAlmostEqual(
            daily.pooled_mse_contribution.sum(), 65
        )

        loo = out["leave_one_date_out"].query(
            "scenario == 'mid'"
        ).set_index("excluded_date")

        self.assertEqual(
            loo.loc["2023-09-01", "comparisons"], 1
        )
        self.assertEqual(
            loo.loc["2023-09-05", "comparisons"], 2
        )

    def test_single_date_has_no_leave_one_date_out_estimate(self):
        out = PilotHedgeAttribution().run(
            *example(two_dates=False)
        )
        self.assertTrue(out["leave_one_date_out"].empty)
        self.assertIn("scenario", out["leave_one_date_out"])

    def test_midpoint_only_run_has_empty_cost_shifts(self):
        out = PilotHedgeAttribution().run(*example(costs=False))
        self.assertTrue(out["cost_shifts"].empty)

    def test_exclusions_remain_in_coverage(self):
        t, p, c = example()
        extra = pd.DataFrame([{
            "scenario": "mid",
            "entry_id": "last",
            "status": "sample_end",
        }])

        out = PilotHedgeAttribution().run(
            t, p, pd.concat([c, extra], ignore_index=True)
        )

        self.assertEqual(
            out["coverage"].status.eq("sample_end").sum(), 1
        )
        self.assertNotIn(
            "last", out["attribution"].entry_id.tolist()
        )

    def test_duplicate_or_incomplete_groups_are_rejected(self):
        t, p, c = example()

        for bad in (
            pd.concat([t, t.iloc[:1]]), t.iloc[1:]
        ):
            with self.assertRaises(ValueError):
                PilotHedgeAttribution().run(bad, p, c)

    def test_corrupt_pnl_or_saved_metrics_are_rejected(self):
        t, p, c = example()
        bad = t.copy()
        bad.loc[0, "net_pnl"] += 1

        with self.assertRaises(ValueError):
            PilotHedgeAttribution().run(bad, p, c)

        bad_p = p.copy()
        bad_p.loc[0, "squared_error_improvement"] += 1

        with self.assertRaises(ValueError):
            PilotHedgeAttribution().run(t, bad_p, c)

    def test_nonfinite_values_are_rejected(self):
        t, p, c = example()
        t.loc[0, "entry_delta"] = np.inf

        with self.assertRaises(ValueError):
            PilotHedgeAttribution().run(t, p, c)

    def test_contract_mismatch_and_missing_coverage_are_rejected(self):
        t, p, c = example()
        bad = t.copy()
        bad.loc[0, "strike"] = 101

        with self.assertRaises(ValueError):
            PilotHedgeAttribution().run(bad, p, c)

        with self.assertRaises(ValueError):
            PilotHedgeAttribution().run(t, p, c.iloc[1:])

    def test_same_date_index_moves_must_agree(self):
        t, p, c = example()
        t.loc[
            t.entry_id.eq("e1"), "quote_date"
        ] = "2023-09-01"

        with self.assertRaises(ValueError):
            PilotHedgeAttribution().run(t, p, c)

    def test_inputs_are_unchanged(self):
        frames = example()
        copies = [f.copy(deep=True) for f in frames]

        PilotHedgeAttribution().run(*frames)

        for actual, before in zip(frames, copies):
            pd.testing.assert_frame_equal(actual, before)

    def test_reader_checks_hashes_and_upstream_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            t, p, c = example()

            for name, frame in zip(
                ("strategy_results", "paired_results", "coverage"),
                (t, p, c),
            ):
                frame.to_csv(folder / f"{name}.csv", index=False)

            audit = {
                "status": "completed",
                "future_information_used_in_entry_hedges": False,
                "models_refitted": False,
                "prices_clipped": False,
                "dividend_per_unit": 0,
                "coverage_by_scenario": {
                    case: g.status.value_counts().to_dict()
                    for case, g in c.groupby("scenario")
                },
                "output_sha256": {
                    f.name: sha256(f)
                    for f in folder.glob("*.csv")
                },
            }
            path = folder / "audit.json"
            path.write_text(json.dumps(audit))

            self.assertEqual(
                len(read_comparison(folder)[0]), len(t)
            )

            audit["prices_clipped"] = True
            path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                read_comparison(folder)

            audit["prices_clipped"] = False
            path.write_text(json.dumps(audit))

            with (folder / "strategy_results.csv").open("a") as handle:
                handle.write("\n")

            with self.assertRaises(ValueError):
                read_comparison(folder)


if __name__ == "__main__":
    unittest.main()