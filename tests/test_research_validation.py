"""Independent parity, Black shock and saved-study integrity controls."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import sha256
from fixtures import surface_fixture, study_research_repository
from pricing import BlackPricer
from research_validation import (
    CalibrationVegaSettings,
    CalibrationVegaValidator,
    GreekPanelSettings,
    GreekPanelValidator,
    contract_panel,
)
from study import HistoricalStudy


class NumericalResearchControls(unittest.TestCase):
    def test_broader_panel_preserves_call_put_price_and_greek_parity(self):
        model, _, carry = surface_fixture()
        settings = GreekPanelSettings(
            radii=(0.0, 0.001),
            log_strikes=(-0.02, 0.0, 0.02),
            spot_changes=(-0.01, 0.0, 0.01),
            coarse_intervals=800,
            fine_intervals=1600,
            steps_per_day=4,
        )
        tables = GreekPanelValidator(settings).run(
            model, carry, progress=lambda _: None
        )
        self.assertTrue(tables["validation_failures"].empty)
        values = tables["greek_panel"]
        keys = ["radius", "case", "maturity", "log_strike", "spot_log_change"]
        pair = values[values.kind.eq("call")].merge(
            values[values.kind.eq("put")], on=keys, suffixes=("_c", "_p")
        )
        f = np.array([model.forward(t) for t in pair.maturity])
        d = np.array([model.discount_factor(t) for t in pair.maturity])
        np.testing.assert_allclose(
            pair.price_c - pair.price_p,
            d * (f / model.spot * pair.spot_c - pair.strike_c),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            pair.delta_c - pair.delta_p, d * f / model.spot, atol=2e-7
        )
        np.testing.assert_allclose(pair.gamma_c - pair.gamma_p, 0, atol=2e-6)
        self.assertEqual(
            set(tables["greek_refinement"].check),
            {"space", "time", "finer_space", "shift", "domain"},
        )
        raw = tables["blend_sensitivity"].query("radius == 0")
        self.assertTrue(
            raw[["price_blend_change", "delta_blend_change", "gamma_blend_change"]]
            .eq(0)
            .all()
            .all()
        )

    def test_quote_iv_shock_agrees_with_analytic_black_vega(self):
        _, quotes, carry = surface_fixture()
        original = quotes.copy(deep=True)
        settings = CalibrationVegaSettings(
            bumps=(0.00001,), intervals=(200, 400), steps_per_day=2, radius=0.001
        )
        validator = CalibrationVegaValidator({}, settings)
        up = validator.shocked_quotes(quotes, carry, 0.00001)
        down = validator.shocked_quotes(quotes, carry, -0.00001)
        self.assertTrue(up.shock_supported.all())
        for row in carry.itertuples():
            mask = quotes.expire_date.eq(row.expire_date)
            expected = BlackPricer(
                row.forward, row.discount_factor, row.maturity_years
            ).vega(quotes.loc[mask, "strike"].to_numpy(), 0.2)
            observed = (up.loc[mask, "call_mid"] - down.loc[mask, "call_mid"]) / 0.00002
            np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=1e-8)
        pd.testing.assert_frame_equal(quotes, original)
        np.testing.assert_allclose(
            up.call_ask - up.call_bid, quotes.call_ask - quotes.call_bid, atol=1e-14
        )
        np.testing.assert_array_equal(up.call_half_width, quotes.call_half_width)

    def test_noninvertible_midpoints_are_retained_on_both_shock_sides(self):
        _, quotes, carry = surface_fixture()
        quotes.loc[0, "call_mid"] = -1.0
        validator = CalibrationVegaValidator({}, CalibrationVegaSettings())
        for bump in (-0.0005, 0.0005):
            shocked = validator.shocked_quotes(quotes, carry, bump)
            self.assertEqual(len(shocked), len(quotes))
            self.assertFalse(shocked.loc[0, "shock_supported"])
            self.assertEqual(shocked.loc[0, "call_mid"], -1.0)
            self.assertEqual(shocked.loc[0, "call_bid"], quotes.loc[0, "call_bid"])

    def test_calibration_vega_call_put_parity_and_grid_comparison(self):
        model, quotes, carry = surface_fixture()
        quotes, carry = (
            quotes[quotes.expire_date.eq(carry.expire_date.iloc[0])],
            carry.iloc[:1],
        )
        # The selected saved model must have the same pillars as these inputs.
        from surface import AndreasenHugeSurface

        saved = AndreasenHugeSurface(
            model.grid,
            model.spot,
            model.maturities[:1],
            model.forwards[:1],
            model.discounts[:1],
            model.controls[:1],
            model.parameters[:1],
        )
        settings = CalibrationVegaSettings(
            bumps=(0.0002, 0.0004),
            log_strikes=(0.0,),
            intervals=(800, 1600),
            steps_per_day=4,
            radius=0.005,
        )
        config = dict(
            width=1.0,
            intervals=400,
            control_points=9,
            smoothing=1.0,
            min_proxy_vol=0.005,
            max_proxy_vol=3.0,
            max_evaluations=100,
        )
        validator = CalibrationVegaValidator(config, settings)
        original = quotes.copy(deep=True)
        tables = validator.run(saved, quotes, carry, progress=lambda _: None)
        self.assertTrue(tables["validation_failures"].empty)
        self.assertEqual(len(validator.models), 5)
        self.assertEqual(len(tables["calibration_vega"]), 8)
        sensitivities = tables["calibration_vega"]
        self.assertTrue((sensitivities.native_vega > 0).all())
        self.assertTrue((sensitivities.pde_vega > 0).all())
        paired = sensitivities[sensitivities.kind.eq("call")].merge(
            sensitivities[sensitivities.kind.eq("put")],
            on=["bump", "intervals"],
            suffixes=("_c", "_p"),
        )
        np.testing.assert_allclose(
            paired.native_vega_c, paired.native_vega_p, atol=1e-8
        )
        np.testing.assert_allclose(paired.pde_vega_c, paired.pde_vega_p, atol=2e-7)
        pd.testing.assert_frame_equal(quotes, original)
        self.assertEqual(len(tables["vega_refinement"]), 4)

    def test_panel_rejects_mismatched_carry(self):
        model, _, carry = surface_fixture()
        carry.loc[0, "forward"] += 1
        with self.assertRaises(ValueError):
            contract_panel(model, carry, (0.0,))

    def test_duplicate_shocks_and_invalid_grids_are_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationVegaSettings(bumps=(0.001, 0.001))
        with self.assertRaises(ValueError):
            GreekPanelSettings(fine_intervals=12001)


class CompletedStudyReviewControls(unittest.TestCase):
    def test_scalar_cache_review_preserves_completed_history(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            repo, config, _ = study_research_repository(Path(directory) / "fixture")
            study = HistoricalStudy(config, repo)
            study.prepare_calendar()
            study.freeze()
            study.run(progress=lambda _: None)
            original = {
                str(p): sha256(p) for p in study.output.rglob("*") if p.is_file()
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts/audit_history.py"),
                    "--study",
                    str(study.output),
                    "--output",
                    str(Path(directory) / "review"),
                    "--dates",
                    study.settings.start,
                    "--strategies",
                    "ah_pde",
                ],
                cwd=repository,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(
                (Path(directory) / "review/tail_ledger_summary.csv").is_file()
            )
            table = pd.read_csv(Path(directory) / "review/tail_ledger_summary.csv")
            self.assertEqual(
                set(table.scenario), {"mid", "option_spread", "option_spread_fee"}
            )
            self.assertLess(table.maximum_replay_difference.max(), 1e-8)
            self.assertEqual(
                original,
                {str(p): sha256(p) for p in study.output.rglob("*") if p.is_file()},
            )


if __name__ == "__main__":
    unittest.main()
