"""Saved-data controls for prices, basket identities and audit provenance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pilot_result_audit import (PilotResultAudit, ResultAuditReport, ResultAuditSettings,
                                SavedResearchRun, sha256)


class SavedAuditFixture:
    """Small manually specified saved tables; no optimizer, PDE or market claims."""

    settings = {"target_days": [21, 35], "target_spot_y": [0.0], "kinds": ["call", "put"]}

    @classmethod
    def frames(cls):
        quotes, carry, residuals, panel, raw, buckets, paired, attribution, coverage, daily, manifests, expiry = ([] for _ in range(12))
        for date in ["2023-10-11", "2023-10-19"]:
            group = dict(quote_date=date, root="UNKNOWN", expire_date="2023-11-15")
            c = dict(**group, forward=101.0, discount_factor=0.99, spot=100.0, annual_rate=0.05,
                     maturity_years=(pd.Timestamp(group['expire_date'])-pd.Timestamp(date)).days/365,
                     carry_ready=True, parity_bands_incompatible=False, fitted_parity_outside_bands=False,
                     case="rate_5pct")
            carry.append(c)
            q = dict(**group, strike=100.0, source_kind="put", source_bid=4.0, source_ask=6.0,
                call_bid=4.99, call_ask=6.99, call_mid=5.99, call_half_width=1.0,
                forward=101.0, discount_factor=0.99, case="rate_5pct", preferred_side_missing=False,
                band_disjoint_from_call_bounds=False, midpoint_outside_call_bounds=False)
            quotes.append(q)
            residuals.append(dict(**q, model_call_price=6.0, price_residual_points=0.01,
                                  residual_half_spreads=0.01, outside_original_band=False))
            manifests.append(dict(quote_date=date, root="UNKNOWN", status="fitted", input_quotes=1,
                                  rms_half_spreads=0.01, outside_original_bands=0))
            expiry.append(dict(**group, optimizer_evaluations=3, optimizer_message="controlled saved fixture",
                               active_proxy_bounds=0, rms_half_spreads=0.01, outside_original_bands=0))
            comparisons = 0
            for kind, bid, ask, price in [("call", 5.5, 5.7, 6.3), ("put", 4.0, 6.0, 5.31)]:
                identity = f"{date}-{kind}"
                end_status = "sample_end" if date == "2023-10-19" and kind == "put" else "matched"
                entry = dict(**group, strike=100.0, kind=kind, entry_id=identity,
                    bid=bid, ask=ask, mid=(bid+ask)/2, underlying_last=100.0,
                    calendar_days=round(c["maturity_years"]*365), forward=101.0, discount_factor=0.99,
                    case="rate_5pct", ah_price=price, ah_status="ready", black_status="ready", end_status=end_status)
                panel.append(entry)
                raw.append({name: entry[name] for name in ["quote_date", "root", "expire_date", "strike", "kind", "bid", "ask", "mid", "underlying_last"]})
                for target, status in [(21, "selected"), (35, "duplicate_bucket")]:
                    buckets.append(dict(**group, kind=kind, entry_id=identity, strike=100.0,
                        target_days=target, target_spot_y=0.0, status=status))
                coverage.append(dict(scenario="mid", entry_id=identity,
                                     status="compared" if end_status == "matched" else "sample_end"))
                if end_status != "matched":
                    continue
                comparisons += 1
                p = dict(**group, strike=100.0, kind=kind, scenario="mid", entry_id=identity,
                         ah_net_pnl=2.0, black_net_pnl=1.0, abs_error_improvement=-1.0, squared_error_improvement=-3.0)
                paired.append(p)
                attribution.append(dict(**p, delta_difference=0.1, spot_change=9.0, pnl_gap=1.0,
                    delta_exposure=0.9, funding_difference=0.1, cost_effect=0.0, identity_residual=0.0,
                    maturity_band="28-39d" if entry["calendar_days"] >= 28 else "14-27d"))
            daily.append(dict(quote_date=date, scenario="mid", comparisons=comparisons,
                mae_improvement=-1.0, mse_improvement=-3.0,
                pooled_mae_contribution=-comparisons/3, pooled_mse_contribution=-comparisons))
        tables = dict(calibration_quotes=quotes, primary_carry=carry, quote_residuals=residuals,
            delta_panel=panel, pilot_quotes=raw, selection_buckets=buckets,
            paired_results=paired, attribution=attribution, coverage=coverage,
            daily_attribution=daily, model_manifest=manifests, expiry_summary=expiry,
            group_summary=[dict(scenario="mid", dimension="kind", group="call", comparisons=2)])
        return {name: pd.DataFrame(records) for name, records in tables.items()}

    @classmethod
    def write(cls, parent, legacy=False):
        frames = cls.frames()
        audits, folders = {}, {}
        for stage, names in SavedResearchRun.FILES.items():
            folder = parent / stage
            folder.mkdir()
            folders[stage] = folder
            for name in names:
                frames[name].to_csv(folder / f"{name}.csv", index=False)
            audits[stage] = dict(output_sha256={f"{name}.csv": sha256(folder/f"{name}.csv") for name in names}, input_sha256={})
            if stage == "prepare" and legacy:
                audits[stage].pop("output_sha256")
        audits["panel"].update(settings=cls.settings, marks_replaced_by_model_prices=False,
                              entries_filtered_by_future_availability=False)
        audits["carry"].update(settings={"parity_window": 0.03})
        links = dict(carry=[("prepare", "pilot_quotes.csv")],
            calibration=[("carry", "calibration_quotes.csv"), ("carry", "primary_carry.csv")],
            panel=[("calibration", "audit.json"), ("calibration", "model_manifest.csv"), ("carry", "primary_carry.csv")],
            comparison=[("panel", "delta_panel.csv")],
            attribution=[("comparison", "paired_results.csv"), ("comparison", "coverage.csv")])
        for stage in SavedResearchRun.FILES:
            for producer, filename in links.get(stage, []):
                path = folders[producer] / filename
                audits[stage]["input_sha256"][str(path)] = sha256(path)
            (folders[stage]/"audit.json").write_text(json.dumps(audits[stage]))
        index = parent / "run_index.json"
        index.write_text(json.dumps(dict(months={"2023-10": {
            **{key: str(value) for key, value in folders.items()}, "validation_scope": [], "status": "completed"}})))
        return index


class PilotResultAuditTests(unittest.TestCase):
    def setUp(self):
        self.frames = SavedAuditFixture.frames()
        self.audit = PilotResultAudit(ResultAuditSettings(("2023-10-19",), ("2023-10-11",)))

    def run_audit(self):
        return self.audit.run(self.frames, SavedAuditFixture.settings)

    def test_put_price_uses_saved_call_minus_pinned_parity(self):
        tables, _ = self.run_audit()
        puts = tables["selected_contract_audit"].query("kind == 'put'")
        np.testing.assert_allclose(puts.original_ah_price, 5.01)
        np.testing.assert_allclose(puts.smoothed_minus_original_price, 0.30)

    def test_entry_own_band_is_not_the_opposite_side_training_band(self):
        tables, _ = self.run_audit()
        calls = tables["selected_contract_audit"].query("kind == 'call'")
        self.assertFalse(calls.training_outside_original_band.any())
        self.assertTrue(calls.original_outside_entry_band.all())
        self.assertFalse(calls.entry_side_is_training_side.any())
        np.testing.assert_allclose(calls.original_residual_entry_half_spreads, 4.0)

    def test_duplicates_explain_basket_counts_without_creating_new_entries(self):
        tables, summary = self.run_audit()
        self.assertTrue(tables["selection_daily"].selected_buckets.eq(2).all())
        self.assertTrue(tables["selection_daily"].duplicate_buckets.eq(2).all())
        self.assertEqual(summary["selected_entries"], 4)
        self.assertEqual(summary["midpoint_compared"], 3)
        self.assertTrue(summary["all_entries_retained"])

    def test_sample_end_and_negative_improvement_remain_visible(self):
        tables, _ = self.run_audit()
        mid = tables["midpoint_entry_audit"]
        self.assertEqual(len(mid), 4)
        self.assertEqual(mid.midpoint_comparison_status.eq("not_compared").sum(), 1)
        self.assertTrue(mid.abs_error_improvement.dropna().eq(-1).all())
        self.assertTrue(mid.squared_error_improvement.dropna().eq(-3).all())

    def test_missing_saved_calibration_strike_is_not_interpolated(self):
        for name in ["delta_panel", "pilot_quotes", "selection_buckets", "paired_results", "attribution"]:
            self.frames[name].loc[self.frames[name].kind.eq("call"), "strike"] = 100.5
        tables, summary = self.run_audit()
        missing = tables["selected_contract_audit"].query("kind == 'call'")
        self.assertTrue(missing.original_ah_price.isna().all())
        self.assertTrue(missing.original_outside_entry_band.isna().all())
        self.assertTrue(missing.smoothed_outside_entry_band.notna().all())
        self.assertEqual(summary["original_price_missing"], 2)

    def test_nonready_cached_prices_are_retained_without_price_substitution(self):
        self.frames["delta_panel"].loc[3, ["ah_status", "ah_price"]] = ["failed", np.nan]
        tables, _ = self.run_audit()
        row = tables["selected_contract_audit"].iloc[3]
        self.assertEqual(row.smoothed_price_status, "AH_entry_not_ready")
        self.assertTrue(pd.isna(row.smoothed_outside_entry_band))
        self.assertTrue(np.isfinite(row.original_ah_price))

    def test_calibration_scope_excludes_exit_only_next_month_carry(self):
        for name in ["primary_carry", "calibration_quotes"]:
            extra = self.frames[name].iloc[[0]].copy()
            extra["quote_date"] = "2023-11-01"
            self.frames[name] = pd.concat([self.frames[name], extra], ignore_index=True)
        tables, summary = self.run_audit()
        self.assertEqual(summary["calibration_quotes"], 2)
        self.assertNotIn("2023-11-01", tables["expiry_audit"].quote_date.tolist())

    def test_pooled_rms_weights_quotes_rather_than_averaging_daily_rms(self):
        extra = self.frames["calibration_quotes"].iloc[[0]].copy()
        extra["strike"] = 99.0
        extra["call_bid"], extra["call_ask"], extra["call_mid"] = 5.98, 7.98, 6.98
        self.frames["calibration_quotes"] = pd.concat([self.frames["calibration_quotes"], extra], ignore_index=True)
        residual = self.frames["quote_residuals"].iloc[[0]].copy()
        for name in extra.columns:
            residual[name] = extra[name].values
        residual["model_call_price"] = 7.98
        residual["price_residual_points"] = 1.0
        residual["residual_half_spreads"] = 1.0
        self.frames["quote_residuals"] = pd.concat([self.frames["quote_residuals"], residual], ignore_index=True)
        self.frames["model_manifest"].loc[0, ["input_quotes", "rms_half_spreads"]] = [2, np.sqrt((1+.0001)/2)]
        _, summary = self.run_audit()
        self.assertAlmostEqual(summary["pooled_original_rms_half_spreads"], np.sqrt((1+.0002)/3))

    def test_parity_diagnostic_uses_bid_minus_ask_interval(self):
        tables, _ = self.run_audit()
        pairs = tables["parity_pairs"]
        np.testing.assert_allclose(pairs.parity_bid, -0.5)
        np.testing.assert_allclose(pairs.parity_ask, 1.7)
        self.assertFalse(pairs.outside_parity_band.any())

    def test_duplicate_quote_identity_fails(self):
        self.frames["quote_residuals"] = pd.concat([self.frames["quote_residuals"]]*2)
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            self.run_audit()

    def test_invalid_saved_boolean_fails_instead_of_becoming_truthy(self):
        self.frames["primary_carry"]["carry_ready"] = "no"
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.run_audit()

    def test_entry_mark_difference_from_observed_quote_fails(self):
        self.frames["delta_panel"].loc[0, "mid"] += 0.2
        with self.assertRaisesRegex(ValueError, "entry observed mid"):
            self.run_audit()

    def test_unpinned_forward_difference_fails(self):
        self.frames["delta_panel"].loc[0, "forward"] += 0.2
        with self.assertRaisesRegex(ValueError, "panel forward"):
            self.run_audit()

    def test_zero_entry_spread_fails(self):
        self.frames["delta_panel"].loc[0, "ask"] = 5.5
        with self.assertRaisesRegex(ValueError, "quote widths"):
            self.run_audit()

    def test_bucket_pointing_to_absent_entry_fails(self):
        self.frames["selection_buckets"].loc[0, "entry_id"] = "absent"
        with self.assertRaisesRegex(ValueError, "absent entry"):
            self.run_audit()

    def test_corrupt_attribution_identity_fails(self):
        self.frames["attribution"].loc[0, "funding_difference"] += 0.2
        with self.assertRaisesRegex(ValueError, "P&L attribution"):
            self.run_audit()

    def test_wrong_compared_contract_metadata_fails(self):
        self.frames["paired_results"].loc[0, "kind"] = "put"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.run_audit()

    def test_focus_views_do_not_mutate_inputs_or_filter_full_tables(self):
        before = copy.deepcopy(self.frames)
        tables, summary = self.run_audit()
        self.assertEqual(len(tables["quote_audit"]), 2)
        self.assertEqual(summary["selected_entries"], 4)
        for name in before:
            pd.testing.assert_frame_equal(self.frames[name], before[name])

    def test_missing_focus_date_is_reported_explicitly(self):
        self.audit = PilotResultAudit(ResultAuditSettings(("2023-10-27",), ("2023-10-11",)))
        tables, _ = self.run_audit()
        row = tables["focus_status"].query("quote_date == '2023-10-27'").iloc[0]
        self.assertFalse(row.present_in_manifest)

    def test_saved_report_notebook_contains_no_pricing_execution(self):
        tables, _ = self.run_audit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"audit.ipynb"
            ResultAuditReport(tables, self.audit.settings).notebook(path)
            document = json.loads(path.read_text())
            self.assertEqual(document["nbformat"], 4)
            code = "\n".join("".join(c["source"]) for c in document["cells"] if c["cell_type"] == "code")
            self.assertIn("pd.read_csv", code)
            self.assertNotIn("solve_many", code)
            self.assertNotIn("subprocess", code)


class SavedResearchRunTests(unittest.TestCase):
    @staticmethod
    def add_validation(parent, index, date="2023-10-11", table_date=None):
        folder = parent/"validation"
        folder.mkdir()
        hashes = {}
        for name in ["quote_fit", "sensitivity", "forward_shapes", "forward_backward"]:
            path = folder/f"{name}.csv"
            pd.DataFrame([dict(quote_date=table_date or date, root="UNKNOWN", radius=0.0005)]).to_csv(path, index=False)
            hashes[path.name] = sha256(path)
        audit = dict(output_sha256=hashes, input_sha256={
            str(parent/"calibration/audit.json"): sha256(parent/"calibration/audit.json")})
        (folder/"audit.json").write_text(json.dumps(audit))
        pd.DataFrame([dict(job=f"2023-10/validation/{date}", status="completed", folder=str(folder))]).to_csv(parent/"stage_index.csv", index=False)
        doc = json.loads(index.read_text())
        doc["months"]["2023-10"]["validation_scope"] = [date]
        index.write_text(json.dumps(doc))

    def test_consumed_hashes_and_cross_stage_links_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SavedAuditFixture.write(Path(tmp))
            run = SavedResearchRun(index, "2023-10").load()
            self.assertEqual(len(run.frames["delta_panel"]), 4)
            self.assertTrue(run.frames["validation_coverage"].stage_status.eq("not_requested").all())
            run.verify_unchanged()

    def test_legacy_preparation_requires_existing_carry_input_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SavedAuditFixture.write(Path(tmp), legacy=True)
            SavedResearchRun(index, "2023-10").load()
            path = Path(tmp)/"prepare/pilot_quotes.csv"
            path.write_text(path.read_text().replace("5.5", "5.4"))
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                SavedResearchRun(index, "2023-10").load()

    def test_tampered_saved_table_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SavedAuditFixture.write(Path(tmp))
            path = Path(tmp)/"panel/delta_panel.csv"
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                SavedResearchRun(index, "2023-10").load()

    def test_stale_cross_stage_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SavedAuditFixture.write(Path(tmp))
            path = Path(tmp)/"comparison/audit.json"
            audit = json.loads(path.read_text())
            audit["input_sha256"] = {"old_panel": "0"*64}
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "does not pin panel"):
                SavedResearchRun(index, "2023-10").load()

    def test_consumed_file_change_during_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SavedAuditFixture.write(Path(tmp))
            run = SavedResearchRun(index, "2023-10").load()
            path = Path(tmp)/"carry/primary_carry.csv"
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                run.verify_unchanged()

    def test_output_cannot_overwrite_upstream_run_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SavedAuditFixture.write(Path(tmp))
            run = SavedResearchRun(index, "2023-10").load()
            with self.assertRaisesRegex(ValueError, "separate"):
                run.safe_output(Path(tmp)/"panel/report")

    def test_relative_index_paths_resolve_against_explicit_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            index = SavedAuditFixture.write(parent)
            doc = json.loads(index.read_text())
            for stage in SavedResearchRun.FILES:
                doc["months"]["2023-10"][stage] = stage
            index.write_text(json.dumps(doc))
            run = SavedResearchRun(index, "2023-10", repository=parent).load()
            self.assertEqual(run.folders["panel"], (parent/"panel").resolve())

    def test_cached_validation_is_loaded_without_claiming_all_entry_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            index = SavedAuditFixture.write(parent)
            self.add_validation(parent, index)
            run = SavedResearchRun(index, "2023-10").load()
            scope = run.frames["validation_coverage"]
            self.assertEqual(scope.diagnostics_loaded.sum(), 1)
            self.assertFalse(scope.all_entry_greeks_independently_validated.any())
            self.assertEqual(len(run.frames["validation_sensitivity"]), 1)

    def test_cached_validation_date_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            index = SavedAuditFixture.write(parent)
            self.add_validation(parent, index, table_date="2023-10-19")
            with self.assertRaisesRegex(ValueError, "job/date mismatch"):
                SavedResearchRun(index, "2023-10").load()

    def test_requested_validation_without_saved_job_remains_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            index = SavedAuditFixture.write(parent)
            doc = json.loads(index.read_text())
            doc["months"]["2023-10"]["validation_scope"] = ["2023-10-19"]
            index.write_text(json.dumps(doc))
            run = SavedResearchRun(index, "2023-10").load()
            row = run.frames["validation_coverage"].query("quote_date == '2023-10-19'").iloc[0]
            self.assertEqual(row.stage_status, "requested_job_missing")
            self.assertFalse(row.diagnostics_loaded)

    def test_cached_validation_from_another_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            index = SavedAuditFixture.write(parent)
            self.add_validation(parent, index)
            path = parent/"validation/audit.json"
            doc = json.loads(path.read_text())
            doc["input_sha256"] = {"old_calibration": "0"*64}
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "does not pin this calibration"):
                SavedResearchRun(index, "2023-10").load()


if __name__ == "__main__":
    unittest.main()
