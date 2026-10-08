from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from research_pipeline import (
    PipelineConfig, ProgressStore, REQUIRED, ResearchPipeline, SCRIPTS,
    assert_consumed, atomic_json, combine_csv, rows, sha256, verify_artifact, write_rows,
)


class ResearchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def store(self):
        return ProgressStore(self.root / "progress", {"config": 1})

    def artifact(self, folder, stage="carry", inputs=(), extra=None):
        folder.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED[stage]:
            write_rows(folder / name, [{"value": "saved"}])
        audit = {"status": "completed", "input_sha256": {str(p): sha256(p) for p in inputs},
                 "output_sha256": {name: sha256(folder / name) for name in REQUIRED[stage]}}
        audit.update(extra or {})
        atomic_json(folder / "audit.json", audit)
        return audit

    def repo(self, **kwargs):
        (self.root / "scripts").mkdir(exist_ok=True)
        (self.root / "src").mkdir(exist_ok=True)
        for name in SCRIPTS.values():
            (self.root / "scripts" / name).touch()
        return ResearchPipeline(self.root, PipelineConfig("2023-09-01", "2023-09-29", **kwargs))

    def test_month_windows_include_leap_day_and_partial_months(self):
        config = PipelineConfig("2024-02-12", "2024-03-05")
        self.assertEqual(list(config.months()), [("2024-02", "2024-02-12", "2024-02-29"),
                                                ("2024-03", "2024-03-01", "2024-03-05")])

    def test_invalid_or_out_of_range_dates_are_rejected(self):
        for start, end in (("2023-09-30", "2023-09-01"), ("2023-02-29", "2023-03-01"),
                           ("20230901", "2023-09-29")):
            with self.subTest(start=start), self.assertRaises(ValueError):
                PipelineConfig(start, end)
        with self.assertRaises(ValueError):
            PipelineConfig("2023-09-01", "2023-09-29", validation_dates=["2023-10-02"])

    def test_unknown_parameters_and_checkpoints_fail_early(self):
        for kwargs in ({"calibration": {"typo": 1}}, {"checkpoints": {"2023-10": {}}},
                       {"checkpoints": {"2023-09": {"latest": "x"}}},
                       {"include_next_session": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig("2023-09-01", "2023-09-29", **kwargs)

    def test_invalid_grids_and_nonfinite_rates_are_rejected(self):
        for kwargs in ({"calibration": {"intervals": 2000}}, {"panel": {"intervals": 101}},
                       {"funding": {"lend_rate": float("nan")}},
                       {"carry": {"primary_rate_pct": 4}},
                       {"calibration": {"width": 0.75}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig("2023-09-01", "2023-09-29", **kwargs)

    def test_completed_stage_is_reused_without_action(self):
        store = self.store()
        target = self.root / "result.txt"
        target.write_text("original")
        count = []
        def action(attempt):
            count.append(attempt)
            return dict(status="completed", folder=str(self.root), files={str(target): sha256(target)})
        with store.locked(), redirect_stdout(io.StringIO()):
            store.execute("date1", "same", action)
        with self.store().locked() as resumed, redirect_stdout(io.StringIO()):
            resumed.execute("date1", "same", action)
        self.assertEqual(count, [1])

    def test_failed_date_resumes_without_rerunning_completed_date(self):
        attempted = []
        def successful(attempt):
            attempted.append("good")
            return dict(status="completed", files={})
        def interrupted(attempt):
            raise KeyboardInterrupt("injected interruption")
        with self.store().locked() as store, redirect_stdout(io.StringIO()):
            store.execute("date1", "one", successful)
            with self.assertRaises(KeyboardInterrupt):
                store.execute("date2", "two", interrupted)
        with self.store().locked() as store, redirect_stdout(io.StringIO()):
            store.execute("date1", "one", successful)
            store.execute("date2", "two", successful)
            self.assertEqual(store.data["stages"]["date2"]["attempts"], 2)
        self.assertEqual(attempted, ["good", "good"])

    def test_missing_or_changed_cached_output_is_rejected(self):
        path = self.root / "output.txt"
        path.write_text("original")
        with self.store().locked() as store:
            store.execute("job", "signature", lambda _: dict(status="completed", files={str(path): sha256(path)}))
        path.write_text("changed")
        with self.store().locked() as store, self.assertRaisesRegex(ValueError, "output changed"):
            store.execute("job", "signature", lambda _: self.fail("Must not silently recompute"))
        path.unlink()
        with self.store().locked() as store, self.assertRaises(ValueError):
            store.execute("job", "signature", lambda _: None)

    def test_configuration_and_stage_input_changes_are_rejected(self):
        with self.store().locked() as store:
            store.execute("job", "first", lambda _: dict(status="completed", files={}))
        with self.assertRaisesRegex(ValueError, "environment changed"):
            with ProgressStore(self.root / "progress", {"config": 2}).locked():
                pass
        with self.store().locked() as store, self.assertRaisesRegex(ValueError, "Inputs changed"):
            store.execute("job", "different", lambda _: None)

    def test_concurrent_run_is_blocked_and_lock_is_released(self):
        with self.store().locked():
            with self.assertRaisesRegex(RuntimeError, "Another runner"):
                with self.store().locked():
                    pass
        with self.store().locked():
            pass

    def test_audited_model_failures_are_not_promoted_to_success(self):
        with self.store().locked() as store, redirect_stdout(io.StringIO()):
            store.execute("model", "same", lambda _: dict(status="completed_with_failures", files={}))
            store.execute("model", "same", lambda _: self.fail("Failure must remain visible"))
            self.assertEqual(store.data["stages"]["model"]["status"], "completed_with_failures")

    def test_skipped_validation_is_explicit_and_cached(self):
        runner = self.repo()
        with runner.store.locked(), redirect_stdout(io.StringIO()):
            runner.skip("2023-09", "validation", "No dates requested.")
            runner.skip("2023-09", "validation", "No dates requested.")
            record = runner.store.data["stages"]["2023-09/validation"]
            self.assertEqual(record["status"], "skipped")
            self.assertEqual(record["attempts"], 1)
            self.assertIn("No dates", record["message"])

    def test_output_checksums_and_missing_declarations_are_checked(self):
        folder = self.root / "carry"
        self.artifact(folder)
        verify_artifact(folder, "carry")
        (folder / "calibration_quotes.csv").write_text("tampered")
        with self.assertRaises(ValueError):
            verify_artifact(folder, "carry")
        audit = self.artifact(folder)
        del audit["output_sha256"]["primary_carry.csv"]
        atomic_json(folder / "audit.json", audit)
        with self.assertRaisesRegex(ValueError, "Unrecorded"):
            verify_artifact(folder, "carry")

    def test_artifact_paths_cannot_escape_the_run(self):
        outside = self.root / "outside.csv"
        outside.write_text("outside")
        folder = self.root / "carry"
        audit = self.artifact(folder)
        audit["output_sha256"]["../outside.csv"] = sha256(outside)
        atomic_json(folder / "audit.json", audit)
        with self.assertRaisesRegex(ValueError, "path mismatch"):
            verify_artifact(folder, "carry")

    def test_checkpoint_must_consume_exact_inputs_even_after_relocation(self):
        source = self.root / "quotes.csv"
        source.write_text("source")
        audit = {"input_sha256": {"/old/mac/repo/quotes.csv": sha256(source)}}
        assert_consumed(audit, [source])
        source.write_text("different")
        with self.assertRaisesRegex(ValueError, "selected input"):
            assert_consumed(audit, [source])

    def test_legacy_preparation_without_output_hashes_can_be_pinned(self):
        folder = self.root / "prepared"
        self.artifact(folder, "prepare", extra={"output_sha256": {}})
        _, files = verify_artifact(folder, "prepare")
        self.assertIn(str((folder / "pilot_quotes.csv").resolve()), files)

    def test_csv_aggregation_preserves_columns_and_failure_rows(self):
        first, second, target = [self.root / name for name in ("first.csv", "second.csv", "all.csv")]
        write_rows(first, [dict(date="one", status="fitted", metric="1")])
        write_rows(second, [dict(date="two", status="failed", message="failure")])
        combine_csv([first, second], target)
        combined = rows(target)
        self.assertEqual(combined[1]["status"], "failed")
        self.assertEqual(combined[1]["metric"], "")
        self.assertEqual(combined[0]["message"], "")

    def test_native_model_paths_remain_inside_aggregate_folder(self):
        runner = self.repo()
        parent = runner.output / "2023-09/calibration"
        parts = []
        for d in ("2023-09-01", "2023-09-05"):
            folder = parent / "days" / d / "attempt_001/results/run_test"
            folder.mkdir(parents=True)
            model = folder / "model.json"
            model.write_text('{"model": "fixture"}')
            write_rows(folder / "model_manifest.csv", [dict(quote_date=d, root="UNKNOWN", status="fitted",
                       model_file="model.json", model_sha256=sha256(model))])
            atomic_json(folder / "audit.json", {"settings": {}, "dates": [d], "failed_daily_models": 0,
                "output_sha256": {"model.json": sha256(model), "model_manifest.csv": sha256(folder / "model_manifest.csv")}})
            parts.append(folder)
        with runner.store.locked(), redirect_stdout(io.StringIO()):
            folder = runner.aggregate("2023-09", "calibration", parts, [], ["2023-09-01", "2023-09-05"])
            verify_artifact(folder, "calibration")
        for r in rows(folder / "model_manifest.csv"):
            self.assertTrue((folder / r["model_file"]).resolve().is_relative_to(folder))

    def policy(self, end="2023-12-31"):
        import pandas as pd
        path = self.root / "policy.csv"
        dates = pd.bdate_range("2023-08-01", end)
        write_rows(path, [dict(quote_date=d.strftime("%Y-%m-%d"), reference_session=True,
            early_cash_close=False, session_policy="provisional_standard_session") for d in dates])
        return path

    def test_plan_handles_cross_month_exit_without_changing_entry_end(self):
        self.policy()
        raw = self.root / "raw"
        raw.mkdir()
        for m in ("202309", "202310"):
            (raw / f"spx_eod_{m}.txt").touch()
        runner = self.repo(raw_directory="raw", session_policy="policy.csv", include_next_session=True)
        job = runner.plan()[0]
        self.assertEqual(job["entry_end"], "2023-09-29")
        self.assertEqual(job["observation_end"], "2023-10-02")
        self.assertEqual(len(job["raw_files"]), 2)
        self.assertFalse(runner.output.exists())

    def test_missing_calendar_horizon_or_month_is_not_fabricated(self):
        self.policy("2023-10-31")
        runner = self.repo(raw_directory="raw", session_policy="policy.csv")
        with self.assertRaisesRegex(ValueError, "60-day"):
            runner.plan()
        self.policy()
        with self.assertRaisesRegex(FileNotFoundError, "Missing raw month"):
            runner.plan()

    def test_sources_are_part_of_restart_identity(self):
        runner = self.repo()
        with runner.store.locked():
            runner.store.save()
        (self.root / "src/new_method.py").write_text("# new source\n")
        changed = ResearchPipeline(self.root, runner.config)
        with self.assertRaisesRegex(ValueError, "source code"):
            with changed.store.locked():
                pass

    def test_explicit_checkpoint_is_verified_without_launching_a_process(self):
        input_file = self.root / "pilot_quotes.csv"
        input_file.write_text("input")
        folder = self.root / "saved"
        self.artifact(folder, inputs=[input_file])
        runner = self.repo(checkpoints={"2023-09": {"carry": "saved"}})
        with runner.store.locked(), redirect_stdout(io.StringIO()):
            result = runner.stage("2023-09", "carry", [], [input_file], lambda f, a: None)
            self.assertEqual(result, folder.resolve())
            self.assertEqual(runner.store.data["stages"]["2023-09/carry"]["origin"], "explicit_checkpoint")
        self.assertFalse((runner.output / "2023-09/carry").exists())

    def test_process_failures_leave_logs_and_resume_into_a_new_attempt(self):
        runner = self.repo()
        script = self.root / "scripts" / SCRIPTS["carry"]
        script.write_text('raise RuntimeError("injected process failure")\n')
        # The runner has already pinned this source; change only inside this controlled test.
        with runner.store.locked(), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "no unique completed audit"):
                runner.stage("2023-09", "carry", [], [], lambda f, a: None)
            self.assertEqual(runner.store.data["stages"]["2023-09/carry"]["status"], "failed")
        log = runner.output / "2023-09/carry/attempt_001/process.log"
        self.assertIn("injected process failure", log.read_text())
        self.assertTrue(log.is_file())
        script.write_text('''import argparse, json, hashlib
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--output"); a=p.parse_args()
folder=Path(a.output); folder.mkdir(parents=True)
names=("carry_inputs.csv", "primary_carry.csv", "calibration_quotes.csv", "daily_summary.csv")
for name in names: (folder/name).write_text("value\\nsaved\\n")
hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
(folder/"audit.json").write_text(json.dumps({"output_sha256":hashes, "input_sha256":{}}))
''')
        with runner.store.locked(), redirect_stdout(io.StringIO()):
            result = runner.stage("2023-09", "carry", [], [], lambda f, a: None)
            self.assertEqual(runner.store.data["stages"]["2023-09/carry"]["attempts"], 2)
        self.assertIn("attempt_002", str(result))
        self.assertIn("injected process failure", log.read_text())

    def test_next_month_endpoint_quote_does_not_become_an_entry(self):
        import pandas as pd
        from pilot_hedge_panel import PilotContractSelector
        self.policy()
        stamp = lambda d: pd.Timestamp(f"{d} 16:00", tz="America/New_York").tz_convert("UTC")
        data = []
        for d in ("2023-09-29", "2023-10-02"):
            for kind in ("call", "put"):
                data.append(dict(quote_date=d, root="UNKNOWN", expire_date="2023-10-27", kind=kind,
                    strike=100., underlying_last=100., bid=2., ask=2.2, mid=2.1,
                    quote_timestamp_utc=stamp(d), assumed_fixing_utc=stamp("2023-10-27"),
                    assumed_maturity_years=(stamp("2023-10-27")-stamp(d)).total_seconds()/(365*86400),
                    entry_research_candidate=True))
        q = pd.DataFrame(data)
        timeline = pd.read_csv(self.root / "policy.csv")
        timeline = timeline.loc[timeline.quote_date.between("2023-09-29", "2023-10-02")]
        selector = PilotContractSelector()
        entries, _ = selector.select(q, timeline, ["2023-09-29"])
        pairs = selector.match_next(entries, q, q.iloc[:0].copy(), timeline)
        self.assertEqual(set(entries.quote_date), {"2023-09-29"})
        self.assertEqual(set(pairs.end_date), {"2023-10-02"})
        self.assertTrue(pairs.end_status.eq("matched").all())


if __name__ == "__main__":
    unittest.main()
