from contextlib import redirect_stdout
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/37_run_measured_pipeline.py"
spec = importlib.util.spec_from_file_location("measured_pipeline_script", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MeasuredPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/36_run_research_pipeline.py"
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps({"output": "results", "session_policy": "derived.csv"}))
        self.script.write_text(
            'import argparse, json\n'
            'from pathlib import Path\n'
            'p = argparse.ArgumentParser()\n'
            'p.add_argument("--config")\n'
            'p.add_argument("--repo-root")\n'
            'p.add_argument("--output")\n'
            'p.add_argument("--stop-after")\n'
            'p.add_argument("--ignore-checkpoints", action="store_true")\n'
            'p.add_argument("--plan", action="store_true")\n'
            'p.add_argument("--status", action="store_true")\n'
            'a = p.parse_args()\n'
            'if a.plan or a.status:\n'
            '    print("read-only")\n'
            'else:\n'
            '    config = json.loads(Path(a.config).read_text())\n'
            '    out = Path(a.repo_root) / (a.output or config["output"])\n'
            '    out.mkdir(parents=True, exist_ok=True)\n'
            '    status = "completed" if a.stop_after == "notebook" else "paused"\n'
            '    (out / "pipeline_state.json").write_text(json.dumps({"status": status}))\n'
            '    (out / "stage_index.csv").write_text("job,seconds\\nold/stage,9999\\n")\n'
            '    print("child output")\n'
        )

    def runner(self, **kwargs):
        return module.MeasuredPipeline(self.root, self.config_path, **kwargs)

    def reports(self, output="results"):
        return [json.loads(path.read_text())
                for path in sorted((self.root / output / "runtime").glob("*/timing.json"))]

    def test_elapsed_time_is_measured_and_does_not_sum_saved_stage_times(self):
        runner = self.runner()
        with patch.object(module.time, "perf_counter", side_effect=[100.0, 112.5]), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(), 0)
        report = self.reports()[0]
        self.assertEqual(report["elapsed_wall_seconds"], 12.5)
        self.assertEqual(report["command_returncode"], 0)
        self.assertEqual(report["pipeline_status_at_finish"], "completed")
        self.assertFalse(report["runtime_estimated"])
        self.assertFalse(report["stage_times_summed"])
        self.assertEqual(Path(report["terminal_log"]).read_text(), "child output\n")
        self.assertEqual(report["config_sha256"], module.digest(self.config_path))

    def test_real_child_startup_and_work_are_included(self):
        self.script.write_text('import time\ntime.sleep(0.03)\nprint("measured control")\n')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner().run(), 0)
        self.assertGreaterEqual(self.reports()[0]["elapsed_wall_seconds"], 0.03)

    def test_selected_repository_source_path_is_available_to_the_child(self):
        (self.root / "src").mkdir()
        (self.root / "src/controlled_import.py").write_text('VALUE = "selected repository"\n')
        self.script.write_text('from controlled_import import VALUE\nprint(VALUE)\n')
        with patch.dict(module.os.environ, {"PYTHONPATH": "an unrelated relative directory"}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner().run(), 0)
        self.assertIn("selected repository", Path(self.reports()[0]["terminal_log"]).read_text())

    def test_failed_child_preserves_its_exit_code_log_and_timing(self):
        self.script.write_text('import sys\nprint("controlled failure", file=sys.stderr)\nraise SystemExit(7)\n')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner().run(), 7)
        report = self.reports()[0]
        self.assertEqual(report["timing_status"], "failed")
        self.assertEqual(report["command_returncode"], 7)
        self.assertIn("controlled failure", Path(report["terminal_log"]).read_text())
        self.assertGreater(report["elapsed_wall_seconds"], 0)

    def test_paused_and_resumed_invocations_have_separate_measurements(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner(stop_after="calibration").run(), 0)
            self.assertEqual(self.runner().run(), 0)
        reports = self.reports()
        self.assertEqual(len(reports), 2)
        self.assertEqual([r["pipeline_status_at_finish"] for r in reports], ["paused", "completed"])
        self.assertNotEqual(reports[0]["invocation_id"], reports[1]["invocation_id"])
        self.assertEqual(reports[0]["stop_after"], "calibration")
        with (self.root / "results/runtime_index.csv").open(newline="") as stream:
            self.assertEqual(len(list(module.csv.DictReader(stream))), 2)

    def test_read_only_plan_and_status_do_not_create_runtime_files(self):
        runner = self.runner()
        self.assertEqual(runner.read_only("--plan"), 0)
        self.assertEqual(runner.read_only("--status"), 0)
        self.assertFalse(runner.output.exists())

    def test_output_override_with_spaces_is_forwarded_without_shell_parsing(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner(output="a separate output", ignore_checkpoints=True).run(), 0)
        report = self.reports("a separate output")[0]
        self.assertIn("--ignore-checkpoints", report["command"])
        self.assertIn("a separate output", report["command"])
        self.assertFalse((self.root / "results").exists())

    def test_interrupt_stops_child_group_and_saves_incomplete_invocation(self):
        def interrupted_stream():
            yield "started child\n"
            raise KeyboardInterrupt
        process = Mock(pid=12345)
        process.stdout = Mock()
        process.stdout.__iter__ = Mock(return_value=interrupted_stream())
        process.poll.return_value = None
        process.wait.return_value = 0
        with patch.object(module.subprocess, "Popen", return_value=process), \
                patch.object(module.os, "killpg") as kill, redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner().run(), 130)
        kill.assert_called_once_with(12345, module.signal.SIGINT)
        report = self.reports()[0]
        self.assertEqual(report["timing_status"], "interrupted")
        self.assertEqual(report["command_returncode"], 130)
        self.assertGreaterEqual(report["elapsed_wall_seconds"], 0)


class CalendarSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source, self.target = self.root / "source.csv", self.root / "derived/policy.csv"
        self.original = [dict(quote_date="2023-10-31", reference_session="True",
                              early_cash_close="False", session_policy="provisional_standard_session",
                              raw_rows="8192", status="observed_reference_session", cash_close_ny="",
                              contract_identity_verified="False"),
                         dict(quote_date="2023-12-29", reference_session="True",
                              early_cash_close="False", session_policy="missing_observation",
                              raw_rows="0", status="missing_session_observation", cash_close_ny="",
                              contract_identity_verified="False")]
        self.write_source()

    def write_source(self):
        with self.source.open("w", newline="") as stream:
            writer = module.csv.DictWriter(stream, fieldnames=list(self.original[0]))
            writer.writeheader()
            writer.writerows(self.original)

    def extension(self):
        return module.CalendarPolicyExtension(self.source, self.target)

    def schedule(self):
        # Synthetic provider control, not a claimed historical schedule.
        return {"2024-01-02": datetime.fromisoformat("2024-01-02T16:00:00-05:00"),
                "2024-01-03": datetime.fromisoformat("2024-01-03T13:00:00-05:00")}, "test-provider"

    def test_provider_adapter_uses_cash_closes_in_new_york_and_records_version(self):
        import pandas as pd
        schedule = pd.DataFrame({"open": pd.to_datetime(["2024-01-02T14:30Z", "2024-01-03T14:30Z"]),
                                 "close": pd.to_datetime(["2024-01-02T21:00Z", "2024-01-03T18:00Z"])},
                                index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        provider = SimpleNamespace(__version__="controlled-version", get_calendar=Mock(
            return_value=SimpleNamespace(schedule=schedule)))
        with patch.dict(module.sys.modules, {"exchange_calendars": provider}):
            closes, version = module.xnys_schedule("2023-12-30", "2024-01-31")
        self.assertEqual(version, "controlled-version")
        self.assertEqual(closes["2024-01-02"].hour, 16)
        self.assertEqual(closes["2024-01-03"].hour, 13)
        self.assertEqual(provider.get_calendar.call_args.args, ("XNYS",))

    def test_extension_preserves_audited_rows_and_adds_no_observed_data(self):
        source_hash = module.digest(self.source)
        with patch.object(module, "xnys_schedule", return_value=self.schedule()) as provider, \
                redirect_stdout(io.StringIO()):
            audit = self.extension().prepare()
        provider.assert_called_once_with("2023-12-30", "2024-01-31")
        with self.target.open(newline="") as stream:
            output = list(module.csv.DictReader(stream))
        self.assertEqual(output[:2], self.original)
        self.assertEqual(module.digest(self.source), source_hash)
        added = {row["quote_date"]: row for row in output[2:]}
        self.assertEqual(len(added), 33)
        self.assertEqual(max(added), "2024-01-31")
        self.assertEqual(added["2024-01-02"]["reference_session"], "True")
        self.assertEqual(added["2024-01-03"]["early_cash_close"], "True")
        self.assertEqual(added["2024-01-01"]["reference_session"], "False")
        self.assertTrue(all(row["raw_rows"] == "" for row in added.values()))
        self.assertTrue(all(row["session_policy"] == "calendar_only_not_audited" for row in added.values()))
        self.assertEqual(audit["observed_rows_added"], 0)
        self.assertFalse(audit["calendar_only_rows_eligible_for_entries"])

    def test_repeated_preparation_reuses_verified_policy_and_rejects_changed_output(self):
        with patch.object(module, "xnys_schedule", return_value=self.schedule()) as provider, \
                redirect_stdout(io.StringIO()):
            self.extension().prepare()
            self.extension().prepare()
            self.assertEqual(provider.call_count, 1)
            self.target.write_text(self.target.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                self.extension().prepare()

    def test_changed_source_is_rejected_after_preparation(self):
        with patch.object(module, "xnys_schedule", return_value=self.schedule()), redirect_stdout(io.StringIO()):
            self.extension().prepare()
            self.original[0]["session_policy"] = "snapshot_review"
            self.write_source()
            with self.assertRaisesRegex(ValueError, "differs"):
                self.extension().prepare()

    def test_source_cannot_be_overwritten_and_duplicate_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            module.CalendarPolicyExtension(self.source, self.source)
        self.original.append(dict(self.original[0]))
        self.write_source()
        with self.assertRaisesRegex(ValueError, "unique"):
            self.extension().prepare()
        self.assertFalse(self.target.exists())


class FullOctoberPlanTests(unittest.TestCase):
    def test_full_month_plan_includes_november_exit_and_calendar_support(self):
        from research_pipeline import PipelineConfig, ResearchPipeline, SCRIPTS, write_rows
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "scripts").mkdir()
            (root / "src").mkdir()
            for name in SCRIPTS.values():
                (root / "scripts" / name).touch()
            config_path = Path(__file__).resolve().parents[1] / "config/research_pipeline_october_full.json"
            config = PipelineConfig.load(config_path)
            source = root / "source.csv"
            write_rows(source, [dict(quote_date=d, reference_session=True, early_cash_close=False,
                                    session_policy="provisional_standard_session")
                                for d in ["2023-10-02", "2023-10-03", "2023-10-16", "2023-10-23",
                                          "2023-10-31", "2023-11-01", "2023-12-29"]])
            target = root / config.session_policy
            with patch.object(module, "xnys_schedule", return_value=({}, "test-provider")), \
                    redirect_stdout(io.StringIO()):
                module.CalendarPolicyExtension(source, target).prepare()
            raw = root / config.raw_directory
            raw.mkdir(parents=True)
            for name in ("spx_eod_202310.txt", "spx_eod_202311.txt"):
                (raw / name).touch()
            runner = ResearchPipeline(root, config)
            job = runner.plan()[0]
            self.assertEqual(job["entry_end"], "2023-10-31")
            self.assertEqual(job["observation_end"], "2023-11-01")
            self.assertEqual(len(job["raw_files"]), 2)
            self.assertEqual(job["validation_dates"], config.validation_dates)
            self.assertFalse(job["checkpoints"])
            self.assertFalse(runner.output.exists())


if __name__ == "__main__":
    unittest.main()
