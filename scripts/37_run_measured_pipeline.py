"""Measure the existing pipeline command and prepare explicit calendar support."""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


DEFAULT_SOURCE_POLICY = (
    "outputs/hedging_data_diagnostics/historical_2013_2023/provisional_session_policy.csv"
)
STAGES = ("prepare", "carry", "calibration", "validation", "panel", "comparison",
          "attribution", "notebook")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def xnys_schedule(start, end):
    """Use the same cash-session reference provider as the historical audit."""
    import exchange_calendars as xc
    import pandas as pd

    calendar = xc.get_calendar("XNYS", start=pd.Timestamp(start) - pd.Timedelta(days=7),
                               end=pd.Timestamp(end) + pd.Timedelta(days=7))
    schedule = calendar.schedule.loc[start:end, ["open", "close"]]
    closes = schedule["close"].dt.tz_convert("America/New_York")
    return {stamp.strftime("%Y-%m-%d"): close for stamp, close in closes.items()}, xc.__version__


class CalendarPolicyExtension:
    """Copy historical policy rows and append calendar-only, unaudited dates."""

    def __init__(self, source, target, end="2024-01-31"):
        self.source, self.target = Path(source).resolve(), Path(target).resolve()
        self.end = date.fromisoformat(end)
        if self.end.isoformat() != end:
            raise ValueError("Calendar end must use YYYY-MM-DD.")
        if self.source == self.target:
            raise ValueError("Calendar support must use a separate policy file.")

    def prepare(self):
        with self.source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns, original = list(reader.fieldnames or []), list(reader)
        required = {"quote_date", "reference_session", "early_cash_close", "session_policy"}
        if not required.issubset(columns) or not original:
            raise ValueError("Source policy requires dated, nonempty session-policy rows.")
        dates = [date.fromisoformat(row["quote_date"]) for row in original]
        if (len(set(dates)) != len(dates)
                or any(d.isoformat() != row["quote_date"] for d, row in zip(dates, original))):
            raise ValueError("Source policy dates must be unique YYYY-MM-DD values.")
        if any(row["reference_session"].lower() not in ("true", "false") for row in original):
            raise ValueError("Source reference-session flags must be true or false.")
        if any(row["early_cash_close"].lower() not in ("true", "false")
               for row in original if row["reference_session"].lower() == "true"):
            raise ValueError("Source reference sessions require an early-close flag.")
        audit_path = self.target.with_name(self.target.stem + "_audit.json")
        source_hash = digest(self.source)
        if self.target.exists() or audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (audit.get("source_policy_sha256") != source_hash
                    or audit.get("requested_calendar_end") != self.end.isoformat()
                    or audit.get("output_sha256", {}).get(self.target.name) != digest(self.target)):
                raise ValueError("Saved calendar support differs from its recorded inputs or output.")
            print("REUSE calendar support:", self.target)
            return audit
        start = max(dates) + timedelta(days=1)
        additions, version = [], None
        if start <= self.end:
            schedule, version = xnys_schedule(start.isoformat(), self.end.isoformat())
            current = start
            while current <= self.end:
                value = current.isoformat()
                close = schedule.get(value)
                row = {column: "" for column in columns}
                row.update(quote_date=value, reference_session=str(close is not None),
                           early_cash_close=str(close is not None and close.hour < 16),
                           session_policy="calendar_only_not_audited")
                if "cash_close_ny" in columns and close is not None:
                    row["cash_close_ny"] = close.isoformat()
                if "status" in columns:
                    row["status"] = "calendar_only_not_audited"
                for flag in ("contract_identity_verified", "snapshot_provenance_verified",
                             "daily_carry_verified", "backtest_approved"):
                    if flag in columns:
                        row[flag] = "False"
                additions.append(row)
                current += timedelta(days=1)
        if digest(self.source) != source_hash:
            raise RuntimeError("Source policy changed during calendar preparation.")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.target.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(original)
            writer.writerows(additions)
        temporary.replace(self.target)
        audit = dict(status="completed", source_policy=str(self.source),
                     source_policy_sha256=source_hash, requested_calendar_end=self.end.isoformat(),
                     original_rows=len(original), appended_calendar_rows=len(additions),
                     calendar="XNYS cash-session reference", exchange_calendars_version=version,
                     historical_rows_preserved=True, observed_rows_added=0, quotes_created=False,
                     calendar_only_rows_eligible_for_entries=False,
                     historical_exchange_calendar_verified=False,
                     contract_identity_verified=False, snapshot_provenance_verified=False,
                     scope="Calendar support only; appended rows have no audited observations.",
                     source_sha256={Path(__file__).name: digest(__file__)},
                     output_sha256={self.target.name: digest(self.target)})
        atomic_json(audit_path, audit)
        print(f"Saved calendar support: {self.target}; {len(additions)} calendar-only rows.")
        return audit


class MeasuredPipeline:
    """Record elapsed wall time for each real invocation, including reused work."""

    def __init__(self, repo, config_path, *, output=None, ignore_checkpoints=False,
                 stop_after="notebook"):
        self.repo = Path(repo).resolve()
        self.config_path = (self.repo / config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.output = (self.repo / (output or self.config["output"])).resolve()
        self.script = self.repo / "scripts/36_run_research_pipeline.py"
        if not self.script.is_file():
            raise FileNotFoundError(f"Missing existing pipeline runner: {self.script}")
        self.stop_after = stop_after
        self.command = [sys.executable, "-u", str(self.script), "--config", str(self.config_path),
                        "--repo-root", str(self.repo), "--stop-after", stop_after]
        if output:
            self.command += ["--output", str(output)]
        if ignore_checkpoints:
            self.command.append("--ignore-checkpoints")
        self.environment = os.environ.copy()
        self.environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.repo / "src"),
                                                                     self.environment.get("PYTHONPATH", "")]))

    def read_only(self, mode):
        return subprocess.run([*self.command, mode], cwd=self.repo, env=self.environment).returncode

    def _save(self, report_path, report):
        atomic_json(report_path, report)
        # Separate invocations have separate files; protect the combined index.
        import fcntl
        with (self.output / ".runtime.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reports = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((self.output / "runtime").glob("*/timing.json"))]
            columns = ["invocation_id", "started_utc", "finished_utc", "elapsed_wall_seconds",
                       "timing_status", "command_returncode", "stop_after", "report_file", "terminal_log"]
            temporary = self.output / "runtime_index.csv.tmp"
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows({key: value.get(key, "") for key in columns} for value in reports)
            temporary.replace(self.output / "runtime_index.csv")

    @staticmethod
    def _interrupt(process):
        """Give the runner time to save progress, then clean up its process group."""
        if process.poll() is not None:
            return
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                pass

    def run(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        invocation = "run_" + stamp + "_" + uuid.uuid4().hex[:8]
        folder = self.output / "runtime" / invocation
        folder.mkdir(parents=True, exist_ok=False)
        report_path, log_path = folder / "timing.json", folder / "terminal.log"
        report = dict(invocation_id=invocation, started_utc=None, finished_utc=None,
                      elapsed_wall_seconds=None, timing_status="starting", command_returncode=None,
                      stop_after=self.stop_after, command=self.command, repo_root=str(self.repo),
                      config_file=str(self.config_path), config_sha256=digest(self.config_path),
                      source_sha256={str(self.script): digest(self.script), str(Path(__file__).resolve()): digest(__file__)},
                      report_file=str(report_path), terminal_log=str(log_path),
                      clock="time.perf_counter; monotonic elapsed wall time",
                      scope="Measured execution block from initial invocation registration through child exit/cleanup; includes process startup, output relay, verification/reuse and executed stages. Initial launcher setup and final report saving are excluded.",
                      resumed_runs="Each invocation measured separately; idle time between invocations excluded.",
                      stage_times_summed=False, runtime_estimated=False)
        process = None
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                report["started_utc"] = datetime.now(timezone.utc).isoformat()
                report["timing_status"] = "running"
                started = time.perf_counter()
                try:
                    self._save(report_path, report)
                    process = subprocess.Popen(self.command, cwd=self.repo, env=self.environment, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                               errors="replace", bufsize=1, start_new_session=True)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    returncode = process.wait()
                    report["timing_status"] = "finished" if returncode == 0 else "failed"
                except KeyboardInterrupt:
                    if process is not None:
                        self._interrupt(process)
                    returncode = 130
                    report["timing_status"] = "interrupted"
                except BaseException as error:
                    if process is not None:
                        self._interrupt(process)
                    returncode = 1
                    report["timing_status"] = "failed"
                    report["error"] = f"{type(error).__name__}: {error}"
                    print(report["error"], file=sys.stderr)
                finally:
                    report["elapsed_wall_seconds"] = time.perf_counter() - started
                    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    if process is not None and process.stdout is not None:
                        process.stdout.close()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        report["command_returncode"] = returncode
        state_path = self.output / "pipeline_state.json"
        if state_path.is_file():
            try:
                report["pipeline_status_at_finish"] = json.loads(state_path.read_text())["status"]
            except (OSError, ValueError, KeyError):
                report["pipeline_status_at_finish"] = "unreadable"
        self._save(report_path, report)
        print(f"\nMeasured invocation wall time: {report['elapsed_wall_seconds']:.3f} seconds "
              f"({report['elapsed_wall_seconds'] / 60:.3f} minutes).")
        print("Timing:", report_path)
        print("Terminal log:", log_path)
        print("Runtime history:", self.output / "runtime_index.csv")
        return 128 - returncode if returncode < 0 else returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", help="Override the pipeline output folder.")
    parser.add_argument("--ignore-checkpoints", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES, default="notebook")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-calendar", action="store_true", help="Prepare calendar support only; no pipeline work.")
    mode.add_argument("--plan", action="store_true", help="Delegate the read-only plan; no timing files.")
    mode.add_argument("--status", action="store_true", help="Delegate read-only status; no timing files.")
    parser.add_argument("--source-policy", type=Path, default=Path(DEFAULT_SOURCE_POLICY))
    parser.add_argument("--calendar-end", default="2024-01-31")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    runner = MeasuredPipeline(repo, args.config, output=args.output,
                              ignore_checkpoints=args.ignore_checkpoints, stop_after=args.stop_after)
    if args.prepare_calendar:
        CalendarPolicyExtension(repo / args.source_policy, repo / runner.config["session_policy"],
                                args.calendar_end).prepare()
        return 0
    if args.plan or args.status:
        return runner.read_only("--plan" if args.plan else "--status")
    return runner.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from error
