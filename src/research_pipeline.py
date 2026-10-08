"""Resume the existing research stages without changing their pricing methods.

Paths in a configuration are relative to the repository root. Saved checkpoints
are explicit inputs, never selected by newest-folder sorting. Successful process
execution and numerical/economic validation are deliberately separate statuses.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import calendar
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


STAGES = ("prepare", "carry", "calibration", "validation", "panel", "comparison",
          "attribution", "notebook")
SCRIPTS = {"prepare": "26_prepare_hedging_pilot.py", "carry": "28_prepare_pilot_carry.py",
           "calibration": "29_calibrate_daily_ah.py", "validation": "30_validate_daily_ah.py",
           "panel": "32_build_pilot_hedge_panel.py", "comparison": "33_compare_pilot_hedges.py",
           "attribution": "34_attribute_pilot_hedges.py", "notebook": "35_build_pilot_notebook.py"}
REQUIRED = {
    "prepare": ("pilot_quotes.csv", "zero_bid_bounds.csv", "session_timeline.csv",
                "quote_selection_summary.csv", "expiry_coverage.csv"),
    "carry": ("carry_inputs.csv", "primary_carry.csv", "calibration_quotes.csv", "daily_summary.csv"),
    "calibration": ("model_manifest.csv",),
    "validation": ("study_status.csv",),
    "panel": ("delta_panel.csv", "entries.csv", "selection_buckets.csv", "endpoint_pairs.csv",
              "daily_status.csv", "daily_summary.csv"),
    "comparison": ("coverage.csv", "strategy_results.csv", "paired_results.csv", "summary.csv"),
    "attribution": ("coverage.csv", "summary.csv", "daily_attribution.csv", "leave_one_date_out.csv"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, records, columns=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    columns = columns or list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def combine_csv(paths, target):
    """Stream date reports into a monthly table, preserving the column union."""
    paths = [Path(path) for path in paths if Path(path).is_file()]
    if not paths:
        return False
    columns = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for key in next(csv.reader(stream)):
                if key not in columns:
                    columns.append(key)
    with Path(target).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for path in paths:
            with path.open(newline="", encoding="utf-8-sig") as source:
                writer.writerows(csv.DictReader(source))
    return True


def checked_date(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD.")
    return parsed


@dataclass(frozen=True)
class PipelineConfig:
    start: str
    end: str
    raw_directory: str = "data/raw"
    session_policy: str = "outputs/hedging_data_diagnostics/historical_2013_2023/provisional_session_policy.csv"
    output: str = "outputs/research_pipeline"
    include_next_session: bool = False
    calibration: dict = field(default_factory=lambda: dict(
        intervals=8000, width=1.0, smoothing=1.0, control_points=31, max_evaluations=300))
    carry: dict = field(default_factory=lambda: dict(primary_rate_pct=5.0, rates_pct=[3.0, 5.0, 7.0]))
    panel: dict = field(default_factory=lambda: dict(intervals=24000, steps_per_day=128))
    funding: dict = field(default_factory=lambda: dict(lend_rate=0.05, borrow_rate=0.05, hedge_fee_bps=1.0))
    validation_dates: list = field(default_factory=list)
    build_notebooks: bool = True
    checkpoints: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def __post_init__(self):
        if checked_date(self.start) > checked_date(self.end):
            raise ValueError("Start must not follow end.")
        for name in ("raw_directory", "session_policy", "output"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty path.")
        for name in ("include_next_session", "build_notebooks"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        defaults = {
            "calibration": dict(intervals=8000, width=1.0, smoothing=1.0, control_points=31, max_evaluations=300),
            "carry": dict(primary_rate_pct=5.0, rates_pct=[3.0, 5.0, 7.0]),
            "panel": dict(intervals=24000, steps_per_day=128),
            "funding": dict(lend_rate=0.05, borrow_rate=0.05, hedge_fee_bps=1.0),
        }
        for name, default in defaults.items():
            actual = getattr(self, name)
            if not isinstance(actual, dict) or set(actual) - set(default):
                raise ValueError(f"Unknown {name} setting.")
            object.__setattr__(self, name, {**default, **actual})
        for group in (self.calibration, self.panel, self.funding):
            if any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) for v in group.values()):
                raise ValueError("Numerical settings must be finite.")
        for group, keys in [(self.calibration, ("intervals", "control_points", "max_evaluations")),
                            (self.panel, ("intervals", "steps_per_day"))]:
            if any(not isinstance(group[k], int) or group[k] < 1 for k in keys):
                raise ValueError("Grid counts and iteration limits must be positive integers.")
        if (self.calibration["intervals"] % 2 or self.panel["intervals"] % 2
                or self.calibration["intervals"] < 100 or self.panel["intervals"] < 100
                or self.calibration["control_points"] < 3 or self.calibration["width"] <= 0.902
                or self.calibration["smoothing"] < 0 or self.funding["hedge_fee_bps"] <= 0):
            raise ValueError("Invalid grid, coefficient-domain room, regularization or fee setting.")
        if 2 * self.calibration["width"] / self.calibration["intervals"] > 0.0005 * (1 + 1e-10):
            raise ValueError("The fixed smoothing radius must cover at least one native AH cell.")
        if not isinstance(self.carry["rates_pct"], list) or not self.carry["rates_pct"]:
            raise ValueError("rates_pct must be a nonempty list.")
        rates = [self.carry["primary_rate_pct"], *self.carry["rates_pct"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rates):
            raise ValueError("Carry rates must be finite numbers.")
        if (self.carry["primary_rate_pct"] not in self.carry["rates_pct"]
                or len(set(self.carry["rates_pct"])) != len(self.carry["rates_pct"])):
            raise ValueError("Include the primary rate exactly once among the scenarios.")
        if not isinstance(self.validation_dates, list) or len(set(self.validation_dates)) != len(self.validation_dates):
            raise ValueError("Validation dates must be a unique list.")
        for value in self.validation_dates:
            if not checked_date(self.start) <= checked_date(value) <= checked_date(self.end):
                raise ValueError("Validation dates must lie inside the entry range.")
        valid_months = {m for m, _, _ in self.months()}
        if not isinstance(self.checkpoints, dict) or set(self.checkpoints) - valid_months:
            raise ValueError("Checkpoint months must lie in the configured range.")
        for checkpoint in self.checkpoints.values():
            if (not isinstance(checkpoint, dict) or set(checkpoint) - set(REQUIRED)
                    or any(not isinstance(v, str) or not v for v in checkpoint.values())):
                raise ValueError("Unknown checkpoint stage or invalid folder.")

    def months(self):
        first, end = checked_date(self.start), checked_date(self.end)
        current = first.replace(day=1)
        while current <= end:
            last = current.replace(day=calendar.monthrange(current.year, current.month)[1])
            yield current.strftime("%Y-%m"), max(first, current).isoformat(), min(end, last).isoformat()
            current = last + timedelta(days=1)


def verify_artifact(folder, stage):
    """Verify declared outputs and the manifest's model paths before reuse."""
    folder = Path(folder).resolve()
    audit_path = folder / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    hashes = audit.get("output_sha256", {})
    for filename in REQUIRED[stage]:
        if not (folder / filename).is_file():
            raise ValueError(f"Missing {stage} output: {filename}")
        if stage != "prepare" and filename not in hashes:
            raise ValueError(f"Unrecorded {stage} output: {filename}")
    files = {str(audit_path): sha256(audit_path)}
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or path == audit_path or sha256(path) != expected:
            raise ValueError(f"Output checksum/path mismatch: {stage}/{name}")
        files[str(path)] = expected
    for name in REQUIRED[stage]:
        path = folder / name
        files[str(path)] = sha256(path)
    if stage == "calibration":
        manifest = rows(folder / "model_manifest.csv")
        if not manifest or len({(r["quote_date"], r["root"]) for r in manifest}) != len(manifest):
            raise ValueError("Empty or duplicate calibration manifest.")
        for row in manifest:
            if row["status"] not in ("fitted", "failed"):
                raise ValueError("Unexpected calibration status.")
            if row["status"] == "fitted":
                path = (folder / row["model_file"]).resolve()
                if (not path.is_relative_to(folder) or sha256(path) != row["model_sha256"]
                        or hashes.get(row["model_file"]) != row["model_sha256"]):
                    raise ValueError("Model checksum/path mismatch.")
                files[str(path)] = row["model_sha256"]
    elif stage in ("panel", "comparison", "attribution"):
        if audit.get("status") not in ("completed", "completed_with_accounting_failures"):
            raise ValueError(f"Incomplete {stage} audit.")
    elif stage == "validation":
        if not audit.get("daily_status") or any(r["status"] != "completed" for r in audit["daily_status"]):
            raise ValueError("Incomplete validation run.")
    return audit, files


def assert_consumed(audit, inputs):
    recorded = audit.get("input_sha256", {})
    for path in inputs:
        path = Path(path)
        expected = sha256(path)
        if not any(Path(key).name == path.name and value == expected for key, value in recorded.items()):
            raise ValueError(f"Saved run does not consume the selected input: {path.name}")


class ProgressStore:
    """Atomic stage state and an OS-released lock for interruption-safe restarts."""

    def __init__(self, folder, identity):
        self.folder, self.identity = Path(folder), identity
        self.path = self.folder / "pipeline_state.json"
        self.data = None

    @contextmanager
    def locked(self):
        import fcntl  # macOS/Linux; release is automatic even after process termination.
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / ".pipeline.lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another runner is using this output folder.") from error
            try:
                if self.path.exists():
                    self.data = json.loads(self.path.read_text())
                    if self.data["identity"] != self.identity:
                        raise ValueError("Configuration, source code or environment changed. Use a new output folder.")
                else:
                    self.data = {"identity": self.identity, "status": "running", "stages": {}}
                yield self
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def save(self):
        self.data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, self.data)
        records = [{"job": key, **{k: row.get(k, "") for k in
                    ("status", "origin", "seconds", "attempts", "folder", "message")}}
                   for key, row in self.data["stages"].items()]
        write_rows(self.folder / "stage_index.csv", records,
                   ["job", "status", "origin", "seconds", "attempts", "folder", "message"])

    def execute(self, key, signature, action):
        existing = self.data["stages"].get(key, {})
        if existing.get("status") in ("completed", "completed_with_failures", "skipped"):
            if existing["signature"] != signature:
                raise ValueError(f"Inputs changed for {key}. Use a new output folder.")
            for path, expected in existing.get("files", {}).items():
                if not Path(path).is_file() or sha256(path) != expected:
                    raise ValueError(f"Saved output changed for {key}: {path}")
            print(f"REUSE {key}", flush=True)
            return Path(existing["folder"]) if existing.get("folder") else None
        record = {"status": "running", "signature": signature,
                  "attempts": existing.get("attempts", 0) + 1,
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        self.data["stages"][key] = record
        self.save()
        started = time.monotonic()
        try:
            result = action(record["attempts"])
            record.update(result, seconds=time.monotonic() - started)
        except BaseException as error:
            record.update(status="failed", seconds=time.monotonic() - started,
                          message=f"{type(error).__name__}: {error}")
            self.save()
            raise
        self.save()
        return Path(record["folder"]) if record.get("folder") else None


class ResearchPipeline:
    def __init__(self, repo, config, stop_after="notebook"):
        self.repo, self.config, self.stop_after = Path(repo).resolve(), config, stop_after
        self.output = self.path(config.output)
        self.policy = self.path(config.session_policy)
        if stop_after not in STAGES:
            raise ValueError("Unknown stopping stage.")
        for script in SCRIPTS.values():
            if not (self.repo / "scripts" / script).is_file():
                raise FileNotFoundError(f"Install the existing stage script: scripts/{script}")
        sources = [*sorted((self.repo / "src").glob("*.py")),
                   *[self.repo / "scripts" / name for name in SCRIPTS.values()],
                   self.repo / "scripts/36_run_research_pipeline.py"]
        self.sources = {str(p.relative_to(self.repo)): sha256(p) for p in sources if p.is_file()}
        from dataclasses import asdict
        self.configuration = asdict(config)
        versions = {"python": platform.python_version()}
        for name in ("numpy", "pandas", "scipy", "matplotlib"):
            versions[name] = importlib.metadata.version(name)
        self.identity = dict(configuration=self.configuration, sources=self.sources, versions=versions)
        self.store = ProgressStore(self.output, self.identity)

    def path(self, value):
        return (self.repo / value).resolve()

    def window(self, month, start, end):
        policy = rows(self.policy)
        dates = [r["quote_date"] for r in policy]
        if len(set(dates)) != len(dates):
            raise ValueError("Session policy contains duplicate dates.")
        observed_end = end
        if self.config.include_next_session:
            following = sorted(r["quote_date"] for r in policy if r["quote_date"] > end
                               and r["reference_session"].lower() == "true")
            if not following:
                raise ValueError("No next reference session in the policy; extend calendar coverage first.")
            observed_end = following[0]
        horizon = checked_date(observed_end) + timedelta(days=60)
        first = horizon.replace(day=1)
        third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
        if not dates or min(dates) > start or max(dates) < max(horizon, third_friday).isoformat():
            raise ValueError("Policy must cover the entry window and the full 60-day expiry horizon.")
        months = sorted({month, observed_end[:7]})
        raw = [self.path(self.config.raw_directory) / f"spx_eod_{m.replace('-', '')}.txt" for m in months]
        for path in raw:
            if not path.is_file():
                raise FileNotFoundError(f"Missing raw month: {path}")
        return observed_end, raw

    def plan(self):
        result = []
        for month, start, end in self.config.months():
            observed_end, raw = self.window(month, start, end)
            result.append(dict(month=month, entry_start=start, entry_end=end,
                               observation_end=observed_end, raw_files=[str(p) for p in raw],
                               checkpoints=self.config.checkpoints.get(month, {}),
                               validation_dates=[d for d in self.config.validation_dates if start <= d <= end]))
        return result

    def signature(self, stage, arguments, inputs):
        payload = dict(stage=stage, arguments=arguments,
                       inputs={str(Path(p).resolve()): sha256(p) for p in inputs})
        return hashlib.sha256(canonical(payload).encode()).hexdigest()

    def stage(self, month, stage, arguments, inputs, validator, *, date_key=None, parent=None):
        key = f"{month}/{stage}" + (f"/{date_key}" if date_key else "")
        checkpoint = self.config.checkpoints.get(month, {}).get(stage) if not date_key else None
        signature = self.signature(stage, {"argv": arguments, "checkpoint": checkpoint}, inputs)

        def action(attempt):
            before = {str(p): sha256(p) for p in inputs}
            if checkpoint:
                folder = self.path(checkpoint)
                origin = "explicit_checkpoint"
                print(f"VERIFY {key}: {folder}", flush=True)
            else:
                directory = (parent or self.output / month / stage)
                if date_key:
                    directory = directory / "days" / date_key
                attempt_dir = directory / f"attempt_{attempt:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, "-u", str(self.repo / "scripts" / SCRIPTS[stage]),
                           *map(str, arguments), "--output", str(attempt_dir / "results")]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.repo / "src"),
                                                        environment.get("PYTHONPATH", "")]))
                print(f"RUN {key}", flush=True)
                with (attempt_dir / "process.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=self.repo, env=environment,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    try:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end="", flush=True)
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                    finally:
                        process.stdout.close()
                candidates = [p.parent for p in (attempt_dir / "results").glob("*/audit.json")]
                if (attempt_dir / "results/audit.json").exists():
                    candidates.append(attempt_dir / "results")
                if len(candidates) != 1:
                    raise RuntimeError(f"Stage exit {code}; no unique completed audit. See {attempt_dir / 'process.log'}")
                folder, origin = candidates[0], "executed"
                # Script 29 intentionally exits 1 after exporting audited fit failures.
                audit = json.loads((folder / "audit.json").read_text())
                if code and not (stage == "calibration" and code == 1 and audit.get("failed_daily_models", 0) > 0):
                    raise RuntimeError(f"Stage exit {code}. See {attempt_dir / 'process.log'}")
            audit, files = verify_artifact(folder, stage)
            assert_consumed(audit, inputs)
            validator(folder, audit)
            if any(sha256(path) != digest for path, digest in before.items()):
                raise RuntimeError("Inputs changed during stage execution or checkpoint verification.")
            failed = audit.get("failed_daily_models", 0) > 0 or audit.get("status") == "completed_with_accounting_failures"
            if stage == "panel":
                failed = failed or any(count for field in ("black_status_counts", "ah_status_counts")
                    for status, count in audit.get(field, {}).items() if status != "ready")
            return dict(status="completed_with_failures" if failed else "completed", folder=str(folder),
                        origin=origin, files=files, input_sha256=before,
                        saved_source_sha256=audit.get("source_sha256", {}))
        return self.store.execute(key, signature, action)

    def aggregate(self, month, stage, folders, input_files, dates):
        target = self.output / month / stage
        inputs = [*input_files, *[folder / "audit.json" for folder in folders]]
        signature = self.signature(stage, {"aggregate_dates": dates}, inputs)

        def action(attempt):
            target.mkdir(parents=True, exist_ok=True)
            audits = [verify_artifact(folder, stage)[0] for folder in folders]
            audit = dict(audits[0])
            if stage == "calibration":
                records = []
                for folder in folders:
                    for row in rows(folder / "model_manifest.csv"):
                        if row["status"] == "fitted":
                            row["model_file"] = (folder / row["model_file"]).relative_to(target).as_posix()
                        records.append(row)
                write_rows(target / "model_manifest.csv", records)
                for name in ("quote_residuals", "expiry_summary", "conditioning", "shape_checks"):
                    combine_csv([folder / f"{name}.csv" for folder in folders], target / f"{name}.csv")
                audit.update(dates=dates, successful_daily_models=sum(r["status"] == "fitted" for r in records),
                             failed_daily_models=sum(r["status"] == "failed" for r in records))
            elif stage == "panel":
                for name in REQUIRED["panel"]:
                    combine_csv([folder / name for folder in folders], target / name)
                panel = rows(target / "delta_panel.csv")

                def counts(column):
                    result = {}
                    for row in panel:
                        result[row[column]] = result.get(row[column], 0) + 1
                    return result
                bucket_status = {}
                for row in rows(target / "selection_buckets.csv"):
                    bucket_status[row["status"]] = bucket_status.get(row["status"], 0) + 1
                audit.update(status="completed", entries=len(panel), entry_dates=sorted({r["quote_date"] for r in panel}),
                    requested_entry_dates=dates, selection_status_counts=bucket_status,
                    endpoint_status_counts=counts("end_status"), black_status_counts=counts("black_status"),
                    ah_status_counts=counts("ah_status"),
                    matched_comparison_available=sum(r["matched_comparison_available"].lower() == "true" for r in panel))
            audit.update(input_sha256={str(p.resolve()): sha256(p) for p in inputs},
                         aggregation="Saved independent date jobs; no fit, delta or P&L recomputation.",
                         date_job_audits=[str(p.resolve()) for p in folders],
                         output_sha256={p.relative_to(target).as_posix(): sha256(p)
                             for p in sorted(target.rglob("*")) if p.is_file() and p != target / "audit.json"})
            atomic_json(target / "audit.json", audit)
            _, files = verify_artifact(target, stage)
            failed = audit.get("failed_daily_models", 0) > 0
            if stage == "panel":
                failed = any(count for field in ("black_status_counts", "ah_status_counts")
                    for status, count in audit.get(field, {}).items() if status != "ready")
            return dict(status="completed_with_failures" if failed else "completed",
                        origin="date_aggregation", folder=str(target), files=files)
        return self.store.execute(f"{month}/{stage}", signature, action)

    def skip(self, month, stage, reason, inputs=()):
        signature = self.signature(stage, {"skipped": reason}, inputs)
        return self.store.execute(f"{month}/{stage}", signature,
            lambda _: dict(status="skipped", origin="not_executed", files={}, message=reason))

    def month(self, job):
        c, month, start, end = self.config, job["month"], job["entry_start"], job["entry_end"]
        def stop(stage):
            return STAGES.index(stage) >= STAGES.index(self.stop_after)
        def settings(expected):
            def check(folder, audit):
                for key, value in expected.items():
                    if audit.get("settings", {}).get(key) != value:
                        raise ValueError(f"Saved setting differs: {key}")
            return check

        prepared = self.stage(month, "prepare", ["--raw", *job["raw_files"], "--policy", str(self.policy),
            "--start", start, "--end", job["observation_end"]], [*[Path(p) for p in job["raw_files"]], self.policy],
            settings(dict(start=start, end=job["observation_end"], minimum_days=4, maximum_days=60,
                          calibration_half_width=0.10, entry_minimum_days=14, entry_maximum_days=45, entry_half_width=0.03)))
        if stop("prepare"):
            return {"prepare": str(prepared)}
        carry = self.stage(month, "carry", ["--quotes", str(prepared / "pilot_quotes.csv"),
            "--primary-rate-pct", str(c.carry["primary_rate_pct"]), "--rates-pct", *map(str, c.carry["rates_pct"])],
            [prepared / "pilot_quotes.csv", prepared / "audit.json"],
            settings(dict(primary_rate=c.carry["primary_rate_pct"] / 100,
                          rates=[v / 100 for v in c.carry["rates_pct"]], parity_window=0.03, minimum_pairs=6)))
        result = dict(prepare=str(prepared), carry=str(carry))
        if stop("carry"):
            return result
        available_dates = sorted({r["quote_date"] for r in rows(carry / "calibration_quotes.csv")
                                  if start <= r["quote_date"] <= end})
        policy_window = [r for r in rows(self.policy) if start <= r["quote_date"] <= end]
        coverage = [{"quote_date": r["quote_date"], "session_policy": r["session_policy"],
                     "has_calibration_quotes": r["quote_date"] in available_dates} for r in policy_window]
        write_rows(self.output / month / "date_coverage.csv", coverage)
        if not available_dates:
            result["status"] = "no_calibration_dates"
            return result
        cal_inputs = [carry / "calibration_quotes.csv", carry / "primary_carry.csv"]
        cal_args = ["--quotes", str(cal_inputs[0]), "--carry", str(cal_inputs[1])]
        for key, value in c.calibration.items():
            cal_args += ["--" + key.replace("_", "-"), str(value)]
        def check_calibration(folder, audit, dates=available_dates):
            settings({**c.calibration, "min_proxy_vol": 0.005, "max_proxy_vol": 3.0})(folder, audit)
            if sorted(audit["dates"]) != dates:
                raise ValueError("Calibration checkpoint date set differs from the configured window.")
            if audit.get("short_end_smoothing_applied") is not False or audit.get("future_observations_used") is not False:
                raise ValueError("Unexpected calibration model or information policy.")
        if "calibration" in c.checkpoints.get(month, {}):
            calibration = self.stage(month, "calibration", cal_args + ["--dates", *available_dates],
                                     cal_inputs, check_calibration)
        else:
            cal_parent = self.output / month / "calibration"
            parts = [self.stage(month, "calibration", cal_args + ["--dates", d], cal_inputs,
                lambda f, a, d=d: check_calibration(f, a, [d]), date_key=d, parent=cal_parent)
                for d in available_dates]
            calibration = self.aggregate(month, "calibration", parts, cal_inputs, available_dates)
        result["calibration"] = str(calibration)
        if stop("calibration"):
            return result
        for d in job["validation_dates"]:
            if d not in available_dates:
                raise ValueError(f"Requested validation date has no calibration quotes: {d}")
            self.stage(month, "validation", ["--calibration-run", str(calibration), "--dates", d],
                [calibration / name for name in ("audit.json", "model_manifest.csv", "quote_residuals.csv", "expiry_summary.csv")],
                lambda f, a: None, date_key=d)
        if not job["validation_dates"]:
            self.skip(month, "validation", "No additional numerical validation dates requested.",
                      [calibration / "audit.json"])
        result["validation_scope"] = job["validation_dates"]
        if stop("validation"):
            return result
        # Selection uses only entry-date information. Exit matching happens later.
        import pandas as pd
        from pilot_hedge_panel import PilotContractSelector
        entries, _ = PilotContractSelector().select(pd.read_csv(prepared / "pilot_quotes.csv"),
                                                    pd.read_csv(prepared / "session_timeline.csv"), available_dates)
        entry_dates = sorted(entries.quote_date.unique().tolist()) if len(entries) else []
        if not entry_dates:
            result["status"] = "no_entry_candidates"
            return result
        panel_inputs = [prepared / name for name in ("pilot_quotes.csv", "zero_bid_bounds.csv", "session_timeline.csv")]
        panel_inputs += [carry / "primary_carry.csv", calibration / "audit.json", calibration / "model_manifest.csv"]
        panel_args = ["--calibration-run", str(calibration), "--quotes", str(panel_inputs[0]),
            "--bounds", str(panel_inputs[1]), "--timeline", str(panel_inputs[2]), "--carry", str(panel_inputs[3])]
        for key, value in c.panel.items():
            panel_args += ["--" + key.replace("_", "-"), str(value)]
        def check_panel(folder, audit, dates=entry_dates):
            settings({**c.panel, "radius": 0.0005, "width": 0.75,
                      "target_days": [21, 35, 45], "target_spot_y": [-0.02, 0.0, 0.02],
                      "kinds": ["call", "put"]})(folder, audit)
            if sorted(audit["entry_dates"]) != dates:
                raise ValueError("Panel checkpoint entry dates differ from current-date selection.")
            expected_entries = set(entries.loc[entries.quote_date.isin(dates), "entry_id"])
            actual = rows(folder / "entries.csv")
            if len(actual) != audit["entries"] or {r["entry_id"] for r in actual} != expected_entries:
                raise ValueError("Panel entries differ from current-date contract selection.")
            for flag in ("marks_replaced_by_model_prices", "entries_filtered_by_future_availability", "entry_calculations_use_future_quotes"):
                if audit.get(flag) is not False:
                    raise ValueError(f"Unexpected entry/mark policy: {flag}")
        if "panel" in c.checkpoints.get(month, {}):
            panel = self.stage(month, "panel", panel_args + ["--dates", *entry_dates], panel_inputs, check_panel)
        else:
            parts = [self.stage(month, "panel", panel_args + ["--dates", d], panel_inputs,
                lambda f, a, d=d: check_panel(f, a, [d]), date_key=d,
                parent=self.output / month / "panel") for d in entry_dates]
            panel = self.aggregate(month, "panel", parts, panel_inputs, entry_dates)
        result["panel"] = str(panel)
        if stop("panel"):
            return result
        panel_rows = rows(panel / "delta_panel.csv")
        if not any(r["matched_comparison_available"].lower() == "true" for r in panel_rows):
            for stage in ("comparison", "attribution", "notebook"):
                self.skip(month, stage, "No matched entries with both deltas ready.", [panel / "audit.json"])
            result["status"] = "no_matched_ready_comparisons"
            return result
        compare_args = ["--panel-run", str(panel)]
        for key, value in c.funding.items():
            compare_args += ["--" + key.replace("_", "-"), str(value)]
        comparison = self.stage(month, "comparison", compare_args,
            [panel / "audit.json", panel / "delta_panel.csv"], settings(c.funding))
        result["comparison"] = str(comparison)
        if stop("comparison"):
            return result
        attribution = self.stage(month, "attribution", ["--comparison-run", str(comparison)],
            [comparison / n for n in ("audit.json", "strategy_results.csv", "paired_results.csv", "coverage.csv")],
            settings(c.funding))
        result["attribution"] = str(attribution)
        if stop("attribution"):
            return result
        if c.build_notebooks:
            # Reuse the existing, reviewed notebook template with a period label.
            import importlib.util
            spec = importlib.util.spec_from_file_location("pilot_notebook_template", self.repo / "scripts" / SCRIPTS["notebook"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            paths = {name: str(Path(result[name])) for name in ("calibration", "panel", "comparison", "attribution")}
            path = self.output / month / "research_results.ipynb"
            def notebook_action(attempt):
                notebook = module.build_notebook(paths)
                for cell in notebook["cells"]:
                    cell["source"] = cell["source"].replace("September", month)
                atomic_json(path, notebook)
                return dict(status="completed", origin="saved_result_notebook", folder=str(path.parent),
                            files={str(path): sha256(path)})
            inputs = [Path(result[name]) / "audit.json" for name in paths]
            self.store.execute(f"{month}/notebook", self.signature("notebook", paths, inputs), notebook_action)
            result["notebook"] = str(path)
        else:
            self.skip(month, "notebook", "Notebook generation not requested.")
        result["status"] = "completed"
        return result

    def summary(self, months):
        records = []
        for month, result in months.items():
            row = {"month": month, "status": result.get("status", "paused"),
                   "calibration_dates": 0, "failed_models": 0, "entries": 0,
                   "matched_comparisons": 0, "validation_dates_requested":
                   ";".join(result.get("validation_scope", [])),
                   "all_entry_greeks_independently_validated": False}
            if "calibration" in result:
                manifest = rows(Path(result["calibration"]) / "model_manifest.csv")
                row["calibration_dates"] = len({r["quote_date"] for r in manifest})
                row["failed_models"] = sum(r["status"] == "failed" for r in manifest)
            if "panel" in result:
                panel = rows(Path(result["panel"]) / "delta_panel.csv")
                row["entries"] = len(panel)
                row["matched_comparisons"] = sum(r["matched_comparison_available"].lower() == "true" for r in panel)
            records.append(row)
        write_rows(self.output / "month_summary.csv", records)

    def run(self):
        jobs = self.plan()  # Check raw-file/calendar coverage before any computation.
        with self.store.locked():
            self.store.data["status"] = "running"
            self.store.save()
            try:
                months = {}
                for job in jobs:
                    months[job["month"]] = self.month(job)
                    atomic_json(self.output / "run_index.json", dict(configuration=self.configuration,
                        months=months, scope="Provisional research; one-session synthetic-index hedges.",
                        all_dates_numerically_validated=False, out_of_sample_claim=False,
                        contract_identity_verified=False, funding_curve_verified=False))
                    self.summary(months)
                bad = any(r["status"] == "completed_with_failures" for r in self.store.data["stages"].values())
                self.store.data["status"] = ("paused" if self.stop_after != "notebook" else
                                             "completed_with_failures" if bad else "completed")
            except BaseException:
                self.store.data["status"] = "failed"
                self.store.save()
                raise
            self.store.save()
        return months
