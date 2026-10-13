"""Frozen, resumable orchestration of the historical study."""

from artifacts import CsvTables
from artifacts import ArtifactCache, atomic_json, sha256, signature, study_lock
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from market_data import QuoteHistory, frame
from calibration import HedgeDecisionEngine
from selection import ContractSelector, HedgeSettings
from paths import HistoricalSettings, FixedContractPaths, HedgePathEvaluator
from market_data import KEY
from paths import METHODS
from market_data import empty_quotes
from inference import GainAccumulator


def requested_quotes(quotes, keys):
    if quotes.empty or not keys:
        return quotes.iloc[:0].copy()
    mask = pd.MultiIndex.from_frame(quotes[KEY]).isin(list(keys))
    return quotes.loc[mask]


class HistoricalStudy:

    def __init__(self, config_file, repository=None):
        self.repository = Path(repository or Path.cwd()).resolve()
        self.config_file = Path(config_file).resolve()
        config = json.loads(self.config_file.read_text())
        allowed = {"settings", "raw_directory", "session_policy", "output"}
        if set(config) != allowed:
            raise ValueError(
                f"Expected exactly these configuration keys: {sorted(allowed)}"
            )
        options = dict(config["settings"])
        for key in ("schedules", "block_lengths"):
            if key in options:
                options[key] = tuple(
                    (tuple(x) if isinstance(x, list) else x for x in options[key])
                )
        self.settings = HistoricalSettings(**options)
        self.config = config
        self.raw = self.path(config["raw_directory"])
        self.original_policy = self.path(config["session_policy"])
        self.output = self.path(config["output"])
        if (
            self.output == self.repository
            or self.output.is_relative_to(self.raw)
            or self.raw.is_relative_to(self.output)
        ):
            raise ValueError(
                "Keep study outputs separate from the repository root and raw inputs."
            )
        if self.original_policy.is_relative_to(self.output):
            raise ValueError("The input policy must be separate from study outputs.")
        self.policy_file = self.output / "calendar_support.csv"
        self.freeze_file = self.output / "freeze.json"
        self.cache = ArtifactCache(self.output / "cache")
        self.history = QuoteHistory(
            self.raw, self.policy_file, self.settings, self.cache
        )
        self.engine = HedgeDecisionEngine(self.settings)
        self.index = {}
        self.evaluator = HedgePathEvaluator(self.settings)

    def path(self, name):
        p = Path(name)
        return (p if p.is_absolute() else self.repository / p).resolve()

    @property
    def months(self):
        return [
            str(x)
            for x in pd.period_range(self.settings.start, self.settings.end, freq="M")
        ]

    def prepare_calendar(self):
        """Preserve audited flags; add reference-calendar-only maturity support."""
        if self.freeze_file.exists():
            raise ValueError("Calendar is already frozen; use its saved copy.")
        import exchange_calendars as xc

        original = pd.read_csv(self.original_policy)
        original = ContractSelector.checked_timeline(original)
        if "early_cash_close" not in original:
            raise ValueError("The audited policy must contain early_cash_close.")
        before = pd.Timestamp(self.settings.start) - pd.Timedelta(days=31)
        after = pd.Timestamp(self.settings.end) + pd.Timedelta(days=120)
        calendar = xc.get_calendar("XNYS", start=before, end=after)
        dates = pd.date_range(before, after).strftime("%Y-%m-%d")
        sessions = set(calendar.sessions.strftime("%Y-%m-%d"))
        early = set(calendar.early_closes.strftime("%Y-%m-%d"))
        existing = original.set_index("quote_date")
        records = []
        for date in dates:
            if date in existing.index:
                records.append(
                    existing.loc[date].to_dict()
                    | {"quote_date": date, "calendar_support_origin": "audited_policy"}
                )
            else:
                reference = date in sessions
                in_history = self.settings.start <= date <= self.settings.end
                state = (
                    "missing_observation"
                    if reference and in_history
                    else "calendar_support_only" if reference else "nonreference_date"
                )
                records.append(
                    dict(
                        quote_date=date,
                        reference_session=reference,
                        early_cash_close=date in early,
                        session_policy=state,
                        calendar_support_origin="calendar_only_no_observation",
                    )
                )
        policy = pd.DataFrame(records)
        self.output.mkdir(parents=True, exist_ok=True)
        policy.to_csv(self.policy_file, index=False)
        atomic_json(
            self.output / "calendar_support_audit.json",
            dict(
                original_policy_sha256=sha256(self.original_policy),
                calendar_support_sha256=sha256(self.policy_file),
                exchange_calendars_version=xc.__version__,
                audited_rows_preserved=len(
                    original.loc[
                        original.quote_date.between(str(dates[0]), str(dates[-1]))
                    ]
                ),
                added_calendar_only_rows=int(
                    policy.calendar_support_origin.eq(
                        "calendar_only_no_observation"
                    ).sum()
                ),
                observed_data_inferred=False,
                snapshot_times_inferred=False,
                historical_exchange_calendar_certified=False,
            ),
        )
        return self.policy_file

    def plan(self):
        s = self.settings
        p = ContractSelector.checked_timeline(pd.read_csv(self.original_policy))
        reference = p.loc[p.reference_session & p.quote_date.between(s.start, s.end)]
        return dict(
            start=s.start,
            end=s.end,
            raw_files=len(self.months),
            missing_raw_files=[
                str(self.raw / f"spx_eod_{m.replace('-', '')}.txt")
                for m in self.months
                if not (self.raw / f"spx_eod_{m.replace('-', '')}.txt").exists()
            ],
            audited_reference_dates=len(reference),
            schedules=s.schedules,
            methods=list(METHODS),
            assumed_pricing_rate=0.05,
            funding=s.funding,
            calendar_support_ready=self.policy_file.exists(),
            output=str(self.output),
            evaluation_scope="entire declared dataset; retrospective rolling evaluation",
            genuine_untouched_holdout_declared=False,
            previously_inspected_development=[s.development_start, s.development_end],
            source_and_inputs_frozen=self.freeze_file.exists(),
            per_date_surfaces=True,
            pooled_history_surface=False,
            bootstrap_mean_blocks=s.block_lengths,
            measured_runtime_only=True,
        )

    def frozen_identity(self):
        source_folder = Path(__file__).resolve().parent
        dependencies = [p.stem for p in sorted(source_folder.glob("*.py"))]
        sources = {
            str(source_folder / f"{name}.py"): sha256(source_folder / f"{name}.py")
            for name in dependencies
        }
        script = self.repository / "scripts/run_history.py"
        sources[str(script)] = sha256(script)
        paths = [
            self.config_file,
            self.original_policy,
            self.policy_file,
            self.output / "calendar_support_audit.json",
        ]
        inputs = {str(p): sha256(p) for p in paths}
        for month in self.months:
            path = self.raw / f"spx_eod_{month.replace('-', '')}.txt"
            inputs[str(path)] = sha256(path) if path.exists() else None
        return dict(
            config=self.config,
            input_sha256=inputs,
            source_sha256=sources,
            versions=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                pandas=pd.__version__,
                scipy=scipy.__version__,
            ),
            design=dict(
                scope=f"entire declared range {self.settings.start} to {self.settings.end}",
                genuine_untouched_holdout=False,
                existing_development=[
                    self.settings.development_start,
                    self.settings.development_end,
                ],
                empirical_training="rolling completed one-session labels; endpoint STRICTLY before current timestamp",
                future_availability_used_to_select_entries=False,
                no_fallback_hedge=True,
                dividends=0.0,
                synthetic_fractional_index=True,
                funding_curve_verified=False,
                contract_identity_verified=False,
                snapshot_provenance_verified=False,
                settlement_payoffs_used=False,
                every_entry_greek_independently_validated=False,
            ),
        )

    def freeze(self):
        if not self.policy_file.exists():
            raise ValueError("Run --prepare-calendar before freezing or executing.")
        calendar_audit = json.loads(
            (self.output / "calendar_support_audit.json").read_text()
        )
        if calendar_audit["original_policy_sha256"] != sha256(
            self.original_policy
        ) or calendar_audit["calendar_support_sha256"] != sha256(self.policy_file):
            raise ValueError(
                "Calendar support disagrees with its audited-policy input or saved hash."
            )
        identity = self.frozen_identity()
        if self.freeze_file.exists():
            prior = json.loads(self.freeze_file.read_text())
            if prior["identity"] != identity:
                raise ValueError(
                    "Frozen source, inputs, environment or settings changed. Use a new output directory."
                )
            return prior
        result = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            fingerprint=signature(identity),
            identity=identity,
        )
        atomic_json(self.freeze_file, result)
        return result

    def dated_decisions(self, date, requested, training, freeze, progress):
        data = self.history.month_data(date[:7], freeze)
        now_quotes = self.history.quotes_on(date, freeze)
        requested_keys = sorted([list(key) for key in requested], key=str)
        identity = dict(
            freeze=freeze["fingerprint"],
            date=date,
            requested=requested_keys,
            training_ids=sorted(training.entry_id.tolist()) if len(training) else [],
        )
        folder, metadata = self.cache.get(
            f"dates/{date}/decisions",
            identity,
            self.save_decisions,
            date=date,
            now_quotes=now_quotes,
            requested=requested,
            data=data,
            training=training,
            progress=progress,
        )
        return (frame(folder / "decisions.csv"), folder, metadata)

    def training_labels(self, date, entries, decisions, paths, freeze):
        rows = []
        i = paths.positions[date]
        end = paths.sessions[i + 1] if i + 1 < len(paths.sessions) else None
        keys = set(map(tuple, entries[KEY].to_numpy()))
        lookup = (
            {
                tuple((row[k] for k in KEY)): row
                for row in requested_quotes(
                    self.history.quotes_on(end, freeze), keys
                ).to_dict("records")
            }
            if end
            else {}
        )
        now = {
            tuple((row[k] for k in KEY)): row for row in decisions.to_dict("records")
        }
        for entry in entries.to_dict("records"):
            row = dict(now.get(tuple((entry[k] for k in KEY)), entry))
            row.update(
                entry_id=entry["entry_id"],
                end_status="unavailable",
                end_timestamp=pd.NaT,
                end_mid=np.nan,
                end_spot=np.nan,
                black_status=row.get("black_status", "unavailable"),
            )
            other = lookup.get(tuple((entry[k] for k in KEY)))
            if (
                other is not None
                and paths.policies[end] == "provisional_standard_session"
                and (entry["expire_date"] > end)
            ):
                if pd.Timestamp(other["assumed_fixing_utc"]) != pd.Timestamp(
                    entry["assumed_fixing_utc"]
                ):
                    raise ValueError("Training fixed-contract assumption changed.")
                row.update(
                    end_status="matched",
                    end_timestamp=other["quote_timestamp_utc"],
                    end_mid=other["mid"],
                    end_spot=other["underlying_last"],
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def run(self, until_date=None, progress=print):
        self.index = {}
        self.cache.timing = []
        started = datetime.now(timezone.utc)
        clock = time.perf_counter()
        until = until_date or self.settings.end
        stamp = pd.Timestamp(until)
        if (
            pd.isna(stamp)
            or stamp.tzinfo is not None
            or stamp != stamp.normalize()
            or (until != stamp.strftime("%Y-%m-%d"))
            or (not self.settings.start <= until <= self.settings.end)
        ):
            raise ValueError("--until-date must lie in the frozen study range.")
        invocation = started.strftime("run_%Y%m%dT%H%M%S_%fZ")
        runtime = self.output / "runtime" / invocation
        runtime.mkdir(parents=True, exist_ok=True)
        timing = dict(
            started_utc=started.isoformat(), until_date=until, status="running"
        )
        with study_lock(self.output):
            try:
                freeze = self.freeze()
                policy = ContractSelector.checked_timeline(
                    pd.read_csv(self.policy_file)
                )
                paths = FixedContractPaths(policy, self.settings)
                dates = [
                    date
                    for date in paths.sessions
                    if self.settings.start <= date <= until
                ]
                selector = ContractSelector(HedgeSettings(**self.settings.panel))
                training = empty_quotes().assign(
                    entry_id=pd.Series(dtype=str),
                    black_status=pd.Series(dtype=str),
                    end_status=pd.Series(dtype=str),
                    end_timestamp=pd.Series(dtype="datetime64[ns, UTC]"),
                    end_mid=pd.Series(dtype=float),
                    end_spot=pd.Series(dtype=float),
                )
                active = []
                entries_files = []
                for date in dates:
                    progress(
                        f"{date}: current-date selection and fixed-contract rebalance decisions..."
                    )
                    data = self.history.month_data(date[:7], freeze)
                    q = data["quotes"].loc[data["quotes"].quote_date.eq(date)]
                    new, buckets = selector.select(q, policy, [date])
                    if new.empty:
                        new = empty_quotes().assign(
                            entry_id=pd.Series(dtype=str),
                            contract_id=pd.Series(dtype=str),
                            calendar_days=pd.Series(dtype=float),
                            entry_spot_y=pd.Series(dtype=float),
                        )
                    requested = paths.requests(date, active, new)
                    decisions, folder, metadata = self.dated_decisions(
                        date, requested, training, freeze, progress
                    )
                    selection_folder, selection_meta = self.cache.get(
                        f"dates/{date}/selection",
                        dict(freeze=freeze["fingerprint"], date=date),
                        self.save_selection,
                        new=new,
                        buckets=buckets,
                        session_policy=paths.policies[date],
                    )
                    entries_files.append(selection_folder / "entries.csv")
                    labels = self.training_labels(date, new, decisions, paths, freeze)
                    if len(labels):
                        training = (
                            pd.concat([training, labels], ignore_index=True)
                            if len(training)
                            else labels.copy()
                        )
                    if len(training):
                        keep = sorted(training.quote_date.unique())[
                            -(self.settings.empirical["window_dates"] + 2) :
                        ]
                        training = training.loc[
                            training.quote_date.isin(keep)
                        ].reset_index(drop=True)
                    active.extend(new.to_dict("records"))
                    active = [
                        e
                        for e in active
                        if paths.positions[date] - paths.positions[e["quote_date"]]
                        < self.settings.maximum_holding
                    ]
                    self.index[date] = dict(
                        date=date,
                        **selection_meta,
                        **metadata,
                        prepare_status=data["prepare_status"],
                        carry_status=data["carry_status"],
                        decision_folder=str(folder),
                        entries_file=str(selection_folder / "entries.csv"),
                    )
                    self.write_index(until, freeze, completed=False)
                entries = (
                    pd.concat([frame(p) for p in entries_files], ignore_index=True)
                    if entries_files
                    else pd.DataFrame()
                )
                summary = self.evaluate(entries, paths, dates, freeze, progress)
                completed = until == self.settings.end
                self.write_index(until, freeze, completed=completed)
                timing.update(
                    status="completed" if completed else "prefix_completed",
                    summary=summary,
                )
            except BaseException as error:
                timing.update(
                    status=(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    message=f"{type(error).__name__}: {error}",
                )
                raise
            finally:
                timing.update(
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_wall_seconds=time.perf_counter() - clock,
                )
                atomic_json(runtime / "timing.json", timing)
                pd.DataFrame(self.cache.timing).to_csv(
                    runtime / "stage_timing.csv", index=False
                )
        return timing

    def write_index(self, until, freeze, completed):
        pd.DataFrame(self.index.values()).to_csv(
            self.output / "date_coverage.csv", index=False
        )
        atomic_json(
            self.output / "run_index.json",
            dict(
                status="execution_complete" if completed else "partial",
                frozen_fingerprint=freeze["fingerprint"],
                start=self.settings.start,
                end=self.settings.end,
                processed_through=max(self.index, default=None),
                dates=self.index,
                every_entry_greek_independently_validated=False,
                heldout_claim=False,
                evaluation="entire requested history, rolling past-only estimation",
                aggregate_folder=str(self.output / "summary"),
            ),
        )

    def evaluate(self, entries, paths, dates, freeze, progress):
        completed = set(dates)
        output = self.output / "summary"
        output.mkdir(parents=True, exist_ok=True)
        accum = GainAccumulator(dates, self.settings)
        intersection = GainAccumulator(dates, self.settings, common_strategies=True)
        counts = {}
        summary = dict(
            selected_entries=len(entries),
            processed_reference_dates=len(dates),
            trial_rows=0,
            coverage_rows=0,
            compared_paths=0,
            pending_paths=0,
            max_reconciliation=None,
        )
        writer = CsvTables(
            output,
            (
                "strategy_results.csv",
                "coverage.csv",
                "paired_results.csv",
                "representative_ledgers.csv",
            ),
        )
        ledger_count = 0
        grouped = entries.groupby("quote_date", sort=True) if len(entries) else []
        for date, group in grouped:
            selected_dates = paths.dates(date, self.settings.maximum_holding)
            needed = set(map(tuple, group[KEY].to_numpy()))
            lookup, decisions = ({}, {})
            for day in selected_dates:
                for row in requested_quotes(
                    self.history.quotes_on(day, freeze), needed
                ).to_dict("records"):
                    lookup[day, *[row[k] for k in KEY]] = row
                if day in self.index:
                    for row in requested_quotes(
                        frame(
                            Path(self.index[day]["decision_folder"]) / "decisions.csv"
                        ),
                        needed,
                    ).to_dict("records"):
                        decisions[day, *[row[k] for k in KEY]] = row
            identity = dict(
                freeze=freeze["fingerprint"],
                date=date,
                completed_path_dates=[d for d in selected_dates if d in completed],
            )
            version = signature(identity)[:16]
            folder, _ = self.cache.get(
                f"outcomes/{date}/{version}",
                identity,
                self.evaluate_date,
                group=group,
                paths=paths,
                lookup=lookup,
                decisions=decisions,
                completed=completed,
                limit=(
                    self.settings.representative_ledgers
                    if date == entries.quote_date.min()
                    else 0
                ),
            )
            self.index[date]["outcome_folder"] = str(folder)
            trials = frame(folder / "trials.csv")
            coverage = frame(folder / "coverage.csv")
            writer.append("strategy_results.csv", trials)
            writer.append("coverage.csv", coverage)
            pairs = accum.add(trials)
            intersection.add(trials)
            writer.append("paired_results.csv", pairs)
            for key, n in (
                coverage.groupby(
                    [
                        "holding_sessions",
                        "rebalance_sessions",
                        "scenario",
                        "strategy",
                        "status",
                    ]
                )
                .size()
                .items()
            ):
                counts[key] = counts.get(key, 0) + int(n)
            summary["trial_rows"] += len(trials)
            summary["coverage_rows"] += len(coverage)
            summary["compared_paths"] += int(coverage.status.eq("compared").sum())
            summary["pending_paths"] += int(
                coverage.status.eq("pending_run_prefix").sum()
            )
            if len(trials):
                error = float(trials.max_reconciliation.max())
                summary["max_reconciliation"] = max(
                    error, summary["max_reconciliation"] or 0
                )
            if (
                ledger_count < self.settings.representative_ledgers
                and (folder / "representative_ledgers.csv").exists()
            ):
                detail = frame(folder / "representative_ledgers.csv")
                identity_columns = [
                    "entry_id",
                    "holding_sessions",
                    "rebalance_sessions",
                    "scenario",
                    "strategy",
                ]
                selected = (
                    detail[identity_columns]
                    .drop_duplicates()
                    .head(self.settings.representative_ledgers - ledger_count)
                )
                writer.append(
                    "representative_ledgers.csv",
                    detail.merge(selected, on=identity_columns, validate="many_to_one"),
                )
                ledger_count += len(selected)
            progress(f"{date}: observed-mark path outcomes verified.")
        writer.finish()
        gains, common = (accum.report(), intersection.report())
        if gains.empty:
            gains = pd.DataFrame(
                columns=[
                    "holding_sessions",
                    "rebalance_sessions",
                    "scenario",
                    "strategy",
                    "dimension",
                    "group",
                    "comparisons",
                    "gain",
                    "gain_status",
                    "ci_status",
                    "mean_block_sessions",
                ]
            )
        if common.empty:
            common = pd.DataFrame(columns=gains.columns)
        gains.to_csv(output / "gain_summary.csv", index=False)
        common.to_csv(output / "common_strategy_gain.csv", index=False)
        stats = accum.daily_statistics()
        if stats.empty:
            stats = pd.DataFrame(
                columns=[
                    "quote_date",
                    "holding_sessions",
                    "rebalance_sessions",
                    "scenario",
                    "strategy",
                ]
            )
        stats.to_csv(output / "daily_gain_statistics.csv", index=False)
        coverage_names = [
            "holding_sessions",
            "rebalance_sessions",
            "scenario",
            "strategy",
            "status",
        ]
        pd.DataFrame(
            [
                dict(zip(coverage_names, k), entries=v)
                for k, v in sorted(counts.items())
            ],
            columns=coverage_names + ["entries"],
        ).to_csv(output / "coverage_summary.csv", index=False)
        result = dict(
            **summary,
            every_entry_greek_independently_validated=False,
            heldout_claim=False,
            input_sha256={str(self.freeze_file): sha256(self.freeze_file)},
            output_sha256={p.name: sha256(p) for p in sorted(output.glob("*.csv"))},
        )
        atomic_json(output / "audit.json", result)
        return result

    def save_decisions(
        self, folder, date, now_quotes, requested, data, training, progress
    ):
        result = self.engine.calculate(
            date,
            now_quotes,
            requested,
            data["carry"],
            data["calibration"],
            training,
            progress,
        )
        result.save(folder)
        return result.summary

    def save_selection(self, target, new, buckets, session_policy):
        new.to_csv(target / "entries.csv", index=False)
        buckets.to_csv(target / "selection.csv", index=False)
        return dict(selected_entries=len(new), session_policy=session_policy)

    def evaluate_date(self, folder, group, paths, lookup, decisions, completed, limit):
        t, c, l = ([], [], [])
        for entry in group.to_dict("records"):
            for hold, reb in self.settings.schedules:
                meta, common, failed, observed = paths.path(
                    entry, hold, reb, lookup, decisions, completed
                )
                result, flags, detail = self.evaluator.run(
                    meta,
                    common,
                    failed,
                    observed,
                    representative_limit=max(0, limit - len(l)),
                )
                t.extend(result)
                c.extend(flags)
                if len(l) < limit:
                    l.extend(detail[: limit - len(l)])
        trial_columns = [
            "entry_id",
            "quote_date",
            "strategy",
            "scenario",
            "net_pnl",
            "raw_mark_error",
        ]
        (pd.DataFrame(t) if t else pd.DataFrame(columns=trial_columns)).to_csv(
            folder / "trials.csv", index=False
        )
        pd.DataFrame(c).to_csv(folder / "coverage.csv", index=False)
        if l:
            pd.concat(l, ignore_index=True).to_csv(
                folder / "representative_ledgers.csv", index=False
            )
        return dict(compared=len(t), coverage_records=len(c))
