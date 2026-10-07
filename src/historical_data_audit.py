"""Historical coverage audit; no contract selection, curve fitting or backtest."""

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class HistoricalAuditSettings:
    start: str = "2013-01-01"
    end: str = "2023-12-31"
    minimum_days: int = 1
    maximum_days: int = 60
    log_spot_half_width: float = 0.10
    timestamp_tolerance_seconds: float = 1.0
    spot_tolerance_points: float = 0.01
    readtime_timezone: str = "America/New_York"
    root_column: str | None = None

    def __post_init__(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("Start must precede end.")
        if not 1 <= self.minimum_days <= self.maximum_days:
            raise ValueError("Require 1 <= minimum_days <= maximum_days.")
        for value in (
            self.log_spot_half_width,
            self.timestamp_tolerance_seconds,
            self.spot_tolerance_points,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "Windows and tolerances must be finite and nonnegative."
                )


class HistoricalDataAudit:
    REQUIRED = (
        "quote_unixtime",
        "quote_readtime",
        "quote_date",
        "expire_date",
        "underlying_last",
        "strike",
        "c_bid",
        "c_ask",
        "p_bid",
        "p_ask",
    )
    KEY = ["root", "expire_date", "strike", "kind"]

    FLAGS = (
        "bad_quote_date",
        "file_month_mismatch",
        "bad_timestamp",
        "timestamp_mismatch",
        "date_mismatch",
        "bad_spot",
        "bad_contract",
        "vendor_expiry_date_mismatch",
        "c_bad_quote",
        "p_bad_quote",
        "c_locked_quote",
        "p_locked_quote",
        "c_zero_bid",
        "p_zero_bid",
    )

    def __init__(self, settings=None):
        self.settings = settings or HistoricalAuditSettings()

        import exchange_calendars as xc

        self.calendar_version = xc.__version__
        calendar = xc.get_calendar(
            "XNYS",
            start=pd.Timestamp(self.settings.start) - pd.Timedelta(days=7),
            end=pd.Timestamp(self.settings.end) + pd.Timedelta(days=7),
        )
        self.schedule = calendar.schedule.loc[
            self.settings.start:self.settings.end,
            ["open", "close"],
        ].copy()
        self.schedule.index.name = "quote_date"
        self.schedule["cash_close_ny"] = (
            self.schedule["close"].dt.tz_convert("America/New_York")
        )
        self.schedule["early_cash_close"] = (
            self.schedule["cash_close_ny"].dt.hour.lt(16)
        )
        self._previous = None
        self._adjacent = []

    @staticmethod
    def header(value):
        return str(value).strip().strip("[]").strip().lower()

    def _read(self, path, month, info):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            original = next(csv.reader(stream))

        names = [self.header(c) for c in original]
        info["columns"] = json.dumps(names)

        if len(names) != len(set(names)):
            raise ValueError("Repeated normalized headers.")

        missing = set(self.REQUIRED) - set(names)
        root_name = (
            self.header(self.settings.root_column)
            if self.settings.root_column
            else None
        )
        if root_name and root_name not in names:
            missing.add(root_name)

        info["missing_columns"] = json.dumps(sorted(missing))
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        wanted = set(self.REQUIRED) | {"expire_unix"}
        if root_name:
            wanted.add(root_name)

        selected = [
            column
            for column, normalized in zip(original, names)
            if normalized in wanted
        ]
        f = pd.read_csv(
            path,
            usecols=selected,
            dtype="string",
            encoding="utf-8-sig",
        )
        f.columns = [self.header(c) for c in f]
        info["raw_rows"] = len(f)

        for c in f:
            f[c] = f[c].str.strip()

        for c in (
            "quote_unixtime",
            "underlying_last",
            "strike",
            "c_bid",
            "c_ask",
            "p_bid",
            "p_ask",
        ):
            f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)

        for c in ("quote_date", "expire_date"):
            f[c] = pd.to_datetime(
                f[c], format="%Y-%m-%d", errors="coerce"
            )

        info["observation_dates"] = int(f["quote_date"].nunique())
        for name, value in (
            ("first_date", f["quote_date"].min()),
            ("last_date", f["quote_date"].max()),
        ):
            info[name] = (
                value.strftime("%Y-%m-%d") if pd.notna(value) else ""
            )

        utc = pd.to_datetime(
            f["quote_unixtime"], unit="s", utc=True, errors="coerce"
        )
        local = utc.dt.tz_convert("America/New_York")

        read = pd.to_datetime(
            f["quote_readtime"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        read = read.fillna(
            pd.to_datetime(
                f["quote_readtime"],
                format="%Y-%m-%d %H:%M",
                errors="coerce",
            )
        )
        read = read.dt.tz_localize(
            self.settings.readtime_timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )

        f["timestamp_utc"] = utc
        f["clock_ny"] = local.dt.strftime("%H:%M:%S")
        f["bad_timestamp"] = utc.isna() | read.isna()
        f["timestamp_mismatch"] = (
            (utc - read)
            .dt.total_seconds()
            .abs()
            .gt(self.settings.timestamp_tolerance_seconds)
        )
        f["date_mismatch"] = (
            local.dt.tz_localize(None)
            .dt.normalize()
            .ne(f["quote_date"])
        )
        f["bad_quote_date"] = f["quote_date"].isna()
        f["file_month_mismatch"] = (
            f["quote_date"].notna()
            & f["quote_date"].dt.to_period("M").ne(month)
        )
        f["bad_spot"] = (
            ~np.isfinite(f["underlying_last"])
            | f["underlying_last"].le(0)
        )
        f["bad_contract"] = (
            f["expire_date"].isna()
            | ~np.isfinite(f["strike"])
            | f["strike"].le(0)
            | f["expire_date"].lt(f["quote_date"])
        )
        f["days"] = (f["expire_date"] - f["quote_date"]).dt.days
        f["y"] = np.log(
            f["strike"].where(~f["bad_contract"])
            / f["underlying_last"].where(~f["bad_spot"])
        )
        f["in_scope"] = (
            f["days"].between(
                self.settings.minimum_days,
                self.settings.maximum_days,
            )
            & f["y"].abs().le(self.settings.log_spot_half_width)
        )
        f["root"] = (
            f[root_name]
            .str.upper()
            .replace("", pd.NA)
            .fillna("UNKNOWN")
            if root_name
            else "UNKNOWN"
        )

        vendor = pd.to_datetime(
            pd.to_numeric(
                f.get(
                    "expire_unix",
                    pd.Series(np.nan, index=f.index),
                ),
                errors="coerce",
            ),
            unit="s",
            utc=True,
            errors="coerce",
        )
        vendor_date = (
            vendor.dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        f["vendor_expiry_date_mismatch"] = (
            vendor.notna() & vendor_date.ne(f["expire_date"])
        )
        f["vendor_expiry_timestamp_utc"] = vendor

        for kind in ("c", "p"):
            bid, ask = f[f"{kind}_bid"], f[f"{kind}_ask"]
            valid = (
                np.isfinite(bid)
                & np.isfinite(ask)
                & bid.ge(0)
                & ask.ge(bid)
            )
            f[f"{kind}_bad_quote"] = ~valid
            f[f"{kind}_locked_quote"] = valid & bid.eq(ask)
            f[f"{kind}_zero_bid"] = valid & bid.eq(0)

        return f

    def _day(self, date, g, file_name):
        on_calendar = date in self.schedule.index
        row = {
            "quote_date": date,
            "file": file_name,
            "raw_rows": len(g),
            "reference_session": on_calendar,
            "observed_timestamps": int(g["timestamp_utc"].nunique()),
            "clocks_ny": ";".join(
                sorted(g["clock_ny"].dropna().unique())
            ),
            "minimum_spot": g["underlying_last"].min(),
            "maximum_spot": g["underlying_last"].max(),
            "scope_rows": int(g["in_scope"].sum()),
            "root_unknown_rows": int(g["root"].eq("UNKNOWN").sum()),
        }
        for flag in self.FLAGS:
            row[flag] = int(g[flag].sum())

        if on_calendar:
            close = self.schedule.loc[date, "close"]
            delta = (g["timestamp_utc"] - close).dt.total_seconds()
            row["early_cash_close"] = bool(
                self.schedule.loc[date, "early_cash_close"]
            )
            row["cash_close_ny"] = (
                self.schedule.loc[date, "cash_close_ny"].isoformat()
            )
            row["at_cash_close_rows"] = int(
                delta.abs()
                .le(self.settings.timestamp_tolerance_seconds)
                .sum()
            )
            row["after_cash_close_rows"] = int(
                delta.gt(
                    self.settings.timestamp_tolerance_seconds
                ).sum()
            )

        spot = g.loc[~g["bad_spot"], "underlying_last"]
        span = spot.max() - spot.min()
        single = row["observed_timestamps"] == 1
        consistent = (
            len(spot) > 0
            and span <= self.settings.spot_tolerance_points
        )
        row["snapshot_status"] = (
            "multiple_or_missing_timestamps"
            if not single
            else "inconsistent_spot"
            if not consistent
            else "single_timestamp"
        )

        g = g.copy()
        duplicate = g.duplicated(
            ["root", "expire_date", "strike"], keep=False
        )
        row["duplicate_key_rows"] = int(duplicate.sum())

        bad = g[
            [
                "bad_timestamp",
                "timestamp_mismatch",
                "date_mismatch",
                "bad_spot",
                "bad_contract",
            ]
        ].any(axis=1)
        base = (
            ~bad
            & ~duplicate
            & single
            & consistent
            & g["days"].gt(0)
        )
        for kind in ("c", "p"):
            g[f"{kind}_usable"] = (
                base
                & ~g[f"{kind}_bad_quote"]
                & ~g[f"{kind}_locked_quote"]
            )
            row[f"usable_{kind}_marks"] = int(
                g[f"{kind}_usable"].sum()
            )

        coverage = (
            g.groupby(["root", "expire_date"], dropna=False)
            .agg(
                rows=("strike", "size"),
                days=("days", "first"),
                minimum_y=("y", "min"),
                maximum_y=("y", "max"),
                scope_rows=("in_scope", "sum"),
                usable_calls=("c_usable", "sum"),
                usable_puts=("p_usable", "sum"),
                vendor_timestamp_count=(
                    "vendor_expiry_timestamp_utc", "nunique"
                ),
                vendor_timestamp_first=(
                    "vendor_expiry_timestamp_utc", "min"
                ),
                vendor_timestamp_last=(
                    "vendor_expiry_timestamp_utc", "max"
                ),
            )
            .reset_index()
        )
        coverage.insert(0, "quote_date", date)

        panel = []
        for prefix, kind in (("c", "call"), ("p", "put")):
            q = g.loc[
                ~duplicate & ~g["bad_contract"],
                [
                    "root",
                    "expire_date",
                    "strike",
                    "underlying_last",
                    "in_scope",
                ],
            ].copy()
            q["kind"] = kind
            q["bid"] = g.loc[q.index, f"{prefix}_bid"]
            q["ask"] = g.loc[q.index, f"{prefix}_ask"]
            q["usable"] = g.loc[q.index, f"{prefix}_usable"]
            panel.append(q)

        panel = pd.concat(panel, ignore_index=True).set_index(self.KEY)
        panel["seen"] = True
        return row, coverage, panel

    def _advance(self, date, panel, file_name, availability):
        if self._previous is not None:
            old_date, old, old_file = self._previous
            out = (
                old.join(panel.add_prefix("next_"), how="left")
                if panel is not None
                else old.copy()
            )
            if len(out):
                expiry = out.index.get_level_values("expire_date")
                status = pd.Series("matched", index=out.index)

                if panel is None:
                    status[:] = availability
                else:
                    status.loc[
                        ~out["next_in_scope"].eq(True)
                    ] = "out_of_scope_next"
                    status.loc[
                        ~out["next_usable"].eq(True)
                    ] = "invalid_next_quote"
                    status.loc[
                        ~out["next_seen"].eq(True)
                    ] = "missing_or_ambiguous_next"

                status.loc[
                    expiry <= date
                ] = "expiry_before_or_on_next_session"

                for value, count in status.value_counts().items():
                    self._adjacent.append(
                        {
                            "quote_date": old_date,
                            "next_date": date,
                            "cross_month": (
                                old_date.to_period("M")
                                != date.to_period("M")
                            ),
                            "from_file": old_file,
                            "to_file": file_name,
                            "status": value,
                            "contracts": int(count),
                        }
                    )

        starts = (
            panel.loc[
                panel["usable"] & panel["in_scope"]
            ].copy()
            if panel is not None
            else None
        )
        self._previous = (
            (date, starts, file_name)
            if starts is not None
            else None
        )

    def run(self, files, output, script_path=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        by_month = {}
        for path in sorted(map(Path, files)):
            match = re.search(r"(\d{6})\.txt$", path.name)
            if not match:
                raise ValueError(
                    f"Expected a monthly YYYYMM.txt filename: {path}"
                )
            month = pd.Period(
                match.group(1)[:4] + "-" + match.group(1)[4:],
                freq="M",
            )
            if month in by_month:
                raise ValueError(
                    f"Multiple files for {month}; "
                    "resolve overlapping sources first."
                )
            by_month[month] = path

        self._previous, self._adjacent = None, []
        daily, inventory, coverage, clocks = [], [], [], []

        for month in pd.period_range(
            self.settings.start, self.settings.end, freq="M"
        ):
            path = by_month.get(month)
            info = {
                "month": str(month),
                "file": str(path) if path else "",
                "status": "missing_file",
                "raw_rows": 0,
            }
            f = None
            if path:
                print(f"Auditing {path.name}...", flush=True)
                info["sha256"] = sha256(path)
                try:
                    f = self._read(path, month, info)
                    info["status"] = "audited"
                    for flag in self.FLAGS:
                        info[flag] = int(f[flag].sum())
                except (
                    ValueError,
                    pd.errors.ParserError,
                    UnicodeError,
                    StopIteration,
                ) as exc:
                    info.update(
                        status="file_failed", message=str(exc)
                    )
                    print(f"  File failed: {exc}", flush=True)

            inventory.append(info)
            observed = {}
            if f is not None:
                mask = (
                    f["quote_date"].between(
                        self.settings.start, self.settings.end
                    )
                    & ~f["file_month_mismatch"]
                )
                observed = dict(
                    tuple(
                        f.loc[mask].groupby(
                            "quote_date", sort=True
                        )
                    )
                )

            sessions = self.schedule.index[
                self.schedule.index.to_period("M") == month
            ]
            dates = sorted(set(sessions) | set(observed))

            for date in dates:
                g = observed.get(date)
                panel = None
                if g is None:
                    status = (
                        "missing_session_observation"
                        if info["status"] == "audited"
                        else info["status"]
                    )
                    row = {
                        "quote_date": date,
                        "file": info["file"],
                        "raw_rows": 0,
                        "reference_session": True,
                        "status": status,
                        "cash_close_ny": self.schedule.loc[
                            date, "cash_close_ny"
                        ].isoformat(),
                        "early_cash_close": bool(
                            self.schedule.loc[
                                date, "early_cash_close"
                            ]
                        ),
                    }
                else:
                    row, c, panel = self._day(
                        date, g, info["file"]
                    )
                    row["status"] = (
                        "observed_reference_session"
                        if date in sessions
                        else "observed_nonreference_date"
                    )
                    coverage.append(c)
                    for clock, count in (
                        g["clock_ny"].value_counts().items()
                    ):
                        clocks.append(
                            {
                                "quote_date": date,
                                "clock_ny": clock,
                                "rows": int(count),
                            }
                        )

                daily.append(row)
                if date in sessions:
                    self._advance(
                        date, panel, info["file"], row["status"]
                    )

            del f, observed

        if self._previous is not None:
            date, starts, file_name = self._previous
            if len(starts):
                self._adjacent.append(
                    {
                        "quote_date": date,
                        "next_date": pd.NaT,
                        "cross_month": False,
                        "from_file": file_name,
                        "to_file": "",
                        "status": "sample_end",
                        "contracts": len(starts),
                    }
                )

        daily_frame = (
            pd.DataFrame(daily)
            if daily
            else pd.DataFrame(
                columns=[
                    "quote_date",
                    "file",
                    "raw_rows",
                    "reference_session",
                    "status",
                ]
            )
        )
        tables = {
            "file_summary": pd.DataFrame(inventory),
            "daily_summary": daily_frame,
            "clock_summary": pd.DataFrame(clocks),
            "expiry_coverage": (
                pd.concat(coverage, ignore_index=True)
                if coverage
                else pd.DataFrame()
            ),
            "adjacent_summary": pd.DataFrame(
                self._adjacent,
                columns=[
                    "quote_date",
                    "next_date",
                    "cross_month",
                    "from_file",
                    "to_file",
                    "status",
                    "contracts",
                ],
            ),
        }
        tables["session_anomalies"] = tables[
            "daily_summary"
        ].loc[
            tables["daily_summary"]["status"].ne(
                "observed_reference_session"
            )
        ]

        schemas = (
            tables["file_summary"].dropna(subset=["columns"])
            if "columns" in tables["file_summary"]
            else pd.DataFrame()
        )
        tables["schema_summary"] = (
            schemas.groupby("columns")
            .agg(
                files=("month", "size"),
                first_month=("month", "min"),
                last_month=("month", "max"),
            )
            .reset_index()
            if len(schemas)
            else pd.DataFrame()
        )

        d = tables["daily_summary"].copy()
        d["year"] = pd.to_datetime(d["quote_date"]).dt.year
        d["observed_reference"] = (
            d["reference_session"] & d["raw_rows"].gt(0)
        )
        d["observed_nonreference"] = (
            ~d["reference_session"] & d["raw_rows"].gt(0)
        )
        d["missing_reference"] = (
            d["reference_session"] & d["raw_rows"].eq(0)
        )
        d["usable_snapshot"] = d["reference_session"] & (
            d.get(
                "usable_c_marks", pd.Series(0, index=d.index)
            ).fillna(0).gt(0)
            | d.get(
                "usable_p_marks", pd.Series(0, index=d.index)
            ).fillna(0).gt(0)
        )
        tables["year_summary"] = (
            d.groupby("year")
            .agg(
                observed_rows=("raw_rows", "sum"),
                reference_sessions=("reference_session", "sum"),
                observed_reference_dates=(
                    "observed_reference", "sum"
                ),
                missing_reference_dates=(
                    "missing_reference", "sum"
                ),
                observed_nonreference_dates=(
                    "observed_nonreference", "sum"
                ),
                provisional_usable_snapshot_dates=(
                    "usable_snapshot", "sum"
                ),
            )
            .reset_index()
        )

        self.schedule.to_csv(output / "reference_schedule.csv")
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)

        audit = {
            "settings": asdict(self.settings),
            "calendar": "XNYS cash-session reference",
            "exchange_calendars_version": self.calendar_version,
            "calendar_scope": (
                "Cash-market reference, not certified SPX "
                "trading or fixing conventions."
            ),
            "reference_schedule_sha256": sha256(
                output / "reference_schedule.csv"
            ),
            "file_summary_sha256": sha256(
                output / "file_summary.csv"
            ),
            "files_by_status": (
                tables["file_summary"]["status"]
                .value_counts()
                .to_dict()
            ),
            "dates_by_status": (
                tables["daily_summary"]["status"]
                .value_counts()
                .to_dict()
            ),
            "source_sha256": {
                str(Path(__file__)): sha256(__file__)
            },
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "after_cash_close": (
                "Descriptive timing flag; not proof of invalid "
                "quotes or of the actual SPX close."
            ),
            "continuity": (
                "Consecutive reference sessions; includes "
                "cross-month pairs, never skips a missing session."
            ),
            "snapshot_rule": (
                "One observed epoch timestamp and consistent "
                "spot; no latest-timestamp selection."
            ),
            "clock_scope": (
                "Observed vendor clock only; neither a trade "
                "timestamp nor an EOD convention is certified."
            ),
            "identity": (
                "Root/expiry/strike/type provisional; root "
                "and settlement evidence still required."
            ),
            "scope": (
                "Continuity starts within configured calendar-day "
                "and log(K/spot) windows; next lookup retains "
                "wider coverage."
            ),
            "zero_bids": (
                "Allowed for descriptive marks; separately counted, "
                "not approved for calibration or execution."
            ),
            "selection_scope": (
                "Retrospective audit only; never use future "
                "continuity to screen backtest entries."
            ),
            "contract_identity_certified": False,
            "fixings_certified": False,
            "daily_discount_curves_verified": False,
            "input_modified": False,
            "models_refitted": False,
            "backtest_performed": False,
        }
        if script_path:
            audit["source_sha256"][
                str(Path(script_path))
            ] = sha256(script_path)

        (output / "audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        return tables, audit