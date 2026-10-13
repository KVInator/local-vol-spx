"""Quote validation, filtering and monthly history preparation."""

from dataclasses import asdict, dataclass, field
from datetime import time
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from artifacts import atomic_json
from carry import CarryInputs

KEY = ["root", "expire_date", "strike", "kind"]


@dataclass(frozen=True)
class AuditSettings:
    start: str = "2023-09-01"
    end: str = "2023-09-30"
    close_ny: str = "16:00:00"
    timestamp_tolerance_seconds: float = 1.0
    spot_tolerance_points: float = 0.01
    minimum_days: int = 1
    maximum_days: int = 60
    log_spot_half_width: float = 0.1
    parity_half_width: float = 0.03
    minimum_parity_pairs: int = 5
    root_column: str | None = None

    def __post_init__(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("Start date must precede end date.")
        time.fromisoformat(self.close_ny)
        for value in (self.timestamp_tolerance_seconds, self.spot_tolerance_points):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Timestamp and spot tolerances must be nonnegative.")
        if not 1 <= self.minimum_days <= self.maximum_days:
            raise ValueError("Require 1 <= minimum_days <= maximum_days.")
        if not 0 < self.parity_half_width <= self.log_spot_half_width:
            raise ValueError("Require 0 < parity window <= coverage window.")
        if self.minimum_parity_pairs < 3:
            raise ValueError("At least three parity pairs are required.")


class HedgingDataAudit:
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
    CONTRACT = ["root", "expire_date", "strike", "kind"]

    def __init__(self, settings=None):
        self.settings = settings or AuditSettings()

    @staticmethod
    def header(value):
        return str(value).strip().strip("[]").strip().lower()

    def run(self, raw):
        s = self.settings
        f = raw.copy().reset_index(drop=True)
        f.columns = [self.header(c) for c in f.columns]
        if f.columns.duplicated().any():
            raise ValueError("Duplicate column names after header normalization.")
        columns = []
        for c in f:
            values = f[c].astype("string").str.strip()
            columns.append(
                {
                    "column": c,
                    "missing_rows": int((values.isna() | values.eq("")).sum()),
                }
            )
        column_summary = pd.DataFrame(columns)
        missing = sorted(set(self.REQUIRED) - set(f.columns))
        if missing:
            raise ValueError(f"Missing raw columns: {missing}")
        f["source_row"] = np.arange(1, len(f) + 1)
        for c in (
            "quote_unixtime",
            "underlying_last",
            "strike",
            "c_bid",
            "c_ask",
            "p_bid",
            "p_ask",
        ):
            f[c] = pd.to_numeric(f[c], errors="coerce")
        for c in ("quote_date", "expire_date"):
            f[c] = pd.to_datetime(
                f[c].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce"
            )
        utc = pd.to_datetime(f["quote_unixtime"], unit="s", utc=True, errors="coerce")
        local = utc.dt.tz_convert("America/New_York")
        text = f["quote_readtime"].astype(str).str.strip()
        read = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        read = read.fillna(
            pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="coerce")
        )
        read = read.dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        )
        f["quote_timestamp_utc"] = utc
        f["bad_timestamp"] = utc.isna() | read.isna()
        f["timestamp_mismatch"] = (
            utc - read
        ).dt.total_seconds().abs() > s.timestamp_tolerance_seconds
        f["date_mismatch"] = (
            local.dt.tz_localize(None).dt.normalize().ne(f["quote_date"])
        )
        target = time.fromisoformat(s.close_ny)
        target_seconds = target.hour * 3600 + target.minute * 60 + target.second
        seconds = (
            local.dt.hour * 3600
            + local.dt.minute * 60
            + local.dt.second
            + local.dt.microsecond / 1000000.0
        )
        f["at_close"] = (
            seconds - target_seconds
        ).abs() <= s.timestamp_tolerance_seconds
        f["in_date_range"] = f["quote_date"].between(s.start, s.end)
        selected = f["in_date_range"] & f["at_close"]
        f["bad_spot"] = ~np.isfinite(f["underlying_last"]) | f["underlying_last"].le(0)
        f["bad_contract"] = (
            f["expire_date"].isna()
            | ~np.isfinite(f["strike"])
            | f["strike"].le(0)
            | f["expire_date"].lt(f["quote_date"])
        )
        f["calendar_days"] = (f["expire_date"] - f["quote_date"]).dt.days
        ratio = f["strike"].where(~f["bad_contract"]) / f["underlying_last"].where(
            ~f["bad_spot"]
        )
        f["log_spot_moneyness"] = np.log(ratio)
        f["in_scope"] = (
            selected
            & f["calendar_days"].between(s.minimum_days, s.maximum_days)
            & f["log_spot_moneyness"].abs().le(s.log_spot_half_width)
        )
        if s.root_column:
            column = self.header(s.root_column)
            if column not in f:
                raise ValueError(f"Root column not found: {column}")
            root = f[column].astype("string").str.strip().str.upper()
            f["root"] = root.replace("", pd.NA).fillna("UNKNOWN")
        else:
            f["root"] = "UNKNOWN"
        f["identity_status"] = np.where(
            f["root"].eq("UNKNOWN"), "provisional_no_root", "root_present_unverified"
        )
        f["settlement_status"] = "unverified"
        keys = ["quote_date", "root", "expire_date", "strike"]
        f["duplicate_close_key"] = False
        chosen = f.loc[selected]
        f.loc[chosen.index, "duplicate_close_key"] = chosen.duplicated(keys, keep=False)
        spot_span = chosen.groupby("quote_date")["underlying_last"].agg(
            lambda x: x.max() - x.min()
        )
        f["spot_inconsistent"] = (
            f["quote_date"].map(spot_span).gt(s.spot_tolerance_points)
        )
        expiry_epoch = pd.to_numeric(
            f.get("expire_unix", pd.Series(np.nan, index=f.index)), errors="coerce"
        )
        f["vendor_expiry_timestamp_utc"] = pd.to_datetime(
            expiry_epoch, unit="s", utc=True, errors="coerce"
        )
        vendor_date = (
            f["vendor_expiry_timestamp_utc"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        f["vendor_expiry_date_mismatch"] = vendor_date.notna() & vendor_date.ne(
            f["expire_date"]
        )
        f["vendor_dte"] = pd.to_numeric(
            f.get("dte", pd.Series(np.nan, index=f.index)), errors="coerce"
        )
        f["vendor_dte_minus_calendar_days"] = f["vendor_dte"] - f["calendar_days"]
        bad = f[
            [
                "bad_timestamp",
                "timestamp_mismatch",
                "date_mismatch",
                "bad_spot",
                "bad_contract",
                "duplicate_close_key",
                "spot_inconsistent",
            ]
        ].any(axis=1)
        f["base_usable"] = selected & ~bad & f["calendar_days"].gt(0)
        for prefix in ("c", "p"):
            bid = f[f"{prefix}_bid"]
            ask = f[f"{prefix}_ask"]
            valid = np.isfinite(bid) & np.isfinite(ask) & bid.ge(0) & ask.ge(bid)
            f[f"{prefix}_bad_quote"] = ~valid
            f[f"{prefix}_locked_quote"] = valid & ask.eq(bid)
            f[f"{prefix}_zero_bid"] = valid & bid.eq(0)
            f[f"{prefix}_mark_usable"] = f["base_usable"] & valid & ask.gt(bid)
            f[f"{prefix}_mid"] = bid + (ask - bid) / 2
            f[f"{prefix}_spread"] = ask - bid
        dates = pd.DatetimeIndex(
            sorted(f.loc[f["in_date_range"], "quote_date"].unique())
        )
        if not len(dates):
            raise ValueError("No quote_date values in the requested range.")
        panel = self._panel(f.loc[selected & ~f["duplicate_close_key"]])
        coverage = (
            f.loc[selected]
            .groupby(["quote_date", "root", "expire_date"], dropna=False)
            .agg(
                raw_rows=("source_row", "size"),
                days=("calendar_days", "first"),
                minimum_y=("log_spot_moneyness", "min"),
                maximum_y=("log_spot_moneyness", "max"),
                scope_rows=("in_scope", "sum"),
                usable_calls=("c_mark_usable", "sum"),
                usable_puts=("p_mark_usable", "sum"),
                duplicate_rows=("duplicate_close_key", "sum"),
            )
            .reset_index()
        )
        daily = (
            f.loc[f["in_date_range"]]
            .groupby("quote_date")
            .agg(
                raw_rows=("source_row", "size"),
                close_rows=("at_close", "sum"),
                observed_timestamps=("quote_timestamp_utc", "nunique"),
                minimum_spot=("underlying_last", "min"),
                maximum_spot=("underlying_last", "max"),
                bad_timestamps=("bad_timestamp", "sum"),
                date_mismatches=("date_mismatch", "sum"),
                duplicate_close_rows=("duplicate_close_key", "sum"),
                scope_rows=("in_scope", "sum"),
            )
            .reset_index()
        )
        parity, pairs = self._parity(f, coverage)
        metadata = (
            f.loc[selected]
            .groupby(["root", "expire_date"], dropna=False)
            .agg(
                observations=("source_row", "size"),
                vendor_timestamp_count=("vendor_expiry_timestamp_utc", "nunique"),
                vendor_timestamp_first=("vendor_expiry_timestamp_utc", "min"),
                vendor_timestamp_last=("vendor_expiry_timestamp_utc", "max"),
                vendor_date_mismatch_rows=("vendor_expiry_date_mismatch", "sum"),
            )
            .reset_index()
        )
        metadata["settlement_status"] = "unverified"
        metadata["model_fixing_timestamp_utc"] = ""
        metadata["evidence"] = (
            "Contract identifier and historical fixing evidence required."
        )
        return {
            "column_summary": column_summary,
            "row_audit": f,
            "daily_summary": daily,
            "expiry_coverage": coverage,
            "expiry_metadata": metadata,
            "quote_panel": panel,
            "contract_continuity": self._continuity(panel, dates),
            "adjacent_observations": self._adjacent(panel, dates),
            "parity_summary": parity,
            "parity_pairs": pairs,
        }

    def _panel(self, f):
        keep = [
            "source_row",
            "quote_date",
            "quote_timestamp_utc",
            "root",
            "expire_date",
            "strike",
            "underlying_last",
            "calendar_days",
            "log_spot_moneyness",
            "in_scope",
            "identity_status",
            "settlement_status",
        ]
        parts = []
        for prefix, kind in (("c", "call"), ("p", "put")):
            q = f[keep].copy()
            q["kind"] = kind
            for field in (
                "bid",
                "ask",
                "mid",
                "spread",
                "bad_quote",
                "locked_quote",
                "zero_bid",
                "mark_usable",
            ):
                q[field] = f[f"{prefix}_{field}"]
            q["two_sided_quote"] = q["mark_usable"] & q["bid"].gt(0)
            parts.append(q)
        return pd.concat(parts, ignore_index=True)

    def _continuity(self, panel, dates):
        usable = panel.loc[panel["mark_usable"]].copy()
        contracts = usable.loc[usable["in_scope"], self.CONTRACT].drop_duplicates()
        usable = usable.merge(contracts, on=self.CONTRACT, validate="many_to_one")
        result = (
            usable.groupby(self.CONTRACT)
            .agg(
                first_quote=("quote_date", "min"),
                last_quote=("quote_date", "max"),
                usable_observations=("quote_date", "nunique"),
                scope_observations=("in_scope", "sum"),
                zero_bid_observations=("zero_bid", "sum"),
            )
            .reset_index()
        )
        first = dates.searchsorted(result["first_quote"])
        last = dates.searchsorted(result["last_quote"])
        stop = dates.searchsorted(result["expire_date"])
        result["missing_between_first_and_last"] = (
            last - first + 1 - result["usable_observations"]
        )
        result["unavailable_from_first_to_expiry_or_sample_end"] = (
            stop - first - result["usable_observations"]
        )
        result["observed_at_sample_start"] = result["first_quote"].eq(dates[0])
        result["expiry_after_sample_end"] = result["expire_date"].gt(dates[-1])
        result["identity_status"] = np.where(
            result["root"].eq("UNKNOWN"),
            "provisional_no_root",
            "root_present_unverified",
        )
        return result

    def _adjacent(self, panel, dates):
        start = panel.loc[panel["in_scope"] & panel["mark_usable"]].copy()
        mapping = dict(zip(dates[:-1], dates[1:]))
        start["next_date"] = pd.to_datetime(start["quote_date"].map(mapping))
        fields = self.CONTRACT + [
            "quote_date",
            "bid",
            "ask",
            "mid",
            "mark_usable",
            "in_scope",
            "underlying_last",
            "source_row",
        ]
        end = panel[fields].rename(
            columns={c: f"next_{c}" for c in fields if c not in self.CONTRACT}
        )
        end = end.rename(columns={"next_quote_date": "next_date"})
        out = start.merge(
            end,
            on=self.CONTRACT + ["next_date"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        out["status"] = np.select(
            [
                out["next_date"].isna(),
                out["expire_date"].le(out["next_date"]),
                out["_merge"].eq("left_only"),
                ~out["next_mark_usable"].eq(True),
                ~out["next_in_scope"].eq(True),
            ],
            [
                "sample_end",
                "expiry_before_or_on_next_date",
                "missing_or_ambiguous_next",
                "invalid_next_quote",
                "out_of_scope_next",
            ],
            default="matched",
        )
        matched = out["status"].isin(["matched", "out_of_scope_next"])
        out["unchanged_bid_ask"] = (
            matched & out["bid"].eq(out["next_bid"]) & out["ask"].eq(out["next_ask"])
        )
        out["unchanged_quote_with_spot_move"] = out["unchanged_bid_ask"] & out[
            "underlying_last"
        ].ne(out["next_underlying_last"])
        out["elapsed_calendar_days"] = (out["next_date"] - out["quote_date"]).dt.days
        return out.drop(columns="_merge")

    def _parity(self, f, coverage):
        s = self.settings
        mask = (
            f["c_mark_usable"]
            & f["p_mark_usable"]
            & f["c_bid"].gt(0)
            & f["p_bid"].gt(0)
            & f["log_spot_moneyness"].abs().le(s.parity_half_width)
            & f["calendar_days"].between(s.minimum_days, s.maximum_days)
        )
        groups = {
            key: g
            for key, g in f.loc[mask].groupby(["quote_date", "root", "expire_date"])
        }
        rows, details = ([], [])
        columns = ["quote_date", "root", "expire_date", "days"]
        for values in coverage[columns].itertuples(index=False, name=None):
            key, days = (values[:3], values[3])
            row = dict(zip(("quote_date", "root", "expire_date"), key))
            g = groups.get(key, f.iloc[:0])
            status = (
                "insufficient_pairs"
                if s.minimum_days <= days <= s.maximum_days
                else "outside_day_range"
            )
            row.update(
                days=days, pairs=len(g), status=status, settlement_status="unverified"
            )
            if len(g) >= s.minimum_parity_pairs and g["strike"].nunique() >= 3:
                spot = float(g["underlying_last"].median())
                x = g["strike"].to_numpy() / spot - 1
                lower = (g["c_bid"] - g["p_ask"]).to_numpy()
                upper = (g["c_ask"] - g["p_bid"]).to_numpy()
                mid = (lower + upper) / 2
                half = (upper - lower) / 2
                matrix = np.column_stack((np.ones(len(x)), x))
                initial = np.linalg.lstsq(
                    matrix / half[:, None], mid / half, rcond=None
                )[0]
                fit = least_squares(
                    lambda p: (matrix @ p - mid) / half,
                    initial,
                    loss="soft_l1",
                    f_scale=1.0,
                )
                discount = -fit.x[1] / spot
                forward = spot + fit.x[0] / discount if discount > 0 else np.nan
                predicted = matrix @ fit.x
                residual = (predicted - mid) / half
                valid_fit = fit.success and np.isfinite(forward) and (forward > 0)
                excess = np.maximum.reduce(
                    [lower - predicted, predicted - upper, np.zeros(len(g))]
                )
                row.update(
                    status="diagnostic_fit" if valid_fit else "invalid_fit",
                    fitted_forward=forward,
                    fitted_discount=discount,
                    optimizer_success=bool(fit.success),
                    rms_parity_half_widths=float(np.sqrt(np.mean(residual**2))),
                    outside_parity_bands=int(
                        ((predicted < lower) | (predicted > upper)).sum()
                    ),
                    maximum_band_excess_points=float(excess.max()),
                )
                d = g[
                    ["source_row", "quote_date", "root", "expire_date", "strike"]
                ].copy()
                d["cp_lower"] = lower
                d["cp_upper"] = upper
                d["fitted_cp"] = predicted
                d["residual_half_widths"] = residual
                details.append(d)
            rows.append(row)
        pair_columns = [
            "source_row",
            "quote_date",
            "root",
            "expire_date",
            "strike",
            "cp_lower",
            "cp_upper",
            "fitted_cp",
            "residual_half_widths",
        ]
        pairs = (
            pd.concat(details, ignore_index=True)
            if details
            else pd.DataFrame(columns=pair_columns)
        )
        return (pd.DataFrame(rows), pairs)


@dataclass(frozen=True)
class QuotePolicySettings:
    start: str = "2023-09-01"
    end: str = "2023-09-29"
    minimum_days: int = 4
    maximum_days: int = 60
    calibration_half_width: float = 0.1
    entry_minimum_days: int = 14
    entry_maximum_days: int = 45
    entry_half_width: float = 0.03

    def __post_init__(self):
        dates = [pd.Timestamp(self.start), pd.Timestamp(self.end)]
        if any(
            (pd.isna(d) or d.tzinfo is not None or d != d.normalize() for d in dates)
        ):
            raise ValueError("Use nonmissing, timezone-free calendar dates.")
        if dates[0] > dates[1]:
            raise ValueError("Start must precede end.")
        if (
            not 1
            <= self.minimum_days
            <= self.entry_minimum_days
            <= self.entry_maximum_days
            <= self.maximum_days
        ):
            raise ValueError("Entry day range must lie inside calibration range.")
        if not (
            np.isfinite(self.calibration_half_width)
            and 0 < self.entry_half_width <= self.calibration_half_width
        ):
            raise ValueError("Require finite positive, nested moneyness windows.")


class QuotePolicy:

    def __init__(self, session_policy, settings=None):
        self.settings = settings or QuotePolicySettings()
        p = session_policy.copy()
        required = {
            "quote_date",
            "reference_session",
            "early_cash_close",
            "session_policy",
        }
        if not required.issubset(p):
            raise ValueError(f"Missing policy columns: {sorted(required - set(p))}")
        p["quote_date"] = pd.to_datetime(p["quote_date"], errors="raise")
        if p["quote_date"].isna().any() or p["quote_date"].duplicated().any():
            raise ValueError("Policy dates must be nonmissing and unique.")
        reference = p["reference_session"].astype(str).str.lower()
        if not reference.isin(["true", "false"]).all():
            raise ValueError("Reference-session flags must be true or false.")
        p["reference_session"] = reference.eq("true")
        early = p["early_cash_close"].astype(str).str.lower()
        if not early.loc[p["reference_session"]].isin(["true", "false"]).all():
            raise ValueError("Reference sessions require an early-close flag.")
        p["early_cash_close"] = early.eq("true")
        self.policy = p.sort_values("quote_date").set_index("quote_date")
        self.sessions = self.policy.index[self.policy["reference_session"]]
        s = self.settings
        horizon = pd.Timestamp(s.end) + pd.Timedelta(days=s.maximum_days)
        first = horizon.to_period("M").start_time
        last_monthly = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        required_end = max(horizon, last_monthly)
        if (
            not len(self.sessions)
            or self.policy.index[0] > pd.Timestamp(s.start)
            or self.policy.index[-1] < required_end
        ):
            raise ValueError(
                "Policy must cover the pilot and its full maturity horizon."
            )
        self.regular_expiries = self.policy.index[
            self.policy["reference_session"] & ~self.policy["early_cash_close"]
        ]
        self.monthly_aliases = self._monthly_aliases(horizon)

    def _monthly_aliases(self, horizon):
        aliases = set()
        for month in pd.period_range(self.settings.start, horizon, freq="M"):
            first = month.start_time
            friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
            previous = self.sessions[self.sessions <= friday]
            if not len(previous):
                raise ValueError("Insufficient reference calendar for monthly expiry.")
            fixing = previous[-1]
            prior = self.sessions[self.sessions < fixing]
            if not len(prior):
                raise ValueError("Missing preceding monthly-expiry session.")
            aliases.update([friday, friday + pd.Timedelta(days=1), fixing, prior[-1]])
        return pd.DatetimeIndex(sorted(aliases))

    def prepare(self, raw):
        s = self.settings
        tables = HedgingDataAudit(
            AuditSettings(
                start=s.start,
                end=s.end,
                minimum_days=s.minimum_days,
                maximum_days=s.maximum_days,
                log_spot_half_width=s.calibration_half_width,
                parity_half_width=min(0.03, s.calibration_half_width),
            )
        ).run(raw.astype(object))
        q = tables["quote_panel"].copy()
        same_day_policy = q["quote_date"].map(self.policy["session_policy"])
        ordinary = same_day_policy.eq("provisional_standard_session") & q[
            "quote_date"
        ].isin(self.regular_expiries)
        supported = q["expire_date"].isin(self.regular_expiries)
        ambiguous = q["expire_date"].isin(self.monthly_aliases)
        q["quote_status"] = np.select(
            [
                ~ordinary,
                ~q["mark_usable"],
                ~supported,
                ambiguous,
                ~q["in_scope"],
                q["zero_bid"],
            ],
            [
                "session_excluded",
                "invalid_or_locked_quote",
                "expiry_calendar_excluded",
                "monthly_alias_excluded",
                "outside_calibration_scope",
                "zero_bid_bound",
            ],
            default="research_calibration_quote",
        )
        assumed = (
            (q["expire_date"] + pd.Timedelta(hours=16))
            .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
        q["assumed_fixing_utc"] = assumed
        q["assumed_maturity_years"] = (
            assumed - q["quote_timestamp_utc"]
        ).dt.total_seconds() / (365 * 24 * 3600)
        q["settlement_status"] = "PM_16_NY_assumed_unverified"
        q["contract_identity_verified"] = False
        q["snapshot_provenance_verified"] = False
        q["entry_research_candidate"] = (
            q["quote_status"].eq("research_calibration_quote")
            & q["calendar_days"].between(s.entry_minimum_days, s.entry_maximum_days)
            & q["log_spot_moneyness"].abs().le(s.entry_half_width)
        )
        quotes = q.loc[q["quote_status"].eq("research_calibration_quote")].copy()
        bounds = q.loc[q["quote_status"].eq("zero_bid_bound")].copy()
        if len(quotes) and (
            quotes["assumed_maturity_years"].isna().any()
            or quotes["assumed_maturity_years"].le(0).any()
        ):
            raise ValueError("Selected quotes require a positive assumed maturity.")
        summary = (
            q.groupby(["quote_date", "kind", "quote_status"])
            .size()
            .rename("quotes")
            .reset_index()
        )
        coverage = (
            quotes.groupby(["quote_date", "expire_date", "kind"])
            .agg(
                quotes=("strike", "size"),
                days=("calendar_days", "first"),
                min_y=("log_spot_moneyness", "min"),
                max_y=("log_spot_moneyness", "max"),
                entry_candidates=("entry_research_candidate", "sum"),
            )
            .reset_index()
        )
        timeline = self.policy.loc[s.start : s.end].reset_index()
        row_audit = tables["row_audit"]
        audit = {
            "settings": asdict(s),
            "raw_rows": len(raw),
            "calibration_quotes": len(quotes),
            "zero_bid_bounds": len(bounds),
            "duplicate_rows_quarantined": int(row_audit["duplicate_close_key"].sum()),
            "rows_outside_snapshot_selection": int(
                (~(row_audit["in_date_range"] & row_audit["at_close"])).sum()
            ),
            "monthly_date_aliases_excluded": self.monthly_aliases.strftime(
                "%Y-%m-%d"
            ).tolist(),
            "time_assumption": "16:00 America/New_York expiry; ACT/365F using UTC elapsed time.",
            "root_inferred": False,
            "contract_identity_verified": False,
            "snapshot_provenance_verified": False,
            "daily_carry_verified": False,
            "monthly_exclusion": "Third Friday, following Saturday, scheduled holiday-adjusted session and its preceding session; conservative date aliases, not verified series identities.",
            "entry_selection": "Current-day quote quality, days and moneyness only; no future availability filter.",
            "expiry_calendar": "Reference-session schedule only; future raw-data availability is not used.",
            "scope": "Development quotes for a provisional diffusion study; no settlement payoff or execution claim.",
            "models_refitted": False,
            "backtest_performed": False,
        }
        return (
            {
                "pilot_quotes": quotes,
                "zero_bid_bounds": bounds,
                "quote_selection_summary": summary,
                "expiry_coverage": coverage,
                "session_timeline": timeline,
            },
            audit,
        )


def boolean(series):
    values = series.astype(str).str.lower().map({"true": True, "false": False})
    if values.isna().any():
        raise ValueError(f"Invalid Boolean column: {series.name}")
    return values.astype(bool)


def utc(series):
    values = [pd.Timestamp(value) for value in series]
    if any((pd.isna(value) or value.tzinfo is None for value in values)):
        raise ValueError(f"Timezone-aware timestamps required: {series.name}")
    return pd.Series(pd.to_datetime(values, utc=True), index=series.index)


def identifier(date, values):
    return hashlib.sha256(
        json.dumps([str(date), *map(str, values)], separators=(",", ":")).encode()
    ).hexdigest()[:24]


def empty_quotes():
    return pd.DataFrame(
        columns=[
            "quote_date",
            *KEY,
            "quote_timestamp_utc",
            "assumed_fixing_utc",
            "assumed_maturity_years",
            "underlying_last",
            "bid",
            "ask",
            "mid",
            "entry_research_candidate",
        ]
    )


def frame(path):
    result = pd.read_csv(
        path,
        dtype={
            "quote_date": str,
            "expire_date": str,
            "root": str,
            "entry_id": str,
            "contract_id": str,
        },
    )
    for name in ("quote_timestamp_utc", "assumed_fixing_utc", "end_timestamp"):
        if name in result:
            result[name] = pd.to_datetime(result[name], utc=True)
    return result


class QuoteHistory:

    def __init__(self, raw, policy_file, settings, cache):
        self.raw = Path(raw)
        self.policy_file = Path(policy_file)
        self.settings = settings
        self.cache = cache
        self.month_memory = {}
        self.carry = CarryInputs()

    def month_data(self, month, freeze):
        if month in self.month_memory:
            return self.month_memory[month]
        s = self.settings
        path = self.raw / f"spx_eod_{month.replace('-', '')}.txt"
        identity = dict(
            raw_sha256=freeze["identity"]["input_sha256"].get(str(path)),
            freeze=freeze["fingerprint"],
            month=month,
        )
        folder, metadata = self.cache.get(
            f"months/{month}/prepare",
            identity,
            self.prepare_month,
            month=month,
            path=path,
        )
        if metadata["status"] == "prepared":
            quotes, bounds = (
                frame(folder / "pilot_quotes.csv"),
                frame(folder / "zero_bid_bounds.csv"),
            )
        else:
            quotes = bounds = empty_quotes()
        carry_folder, carry_meta = self.cache.get(
            f"months/{month}/carry", identity, self.prepare_carry, quotes=quotes
        )
        carry = (
            frame(carry_folder / "primary_carry.csv")
            if carry_meta["status"] == "prepared"
            else pd.DataFrame()
        )
        calibration = (
            frame(carry_folder / "calibration_quotes.csv")
            if carry_meta["status"] == "prepared"
            else pd.DataFrame()
        )
        result = dict(
            quotes=quotes,
            bounds=bounds,
            carry=carry,
            calibration=calibration,
            prepare_status=metadata["status"],
            carry_status=carry_meta["status"],
        )
        self.month_memory[month] = result
        while len(self.month_memory) > 2:
            del self.month_memory[next(iter(self.month_memory))]
        return result

    def quotes_on(self, date, freeze):
        if not self.settings.start <= date <= self.settings.end:
            return empty_quotes()
        data = self.month_data(date[:7], freeze)
        parts = [x.loc[x.quote_date.eq(date)] for x in (data["quotes"], data["bounds"])]
        nonempty = [x for x in parts if len(x)]
        return pd.concat(nonempty, ignore_index=True) if nonempty else empty_quotes()

    def prepare_month(self, folder, month, path):
        s = self.settings
        if not path.exists():
            return dict(status="raw_file_missing", message=str(path))
        raw = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
        first = str(max(pd.Period(month).start_time, pd.Timestamp(s.start)).date())
        last = str(
            min(pd.Period(month).end_time.normalize(), pd.Timestamp(s.end)).date()
        )
        policy = pd.read_csv(self.policy_file)
        tables, audit = QuotePolicy(
            policy, QuotePolicySettings(start=first, end=last)
        ).prepare(raw)
        for name in ("pilot_quotes", "zero_bid_bounds", "quote_selection_summary"):
            tables[name].to_csv(folder / f"{name}.csv", index=False)
        atomic_json(folder / "prepare_audit.json", audit)
        return dict(
            status="prepared",
            raw_rows=len(raw),
            calibration_quote_sides=len(tables["pilot_quotes"]),
        )

    def prepare_carry(self, target, quotes):
        if quotes.empty:
            return dict(status="no_calibration_quotes")
        tables, audit = self.carry.prepare(quotes)
        for name in ("primary_carry", "calibration_quotes", "daily_summary"):
            tables[name].to_csv(target / f"{name}.csv", index=False)
        atomic_json(target / "carry_audit.json", audit)
        return dict(status="prepared", primary_quotes=len(tables["calibration_quotes"]))
