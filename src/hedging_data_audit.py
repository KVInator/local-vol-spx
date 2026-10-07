"""Descriptive raw-quote audit. Does not select a backtest universe."""

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class AuditSettings:
    start: str = "2023-09-01"
    end: str = "2023-09-30"
    close_ny: str = "16:00:00"
    timestamp_tolerance_seconds: float = 1.0
    spot_tolerance_points: float = 0.01
    minimum_days: int = 1
    maximum_days: int = 60
    log_spot_half_width: float = 0.10
    parity_half_width: float = 0.03
    minimum_parity_pairs: int = 5
    root_column: str | None = None

    def __post_init__(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("Start date must precede end date.")

        time.fromisoformat(self.close_ny)

        for value in (
            self.timestamp_tolerance_seconds,
            self.spot_tolerance_points,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "Timestamp and spot tolerances must be nonnegative."
                )

        if not 1 <= self.minimum_days <= self.maximum_days:
            raise ValueError(
                "Require 1 <= minimum_days <= maximum_days."
            )

        if not 0 < self.parity_half_width <= self.log_spot_half_width:
            raise ValueError(
                "Require 0 < parity window <= coverage window."
            )

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
            raise ValueError(
                "Duplicate column names after header normalization."
            )

        columns = []
        for c in f:
            values = f[c].astype("string").str.strip()
            columns.append(
                {
                    "column": c,
                    "missing_rows": int(
                        (values.isna() | values.eq("")).sum()
                    ),
                }
            )
        column_summary = pd.DataFrame(columns)

        missing = sorted(set(self.REQUIRED) - set(f.columns))
        if missing:
            raise ValueError(f"Missing raw columns: {missing}")

        # Parsed record number, starting at one after the CSV header.
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
                f[c].astype(str).str.strip(),
                format="%Y-%m-%d",
                errors="coerce",
            )

        utc = pd.to_datetime(
            f["quote_unixtime"],
            unit="s",
            utc=True,
            errors="coerce",
        )
        local = utc.dt.tz_convert("America/New_York")

        text = f["quote_readtime"].astype(str).str.strip()
        read = pd.to_datetime(
            text,
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        read = read.fillna(
            pd.to_datetime(
                text,
                format="%Y-%m-%d %H:%M",
                errors="coerce",
            )
        )
        read = read.dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="NaT",
        )

        f["quote_timestamp_utc"] = utc
        f["bad_timestamp"] = utc.isna() | read.isna()
        f["timestamp_mismatch"] = (
            (utc - read).dt.total_seconds().abs()
            > s.timestamp_tolerance_seconds
        )
        f["date_mismatch"] = (
            local.dt.tz_localize(None)
            .dt.normalize()
            .ne(f["quote_date"])
        )

        target = time.fromisoformat(s.close_ny)
        target_seconds = (
            target.hour * 3600 + target.minute * 60 + target.second
        )
        seconds = (
            local.dt.hour * 3600
            + local.dt.minute * 60
            + local.dt.second
            + local.dt.microsecond / 1e6
        )

        f["at_close"] = (
            (seconds - target_seconds).abs()
            <= s.timestamp_tolerance_seconds
        )
        f["in_date_range"] = f["quote_date"].between(s.start, s.end)
        selected = f["in_date_range"] & f["at_close"]

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

        f["calendar_days"] = (
            f["expire_date"] - f["quote_date"]
        ).dt.days

        ratio = (
            f["strike"].where(~f["bad_contract"])
            / f["underlying_last"].where(~f["bad_spot"])
        )
        f["log_spot_moneyness"] = np.log(ratio)

        # Reporting subset only. Wider close quotes remain exported.
        f["in_scope"] = (
            selected
            & f["calendar_days"].between(
                s.minimum_days, s.maximum_days
            )
            & f["log_spot_moneyness"]
            .abs()
            .le(s.log_spot_half_width)
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
            f["root"].eq("UNKNOWN"),
            "provisional_no_root",
            "root_present_unverified",
        )
        f["settlement_status"] = "unverified"

        keys = ["quote_date", "root", "expire_date", "strike"]
        f["duplicate_close_key"] = False
        chosen = f.loc[selected]

        # Quarantine every member of a duplicate group.
        f.loc[chosen.index, "duplicate_close_key"] = (
            chosen.duplicated(keys, keep=False)
        )

        spot_span = chosen.groupby("quote_date")[
            "underlying_last"
        ].agg(lambda x: x.max() - x.min())

        f["spot_inconsistent"] = (
            f["quote_date"]
            .map(spot_span)
            .gt(s.spot_tolerance_points)
        )

        expiry_epoch = pd.to_numeric(
            f.get(
                "expire_unix",
                pd.Series(np.nan, index=f.index),
            ),
            errors="coerce",
        )
        f["vendor_expiry_timestamp_utc"] = pd.to_datetime(
            expiry_epoch,
            unit="s",
            utc=True,
            errors="coerce",
        )
        vendor_date = (
            f["vendor_expiry_timestamp_utc"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        f["vendor_expiry_date_mismatch"] = (
            vendor_date.notna()
            & vendor_date.ne(f["expire_date"])
        )

        f["vendor_dte"] = pd.to_numeric(
            f.get("dte", pd.Series(np.nan, index=f.index)),
            errors="coerce",
        )
        f["vendor_dte_minus_calendar_days"] = (
            f["vendor_dte"] - f["calendar_days"]
        )

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

        # Expiry-day marks require independently verified fixing rules.
        f["base_usable"] = (
            selected & ~bad & f["calendar_days"].gt(0)
        )

        for prefix in ("c", "p"):
            bid = f[f"{prefix}_bid"]
            ask = f[f"{prefix}_ask"]

            valid = (
                np.isfinite(bid)
                & np.isfinite(ask)
                & bid.ge(0)
                & ask.ge(bid)
            )

            f[f"{prefix}_bad_quote"] = ~valid
            f[f"{prefix}_locked_quote"] = valid & ask.eq(bid)
            f[f"{prefix}_zero_bid"] = valid & bid.eq(0)
            f[f"{prefix}_mark_usable"] = (
                f["base_usable"] & valid & ask.gt(bid)
            )
            f[f"{prefix}_mid"] = bid + (ask - bid) / 2
            f[f"{prefix}_spread"] = ask - bid

        # Keep dates with unusable closes in the observation calendar.
        dates = pd.DatetimeIndex(
            sorted(
                f.loc[f["in_date_range"], "quote_date"].unique()
            )
        )
        if not len(dates):
            raise ValueError(
                "No quote_date values in the requested range."
            )

        panel = self._panel(
            f.loc[selected & ~f["duplicate_close_key"]]
        )

        coverage = (
            f.loc[selected]
            .groupby(
                ["quote_date", "root", "expire_date"],
                dropna=False,
            )
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
                vendor_timestamp_count=(
                    "vendor_expiry_timestamp_utc", "nunique"
                ),
                vendor_timestamp_first=(
                    "vendor_expiry_timestamp_utc", "min"
                ),
                vendor_timestamp_last=(
                    "vendor_expiry_timestamp_utc", "max"
                ),
                vendor_date_mismatch_rows=(
                    "vendor_expiry_date_mismatch", "sum"
                ),
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

            q["two_sided_quote"] = (
                q["mark_usable"] & q["bid"].gt(0)
            )
            parts.append(q)

        return pd.concat(parts, ignore_index=True)

    def _continuity(self, panel, dates):
        usable = panel.loc[panel["mark_usable"]].copy()

        contracts = usable.loc[
            usable["in_scope"], self.CONTRACT
        ].drop_duplicates()

        usable = usable.merge(
            contracts,
            on=self.CONTRACT,
            validate="many_to_one",
        )

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
        result["observed_at_sample_start"] = (
            result["first_quote"].eq(dates[0])
        )
        result["expiry_after_sample_end"] = (
            result["expire_date"].gt(dates[-1])
        )
        result["identity_status"] = np.where(
            result["root"].eq("UNKNOWN"),
            "provisional_no_root",
            "root_present_unverified",
        )

        return result

    def _adjacent(self, panel, dates):
        start = panel.loc[
            panel["in_scope"] & panel["mark_usable"]
        ].copy()

        mapping = dict(zip(dates[:-1], dates[1:]))
        start["next_date"] = pd.to_datetime(
            start["quote_date"].map(mapping)
        )

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
            columns={
                c: f"next_{c}"
                for c in fields
                if c not in self.CONTRACT
            }
        )
        end = end.rename(
            columns={"next_quote_date": "next_date"}
        )

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

        matched = out["status"].isin(
            ["matched", "out_of_scope_next"]
        )
        out["unchanged_bid_ask"] = (
            matched
            & out["bid"].eq(out["next_bid"])
            & out["ask"].eq(out["next_ask"])
        )
        out["unchanged_quote_with_spot_move"] = (
            out["unchanged_bid_ask"]
            & out["underlying_last"].ne(
                out["next_underlying_last"]
            )
        )
        out["elapsed_calendar_days"] = (
            out["next_date"] - out["quote_date"]
        ).dt.days

        return out.drop(columns="_merge")

    def _parity(self, f, coverage):
        s = self.settings

        mask = (
            f["c_mark_usable"]
            & f["p_mark_usable"]
            & f["c_bid"].gt(0)
            & f["p_bid"].gt(0)
            & f["log_spot_moneyness"]
            .abs()
            .le(s.parity_half_width)
            & f["calendar_days"].between(
                s.minimum_days, s.maximum_days
            )
        )

        groups = {
            key: g
            for key, g in f.loc[mask].groupby(
                ["quote_date", "root", "expire_date"]
            )
        }

        rows, details = [], []
        columns = ["quote_date", "root", "expire_date", "days"]

        for values in coverage[columns].itertuples(
            index=False, name=None
        ):
            key, days = values[:3], values[3]
            row = dict(
                zip(("quote_date", "root", "expire_date"), key)
            )
            g = groups.get(key, f.iloc[:0])

            status = (
                "insufficient_pairs"
                if s.minimum_days <= days <= s.maximum_days
                else "outside_day_range"
            )
            row.update(
                days=days,
                pairs=len(g),
                status=status,
                settlement_status="unverified",
            )

            if (
                len(g) >= s.minimum_parity_pairs
                and g["strike"].nunique() >= 3
            ):
                spot = float(g["underlying_last"].median())
                x = g["strike"].to_numpy() / spot - 1

                lower = (g["c_bid"] - g["p_ask"]).to_numpy()
                upper = (g["c_ask"] - g["p_bid"]).to_numpy()
                mid = (lower + upper) / 2
                half = (upper - lower) / 2

                matrix = np.column_stack(
                    (np.ones(len(x)), x)
                )
                initial = np.linalg.lstsq(
                    matrix / half[:, None],
                    mid / half,
                    rcond=None,
                )[0]

                fit = least_squares(
                    lambda p: (matrix @ p - mid) / half,
                    initial,
                    loss="soft_l1",
                    f_scale=1.0,
                )

                discount = -fit.x[1] / spot
                forward = (
                    spot + fit.x[0] / discount
                    if discount > 0
                    else np.nan
                )
                predicted = matrix @ fit.x
                residual = (predicted - mid) / half

                valid_fit = (
                    fit.success
                    and np.isfinite(forward)
                    and forward > 0
                )
                excess = np.maximum.reduce(
                    [
                        lower - predicted,
                        predicted - upper,
                        np.zeros(len(g)),
                    ]
                )

                row.update(
                    status=(
                        "diagnostic_fit"
                        if valid_fit
                        else "invalid_fit"
                    ),
                    fitted_forward=forward,
                    fitted_discount=discount,
                    optimizer_success=bool(fit.success),
                    rms_parity_half_widths=float(
                        np.sqrt(np.mean(residual**2))
                    ),
                    outside_parity_bands=int(
                        (
                            (predicted < lower)
                            | (predicted > upper)
                        ).sum()
                    ),
                    maximum_band_excess_points=float(
                        excess.max()
                    ),
                )

                d = g[
                    [
                        "source_row",
                        "quote_date",
                        "root",
                        "expire_date",
                        "strike",
                    ]
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

        return pd.DataFrame(rows), pairs