"""Prepare an explicitly provisional quote panel; no calibration or backtest."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hedging_data_audit import AuditSettings, HedgingDataAudit


@dataclass(frozen=True)
class PilotSettings:
    start: str = "2023-09-01"
    end: str = "2023-09-29"
    minimum_days: int = 4
    maximum_days: int = 60
    calibration_half_width: float = 0.10
    entry_minimum_days: int = 14
    entry_maximum_days: int = 45
    entry_half_width: float = 0.03

    def __post_init__(self):
        dates = [pd.Timestamp(self.start), pd.Timestamp(self.end)]
        if any(
            pd.isna(d) or d.tzinfo is not None or d != d.normalize()
            for d in dates
        ):
            raise ValueError("Use nonmissing, timezone-free calendar dates.")
        if dates[0] > dates[1]:
            raise ValueError("Start must precede end.")
        if not (
            1 <= self.minimum_days <= self.entry_minimum_days
            <= self.entry_maximum_days <= self.maximum_days
        ):
            raise ValueError("Entry day range must lie inside calibration range.")
        if not (
            np.isfinite(self.calibration_half_width)
            and 0 < self.entry_half_width <= self.calibration_half_width
        ):
            raise ValueError("Require finite positive, nested moneyness windows.")


class HedgingPilot:
    def __init__(self, session_policy, settings=None):
        self.settings = settings or PilotSettings()
        p = session_policy.copy()
        required = {
            "quote_date", "reference_session",
            "early_cash_close", "session_policy",
        }
        if not required.issubset(p):
            raise ValueError(
                f"Missing policy columns: {sorted(required - set(p))}"
            )

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
        last_monthly = first + pd.Timedelta(
            days=(4 - first.weekday()) % 7 + 14
        )
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
            friday = first + pd.Timedelta(
                days=(4 - first.weekday()) % 7 + 14
            )
            previous = self.sessions[self.sessions <= friday]
            if not len(previous):
                raise ValueError(
                    "Insufficient reference calendar for monthly expiry."
                )
            fixing = previous[-1]
            prior = self.sessions[self.sessions < fixing]
            if not len(prior):
                raise ValueError("Missing preceding monthly-expiry session.")
            aliases.update([
                friday, friday + pd.Timedelta(days=1), fixing, prior[-1],
            ])
        return pd.DatetimeIndex(sorted(aliases))

    def prepare(self, raw):
        s = self.settings
        tables = HedgingDataAudit(AuditSettings(
            start=s.start,
            end=s.end,
            minimum_days=s.minimum_days,
            maximum_days=s.maximum_days,
            log_spot_half_width=s.calibration_half_width,
            parity_half_width=min(0.03, s.calibration_half_width),
        )).run(raw.astype(object))

        q = tables["quote_panel"].copy()
        same_day_policy = q["quote_date"].map(self.policy["session_policy"])
        ordinary = (
            same_day_policy.eq("provisional_standard_session")
            & q["quote_date"].isin(self.regular_expiries)
        )
        supported = q["expire_date"].isin(self.regular_expiries)
        ambiguous = q["expire_date"].isin(self.monthly_aliases)

        q["quote_status"] = np.select(
            [
                ~ordinary, ~q["mark_usable"], ~supported,
                ambiguous, ~q["in_scope"], q["zero_bid"],
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
            q["expire_date"] + pd.Timedelta(hours=16)
        ).dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")

        q["assumed_fixing_utc"] = assumed
        q["assumed_maturity_years"] = (
            assumed - q["quote_timestamp_utc"]
        ).dt.total_seconds() / (365 * 24 * 3600)
        q["settlement_status"] = "PM_16_NY_assumed_unverified"
        q["contract_identity_verified"] = False
        q["snapshot_provenance_verified"] = False

        q["entry_research_candidate"] = (
            q["quote_status"].eq("research_calibration_quote")
            & q["calendar_days"].between(
                s.entry_minimum_days, s.entry_maximum_days
            )
            & q["log_spot_moneyness"].abs().le(s.entry_half_width)
        )

        quotes = q.loc[
            q["quote_status"].eq("research_calibration_quote")
        ].copy()
        bounds = q.loc[q["quote_status"].eq("zero_bid_bound")].copy()

        if len(quotes) and (
            quotes["assumed_maturity_years"].isna().any()
            or quotes["assumed_maturity_years"].le(0).any()
        ):
            raise ValueError(
                "Selected quotes require a positive assumed maturity."
            )

        summary = (
            q.groupby(["quote_date", "kind", "quote_status"])
            .size().rename("quotes").reset_index()
        )
        coverage = quotes.groupby(
            ["quote_date", "expire_date", "kind"]
        ).agg(
            quotes=("strike", "size"),
            days=("calendar_days", "first"),
            min_y=("log_spot_moneyness", "min"),
            max_y=("log_spot_moneyness", "max"),
            entry_candidates=("entry_research_candidate", "sum"),
        ).reset_index()
        timeline = self.policy.loc[s.start:s.end].reset_index()

        row_audit = tables["row_audit"]
        audit = {
            "settings": asdict(s),
            "raw_rows": len(raw),
            "calibration_quotes": len(quotes),
            "zero_bid_bounds": len(bounds),
            "duplicate_rows_quarantined": int(
                row_audit["duplicate_close_key"].sum()
            ),
            "rows_outside_snapshot_selection": int((
                ~(row_audit["in_date_range"] & row_audit["at_close"])
            ).sum()),
            "monthly_date_aliases_excluded": (
                self.monthly_aliases.strftime("%Y-%m-%d").tolist()
            ),
            "time_assumption": (
                "16:00 America/New_York expiry; ACT/365F using UTC elapsed time."
            ),
            "root_inferred": False,
            "contract_identity_verified": False,
            "snapshot_provenance_verified": False,
            "daily_carry_verified": False,
            "monthly_exclusion": (
                "Third Friday, following Saturday, scheduled holiday-adjusted "
                "session and its preceding session; conservative date aliases, "
                "not verified series identities."
            ),
            "entry_selection": (
                "Current-day quote quality, days and moneyness only; "
                "no future availability filter."
            ),
            "expiry_calendar": (
                "Reference-session schedule only; "
                "future raw-data availability is not used."
            ),
            "scope": (
                "Development quotes for a provisional diffusion study; "
                "no settlement payoff or execution claim."
            ),
            "models_refitted": False,
            "backtest_performed": False,
        }
        return {
            "pilot_quotes": quotes,
            "zero_bid_bounds": bounds,
            "quote_selection_summary": summary,
            "expiry_coverage": coverage,
            "session_timeline": timeline,
        }, audit