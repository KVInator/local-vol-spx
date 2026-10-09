"""Current-date contract selection."""

from market_data import boolean, utc
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeSettings:
    target_days: tuple = (21, 35, 45)
    target_spot_y: tuple = (-0.02, 0.0, 0.02)
    kinds: tuple = ("call", "put")
    radius: float = 0.0005
    width: float = 0.75
    intervals: int = 24000
    steps_per_day: int = 128

    def __post_init__(self):
        if (
            not self.target_days
            or not self.target_spot_y
            or (not self.kinds)
            or (not np.isfinite(self.target_days + self.target_spot_y).all())
            or any((not 14 <= day <= 45 for day in self.target_days))
            or any((abs(y) > 0.03 for y in self.target_spot_y))
            or (len(set(self.kinds)) != len(self.kinds))
            or (not set(self.kinds).issubset({"call", "put"}))
            or (not np.isfinite([self.radius, self.width]).all())
            or (self.radius <= 0)
            or (self.width <= 0.1)
            or (not isinstance(self.intervals, int))
            or (self.intervals < 100)
            or (self.intervals % 2)
            or (not isinstance(self.steps_per_day, int))
            or (self.steps_per_day < 1)
        ):
            raise ValueError("Invalid entry basket or PDE settings")


class ContractSelector:
    KEY = ["root", "expire_date", "strike", "kind"]

    def __init__(self, settings=None):
        self.settings = settings or HedgeSettings()

    @staticmethod
    def checked_quotes(frame, allow_zero_bid=False):
        q = frame.copy()
        required = ContractSelector.KEY + [
            "quote_date",
            "quote_timestamp_utc",
            "assumed_fixing_utc",
            "assumed_maturity_years",
            "underlying_last",
            "bid",
            "ask",
            "mid",
        ]
        if not set(required).issubset(q):
            raise ValueError("Missing pilot quote columns")
        if q[required].isna().any().any():
            raise ValueError("Missing quote identity, timestamp or price")
        for name in ["quote_date", "expire_date"]:
            dates = pd.to_datetime(q[name], errors="raise")
            if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
                raise ValueError("Use timezone-free calendar dates")
            q[name] = dates.dt.strftime("%Y-%m-%d")
        q["root"] = q["root"].astype(str)
        for name in ["quote_timestamp_utc", "assumed_fixing_utc"]:
            q[name] = utc(q[name])
        numbers = [
            "strike",
            "assumed_maturity_years",
            "underlying_last",
            "bid",
            "ask",
            "mid",
        ]
        q[numbers] = q[numbers].apply(pd.to_numeric, errors="raise").astype(float)
        if not np.isfinite(q[numbers].to_numpy(float)).all():
            raise ValueError("Nonfinite quote inputs")
        bad_bid = q.bid.lt(0) if allow_zero_bid else q.bid.le(0)
        elapsed = (q.assumed_fixing_utc - q.quote_timestamp_utc).dt.total_seconds() / (
            365 * 86400
        )
        if (
            bad_bid.any()
            or (q.ask <= q.bid).any()
            or q[["strike", "underlying_last", "assumed_maturity_years"]]
            .le(0)
            .any()
            .any()
            or (not q.kind.isin(["call", "put"]).all())
            or (not np.allclose(q.mid, (q.bid + q.ask) / 2, rtol=0, atol=1e-08))
            or (
                not np.allclose(
                    q.assumed_maturity_years, elapsed, rtol=1e-12, atol=1e-12
                )
            )
            or (
                not q.quote_timestamp_utc.dt.tz_convert("America/New_York")
                .dt.strftime("%Y-%m-%d")
                .eq(q.quote_date)
                .all()
            )
            or (
                not q.assumed_fixing_utc.dt.tz_convert("America/New_York")
                .dt.strftime("%Y-%m-%d")
                .eq(q.expire_date)
                .all()
            )
        ):
            raise ValueError("Invalid quotes or inconsistent assumed maturities")
        if q.duplicated(["quote_date"] + ContractSelector.KEY).any():
            raise ValueError("Duplicate provisional quote key; no row is chosen")
        if len(q) and q.groupby("quote_date").quote_timestamp_utc.nunique().gt(1).any():
            raise ValueError("A date has multiple snapshot timestamps")
        return q

    @staticmethod
    def checked_timeline(frame):
        p = frame.copy()
        required = {"quote_date", "reference_session", "session_policy"}
        if not required.issubset(p):
            raise ValueError("Missing session timeline columns")
        dates = pd.to_datetime(p.quote_date, errors="raise")
        if (
            dates.isna().any()
            or dates.duplicated().any()
            or dates.dt.tz is not None
            or (not dates.eq(dates.dt.normalize()).all())
        ):
            raise ValueError("Invalid or duplicate session dates")
        p["quote_date"] = dates.dt.strftime("%Y-%m-%d")
        p["reference_session"] = boolean(p.reference_session)
        return p.sort_values("quote_date").reset_index(drop=True)

    def select(self, quotes, timeline, dates=None):
        q = self.checked_quotes(quotes)
        if "entry_research_candidate" not in q:
            raise ValueError("Missing current-date entry flags")
        q["entry_research_candidate"] = boolean(q.entry_research_candidate)
        p = self.checked_timeline(timeline)
        sessions = p.loc[p.reference_session, "quote_date"].tolist()
        requested = sessions if dates is None else sorted(set(dates))
        if not set(requested).issubset(sessions):
            raise ValueError("Entry dates must occur in the reference timeline")
        selected, buckets = ({}, [])
        for date in requested:
            current = q.loc[q.quote_date.eq(date)].copy()
            policy = p.loc[p.quote_date.eq(date), "session_policy"].iloc[0]
            if current.empty or policy != "provisional_standard_session":
                buckets.append(
                    {
                        "quote_date": date,
                        "status": "no_current_entries",
                        "session_policy": policy,
                    }
                )
                continue
            current["calendar_days"] = (
                pd.to_datetime(current.expire_date) - pd.Timestamp(date)
            ).dt.days
            current["entry_spot_y"] = np.log(current.strike / current.underlying_last)
            eligible = current.loc[
                current.entry_research_candidate
                & current.calendar_days.between(14, 45)
                & current.entry_spot_y.abs().le(0.03 + 1e-12)
            ]
            for root, group in eligible.groupby("root", sort=True):
                expiries = group[["expire_date", "calendar_days"]].drop_duplicates()
                for target_day in self.settings.target_days:
                    expiry = (
                        expiries.assign(
                            distance=abs(expiries.calendar_days - target_day)
                        )
                        .sort_values(["distance", "expire_date"])
                        .expire_date.iloc[0]
                    )
                    for kind in self.settings.kinds:
                        available = group.loc[
                            group.expire_date.eq(expiry) & group.kind.eq(kind)
                        ]
                        for target_y in self.settings.target_spot_y:
                            bucket = {
                                "quote_date": date,
                                "root": root,
                                "target_days": target_day,
                                "target_spot_y": target_y,
                                "kind": kind,
                                "expire_date": expiry,
                            }
                            if available.empty:
                                buckets.append({**bucket, "status": "side_unavailable"})
                                continue
                            row = (
                                available.assign(
                                    distance=abs(available.entry_spot_y - target_y)
                                )
                                .sort_values(["distance", "strike"])
                                .iloc[0]
                                .drop("distance")
                                .to_dict()
                            )
                            identity = [str(row[name]) for name in self.KEY]
                            contract_id = hashlib.sha256(
                                json.dumps(identity).encode()
                            ).hexdigest()[:20]
                            entry_id = hashlib.sha256(
                                f"{date}:{contract_id}".encode()
                            ).hexdigest()[:20]
                            duplicate = entry_id in selected
                            row.update(contract_id=contract_id, entry_id=entry_id)
                            selected[entry_id] = row
                            buckets.append(
                                {
                                    **bucket,
                                    "entry_id": entry_id,
                                    "strike": row["strike"],
                                    "actual_spot_y": row["entry_spot_y"],
                                    "status": (
                                        "duplicate_bucket" if duplicate else "selected"
                                    ),
                                }
                            )
            if eligible.empty:
                buckets.append({"quote_date": date, "status": "no_current_candidates"})
        return (pd.DataFrame(selected.values()), pd.DataFrame(buckets))

    def match_next(self, entries, quotes, bounds, timeline):
        q = self.checked_quotes(quotes)
        b = self.checked_quotes(bounds, allow_zero_bid=True)
        parts = [q.assign(end_quote_source="pilot_quotes")]
        if not b.empty:
            parts.append(b.assign(end_quote_source="zero_bid_bounds"))
        all_quotes = pd.concat(parts, ignore_index=True)
        keys = ["quote_date"] + self.KEY
        if all_quotes.duplicated(keys).any():
            raise ValueError("Duplicate endpoint key across quote inputs")
        lookup = all_quotes.set_index(keys)
        p = self.checked_timeline(timeline)
        sessions = p.loc[p.reference_session, "quote_date"].tolist()
        next_date = dict(zip(sessions[:-1], sessions[1:]))
        policies = p.set_index("quote_date").session_policy
        records = []
        for row in entries.to_dict("records"):
            end = next_date.get(row["quote_date"])
            out = {
                "entry_id": row["entry_id"],
                "contract_id": row["contract_id"],
                "quote_date": row["quote_date"],
                "end_date": end,
                "end_status": "sample_end",
            }
            if end is not None:
                out["end_status"] = "next_session_excluded"
                if policies[end] == "provisional_standard_session":
                    out["end_status"] = "expiry_before_or_on_next_session"
                    if row["expire_date"] > end:
                        out["end_status"] = "next_quote_unavailable_in_pilot"
                        key = (end,) + tuple((row[name] for name in self.KEY))
                        if key in lookup.index:
                            other = lookup.loc[key]
                            if other.assumed_fixing_utc != row["assumed_fixing_utc"]:
                                raise ValueError(
                                    "Fixing assumption changed for a fixed contract"
                                )
                            out.update(
                                end_status="matched",
                                end_timestamp=other.quote_timestamp_utc,
                                end_spot=float(other.underlying_last),
                                end_bid=float(other.bid),
                                end_ask=float(other.ask),
                                end_mid=float(other.mid),
                                end_quote_source=other.end_quote_source,
                            )
            records.append(out)
        return pd.DataFrame(records)
