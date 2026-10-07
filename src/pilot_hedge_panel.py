"""Current-date contract selection and deltas for a one-session research pilot."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from ah_backward_pricer import AHBackwardPricer
from daily_ah_validation import AHShortEndVariance


def boolean(series):
    values = series.astype(str).str.lower().map({
        "true": True,
        "false": False,
    })
    if values.isna().any():
        raise ValueError(f"Invalid Boolean column: {series.name}")
    return values.astype(bool)


def utc(series):
    values = [pd.Timestamp(value) for value in series]
    if any(pd.isna(value) or value.tzinfo is None for value in values):
        raise ValueError(
            f"Timezone-aware timestamps required: {series.name}"
        )
    return pd.Series(
        pd.to_datetime(values, utc=True),
        index=series.index,
    )


@dataclass(frozen=True)
class HedgePanelSettings:
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
            or not self.kinds
            or not np.isfinite(
                self.target_days + self.target_spot_y
            ).all()
            or any(not 14 <= day <= 45 for day in self.target_days)
            or any(abs(y) > 0.03 for y in self.target_spot_y)
            or len(set(self.kinds)) != len(self.kinds)
            or not set(self.kinds).issubset({"call", "put"})
            or not np.isfinite([self.radius, self.width]).all()
            or self.radius <= 0
            or self.width <= 0.10
            or not isinstance(self.intervals, int)
            or self.intervals < 100
            or self.intervals % 2
            or not isinstance(self.steps_per_day, int)
            or self.steps_per_day < 1
        ):
            raise ValueError("Invalid entry basket or PDE settings")


class PilotContractSelector:
    KEY = ["root", "expire_date", "strike", "kind"]

    def __init__(self, settings=None):
        self.settings = settings or HedgePanelSettings()

    @staticmethod
    def checked_quotes(frame, allow_zero_bid=False):
        q = frame.copy()
        required = PilotContractSelector.KEY + [
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
            raise ValueError(
                "Missing quote identity, timestamp or price"
            )

        for name in ["quote_date", "expire_date"]:
            dates = pd.to_datetime(q[name], errors="raise")
            if (
                dates.dt.tz is not None
                or not dates.eq(dates.dt.normalize()).all()
            ):
                raise ValueError(
                    "Use timezone-free calendar dates"
                )
            q[name] = dates.dt.strftime("%Y-%m-%d")

        q["root"] = q["root"].astype(str)

        for name in [
            "quote_timestamp_utc",
            "assumed_fixing_utc",
        ]:
            q[name] = utc(q[name])

        numbers = [
            "strike",
            "assumed_maturity_years",
            "underlying_last",
            "bid",
            "ask",
            "mid",
        ]
        q[numbers] = q[numbers].apply(
            pd.to_numeric, errors="raise"
        ).astype(float)

        if not np.isfinite(q[numbers].to_numpy(float)).all():
            raise ValueError("Nonfinite quote inputs")

        bad_bid = q.bid.lt(0) if allow_zero_bid else q.bid.le(0)
        elapsed = (
            q.assumed_fixing_utc - q.quote_timestamp_utc
        ).dt.total_seconds() / (365 * 86400)

        if (
            bad_bid.any()
            or (q.ask <= q.bid).any()
            or q[
                ["strike", "underlying_last", "assumed_maturity_years"]
            ].le(0).any().any()
            or not q.kind.isin(["call", "put"]).all()
            or not np.allclose(
                q.mid, (q.bid + q.ask) / 2,
                rtol=0, atol=1e-8,
            )
            or not np.allclose(
                q.assumed_maturity_years, elapsed,
                rtol=1e-12, atol=1e-12,
            )
            or not q.quote_timestamp_utc.dt.tz_convert(
                "America/New_York"
            ).dt.strftime("%Y-%m-%d").eq(q.quote_date).all()
            or not q.assumed_fixing_utc.dt.tz_convert(
                "America/New_York"
            ).dt.strftime("%Y-%m-%d").eq(q.expire_date).all()
        ):
            raise ValueError(
                "Invalid quotes or inconsistent assumed maturities"
            )

        if q.duplicated(
            ["quote_date"] + PilotContractSelector.KEY
        ).any():
            raise ValueError(
                "Duplicate provisional quote key; no row is chosen"
            )

        if (
            len(q)
            and q.groupby("quote_date")
            .quote_timestamp_utc.nunique().gt(1).any()
        ):
            raise ValueError(
                "A date has multiple snapshot timestamps"
            )

        return q

    @staticmethod
    def checked_timeline(frame):
        p = frame.copy()
        required = {
            "quote_date",
            "reference_session",
            "session_policy",
        }
        if not required.issubset(p):
            raise ValueError("Missing session timeline columns")

        dates = pd.to_datetime(p.quote_date, errors="raise")
        if (
            dates.isna().any()
            or dates.duplicated().any()
            or dates.dt.tz is not None
            or not dates.eq(dates.dt.normalize()).all()
        ):
            raise ValueError("Invalid or duplicate session dates")

        p["quote_date"] = dates.dt.strftime("%Y-%m-%d")
        p["reference_session"] = boolean(p.reference_session)
        return p.sort_values("quote_date").reset_index(drop=True)

    def select(self, quotes, timeline, dates=None):
        q = self.checked_quotes(quotes)

        if "entry_research_candidate" not in q:
            raise ValueError("Missing current-date entry flags")

        q["entry_research_candidate"] = boolean(
            q.entry_research_candidate
        )

        p = self.checked_timeline(timeline)
        sessions = p.loc[p.reference_session, "quote_date"].tolist()
        requested = (
            sessions if dates is None else sorted(set(dates))
        )

        if not set(requested).issubset(sessions):
            raise ValueError(
                "Entry dates must occur in the reference timeline"
            )

        selected, buckets = {}, []

        for date in requested:
            current = q.loc[q.quote_date.eq(date)].copy()
            policy = p.loc[
                p.quote_date.eq(date), "session_policy"
            ].iloc[0]

            if (
                current.empty
                or policy != "provisional_standard_session"
            ):
                buckets.append({
                    "quote_date": date,
                    "status": "no_current_entries",
                    "session_policy": policy,
                })
                continue

            current["calendar_days"] = (
                pd.to_datetime(current.expire_date)
                - pd.Timestamp(date)
            ).dt.days

            current["entry_spot_y"] = np.log(
                current.strike / current.underlying_last
            )

            eligible = current.loc[
                current.entry_research_candidate
                & current.calendar_days.between(14, 45)
                & current.entry_spot_y.abs().le(0.03 + 1e-12)
            ]

            for root, group in eligible.groupby("root", sort=True):
                expiries = group[
                    ["expire_date", "calendar_days"]
                ].drop_duplicates()

                for target_day in self.settings.target_days:
                    expiry = expiries.assign(
                        distance=abs(
                            expiries.calendar_days - target_day
                        )
                    ).sort_values(
                        ["distance", "expire_date"]
                    ).expire_date.iloc[0]

                    for kind in self.settings.kinds:
                        available = group.loc[
                            group.expire_date.eq(expiry)
                            & group.kind.eq(kind)
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
                                buckets.append({
                                    **bucket,
                                    "status": "side_unavailable",
                                })
                                continue

                            row = available.assign(
                                distance=abs(
                                    available.entry_spot_y - target_y
                                )
                            ).sort_values(
                                ["distance", "strike"]
                            ).iloc[0].drop("distance").to_dict()

                            identity = [
                                str(row[name]) for name in self.KEY
                            ]
                            contract_id = hashlib.sha256(
                                json.dumps(identity).encode()
                            ).hexdigest()[:20]

                            entry_id = hashlib.sha256(
                                f"{date}:{contract_id}".encode()
                            ).hexdigest()[:20]

                            duplicate = entry_id in selected
                            row.update(
                                contract_id=contract_id,
                                entry_id=entry_id,
                            )
                            selected[entry_id] = row

                            buckets.append({
                                **bucket,
                                "entry_id": entry_id,
                                "strike": row["strike"],
                                "actual_spot_y": row["entry_spot_y"],
                                "status": (
                                    "duplicate_bucket"
                                    if duplicate else "selected"
                                ),
                            })

            if eligible.empty:
                buckets.append({
                    "quote_date": date,
                    "status": "no_current_candidates",
                })

        return (
            pd.DataFrame(selected.values()),
            pd.DataFrame(buckets),
        )

    def match_next(self, entries, quotes, bounds, timeline):
        q = self.checked_quotes(quotes)
        b = self.checked_quotes(bounds, allow_zero_bid=True)

        parts = [
            q.assign(end_quote_source="pilot_quotes")
        ]
        if not b.empty:
            parts.append(
                b.assign(end_quote_source="zero_bid_bounds")
            )
        all_quotes = pd.concat(parts, ignore_index=True)

        keys = ["quote_date"] + self.KEY
        if all_quotes.duplicated(keys).any():
            raise ValueError(
                "Duplicate endpoint key across quote inputs"
            )

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
                    out["end_status"] = (
                        "expiry_before_or_on_next_session"
                    )

                    if row["expire_date"] > end:
                        out["end_status"] = (
                            "next_quote_unavailable_in_pilot"
                        )
                        key = (end,) + tuple(
                            row[name] for name in self.KEY
                        )

                        if key in lookup.index:
                            other = lookup.loc[key]

                            if (
                                other.assumed_fixing_utc
                                != row["assumed_fixing_utc"]
                            ):
                                raise ValueError(
                                    "Fixing assumption changed "
                                    "for a fixed contract"
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


class BlackFixedIVBenchmark:
    """Observed-price IV; spot delta holds IV and F/S fixed. No clipping."""

    @staticmethod
    def price(forward, discount, strike, stddev, kind):
        if stddev == 0:
            sign = 1 if kind == "call" else -1
            return discount * max(
                (forward - strike) * sign, 0
            )

        d1 = np.log(forward / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev

        if kind == "call":
            return discount * (
                forward * ndtr(d1) - strike * ndtr(d2)
            )

        return discount * (
            strike * ndtr(-d2) - forward * ndtr(-d1)
        )

    def estimate(
        self, spot, strike, time, forward, discount, mark, kind
    ):
        values = [spot, strike, time, forward, discount, mark]
        if (
            not np.isfinite(values).all()
            or min(values[:5]) <= 0
            or kind not in {"call", "put"}
        ):
            raise ValueError("Invalid Black benchmark inputs")

        lower = self.price(
            forward, discount, strike, 0, kind
        )
        upper = discount * (
            forward if kind == "call" else strike
        )

        if not lower < mark < upper:
            return {
                "black_status": "no_finite_positive_iv",
                "black_lower_bound": lower,
                "black_upper_bound": upper,
            }

        std = brentq(
            lambda x: (
                self.price(forward, discount, strike, x, kind)
                - mark
            ) / (discount * forward),
            0,
            32,
            xtol=1e-13,
            rtol=1e-13,
        )

        d1 = np.log(forward / strike) / std + std / 2
        ratio = discount * forward / spot

        delta = ratio * (ndtr(d1) - (kind == "put"))
        gamma = (
            ratio
            * np.exp(-0.5 * d1**2)
            / (np.sqrt(2 * np.pi) * spot * std)
        )

        price = self.price(
            forward, discount, strike, std, kind
        )

        return {
            "black_status": "ready",
            "black_iv": std / np.sqrt(time),
            "black_price": price,
            "black_delta": delta,
            "black_gamma": gamma,
            "black_repricing_error": price - mark,
        }


class DailyHedgeDeltas:
    def __init__(self, settings=None):
        self.settings = settings or HedgePanelSettings()

    def calculate(self, entries, carry, model=None, progress=print):
        if entries.empty:
            return entries.copy()

        keys = ["quote_date", "root", "expire_date"]
        required = keys + [
            "case",
            "carry_ready",
            "spot",
            "maturity_years",
            "forward",
            "discount_factor",
            "annual_rate",
            "quote_timestamp_utc",
            "assumed_fixing_utc",
        ]

        if (
            not set(required).issubset(carry)
            or carry.duplicated(keys).any()
        ):
            raise ValueError("Missing or duplicate daily carry")

        if len(
            entries[["quote_date", "root"]].drop_duplicates()
        ) != 1:
            raise ValueError(
                "Calculate one entry date and root at a time"
            )

        date, root = entries[keys[:2]].iloc[0]
        c = carry.loc[
            carry.quote_date.eq(date) & carry.root.eq(root)
        ].copy()

        c["carry_ready"] = boolean(c.carry_ready)

        for name in [
            "quote_timestamp_utc",
            "assumed_fixing_utc",
        ]:
            c[name] = utc(c[name])

        numbers = [
            "spot", "maturity_years", "forward",
            "discount_factor", "annual_rate",
        ]
        c[numbers] = c[numbers].apply(
            pd.to_numeric, errors="raise"
        )

        if len(c) and (
            c["case"].nunique() != 1
            or c.annual_rate.nunique() != 1
        ):
            raise ValueError(
                "Use one current-date carry scenario"
            )

        if model is not None:
            ordered = c.sort_values("maturity_years")
            comparisons = [
                ("maturity_years", model.maturities),
                ("forward", model.forwards),
                ("discount_factor", model.discounts),
            ]

            if (
                not c.carry_ready.all()
                or len(c) != len(model.maturities)
                or not np.allclose(
                    c.spot, model.spot, rtol=0, atol=1e-8
                )
                or any(
                    not np.allclose(
                        ordered[name], values,
                        rtol=1e-12, atol=1e-10,
                    )
                    for name, values in comparisons
                )
            ):
                raise ValueError(
                    "Current-day model and carry disagree"
                )

            variance = AHShortEndVariance(
                model, self.settings.radius
            )
            solver = AHBackwardPricer(
                variance,
                domain_width=self.settings.width,
                space_intervals=self.settings.intervals,
                steps_per_day=self.settings.steps_per_day,
                early_time_power=3.0,
            )

        rows = []
        benchmark = BlackFixedIVBenchmark()

        for expiry, group in entries.groupby(
            "expire_date", sort=True
        ):
            match = c.loc[c.expire_date.eq(expiry)]

            if match.empty or not match.carry_ready.iloc[0]:
                rows.extend(
                    {
                        **row,
                        "black_status": "carry_unavailable",
                        "ah_status": "carry_unavailable",
                    }
                    for row in group.to_dict("records")
                )
                continue

            item = match.iloc[0]

            if (
                not np.isfinite(
                    item[numbers].to_numpy(float)
                ).all()
                or min(
                    item.spot, item.maturity_years,
                    item.forward, item.discount_factor,
                ) <= 0
                or not np.isclose(
                    item.discount_factor,
                    np.exp(
                        -item.annual_rate * item.maturity_years
                    ),
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise ValueError("Invalid assumed carry")

            for row in group.itertuples(index=False):
                if (
                    row.quote_timestamp_utc
                    != item.quote_timestamp_utc
                    or row.assumed_fixing_utc
                    != item.assumed_fixing_utc
                    or not np.isclose(
                        row.underlying_last, item.spot,
                        rtol=0, atol=1e-8,
                    )
                    or not np.isclose(
                        row.assumed_maturity_years,
                        item.maturity_years,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        "Entry quote and current carry disagree"
                    )

            solved, failure = {}, ""

            if model is not None:
                strikes = np.sort(group.strike.unique())
                progress(
                    f"{date}: {expiry}, "
                    f"{len(strikes)} unique strikes, "
                    f"{365 * item.maturity_years:.4g} days..."
                )

                try:
                    grids = solver.solve_many(
                        strikes, float(item.maturity_years)
                    )
                    solved = {
                        float(k): grid.greeks(model.spot)
                        for k, grid in zip(strikes, grids)
                    }
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    np.linalg.LinAlgError,
                ) as error:
                    failure = (
                        f"{type(error).__name__}: {error}"
                    )

            for entry in group.to_dict("records"):
                out = {
                    **entry,
                    "case": item["case"],
                    "annual_rate": float(item.annual_rate),
                    "forward": float(item.forward),
                    "discount_factor": float(item.discount_factor),
                    "forward_log_moneyness": float(
                        np.log(entry["strike"] / item.forward)
                    ),
                    "ah_status": "model_unavailable",
                    "ah_message": failure,
                }

                out.update(benchmark.estimate(
                    item.spot,
                    entry["strike"],
                    item.maturity_years,
                    item.forward,
                    item.discount_factor,
                    entry["mid"],
                    entry["kind"],
                ))

                if model is not None:
                    out["ah_status"] = "solve_failed"

                    if solved:
                        value = solved[float(entry["strike"])]
                        put = entry["kind"] == "put"
                        ratio = (
                            item.discount_factor
                            * item.forward
                            / item.spot
                        )

                        price = float(value.price) - (
                            put
                            * item.discount_factor
                            * (item.forward - entry["strike"])
                        )
                        delta = float(value.delta) - put * ratio
                        gamma = float(value.gamma)

                        if np.isfinite(
                            [price, delta, gamma]
                        ).all():
                            sign = 1 if not put else -1
                            lower = item.discount_factor * max(
                                (item.forward - entry["strike"])
                                * sign,
                                0,
                            )
                            upper = item.discount_factor * (
                                item.forward
                                if not put else entry["strike"]
                            )

                            flag = (
                                gamma < -1e-8
                                or not (
                                    -put * ratio - 1e-6
                                    <= delta
                                    <= (1 - put) * ratio + 1e-6
                                )
                            )
                            flag |= not (
                                lower - 1e-6
                                <= price
                                <= upper + 1e-6
                            )

                            out.update(
                                ah_status=(
                                    "shape_flag" if flag else "ready"
                                ),
                                ah_price=price,
                                ah_delta=delta,
                                ah_gamma=gamma,
                                ah_price_minus_mid=(
                                    price - entry["mid"]
                                ),
                                **solver.last_diagnostics,
                            )

                rows.append(out)

        return pd.DataFrame(rows)