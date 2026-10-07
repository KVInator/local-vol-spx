"""Paired one-session research hedges using observed option marks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedge_ledger import HedgeLedgerSettings, SelfFinancingHedgeLedger


@dataclass(frozen=True)
class PilotComparisonSettings:
    lend_rate: float = 0.05
    borrow_rate: float = 0.05
    hedge_fee_bps: float = 1.0

    def __post_init__(self):
        HedgeLedgerSettings(
            lend_rate=self.lend_rate,
            borrow_rate=self.borrow_rate,
            hedge_fee_bps=self.hedge_fee_bps,
        )
        if self.hedge_fee_bps <= 0:
            raise ValueError(
                "Use a positive fee for the separate hedge-fee scenario"
            )


@dataclass(frozen=True)
class PilotComparisonResult:
    trials: pd.DataFrame
    coverage: pd.DataFrame
    ledgers: pd.DataFrame
    paired: pd.DataFrame


class PilotHedgeComparison:
    STRATEGIES = ("unhedged", "black", "ah")
    META = (
        "entry_id", "contract_id", "quote_date", "end_date", "root",
        "expire_date", "strike", "kind", "calendar_days", "entry_spot_y",
    )

    def __init__(self, settings=None):
        self.settings = settings or PilotComparisonSettings()

    @property
    def cases(self):
        s = self.settings
        return {
            "mid": HedgeLedgerSettings(
                lend_rate=s.lend_rate,
                borrow_rate=s.borrow_rate,
            ),
            "option_spread": HedgeLedgerSettings(
                lend_rate=s.lend_rate,
                borrow_rate=s.borrow_rate,
                execution="bid_ask",
            ),
            "option_spread_fee": HedgeLedgerSettings(
                lend_rate=s.lend_rate,
                borrow_rate=s.borrow_rate,
                execution="bid_ask",
                hedge_fee_bps=s.hedge_fee_bps,
            ),
        }

    @staticmethod
    def eligibility(row):
        if row["end_status"] != "matched":
            return str(row["end_status"])
        if row["black_status"] != "ready":
            return "black_not_ready"
        if row["ah_status"] != "ready":
            return "ah_not_ready"
        return "eligible"

    def checked(self, panel):
        required = set(self.META) | {
            "quote_timestamp_utc", "assumed_fixing_utc",
            "underlying_last", "mid", "bid", "ask",
            "end_status", "end_timestamp", "end_spot",
            "end_mid", "end_bid", "end_ask",
            "black_status", "ah_status", "black_delta", "ah_delta",
        }
        if (
            panel.empty
            or panel.columns.has_duplicates
            or not required.issubset(panel)
        ):
            missing = sorted(required - set(panel))
            raise ValueError(
                f"Empty panel or invalid columns; missing {missing}"
            )

        data = panel.copy(deep=True)
        missing_meta = data[list(self.META)].isna().drop(columns="end_date")
        if missing_meta.any().any():
            raise ValueError("Missing entry identity or selection information")
        if data.entry_id.duplicated().any():
            raise ValueError("Duplicate entry_id")

        data["eligibility"] = [
            self.eligibility(row) for row in data.to_dict("records")
        ]

        eligible = data.loc[data.eligibility.eq("eligible")]
        for row in eligible.to_dict("records"):
            numbers = [
                row[name] for name in (
                    "underlying_last", "end_spot", "mid", "end_mid",
                    "bid", "ask", "end_bid", "end_ask",
                    "black_delta", "ah_delta", "strike",
                )
            ]
            if (
                not np.isfinite(np.asarray(numbers, float)).all()
                or min(numbers[:2]) <= 0
                or row["strike"] <= 0
            ):
                raise ValueError("Invalid price or ready delta")

            for prefix in ("", "end_"):
                bid, ask, mid = (
                    float(row[prefix + name])
                    for name in ("bid", "ask", "mid")
                )
                if (
                    not 0 <= bid < ask
                    or not np.isclose(
                        mid, (bid + ask) / 2, rtol=0, atol=1e-8
                    )
                ):
                    raise ValueError(
                        "Invalid observed quote or non-midpoint mark"
                    )

            if row["bid"] <= 0 or row["kind"] not in {"call", "put"}:
                raise ValueError("Invalid entry quote or option type")

            times = [
                pd.Timestamp(row[name])
                for name in (
                    "quote_timestamp_utc",
                    "end_timestamp",
                    "assumed_fixing_utc",
                )
            ]
            if (
                any(pd.isna(t) or t.tzinfo is None for t in times)
                or not times[0] < times[1] < times[2]
            ):
                raise ValueError(
                    "Require timezone-aware entry < exit < assumed fixing"
                )

            dates = [
                t.tz_convert("America/New_York").date().isoformat()
                for t in times
            ]
            if dates != [
                row["quote_date"], row["end_date"], row["expire_date"]
            ]:
                raise ValueError("Timestamp and contract dates disagree")

        return data

    @staticmethod
    def observations(row, delta):
        data = pd.DataFrame({
            "timestamp": [
                row["quote_timestamp_utc"], row["end_timestamp"]
            ],
            "spot": [row["underlying_last"], row["end_spot"]],
            "option_mark": [row["mid"], row["end_mid"]],
            "option_bid": [row["bid"], row["end_bid"]],
            "option_ask": [row["ask"], row["end_ask"]],
            "delta": [delta, 0.0],
            "dividend_per_unit": [0.0, 0.0],
        })

        # Synthetic index fills at its mark.
        # These are not observed hedge quotes.
        data["hedge_bid"] = data["hedge_ask"] = data["spot"]

        for name in ("contract_id", "root", "expire_date", "strike", "kind"):
            data[name] = row[name]
        return data

    @staticmethod
    def direct_price(row, delta, settings):
        """Independent cash calculation for a two-event short position."""
        first = (
            row["mid"] if settings.execution == "mid" else row["bid"]
        )
        last = (
            row["end_mid"]
            if settings.execution == "mid"
            else row["end_ask"]
        )

        entry_fee = (
            abs(delta) * row["underlying_last"]
            * settings.hedge_fee_bps / 10000
        )
        exit_fee = (
            abs(delta) * row["end_spot"]
            * settings.hedge_fee_bps / 10000
        )
        cash = first - delta * row["underlying_last"] - entry_fee
        rate = settings.lend_rate if cash >= 0 else settings.borrow_rate

        elapsed = (
            pd.Timestamp(row["end_timestamp"])
            - pd.Timestamp(row["quote_timestamp_utc"])
        ).total_seconds()
        years = elapsed / (365 * 86400)

        return (
            cash * np.exp(rate * years)
            + delta * row["end_spot"]
            - last
            - exit_fee
        )

    def run(self, panel):
        data = self.checked(panel)
        trials, coverage, ledgers = [], [], []

        for row in data.to_dict("records"):
            meta = {name: row[name] for name in self.META}

            for case, settings in self.cases.items():
                status = {
                    **meta,
                    "scenario": case,
                    "status": row["eligibility"],
                    "message": "",
                }

                if row["eligibility"] == "eligible":
                    group, paths = [], []
                    try:
                        for strategy in self.STRATEGIES:
                            delta = (
                                0.0 if strategy == "unhedged"
                                else float(row[strategy + "_delta"])
                            )
                            result = SelfFinancingHedgeLedger(settings).run(
                                self.observations(row, delta)
                            )
                            direct = self.direct_price(row, delta, settings)
                            error = result.summary["net_pnl"] - direct

                            scale = max(
                                1,
                                abs(direct),
                                abs(delta * row["underlying_last"]),
                            )
                            if abs(error) > 1e-9 + 1e-12 * scale:
                                raise ArithmeticError(
                                    "Two-event closed-form reconciliation failed"
                                )

                            group.append({
                                **meta,
                                "scenario": case,
                                "strategy": strategy,
                                "entry_delta": delta,
                                "delta_difference": (
                                    row["ah_delta"] - row["black_delta"]
                                ),
                                "spot_change": (
                                    row["end_spot"] - row["underlying_last"]
                                ),
                                "option_mark_change": (
                                    row["end_mid"] - row["mid"]
                                ),
                                **result.summary,
                                "closed_form_error": error,
                            })
                            paths.append(
                                result.ledger.assign(
                                    entry_id=row["entry_id"],
                                    scenario=case,
                                    strategy=strategy,
                                )
                            )

                        trials.extend(group)
                        ledgers.extend(paths)
                        status["status"] = "compared"

                    except ArithmeticError as error:
                        # Retain the failure without a partial strategy group.
                        status.update(
                            status="accounting_failed",
                            message=str(error),
                        )

                coverage.append(status)

        values = pd.DataFrame(trials)
        paired = self.paired(values) if len(values) else pd.DataFrame()

        return PilotComparisonResult(
            values,
            pd.DataFrame(coverage),
            pd.concat(ledgers, ignore_index=True)
            if ledgers else pd.DataFrame(),
            paired,
        )

    @staticmethod
    def paired(trials):
        indexed = trials.pivot(
            index=["scenario", "entry_id"],
            columns="strategy",
            values="net_pnl",
        )
        if indexed[
            list(PilotHedgeComparison.STRATEGIES)
        ].isna().any().any():
            raise ValueError("Incomplete paired strategy group")

        out = indexed.rename(
            columns={name: name + "_net_pnl" for name in indexed}
        ).reset_index()

        meta = trials.loc[
            trials.strategy.eq("ah"),
            list(PilotHedgeComparison.META) + ["scenario"],
        ]
        out = out.merge(
            meta,
            on=["scenario", "entry_id"],
            validate="one_to_one",
        )

        out["ah_minus_black_net_pnl"] = (
            out.ah_net_pnl - out.black_net_pnl
        )
        out["abs_error_improvement"] = (
            out.black_net_pnl.abs() - out.ah_net_pnl.abs()
        )
        out["squared_error_improvement"] = (
            out.black_net_pnl**2 - out.ah_net_pnl**2
        )
        out["ah_lower_abs_error"] = out.abs_error_improvement > 1e-10
        out["abs_error_tie"] = out.abs_error_improvement.abs() <= 1e-10
        return out


def summarize(trials, keys):
    """Zero-centred RMS and absolute P&L; no annualization or demeaning."""
    return trials.groupby(keys, dropna=False, sort=True).agg(
        comparisons=("entry_id", "size"),
        entry_dates=("quote_date", "nunique"),
        mean_net_pnl=("net_pnl", "mean"),
        std_net_pnl=("net_pnl", "std"),
        rms_net_pnl=(
            "net_pnl",
            lambda x: float(np.sqrt(np.mean(x**2))),
        ),
        mean_abs_net_pnl=(
            "net_pnl",
            lambda x: float(np.mean(np.abs(x))),
        ),
        p05_net_pnl=("net_pnl", lambda x: float(x.quantile(0.05))),
        p95_net_pnl=("net_pnl", lambda x: float(x.quantile(0.95))),
        mean_direct_costs=("direct_costs", "mean"),
        mean_interest=("total_interest", "mean"),
        mean_hedge_turnover=("total_hedge_turnover", "mean"),
        max_reconciliation=("max_reconciliation", "max"),
    ).reset_index()