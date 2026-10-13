"""Self-financing cash accounting and funding scenarios."""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeLedgerSettings:
    option_quantity: float = -1.0
    option_multiplier: float = 1.0
    hedge_multiplier: float = 1.0
    initial_capital: float = 0.0
    lend_rate: float = 0.05
    borrow_rate: float = 0.05
    execution: str = "mid"
    hedge_fee_bps: float = 0.0
    hedge_fee_per_trade: float = 0.0
    option_fee_per_contract: float = 0.0
    liquidate_final: bool = True

    def __post_init__(self):
        numeric = (
            "option_quantity",
            "option_multiplier",
            "hedge_multiplier",
            "initial_capital",
            "lend_rate",
            "borrow_rate",
            "hedge_fee_bps",
            "hedge_fee_per_trade",
            "option_fee_per_contract",
        )
        for name in numeric:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.option_quantity == 0:
            raise ValueError("option_quantity must be nonzero")
        if self.option_multiplier <= 0 or self.hedge_multiplier <= 0:
            raise ValueError("Multipliers must be positive")
        if (
            min(
                self.hedge_fee_bps,
                self.hedge_fee_per_trade,
                self.option_fee_per_contract,
            )
            < 0
        ):
            raise ValueError("Fees must be nonnegative")
        if self.execution not in {"mid", "bid_ask"}:
            raise ValueError("execution must be 'mid' or 'bid_ask'")
        if not isinstance(self.liquidate_final, bool):
            raise ValueError("liquidate_final must be Boolean")


@dataclass(frozen=True)
class HedgeLedgerResult:
    ledger: pd.DataFrame
    summary: dict


class SelfFinancingHedgeLedger:
    """Prices, cash and P&L use the same currency unit.

    Required columns: timestamp, spot, option_mark, delta. Timestamps must
    include a timezone and be strictly increasing. The option is fixed for
    the whole path; optional identity columns are checked for constancy.

    For bid/ask execution also supply option_bid, option_ask, hedge_bid and
    hedge_ask. 'mid' execution fills at the supplied marks, without imposing
    that those marks equal an arithmetic quote midpoint.

    Optional dividend_per_unit is cash paid at the right endpoint, per unit
    of underlying price, to the hedge held during the preceding interval.
    Its first value must be zero. It defaults to zero and is never inferred
    from an option-implied yield. Financing uses ACT/365F elapsed seconds and
    continuous compounding, with the rate selected by the opening cash sign.

    Hedge bps fees use absolute traded notional at the supplied spot mark.
    Fixed hedge fees apply only to nonzero trades; option fees use contracts.
    """

    def __init__(self, settings=None):
        self.settings = settings or HedgeLedgerSettings()

    def _inputs(self, observations):
        if not isinstance(observations, pd.DataFrame):
            raise TypeError("observations must be a DataFrame")
        if observations.columns.has_duplicates:
            raise ValueError("Duplicate column names")
        required = {"timestamp", "spot", "option_mark", "delta"}
        missing = required - set(observations.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if len(observations) < 2:
            raise ValueError("At least two observations are required")
        data = observations.copy(deep=True).reset_index(drop=True)
        times = [pd.Timestamp(value) for value in data["timestamp"]]
        if any((pd.isna(value) or value.tzinfo is None for value in times)):
            raise ValueError("Every timestamp must be valid and timezone-aware")
        data["timestamp"] = pd.to_datetime(times, utc=True)
        if (
            not data["timestamp"].is_monotonic_increasing
            or data["timestamp"].duplicated().any()
        ):
            raise ValueError(
                "Timestamps must be strictly increasing; inputs are not sorted"
            )
        for name in ("contract_id", "strike", "expire_date", "kind", "root"):
            if name in data:
                if data[name].isna().any() or data[name].nunique() != 1:
                    raise ValueError(f"{name} must identify one fixed contract")
        if "dividend_per_unit" not in data:
            data["dividend_per_unit"] = 0.0
        numeric = ["spot", "option_mark", "delta", "dividend_per_unit"]
        for prefix in ("option", "hedge"):
            bid, ask = (f"{prefix}_bid", f"{prefix}_ask")
            present = (bid in data, ask in data)
            if present[0] != present[1]:
                raise ValueError(f"Supply both {bid} and {ask}")
            if self.settings.execution == "bid_ask" and (not all(present)):
                raise ValueError(f"Bid/ask execution requires {bid} and {ask}")
            if all(present):
                numeric.extend([bid, ask])
        for name in numeric:
            data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
            if not np.isfinite(data[name]).all():
                raise ValueError(f"{name} must be finite at every observation")
        if (data["spot"] <= 0).any() or (data["option_mark"] < 0).any():
            raise ValueError("spot must be positive and option_mark nonnegative")
        if (data["dividend_per_unit"] < 0).any() or data.loc[
            0, "dividend_per_unit"
        ] != 0:
            raise ValueError("Dividends must be nonnegative, with zero at entry")
        for prefix, mark in (("option", "option_mark"), ("hedge", "spot")):
            bid, ask = (f"{prefix}_bid", f"{prefix}_ask")
            if bid in data:
                bad = (data[bid] < 0) | (data[bid] > data[ask])
                bad |= (data[mark] < data[bid]) | (data[mark] > data[ask])
                if bad.any():
                    raise ValueError(f"Invalid {prefix} quotes or mark outside bid/ask")
        return data

    def _fill(self, row, prefix, trade, mark):
        if trade == 0 or self.settings.execution == "mid":
            return float(mark)
        side = "ask" if trade > 0 else "bid"
        return float(row[f"{prefix}_{side}"])

    def run(self, observations):
        data = self._inputs(observations)
        s = self.settings
        cash = previous_equity = s.initial_capital
        option_position = hedge_position = 0.0
        cumulative_gross = cumulative_costs = 0.0
        records = []
        previous = None
        for index, row in data.iterrows():
            spot = float(row["spot"])
            mark = float(row["option_mark"])
            elapsed = interest = dividend = option_pnl = hedge_pnl = 0.0
            rate = np.nan
            if previous is not None:
                elapsed = (row["timestamp"] - previous["timestamp"]).total_seconds() / (
                    365 * 86400
                )
                rate = s.lend_rate if cash >= 0 else s.borrow_rate
                interest = cash * math.expm1(rate * elapsed)
                dividend = (
                    hedge_position * s.hedge_multiplier * row["dividend_per_unit"]
                )
                option_pnl = (
                    option_position
                    * s.option_multiplier
                    * (mark - previous["option_mark"])
                )
                hedge_pnl = (
                    hedge_position * s.hedge_multiplier * (spot - previous["spot"])
                )
            cash_before_trade = cash + interest + dividend
            closing = s.liquidate_final and index == len(data) - 1
            target_option = 0.0 if closing else s.option_quantity
            target_hedge = (
                0.0
                if closing
                else -s.option_quantity
                * s.option_multiplier
                * row["delta"]
                / s.hedge_multiplier
            )
            option_trade = target_option - option_position
            hedge_trade = target_hedge - hedge_position
            option_fill = self._fill(row, "option", option_trade, mark)
            hedge_fill = self._fill(row, "hedge", hedge_trade, spot)
            option_slippage = option_trade * s.option_multiplier * (option_fill - mark)
            hedge_slippage = hedge_trade * s.hedge_multiplier * (hedge_fill - spot)
            option_fee = abs(option_trade) * s.option_fee_per_contract
            hedge_fee = (
                abs(hedge_trade) * s.hedge_multiplier * spot * s.hedge_fee_bps / 10000
            )
            if hedge_trade != 0:
                hedge_fee += s.hedge_fee_per_trade
            costs = option_slippage + hedge_slippage + option_fee + hedge_fee
            cash = cash_before_trade - option_trade * s.option_multiplier * option_fill
            cash -= (
                hedge_trade * s.hedge_multiplier * hedge_fill + option_fee + hedge_fee
            )
            option_position = target_option
            hedge_position = target_hedge
            equity = (
                cash
                + option_position * s.option_multiplier * mark
                + hedge_position * s.hedge_multiplier * spot
            )
            gross = option_pnl + hedge_pnl + interest + dividend
            cumulative_gross += gross
            cumulative_costs += costs
            residual = equity - previous_equity - gross + costs
            cumulative_residual = (
                equity - s.initial_capital - cumulative_gross + cumulative_costs
            )
            scale = max(
                1.0,
                abs(cash),
                abs(equity),
                abs(previous_equity),
                abs(cash_before_trade),
                abs(option_trade * s.option_multiplier * option_fill),
                abs(hedge_trade * s.hedge_multiplier * hedge_fill),
                abs(option_position * s.option_multiplier * mark),
                abs(hedge_position * s.hedge_multiplier * spot),
                abs(cumulative_gross),
                abs(cumulative_costs),
            )
            values = (cash, equity, gross, costs, residual, cumulative_residual, scale)
            if not all((math.isfinite(value) for value in values)):
                raise ArithmeticError("Nonfinite accounting value")
            tolerance = 1e-10 + 1e-12 * scale
            if max(abs(residual), abs(cumulative_residual)) > tolerance:
                raise ArithmeticError("Self-financing reconciliation failed")
            record = row.to_dict()
            record.update(
                event=(
                    "liquidation" if closing else "entry" if index == 0 else "rebalance"
                ),
                interval_years=elapsed,
                funding_rate_used=rate,
                cash_before_trade=cash_before_trade,
                cash=cash,
                option_position=option_position,
                hedge_position=hedge_position,
                option_trade=option_trade,
                hedge_trade=hedge_trade,
                option_fill=option_fill,
                hedge_fill=hedge_fill,
                option_pnl=option_pnl,
                hedge_pnl=hedge_pnl,
                interest=interest,
                dividend_income=dividend,
                option_slippage=option_slippage,
                hedge_slippage=hedge_slippage,
                option_fee=option_fee,
                hedge_fee=hedge_fee,
                direct_costs=costs,
                equity=equity,
                gross_pnl_increment=gross,
                cumulative_gross_pnl=cumulative_gross,
                cumulative_direct_costs=cumulative_costs,
                net_pnl=equity - s.initial_capital,
                reconciliation=residual,
                cumulative_reconciliation=cumulative_residual,
                hedge_turnover=abs(hedge_trade) * s.hedge_multiplier * spot,
            )
            records.append(record)
            previous = row
            previous_equity = equity
        ledger = pd.DataFrame(records)
        last = ledger.iloc[-1]
        summary = {
            "observations": len(ledger),
            "start": ledger.iloc[0]["timestamp"].isoformat(),
            "end": last["timestamp"].isoformat(),
            "liquidated": s.liquidate_final,
            "initial_capital": s.initial_capital,
            "final_cash": float(last["cash"]),
            "final_equity": float(last["equity"]),
            "final_option_position": float(last["option_position"]),
            "final_hedge_position": float(last["hedge_position"]),
            "gross_pnl": float(last["cumulative_gross_pnl"]),
            "direct_costs": float(last["cumulative_direct_costs"]),
            "net_pnl": float(last["net_pnl"]),
            "max_reconciliation": float(
                ledger[["reconciliation", "cumulative_reconciliation"]]
                .abs()
                .to_numpy()
                .max()
            ),
        }
        for name in (
            "option_pnl",
            "hedge_pnl",
            "interest",
            "dividend_income",
            "hedge_turnover",
        ):
            summary[f"total_{name}"] = float(ledger[name].sum())
        return HedgeLedgerResult(ledger, summary)


@dataclass(frozen=True)
class ComparisonSettings:
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
            raise ValueError("Use a positive fee for the separate hedge-fee scenario")


class FundingScenarios:

    def __init__(self, settings=None):
        self.settings = settings or ComparisonSettings()

    @property
    def cases(self):
        s = self.settings
        return {
            "mid": HedgeLedgerSettings(
                lend_rate=s.lend_rate, borrow_rate=s.borrow_rate
            ),
            "option_spread": HedgeLedgerSettings(
                lend_rate=s.lend_rate, borrow_rate=s.borrow_rate, execution="bid_ask"
            ),
            "option_spread_fee": HedgeLedgerSettings(
                lend_rate=s.lend_rate,
                borrow_rate=s.borrow_rate,
                execution="bid_ask",
                hedge_fee_bps=s.hedge_fee_bps,
            ),
        }
