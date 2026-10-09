"""Fixed-contract schedules and multi-session hedge evaluation."""

from market_data import KEY, identifier
from hedging import STRATEGIES

METHODS = dict(STRATEGIES, unhedged=(None, None))
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from ledger import SelfFinancingHedgeLedger
from ledger import FundingScenarios, ComparisonSettings
from selection import ContractSelector, HedgeSettings
from calibration import DailyAHSettings
from hedging import SmileSettings, EmpiricalMVSettings


@dataclass(frozen=True)
class HistoricalSettings:
    start: str = "2013-01-01"
    end: str = "2023-12-31"
    schedules: tuple = ((1, 1), (5, 1), (5, 5), (10, 1), (10, 5))
    bootstrap_replicates: int = 500
    block_lengths: tuple = (10, 20, 40)
    seed: int = 20261008
    representative_ledgers: int = 24
    development_start: str = "2023-09-01"
    development_end: str = "2023-10-31"
    calibration: dict = field(default_factory=lambda: asdict(DailyAHSettings()))
    panel: dict = field(default_factory=lambda: asdict(HedgeSettings()))
    smile: dict = field(default_factory=lambda: asdict(SmileSettings()))
    empirical: dict = field(default_factory=lambda: asdict(EmpiricalMVSettings()))
    funding: dict = field(default_factory=lambda: asdict(ComparisonSettings()))

    def __post_init__(self):
        for name in ("start", "end", "development_start", "development_end"):
            t = pd.Timestamp(getattr(self, name))
            if (
                pd.isna(t)
                or t.tzinfo is not None
                or t != t.normalize()
                or (getattr(self, name) != t.strftime("%Y-%m-%d"))
            ):
                raise ValueError("Use finite timezone-free calendar dates.")
        if self.start > self.end or self.development_start > self.development_end:
            raise ValueError("Invalid date order.")
        if (
            not self.schedules
            or len(set(map(tuple, self.schedules))) != len(self.schedules)
            or any(
                (
                    len(x) != 2
                    or any((type(v) is not int for v in x))
                    or (not 1 <= x[1] <= x[0] <= 20)
                    for x in self.schedules
                )
            )
        ):
            raise ValueError(
                "Schedules are distinct (holding sessions, rebalance sessions)."
            )
        if (
            type(self.bootstrap_replicates) is not int
            or self.bootstrap_replicates < 100
            or type(self.seed) is not int
            or (self.seed < 0)
            or (type(self.representative_ledgers) is not int)
            or (self.representative_ledgers < 0)
            or (not self.block_lengths)
            or (len(set(self.block_lengths)) != len(self.block_lengths))
            or any(
                (
                    type(x) is not int or x < self.maximum_holding
                    for x in self.block_lengths
                )
            )
        ):
            raise ValueError(
                "Invalid bootstrap/ledger controls; blocks must cover the longest horizon."
            )
        DailyAHSettings(**self.calibration)
        HedgeSettings(**self.panel)
        if self.panel["radius"] < 2 * self.calibration["width"] / self.calibration[
            "intervals"
        ] * (1 - 1e-10):
            raise ValueError("Smoothing radius must cover at least one native AH cell.")
        if (
            self.panel["width"]
            >= self.calibration["width"]
            - 2 * self.calibration["width"] / self.calibration["intervals"]
        ):
            raise ValueError(
                "The PDE domain must stay inside the native AH coefficient domain."
            )
        SmileSettings(**self.smile)
        EmpiricalMVSettings(**self.empirical)
        ComparisonSettings(**self.funding)

    @property
    def maximum_holding(self):
        return max((x[0] for x in self.schedules))

    def phase(self, date):
        return (
            "inspected_development"
            if self.development_start <= date <= self.development_end
            else "retrospective_history"
        )


class FixedContractPaths:
    """Plan from reference sessions, never screen entries by future availability."""

    def __init__(self, timeline, settings=None):
        self.settings = settings or HistoricalSettings()
        p = ContractSelector.checked_timeline(timeline)
        self.sessions = p.loc[
            p.reference_session
            & p.quote_date.between(self.settings.start, self.settings.end),
            "quote_date",
        ].tolist()
        self.positions = {date: i for i, date in enumerate(self.sessions)}
        self.policies = p.set_index("quote_date").session_policy.to_dict()

    def dates(self, entry_date, holding):
        i = self.positions[entry_date]
        return self.sessions[i : i + holding + 1]

    def requests(self, date, previous_entries, new_entries):
        """Union current contract requests over all rebalance schedules."""
        items = {
            tuple((row[k] for k in KEY)): row for row in new_entries.to_dict("records")
        }
        i = self.positions[date]
        for row in previous_entries:
            offset = i - self.positions[row["quote_date"]]
            if any(
                (
                    0 < offset < hold and offset % reb == 0
                    for hold, reb in self.settings.schedules
                )
            ):
                if row["expire_date"] > date:
                    items.setdefault(tuple((row[k] for k in KEY)), row)
        return items

    def path(self, entry, holding, rebalance, quote_lookup, decisions, completed_dates):
        dates = self.dates(entry["quote_date"], holding)
        meta = dict(
            entry_id=entry["entry_id"],
            contract_id=entry["contract_id"],
            quote_date=entry["quote_date"],
            end_date=dates[-1],
            root=entry["root"],
            expire_date=entry["expire_date"],
            strike=entry["strike"],
            kind=entry["kind"],
            calendar_days=entry["calendar_days"],
            entry_spot_y=entry["entry_spot_y"],
            holding_sessions=holding,
            rebalance_sessions=rebalance,
            phase=self.settings.phase(entry["quote_date"]),
        )
        common = "ready"
        failure_date = ""
        if len(dates) != holding + 1:
            common = "sample_end"
        rows = []
        for j, date in enumerate(dates):
            if common != "ready":
                break
            if date not in completed_dates:
                common, failure_date = ("pending_run_prefix", date)
                break
            if self.policies[date] != "provisional_standard_session":
                common, failure_date = ("session_excluded", date)
                break
            if date >= entry["expire_date"]:
                common, failure_date = ("expiry_before_or_on_path_session", date)
                break
            key = (date, *[entry[k] for k in KEY])
            observation = quote_lookup.get(key)
            if observation is None:
                common, failure_date = ("quote_unavailable_in_research_scope", date)
                break
            if pd.Timestamp(observation["assumed_fixing_utc"]) != pd.Timestamp(
                entry["assumed_fixing_utc"]
            ):
                raise ValueError("A fixed contract changed its fixing assumption.")
            first = pd.Timestamp(entry["quote_timestamp_utc"])
            current = pd.Timestamp(observation["quote_timestamp_utc"])
            fixing = pd.Timestamp(observation["assumed_fixing_utc"])
            if (
                any((t.tzinfo is None for t in (first, current, fixing)))
                or not first <= current < fixing
            ):
                raise ValueError("Invalid fixed-contract timestamp order.")
            rows.append(
                dict(
                    observation,
                    rebalance=j < holding and j % rebalance == 0,
                    decision=decisions.get(key),
                )
            )
        return (meta, common, failure_date, rows)


def funded_summaries(observations, delta_matrix, settings):
    """Vectorize methods on one path; verify both cash and P&L identities.

    Fixed unit short option, zero initial capital, zero dividend cash and
    fractional index fills. These are exactly the pilot's research conventions.
    """
    if (
        settings.option_quantity != -1
        or settings.option_multiplier != 1
        or settings.hedge_multiplier != 1
        or (settings.initial_capital != 0)
        or (settings.hedge_fee_per_trade != 0)
        or (settings.option_fee_per_contract != 0)
        or (not settings.liquidate_final)
    ):
        raise ValueError("Use the declared unit short-option conventions.")
    n = len(delta_matrix)
    stamps = pd.to_datetime(
        [row["quote_timestamp_utc"] for row in observations], utc=True
    )
    spots = np.array([row["underlying_last"] for row in observations], float)
    marks = np.array([row["mid"] for row in observations], float)
    bids = np.array([row["bid"] for row in observations], float)
    asks = np.array([row["ask"] for row in observations], float)
    if (
        len(stamps) < 2
        or not stamps.is_monotonic_increasing
        or stamps.has_duplicates
        or (
            not np.isfinite(np.r_[spots, marks, bids, asks, delta_matrix.ravel()]).all()
        )
        or (spots <= 0).any()
        or (bids < 0).any()
        or (asks <= bids).any()
        or (not np.allclose(marks, (bids + asks) / 2, rtol=0, atol=1e-08))
    ):
        raise ValueError("Invalid observed path or strategy deltas.")
    cash = np.zeros(n)
    previous_equity = np.zeros(n)
    hedge = np.zeros(n)
    option = 0.0
    gross_total = np.zeros(n)
    cost_total = np.zeros(n)
    interest_total = np.zeros(n)
    hedge_pnl_total = np.zeros(n)
    option_pnl_total = 0.0
    turnover = np.zeros(n)
    max_error = np.zeros(n)
    for j, (spot, mark) in enumerate(zip(spots, marks)):
        interest = np.zeros(n)
        op_pnl = 0.0
        h_pnl = np.zeros(n)
        if j:
            elapsed = (stamps[j] - stamps[j - 1]).total_seconds() / (365 * 86400)
            rates = np.where(cash >= 0, settings.lend_rate, settings.borrow_rate)
            interest = cash * np.expm1(rates * elapsed)
            op_pnl = option * (mark - marks[j - 1])
            h_pnl = hedge * (spot - spots[j - 1])
        closing = j == len(spots) - 1
        target_option = 0.0 if closing else -1.0
        target_hedge = np.zeros(n) if closing else delta_matrix[:, j]
        option_trade = target_option - option
        hedge_trade = target_hedge - hedge
        fill = mark
        if settings.execution == "bid_ask":
            fill = (
                asks[j] if option_trade > 0 else bids[j] if option_trade < 0 else mark
            )
        notional = abs(hedge_trade) * spot
        fee = notional * settings.hedge_fee_bps / 10000
        costs = option_trade * (fill - mark) + fee
        cash = cash + interest - option_trade * fill - hedge_trade * spot - fee
        option, hedge = (target_option, target_hedge)
        equity = cash + option * mark + hedge * spot
        gross = op_pnl + h_pnl + interest
        gross_total += gross
        cost_total += costs
        interest_total += interest
        hedge_pnl_total += h_pnl
        option_pnl_total += op_pnl
        turnover += notional
        residual = equity - previous_equity - gross + costs
        cumulative = equity - gross_total + cost_total
        error = np.maximum(abs(residual), abs(cumulative))
        scale = np.maximum.reduce(
            [
                np.ones(n),
                abs(cash),
                abs(equity),
                abs(previous_equity),
                abs(gross_total),
                abs(cost_total),
                abs(hedge * spot),
            ]
        )
        if not np.isfinite(cash).all() or np.any(error > 1e-10 + 1e-12 * scale):
            raise ArithmeticError(
                "Vectorized observed-mark ledger reconciliation failed."
            )
        max_error = np.maximum(max_error, error)
        previous_equity = equity
    rows = []
    for i in range(n):
        rows.append(
            dict(
                observations=len(spots),
                start=stamps[0].isoformat(),
                end=stamps[-1].isoformat(),
                liquidated=True,
                initial_capital=0.0,
                final_cash=float(cash[i]),
                final_equity=float(cash[i]),
                final_option_position=0.0,
                final_hedge_position=0.0,
                gross_pnl=float(gross_total[i]),
                direct_costs=float(cost_total[i]),
                net_pnl=float(cash[i]),
                max_reconciliation=float(max_error[i]),
                total_option_pnl=float(option_pnl_total),
                total_hedge_pnl=float(hedge_pnl_total[i]),
                total_interest=float(interest_total[i]),
                total_dividend_income=0.0,
                total_hedge_turnover=float(turnover[i]),
            )
        )
    return rows


class HedgePathEvaluator:
    """Account for every ready strategy along one fixed-contract path."""

    def __init__(self, settings=None):
        self.settings = settings or HistoricalSettings()
        self.scenarios = FundingScenarios(ComparisonSettings(**self.settings.funding))

    def run(self, meta, common, failure_date, path, representative_limit=None):
        """Evaluate each ready hedge under the shared funding and cost scenarios."""
        settings = self.settings
        comparator = self.scenarios
        trials, coverage, ledgers = ([], [], [])
        ready_methods, delta_rows = ([], [])
        for strategy, (status_column, delta_column) in METHODS.items():
            status, failed = (common, failure_date)
            delta = 0.0
            observations = []
            black_entry_delta = np.nan
            if status == "ready":
                first_decision = path[0]["decision"] or {}
                black_entry_delta = first_decision.get("black_delta", np.nan)
                for row in path:
                    if row["rebalance"] and strategy != "unhedged":
                        decision = row["decision"] or {}
                        status = decision.get(status_column, "decision_unavailable")
                        if status != "ready":
                            failed = row["quote_date"]
                            break
                        delta = float(decision[delta_column])
                        if not np.isfinite(delta):
                            raise ValueError("A ready hedge has a nonfinite delta.")
                    observations.append(
                        dict(
                            timestamp=row["quote_timestamp_utc"],
                            spot=row["underlying_last"],
                            option_mark=row["mid"],
                            option_bid=row["bid"],
                            option_ask=row["ask"],
                            hedge_bid=row["underlying_last"],
                            hedge_ask=row["underlying_last"],
                            delta=delta,
                            dividend_per_unit=0.0,
                            **{k: meta[k] for k in ["contract_id", *KEY]}
                        )
                    )
            if status == "ready":
                ready_methods.append((strategy, observations, black_entry_delta))
                delta_rows.append([row["delta"] for row in observations])
            else:
                for scenario in comparator.cases:
                    coverage.append(
                        dict(
                            **meta,
                            scenario=scenario,
                            strategy=strategy,
                            status=status,
                            failure_date=failed
                        )
                    )
        if ready_methods:
            deltas = np.asarray(delta_rows, float)
            spots = np.array([row["underlying_last"] for row in path], float)
            marks = np.array([row["mid"] for row in path], float)
            raw_errors = marks[-1] - marks[0] - deltas[:, :-1] @ np.diff(spots)
            for scenario, ledger_settings in comparator.cases.items():
                summaries = funded_summaries(path, deltas, ledger_settings)
                for i, (strategy, observations, black_delta) in enumerate(
                    ready_methods
                ):
                    summaries[i]["max_closed_form_error"] = abs(
                        summaries[i]["net_pnl"]
                        - (
                            -raw_errors[i]
                            + summaries[i]["total_interest"]
                            - summaries[i]["direct_costs"]
                        )
                    )
                    if summaries[i]["max_closed_form_error"] > 1e-08:
                        raise ArithmeticError("Raw error/funding/cost identity failed.")
                    trials.append(
                        dict(
                            **meta,
                            strategy=strategy,
                            scenario=scenario,
                            black_entry_delta=black_delta,
                            entry_delta=float(deltas[i, 0]),
                            raw_mark_error=float(raw_errors[i]),
                            **summaries[i]
                        )
                    )
                    coverage.append(
                        dict(
                            **meta,
                            scenario=scenario,
                            strategy=strategy,
                            status="compared",
                            failure_date=""
                        )
                    )
                    if (
                        representative_limit is None
                        or len(ledgers) < representative_limit
                    ):
                        result = SelfFinancingHedgeLedger(ledger_settings).run(
                            pd.DataFrame(observations)
                        )
                        for name in (
                            "net_pnl",
                            "direct_costs",
                            "gross_pnl",
                            "total_interest",
                            "total_hedge_turnover",
                        ):
                            if abs(result.summary[name] - summaries[i][name]) > 1e-08:
                                raise ArithmeticError(
                                    "Vectorized and scalar hedge ledgers disagree."
                                )
                        ledgers.append(
                            result.ledger.assign(
                                **meta, scenario=scenario, strategy=strategy
                            )
                        )
        return (trials, coverage, ledgers)
