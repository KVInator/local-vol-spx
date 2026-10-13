"""Paired hedge statistics and entry-date block resampling."""

from paths import HistoricalSettings
from hedging import STRATEGIES

METHODS = dict(STRATEGIES, unhedged=(None, None))
import numpy as np
import pandas as pd


def matched_pairs(trials, common_strategies=False):
    """Common observations for each candidate/Black; optionally all methods."""
    if trials.empty:
        return pd.DataFrame()
    identity = ["entry_id", "holding_sessions", "rebalance_sessions", "scenario"]
    if trials.duplicated(identity + ["strategy"]).any():
        raise ValueError("Duplicate strategy result.")
    data = trials.copy()
    if common_strategies:
        counts = data.groupby(identity).strategy.nunique()
        good = counts.loc[counts.eq(len(METHODS))].reset_index()[identity]
        data = data.merge(good, on=identity, validate="many_to_one")
    black = data.loc[
        data.strategy.eq("black"), identity + ["net_pnl", "raw_mark_error"]
    ]
    paired = data.loc[~data.strategy.eq("black")].merge(
        black, on=identity, suffixes=("", "_black"), validate="many_to_one"
    )
    if paired.empty:
        return paired
    paired["maturity_bucket"] = pd.cut(
        paired.calendar_days, [0, 27, 39, np.inf], labels=["<=27d", "28-39d", ">=40d"]
    ).astype(str)
    paired["moneyness_bucket"] = np.select(
        [paired.entry_spot_y < -0.01, paired.entry_spot_y > 0.01],
        ["K_below_spot", "K_above_spot"],
        default="near_spot",
    )
    paired["delta_bucket"] = pd.cut(
        abs(paired.black_entry_delta),
        [0, 0.25, 0.5, 0.75, np.inf],
        labels=["0-.25", ".25-.50", ".50-.75", ">.75"],
        include_lowest=True,
    ).astype(str)
    paired["year"] = paired.quote_date.str[:4]
    return paired


def stationary_weights(n, replicates, mean_block, rng):
    """Circular stationary bootstrap; all rows on a date get the same weight."""
    if n < 1:
        raise ValueError("At least one reference date is required.")
    indices = np.empty((replicates, n), np.int32)
    indices[:, 0] = rng.integers(n, size=replicates)
    for j in range(1, n):
        restart = rng.random(replicates) < 1.0 / mean_block
        indices[:, j] = np.where(
            restart, rng.integers(n, size=replicates), (indices[:, j - 1] + 1) % n
        )
    weights = np.zeros((replicates, n), np.int32)
    np.add.at(weights, (np.arange(replicates)[:, None], indices), 1)
    return weights


class GainAccumulator:
    """Sufficient daily statistics; retain no eleven-year row-level dataframe."""

    COLUMNS = (
        "rows",
        "strategy_sse",
        "black_sse",
        "raw_strategy_sse",
        "raw_black_sse",
        "strategy_sum",
        "black_sum",
        "strategy_absolute",
        "black_absolute",
        "wins",
    )

    def __init__(self, reference_dates, settings=None, common_strategies=False):
        self.dates = list(reference_dates)
        if self.dates != sorted(set(self.dates)):
            raise ValueError("Reference dates must be unique and chronological.")
        self.positions = {d: i for i, d in enumerate(self.dates)}
        self.settings = settings or HistoricalSettings()
        self.common = common_strategies
        self.data = {}

    def add(self, trials):
        pairs = matched_pairs(trials, self.common)
        if pairs.empty:
            return pairs
        by = ["holding_sessions", "rebalance_sessions", "scenario", "strategy"]
        for values, candidate in pairs.groupby(by, sort=True):
            groups = [("overall", "all", candidate)]
            for dimension in (
                "phase",
                "year",
                "kind",
                "maturity_bucket",
                "moneyness_bucket",
                "delta_bucket",
            ):
                groups += [
                    (dimension, str(name), group)
                    for name, group in candidate.groupby(dimension, sort=True)
                ]
            for dimension, name, group in groups:
                key = (*values, dimension, name)
                values_by_date = self.data.setdefault(key, {})
                for date, g in group.groupby("quote_date"):
                    if date not in self.positions:
                        raise ValueError("Result date escapes reference calendar.")
                    a, b = (g.net_pnl.to_numpy(float), g.net_pnl_black.to_numpy(float))
                    ar, br = (
                        g.raw_mark_error.to_numpy(float),
                        g.raw_mark_error_black.to_numpy(float),
                    )
                    stat = np.array(
                        [
                            len(g),
                            a @ a,
                            b @ b,
                            ar @ ar,
                            br @ br,
                            a.sum(),
                            b.sum(),
                            abs(a).sum(),
                            abs(b).sum(),
                            np.sum(abs(a) < abs(b)),
                        ],
                        float,
                    )
                    if date in values_by_date:
                        values_by_date[date] += stat
                    else:
                        values_by_date[date] = stat
        return pairs

    def report(self):
        rng = np.random.default_rng(self.settings.seed)
        weights = (
            {
                block: stationary_weights(
                    len(self.dates), self.settings.bootstrap_replicates, block, rng
                )
                for block in self.settings.block_lengths
            }
            if self.dates
            else {}
        )
        records = []
        fields = [
            "holding_sessions",
            "rebalance_sessions",
            "scenario",
            "strategy",
            "dimension",
            "group",
        ]
        for key, daily in sorted(self.data.items()):
            matrix = np.zeros((len(self.dates), len(self.COLUMNS)))
            for date, stat in daily.items():
                matrix[self.positions[date]] = stat
            n, aa, bb, ra, rb, sa, sb, abs_a, abs_b, wins = matrix.sum(axis=0)
            out = dict(
                zip(fields, key),
                comparisons=int(n),
                entry_dates=len(daily),
                strategy_sse=aa,
                black_sse=bb,
                gain=1 - aa / bb if bb > 1e-20 else np.nan,
                gain_status="ready" if bb > 1e-20 else "zero_black_sse",
                raw_gain=1 - ra / rb if rb > 1e-20 else np.nan,
                mae_improvement=(abs_b - abs_a) / n,
                mse_improvement=(bb - aa) / n,
                strategy_rms=np.sqrt(aa / n),
                black_rms=np.sqrt(bb / n),
                strategy_mean=sa / n,
                black_mean=sb / n,
                fraction_lower_abs_error=wins / n,
                sample_scope=(
                    "all_method_intersection"
                    if self.common
                    else "candidate_Black_pairs"
                ),
            )
            sufficient = np.column_stack(
                [matrix[:, 1], matrix[:, 2], matrix[:, 8] - matrix[:, 7], matrix[:, 0]]
            )
            for block, w in weights.items():
                rec = dict(
                    out,
                    mean_block_sessions=block,
                    bootstrap_replicates=self.settings.bootstrap_replicates,
                    ci_status="insufficient_dates",
                    gain_ci_low=np.nan,
                    gain_ci_high=np.nan,
                    mae_ci_low=np.nan,
                    mae_ci_high=np.nan,
                    resolved_gain_sign=0,
                )
                if len(daily) >= max(20, 2 * block) and bb > 1e-20:
                    totals = w @ sufficient
                    valid = (totals[:, 1] > 1e-20) & (totals[:, 3] > 0)
                    rec["valid_bootstrap_replicates"] = int(valid.sum())
                    if valid.sum() >= 0.95 * self.settings.bootstrap_replicates:
                        gain = 1 - totals[valid, 0] / totals[valid, 1]
                        mae = totals[valid, 2] / totals[valid, 3]
                        lo, hi = np.quantile(gain, [0.025, 0.975])
                        rec.update(
                            ci_status="ready",
                            gain_ci_low=float(lo),
                            gain_ci_high=float(hi),
                            mae_ci_low=float(np.quantile(mae, 0.025)),
                            mae_ci_high=float(np.quantile(mae, 0.975)),
                            resolved_gain_sign=(
                                1 if lo > 1e-10 else -1 if hi < -1e-10 else 0
                            ),
                        )
                    else:
                        rec["ci_status"] = "unstable_denominator"
                records.append(rec)
        return pd.DataFrame(records)

    def daily_statistics(self):
        rows = []
        names = [
            "holding_sessions",
            "rebalance_sessions",
            "scenario",
            "strategy",
            "dimension",
            "group",
        ]
        for key, daily in sorted(self.data.items()):
            if key[-2] != "overall":
                continue
            for date, values in sorted(daily.items()):
                rows.append(
                    dict(
                        zip(names, key),
                        quote_date=date,
                        **dict(zip(self.COLUMNS, values))
                    )
                )
        return pd.DataFrame(rows)
