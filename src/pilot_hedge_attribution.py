"""Explain saved one-session hedge comparisons without fitting a model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_comparison(folder):
    folder = Path(folder)
    audit_path = folder / "audit.json"
    hashes = {str(audit_path.resolve()): sha256(audit_path)}
    audit = json.loads(audit_path.read_text())

    if audit.get("status") not in {
        "completed", "completed_with_accounting_failures"
    }:
        raise ValueError("Comparison run is incomplete")

    for key in (
        "future_information_used_in_entry_hedges",
        "models_refitted",
        "prices_clipped",
    ):
        if audit.get(key) is not False:
            raise ValueError(f"Unexpected upstream policy: {key}")

    if audit.get("dividend_per_unit") != 0:
        raise ValueError(
            "This diagnostic requires the zero-dividend pilot convention"
        )

    frames = []
    for name in ("strategy_results", "paired_results", "coverage"):
        path = folder / f"{name}.csv"
        actual = sha256(path)

        if actual != audit.get("output_sha256", {}).get(path.name):
            raise ValueError(f"Checksum mismatch: {path.name}")

        hashes[str(path.resolve())] = actual
        frames.append(
            pd.read_csv(
                path,
                dtype={
                    key: str
                    for key in (
                        "entry_id", "contract_id", "quote_date",
                        "end_date", "expire_date", "root", "kind",
                    )
                },
            )
        )

    actual_coverage = {
        case: group.status.value_counts().to_dict()
        for case, group in frames[2].groupby("scenario")
    }
    if actual_coverage != audit.get("coverage_by_scenario"):
        raise ValueError("Coverage and audit counts disagree")

    return (*frames, audit, hashes)


class PilotHedgeAttribution:
    KEYS = ["scenario", "entry_id"]

    META = [
        "contract_id", "quote_date", "end_date", "root", "expire_date",
        "strike", "kind", "calendar_days", "entry_spot_y",
    ]

    NUMBERS = [
        "entry_delta", "delta_difference", "spot_change",
        "option_mark_change", "total_option_pnl", "total_hedge_pnl",
        "total_interest", "total_dividend_income", "direct_costs",
        "net_pnl",
    ]

    STRATEGIES = {"ah", "black", "unhedged"}
    CASES = {"mid", "option_spread", "option_spread_fee"}

    @staticmethod
    def close(actual, expected, label):
        if not np.allclose(actual, expected, rtol=1e-11, atol=1e-8):
            raise ValueError(f"Reconciliation failed: {label}")

    def checked(self, trials, paired, coverage):
        required = self.KEYS + self.META + self.NUMBERS + [
            "strategy", "observations", "liquidated", "initial_capital",
            "final_option_position", "final_hedge_position",
        ]

        if (
            trials.empty
            or trials.columns.has_duplicates
            or not set(required).issubset(trials)
        ):
            raise ValueError("Empty or incomplete strategy results")

        d = trials.copy(deep=True)

        if (
            d[required].isna().any().any()
            or d.duplicated(self.KEYS + ["strategy"]).any()
        ):
            raise ValueError("Missing values or duplicate strategy rows")

        numeric = self.NUMBERS + [
            "strike", "calendar_days", "entry_spot_y"
        ]
        if not np.isfinite(d[numeric].to_numpy(float)).all():
            raise ValueError("Nonfinite comparison values")

        if (
            not set(d.scenario).issubset(self.CASES)
            or "mid" not in set(d.scenario)
        ):
            raise ValueError("Unknown scenarios or missing midpoint baseline")

        if (
            not d.kind.isin(["call", "put"]).all()
            or (d.strike <= 0).any()
            or (d.calendar_days <= 0).any()
        ):
            raise ValueError("Invalid contract metadata")

        if (d.direct_costs < 0).any():
            raise ValueError("Negative direct costs")

        if (
            not d.observations.eq(2).all()
            or not d.liquidated.eq(True).all()
        ):
            raise ValueError("Require two-event liquidated experiments")

        self.close(
            d[[
                "initial_capital",
                "final_option_position",
                "final_hedge_position",
            ]],
            0,
            "zero initial capital and closed final positions",
        )

        groups = d.groupby(self.KEYS, sort=False)

        complete = groups.strategy.agg(
            lambda x: set(x) == self.STRATEGIES
        )
        if not complete.all():
            raise ValueError("Incomplete strategy groups")

        shared = self.META + [
            "spot_change", "option_mark_change", "delta_difference"
        ]
        if groups[shared].nunique(dropna=False).gt(1).any().any():
            raise ValueError(
                "Strategies disagree on the contract or observed movement"
            )

        self.close(
            d.total_dividend_income, 0, "zero-dividend convention"
        )
        self.close(
            d.total_option_pnl,
            -d.option_mark_change,
            "short-option mark change",
        )
        self.close(
            d.total_hedge_pnl,
            d.entry_delta * d.spot_change,
            "held entry delta",
        )

        expected = (
            d.total_option_pnl
            + d.total_hedge_pnl
            + d.total_interest
            - d.direct_costs
        )
        self.close(d.net_pnl, expected, "strategy P&L components")
        self.close(
            d.loc[d.strategy.eq("unhedged"), "entry_delta"],
            0,
            "unhedged delta",
        )
        self.close(
            d.loc[d.scenario.eq("mid"), "direct_costs"],
            0,
            "midpoint costs",
        )

        for frame, label in (
            (paired, "paired results"), (coverage, "coverage")
        ):
            if (
                frame.columns.has_duplicates
                or not set(self.KEYS).issubset(frame)
                or frame.duplicated(self.KEYS).any()
            ):
                raise ValueError(f"Invalid keys in {label}")

        pair_fields = [
            "ah_net_pnl", "black_net_pnl", "ah_minus_black_net_pnl",
            "abs_error_improvement", "squared_error_improvement",
        ]
        if (
            not set(pair_fields).issubset(paired)
            or not np.isfinite(
                paired[pair_fields].to_numpy(float)
            ).all()
        ):
            raise ValueError("Missing or nonfinite saved paired metrics")

        keys = pd.MultiIndex.from_frame(
            d[self.KEYS].drop_duplicates()
        )
        if set(keys) != set(
            pd.MultiIndex.from_frame(paired[self.KEYS])
        ):
            raise ValueError("Paired results and strategy keys disagree")

        if "status" not in coverage:
            raise ValueError("Missing coverage status")

        covered = pd.MultiIndex.from_frame(
            coverage.loc[coverage.status.eq("compared"), self.KEYS]
        )
        if set(keys) != set(covered):
            raise ValueError("Compared coverage and strategy keys disagree")

        baseline_ids = set(
            d.loc[d.scenario.eq("mid"), "entry_id"]
        )
        if any(
            set(g.entry_id) != baseline_ids
            for _, g in d.groupby("scenario")
        ):
            raise ValueError("Scenarios use different comparison samples")

        shared = self.META + ["spot_change", "option_mark_change"]
        if d.groupby("entry_id")[shared].nunique().gt(1).any().any():
            raise ValueError(
                "Scenarios disagree on contracts or observed movements"
            )

        return d

    @staticmethod
    def metrics(d):
        a = d.ah_net_pnl.to_numpy()
        b = d.black_net_pnl.to_numpy()
        mse_a, mse_b = np.mean(a * a), np.mean(b * b)

        return {
            "comparisons": len(d),
            "entry_dates": d.quote_date.nunique(),
            "ah_mean_pnl": np.mean(a),
            "black_mean_pnl": np.mean(b),
            "ah_rms": np.sqrt(mse_a),
            "black_rms": np.sqrt(mse_b),
            "ah_mae": np.mean(np.abs(a)),
            "black_mae": np.mean(np.abs(b)),
            "mae_improvement": np.mean(np.abs(b) - np.abs(a)),
            "mse_improvement": mse_b - mse_a,
            "variance_improvement": np.var(b) - np.var(a),
            "squared_mean_improvement": np.mean(b)**2 - np.mean(a)**2,
            "fraction_ah_lower_abs_error": np.mean(
                np.abs(b) - np.abs(a) > 1e-10
            ),
            "fraction_abs_error_ties": np.mean(
                np.abs(np.abs(b) - np.abs(a)) <= 1e-10
            ),
            "mean_delta_difference": d.delta_difference.mean(),
            "mean_pnl_gap": d.pnl_gap.mean(),
            "mean_delta_exposure": d.delta_exposure.mean(),
            "mean_funding_difference": d.funding_difference.mean(),
            "mean_cost_effect": d.cost_effect.mean(),
            "max_identity_residual": d.identity_residual.abs().max(),
        }

    def grouped(self, d, keys):
        rows = []
        for key, group in d.groupby(
            keys, sort=True, dropna=False
        ):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({
                **dict(zip(keys, key)),
                **self.metrics(group),
            })
        return pd.DataFrame(rows)

    def run(self, trials, paired, coverage):
        t = self.checked(trials, paired, coverage)

        a = t.loc[t.strategy.eq("ah")].drop(columns="strategy")
        b = t.loc[
            t.strategy.eq("black"), self.KEYS + self.NUMBERS
        ]
        d = a.merge(
            b,
            on=self.KEYS,
            suffixes=("_ah", "_black"),
            validate="one_to_one",
        )

        d["ah_net_pnl"] = d.net_pnl_ah
        d["black_net_pnl"] = d.net_pnl_black
        d["delta_difference"] = (
            d.entry_delta_ah - d.entry_delta_black
        )
        self.close(
            d.delta_difference,
            d.delta_difference_ah,
            "saved delta difference",
        )

        d["spot_change"] = d.spot_change_ah
        d["entry_spot"] = d.strike * np.exp(-d.entry_spot_y)
        d["spot_return"] = d.spot_change / d.entry_spot

        if (
            not np.isfinite(d.entry_spot).all()
            or (d.entry_spot <= 0).any()
        ):
            raise ValueError(
                "Invalid spot reconstructed from log(K/spot)"
            )

        for _, group in d.groupby("quote_date"):
            self.close(
                group.entry_spot,
                group.entry_spot.iloc[0],
                "same-date index level",
            )
            self.close(
                group.spot_change,
                group.spot_change.iloc[0],
                "same-date index movement",
            )

        d["pnl_gap"] = d.ah_net_pnl - d.black_net_pnl
        d["delta_exposure"] = (
            d.delta_difference * d.spot_change
        )
        d["funding_difference"] = (
            d.total_interest_ah - d.total_interest_black
        )
        d["cost_effect"] = (
            d.direct_costs_black - d.direct_costs_ah
        )
        d["option_difference"] = (
            d.total_option_pnl_ah - d.total_option_pnl_black
        )
        d["identity_residual"] = (
            d.pnl_gap
            - d.delta_exposure
            - d.funding_difference
            - d.cost_effect
        )

        self.close(
            d.option_difference, 0, "common observed option P&L"
        )
        self.close(d.identity_residual, 0, "paired attribution")

        d["abs_error_improvement"] = (
            d.black_net_pnl.abs() - d.ah_net_pnl.abs()
        )
        d["squared_error_improvement"] = (
            d.black_net_pnl**2 - d.ah_net_pnl**2
        )

        saved = paired[self.KEYS + [
            "ah_net_pnl", "black_net_pnl",
            "ah_minus_black_net_pnl", "abs_error_improvement",
            "squared_error_improvement",
        ]]
        check = d.merge(
            saved,
            on=self.KEYS,
            suffixes=("", "_saved"),
            validate="one_to_one",
        )

        for name in (
            "ah_net_pnl", "black_net_pnl",
            "abs_error_improvement", "squared_error_improvement",
        ):
            self.close(
                check[name],
                check[name + "_saved"],
                f"saved {name}",
            )
        self.close(
            check.pnl_gap,
            check.ah_minus_black_net_pnl,
            "saved P&L gap",
        )

        d["spot_direction"] = np.select(
            [d.spot_change < -1e-8, d.spot_change > 1e-8],
            ["falling", "rising"],
            default="unchanged",
        )
        d["moneyness_band"] = np.select(
            [d.entry_spot_y < -0.01, d.entry_spot_y > 0.01],
            ["K_below_spot", "K_above_spot"],
            default="near_spot",
        )
        d["maturity_band"] = np.select(
            [
                d.calendar_days.between(14, 27),
                d.calendar_days.between(28, 39),
                d.calendar_days.between(40, 45),
            ],
            ["14-27d", "28-39d", "40-45d"],
            default="outside_pilot",
        )

        summary = self.grouped(d, ["scenario"])
        daily = self.grouped(d, ["scenario", "quote_date"])

        dates = d.groupby(["scenario", "quote_date"]).agg(
            spot_change=("spot_change", "first"),
            spot_return=("spot_return", "first"),
            spot_direction=("spot_direction", "first"),
        )
        daily = daily.merge(
            dates.reset_index(),
            on=["scenario", "quote_date"],
            validate="one_to_one",
        )

        counts = d.groupby("scenario").size()
        daily["pooled_mae_contribution"] = (
            daily.mae_improvement
            * daily.comparisons
            / daily.scenario.map(counts)
        )
        daily["pooled_mse_contribution"] = (
            daily.mse_improvement
            * daily.comparisons
            / daily.scenario.map(counts)
        )

        groups = []
        for dimension in (
            "kind", "maturity_band",
            "moneyness_band", "spot_direction",
        ):
            g = self.grouped(
                d, ["scenario", dimension]
            ).rename(columns={dimension: "group"})
            g.insert(1, "dimension", dimension)
            groups.append(g)

        loo = []
        for case, group in d.groupby("scenario"):
            full = self.metrics(group)
            if group.quote_date.nunique() < 2:
                continue

            for date in sorted(group.quote_date.unique()):
                m = self.metrics(
                    group.loc[group.quote_date.ne(date)]
                )
                loo.append({
                    "scenario": case,
                    "excluded_date": date,
                    **m,
                    "full_mae_improvement": full["mae_improvement"],
                    "full_mse_improvement": full["mse_improvement"],
                })

        shifts = self.cost_shifts(d)

        return {
            "attribution": d,
            "summary": summary,
            "daily_attribution": daily,
            "group_summary": pd.concat(groups, ignore_index=True),
            "leave_one_date_out": pd.DataFrame(
                loo,
                columns=[
                    "scenario", "excluded_date"
                ] + list(self.metrics(d)) + [
                    "full_mae_improvement", "full_mse_improvement"
                ],
            ),
            "cost_shifts": shifts,
            "coverage": coverage.copy(deep=True),
        }

    def cost_shifts(self, d):
        fields = [
            "ah_net_pnl", "black_net_pnl",
            "entry_delta_ah", "entry_delta_black", "spot_change",
            "total_interest_ah", "total_interest_black",
            "direct_costs_ah", "direct_costs_black",
            "abs_error_improvement",
        ]
        mid = d.loc[
            d.scenario.eq("mid"), ["entry_id"] + fields
        ]
        out = d.loc[d.scenario.ne("mid")].merge(
            mid,
            on="entry_id",
            suffixes=("", "_mid"),
            validate="many_to_one",
        )

        for name in (
            "entry_delta_ah", "entry_delta_black", "spot_change"
        ):
            self.close(
                out[name],
                out[name + "_mid"],
                f"unchanged scenario {name}",
            )

        for strategy in ("ah", "black"):
            out[strategy + "_pnl_shift"] = (
                out[strategy + "_net_pnl"]
                - out[strategy + "_net_pnl_mid"]
            )
            out[strategy + "_funding_shift"] = (
                out["total_interest_" + strategy]
                - out["total_interest_" + strategy + "_mid"]
            )
            out[strategy + "_cost_effect"] = (
                out["direct_costs_" + strategy + "_mid"]
                - out["direct_costs_" + strategy]
            )

            self.close(
                out[strategy + "_pnl_shift"],
                out[strategy + "_funding_shift"]
                + out[strategy + "_cost_effect"],
                "cost and funding shift",
            )

        out["abs_improvement_change"] = (
            out.abs_error_improvement
            - out.abs_error_improvement_mid
        )

        return out[[
            "scenario", "entry_id", "quote_date", "kind",
            "ah_pnl_shift", "black_pnl_shift",
            "ah_funding_shift", "black_funding_shift",
            "ah_cost_effect", "black_cost_effect",
            "abs_improvement_change",
        ]]