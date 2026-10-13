"""Read saved studies and draw their figures without rerunning the study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import ArtifactCache, PinnedInputs, sha256
from surface import AndreasenHugeSurface
from diagnostics import SurfaceEvidence, SurfaceEvidenceSettings

DTYPES = {
    name: str
    for name in (
        "quote_date",
        "root",
        "expire_date",
        "entry_id",
        "contract_id",
        "kind",
        "case",
    )
}


class StudyResults(PinnedInputs):
    """Inspect either the original historical run or a new study."""

    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()
        super().__init__(self.folder / "run_index.json")
        self.index = self.read_json(self.index_file)
        self.freeze = self.read_json(self.folder / "freeze.json")
        if self.index["frozen_fingerprint"] != self.freeze["fingerprint"]:
            raise ValueError("The index and freeze belong to different studies.")
        self.folders = {"study": self.folder}
        self.summary = self.folder / "summary"

    @property
    def dates(self):
        return tuple(sorted(self.index["dates"]))

    @property
    def summary_ready(self):
        return (self.summary / "audit.json").exists()

    def table(self, name):
        audit = self.read_json(self.summary / "audit.json")
        path = (self.summary / f"{name}.csv").resolve()
        if not path.is_relative_to(self.summary):
            raise ValueError("Summary table lies outside its folder.")
        return self.read_pinned_csv(
            path,
            audit["output_sha256"].get(f"{name}.csv"),
            DTYPES,
        )

    def dated_folder(self, date):
        # Relative paths also allow a saved study to be copied to another machine.
        return self.folder / "cache" / "dates" / date / "decisions"

    def dated_table(self, date, name):
        if date not in self.index["dates"]:
            raise KeyError(f"No saved decisions for {date}.")
        return self.cache_table(self.dated_folder(date), name)

    def carry(self, date):
        folder = self.folder / "cache" / "months" / date[:7] / "carry"
        audit = self.read_json(folder / "cache.json")
        data = self.read_pinned_csv(
            folder / "primary_carry.csv",
            audit["output_sha256"].get("primary_carry.csv"),
            DTYPES,
        )
        return data.loc[data.quote_date.eq(date)].copy()

    def cache_table(self, folder, name):
        """Read a pinned table under this study, using paths relative to the cache."""
        folder = Path(folder).resolve()
        if not folder.is_relative_to(self.folder / "cache"):
            raise ValueError("Cache folder lies outside the saved study.")
        path = (folder / f"{name}.csv").resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Table lies outside its cache folder.")
        audit = self.read_json(folder / "cache.json")
        return self.read_pinned_csv(path, audit["output_sha256"].get(path.name), DTYPES)

    def selection(self, date):
        if date not in self.dates:
            raise KeyError(f"No saved selection for {date}.")
        return self.cache_table(
            self.folder / "cache/dates" / date / "selection", "entries"
        )

    def observations(self, date):
        data = self.cache_table(
            self.folder / "cache/months" / date[:7] / "prepare", "pilot_quotes"
        )
        return data.loc[data.quote_date.eq(date)].copy()

    def outcome_folder(self, date):
        """Relocate the producer's absolute index path without changing the index."""
        value = Path(self.index["dates"][date]["outcome_folder"])
        folder = self.folder / "cache/outcomes" / date / value.name
        if value.parent.name != date or not folder.is_dir():
            raise FileNotFoundError(f"Missing saved outcome cache: {folder}")
        return folder

    def outcomes(self, date, name="trials"):
        return self.cache_table(self.outcome_folder(date), name)

    def check_cache(self, folder):
        folder = Path(folder).resolve()
        if not folder.is_relative_to(self.folder / "cache"):
            raise ValueError("Cache folder lies outside the saved study.")
        return ArtifactCache.inspect(folder)

    def carry_exclusions(self):
        """Identify every recorded failed-model expiry without removing any pillar."""
        from market_data import boolean

        rows = []
        for date, record in sorted(self.index["dates"].items()):
            if not record.get("failed_models", 0):
                continue
            manifest = self.dated_table(date, "model_manifest")
            carry = self.carry(date)
            for failure in manifest.loc[manifest.status.ne("fitted")].to_dict(
                "records"
            ):
                bad = carry.loc[
                    carry.root.eq(failure["root"]) & ~boolean(carry.carry_ready)
                ]
                if bad.empty:
                    rows.append(
                        dict(
                            quote_date=date,
                            root=failure["root"],
                            expire_date="",
                            carry_status="no_unavailable_expiry_recorded",
                            model_message=failure.get("message", ""),
                        )
                    )
                else:
                    rows.extend(
                        dict(row, model_message=failure.get("message", ""))
                        for row in bad.to_dict("records")
                    )
        return pd.DataFrame(rows)

    def replay_tail_paths(
        self,
        dates,
        strategies=("ah_pde", "surface_sticky_delta"),
        holding=5,
        rebalance=1,
    ):
        """Replay each date/strategy's largest excess-SSE contract with the scalar ledger.

        The selection is diagnostic and uses outcomes. It never changes hedge entries,
        saved decisions, or the historical experiment. All scenarios are replayed.
        """
        from market_data import KEY
        from hedging import STRATEGIES
        from paths import FixedContractPaths, HistoricalSettings
        from ledger import (
            FundingScenarios,
            ComparisonSettings,
            SelfFinancingHedgeLedger,
        )

        frozen = self.freeze["identity"]
        calendar_file = self.folder / "calendar_support.csv"
        expected = next(
            (
                digest
                for path, digest in frozen["input_sha256"].items()
                if Path(path).name == calendar_file.name
            ),
            None,
        )
        calendar = self.read_pinned_csv(calendar_file, expected, DTYPES)
        settings = HistoricalSettings(**frozen["config"]["settings"])
        planner = FixedContractPaths(calendar, settings)
        scenarios = FundingScenarios(ComparisonSettings(**settings.funding)).cases
        quotes_by_month, decisions_by_date = {}, {}

        def quotes(date):
            month = date[:7]
            if month not in quotes_by_month:
                quotes_by_month[month] = self.cache_table(
                    self.folder / "cache/months" / month / "prepare", "pilot_quotes"
                )
            frame = quotes_by_month[month]
            return frame.loc[frame.quote_date.eq(date)]

        rows, intervals, exclusions = [], [], []
        for date in dates:
            trials = self.outcomes(date)
            coverage = self.outcomes(date, "coverage")
            selected_coverage = coverage.loc[
                coverage.holding_sessions.eq(holding)
                & coverage.rebalance_sessions.eq(rebalance)
                & coverage.scenario.eq("mid")
            ]
            exclusions.extend(
                dict(quote_date=date, strategy=s, status=status, entries=len(group))
                for (s, status), group in selected_coverage.groupby(
                    ["strategy", "status"]
                )
            )
            mid = trials.loc[
                trials.holding_sessions.eq(holding)
                & trials.rebalance_sessions.eq(rebalance)
                & trials.scenario.eq("mid")
            ]
            black = mid.loc[mid.strategy.eq("black"), ["entry_id", "net_pnl"]].rename(
                columns={"net_pnl": "black_net_pnl"}
            )
            entries = self.selection(date).set_index("entry_id")
            for strategy in strategies:
                paired = mid.loc[mid.strategy.eq(strategy)].merge(
                    black, on="entry_id", validate="one_to_one"
                )
                if paired.empty:
                    raise ValueError(
                        f"No saved {strategy}/Black paired path for {date}."
                    )
                paired["excess_sse"] = paired.net_pnl**2 - paired.black_net_pnl**2
                chosen = paired.loc[paired.excess_sse.idxmax()]
                entry = dict(entries.loc[chosen.entry_id], entry_id=chosen.entry_id)
                path_dates = planner.dates(date, holding)
                observations, decisions = {}, {}
                for current in path_dates:
                    for row in quotes(current).to_dict("records"):
                        observations[(current, *[row[k] for k in KEY])] = row
                    if current not in decisions_by_date:
                        decisions_by_date[current] = self.dated_table(
                            current, "decisions"
                        )
                    for row in decisions_by_date[current].to_dict("records"):
                        decisions[(current, *[row[k] for k in KEY])] = row
                meta, status, failure, path = planner.path(
                    entry, holding, rebalance, observations, decisions, set(self.dates)
                )
                if status != "ready":
                    raise ValueError(
                        f"Cached compared path is unavailable at {failure}: {status}"
                    )
                method_results = {}
                for method in ("black", strategy):
                    status_column, delta_column = STRATEGIES[method]
                    delta = 0.0
                    scalar = []
                    for j, observation in enumerate(path):
                        decision = observation["decision"] or {}
                        if observation["rebalance"]:
                            if decision.get(status_column) != "ready":
                                raise ValueError(
                                    f'Unavailable saved rebalance: {observation["quote_date"]}/{method}'
                                )
                            delta = float(decision[delta_column])
                        scalar.append(
                            dict(
                                timestamp=observation["quote_timestamp_utc"],
                                spot=observation["underlying_last"],
                                option_mark=observation["mid"],
                                option_bid=observation["bid"],
                                option_ask=observation["ask"],
                                hedge_bid=observation["underlying_last"],
                                hedge_ask=observation["underlying_last"],
                                delta=delta,
                                dividend_per_unit=0.0,
                                **{k: meta[k] for k in ("contract_id", *KEY)},
                            )
                        )
                    scalar = pd.DataFrame(scalar)
                    for scenario, ledger_settings in scenarios.items():
                        result = SelfFinancingHedgeLedger(ledger_settings).run(scalar)
                        cached = trials.loc[
                            trials.entry_id.eq(chosen.entry_id)
                            & trials.strategy.eq(method)
                            & trials.scenario.eq(scenario)
                            & trials.holding_sessions.eq(holding)
                            & trials.rebalance_sessions.eq(rebalance)
                        ]
                        if len(cached) != 1:
                            raise ValueError("No unique cached comparison outcome.")
                        errors = {
                            field: abs(
                                result.summary[field] - float(cached.iloc[0][field])
                            )
                            for field in (
                                "net_pnl",
                                "direct_costs",
                                "gross_pnl",
                                "total_interest",
                                "total_hedge_turnover",
                                "total_option_pnl",
                                "total_hedge_pnl",
                            )
                        }
                        if max(errors.values()) > 1e-8:
                            raise ArithmeticError(
                                f"Scalar/cache ledger mismatch: {date}/{method}/{scenario}: {errors}"
                            )
                        method_results[(method, scenario)] = (result, errors)
                candidate, _ = method_results[(strategy, "mid")]
                reference, _ = method_results[("black", "mid")]
                spot_changes = np.diff([p["underlying_last"] for p in path])
                delta_diff = (
                    candidate.ledger.delta.to_numpy()[:-1]
                    - reference.ledger.delta.to_numpy()[:-1]
                )
                contributions = delta_diff * spot_changes
                for j, amount in enumerate(contributions):
                    intervals.append(
                        dict(
                            quote_date=date,
                            entry_id=chosen.entry_id,
                            strategy=strategy,
                            interval_start=path_dates[j],
                            interval_end=path_dates[j + 1],
                            black_delta=float(reference.ledger.delta.iloc[j]),
                            candidate_delta=float(candidate.ledger.delta.iloc[j]),
                            spot_change=float(spot_changes[j]),
                            hedge_pnl_difference=float(amount),
                            option_mark_change=float(
                                path[j + 1]["mid"] - path[j]["mid"]
                            ),
                            decision_timestamp=path[j]["quote_timestamp_utc"],
                        )
                    )
                for scenario in scenarios:
                    c, ce = method_results[(strategy, scenario)]
                    b, be = method_results[("black", scenario)]
                    rows.append(
                        dict(
                            quote_date=date,
                            entry_id=chosen.entry_id,
                            strategy=strategy,
                            scenario=scenario,
                            expire_date=chosen.expire_date,
                            strike=chosen.strike,
                            kind=chosen.kind,
                            holding_sessions=holding,
                            rebalance_sessions=rebalance,
                            entry_delta_difference=float(delta_diff[0]),
                            entry_interval_hedge_pnl_difference=float(contributions[0]),
                            later_intervals_hedge_pnl_difference=float(
                                contributions[1:].sum()
                            ),
                            option_mark_pnl=c.summary["total_option_pnl"],
                            funding_difference=c.summary["total_interest"]
                            - b.summary["total_interest"],
                            cost_difference=c.summary["direct_costs"]
                            - b.summary["direct_costs"],
                            candidate_net_pnl=c.summary["net_pnl"],
                            black_net_pnl=b.summary["net_pnl"],
                            excess_sse=c.summary["net_pnl"] ** 2
                            - b.summary["net_pnl"] ** 2,
                            maximum_replay_difference=max(*ce.values(), *be.values()),
                        )
                    )
                    rows[-1].update(
                        date_comparisons=len(paired),
                        date_candidate_sse=float((paired.net_pnl**2).sum()),
                        date_black_sse=float((paired.black_net_pnl**2).sum()),
                    )
        return {
            "tail_ledger_summary": pd.DataFrame(rows),
            "tail_intervals": pd.DataFrame(intervals),
            "tail_coverage": pd.DataFrame(exclusions),
        }

    def model_record(self, date, root):
        manifest = self.dated_table(date, "model_manifest")
        rows = manifest.loc[manifest.root.eq(root) & manifest.status.eq("fitted")]
        if len(rows) != 1:
            raise ValueError(f"No unique fitted surface for {date}/{root}.")
        return rows.iloc[0]

    def model(self, date, root="UNKNOWN"):
        folder = self.dated_folder(date)
        path = (folder / self.model_record(date, root).model_file).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("The model file lies outside its decision folder.")
        audit = self.read_json(folder / "cache.json")
        if sha256(path) != audit["output_sha256"].get(path.name):
            raise ValueError(f"Model checksum mismatch: {path}")
        self.inputs[str(path)] = sha256(path)
        return AndreasenHugeSurface.load(path)

    def calibration_quotes(self, date, root="UNKNOWN"):
        prefix = self.model_record(date, root).model_file.removesuffix(
            "_ah_surface.json"
        )
        return self.dated_table(date, f"{prefix}_quote_residuals")

    def surface_evidence(self, date, root="UNKNOWN", settings=None):
        if settings is None:
            radius = self.freeze["identity"]["config"]["settings"]["panel"]["radius"]
            settings = SurfaceEvidenceSettings(radius=radius)
        model = self.model(date, root)
        carry = self.carry(date)
        carry = carry.loc[carry.root.eq(root)]
        return SurfaceEvidence(settings).run(
            model,
            self.calibration_quotes(date, root),
            carry,
            {"quote_date": date, "root": root},
        )

    def plot_surface(self, date, root="UNKNOWN"):
        import matplotlib.pyplot as plt

        samples = self.surface_evidence(date, root)["surface_samples"]
        samples = samples.drop_duplicates(["T", "y"]).sort_values(["T", "y"])
        fields = (
            ("ah_iv", "Implied volatility (%)", 100),
            ("ah_w", "Total variance", 1),
            ("ah_actual_lv", "Local volatility (%)", 100),
        )
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, (field, label, scale) in zip(axes, fields):
            table = samples.pivot(index="T", columns="y", values=field)
            mesh = ax.pcolormesh(
                table.columns, table.index * 365, scale * table, shading="nearest"
            )
            ax.set(xlabel="log(K / forward)", ylabel="Days to expiry", title=label)
            fig.colorbar(mesh, ax=ax)
        fig.suptitle(f"{date}: saved AH surface")
        return fig

    def plot_gains(self, holding=5, rebalance=1, scenario="mid", block=20):
        import matplotlib.pyplot as plt

        data = self.table("gain_summary")
        data = data.loc[
            data.holding_sessions.eq(holding)
            & data.rebalance_sessions.eq(rebalance)
            & data.scenario.eq(scenario)
            & data.dimension.eq("overall")
        ]
        if block not in set(data.mean_block_sessions):
            block = data.mean_block_sessions.min()
        data = data.loc[data.mean_block_sessions.eq(block)].sort_values("strategy")
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        if data.empty:
            ax.text(
                0.5,
                0.5,
                "No completed comparisons for this schedule",
                ha="center",
                transform=ax.transAxes,
            )
            return fig
        # A scatter with asymmetric intervals shows negative gains faithfully.
        ax.scatter(100 * data.gain, data.strategy, color="tab:blue")
        ready = data.loc[data.ci_status.eq("ready")]
        for row in ready.itertuples():
            ax.plot(
                [100 * row.gain_ci_low, 100 * row.gain_ci_high],
                [row.strategy, row.strategy],
                color="tab:blue",
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Gain against matched Black hedges (%)",
            title=f"{holding} sessions, rebalance every {rebalance}, {scenario}",
        )
        return fig

    def plot_coverage(self, holding=5, rebalance=1, scenario="mid"):
        import matplotlib.pyplot as plt

        data = self.table("coverage_summary")
        data = data.loc[
            data.holding_sessions.eq(holding)
            & data.rebalance_sessions.eq(rebalance)
            & data.scenario.eq(scenario)
        ]
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        if len(data):
            data.pivot_table(
                index="strategy",
                columns="status",
                values="entries",
                aggfunc="sum",
                fill_value=0,
            ).plot.bar(stacked=True, ax=ax)
        ax.set(ylabel="Selected entries", title="Completed and incomplete paths")
        ax.tick_params(axis="x", rotation=25)
        return fig

    def export(self, output):
        import matplotlib.pyplot as plt

        output = self.safe_output(output)
        output.mkdir(parents=True, exist_ok=True)
        if not self.summary_ready:
            raise ValueError("The runner has not written aggregate tables yet.")
        for filename, plot in (
            ("gain.png", self.plot_gains),
            ("coverage.png", self.plot_coverage),
        ):
            fig = plot()
            fig.savefig(output / filename, dpi=160)
            plt.close(fig)
        gain = self.table("gain_summary")
        gain.to_csv(output / "gain.csv", index=False)
        self.table("coverage_summary").to_csv(output / "coverage.csv", index=False)
        audit = self.read_json(self.summary / "audit.json")
        block = (
            20
            if 20 in set(gain.mean_block_sessions)
            else gain.mean_block_sessions.min()
        )
        selected = gain.loc[
            gain.dimension.eq("overall")
            & gain.scenario.eq("mid")
            & gain.mean_block_sessions.eq(block)
        ].copy()
        selected["gain_percent"] = (100 * selected.gain).round(2)
        selected["strategy"] = selected.strategy.replace(
            {
                "ah_pde": "AH PDE",
                "empirical_mv": "Empirical MV correction",
                "lv_smile": "LV smile approximation",
                "surface_sticky_strike": "Surface sticky strike",
                "surface_sticky_delta": "Surface sticky delta",
                "unhedged": "Unhedged",
            }
        )
        selected["gain_percent"] = selected.gain_percent.map(
            lambda value: f"{value:+.2f}%" if pd.notna(value) else "n/a"
        )
        selected["ci_status"] = selected.ci_status.replace(
            {
                "ready": "Available",
                "insufficient_dates": "Too few entry dates",
                "unstable_denominator": "Unstable Black error denominator",
            }
        )
        selected = selected[
            [
                "strategy",
                "holding_sessions",
                "rebalance_sessions",
                "comparisons",
                "gain_percent",
                "ci_status",
            ]
        ].rename(
            columns={
                "strategy": "Strategy",
                "holding_sessions": "Holding (sessions)",
                "rebalance_sessions": "Rebalance (sessions)",
                "comparisons": "Matched entries",
                "gain_percent": "Gain",
                "ci_status": "Confidence interval",
            }
        )
        table_lines = [
            "| " + " | ".join(selected.columns) + " |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        table_lines.extend(
            "| " + " | ".join(map(str, row)) + " |"
            for row in selected.itertuples(index=False, name=None)
        )
        final_summary = self.index.get("status") == "execution_complete" and audit[
            "processed_reference_dates"
        ] == len(self.dates)
        scope_note = (
            "The daily-model pass and hedge evaluation are complete for the saved study."
            if final_summary
            else "This is an interim summary. The daily-model pass and hedge evaluation "
            "can cover different numbers of dates while the run is in progress."
        )
        if audit["processed_reference_dates"] < len(self.dates):
            scope_note += (
                " Daily decisions extend beyond this hedge summary; the aggregate "
                "tables will be updated after the daily-model pass."
            )
        comparison = (
            "\n".join(table_lines)
            + f"\n\nConfidence intervals use entry-date blocks with a mean length of {block:g} sessions. "
            "The interval status records whether the saved sample supports an estimate."
            if len(selected)
            else "No completed matched Gain estimates are available in this saved summary."
        )
        default_schedule = gain.loc[
            gain.holding_sessions.eq(5)
            & gain.rebalance_sessions.eq(1)
            & gain.scenario.eq("mid")
            & gain.dimension.eq("overall")
        ]
        figure_note = "The figures use five-session holdings with daily rebalancing and midpoint marks."
        if default_schedule.empty:
            figure_note += (
                " This schedule has no matched Gain estimates in the saved summary."
            )
        (output / "results.md").write_text(
            "# Historical Hedge Results\n\n"
            "This report compares delta-hedging methods on the option entries in the saved study. "
            "Each candidate is measured against Black using the same matched entries.\n\n"
            "## Evaluation Scope\n\n"
            f"{scope_note}\n\n"
            "| Metric | Saved scope |\n| --- | ---: |\n"
            f"| Configured date range | {self.index['start']} to {self.index['end']} |\n"
            f"| Daily decisions saved | {len(self.dates):,} reference dates |\n"
            f"| Hedge-summary scope | {audit['processed_reference_dates']:,} reference dates |\n"
            f"| Selected option entries | {audit['selected_entries']:,} |\n"
            f"| Completed comparison paths | {audit['compared_paths']:,} |\n"
            f"| Pending paths | {audit['pending_paths']:,} |\n\n"
            "## Strategy Comparison\n\n"
            "The table uses midpoint option marks. Gain is one minus the ratio of candidate "
            "squared hedge errors to matched Black squared errors. A positive value means "
            "lower squared error; a negative value means higher squared error.\n\n"
            f"{comparison}\n\n"
            "Squared errors are measured around zero. Mean error and a few larger outcomes "
            "can therefore affect Gain differently from MAE. Matched samples can also differ "
            "across strategies, so coverage and the common-entry comparison matter when "
            "interpreting a ranking.\n\n"
            "## Visual Outputs\n\n"
            f"{figure_note}\n\n"
            "### Gain Relative to Black\n\n"
            "![Gain relative to matched Black hedges](gain.png)\n\n"
            "### Completed and Incomplete Paths\n\n"
            "![Historical hedge-path coverage](coverage.png)\n\n"
            "Coverage records the entries available to each method and the reasons for "
            "incomplete paths. `pending_run_prefix` means the saved evaluation ends before "
            "the path can finish. Other statuses distinguish missing observations, model "
            "failures and remaining exclusions.\n\n"
            "## Exported Tables\n\n"
            "- [Gain estimates](gain.csv): saved schedules, scenarios, buckets and uncertainty estimates\n"
            "- [Coverage](coverage.csv): entry counts by schedule, scenario, strategy and status\n\n"
            "The CSV files retain the saved column names and values. The reported P&L follows "
            "the study's pricing, funding and execution conventions, which are recorded in "
            "its frozen configuration.\n",
            encoding="utf-8",
        )
        return output


class SimulationResults:
    """Compare costs and hedge frequency on the same axes."""

    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()

    def table(self, name):
        return pd.read_csv(self.folder / f"{name}.csv")

    def plot_frequency(self):
        import matplotlib.pyplot as plt

        data = self.table("frequency_summary")
        models = list(data.model.unique())
        strikes = np.sort(data.strike.unique())
        strike = strikes[len(strikes) // 2]
        fig, axes = plt.subplots(
            1, len(models), figsize=(12, 4), squeeze=False, constrained_layout=True
        )
        for ax, model in zip(axes[0], models):
            selected = data.loc[
                data.model.eq(model)
                & data.strike.eq(strike)
                & data.kind.eq("call")
                & data.strategy.eq("analytical_delta")
            ]
            for scenario, group in selected.groupby("scenario"):
                group = group.sort_values("intervals")
                ax.plot(
                    group.intervals,
                    group.rms,
                    marker="o",
                    label=scenario.replace("_", " "),
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Rebalance intervals",
                ylabel="RMS hedge error",
                title=model,
            )
            ax.legend()
            ax.grid(alpha=0.2)
        return fig

    def plot_costs(self):
        import matplotlib.pyplot as plt

        data = self.table("frequency_summary")
        models = list(data.model.unique())
        strikes = np.sort(data.strike.unique())
        strike = strikes[len(strikes) // 2]
        fig, axes = plt.subplots(
            1, len(models), figsize=(12, 4), squeeze=False, constrained_layout=True
        )
        for ax, model in zip(axes[0], models):
            data_model = data.loc[
                data.model.eq(model)
                & data.strike.eq(strike)
                & data.kind.eq("call")
                & data.strategy.eq("analytical_delta")
                & data.scenario.eq("hedge_fee")
            ]
            ax.plot(
                data_model.intervals,
                data_model.mean_funded_costs,
                marker="o",
                label="Funded fees",
            )
            ax.plot(
                data_model.intervals,
                -data_model.pnl_mean,
                marker="s",
                label="Negative mean P&L",
            )
            ax.set(
                xscale="log",
                xlabel="Rebalance intervals",
                ylabel="Option price units",
                title=model,
            )
            ax.legend()
        return fig


class SavedValidationResults(PinnedInputs):
    """Read producer-pinned diagnostics only within their recorded date/root scope."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        super().__init__(self.folder / "audit.json")
        self.audit = self.read_json(self.index_file)
        self.audit_sha256 = self.inputs[str(self.index_file)]
        self.folders = {"validation": self.folder}

    def table(self, name, date, root="UNKNOWN"):
        if self.audit.get("date", date) != date or self.audit.get("root", root) != root:
            return pd.DataFrame()
        filename = f"{name}.csv"
        expected = self.audit.get("output_sha256", {}).get(filename)
        if expected is None:
            return pd.DataFrame()
        path = (self.folder / filename).resolve()
        if not path.is_relative_to(self.folder):
            raise ValueError("Validation table escapes its folder.")
        data = self.read_pinned_csv(path, expected, DTYPES)
        if "quote_date" in data:
            data = data.loc[data.quote_date.eq(date)]
        elif "date" in data:
            data = data.loc[data.date.eq(date)]
        elif self.audit.get("date") != date:
            return pd.DataFrame()
        if "root" in data:
            data = data.loc[data.root.eq(root)]
        return data.copy()

    def ssvi(self, date, root="UNKNOWN"):
        if self.audit.get("date") != date or self.audit.get("root") != root:
            return None
        from surface import SSVISurface

        path = self.folder / "ssvi_surface.json"
        expected = self.audit.get("output_sha256", {}).get(path.name)
        if expected is None:
            return None
        if sha256(path) != expected:
            raise ValueError(f"SSVI checksum mismatch: {path}")
        self.inputs[str(path)] = expected
        return SSVISurface.load(path)
