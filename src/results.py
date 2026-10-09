"""Read saved studies and draw their figures without rerunning the study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import PinnedInputs, sha256
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
        return self.read_pinned_csv(
            self.summary / f"{name}.csv",
            audit["output_sha256"].get(f"{name}.csv"),
            DTYPES,
        )

    def dated_folder(self, date):
        # Relative paths also allow a saved study to be copied to another machine.
        return self.folder / "cache" / "dates" / date / "decisions"

    def dated_table(self, date, name):
        if date not in self.index["dates"]:
            raise KeyError(f"No saved decisions for {date}.")
        folder = self.dated_folder(date)
        audit = self.read_json(folder / "cache.json")
        return self.read_pinned_csv(
            folder / f"{name}.csv", audit["output_sha256"].get(f"{name}.csv"), DTYPES
        )

    def carry(self, date):
        folder = self.folder / "cache" / "months" / date[:7] / "carry"
        audit = self.read_json(folder / "cache.json")
        data = self.read_pinned_csv(
            folder / "primary_carry.csv",
            audit["output_sha256"].get("primary_carry.csv"),
            DTYPES,
        )
        return data.loc[data.quote_date.eq(date)].copy()

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
        final_summary = (
            self.index.get("status") == "execution_complete"
            and audit["processed_reference_dates"] == len(self.dates)
        )
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
        figure_note = (
            "The figures use five-session holdings with daily rebalancing and midpoint marks."
        )
        if default_schedule.empty:
            figure_note += " This schedule has no matched Gain estimates in the saved summary."
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


from artifacts import require


class PilotResults(PinnedInputs):
    """Resolve an explicit run index and verify every consumed producer file."""

    FILES = {
        "prepare": ("pilot_quotes",),
        "carry": ("primary_carry", "calibration_quotes"),
        "calibration": ("model_manifest", "quote_residuals", "expiry_summary"),
        "panel": ("delta_panel", "selection_buckets"),
        "comparison": ("paired_results", "coverage"),
        "attribution": ("attribution", "daily_attribution", "group_summary"),
    }

    def __init__(self, index_file, month, repository=None):
        super().__init__(index_file, repository)
        self.month = month
        self.audits, self.folders, self.frames = ({}, {}, {})
        self.index = self.read_json(self.index_file)
        if month not in self.index.get("months", {}):
            raise ValueError(f"Month {month} is absent from the explicit run index")
        self.record = self.index["months"][month]

    def read_table(self, folder, audit, name, legacy_input_pins=()):
        path = (folder / f"{name}.csv").resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Producer file escapes its run folder")
        expected = audit.get("output_sha256", {}).get(f"{name}.csv")
        return self.read_pinned_csv(path, expected, DTYPES, legacy_input_pins)

    def check_link(self, consumer, producer, name):
        path = self.folders[producer] / name
        digest = self.inputs[str(path.resolve())]
        if digest not in self.audits[consumer].get("input_sha256", {}).values():
            raise ValueError(f"{consumer} does not pin {producer}/{name}")

    def load(self):
        for stage, names in self.FILES.items():
            if stage not in self.record:
                raise ValueError(f"Run index is missing stage {stage}")
            folder = self.resolve(self.record[stage])
            audit = self.read_json(folder / "audit.json")
            self.folders[stage], self.audits[stage] = (folder, audit)
        for stage, names in self.FILES.items():
            folder, audit = (self.folders[stage], self.audits[stage])
            for name in names:
                pins = (
                    self.audits["carry"].get("input_sha256", {}).values()
                    if stage == "prepare"
                    else ()
                )
                self.frames[name] = self.read_table(folder, audit, name, pins)
        for consumer, producer, names in (
            ("carry", "prepare", ["pilot_quotes.csv"]),
            ("calibration", "carry", ["calibration_quotes.csv", "primary_carry.csv"]),
            ("panel", "calibration", ["audit.json", "model_manifest.csv"]),
            ("panel", "carry", ["primary_carry.csv"]),
            ("comparison", "panel", ["delta_panel.csv"]),
            ("attribution", "comparison", ["paired_results.csv", "coverage.csv"]),
        ):
            for name in names:
                self.check_link(consumer, producer, name)
        if self.audits["panel"].get("marks_replaced_by_model_prices") is not False:
            raise ValueError("This audit requires observed hedge-panel marks")
        if (
            self.audits["panel"].get("entries_filtered_by_future_availability")
            is not False
        ):
            raise ValueError("Unexpected future-availability entry screening")
        self.load_validation()
        self.verify_unchanged()
        return self

    def load_validation(self):
        requested = set(self.record.get("validation_scope", []))
        index = self.index_file.parent / "stage_index.csv"
        validation = {}
        if index.exists():
            self.inputs[str(index)] = sha256(index)
            stages = pd.read_csv(index, dtype=str).fillna("")
            require(stages, ["job", "status", "folder"], "stage index", ["job"])
            prefix = f"{self.month}/validation/"
            for row in stages.loc[stages.job.str.startswith(prefix)].to_dict("records"):
                date = row["job"].removeprefix(prefix)
                validation[date] = row
        pieces = {
            name: []
            for name in [
                "quote_fit",
                "sensitivity",
                "forward_shapes",
                "forward_backward",
            ]
        }
        scope = []
        dates = sorted(
            set(self.frames["model_manifest"].quote_date) | requested | set(validation)
        )
        for date in dates:
            row = validation.get(date, {})
            status = row.get(
                "status",
                "requested_job_missing" if date in requested else "not_requested",
            )
            item = dict(
                quote_date=date,
                requested=date in requested,
                stage_status=status,
                diagnostics_loaded=False,
                all_entry_greeks_independently_validated=False,
            )
            if status in {"completed", "completed_with_failures"} and row.get("folder"):
                folder = self.resolve(row["folder"])
                audit = self.read_json(folder / "audit.json")
                if (
                    self.inputs[str(self.folders["calibration"] / "audit.json")]
                    not in audit.get("input_sha256", {}).values()
                ):
                    raise ValueError(
                        f"Validation {date} does not pin this calibration run"
                    )
                for name in pieces:
                    frame = self.read_table(folder, audit, name)
                    if "quote_date" not in frame or not frame.quote_date.eq(date).all():
                        raise ValueError(f"Validation job/date mismatch: {date}/{name}")
                    pieces[name].append(frame)
                item["diagnostics_loaded"] = True
            scope.append(item)
        self.frames["validation_coverage"] = pd.DataFrame(scope)
        for name, parts in pieces.items():
            self.frames[f"validation_{name}"] = (
                pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            )
