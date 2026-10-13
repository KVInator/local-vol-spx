"""Aggregate saved sufficient statistics and present completed historical results."""
from __future__ import annotations
import numpy as np
import pandas as pd
from results import StudyResults


class HistoricalAnalysis:
    """Keep full-scope bootstrap estimates separate from descriptive date ranges."""

    def __init__(self, saved: StudyResults):
        if saved.index.get("status") != "execution_complete":
            raise ValueError("Historical analysis requires a completed study.")
        self.saved = saved
        self.pairwise = saved.table("gain_summary")
        self.common = saved.table("common_strategy_gain")
        self.coverage = saved.table("coverage_summary")
        self._daily = None
        self.widget = None
        self.current = None
        self.figures = []

    @staticmethod
    def aggregate_daily(daily: pd.DataFrame) -> pd.DataFrame:
        """Combine counts and SSE before Gain; no daily-percentage averaging."""
        keys = ["strategy", "dimension", "group"]
        if daily.empty:
            return pd.DataFrame(columns=keys + ["comparisons", "entry_dates", "gain"])
        groups = daily.groupby(keys, sort=True, observed=True)
        result = groups.agg(
            comparisons=("rows", "sum"),
            entry_dates=("quote_date", "nunique"),
            strategy_sse=("strategy_sse", "sum"),
            black_sse=("black_sse", "sum"),
            strategy_sum=("strategy_sum", "sum"),
            black_sum=("black_sum", "sum"),
            strategy_absolute=("strategy_absolute", "sum"),
            black_absolute=("black_absolute", "sum"),
        ).reset_index()
        ready = result.black_sse > 1e-20
        result["gain"] = np.nan
        result.loc[ready, "gain"] = (
            1 - result.loc[ready, "strategy_sse"] / result.loc[ready, "black_sse"]
        )
        result["gain_status"] = np.where(ready, "ready", "zero_black_sse")
        result["strategy_rms"] = np.sqrt(result.strategy_sse / result.comparisons)
        result["black_rms"] = np.sqrt(result.black_sse / result.comparisons)
        result["strategy_mean"] = result.strategy_sum / result.comparisons
        result["black_mean"] = result.black_sum / result.comparisons
        result["sample_scope"] = "descriptive_candidate_Black_pairs"
        result["ci_status"] = "not_estimated_for_date_range"
        return result

    @property
    def daily(self) -> pd.DataFrame:
        if self._daily is None:
            self._daily = self.saved.table("daily_gain_statistics")
        return self._daily

    def select(
        self,
        holding: int = 5,
        rebalance: int = 1,
        scenario: str = "mid",
        sample: str = "pairwise",
        block: int = 20,
        dimension: str = "overall",
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        if sample not in ("pairwise", "common"):
            raise ValueError("Sample must be pairwise or common.")
        if (start is None) != (end is None) or (start and start > end):
            raise ValueError("Provide both range endpoints in date order.")
        if start is not None and sample != "pairwise":
            raise ValueError(
                "Saved common-entry intervals support their full recorded scope only."
            )
        table = self.pairwise if sample == "pairwise" else self.common
        if dimension not in set(table.dimension) or block not in set(
            table.mean_block_sessions
        ):
            raise ValueError("Use a saved grouping and bootstrap block length.")
        schedule = (holding, rebalance, scenario)
        if schedule not in set(
            zip(table.holding_sessions, table.rebalance_sessions, table.scenario)
        ):
            raise ValueError("No saved schedule/scenario combination.")

        def select_schedule(frame):
            return frame.loc[
                frame.holding_sessions.eq(holding)
                & frame.rebalance_sessions.eq(rebalance)
                & frame.scenario.eq(scenario)
            ]

        if start is not None:
            daily = select_schedule(self.daily)
            result = self.aggregate_daily(
                daily.loc[
                    daily.dimension.eq(dimension) & daily.quote_date.between(start, end)
                ]
            )
        else:
            result = select_schedule(table)
            result = result.loc[
                result.mean_block_sessions.eq(block) & result.dimension.eq(dimension)
            ].copy()
        coverage = (
            select_schedule(self.coverage).copy() if start is None else pd.DataFrame()
        )
        # Range exclusions are not stored as daily sufficient statistics.
        self.current = {
            "results": result,
            "coverage": coverage,
            "range": (start, end),
            "holding": holding,
            "rebalance": rebalance,
            "scenario": scenario,
            "sample": sample,
            "block": block,
            "dimension": dimension,
        }
        return self.current

    def tail_contributions(
        self, holding=5, rebalance=1, scenario="mid", start=None, end=None
    ):
        daily = self.daily
        selected = daily.loc[
            daily.holding_sessions.eq(holding)
            & daily.rebalance_sessions.eq(rebalance)
            & daily.scenario.eq(scenario)
            & daily.dimension.eq("overall")
        ].copy()
        if start is not None:
            selected = selected.loc[selected.quote_date.between(start, end)]
        selected["excess_sse"] = selected.strategy_sse - selected.black_sse
        selected["candidate_sse_share"] = selected.strategy_sse / selected.groupby(
            "strategy"
        ).strategy_sse.transform("sum")
        return (
            selected.sort_values("excess_sse", ascending=False)
            .groupby("strategy", sort=True)
            .head(3)
        )

    def build_widget(self):
        import ipywidgets as widgets

        schedules = sorted(
            set(zip(self.pairwise.holding_sessions, self.pairwise.rebalance_sessions))
        )
        self.schedule = widgets.Dropdown(
            options=[
                (f"{h} holding / {r} rebalance sessions", (h, r)) for h, r in schedules
            ],
            value=(5, 1) if (5, 1) in schedules else schedules[0],
            description="Schedule",
        )
        self.scenario = widgets.Dropdown(
            options=sorted(self.pairwise.scenario.unique()),
            value="mid",
            description="Execution",
        )
        self.sample = widgets.Dropdown(
            options=["pairwise", "common"], description="Sample"
        )
        blocks = sorted(self.pairwise.mean_block_sessions.unique())
        self.block = widgets.Dropdown(
            options=blocks,
            value=20 if 20 in blocks else blocks[0],
            description="Mean block",
        )
        self.grouping = widgets.Dropdown(
            options=sorted(self.pairwise.dimension.unique()),
            value="overall",
            description="Grouping",
        )
        self.show_unhedged = widgets.Checkbox(
            value=False, description="Include unhedged in chart"
        )
        self.range_enabled = widgets.Checkbox(
            value=False, description="Descriptive entry-date range"
        )
        from datetime import date

        self.start = widgets.DatePicker(
            description="From", value=date.fromisoformat(self.saved.dates[0])
        )
        self.end = widgets.DatePicker(
            description="To", value=date.fromisoformat(self.saved.dates[-1])
        )
        self.status = widgets.HTML()
        self.output = widgets.Output()
        self.widget = widgets.VBox(
            [
                widgets.HBox([self.schedule, self.scenario]),
                widgets.HBox(
                    [self.sample, self.block, self.grouping, self.show_unhedged]
                ),
                widgets.HBox([self.range_enabled, self.start, self.end]),
                self.status,
                self.output,
            ]
        )
        for control in (
            self.schedule,
            self.scenario,
            self.sample,
            self.block,
            self.grouping,
            self.range_enabled,
            self.start,
            self.end,
            self.show_unhedged,
        ):
            control.observe(self._refresh, names="value")
        self.start.disabled = self.end.disabled = True
        return self.widget

    def _refresh(self, change=None):
        from html import escape
        import matplotlib.pyplot as plt

        self.current = None
        for figure in self.figures:
            plt.close(figure)
        self.figures = []
        self.output.outputs = ()
        active = self.range_enabled.value
        self.start.disabled = self.end.disabled = not active
        self.block.disabled = active
        try:
            if active and (self.start.value is None or self.end.value is None):
                raise ValueError("Choose both range endpoints.")
            self.select(
                *self.schedule.value,
                scenario=self.scenario.value,
                sample=self.sample.value,
                block=self.block.value,
                dimension=self.grouping.value,
                start=self.start.value.isoformat() if active else None,
                end=self.end.value.isoformat() if active else None,
            )
            self.render()
        except (ValueError, OSError) as error:
            self.status.value = f"<b>{escape(str(error))}</b>"

    def _show(self, *objects):
        """Append explicitly owned output without frontend capture/reply traffic."""
        for obj in objects:
            self.output.append_display_data(obj)

    def render(self):
        from IPython.display import Markdown
        import matplotlib.pyplot as plt

        if self.current is None:
            return
        state = self.current
        data = state["results"]
        ranged = state["range"][0] is not None
        note = (
            "Descriptive pairwise date range; no confidence intervals or range coverage inferred."
            if ranged
            else f"Full recorded {state['sample']} sample; saved {state['block']:g}-session mean-block intervals."
        )
        self.status.value = f"<b>{note}</b>"
        with self.output.hold_sync():
            columns = [
                n
                for n in (
                    "strategy",
                    "group",
                    "comparisons",
                    "entry_dates",
                    "gain",
                    "gain_status",
                    "gain_ci_low",
                    "gain_ci_high",
                    "ci_status",
                    "strategy_rms",
                    "black_rms",
                    "strategy_mean",
                    "black_mean",
                )
                if n in data
            ]
            self._show(
                Markdown(
                    "**Gain is a fraction: 1 − candidate SSE / matched Black SSE. Errors are funded, around zero.**"
                ),
                data[columns],
            )
            plotted = (
                data
                if self.show_unhedged.value
                else data.loc[data.strategy.ne("unhedged")]
            )
            if not plotted.empty:
                fig, ax = plt.subplots(
                    figsize=(10, max(4, 0.22 * len(plotted))), constrained_layout=True
                )
                labels = (
                    plotted.strategy.astype(str) + " / " + plotted.group.astype(str)
                )
                positions = np.arange(len(plotted))
                ax.scatter(100 * plotted.gain, positions)
                if not ranged:
                    for i, row in enumerate(plotted.itertuples()):
                        if row.ci_status == "ready":
                            ax.plot(
                                100 * np.array([row.gain_ci_low, row.gain_ci_high]),
                                [i, i],
                                color="tab:blue",
                            )
                ax.set_yticks(positions, labels)
                ax.axvline(0, color="black", linewidth=0.7)
                ax.set(
                    xlabel="Gain (%)",
                    title="Saved paired squared funded errors"
                    + ("" if self.show_unhedged.value else ": hedged methods"),
                )
                self._show(fig)
                self.figures.append(fig)
                plt.close(fig)
            if not state["coverage"].empty:
                self._show(
                    Markdown(
                        "**Coverage and incomplete-path reasons (full selected-entry population)**"
                    ),
                    state["coverage"],
                )
            tail = self.tail_contributions(
                state["holding"], state["rebalance"], state["scenario"], *state["range"]
            )
            self._show(
                Markdown(
                    "**Largest entry-date excess SSE contributions; pairwise sample**"
                ),
                tail[
                    [
                        "strategy",
                        "quote_date",
                        "rows",
                        "strategy_sse",
                        "black_sse",
                        "excess_sse",
                        "candidate_sse_share",
                    ]
                ],
            )

    def display(self):
        from IPython.display import display

        if self.widget is None:
            self.build_widget()
        display(self.widget)
        self._refresh()
