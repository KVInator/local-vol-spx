"""Synchronized, read-only notebook views of a saved daily model."""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date as calendar_date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import sha256
from diagnostics import SurfaceEvidenceSettings
from hedging import STRATEGIES
from results import StudyResults, SavedValidationResults
from surface import SSVISurface


@dataclass
class DailyView:
    date: str
    root: str
    status: str
    message: str
    tables: dict[str, pd.DataFrame]
    samples: pd.DataFrame
    ssvi: SSVISurface | None = None


class SurfaceExplorer:
    """Load one date at a time; all views share the same atomic selection."""

    CHECKS = (
        "greek_refinement",
        "blend_sensitivity",
        "vega_refinement",
        "saved_decision_refinement",
        "smile_replay_checks",
        "empirical_chronology_checks",
        "ssvi_quote_fit",
    )

    def __init__(self, saved: StudyResults, validation_folders=(), cache_size: int = 3):
        if not saved.dates or type(cache_size) is not int or cache_size < 1:
            raise ValueError("Require saved dates and a positive cache size.")
        self.saved = saved
        self.validation = []
        self.evidence_availability = []
        freeze_pin = sha256(saved.folder / "freeze.json")
        for folder in validation_folders:
            folder = Path(folder).resolve()
            if not folder.exists():
                self.evidence_availability.append(
                    dict(
                        producer=folder.name,
                        recorded_date="unknown",
                        status="unavailable_missing_folder",
                        reason=f"Optional evidence folder is absent: {folder}",
                    )
                )
                continue
            result = SavedValidationResults(folder)
            if freeze_pin not in result.audit.get("input_sha256", {}).values():
                raise ValueError(
                    f"Validation does not pin this study freeze: {result.folder}"
                )
            self.validation.append(result)
            self.evidence_availability.append(
                dict(
                    producer=folder.name,
                    recorded_date=result.audit.get("date", "row-level scope"),
                    status="available",
                    reason="Producer pins checked on consumption",
                )
            )
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.current = None
        self.requested_date: str | None = None
        self.figures = []
        self._updating = False
        self.widget = None

    def select(
        self, date: str, root: str = "UNKNOWN", half_width: float = 0.09
    ) -> DailyView:
        """A failed selection clears current state before any cache read."""
        self.current = None
        self.requested_date = date
        if date not in self.saved.dates:
            raise KeyError(f"No saved date: {date}")
        radius = self.saved.freeze["identity"]["config"]["settings"]["panel"]["radius"]
        settings = SurfaceEvidenceSettings(
            half_width=half_width, y_points=81, radius=radius
        )
        key = (date, root, half_width)
        if key in self.cache:
            view, pins = self.cache.pop(key)
            for path, digest in pins.items():
                if sha256(path) != digest:
                    raise ValueError(f"Cached explorer input changed: {path}")
            self.cache[key] = (view, pins)
            self.current = view
            return view
        # New selections use a fresh pin set, bounding checks to the selected day.
        day = StudyResults(self.saved.folder)
        tables = {
            "manifest": day.dated_table(date, "model_manifest"),
            "carry": day.carry(date),
            "decisions": day.dated_table(date, "decisions"),
        }
        tables = {n: f.loc[f.root.eq(root)].copy() for n, f in tables.items()}
        tables["evidence_availability"] = pd.DataFrame(
            self.evidence_availability,
            columns=["producer", "recorded_date", "status", "reason"],
        )
        manifest = tables["manifest"]
        fitted = manifest.loc[manifest.status.eq("fitted")]
        status, message, samples, ssvi = "ready", "", pd.DataFrame(), None
        if len(fitted) != 1:
            status = "model_unavailable"
            message = "; ".join(
                manifest.get("message", pd.Series(dtype=str)).dropna().astype(str)
            )
            message = message or f"No unique fitted model for {date}/{root}."
        else:
            record = fitted.iloc[0]
            prefix = record.model_file.removesuffix("_ah_surface.json")
            tables["quotes"] = day.calibration_quotes(date, root)
            tables["expiry_fit"] = day.dated_table(date, f"{prefix}_expiry_summary")
            # Reuse the core evidence object, including its derivative stability gaps.
            evidence = day.surface_evidence(date, root, settings)
            samples = evidence["surface_samples"]
            tables["observations"] = evidence["quote_inversions"]
        for reader in self.validation:
            if sha256(reader.index_file) != reader.audit_sha256:
                raise ValueError(
                    f"Validation producer audit changed: {reader.index_file}"
                )
            reader.inputs = {str(reader.index_file): reader.audit_sha256}
            for name in self.CHECKS:
                frame = reader.table(name, date, root)
                if not frame.empty:
                    tables["validation/" + name] = frame
            if reader.audit.get("method") == "ssvi":
                ssvi = reader.ssvi(date, root)
            day.inputs.update(reader.inputs)
        day.verify_unchanged()
        view = DailyView(date, root, status, message, tables, samples, ssvi)
        self.cache[key] = (view, dict(day.inputs))
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        self.current = view
        return view

    def close_figures(self):
        import matplotlib.pyplot as plt

        for figure in self.figures:
            plt.close(figure)
        self.figures = []

    def surface_figure(self, view, scale="linear"):
        import matplotlib.pyplot as plt

        if view.status != "ready":
            return None
        samples = view.samples.drop_duplicates(["T", "y"], keep="last")
        routes = [
            ("AH", ("ah_iv", "ah_w", "ah_actual_lv")),
            ("Quote interpolation", ("quote_iv", "quote_w", "quote_local_vol")),
        ]
        if view.ssvi is not None:
            routes.append(("Saved SSVI", ("ssvi_iv", "ssvi_w", "ssvi_lv")))
            samples = samples.copy()
            from diagnostics import dupire

            for time, indices in samples.groupby("T").groups.items():
                y = samples.loc[indices, "y"].to_numpy()
                state = view.ssvi.evaluate(y, float(time))
                values = dupire(y, state["w"], state["wy"], state["wyy"], state["wt"])
                samples.loc[indices, "ssvi_w"] = state["w"]
                samples.loc[indices, "ssvi_iv"] = np.sqrt(state["w"] / time)
                samples.loc[indices, "ssvi_lv"] = np.sqrt(values["variance"])
        fig = plt.figure(figsize=(15, 4 * len(routes)), constrained_layout=True)
        labels = (
            "Implied volatility (%)",
            "Total variance (year × vol²)",
            "Local volatility (%)",
        )
        for i, (route, fields) in enumerate(routes):
            for j, field in enumerate(fields):
                table = samples.pivot(index="T", columns="y", values=field)
                z = table.to_numpy() * (1 if j == 1 else 100)
                label = labels[j]
                if j == 2 and scale == "log":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        z = np.where(z > 0, np.log10(z), np.nan)
                    label = "log10(local volatility in %)"
                ax = fig.add_subplot(len(routes), 3, i * 3 + j + 1, projection="3d")
                x, t = np.meshgrid(table.columns, table.index * 365)
                ax.plot_surface(
                    x,
                    t,
                    z,
                    cmap="viridis",
                    linewidth=0,
                    rcount=len(t),
                    ccount=len(x[0]),
                )
                ax.set(
                    xlabel="log(K / forward)",
                    ylabel="Days to expiry",
                    zlabel=label,
                    title=f"{route}: {labels[j]}",
                )
                ax.view_init(elev=25, azim=-125)
        fig.suptitle(f"{view.date} / {view.root}; gaps retained, coefficients uncapped")
        return fig

    def slice_figure(self, view, expiry, scale="linear"):
        import matplotlib.pyplot as plt

        if view.status != "ready":
            return None
        carry = view.tables["carry"].set_index("expire_date")
        time = float(carry.loc[expiry, "maturity_years"])
        nearest = view.samples.loc[np.isclose(view.samples["T"], time)]
        sample = nearest.drop_duplicates("y", keep="last").sort_values("y")
        quotes = (
            view.tables["quotes"]
            .loc[lambda f: f.expire_date.eq(expiry)]
            .sort_values("strike")
        )
        obs = (
            view.tables["observations"]
            .loc[lambda f: f.expire_date.eq(expiry)]
            .sort_values("strike")
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
        for ax, fields, label, multiplier in (
            (axes[0, 0], ("ah_iv", "quote_iv"), "IV (%)", 100),
            (axes[0, 1], ("ah_w", "quote_w"), "Total variance", 1),
            (
                axes[0, 2],
                ("ah_actual_lv", "smoothed_actual_lv", "quote_local_vol"),
                "Local volatility (%)",
                100,
            ),
        ):
            for field in fields:
                ax.plot(
                    sample.y, sample[field] * multiplier, label=field.replace("_", " ")
                )
            if view.ssvi is not None:
                state = view.ssvi.evaluate(sample.y.to_numpy(), time)
                from diagnostics import dupire

                values = dupire(
                    sample.y, state["w"], state["wy"], state["wyy"], state["wt"]
                )
                curve = (
                    state["w"]
                    if label == "Total variance"
                    else np.sqrt(state["w"] / time)
                    if label == "IV (%)"
                    else np.sqrt(values["variance"])
                )
                ax.plot(sample.y, multiplier * curve, label="saved SSVI")
            ax.set(xlabel="log(K / forward)", ylabel=label)
            ax.legend(fontsize=8)
        axes[0, 0].scatter(
            obs.y, 100 * obs.observed_iv, s=8, color="black", label="observed IV"
        )
        axes[0, 0].legend(fontsize=8)
        if scale == "log":
            axes[0, 2].set_yscale("log", nonpositive="mask")
        axes[1, 0].errorbar(
            quotes.strike,
            quotes.call_mid,
            yerr=quotes.call_half_width,
            fmt=".",
            label="observed equivalent call ± half spread",
        )
        axes[1, 0].plot(quotes.strike, quotes.model_call_price, label="saved AH fit")
        axes[1, 0].set(
            xlabel="Strike (index points)", ylabel="Call price (index points)"
        )
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(quotes.strike, quotes.residual_half_spreads, ".")
        for level in (-1, 1):
            axes[1, 1].axhline(level, color="black", linewidth=0.7)
        axes[1, 1].set(xlabel="Strike", ylabel="Residual / original half spread")
        decision = view.tables["decisions"].loc[lambda f: f.expire_date.eq(expiry)]
        for strategy, (status, field) in STRATEGIES.items():
            if field in decision:
                for kind, group in decision.groupby("kind"):
                    group = group.sort_values("strike")
                    axes[1, 2].plot(
                        group.strike,
                        group[field].where(group[status].eq("ready")),
                        ".-",
                        label=f"{strategy} {kind}",
                        linewidth=0.8,
                    )
        axes[1, 2].set(
            xlabel="Strike", ylabel="Saved hedge delta (index units / option)"
        )
        if axes[1, 2].get_legend_handles_labels()[0]:
            axes[1, 2].legend(fontsize=6)
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "No saved hedge decisions for this expiry",
                ha="center",
                transform=axes[1, 2].transAxes,
            )
        for ax in axes.ravel():
            ax.grid(alpha=0.2)
        lower, upper = float(sample.y.min()), float(sample.y.max())
        for ax in axes[0]:
            ax.set_xlim(lower, upper)
        forward = float(carry.loc[expiry, "forward"])
        for ax in axes[1]:
            ax.set_xlim(forward * np.exp(lower), forward * np.exp(upper))
        fig.suptitle(f"{view.date}: expiry {expiry}, {365 * time:.2f} days")
        return fig

    def build_widget(self, initial_date=None):
        import ipywidgets as widgets

        dates = self.saved.dates
        initial_date = initial_date or (
            "2015-06-19" if "2015-06-19" in dates else dates[0]
        )
        self.requested_date = initial_date
        self.slider = widgets.SelectionSlider(
            options=dates,
            value=initial_date,
            description="Date",
            continuous_update=False,
            layout=widgets.Layout(width="650px"),
        )
        self.picker = widgets.DatePicker(
            description="Select date", value=calendar_date.fromisoformat(initial_date)
        )
        self.previous = widgets.Button(description="Previous")
        self.next = widgets.Button(description="Next")
        self.root = widgets.Dropdown(options=["UNKNOWN"], description="Root")
        self.maturity = widgets.Dropdown(options=[], description="Expiry")
        self.domain = widgets.Dropdown(
            options=[
                ("±0.04 log K/F", 0.04),
                ("±0.09 log K/F", 0.09),
                ("±0.18 log K/F", 0.18),
            ],
            value=0.09,
            description="Domain",
        )
        self.scale = widgets.Dropdown(options=["linear", "log"], description="LV scale")
        self.status = widgets.HTML()
        self.output = widgets.Output()
        self.widget = widgets.VBox(
            [
                widgets.HBox([self.previous, self.next, self.picker]),
                self.slider,
                widgets.HBox([self.root, self.maturity, self.domain, self.scale]),
                self.status,
                self.output,
            ]
        )
        self.slider.observe(self._date_changed, names="value")
        self.picker.observe(self._picker_changed, names="value")
        self.previous.on_click(lambda _: self._step(-1))
        self.next.on_click(lambda _: self._step(1))
        for control in (self.root, self.domain):
            control.observe(self._refresh, names="value")
        for control in (self.maturity, self.scale):
            control.observe(self._render_changed, names="value")
        return self.widget

    def _step(self, amount):
        if self.requested_date is None:
            index = 0 if amount > 0 else len(self.saved.dates) - 1
        else:
            index = (
                bisect_right(self.saved.dates, self.requested_date)
                if amount > 0
                else bisect_left(self.saved.dates, self.requested_date) - 1
            )
        if 0 <= index < len(self.saved.dates):
            self._request_date(self.saved.dates[index])

    def _request_date(self, value: str | None):
        """Synchronize controls to the requested date, including unavailable dates."""
        self.requested_date = value
        self._updating = True
        values = sorted(
            set(self.saved.dates) | ({value} if value is not None else set())
        )
        options = [(date, date) for date in values]
        if value is None:
            options.insert(0, ("No date selected", None))
        self.slider.options = options
        self.slider.value = value
        self.picker.value = calendar_date.fromisoformat(value) if value else None
        self._updating = False
        self._refresh()

    def _picker_changed(self, change):
        if self._updating:
            return
        self._request_date(change["new"].isoformat() if change["new"] else None)

    def _date_changed(self, change):
        if self._updating:
            return
        self._request_date(change["new"])

    def _render_changed(self, change):
        if not self._updating and self.current is not None:
            self.render()

    def _refresh(self, change=None):
        if self._updating:
            return
        self.close_figures()
        self.output.outputs = ()
        requested = self.requested_date
        if requested not in self.saved.dates:
            self.current = None
            self._updating = True
            self.maturity.options = []
            self._updating = False
            self.previous.disabled = (
                requested is not None and requested <= self.saved.dates[0]
            )
            self.next.disabled = (
                requested is not None and requested >= self.saved.dates[-1]
            )
            self.status.value = (
                f"<b>{escape(requested)}: no saved decisions.</b>"
                if requested
                else "<b>Select a saved date.</b>"
            )
            return
        self.status.value = f"Loading {escape(requested)}…"
        try:
            self._updating = True
            roots = sorted(
                self.saved.dated_table(requested, "model_manifest").root.unique()
            )
            previous_root = self.root.value
            self.root.options = roots or ["UNKNOWN"]
            if previous_root in self.root.options:
                self.root.value = previous_root
            self._updating = False
            view = self.select(requested, self.root.value, self.domain.value)
            self._updating = True
            options = [
                (f"{r.expire_date} ({365 * r.maturity_years:.1f}d)", r.expire_date)
                for r in view.tables["carry"].sort_values("maturity_years").itertuples()
            ]
            previous = self.maturity.value
            self.maturity.options = options
            if options:
                self.maturity.value = (
                    previous
                    if previous in dict((v, k) for k, v in options)
                    else min(
                        view.tables["carry"].itertuples(),
                        key=lambda r: abs(365 * r.maturity_years - 35),
                    ).expire_date
                )
            self.previous.disabled = requested == self.saved.dates[0]
            self.next.disabled = requested == self.saved.dates[-1]
            self._updating = False
            self.render()
        except (OSError, ValueError, KeyError) as error:
            self._updating = False
            self.current = None
            self.close_figures()
            self.output.outputs = ()
            self.status.value = (
                f"<b>{escape(requested)}: unavailable — {escape(str(error))}</b>"
            )

    def _show(self, *objects):
        """Append explicitly owned output without frontend capture/reply traffic."""
        for obj in objects:
            self.output.append_display_data(obj)

    def render(self):
        from IPython.display import Markdown
        import matplotlib.pyplot as plt

        if self.current is None:
            return
        view = self.current
        self.close_figures()
        self.output.outputs = ()
        self.status.value = f"<b>{view.date} / {escape(view.root)}: {view.status}</b> {escape(view.message)}"
        with self.output.hold_sync():
            for figure in (
                self.surface_figure(view, self.scale.value),
                self.slice_figure(view, self.maturity.value, self.scale.value),
            ):
                if figure is not None:
                    self.figures.append(figure)
                    self._show(figure)
                    # Manual display belongs to the widget; inline auto-display
                    # would create stale figures outside its clearable output.
                    plt.close(figure)
            for name in ("manifest", "expiry_fit", "carry", "evidence_availability"):
                if name in view.tables:
                    self._show(
                        Markdown(f'**{view.date}: {name.replace("_", " ")}**'),
                        view.tables[name],
                    )
            data = view.tables["decisions"]
            availability = pd.DataFrame(
                [
                    dict(strategy=s, status=status, contracts=int(count))
                    for s, (column, _) in STRATEGIES.items()
                    if column in data
                    for status, count in data[column].value_counts(dropna=False).items()
                ]
            )
            self._show(
                Markdown(f"**{view.date}: saved delta availability**"), availability
            )
            selected = data.loc[data.expire_date.eq(self.maturity.value)]
            columns = ["expire_date", "strike", "kind"] + [
                n for pair in STRATEGIES.values() for n in pair if n in selected
            ]
            self._show(selected[columns])
            checks = [n for n in view.tables if n.startswith("validation/")]
            if not checks:
                self._show(
                    Markdown(
                        f"**{view.date}: saved numerical validation unavailable for this date.**"
                    )
                )
            for name in checks:
                frame = view.tables[name]
                self._show(
                    Markdown(
                        f"**{view.date}: {name}; {len(frame)} recorded rows (first 12 shown)**"
                    ),
                    frame.head(12),
                )

    def display(self):
        from IPython.display import display

        if self.widget is None:
            self.build_widget()
        display(self.widget)
        self._refresh()
