"""Create a clean notebook for the saved September calibration and hedge pilot."""

import argparse
import json
from pathlib import Path
from textwrap import dedent


DEFAULTS = {
    "calibration": "data/processed/hedging_pilot/september_2023/daily_ah/run_20261007T040538_966878Z",
    "panel": "data/processed/hedging_pilot/september_2023/hedge_panel/run_20261007T062726_950243Z",
    "comparison": "outputs/hedging_pilot_diagnostics/september_2023/run_20261007T074506_549964Z",
    "attribution": "outputs/hedging_pilot_diagnostics/september_2023/attribution/run_20261007T081012_331452Z",
}


def build_notebook(paths):
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"pilot-{len(cells):02d}",
                "metadata": {}, "source": dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
        # September SPX local-volatility and hedging pilot

        Explore every available September calibration date, then inspect the saved
        one-session hedge comparison and attribution. This is a development sample
        with provisional contract identity and snapshot provenance.

        **Original AH implied volatility** comes from the calibrated price curve.
        **Hedge-diffusion local volatility** includes the accepted first-period
        variance smoothing. These are different models. Calibration fit and
        numerical sensitivity do not establish out-of-sample hedge superiority.

        Run the cells in order. Change `DATE` to explore another date. This notebook
        loads saved models and results; it performs no optimizer or hedge PDE run.
    """)
    add("code", """
        from pathlib import Path
        from functools import lru_cache
        import hashlib
        import json
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        try:
            from IPython.display import display
        except ImportError:
            def display(value):
                print(value)

        ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / "src").is_dir() and (p / "data").is_dir()), None)
        if ROOT is None:
            raise FileNotFoundError("Open this notebook inside local-vol-spx.")
        sys.path.insert(0, str(ROOT / "src"))
        from andreasen_huge import AndreasenHugeSurface
        from daily_ah_validation import AHShortEndVariance

        pd.set_option("display.max_columns", 20)
        plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                             "grid.alpha": 0.2, "font.size": 10})
    """)
    add("code", "RUN_PATHS = " + repr(paths) + "\n" + dedent("""
        RUNS = {name: ROOT / path for name, path in RUN_PATHS.items()}
        # Explicit run folders keep the notebook tied to reviewed results.
        AUDITS = {name: json.loads((path / "audit.json").read_text())
                  for name, path in RUNS.items()}
        for name in ["panel", "comparison", "attribution"]:
            if AUDITS[name].get("status") != "completed":
                raise ValueError(f"Incomplete {name} run.")

        def sha256(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def checked_path(run, filename):
            folder = RUNS[run].resolve()
            path = (folder / filename).resolve()
            if not path.is_relative_to(folder):
                raise ValueError("An artifact path leaves its run folder.")
            expected = AUDITS[run].get("output_sha256", {}).get(filename)
            if expected is None or sha256(path) != expected:
                raise ValueError(f"Missing or mismatched checksum: {run}/{filename}")
            return path

        def read_table(run, filename):
            return pd.read_csv(checked_path(run, filename))

        for child, parent, filenames in [
            ("panel", "calibration", ["audit.json", "model_manifest.csv"]),
            ("comparison", "panel", ["audit.json", "delta_panel.csv"]),
            ("attribution", "comparison",
             ["audit.json", "strategy_results.csv", "paired_results.csv", "coverage.csv"]),
        ]:
            recorded = AUDITS[child].get("input_sha256", {})
            for filename in filenames:
                digest = sha256(RUNS[parent] / filename)
                if not any(Path(key).name == filename and value == digest
                           for key, value in recorded.items()):
                    raise ValueError(f"{child} does not use the selected {parent}/{filename}.")

        manifest = read_table("calibration", "model_manifest.csv")
        quotes = read_table("calibration", "quote_residuals.csv")
        conditioning = read_table("calibration", "conditioning.csv")
        shape = read_table("calibration", "shape_checks.csv")
        panel = read_table("panel", "delta_panel.csv")
        coverage = read_table("comparison", "coverage.csv")
        strategies = read_table("comparison", "strategy_results.csv")
        summary = read_table("comparison", "summary.csv")
        attribution = read_table("attribution", "attribution.csv")
        attribution_summary = read_table("attribution", "summary.csv")
        daily_attribution = read_table("attribution", "daily_attribution.csv")
        groups = read_table("attribution", "group_summary.csv")
        leave_one_out = read_table("attribution", "leave_one_date_out.csv")
        DATES = sorted(manifest.quote_date.unique())
        RADIUS = float(AUDITS["panel"]["settings"]["radius"])
        print(f"Verified {len(DATES)} calibration dates, {len(panel)} entries.")
        print(f"Saved hedge diffusion radius: {RADIUS:g}; pricing carry: "
              f"{AUDITS['calibration']['carry_case']}.")
        display(pd.DataFrame({"run": RUNS.keys(),
                              "folder": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT)
                                         else str(p) for p in RUNS.values()]}))
    """))
    add("markdown", """
        ## Assumptions and units

        Observed quote marks and vendor index levels remain the historical inputs.
        Estimated forwards, fitted surfaces, and hedge deltas are derived quantities.
        The index hedge, funding, and fees below are research conventions. No
        simulated historical prices or missing-session replacement marks are used.
    """)
    add("code", """
        ca, ha = AUDITS["calibration"], AUDITS["comparison"]
        display(pd.DataFrame([
            ("Historical inputs", "Observed option bid/ask and vendor index levels"),
            ("Identity / snapshot provenance", "Unverified; UNKNOWN remains a label"),
            ("Pricing rate", f"Assumed {100 * ca['assumed_annual_rate']:.1f}% annual rate"),
            ("Forward", "Same-date conditional put-call parity estimate"),
            ("Local-volatility model", f"AH; first-period Gaussian variance blend, radius {RADIUS:g}"),
            ("Greek convention", "Physical local-volatility coefficient and original carry fixed"),
            ("Option / hedge units", ha["unit"]),
            ("Holding period", ha["holding"]),
            ("Hedge fills", ha["hedge_execution"]),
            ("Funding", str(ha["settings"])),
            ("Dividend cash", f"{ha['dividend_per_unit']}; not inferred from the pricing yield proxy"),
            ("Evidence scope", "September development sample; no executable-return or OOS claim"),
        ], columns=["item", "convention"]))
    """)
    add("markdown", """
        ## Coverage and calibration across dates

        Counts include unsuccessful or unavailable entries. Calibration residuals
        use original half-spreads; shape checks apply to sampled finite-grid prices.
        The conditioning table describes the **original AH** nodal coefficient.
    """)
    add("code", """
        display(manifest[["quote_date", "root", "status", "input_expiries",
                          "calibration_quotes", "rms_half_spreads",
                          "outside_original_bands", "active_proxy_bounds",
                          "sampled_shape_violations"]])
        flags = ["increasing_prices", "vertical_spread_violations", "negative_butterflies",
                 "intrinsic_shortfalls", "upper_bound_excesses", "calendar_violations"]
        display(shape.groupby("quote_date")[flags].sum())
        display(conditioning.groupby("quote_date").agg(
            original_AH_sampled_peak_lv_pct=("peak_local_vol_pct", "max"),
            unresolved_original_nodal_samples=("unresolved_variance_nodes", "sum")))
        display(coverage.groupby(["scenario", "status"]).size().rename("entries").to_frame())
        counts = panel.groupby("quote_date").agg(
            entries=("entry_id", "size"), ready=("both_deltas_ready", "sum"),
            matched=("matched_comparison_available", "sum")).reindex(DATES)
        display(counts)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        fit = manifest.groupby("quote_date").agg(
            rms=("rms_half_spreads", "max"), outside=("outside_original_bands", "sum"))
        fit.rms.plot(ax=ax[0], marker="o", title="Daily calibration RMS")
        ax[0].set_ylabel("Half-spreads; max across roots")
        fit.outside.plot.bar(ax=ax[1], title="Quotes outside original bands")
        counts[["entries", "matched"]].plot.bar(ax=ax[2], title="Entry and endpoint coverage")
        for a in ax:
            a.tick_params(axis="x", rotation=60)
        plt.show()
    """)
    add("markdown", r"""
        ## Date explorer

        `DATE` can be any date printed in `DATES`. Surface maps use
        $y=\log(K/F(T))$ over [-0.09, 0.09]. IV starts at the first fitted maturity.
        Local volatility includes positive short-end times; day-zero coefficients
        are never requested. Pillars use their right-hand coefficient, with the
        left-hand value at the terminal pillar. The separate jump table checks
        both sides on a denser sample. Sampled peaks are not global maxima.

        Local-volatility colors use a logarithmic scale. Values are not capped or
        clipped. Blank hedge results mean unavailable observations, not zero P&L.
    """)
    add("code", """
        @lru_cache(maxsize=3)
        def load_model(date, root):
            rows = manifest[(manifest.quote_date == date) & (manifest.root == root)]
            if len(rows) != 1 or rows.iloc[0].status != "fitted":
                raise ValueError("Choose one successfully fitted date/root.")
            row = rows.iloc[0]
            path = checked_path("calibration", row.model_file)
            if sha256(path) != row.model_sha256:
                raise ValueError("Model checksum does not match its manifest.")
            model = AndreasenHugeSurface.load(path)
            return model, AHShortEndVariance(model, RADIUS)

        @lru_cache(maxsize=3)
        def surface_samples(date, root):
            model, diffusion = load_model(date, root)
            y = np.linspace(-0.09, 0.09, 81)
            pillars = model.maturities
            times = np.unique(np.r_[pillars, 0.5 * (pillars[:-1] + pillars[1:])])
            iv = np.array([100 * model.implied_volatility(model.forward(t) * np.exp(y), t)
                           for t in times])
            lv_times = np.unique(np.r_[pillars[0] * np.array([0.0625, 0.25, 0.5]), times])
            lv = np.array([100 * np.sqrt(diffusion.normalized_variance(
                np.exp(y), t, "left" if t == pillars[-1] else "right")) for t in lv_times])
            if not np.isfinite(iv).all() or not np.isfinite(lv).all() or (lv <= 0).any():
                raise ArithmeticError("Surface contains unresolved values.")
            return y, times * 365, iv, lv_times * 365, lv

        def explore_day(date, root="UNKNOWN"):
            model, diffusion = load_model(date, root)
            print(f"{date}: spot {model.spot:.2f}; {len(model.maturities)} expiry pillars.")
            q = quotes[(quotes.quote_date == date) & (quotes.root == root)]
            p = panel[(panel.quote_date == date) & (panel.root == root)]
            y, days, iv, lv_days, lv = surface_samples(date, root)
            fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
            im = ax[0, 0].pcolormesh(y, days, iv, shading="nearest", cmap="viridis")
            fig.colorbar(im, ax=ax[0, 0], label="Implied vol (%)")
            ax[0, 0].set_title("Original AH calibration implied volatility")
            im = ax[0, 1].pcolormesh(y, lv_days, lv, shading="nearest", cmap="magma",
                                   norm=LogNorm(vmin=lv.min(), vmax=lv.max()))
            fig.colorbar(im, ax=ax[0, 1], label="Local vol (%), log scale")
            ax[0, 1].set_title(f"Hedge-diffusion local volatility; radius {RADIUS:g}")
            for a in ax[0]:
                a.set_xlabel("Forward log-moneyness")
                a.set_ylabel("Calendar days remaining")
                a.set_xlim(y[0], y[-1])
            ax[0, 0].set_ylim(days[0], days[-1])
            ax[0, 1].set_ylim(lv_days[0], lv_days[-1])
            scatter = ax[1, 0].scatter(q.forward_log_moneyness, q.residual_half_spreads,
                                       c=q.assumed_maturity_years * 365, s=12, alpha=0.65)
            fig.colorbar(scatter, ax=ax[1, 0], label="Calendar days remaining")
            ax[1, 0].axhline(1, color="grey", linestyle="--")
            ax[1, 0].axhline(-1, color="grey", linestyle="--")
            ax[1, 0].set(xlabel="Forward log-moneyness", ylabel="Residual / half-spread",
                         title="Original AH quote residuals")
            if len(p):
                for kind, marker in [("call", "o"), ("put", "s")]:
                    ready = p[(p.kind == kind) & p.both_deltas_ready]
                    ax[1, 1].scatter(ready.black_delta, ready.ah_delta, marker=marker,
                                     label=kind, alpha=0.8)
                lo = min(p.black_delta.min(), p.ah_delta.min())
                hi = max(p.black_delta.max(), p.ah_delta.max())
                if np.isfinite([lo, hi]).all():
                    ax[1, 1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
                ax[1, 1].legend()
            else:
                ax[1, 1].text(0.5, 0.5, "No saved hedge entries for this date",
                               ha="center", transform=ax[1, 1].transAxes)
            ax[1, 1].set(xlabel="Black delta", ylabel="AH smoothed-diffusion delta",
                         title="Saved entry deltas")
            fig.suptitle(f"September pilot: {date}", fontsize=14)
            plt.show()

            jump_y = np.linspace(-0.09, 0.09, 721)
            jumps = []
            for t in model.maturities[:-1]:
                left = 100 * np.sqrt(diffusion.normalized_variance(np.exp(jump_y), t, "left"))
                right = 100 * np.sqrt(diffusion.normalized_variance(np.exp(jump_y), t, "right"))
                index = np.argmax(abs(right - left))
                jumps.append({"days": 365 * t, "left_peak_pct": left.max(),
                              "right_peak_pct": right.max(), "jump_y": jump_y[index],
                              "signed_jump_pp": (right - left)[index]})
            display(pd.DataFrame(jumps))
            if len(p):
                display(p[["expire_date", "strike", "kind", "black_delta", "ah_delta",
                           "ah_gamma", "ah_price_minus_mid", "end_date", "end_status"]])
            outcomes = strategies[(strategies.quote_date == date) & (strategies.root == root)]
            if outcomes.empty:
                print("No matched hedge comparison on this date. Entry coverage is retained above.")
            else:
                display(outcomes.groupby(["scenario", "strategy"]).agg(
                    comparisons=("net_pnl", "size"), mean_pnl=("net_pnl", "mean"),
                    mae=("net_pnl", lambda v: np.mean(abs(v))),
                    rms=("net_pnl", lambda v: np.sqrt(np.mean(v ** 2)))))
            return None

    """)
    add("code", """
        print("Available calibration dates:", DATES)
        fitted_models = manifest.loc[manifest.status.eq("fitted")].sort_values(
            ["quote_date", "root"])
        if fitted_models.empty:
            raise ValueError("No fitted model is available for the date explorer.")
        first_model = fitted_models.iloc[0]
        DATE, ROOT_LABEL = first_model.quote_date, first_model.root
        explore_day(DATE, ROOT_LABEL)
    """)
    add("markdown", """
        ## Hedge error and attribution

        Each observation is a separately opened and liquidated short-option hedge.
        These distributions and date contributions are not portfolio wealth curves.
        Positive MAE/MSE improvement favors AH. Zero-centered RMS includes P&L bias.
        Cost scenarios can offset positive P&L without improving the hedge itself.
    """)
    add("code", """
        display(summary[["scenario", "strategy", "comparisons", "entry_dates",
                         "mean_net_pnl", "rms_net_pnl", "mean_abs_net_pnl", "mean_direct_costs"]])
        display(attribution_summary[["scenario", "mae_improvement", "mse_improvement",
                                     "variance_improvement", "squared_mean_improvement",
                                     "fraction_ah_lower_abs_error", "mean_pnl_gap",
                                     "mean_delta_exposure", "mean_funding_difference",
                                     "mean_cost_effect", "max_identity_residual"]])
        fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        midpoint = strategies[strategies.scenario == "mid"]
        bins = np.histogram_bin_edges(midpoint.net_pnl, bins=35)
        for strategy in ["ah", "black", "unhedged"]:
            values = midpoint[midpoint.strategy == strategy].net_pnl
            ax[0, 0].hist(values, bins=bins, alpha=0.45, label=strategy)
        ax[0, 0].set(xlabel="Net P&L (index points)", ylabel="Comparisons",
                     title="Midpoint P&L distribution")
        ax[0, 0].legend()
        daily = daily_attribution[daily_attribution.scenario == "mid"].set_index("quote_date")
        for column, a, label, unit in [
            ("pooled_mae_contribution", ax[0, 1], "MAE", "Index points"),
            ("pooled_mse_contribution", ax[0, 2], "MSE", "Squared index points"),
        ]:
            daily[column].plot.bar(ax=a, title=f"Midpoint date contribution to {label}")
            a.axhline(0, color="black", linewidth=1)
            a.set_ylabel(unit)
        summary.pivot(index="scenario", columns="strategy", values="rms_net_pnl").plot.bar(
            ax=ax[1, 0], title="Zero-centered RMS by cost scenario")
        ax[1, 0].set_ylabel("Index points")
        omitted = leave_one_out[leave_one_out.scenario == "mid"].set_index("excluded_date")
        for column, a, label, unit in [
            ("mae_improvement", ax[1, 1], "MAE", "Index points"),
            ("mse_improvement", ax[1, 2], "MSE", "Squared index points"),
        ]:
            if len(omitted):
                omitted[column].plot.bar(ax=a, title=f"Omit one entry date: {label} improvement")
                a.axhline(0, color="black", linewidth=1)
                a.axhline(float(attribution_summary.set_index("scenario").loc["mid", column]),
                          color="darkorange", linestyle="--", label="Full sample")
                a.set_ylabel(unit)
                a.legend()
            else:
                a.text(0.5, 0.5, "Requires multiple entry dates",
                       ha="center", transform=a.transAxes)
        for a in ax.ravel()[1:]:
            a.tick_params(axis="x", rotation=60)
        plt.show()
        display(groups[groups.scenario == "mid"][[
            "dimension", "group", "comparisons", "mae_improvement", "mse_improvement",
            "fraction_ah_lower_abs_error"]])
        display(daily.sort_values("pooled_mse_contribution", ascending=False))
    """)
    add("markdown", """
        ## What these results support

        The ledger and attribution reconcile; the pilot measures differences
        between two hedge rules on common observed endpoints. Aggregate results
        must be read alongside bias, date influence, contract groups, and coverage.
        Keep influential dates in the sample. Leave-one-date-out is a diagnostic,
        not a rule for excluding unfavorable results.

        Remaining work: freeze the experiment settings, run a chronological batch
        beyond this development month, add the agreed comparison baselines and
        sensitivity scenarios, and assess uncertainty with shared dates/contracts
        accounted for. The full-history data audit is complete; full-history
        calibration and hedging are not complete. Missing identity, fixing,
        snapshot, and funding evidence remain documented limitations.
    """)
    add("code", """
        mid = attribution_summary.set_index("scenario").loc["mid"]
        print(f"Midpoint AH RMS: {mid.ah_rms:.6f}; Black RMS: {mid.black_rms:.6f}.")
        print(f"Midpoint MAE improvement: {mid.mae_improvement:+.6f} index points.")
        print(f"AH has lower absolute P&L in {100 * mid.fraction_ah_lower_abs_error:.1f}% of pairs.")
        if len(omitted):
            display(omitted[["mae_improvement", "mse_improvement"]].agg(["min", "max"]))
            print("Both signs can occur after omission:",
                  bool((omitted.mae_improvement < 0).any() and (omitted.mae_improvement > 0).any()))
        print("Scope: September development pilot; synthetic index hedges.")
    """)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python (volspx)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}-run", default=default)
    parser.add_argument("--output", default="notebooks/september_pilot.ipynb")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: getattr(args, f"{name}_run") for name in DEFAULTS}
    output.write_text(json.dumps(build_notebook(paths), indent=2) + "\n", encoding="utf-8")
    print(f"Created {output}; open it with the volspx kernel and run the cells in order.")


if __name__ == "__main__":
    main()
