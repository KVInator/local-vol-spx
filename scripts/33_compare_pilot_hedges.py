"""Compare one-session hedges and save diagnostics; no PDE rerun."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hedge_ledger
import pilot_hedge_comparison
from pilot_hedge_comparison import (
    PilotComparisonSettings,
    PilotHedgeComparison,
    summarize,
)


COLORS = {
    "unhedged": "#737373",
    "black": "#2471a3",
    "ah": "#c05a28",
}
LABELS = {
    "unhedged": "Unhedged",
    "black": "Black",
    "ah": "Smoothed AH",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(folder):
    path = folder / "delta_panel.csv"
    audit_path = folder / "audit.json"
    hashes = {
        str(p.resolve()): digest(p)
        for p in (path, audit_path)
    }
    audit = json.loads(audit_path.read_text())

    if (
        audit.get("status") != "completed"
        or digest(path)
        != audit["output_sha256"].get("delta_panel.csv")
    ):
        raise ValueError(
            "Incomplete panel run or panel checksum mismatch"
        )

    for name in (
        "marks_replaced_by_model_prices",
        "entries_filtered_by_future_availability",
        "entry_calculations_use_future_quotes",
        "values_clipped",
    ):
        if audit.get(name) is not False:
            raise ValueError(f"Unexpected upstream policy: {name}")

    panel = pd.read_csv(
        path,
        dtype={
            name: str for name in (
                "entry_id", "contract_id", "quote_date", "end_date",
                "expire_date", "root", "kind",
            )
        },
    )
    if len(panel) != audit["entries"]:
        raise ValueError("Panel and audit entry counts disagree")

    return panel, audit, hashes


def save_plots(panel, result, summary, daily, output, settings):
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    folder = output / "plots"
    folder.mkdir()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(
            folder / name,
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(fig)

    ready = panel.loc[
        panel.black_status.eq("ready")
        & panel.ah_status.eq("ready")
    ].copy()
    ready["difference"] = ready.ah_delta - ready.black_delta

    fig, ax = plt.subplots(figsize=(7, 5))
    for kind, color in (
        ("call", COLORS["black"]),
        ("put", COLORS["ah"]),
    ):
        group = ready.loc[ready.kind.eq(kind)]
        ax.scatter(
            group.black_delta,
            group.ah_delta,
            s=24,
            alpha=0.65,
            color=color,
            label=kind.title(),
        )

    lo = float(ready[["black_delta", "ah_delta"]].min().min())
    hi = float(ready[["black_delta", "ah_delta"]].max().max())
    ax.plot(
        [lo, hi], [lo, hi],
        "k--", lw=1, label="Equal deltas",
    )
    ax.set(
        xlabel="Black spot delta",
        ylabel="Smoothed AH spot delta",
        title=f"Entry hedge ratios: all {len(ready)} ready entries",
    )
    ax.legend()
    save(fig, "01_delta_comparison.png")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        sharey=True,
        layout="constrained",
    )
    for ax, kind in zip(axes, ("call", "put")):
        group = ready.loc[ready.kind.eq(kind)]
        points = ax.scatter(
            group.entry_spot_y,
            group.difference,
            c=group.calendar_days,
            vmin=14,
            vmax=45,
            cmap="viridis",
            s=28,
            alpha=0.75,
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            xlabel="Entry log(K / spot)",
            title=kind.title(),
        )

    axes[0].set_ylabel("AH delta minus Black delta")
    fig.colorbar(
        points,
        ax=axes,
        label="Calendar days to expiry",
    )
    fig.suptitle("Delta differences by moneyness and maturity")
    fig.savefig(
        folder / "02_delta_difference.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)

    mid = result.trials.loc[result.trials.scenario.eq("mid")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.histogram_bin_edges(mid.net_pnl, bins="fd")

    for strategy in PilotHedgeComparison.STRATEGIES:
        values = mid.loc[
            mid.strategy.eq(strategy), "net_pnl"
        ].to_numpy()

        axes[0].hist(
            values,
            bins=bins,
            histtype="step",
            lw=1.7,
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
        absolute = np.sort(np.abs(values))
        axes[1].step(
            absolute,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=COLORS[strategy],
            label=LABELS[strategy],
        )

    axes[0].set(
        xlabel="Funded net P&L, index points",
        ylabel="Comparisons",
        title="Signed P&L",
    )
    axes[1].set(
        xlabel="Absolute funded net P&L, index points",
        ylabel="Cumulative fraction",
        title="Absolute hedge error",
    )
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Midpoint baseline: separate one-session short-option experiments"
    )
    save(fig, "03_pnl_distributions.png")

    paired = result.paired.loc[result.paired.scenario.eq("mid")]
    fig, ax = plt.subplots(figsize=(7, 5))

    for kind, color in (
        ("call", COLORS["black"]),
        ("put", COLORS["ah"]),
    ):
        group = paired.loc[paired.kind.eq(kind)]
        ax.scatter(
            group.black_net_pnl.abs(),
            group.ah_net_pnl.abs(),
            s=24,
            alpha=0.65,
            color=color,
            label=kind.title(),
        )

    maximum = max(
        float(paired.black_net_pnl.abs().max()),
        float(paired.ah_net_pnl.abs().max()),
        1e-12,
    )
    ax.plot([0, maximum], [0, maximum], "k--", lw=1)
    ax.set(
        xlabel="Black absolute net P&L, index points",
        ylabel="AH absolute net P&L, index points",
        title="Paired midpoint errors: below the line favours AH",
    )
    ax.legend()
    save(fig, "04_paired_errors.png")

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True
    )
    for strategy in PilotHedgeComparison.STRATEGIES:
        group = daily.loc[
            daily.scenario.eq("mid")
            & daily.strategy.eq(strategy)
        ]
        axes[0].plot(
            pd.to_datetime(group.quote_date),
            group.rms_net_pnl,
            marker="o",
            ms=3,
            color=COLORS[strategy],
            label=LABELS[strategy],
        )

    improvement = paired.groupby(
        "quote_date"
    ).abs_error_improvement.mean()
    axes[1].plot(
        pd.to_datetime(improvement.index),
        improvement,
        marker="o",
        color=COLORS["ah"],
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[0].set(
        ylabel="RMS net P&L, index points",
        title="Errors grouped by entry date",
    )
    axes[1].set(
        ylabel="Mean paired absolute-error improvement",
        xlabel="Entry date",
        title=(
            "Black absolute error minus AH absolute error; "
            "positive favours AH"
        ),
    )

    dates = pd.to_datetime(mid.quote_date.unique())
    axes[1].set_xlim(
        dates.min() - pd.Timedelta(days=1),
        dates.max() + pd.Timedelta(days=1),
    )
    axes[0].legend()
    fig.autofmt_xdate()
    save(fig, "05_daily_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cases = list(PilotHedgeComparison(settings).cases)

    for i, strategy in enumerate(PilotHedgeComparison.STRATEGIES):
        group = (
            summary.loc[summary.strategy.eq(strategy)]
            .set_index("scenario")
            .reindex(cases)
        )
        axes[0].bar(
            np.arange(3) + (i - 1) * 0.24,
            group.mean_direct_costs,
            width=0.24,
            color=COLORS[strategy],
            label=LABELS[strategy],
        )
        axes[1].bar(
            i,
            float(group.loc["mid", "mean_hedge_turnover"]),
            color=COLORS[strategy],
        )

    axes[0].set_xticks(
        range(3),
        [
            "Midpoint",
            "Option spread",
            f"Option spread\n+ {settings.hedge_fee_bps:g} bp hedge fee",
        ],
    )
    axes[0].set(
        ylabel="Mean direct cost, index points",
        title="Observed option spread and assumed hedge fee",
    )
    axes[0].legend()
    axes[1].set_xticks(
        range(3),
        [LABELS[s] for s in PilotHedgeComparison.STRATEGIES],
    )
    axes[1].set(
        ylabel="Mean hedge turnover, index-point notional",
        title="Entry plus liquidation turnover",
    )
    save(fig, "06_costs_and_turnover.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/hedging_pilot_diagnostics/september_2023"
        ),
    )
    parser.add_argument("--lend-rate", type=float, default=0.05)
    parser.add_argument("--borrow-rate", type=float, default=0.05)
    parser.add_argument("--hedge-fee-bps", type=float, default=1.0)
    args = parser.parse_args()

    settings = PilotComparisonSettings(
        args.lend_rate,
        args.borrow_rate,
        args.hedge_fee_bps,
    )
    panel, parent, hashes = read_panel(args.panel_run)
    result = PilotHedgeComparison(settings).run(panel)

    output = args.output / datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    result.coverage.to_csv(output / "coverage.csv", index=False)

    if result.trials.empty:
        raise ValueError(
            f"No complete comparisons. Coverage saved in {output}"
        )

    summary = summarize(
        result.trials, ["scenario", "strategy"]
    )
    daily = summarize(
        result.trials, ["scenario", "quote_date", "strategy"]
    )

    breakdown = result.trials.copy()
    breakdown["maturity_band"] = pd.cut(
        breakdown.calendar_days,
        [13, 27, 39, 45],
        labels=["14-27d", "28-39d", "40-45d"],
    ).astype(str)
    grouped = summarize(
        breakdown,
        ["scenario", "kind", "maturity_band", "strategy"],
    )

    pairs = result.paired.groupby("scenario").agg(
        comparisons=("entry_id", "size"),
        entry_dates=("quote_date", "nunique"),
        mean_abs_error_improvement=("abs_error_improvement", "mean"),
        mean_squared_error_improvement=(
            "squared_error_improvement", "mean"
        ),
        fraction_ah_lower_abs_error=("ah_lower_abs_error", "mean"),
        fraction_abs_error_ties=("abs_error_tie", "mean"),
    ).reset_index()

    tables = {
        "strategy_results": result.trials,
        "ledgers": result.ledgers,
        "paired_results": result.paired,
        "summary": summary,
        "daily_summary": daily,
        "group_summary": grouped,
        "paired_summary": pairs,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    save_plots(
        panel, result, summary, daily, output, settings
    )

    if any(
        digest(path) != value
        for path, value in hashes.items()
    ):
        raise RuntimeError("Input changed during comparison")

    sources = [
        Path(__file__).resolve(),
        Path(pilot_hedge_comparison.__file__).resolve(),
        Path(hedge_ledger.__file__).resolve(),
    ]
    accounting_failed = result.coverage.status.eq(
        "accounting_failed"
    ).any()

    audit = {
        "status": (
            "completed_with_accounting_failures"
            if accounting_failed else "completed"
        ),
        "settings": asdict(settings),
        "entries": len(panel),
        "coverage_by_scenario": {
            case: group.status.value_counts().to_dict()
            for case, group in result.coverage.groupby("scenario")
        },
        "paired_scope": (
            "Common entries with observed endpoints and both ready deltas; "
            "failures retained in coverage."
        ),
        "unit": (
            "Index points per short research option; "
            "both multipliers equal one."
        ),
        "holding": (
            "Entry delta held to the next supplied reference session; "
            "both positions then liquidated."
        ),
        "marks": (
            "Original observed option midpoints and vendor index levels; "
            "no model-price substitution."
        ),
        "option_execution": (
            "Midpoint baseline; observed entry bid and exit ask "
            "in the spread scenarios."
        ),
        "hedge_execution": (
            "Synthetic fractional index fills at its mark. "
            "Bid=ask=spot is a convention, not observed quotes."
        ),
        "hedge_fee": (
            "Assumed fee on entry and liquidation notional; "
            "not a market execution-cost estimate."
        ),
        "dividend_per_unit": 0.0,
        "dividend_income_inferred": False,
        "funding": (
            "Assumed continuous lend/borrow rates, selected by opening "
            "cash sign; ACT/365F elapsed seconds."
        ),
        "pricing_carry_case": parent["carry_case"],
        "pricing_assumed_annual_rate": parent["assumed_annual_rate"],
        "AH_settings": parent["settings"],
        "funding_sensitivity_refits_pricing_carry": False,
        "error_definition": (
            "Absolute P&L and zero-centred RMS funded P&L; "
            "higher signed profit is not the hedge-quality objective."
        ),
        "cost_interpretation": (
            "Use the midpoint case to compare hedge errors. "
            "Costs can offset positive P&L and reduce absolute P&L "
            "without improving the hedge."
        ),
        "statistical_scope": (
            "September development pilot; rows share dates and contracts. "
            "No independent-row significance test or annualization."
        ),
        "portfolio_wealth_curve_constructed": False,
        "out_of_sample_claim": False,
        "contract_identity_verified": False,
        "snapshot_provenance_verified": False,
        "funding_curve_verified": False,
        "executable_hedge_returns_claimed": False,
        "future_information_used_in_entry_hedges": False,
        "models_refitted": False,
        "prices_clipped": False,
        "max_ledger_reconciliation": float(
            result.trials.max_reconciliation.max()
        ),
        "max_closed_form_error": float(
            result.trials.closed_form_error.abs().max()
        ),
        "input_sha256": hashes,
        "source_sha256": {
            str(path): digest(path) for path in sources
        },
        "output_sha256": {
            path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print(
        "Coverage:\n"
        + result.coverage.groupby("scenario").status.value_counts().to_string()
    )
    print(
        "\nStrategy summary:\n"
        + summary[[
            "scenario", "strategy", "comparisons", "entry_dates",
            "mean_net_pnl", "std_net_pnl", "rms_net_pnl",
            "mean_abs_net_pnl", "mean_direct_costs",
        ]].to_string(index=False)
    )
    print(
        "\nPaired AH versus Black:\n"
        + pairs.to_string(index=False)
    )
    print(
        "\nMaximum accounting reconciliation: "
        f"{audit['max_ledger_reconciliation']:.3g}"
    )
    print(f"Diagnostics: {output.resolve()}")
    print(
        "Six figures saved under plots/. Research synthetic-index "
        "hedges; September development sample."
    )


if __name__ == "__main__":
    main()