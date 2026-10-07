"""Attribute existing hedge results and measure entry-date concentration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_hedge_attribution
from pilot_hedge_attribution import (
    PilotHedgeAttribution, read_comparison, sha256,
)


def plots(tables, folder):
    folder.mkdir()
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    daily = tables["daily_attribution"].query(
        "scenario == 'mid'"
    ).sort_values("quote_date")

    x = np.arange(len(daily))
    labels = daily.quote_date.str.slice(5)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(
        x - 0.18,
        daily.mean_delta_exposure,
        0.36,
        label="Delta difference × index move",
        color="#2471a3",
    )
    ax.bar(
        x + 0.18,
        daily.mean_funding_difference,
        0.36,
        label="Funding difference",
        color="#737373",
    )
    ax.plot(
        x, daily.mean_pnl_gap, "o",
        color="#c05a28", ms=4,
        label="Observed AH minus Black P&L",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=60)
    ax.set(
        xlabel="Entry date",
        ylabel="Mean signed P&L difference, index points",
        title=(
            "Midpoint P&L attribution; "
            "positive does not necessarily mean lower error"
        ),
    )
    ax.legend()
    save(fig, "01_pnl_attribution.png")

    mid = tables["attribution"].query("scenario == 'mid'")
    fig, ax = plt.subplots(figsize=(7, 5))

    for kind, color in (
        ("call", "#2471a3"), ("put", "#c05a28")
    ):
        group = mid.loc[mid.kind.eq(kind)]
        ax.scatter(
            100 * group.spot_return,
            group.pnl_gap,
            s=23,
            alpha=0.65,
            color=color,
            label=kind.title(),
        )

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Observed holding-period index return, %",
        ylabel="AH minus Black signed P&L, index points",
        title=(
            "Directional exposure diagnostic; "
            "contracts share date-level moves"
        ),
    )
    ax.legend()
    save(fig, "02_move_and_pnl_gap.png")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )

    for ax, column, units in zip(
        axes,
        ["pooled_mse_contribution", "pooled_mae_contribution"],
        ["Squared index points", "Index points"],
    ):
        values = daily[column]
        ax.bar(
            x,
            values,
            color=np.where(values >= 0, "#2471a3", "#c05a28"),
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(units)
        metric = "MSE" if "mse" in column else "MAE"
        ax.set_title(
            f"Date contribution to pooled {metric} improvement"
        )

    axes[1].set_xticks(x, labels, rotation=60)
    axes[1].set_xlabel("Entry date; positive favours AH")
    save(fig, "03_date_contributions.png")

    loo = tables["leave_one_date_out"].query(
        "scenario == 'mid'"
    ).sort_values("excluded_date")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )

    for ax, metric, units in zip(
        axes,
        ["mse_improvement", "mae_improvement"],
        ["Squared index points", "Index points"],
    ):
        ax.axhline(0, color="black", lw=0.8)

        if len(loo):
            ax.plot(
                np.arange(len(loo)),
                loo[metric],
                "o-",
                color="#2471a3",
                ms=4,
            )
            ax.axhline(
                loo["full_" + metric].iloc[0],
                color="#c05a28",
                ls="--",
                label="Full sample",
            )
            ax.legend()
        else:
            ax.text(
                0.5, 0.5,
                "At least two entry dates are required",
                transform=ax.transAxes,
                ha="center",
            )

        name = "MSE" if metric.startswith("mse") else "MAE"
        ax.set(
            ylabel=units,
            title=f"{name} improvement after omitting one date",
        )

    if len(loo):
        axes[1].set_xticks(
            np.arange(len(loo)),
            loo.excluded_date.str.slice(5),
            rotation=60,
        )

    axes[1].set_xlabel(
        "Omitted entry date; sensitivity diagnostic, "
        "not a confidence interval"
    )
    save(fig, "04_leave_one_date_out.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-run", type=Path, required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/hedging_pilot_diagnostics/"
            "september_2023/attribution"
        ),
    )
    args = parser.parse_args()

    trials, paired, coverage, parent, inputs = read_comparison(
        args.comparison_run
    )
    tables = PilotHedgeAttribution().run(
        trials, paired, coverage
    )

    output = args.output / datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)

    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    plots(tables, output / "plots")

    if any(
        sha256(path) != digest
        for path, digest in inputs.items()
    ):
        raise RuntimeError(
            "Comparison inputs changed during attribution"
        )

    summary = tables["summary"]
    sources = [
        Path(__file__).resolve(),
        Path(pilot_hedge_attribution.__file__).resolve(),
    ]

    audit = {
        "status": "completed",
        "comparison_run": str(args.comparison_run.resolve()),
        "upstream_status": parent["status"],
        "settings": parent["settings"],
        "identity": (
            "AH-Black net P&L = delta difference * observed "
            "spot change + interest difference "
            "- direct-cost difference."
        ),
        "spot_return": (
            "Entry spot reconstructed as K*exp(-entry_spot_y); "
            "entry_spot_y is log(K/observed spot)."
        ),
        "mse_decomposition": (
            "Population variance plus squared sample mean; "
            "neither hedge P&L is demeaned for ranking."
        ),
        "date_contributions": (
            "Date mean improvement times its comparison count "
            "divided by the scenario count; "
            "sums to pooled improvement."
        ),
        "leave_one_date_out": (
            "Omit every comparison sharing an entry date; "
            "do not refit, retune or reselect contracts."
        ),
        "cost_shift": (
            "Same deltas and marks; scenario P&L shifts equal "
            "funding shifts minus direct costs."
        ),
        "group_scope": (
            "Separate marginal breakdowns; realized move "
            "direction is descriptive, not an entry rule."
        ),
        "moneyness_bands": (
            "log(K/spot) below -0.01, within [-0.01,0.01], "
            "or above 0.01."
        ),
        "maturity_bands": (
            "Actual calendar days 14-27, 28-39, 40-45; "
            "other values are explicitly retained."
        ),
        "max_identity_residual": float(
            tables["attribution"].identity_residual.abs().max()
        ),
        "statistical_scope": (
            "Development-sample diagnostics; shared dates "
            "and contracts; no IID-row inference "
            "or annualization."
        ),
        "confidence_intervals_computed": False,
        "out_of_sample_claim": False,
        "portfolio_wealth_curve_constructed": False,
        "executable_hedge_returns_claimed": False,
        "models_refitted": False,
        "pde_rerun": False,
        "new_market_data_used": False,
        "deltas_changed": False,
        "prices_clipped": False,
        "future_information_used_for_entry": False,
        "contract_identity_verified": parent.get(
            "contract_identity_verified", False
        ),
        "snapshot_provenance_verified": parent.get(
            "snapshot_provenance_verified", False
        ),
        "funding_curve_verified": parent.get(
            "funding_curve_verified", False
        ),
        "input_sha256": inputs,
        "source_sha256": {
            str(p): sha256(p) for p in sources
        },
        "output_sha256": {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
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
        "Signed P&L attribution:\n"
        + summary[[
            "scenario", "comparisons", "entry_dates",
            "mean_pnl_gap", "mean_delta_exposure",
            "mean_funding_difference", "mean_cost_effect",
        ]].to_string(index=False)
    )

    print(
        "\nHedge-error comparison; "
        "positive improvement favours AH:\n"
        + summary[[
            "scenario", "ah_rms", "black_rms",
            "mae_improvement", "mse_improvement",
            "variance_improvement",
            "squared_mean_improvement",
        ]].to_string(index=False)
    )

    loo = tables["leave_one_date_out"]
    if len(loo):
        print("\nLeave-one-date-out ranges:")
        print(
            loo.groupby("scenario")[[
                "mse_improvement", "mae_improvement"
            ]].agg(["min", "max"]).to_string()
        )

    daily = tables["daily_attribution"].query(
        "scenario == 'mid'"
    )
    largest = daily.loc[
        daily.pooled_mse_contribution.abs().sort_values(
            ascending=False
        ).index
    ].head(5)

    print(
        "\nLargest midpoint date contributions; "
        "diagnostic sums, not portfolio returns:"
    )
    print(
        largest[[
            "quote_date", "spot_change",
            "pooled_mse_contribution",
            "pooled_mae_contribution",
        ]].to_string(index=False)
    )

    print(
        "\nMaximum attribution residual: "
        f"{audit['max_identity_residual']:.3g}"
    )
    print(f"Diagnostics: {output.resolve()}")
    print(
        "Four plots saved. "
        "No calibration, PDE run or parameter change."
    )


if __name__ == "__main__":
    main()