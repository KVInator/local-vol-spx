"""Build and audit OTM IVs and call-equivalent calibration prices."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black import BlackPricer
from calibration_data import CalibrationQuoteBuilder
from implied_vol import ImpliedVolSolver


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SPX calibration quotes from audited carry."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--carry-audit", type=Path, required=True)
    parser.add_argument(
        "--carry-mode",
        choices=("fixed_discount", "free_discount"),
        default="fixed_discount",
    )
    parser.add_argument("--carry-window-pct", type=float, default=3.0)
    parser.add_argument("--y-min", type=float, default=-0.10)
    parser.add_argument("--y-max", type=float, default=0.10)
    return parser.parse_args()


def save_smile(
    quotes: pd.DataFrame,
    forward: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for kind, color in (("put", "tab:orange"), ("call", "tab:blue")):
        part = quotes.loc[quotes["option_kind"].eq(kind)]

        if part.empty:
            continue

        ax.vlines(
            part["strike"],
            100.0 * part["iv_bid"],
            100.0 * part["iv_ask"],
            color=color,
            alpha=0.6,
            linewidth=0.8,
        )
        ax.plot(
            part["strike"],
            100.0 * part["iv_mid"],
            "o",
            color=color,
            markersize=3,
            label=f"Selected OTM {kind}s",
        )

    ax.axvline(
        forward,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Forward",
    )
    ax.set_xlabel("Strike, index points")
    ax.set_ylabel("Implied volatility, %")
    ax.set_title("Observed SPX IVs and bid–ask intervals")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    carry_audit = json.loads(
        args.carry_audit.read_text(encoding="utf-8")
    )
    snapshot_hash = file_sha256(args.snapshot)

    if snapshot_hash != carry_audit["snapshot_sha256"]:
        raise ValueError(
            "The snapshot differs from the file used for carry estimation."
        )

    matching_fits = [
        fit
        for fit in carry_audit["fits"]
        if fit["mode"] == args.carry_mode
        and float(fit["window_pct"]) == args.carry_window_pct
    ]

    if len(matching_fits) != 1:
        raise ValueError(
            "The requested carry fit must match exactly one audit entry."
        )

    selected_fit = matching_fits[0]
    maturity = float(
        carry_audit["maturity_audit"]["maturity_years"]
    )

    pricer = BlackPricer(
        forward=float(selected_fit["forward"]),
        discount_factor=float(selected_fit["discount_factor"]),
        maturity=maturity,
    )
    builder = CalibrationQuoteBuilder(
        solver=ImpliedVolSolver(pricer=pricer),
        min_log_moneyness=args.y_min,
        max_log_moneyness=args.y_max,
    )

    source = pd.read_csv(args.snapshot)
    selection = builder.build(source)

    retained = (
        selection.loc[selection["eligible"]]
        .sort_values("strike")
        .reset_index(drop=True)
    )

    project_root = Path(__file__).resolve().parents[1]
    data_directory = (
        project_root / "data" / "processed" / "calibration"
    )
    output_directory = (
        project_root / "outputs" / "calibration_diagnostics"
    )

    data_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{args.snapshot.stem}_{args.carry_mode}_"
        f"{args.carry_window_pct:g}pct"
    )

    selection_path = data_directory / f"{stem}_selection.csv"
    quotes_path = data_directory / f"{stem}_quotes.csv"
    audit_path = output_directory / f"{stem}_audit.json"
    figure_path = output_directory / f"{stem}_ivs.png"

    selection.to_csv(selection_path, index=False)

    if retained.empty:
        raise ValueError(
            f"No quotes qualified. Inspect {selection_path}"
        )

    retained.to_csv(quotes_path, index=False)

    counts = selection["selection_reason"].value_counts()
    option_counts = retained["option_kind"].value_counts()

    native_error_columns = [
        f"native_{level}_repricing_error"
        for level in ("bid", "mid", "ask")
    ]
    call_error_columns = [
        f"call_{level}_repricing_error"
        for level in ("bid", "mid", "ask")
    ]

    max_native_error = float(
        retained[native_error_columns].abs().to_numpy().max()
    )
    max_call_error = float(
        retained[call_error_columns].abs().to_numpy().max()
    )

    widest = retained.loc[retained["iv_width_pp"].idxmax()]

    audit = {
        "snapshot_file": args.snapshot.name,
        "snapshot_sha256": snapshot_hash,
        "carry_audit_file": args.carry_audit.name,
        "carry_audit_sha256": file_sha256(args.carry_audit),
        "carry_audit": carry_audit,
        "selected_carry_fit": selected_fit,
        "maturity_years": maturity,
        "log_moneyness_domain": [args.y_min, args.y_max],
        "source_rows": len(source),
        "retained_rows": len(retained),
        "selection_counts": {
            str(reason): int(count)
            for reason, count in counts.items()
        },
        "option_counts": {
            str(kind): int(count)
            for kind, count in option_counts.items()
        },
        "strike_range": [
            float(retained["strike"].min()),
            float(retained["strike"].max()),
        ],
        "midpoint_iv_range_pct": [
            100.0 * float(retained["iv_mid"].min()),
            100.0 * float(retained["iv_mid"].max()),
        ],
        "maximum_iv_interval_pp": float(widest["iv_width_pp"]),
        "widest_interval_strike": float(widest["strike"]),
        "widest_interval_kind": str(widest["option_kind"]),
        "maximum_native_repricing_error_points": max_native_error,
        "maximum_call_repricing_error_points": max_call_error,
        "iv_units": "decimal annualised volatility",
        "price_units": "index points",
        "call_price_basis": (
            "Selected calls; selected puts translated using "
            "the recorded forward and discount factor."
        ),
    }

    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_smile(retained, pricer.forward, figure_path)

    print(f"Forward:          {pricer.forward:.9f}")
    print(f"Discount factor:  {pricer.discount_factor:.8f}")
    print(f"Maturity:         {maturity:.10f} years")
    print(f"Retained quotes:  {len(retained)} / {len(source)}")
    print(
        "Puts / calls:     "
        f"{int(option_counts.get('put', 0))} / "
        f"{int(option_counts.get('call', 0))}"
    )
    print(
        "Strike range:     "
        f"{retained['strike'].min():.2f} to "
        f"{retained['strike'].max():.2f}"
    )
    print(
        "Midpoint IV range: "
        f"{100.0 * retained['iv_mid'].min():.4f}% to "
        f"{100.0 * retained['iv_mid'].max():.4f}%"
    )
    print(
        "Widest IV interval: "
        f"{widest['iv_width_pp']:.4f} percentage points "
        f"at {widest['strike']:.2f} ({widest['option_kind']})"
    )
    print(f"Maximum native repricing error: {max_native_error:.3e}")
    print(f"Maximum call repricing error:   {max_call_error:.3e}")
    print(f"\nSelection counts: {audit['selection_counts']}")
    print(f"\nQuotes saved to:    {quotes_path}")
    print(f"Selection saved to: {selection_path}")
    print(f"Audit saved to:     {audit_path}")
    print(f"Figure saved to:    {figure_path}")


if __name__ == "__main__":
    main()