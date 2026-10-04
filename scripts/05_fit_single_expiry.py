"""Fit a constrained call-price spline to one calibration quote set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black import BlackPricer
from convex_spline import ConvexSplineCalibrator
from implied_vol import ImpliedVolSolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def unique_number(quotes: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(quotes[column], errors="raise")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{column} contains nonfinite values.")

    if values.nunique() != 1:
        raise ValueError(f"{column} must have one value.")

    return float(values.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", required=True)
    parser.add_argument("--intervals", type=int, default=30)
    parser.add_argument("--weight", type=float, default=1e-3)
    parser.add_argument("--band-slack", type=float, default=1.5)
    args = parser.parse_args()

    quotes_path = resolve_path(args.quotes)
    quotes = pd.read_csv(quotes_path)

    required = {
        "strike",
        "call_mid",
        "call_half_width",
        "forward",
        "discount_factor",
        "maturity_years",
        "eligible",
        "iv_bid",
        "iv_mid",
        "iv_ask",
    }
    missing = required.difference(quotes.columns)

    if missing:
        raise ValueError(
            f"Missing calibration columns: {sorted(missing)}"
        )

    if quotes.empty:
        raise ValueError("The calibration quote file is empty.")

    if not (
        quotes["eligible"]
        .astype(str)
        .str.lower()
        .eq("true")
        .all()
    ):
        raise ValueError(
            "Use the retained quotes file, not the full selection file."
        )

    quotes = quotes.sort_values(
        "strike",
        kind="stable",
    ).reset_index(drop=True)

    pricer = BlackPricer(
        forward=unique_number(quotes, "forward"),
        discount_factor=unique_number(
            quotes,
            "discount_factor",
        ),
        maturity=unique_number(quotes, "maturity_years"),
    )
    calibrator = ConvexSplineCalibrator(
        pricer=pricer,
        n_intervals=args.intervals,
        smoothing_weight=args.weight,
        band_slack=args.band_slack,
    )

    strikes = quotes["strike"].to_numpy(dtype=float)
    midpoints = quotes["call_mid"].to_numpy(dtype=float)
    half_widths = quotes["call_half_width"].to_numpy(dtype=float)

    fitted = calibrator.fit(
        strikes,
        midpoints,
        half_widths,
    )

    grid = np.linspace(
        fitted.strike_origin,
        fitted.strike_max,
        1001,
    )
    grid_prices = np.asarray(fitted.price(grid))
    iv_solver = ImpliedVolSolver(pricer)

    grid_ivs = np.array(
        [
            iv_solver.solve(
                price=float(price),
                strike=float(strike),
                kind="call",
            ).volatility
            for strike, price in zip(grid, grid_prices)
        ]
    )

    curve = pd.DataFrame(
        {
            "strike": grid,
            "log_moneyness": np.log(grid / pricer.forward),
            "call_price": grid_prices,
            "strike_slope": fitted.strike_slope(grid),
            "strike_curvature": fitted.strike_curvature(grid),
            "implied_volatility": grid_ivs,
        }
    )

    quote_diagnostics = quotes.copy()
    quote_diagnostics["fitted_call_price"] = fitted.price(strikes)
    quote_diagnostics["move_points"] = (
        quote_diagnostics["fitted_call_price"]
        - quote_diagnostics["call_mid"]
    )
    quote_diagnostics["move_half_spreads"] = (
        quote_diagnostics["move_points"]
        / quote_diagnostics["call_half_width"]
    )
    quote_diagnostics["outside_original_band"] = (
        quote_diagnostics["move_half_spreads"].abs()
        > 1.0 + 1e-7
    )

    stem = quotes_path.stem.removesuffix("_quotes")
    model_directory = (
        PROJECT_ROOT / "data/processed/splines"
    )
    output_directory = (
        PROJECT_ROOT / "outputs/spline_diagnostics"
    )
    model_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_path = model_directory / f"{stem}_spline.npz"
    curve_path = model_directory / f"{stem}_curve.csv"
    diagnostics_path = (
        output_directory / f"{stem}_quote_movements.csv"
    )
    report_path = output_directory / f"{stem}_audit.json"
    figure_path = output_directory / f"{stem}_fit.png"

    fitted.save(model_path)
    curve.to_csv(curve_path, index=False)
    quote_diagnostics.to_csv(diagnostics_path, index=False)

    with quotes_path.open("rb") as source:
        source_hash = hashlib.file_digest(
            source,
            "sha256",
        ).hexdigest()

    checks = fitted.shape_checks()
    report = {
        "quotes_file": quotes_path.name,
        "quotes_sha256": source_hash,
        "observations": len(quotes),
        "band_slack": args.band_slack,
        "model": fitted.metadata(),
        "shape_checks": checks,
        "quote_cap_passed": bool(
            fitted.max_half_spreads
            <= fitted.band_cap + 1e-6
        ),
        "domain_policy": "No strike extrapolation.",
        "curvature_units": "1 / index point",
        "roughness_definition": (
            "Integral of squared third derivative of "
            "u(x)=C(K(x))/strike_span over x in [0,1]."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].vlines(
        strikes,
        100.0 * quotes["iv_bid"],
        100.0 * quotes["iv_ask"],
        color="0.65",
        linewidth=1.0,
        label="Observed IV intervals",
    )
    axes[0].scatter(
        strikes,
        100.0 * quotes["iv_mid"],
        s=12,
        color="0.35",
        label="Observed midpoint IV",
    )
    axes[0].plot(
        grid,
        100.0 * grid_ivs,
        color="tab:blue",
        label="Fitted curve IV",
    )
    axes[0].set_ylabel("Implied volatility (%)")
    axes[0].set_title("Constrained single-expiry calibration")
    axes[0].legend()

    axes[1].plot(
        grid,
        curve["strike_curvature"],
        color="tab:blue",
    )
    axes[1].axhline(
        fitted.curvature_floor,
        color="tab:red",
        linestyle=":",
        label="Curvature floor",
    )
    axes[1].set_ylabel("Call curvature\n(1 / index point)")
    axes[1].legend()

    axes[2].scatter(
        strikes,
        quote_diagnostics["move_half_spreads"],
        s=14,
        color="tab:blue",
    )
    axes[2].axhspan(-1.0, 1.0, color="0.9", zorder=0)
    for boundary in (-fitted.band_cap, fitted.band_cap):
        axes[2].axhline(
            boundary,
            color="tab:red",
            linestyle=":",
        )
    axes[2].set_ylabel("Move from midpoint\n(half-spreads)")
    axes[2].set_xlabel("Strike")

    for axis in axes:
        axis.axvline(
            pricer.forward,
            color="0.5",
            linestyle="--",
            linewidth=0.8,
        )
        axis.grid(alpha=0.2)

    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Price observations: {len(quotes)}")
    print(f"Spline intervals:   {fitted.n_intervals}")
    print(f"Coefficients:       {len(fitted.coefficients)}")
    print(f"Fit iterations:     {fitted.iterations}")
    print()
    print(
        "Minimum multiplier: "
        f"{fitted.minimum_multiplier:.8f}"
    )
    print(f"Allowed band cap:   {fitted.band_cap:.8f}")
    print(
        "RMS half-spreads:   "
        f"{fitted.rms_half_spreads:.6f}"
    )
    print(
        "Max half-spreads:   "
        f"{fitted.max_half_spreads:.6f}"
    )
    print(f"Outside old bands:  {fitted.outside_bands}")
    print(f"Roughness:          {fitted.roughness:.6f}")
    print()
    print(
        "Minimum curvature:  "
        f"{checks['minimum_curvature']:.6e}"
    )
    print(f"Left slope:         {checks['left_slope']:.8f}")
    print(f"Right slope:        {checks['right_slope']:.8f}")
    print(f"Shape checks pass:  {checks['passed']}")
    print(f"Quote cap passes:   {report['quote_cap_passed']}")
    print()
    print(f"Model saved to: {model_path}")
    print(f"Curve saved to: {curve_path}")
    print(f"Audit saved to: {report_path}")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()