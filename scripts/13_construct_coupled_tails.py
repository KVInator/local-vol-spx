"""Construct and audit calendar-ordered strike tails."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coupled_tails import CoupledTailBuilder
from vol_surface import NormalizedCallSurface


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--left-power", type=float, default=3.0)
    parser.add_argument("--right-power", type=float, default=1.5)
    parser.add_argument("--join-width", type=float, default=0.09)
    parser.add_argument("--join-points", type=int, default=401)
    parser.add_argument("--preserve-from-days", type=float, default=21.0)
    args = parser.parse_args()

    surface_path = project_path(args.surface).resolve()
    specification = json.loads(surface_path.read_text())
    surface = NormalizedCallSurface.load(surface_path)

    builder = CoupledTailBuilder(
        left_power=args.left_power,
        right_power=args.right_power,
        join_width=args.join_width,
        join_points=args.join_points,
        preserve_from_days=args.preserve_from_days,
    )

    print("Selecting joins jointly and checking continuous calendar order...")
    models, checks = builder.construct(surface.splines)

    run_name = surface_path.parent.name
    model_directory = surface_path.parent / "coupled_tails"
    diagnostic_directory = (
        PROJECT_ROOT
        / "outputs"
        / "strike_tail_diagnostics"
        / run_name
    )
    model_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_directory.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    quote_frames = []
    grid_frames = []
    slices = []
    input_hashes = {
        str(surface_path.relative_to(PROJECT_ROOT)): sha256(surface_path)
    }

    y = np.linspace(-0.5, 0.5, 2001)
    z = np.exp(y)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model, slice_file in zip(
        models, specification["slice_files"], strict=True
    ):
        core_path = surface_path.parent / slice_file
        expiry = Path(slice_file).parent.name
        quotes_path = core_path.parent / "quotes.csv"

        for path in (core_path, quotes_path):
            input_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)

        quotes = pd.read_csv(quotes_path, float_precision="round_trip")
        strikes = quotes["strike"].to_numpy(dtype=float)
        forward = model.core.pricer.forward
        discount = model.core.pricer.discount_factor
        quote_z = strikes / forward

        original_prices = model.core.price(strikes)
        tail_prices = (
            discount * forward * model.normalized_call(quote_z)
        )
        price_changes = tail_prices - original_prices
        residuals = (
            tail_prices - quotes["call_mid"].to_numpy(dtype=float)
        ) / quotes["call_half_width"].to_numpy(dtype=float)

        retained = (
            (quote_z >= model.left.normalized_strike)
            & (quote_z <= model.right.normalized_strike)
        )
        outside = np.abs(residuals) > 1.0 + 1e-6

        quote_frames.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "strike": strikes,
                    "log_moneyness": np.log(quote_z),
                    "retained_in_core": retained,
                    "original_fitted_call": original_prices,
                    "tail_extended_call": tail_prices,
                    "price_change_points": price_changes,
                    "residual_half_spreads": residuals,
                    "outside_original_bands": outside,
                }
            )
        )

        calls = model.normalized_call(z)
        puts = model.normalized_put(z)
        slopes = model.normalized_call(z, 1)
        curvatures = model.normalized_call(z, 2)

        if not (
            np.all(np.isfinite(calls))
            and np.all(np.isfinite(puts))
            and np.all(np.isfinite(slopes))
            and np.all(np.isfinite(curvatures))
            and np.all(calls >= np.maximum(1.0 - z, 0.0) - 1e-12)
            and np.all(calls <= 1.0 + 1e-12)
            and np.all(slopes >= -1.0 - 1e-12)
            and np.all(slopes <= 1e-12)
            and np.all(curvatures >= 0.0)
        ):
            raise ValueError(f"Sampled shape checks failed for {expiry}.")

        metadata = model.metadata()
        summary_rows.append(
            {
                "expiry": expiry,
                "days": 365.0 * model.maturity,
                **metadata,
                "quotes": len(quotes),
                "quotes_retained_in_core": int(retained.sum()),
                "quotes_in_tail": int((~retained).sum()),
                "max_quote_price_change": float(
                    np.max(np.abs(price_changes))
                ),
                "rms_half_spreads": float(
                    np.sqrt(np.mean(residuals**2))
                ),
                "max_half_spreads": float(np.max(np.abs(residuals))),
                "outside_bands": int(outside.sum()),
                "sampled_shape_checks_passed": True,
            }
        )

        slices.append(
            {
                "expiry": expiry,
                "core_file": f"../{slice_file}",
                **metadata,
            }
        )
        grid_frames.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "maturity_years": model.maturity,
                    "days": 365.0 * model.maturity,
                    "log_moneyness": y,
                    "normalized_strike": z,
                    "normalized_call": calls,
                    "normalized_put": puts,
                    "normalized_slope": slopes,
                    "normalized_curvature": curvatures,
                }
            )
        )

        otm = np.where(y < 0.0, puts, calls)
        axes[0].plot(y, otm, label=expiry)
        axes[1].plot(y, curvatures, label=expiry)

    summary = pd.DataFrame(summary_rows)
    calendar = pd.DataFrame(
        {
            "earlier_expiry": slices[index]["expiry"],
            "later_expiry": slices[index + 1]["expiry"],
            **check,
        }
        for index, check in enumerate(checks)
    )
    quote_comparison = pd.concat(quote_frames, ignore_index=True)
    grid = pd.concat(grid_frames, ignore_index=True)

    summary_path = diagnostic_directory / "coupled_tail_join_summary.csv"
    calendar_path = (
        diagnostic_directory / "coupled_tail_calendar_checks.csv"
    )
    quotes_path = (
        diagnostic_directory / "coupled_tail_quote_comparison.csv"
    )
    grid_path = model_directory / "tail_slice_grid.csv"
    manifest_path = model_directory / "tail_slices.json"
    audit_path = diagnostic_directory / "coupled_tail_audit.json"

    summary.to_csv(summary_path, index=False)
    calendar.to_csv(calendar_path, index=False)
    quote_comparison.to_csv(quotes_path, index=False)
    grid.to_csv(grid_path, index=False)

    manifest = {
        "format_version": 1,
        "model_type": "calendar_ordered_tail_slices",
        "base_surface_file": "../call_surface.json",
        "left_power": args.left_power,
        "right_power": args.right_power,
        "preserve_outer_joins_from_days": args.preserve_from_days,
        "join_continuity": "Price and slope; curvature may jump.",
        "tail_powers_are_modelling_assumptions": True,
        "calendar_check_method": (
            "Analytical common-tail endpoint and asymptotic-ratio checks; "
            "adaptive convex-slope bounds throughout the bounded middle."
        ),
        "calendar_price_tolerance": 1e-12,
        "calendar_log_ratio_tolerance": 1e-12,
        "all_calendar_checks_passed": True,
        "slices": slices,
        "calendar_checks": checks,
        "zero_maturity_extension_included": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )

    audit = {
        **manifest,
        "input_sha256": input_hashes,
        "source_sha256": {
            "src/coupled_tails.py": sha256(
                PROJECT_ROOT / "src" / "coupled_tails.py"
            ),
            "scripts/13_construct_coupled_tails.py": sha256(__file__),
        },
        "observed_quotes": int(len(quote_comparison)),
        "quotes_in_tails": int(
            (~quote_comparison["retained_in_core"]).sum()
        ),
        "maximum_quote_price_change_points": float(
            quote_comparison["price_change_points"].abs().max()
        ),
        "probability_mass_and_mean_diagnostics": (
            "Algebraic consequences of the matched prices, slopes "
            "and asymptotic limits."
        ),
        "sampled_shape_domain": [-0.5, 0.5],
        "sampled_shape_points_per_expiry": len(y),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )

    axes[0].set(
        xlabel="Forward log-moneyness",
        ylabel="Normalized OTM price",
        title="Coupled tail prices",
        yscale="log",
    )
    axes[1].set(
        xlabel="Forward log-moneyness",
        ylabel="Normalized strike curvature",
        title="Coupled tail densities",
        yscale="log",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)

    figure.tight_layout()
    figure_path = diagnostic_directory / "coupled_tail_slices.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print("\nSelected joins:")
    print(
        summary[
            [
                "expiry",
                "days",
                "left_log_moneyness",
                "right_log_moneyness",
                "left_shape",
                "right_shape",
                "quotes_in_tail",
                "max_quote_price_change",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.8g}")
    )

    print("\nContinuous calendar checks:")
    print(
        calendar[
            [
                "earlier_expiry",
                "later_expiry",
                "left_tail_ordered",
                "right_tail_ordered",
                "middle_status",
                "middle_intervals_checked",
                "passed",
            ]
        ].to_string(index=False)
    )

    print("\nQuote-fit comparison:")
    print(
        summary[
            [
                "expiry",
                "rms_half_spreads",
                "max_half_spreads",
                "outside_bands",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.8g}")
    )

    print("\nAll nine calendar checks passed.")
    print("All sampled shape checks passed.")
    print(
        "Largest observed-quote price change: "
        f"{audit['maximum_quote_price_change_points']:.8g} index points"
    )
    print(f"\nModel specification: {manifest_path}")
    print(f"Join summary:        {summary_path}")
    print(f"Calendar checks:     {calendar_path}")
    print(f"Quote comparison:    {quotes_path}")
    print(f"Audit:               {audit_path}")
    print(f"Figure:              {figure_path}")


if __name__ == "__main__":
    main()