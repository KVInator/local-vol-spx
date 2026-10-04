"""Compare strike smoothing while preserving other calibration inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convex_spline import ConvexSplineCalibrator
from local_vol import DupireLocalVolatility
from vol_surface import NormalizedCallSurface


def refit_slices(baseline, quote_tables, weight_factor):
    models = []

    for original, quotes in zip(baseline.splines, quote_tables):
        calibrator = ConvexSplineCalibrator(
            pricer=original.pricer,
            n_intervals=original.n_intervals,
            smoothing_weight=original.smoothing_weight * weight_factor,
            band_slack=(
                original.band_cap
                - original.minimum_multiplier
                - 1e-6
            ),
            curvature_floor=original.curvature_floor,
        )
        fitted = calibrator.fit(
            quotes["strike"].to_numpy(),
            quotes["call_mid"].to_numpy(),
            quotes["call_half_width"].to_numpy(),
        )

        if abs(fitted.band_cap - original.band_cap) > 1e-6:
            raise RuntimeError(
                "The quote-band cap changed beyond numerical tolerance."
            )

        models.append(fitted)

    return NormalizedCallSurface(
        splines=tuple(models),
        min_log_moneyness=baseline.min_log_moneyness,
        max_log_moneyness=baseline.max_log_moneyness,
        calendar_tolerance=baseline.calendar_tolerance,
    )


def endpoint_grid(surface, log_moneyness):
    """Evaluate both time sides bounding every maturity interval."""
    model = DupireLocalVolatility(surface)
    z = np.exp(log_moneyness)
    frames = []

    for interval in range(len(surface.maturities) - 1):
        endpoints = (
            (interval, "right"),
            (interval + 1, "left"),
        )

        for pillar, side in endpoints:
            t = surface.maturities[pillar]
            spline = surface.splines[pillar]

            curvature = np.asarray(
                surface.normalized_call(z, t, derivative=2)
            )
            floor = (
                spline.curvature_floor
                * spline.pricer.forward
                / spline.pricer.discount_factor
            )

            frames.append(
                pd.DataFrame(
                    {
                        "pillar": pillar,
                        "maturity_years": t,
                        "elapsed_days": 365.0 * t,
                        "time_side": side,
                        "log_moneyness": log_moneyness,
                        "normalized_strike_curvature": curvature,
                        "at_curvature_floor": (
                            (floor > 0.0)
                            & (curvature <= floor * 1.0001)
                        ),
                        "local_volatility": (
                            model.normalized_volatility(z, t, side=side)
                        ),
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)


def summarize(case, surface, baseline, quote_tables, grid):
    count = sum(len(quotes) for quotes in quote_tables)

    weighted_squared_error = sum(
        len(quotes) * spline.rms_half_spreads**2
        for spline, quotes in zip(surface.splines, quote_tables)
    )

    price_changes = []
    for fitted, original, quotes in zip(
        surface.splines,
        baseline.splines,
        quote_tables,
    ):
        strikes = quotes["strike"].to_numpy()
        price_changes.append(
            float(
                np.max(
                    np.abs(
                        fitted.price(strikes)
                        - original.price(strikes)
                    )
                )
            )
        )

    largest_jump = 0.0
    for pillar in range(1, len(surface.maturities) - 1):
        group = grid.loc[grid["pillar"] == pillar]
        left = group.loc[
            group["time_side"] == "left", "local_volatility"
        ].to_numpy()
        right = group.loc[
            group["time_side"] == "right", "local_volatility"
        ].to_numpy()

        largest_jump = max(
            largest_jump,
            float(np.max(100.0 * np.abs(right - left))),
        )

    peak = grid.loc[grid["local_volatility"].idxmax()]

    return {
        "case": case,
        "status": "passed",
        "quotes": count,
        "rms_half_spreads": float(
            np.sqrt(weighted_squared_error / count)
        ),
        "max_half_spreads": float(
            max(spline.max_half_spreads for spline in surface.splines)
        ),
        "outside_bands": int(
            sum(spline.outside_bands for spline in surface.splines)
        ),
        "max_quote_price_change": max(price_changes),
        "max_lv_pct": float(100.0 * peak["local_volatility"]),
        "peak_days": float(peak["elapsed_days"]),
        "peak_log_moneyness": float(peak["log_moneyness"]),
        "peak_time_side": str(peak["time_side"]),
        "max_pillar_jump_pp": largest_jump,
        "floor_endpoint_samples": int(
            grid["at_curvature_floor"].sum()
        ),
    }


def save_candidate(surface, manifest, directory):
    directory.mkdir(parents=True, exist_ok=True)
    slice_files = []

    for spline, source_name in zip(
        surface.splines,
        manifest["slice_files"],
    ):
        relative_path = Path(source_name)
        destination = directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        spline.save(destination)
        slice_files.append(relative_path.as_posix())

    metadata = surface.metadata()
    metadata["slice_files"] = slice_files

    (directory / "call_surface.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, type=Path)
    args = parser.parse_args()

    model_path = args.surface.resolve()
    baseline = NormalizedCallSurface.load(model_path)
    manifest = json.loads(model_path.read_text())

    quote_paths = [
        model_path.parent / Path(name).parent / "quotes.csv"
        for name in manifest["slice_files"]
    ]
    quote_tables = [
        pd.read_csv(path, float_precision="round_trip")
        .sort_values("strike")
        .reset_index(drop=True)
        for path in quote_paths
    ]

    data_directory = (
        model_path.parent.parent
        / f"{model_path.parent.name}_smoothing_study"
    )
    output_directory = (
        Path("outputs/local_volatility_diagnostics")
        / model_path.parent.name
        / "smoothing_study"
    ).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    log_moneyness = np.linspace(
        baseline.min_log_moneyness,
        baseline.max_log_moneyness,
        2001,
    )

    cases = (
        ("current", 1.0),
        ("smoothing_x3", 3.0),
        ("smoothing_x10", 10.0),
    )
    records = []
    grids = {}

    for case, factor in cases:
        print(f"Evaluating {case}...", flush=True)

        try:
            surface = (
                baseline
                if factor == 1.0
                else refit_slices(baseline, quote_tables, factor)
            )
            grid = endpoint_grid(surface, log_moneyness)
            record = summarize(
                case,
                surface,
                baseline,
                quote_tables,
                grid,
            )
            record["weight_factor"] = factor

            candidate_directory = data_directory / case
            save_candidate(surface, manifest, candidate_directory)
            grid.to_csv(
                candidate_directory / "local_vol_endpoint_grid.csv",
                index=False,
            )
            grids[case] = grid
            records.append(record)

        except (ValueError, RuntimeError) as error:
            if factor == 1.0:
                raise

            records.append(
                {
                    "case": case,
                    "weight_factor": factor,
                    "status": "failed",
                    "message": str(error),
                }
            )
            print(f"  Failed: {error}", flush=True)

    summary = pd.DataFrame(records)
    summary_path = output_directory / "smoothing_summary.csv"
    summary.to_csv(summary_path, index=False)

    passed = summary.loc[summary["status"] == "passed"]

    print("\nQuote-fit comparison:")
    print(
        passed[
            [
                "case",
                "rms_half_spreads",
                "max_half_spreads",
                "outside_bands",
                "max_quote_price_change",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\nLocal-volatility comparison:")
    print(
        passed[
            [
                "case",
                "max_lv_pct",
                "peak_days",
                "peak_log_moneyness",
                "peak_time_side",
                "max_pillar_jump_pp",
                "floor_endpoint_samples",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for case, grid in grids.items():
        first_pillar = grid.loc[
            (grid["pillar"] == 0)
            & (grid["time_side"] == "right")
        ]
        axes[0].plot(
            first_pillar["log_moneyness"],
            100.0 * first_pillar["local_volatility"],
            label=case,
        )
        axes[1].plot(
            first_pillar["log_moneyness"],
            first_pillar["normalized_strike_curvature"],
            label=case,
        )

    axes[0].set_title("First-pillar local volatility")
    axes[0].set_ylabel("Annualized local volatility (%)")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].set_title("First-pillar normalized curvature")
    axes[1].set_ylabel("c_zz")
    axes[1].set_yscale("log")

    for axis in axes:
        axis.set_xlabel("Forward log-moneyness")
        axis.grid(alpha=0.25, which="both")

    fig.tight_layout()
    figure_path = output_directory / "smoothing_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    input_files = [
        model_path,
        *quote_paths,
        *[
            model_path.parent / name
            for name in manifest["slice_files"]
        ],
    ]
    audit = {
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_files
        },
        "weight_factors": [factor for _, factor in cases],
        "log_moneyness_points": len(log_moneyness),
        "maturity_sampling": "both sides bounding every interpolation interval",
        "held_fixed": [
            "market quotes",
            "forward prices",
            "discount factors",
            "spline intervals",
            "curvature floors",
            "quote-band caps within numerical tolerance",
            "surface domain",
        ],
        "results": records,
    }
    (output_directory / "smoothing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print(f"\nSummary: {summary_path}")
    print(f"Figure:  {figure_path}")
    print(f"Models:  {data_directory}")
    print("The primary calibrated surface has not been replaced.")


if __name__ == "__main__":
    main()