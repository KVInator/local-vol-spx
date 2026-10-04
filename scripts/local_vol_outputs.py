"""Produce raw Dupire local-volatility grids and diagnostics."""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local_vol import DupireLocalVolatility
from vol_surface import NormalizedCallSurface


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_local_volatility(
    surface_path: Path,
    output_directory: Path | None = None,
) -> dict:
    model_path = Path(surface_path).resolve()
    grid_path = model_path.parent / "interpolated_surface.csv"

    if output_directory is None:
        output_directory = (
            PROJECT_ROOT
            / "outputs/local_volatility_diagnostics"
            / model_path.parent.name
        )

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    surface = NormalizedCallSurface.load(model_path)
    local_vol = DupireLocalVolatility(surface)

    grid = pd.read_csv(grid_path, float_precision="round_trip")
    maturity = grid["maturity_years"].to_numpy()
    z = np.exp(grid["log_moneyness"].to_numpy())

    curvature = np.asarray(
        surface.normalized_call(z, maturity, derivative=2)
    )
    time_derivative = np.asarray(
        surface.normalized_time_derivative(
            z,
            maturity,
            side="right",
        )
    )
    variance = np.asarray(
        local_vol.normalized_variance(
            z,
            maturity,
            side="right",
        )
    )
    volatility = np.sqrt(variance)

    pillar_floors = np.array(
        [
            spline.curvature_floor
            * spline.pricer.forward
            / spline.pricer.discount_factor
            for spline in surface.splines
        ]
    )
    curvature_floor = np.interp(
        maturity,
        surface.maturities,
        pillar_floors,
    )
    floor_multiple = np.divide(
        curvature,
        curvature_floor,
        out=np.full_like(curvature, np.nan),
        where=curvature_floor > 0.0,
    )

    # This reports numerical floor contact, not model reliability.
    floor_contact = (
        (curvature_floor > 0.0)
        & (floor_multiple <= 1.0001)
    )

    result = grid[
        [
            "maturity_years",
            "elapsed_days",
            "log_moneyness",
            "strike",
            "implied_volatility",
        ]
    ].copy()

    result["normalized_strike_curvature"] = curvature
    result["normalized_time_derivative_right"] = time_derivative
    result["normalized_curvature_floor"] = curvature_floor
    result["curvature_floor_multiple"] = floor_multiple
    result["at_curvature_floor"] = floor_contact
    result["local_variance"] = variance
    result["local_volatility"] = volatility

    result_path = model_path.parent / "local_volatility_grid.csv"
    result.to_csv(result_path, index=False)

    pillar_mask = np.isclose(
        maturity[:, None],
        surface.maturities[None, :],
        rtol=0.0,
        atol=1e-14,
    ).any(axis=1)
    pillar_grid = result.loc[pillar_mask]

    summary_rows = []
    for t, group in pillar_grid.groupby("maturity_years", sort=True):
        summary_rows.append(
            {
                "days": 365.0 * t,
                "min_lv_pct": 100.0 * group["local_volatility"].min(),
                "median_lv_pct": (
                    100.0 * group["local_volatility"].median()
                ),
                "max_lv_pct": 100.0 * group["local_volatility"].max(),
                "min_curvature": (
                    group["normalized_strike_curvature"].min()
                ),
                "floor_points": int(
                    group["at_curvature_floor"].sum()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_directory / "pillar_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Use the finer grid from the smoothing study for pillar jumps.
    jump_y = np.linspace(
        surface.min_log_moneyness,
        surface.max_log_moneyness,
        2001,
    )
    query_z = np.exp(jump_y)

    jump_frames = []
    jump_summary_rows = []

    for t in surface.maturities[1:-1]:
        left = np.asarray(
            local_vol.normalized_volatility(
                query_z,
                t,
                side="left",
            )
        )
        right = np.asarray(
            local_vol.normalized_volatility(
                query_z,
                t,
                side="right",
            )
        )
        jump_pp = 100.0 * (right - left)
        largest = int(np.argmax(np.abs(jump_pp)))

        jump_frames.append(
            pd.DataFrame(
                {
                    "maturity_years": t,
                    "elapsed_days": 365.0 * t,
                    "log_moneyness": jump_y,
                    "left_local_volatility": left,
                    "right_local_volatility": right,
                    "jump_percentage_points": jump_pp,
                }
            )
        )
        jump_summary_rows.append(
            {
                "days": 365.0 * t,
                "max_abs_jump_pp": abs(jump_pp[largest]),
                "signed_jump_pp": jump_pp[largest],
                "log_moneyness_at_largest_jump": jump_y[largest],
            }
        )

    jumps = pd.concat(jump_frames, ignore_index=True)
    jumps_path = output_directory / "pillar_time_jumps.csv"
    jumps.to_csv(jumps_path, index=False)

    jump_summary = pd.DataFrame(jump_summary_rows)
    jump_summary_path = (
        output_directory / "pillar_time_jump_summary.csv"
    )
    jump_summary.to_csv(jump_summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for t, group in pillar_grid.groupby("maturity_years", sort=True):
        label = f"{365.0 * t:.0f} days"

        axes[0].plot(
            group["log_moneyness"],
            100.0 * group["local_volatility"],
            label=label,
        )
        axes[1].plot(
            group["log_moneyness"],
            group["normalized_strike_curvature"],
            label=label,
        )

    axes[0].set_title("Raw Dupire local volatility")
    axes[0].set_ylabel("Annualized local volatility (%)")

    if np.any(volatility == 0.0):
        axes[0].set_yscale("symlog", linthresh=1.0)
    else:
        axes[0].set_yscale("log")

    axes[0].legend()

    axes[1].set_title("Normalized strike curvature")
    axes[1].set_ylabel("c_zz")
    axes[1].set_yscale("log")

    for axis in axes:
        axis.set_xlabel("Forward log-moneyness")
        axis.grid(alpha=0.25, which="both")

    fig.tight_layout()
    figure_path = output_directory / "raw_local_volatility.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    percentiles = {
        str(p): float(np.percentile(100.0 * volatility, p))
        for p in (0, 50, 95, 99, 100)
    }
    maximum_index = int(np.argmax(volatility))

    manifest = json.loads(model_path.read_text(encoding="utf-8"))
    input_files = [
        model_path,
        grid_path,
        *[
            model_path.parent / name
            for name in manifest["slice_files"]
        ],
    ]

    audit_path = output_directory / "local_volatility_audit.json"
    audit = {
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_files
        },
        "evaluation_points": len(result),
        "time_derivative_side": (
            "right; available interior side at endpoints"
        ),
        "local_volatility_percentiles_pct": percentiles,
        "curvature_floor_points": int(floor_contact.sum()),
        "floor_contact_relative_tolerance": 1e-4,
        "pillar_jump_log_moneyness_points": len(jump_y),
        "maximum_location": {
            "days": float(maturity[maximum_index] * 365.0),
            "log_moneyness": float(
                grid["log_moneyness"].iloc[maximum_index]
            ),
            "local_volatility_pct": float(
                volatility[maximum_index] * 100.0
            ),
        },
        "values_modified": False,
        "output_files": {
            "grid": str(result_path),
            "pillar_summary": str(summary_path),
            "pillar_time_jumps": str(jumps_path),
            "pillar_time_jump_summary": str(jump_summary_path),
            "figure": str(figure_path),
            "audit": str(audit_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nRaw local volatility percentiles (%):")
    for percentile, value in percentiles.items():
        print(f"  {percentile:>3}: {value:.6f}")

    print("\nExpiry-pillar diagnostics:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print("\nOne-sided local-volatility jumps:")
    print(
        jump_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print(f"\nPillar-jump strike grid: {len(jump_y):,} points")
    print(f"Curvature-floor grid points: {int(floor_contact.sum())}")
    print(f"Grid:   {result_path}")
    print(f"Audit:  {audit_path}")
    print(f"Figure: {figure_path}")

    return audit