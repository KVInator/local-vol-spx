"""Generate saved models and diagnostics for the interpolated surface."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black import BlackPricer
from vol_surface import NormalizedCallSurface


MATURITY_MERGE_TOLERANCE_YEARS = 1e-12


def _evaluation_maturities(pillars, maturity_count):
    """Combine regular evaluation times with exact expiry pillars."""
    regular_times = np.linspace(
        pillars[0],
        pillars[-1],
        maturity_count,
    )

    near_pillar = np.any(
        np.isclose(
            regular_times[:, None],
            pillars[None, :],
            rtol=0.0,
            atol=MATURITY_MERGE_TOLERANCE_YEARS,
        ),
        axis=1,
    )

    return np.sort(
        np.concatenate(
            [
                pillars,
                regular_times[~near_pillar],
            ]
        )
    )


def build_interpolated_surface(
    results,
    grid_settings,
    data_directory,
    diagnostic_directory,
):
    data_directory = Path(data_directory)
    diagnostic_directory = Path(diagnostic_directory)
    splines = tuple(result.spline for result in results)

    lower = max(
        float(grid_settings["min_log_moneyness"]),
        max(
            np.log(model.strike_origin / model.pricer.forward)
            for model in splines
        ),
    )
    upper = min(
        float(grid_settings["max_log_moneyness"]),
        min(
            np.log(model.strike_max / model.pricer.forward)
            for model in splines
        ),
    )

    surface = NormalizedCallSurface(
        splines=splines,
        min_log_moneyness=lower,
        max_log_moneyness=upper,
    )

    calendar = pd.DataFrame(surface.calendar_checks())
    calendar_path = (
        diagnostic_directory / "continuous_calendar_checks.csv"
    )
    calendar.to_csv(calendar_path, index=False)

    print("\nContinuous calendar checks:")
    print(calendar.to_string(index=False))

    point_count = int(grid_settings["points"])
    maturity_count = int(grid_settings.get("maturity_points", 41))

    if point_count < 3 or maturity_count < 2:
        raise ValueError("Insufficient surface evaluation points.")

    pillars = surface.maturities
    times = _evaluation_maturities(pillars, maturity_count)

    y = np.linspace(lower, upper, point_count)
    z = np.exp(y)

    frames = []
    iv_rows = []

    for time in times:
        forward = surface.forward(time)
        discount = surface.discount_factor(time)
        strikes = forward * z

        prices = surface.call_price(strikes, time)
        volatility = surface.implied_volatility(strikes, time)
        normalized_prices = surface.normalized_call(z, time)
        normalized_curvature = surface.normalized_call(
            z, time, derivative=2
        )
        time_derivative = surface.normalized_time_derivative(z, time)

        pricer = BlackPricer(forward, discount, float(time))
        repricing_error = (
            pricer.price(strikes, volatility, "call") - prices
        )

        frames.append(
            pd.DataFrame(
                {
                    "maturity_years": float(time),
                    "elapsed_days": float(365.0 * time),
                    "log_moneyness": y,
                    "strike": strikes,
                    "forward": forward,
                    "discount_factor": discount,
                    "call_price": prices,
                    "normalized_call_price": normalized_prices,
                    "normalized_strike_curvature": normalized_curvature,
                    "normalized_time_derivative_right": time_derivative,
                    "strike_curvature": surface.strike_curvature(
                        strikes, time
                    ),
                    "implied_volatility": volatility,
                    "total_variance": volatility**2 * time,
                    "iv_repricing_error": repricing_error,
                }
            )
        )
        iv_rows.append(volatility)

    grid = pd.concat(frames, ignore_index=True)
    grid_path = data_directory / "interpolated_surface.csv"
    grid.to_csv(grid_path, index=False)

    iv_matrix = np.stack(iv_rows)
    np.savez_compressed(
        data_directory / "interpolated_surface_grid.npz",
        log_moneyness=y,
        maturity_years=times,
        implied_volatility=iv_matrix,
        total_variance=iv_matrix**2 * times[:, None],
    )

    specification = surface.metadata()
    specification["slice_files"] = [
        f"{result.expiry_date.isoformat()}/spline.npz"
        for result in results
    ]

    manifest_path = data_directory / "call_surface.json"
    manifest_path.write_text(
        json.dumps(specification, indent=2) + "\n",
        encoding="utf-8",
    )

    maximum_repricing_error = float(
        grid["iv_repricing_error"].abs().max()
    )

    y_grid, day_grid = np.meshgrid(y, 365.0 * times)
    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")

    plotted = axis.plot_surface(
        y_grid,
        day_grid,
        100.0 * iv_matrix,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=len(times),
        ccount=len(y),
    )
    axis.set_xlabel("Forward log-moneyness")
    axis.set_ylabel("Days to model fixing")
    axis.set_zlabel("Implied volatility (%)")
    axis.set_title(
        "Interpolated implied-volatility surface\n"
        "Linear interpolation of normalized call prices"
    )
    figure.colorbar(
        plotted,
        ax=axis,
        shrink=0.65,
        pad=0.10,
        label="Implied volatility (%)",
    )

    figure_path = diagnostic_directory / "iv_surface.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    audit = {
        **surface.metadata(),
        "evaluation_maturities": int(len(times)),
        "evaluation_points_per_maturity": int(len(y)),
        "evaluation_points": int(len(grid)),
        "regular_maturity_points_requested": maturity_count,
        "maturity_merge_tolerance_years": (
            MATURITY_MERGE_TOLERANCE_YEARS
        ),
        "maturity_grid_policy": (
            "Preserve exact expiry pillars; omit regular evaluation "
            "times within the absolute merge tolerance of a pillar."
        ),
        "minimum_normalized_strike_curvature": float(
            grid["normalized_strike_curvature"].min()
        ),
        "minimum_normalized_time_derivative": float(
            grid["normalized_time_derivative_right"].min()
        ),
        "maximum_iv_repricing_error": maximum_repricing_error,
    }

    audit_path = (
        diagnostic_directory / "interpolated_surface_audit.json"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nInterpolated surface:")
    print(f"Expiry pillars:           {len(pillars)}")
    print(f"Evaluation maturities:    {len(times)}")
    print(f"Evaluation points:        {len(grid):,}")
    print(f"Log-moneyness domain:     [{lower:.4f}, {upper:.4f}]")
    print(
        "Maximum IV repricing error: "
        f"{maximum_repricing_error:.3e} index points"
    )
    print(f"\nSurface model: {manifest_path}")
    print(f"Surface grid:  {grid_path}")
    print(f"Audit:         {audit_path}")
    print(f"Figure:        {figure_path}")

    return audit