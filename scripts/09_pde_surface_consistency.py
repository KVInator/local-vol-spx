"""Check forward PDE propagation against a fitted call surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forward_pde import ForwardPDESolver
from local_vol import DupireLocalVolatility
from vol_surface import NormalizedCallSurface


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_case(
    surface,
    local_volatility,
    times,
    pillars,
    price_scales,
    intervals,
    max_time_step,
):
    solver = ForwardPDESolver(
        min_log_moneyness=surface.min_log_moneyness,
        max_log_moneyness=surface.max_log_moneyness,
        n_space_intervals=intervals,
        max_time_step=max_time_step,
        theta=0.5,
        rannacher_steps=0,
    )

    strikes = solver.normalized_strikes
    endpoint_strikes = strikes[[0, -1]]

    def local_variance(z, maturity, side):
        return local_volatility.normalized_variance(
            z,
            maturity,
            side=side,
        )

    def boundaries(maturity):
        values = surface.normalized_call(
            endpoint_strikes,
            maturity,
        )
        return float(values[0]), float(values[1])

    result = solver.solve(
        maturities=times,
        initial_prices=surface.normalized_call(
            strikes,
            float(times[0]),
        ),
        local_variance=local_variance,
        boundary_values=boundaries,
        time_breaks=pillars[1:-1],
    )

    target = np.vstack([
        surface.normalized_call(strikes, float(maturity))
        for maturity in times
    ])

    price_errors = (
        result.normalized_calls - target
    ) * price_scales[:, None]

    return result, target, price_errors


def sampled_shape_checks(prices, normalized_strikes):
    price_tolerance = 1e-10
    slope_tolerance = 1e-8

    widths = np.diff(normalized_strikes)
    price_changes = np.diff(prices, axis=1)
    slopes = price_changes / widths[None, :]

    left_weights = (
        widths[1:] / (widths[:-1] + widths[1:])
    )

    butterflies = (
        left_weights[None, :] * prices[:, :-2]
        + (1.0 - left_weights)[None, :] * prices[:, 2:]
        - prices[:, 1:-1]
    )

    return {
        "increasing_price_intervals": int(np.sum(
            price_changes > price_tolerance
        )),
        "vertical_spread_bound_violations": int(np.sum(
            (slopes < -1.0 - slope_tolerance)
            | (slopes > slope_tolerance)
        )),
        "negative_butterflies": int(np.sum(
            butterflies < -price_tolerance
        )),
        "normalized_price_tolerance": price_tolerance,
        "normalized_slope_tolerance": slope_tolerance,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Propagate normalized calls using Dupire local variance "
            "and compare with the fitted surface."
        )
    )
    parser.add_argument(
        "--surface",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--space-intervals",
        type=int,
        nargs="+",
        default=[100, 200, 400, 800],
    )
    parser.add_argument(
        "--steps-per-day",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    if args.steps_per_day <= 0:
        parser.error("--steps-per-day must be positive.")

    intervals_to_run = sorted(set(args.space_intervals))

    if any(intervals < 4 for intervals in intervals_to_run):
        parser.error("Spatial interval counts must be at least four.")

    surface_path = args.surface.expanduser()
    if not surface_path.is_absolute():
        surface_path = PROJECT_ROOT / surface_path
    surface_path = surface_path.resolve()

    surface = NormalizedCallSurface.load(surface_path)
    local_volatility = DupireLocalVolatility(surface)

    pillars = np.asarray(surface.maturities, dtype=float)
    times = np.unique(np.concatenate([
        np.linspace(pillars[0], pillars[-1], 41),
        pillars,
    ]))

    forwards = np.array([
        surface.forward(float(maturity))
        for maturity in times
    ])
    discount_factors = np.array([
        surface.discount_factor(float(maturity))
        for maturity in times
    ])
    price_scales = forwards * discount_factors

    max_time_step = 1.0 / (
        365.0 * args.steps_per_day
    )

    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "pde_diagnostics"
        / surface_path.parent.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    convergence_rows = []
    previous_error = None
    previous_spacing = None

    for intervals in intervals_to_run:
        print(f"Solving with {intervals} spatial intervals...")

        result, target, errors = run_case(
            surface=surface,
            local_volatility=local_volatility,
            times=times,
            pillars=pillars,
            price_scales=price_scales,
            intervals=intervals,
            max_time_step=max_time_step,
        )

        # Exclude the imposed initial curve and boundary prices.
        comparison_errors = errors[1:, 1:-1]

        maximum_error = float(np.max(
            np.abs(comparison_errors)
        ))
        rms_error = float(np.sqrt(
            np.mean(comparison_errors**2)
        ))
        spacing = float(
            result.log_moneyness[1]
            - result.log_moneyness[0]
        )

        order = None
        if (
            previous_error is not None
            and previous_error > 0.0
            and maximum_error > 0.0
        ):
            order = float(
                np.log(previous_error / maximum_error)
                / np.log(previous_spacing / spacing)
            )

        convergence_rows.append({
            "space_intervals": int(intervals),
            "log_spacing": spacing,
            "actual_time_steps": int(result.time_steps),
            "max_price_error_points": maximum_error,
            "rms_price_error_points": rms_error,
            "observed_space_order": order,
        })

        previous_error = maximum_error
        previous_spacing = spacing

        finest_result = result
        finest_target = target
        finest_errors = errors

    convergence = pd.DataFrame(convergence_rows)

    maturity_summary = pd.DataFrame({
        "maturity_years": times,
        "days": 365.0 * times,
        "expiry_pillar": np.isin(times, pillars),
        "max_price_error_points": np.max(
            np.abs(finest_errors[:, 1:-1]),
            axis=1,
        ),
        "rms_price_error_points": np.sqrt(
            np.mean(finest_errors[:, 1:-1]**2, axis=1)
        ),
    })

    y = finest_result.log_moneyness
    z = np.exp(y)

    time_grid, y_grid = np.meshgrid(
        times,
        y,
        indexing="ij",
    )

    fitted_prices = finest_target * price_scales[:, None]
    pde_prices = (
        finest_result.normalized_calls
        * price_scales[:, None]
    )

    comparison_grid = pd.DataFrame({
        "maturity_years": time_grid.ravel(),
        "elapsed_days": (365.0 * time_grid).ravel(),
        "log_moneyness": y_grid.ravel(),
        "normalized_strike": np.exp(y_grid).ravel(),
        "strike": (
            forwards[:, None] * z[None, :]
        ).ravel(),
        "pde_normalized_call": (
            finest_result.normalized_calls.ravel()
        ),
        "fitted_normalized_call": finest_target.ravel(),
        "pde_call_price": pde_prices.ravel(),
        "fitted_call_price": fitted_prices.ravel(),
        "price_error_points": finest_errors.ravel(),
    })

    shape_checks = sampled_shape_checks(
        finest_result.normalized_calls,
        z,
    )

    convergence.to_csv(
        output_directory / "space_convergence.csv",
        index=False,
    )
    maturity_summary.to_csv(
        output_directory / "maturity_errors.csv",
        index=False,
    )
    comparison_grid.to_csv(
        output_directory / "finest_comparison_grid.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    pillar_indices = np.flatnonzero(
        np.isin(times, pillars)
    )

    for index in pillar_indices[1:]:
        axes[0, 0].plot(
            y,
            finest_errors[index],
            label=f"{365.0 * times[index]:g} days",
        )

    axes[0, 0].axhline(
        0.0,
        color="black",
        linewidth=0.7,
    )
    axes[0, 0].set(
        title="Expiry-pillar pricing errors",
        xlabel="Forward log-moneyness",
        ylabel="PDE minus fitted price, index points",
    )
    axes[0, 0].legend()

    axes[0, 1].loglog(
        convergence["log_spacing"],
        convergence["max_price_error_points"],
        marker="o",
    )
    axes[0, 1].set(
        title="Spatial refinement",
        xlabel="Log-moneyness spacing",
        ylabel="Maximum price error, index points",
    )

    colour_limit = max(
        float(np.max(np.abs(finest_errors))),
        np.finfo(float).eps,
    )
    heatmap = axes[1, 0].pcolormesh(
        y,
        365.0 * times,
        finest_errors,
        shading="auto",
        cmap="coolwarm",
        vmin=-colour_limit,
        vmax=colour_limit,
    )
    axes[1, 0].set(
        title="Pricing error across the supported domain",
        xlabel="Forward log-moneyness",
        ylabel="Maturity, days",
    )
    fig.colorbar(
        heatmap,
        ax=axes[1, 0],
        label="Index points",
    )

    near_forward = int(np.argmin(np.abs(y)))

    axes[1, 1].plot(
        365.0 * times,
        fitted_prices[:, near_forward],
        label="Fitted surface",
    )
    axes[1, 1].plot(
        365.0 * times,
        pde_prices[:, near_forward],
        linestyle="--",
        label="PDE",
    )
    axes[1, 1].set(
        title=f"Call prices at log-moneyness {y[near_forward]:.4f}",
        xlabel="Maturity, days",
        ylabel="Call price, index points",
    )
    axes[1, 1].legend()

    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        axis.grid(True, alpha=0.25)

    figure_path = output_directory / "surface_consistency.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    manifest = json.loads(
        surface_path.read_text(encoding="utf-8")
    )

    input_files = [surface_path]
    input_files.extend(
        surface_path.parent / filename
        for filename in manifest["slice_files"]
    )
    input_files.extend([
        PROJECT_ROOT / "src" / "forward_pde.py",
        PROJECT_ROOT / "src" / "local_vol.py",
        PROJECT_ROOT / "src" / "vol_surface.py",
        Path(__file__).resolve(),
    ])

    row, column = np.unravel_index(
        np.argmax(np.abs(finest_errors[1:, 1:-1])),
        finest_errors[1:, 1:-1].shape,
    )
    row += 1
    column += 1

    audit = {
        "input_sha256": {
            str(path): sha256_file(path)
            for path in input_files
        },
        "validation_scope": (
            "Bounded-domain propagation from the first fitted "
            "maturity to the last fitted maturity."
        ),
        "initial_condition": (
            "Fitted normalized call curve at the first maturity."
        ),
        "boundary_conditions": (
            "Fitted normalized call prices at both spatial boundaries."
        ),
        "local_variance_source": (
            "Dupire variance from the loaded call surface."
        ),
        "initial_days": float(365.0 * times[0]),
        "final_days": float(365.0 * times[-1]),
        "evaluation_maturities": int(len(times)),
        "log_moneyness_domain": [
            float(surface.min_log_moneyness),
            float(surface.max_log_moneyness),
        ],
        "internal_time_breaks_years": pillars[1:-1].tolist(),
        "max_time_step_years": max_time_step,
        "theta": 0.5,
        "rannacher_nominal_steps": 0,
        "time_derivative_sides": {
            "old_time": "right",
            "new_time": "left",
        },
        "error_comparison": (
            "PDE prices against fitted surface prices, excluding "
            "the imposed initial curve and spatial boundaries."
        ),
        "space_convergence": convergence_rows,
        "space_errors_decrease": bool(np.all(
            np.diff(
                convergence["max_price_error_points"].to_numpy()
            ) < 0.0
        )),
        "maximum_error_location": {
            "days": float(365.0 * times[row]),
            "log_moneyness": float(y[column]),
            "signed_price_error_points": float(
                finest_errors[row, column]
            ),
        },
        "sampled_grid_shape_checks": shape_checks,
        "prices_modified_after_solve": False,
        "extrapolation_used": False,
    }

    audit_path = output_directory / "consistency_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nSpatial convergence:")
    print(convergence.to_string(
        index=False,
        float_format=lambda value: f"{value:.6e}",
    ))

    print("\nExpiry-pillar errors, finest spatial grid:")
    print(
        maturity_summary.loc[
            maturity_summary["expiry_pillar"],
            [
                "days",
                "max_price_error_points",
                "rms_price_error_points",
            ],
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6e}",
        )
    )

    print("\nSampled shape checks, finest PDE grid:")
    for name in (
        "increasing_price_intervals",
        "vertical_spread_bound_violations",
        "negative_butterflies",
    ):
        print(f"{name}: {shape_checks[name]}")

    print(
        "\nSpatial errors decrease:",
        audit["space_errors_decrease"],
    )
    print(
        "Largest error location:",
        audit["maximum_error_location"],
    )

    print("\nTables:", output_directory)
    print("Figure:", figure_path)
    print("Audit: ", audit_path)
    print(
        "\nThe first maturity and both spatial boundaries "
        "are supplied by the fitted surface."
    )


if __name__ == "__main__":
    main()