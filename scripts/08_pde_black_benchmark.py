"""Report spatial and temporal convergence of the forward PDE."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black import BlackPricer
from forward_pde import ForwardPDESolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VOLATILITY = 0.20
MATURITY = 0.50

# Scaling for readable benchmark price errors.
BENCHMARK_FORWARD = 100.0
BENCHMARK_DISCOUNT_FACTOR = 0.98
PRICE_SCALE = BENCHMARK_FORWARD * BENCHMARK_DISCOUNT_FACTOR


def normalized_black(strikes, maturity):
    pricer = BlackPricer(
        forward=1.0,
        discount_factor=1.0,
        maturity=float(maturity),
    )

    return np.asarray(
        pricer.price(
            strikes,
            VOLATILITY,
            kind="call",
        ),
        dtype=float,
    )


def run_solver(intervals, max_time_step):
    solver = ForwardPDESolver(
        min_log_moneyness=-1.0,
        max_log_moneyness=1.0,
        n_space_intervals=intervals,
        max_time_step=max_time_step,
        theta=0.5,
        rannacher_steps=2,
    )

    strikes = solver.normalized_strikes
    endpoint_strikes = strikes[[0, -1]]

    def boundaries(maturity):
        prices = normalized_black(endpoint_strikes, maturity)
        return float(prices[0]), float(prices[1])

    result = solver.solve(
        maturities=[0.0, MATURITY],
        initial_prices=np.maximum(1.0 - strikes, 0.0),
        local_variance=lambda z, t, side: VOLATILITY**2,
        boundary_values=boundaries,
    )

    return result


def main():
    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "pde_diagnostics"
        / "black_benchmark"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    space_rows = []
    space_results = []
    previous_error = None
    previous_spacing = None

    for intervals in (100, 200, 400, 800):
        result = run_solver(
            intervals=intervals,
            max_time_step=MATURITY / 2000,
        )

        strikes = np.exp(result.log_moneyness)
        analytical = normalized_black(strikes, MATURITY)
        error = result.normalized_calls[-1] - analytical

        central = np.abs(result.log_moneyness) <= 0.25
        central_error = error[central]
        spacing = 2.0 / intervals
        maximum_error = float(np.max(np.abs(central_error)))

        order = np.nan
        if previous_error is not None:
            order = (
                np.log(previous_error / maximum_error)
                / np.log(previous_spacing / spacing)
            )

        space_rows.append({
            "space_intervals": intervals,
            "log_spacing": spacing,
            "actual_time_steps": result.time_steps,
            "max_price_error_points": PRICE_SCALE * maximum_error,
            "rms_price_error_points": PRICE_SCALE * float(
                np.sqrt(np.mean(central_error**2))
            ),
            "observed_space_order": order,
        })
        space_results.append((intervals, result, error))

        previous_error = maximum_error
        previous_spacing = spacing

    space_summary = pd.DataFrame(space_rows)

    reference = run_solver(
        intervals=800,
        max_time_step=MATURITY / 2048,
    )
    central = np.abs(reference.log_moneyness) <= 0.25
    reference_prices = reference.normalized_calls[-1]
    analytical = normalized_black(
        np.exp(reference.log_moneyness),
        MATURITY,
    )

    time_rows = []
    previous_error = None
    previous_step = None

    for nominal_steps in (16, 32, 64, 128):
        max_time_step = MATURITY / nominal_steps

        result = run_solver(
            intervals=800,
            max_time_step=max_time_step,
        )

        reference_error = (
            result.normalized_calls[-1] - reference_prices
        )
        black_error = (
            result.normalized_calls[-1] - analytical
        )

        maximum_error = float(np.max(np.abs(
            reference_error[central]
        )))

        order = np.nan
        if previous_error is not None:
            order = (
                np.log(previous_error / maximum_error)
                / np.log(previous_step / max_time_step)
            )

        time_rows.append({
            "nominal_time_steps": nominal_steps,
            "max_time_step_years": max_time_step,
            "actual_time_steps": result.time_steps,
            "max_vs_reference_points": PRICE_SCALE * maximum_error,
            "max_black_error_points": PRICE_SCALE * float(
                np.max(np.abs(black_error[central]))
            ),
            "observed_time_order": order,
        })

        previous_error = maximum_error
        previous_step = max_time_step

    time_summary = pd.DataFrame(time_rows)

    _, finest, finest_error = space_results[-1]
    finest_strikes = np.exp(finest.log_moneyness)
    finest_analytical = normalized_black(
        finest_strikes,
        MATURITY,
    )

    finest_grid = pd.DataFrame({
        "log_moneyness": finest.log_moneyness,
        "normalized_strike": finest_strikes,
        "normalized_pde_call": finest.normalized_calls[-1],
        "normalized_black_call": finest_analytical,
        "price_error_points": PRICE_SCALE * finest_error,
    })

    space_summary.to_csv(
        output_directory / "space_convergence.csv",
        index=False,
    )
    time_summary.to_csv(
        output_directory / "time_convergence.csv",
        index=False,
    )
    finest_grid.to_csv(
        output_directory / "finest_grid.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    central = np.abs(finest.log_moneyness) <= 0.25

    axes[0, 0].plot(
        finest.log_moneyness[central],
        PRICE_SCALE * finest_analytical[central],
        label="Analytical Black",
    )
    axes[0, 0].plot(
        finest.log_moneyness[central],
        PRICE_SCALE * finest.normalized_calls[-1, central],
        linestyle="--",
        label="PDE: 800 intervals",
    )
    axes[0, 0].set(
        title="Call-price comparison",
        xlabel="Forward log-moneyness",
        ylabel="Benchmark price",
    )
    axes[0, 0].legend()

    for intervals, result, error in space_results:
        mask = np.abs(result.log_moneyness) <= 0.25

        axes[0, 1].plot(
            result.log_moneyness[mask],
            PRICE_SCALE * error[mask],
            label=f"{intervals} intervals",
        )

    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set(
        title="Signed spatial errors",
        xlabel="Forward log-moneyness",
        ylabel="PDE minus Black",
    )
    axes[0, 1].legend()

    axes[1, 0].loglog(
        space_summary["log_spacing"],
        space_summary["max_price_error_points"],
        marker="o",
    )
    axes[1, 0].set(
        title="Spatial convergence",
        xlabel="Log-moneyness spacing",
        ylabel="Maximum absolute price error",
    )

    axes[1, 1].loglog(
        time_summary["max_time_step_years"],
        time_summary["max_vs_reference_points"],
        marker="o",
    )
    axes[1, 1].set(
        title="Time convergence",
        xlabel="Maximum time step, years",
        ylabel="Maximum difference from finer time solve",
    )

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    figure_path = output_directory / "black_convergence.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    audit = {
        "equation": "u_T = 0.5 * variance * (u_yy - u_y)",
        "volatility": VOLATILITY,
        "maturity_years": MATURITY,
        "log_moneyness_domain": [-1.0, 1.0],
        "reported_error_domain": [-0.25, 0.25],
        "benchmark_forward": BENCHMARK_FORWARD,
        "benchmark_discount_factor": BENCHMARK_DISCOUNT_FACTOR,
        "boundary_condition": "Exact analytical Black prices",
        "theta": 0.5,
        "rannacher_nominal_steps": 2,
        "time_reference_space_intervals": 800,
        "time_reference_max_step_years": MATURITY / 2048,
        "space_errors_decrease": bool(np.all(
            np.diff(
                space_summary["max_price_error_points"].to_numpy()
            ) < 0.0
        )),
        "time_errors_decrease": bool(np.all(
            np.diff(
                time_summary["max_vs_reference_points"].to_numpy()
            ) < 0.0
        )),
        "finest_max_price_error_points": float(
            space_summary.iloc[-1]["max_price_error_points"]
        ),
    }

    audit_path = output_directory / "benchmark_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Volatility: {100.0 * VOLATILITY:.2f}%")
    print(f"Maturity:   {MATURITY:.4f} years")
    print(
        "Price-error scaling: "
        f"forward={BENCHMARK_FORWARD:.2f}, "
        f"discount factor={BENCHMARK_DISCOUNT_FACTOR:.2f}"
    )
    print("Error comparison domain: log-moneyness [-0.25, 0.25]")

    print("\nSpatial convergence:")
    print(space_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6e}",
    ))

    print("\nTime convergence:")
    print(time_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6e}",
    ))

    print(
        "\nSpatial errors decrease:",
        audit["space_errors_decrease"],
    )
    print(
        "Time errors decrease:",
        audit["time_errors_decrease"],
    )
    print("\nTables:", output_directory)
    print("Figure:", figure_path)
    print("Audit: ", audit_path)


if __name__ == "__main__":
    main()