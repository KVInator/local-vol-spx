"""Validate AH backward prices and fixed-model delta/gamma sensitivities."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ah_backward_pricer import AHBackwardPricer
from ah_local_vol import AHLocalVariance
from andreasen_huge import AndreasenHugeSurface
from backward_pricer import BackwardPriceGrid


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--model", type=Path, required=True
    )
    parser.add_argument(
        "--space-intervals",
        type=int,
        nargs="+",
        default=[2000, 4000, 8000],
    )
    parser.add_argument(
        "--steps-per-day",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--domain-width",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--days",
        type=float,
        nargs="+",
        default=[4, 35, 60],
    )
    parser.add_argument(
        "--log-strikes",
        type=float,
        nargs="+",
        default=[-0.02, 0, 0.02],
    )
    parser.add_argument(
        "--run-name",
        default="ah_backward_validation",
    )
    args = parser.parse_args()

    counts = sorted(set(args.space_intervals))
    days = sorted(set(args.days))
    log_strikes = sorted(set(args.log_strikes))

    if (
        len(counts) < 3
        or min(counts) < 100
        or any(n % 4 for n in counts)
        or args.steps_per_day < 1
        or not np.isfinite(args.domain_width)
        or args.domain_width <= 0
        or not np.all(np.isfinite(days))
        or min(days) <= 0
        or not np.all(np.isfinite(log_strikes))
        or not log_strikes
        or Path(args.run_name).name != args.run_name
        or args.run_name in ("", ".", "..")
    ):
        raise ValueError(
            "Invalid study settings or run name."
        )

    model = AndreasenHugeSurface.load(
        args.model
    )

    if (
        days[-1] / 365
        > model.maturities[-1] + 1e-14
    ):
        raise ValueError(
            "Requested expiry exceeds the AH model."
        )

    for day in days:
        model.forward(day / 365)

    output = (
        Path("outputs/backward_pde_diagnostics")
        / args.run_name
    )
    if output.exists():
        raise FileExistsError(
            "Choose a fresh --run-name "
            "to preserve previous results."
        )
    output.mkdir(parents=True)

    width = args.domain_width
    steps = args.steps_per_day
    finest = counts[-1]

    cases = [
        {
            "case": f"space_{n}",
            "width": width,
            "intervals": n,
            "steps_per_day": steps,
            "early_power": 2,
            "state_shift": 0,
        }
        for n in counts
    ]
    cases.extend([
        {
            "case": "time_medium",
            "width": width,
            "intervals": finest,
            "steps_per_day": 2 * steps,
            "early_power": 2,
            "state_shift": 0,
        },
        {
            "case": "time_fine",
            "width": width,
            "intervals": finest,
            "steps_per_day": 4 * steps,
            "early_power": 2,
            "state_shift": 0,
        },
        {
            "case": "domain_wide",
            "width": 1.5 * width,
            "intervals": 3 * finest // 2,
            "steps_per_day": 4 * steps,
            "early_power": 2,
            "state_shift": 0,
        },
        {
            "case": "early_mesh_cubic",
            "width": width,
            "intervals": finest,
            "steps_per_day": 4 * steps,
            "early_power": 3,
            "state_shift": 0,
        },
        {
            "case": "half_cell_shift",
            "width": width,
            "intervals": finest,
            "steps_per_day": 4 * steps,
            "early_power": 2,
            "state_shift": width / finest,
        },
    ])

    for case in cases:
        AHBackwardPricer(
            model,
            domain_width=case["width"],
            space_intervals=case["intervals"],
            steps_per_day=case["steps_per_day"],
            early_time_power=case["early_power"],
            state_shift=case["state_shift"],
        )

    spot_y = np.linspace(-0.02, 0.02, 161)
    spots = model.spot * np.exp(spot_y)
    original = []
    curves = []
    bumps = []
    diagnostics = []

    print(
        f"Loaded one fixed AH model: "
        f"{model.grid.intervals:,} native intervals."
    )
    print(
        "Positive midpoint coefficients; "
        "no zero-time coefficient or volatility cap."
    )

    for case in cases:
        pricer = AHBackwardPricer(
            model,
            domain_width=case["width"],
            space_intervals=case["intervals"],
            steps_per_day=case["steps_per_day"],
            early_time_power=case["early_power"],
            state_shift=case["state_shift"],
        )

        for day in days:
            time = day / 365
            strikes = (
                model.forward(time)
                * np.exp(log_strikes)
            )
            print(
                f"{case['case']}: {day:g} days, "
                f"{len(strikes)} calls...",
                flush=True,
            )

            results = pricer.solve_many(
                strikes, time
            )
            diagnostics.append({
                **case,
                "days": day,
                **pricer.last_diagnostics,
            })

            for y, strike, result in zip(
                log_strikes, strikes, results
            ):
                value = result.greeks(model.spot)
                target = float(
                    model.call_price(strike, time)
                )

                original.append({
                    **case,
                    "days": day,
                    "log_strike": y,
                    "strike": strike,
                    "spot": model.spot,
                    "price": float(value.price),
                    "candidate_price": target,
                    "price_error_points": (
                        float(value.price) - target
                    ),
                    "delta": float(value.delta),
                    "gamma": float(value.gamma),
                    "actual_steps": result.time_steps,
                })

                grid_values = result.greeks(spots)
                curves.append(pd.DataFrame({
                    "case": case["case"],
                    "days": day,
                    "log_strike": y,
                    "spot_log_change": spot_y,
                    "spot": spots,
                    "price": grid_values.price,
                    "delta": grid_values.delta,
                    "gamma": grid_values.gamma,
                }))

                if case["case"] == "time_fine":
                    for fraction in (
                        0.002, 0.001, 0.0005, 0.00025,
                    ):
                        bump = fraction * model.spot
                        low, centre, high = result.price([
                            model.spot - bump,
                            model.spot,
                            model.spot + bump,
                        ])
                        fd_delta = float(
                            (high - low) / (2 * bump)
                        )
                        fd_gamma = float(
                            (high - 2 * centre + low)
                            / bump**2
                        )

                        bumps.append({
                            "days": day,
                            "log_strike": y,
                            "bump_points": bump,
                            "delta": float(value.delta),
                            "fd_delta": fd_delta,
                            "delta_difference": (
                                fd_delta
                                - float(value.delta)
                            ),
                            "gamma": float(value.gamma),
                            "fd_gamma": fd_gamma,
                            "gamma_difference": (
                                fd_gamma
                                - float(value.gamma)
                            ),
                        })

    contracts = pd.DataFrame(original)
    curves = pd.concat(
        curves, ignore_index=True
    )

    keys = [
        "days", "log_strike", "spot_log_change",
    ]
    reference = curves[
        curves["case"] == "time_fine"
    ][keys + ["price", "delta", "gamma"]]
    reference = reference.rename(
        columns={
            name: f"reference_{name}"
            for name in ("price", "delta", "gamma")
        }
    )
    curves = curves.merge(
        reference,
        on=keys,
        validate="many_to_one",
    )

    for name in ("price", "delta", "gamma"):
        curves[f"{name}_change_vs_reference"] = (
            curves[name]
            - curves[f"reference_{name}"]
        )

    summary = []

    for case in cases:
        rows = curves[
            curves["case"] == case["case"]
        ]
        at_spot = contracts[
            contracts["case"] == case["case"]
        ]

        slope_bounds = np.array([
            model.discount_factor(day / 365)
            * model.forward(day / 365)
            / model.spot
            for day in rows["days"]
        ])

        summary.append({
            **case,
            "max_original_spot_price_error": float(
                at_spot["price_error_points"]
                .abs().max()
            ),
            "max_curve_price_change": float(
                rows["price_change_vs_reference"]
                .abs().max()
            ),
            "max_curve_delta_change": float(
                rows["delta_change_vs_reference"]
                .abs().max()
            ),
            "max_curve_gamma_change": float(
                rows["gamma_change_vs_reference"]
                .abs().max()
            ),
            "negative_gamma_samples": int(
                np.sum(rows["gamma"] < -1e-8)
            ),
            "delta_bound_violations": int(
                np.sum(
                    (rows["delta"] < -1e-6)
                    | (
                        rows["delta"]
                        > slope_bounds + 1e-6
                    )
                )
            ),
        })

    summary = pd.DataFrame(summary)

    contracts.to_csv(
        output / "contract_results.csv",
        index=False,
    )
    curves.to_csv(
        output / "conditional_curves.csv",
        index=False,
    )
    summary.to_csv(
        output / "sensitivity_summary.csv",
        index=False,
    )
    pd.DataFrame(bumps).to_csv(
        output / "spot_bump_checks.csv",
        index=False,
    )

    figure, axes = plt.subplots(
        1, 3,
        figsize=(14, 4),
        constrained_layout=True,
    )
    displayed_day = days[len(days) // 2]
    displayed_strike = min(
        log_strikes, key=abs
    )
    subset = curves[
        (curves["days"] == displayed_day)
        & (
            curves["log_strike"]
            == displayed_strike
        )
    ]

    for axis, name in zip(
        axes, ("price", "delta", "gamma")
    ):
        for case_name, frame in subset.groupby(
            "case", sort=False
        ):
            axis.plot(
                frame["spot"],
                frame[name],
                label=case_name,
                linewidth=1,
            )
        axis.set(
            xlabel="Spot, fixed model",
            ylabel=name.capitalize(),
            title=(
                f"{displayed_day:g} days; "
                "forward log-strike "
                f"{displayed_strike:+g}"
            ),
        )

    axes[-1].legend(fontsize=7)
    figure.savefig(
        output / "backward_checks.png",
        dpi=160,
    )
    plt.close(figure)

    source_paths = [
        Path(__file__),
        Path(inspect.getfile(AHBackwardPricer)),
        Path(inspect.getfile(AHLocalVariance)),
        Path(inspect.getfile(AndreasenHugeSurface)),
        Path(inspect.getfile(BackwardPriceGrid)),
    ]

    audit = {
        "input_sha256": {
            str(args.model): digest(args.model),
        },
        "source_sha256": {
            str(path): digest(path)
            for path in source_paths
        },
        "native_AH_intervals": model.grid.intervals,
        "cases": cases,
        "coefficient_time_checks": diagnostics,
        "initial_condition": (
            "Terminal vanilla payoffs; backwards "
            "propagation to calendar zero."
        ),
        "boundary_policy": (
            "Martingale payoff asymptotes, "
            "not AH prices."
        ),
        "scheme": (
            "Midpoint-frozen Crank-Nicolson; two "
            "terminal Rannacher nominal steps."
        ),
        "zero_time_coefficient_requested": False,
        "coefficient_interpolation": (
            "Recovered AH variance linear "
            "in log-state."
        ),
        "time_mesh": (
            "Split at every expiry pillar; "
            "graded early calendar time."
        ),
        "greek_convention": (
            "Original physical local-volatility "
            "function and deterministic carry "
            "held fixed."
        ),
        "greek_method": (
            "Derivatives of cubic conditional-value "
            "interpolation in log-state."
        ),
        "reference_case": (
            "time_fine; finite-grid reference, "
            "not exact Greek truth."
        ),
        "curve_comparison_spot_log_range": [
            -0.02, 0.02,
        ],
        "price_error_scope": (
            "Generated calls at the original spot "
            "against this fixed AH model."
        ),
        "bump_scope": (
            "Finite differences of the same solved "
            "value curve; Black tests supply "
            "independent Greek checks."
        ),
        "model_refitted": False,
        "values_clipped": False,
        "candidate_promoted": False,
        "economic_wing_robustness_validated": False,
    }

    (output / "backward_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print("\nSensitivity summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print(
        "\nFinest-time values at the original spot:"
    )
    print(
        contracts[
            contracts["case"] == "time_fine"
        ][[
            "days",
            "log_strike",
            "price",
            "candidate_price",
            "price_error_points",
            "delta",
            "gamma",
        ]].to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print(
        f"\nDiagnostics: {output.resolve()}"
    )
    print(
        "Greek changes measure numerical sensitivity; "
        "economic wing robustness remains open."
    )


if __name__ == "__main__":
    main()