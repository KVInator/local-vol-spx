"""Propagate AH prices from the day-zero payoff with independent PDE grids."""

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

from ah_local_vol import AHLocalVariance
from andreasen_huge import AndreasenHugeSurface
from forward_pde import ForwardPDESolver


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_calls(model, strikes, time):
    """Evaluate existing AH prices without calculating unused variance."""
    time = model._time(time)
    z = np.asarray(strikes, float)
    if np.any(z < model.grid.z[0]) or np.any(z > model.grid.z[-1]):
        raise ValueError("Price extrapolation is unsupported.")

    index = min(
        int(np.searchsorted(model.maturities, time, side="right")),
        len(model.maturities) - 1,
    )
    start = 0 if index == 0 else model.maturities[index - 1]
    calls, _, _ = model.grid.step(
        model._calls[index],
        model._densities[index],
        model._variances[index],
        time - start,
    )
    return np.interp(z, model.grid.z, calls)


def shape_counts(z, calls, previous):
    slopes = np.diff(calls) / np.diff(z)
    return {
        "increasing_prices": int(np.sum(slopes > 1e-8)),
        "vertical_spread_violations": int(np.sum(slopes < -1 - 1e-8)),
        "negative_butterflies": int(np.sum(np.diff(slopes) < -1e-8)),
        "intrinsic_shortfalls": int(
            np.sum(calls < np.maximum(1 - z, 0) - 1e-10)
        ),
        "upper_bound_excesses": int(np.sum(calls > 1 + 1e-10)),
        "calendar_violations": int(np.sum(calls < previous - 1e-10)),
    }


def load_quotes(root, model):
    found = {}
    required = {
        "strike", "call_mid", "call_half_width", "maturity_years",
        "forward", "discount_factor",
    }
    for path in sorted(root.glob("*/quotes.csv")):
        frame = pd.read_csv(path)
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"Missing retained-quote fields: {path}")

        match = np.flatnonzero(np.isclose(
            model.maturities,
            frame["maturity_years"].iloc[0],
            rtol=0,
            atol=1e-12,
        ))
        if len(match) != 1 or int(match[0]) in found:
            raise ValueError(f"Quote maturity is missing or duplicated: {path}")

        i = int(match[0])
        for column, target, tolerance in (
            ("maturity_years", model.maturities[i], 1e-12),
            ("forward", model.forwards[i], 1e-7),
            ("discount_factor", model.discounts[i], 1e-10),
        ):
            if (
                not np.all(np.isfinite(frame[column]))
                or not np.allclose(
                    frame[column], target, rtol=0, atol=tolerance
                )
            ):
                raise ValueError(f"Inconsistent {column}: {path}")

        data = frame[
            ["strike", "call_mid", "call_half_width"]
        ].to_numpy(float)
        if not np.all(np.isfinite(data)) or np.any(data[:, [0, 2]] <= 0):
            raise ValueError(f"Invalid retained prices or spreads: {path}")
        found[i] = (path, data)

    if len(found) != len(model.maturities):
        raise ValueError("Need retained quotes for every AH maturity.")
    return [found[i] for i in range(len(model.maturities))]


def evaluation_times(pillars):
    times = [0.0, *map(float, pillars)]
    candidates = np.r_[
        np.array([0.25, 0.5, 1, 2, 3]) / 365,
        np.linspace(pillars[0], pillars[-1], 31),
    ]
    for time in candidates:
        if (
            time <= pillars[-1]
            and min(abs(np.asarray(times) - time)) > 1e-12
        ):
            times.append(float(time))
    return np.array(sorted(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--quotes-root", type=Path, required=True)
    parser.add_argument(
        "--space-intervals",
        type=int,
        nargs="+",
        default=[2000, 4000, 8000],
    )
    parser.add_argument("--steps-per-day", type=int, default=32)
    parser.add_argument("--output-name", default="ah_day_zero_validation")
    args = parser.parse_args()

    counts = sorted(set(args.space_intervals))
    if (
        len(counts) < 3
        or any(n < 100 or n % 4 for n in counts)
        or args.steps_per_day < 1
        or args.output_name in ("", ".", "..")
        or Path(args.output_name).name != args.output_name
    ):
        raise ValueError(
            "Use three counts divisible by four and a fresh output name."
        )

    model_paths = [
        args.models_root / f"grid_{n}.json" for n in counts
    ]
    models = {
        n: AndreasenHugeSurface.load(path)
        for n, path in zip(counts, model_paths)
    }
    reference = models[counts[-1]]

    for n, model in models.items():
        if model.grid.intervals != n or model.grid.width != 1.0:
            raise ValueError(
                "Expected width-one AH models with matching grid counts."
            )
        for name in ("maturities", "forwards", "discounts"):
            if not np.array_equal(
                getattr(model, name), getattr(reference, name)
            ):
                raise ValueError(
                    "AH refinement models must share carry and maturities."
                )
        if model.spot != reference.spot:
            raise ValueError("AH refinement models must share spot.")

    quotes = load_quotes(args.quotes_root, reference)
    times = evaluation_times(reference.maturities)
    report_y = np.linspace(-0.09, 0.09, 361)
    report_z = np.exp(report_y)

    directory = args.quotes_root / args.output_name
    diagnostics = (
        Path("outputs/andreasen_huge_diagnostics")
        / args.quotes_root.name
        / args.output_name
    )
    if directory.exists() or diagnostics.exists():
        raise FileExistsError("Output exists. Choose a fresh --output-name.")

    directory.mkdir(parents=True)
    diagnostics.mkdir(parents=True)

    cases, errors, shapes, quote_frames, solved = [], [], [], [], {}

    print(
        f"Loaded {len(reference.maturities)} expiries and "
        f"{sum(len(d) for _, d in quotes):,} retained quotes.",
        flush=True,
    )
    print("Starting from intrinsic payoff; no fitted initial curve is imposed.")
    print("Zero-time local variance is undefined and is never requested.")

    def run_case(name, n, width, intervals, steps):
        print(
            f"PDE {name}: {intervals} intervals, {steps} steps/day...",
            flush=True,
        )
        model = models[n]
        adapter = AHLocalVariance(model)
        solver = ForwardPDESolver(
            -width,
            width,
            n_space_intervals=intervals,
            max_time_step=1 / (365 * steps),
            theta=0.5,
            rannacher_steps=2,
        )
        z = solver.normalized_strikes

        early_count = max(
            8, int(np.ceil(365 * model.maturities[0] * steps))
        )
        early_breaks = (
            model.maturities[0]
            * np.linspace(0, 1, early_count + 1) ** 2
        )
        result = solver.solve(
            times,
            np.maximum(1 - z, 0),
            adapter,
            lambda t: tuple(normalized_calls(model, z[[0, -1]], t)),
            time_breaks=np.unique(
                np.r_[model.maturities, early_breaks]
            ),
        )

        scales = np.array([
            model.forward(t) * model.discount_factor(t)
            for t in times
        ])
        sampled = np.stack([
            np.interp(report_z, z, c)
            for c in result.normalized_calls
        ])
        target = np.stack([
            normalized_calls(model, report_z, t) for t in times
        ])
        difference = (sampled - target) * scales[:, None]
        solved[name] = sampled * scales[:, None]

        total_shape = 0
        for i in range(1, len(times)):
            values = shape_counts(
                z,
                result.normalized_calls[i],
                result.normalized_calls[i - 1],
            )
            total_shape += sum(values.values())
            slopes = np.diff(result.normalized_calls[i]) / np.diff(z)
            shapes.append({
                "case": name,
                "days": 365 * times[i],
                **values,
                "minimum_butterfly_slope_change": float(
                    np.diff(slopes).min()
                ),
            })
            errors.append({
                "case": name,
                "days": 365 * times[i],
                "expiry_pillar": bool(np.any(
                    abs(model.maturities - times[i]) < 1e-12
                )),
                "max_price_error_points": float(
                    np.max(abs(difference[i]))
                ),
                "rms_price_error_points": float(
                    np.sqrt(np.mean(difference[i] ** 2))
                ),
            })

        numerical, residuals, candidate_residuals = [], [], []
        for (path, data), t in zip(quotes, model.maturities):
            i = int(np.argmin(abs(times - t)))
            normalized = data[:, 0] / model.forward(t)
            if (
                np.any(normalized <= z[0])
                or np.any(normalized >= z[-1])
            ):
                raise ValueError("Retained quote lies outside the PDE domain.")

            price = scales[i] * np.interp(
                normalized, z, result.normalized_calls[i]
            )
            candidate = scales[i] * normalized_calls(
                model, normalized, t
            )
            numerical.extend(price - candidate)
            residuals.extend((price - data[:, 1]) / data[:, 2])
            candidate_residuals.extend(
                (candidate - data[:, 1]) / data[:, 2]
            )
            quote_frames.append(pd.DataFrame({
                "case": name,
                "expiry": path.parent.name,
                "strike": data[:, 0],
                "call_mid": data[:, 1],
                "call_half_width": data[:, 2],
                "candidate_price": candidate,
                "PDE_price": price,
                "PDE_minus_candidate": price - candidate,
            }))

        row = {
            "case": name,
            "AH_intervals": n,
            "PDE_intervals": intervals,
            "width": width,
            "steps_per_day": steps,
            "actual_steps": result.time_steps,
            "max_price_error_points": float(
                np.max(abs(difference[1:]))
            ),
            "rms_price_error_points": float(
                np.sqrt(np.mean(difference[1:] ** 2))
            ),
            "max_quote_error_points": float(
                np.max(abs(np.asarray(numerical)))
            ),
            "candidate_rms_half_spreads": float(
                np.sqrt(np.mean(np.square(candidate_residuals)))
            ),
            "PDE_rms_half_spreads": float(
                np.sqrt(np.mean(np.square(residuals)))
            ),
            "sampled_shape_violations": total_shape,
        }
        cases.append(row)

        if name == "time_fine":
            np.savez_compressed(
                directory / "PDE_grid.npz",
                maturities=times,
                log_moneyness=result.log_moneyness,
                normalized_calls=result.normalized_calls,
            )

        print(
            f"  Max error {row['max_price_error_points']:.6g}; "
            f"quote error {row['max_quote_error_points']:.6g} points.",
            flush=True,
        )

    for n in counts:
        run_case(f"joint_{n}", n, 0.5, n, args.steps_per_day)

    finest = counts[-1]
    run_case(
        "time_medium", finest, 0.5, finest, 2 * args.steps_per_day
    )
    run_case(
        "time_fine", finest, 0.5, finest, 4 * args.steps_per_day
    )
    run_case(
        "domain_0p75",
        finest,
        0.75,
        3 * finest // 2,
        2 * args.steps_per_day,
    )
    run_case(
        "matched_nodes",
        finest,
        0.5,
        finest // 2,
        4 * args.steps_per_day,
    )

    for row in cases:
        if row["AH_intervals"] == finest:
            row["max_change_vs_time_reference"] = float(np.max(abs(
                solved[row["case"]][1:]
                - solved["time_fine"][1:]
            )))

    domain_change = float(np.max(abs(
        solved["domain_0p75"][1:] - solved["time_medium"][1:]
    )))

    joint = pd.DataFrame(cases[:len(counts)])
    joint["observed_joint_order"] = np.r_[
        np.nan,
        np.log(
            joint["max_price_error_points"].to_numpy()[:-1]
            / joint["max_price_error_points"].to_numpy()[1:]
        ) / np.log(
            np.array(counts[1:]) / np.array(counts[:-1])
        ),
    ]

    joint.to_csv(diagnostics / "joint_refinement.csv", index=False)
    pd.DataFrame(cases).to_csv(
        diagnostics / "PDE_cases.csv", index=False
    )
    pd.DataFrame(errors).to_csv(
        diagnostics / "maturity_errors.csv", index=False
    )
    pd.DataFrame(shapes).to_csv(
        diagnostics / "PDE_shape_checks.csv", index=False
    )
    pd.concat(quote_frames, ignore_index=True).to_csv(
        diagnostics / "quote_repricing.csv", index=False
    )

    adapter = AHLocalVariance(reference)
    conditioning = []
    requests = [
        (d / 365, "right")
        for d in (0.001, 0.01, 0.25, 1, 2, 3)
        if d / 365 < reference.maturities[0]
    ]
    requests += [
        (float(reference.maturities[0]), side)
        for side in ("left", "right")
    ]
    for t, side in requests:
        for width in (0.01, 0.09, 0.5):
            mask = abs(adapter.y) <= width + 1e-12
            lv = 100 * np.sqrt(adapter(
                reference.grid.z[1:-1][mask], t, side
            ))
            i = int(np.argmax(lv))
            conditioning.append({
                "days": 365 * t,
                "side": side,
                "half_width": width,
                "peak_local_vol_pct": float(lv[i]),
                "peak_log_moneyness": float(adapter.y[mask][i]),
            })
    pd.DataFrame(conditioning).to_csv(
        diagnostics / "early_local_variance.csv", index=False
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(
        joint["AH_intervals"],
        joint["max_price_error_points"],
        "o-",
    )
    axes[0].set(
        xlabel="AH intervals; PDE uses half the log spacing",
        ylabel="Maximum PDE-minus-AH price error, points",
    )

    error_frame = pd.DataFrame(errors)
    for name in [f"joint_{n}" for n in counts] + ["time_fine"]:
        frame = error_frame[error_frame["case"] == name]
        axes[1].semilogy(
            frame["days"],
            frame["max_price_error_points"],
            label=name,
        )
    axes[1].set(
        xlabel="Days from valuation",
        ylabel="Maximum price error, points",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        diagnostics / "AH_day_zero_validation.png", dpi=160
    )
    plt.close(figure)

    matched_change = float(np.max(abs(
        solved["matched_nodes"][1:] - solved["time_fine"][1:]
    )))
    audit = {
        "input_sha256": {
            str(p): digest(p)
            for p in [*model_paths, *[p for p, _ in quotes]]
        },
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path(inspect.getfile(AHLocalVariance)),
                Path(inspect.getfile(AndreasenHugeSurface)),
                Path(inspect.getfile(ForwardPDESolver)),
            ]
        },
        "initial_condition": (
            "Intrinsic normalized call payoff at day zero."
        ),
        "boundaries": (
            "AH prices at both finite PDE boundaries."
        ),
        "coefficient": (
            "Logarithmic density/time-derivative solves; "
            "variance linear in log-strike."
        ),
        "initial_density": (
            "Analytical discrete unit slope jump at z=1; "
            "no flat-wing differencing."
        ),
        "zero_time_variance": (
            "Undefined; initial implicit half-steps "
            "request positive times only."
        ),
        "first_period": (
            "Original AH resolvent interpolation, "
            "with no added short-end model."
        ),
        "early_time_mesh": (
            "Quadratically spaced time breaks through the first expiry."
        ),
        "report_domain": [-0.09, 0.09],
        "evaluation_maturities": len(times),
        "joint_errors_decrease": bool(np.all(
            np.diff(joint["max_price_error_points"]) < 0
        )),
        "time_reference_shape_violations": next(
            r["sampled_shape_violations"]
            for r in cases if r["case"] == "time_fine"
        ),
        "max_PDE_domain_change_points": domain_change,
        "max_matched_node_change_points": matched_change,
        "shape_tolerances": {
            "normalized_price": 1e-10,
            "normalized_slope": 1e-8,
            "butterfly_slope_change": 1e-8,
        },
        "economic_wing_robustness_validated": False,
        "backward_prices_and_greeks_validated": False,
        "values_clipped": False,
        "models_refitted": False,
        "candidate_promoted": False,
    }
    (diagnostics / "day_zero_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nJoint day-zero refinement:")
    print(joint[[
        "AH_intervals",
        "PDE_intervals",
        "max_price_error_points",
        "max_quote_error_points",
        "observed_joint_order",
    ]].to_string(index=False))

    print("\nTime and domain checks:")
    print(pd.DataFrame(cases)[[
        "case",
        "steps_per_day",
        "max_price_error_points",
        "max_change_vs_time_reference",
        "sampled_shape_violations",
    ]].to_string(index=False))

    print(f"\nPDE domain change: {domain_change:.6g} points")
    print(f"Matched-node price change: {matched_change:.6g} points")
    print(f"Joint errors decrease: {audit['joint_errors_decrease']}")
    print(
        "Finest-time shape violations: "
        f"{audit['time_reference_shape_violations']}"
    )
    print(f"\nPDE grid: {directory.resolve() / 'PDE_grid.npz'}")
    print(f"Diagnostics: {diagnostics.resolve()}")
    print(
        "Day-zero convergence study; "
        "wing robustness and Greeks remain unvalidated."
    )


if __name__ == "__main__":
    main()