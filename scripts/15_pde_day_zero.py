"""Reprice a coupled-tail candidate from the day-zero payoff."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from forward_pde import ForwardPDESolver
from tailed_surface import TailedCallSurface, TailedShortEndSurface


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_times(pillars):
    """Keep exact pillars and remove nearby floating-point duplicates."""
    times = [0.0, *map(float, pillars)]
    candidates = [
        *(np.array([0.25, 0.5, 1.0, 2.0, 3.0]) / 365.0),
        *np.linspace(0.0, pillars[-1], 41),
    ]
    for time in candidates:
        if 0.0 <= time <= pillars[-1] and min(abs(time - old) for old in times) > 1e-12:
            times.append(float(time))
    return np.array(sorted(times))


def sample_result(result, normalized_strikes):
    """Linear interpolation in strike preserves the discrete convexity check."""
    grid = np.exp(result.log_moneyness)
    return np.stack([
        np.interp(normalized_strikes, grid, row)
        for row in result.normalized_calls
    ])


def shape_checks(result, comparison_width):
    y, z = result.log_moneyness, np.exp(result.log_moneyness)
    mask = np.abs(y) <= comparison_width + 1e-12
    z = z[mask]
    if len(z) < 3:
        raise ValueError("The comparison domain needs at least three PDE grid points.")
    calls = result.normalized_calls[1:, mask]
    slopes = np.diff(calls, axis=1) / np.diff(z)
    return {
        "increasing_price_intervals": int(np.sum(np.diff(calls, axis=1) > 1e-10)),
        "vertical_spread_bound_violations": int(np.sum(slopes < -1.0 - 1e-8)),
        "negative_butterflies": int(np.sum(np.diff(slopes, axis=1) < -1e-8)),
        "intrinsic_shortfalls": int(np.sum(calls < np.maximum(1.0 - z, 0.0) - 1e-10)),
        "upper_bound_excesses": int(np.sum(calls > 1.0 + 1e-10)),
    }


def quote_comparison(result, surface, rows, quotes_root):
    frames = []
    for row in rows:
        expiry, time = row["expiry"], float(row["maturity_years"])
        quotes = pd.read_csv(quotes_root / expiry / "quotes.csv")
        required = {"strike", "call_mid", "call_half_width"}
        if not required.issubset(quotes.columns):
            raise ValueError(f"Missing quote columns for {expiry}: {required - set(quotes.columns)}")
        half_width = quotes["call_half_width"].to_numpy(float)
        if not np.all(np.isfinite(half_width)) or np.any(half_width <= 0.0):
            raise ValueError(f"Invalid half-spreads for {expiry}.")
        index = int(np.argmin(abs(result.maturities - time)))
        if abs(result.maturities[index] - time) > 1e-12:
            raise ValueError("A fitted maturity is absent from the PDE output.")
        forward, discount = surface.forward(time), surface.discount_factor(time)
        strikes = quotes["strike"].to_numpy(float)
        midpoint = quotes["call_mid"].to_numpy(float)
        if not np.all(np.isfinite(strikes)) or not np.all(np.isfinite(midpoint)):
            raise ValueError(f"Nonfinite calibration data for {expiry}.")
        z = strikes / forward
        if np.any(z <= np.exp(result.log_moneyness[0])) or np.any(z >= np.exp(result.log_moneyness[-1])):
            raise ValueError(f"Some {expiry} quotes fall outside the PDE interior.")
        pde = discount * forward * np.interp(
            z, np.exp(result.log_moneyness), result.normalized_calls[index]
        )
        target = surface.call_price(strikes, time)
        frames.append(pd.DataFrame({
            "expiry": expiry,
            "strike": strikes,
            "log_moneyness": np.log(z),
            "call_mid": midpoint,
            "call_half_width": half_width,
            "candidate_call": target,
            "pde_call": pde,
            "pde_minus_candidate_points": pde - target,
            "candidate_residual_half_spreads": (target - midpoint) / half_width,
            "pde_residual_half_spreads": (pde - midpoint) / half_width,
        }))
    return pd.concat(frames, ignore_index=True)


def observed_orders(frame, coordinate, error):
    frame = frame.copy()
    orders = [np.nan]
    for index in range(1, len(frame)):
        old, new = frame.iloc[index - 1], frame.iloc[index]
        ratio = old[error] / new[error] if new[error] > 0.0 else np.nan
        refinement = new[coordinate] / old[coordinate]
        orders.append(float(np.log(ratio) / np.log(refinement)) if ratio > 0.0 else np.nan)
    frame["observed_order"] = orders
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tails", type=Path, required=True)
    parser.add_argument("--quotes-root", type=Path, required=True)
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--space-intervals", nargs="+", type=int, default=[400, 800, 1600])
    parser.add_argument("--time-steps-per-day", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--time-reference-steps-per-day", type=int, default=64)
    parser.add_argument("--space-study-steps-per-day", type=int, default=16)
    parser.add_argument("--domain-widths", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--comparison-width", type=float, default=0.09)
    args = parser.parse_args()
    intervals = sorted(set(args.space_intervals))
    steps = sorted(set(args.time_steps_per_day))
    widths = sorted(set(args.domain_widths))
    if (
        len(intervals) < 2 or min(intervals) < 4
        or any(value % 2 for value in intervals)
        or len(steps) < 2 or min(steps) <= 0 or args.space_study_steps_per_day <= 0
        or args.time_reference_steps_per_day <= max(steps)
        or len(widths) < 2 or not np.all(np.isfinite(widths))
        or not 0.0 < args.comparison_width < widths[0]
    ):
        raise ValueError("Invalid refinement settings; use even spatial counts and a finer time reference.")

    candidate_name = args.tails.parent.name
    diagnostic_dir = Path("outputs/pde_diagnostics") / args.quotes_root.name / f"day_zero_{candidate_name}"
    data_dir = args.quotes_root / f"day_zero_{candidate_name}"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    specification = json.loads(args.tails.read_text(encoding="utf-8"))
    surface = TailedShortEndSurface(
        TailedCallSurface.load(args.tails, -widths[-1], widths[-1]), args.spot
    )
    pillars = surface.surface.maturities
    times = evaluation_times(pillars)
    report_y = np.linspace(-args.comparison_width, args.comparison_width, 401)
    report_z = np.exp(report_y)
    scales = np.array([surface.discount_factor(time) * surface.forward(time) for time in times])
    target = np.stack([surface.normalized_call(report_z, time) for time in times])

    # Sample the widest domain before starting any PDE solve.
    probe_z = np.unique(np.concatenate([
        np.exp(np.linspace(-widths[-1], widths[-1], 4001)),
        *[
            np.array([np.nextafter(join, 0.0), join, np.nextafter(join, np.inf)])
            for model in surface.surface.slices
            for join in (model.left.normalized_strike, model.right.normalized_strike)
            if abs(np.log(join)) < widths[-1]
        ],
    ]))
    lv_zero = np.sqrt(surface.normalized_variance(probe_z, 0.0))
    lv_first = np.sqrt(surface.normalized_variance(probe_z, surface.first_maturity, "left"))
    print(f"Loaded {len(pillars)} pillars; recomputed calendar checks passed.", flush=True)
    print(f"First maturity: {365.0 * surface.first_maturity:.8f} days", flush=True)
    print(f"Widest numerical domain: [-{widths[-1]:.2f}, {widths[-1]:.2f}]", flush=True)
    print(f"Sampled maximum day-zero local vol: {100.0 * lv_zero.max():.4f}%", flush=True)
    print(f"Sampled maximum first-expiry left local vol: {100.0 * lv_first.max():.4f}%", flush=True)

    cache = {}
    case_rows = []

    def run_case(width, count, steps_per_day, boundary_kind="surface"):
        key = (float(width), int(count), int(steps_per_day), boundary_kind)
        if key in cache:
            return cache[key]
        label = f"width={width:g}, intervals={count}, steps/day={steps_per_day}, boundary={boundary_kind}"
        print(f"Solving {label}...", flush=True)
        solver = ForwardPDESolver(
            -width, width, n_space_intervals=count,
            max_time_step=1.0 / (365.0 * steps_per_day),
            theta=0.5, rannacher_steps=2,
        )
        boundary_z = solver.normalized_strikes[[0, -1]]
        def boundaries(time):
            if boundary_kind == "surface":
                return tuple(surface.normalized_call(boundary_z, time))
            return tuple(np.maximum(1.0 - boundary_z, 0.0))
        result = solver.solve(
            times,
            np.maximum(1.0 - solver.normalized_strikes, 0.0),
            surface.normalized_variance,
            boundaries,
            time_breaks=pillars,
        )
        sampled = sample_result(result, report_z)
        errors = (sampled - target) * scales[:, None]
        comparison = quote_comparison(result, surface, specification["slices"], args.quotes_root)
        metrics = {
            "domain_width": width, "space_intervals": count,
            "log_spacing": 2.0 * width / count,
            "steps_per_day": steps_per_day, "boundary": boundary_kind,
            "actual_time_steps": result.time_steps,
            "max_price_error_points": float(np.max(np.abs(errors[1:]))),
            "rms_price_error_points": float(np.sqrt(np.mean(errors[1:] ** 2))),
            "max_quote_repricing_error_points": float(comparison["pde_minus_candidate_points"].abs().max()),
            "pde_quote_rms_half_spreads": float(np.sqrt(np.mean(comparison["pde_residual_half_spreads"] ** 2))),
            **shape_checks(result, args.comparison_width),
        }
        case_rows.append(metrics)
        cache[key] = (result, sampled, errors, metrics, comparison)
        print(f"  Max surface error {metrics['max_price_error_points']:.6g}; max quote repricing error {metrics['max_quote_repricing_error_points']:.6g} points.", flush=True)
        return cache[key]

    width, finest = widths[0], intervals[-1]
    spatial = [run_case(width, count, args.space_study_steps_per_day)[3] for count in intervals]
    spatial_table = observed_orders(pd.DataFrame(spatial), "space_intervals", "max_price_error_points")
    reference = run_case(width, finest, args.time_reference_steps_per_day)
    time_rows = []
    for step_count in steps:
        case = run_case(width, finest, step_count)
        difference = (case[1] - reference[1]) * scales[:, None]
        time_rows.append({**case[3], "max_vs_time_reference_points": float(np.max(np.abs(difference[1:])))})
    time_table = observed_orders(pd.DataFrame(time_rows), "steps_per_day", "max_vs_time_reference_points")

    # Match spacing across widened domains, then compare both boundary policies.
    reference_spacing = 2.0 * width / finest
    baseline = run_case(width, finest, args.space_study_steps_per_day)
    domain_rows = []
    for domain_width in widths:
        count = 2 * int(round(domain_width / reference_spacing))
        case = run_case(domain_width, count, args.space_study_steps_per_day)
        difference = (case[1] - baseline[1]) * scales[:, None]
        domain_rows.append({**case[3], "max_change_vs_narrow_domain_points": float(np.max(np.abs(difference[1:])))})
    domain_table = pd.DataFrame(domain_rows)
    boundary_rows = []
    for domain_width in (widths[0], widths[-1]):
        count = 2 * int(round(domain_width / reference_spacing))
        exact = run_case(domain_width, count, args.space_study_steps_per_day)
        asymptotic = run_case(domain_width, count, args.space_study_steps_per_day, "intrinsic")
        difference = (asymptotic[1] - exact[1]) * scales[:, None]
        boundary_rows.append({**asymptotic[3], "max_boundary_policy_change_points": float(np.max(np.abs(difference[1:])))})
    boundary_table = pd.DataFrame(boundary_rows)

    result, sampled, errors, _, quotes = reference
    quotes.to_csv(diagnostic_dir / "quote_repricing.csv", index=False)
    maturity_table = pd.DataFrame({
        "days": 365.0 * times[1:],
        "expiry_pillar": [bool(np.any(abs(pillars - time) <= 1e-12)) for time in times[1:]],
        "max_price_error_points": np.max(np.abs(errors[1:]), axis=1),
        "rms_price_error_points": np.sqrt(np.mean(errors[1:] ** 2, axis=1)),
    })
    maturity_table.to_csv(diagnostic_dir / "maturity_errors.csv", index=False)
    for name, table in (
        ("space_convergence", spatial_table), ("time_convergence", time_table),
        ("domain_sensitivity", domain_table), ("boundary_sensitivity", boundary_table),
        ("all_cases", pd.DataFrame(case_rows)),
    ):
        table.to_csv(diagnostic_dir / f"{name}.csv", index=False)
    np.savez_compressed(
        data_dir / "day_zero_pde_grid.npz",
        maturity_years=result.maturities, log_moneyness=result.log_moneyness,
        normalized_calls=result.normalized_calls,
        report_log_moneyness=report_y, report_price_errors=errors,
    )
    model_path = data_dir / "day_zero_surface.json"
    model_path.write_text(json.dumps({
        **surface.metadata(),
        "tail_slices_file": os.path.relpath(args.tails.resolve(), data_dir.resolve()),
    }, indent=2) + "\n", encoding="utf-8")

    inputs = [args.tails, *[
        args.tails.parent / row["core_file"] for row in specification["slices"]
    ], *[
        args.quotes_root / row["expiry"] / "quotes.csv" for row in specification["slices"]
    ]]
    source_root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *[
        source_root / "src" / name for name in
        ("tailed_surface.py", "short_end.py", "forward_pde.py", "coupled_tails.py", "black.py", "implied_vol.py", "local_vol.py", "convex_spline.py", "vol_surface.py")
    ]]
    audit = {
        "validation_scope": "Day-zero payoff propagation with assumed short-end and strike tails; numerical consistency, not independent model validation.",
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "source_sha256": {str(path): sha256(path) for path in sources},
        "candidate": str(args.tails), "spot": args.spot,
        "runtime_versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "theta": 0.5, "rannacher_nominal_steps": 2,
        "initial_condition": "Intrinsic normalized call payoff at time zero.",
        "surface_boundary": "Constructed surface prices at both finite boundaries.",
        "alternative_boundary": "Intrinsic payoff on the left and zero on the right, at every maturity.",
        "derivative_sides": {"old_time": "right", "new_time": "left"},
        "time_breaks_years": pillars.tolist(),
        "evaluation_maturities": len(times),
        "comparison_log_moneyness": [-args.comparison_width, args.comparison_width],
        "preflight_scope": "Sampled widest domain plus join-adjacent points; not a continuous short-end certificate.",
        "preflight_points": len(probe_z),
        "max_day_zero_local_volatility_pct": float(100.0 * lv_zero.max()),
        "max_first_expiry_left_local_volatility_pct": float(100.0 * lv_first.max()),
        "continuous_calendar_checks": surface.surface.calendar_checks(),
        "candidate_quote_rms_half_spreads": float(np.sqrt(np.mean(quotes["candidate_residual_half_spreads"] ** 2))),
        "reference_case": reference[3],
        "all_cases": case_rows,
        "space_errors_decrease": bool(np.all(np.diff(spatial_table["max_price_error_points"]) < 0.0)),
        "time_reference_errors_decrease": bool(np.all(np.diff(time_table["max_vs_time_reference_points"]) < 0.0)),
        "original_models_modified": False, "prices_or_variances_clipped": False,
    }
    (diagnostic_dir / "day_zero_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].loglog(spatial_table["space_intervals"], spatial_table["max_price_error_points"], "o-")
    axes[0, 0].set(xlabel="Spatial intervals", ylabel="Max price error (points)", title="Spatial refinement")
    axes[0, 1].loglog(time_table["steps_per_day"], time_table["max_vs_time_reference_points"], "o-")
    axes[0, 1].set(xlabel="Steps per day", ylabel="Difference from time reference (points)", title="Time refinement at fixed spatial grid")
    axes[1, 0].plot(maturity_table["days"], maturity_table["max_price_error_points"])
    axes[1, 0].set(xlabel="Days", ylabel="Max price error (points)", title="Day-zero propagation: reference case")
    axes[1, 1].plot(domain_table["domain_width"], domain_table["max_change_vs_narrow_domain_points"], "o-", label="Domain change versus narrow case")
    axes[1, 1].plot(boundary_table["domain_width"], boundary_table["max_boundary_policy_change_points"], "s--", label="Change from boundary policy")
    axes[1, 1].set(xlabel="Symmetric domain half-width", ylabel="Max central price change (points)", title="Domain and boundary sensitivity")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
    axes[1, 1].legend()
    figure.savefig(diagnostic_dir / "day_zero_repricing.png", dpi=160)
    plt.close(figure)

    with pd.option_context("display.max_columns", None, "display.width", 180):
        for title, table, columns in (
            ("Spatial convergence", spatial_table, ["space_intervals", "log_spacing", "actual_time_steps", "max_price_error_points", "max_quote_repricing_error_points", "observed_order"]),
            ("Time convergence", time_table, ["steps_per_day", "actual_time_steps", "max_vs_time_reference_points", "max_price_error_points", "observed_order"]),
            ("Domain sensitivity", domain_table, ["domain_width", "space_intervals", "max_price_error_points", "max_change_vs_narrow_domain_points"]),
            ("Boundary sensitivity", boundary_table, ["domain_width", "max_price_error_points", "max_boundary_policy_change_points"]),
        ):
            print(f"\n{title}:\n{table[columns].to_string(index=False)}")
    print("\nReference-case shape checks:", shape_checks(result, args.comparison_width))
    print(f"Candidate quote RMS: {audit['candidate_quote_rms_half_spreads']:.6f} half-spreads")
    print(f"PDE quote RMS:       {reference[3]['pde_quote_rms_half_spreads']:.6f} half-spreads")
    print(f"\nDiagnostics: {diagnostic_dir.resolve()}")
    print(f"Loadable day-zero model: {model_path.resolve()}")
    print("The primary calibrated models have not been replaced.")


if __name__ == "__main__":
    main()
