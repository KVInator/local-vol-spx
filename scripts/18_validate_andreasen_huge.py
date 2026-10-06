"""Refine AH calibration and check positive-maturity PDE consistency."""

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from andreasen_huge import (
    AHGrid, AndreasenHugeCalibrator, AndreasenHugeSurface,
)
from forward_pde import ForwardPDESolver


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def price_metrics(residual):
    return {
        "rms_half_spreads": float(np.sqrt(np.mean(residual ** 2))),
        "max_half_spreads": float(np.max(np.abs(residual))),
        "outside_original_bands": int(np.sum(np.abs(residual) > 1 + 1e-8)),
    }


def shape_counts(z, calls, previous=None):
    slopes = np.diff(calls) / np.diff(z)
    return {
        "increasing_prices": int(np.sum(slopes > 1e-8)),
        "vertical_spread_violations": int(np.sum(slopes < -1 - 1e-8)),
        "negative_butterflies": int(np.sum(np.diff(slopes) < -1e-8)),
        "intrinsic_shortfalls": int(np.sum(calls < np.maximum(1 - z, 0) - 1e-10)),
        "upper_bound_excesses": int(np.sum(calls > 1 + 1e-10)),
        "calendar_violations": 0 if previous is None else int(
            np.sum(calls < previous - 1e-10)
        ),
    }


def load_quotes(root, reference):
    matches = {}
    for path in sorted(root.glob("*/quotes.csv")):
        frame = pd.read_csv(path)
        required = ["strike", "call_mid", "call_half_width",
                    "maturity_years", "forward", "discount_factor"]
        if not set(required).issubset(frame.columns) or frame.empty:
            raise ValueError(f"Missing retained-quote fields: {path}")
        time = float(frame["maturity_years"].iloc[0])
        indices = np.flatnonzero(np.isclose(
            reference.maturities, time, rtol=0, atol=1e-12
        ))
        if len(indices) != 1:
            raise ValueError(f"Quote maturity does not match the AH model: {path}")
        i = int(indices[0])
        if i in matches:
            raise ValueError(f"Duplicate quote maturity: {path}")
        for column, target, tolerance in (
            ("maturity_years", reference.maturities[i], 1e-12),
            ("forward", reference.forwards[i], 1e-7),
            ("discount_factor", reference.discounts[i], 1e-10),
        ):
            if not np.all(np.isfinite(frame[column])) or not np.allclose(
                frame[column], target, rtol=0, atol=tolerance
            ):
                raise ValueError(f"Inconsistent {column}: {path}")
        matches[i] = (path, frame.sort_values("strike").reset_index(drop=True))
    if len(matches) != len(reference.maturities):
        raise ValueError("Need one retained-quote file for every AH maturity.")
    rows = [matches[i] for i in range(len(reference.maturities))]
    dates = set()
    for _, frame in rows:
        if "quote_date" in frame:
            dates.update(frame["quote_date"].astype(str))
        if "underlying_last" in frame and not np.allclose(
            frame["underlying_last"], reference.spot, rtol=0, atol=1e-6
        ):
            raise ValueError("Retained quotes and AH spot disagree.")
    if len(dates) > 1:
        raise ValueError("Retained quotes must share one valuation date.")
    data = [f[["strike", "call_mid", "call_half_width"]].to_numpy(float)
            for _, f in rows]
    return rows, data


class NodeVariance:
    """Diagnostic continuation of discrete variance in log-strike."""

    def __init__(self, model):
        self.model = model
        self.y = np.log(model.grid.z[1:-1])

    def __call__(self, strikes, time, side):
        if np.asarray(time).ndim != 0 or float(time) <= 0:
            raise ValueError("This variance adapter needs a positive scalar maturity.")
        z = np.asarray(strikes, float)
        if not np.all(np.isfinite(z)) or np.any(z <= 0):
            raise ValueError("Normalized strikes must be finite and positive.")
        y = np.log(z)
        if y.min() < self.y[0] or y.max() > self.y[-1]:
            raise ValueError("Variance extrapolation is not supported.")
        variance = self.model.node_state(float(time), side)["local_variance"]
        first = max(0, int(np.searchsorted(self.y, y.min())) - 1)
        last = min(len(self.y), int(np.searchsorted(self.y, y.max(), side="right")) + 1)
        values = variance[first:last]
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Unresolved AH variance in the PDE domain.")
        return np.interp(y.ravel(), self.y[first:last], values).reshape(z.shape)


def evaluation_times(pillars):
    # Preserve exact pillars when a regular-grid time nearly coincides with one.
    times = list(pillars)
    for time in np.linspace(pillars[0], pillars[-1], 31):
        if np.min(np.abs(np.asarray(times) - time)) > 1e-12:
            times.append(float(time))
    return np.array(sorted(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quotes-root", type=Path, required=True)
    parser.add_argument("--output-name", default="ah_1_validation")
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--control-points", type=int, default=31)
    parser.add_argument("--space-intervals", type=int, nargs="+", default=[2000, 4000, 8000])
    parser.add_argument("--steps-per-day", type=int, default=32)
    args = parser.parse_args()
    counts = sorted(set(args.space_intervals))
    if (
        len(counts) < 3 or any(n < 100 or n % 4 for n in counts)
        or args.steps_per_day < 1 or args.control_points < 3
        or not np.isfinite(args.smoothing) or args.smoothing < 0
        or args.output_name in ("", ".", "..")
        or Path(args.output_name).name != args.output_name
    ):
        raise ValueError("Use three grid counts divisible by four and valid study settings.")
    reference = AndreasenHugeSurface.load(args.model)
    if len(reference.maturities) < 2:
        raise ValueError("PDE propagation requires at least two fitted maturities.")
    rows, data = load_quotes(args.quotes_root, reference)
    times = evaluation_times(reference.maturities)
    report_y = np.linspace(-0.09, 0.09, 361)
    report_z = np.exp(report_y)
    directory = args.quotes_root / args.output_name
    diagnostics = Path("outputs/andreasen_huge_diagnostics") / args.quotes_root.name / args.output_name
    if directory.exists() or diagnostics.exists():
        raise FileExistsError("Study output already exists. Choose a fresh --output-name.")
    directory.mkdir(parents=True)
    diagnostics.mkdir(parents=True)
    models, all_prices = {}, {}
    calibration, expiry_rows, bound_rows, conditioning, shapes = [], [], [], [], []
    original_prices = np.concatenate([
        reference.call_price(d[:, 0], t) for d, t in zip(data, reference.maturities)
    ])
    mid = np.concatenate([d[:, 1] for d in data])
    half_width = np.concatenate([d[:, 2] for d in data])
    specifications = [(f"grid_{n}", 1.0, n, 3.0) for n in counts]
    finest = f"grid_{counts[-1]}"
    specifications += [
        ("domain_0p75", 0.75, 3 * counts[-1] // 4, 3.0),
        ("cap_100pct", 1.0, counts[0], 1.0),
        ("cap_600pct", 1.0, counts[0], 6.0),
    ]
    requests = [(t / 365, "right") for t in (0.25, 1.0)
                if t / 365 < reference.maturities[0]]
    for i, t in enumerate(reference.maturities):
        requests.append((float(t), "left"))
        if i < len(reference.maturities) - 1:
            requests.append((float(t), "right"))

    print(f"Loaded {len(data)} expiries and {len(mid):,} retained quotes.", flush=True)
    for name, width, count, cap in specifications:
        print(f"Refitting {name}...", flush=True)
        model, reports = AndreasenHugeCalibrator(
            AHGrid(width, count), control_points=args.control_points,
            smoothing=args.smoothing, max_proxy_vol=cap,
        ).calibrate(reference.spot, reference.maturities, reference.forwards,
                    reference.discounts, data)
        model.save(directory / f"{name}.json")
        models[name] = model
        prices = np.concatenate([
            model.call_price(d[:, 0], t) for d, t in zip(data, model.maturities)
        ])
        all_prices[name] = prices
        calibration.append({
            "case": name, "width": width, "AH_intervals": count,
            "proxy_cap_pct": 100 * cap, **price_metrics((prices - mid) / half_width),
            "active_proxy_bounds": sum(r["active_proxy_bounds"] for r in reports),
            "max_quote_change_vs_input_model": float(np.max(np.abs(prices - original_prices))),
        })
        for i, ((path, _), d, t, nodes, parameters) in enumerate(zip(
            rows, data, model.maturities, model.controls, model.parameters
        )):
            residual = (model.call_price(d[:, 0], t) - d[:, 1]) / d[:, 2]
            expiry_rows.append({"case": name, "expiry": path.parent.name,
                               "days": 365 * t, **price_metrics(residual), **reports[i]})
            for y, p in zip(nodes, parameters):
                vol = float(np.exp(p))
                bound_rows.append({
                    "case": name, "expiry": path.parent.name, "log_moneyness": y,
                    "proxy_vol_pct": 100 * vol,
                    "near_upper_bound": bool(vol >= cap * (1 - 1e-6)),
                    "near_lower_bound": bool(vol <= 0.005 * (1 + 1e-6)),
                })
        native_y = np.log(model.grid.z[1:-1])
        for t, side in requests:
            state = model.node_state(t, side)
            for width in (0.01, 0.03, 0.09):
                mask = np.abs(native_y) <= width + 1e-12
                v = state["local_variance"][mask]
                valid = np.isfinite(v) & (v >= 0)
                lv = np.full(v.shape, np.nan)
                lv[valid] = 100 * np.sqrt(v[valid])
                index = int(np.nanargmax(lv)) if valid.any() else None
                conditioning.append({
                    "case": name, "days": 365 * t, "side": side,
                    "report_half_width": width, "nodes": int(mask.sum()),
                    "unresolved_nodes": int((~valid).sum()),
                    "peak_lv_pct": float(lv[index]) if index is not None else None,
                    "peak_y": float(native_y[mask][index]) if index is not None else None,
                    "z2_curvature_at_peak": float((model.grid.z[1:-1] ** 2 * state["curvature"])[mask][index]) if index is not None else None,
                    "time_derivative_at_peak": float(state["time_derivative"][mask][index]) if index is not None else None,
                })
        previous = np.maximum(1 - model.grid.z, 0)
        for t in times:
            calls = model.node_state(t)["calls"]
            shapes.append({"case": name, "days": 365 * t,
                           **shape_counts(model.grid.z, calls, previous)})
            previous = calls

    pd.DataFrame(calibration).to_csv(diagnostics / "calibration_refinement.csv", index=False)
    pd.DataFrame(expiry_rows).to_csv(diagnostics / "expiry_fit.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(diagnostics / "proxy_bound_contacts.csv", index=False)
    pd.DataFrame(conditioning).to_csv(diagnostics / "conditioning.csv", index=False)
    pd.DataFrame(shapes).to_csv(diagnostics / "AH_shape_checks.csv", index=False)
    domain_change = float(np.max(np.abs(all_prices[finest] - all_prices["domain_0p75"])))
    pde_rows, error_frames, quote_frames, pde_shapes, solved = [], [], [], [], {}

    def solve_case(name, model_name, width, count, steps):
        print(f"PDE {name}: {count} intervals, {steps} steps/day...", flush=True)
        model = models[model_name]
        solver = ForwardPDESolver(
            -width, width, n_space_intervals=count,
            max_time_step=1 / (365 * steps), theta=0.5, rannacher_steps=0,
        )
        z = solver.normalized_strikes
        result = solver.solve(
            times, model.normalized_call(z, times[0]), NodeVariance(model),
            lambda t: tuple(model.normalized_call(z[[0, -1]], t)),
            time_breaks=model.maturities,
        )
        pde_z = np.exp(result.log_moneyness)
        sampled = np.stack([np.interp(report_z, pde_z, c)
                            for c in result.normalized_calls])
        target = np.stack([model.normalized_call(report_z, t) for t in times])
        scales = np.array([model.forward(t) * model.discount_factor(t) for t in times])
        errors = (sampled - target) * scales[:, None]
        shape_total = 0
        for i, t in enumerate(times):
            error_frames.append({"case": name, "days": 365 * t,
                                 "initial_condition": i == 0,
                                 "max_price_error_points": float(np.max(np.abs(errors[i])))})
            calls = result.normalized_calls[i]
            counts_at_time = shape_counts(
                pde_z, calls, result.normalized_calls[i - 1] if i else None
            )
            slopes = np.diff(calls) / np.diff(pde_z)
            pde_shapes.append({
                "case": name, "days": 365 * t, **counts_at_time,
                "minimum_slope": float(slopes.min()),
                "maximum_slope": float(slopes.max()),
                "minimum_butterfly_slope_change": float(np.diff(slopes).min()),
            })
            shape_total += sum(counts_at_time.values())
        numerical_errors = []
        for (path, _), d, t in zip(rows, data, model.maturities):
            i = int(np.argmin(np.abs(times - t)))
            price = scales[i] * np.interp(d[:, 0] / model.forward(t), pde_z,
                                         result.normalized_calls[i])
            candidate = model.call_price(d[:, 0], t)
            if i > 0:
                numerical_errors.extend(price - candidate)
            quote_frames.append(pd.DataFrame({
                "case": name, "expiry": path.parent.name, "strike": d[:, 0],
                "initial_condition": i == 0, "call_mid": d[:, 1],
                "call_half_width": d[:, 2], "candidate_price": candidate,
                "PDE_price": price, "PDE_minus_candidate": price - candidate,
            }))
        row = {
            "case": name, "AH_model": model_name, "AH_intervals": model.grid.intervals,
            "PDE_width": width, "PDE_intervals": count, "steps_per_day": steps,
            "actual_steps": result.time_steps,
            "max_price_error_points": float(np.max(np.abs(errors[1:]))),
            "rms_price_error_points": float(np.sqrt(np.mean(errors[1:] ** 2))),
            "max_quote_error_points": float(np.max(np.abs(numerical_errors))),
            "sampled_shape_violations": int(shape_total),
        }
        pde_rows.append(row)
        solved[name] = sampled * scales[:, None]
        if name == "time_fine":
            np.savez_compressed(directory / "PDE_grid.npz",
                                log_moneyness=result.log_moneyness,
                                maturities=times, normalized_calls=result.normalized_calls)
        return row

    for n in counts:
        solve_case(f"joint_{n}", f"grid_{n}", 0.5, n, args.steps_per_day)
    solve_case("time_medium", finest, 0.5, counts[-1], 2 * args.steps_per_day)
    solve_case("time_fine", finest, 0.5, counts[-1], 4 * args.steps_per_day)
    solve_case("PDE_domain_0p75", finest, 0.75, 3 * counts[-1] // 2, 2 * args.steps_per_day)
    for row in pde_rows:
        if row["AH_model"] == finest:
            row["max_change_vs_time_reference"] = float(np.max(np.abs(
                solved[row["case"]][1:] - solved["time_fine"][1:]
            )))
    pde_domain_change = float(np.max(np.abs(
        solved["PDE_domain_0p75"][1:] - solved["time_medium"][1:]
    )))
    joint = pd.DataFrame(pde_rows[:len(counts)])
    joint["observed_joint_order"] = np.r_[np.nan, np.log(
        joint["max_price_error_points"].to_numpy()[:-1] /
        joint["max_price_error_points"].to_numpy()[1:]
    ) / np.log(np.array(counts[1:]) / np.array(counts[:-1]))]
    joint.to_csv(diagnostics / "joint_refinement.csv", index=False)
    pd.DataFrame(pde_rows).to_csv(diagnostics / "PDE_cases.csv", index=False)
    pd.DataFrame(error_frames).to_csv(diagnostics / "maturity_errors.csv", index=False)
    pd.DataFrame(pde_shapes).to_csv(diagnostics / "PDE_shape_checks.csv", index=False)
    pd.concat(quote_frames, ignore_index=True).to_csv(diagnostics / "PDE_quote_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(joint["AH_intervals"], joint["max_price_error_points"], "o-")
    axes[0].set(xlabel="AH intervals; PDE uses half the log spacing",
                ylabel="Maximum PDE-minus-AH price error, points")
    for row in pde_rows[:len(counts)]:
        frame = pd.DataFrame(error_frames)
        frame = frame[(frame["case"] == row["case"]) & ~frame["initial_condition"]]
        axes[1].semilogy(frame["days"], frame["max_price_error_points"], label=row["case"])
    axes[1].set(xlabel="Days", ylabel="Maximum price error, points")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(diagnostics / "AH_validation.png", dpi=160)
    plt.close(figure)
    audit = {
        "input_sha256": {str(p): sha256(p) for p in [args.model, *[p for p, _ in rows]]},
        "source_sha256": {str(p): sha256(p) for p in [
            Path(__file__), Path(inspect.getfile(AndreasenHugeSurface)),
            Path(inspect.getfile(ForwardPDESolver)),
        ]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "settings": {"smoothing": args.smoothing, "control_points": args.control_points,
                     "AH_intervals": counts, "base_steps_per_day": args.steps_per_day},
        "PDE_scope": "First fitted maturity to last; initial curve and boundaries supplied by AH.",
        "PDE_variance": "Linear interpolation in log-strike of recovered discrete Dupire variance.",
        "time_sides": "Old endpoint right; new endpoint left; steps split at every pillar.",
        "error_scope": "Initial maturity excluded; report strikes lie inside PDE boundaries.",
        "report_domain": [-0.09, 0.09], "evaluation_maturities": len(times),
        "conditioning_scope": "Native AH nodes in each reported window; positive sampled maturities.",
        "max_AH_domain_quote_change_points": domain_change,
        "max_PDE_domain_price_change_points": pde_domain_change,
        "joint_errors_decrease": bool(np.all(np.diff(joint["max_price_error_points"]) < 0)),
        "AH_sampled_shape_violations": int(pd.DataFrame(shapes).iloc[:, 2:].to_numpy().sum()),
        "PDE_sampled_shape_violations": int(sum(r["sampled_shape_violations"] for r in pde_rows)),
        "time_reference_shape_violations": next(
            r["sampled_shape_violations"] for r in pde_rows if r["case"] == "time_fine"
        ),
        "shape_tolerances": {"normalized_price": 1e-10,
                             "normalized_slope": 1e-8, "slope_change": 1e-8},
        "values_clipped": False, "day_zero_validated": False,
        "backward_prices_and_greeks_validated": False,
        "candidate_promoted": False,
    }
    (diagnostics / "validation_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("\nCalibration refinement:")
    print(pd.DataFrame(calibration)[[
        "case", "rms_half_spreads", "outside_original_bands", "active_proxy_bounds",
        "max_quote_change_vs_input_model",
    ]].to_string(index=False))
    print("\nJoint AH/PDE refinement:")
    print(joint[["AH_intervals", "PDE_intervals", "max_price_error_points",
                 "max_quote_error_points", "observed_joint_order"]].to_string(index=False))
    print("\nPDE time/domain cases:")
    print(pd.DataFrame(pde_rows)[[
        "case", "steps_per_day", "max_price_error_points",
        "max_change_vs_time_reference", "sampled_shape_violations",
    ]].to_string(index=False))
    print(f"\nAH domain quote-price change: {domain_change:.6g} points")
    print(f"PDE domain price change:     {pde_domain_change:.6g} points")
    print(f"Joint errors decrease:      {audit['joint_errors_decrease']}")
    print(f"AH sampled shape violations: {audit['AH_sampled_shape_violations']}")
    print(f"PDE sampled shape violations: {audit['PDE_sampled_shape_violations']}")
    print(f"Time-reference shape violations: {audit['time_reference_shape_violations']}")
    print(f"\nModels and PDE grid: {directory.resolve()}")
    print(f"Diagnostics:        {diagnostics.resolve()}")
    print("Scope: positive-maturity consistency; day zero and Greeks remain unvalidated.")


if __name__ == "__main__":
    main()