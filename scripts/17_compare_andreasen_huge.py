"""Compare regularized AH candidates with the existing spline/tail model."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from andreasen_huge import AHGrid, AndreasenHugeCalibrator
from black import BlackPricer
from tailed_surface import TailedShortEndSurface


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_metrics(values):
    values = np.asarray(values, float)
    return {
        "rms_half_spreads": float(np.sqrt(np.mean(values ** 2))),
        "max_half_spreads": float(np.max(abs(values))),
        "outside_original_bands": int(
            np.sum(abs(values) > 1.0 + 1e-8)
        ),
    }


def shape_metrics(grid, calls, previous):
    slopes = np.diff(calls) / np.diff(grid.z)
    return {
        "increasing_price_intervals": int(np.sum(slopes > 1e-8)),
        "vertical_spread_violations": int(
            np.sum(slopes < -1.0 - 1e-8)
        ),
        "negative_butterflies": int(
            np.sum(np.diff(slopes) < -1e-8)
        ),
        "intrinsic_shortfalls": int(np.sum(
            calls < np.maximum(1.0 - grid.z, 0.0) - 1e-10
        )),
        "upper_bound_excesses": int(np.sum(calls > 1.0 + 1e-10)),
        "calendar_violations": int(
            np.sum(calls < previous - 1e-10)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--quotes-root", type=Path, required=True)
    parser.add_argument(
        "--smoothing-weights",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0]
    )
    parser.add_argument("--control-points", type=int, default=31)
    parser.add_argument("--space-intervals", type=int, default=2000)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--report-width", type=float, default=0.09)
    parser.add_argument("--max-evaluations", type=int, default=300)
    parser.add_argument(
        "--output-name", default="andreasen_huge_comparison"
    )
    args = parser.parse_args()
    weights = sorted(set(args.smoothing_weights))

    if (
        not weights or not np.all(np.isfinite(weights))
        or min(weights) < 0.0
        or not np.isfinite(args.report_width)
        or not 0.0 < args.report_width < args.width
        or Path(args.output_name).name != args.output_name
        or args.output_name in ("", ".", "..")
    ):
        raise ValueError("Invalid study settings or output name.")

    baseline = TailedShortEndSurface.load(args.surface)
    if args.report_width > min(
        -baseline.min_log_moneyness,
        baseline.max_log_moneyness
    ):
        raise ValueError(
            "The report domain exceeds the existing candidate's domain."
        )

    manifest = json.loads(args.surface.read_text(encoding="utf-8"))
    tails_path = (
        args.surface.parent / manifest["tail_slices_file"]
    ).resolve()
    tails = json.loads(tails_path.read_text(encoding="utf-8"))
    rows = tails["slices"]
    times = baseline.surface.maturities
    if len(rows) != len(times):
        raise ValueError("Expiry manifest and loaded surface disagree.")

    inputs = {
        str(args.surface): sha256(args.surface),
        str(tails_path): sha256(tails_path)
    }
    quote_frames, quotes = [], []

    for row, time in zip(rows, times):
        if abs(float(row["maturity_years"]) - time) > 1e-14:
            raise ValueError("Expiry maturity mismatch.")
        path = args.quotes_root / row["expiry"] / "quotes.csv"
        frame = pd.read_csv(path)
        required = ["strike", "call_mid", "call_half_width"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Missing quote columns in {path}.")
        inputs[str(path)] = sha256(path)
        core_path = (
            tails_path.parent / row["core_file"]
        ).resolve()
        inputs[str(core_path)] = sha256(core_path)
        quote_frames.append(frame)
        quotes.append(frame[required].to_numpy(float))

    forwards = np.array([baseline.forward(t) for t in times])
    discounts = np.array([baseline.discount_factor(t) for t in times])
    grid = AHGrid(args.width, args.space_intervals)
    data_root = args.quotes_root / args.output_name
    output_root = (
        Path("outputs/andreasen_huge_diagnostics")
        / args.quotes_root.name
        / args.output_name
    )
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    y = np.log(grid.z[1:-1])
    mask = abs(y) <= args.report_width + 1e-12
    if np.sum(mask) < 21:
        raise ValueError(
            "Increase spatial intervals: too few nodes in the report domain."
        )
    report_z, report_y = grid.z[1:-1][mask], y[mask]
    summary, expiry_reports, conditioning, shape_rows = [], [], [], []
    all_quotes, diagnostic_frames = [], []

    slice_requests = [
        (0.25 / 365.0, "right"),
        (1.0 / 365.0, "right")
    ]
    slice_requests = [
        (t, side) for t, side in slice_requests if t < times[-1]
    ]
    for index, time in enumerate(times):
        slice_requests.append((float(time), "left"))
        if index < len(times) - 1:
            slice_requests.append((float(time), "right"))

    models = {"current": baseline}
    baseline_prices = [
        baseline.call_price(data[:, 0], t)
        for t, data in zip(times, quotes)
    ]

    def record_model(name, model, optimizer_reports=None):
        quote_parts = []
        for index, (row, t, data) in enumerate(zip(rows, times, quotes)):
            prices = model.call_price(data[:, 0], t)
            residual = (prices - data[:, 1]) / data[:, 2]
            quote_parts.append(residual)
            report = {
                "case": name,
                "expiry": row["expiry"],
                "days": float(365.0 * t),
                "quotes": len(data),
                **residual_metrics(residual),
                "max_price_change_vs_current": float(np.max(abs(
                    prices - baseline_prices[index]
                ))),
            }
            if optimizer_reports is not None:
                report.update(optimizer_reports[index])
            expiry_reports.append(report)

            frame = quote_frames[index].copy()
            frame["case"], frame["expiry"] = name, row["expiry"]
            frame["model_call_price"] = prices
            frame["residual_half_spreads"] = residual
            frame["current_candidate_call"] = baseline_prices[index]
            all_quotes.append(frame)

        peak = 0.0
        invalid_total = 0
        for t, side in slice_requests:
            if name == "current":
                variance = np.asarray(
                    model.normalized_variance(report_z, t, side)
                )
                curvature = np.asarray(
                    model.normalized_call(report_z, t, derivative=2)
                )
                derivative = np.asarray(
                    model.normalized_time_derivative(report_z, t, side)
                )
            else:
                state = model.node_state(t, side)
                variance, curvature, derivative = [
                    state[key][mask] for key in (
                        "local_variance",
                        "curvature",
                        "time_derivative"
                    )
                ]

            valid = np.isfinite(variance) & (variance >= 0.0)
            lv = np.full(len(report_z), np.nan)
            lv[valid] = 100.0 * np.sqrt(variance[valid])
            invalid = int(np.sum(~valid))
            invalid_total += invalid
            maximum = (
                float(np.nanmax(lv)) if np.any(valid) else np.nan
            )
            if np.any(valid):
                peak = max(peak, maximum)

            conditioning.append({
                "case": name,
                "days": float(365.0 * t),
                "side": side,
                "peak_local_vol_pct": maximum,
                "peak_log_moneyness": (
                    float(report_y[np.nanargmax(lv)])
                    if np.any(valid) else np.nan
                ),
                "minimum_z2_curvature": float(
                    np.min(report_z ** 2 * curvature)
                ),
                "minimum_time_derivative": float(
                    np.min(derivative)
                ),
                "unresolved_local_variance_nodes": invalid,
            })
            diagnostic_frames.append(pd.DataFrame({
                "case": name,
                "days": float(365.0 * t),
                "side": side,
                "log_moneyness": report_y,
                "local_volatility_pct": lv,
                "z2_curvature": report_z ** 2 * curvature,
                "normalized_time_derivative": derivative,
            }))

        checks = []
        if name != "current":
            evaluation_times = sorted(set([
                *times,
                *np.linspace(times[0] / 4.0, times[-1], 31)
            ]))
            previous = np.maximum(1.0 - grid.z, 0.0)
            for t in evaluation_times:
                calls = model.node_state(t)["calls"]
                check = shape_metrics(grid, calls, previous)
                checks.append(sum(check.values()))
                shape_rows.append({
                    "case": name,
                    "days": float(365.0 * t),
                    **check
                })
                previous = calls

        summary.append({
            "case": name,
            **residual_metrics(np.concatenate(quote_parts)),
            "max_sampled_local_vol_pct": peak,
            "unresolved_local_variance_samples": invalid_total,
            "discrete_shape_violations": (
                int(sum(checks)) if checks else None
            ),
        })

    record_model("current", baseline)
    print(
        f"Loaded {len(times)} expiries and "
        f"{sum(map(len, quotes)):,} retained quotes.",
        flush=True
    )

    for weight in weights:
        name = "ah_" + f"{weight:g}".replace(".", "p")
        print(f"Calibrating {name}...", flush=True)
        model, reports = AndreasenHugeCalibrator(
            grid,
            args.control_points,
            weight,
            max_evaluations=args.max_evaluations
        ).calibrate(
            baseline.spot,
            times,
            forwards,
            discounts,
            quotes,
            progress=lambda index, t: print(
                f"  Fitting {rows[index]['expiry']}...",
                flush=True
            )
        )

        directory = data_root / name
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory / "ah_surface.json")
        models[name] = model
        record_model(name, model, reports)

        iv_y = np.linspace(-args.report_width, args.report_width, 61)
        iv_times = np.unique(np.r_[
            times,
            np.linspace(times[0], times[-1], 25)
        ])
        iv_rows, price_errors = [], []

        for t in iv_times:
            f, d = model.forward(t), model.discount_factor(t)
            strikes = f * np.exp(iv_y)
            iv = model.implied_volatility(strikes, t)
            errors = (
                BlackPricer(f, d, t).price(strikes, iv, "call")
                - model.call_price(strikes, t)
            )
            price_errors.extend(np.abs(errors))
            iv_rows.append(pd.DataFrame({
                "maturity_years": t,
                "days": 365.0 * t,
                "log_moneyness": iv_y,
                "strike": strikes,
                "implied_volatility": iv,
                "iv_repricing_error_points": errors,
            }))

        iv_data = pd.concat(iv_rows, ignore_index=True)
        iv_data.to_csv(directory / "iv_grid.csv", index=False)
        summary[-1]["maximum_iv_repricing_error_points"] = float(
            max(price_errors)
        )

        yy, tt = np.meshgrid(iv_y, 365.0 * iv_times)
        figure = plt.figure(figsize=(10, 7))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_surface(
            yy,
            tt,
            100.0 * iv_data["implied_volatility"].to_numpy().reshape(
                yy.shape
            ),
            cmap="viridis",
            linewidth=0
        )
        axis.set_xlabel("Forward log-moneyness")
        axis.set_ylabel("Days")
        axis.set_zlabel("Implied volatility (%)")
        axis.set_title(name + ": discrete AH price interpolation")
        figure.savefig(
            output_root / (name + "_iv_surface.png"),
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(figure)

    comparison = pd.DataFrame(summary)
    comparison.to_csv(
        output_root / "comparison_summary.csv", index=False
    )
    pd.DataFrame(expiry_reports).to_csv(
        output_root / "expiry_comparison.csv", index=False
    )
    pd.DataFrame(conditioning).to_csv(
        output_root / "conditioning_summary.csv", index=False
    )
    pd.DataFrame(shape_rows).to_csv(
        output_root / "discrete_shape_checks.csv", index=False
    )
    pd.concat(all_quotes, ignore_index=True).to_csv(
        output_root / "quote_comparison.csv", index=False
    )
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostics.to_csv(
        output_root / "local_variance_diagnostics.csv", index=False
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(
        comparison["case"], comparison["rms_half_spreads"]
    )
    axes[0, 0].set_ylabel("Quote RMS, half-spreads")
    axes[0, 0].tick_params(axis="x", rotation=25)

    selected = [
        (times[0], "left"),
        (times[0], "right"),
        (times[-1], "left")
    ]
    for axis, (t, side) in zip(axes.flat[1:], selected):
        for name in models:
            frame = diagnostics[
                (diagnostics["case"] == name)
                & np.isclose(
                    diagnostics["days"], 365.0 * t, atol=1e-9
                )
                & (diagnostics["side"] == side)
            ]
            axis.plot(
                frame["log_moneyness"],
                frame["local_volatility_pct"],
                label=name
            )
        axis.set_title(f"{365.0 * t:g} days, {side}")
        axis.set_xlabel("Forward log-moneyness")
        axis.set_ylabel("Local volatility (%)")
        axis.set_yscale("log")
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output_root / "comparison.png", dpi=160)
    plt.close(figure)

    audit = {
        "input_sha256": inputs,
        "source_sha256": {
            name: sha256(Path("src") / name)
            for name in (
                "andreasen_huge.py",
                "black.py",
                "implied_vol.py",
                "tailed_surface.py",
                "short_end.py",
                "coupled_tails.py",
                "convex_spline.py",
                "local_vol.py"
            )
        },
        "script_sha256": sha256(__file__),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "same_retained_quotes_and_carry": True,
        "calibration_target": (
            "Original retained call-equivalent midpoints, "
            "weighted by half-spread."
        ),
        "regularization": (
            "smoothing * sum(diff(log(proxy_volatility))**2)"
        ),
        "proxy_volatility_bounds": [0.005, 3.0],
        "bounds_apply_to": (
            "Calibration proxy, not recovered Dupire local volatility."
        ),
        "AH_derivative_scope": (
            "Finite-difference curvature and resolvent time derivative "
            "at interior nodes."
        ),
        "shape_check_scope": (
            "Discrete strike grid and sampled maturities on finite domain."
        ),
        "local_vol_peak_scope": (
            "Sampled positive maturities and both sides of pillars "
            "within report domain."
        ),
        "report_grid_points": int(len(report_z)),
        "zero_time_local_vol_evaluated": False,
        "global_tail_extension": False,
        "candidate_selected_automatically": False,
        "existing_models_modified": False,
        "independent_pde_repricing_performed": False,
    }
    (output_root / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nDiagnostics: {output_root}")
    print(f"AH model files: {data_root}")
    print(
        "Discrete AH candidates only; grid/domain refinement "
        "and independent PDE validation remain."
    )


if __name__ == "__main__":
    main()