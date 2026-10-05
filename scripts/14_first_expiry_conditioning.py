"""Study first-expiry conditioning before day-zero PDE pricing.

The study compares stronger smoothing of the first fitted expiry.
Other expiry curves retain their existing fitted models.

For the short-end assumption w(y,T) = (T/T1) * w(y,T1), the
curvature factor at alpha = T/T1 can be written as

    g(alpha) = (1-alpha)*g0 + alpha*g1
               + alpha*(1-alpha)*w_y**2/16.

The endpoint factors therefore bound the local variance at each
evaluated strike throughout the short-end interval.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from black import BlackPricer
from convex_spline import ConvexCallSpline, ConvexSplineCalibrator
from coupled_tails import (
    CoupledTailBuilder,
    CoupledTailSlice,
    check_coupled_calendar,
)
from implied_vol import ImpliedVolSolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(manifest_path):
    specification = json.loads(manifest_path.read_text())

    if specification["model_type"] != "calendar_ordered_tail_slices":
        raise ValueError("Expected a coupled-tail slice specification.")

    models = []
    core_paths = []

    for item in specification["slices"]:
        core_path = (
            manifest_path.parent / item["core_file"]
        ).resolve()
        core = ConvexCallSpline.load(core_path)

        models.append(
            CoupledTailSlice(
                core=core,
                left_log_moneyness=item["left_log_moneyness"],
                right_log_moneyness=item["right_log_moneyness"],
                left_power=item["left_power"],
                right_power=item["right_power"],
            )
        )
        core_paths.append(core_path)

    checks = tuple(
        check_coupled_calendar(earlier, later)
        for earlier, later in zip(models[:-1], models[1:])
    )
    if not all(check["passed"] for check in checks):
        raise ValueError("The loaded coupled tails failed calendar checks.")

    return specification, tuple(models), tuple(core_paths), checks


def short_end_diagnostics(model, min_y, max_y, points, curvature_floor):
    """Evaluate the frozen-first-IV assumption at its time endpoints."""
    regular_z = np.exp(np.linspace(min_y, max_y, points))

    joins = []
    for anchor in (model.left, model.right):
        z = anchor.normalized_strike
        joins.extend(
            [np.nextafter(z, 0.0), z, np.nextafter(z, np.inf)]
        )

    z = np.unique(np.concatenate([regular_z, joins]))
    y = np.log(z)

    forward = model.core.pricer.forward
    discount = model.core.pricer.discount_factor
    maturity = model.maturity
    scale = discount * forward

    calls = model.normalized_call(z)
    puts = model.normalized_put(z)
    slopes = model.normalized_call(z, 1)
    curvatures = model.normalized_call(z, 2)

    pricer = BlackPricer(forward, discount, maturity)
    solver = ImpliedVolSolver(pricer)

    volatility = np.empty_like(z)
    repricing_errors = np.empty_like(z)

    for index, normalized_strike in enumerate(z):
        kind = "put" if normalized_strike < 1.0 else "call"
        native_price = (
            puts[index] if kind == "put" else calls[index]
        )

        result = solver.solve(
            price=float(scale * native_price),
            strike=float(forward * normalized_strike),
            kind=kind,
        )

        volatility[index] = result.volatility
        repricing_errors[index] = abs(result.price_error)

    w = volatility**2 * maturity
    root_w = np.sqrt(w)
    d1 = -y / root_w + 0.5 * root_w
    d2 = d1 - root_w
    density_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

    if not (
        np.all(np.isfinite(w))
        and np.all(w > 0.0)
        and np.all(density_d1 > 0.0)
        and np.all(curvatures > 0.0)
    ):
        raise ValueError(
            "The anchor smile has invalid or numerically unresolved "
            "variance or curvature."
        )

    black_w_derivative = density_d1 / (2.0 * root_w)
    black_y_derivative = -z * ndtr(d2)

    w_y = (
        z * slopes - black_y_derivative
    ) / black_w_derivative

    g0 = (1.0 - y * w_y / (2.0 * w)) ** 2

    # This density identity avoids cancellation when g1 is small.
    g1 = curvatures * z**2 * root_w / density_d1

    if not (
        np.all(np.isfinite(g0))
        and np.all(np.isfinite(g1))
        and np.all(g0 > 0.0)
        and np.all(g1 > 0.0)
    ):
        raise ValueError(
            "The short-end curvature factors are nonpositive or nonfinite."
        )

    lv0 = volatility / np.sqrt(g0)
    lv1 = volatility / np.sqrt(g1)

    retained = (
        (z >= model.left.normalized_strike)
        & (z <= model.right.normalized_strike)
    )
    index_curvature = curvatures * discount / forward
    floor_contact = retained & (
        index_curvature <= curvature_floor * (1.0 + 1e-4)
    )

    if not (
        np.all(np.isfinite(lv0))
        and np.all(np.isfinite(lv1))
    ):
        raise ValueError("The short-end local volatility is nonfinite.")

    frame = pd.DataFrame(
        {
            "log_moneyness": y,
            "normalized_strike": z,
            "retained_in_core": retained,
            "first_expiry_implied_volatility": volatility,
            "first_expiry_total_variance": w,
            "first_expiry_variance_y_derivative": w_y,
            "normalized_strike_curvature": curvatures,
            "index_strike_curvature": index_curvature,
            "core_curvature_floor_contact": floor_contact,
            "g_zero_time": g0,
            "g_first_expiry": g1,
            "local_volatility_zero_time_limit": lv0,
            "local_volatility_first_expiry_left": lv1,
            "iv_repricing_error_points": repricing_errors,
        }
    )

    worst = int(np.argmax(lv1))
    metrics = {
        "max_lv0_pct": float(100.0 * lv0.max()),
        "max_lv1_pct": float(100.0 * lv1.max()),
        "max_lv1_core_pct": float(100.0 * lv1[retained].max()),
        "max_lv1_tail_pct": float(100.0 * lv1[~retained].max()),
        "peak_lv1_log_moneyness": float(y[worst]),
        "peak_lv1_in_core": bool(retained[worst]),
        "minimum_core_curvature": float(
            index_curvature[retained].min()
        ),
        "core_floor_points": int(floor_contact.sum()),
        "minimum_g0": float(g0.min()),
        "minimum_g1": float(g1.min()),
        "maximum_iv_repricing_error_points": float(
            repricing_errors.max()
        ),
    }

    return frame, metrics


def quote_metrics(model, quotes, current_prices):
    forward = model.core.pricer.forward
    discount = model.core.pricer.discount_factor
    strikes = quotes["strike"].to_numpy(dtype=float)

    prices = (
        discount
        * forward
        * model.normalized_call(strikes / forward)
    )
    residuals = (
        prices - quotes["call_mid"].to_numpy(dtype=float)
    ) / quotes["call_half_width"].to_numpy(dtype=float)

    return {
        "rms_half_spreads": float(
            np.sqrt(np.mean(residuals**2))
        ),
        "max_half_spreads": float(np.abs(residuals).max()),
        "outside_bands": int(
            (np.abs(residuals) > 1.0 + 1e-6).sum()
        ),
        "max_change_vs_current_points": float(
            np.abs(prices - current_prices).max()
        ),
    }


def save_candidate(
    case_directory,
    specification,
    models,
    core_paths,
    checks,
    first_core,
):
    case_directory.mkdir(parents=True, exist_ok=True)
    first_core_path = case_directory / "first_expiry_spline.npz"
    first_core.save(first_core_path)

    slices = []
    for index, (model, item) in enumerate(
        zip(models, specification["slices"], strict=True)
    ):
        core_path = first_core_path if index == 0 else core_paths[index]
        slices.append(
            {
                "expiry": item["expiry"],
                "core_file": os.path.relpath(
                    core_path, case_directory
                ),
                **model.metadata(),
            }
        )

    manifest = {
        "format_version": 1,
        "model_type": "calendar_ordered_tail_slices",
        "left_power": models[0].left_power,
        "right_power": models[0].right_power,
        "join_continuity": "Price and slope; curvature may jump.",
        "tail_powers_are_modelling_assumptions": True,
        "all_calendar_checks_passed": True,
        "slices": slices,
        "calendar_checks": checks,
        "zero_maturity_extension_included": False,
    }

    (case_directory / "tail_slices.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tails", required=True)
    parser.add_argument("--baseline-weight", type=float, default=0.01)
    parser.add_argument(
        "--weight-factors",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0, 30.0, 100.0],
    )
    parser.add_argument("--band-slack", type=float, default=1.5)
    parser.add_argument("--curvature-floor", type=float, default=1e-8)
    parser.add_argument("--min-log-moneyness", type=float, default=-0.5)
    parser.add_argument("--max-log-moneyness", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=2001)
    args = parser.parse_args()

    if args.baseline_weight <= 0.0:
        raise ValueError("The baseline smoothing weight must be positive.")
    if any(factor <= 0.0 for factor in args.weight_factors):
        raise ValueError("Smoothing factors must be positive.")
    if not args.min_log_moneyness < 0.0 < args.max_log_moneyness:
        raise ValueError("The study domain must contain zero.")
    if args.points < 101:
        raise ValueError("Use at least 101 evaluation points.")

    manifest_path = project_path(args.tails).resolve()
    specification, current_models, core_paths, current_checks = (
        load_models(manifest_path)
    )

    original_cores = tuple(model.core for model in current_models)
    original_first = original_cores[0]
    current_first = current_models[0]

    quotes_path = core_paths[0].parent / "quotes.csv"
    quotes = pd.read_csv(quotes_path, float_precision="round_trip")
    strikes = quotes["strike"].to_numpy(dtype=float)
    midpoints = quotes["call_mid"].to_numpy(dtype=float)
    half_widths = quotes["call_half_width"].to_numpy(dtype=float)

    current_prices = (
        original_first.pricer.discount_factor
        * original_first.pricer.forward
        * current_first.normalized_call(
            strikes / original_first.pricer.forward
        )
    )

    run_directory = manifest_path.parent.parent
    run_name = run_directory.name
    data_directory = run_directory / "first_expiry_conditioning"
    diagnostic_directory = (
        PROJECT_ROOT
        / "outputs"
        / "short_end_diagnostics"
        / run_name
        / "first_expiry_conditioning"
    )
    data_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_directory.mkdir(parents=True, exist_ok=True)

    builder = CoupledTailBuilder(
        left_power=current_first.left_power,
        right_power=current_first.right_power,
    )

    factors = sorted(set([1.0, *args.weight_factors]))
    rows = []
    successful_frames = {}

    for factor in factors:
        weight = args.baseline_weight * factor
        case = "weight_" + f"{weight:.8g}".replace(".", "p")
        print(f"Evaluating {case}...", flush=True)

        try:
            if factor == 1.0:
                first_core = original_first
                models = current_models
                checks = current_checks
            else:
                first_core = ConvexSplineCalibrator(
                    pricer=original_first.pricer,
                    n_intervals=original_first.n_intervals,
                    smoothing_weight=weight,
                    band_slack=args.band_slack,
                    curvature_floor=args.curvature_floor,
                ).fit(strikes, midpoints, half_widths)

                candidate_cores = (
                    first_core,
                    *original_cores[1:],
                )
                models, checks = builder.construct(candidate_cores)

            frame, conditioning = short_end_diagnostics(
                models[0],
                args.min_log_moneyness,
                args.max_log_moneyness,
                args.points,
                args.curvature_floor,
            )
            fit = quote_metrics(models[0], quotes, current_prices)

            row = {
                "case": case,
                "status": "evaluated",
                "message": "",
                "smoothing_weight": weight,
                "weight_factor": factor,
                "left_join": models[0].left_log_moneyness,
                "right_join": models[0].right_log_moneyness,
                "calendar_checks_passed": all(
                    check["passed"] for check in checks
                ),
                **fit,
                **conditioning,
            }
            rows.append(row)
            successful_frames[case] = frame

            frame.to_csv(
                diagnostic_directory / f"{case}_anchor_diagnostics.csv",
                index=False,
            )

            save_candidate(
                data_directory / case,
                specification,
                models,
                core_paths,
                checks,
                first_core,
            )

        except (ValueError, RuntimeError, ArithmeticError) as error:
            rows.append(
                {
                    "case": case,
                    "status": "failed",
                    "message": str(error),
                    "smoothing_weight": weight,
                    "weight_factor": factor,
                }
            )
            print(f"  Failed: {error}", flush=True)

    summary = pd.DataFrame(rows)
    summary_path = diagnostic_directory / "conditioning_summary.csv"
    summary.to_csv(summary_path, index=False)

    evaluated = summary[summary["status"].eq("evaluated")]

    print("\nQuote-fit comparison:")
    if not evaluated.empty:
        print(
            evaluated[
                [
                    "case",
                    "smoothing_weight",
                    "rms_half_spreads",
                    "max_half_spreads",
                    "outside_bands",
                    "max_change_vs_current_points",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.8g}",
            )
        )

        print("\nShort-end conditioning:")
        print(
            evaluated[
                [
                    "case",
                    "max_lv0_pct",
                    "max_lv1_core_pct",
                    "max_lv1_tail_pct",
                    "core_floor_points",
                    "peak_lv1_log_moneyness",
                    "peak_lv1_in_core",
                    "calendar_checks_passed",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.8g}",
            )
        )

        print("\nSelected first-expiry joins:")
        print(
            evaluated[
                ["case", "left_join", "right_join"]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.8g}",
            )
        )

    failed = summary[summary["status"].eq("failed")]
    if not failed.empty:
        print("\nCases without completed diagnostics:")
        print(failed[["case", "message"]].to_string(index=False))

    if successful_frames:
        figure, axes = plt.subplots(1, 3, figsize=(17, 5))

        for case, frame in successful_frames.items():
            y = frame["log_moneyness"]

            axes[0].plot(
                y,
                frame["normalized_strike_curvature"],
                label=case,
            )
            axes[1].plot(
                y,
                100.0 * frame["local_volatility_zero_time_limit"],
                label=case,
            )
            axes[2].plot(
                y,
                100.0 * frame["local_volatility_first_expiry_left"],
                label=case,
            )

        titles = (
            "First-expiry strike curvature",
            "Day-zero local-volatility limit",
            "First-expiry local volatility, left side",
        )
        labels = (
            "Normalized strike curvature",
            "Local volatility (%)",
            "Local volatility (%)",
        )

        for axis, title, label in zip(axes, titles, labels, strict=True):
            axis.set(
                xlabel="Forward log-moneyness",
                ylabel=label,
                title=title,
                yscale="log",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)

        figure.tight_layout()
        figure_path = diagnostic_directory / "conditioning_comparison.png"
        figure.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"\nFigure: {figure_path}")

    sources = [manifest_path, quotes_path, *core_paths]
    audit = {
        "first_expiry": specification["slices"][0]["expiry"],
        "first_maturity_days": 365.0 * original_first.pricer.maturity,
        "baseline_smoothing_weight": args.baseline_weight,
        "weight_factors": factors,
        "band_slack": args.band_slack,
        "curvature_floor": args.curvature_floor,
        "study_log_moneyness_domain": [
            args.min_log_moneyness,
            args.max_log_moneyness,
        ],
        "short_end_assumption": (
            "First-expiry IV held constant at fixed forward moneyness."
        ),
        "local_volatility_time_sides": {
            "zero_time": "right-hand limit",
            "first_expiry": "left-hand value",
        },
        "endpoint_factor_method": (
            "g0 from first-expiry total-variance slope; "
            "g1 directly from first-expiry strike curvature."
        ),
        "other_expiry_core_models_refitted": False,
        "original_model_files_modified": False,
        "candidate_selected_automatically": False,
        "pde_pricing_performed": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in sources
        },
        "script_sha256": sha256(__file__),
    }

    audit_path = diagnostic_directory / "conditioning_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )

    print(f"\nSummary: {summary_path}")
    print(f"Audit:   {audit_path}")
    print(f"Models:  {data_directory}")
    print("This study has not replaced the primary calibrated models.")


if __name__ == "__main__":
    main()