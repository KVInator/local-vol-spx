"""Validate selected daily AH diffusions before research hedging."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import ah_backward_pricer
import ah_local_vol
import andreasen_huge
import backward_pricer
import daily_ah
import daily_ah_validation
import forward_pde
from andreasen_huge import AndreasenHugeSurface
from daily_ah import DailyAHCalibrator
from daily_ah_validation import DailyAHValidator, DailyValidationSettings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", default=[
        "2023-09-01", "2023-09-08", "2023-09-12", "2023-09-19", "2023-09-21",
    ])
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/hedging_data_diagnostics/september_2023/daily_validation"),
    )
    parser.add_argument("--radius", type=float, default=0.0005)
    parser.add_argument("--width", type=float, default=0.75)
    parser.add_argument("--wide-width", type=float, default=0.90)
    parser.add_argument("--space-intervals", type=int, nargs=2, default=[12000, 24000])
    parser.add_argument("--steps-per-day", type=int, default=64)
    args = parser.parse_args()

    settings = DailyValidationSettings(
        args.radius, args.width, args.wide_width,
        args.space_intervals[0], args.space_intervals[1], args.steps_per_day,
    )
    base = args.calibration_run.resolve()
    inputs = [
        base / name for name in [
            "model_manifest.csv", "quote_residuals.csv", "expiry_summary.csv", "audit.json"
        ]
    ]
    parent_audit = json.loads(inputs[3].read_text(encoding="utf-8"))
    for path in inputs[:3]:
        expected = parent_audit.get("output_sha256", {}).get(path.name)
        if expected is None or digest(path) != expected:
            raise ValueError(f"Calibration output checksum mismatch: {path.name}.")
    manifest, quotes, carry = [
        pd.read_csv(path, dtype={"quote_date": str, "root": str})
        for path in inputs[:3]
    ]

    dates = sorted(set(args.dates))
    selected = manifest.loc[manifest.quote_date.isin(dates)].copy()
    if set(selected.quote_date) != set(dates) or not selected.status.eq("fitted").all():
        raise ValueError("Every requested date must have a fitted calibration model.")
    if selected.duplicated(["quote_date", "root"]).any():
        raise ValueError("Duplicate daily model in the manifest.")

    jobs = []
    checker = DailyAHCalibrator()
    for row in selected.sort_values(["quote_date", "root"]).itertuples(index=False):
        path = (base / row.model_file).resolve()
        if not path.is_relative_to(base) or digest(path) != row.model_sha256:
            raise ValueError(f"Model path or checksum mismatch for {row.quote_date}.")
        model = AndreasenHugeSurface.load(path)
        q = quotes.loc[
            quotes.quote_date.eq(row.quote_date) & quotes.root.eq(row.root)
        ]
        c = carry.loc[
            carry.quote_date.eq(row.quote_date) & carry.root.eq(row.root)
        ]
        q, c = checker._checked(q, c)
        if (len(c) != len(model.maturities) or len(q) != row.input_quotes
                or not np.isclose(model.spot, c.spot.iloc[0], rtol=0, atol=1e-8)
                or any(
                    not np.allclose(c[field], target, rtol=1e-12, atol=1e-10)
                    for field, target in [
                        ("maturity_years", model.maturities),
                        ("forward", model.forwards),
                        ("discount_factor", model.discounts),
                    ]
                )):
            raise ValueError(f"Model/carry mismatch for {row.quote_date}.")
        jobs.append((row, path, model, q, c))
        inputs.append(path)

    output = args.output / datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    frames, status = {}, []
    validator = DailyAHValidator(settings)
    print(
        f"{len(jobs)} fixed daily models; raw and radius={settings.radius:g} diffusions.",
        flush=True,
    )
    print(
        "No refit, zero-time coefficient request, volatility clipping or automatic promotion.",
        flush=True,
    )

    for row, path, model, q, c in jobs:
        record = {
            "quote_date": row.quote_date, "root": row.root,
            "status": "failed", "message": "",
        }
        try:
            result = validator.run(
                model, q, c, progress=lambda message: print(message, flush=True)
            )
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
            record["message"] = f"{type(error).__name__}: {error}"
            print(f"FAILED {row.quote_date}: {record['message']}", flush=True)
        else:
            record["status"] = "completed"
            for name, frame in result.items():
                frames.setdefault(name, []).append(frame)
            day = (
                output / row.quote_date
                / hashlib.sha256(row.root.encode()).hexdigest()[:12]
            )
            day.mkdir(parents=True)
            for name, frame in result.items():
                frame.to_csv(day / f"{name}.csv", index=False)
        status.append(record)
        pd.DataFrame(status).to_csv(output / "study_status.csv", index=False)

    combined = {}
    for name, parts in frames.items():
        combined[name] = pd.concat(parts, ignore_index=True)
        combined[name].to_csv(output / f"{name}.csv", index=False)

    modules = [
        daily_ah_validation, daily_ah, ah_backward_pricer, ah_local_vol,
        andreasen_huge, backward_pricer, forward_pde,
    ]
    sources = [Path(__file__)] + [Path(module.__file__) for module in modules]
    audit = {
        "settings": asdict(settings),
        "cases": settings.cases(),
        "requested_dates": dates,
        "daily_status": status,
        "completed_dates": sum(r["status"] == "completed" for r in status),
        "input_sha256": {str(path): digest(path) for path in inputs},
        "source_sha256": {str(path): digest(path) for path in sources},
        "coefficient_change": (
            "Gaussian native-node variance average, nearest endpoints, four-sigma "
            "kernel; blend (1-t/T1)^2 before T1."
        ),
        "contracts": (
            "Calls at first pillar, nearest pillar to 35 days and final pillar; "
            "forward log-strikes -0.02, 0, +0.02."
        ),
        "greek_convention": (
            "Physical local-volatility function and original carry held fixed "
            "separately for each diffusion."
        ),
        "sensitivity_scope": (
            "Common physical spots inside log-spot +/-0.01; "
            "reference is numerical, not exact Greek truth."
        ),
        "bump_scope": (
            "Finite differences of the same solved conditional price curve; "
            "not an independent pricing benchmark."
        ),
        "forward_start": (
            "Intrinsic payoff at zero; initial implicit half-steps "
            "request positive-time coefficients only."
        ),
        "boundaries": "Martingale payoff asymptotes; finite PDE domains.",
        "forward_backward_scope": (
            "Original spot, generated calls, finest time grid; "
            "both solvers use the same coefficient."
        ),
        "price_minus_original_AH_scope": (
            "For smoothing, includes a changed diffusion and numerical error."
        ),
        "shape_tolerances": {
            "normalized_price": 1e-10, "slope": 1e-8,
            "slope_change": 1e-8, "conditional_delta": 1e-6,
            "conditional_price_points": 1e-6, "conditional_gamma": 1e-8,
        },
        "quote_band_tolerance_points": 1e-6,
        "native_AH_grid_held_fixed": True,
        "forward_shape_scope": (
            "Full finite domain and fixed log-strike +/-0.10; this window "
            "is not identical to each expiry's observed quote range."
        ),
        "native_grid_refinement_performed": False,
        "proxy_cap_sensitivity_performed": False,
        "funding_curve_verified": False,
        "contract_identity_verified": False,
        "snapshot_provenance_verified": False,
        "wing_robustness_certified": False,
        "parameters_refitted": False,
        "input_models_modified": False,
        "zero_time_coefficient_requested": False,
        "values_clipped": False,
        "radius_selected_automatically": False,
        "candidate_promoted": False,
        "backtest_performed": False,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
        },
        "output_sha256": {
            path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    if combined:
        fit = combined["quote_fit"]
        print("\nQuote fit:\n" + fit[[
            "quote_date", "source", "rms_half_spreads",
            "outside_original_bands", "max_PDE_minus_AH_points",
        ]].to_string(index=False))
        print("\nGreek sensitivity:\n" + combined["sensitivity"][[
            "quote_date", "radius", "check", "max_original_spot_delta_change",
            "max_curve_delta_change", "max_curve_gamma_change",
        ]].to_string(index=False))
        check = combined["forward_backward"]
        print("\nMaximum absolute forward/backward difference, points:\n" +
              check.assign(error=check.forward_minus_backward_points.abs()).groupby(
                  ["quote_date", "radius"]
              ).error.max().to_string())

    print(f"\nDiagnostics: {output.resolve()}")
    print("Completion means diagnostics were produced, not that every sensitivity check passed.")
    if any(row["status"] != "completed" for row in status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
