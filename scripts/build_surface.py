"""Calibrate expiry slices and assess their cross-maturity consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surface_outputs import build_interpolated_surface

from calibration_pipeline import (
    CalibratedExpiry,
    ExpirySliceCalibrator,
)
from discount_curve import TreasuryYieldProxy
from implied_vol import ImpliedVolSolver
from market_data import SPXQuoteFile
from maturity import ExpiryConvention


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, contents: dict) -> None:
    path.write_text(
        json.dumps(contents, indent=2) + "\n",
        encoding="utf-8",
    )


def save_slice(
    result: CalibratedExpiry,
    data_directory: Path,
    diagnostic_directory: Path,
) -> None:
    expiry = result.expiry_date.isoformat()
    slice_directory = data_directory / expiry
    slice_directory.mkdir(parents=True, exist_ok=True)

    result.spline.save(slice_directory / "spline.npz")
    result.quotes.to_csv(
        slice_directory / "quotes.csv",
        index=False,
    )
    result.quote_selection.to_csv(
        slice_directory / "quote_selection.csv",
        index=False,
    )
    result.parity_selection.to_csv(
        slice_directory / "parity_selection.csv",
        index=False,
    )
    result.parity_diagnostics.to_csv(
        slice_directory / "parity_diagnostics.csv",
        index=False,
    )
    write_json(
        diagnostic_directory / f"{expiry}_audit.json",
        result.audit,
    )


def compare_slices(
    results: list[CalibratedExpiry],
    grid_settings: dict,
    data_directory: Path,
    diagnostic_directory: Path,
) -> dict:
    requested_lower = float(
        grid_settings["min_log_moneyness"]
    )
    requested_upper = float(
        grid_settings["max_log_moneyness"]
    )
    points = int(grid_settings["points"])
    tolerance = float(grid_settings["calendar_tolerance"])

    if (
        not np.isfinite([requested_lower, requested_upper]).all()
        or requested_lower >= requested_upper
        or points < 3
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("Invalid comparison-grid settings.")

    lower = max(
        requested_lower,
        max(
            np.log(
                result.spline.strike_origin
                / result.spline.pricer.forward
            )
            for result in results
        ),
    )
    upper = min(
        requested_upper,
        min(
            np.log(
                result.spline.strike_max
                / result.spline.pricer.forward
            )
            for result in results
        ),
    )

    if lower >= upper:
        raise ValueError(
            "The fitted slices have no shared comparison domain."
        )

    y_grid = np.linspace(lower, upper, points)
    times = np.array(
        [result.spline.pricer.maturity for result in results]
    )
    days = 365.0 * times

    if np.any(np.diff(times) <= 0.0):
        raise ValueError(
            "Slice maturities must be strictly increasing."
        )

    iv_rows = []
    comparison_frames = []

    for result in results:
        spline = result.spline
        pricer = spline.pricer

        # The shared domain is already inside every spline domain.
        # Clip only the endpoint roundoff from the log/exp conversion.
        strikes = np.clip(
            pricer.forward * np.exp(y_grid),
            spline.strike_origin,
            spline.strike_max,
        )
        prices = np.asarray(spline.price(strikes))
        solver = ImpliedVolSolver(pricer)

        recovered = [
            solver.solve(
                price=float(price),
                strike=float(strike),
                kind="call",
            )
            for strike, price in zip(strikes, prices)
        ]
        ivs = np.array(
            [recovery.volatility for recovery in recovered]
        )
        iv_rows.append(ivs)

        comparison_frames.append(
            pd.DataFrame(
                {
                    "expiry_date": result.expiry_date.isoformat(),
                    "maturity_years": pricer.maturity,
                    "elapsed_days": 365.0 * pricer.maturity,
                    "log_moneyness": y_grid,
                    "strike": strikes,
                    "forward": pricer.forward,
                    "discount_factor": pricer.discount_factor,
                    "call_price": prices,
                    "normalized_call_price": (
                        prices
                        / (
                            pricer.discount_factor
                            * pricer.forward
                        )
                    ),
                    "implied_volatility": ivs,
                    "total_variance": ivs**2 * pricer.maturity,
                    "strike_curvature": (
                        spline.strike_curvature(strikes)
                    ),
                    "iv_repricing_error": [
                        recovery.price_error
                        for recovery in recovered
                    ],
                }
            )
        )

    iv_matrix = np.vstack(iv_rows)
    variance_matrix = iv_matrix**2 * times[:, None]
    variance_changes = np.diff(variance_matrix, axis=0)

    calendar_rows = []
    calendar_summary = []

    for index, changes in enumerate(variance_changes):
        earlier = results[index].expiry_date.isoformat()
        later = results[index + 1].expiry_date.isoformat()
        violations = changes < -tolerance

        calendar_rows.append(
            pd.DataFrame(
                {
                    "earlier_expiry": earlier,
                    "later_expiry": later,
                    "log_moneyness": y_grid,
                    "earlier_total_variance": (
                        variance_matrix[index]
                    ),
                    "later_total_variance": (
                        variance_matrix[index + 1]
                    ),
                    "variance_change": changes,
                    "calendar_violation": violations,
                }
            )
        )
        calendar_summary.append(
            {
                "earlier_expiry": earlier,
                "later_expiry": later,
                "grid_points": points,
                "violations": int(violations.sum()),
                "minimum_variance_change": float(changes.min()),
            }
        )

    comparison = pd.concat(
        comparison_frames,
        ignore_index=True,
    )
    comparison.to_csv(
        data_directory / "comparison_grid.csv",
        index=False,
    )
    pd.concat(calendar_rows, ignore_index=True).to_csv(
        diagnostic_directory / "calendar_checks.csv",
        index=False,
    )
    calendar_summary_frame = pd.DataFrame(calendar_summary)
    calendar_summary_frame.to_csv(
        diagnostic_directory / "calendar_summary.csv",
        index=False,
    )

    np.savez_compressed(
        data_directory / "comparison_grid.npz",
        log_moneyness=y_grid,
        maturities=times,
        implied_volatility=iv_matrix,
        total_variance=variance_matrix,
        expiry_dates=np.array(
            [
                result.expiry_date.isoformat()
                for result in results
            ]
        ),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )
    for index, result in enumerate(results):
        label = (
            f"{result.expiry_date.isoformat()} "
            f"({days[index]:.0f} days)"
        )
        axes[0].plot(
            y_grid,
            100.0 * iv_matrix[index],
            label=label,
        )
        axes[1].plot(
            y_grid,
            variance_matrix[index],
            label=label,
        )

    axes[0].set_title("Independently calibrated expiry slices")
    axes[0].set_ylabel("Implied volatility (%)")
    axes[0].legend()
    axes[1].set_ylabel("Total variance")
    axes[1].set_xlabel("Forward log-moneyness")

    for axis in axes:
        axis.axvline(0.0, color="0.5", linestyle="--")
        axis.grid(alpha=0.2)

    fig.savefig(
        diagnostic_directory / "expiry_comparison.png",
        dpi=160,
    )
    plt.close(fig)

    y_mesh, days_mesh = np.meshgrid(y_grid, days)
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        y_mesh,
        days_mesh,
        100.0 * iv_matrix,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.9,
    )
    axis.set_title(
        "Independent-slice IV preview\n"
        "Calendar repair and maturity interpolation pending"
    )
    axis.set_xlabel("Forward log-moneyness")
    axis.set_ylabel("Days to fixing")
    axis.set_zlabel("Implied volatility (%)")
    fig.colorbar(
        surface,
        ax=axis,
        shrink=0.65,
        pad=0.1,
        label="Implied volatility (%)",
    )
    fig.savefig(
        diagnostic_directory / "iv_surface_preview.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nSampled calendar checks:")
    print(
        calendar_summary_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6e}",
        )
    )

    return {
        "requested_log_moneyness_domain": [
            requested_lower,
            requested_upper,
        ],
        "shared_log_moneyness_domain": [
            float(lower),
            float(upper),
        ],
        "points_per_expiry": points,
        "calendar_tolerance": tolerance,
        "calendar_violation_count": int(
            np.count_nonzero(
                variance_changes < -tolerance
            )
        ),
        "minimum_variance_change": float(
            variance_changes.min()
        ),
        "maximum_grid_iv_repricing_error": float(
            comparison["iv_repricing_error"].abs().max()
        ),
        "calendar_check_assumption": (
            "Deterministic carry and proportional dividends."
        ),
        "calendar_check_scope": (
            "Adjacent calibrated maturities on the shared sampled "
            "log-moneyness grid."
        ),
        "calendar_repair_applied": False,
        "maturity_interpolation": (
            "Not yet implemented. The 3D plot connects slices "
            "for display only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    quote_date = date.fromisoformat(config["quote_date"])
    quote_time = pd.Timestamp(config["quote_timestamp_utc"])
    if quote_time.tzinfo is None:
        raise ValueError("Quote timestamp must include a timezone.")
    quote_time = quote_time.tz_convert("UTC")

    expiries = [
        date.fromisoformat(value)
        for value in config["expiries"]
    ]
    if len(expiries) < 2 or len(set(expiries)) != len(expiries):
        raise ValueError(
            "Provide at least two distinct expiry dates."
        )
    expiries.sort()

    raw_path = project_path(config["raw_file"])
    discount_path = project_path(config["discount_reference"])
    proxy = TreasuryYieldProxy.from_json(discount_path)

    if proxy.as_of != quote_date:
        raise ValueError(
            "Discount-reference date does not match run date."
        )

    calibrator = ExpirySliceCalibrator(
        discount_proxy=proxy,
        **config["calibration"],
    )
    source = SPXQuoteFile(raw_path)

    run_name = config["run_name"]
    if (
        not isinstance(run_name, str)
        or not run_name
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789_-"
            )
            for character in run_name
        )
    ):
        raise ValueError(
            "run_name must contain only letters, digits, '-' or '_'."
        )

    data_directory = (
        PROJECT_ROOT / "data/processed/surfaces" / run_name
    )
    diagnostic_directory = (
        PROJECT_ROOT / "outputs/surface_diagnostics" / run_name
    )
    data_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_directory.mkdir(parents=True, exist_ok=True)

    records = []
    results = []
    settlement = config["settlement"]

    for expiry in expiries:
        print(f"Calibrating {expiry.isoformat()}...", flush=True)

        try:
            snapshot = source.snapshot(
                quote_date,
                expiry,
                quote_timestamp_utc=quote_time,
            )
            convention = ExpiryConvention(
                expiry_date=expiry,
                settlement_time_ny=settlement["time_ny"],
                settlement_kind=settlement["kind"],
                verification_status=settlement[
                    "verification_status"
                ],
                evidence=settlement["evidence"],
            )
            result = calibrator.calibrate(
                snapshot,
                convention,
            )
        except (ValueError, RuntimeError) as error:
            records.append(
                {
                    "expiry": expiry.isoformat(),
                    "status": "failed",
                    "message": str(error),
                }
            )
            print(f"  Failed: {error}", flush=True)
            continue

        save_slice(
            result,
            data_directory,
            diagnostic_directory,
        )
        results.append(result)

        spline = result.spline
        records.append(
            {
                "expiry": expiry.isoformat(),
                "status": "fitted",
                "message": "",
                "days": result.audit["elapsed_days"],
                "raw_rows": result.audit["raw_rows"],
                "parity_pairs": result.audit["parity"]["pairs"],
                "quotes": len(result.quotes),
                "forward": spline.pricer.forward,
                "discount_factor": spline.pricer.discount_factor,
                "minimum_multiplier": spline.minimum_multiplier,
                "band_cap": spline.band_cap,
                "rms_half_spreads": spline.rms_half_spreads,
                "max_half_spreads": spline.max_half_spreads,
                "outside_bands": spline.outside_bands,
                "roughness": spline.roughness,
                "min_curvature": spline.shape_checks()[
                    "minimum_curvature"
                ],
            }
        )
        print(
            f"  Fitted {len(result.quotes)} quotes; "
            f"RMS movement {spline.rms_half_spreads:.4f} "
            "half-spreads.",
            flush=True,
        )

    summary = pd.DataFrame(records)
    summary_path = diagnostic_directory / "expiry_summary.csv"
    summary.to_csv(summary_path, index=False)

    run_audit = {
        "stage": "independent_expiry_calibration",
        "configuration": config,
        "config_sha256": sha256(config_path),
        "raw_file": raw_path.name,
        "raw_sha256": sha256(raw_path),
        "data_provider": "unconfirmed",
        "discount_reference_sha256": sha256(discount_path),
        "discount_proxy": proxy.metadata(),
        "python_version": sys.version.split()[0],
        "package_versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "matplotlib")
        },
        "requested_expiries": len(expiries),
        "fitted_expiries": len(results),
        "expiry_results": records,
    }
    audit_path = diagnostic_directory / "run_audit.json"
    write_json(audit_path, run_audit)

    if len(results) != len(expiries):
        raise RuntimeError(
            "Some requested expiries failed. Successful models and "
            f"failure details were saved. Review {summary_path}"
        )

    print("\nExpiry fitting summary:")
    print(
        summary[
            [
                "expiry",
                "days",
                "quotes",
                "forward",
                "minimum_multiplier",
                "rms_half_spreads",
                "outside_bands",
                "min_curvature",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    run_audit["comparison"] = compare_slices(
        results,
        config["comparison_grid"],
        data_directory,
        diagnostic_directory,
    )
    run_audit["interpolated_surface"] = build_interpolated_surface(
    results,
    config["comparison_grid"],
    data_directory,
    diagnostic_directory,
    )

    run_audit["stage"] = "interpolated_call_price_surface"
    
    write_json(audit_path, run_audit)

    print(
        "\nShared log-moneyness domain:",
        run_audit["comparison"]["shared_log_moneyness_domain"],
    )
    print(
        "Sampled calendar violations:",
        run_audit["comparison"]["calendar_violation_count"],
    )
    print(f"\nModels and comparison grid: {data_directory}")
    print(f"Summary: {summary_path}")
    print(f"Audit:   {audit_path}")
    print(
        "Smile comparison:",
        diagnostic_directory / "expiry_comparison.png",
    )
    print(
        "3D preview:",
        diagnostic_directory / "iv_surface_preview.png",
    )


if __name__ == "__main__":
    main()