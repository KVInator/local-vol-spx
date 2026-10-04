"""Evaluate and audit the extension from zero to the first fitted expiry."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from short_end import ShortEndSurface
from vol_surface import NormalizedCallSurface


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, contents):
    Path(path).write_text(
        json.dumps(contents, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--points", type=int, default=201)
    args = parser.parse_args()

    if args.points < 3:
        raise ValueError("At least three strike points are required.")

    manifest_path = project_path(args.surface).resolve()
    base = NormalizedCallSurface.load(manifest_path)
    extension = ShortEndSurface(base, spot=args.spot)

    data_directory = manifest_path.parent
    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "short_end_diagnostics"
        / data_directory.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    y = np.linspace(
        extension.min_log_moneyness,
        extension.max_log_moneyness,
        args.points,
    )
    z = np.exp(y)
    first_time = extension.first_maturity

    times = first_time * np.array(
        [0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
    )

    frames = []
    summaries = []
    tolerance = 1e-10

    for time in times:
        side = "left" if time == first_time else "right"
        forward = extension.forward(time)
        discount = extension.discount_factor(time)
        normalized_prices = extension.normalized_call(z, time)
        local_variance = extension.normalized_variance(
            z, time, side=side
        )
        local_volatility = np.sqrt(local_variance)

        if time == 0.0:
            iv = np.full_like(z, np.nan)
            slope = np.full_like(z, np.nan)
            curvature = np.full_like(z, np.nan)
            maturity_derivative = np.full_like(z, np.nan)
            shape_passed = True
        else:
            iv = extension.implied_volatility(forward * z, time)
            slope = extension.normalized_call(
                z, time, derivative=1
            )
            curvature = extension.normalized_call(
                z, time, derivative=2
            )
            maturity_derivative = (
                extension.normalized_time_derivative(
                    z, time, side=side
                )
            )
            shape_passed = bool(
                np.all(np.isfinite(slope))
                and np.all(np.isfinite(curvature))
                and np.all(np.isfinite(maturity_derivative))
                and np.all(slope >= -1.0 - tolerance)
                and np.all(slope <= tolerance)
                and np.all(curvature >= 0.0)
                and np.all(maturity_derivative >= 0.0)
            )

        bounds_passed = bool(
            np.all(np.isfinite(normalized_prices))
            and np.all(
                normalized_prices
                >= np.maximum(1.0 - z, 0.0) - tolerance
            )
            and np.all(normalized_prices <= 1.0 + tolerance)
        )
        variance_passed = bool(
            np.all(np.isfinite(local_variance))
            and np.all(local_variance > 0.0)
        )

        if not (shape_passed and bounds_passed and variance_passed):
            raise RuntimeError(
                f"Short-end checks failed at {365.0 * time:.8f} days."
            )

        frames.append(
            pd.DataFrame(
                {
                    "maturity_years": float(time),
                    "elapsed_days": float(365.0 * time),
                    "log_moneyness": y,
                    "strike": forward * z,
                    "forward": forward,
                    "discount_factor": discount,
                    "normalized_call_price": normalized_prices,
                    "call_price": (
                        discount * forward * normalized_prices
                    ),
                    "implied_volatility": iv,
                    "normalized_strike_slope": slope,
                    "normalized_strike_curvature": curvature,
                    "normalized_time_derivative": maturity_derivative,
                    "local_variance": local_variance,
                    "local_volatility": local_volatility,
                    "time_side": (
                        "right limit" if time == 0.0 else side
                    ),
                }
            )
        )

        summaries.append(
            {
                "days": float(365.0 * time),
                "time_side": (
                    "right limit" if time == 0.0 else side
                ),
                "min_lv_pct": float(100.0 * local_volatility.min()),
                "median_lv_pct": float(
                    100.0 * np.median(local_volatility)
                ),
                "max_lv_pct": float(100.0 * local_volatility.max()),
                "shape_passed": shape_passed,
                "bounds_passed": bounds_passed,
            }
        )

    grid = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)

    grid_path = data_directory / "short_end_grid.csv"
    summary_path = output_directory / "short_end_summary.csv"
    grid.to_csv(grid_path, index=False)
    summary.to_csv(summary_path, index=False)

    just_before = np.nextafter(first_time, 0.0)
    anchor_price_error = float(
        np.max(
            np.abs(
                extension.normalized_call(z, just_before)
                - base.normalized_call(z, first_time)
            )
        )
        * base.forward(first_time)
        * base.discount_factor(first_time)
    )

    left_lv = np.sqrt(
        extension.normalized_variance(z, first_time, side="left")
    )
    right_lv = np.sqrt(
        extension.normalized_variance(z, first_time, side="right")
    )

    specification = extension.metadata()
    specification["base_surface_file"] = manifest_path.name
    extension_path = data_directory / "short_end_extension.json"
    write_json(extension_path, specification)

    base_specification = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    input_paths = [
        manifest_path,
        PROJECT_ROOT / "src" / "short_end.py",
        PROJECT_ROOT / "src" / "vol_surface.py",
        PROJECT_ROOT / "src" / "local_vol.py",
        Path(__file__).resolve(),
    ]
    input_paths.extend(
        data_directory / name
        for name in base_specification["slice_files"]
    )

    audit = {
        **specification,
        "input_sha256": {
            str(path): sha256(path) for path in input_paths
        },
        "evaluation_points": int(len(grid)),
        "evaluation_days": [float(365.0 * time) for time in times],
        "anchor_price_join_error_points": anchor_price_error,
        "first_pillar_max_lv_jump_pp": float(
            100.0 * np.max(np.abs(right_lv - left_lv))
        ),
        "all_sampled_checks_passed": True,
        "zero_time_strike_and_time_derivatives": (
            "Not evaluated because the initial payoff has a kink."
        ),
        "short_end_endpoint_side": "left",
        "values_modified": False,
        "validation_scope": (
            "Sampled short-end prices, derivatives and local variance "
            "within the fitted surface's supported strike domain."
        ),
    }

    audit_path = output_directory / "short_end_audit.json"
    write_json(audit_path, audit)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for time, frame in zip(times, frames):
        label = f"{365.0 * time:g} days"
        axes[0].plot(
            y,
            frame["normalized_call_price"],
            label=label,
        )
        axes[1].plot(
            y,
            100.0 * frame["local_volatility"],
            label=label,
        )

    axes[0].set_title("Short-end normalized call prices")
    axes[0].set_ylabel("Normalized call price")
    axes[1].set_title("Short-end local volatility")
    axes[1].set_ylabel("Annualised local volatility (%)")

    for axis in axes:
        axis.set_xlabel("Forward log-moneyness")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure_path = output_directory / "short_end_validation.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(
        f"First fitted maturity: {365.0 * first_time:.8f} days"
    )
    print(
        "Short-end assumption: first-expiry IV held constant "
        "at fixed forward moneyness."
    )
    print(
        "Day-zero local volatility uses its right-hand limit; "
        "the first-expiry row uses its left-hand value."
    )
    print("\nShort-end diagnostics:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )
    print(
        "\nAnchor price join error: "
        f"{anchor_price_error:.3e} index points"
    )
    print(
        "Maximum local-volatility jump at the first expiry: "
        f"{audit['first_pillar_max_lv_jump_pp']:.6f} percentage points"
    )
    print("\nAll sampled checks passed.")
    print(f"Extension model: {extension_path}")
    print(f"Grid:            {grid_path}")
    print(f"Summary:         {summary_path}")
    print(f"Audit:           {audit_path}")
    print(f"Figure:          {figure_path}")


if __name__ == "__main__":
    main()