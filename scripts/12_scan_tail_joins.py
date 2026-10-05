"""Export fitted anchor values for coupled strike-tail construction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface import NormalizedCallSurface


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEFT_POWER = 3.0
RIGHT_POWER = 1.5
ENDPOINT_MARGIN = 1e-8


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def candidate_grid(start, end, points):
    if start == end:
        return np.array([start], dtype=float)

    return np.linspace(start, end, points)


def scan_side(core, expiry, side, y):
    forward = core.pricer.forward
    discount = core.pricer.discount_factor
    z = np.exp(y)
    strikes = forward * z

    call = np.asarray(core.price(strikes), dtype=float) / (
        discount * forward
    )
    slope = (
        np.asarray(core.strike_slope(strikes), dtype=float)
        / discount
    )
    curvature = (
        np.asarray(core.strike_curvature(strikes), dtype=float)
        * forward
        / discount
    )
    put = call + z - 1.0

    if side == "left":
        time_value = put
        numerator = z * (1.0 + slope)
        power = LEFT_POWER
    else:
        time_value = call
        numerator = -z * slope
        power = RIGHT_POWER

    elasticity = np.divide(
        numerator,
        time_value,
        out=np.full_like(z, np.nan),
        where=time_value > 0.0,
    )

    if side == "left":
        shape = elasticity - power
    else:
        shape = elasticity / power

    valid = (
        np.isfinite(time_value)
        & (time_value > 0.0)
        & np.isfinite(elasticity)
        & np.isfinite(curvature)
        & (curvature > 0.0)
        & (slope > -1.0)
        & (slope < 0.0)
    )

    if side == "left":
        valid &= shape >= 0.0
    else:
        valid &= shape > 0.0

    log_coefficient = np.full_like(z, np.nan)
    curvature_ratio = np.full_like(z, np.nan)
    right_conditional_mean = np.full_like(z, np.nan)

    if side == "left":
        # Proposed put tail:
        # P(z) = P(a) * (z/a)**p / [1 + B * (1 - z/a)]
        #
        # B = a * P_z(a) / P(a) - p.
        #
        # Near zero, P(z) behaves as A * z**p.
        log_coefficient[valid] = (
            np.log(time_value[valid])
            - power * y[valid]
            - np.log1p(shape[valid])
        )

        tail_curvature = (
            time_value[valid]
            / z[valid]**2
            * (
                power * (power - 1.0)
                + 2.0 * power * shape[valid]
                + 2.0 * shape[valid]**2
            )
        )
    else:
        # Proposed call tail:
        # C(z) = C(b) * [1 + B * (z/b - 1)]**(-q)
        #
        # B = -b * C_z(b) / [q * C(b)].
        #
        # At infinity, C(z) behaves as A * z**(-q).
        log_coefficient[valid] = (
            np.log(time_value[valid])
            + power * (y[valid] - np.log(shape[valid]))
        )

        tail_curvature = (
            power
            * (power + 1.0)
            * time_value[valid]
            * shape[valid]**2
            / z[valid]**2
        )

        right_conditional_mean[valid] = (
            z[valid]
            * (elasticity[valid] + 1.0)
            / elasticity[valid]
        )

    curvature_ratio[valid] = (
        tail_curvature / curvature[valid]
    )

    return pd.DataFrame(
        {
            "expiry": expiry,
            "maturity_years": core.pricer.maturity,
            "days": 365.0 * core.pricer.maturity,
            "side": side,
            "log_moneyness": y,
            "normalized_strike": z,
            "normalized_call_price": call,
            "normalized_put_price": put,
            "normalized_call_slope": slope,
            "normalized_strike_curvature": curvature,
            "anchor_time_value": time_value,
            "anchor_slope_elasticity": elasticity,
            "proposed_far_tail_power": power,
            "proposed_shape_parameter": shape,
            "candidate_admissible": valid,
            "log_asymptotic_coefficient": log_coefficient,
            "tail_to_core_curvature_ratio": curvature_ratio,
            "right_tail_conditional_mean_over_forward": (
                right_conditional_mean
            ),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--points", type=int, default=401)
    args = parser.parse_args()

    if args.points < 3:
        raise ValueError("At least three candidate points are required.")

    surface_path = project_path(args.surface)
    specification = json.loads(surface_path.read_text())
    surface = NormalizedCallSurface.load(surface_path)

    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "strike_tail_diagnostics"
        / surface_path.parent.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    frames = []
    summaries = []
    source_paths = [surface_path]

    for core, relative_file in zip(
        surface.splines,
        specification["slice_files"],
        strict=True,
    ):
        spline_path = surface_path.parent / relative_file
        expiry = spline_path.parent.name
        source_paths.append(spline_path)

        available_lower = float(
            np.log(core.strike_origin / core.pricer.forward)
        )
        available_upper = float(
            np.log(core.strike_max / core.pricer.forward)
        )

        left_start = max(
            -0.09, available_lower + ENDPOINT_MARGIN
        )
        right_start = min(
            0.09, available_upper - ENDPOINT_MARGIN
        )

        if not left_start < 0.0 < right_start:
            raise ValueError(
                f"{expiry} does not support joins on both sides of forward."
            )

        grids = {
            "left": candidate_grid(
                left_start,
                max(left_start, -0.01),
                args.points,
            ),
            "right": candidate_grid(
                right_start,
                min(right_start, 0.01),
                args.points,
            ),
        }

        for side, y in grids.items():
            frame = scan_side(core, expiry, side, y)
            frames.append(frame)
            admissible = frame.loc[frame["candidate_admissible"]]

            summaries.append(
                {
                    "expiry": expiry,
                    "days": 365.0 * core.pricer.maturity,
                    "side": side,
                    "candidates": len(frame),
                    "admissible": len(admissible),
                    "outermost_admissible_y": (
                        float(admissible.iloc[0]["log_moneyness"])
                        if len(admissible)
                        else np.nan
                    ),
                    "minimum_log_coefficient": (
                        float(
                            admissible["log_asymptotic_coefficient"].min()
                        )
                        if len(admissible)
                        else np.nan
                    ),
                    "maximum_log_coefficient": (
                        float(
                            admissible["log_asymptotic_coefficient"].max()
                        )
                        if len(admissible)
                        else np.nan
                    ),
                }
            )

    anchors = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)

    anchors_path = output_directory / "candidate_tail_anchors.csv"
    summary_path = output_directory / "candidate_tail_anchor_summary.csv"

    anchors.to_csv(anchors_path, index=False)
    summary.to_csv(summary_path, index=False)

    audit = {
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in source_paths
        },
        "script_sha256": sha256(Path(__file__)),
        "proposed_left_power": LEFT_POWER,
        "proposed_right_power": RIGHT_POWER,
        "candidate_rows": len(anchors),
        "purpose": (
            "Export fitted prices and derivatives for joint tail-join "
            "selection. Admissibility concerns individual anchors; "
            "calendar consistency has not been certified."
        ),
        "models_replaced": False,
    }

    audit_path = output_directory / "candidate_tail_anchor_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nCandidate anchor coverage:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )
    print(f"\nCandidate rows: {len(anchors):,}")
    print(f"Anchor data: {anchors_path}")
    print(f"Summary:     {summary_path}")
    print(f"Audit:       {audit_path}")


if __name__ == "__main__":
    main()