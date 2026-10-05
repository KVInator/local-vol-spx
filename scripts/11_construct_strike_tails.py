"""Construct and audit strike-tail candidates from saved expiry models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from strike_tails import PowerTailSlice, check_tail_calendar
from vol_surface import NormalizedCallSurface


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Positive numerical allowance for the left-tail admissibility condition.
# Units are normalized call price.
LEFT_TANGENT_MARGIN = 1e-8

# Search toward the previously validated central region when necessary.
LEFT_INNER_LIMIT = -0.01

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


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def left_tangent_margin(core, log_moneyness):
    """Return the margin required for a left power greater than one.

    For normalized call price c and normalized strike z:

        margin = 1 - c + z * c_z
               = z * (1 + c_z) - normalized_put.

    The left-tail power satisfies:

        power - 1 = margin / normalized_put.

    For a convex core, the margin increases with strike because its
    derivative with respect to z is z * c_zz.
    """
    forward = core.pricer.forward
    discount = core.pricer.discount_factor
    z = float(np.exp(log_moneyness))
    strike = forward * z

    call = float(core.price(strike)) / (discount * forward)
    slope = float(core.strike_slope(strike)) / discount
    put = call + z - 1.0

    margin = float(z * (1.0 + slope) - put)

    if not np.isfinite(margin):
        raise ValueError("The left-anchor tangent margin is not finite.")

    return margin


def select_left_join(core, requested_join):
    """Retain the requested join or find an admissible inward join."""
    requested_margin = left_tangent_margin(core, requested_join)

    if requested_margin >= LEFT_TANGENT_MARGIN:
        return {
            "join": float(requested_join),
            "adjusted": False,
            "requested_margin": requested_margin,
            "selected_margin": requested_margin,
        }

    inner_join = max(float(requested_join), LEFT_INNER_LIMIT)
    inner_margin = left_tangent_margin(core, inner_join)

    if (
        inner_join <= requested_join
        or inner_margin < LEFT_TANGENT_MARGIN
    ):
        raise ValueError(
            "No admissible left join was found between the requested "
            "join and the central search limit."
        )

    selected_join = float(
        brentq(
            lambda y: (
                left_tangent_margin(core, y)
                - LEFT_TANGENT_MARGIN
            ),
            float(requested_join),
            inner_join,
            xtol=1e-12,
            rtol=1e-12,
        )
    )
    selected_margin = left_tangent_margin(core, selected_join)

    if selected_margin < 0.99 * LEFT_TANGENT_MARGIN:
        raise ValueError(
            "The selected left join failed its numerical margin check."
        )

    return {
        "join": selected_join,
        "adjusted": True,
        "requested_margin": requested_margin,
        "selected_margin": selected_margin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--join-width", type=float, default=0.09)
    args = parser.parse_args()

    if not np.isfinite(args.join_width) or args.join_width <= 0.0:
        raise ValueError("Join width must be finite and positive.")

    surface_path = project_path(args.surface)
    source_directory = surface_path.parent
    specification = json.loads(surface_path.read_text())
    surface = NormalizedCallSurface.load(surface_path)

    slice_files = specification["slice_files"]

    if len(slice_files) != len(surface.splines):
        raise ValueError("Surface manifest and loaded slices disagree.")

    data_directory = source_directory / "strike_tail_candidates"
    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "strike_tail_diagnostics"
        / source_directory.name
    )
    data_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    models = []
    expiry_names = []
    summaries = []
    model_specs = []
    quote_frames = []
    source_paths = [surface_path]

    for core, relative_file in zip(surface.splines, slice_files):
        spline_path = source_directory / relative_file
        quotes_path = spline_path.parent / "quotes.csv"
        expiry = spline_path.parent.name
        forward = core.pricer.forward
        discount = core.pricer.discount_factor

        source_paths.extend([spline_path, quotes_path])

        available_lower = float(
            np.log(core.strike_origin / forward)
        )
        available_upper = float(
            np.log(core.strike_max / forward)
        )

        requested_lower = max(
            -args.join_width,
            available_lower + ENDPOINT_MARGIN,
        )
        upper = min(
            args.join_width,
            available_upper - ENDPOINT_MARGIN,
        )

        print(f"Constructing tails for {expiry}...")

        try:
            left_selection = select_left_join(core, requested_lower)
            lower = left_selection["join"]
            model = PowerTailSlice(core, lower, upper)
        except ValueError as error:
            summaries.append(
                {
                    "expiry": expiry,
                    "status": "invalid_anchor",
                    "message": str(error),
                    "days": 365.0 * core.pricer.maturity,
                    "requested_left_log_moneyness": requested_lower,
                    "right_log_moneyness": upper,
                }
            )
            continue

        if left_selection["adjusted"]:
            print(
                "  Left join adjusted: "
                f"{requested_lower:.8f} -> {lower:.8f}"
            )

        quotes = pd.read_csv(
            quotes_path, float_precision="round_trip"
        )
        strikes = quotes["strike"].to_numpy(dtype=float)
        z_quotes = strikes / forward

        original_prices = core.price(strikes)
        tail_prices = (
            discount
            * forward
            * model.normalized_call(z_quotes)
        )
        half_widths = quotes["call_half_width"].to_numpy(dtype=float)
        midpoint = quotes["call_mid"].to_numpy(dtype=float)

        residuals = (tail_prices - midpoint) / half_widths
        in_core = (
            (z_quotes >= model.left.normalized_strike)
            & (z_quotes <= model.right.normalized_strike)
        )

        quote_frames.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "strike": strikes,
                    "log_moneyness": np.log(z_quotes),
                    "retained_in_core": in_core,
                    "original_fitted_call": original_prices,
                    "tail_extended_call": tail_prices,
                    "price_change_points": tail_prices - original_prices,
                    "residual_half_spreads": residuals,
                    "outside_original_bands": (
                        np.abs(residuals) > 1.0 + 1e-8
                    ),
                }
            )
        )

        metadata = model.metadata()
        selection_metadata = {
            "requested_left_log_moneyness": requested_lower,
            "left_join_adjusted": left_selection["adjusted"],
            "requested_left_tangent_margin": (
                left_selection["requested_margin"]
            ),
            "selected_left_tangent_margin": (
                left_selection["selected_margin"]
            ),
        }

        summaries.append(
            {
                "expiry": expiry,
                "status": "constructed",
                "message": "",
                "days": 365.0 * model.maturity,
                "quotes": len(quotes),
                "quotes_retained_in_core": int(in_core.sum()),
                "quotes_in_tail": int((~in_core).sum()),
                **metadata,
                **selection_metadata,
                "max_quote_price_change": float(
                    np.max(np.abs(tail_prices - original_prices))
                ),
                "rms_half_spreads": float(
                    np.sqrt(np.mean(residuals**2))
                ),
                "max_half_spreads": float(
                    np.max(np.abs(residuals))
                ),
                "outside_bands": int(
                    np.sum(np.abs(residuals) > 1.0 + 1e-8)
                ),
            }
        )

        models.append(model)
        expiry_names.append(expiry)
        model_specs.append(
            {
                "expiry": expiry,
                "source_spline_file": f"../{relative_file}",
                **metadata,
                **selection_metadata,
            }
        )

    summary = pd.DataFrame(summaries)
    summary_path = output_directory / "tail_join_summary.csv"
    summary.to_csv(summary_path, index=False)

    audit_inputs = {
        str(path.relative_to(PROJECT_ROOT)): sha256(path)
        for path in source_paths
    }

    if len(models) != len(surface.splines):
        failures = summary.loc[
            summary["status"] != "constructed",
            ["expiry", "message"],
        ]

        write_json(
            output_directory / "tail_audit.json",
            {
                "input_sha256": audit_inputs,
                "status": "anchor_construction_failed",
                "requested_join_width": args.join_width,
                "left_tangent_margin_required": LEFT_TANGENT_MARGIN,
                "left_inner_search_limit": LEFT_INNER_LIMIT,
                "failures": failures.to_dict(orient="records"),
            },
        )

        print("\nTail construction failures:")
        print(failures.to_string(index=False))
        print(f"\nSummary: {summary_path}")

        raise SystemExit(
            "Some anchors do not admit these tails. "
            "Review the saved summary before proceeding."
        )

    calendar_rows = []

    for index in range(len(models) - 1):
        earlier_expiry = expiry_names[index]
        later_expiry = expiry_names[index + 1]

        print(
            f"Checking calendar order: "
            f"{earlier_expiry} to {later_expiry}..."
        )

        calendar_rows.append(
            {
                "earlier_expiry": earlier_expiry,
                "later_expiry": later_expiry,
                **check_tail_calendar(
                    models[index], models[index + 1]
                ),
            }
        )

    calendar = pd.DataFrame(calendar_rows)
    calendar_path = output_directory / "tail_calendar_checks.csv"
    calendar.to_csv(calendar_path, index=False)

    quote_path = output_directory / "tail_quote_comparison.csv"
    pd.concat(quote_frames, ignore_index=True).to_csv(
        quote_path, index=False
    )

    y = np.linspace(-0.5, 0.5, 2001)
    z = np.exp(y)
    grid_frames = []

    for expiry, model in zip(expiry_names, models):
        call = model.normalized_call(z)
        slope = model.normalized_call(z, derivative=1)
        curvature = model.normalized_call(z, derivative=2)

        if (
            np.any(~np.isfinite(call))
            or np.any(~np.isfinite(slope))
            or np.any(~np.isfinite(curvature))
            or np.any(call < np.maximum(1.0 - z, 0.0) - 1e-12)
            or np.any(call > 1.0 + 1e-12)
            or np.any(slope < -1.0 - 1e-12)
            or np.any(slope > 1e-12)
            or np.any(curvature < -1e-12)
        ):
            raise RuntimeError(f"Tail shape check failed for {expiry}.")

        grid_frames.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "maturity_years": model.maturity,
                    "log_moneyness": y,
                    "normalized_strike": z,
                    "normalized_call_price": call,
                    "normalized_put_price": model.normalized_put(z),
                    "normalized_strike_slope": slope,
                    "normalized_strike_curvature": curvature,
                }
            )
        )

    grid_path = data_directory / "tail_slice_grid.csv"
    pd.concat(grid_frames, ignore_index=True).to_csv(
        grid_path, index=False
    )

    calendar_passed = bool(calendar["passed"].all())

    manifest_path = data_directory / "tail_candidates.json"
    write_json(
        manifest_path,
        {
            "format_version": 1,
            "model_type": "power_tail_slice_candidates",
            "base_surface_file": f"../{surface_path.name}",
            "requested_join_width": args.join_width,
            "endpoint_margin": ENDPOINT_MARGIN,
            "left_tangent_margin_required": LEFT_TANGENT_MARGIN,
            "left_inner_search_limit": LEFT_INNER_LIMIT,
            "left_join_selection": (
                "Retain the requested join when admissible; otherwise "
                "solve for an inward join with a positive tangent margin."
            ),
            "slices": model_specs,
            "calendar_checks": calendar_rows,
            "all_calendar_checks_passed": calendar_passed,
            "candidate_only": True,
            "short_end_revalidated": False,
            "pde_from_zero_validated": False,
        },
    )

    plot_y = np.linspace(-0.2, 0.2, 1001)
    plot_z = np.exp(plot_y)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    for model in models:
        call = model.normalized_call(plot_z)
        put = model.normalized_put(plot_z)
        otm = np.where(plot_z < 1.0, put, call)
        positive = otm > 0.0
        label = f"{365.0 * model.maturity:.0f} days"

        axes[0].semilogy(
            plot_y[positive],
            otm[positive],
            label=label,
        )
        axes[1].plot(
            plot_y,
            model.normalized_call(plot_z, derivative=2),
            label=label,
        )

    axes[0].set_title("Normalized OTM prices with power tails")
    axes[0].set_ylabel("Normalized option price")
    axes[1].set_title("Density and curvature joins")
    axes[1].set_ylabel("Normalized strike curvature")

    for axis in axes:
        axis.set_xlabel("Forward log-moneyness")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure_path = output_directory / "strike_tail_candidates.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    write_json(
        output_directory / "tail_audit.json",
        {
            "input_sha256": audit_inputs,
            "source_sha256": {
                "src/strike_tails.py": sha256(
                    PROJECT_ROOT / "src/strike_tails.py"
                ),
                "scripts/11_construct_strike_tails.py": sha256(
                    Path(__file__)
                ),
            },
            "slices_constructed": len(models),
            "left_joins_adjusted": int(
                summary["left_join_adjusted"].sum()
            ),
            "left_tangent_margin_required": LEFT_TANGENT_MARGIN,
            "left_inner_search_limit": LEFT_INNER_LIMIT,
            "all_calendar_checks_passed": calendar_passed,
            "calendar_price_tolerance": 1e-12,
            "calendar_log_ratio_tolerance": 1e-12,
            "calendar_scope": (
                "Analytical comparisons on both unbounded tails; "
                "adaptive convex-slope bounds on the bounded middle."
            ),
            "join_continuity": "Price and slope; curvature may jump.",
            "original_saved_models_replaced": False,
            "short_end_revalidated": False,
            "pde_from_zero_validated": False,
        },
    )

    print("\nTail joins:")
    print(
        summary[
            [
                "expiry",
                "days",
                "left_log_moneyness",
                "right_log_moneyness",
                "left_join_adjusted",
                "left_power",
                "right_power",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print("\nQuote-fit comparison:")
    print(
        summary[
            [
                "expiry",
                "quotes_in_tail",
                "max_quote_price_change",
                "rms_half_spreads",
                "max_half_spreads",
                "outside_bands",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print("\nGlobal calendar checks:")
    print(
        calendar[
            [
                "earlier_expiry",
                "later_expiry",
                "left_tail_ordered",
                "right_tail_ordered",
                "middle_status",
                "middle_intervals_checked",
                "passed",
            ]
        ].to_string(index=False)
    )

    print(
        "\nMaximum probability-mass error: "
        f"{(summary['total_probability_mass'] - 1.0).abs().max():.3e}"
    )
    print(
        "Maximum normalized-mean error: "
        f"{(summary['normalized_mean'] - 1.0).abs().max():.3e}"
    )
    print(f"All calendar checks passed: {calendar_passed}")
    print(f"\nJoin summary: {summary_path}")
    print(f"Calendar checks: {calendar_path}")
    print(f"Quote comparison: {quote_path}")
    print(f"Candidate specification: {manifest_path}")
    print(f"Figure: {figure_path}")

    if not calendar_passed:
        raise SystemExit(
            "Tail candidates were saved, but calendar checks did not "
            "all pass. Resolve those results before PDE use."
        )


if __name__ == "__main__":
    main()