"""Study early AH variance smoothing, Greek sensitivity and quote-fit cost."""

import argparse
import hashlib
import inspect
import json
import platform
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr

from ah_backward_pricer import AHBackwardPricer
from ah_local_vol import AHLocalVariance
from andreasen_huge import AndreasenHugeSurface
from backward_pricer import BackwardPriceGrid
from forward_pde import ForwardPDESolver


class EarlyVarianceExperiment:
    """Experimental coefficient change, not a recalibrated AH surface.

    For 0 < t < T1, blend the recovered native-node variance with its
    Gaussian average in log-state. The blend weight is (1-t/T1)^2.
    The averaging width is fixed in log-state, independent of the PDE grid.
    Carry and all coefficients at/after T1 retain their original values.
    Prices produced by this changed diffusion must be re-evaluated.
    """

    def __init__(self, model, radius):
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Averaging radius must be finite and nonnegative.")

        self.model = model
        self.radius = float(radius)
        self.base = AHLocalVariance(model)
        self.spot = model.spot
        self.maturities = model.maturities.copy()
        self.min_log_moneyness = float(self.base.y[0])
        self.max_log_moneyness = float(self.base.y[-1])
        self.native_z = model.grid.z[1:-1]
        self.spacing = float(np.mean(np.diff(self.base.y)))
        self._nodes = lru_cache(maxsize=2)(self._averaged_nodes)

    def forward(self, time):
        return self.model.forward(time)

    def discount_factor(self, time):
        return self.model.discount_factor(time)

    def _averaged_nodes(self, time, side):
        raw = self.base.normalized_variance(self.native_z, time, side)
        averaged = gaussian_filter1d(
            raw,
            self.radius / self.spacing,
            mode="nearest",
            truncate=4.0,
        )
        weight = (1.0 - time / self.maturities[0])**2
        values = (1.0 - weight) * raw + weight * averaged

        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ArithmeticError("Invalid experimental local variance.")

        return values

    def normalized_variance(self, states, time, side="right"):
        time = float(time)

        if time <= 0:
            raise ValueError("Only positive-time variance is supported.")

        if self.radius == 0 or time >= self.maturities[0]:
            return self.base.normalized_variance(states, time, side)

        z = np.asarray(states, float)

        if z.size == 0 or not np.all(np.isfinite(z)) or np.any(z <= 0):
            raise ValueError("States must be finite, positive and nonempty.")

        y = np.log(z)

        if (
            np.any(y < self.base.y[0])
            or np.any(y > self.base.y[-1])
            or side not in ("left", "right")
        ):
            raise ValueError("Invalid states or derivative side.")

        values = self._nodes(time, side)
        return np.interp(y.ravel(), self.base.y, values).reshape(z.shape)


class AnalyticalControl:
    """Black or continuum first-resolvent AH with a constant proxy."""

    spot = 100.0
    sigma = 0.2
    min_log_moneyness = -1.0
    max_log_moneyness = 1.0

    def __init__(self, maturity, kind):
        self.maturities = np.array([maturity])
        self.kind = kind

    def forward(self, time):
        return self.spot

    def discount_factor(self, time):
        return 1.0

    def normalized_variance(self, states, time, side="right"):
        if time <= 0:
            raise ValueError("Control requires positive time.")

        z = np.asarray(states, float)

        if self.kind == "Black":
            return np.full_like(z, self.sigma**2)

        d = np.sqrt(1.0 + 8.0 / (self.sigma**2 * time))
        return 4.0 / (time * d) * (
            1.0 / d + np.abs(np.log(z)) / 2.0
        )

    def original_spot_price(self):
        time = self.maturities[0]

        if self.kind == "Black":
            return self.spot * (
                2.0 * ndtr(0.5 * self.sigma * np.sqrt(time)) - 1.0
            )

        return self.spot / np.sqrt(
            1.0 + 8.0 / (self.sigma**2 * time)
        )


def grid_greeks(result, spot):
    """Nearest native three-point differences in physical spot.

    This bypasses cubic interpolation. A shifted-grid estimate is located
    at its nearest node, whose spot is reported rather than hidden.
    """
    spots = result.valuation_forward * np.exp(result.log_states)
    i = int(np.argmin(abs(spots - spot)))

    if not 0 < i < len(spots) - 1:
        raise ValueError("Greek evaluation requires an interior node.")

    left = spots[i] - spots[i - 1]
    right = spots[i + 1] - spots[i]
    values = (
        result.price_scale
        * result.normalized_values[i - 1:i + 2]
    )

    slope_left = (values[1] - values[0]) / left
    slope_right = (values[2] - values[1]) / right
    delta = (
        right * slope_left + left * slope_right
    ) / (left + right)
    gamma = 2.0 * (
        slope_right - slope_left
    ) / (left + right)

    return float(spots[i]), float(delta), float(gamma)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_quotes(root, model):
    found = {}
    columns = ["strike", "call_mid", "call_half_width"]

    for path in sorted(root.glob("*/quotes.csv")):
        frame = pd.read_csv(path)
        required = columns + [
            "maturity_years",
            "forward",
            "discount_factor",
        ]

        if frame.empty or not set(required).issubset(frame.columns):
            raise ValueError(f"Missing retained quote fields: {path}")

        matches = np.flatnonzero(
            np.isclose(
                model.maturities,
                frame["maturity_years"].iloc[0],
                rtol=0,
                atol=1e-12,
            )
        )

        if len(matches) != 1 or int(matches[0]) in found:
            raise ValueError(
                f"Missing or duplicate quote maturity: {path}"
            )

        i = int(matches[0])

        for column, target, tolerance in (
            ("maturity_years", model.maturities[i], 1e-12),
            ("forward", model.forwards[i], 1e-7),
            ("discount_factor", model.discounts[i], 1e-10),
        ):
            if not np.allclose(
                frame[column],
                target,
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError(f"Inconsistent {column}: {path}")

        data = frame[columns].to_numpy(float)

        if (
            not np.all(np.isfinite(data))
            or np.any(data[:, [0, 2]] <= 0)
        ):
            raise ValueError(
                f"Invalid prices, strikes or spreads: {path}"
            )

        found[i] = (path, data)

    if len(found) != len(model.maturities):
        raise ValueError(
            "Retained quotes are needed for every AH pillar."
        )

    return [found[i] for i in range(len(model.maturities))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quotes-root", type=Path, required=True)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.0005, 0.001],
    )
    parser.add_argument(
        "--space-intervals",
        type=int,
        nargs=2,
        default=[8000, 16000],
    )
    parser.add_argument("--steps-per-day", type=int, default=64)
    parser.add_argument(
        "--days",
        type=float,
        nargs="+",
        default=[4, 35, 60],
    )
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument(
        "--run-name",
        default="ah_short_end_study",
    )
    args = parser.parse_args()

    counts = sorted(set(args.space_intervals))
    radii = sorted(set([0.0, *args.radii]))
    days = sorted(set(args.days))

    if (
        len(counts) != 2
        or min(counts) < 100
        or any(n % 4 for n in counts)
        or args.steps_per_day < 1
        or not np.isfinite(args.width)
        or args.width <= 0
        or not np.all(np.isfinite(radii))
        or min(radii) < 0
        or len(radii) < 2
        or not np.all(np.isfinite(days))
        or min(days) <= 0
        or Path(args.run_name).name != args.run_name
        or args.run_name in ("", ".", "..")
    ):
        raise ValueError("Invalid study settings.")

    model = AndreasenHugeSurface.load(args.model)
    quotes = load_quotes(args.quotes_root, model)

    for day in days:
        model.forward(day / 365.0)

    variants = {
        radius: EarlyVarianceExperiment(model, radius)
        for radius in radii
    }
    native_spacing = variants[0.0].spacing

    if any(0 < radius < native_spacing for radius in radii):
        raise ValueError(
            "Use smoothing radii at least as large as the native spacing."
        )

    if (
        args.width + 4.0 * max(radii)
        >= variants[0.0].max_log_moneyness
    ):
        raise ValueError(
            "Leave room for the averaging kernel inside the native domain."
        )

    settings = [
        ("coarse", counts[0], args.steps_per_day, 0.0),
        ("fine", counts[1], args.steps_per_day, 0.0),
        (
            "shifted",
            counts[1],
            args.steps_per_day,
            args.width / counts[1],
        ),
        ("time_fine", counts[1], 2 * args.steps_per_day, 0.0),
    ]

    for variant in variants.values():
        for _, n, steps, shift in settings:
            AHBackwardPricer(
                variant,
                domain_width=args.width,
                space_intervals=n,
                steps_per_day=steps,
                early_time_power=3,
                state_shift=shift,
            )

    output = (
        Path("outputs/backward_pde_diagnostics")
        / args.run_name
    )

    if output.exists():
        raise FileExistsError(
            "Choose a fresh --run-name to preserve previous results."
        )

    output.mkdir(parents=True)

    print(
        f"One fixed AH input; "
        f"{sum(len(data) for _, data in quotes):,} retained quotes."
    )
    print(
        "Experimental first-period coefficient changes; "
        "no refit or promotion."
    )

    records = []
    curves = []
    bumps = []
    controls = []
    quote_rows = []
    forward_checks = []
    consistency = []

    spot_log = np.linspace(-0.01, 0.01, 401)
    spots = model.spot * np.exp(spot_log)
    log_strikes = np.array([-0.02, 0.0, 0.02])

    for radius, variant in variants.items():
        for case, n, steps, shift in settings:
            solver = AHBackwardPricer(
                variant,
                domain_width=args.width,
                space_intervals=n,
                steps_per_day=steps,
                early_time_power=3,
                state_shift=shift,
            )

            for day in days:
                print(
                    f"Backward radius={radius:g}, "
                    f"{case}, {day:g} days...",
                    flush=True,
                )
                time = day / 365.0
                strikes = (
                    model.forward(time) * np.exp(log_strikes)
                )
                results = solver.solve_many(strikes, time)

                for y, strike, result in zip(
                    log_strikes, strikes, results
                ):
                    values = result.greeks(model.spot)
                    node_spot, node_delta, node_gamma = grid_greeks(
                        result, model.spot
                    )
                    target = float(model.call_price(strike, time))

                    records.append(
                        {
                            "radius": radius,
                            "case": case,
                            "intervals": n,
                            "steps_per_day": steps,
                            "days": day,
                            "log_strike": y,
                            "strike": strike,
                            "price": float(values.price),
                            "AH_price": target,
                            "price_minus_AH": (
                                float(values.price) - target
                            ),
                            "delta": float(values.delta),
                            "gamma": float(values.gamma),
                            "node_spot": node_spot,
                            "node_delta": node_delta,
                            "node_gamma": node_gamma,
                            **solver.last_diagnostics,
                        }
                    )

                    if y == 0:
                        profile = result.greeks(spots)
                        curves.append(
                            pd.DataFrame(
                                {
                                    "radius": radius,
                                    "case": case,
                                    "days": day,
                                    "spot": spots,
                                    "gamma": profile.gamma,
                                    "delta": profile.delta,
                                }
                            )
                        )

                    if case == "time_fine":
                        for fraction in (
                            0.001, 0.0005, 0.00025, 0.000125
                        ):
                            bump = fraction * model.spot
                            low, centre, high = result.price(
                                [
                                    model.spot - bump,
                                    model.spot,
                                    model.spot + bump,
                                ]
                            )
                            bumps.append(
                                {
                                    "radius": radius,
                                    "days": day,
                                    "log_strike": y,
                                    "bump_points": bump,
                                    "gamma": float(values.gamma),
                                    "fd_gamma": float(
                                        (high - 2 * centre + low)
                                        / bump**2
                                    ),
                                    "delta": float(values.delta),
                                    "fd_delta": float(
                                        (high - low) / (2 * bump)
                                    ),
                                }
                            )

    records = pd.DataFrame(records)
    sensitivity = []
    keys = ["days", "log_strike"]

    for radius in radii:
        subset = records[records["radius"] == radius]
        fine = subset[
            subset["case"] == "fine"
        ].set_index(keys)
        row = {"radius": radius}

        for label, case in (
            ("space", "coarse"),
            ("shift", "shifted"),
            ("time", "time_fine"),
        ):
            other = subset[
                subset["case"] == case
            ].set_index(keys)

            for field in ("price", "delta", "gamma"):
                row[f"max_{label}_{field}_change"] = float(
                    (other[field] - fine[field]).abs().max()
                )

            atm = (
                fine.index.get_level_values("log_strike") == 0
            )
            row[
                f"max_{label}_ATM_gamma_change_pct"
            ] = float(
                100
                * (
                    other.loc[atm, "gamma"]
                    / fine.loc[atm, "gamma"]
                    - 1
                ).abs().max()
            )

        ref = subset[subset["case"] == "time_fine"]
        row["max_price_change_vs_AH"] = float(
            ref["price_minus_AH"].abs().max()
        )
        sensitivity.append(row)

    for kind in ("Black", "flat_proxy_AH"):
        control = AnalyticalControl(
            float(model.maturities[0]), kind
        )

        for n in (1000, 2000, 4000, 8000):
            result = AHBackwardPricer(
                control,
                space_intervals=n,
                domain_width=0.5,
                steps_per_day=2 * args.steps_per_day,
                early_time_power=3,
            ).solve(100.0, control.maturities[0])

            values = result.greeks(100.0)
            _, _, native_gamma = grid_greeks(result, 100.0)
            root = (
                control.sigma * np.sqrt(control.maturities[0])
            )
            exact_gamma = (
                np.exp(-root**2 / 8)
                / (np.sqrt(2 * np.pi) * 100 * root)
            )

            controls.append(
                {
                    "control": kind,
                    "intervals": n,
                    "price_error": (
                        float(values.price)
                        - control.original_spot_price()
                    ),
                    "delta": float(values.delta),
                    "gamma": float(values.gamma),
                    "node_gamma": native_gamma,
                    "exact_gamma": (
                        exact_gamma if kind == "Black" else np.nan
                    ),
                }
            )

    for radius, variant in variants.items():
        print(
            f"Forward quote check radius={radius:g}...",
            flush=True,
        )
        steps = 2 * args.steps_per_day
        solver = ForwardPDESolver(
            -args.width,
            args.width,
            n_space_intervals=counts[-1],
            max_time_step=1 / (365 * steps),
            rannacher_steps=2,
        )
        z = solver.normalized_strikes
        times = np.unique(
            np.r_[
                0.0,
                model.maturities,
                [model._time(day / 365) for day in days],
            ]
        )
        first = model.maturities[0]
        early_count = max(
            8, int(np.ceil(365 * first * steps))
        )
        breaks = (
            first * np.linspace(0, 1, early_count + 1)**3
        )

        result = solver.solve(
            times,
            np.maximum(1 - z, 0),
            variant.normalized_variance,
            lambda t: (1 - z[0], 0.0),
            time_breaks=breaks,
        )

        reference = records[
            (records["radius"] == radius)
            & (records["case"] == "time_fine")
        ]

        for row in reference.itertuples(index=False):
            i = int(
                np.argmin(abs(times - row.days / 365))
            )
            scale = (
                model.forward(times[i])
                * model.discount_factor(times[i])
            )
            forward_price = scale * np.interp(
                row.strike / model.forward(times[i]),
                z,
                result.normalized_calls[i],
            )
            consistency.append(
                {
                    "radius": radius,
                    "days": row.days,
                    "log_strike": row.log_strike,
                    "backward_price": row.price,
                    "forward_price": forward_price,
                    "forward_minus_backward": (
                        forward_price - row.price
                    ),
                }
            )

        for i, (path, data) in enumerate(quotes):
            time = model.maturities[i]
            checkpoint = int(
                np.argmin(abs(times - time))
            )
            normalized = data[:, 0] / model.forward(time)

            if (
                np.any(normalized <= z[0])
                or np.any(normalized >= z[-1])
            ):
                raise ValueError(
                    "A retained quote exceeds the forward PDE domain."
                )

            scale = (
                model.forward(time)
                * model.discount_factor(time)
            )
            prices = scale * np.interp(
                normalized,
                z,
                result.normalized_calls[checkpoint],
            )
            target = model.call_price(data[:, 0], time)

            quote_rows.append(
                pd.DataFrame(
                    {
                        "radius": radius,
                        "expiry": path.parent.name,
                        "strike": data[:, 0],
                        "call_mid": data[:, 1],
                        "half_width": data[:, 2],
                        "AH_price": target,
                        "PDE_price": prices,
                        "residual_half_spreads": (
                            (prices - data[:, 1]) / data[:, 2]
                        ),
                    }
                )
            )

        for i in range(1, len(times)):
            calls = result.normalized_calls[i]
            slopes = np.diff(calls) / np.diff(z)

            forward_checks.append(
                {
                    "radius": radius,
                    "days": 365 * times[i],
                    "increasing_prices": int(
                        np.sum(slopes > 1e-8)
                    ),
                    "vertical_spread_violations": int(
                        np.sum(slopes < -1 - 1e-8)
                    ),
                    "negative_butterflies": int(
                        np.sum(np.diff(slopes) < -1e-8)
                    ),
                    "intrinsic_shortfalls": int(
                        np.sum(
                            calls
                            < np.maximum(1 - z, 0) - 1e-10
                        )
                    ),
                    "upper_bound_excesses": int(
                        np.sum(calls > 1 + 1e-10)
                    ),
                    "calendar_violations": int(
                        np.sum(
                            calls
                            < result.normalized_calls[i - 1] - 1e-10
                        )
                    ),
                }
            )

    quotes_out = pd.concat(quote_rows, ignore_index=True)
    fit = []
    baseline = quotes_out[
        quotes_out["radius"] == 0
    ].set_index(["expiry", "strike"])

    for radius, frame in quotes_out.groupby(
        "radius", sort=True
    ):
        residual = frame["residual_half_spreads"].to_numpy()
        current = frame.set_index(["expiry", "strike"])

        fit.append(
            {
                "radius": radius,
                "quotes": len(frame),
                "rms_half_spreads": float(
                    np.sqrt(np.mean(residual**2))
                ),
                "outside_bands": int(
                    np.sum(abs(residual) > 1 + 1e-8)
                ),
                "max_price_change_vs_baseline": float(
                    (
                        current["PDE_price"]
                        - baseline["PDE_price"]
                    ).abs().max()
                ),
            }
        )

    summary = pd.DataFrame(sensitivity).merge(
        pd.DataFrame(fit), on="radius"
    )
    expiry_fit = []

    for (radius, expiry), frame in quotes_out.groupby(
        ["radius", "expiry"], sort=True
    ):
        residual = frame["residual_half_spreads"].to_numpy()

        expiry_fit.append(
            {
                "radius": radius,
                "expiry": expiry,
                "rms_half_spreads": float(
                    np.sqrt(np.mean(residual**2))
                ),
                "outside_bands": int(
                    np.sum(abs(residual) > 1 + 1e-8)
                ),
            }
        )

    expiry_fit = pd.DataFrame(expiry_fit)

    for name, frame in (
        ("summary", summary),
        ("contracts", records),
        ("bump_checks", pd.DataFrame(bumps)),
        ("controls", pd.DataFrame(controls)),
        ("quotes", quotes_out),
        ("forward_shape_checks", pd.DataFrame(forward_checks)),
        (
            "forward_backward_consistency",
            pd.DataFrame(consistency),
        ),
        ("quote_fit_by_expiry", expiry_fit),
        ("gamma_curves", pd.concat(curves, ignore_index=True)),
    ):
        frame.to_csv(output / f"{name}.csv", index=False)

    figure, axes = plt.subplots(
        1,
        len(days),
        figsize=(5 * len(days), 4),
        squeeze=False,
    )
    curve_frame = pd.concat(curves, ignore_index=True)

    for axis, day in zip(axes[0], days):
        for radius in radii:
            for case, style in (
                ("fine", "-"),
                ("shifted", "--"),
            ):
                rows = curve_frame[
                    (curve_frame["radius"] == radius)
                    & (curve_frame["case"] == case)
                    & (curve_frame["days"] == day)
                ]
                axis.plot(
                    rows["spot"],
                    rows["gamma"],
                    style,
                    label=f"r={radius:g}, {case}",
                )

        axis.set(
            xlabel="Spot, fixed coefficient experiment",
            ylabel="Gamma per index point",
            title=f"{day:g}-day ATM call",
        )

    axes[0, -1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(
        output / "gamma_comparison.png", dpi=160
    )
    plt.close(figure)

    sources = [
        Path(__file__),
        Path(inspect.getfile(AHBackwardPricer)),
        Path(inspect.getfile(AHLocalVariance)),
        Path(inspect.getfile(AndreasenHugeSurface)),
        Path(inspect.getfile(BackwardPriceGrid)),
        Path(inspect.getfile(ForwardPDESolver)),
    ]
    audit = {
        "input_sha256": {
            str(p): digest(p)
            for p in [args.model, *[p for p, _ in quotes]]
        },
        "source_sha256": {
            str(p): digest(p) for p in sources
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "radii_in_log_state": radii,
        "native_log_spacing": native_spacing,
        "settings": [
            {
                "case": c,
                "intervals": n,
                "steps_per_day": s,
                "shift": h,
            }
            for c, n, s, h in settings
        ],
        "coefficient_change": (
            "Positive Gaussian variance average; "
            "blend (1-t/T1)^2 before T1."
        ),
        "kernel": (
            "Four standard deviations; nearest endpoint "
            "continuation on native nodes."
        ),
        "forward_boundaries": (
            "Intrinsic martingale asymptotes "
            "on the finite study domain."
        ),
        "control_AH": (
            "Analytical continuum first resolvent with "
            "constant proxy; actual Dupire coefficient."
        ),
        "native_greeks": (
            "Three-point physical-spot differences "
            "at the reported nearest grid node."
        ),
        "bump_scope": (
            "Same solved curve; independent Black control is included."
        ),
        "modified_diffusion_preserves_original_AH_prices": False,
        "volatility_cap_added": False,
        "prices_clipped": False,
        "input_models_modified": False,
        "models_refitted": False,
        "candidate_promoted": False,
        "wing_robustness_validated": False,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )

    print("\nGreek sensitivity and quote-fit cost:")
    print(
        summary[
            [
                "radius",
                "rms_half_spreads",
                "outside_bands",
                "max_price_change_vs_baseline",
                "max_shift_ATM_gamma_change_pct",
                "max_space_ATM_gamma_change_pct",
                "max_time_ATM_gamma_change_pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\nFirst-expiry quote fit:")
    print(
        expiry_fit[
            expiry_fit["expiry"] == quotes[0][0].parent.name
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\nFinest-time ATM values:")
    print(
        records[
            (records["case"] == "time_fine")
            & (records["log_strike"] == 0)
        ][
            [
                "radius",
                "days",
                "price_minus_AH",
                "delta",
                "gamma",
                "node_gamma",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\nAnalytical controls:")
    print(
        pd.DataFrame(controls).to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\nMaximum forward/backward difference:")
    print(
        pd.DataFrame(consistency)
        .groupby("radius")["forward_minus_backward"]
        .agg(lambda x: x.abs().max())
        .to_string()
    )

    print("\nForward shape-violation totals:")
    print(
        pd.DataFrame(forward_checks)
        .drop(columns="days")
        .groupby("radius")
        .sum()
        .to_string()
    )

    print(f"\nDiagnostics: {output.resolve()}")
    print(
        "Changed coefficients define different diffusions. "
        "No candidate has been promoted."
    )


if __name__ == "__main__":
    main()