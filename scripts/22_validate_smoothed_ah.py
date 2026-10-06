"""Fixed-radius AH grid, numerical-domain and unquoted-wing sensitivity."""

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

from ah_backward_pricer import AHBackwardPricer
from ah_local_vol import AHLocalVariance
from andreasen_huge import AHGrid, AndreasenHugeSurface
from backward_pricer import BackwardPriceGrid
from forward_pde import ForwardPDESolver


class StudyVariance:
    """Same early average as study 21; optional unquoted-wing stress.

    Radius is fixed in log-state units, never in grid cells. Wing factors
    multiply volatility, with a smooth transition outside the observed
    quote range of the active expiry period. They change the diffusion.
    """

    def __init__(
        self,
        model,
        radius,
        quote_bounds,
        wing_factor=1.0,
        transition=0.02,
    ):
        self.model, self.radius = model, float(radius)
        self.base = AHLocalVariance(model)
        self.spot, self.maturities = model.spot, model.maturities.copy()
        self.min_log_moneyness, self.max_log_moneyness = self.base.y[[0, -1]]
        self.spacing = float(np.mean(np.diff(self.base.y)))
        self.bounds = np.asarray(quote_bounds, float)
        self.wing_factor = float(wing_factor)
        self.transition = float(transition)

        if (
            not np.isfinite(radius)
            or radius < 0
            or transition <= 0
            or not np.isfinite(wing_factor)
            or wing_factor <= 0
            or self.bounds.shape != (len(self.maturities), 2)
        ):
            raise ValueError("Invalid coefficient settings.")

        self._nodes = lru_cache(maxsize=2)(self._averaged_nodes)

    def forward(self, time):
        return self.model.forward(time)

    def discount_factor(self, time):
        return self.model.discount_factor(time)

    def _averaged_nodes(self, time, side):
        raw = self.base.normalized_variance(
            self.model.grid.z[1:-1], time, side
        )
        averaged = gaussian_filter1d(
            raw,
            self.radius / self.spacing,
            mode="nearest",
            truncate=4.0,
        )
        weight = (1.0 - time / self.maturities[0]) ** 2
        return (1.0 - weight) * raw + weight * averaged

    def normalized_variance(self, states, time, side="right"):
        time = self.model._time(time)
        z = np.asarray(states, float)

        if (
            time <= 0
            or side not in ("left", "right")
            or z.size == 0
            or not np.all(np.isfinite(z))
            or np.any(z <= 0)
        ):
            raise ValueError(
                "Require positive time/states and a valid side."
            )

        y = np.log(z)
        if np.any(y < self.base.y[0]) or np.any(y > self.base.y[-1]):
            raise ValueError("Coefficient extrapolation is unsupported.")

        if self.radius == 0 or time >= self.maturities[0]:
            values = self.base.normalized_variance(z, time, side)
        else:
            values = np.interp(
                y.ravel(),
                self.base.y,
                self._nodes(time, side),
            ).reshape(z.shape)

        if self.wing_factor != 1.0:
            i = min(
                int(np.searchsorted(self.maturities, time, side=side)),
                len(self.maturities) - 1,
            )
            lower, upper = self.bounds[i]
            distance = np.maximum(
                np.maximum(lower - y, y - upper), 0.0
            )
            ramp = np.minimum(distance / self.transition, 1.0)
            ramp = ramp**2 * (3.0 - 2.0 * ramp)
            values = values * (
                1.0 + (self.wing_factor**2 - 1.0) * ramp
            )

        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ArithmeticError("Invalid experimental local variance.")

        return values


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
            raise ValueError(f"Missing retained-quote fields: {path}")

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
                frame[column], target, rtol=0, atol=tolerance
            ):
                raise ValueError(f"Inconsistent {column}: {path}")

        data = frame.sort_values("strike")[columns].to_numpy(float)
        if (
            not np.all(np.isfinite(data))
            or np.any(data[:, [0, 2]] <= 0)
            or len(np.unique(data[:, 0])) != len(data)
        ):
            raise ValueError(
                f"Invalid prices, strikes or spreads: {path}"
            )

        found[i] = (path, data)

    if len(found) != len(model.maturities):
        raise ValueError("Need retained quotes for every AH pillar.")

    return [found[i] for i in range(len(model.maturities))]


def fit_metrics(residual):
    return {
        "rms_half_spreads": float(np.sqrt(np.mean(residual**2))),
        "outside_bands": int(np.sum(abs(residual) > 1.0 + 1e-8)),
    }


def node_gamma(result, spot):
    states = result.valuation_forward * np.exp(result.log_states)
    i = int(np.argmin(abs(states - spot)))

    if not 0 < i < len(states) - 1:
        raise ValueError("Native gamma requires an interior node.")

    left = states[i] - states[i - 1]
    right = states[i + 1] - states[i]
    prices = (
        result.price_scale * result.normalized_values[i - 1 : i + 2]
    )
    gamma = (
        2.0
        * (
            (prices[2] - prices[1]) / right
            - (prices[1] - prices[0]) / left
        )
        / (left + right)
    )
    return float(states[i]), float(gamma)


def shape_counts(z, calls, previous):
    slopes = np.diff(calls) / np.diff(z)
    return {
        "increasing_prices": int(np.sum(slopes > 1e-8)),
        "vertical_spread_violations": int(
            np.sum(slopes < -1.0 - 1e-8)
        ),
        "negative_butterflies": int(
            np.sum(np.diff(slopes) < -1e-8)
        ),
        "intrinsic_shortfalls": int(
            np.sum(calls < np.maximum(1 - z, 0) - 1e-10)
        ),
        "upper_bound_excesses": int(np.sum(calls > 1 + 1e-10)),
        "calendar_violations": int(
            np.sum(calls < previous - 1e-10)
        ),
    }


def shape_audit(z, calls, previous, windows):
    """Keep full-domain flags and also report the requested windows."""
    scoped = []

    for window in windows:
        mask = abs(np.log(z)) <= window + 1e-12
        if mask.sum() < 3:
            raise ValueError(
                "A shape-check window needs at least three PDE nodes."
            )
        scoped.append(
            {
                "log_strike_half_width": window,
                **shape_counts(
                    z[mask], calls[mask], previous[mask]
                ),
            }
        )

    changes = np.diff(np.diff(calls) / np.diff(z))
    locations = [
        {
            "log_moneyness": float(np.log(z[i + 1])),
            "butterfly_slope_change": float(changes[i]),
        }
        for i in np.flatnonzero(changes < -1e-8)
    ]
    return shape_counts(z, calls, previous), scoped, locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quotes-root", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.0005)
    parser.add_argument(
        "--native-intervals",
        type=int,
        nargs="+",
        default=[4000, 8000, 16000],
    )
    parser.add_argument("--pde-intervals", type=int, default=16000)
    parser.add_argument("--steps-per-day", type=int, default=128)
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument(
        "--days", type=float, nargs="+", default=[4, 35, 60]
    )
    parser.add_argument(
        "--log-strikes",
        type=float,
        nargs="+",
        default=[-0.06, -0.02, 0, 0.02, 0.06],
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.09],
    )
    parser.add_argument(
        "--wing-factors", type=float, nargs=2, default=[0.8, 1.2]
    )
    parser.add_argument(
        "--wing-transition", type=float, default=0.02
    )
    parser.add_argument(
        "--run-name", default="ah_smoothed_robustness"
    )
    args = parser.parse_args()

    counts = sorted(set(args.native_intervals))
    days = sorted(set(args.days))
    strikes_y = sorted(set(args.log_strikes))
    windows = sorted(set(args.windows))
    numeric = [
        args.radius,
        args.width,
        args.wing_transition,
        *days,
        *strikes_y,
        *windows,
        *args.wing_factors,
    ]

    if (
        not np.all(np.isfinite(numeric))
        or args.radius <= 0
        or args.width <= 0
        or args.wing_transition <= 0
        or len(counts) < 3
        or min(counts) < 100
        or any(n % 4 for n in counts)
        or args.pde_intervals < 100
        or args.pde_intervals % 4
        or args.steps_per_day < 2
        or args.steps_per_day % 2
        or not days
        or min(days) <= 0
        or not strikes_y
        or not windows
        or min(windows) <= 0
        or max(windows) >= args.width
        or max(abs(np.array(strikes_y))) >= args.width
        or min(args.wing_factors) <= 0
        or len(set(args.wing_factors)) != 2
        or any(f == 1.0 for f in args.wing_factors)
        or args.run_name in ("", ".", "..")
        or Path(args.run_name).name != args.run_name
    ):
        raise ValueError("Invalid study settings.")

    original = AndreasenHugeSurface.load(args.model)
    quotes = load_quotes(args.quotes_root, original)
    times = np.array(
        [original._time(day / 365.0) for day in days]
    )
    days = list(365.0 * times)

    bounds = np.array(
        [
            [
                np.log(d[:, 0] / f).min(),
                np.log(d[:, 0] / f).max(),
            ]
            for (_, d), f in zip(quotes, original.forwards)
        ]
    )
    if args.radius < 2.0 * original.grid.width / counts[0]:
        raise ValueError(
            "Radius must be at least the coarsest native spacing."
        )

    models = {
        n: AndreasenHugeSurface(
            AHGrid(original.grid.width, n),
            original.spot,
            original.maturities,
            original.forwards,
            original.discounts,
            original.controls,
            original.parameters,
        )
        for n in counts
    }
    variants = {
        n: StudyVariance(m, args.radius, bounds)
        for n, m in models.items()
    }

    n = args.pde_intervals
    steps = args.steps_per_day
    width = args.width

    cases = [
        (f"native_{k}", k, width, n, steps, 0.0, 1.0)
        for k in counts
    ]
    reference_name = f"native_{counts[-1]}"

    cases += [
        (
            "pde_coarse",
            counts[-1],
            width,
            n // 2,
            steps,
            0.0,
            1.0,
        ),
        (
            "time_coarse",
            counts[-1],
            width,
            n,
            steps // 2,
            0.0,
            1.0,
        ),
        (
            "half_cell_shift",
            counts[-1],
            width,
            n,
            steps,
            width / n,
            1.0,
        ),
        (
            "domain_wide",
            counts[-1],
            1.5 * width,
            3 * n // 2,
            steps,
            0.0,
            1.0,
        ),
    ]
    cases += [
        (
            f"wing_{factor:g}",
            counts[-1],
            width,
            n,
            steps,
            0.0,
            factor,
        )
        for factor in args.wing_factors
    ]

    for name, k, w, intervals, s, shift, factor in cases:
        variant = StudyVariance(
            models[k],
            args.radius,
            bounds,
            factor,
            args.wing_transition,
        )
        if (
            w + abs(shift) + 4.0 * args.radius
            >= variant.max_log_moneyness
        ):
            raise ValueError(
                "Leave room for the Gaussian kernel inside the AH domain."
            )

        AHBackwardPricer(
            variant,
            domain_width=w,
            space_intervals=intervals,
            steps_per_day=s,
            early_time_power=3,
            state_shift=shift,
        )
        if windows[0] < 2.0 * w / intervals:
            raise ValueError(
                "Shape-check windows must span at least three PDE nodes."
            )

    output = (
        Path("outputs/backward_pde_diagnostics") / args.run_name
    )
    model_output = args.quotes_root / args.run_name
    if output.exists() or model_output.exists():
        raise FileExistsError(
            "Use a fresh --run-name to preserve previous results."
        )
    output.mkdir(parents=True)
    model_output.mkdir(parents=True)

    for k, model in models.items():
        model.save(model_output / f"frozen_proxy_{k}.json")

    print(
        f"Fixed radius {args.radius:g}; proxy controls and carry "
        "are unchanged.",
        flush=True,
    )
    print(
        "Native grids are rebuilt without optimization. "
        "Wing stresses change the diffusion.",
        flush=True,
    )

    envelopes = pd.DataFrame(
        {
            "days": 365 * original.maturities,
            "observed_min_y": bounds[:, 0],
            "observed_max_y": bounds[:, 1],
        }
    )
    envelopes.to_csv(output / "quote_coverage.csv", index=False)

    mid = np.concatenate([d[:, 1] for _, d in quotes])
    spreads = np.concatenate([d[:, 2] for _, d in quotes])
    input_prices = np.concatenate(
        [
            original.call_price(d[:, 0], t)
            for (_, d), t in zip(quotes, original.maturities)
        ]
    )

    grid_rows = []
    for k, model in models.items():
        prices = np.concatenate(
            [
                model.call_price(d[:, 0], t)
                for (_, d), t in zip(quotes, model.maturities)
            ]
        )
        grid_rows.append(
            {
                "native_intervals": k,
                **fit_metrics((prices - mid) / spreads),
                "max_quote_change_vs_input": float(
                    abs(prices - input_prices).max()
                ),
            }
        )
    pd.DataFrame(grid_rows).to_csv(
        output / "frozen_proxy_quote_fit.csv", index=False
    )

    spot_y = np.linspace(-windows[-1], windows[-1], 1801)
    spots = original.spot * np.exp(spot_y)
    curves, contracts, case_settings = {}, [], []

    for name, k, w, intervals, s, shift, factor in cases:
        model = models[k]
        variant = (
            variants[k]
            if factor == 1.0
            else StudyVariance(
                model,
                args.radius,
                bounds,
                factor,
                args.wing_transition,
            )
        )
        solver = AHBackwardPricer(
            variant,
            domain_width=w,
            space_intervals=intervals,
            steps_per_day=s,
            early_time_power=3,
            state_shift=shift,
        )
        case_settings.append(
            {
                "case": name,
                "native_intervals": k,
                "width": w,
                "pde_intervals": intervals,
                "steps_per_day": s,
                "state_shift": shift,
                "wing_vol_factor": factor,
            }
        )

        frames = []
        for day, time in zip(days, times):
            print(
                f"Backward {name}: {day:g} days, "
                f"{len(strikes_y)} calls...",
                flush=True,
            )
            strikes = model.forward(time) * np.exp(strikes_y)
            results = solver.solve_many(strikes, time)

            for y, strike, result in zip(
                strikes_y, strikes, results
            ):
                values = result.greeks(spots)
                at_spot = result.greeks(original.spot)
                native_spot, gamma = node_gamma(
                    result, original.spot
                )

                contracts.append(
                    {
                        "case": name,
                        "days": day,
                        "log_strike": y,
                        "price": float(at_spot.price),
                        "delta": float(at_spot.delta),
                        "gamma": float(at_spot.gamma),
                        "node_spot": native_spot,
                        "node_gamma": gamma,
                        "price_minus_original_AH": (
                            float(at_spot.price)
                            - float(original.call_price(strike, time))
                        ),
                        **solver.last_diagnostics,
                    }
                )

                scale = (
                    model.forward(time) * model.discount_factor(time)
                )
                lower = scale * np.maximum(
                    spots / original.spot
                    - strike / model.forward(time),
                    0,
                )

                frames.append(
                    pd.DataFrame(
                        {
                            "days": day,
                            "log_strike": y,
                            "spot_log_change": spot_y,
                            "spot": spots,
                            "price": values.price,
                            "delta": values.delta,
                            "gamma": values.gamma,
                            "negative_gamma": values.gamma < -1e-8,
                            "delta_bound_violation": (
                                (values.delta < -1e-6)
                                | (
                                    values.delta
                                    > scale / original.spot + 1e-6
                                )
                            ),
                            "price_bound_violation": (
                                (values.price < lower - 1e-6)
                                | (
                                    values.price
                                    > scale * spots / original.spot
                                    + 1e-6
                                )
                            ),
                        }
                    )
                )

        curves[name] = pd.concat(frames, ignore_index=True)
        curves[name].to_csv(
            output / f"curves_{name}.csv", index=False
        )
        pd.DataFrame(contracts).to_csv(
            output / "contracts.csv", index=False
        )

    comparisons = []
    reference = curves[reference_name]

    for name, frame in curves.items():
        for window in windows:
            mask = (
                abs(frame.spot_log_change) <= window + 1e-12
            )
            row = {
                "case": name,
                "spot_log_half_width": window,
            }

            for field in ("price", "delta", "gamma"):
                difference = (
                    frame[field].to_numpy()
                    - reference[field].to_numpy()
                )
                indices = np.flatnonzero(mask.to_numpy())
                i = indices[
                    int(np.argmax(abs(difference[indices])))
                ]

                row[f"max_{field}_change"] = float(
                    abs(difference[i])
                )
                row[f"{field}_max_change_days"] = float(
                    frame.days.iloc[i]
                )
                row[f"{field}_max_change_log_strike"] = float(
                    frame.log_strike.iloc[i]
                )
                row[f"{field}_max_change_spot_y"] = float(
                    frame.spot_log_change.iloc[i]
                )

            for field in (
                "negative_gamma",
                "delta_bound_violation",
                "price_bound_violation",
            ):
                row[field + "_samples"] = int(
                    frame.loc[mask, field].sum()
                )

            comparisons.append(row)

    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(
        output / "conditional_sensitivity.csv", index=False
    )
    contracts = pd.DataFrame(contracts)

    reference_contracts = contracts[
        contracts.case == reference_name
    ].set_index(["days", "log_strike"])
    at_spot = []

    for name in curves:
        frame = contracts[
            contracts.case == name
        ].set_index(["days", "log_strike"])
        row = {"case": name}

        for field in ("price", "delta", "gamma"):
            row[f"max_{field}_change"] = float(
                abs(
                    frame[field] - reference_contracts[field]
                ).max()
            )

        if 0.0 in strikes_y:
            ref = reference_contracts.xs(
                0.0, level="log_strike"
            ).gamma
            gamma = frame.xs(
                0.0, level="log_strike"
            ).gamma
            row["max_ATM_gamma_change_pct"] = (
                float(100 * abs(gamma / ref - 1).max())
                if np.all(abs(ref) > 1e-12)
                else None
            )

        at_spot.append(row)

    pd.DataFrame(at_spot).to_csv(
        output / "original_spot_sensitivity.csv", index=False
    )

    quote_frames, shape_rows = [], []
    consistency, fit_rows, expiry_rows = [], [], []
    window_shapes, butterfly_locations = [], []

    forward_cases = cases[: len(counts)] + [
        c
        for c in cases
        if c[0] == "domain_wide" or c[-1] != 1.0
    ]

    for name, k, w, intervals, s, shift, factor in forward_cases:
        print(
            f"Forward retained-quote check: {name}...",
            flush=True,
        )
        model = models[k]
        variant = (
            variants[k]
            if factor == 1.0
            else StudyVariance(
                model,
                args.radius,
                bounds,
                factor,
                args.wing_transition,
            )
        )
        solver = ForwardPDESolver(
            -w,
            w,
            n_space_intervals=intervals,
            max_time_step=1 / (365 * s),
            rannacher_steps=2,
        )
        z = solver.normalized_strikes
        checkpoints = np.unique(
            np.r_[0.0, model.maturities, times]
        )
        breaks = model.maturities[0] * np.linspace(
            0,
            1,
            max(
                8,
                int(
                    np.ceil(
                        365 * model.maturities[0] * s
                    )
                ),
            )
            + 1,
        ) ** 3

        result = solver.solve(
            checkpoints,
            np.maximum(1 - z, 0),
            variant.normalized_variance,
            lambda t: (1 - z[0], 0.0),
            time_breaks=breaks,
        )

        rows = []
        for (path, data), time in zip(
            quotes, model.maturities
        ):
            i = int(np.argmin(abs(checkpoints - time)))
            normalized = data[:, 0] / model.forward(time)

            if (
                np.any(normalized <= z[0])
                or np.any(normalized >= z[-1])
            ):
                raise ValueError(
                    "Retained quote outside the forward domain."
                )

            scale = (
                model.forward(time) * model.discount_factor(time)
            )
            prices = scale * np.interp(
                normalized,
                z,
                result.normalized_calls[i],
            )
            residual = (prices - data[:, 1]) / data[:, 2]

            rows.append(
                pd.DataFrame(
                    {
                        "case": name,
                        "expiry": path.parent.name,
                        "strike": data[:, 0],
                        "PDE_price": prices,
                        "mid": data[:, 1],
                        "half_width": data[:, 2],
                        "residual_half_spreads": residual,
                    }
                )
            )
            expiry_rows.append(
                {
                    "case": name,
                    "expiry": path.parent.name,
                    **fit_metrics(residual),
                }
            )

        frame = pd.concat(rows, ignore_index=True)
        quote_frames.append(frame)
        fit_rows.append(
            {
                "case": name,
                **fit_metrics(
                    frame.residual_half_spreads.to_numpy()
                ),
            }
        )

        for i in range(1, len(checkpoints)):
            label = {
                "case": name,
                "days": float(365 * checkpoints[i]),
            }
            full, scoped, locations = shape_audit(
                z,
                result.normalized_calls[i],
                result.normalized_calls[i - 1],
                windows,
            )
            shape_rows.append({**label, **full})
            window_shapes.extend(
                [{**label, **row} for row in scoped]
            )
            butterfly_locations.extend(
                [{**label, **row} for row in locations]
            )

        for row in contracts[
            contracts.case == name
        ].itertuples(index=False):
            time = model._time(row.days / 365)
            i = int(np.argmin(abs(checkpoints - time)))
            price = (
                model.forward(time)
                * model.discount_factor(time)
                * np.interp(
                    np.exp(row.log_strike),
                    z,
                    result.normalized_calls[i],
                )
            )
            consistency.append(
                {
                    "case": name,
                    "days": row.days,
                    "log_strike": row.log_strike,
                    "forward_minus_backward": float(
                        price - row.price
                    ),
                }
            )

    quote_output = pd.concat(
        quote_frames, ignore_index=True
    )
    baseline = quote_output[
        quote_output.case == reference_name
    ].PDE_price.to_numpy()

    for row, frame in zip(fit_rows, quote_frames):
        row["max_quote_change_vs_reference"] = float(
            abs(
                frame.PDE_price.to_numpy() - baseline
            ).max()
        )

    fit = pd.DataFrame(fit_rows)
    shapes = pd.DataFrame(shape_rows)

    for name, frame in (
        ("quotes", quote_output),
        ("diffusion_quote_fit", fit),
        ("expiry_quote_fit", pd.DataFrame(expiry_rows)),
        ("forward_shapes", shapes),
        (
            "forward_window_shapes",
            pd.DataFrame(window_shapes),
        ),
        (
            "forward_butterfly_locations",
            pd.DataFrame(
                butterfly_locations,
                columns=[
                    "case",
                    "days",
                    "log_moneyness",
                    "butterfly_slope_change",
                ],
            ),
        ),
        (
            "forward_backward",
            pd.DataFrame(consistency),
        ),
    ):
        frame.to_csv(
            output / f"{name}.csv", index=False
        )

    figure, axes = plt.subplots(
        1,
        len(days),
        figsize=(5 * len(days), 4),
        squeeze=False,
    )
    displayed = min(strikes_y, key=abs)

    for axis, day in zip(axes[0], days):
        for name in curves:
            if name in ("pde_coarse", "time_coarse"):
                continue
            frame = curves[name]
            frame = frame[
                (frame.days == day)
                & (frame.log_strike == displayed)
                & (
                    abs(frame.spot_log_change)
                    <= windows[0] + 1e-12
                )
            ]
            axis.plot(
                frame.spot,
                frame.gamma,
                label=name,
                linewidth=1,
            )

        axis.set(
            xlabel="Spot; fixed carry and coefficient per case",
            ylabel="Gamma per index point",
            title=(
                f"{day:g} days; log-strike "
                f"{displayed:+g}"
            ),
        )

    axes[0, -1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(
        output / "gamma_robustness.png", dpi=160
    )
    plt.close(figure)

    worst = comparisons[
        (comparisons.case == "domain_wide")
        & (
            comparisons.spot_log_half_width
            == windows[-1]
        )
    ].iloc[0]
    mask = (
        (reference.days == worst.price_max_change_days)
        & (
            reference.log_strike
            == worst.price_max_change_log_strike
        )
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4)
    )
    plotted_cases = [
        "domain_wide",
        *[
            c[0]
            for c in cases
            if c[-1] != 1.0
        ],
    ]

    for axis, field in zip(
        axes, ("price", "delta", "gamma")
    ):
        for name in plotted_cases:
            axis.plot(
                spot_y,
                (
                    curves[name].loc[mask, field]
                    - reference.loc[mask, field]
                ),
                label=name,
            )
        axis.set(
            xlabel="Log spot change",
            ylabel=(
                f"{field.capitalize()} change vs reference"
            ),
        )
        axis.axhline(
            0.0, color="black", linewidth=0.5
        )

    axes[-1].legend(fontsize=8)
    figure.suptitle(
        "Largest domain price-change contract: "
        f"{worst.price_max_change_days:g} days, "
        "forward log-strike "
        f"{worst.price_max_change_log_strike:+g}"
    )
    figure.tight_layout()
    figure.savefig(
        output / "conditional_wing_changes.png", dpi=160
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
            for p in [
                args.model,
                *[p for p, _ in quotes],
            ]
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
        "radius_in_log_state": args.radius,
        "cases": case_settings,
        "reference_case": reference_name,
        "spot_log_windows": windows,
        "reference_is_exact_greek_truth": False,
        "generated_forward_log_strikes": strikes_y,
        "first_period_change": (
            "Gaussian variance average; "
            "blend (1-t/T1)^2 before T1."
        ),
        "wing_stress": (
            "Volatility factors outside active-period "
            "observed quote range; cubic ramp."
        ),
        "wing_transition_log_width": args.wing_transition,
        "native_grid_scope": (
            "Same fitted proxy control nodes/values "
            "on rebuilt grids; no optimizer."
        ),
        "conditional_scope": (
            "Generated vanilla contracts on sampled "
            "spot windows at calendar zero."
        ),
        "boundaries": (
            "Martingale payoff asymptotes "
            "on each finite domain."
        ),
        "price_minus_original_AH_scope": (
            "Includes the diffusion change, "
            "not solely numerical error."
        ),
        "greek_convention": (
            "Physical coefficient and original carry "
            "fixed separately for each case."
        ),
        "zero_time_coefficient_requested": False,
        "prices_clipped": False,
        "shape_tolerances": {
            "normalized_price": 1e-10,
            "normalized_slope": 1e-8,
            "butterfly_slope_change": 1e-8,
            "conditional_delta": 1e-6,
            "conditional_price_points": 1e-6,
            "conditional_gamma": 1e-8,
        },
        "input_models_modified": False,
        "parameters_refitted": False,
        "candidate_promoted": False,
        "global_wing_robustness_certified": False,
        "status": "completed",
    }
    (output / "audit.json").write_text(
        json.dumps(
            audit, indent=2, allow_nan=False
        )
        + "\n"
    )

    print("\nFrozen proxy quote fit:")
    print(
        pd.DataFrame(grid_rows).to_string(index=False)
    )

    print(
        "\nOriginal-spot sensitivity against "
        "the finest native grid:"
    )
    print(
        pd.DataFrame(at_spot).to_string(index=False)
    )

    print("\nConditional sensitivity by spot window:")
    print(
        comparisons[
            [
                "case",
                "spot_log_half_width",
                "max_price_change",
                "max_delta_change",
                "max_gamma_change",
                "negative_gamma_samples",
                "delta_bound_violation_samples",
                "price_bound_violation_samples",
            ]
        ].to_string(index=False)
    )

    print("\nChanged-diffusion quote fit:")
    print(fit.to_string(index=False))

    print("\nMaximum forward/backward difference:")
    print(
        pd.DataFrame(consistency)
        .groupby("case")
        .forward_minus_backward
        .agg(lambda x: x.abs().max())
        .to_string()
    )

    print("\nForward sampled shape totals:")
    print(
        shapes.drop(columns="days")
        .groupby("case")
        .sum()
        .to_string()
    )

    print(
        "\nForward sampled shape totals "
        "inside reported strike windows:"
    )
    print(
        pd.DataFrame(window_shapes)
        .drop(columns="days")
        .groupby(["case", "log_strike_half_width"])
        .sum()
        .to_string()
    )

    print(f"\nDiagnostics: {output.resolve()}")
    print(
        "Rebuilt fixed-proxy grids: "
        f"{model_output.resolve()}"
    )
    print(
        "Finite-domain sensitivity study; "
        "no promotion or economic hedging validation."
    )


if __name__ == "__main__":
    main()