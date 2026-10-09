"""Surface price and Greek refinement checks."""

from surface import AHShortEndVariance
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pde import AHBackwardPricer
from pde import ForwardPDESolver


@dataclass(frozen=True)
class DailyValidationSettings:
    radius: float = 0.0005
    width: float = 0.75
    wide_width: float = 0.9
    coarse_intervals: int = 12000
    fine_intervals: int = 24000
    steps_per_day: int = 64

    def __post_init__(self):
        if (
            not np.isfinite([self.radius, self.width, self.wide_width]).all()
            or self.radius <= 0
            or (not 0.1 <= self.width < self.wide_width)
            or (not isinstance(self.steps_per_day, int))
            or (self.steps_per_day < 1)
            or any(
                (
                    not isinstance(n, int) or n < 100 or n % 2
                    for n in [self.coarse_intervals, self.fine_intervals]
                )
            )
            or (self.fine_intervals != 2 * self.coarse_intervals)
        ):
            raise ValueError(
                "Use positive settings and a two-to-one spatial refinement."
            )
        wide_count = self.fine_intervals * self.wide_width / self.width
        if abs(wide_count - round(wide_count)) > 1e-08 or round(wide_count) % 2:
            raise ValueError(
                "Wide domain must preserve spacing with an even interval count."
            )

    def cases(self):
        s = self
        return [
            ("space_coarse", s.width, s.coarse_intervals, s.steps_per_day, 0.0),
            ("space_fine", s.width, s.fine_intervals, s.steps_per_day, 0.0),
            ("time_fine", s.width, s.fine_intervals, 2 * s.steps_per_day, 0.0),
            (
                "half_cell_shift",
                s.width,
                s.fine_intervals,
                2 * s.steps_per_day,
                s.width / s.fine_intervals,
            ),
            (
                "domain_wide",
                s.wide_width,
                int(round(s.fine_intervals * s.wide_width / s.width)),
                2 * s.steps_per_day,
                0.0,
            ),
        ]


def quote_metrics(prices, quotes):
    error = np.asarray(prices) - quotes.call_mid.to_numpy()
    scaled = error / quotes.call_half_width.to_numpy()
    outside = (prices < quotes.call_bid.to_numpy() - 1e-06) | (
        prices > quotes.call_ask.to_numpy() + 1e-06
    )
    return {
        "quotes": len(quotes),
        "rms_half_spreads": float(np.sqrt(np.mean(scaled**2))),
        "max_half_spreads": float(abs(scaled).max()),
        "outside_original_bands": int(outside.sum()),
        "max_price_residual_points": float(abs(error).max()),
    }


class DailyAHValidator:

    def __init__(self, settings=None):
        self.settings = settings or DailyValidationSettings()

    def run(self, model, quotes, carry, progress=print):
        s = self.settings
        labels = {"quote_date": carry.quote_date.iloc[0], "root": carry.root.iloc[0]}
        selected = sorted(
            {
                0,
                int(np.argmin(abs(365 * model.maturities - 35))),
                len(model.maturities) - 1,
            }
        )
        spot_y = np.linspace(-0.01, 0.01, 401)
        spots = model.spot * np.exp(spot_y)
        strikes_y = np.array([-0.02, 0.0, 0.02])
        contracts, profiles, bumps, fits, expiry_fits = ([], [], [], [], [])
        shapes, consistency, conditioning, quote_outputs = ([], [], [], [])
        comparisons = []
        for radius in [0.0, s.radius]:
            variant = AHShortEndVariance(model, radius)
            if s.wide_width + 4 * radius >= variant.max_log_moneyness:
                raise ValueError(
                    "Leave room for the smoothing kernel inside the AH domain."
                )
            results_by_case = {}
            for case, width, intervals, steps, shift in s.cases():
                solver = AHBackwardPricer(
                    variant,
                    domain_width=width,
                    space_intervals=intervals,
                    steps_per_day=steps,
                    early_time_power=3.0,
                    state_shift=shift,
                )
                curves = []
                for i in selected:
                    time = float(model.maturities[i])
                    expiry = carry.expire_date.iloc[i]
                    progress(
                        f"{labels['quote_date']} radius={radius:g} {case}: {365 * time:.4g} days..."
                    )
                    strikes = model.forwards[i] * np.exp(strikes_y)
                    solved = solver.solve_many(strikes, time)
                    for y, strike, grid in zip(strikes_y, strikes, solved):
                        values = grid.greeks(spots)
                        at_spot = grid.greeks(model.spot)
                        bound = model.discounts[i] * model.forwards[i] / model.spot
                        low = np.maximum(bound * spots - model.discounts[i] * strike, 0)
                        frame = pd.DataFrame(
                            {
                                **labels,
                                "radius": radius,
                                "case": case,
                                "expire_date": expiry,
                                "days": 365 * time,
                                "log_strike": y,
                                "spot_log_change": spot_y,
                                "spot": spots,
                                "price": values.price,
                                "delta": values.delta,
                                "gamma": values.gamma,
                                "negative_gamma": values.gamma < -1e-08,
                                "delta_bound_violation": (values.delta < -1e-06)
                                | (values.delta > bound + 1e-06),
                                "price_bound_violation": (values.price < low - 1e-06)
                                | (values.price > bound * spots + 1e-06),
                            }
                        )
                        if (
                            not np.isfinite(frame[["price", "delta", "gamma"]])
                            .all()
                            .all()
                        ):
                            raise ArithmeticError(
                                "Nonfinite conditional prices or Greeks."
                            )
                        curves.append(frame)
                        contracts.append(
                            {
                                **labels,
                                "radius": radius,
                                "case": case,
                                "expire_date": expiry,
                                "days": 365 * time,
                                "log_strike": y,
                                "strike": strike,
                                "price": float(at_spot.price),
                                "delta": float(at_spot.delta),
                                "gamma": float(at_spot.gamma),
                                "AH_price": float(model.call_price(strike, time)),
                                **solver.last_diagnostics,
                            }
                        )
                        if case == "time_fine":
                            for fraction in [0.001, 0.0005, 0.00025, 0.000125]:
                                h = fraction * model.spot
                                lo, mid, hi = grid.price(
                                    [model.spot - h, model.spot, model.spot + h]
                                )
                                bumps.append(
                                    {
                                        **labels,
                                        "radius": radius,
                                        "expire_date": expiry,
                                        "days": 365 * time,
                                        "log_strike": y,
                                        "bump_points": h,
                                        "delta": float(at_spot.delta),
                                        "fd_delta": float((hi - lo) / (2 * h)),
                                        "delta_difference": float(
                                            (hi - lo) / (2 * h) - at_spot.delta
                                        ),
                                        "gamma": float(at_spot.gamma),
                                        "fd_gamma": float((hi - 2 * mid + lo) / h**2),
                                        "gamma_difference": float(
                                            (hi - 2 * mid + lo) / h**2 - at_spot.gamma
                                        ),
                                    }
                                )
                results_by_case[case] = pd.concat(curves, ignore_index=True)
                profiles.append(results_by_case[case])
            for check, name, reference in [
                ("space", "space_coarse", "space_fine"),
                ("time", "space_fine", "time_fine"),
                ("shift", "half_cell_shift", "time_fine"),
                ("domain", "domain_wide", "time_fine"),
            ]:
                other, ref = (results_by_case[name], results_by_case[reference])
                keys = ["expire_date", "log_strike", "spot_log_change"]
                pair = other.merge(
                    ref, on=keys, suffixes=("", "_ref"), validate="one_to_one"
                )
                row = {
                    **labels,
                    "radius": radius,
                    "check": check,
                    "case": name,
                    "reference_case": reference,
                }
                for field in ["price", "delta", "gamma"]:
                    change = abs(pair[field] - pair[field + "_ref"])
                    row["max_curve_" + field + "_change"] = float(change.max())
                    row["max_original_spot_" + field + "_change"] = float(
                        change.loc[pair.spot_log_change.eq(0)].max()
                    )
                for field in [
                    "negative_gamma",
                    "delta_bound_violation",
                    "price_bound_violation",
                ]:
                    row[field + "_samples"] = int(other[field].sum())
                comparisons.append(row)
            progress(
                f"{labels['quote_date']} radius={radius:g}: forward quote and consistency checks..."
            )
            solver = ForwardPDESolver(
                -s.width,
                s.width,
                n_space_intervals=s.fine_intervals,
                max_time_step=1 / (365 * 2 * s.steps_per_day),
                rannacher_steps=2,
            )
            z = solver.normalized_strikes
            times = np.unique(
                np.r_[
                    0.0,
                    model.maturities,
                    (np.r_[0.0, model.maturities[:-1]] + model.maturities) / 2,
                    [d / 365 for d in [0.25, 1.0] if d / 365 < model.maturities[0]],
                ]
            )
            first = float(model.maturities[0])
            count = max(8, int(np.ceil(365 * first * 2 * s.steps_per_day)))
            breaks = first * np.linspace(0, 1, count + 1) ** 3
            forward = solver.solve(
                times,
                np.maximum(1 - z, 0),
                variant.normalized_variance,
                lambda t: (1 - z[0], 0.0),
                time_breaks=breaks,
            )
            frames = []
            for i, expiry in enumerate(carry.expire_date):
                group = quotes.loc[quotes.expire_date.eq(expiry)].copy()
                at = int(np.searchsorted(times, model.maturities[i]))
                normalized = group.strike.to_numpy() / model.forwards[i]
                if np.any(normalized <= z[0]) or np.any(normalized >= z[-1]):
                    raise ValueError("A calibration strike exceeds the forward domain.")
                prices = (
                    model.discounts[i]
                    * model.forwards[i]
                    * np.interp(normalized, z, forward.normalized_calls[at])
                )
                group["radius"] = radius
                group["PDE_price"] = prices
                group["AH_price"] = model.call_price(
                    group.strike.to_numpy(), float(model.maturities[i])
                )
                group["PDE_minus_AH_points"] = prices - group.AH_price
                frames.append(group)
                expiry_fits.append(
                    {
                        **labels,
                        "radius": radius,
                        "expire_date": expiry,
                        **quote_metrics(prices, group),
                    }
                )
            frame = pd.concat(frames, ignore_index=True)
            quote_outputs.append(frame)
            fits.append(
                {
                    **labels,
                    "radius": radius,
                    **quote_metrics(frame.PDE_price.to_numpy(), frame),
                    "max_PDE_minus_AH_points": float(
                        abs(frame.PDE_minus_AH_points).max()
                    ),
                }
            )
            if radius == 0:
                fits.append(
                    {
                        **labels,
                        "radius": np.nan,
                        "source": "original_AH",
                        **quote_metrics(frame.AH_price.to_numpy(), frame),
                        "max_PDE_minus_AH_points": 0.0,
                    }
                )
            for row in contracts:
                if row["radius"] == radius and row["case"] == "time_fine":
                    at = int(np.argmin(abs(times - row["days"] / 365)))
                    time = float(times[at])
                    price = (
                        model.discount_factor(time)
                        * model.forward(time)
                        * np.interp(
                            row["strike"] / model.forward(time),
                            z,
                            forward.normalized_calls[at],
                        )
                    )
                    consistency.append(
                        {
                            **labels,
                            "radius": radius,
                            "expire_date": row["expire_date"],
                            "log_strike": row["log_strike"],
                            "forward_price": price,
                            "backward_price": row["price"],
                            "forward_minus_backward_points": price - row["price"],
                        }
                    )
            for i in range(1, len(times)):
                calls = forward.normalized_calls[i]
                previous = forward.normalized_calls[i - 1]
                slopes = np.diff(calls) / np.diff(z)
                changes = np.diff(slopes)
                for scope, half_width in [
                    ("full_domain", s.width),
                    ("report_window", 0.1),
                ]:
                    mask = abs(np.log(z)) <= half_width + 1e-12
                    edges = mask[:-1] & mask[1:]
                    centers = mask[:-2] & mask[1:-1] & mask[2:]
                    breached = centers & (changes < -1e-08)
                    locations = np.log(z[1:-1])[breached]
                    shapes.append(
                        {
                            **labels,
                            "radius": radius,
                            "days": 365 * times[i],
                            "scope": scope,
                            "increasing_prices": int((slopes[edges] > 1e-08).sum()),
                            "vertical_spread_violations": int(
                                (slopes[edges] < -1 - 1e-08).sum()
                            ),
                            "negative_butterflies": int(breached.sum()),
                            "minimum_slope_change": float(changes[centers].min()),
                            "breach_min_y": (
                                float(locations.min()) if len(locations) else np.nan
                            ),
                            "breach_max_y": (
                                float(locations.max()) if len(locations) else np.nan
                            ),
                            "intrinsic_shortfalls": int(
                                (calls[mask] < np.maximum(1 - z[mask], 0) - 1e-10).sum()
                            ),
                            "upper_bound_excesses": int(
                                (calls[mask] > 1 + 1e-10).sum()
                            ),
                            "calendar_violations": int(
                                (calls[mask] < previous[mask] - 1e-10).sum()
                            ),
                        }
                    )
            window = abs(variant.base.y) <= 0.1 + 1e-12
            samples = [
                (d / 365, "right") for d in [0.001, 0.25, 1.0] if d / 365 < first
            ]
            samples += [(first, "left")]
            if len(model.maturities) > 1:
                samples += [(first, "right")]
            for time, side in samples:
                variance = variant.normalized_variance(
                    variant.native_z[window], time, side
                )
                conditioning.append(
                    {
                        **labels,
                        "radius": radius,
                        "days": 365 * time,
                        "side": side,
                        "peak_local_vol_pct": float(100 * np.sqrt(variance.max())),
                        "peak_log_moneyness": float(
                            variant.base.y[window][np.argmax(variance)]
                        ),
                    }
                )
        quote_output = pd.concat(quote_outputs, ignore_index=True)
        pair = quote_output.loc[quote_output.radius.eq(s.radius)].merge(
            quote_output.loc[
                quote_output.radius.eq(0), ["expire_date", "strike", "PDE_price"]
            ],
            on=["expire_date", "strike"],
            suffixes=("", "_raw"),
            validate="one_to_one",
        )
        for row in fits:
            row.setdefault(
                "source", "raw_PDE" if row["radius"] == 0 else "smoothed_PDE"
            )
            if row["source"] == "smoothed_PDE":
                row["max_quote_change_vs_raw_PDE_points"] = float(
                    abs(pair.PDE_price - pair.PDE_price_raw).max()
                )
        return {
            "quote_fit": pd.DataFrame(fits),
            "expiry_quote_fit": pd.DataFrame(expiry_fits),
            "quote_prices": quote_output,
            "spot_greeks": pd.DataFrame(contracts),
            "sensitivity": pd.DataFrame(comparisons),
            "profiles": pd.concat(profiles, ignore_index=True),
            "spot_bumps": pd.DataFrame(bumps),
            "forward_backward": pd.DataFrame(consistency),
            "forward_shapes": pd.DataFrame(shapes),
            "conditioning": pd.DataFrame(conditioning),
        }
