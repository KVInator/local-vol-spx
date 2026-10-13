"""Black, PDE, smile and empirical hedge estimators."""

from market_data import boolean, utc
from selection import HedgeSettings
from pricing import BlackFixedIVBenchmark
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from pde import AHBackwardPricer
from surface import AHShortEndVariance
from pricing import black_time_value, invert_time_value


class DailyDeltas:

    def __init__(self, settings=None):
        self.settings = settings or HedgeSettings()
        self.benchmark = BlackFixedIVBenchmark()

    def calculate(self, entries, carry, model=None, progress=print):
        if entries.empty:
            return entries.copy()
        keys = ["quote_date", "root", "expire_date"]
        required = keys + [
            "case",
            "carry_ready",
            "spot",
            "maturity_years",
            "forward",
            "discount_factor",
            "annual_rate",
            "quote_timestamp_utc",
            "assumed_fixing_utc",
        ]
        if not set(required).issubset(carry) or carry.duplicated(keys).any():
            raise ValueError("Missing or duplicate daily carry")
        if len(entries[["quote_date", "root"]].drop_duplicates()) != 1:
            raise ValueError("Calculate one entry date and root at a time")
        date, root = entries[keys[:2]].iloc[0]
        c = carry.loc[carry.quote_date.eq(date) & carry.root.eq(root)].copy()
        c["carry_ready"] = boolean(c.carry_ready)
        for name in ["quote_timestamp_utc", "assumed_fixing_utc"]:
            c[name] = utc(c[name])
        numbers = [
            "spot",
            "maturity_years",
            "forward",
            "discount_factor",
            "annual_rate",
        ]
        c[numbers] = c[numbers].apply(pd.to_numeric, errors="raise")
        if len(c) and (c["case"].nunique() != 1 or c.annual_rate.nunique() != 1):
            raise ValueError("Use one current-date carry scenario")
        if model is not None:
            ordered = c.sort_values("maturity_years")
            comparisons = [
                ("maturity_years", model.maturities),
                ("forward", model.forwards),
                ("discount_factor", model.discounts),
            ]
            if (
                not c.carry_ready.all()
                or len(c) != len(model.maturities)
                or (not np.allclose(c.spot, model.spot, rtol=0, atol=1e-08))
                or any(
                    (
                        not np.allclose(ordered[name], values, rtol=1e-12, atol=1e-10)
                        for name, values in comparisons
                    )
                )
            ):
                raise ValueError("Current-day model and carry disagree")
            variance = AHShortEndVariance(model, self.settings.radius)
            solver = AHBackwardPricer(
                variance,
                domain_width=self.settings.width,
                space_intervals=self.settings.intervals,
                steps_per_day=self.settings.steps_per_day,
                early_time_power=3.0,
            )
        rows = []
        benchmark = self.benchmark
        for expiry, group in entries.groupby("expire_date", sort=True):
            match = c.loc[c.expire_date.eq(expiry)]
            if match.empty or not match.carry_ready.iloc[0]:
                rows.extend(
                    (
                        {
                            **row,
                            "black_status": "carry_unavailable",
                            "ah_status": "carry_unavailable",
                        }
                        for row in group.to_dict("records")
                    )
                )
                continue
            item = match.iloc[0]
            if (
                not np.isfinite(item[numbers].to_numpy(float)).all()
                or min(
                    item.spot, item.maturity_years, item.forward, item.discount_factor
                )
                <= 0
                or (
                    not np.isclose(
                        item.discount_factor,
                        np.exp(-item.annual_rate * item.maturity_years),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )
            ):
                raise ValueError("Invalid assumed carry")
            for row in group.itertuples(index=False):
                if (
                    row.quote_timestamp_utc != item.quote_timestamp_utc
                    or row.assumed_fixing_utc != item.assumed_fixing_utc
                    or (
                        not np.isclose(
                            row.underlying_last, item.spot, rtol=0, atol=1e-08
                        )
                    )
                    or (
                        not np.isclose(
                            row.assumed_maturity_years,
                            item.maturity_years,
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    )
                ):
                    raise ValueError("Entry quote and current carry disagree")
            solved, failure = ({}, "")
            if model is not None:
                strikes = np.sort(group.strike.unique())
                progress(
                    f"{date}: {expiry}, {len(strikes)} unique strikes, {365 * item.maturity_years:.4g} days..."
                )
                try:
                    grids = solver.solve_many(strikes, float(item.maturity_years))
                    solved = {
                        float(k): grid.greeks(model.spot)
                        for k, grid in zip(strikes, grids)
                    }
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    np.linalg.LinAlgError,
                ) as error:
                    failure = f"{type(error).__name__}: {error}"
            for entry in group.to_dict("records"):
                out = {
                    **entry,
                    "case": item["case"],
                    "annual_rate": float(item.annual_rate),
                    "forward": float(item.forward),
                    "discount_factor": float(item.discount_factor),
                    "forward_log_moneyness": float(
                        np.log(entry["strike"] / item.forward)
                    ),
                    "ah_status": "model_unavailable",
                    "ah_message": failure,
                }
                out.update(
                    benchmark.estimate(
                        item.spot,
                        entry["strike"],
                        item.maturity_years,
                        item.forward,
                        item.discount_factor,
                        entry["mid"],
                        entry["kind"],
                    )
                )
                if model is not None:
                    out["ah_status"] = "solve_failed"
                    if solved:
                        value = solved[float(entry["strike"])]
                        put = entry["kind"] == "put"
                        ratio = item.discount_factor * item.forward / item.spot
                        price = float(value.price) - put * item.discount_factor * (
                            item.forward - entry["strike"]
                        )
                        delta = float(value.delta) - put * ratio
                        gamma = float(value.gamma)
                        if np.isfinite([price, delta, gamma]).all():
                            sign = 1 if not put else -1
                            lower = item.discount_factor * max(
                                (item.forward - entry["strike"]) * sign, 0
                            )
                            upper = item.discount_factor * (
                                item.forward if not put else entry["strike"]
                            )
                            flag = (
                                gamma < -1e-08
                                or not -put * ratio - 1e-06
                                <= delta
                                <= (1 - put) * ratio + 1e-06
                            )
                            flag |= not lower - 1e-06 <= price <= upper + 1e-06
                            out.update(
                                ah_status="shape_flag" if flag else "ready",
                                ah_price=price,
                                ah_delta=delta,
                                ah_gamma=gamma,
                                ah_price_minus_mid=price - entry["mid"],
                                **solver.last_diagnostics,
                            )
                rows.append(out)
        return pd.DataFrame(rows)


def black_greeks(spot, strike, maturity, forward, discount, sigma, kind):
    """Fixed IV spot delta; vega is per 1.0 decimal annual volatility."""
    values = np.asarray([spot, strike, maturity, forward, discount, sigma], float)
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or kind not in ("call", "put")
    ):
        raise ValueError("Require positive finite inputs and call/put.")
    std = sigma * np.sqrt(maturity)
    d1 = np.log(forward / strike) / std + std / 2
    phi = np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi)
    ratio = discount * forward / spot
    price = (
        discount
        * forward
        * float(black_time_value(np.log(strike / forward), std * std))
    )
    price += discount * max((forward - strike) * (1 if kind == "call" else -1), 0)
    return dict(
        price=price,
        delta=ratio * (ndtr(d1) - (kind == "put")),
        vega=discount * forward * phi * np.sqrt(maturity),
        forward_call_delta=ndtr(d1),
        d1=d1,
    )


def smile_conventions(spot, strike, maturity, forward, discount, sigma, sigma_y, kind):
    """Chain rule for a frozen forward-delta smile, and the HW LV approximation.

    F/S and D held fixed; y=log(K/F). Sticky delta uses unadjusted normalized
    forward call delta N(d1), not spot/premium-adjusted FX delta. Its local
    representation requires decreasing, invertible delta versus strike.
    """
    if not np.isfinite(sigma_y):
        raise ValueError("Nonfinite smile slope.")
    b = black_greeks(spot, strike, maturity, forward, discount, sigma, kind)
    y = np.log(strike / forward)
    phi = np.exp(-b["d1"] ** 2 / 2) / np.sqrt(2 * np.pi)
    q_y = (
        phi
        / (sigma * np.sqrt(maturity))
        * (-1 + (y / sigma + sigma * maturity / 2) * sigma_y)
    )
    return dict(
        surface_iv=sigma,
        surface_sigma_y=sigma_y,
        surface_sigma_k=sigma_y / strike,
        surface_vega=b["vega"],
        surface_sticky_strike_delta=b["delta"],
        surface_sticky_delta_delta=b["delta"] - b["vega"] * sigma_y / spot,
        surface_delta_coordinate_slope=q_y,
        surface_hw_lv_delta=b["delta"] + b["vega"] * sigma_y / strike,
    )


@dataclass(frozen=True)
class SmileSettings:
    fine_cells: int = 4
    delta_tolerance: float = 0.0005
    slope_absolute_tolerance: float = 0.01
    slope_relative_tolerance: float = 0.05

    def __post_init__(self):
        if (
            not isinstance(self.fine_cells, int)
            or self.fine_cells < 2
            or (
                not np.isfinite(
                    [
                        self.delta_tolerance,
                        self.slope_absolute_tolerance,
                        self.slope_relative_tolerance,
                    ]
                ).all()
            )
            or (
                min(
                    self.delta_tolerance,
                    self.slope_absolute_tolerance,
                    self.slope_relative_tolerance,
                )
                <= 0
            )
        ):
            raise ValueError("Invalid smile stencil settings.")


class AHSmileHedges:
    """Original AH prices only; finite-width derivatives and honest support flags."""

    def __init__(self, settings=None):
        self.settings = settings or SmileSettings()

    def calculate(self, entries, model, calibration_quotes):
        rows = []
        s = self.settings
        native_h = float(np.diff(np.log(model.grid.z)).mean())
        h = s.fine_cells * native_h
        for maturity, group in entries.groupby("assumed_maturity_years", sort=True):
            maturity = float(maturity)
            f, d = (model.forward(maturity), model.discount_factor(maturity))
            state = model.node_state(maturity)
            for row in group.to_dict("records"):
                out = dict(
                    entry_id=row["entry_id"],
                    smile_status="unresolved",
                    smile_message="",
                    smile_fine_log_width=h,
                    smile_coarse_log_width=2 * h,
                    sticky_delta_status="unresolved",
                    lv_smile_status="unresolved",
                )
                try:
                    if (
                        not np.isclose(
                            row["underlying_last"], model.spot, rtol=0, atol=1e-08
                        )
                        or not np.isclose(row["forward"], f, rtol=1e-12, atol=1e-08)
                        or (
                            not np.isclose(
                                row["discount_factor"], d, rtol=1e-12, atol=1e-12
                            )
                        )
                    ):
                        raise ValueError("Saved entry and original AH carry disagree.")
                    spot, strike = (model.spot, float(row["strike"]))
                    y = np.log(strike / f)
                    quotes = calibration_quotes.loc[
                        calibration_quotes.expire_date.eq(row["expire_date"])
                    ]
                    observed = np.log(quotes.strike.to_numpy(float) / f)
                    points = y + h * np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
                    if (
                        not len(observed)
                        or points[0] < observed.min()
                        or points[-1] > observed.max()
                        or (points[0] <= np.log(model.grid.z[0]))
                        or (points[-1] >= np.log(model.grid.z[-1]))
                    ):
                        out["smile_status"] = "outside_observed_stencil_support"
                        rows.append(out)
                        continue
                    tv = np.interp(np.exp(points), model.grid.z, state["time_value"])
                    w, status = invert_time_value(points, tv)
                    if not (status == "ready").all():
                        out["smile_status"] = "iv_stencil_unresolved"
                        rows.append(out)
                        continue
                    sigmas = np.sqrt(w / maturity)
                    fine = (sigmas[5] - sigmas[1]) / (2 * h)
                    coarse = (sigmas[6] - sigmas[0]) / (4 * h)
                    g = smile_conventions(
                        spot, strike, maturity, f, d, sigmas[3], fine, row["kind"]
                    )
                    gc = smile_conventions(
                        spot, strike, maturity, f, d, sigmas[3], coarse, row["kind"]
                    )

                    def bump_delta(width):
                        spots = spot * np.exp(np.array([-width, width]))
                        yy = y - np.log(spots / spot)
                        calls = np.interp(np.exp(yy), model.grid.z, state["calls"])
                        prices = d * f * spots / spot * calls
                        if row["kind"] == "put":
                            prices -= d * (f * spots / spot - strike)
                        return float(np.diff(prices)[0] / np.diff(spots)[0])

                    bumped = bump_delta(h / 2)
                    bump_change = abs(bumped - bump_delta(h))
                    chain_error = abs(bumped - g["surface_sticky_delta_delta"])
                    width_delta = abs(
                        g["surface_sticky_delta_delta"]
                        - gc["surface_sticky_delta_delta"]
                    )
                    density = float(
                        np.interp(strike / f, model.grid.z[1:-1], state["curvature"])
                    )
                    out.update(
                        g,
                        smile_sigma_y_coarse=coarse,
                        smile_slope_width_change=abs(fine - coarse),
                        smile_delta_width_change=width_delta,
                        smile_bump_delta=bumped,
                        smile_bump_width_change=bump_change,
                        smile_chain_minus_bump=chain_error,
                        smile_original_density_z=density,
                    )
                    slope_ok = abs(
                        fine - coarse
                    ) <= s.slope_absolute_tolerance + s.slope_relative_tolerance * abs(
                        fine
                    )
                    if density < -1e-10:
                        out["smile_status"] = "negative_original_density"
                    elif (
                        not slope_ok
                        or max(width_delta, chain_error, bump_change)
                        > s.delta_tolerance
                    ):
                        out["smile_status"] = "stencil_disagreement"
                    else:
                        out["smile_status"] = "ready"
                        out["sticky_delta_status"] = (
                            "ready"
                            if g["surface_delta_coordinate_slope"] < -1e-08
                            else "delta_coordinate_not_invertible"
                        )
                        if row["black_status"] == "ready":
                            b = black_greeks(
                                spot,
                                strike,
                                maturity,
                                f,
                                d,
                                float(row["black_iv"]),
                                row["kind"],
                            )
                            out["lv_smile_delta"] = (
                                float(row["black_delta"]) + b["vega"] * fine / strike
                            )
                            out["lv_smile_status"] = "ready"
                except (ValueError, ArithmeticError) as error:
                    out.update(
                        smile_status="evaluation_failed",
                        smile_message=f"{type(error).__name__}: {error}",
                    )
                rows.append(out)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class EmpiricalMVSettings:
    window_dates: int = 60
    minimum_dates: int = 10
    minimum_rows: int = 60
    condition_limit: float = 100000000.0

    def __post_init__(self):
        if (
            any(
                (
                    not isinstance(x, int)
                    for x in (self.window_dates, self.minimum_dates, self.minimum_rows)
                )
            )
            or not 3 <= self.minimum_dates <= self.window_dates
            or self.minimum_rows < 6
            or (not np.isfinite(self.condition_limit))
            or (self.condition_limit <= 1)
        ):
            raise ValueError("Invalid fixed MV window or rank controls.")


class PastOnlyEmpiricalMV:
    """Hull--White quadratic correction, separate root/kind fits, equal row OLS.

    Minimise SSE of raw observed price changes, without an intercept. This is the
    conventional empirical MV specification, not an exact finite-horizon
    conditional-variance guarantee. Funding/cost P&L is evaluated separately.
    Only endpoints STRICTLY before the prediction timestamp are available.
    """

    def __init__(self, settings=None):
        self.settings = settings or EmpiricalMVSettings()

    @staticmethod
    def feature(row):
        b = black_greeks(
            float(row["underlying_last"]),
            float(row["strike"]),
            float(row["assumed_maturity_years"]),
            float(row["forward"]),
            float(row["discount_factor"]),
            float(row["black_iv"]),
            row["kind"],
        )
        if not np.isclose(
            float(row["black_delta"]), b["delta"], rtol=1e-10, atol=1e-08
        ):
            raise ValueError("Saved observed-IV Black delta and carry disagree.")
        delta = float(row["black_delta"])
        scale = b["vega"] / (
            row["underlying_last"] * np.sqrt(row["assumed_maturity_years"])
        )
        return scale * np.array([1.0, delta, delta * delta])

    def predict(self, entries, training):
        s = self.settings
        if training.entry_id.duplicated().any() or entries.entry_id.duplicated().any():
            raise ValueError("Duplicate training or prediction entry IDs.")
        data = training.copy(deep=True)
        for column in ("quote_timestamp_utc", "end_timestamp"):
            for value in data[column].dropna():
                if pd.Timestamp(value).tzinfo is None:
                    raise ValueError("Training timestamps must be timezone-aware.")
        data["_entry"] = pd.to_datetime(data.quote_timestamp_utc, utc=True)
        data["_end"] = pd.to_datetime(data.end_timestamp, utc=True)
        predictions, fits, membership = ([], [], [])
        for (stamp, root, kind), group in entries.groupby(
            ["quote_timestamp_utc", "root", "kind"], sort=True
        ):
            now = pd.Timestamp(stamp)
            if now.tzinfo is None:
                raise ValueError("Prediction timestamp must be timezone-aware.")
            available = data.loc[
                data.root.eq(root)
                & data.kind.eq(kind)
                & data.end_status.eq("matched")
                & data.black_status.eq("ready")
                & data._entry.lt(now)
                & data._end.lt(now)
            ].copy()
            if len(available) and (not (available._entry < available._end).all()):
                raise ValueError("Invalid training endpoint order.")
            dates = sorted(available.quote_date.unique())[-s.window_dates :]
            available = available.loc[available.quote_date.isin(dates)].sort_values(
                ["quote_date", "entry_id"]
            )
            fit_id = hashlib.sha256(
                f"{now.isoformat()}|{root}|{kind}".encode()
            ).hexdigest()[:20]
            fit = dict(
                fit_id=fit_id,
                prediction_timestamp=now.isoformat(),
                quote_date=group.quote_date.iloc[0],
                root=root,
                kind=kind,
                status="warmup",
                available_dates=len(dates),
                training_rows=len(available),
                training_first_date=dates[0] if dates else "",
                training_last_date=dates[-1] if dates else "",
                maximum_training_endpoint=(
                    available._end.max().isoformat() if len(available) else ""
                ),
                message="",
            )
            beta = None
            if len(dates) >= s.minimum_dates and len(available) >= s.minimum_rows:
                features = np.vstack(
                    [self.feature(row) for row in available.to_dict("records")]
                )
                ds = (available.end_spot - available.underlying_last).to_numpy(float)
                target = (
                    available.end_mid - available.mid - available.black_delta * ds
                ).to_numpy(float)
                design = features * ds[:, None]
                if not np.isfinite(np.r_[design.ravel(), target]).all():
                    raise ValueError("Invalid available training observations.")
                norms = np.linalg.norm(design, axis=0)
                if (norms <= 0).any():
                    fit["status"] = "rank_deficient"
                else:
                    scaled = design / norms
                    singular = np.linalg.svd(scaled, compute_uv=False)
                    condition = (
                        float(singular[0] / singular[-1])
                        if singular[-1] > 0
                        else np.inf
                    )
                    rank = int(np.linalg.matrix_rank(scaled))
                    fit.update(design_rank=rank, scaled_condition_number=condition)
                    if rank < 3:
                        fit["status"] = "rank_deficient"
                    elif condition > s.condition_limit:
                        fit["status"] = "ill_conditioned"
                    else:
                        beta = np.linalg.lstsq(scaled, target, rcond=None)[0] / norms
                        error = target - design @ beta
                        fit.update(
                            status="ready",
                            a=float(beta[0]),
                            b=float(beta[1]),
                            c=float(beta[2]),
                            training_black_sse=float(target @ target),
                            training_mv_sse=float(error @ error),
                            training_mv_mean_error=float(error.mean()),
                            training_mv_variance=float(error.var()),
                            training_delta_min=float(available.black_delta.min()),
                            training_delta_max=float(available.black_delta.max()),
                        )
            fits.append(fit)
            for row in available.to_dict("records"):
                membership.append(
                    dict(
                        fit_id=fit_id,
                        training_entry_id=row["entry_id"],
                        training_date=row["quote_date"],
                        training_endpoint=row["_end"].isoformat(),
                    )
                )
            for row in group.to_dict("records"):
                out = dict(
                    entry_id=row["entry_id"],
                    empirical_mv_status=fit["status"],
                    empirical_mv_fit_id=fit_id,
                )
                if row["black_status"] != "ready":
                    out["empirical_mv_status"] = "black_not_ready"
                elif beta is not None:
                    correction = float(self.feature(row) @ beta)
                    out.update(
                        empirical_mv_correction=correction,
                        empirical_mv_delta=float(row["black_delta"]) + correction,
                        empirical_mv_delta_extrapolated=not fit["training_delta_min"]
                        <= row["black_delta"]
                        <= fit["training_delta_max"],
                    )
                predictions.append(out)
        return (
            pd.DataFrame(predictions),
            pd.DataFrame(fits),
            pd.DataFrame(
                membership,
                columns=[
                    "fit_id",
                    "training_entry_id",
                    "training_date",
                    "training_endpoint",
                ],
            ),
        )


STRATEGIES = {
    "black": ("black_status", "black_delta"),
    "ah_pde": ("ah_status", "ah_delta"),
    "surface_sticky_strike": ("smile_status", "surface_sticky_strike_delta"),
    "surface_sticky_delta": ("sticky_delta_status", "surface_sticky_delta_delta"),
    "lv_smile": ("lv_smile_status", "lv_smile_delta"),
    "empirical_mv": ("empirical_mv_status", "empirical_mv_delta"),
}


def analytical_controls():
    """Known smiles with both signs; these controls are not historical data."""
    rows = []
    for slope in (0.0, -0.3, 0.2):
        for kind in ("call", "put"):
            spot, strike, t, f, d, sigma = (100.0, 101.0, 0.2, 102.0, 0.99, 0.25)
            y = np.log(strike / f)
            values = smile_conventions(spot, strike, t, f, d, sigma, slope, kind)
            errors = []
            for width in (0.001, 0.0005, 0.0001):
                spots = spot * np.exp(np.array([-width, width]))
                prices = [
                    black_greeks(
                        x,
                        strike,
                        t,
                        f * x / spot,
                        d,
                        sigma + slope * (np.log(strike / (f * x / spot)) - y),
                        kind,
                    )["price"]
                    for x in spots
                ]
                errors.append(
                    abs(
                        (prices[1] - prices[0]) / (spots[1] - spots[0])
                        - values["surface_sticky_delta_delta"]
                    )
                )
            rows.append(
                dict(
                    control="analytic_linear_iv",
                    kind=kind,
                    sigma_y=slope,
                    sticky_delta_delta=values["surface_sticky_delta_delta"],
                    sticky_strike_delta=values["surface_sticky_strike_delta"],
                    hw_lv_smile_delta=values["surface_hw_lv_delta"],
                    bump_error_1e_3=errors[0],
                    bump_error_5e_4=errors[1],
                    bump_error_1e_4=errors[2],
                )
            )
    result = pd.DataFrame(rows)
    if result.bump_error_1e_4.max() > 1e-06:
        raise ArithmeticError("Analytical smile control failed.")
    return result
