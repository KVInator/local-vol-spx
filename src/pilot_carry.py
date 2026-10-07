"""Option-implied carry diagnostics. Estimates are not an approved funding curve."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linprog


@dataclass(frozen=True)
class CarrySettings:
    windows: tuple = (0.01, 0.03, 0.05)
    primary_window: float = 0.03
    minimum_pairs: int = 6
    rate_scenarios: tuple = (0.00, 0.03, 0.05, 0.07)

    def __post_init__(self):
        if (
            not self.windows
            or len(set(self.windows)) != len(self.windows)
            or any(
                not np.isfinite(w) or not 0 < w <= 0.10
                for w in self.windows
            )
            or self.primary_window not in self.windows
        ):
            raise ValueError(
                "Require distinct windows in (0, 0.10], "
                "including the primary window."
            )
        if (
            self.minimum_pairs < 3
            or not np.isfinite(self.rate_scenarios).all()
        ):
            raise ValueError(
                "Require at least three pairs and finite rate scenarios."
            )


class PilotCarryEstimator:
    GROUP = ["quote_date", "root", "expire_date"]
    KEY = GROUP + ["strike"]

    def __init__(self, settings=None):
        self.settings = settings or CarrySettings()

    def _checked(self, quotes):
        required = self.KEY + [
            "kind", "bid", "ask", "underlying_last",
            "quote_timestamp_utc", "assumed_fixing_utc",
            "assumed_maturity_years", "quote_status",
        ]
        if not set(required).issubset(quotes):
            raise ValueError(
                f"Missing panel columns: {sorted(set(required) - set(quotes))}"
            )
        q = quotes[required].copy()
        if q.empty:
            raise ValueError("The pilot quote panel is empty.")

        for name in ["quote_date", "expire_date"]:
            q[name] = pd.to_datetime(q[name], errors="raise")
            if (
                q[name].dt.tz is not None
                or not q[name].eq(q[name].dt.normalize()).all()
            ):
                raise ValueError(
                    "Panel dates must be timezone-free calendar dates."
                )

        for name in ["quote_timestamp_utc", "assumed_fixing_utc"]:
            q[name] = pd.to_datetime(q[name], utc=True, errors="raise")

        numeric = [
            "strike", "bid", "ask", "underlying_last",
            "assumed_maturity_years",
        ]
        for name in numeric:
            q[name] = pd.to_numeric(q[name], errors="raise").astype(float)

        if (
            q[required].isna().any().any()
            or not np.isfinite(q[numeric]).all().all()
        ):
            raise ValueError("Missing or nonfinite quote metadata.")

        positive = [
            "strike", "bid", "underlying_last", "assumed_maturity_years",
        ]
        if (
            not q["kind"].isin(["call", "put"]).all()
            or not q["quote_status"].eq("research_calibration_quote").all()
            or not q[positive].gt(0).all().all()
            or not q["ask"].gt(q["bid"]).all()
        ):
            raise ValueError(
                "Use positive-bid, positive-spread pilot calibration quotes only."
            )

        if q.duplicated(self.KEY + ["kind"]).any():
            raise ValueError(
                "Duplicate contract sides; no row is chosen or averaged."
            )

        for stamp, date in [
            ("quote_timestamp_utc", "quote_date"),
            ("assumed_fixing_utc", "expire_date"),
        ]:
            local_date = (
                q[stamp].dt.tz_convert("America/New_York")
                .dt.tz_localize(None).dt.normalize()
            )
            if not local_date.eq(q[date]).all():
                raise ValueError(
                    "Panel dates disagree with the assumed New York timestamps."
                )

        elapsed = (
            q["assumed_fixing_utc"] - q["quote_timestamp_utc"]
        ).dt.total_seconds() / (365 * 86400)
        if not np.allclose(
            elapsed, q["assumed_maturity_years"], rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Stored maturity disagrees with the assumed UTC timestamps."
            )
        return q

    @staticmethod
    def _robust(matrix, target, half):
        jac = matrix / half[:, None]
        initial = np.linalg.lstsq(jac, target / half, rcond=None)[0]
        return least_squares(
            lambda p: (matrix @ p - target) / half,
            initial,
            jac=lambda p: jac.copy(),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=300,
        )

    @staticmethod
    def _residuals(predicted, mid, half):
        residual = (predicted - mid) / half
        return {
            "rms_parity_half_widths": float(
                np.sqrt(np.mean(residual ** 2))
            ),
            "max_parity_half_widths": float(np.max(np.abs(residual))),
            "outside_parity_bands": int(
                (np.abs(predicted - mid) > half + 1e-6).sum()
            ),
        }

    @staticmethod
    def _discount_bounds(matrix, mid, half, maturity):
        # Variables: centred intercept, discount, half-band multiplier.
        a = np.vstack([
            np.column_stack([matrix, -half]),
            np.column_stack([-matrix, -half]),
            [-1.0, -1.0, 0.0],  # Nonnegative prepaid forward.
        ])
        b = np.r_[mid, -mid, 0.0]
        options = {
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        }
        relaxation = linprog(
            [0, 0, 1],
            A_ub=a,
            b_ub=b,
            bounds=[(None, None), (0, None), (0, None)],
            method="highs",
            options=options,
        )
        if not relaxation.success:
            return {"band_bounds_status": "relaxation_failed"}

        minimum = max(0.0, float(relaxation.fun))
        feasible = bool(minimum <= 1.0 + 1e-8)
        result = {
            "minimum_band_multiplier": minimum,
            "original_bands_feasible": feasible,
            "discount_bounds_band_multiplier": 1.0,
            "band_bounds_status": (
                "original_bands_infeasible"
                if not feasible else "bounds_failed"
            ),
        }
        if not feasible:
            return result

        a2 = np.vstack([matrix, -matrix, [-1.0, -1.0]])
        b2 = np.r_[mid + half, -mid + half, 0.0]
        limits = [
            linprog(
                [0, sign],
                A_ub=a2,
                b_ub=b2,
                bounds=[(None, None), (0, None)],
                method="highs",
                options=options,
            )
            for sign in (1, -1)
        ]
        result["band_bounds_status"] = (
            "computed" if all(r.success for r in limits)
            else "bounds_failed"
        )
        if all(r.success for r in limits):
            lower, upper = [float(r.x[1]) for r in limits]
            rate_low = (
                -np.log(upper) / maturity * 100
                if upper > 0 else np.nan
            )
            rate_high = (
                -np.log(lower) / maturity * 100
                if lower > 0 else np.inf
            )
            result.update(
                discount_lower=lower,
                discount_upper=upper,
                rate_band_width_pp=float(rate_high - rate_low),
            )
        return result

    def run(self, quotes):
        q = self._checked(quotes)
        metadata = [
            "underlying_last", "quote_timestamp_utc",
            "assumed_fixing_utc", "assumed_maturity_years",
        ]
        calls = q.loc[
            q["kind"].eq("call"),
            self.KEY + metadata + ["bid", "ask"],
        ].rename(columns={"bid": "c_bid", "ask": "c_ask"})
        puts = q.loc[
            q["kind"].eq("put"), self.KEY + ["bid", "ask"],
        ].rename(columns={"bid": "p_bid", "ask": "p_ask"})

        pairs = calls.merge(puts, on=self.KEY, validate="one_to_one")
        pairs["parity_lower"] = pairs["c_bid"] - pairs["p_ask"]
        pairs["parity_upper"] = pairs["c_ask"] - pairs["p_bid"]
        pairs["parity_mid"] = (
            pairs["parity_lower"] + pairs["parity_upper"]
        ) / 2
        pairs["parity_half_width"] = (
            pairs["parity_upper"] - pairs["parity_lower"]
        ) / 2

        paired = {key: g for key, g in pairs.groupby(self.GROUP)}
        fits, scenarios, details = [], [], []

        for key, g in q.groupby(self.GROUP, sort=True):
            if (
                g["underlying_last"].max()
                - g["underlying_last"].min() > 0.01
                or any(g[c].nunique() != 1 for c in metadata[1:])
            ):
                raise ValueError(
                    f"Inconsistent snapshot metadata for {key}."
                )

            spot = float(g["underlying_last"].median())
            maturity = float(g["assumed_maturity_years"].iloc[0])
            all_pairs = paired.get(key, pairs.iloc[:0])
            base = dict(zip(self.GROUP, key))
            base.update(
                spot=spot,
                maturity_years=maturity,
                days=float(
                    (g["expire_date"].iloc[0] - g["quote_date"].iloc[0]).days
                ),
                available_pair_count=len(all_pairs),
                unpaired_quote_sides=len(g) - 2 * len(all_pairs),
            )

            for window in self.settings.windows:
                p = all_pairs.loc[
                    np.abs(np.log(all_pairs["strike"] / spot)) <= window
                ]
                row = dict(
                    base, window=window, pairs=len(p),
                    status="insufficient_pairs",
                )
                if len(p) < self.settings.minimum_pairs:
                    fits.append(row)
                    continue

                x = p["strike"].to_numpy() / spot - 1
                if p["strike"].nunique() < 3 or np.ptp(x) <= 1e-8:
                    fits.append(dict(row, status="degenerate_strikes"))
                    continue

                mid = p["parity_mid"].to_numpy() / spot
                half = p["parity_half_width"].to_numpy() / spot
                matrix = np.column_stack([np.ones(len(p)), -x])
                fit = self._robust(matrix, mid, half)
                intercept, discount = fit.x
                forward = (
                    spot * (1 + intercept / discount)
                    if discount > 0 else np.nan
                )
                if not fit.success:
                    status = "optimizer_failed"
                elif not np.isfinite(forward) or forward <= 0:
                    status = "invalid_discount_or_forward"
                else:
                    status = "fitted"

                predicted = matrix @ fit.x * spot
                row.update(
                    status=status,
                    optimizer_evaluations=int(fit.nfev),
                    weighted_condition=float(
                        np.linalg.cond(matrix / half[:, None])
                    ),
                    discount_factor=float(discount),
                    forward=float(forward),
                )
                row.update(
                    self._residuals(predicted, mid * spot, half * spot)
                )
                if status == "fitted":
                    row.update(
                        implied_zero_rate_pct=float(
                            -np.log(discount) / maturity * 100
                        ),
                        implied_net_carry_pct=float(
                            np.log(forward / spot) / maturity * 100
                        ),
                    )
                    row.update(
                        self._discount_bounds(matrix, mid, half, maturity)
                    )
                fits.append(row)

                if window == self.settings.primary_window:
                    fields = self.KEY + [
                        "parity_lower", "parity_upper",
                        "parity_mid", "parity_half_width",
                    ]
                    d = p[fields].copy()
                    d["fitted_parity"] = predicted
                    d["residual_half_widths"] = (
                        predicted / spot - mid
                    ) / half
                    details.append(d)

                    for rate in self.settings.rate_scenarios:
                        fixed_discount = np.exp(-rate * maturity)
                        fixed = self._robust(
                            np.ones((len(p), 1)),
                            mid + fixed_discount * x,
                            half,
                        )
                        fixed_forward = spot * (
                            1 + fixed.x[0] / fixed_discount
                        )
                        candidate = (
                            fixed.x[0] - fixed_discount * x
                        ) * spot
                        forward_low = float((
                            p["strike"]
                            + p["parity_lower"] / fixed_discount
                        ).max())
                        forward_high = float((
                            p["strike"]
                            + p["parity_upper"] / fixed_discount
                        ).min())
                        entry = dict(
                            base,
                            annual_rate_pct=rate * 100,
                            discount_factor=float(fixed_discount),
                            forward=float(fixed_forward),
                            status=(
                                "conditional_fit"
                                if fixed.success and fixed_forward > 0
                                else "failed"
                            ),
                            forward_change_vs_free_fit=float(
                                fixed_forward - forward
                            ),
                            forward_band_lower=forward_low,
                            forward_band_upper=forward_high,
                            original_bands_feasible=bool(
                                forward_low <= forward_high
                            ),
                        )
                        entry.update(self._residuals(
                            candidate, mid * spot, half * spot
                        ))
                        scenarios.append(entry)

        fitted = pd.DataFrame(fits)
        for column in [
            "forward", "implied_zero_rate_pct", "discount_factor",
            "rms_parity_half_widths", "minimum_band_multiplier",
            "rate_band_width_pp",
        ]:
            if column not in fitted:
                fitted[column] = np.nan

        sensitivity = fitted.loc[
            fitted["status"].eq("fitted")
        ].groupby(self.GROUP).agg(
            fitted_windows=("window", "size"),
            forward_min=("forward", "min"),
            forward_max=("forward", "max"),
            rate_min_pct=("implied_zero_rate_pct", "min"),
            rate_max_pct=("implied_zero_rate_pct", "max"),
        ).reset_index()
        sensitivity["forward_window_range_points"] = (
            sensitivity["forward_max"] - sensitivity["forward_min"]
        )
        sensitivity["rate_window_range_pp"] = (
            sensitivity["rate_max_pct"] - sensitivity["rate_min_pct"]
        )
        sensitivity.loc[
            sensitivity["fitted_windows"].lt(2),
            ["forward_window_range_points", "rate_window_range_pp"],
        ] = np.nan

        audit = {
            "settings": asdict(self.settings),
            "input_quote_sides": len(q),
            "matched_pairs": len(pairs),
            "unpaired_quote_sides": len(q) - 2 * len(pairs),
            "fit": (
                "C-P=D(F-K); soft-L1 residuals scaled by parity half-widths."
            ),
            "band_bounds": (
                "Discount extrema inside original parity bands only; "
                "D>=0 closure and nonnegative prepaid forward. "
                "Infeasible groups receive no discount range."
            ),
            "minimum_band_multiplier": (
                "Diagnostic of consistency only; quotes and bounds "
                "are never expanded for estimation."
            ),
            "band_bounds_are_confidence_intervals": False,
            "fixed_rates_are_market_observations": False,
            "rate_scenarios": (
                "Continuously compounded, constant annual rates; "
                "sensitivity only."
            ),
            "same_date_pairs_only": True,
            "carry_interpolated": False,
            "contract_identity_verified": False,
            "snapshot_provenance_verified": False,
            "daily_discount_curve_verified": False,
            "candidate_promoted": False,
            "quotes_modified": False,
            "diffusion_models_refitted": False,
            "backtest_performed": False,
        }

        detail_columns = self.KEY + [
            "parity_lower", "parity_upper",
            "parity_mid", "parity_half_width",
        ]
        empty_details = pairs[detail_columns].iloc[:0].assign(
            fitted_parity=np.nan, residual_half_widths=np.nan
        )
        scenario_columns = self.GROUP + [
            "spot", "maturity_years", "days", "available_pair_count",
            "unpaired_quote_sides", "annual_rate_pct", "discount_factor",
            "forward", "status", "forward_change_vs_free_fit",
            "forward_band_lower", "forward_band_upper",
            "original_bands_feasible", "rms_parity_half_widths",
            "max_parity_half_widths", "outside_parity_bands",
        ]
        return {
            "carry_estimates": fitted,
            "window_sensitivity": sensitivity,
            "fixed_rate_sensitivity": pd.DataFrame(scenarios).reindex(
                columns=scenario_columns
            ),
            "matched_pairs": pairs,
            "primary_pair_residuals": (
                pd.concat(details, ignore_index=True)
                if details else empty_details
            ),
        }, audit