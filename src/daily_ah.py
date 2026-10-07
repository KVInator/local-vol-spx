"""Independent daily AH research calibration from prepared carry inputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from andreasen_huge import AHGrid, AndreasenHugeCalibrator


@dataclass(frozen=True)
class DailyAHSettings:
    intervals: int = 8000
    width: float = 1.0
    smoothing: float = 1.0
    control_points: int = 31
    min_proxy_vol: float = 0.005
    max_proxy_vol: float = 3.0
    max_evaluations: int = 300

    def __post_init__(self):
        AHGrid(self.width, self.intervals)
        if (
            not isinstance(self.control_points, int)
            or self.control_points < 3
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations < 1
            or not np.isfinite(self.smoothing)
            or self.smoothing < 0
            or not 0 < self.min_proxy_vol < self.max_proxy_vol < np.inf
        ):
            raise ValueError("Invalid daily calibration settings.")


class DailyAHCalibrator:
    GROUP = ["quote_date", "root", "expire_date"]
    FLAGS = [
        "preferred_side_missing",
        "band_disjoint_from_call_bounds",
        "midpoint_outside_call_bounds",
    ]

    def __init__(self, settings=None):
        self.settings = settings or DailyAHSettings()
        s = self.settings
        self.calibrator = AndreasenHugeCalibrator(
            AHGrid(s.width, s.intervals),
            s.control_points,
            s.smoothing,
            s.min_proxy_vol,
            s.max_proxy_vol,
            s.max_evaluations,
        )

    @staticmethod
    def _boolean(series):
        converted = series.map({
            True: True,
            False: False,
            "True": True,
            "False": False,
        })
        if converted.isna().any():
            raise ValueError(f"Invalid boolean column: {series.name}.")
        return converted.astype(bool)

    def _checked(self, quotes, carry):
        q, c = quotes.copy(), carry.copy()
        q_required = self.GROUP + self.FLAGS + [
            "case",
            "strike",
            "underlying_last",
            "assumed_maturity_years",
            "forward",
            "discount_factor",
            "carry_ready",
            "source_kind",
            "source_bid",
            "source_ask",
            "call_bid",
            "call_ask",
            "call_mid",
            "call_half_width",
            "quote_timestamp_utc",
            "assumed_fixing_utc",
        ]
        c_required = self.GROUP + [
            "case",
            "spot",
            "maturity_years",
            "forward",
            "discount_factor",
            "annual_rate",
            "carry_ready",
            "parity_bands_incompatible",
            "fitted_parity_outside_bands",
            "quote_timestamp_utc",
            "assumed_fixing_utc",
        ]

        for frame, required in [(q, q_required), (c, c_required)]:
            missing = sorted(set(required) - set(frame.columns))
            if missing or frame.empty:
                raise ValueError(f"Empty input or missing columns: {missing}.")
            if frame[self.GROUP + ["case"]].isna().any().any():
                raise ValueError("Missing group identity or carry case.")
            if len(frame[["quote_date", "root"]].drop_duplicates()) != 1:
                raise ValueError("Calibrate exactly one date and root at a time.")
            frame["carry_ready"] = self._boolean(frame["carry_ready"])
            if not frame["carry_ready"].all():
                raise ValueError(
                    "Unavailable carry; no expiry is silently removed."
                )

        if (
            q.duplicated(self.GROUP + ["strike"]).any()
            or c.duplicated(self.GROUP).any()
        ):
            raise ValueError("Duplicate calibration strike or carry group.")
        if (
            set(map(tuple, q[self.GROUP].values))
            != set(map(tuple, c[self.GROUP].values))
        ):
            raise ValueError("Quote and carry expiry groups differ.")
        if q["case"].nunique() != 1 or set(q["case"]) != set(c["case"]):
            raise ValueError("Use one matching carry scenario.")

        for column in self.FLAGS:
            q[column] = self._boolean(q[column])
        for column in [
            "parity_bands_incompatible",
            "fitted_parity_outside_bands",
        ]:
            c[column] = self._boolean(c[column])

        q_numbers = [
            "strike",
            "underlying_last",
            "assumed_maturity_years",
            "forward",
            "discount_factor",
            "source_bid",
            "source_ask",
            "call_bid",
            "call_ask",
            "call_mid",
            "call_half_width",
        ]
        c_numbers = [
            "spot",
            "maturity_years",
            "forward",
            "discount_factor",
            "annual_rate",
        ]
        for frame, columns in [(q, q_numbers), (c, c_numbers)]:
            frame[columns] = frame[columns].apply(
                pd.to_numeric, errors="raise"
            )
            if not np.isfinite(frame[columns].to_numpy(float)).all():
                raise ValueError("Nonfinite numerical calibration inputs.")

        if (
            q["strike"].le(0).any()
            or q["source_bid"].le(0).any()
            or (q["source_ask"] <= q["source_bid"]).any()
            or q["call_half_width"].le(0).any()
            or not q["source_kind"].isin(["call", "put"]).all()
        ):
            raise ValueError("Invalid source quote or half-spread.")
        if (
            c[["spot", "maturity_years", "forward", "discount_factor"]]
            .le(0).any().any()
            or c["spot"].max() - c["spot"].min() > 0.01
            or c["annual_rate"].nunique() != 1
        ):
            raise ValueError("Inconsistent daily spot or assumed carry.")

        merged = q.merge(
            c,
            on=self.GROUP,
            suffixes=("", "_carry"),
            validate="many_to_one",
        )
        for left, right in [
            ("underlying_last", "spot"),
            ("assumed_maturity_years", "maturity_years"),
            ("forward", "forward_carry"),
            ("discount_factor", "discount_factor_carry"),
        ]:
            if not np.allclose(
                merged[left], merged[right], rtol=1e-12, atol=1e-10
            ):
                raise ValueError(f"Quote/carry mismatch: {left}.")

        for name in ["quote_timestamp_utc", "assumed_fixing_utc"]:
            if not pd.to_datetime(merged[name], utc=True).eq(
                pd.to_datetime(merged[name + "_carry"], utc=True)
            ).all():
                raise ValueError(f"Quote/carry mismatch: {name}.")

        for frame, maturity in [
            (q, "assumed_maturity_years"),
            (c, "maturity_years"),
        ]:
            timestamp = pd.to_datetime(
                frame["quote_timestamp_utc"], utc=True, errors="raise"
            )
            fixing = pd.to_datetime(
                frame["assumed_fixing_utc"], utc=True, errors="raise"
            )
            elapsed = (
                (fixing - timestamp).dt.total_seconds()
                / (365.0 * 86400.0)
            )
            if (
                timestamp.isna().any()
                or fixing.isna().any()
                or timestamp.nunique() != 1
                or not np.allclose(
                    frame[maturity], elapsed, rtol=1e-12, atol=1e-12
                )
                or not timestamp.dt.tz_convert("America/New_York")
                .dt.strftime("%Y-%m-%d").eq(frame["quote_date"]).all()
                or not fixing.dt.tz_convert("America/New_York")
                .dt.strftime("%Y-%m-%d").eq(frame["expire_date"]).all()
            ):
                raise ValueError(
                    "Snapshot dates or assumed UTC maturities disagree."
                )

        if not np.allclose(
            c["discount_factor"],
            np.exp(-c["annual_rate"] * c["maturity_years"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Discount factors disagree with the assumed rate."
            )

        offset = np.where(
            q["source_kind"].eq("put"),
            q["discount_factor"] * (q["forward"] - q["strike"]),
            0.0,
        )
        checks = [
            (q["call_bid"], q["source_bid"] + offset),
            (q["call_ask"], q["source_ask"] + offset),
            (q["call_mid"], (q["call_bid"] + q["call_ask"]) / 2),
            (
                q["call_half_width"],
                (q["source_ask"] - q["source_bid"]) / 2,
            ),
        ]
        if any(
            not np.allclose(a, b, rtol=1e-12, atol=1e-9)
            for a, b in checks
        ):
            raise ValueError(
                "Prepared prices differ from the original parity conversion."
            )

        lower = q["discount_factor"] * np.maximum(
            q["forward"] - q["strike"], 0
        )
        upper = q["discount_factor"] * q["forward"]
        expected = {
            "band_disjoint_from_call_bounds": (
                (q["call_ask"] < lower - 1e-6)
                | (q["call_bid"] > upper + 1e-6)
            ),
            "midpoint_outside_call_bounds": (
                (q["call_mid"] < lower - 1e-6)
                | (q["call_mid"] > upper + 1e-6)
            ),
        }
        if any(
            not q[name].eq(value).all()
            for name, value in expected.items()
        ):
            raise ValueError(
                "Prepared call-bound flags disagree with the original prices."
            )

        return (
            q.sort_values(self.GROUP + ["strike"]),
            c.sort_values("maturity_years"),
        )

    def calibrate(self, quotes, carry, progress=None):
        q, c = self._checked(quotes, carry)
        grouped = [
            q.loc[q["expire_date"].eq(expiry)].copy()
            for expiry in c["expire_date"]
        ]
        if any(len(group) < 3 for group in grouped):
            raise ValueError(
                "Every expiry needs at least three unique strikes."
            )

        model, optimizer = self.calibrator.calibrate(
            float(c["spot"].iloc[0]),
            c["maturity_years"].to_numpy(),
            c["forward"].to_numpy(),
            c["discount_factor"].to_numpy(),
            [
                group[["strike", "call_mid", "call_half_width"]].to_numpy()
                for group in grouped
            ],
            progress=progress,
        )

        residuals, expiry_rows, conditioning = [], [], []
        y = np.log(model.grid.z[1:-1])
        window = np.abs(y) <= 0.10 + 1e-12

        for index, (group, (_, row), report) in enumerate(
            zip(grouped, c.iterrows(), optimizer)
        ):
            t = float(row["maturity_years"])
            group["model_call_price"] = model.call_price(
                group["strike"].to_numpy(), t
            )
            group["price_residual_points"] = (
                group["model_call_price"] - group["call_mid"]
            )
            group["residual_half_spreads"] = (
                group["price_residual_points"] / group["call_half_width"]
            )
            group["outside_original_band"] = (
                (group["model_call_price"] < group["call_bid"] - 1e-6)
                | (group["model_call_price"] > group["call_ask"] + 1e-6)
            )
            residuals.append(group)
            expiry_rows.append({
                **row.to_dict(),
                **report,
                **self.fit_metrics(group),
                **{
                    name: int(group[name].sum())
                    for name in self.FLAGS
                },
            })

            sides = (
                ["left", "right"]
                if index < len(c) - 1 else ["left"]
            )
            for side in sides:
                state = model.node_state(t, side)
                v = state["local_variance"][window]
                valid = np.isfinite(v) & (v >= 0)
                conditioning.append({
                    **{name: row[name] for name in self.GROUP},
                    "days": t * 365,
                    "side": side,
                    "half_width": 0.10,
                    "native_nodes": int(window.sum()),
                    "unresolved_variance_nodes": int((~valid).sum()),
                    "peak_local_vol_pct": (
                        float(100 * np.sqrt(v[valid].max()))
                        if valid.any() else np.nan
                    ),
                    "minimum_z2_curvature": float(
                        (
                            model.grid.z[1:-1][window] ** 2
                            * state["curvature"][window]
                        ).min()
                    ),
                })

        times = np.sort(np.r_[
            model.maturities,
            (
                np.r_[0, model.maturities[:-1]]
                + model.maturities
            ) / 2,
        ])
        previous = np.maximum(1 - model.grid.z, 0)
        shapes = []

        for time in times:
            calls = model.node_state(float(time))["calls"]
            if not np.isfinite(calls).all():
                raise FloatingPointError(
                    "Nonfinite model prices in the shape study."
                )

            slopes = np.diff(calls) / np.diff(model.grid.z)
            shapes.append({
                "quote_date": c["quote_date"].iloc[0],
                "root": c["root"].iloc[0],
                "days": float(time * 365),
                "increasing_prices": int((slopes > 1e-8).sum()),
                "vertical_spread_violations": int(
                    (slopes < -1 - 1e-8).sum()
                ),
                "negative_butterflies": int(
                    (np.diff(slopes) < -1e-8).sum()
                ),
                "intrinsic_shortfalls": int(
                    (
                        calls
                        < np.maximum(1 - model.grid.z, 0) - 1e-10
                    ).sum()
                ),
                "upper_bound_excesses": int(
                    (calls > 1 + 1e-10).sum()
                ),
                "calendar_violations": int(
                    (calls < previous - 1e-10).sum()
                ),
            })
            previous = calls

        return model, {
            "quote_residuals": pd.concat(residuals, ignore_index=True),
            "expiry_summary": pd.DataFrame(expiry_rows),
            "conditioning": pd.DataFrame(conditioning),
            "shape_checks": pd.DataFrame(shapes),
        }

    @staticmethod
    def fit_metrics(frame):
        r = frame["residual_half_spreads"].to_numpy(float)
        return {
            "calibration_quotes": len(frame),
            "rms_half_spreads": float(np.sqrt(np.mean(r * r))),
            "max_half_spreads": float(np.abs(r).max()),
            "outside_original_bands": int(
                frame["outside_original_band"].sum()
            ),
            "max_price_residual_points": float(
                frame["price_residual_points"].abs().max()
            ),
        }