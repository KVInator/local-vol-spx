from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class SurfaceBuildResult:
    quote_date: pd.Timestamp
    spot: float
    y_grid: np.ndarray
    target_maturities: np.ndarray
    pillar_expiries: np.ndarray
    pillar_maturities: np.ndarray
    pillar_total_variance: np.ndarray
    repaired_pillar_total_variance: np.ndarray
    target_total_variance: np.ndarray
    target_implied_vol: np.ndarray
    surface_points: pd.DataFrame
    expiry_coverage: pd.DataFrame
    diagnostics: dict[str, Any]


def prepare_surface_points_for_date(
    clean_calls: pd.DataFrame,
    quote_date: str | pd.Timestamp,
    flat_rate: float = 0.0,
    flat_dividend_yield: float = 0.0,
    input_y_min: float = -0.25,
    input_y_max: float = 0.20,
) -> tuple[pd.DataFrame, float]:
    quote_date = pd.Timestamp(quote_date)

    out = clean_calls.loc[clean_calls["quote_date"].eq(quote_date)].copy()
    if out.empty:
        raise ValueError(f"No clean call quotes found for quote_date={quote_date.date()}")

    spot = float(out["underlying_spot"].median())
    out["rate"] = float(flat_rate)
    out["dividend_yield"] = float(flat_dividend_yield)
    out["forward"] = spot * np.exp((flat_rate - flat_dividend_yield) * out["time_to_expiry"])
    out["log_moneyness"] = np.log(out["strike"] / out["forward"])
    out["total_variance"] = out["time_to_expiry"] * (out["implied_vol"] ** 2)

    out = out.loc[out["log_moneyness"].between(input_y_min, input_y_max)].copy()
    out = out.sort_values(["expiry", "strike"]).reset_index(drop=True)

    if out.empty:
        raise ValueError(
            f"No quotes remained after moneyness filtering for quote_date={quote_date.date()}"
        )

    return out, spot


def build_pillar_total_variance_surface(
    surface_points: pd.DataFrame,
    y_grid: np.ndarray,
    min_points_per_expiry: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    y_grid = np.asarray(y_grid, dtype=float)

    pillar_expiries: list[pd.Timestamp] = []
    pillar_maturities: list[float] = []
    pillar_values: list[np.ndarray] = []
    coverage_rows: list[dict[str, float]] = []
    edge_fill_count = 0

    for expiry, grp in surface_points.groupby("expiry", sort=True):
        g = grp[
            ["strike", "log_moneyness", "total_variance", "implied_vol", "time_to_expiry"]
        ].dropna()

        g = g.sort_values(["strike", "log_moneyness"])
        g = (
            g.groupby("strike", as_index=False)
            .agg(
                log_moneyness=("log_moneyness", "mean"),
                total_variance=("total_variance", "mean"),
                implied_vol=("implied_vol", "mean"),
                time_to_expiry=("time_to_expiry", "first"),
            )
            .sort_values("log_moneyness")
            .drop_duplicates("log_moneyness")
        )

        if len(g) < min_points_per_expiry:
            continue

        y_native = g["log_moneyness"].to_numpy(dtype=float)
        w_native = g["total_variance"].to_numpy(dtype=float)

        interp = interp1d(
            y_native,
            w_native,
            kind="linear",
            bounds_error=False,
            fill_value=(float(w_native[0]), float(w_native[-1])),
        )
        w_eval = interp(y_grid)

        edge_fill_count += int((y_grid < y_native.min()).sum() + (y_grid > y_native.max()).sum())

        pillar_expiries.append(pd.Timestamp(expiry))
        pillar_maturities.append(float(g["time_to_expiry"].iloc[0]))
        pillar_values.append(w_eval)
        coverage_rows.append(
            {
                "expiry": pd.Timestamp(expiry),
                "time_to_expiry": float(g["time_to_expiry"].iloc[0]),
                "y_min_native": float(y_native.min()),
                "y_max_native": float(y_native.max()),
                "n_points": int(len(g)),
            }
        )

    if len(pillar_maturities) < 2:
        raise ValueError("Fewer than two expiries survived filtering. Cannot build a surface.")

    pillar_expiries_arr = np.array(pillar_expiries, dtype="datetime64[ns]")
    pillar_maturities_arr = np.asarray(pillar_maturities, dtype=float)
    pillar_values_arr = np.vstack(pillar_values)

    order = np.argsort(pillar_maturities_arr)
    pillar_expiries_arr = pillar_expiries_arr[order]
    pillar_maturities_arr = pillar_maturities_arr[order]
    pillar_values_arr = pillar_values_arr[order, :]

    coverage_df = pd.DataFrame(coverage_rows).sort_values("time_to_expiry").reset_index(drop=True)

    return (
        pillar_expiries_arr,
        pillar_maturities_arr,
        pillar_values_arr,
        coverage_df,
        edge_fill_count,
    )


def count_calendar_violations(pillar_total_variance: np.ndarray) -> int:
    diffs = np.diff(pillar_total_variance, axis=0)
    return int(np.sum(diffs < -1e-12))


def repair_calendar_monotonicity(pillar_total_variance: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(pillar_total_variance, axis=0)


def interpolate_target_surface(
    pillar_maturities: np.ndarray,
    repaired_pillar_total_variance: np.ndarray,
    target_maturities: np.ndarray,
) -> np.ndarray:
    pillar_maturities = np.asarray(pillar_maturities, dtype=float)
    target_maturities = np.asarray(target_maturities, dtype=float)

    out = np.empty((len(target_maturities), repaired_pillar_total_variance.shape[1]), dtype=float)

    for j in range(repaired_pillar_total_variance.shape[1]):
        out[:, j] = np.interp(
            target_maturities,
            pillar_maturities,
            repaired_pillar_total_variance[:, j],
            left=np.nan,
            right=np.nan,
        )

    return out


def total_variance_to_implied_vol(target_total_variance: np.ndarray, target_maturities: np.ndarray) -> np.ndarray:
    target_maturities = np.asarray(target_maturities, dtype=float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = np.sqrt(np.maximum(target_total_variance, 0.0) / target_maturities)
    return iv


def build_daily_surface(
    clean_calls: pd.DataFrame,
    quote_date: str | pd.Timestamp,
    y_grid: np.ndarray,
    target_maturities: np.ndarray,
    flat_rate: float = 0.0,
    flat_dividend_yield: float = 0.0,
    input_y_min: float = -0.25,
    input_y_max: float = 0.20,
    min_points_per_expiry: int = 8,
) -> SurfaceBuildResult:
    surface_points, spot = prepare_surface_points_for_date(
        clean_calls=clean_calls,
        quote_date=quote_date,
        flat_rate=flat_rate,
        flat_dividend_yield=flat_dividend_yield,
        input_y_min=input_y_min,
        input_y_max=input_y_max,
    )

    (
        pillar_expiries,
        pillar_maturities,
        pillar_total_variance,
        expiry_coverage,
        edge_fill_count,
    ) = build_pillar_total_variance_surface(
        surface_points=surface_points,
        y_grid=y_grid,
        min_points_per_expiry=min_points_per_expiry,
    )

    calendar_violations_before = count_calendar_violations(pillar_total_variance)
    repaired_pillar_total_variance = repair_calendar_monotonicity(pillar_total_variance)
    calendar_violations_after = count_calendar_violations(repaired_pillar_total_variance)

    target_total_variance = interpolate_target_surface(
        pillar_maturities=pillar_maturities,
        repaired_pillar_total_variance=repaired_pillar_total_variance,
        target_maturities=target_maturities,
    )
    target_implied_vol = total_variance_to_implied_vol(
        target_total_variance=target_total_variance,
        target_maturities=target_maturities,
    )

    diagnostics = {
        "quote_date": str(pd.Timestamp(quote_date).date()),
        "spot": float(spot),
        "n_surface_points": int(len(surface_points)),
        "n_input_expiries": int(surface_points["expiry"].nunique()),
        "n_kept_expiries": int(len(pillar_maturities)),
        "y_grid_min": float(np.min(y_grid)),
        "y_grid_max": float(np.max(y_grid)),
        "input_y_min": float(input_y_min),
        "input_y_max": float(input_y_max),
        "min_target_maturity": float(np.min(target_maturities)),
        "max_target_maturity": float(np.max(target_maturities)),
        "calendar_violations_before": int(calendar_violations_before),
        "calendar_violations_after": int(calendar_violations_after),
        "edge_fill_count": int(edge_fill_count),
        "surface_iv_min": float(np.nanmin(target_implied_vol)),
        "surface_iv_max": float(np.nanmax(target_implied_vol)),
    }

    return SurfaceBuildResult(
        quote_date=pd.Timestamp(quote_date),
        spot=spot,
        y_grid=np.asarray(y_grid, dtype=float),
        target_maturities=np.asarray(target_maturities, dtype=float),
        pillar_expiries=pillar_expiries,
        pillar_maturities=pillar_maturities,
        pillar_total_variance=pillar_total_variance,
        repaired_pillar_total_variance=repaired_pillar_total_variance,
        target_total_variance=target_total_variance,
        target_implied_vol=target_implied_vol,
        surface_points=surface_points,
        expiry_coverage=expiry_coverage,
        diagnostics=diagnostics,
    )


def surface_to_long_frame(result: SurfaceBuildResult) -> pd.DataFrame:
    tt, yy = np.meshgrid(result.target_maturities, result.y_grid, indexing="ij")
    out = pd.DataFrame(
        {
            "quote_date": result.quote_date,
            "time_to_expiry": tt.ravel(),
            "log_moneyness": yy.ravel(),
            "total_variance": result.target_total_variance.ravel(),
            "implied_vol": result.target_implied_vol.ravel(),
        }
    )
    return out


def diagnostics_to_frame(result: SurfaceBuildResult) -> pd.DataFrame:
    return pd.DataFrame([result.diagnostics])