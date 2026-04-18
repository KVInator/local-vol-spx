from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lv_project.config import (
    DATA_PROCESSED_DIR,
    OUTPUT_DIAGNOSTICS_DIR,
    OUTPUT_INTERACTIVE_DIR,
    default_target_maturities_years,
)
from lv_project.plotting import make_3d_surface, make_smile_slice_figure, save_interactive_html
from lv_project.surface import build_daily_surface, diagnostics_to_frame, surface_to_long_frame


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real implied-vol surface for one quote date.")
    parser.add_argument("--quote-date", type=str, required=True)
    parser.add_argument("--clean-path", type=str, default=str(DATA_PROCESSED_DIR / "spx_calls_clean.csv"))
    parser.add_argument("--y-min", type=float, default=-0.15)
    parser.add_argument("--y-max", type=float, default=0.10)
    parser.add_argument("--n-y", type=int, default=50)
    parser.add_argument("--flat-rate", type=float, default=0.0)
    parser.add_argument("--flat-dividend-yield", type=float, default=0.0)
    parser.add_argument("--input-y-pad-left", type=float, default=0.10)
    parser.add_argument("--input-y-pad-right", type=float, default=0.10)
    parser.add_argument("--min-points-per-expiry", type=int, default=8)
    args = parser.parse_args()

    clean_path = Path(args.clean_path)
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {clean_path}. Run scripts/01_prepare_quotes.py first."
        )

    OUTPUT_INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    surfaces_dir = DATA_PROCESSED_DIR / "surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    clean_calls = pd.read_csv(clean_path, parse_dates=["quote_date", "expiry"])

    y_grid = np.linspace(args.y_min, args.y_max, args.n_y)
    
    target_maturities = default_target_maturities_years()

    result = build_daily_surface(
        clean_calls=clean_calls,
        quote_date=args.quote_date,
        y_grid=y_grid,
        target_maturities=target_maturities,
        flat_rate=args.flat_rate,
        flat_dividend_yield=args.flat_dividend_yield,
        input_y_min=args.y_min - args.input_y_pad_left,
        input_y_max=args.y_max + args.input_y_pad_right,
        min_points_per_expiry=args.min_points_per_expiry,
    )

    date_str = pd.Timestamp(args.quote_date).strftime("%Y-%m-%d")

    surface_long = surface_to_long_frame(result)
    diagnostics_df = diagnostics_to_frame(result)

    surface_csv = surfaces_dir / f"surface_{date_str}.csv"
    coverage_csv = surfaces_dir / f"pillar_coverage_{date_str}.csv"
    diagnostics_csv = OUTPUT_DIAGNOSTICS_DIR / f"surface_diagnostics_{date_str}.csv"
    diagnostics_json = OUTPUT_DIAGNOSTICS_DIR / f"surface_diagnostics_{date_str}.json"

    surface_long.to_csv(surface_csv, index=False)
    result.expiry_coverage.to_csv(coverage_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    with diagnostics_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(result.diagnostics), f, indent=2)

    iv_fig = make_3d_surface(
        x=result.y_grid,
        y=result.target_maturities,
        z=result.target_implied_vol,
        title=f"SPX Implied Volatility Surface | {date_str}",
        x_label="Log-Moneyness",
        y_label="Time to Maturity",
        z_label="Implied Volatility",
    )
    iv_html = OUTPUT_INTERACTIVE_DIR / f"implied_vol_surface_{date_str}.html"
    save_interactive_html(iv_fig, iv_html)

    w_fig = make_3d_surface(
        x=result.y_grid,
        y=result.target_maturities,
        z=result.target_total_variance,
        title=f"SPX Total Variance Surface | {date_str}",
        x_label="Log-Moneyness",
        y_label="Time to Maturity",
        z_label="Total Variance",
    )
    w_html = OUTPUT_INTERACTIVE_DIR / f"total_variance_surface_{date_str}.html"
    save_interactive_html(w_fig, w_html)

    preferred_days = [14, 30, 60, 90, 180, 360]
    curves = {}
    for days in preferred_days:
        t = days / 365.0
        idx = int(abs(result.target_maturities - t).argmin())
        curves[f"T={result.target_maturities[idx]:.3f}"] = result.target_implied_vol[idx, :]

    slices_fig = make_smile_slice_figure(
        x=result.y_grid,
        curves=curves,
        title=f"SPX Smile Slices | {date_str}",
        x_label="Log-Moneyness",
        y_label="Implied Volatility",
    )
    slices_html = OUTPUT_INTERACTIVE_DIR / f"smile_slices_{date_str}.html"
    save_interactive_html(slices_fig, slices_html)

    print("Built daily implied-vol surface")
    print(f"quote date                : {date_str}")
    print(f"spot                      : {result.spot:.4f}")
    print(f"surface points            : {len(result.surface_points):,}")
    print(f"kept expiries             : {len(result.pillar_maturities):,}")
    print(f"calendar violations       : {result.diagnostics['calendar_violations_before']} -> {result.diagnostics['calendar_violations_after']}")
    print(f"surface IV range          : {result.diagnostics['surface_iv_min']:.4f} -> {result.diagnostics['surface_iv_max']:.4f}")
    print(f"surface csv saved to      : {surface_csv}")
    print(f"coverage csv saved to     : {coverage_csv}")
    print(f"diagnostics json saved to : {diagnostics_json}")
    print(f"interactive IV html       : {iv_html}")
    print(f"interactive TV html       : {w_html}")
    print(f"interactive slices html   : {slices_html}")


if __name__ == "__main__":
    main()