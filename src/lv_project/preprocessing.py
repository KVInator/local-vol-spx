from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = {
    "quote_date",
    "underlying_last",
    "expire_date",
    "dte",
    "strike",
    "c_bid",
    "c_ask",
    "c_iv",
}


RAW_NUMERIC_COLUMNS = [
    "quote_unixtime",
    "quote_time_hours",
    "underlying_last",
    "expire_unix",
    "dte",
    "c_delta",
    "c_gamma",
    "c_vega",
    "c_theta",
    "c_rho",
    "c_iv",
    "c_volume",
    "c_last",
    "c_bid",
    "c_ask",
    "strike",
    "p_bid",
    "p_ask",
    "p_last",
    "p_delta",
    "p_gamma",
    "p_vega",
    "p_theta",
    "p_rho",
    "p_iv",
    "p_volume",
    "strike_distance",
    "strike_distance_pct",
]


def _normalise_raw_column_name(name: str) -> str:
    name = re.sub(r"[\[\]]", "", name)
    return name.strip().lower()


def parse_raw_option_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    df = pd.read_csv(
        path,
        sep=",",
        engine="python",
        skipinitialspace=True,
        na_values=["", " "],
    )
    df.columns = [_normalise_raw_column_name(c) for c in df.columns]

    missing = RAW_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    for col in ["quote_readtime", "quote_date", "expire_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in RAW_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_file"] = path.name
    return df


def load_raw_option_directory(raw_dir: str | Path, pattern: str = "spx_eod_*.txt") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {raw_dir}")

    frames = [parse_raw_option_file(path) for path in files]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["quote_date", "expire_date", "strike"]).reset_index(drop=True)
    return out


def build_clean_call_dataset(
    raw_df: pd.DataFrame,
    flat_rate: float = 0.0,
    flat_dividend_yield: float = 0.0,
    min_dte: int = 7,
    max_dte: int = 365,
    min_bid: float = 0.05,
    max_spread_pct_mid: float = 0.35,
    min_iv: float = 0.01,
    max_iv: float = 2.0,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "quote_date": raw_df["quote_date"],
            "expiry": raw_df["expire_date"],
            "underlying_spot": raw_df["underlying_last"],
            "days_to_expiry": raw_df["dte"],
            "time_to_expiry": raw_df["dte"] / 365.0,
            "strike": raw_df["strike"],
            "bid": raw_df["c_bid"],
            "ask": raw_df["c_ask"],
            "implied_vol": raw_df["c_iv"],
            "delta": raw_df["c_delta"] if "c_delta" in raw_df.columns else np.nan,
            "gamma": raw_df["c_gamma"] if "c_gamma" in raw_df.columns else np.nan,
            "vega": raw_df["c_vega"] if "c_vega" in raw_df.columns else np.nan,
            "theta": raw_df["c_theta"] if "c_theta" in raw_df.columns else np.nan,
            "rho": raw_df["c_rho"] if "c_rho" in raw_df.columns else np.nan,
            "volume": raw_df["c_volume"] if "c_volume" in raw_df.columns else np.nan,
            "source_file": raw_df["source_file"] if "source_file" in raw_df.columns else "",
        }
    )

    out["option_type"] = "C"
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["spread"] = out["ask"] - out["bid"]
    out["spread_pct_mid"] = np.where(out["mid"] > 0.0, out["spread"] / out["mid"], np.nan)
    out["rate"] = float(flat_rate)
    out["dividend_yield"] = float(flat_dividend_yield)

    mask = (
        out["quote_date"].notna()
        & out["expiry"].notna()
        & out["underlying_spot"].gt(0.0)
        & out["strike"].gt(0.0)
        & out["days_to_expiry"].ge(min_dte)
        & out["days_to_expiry"].le(max_dte)
        & out["time_to_expiry"].gt(0.0)
        & out["bid"].ge(min_bid)
        & out["ask"].ge(out["bid"])
        & out["mid"].gt(0.0)
        & out["implied_vol"].gt(min_iv)
        & out["implied_vol"].lt(max_iv)
        & out["spread_pct_mid"].le(max_spread_pct_mid)
    )

    out = out.loc[mask].copy()
    out = out.sort_values(["quote_date", "expiry", "strike"]).reset_index(drop=True)
    return out


def make_daily_summary(clean_calls: pd.DataFrame) -> pd.DataFrame:
    summary = (
        clean_calls.groupby("quote_date", as_index=False)
        .agg(
            n_quotes=("strike", "size"),
            n_expiries=("expiry", "nunique"),
            spot=("underlying_spot", "median"),
            min_dte=("days_to_expiry", "min"),
            max_dte=("days_to_expiry", "max"),
            median_iv=("implied_vol", "median"),
            median_spread_pct_mid=("spread_pct_mid", "median"),
        )
        .sort_values("quote_date")
        .reset_index(drop=True)
    )
    return summary