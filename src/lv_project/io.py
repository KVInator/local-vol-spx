from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_BASE_COLUMNS = {
    "quote_date",
    "expiry",
    "option_type",
    "strike",
    "underlying_spot",
}

OPTIONAL_PRICE_COLUMNS = {"mid", "price", "bid", "ask"}
OPTIONAL_VOL_COLUMNS = {"implied_vol"}


def load_option_quotes_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not (OPTIONAL_PRICE_COLUMNS & set(df.columns) or OPTIONAL_VOL_COLUMNS & set(df.columns)):
        raise ValueError("CSV must contain either price fields or implied_vol.")

    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.normalize()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["underlying_spot"] = pd.to_numeric(df["underlying_spot"], errors="coerce")

    if "rate" not in df.columns:
        df["rate"] = 0.0
    else:
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0.0)

    if "dividend_yield" not in df.columns:
        df["dividend_yield"] = 0.0
    else:
        df["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce").fillna(0.0)

    if "mid" in df.columns:
        df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    elif "price" in df.columns:
        df["mid"] = pd.to_numeric(df["price"], errors="coerce")
    elif {"bid", "ask"}.issubset(df.columns):
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
    else:
        df["mid"] = np.nan

    if "bid" in df.columns:
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    else:
        df["bid"] = np.nan

    if "ask" in df.columns:
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    else:
        df["ask"] = np.nan

    if "implied_vol" in df.columns:
        df["implied_vol"] = pd.to_numeric(df["implied_vol"], errors="coerce")
    else:
        df["implied_vol"] = np.nan

    df["days_to_expiry"] = (df["expiry"] - df["quote_date"]).dt.days
    df["time_to_expiry"] = df["days_to_expiry"] / 365.0
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct_mid"] = np.where(df["mid"] > 0.0, df["spread"] / df["mid"], np.nan)

    df = df.sort_values(["quote_date", "expiry", "strike"]).reset_index(drop=True)
    return df


def basic_quote_sanity_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mask = (
        out["option_type"].eq("C")
        & out["strike"].gt(0.0)
        & out["underlying_spot"].gt(0.0)
        & out["time_to_expiry"].gt(0.0)
    )

    if "mid" in out.columns:
        mask &= out["mid"].gt(0.0) | out["mid"].isna()

    if "implied_vol" in out.columns:
        mask &= out["implied_vol"].isna() | out["implied_vol"].gt(0.0)

    return out.loc[mask].reset_index(drop=True)