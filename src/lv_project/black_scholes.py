from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive.")
    if strike <= 0.0:
        raise ValueError("strike must be positive.")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive.")
    if vol <= 0.0:
        raise ValueError("vol must be positive.")


def d1(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> float:
    _validate_inputs(spot, strike, maturity, vol)
    numerator = math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * maturity
    denominator = vol * math.sqrt(maturity)
    return numerator / denominator


def d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> float:
    return d1(spot, strike, maturity, rate, dividend_yield, vol) - vol * math.sqrt(maturity)


def call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> float:
    d1_val = d1(spot, strike, maturity, rate, dividend_yield, vol)
    d2_val = d2(spot, strike, maturity, rate, dividend_yield, vol)
    return (
        math.exp(-dividend_yield * maturity) * spot * norm.cdf(d1_val)
        - math.exp(-rate * maturity) * strike * norm.cdf(d2_val)
    )


def put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> float:
    d1_val = d1(spot, strike, maturity, rate, dividend_yield, vol)
    d2_val = d2(spot, strike, maturity, rate, dividend_yield, vol)
    return (
        math.exp(-rate * maturity) * strike * norm.cdf(-d2_val)
        - math.exp(-dividend_yield * maturity) * spot * norm.cdf(-d1_val)
    )


def call_delta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> float:
    d1_val = d1(spot, strike, maturity, rate, dividend_yield, vol)
    return math.exp(-dividend_yield * maturity) * norm.cdf(d1_val)


def call_vega(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
) -> float:
    d1_val = d1(spot, strike, maturity, rate, dividend_yield, vol)
    return math.exp(-dividend_yield * maturity) * spot * math.sqrt(maturity) * norm.pdf(d1_val)


def forward_price(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> float:
    if spot <= 0.0:
        raise ValueError("spot must be positive.")
    if maturity < 0.0:
        raise ValueError("maturity must be non-negative.")
    return spot * math.exp((rate - dividend_yield) * maturity)


def implied_vol_call(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol_lower: float = 1e-4,
    vol_upper: float = 5.0,
) -> float:
    if price <= 0.0:
        raise ValueError("price must be positive.")
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive.")

    intrinsic = max(
        math.exp(-dividend_yield * maturity) * spot - math.exp(-rate * maturity) * strike,
        0.0,
    )
    upper_bound = math.exp(-dividend_yield * maturity) * spot

    if not (intrinsic <= price <= upper_bound):
        raise ValueError("price is outside no-arbitrage bounds for a European call.")

    def objective(vol: float) -> float:
        return call_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            vol=vol,
        ) - price

    return float(brentq(objective, vol_lower, vol_upper, maxiter=200))


def vectorized_call_price(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    vol: np.ndarray,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    rate = np.asarray(rate, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)
    vol = np.asarray(vol, dtype=float)

    if np.any(spot <= 0.0) or np.any(strike <= 0.0) or np.any(maturity <= 0.0) or np.any(vol <= 0.0):
        raise ValueError("spot, strike, maturity, and vol must all be positive.")

    sigma_root_t = vol * np.sqrt(maturity)
    d1_val = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * maturity) / sigma_root_t
    d2_val = d1_val - sigma_root_t

    return (
        np.exp(-dividend_yield * maturity) * spot * norm.cdf(d1_val)
        - np.exp(-rate * maturity) * strike * norm.cdf(d2_val)
    )