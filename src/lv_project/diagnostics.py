"""Support-aware European call diagnostics for an implied-variance surface."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


def reconstruct_call_prices(
    spot: float,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return strikes, discounted calls, forwards and discounts on (T, y).

    F = S exp((r-q)T), D = exp(-rT), K = F exp(y), and
    C = D [F N(d1) - K N(d2)], d1 = -y/sqrt(w) + sqrt(w)/2.
    Zero variance uses discounted intrinsic value; missing/negative variance
    remains NaN. The convention does not infer vendor carry from vendor IV.
    """
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or w.shape != (len(t), len(y)):
        raise ValueError('Expected one-dimensional T/y and variance shape (n_T, n_y).')
    if (not np.isfinite(spot) or spot <= 0 or not np.isfinite([rate, dividend_yield]).all()
            or not np.isfinite(t).all() or np.any(t <= 0) or not np.isfinite(y).all()):
        raise ValueError('Spot and maturities must be positive; coordinates/carry must be finite.')
    forward = spot * np.exp((rate - dividend_yield) * t)
    discount = np.exp(-rate * t)
    strike = forward[:, None] * np.exp(y[None, :])
    prices = np.full_like(w, np.nan)
    valid = np.isfinite(w) & (w > 0)
    root_w = np.sqrt(np.where(valid, w, np.nan))
    d1 = -y[None, :] / root_w + root_w / 2
    d2 = d1 - root_w
    values = discount[:, None] * (forward[:, None] * ndtr(d1) - strike * ndtr(d2))
    prices[valid] = values[valid]
    zero = w == 0
    intrinsic = discount[:, None] * np.maximum(forward[:, None] - strike, 0)
    prices[zero] = intrinsic[zero]
    return strike, prices, forward, discount


def check_call_slice(
    strike: np.ndarray,
    price: np.ndarray,
    forward: float,
    discount: float,
    price_tolerance: float = 1e-8,
    slope_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Check bounds and nonuniform-strike secants, with explicit valid masks.

    A convex slice has nondecreasing secant slopes. At each interior strike,
    butterfly_cost is the linearly weighted neighbor prices minus the center
    price (nonnegative for a convex call). Endpoints have no butterfly check.
    A missing node invalidates its adjacent pairs/triples; no gap is bridged.
    """
    k = np.asarray(strike, dtype=float)
    c = np.asarray(price, dtype=float)
    if (k.ndim != 1 or c.shape != k.shape or k.size < 3
            or not np.isfinite(k).all() or np.any(k <= 0) or np.any(np.diff(k) <= 0)):
        raise ValueError('Need at least three finite positive increasing strikes and matching prices.')
    if (not np.isfinite([forward, discount, price_tolerance, slope_tolerance]).all()
            or forward <= 0 or discount <= 0 or price_tolerance < 0 or slope_tolerance < 0):
        raise ValueError('Forward/discount must be positive and tolerances nonnegative.')
    finite = np.isfinite(c)
    lower = discount * np.maximum(forward - k, 0)
    upper = discount * forward
    slope = np.diff(c) / np.diff(k)
    pair_valid = finite[:-1] & finite[1:]
    triple_valid = finite[:-2] & finite[1:-1] & finite[2:]
    slope_change = np.diff(slope)
    butterfly = ((k[2:] - k[1:-1]) * c[:-2]
                 + (k[1:-1] - k[:-2]) * c[2:]) / (k[2:] - k[:-2]) - c[1:-1]
    return pd.DataFrame({
        'strike': k, 'call_price': c, 'lower_bound': lower, 'upper_bound': upper,
        'price_valid': finite,
        'bounds_violation': finite & ((c < lower-price_tolerance) | (c > upper+price_tolerance)),
        'right_pair_valid': np.r_[pair_valid, False],
        'right_slope': np.r_[slope, np.nan],
        'monotonicity_violation': np.r_[pair_valid & (slope > slope_tolerance), False],
        'vertical_spread_violation': np.r_[pair_valid & (slope < -discount-slope_tolerance), False],
        'butterfly_valid': np.r_[False, triple_valid, False],
        'slope_change': np.r_[np.nan, slope_change, np.nan],
        'strike_curvature': np.r_[np.nan, 2*slope_change/(k[2:]-k[:-2]), np.nan],
        'butterfly_cost': np.r_[np.nan, butterfly, np.nan],
        'butterfly_violation': np.r_[False, triple_valid & (slope_change < -slope_tolerance), False],
    })
