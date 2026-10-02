"""Price-based quote matching, put/call parity and forward-native Black pricing.

These diagnostics do not change the call-only cleaner or surface model. Prices
are index points, D discounts to the quote snapshot, and T is ACT/365F elapsed
seconds. Bid/ask feasible ranges are deterministic sets, not confidence bands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog
from scipy.special import ndtr


def prepare_price_pairs(raw: pd.DataFrame, min_days=7, max_days=365,
                        min_bid=.05, max_relative_spread=.35) -> pd.DataFrame:
    """Audit vendor wide rows; never infer a match across different snapshots.

    Preserve excluded rows and vendor IV. Exact repeats keep one representative;
    conflicting duplicates exclude every member. Available contract metadata
    participates in the key, so different roots/types cannot be pooled later.
    """
    required = ['quote_unixtime', 'expire_unix', 'expire_date', 'strike',
                'underlying_last', 'c_bid', 'c_ask', 'p_bid', 'p_ask']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing pairing fields: {sorted(missing)}')
    q = raw.copy().reset_index(drop=True)
    q['quote_id'] = np.arange(len(q))
    q['elapsed_days'] = (q.expire_unix-q.quote_unixtime)/86400
    q['maturity'] = q.elapsed_days/365
    metadata = [c for c in ['contract_root', 'contract_type', 'settlement_type'] if c in q]
    keys = ['quote_unixtime', 'expire_unix', 'expire_date', 'strike'] + metadata
    q['pair_group'] = q.groupby([c for c in keys if c != 'strike'], dropna=False).ngroup()
    reasons = [[] for _ in range(len(q))]

    def flag(mask, reason):
        for i in np.flatnonzero(np.asarray(mask)):
            reasons[i].append(reason)

    identity = [c for c in raw.columns if c != 'source_file']
    repeat = q.duplicated(identity, keep='first')
    unique = q.loc[~repeat]
    conflict = unique.duplicated(keys, keep=False)
    conflict_ids = set(unique.loc[conflict].set_index(keys).index)
    conflict_mask = pd.Series([key in conflict_ids for key in q.set_index(keys).index])
    flag(repeat, 'identical_duplicate')
    flag(conflict_mask, 'conflicting_duplicate')
    meta_valid = np.isfinite(q[required[:2]+['strike', 'underlying_last']]).all(axis=1)
    meta_valid &= q.strike.gt(0) & q.underlying_last.gt(0) & q.expire_date.notna()
    for col in metadata:
        meta_valid &= q[col].notna()
    flag(~meta_valid, 'invalid_metadata')
    flag(~q.elapsed_days.between(min_days, max_days), 'outside_maturity_window')
    synchronous = pd.Series(True, index=q.index)
    for c in ['c_quote_unixtime', 'p_quote_unixtime']:
        if c in q:
            synchronous &= q[c].eq(q.quote_unixtime)
    flag(~synchronous, 'asynchronous_sides')
    for side in ['c', 'p']:
        b, a = q[side+'_bid'], q[side+'_ask']
        q[side+'_mid'] = (b+a)/2
        q[side+'_relative_spread'] = (a-b)/q[side+'_mid'].where(q[side+'_mid'].gt(0))
        valid = np.isfinite(b) & np.isfinite(a) & b.ge(min_bid) & a.ge(b)
        valid &= q[side+'_mid'].gt(0) & q[side+'_relative_spread'].le(max_relative_spread)
        q[side+'_price_usable'] = valid & meta_valid & synchronous & ~repeat & ~conflict_mask
        q[side+'_price_usable'] &= q.elapsed_days.between(min_days, max_days)
        flag(~valid, side+'_price_liquidity')
        if side+'_iv' not in q:
            q[side+'_iv'] = np.nan
        if side+'_size' in q:
            sizes = q[side+'_size'].astype(str).str.extract(r'^\s*(\d+)\s*x\s*(\d+)\s*$')
            q[side+'_bid_size'] = pd.to_numeric(sizes[0], errors='coerce')
            q[side+'_ask_size'] = pd.to_numeric(sizes[1], errors='coerce')
    q['pair_usable'] = q.c_price_usable & q.p_price_usable
    q['exclusion_reasons'] = [';'.join(x) for x in reasons]
    q['parity_mid'] = q.c_mid-q.p_mid
    q['parity_lower'] = q.c_bid-q.p_ask
    q['parity_upper'] = q.c_ask-q.p_bid
    q['parity_half_width'] = (q.parity_upper-q.parity_lower)/2
    q['log_strike_spot'] = np.log(q.strike.where(q.strike.gt(0))/q.underlying_last.where(q.underlying_last.gt(0)))
    return q


def _interval_ranges(k, lo, hi, center, scale):
    """LP in centered variables, then a linear-fractional change for F bounds."""
    x = (k-center)/scale
    a = np.r_[np.c_[np.ones(len(k)), -x], np.c_[-np.ones(len(k)), x]]
    rhs = np.r_[hi, -lo]
    bounds = [(None, None), (1e-10*scale, None)]
    slack = linprog([0, 0, 1], A_ub=np.c_[a, -np.ones(2*len(k))],
                    b_ub=rhs, bounds=bounds+[(0, None)], method='highs')
    result = dict(interval_feasible=False, minimum_extra_half_width=np.nan,
                  discount_lower=np.nan, discount_upper=np.nan,
                  forward_lower=np.nan, forward_upper=np.nan)
    if not slack.success:
        return result
    result['minimum_extra_half_width'] = float(slack.x[2])
    feasible = linprog([0, 0], A_ub=a, b_ub=rhs, bounds=bounds, method='highs')
    if not feasible.success:
        return result
    result['interval_feasible'] = True
    for sign, name in [(1, 'discount_lower'), (-1, 'discount_upper')]:
        lp = linprog([0, sign], A_ub=a, b_ub=rhs, bounds=bounds, method='highs')
        if lp.success:
            result[name] = float(lp.x[1]/scale)
    # z=(F-center)/scale, t=1/D: scale*z-hi*t <= K-center;
    # -scale*z+lo*t <= -(K-center). t=0 is the closure at infinite D.
    af = np.r_[np.c_[np.full(len(k), scale), -hi],
               np.c_[np.full(len(k), -scale), lo]]
    bf = np.r_[k-center, -(k-center)]
    for sign, name in [(1, 'forward_lower'), (-1, 'forward_upper')]:
        lp = linprog([sign, 0], A_ub=af, b_ub=bf,
                     bounds=[(None, None), (0, 1e10)], method='highs')
        if lp.success:
            result[name] = float(center+scale*lp.x[0])
    return result


def estimate_parity(pairs: pd.DataFrame, min_pairs=8, min_span_fraction=.04) -> dict:
    """Weighted mid-price regression; bid/ask widths determine weights/sets.

    Fit C-P=a-D(K-K0) on centered, scaled strikes, with weight
    1/max((call spread+put spread)/2, .05)^2. Weak/invalid estimates remain
    explicitly flagged; there is no spot/zero-carry fallback.
    """
    result = dict(pair_count=len(pairs), status='weak', reasons='', forward=np.nan,
                  discount=np.nan, interval_feasible=False)
    if len(pairs) < 2:
        result['reasons'] = 'insufficient_pairs'
        return result
    k = pairs.strike.to_numpy(float)
    y = pairs.parity_mid.to_numpy(float)
    lo, hi = pairs.parity_lower.to_numpy(float), pairs.parity_upper.to_numpy(float)
    if not np.isfinite(np.r_[k, y, lo, hi]).all() or np.any(lo > hi):
        raise ValueError('Parity inputs must be finite with ordered intervals')
    h = np.maximum((hi-lo)/2, .05)
    weights = 1/h**2
    center = np.average(k, weights=weights)
    scale = np.sqrt(np.average((k-center)**2, weights=weights))
    spot = float(pairs.underlying_last.median())
    result.update(strike_span=float(np.ptp(k)), spot=spot,
                  effective_pair_count=float(weights.sum()**2/(weights@weights)))
    if scale <= 1e-8*max(spot, 1):
        result['reasons'] = 'unidentified_strike_slope'
        return result
    x = (k-center)/scale
    design = np.c_[np.ones(len(k)), x]/h[:, None]
    beta = np.linalg.lstsq(design, y/h, rcond=None)[0]
    discount = -beta[1]/scale
    forward = center+beta[0]/discount if discount > 0 else np.nan
    predicted = beta[0]+beta[1]*x
    residual = y-predicted
    excess = np.maximum(np.maximum(lo-predicted, predicted-hi), 0)
    result.update(forward=float(forward), discount=float(discount),
                  strike_center=float(center), strike_scale=float(scale),
                  scaled_condition=float(np.linalg.cond(design)),
                  unscaled_condition=float(np.linalg.cond(np.c_[np.ones(len(k)), k]/h[:, None])),
                  residual_rmse=float(np.sqrt(np.mean(residual**2))),
                  weighted_residual_rmse=float(np.sqrt(np.average(residual**2, weights=weights))),
                  residual_max_abs=float(np.max(np.abs(residual))),
                  interval_coverage=float(np.mean(excess <= 1e-7)),
                  max_interval_excess=float(excess.max()))
    result.update(_interval_ranges(k, lo, hi, center, scale))
    reasons = []
    if len(k) < min_pairs:
        reasons.append('few_pairs')
    if np.ptp(k)/spot < min_span_fraction:
        reasons.append('narrow_strike_span')
    if result['effective_pair_count'] < min_pairs:
        reasons.append('low_effective_pair_count')
    if discount <= 0 or not np.isfinite(forward) or forward <= 0:
        reasons.append('nonpositive_forward_or_discount')
    if not result['interval_feasible']:
        reasons.append('inconsistent_bid_ask_intervals')
    if result['max_interval_excess'] > 1e-7:
        reasons.append('mid_fit_outside_intervals')
    if not np.isfinite(result['discount_upper']) or result['discount_upper']-result['discount_lower'] > .05:
        reasons.append('wide_discount_range')
    if not np.isfinite(result['forward_upper']) or (result['forward_upper']-result['forward_lower'])/spot > .005:
        reasons.append('wide_forward_range')
    result['status'] = 'weak' if reasons else 'identified'
    result['reasons'] = ';'.join(reasons)
    return result


def forward_option_price(forward, strike, discount, maturity, volatility, option='call'):
    """Black price, using the OTM leg plus parity to avoid ITM cancellation."""
    if option not in ('call', 'put'):
        raise ValueError('option must be call or put')
    f, k, d, t, v = np.broadcast_arrays(*[np.asarray(a, float) for a in
                                         (forward, strike, discount, maturity, volatility)])
    if not np.isfinite(np.array([f, k, d, t, v])).all() or np.any((f <= 0) | (k <= 0) | (d <= 0) | (t <= 0) | (v < 0)):
        raise ValueError('Finite positive F, K, D, T and nonnegative volatility required')
    s = v*np.sqrt(t)
    safe = np.where(s > 0, s, 1)
    d1 = np.log(f/k)/safe+safe/2
    d2 = d1-safe
    call_otm = d*(f*ndtr(d1)-k*ndtr(d2))
    put_otm = d*(k*ndtr(-d2)-f*ndtr(-d1))
    intrinsic = d*np.maximum(f-k if option == 'call' else k-f, 0)
    time_value = np.where(f <= k, call_otm, put_otm)
    price = intrinsic+np.where(s > 0, time_value, 0)
    return float(price) if price.ndim == 0 else price


@dataclass(frozen=True)
class IVResult:
    volatility: float
    status: str
    lower_bound: float = np.nan
    upper_bound: float = np.nan
    repricing_error: float = np.nan
    vega: float = np.nan


def invert_forward_iv(price, forward, strike, discount, maturity, option='call',
                      price_tolerance=1e-8, max_volatility=10.) -> IVResult:
    """Brent inversion with separate bounds, zero-variance and cap statuses.

    No clipping of observed prices. A price at intrinsic can have unidentifiable
    tiny time value; it is not counted as a positive-IV inversion success.
    """
    if option not in ('call', 'put'):
        raise ValueError('option must be call or put')
    if not np.isfinite([price, forward, strike, discount, maturity, max_volatility]).all():
        return IVResult(np.nan, 'nonfinite_input')
    if min(forward, strike, discount, maturity, max_volatility) <= 0 or price_tolerance < 0:
        return IVResult(np.nan, 'invalid_convention')
    lower = discount*max(forward-strike if option == 'call' else strike-forward, 0)
    upper = discount*(forward if option == 'call' else strike)
    if price < lower-price_tolerance:
        return IVResult(np.nan, 'below_lower_bound', lower, upper)
    if price > upper+price_tolerance:
        return IVResult(np.nan, 'above_upper_bound', lower, upper)
    if abs(price-lower) <= price_tolerance:
        return IVResult(0., 'at_intrinsic_limit', lower, upper, lower-price, 0.)
    if abs(price-upper) <= price_tolerance:
        return IVResult(np.nan, 'at_infinite_volatility_limit', lower, upper)
    def error(vol):
        return forward_option_price(forward, strike, discount, maturity, vol, option)-price
    if error(max_volatility) < 0:
        return IVResult(np.nan, 'volatility_cap_exceeded', lower, upper)
    try:
        vol = brentq(error, 0, max_volatility, xtol=1e-13, rtol=1e-13, maxiter=200)
    except (ValueError, RuntimeError):
        return IVResult(np.nan, 'solver_failure', lower, upper)
    s = vol*np.sqrt(maturity)
    d1 = np.log(forward/strike)/s+s/2
    vega = discount*forward*np.sqrt(maturity)*np.exp(-d1*d1/2)/np.sqrt(2*np.pi)
    return IVResult(float(vol), 'ok', lower, upper, float(error(vol)), float(vega))
