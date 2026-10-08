"""Saved AH surfaces and a separate, support-limited total-variance benchmark.

No calibration, clipping, arbitrage repair, hedge selection or PDE is performed.
The quote benchmark is PCHIP in y and linear in T, not a promoted diffusion.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import log_ndtr, ndtr

from ah_local_vol import AHLocalVariance
from andreasen_huge import AndreasenHugeSurface
from daily_ah_validation import AHShortEndVariance


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def black_time_value(y, w):
    """Normalized OTM price using logarithmic CDFs to reduce cancellation."""
    y, w = np.broadcast_arrays(np.asarray(y, float), np.asarray(w, float))
    result = np.full(y.shape, np.nan)
    zero = np.isfinite(y) & (w == 0)
    result[zero] = 0.0
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
    yy, ww = y[valid], w[valid]
    s = np.sqrt(ww)
    d1, d2 = -yy / s + s / 2, -yy / s - s / 2
    call = yy >= 0
    a = np.where(call, log_ndtr(d1), yy + log_ndtr(-d2))
    b = np.where(call, yy + log_ndtr(d2), log_ndtr(-d1))
    result[valid] = np.exp(a) * (-np.expm1(b - a))
    return result


def black_call(y, w):
    return np.maximum(1 - np.exp(y), 0) + black_time_value(y, w)


def invert_time_value(y, time_value):
    """Return w and explicit status. Invalid prices remain invalid, never clipped."""
    y, price = np.broadcast_arrays(np.asarray(y, float), np.asarray(time_value, float))
    w = np.full(y.shape, np.nan)
    status = np.full(y.shape, 'nonfinite', dtype='<U32')
    finite = np.isfinite(y) & np.isfinite(price)
    status[finite & (price <= 0)] = 'at_or_below_intrinsic'
    upper_price = np.minimum(1.0, np.exp(y))
    status[finite & (price >= upper_price)] = 'at_or_above_upper_bound'
    valid = finite & (price > 0) & (price < upper_price)
    if not valid.any():
        return w, status
    yy, target = y[valid], price[valid]
    upper = np.full(yy.shape, 0.04)
    for _ in range(20):
        small = black_time_value(yy, upper) < target
        if not small.any():
            break
        upper[small] *= 4
    bracketed = black_time_value(yy, upper) >= target
    lower = np.zeros_like(upper)
    for _ in range(64):
        middle = (upper + lower) / 2
        small = black_time_value(yy, middle) < target
        lower = np.where(small, middle, lower)
        upper = np.where(small, upper, middle)
    candidate = (upper + lower) / 2
    w[valid] = np.where(bracketed, candidate, np.nan)
    status[valid] = np.where(bracketed, 'ready', 'inversion_unresolved')
    return w, status


def dupire(y, w, wy, wyy, wt, denominator_floor=1e-8):
    """w_T / g and signed density in z=K/F. No negative value is repaired."""
    y, w, wy, wyy, wt = np.broadcast_arrays(y, w, wy, wyy, wt)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        g = (1 - y * wy / (2 * w))**2 - wy**2 / 4 * (1 / w + 0.25) + wyy / 2
        d2 = -y / np.sqrt(w) - np.sqrt(w) / 2
        density = np.exp(-d2**2 / 2) / np.sqrt(2 * np.pi * w) / np.exp(y) * g
        a = wt / g
    admissible = (np.isfinite(a) & np.isfinite(g) & np.isfinite(density)
                  & (w > 0) & (wt >= 0) & (g > denominator_floor))
    return dict(g=g, density_z=density, variance=np.where(admissible, a, np.nan),
                admissible=admissible)


class TotalVarianceSurface:
    """Independent quote IVs; contiguous valid segments, no wing/short extrapolation.

    PCHIP is C1 in y. The second derivative is one-sided at its knots.
    Both strike and expiry derivative sides are exposed, rather than averaged.
    A missing IV splits a segment: no interpolation across an invalid quote.
    """

    def __init__(self, quotes, carry):
        self.carry = carry.sort_values('maturity_years').copy()
        self.times = self.carry.maturity_years.to_numpy(float)
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError('The total-variance benchmark requires two ordered expiries.')
        self.observations, self.slices = [], []
        for row in self.carry.to_dict('records'):
            q = quotes.loc[quotes.expire_date.eq(row['expire_date'])].sort_values('strike').copy()
            if len(q) < 2 or q.strike.duplicated().any():
                raise ValueError('Every expiry requires at least two unique observed strikes.')
            y = np.log(q.strike.to_numpy(float) / row['forward'])
            normalized = q.call_mid.to_numpy(float) / (row['discount_factor'] * row['forward'])
            tv = normalized - np.maximum(1 - np.exp(y), 0)
            w, status = invert_time_value(y, tv)
            q['y'], q['observed_w'], q['iv_status'] = y, w, status
            q['observed_iv'] = np.sqrt(w / row['maturity_years'])
            self.observations.append(q)
            indices = np.flatnonzero(status == 'ready')
            runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            segments = []
            for run in runs:
                if len(run) >= 2:
                    segments.append(PchipInterpolator(y[run], w[run], extrapolate=False))
            self.slices.append(segments)
        self.observations = pd.concat(self.observations, ignore_index=True)

    def _slice(self, index, y, strike_side):
        out = [np.full(y.shape, np.nan) for _ in range(3)]
        knot = np.zeros(y.shape, bool)
        for curve in self.slices[index]:
            valid = (y >= curve.x[0]) & (y <= curve.x[-1])
            yy = y[valid]
            if not len(yy):
                continue
            # Explicit polynomial evaluation chooses the requested knot side.
            cell = np.searchsorted(curve.x, yy, side=strike_side) - 1
            cell = np.clip(cell, 0, len(curve.x) - 2)
            x, c = yy - curve.x[cell], curve.c[:, cell]
            out[0][valid] = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
            out[1][valid] = (3 * c[0] * x + 2 * c[1]) * x + c[2]
            out[2][valid] = 6 * c[0] * x + 2 * c[1]
            knot[valid] = np.any(np.isclose(yy[:, None], curve.x[None, 1:-1],
                                             rtol=0, atol=1e-12), axis=1)
        return (*out, knot)

    def evaluate(self, y, time, side='right', strike_side='right'):
        y = np.atleast_1d(np.asarray(y, float))
        if y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(time):
            raise ValueError('Require finite one-dimensional y and scalar T.')
        if side not in ('left', 'right') or strike_side not in ('left', 'right'):
            raise ValueError('Derivative sides must be left or right.')
        nearest = int(np.argmin(abs(self.times-time)))
        if abs(self.times[nearest]-time) <= 1e-14:
            time = float(self.times[nearest])
        empty = np.full(y.shape, np.nan)
        if not self.times[0] <= time <= self.times[-1]:
            return dict(w=empty.copy(), wy=empty.copy(), wyy=empty.copy(), wt=empty.copy(),
                        supported=np.zeros(y.shape, bool), at_strike_knot=np.zeros(y.shape, bool))
        if (time == self.times[0] and side == 'left') or (time == self.times[-1] and side == 'right'):
            index = 0 if time == self.times[0] else len(self.times)-1
            w, wy, wyy, knot = self._slice(index, y, strike_side)
            return dict(w=w, wy=wy, wyy=wyy, wt=empty.copy(),
                        supported=np.zeros(y.shape, bool), at_strike_knot=knot)
        i = int(np.searchsorted(self.times, time, side=side)) - 1
        i = min(max(i, 0), len(self.times) - 2)
        lo, hi = self._slice(i, y, strike_side), self._slice(i + 1, y, strike_side)
        alpha = (time - self.times[i]) / (self.times[i + 1] - self.times[i])
        parts = [(1 - alpha) * lo[k] + alpha * hi[k] for k in range(3)]
        # The price itself at a pillar does not need the neighbouring support.
        # Its time derivative does. Save both masks, do not extrapolate either.
        if alpha == 0:
            parts[0] = lo[0].copy()
        elif alpha == 1:
            parts[0] = hi[0].copy()
        wt = (hi[0] - lo[0]) / (self.times[i + 1] - self.times[i])
        supported = np.isfinite(np.vstack(parts + [wt])).all(axis=0)
        return dict(w=parts[0], wy=parts[1], wyy=parts[2], wt=wt,
                    supported=supported, at_strike_knot=lo[3] | hi[3])


@dataclass(frozen=True)
class SurfaceEvidenceSettings:
    half_width: float = 0.09
    y_points: int = 181
    time_subdivisions: int = 2
    radius: float = 0.0005
    denominator_floor: float = 1e-8
    stencil_relative_tolerance: float = 0.05
    stencil_absolute_tolerance: float = 1e-4

    def __post_init__(self):
        if (not np.isfinite([self.half_width, self.radius, self.denominator_floor,
                            self.stencil_relative_tolerance, self.stencil_absolute_tolerance]).all()
                or not 0 < self.half_width <= 0.25 or self.radius < 0
                or self.denominator_floor <= 0 or self.stencil_relative_tolerance <= 0
                or self.stencil_absolute_tolerance <= 0
                or not isinstance(self.y_points, (int, np.integer)) or self.y_points < 21
                or self.y_points % 2 == 0
                or not isinstance(self.time_subdivisions, (int, np.integer)) or self.time_subdivisions < 1):
            raise ValueError('Invalid evidence grid, radius or tolerances.')


class SurfaceEvidence:
    def __init__(self, settings=None):
        self.settings = settings or SurfaceEvidenceSettings()

    @staticmethod
    def ah_variance_derivatives(model, time, side):
        """Discrete chain rule using native price diagnostics, not d2 linear prices."""
        state = model.node_state(time, side)
        z = model.grid.z[1:-1]
        y = np.log(z)
        w, status = invert_time_value(y, state['time_value'][1:-1])
        slopes = np.diff(state['calls']) / np.diff(model.grid.z)
        tv_slopes = np.diff(state['time_value']) / np.diff(model.grid.z)
        left, right = np.diff(model.grid.z)[:-1], np.diff(model.grid.z)[1:]
        cz = (right * slopes[:-1] + left * slopes[1:]) / (left + right)
        tv_cz = (right * tv_slopes[:-1] + left * tv_slopes[1:]) / (left + right)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            s = np.sqrt(w)
            d1, d2 = -y / s + s / 2, -y / s - s / 2
            phi1, phi2 = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi), np.exp(-d2**2 / 2) / np.sqrt(2 * np.pi)
            cw = phi1 / (2 * s)
            # In the ITM wing, cancel the intrinsic terms symbolically rather
            # than subtracting almost-equal prices/slopes of order one.
            cy_minus_by = np.where(y < 0, z*(tv_cz-ndtr(-d2)), z*(cz+ndtr(d2)))
            cyy_minus_byy = z**2*state['curvature']+cy_minus_by-z*phi2/s
            byw = z * phi2 / (2 * s) * (0.5 - y / w)
            bww = cw * (y**2 / (2 * w**2) - 1 / 8 - 1 / (2 * w))
            wy = cy_minus_by / cw
            wyy = (cyy_minus_byy - 2 * byw * wy - bww * wy**2) / cw
            wt = state['time_derivative'] / cw
        return dict(y=y, w=w, wy=wy, wyy=wyy, wt=wt, iv_status=status,
                    density_z=state['curvature'], ct=state['time_derivative'])

    def run(self, model, quotes, carry, labels):
        from daily_ah import DailyAHCalibrator
        q, c = DailyAHCalibrator()._checked(quotes, carry)
        if (not np.allclose(c.maturity_years, model.maturities, rtol=1e-12, atol=1e-12)
                or not np.allclose(c.forward, model.forwards, rtol=1e-12, atol=1e-9)
                or not np.allclose(c.discount_factor, model.discounts, rtol=1e-12, atol=1e-12)
                or not np.isclose(c.spot.iloc[0], model.spot, rtol=0, atol=1e-8)):
            raise ValueError('Model and saved same-date carry disagree.')
        s = self.settings
        benchmark = TotalVarianceSurface(q, c)
        raw, smooth = AHLocalVariance(model), AHShortEndVariance(model, s.radius)
        y = np.linspace(-s.half_width, s.half_width, s.y_points)
        h = float(np.diff(np.log(model.grid.z)).mean())
        if s.half_width + 8 * h >= raw.y[-1]:
            raise ValueError('Evidence stencils exceed the AH coefficient domain.')
        times = list(model.maturities)
        starts = np.r_[0.0, model.maturities[:-1]]
        for a, b in zip(starts, model.maturities):
            times.extend(np.linspace(a, b, s.time_subdivisions + 1)[1:-1])
        times.extend(t for t in (0.25 / 365, 1 / 365) if t < model.maturities[0])
        rows, shapes, moments, seams, numerical = [], [], [], [], []
        previous_calls = np.maximum(1-model.grid.z, 0)
        for time in np.unique(times):
            is_pillar = np.any(abs(model.maturities - time) <= 1e-14)
            sides = ('left', 'right') if is_pillar and time < model.maturities[-1] else ('left',) if is_pillar else ('right',)
            state = model.node_state(float(time))
            tv = np.interp(np.exp(y), model.grid.z, state['time_value'])
            aw, iv_status = invert_time_value(y, tv)
            native = self.ah_variance_derivatives(model, float(time), sides[0])
            for side in sides:
                av = raw.normalized_variance(np.exp(y), float(time), side)
                sv = smooth.normalized_variance(np.exp(y), float(time), side)
                if side != sides[0]:
                    native = self.ah_variance_derivatives(model, float(time), side)
                derived = dupire(native['y'], native['w'], native['wy'], native['wyy'], native['wt'], s.denominator_floor)
                mapped = {key: np.interp(y, native['y'], value) for key, value in native.items()
                          if key in ('wy', 'wyy', 'wt', 'density_z', 'ct')}
                recovered = np.interp(y, native['y'], derived['variance'])
                ah_g = np.interp(y, native['y'], derived['g'])
                independent = benchmark.evaluate(y, float(time), side)
                independent_left = benchmark.evaluate(y, float(time), side, 'left')
                tvd = dupire(y, independent['w'], independent['wy'], independent['wyy'], independent['wt'], s.denominator_floor)
                # Two spatial widths. In time, stay strictly in the same expiry interval.
                fd_values = []
                interval = min(int(np.searchsorted(model.maturities, time, side=side)), len(model.maturities)-1)
                start = 0 if interval == 0 else model.maturities[interval - 1]
                stop = model.maturities[interval]
                dt = min((stop-start) * 1e-3, time * 1e-3)
                for width in (8 * h, 4 * h):
                    ym, yp = y - width, y + width
                    wm = invert_time_value(ym, np.interp(np.exp(ym), model.grid.z, state['time_value']))[0]
                    wp = invert_time_value(yp, np.interp(np.exp(yp), model.grid.z, state['time_value']))[0]
                    wy, wyy = (wp-wm)/(2*width), (wp-2*aw+wm)/width**2
                    if time == start:
                        t0, t1 = time, time + dt
                    elif time == stop:
                        t0, t1 = time - dt, time
                    else:
                        dt = min(dt, (time-start)/2, (stop-time)/2)
                        t0, t1 = time-dt, time+dt
                    def at(t):
                        values = model.node_state(float(t))['time_value']
                        return invert_time_value(y, np.interp(np.exp(y), model.grid.z, values))[0]
                    wt = (at(t1)-at(t0))/(t1-t0)
                    fd_values.append(dupire(y, aw, wy, wyy, wt, s.denominator_floor)['variance'])
                fd_coarse, fd_fine = fd_values
                fd_change = abs(fd_fine-fd_coarse)
                fd_stable = np.isfinite(fd_change) & (fd_change <= s.stencil_absolute_tolerance + s.stencil_relative_tolerance * abs(fd_fine))
                # Analytical quote-PCHIP derivatives versus finite differences.
                dy = (y[-1]-y[0])/(len(y)-1)/4
                bm, bp = benchmark.evaluate(y-dy, float(time), side), benchmark.evaluate(y+dy, float(time), side)
                bm2, bp2 = benchmark.evaluate(y-dy/2, float(time), side), benchmark.evaluate(y+dy/2, float(time), side)
                bfd = []
                for minus, plus, width in ((bm, bp, dy), (bm2, bp2, dy/2)):
                    bfd.append(dupire(y, independent['w'], (plus['w']-minus['w'])/(2*width),
                        (plus['w']-2*independent['w']+minus['w'])/width**2, independent['wt'], s.denominator_floor)['variance'])
                bchange = abs(bfd[1]-bfd[0])
                bagree = abs(bfd[1]-tvd['variance'])
                bstable = (np.isfinite(bchange) & np.isfinite(bagree)
                    & (bchange <= s.stencil_absolute_tolerance + s.stencil_relative_tolerance*abs(tvd['variance']))
                    & (bagree <= s.stencil_absolute_tolerance + s.stencil_relative_tolerance*abs(tvd['variance']))
                    & ~independent['at_strike_knot'])
                supported = independent['supported']
                with np.errstate(divide='ignore', invalid='ignore'):
                    bw = independent['w']
                    d1 = -y/np.sqrt(bw)+np.sqrt(bw)/2
                    d2 = d1-np.sqrt(bw)
                    quote_cz = -ndtr(d2)+np.exp(-d1**2/2)/np.sqrt(2*np.pi)/(2*np.sqrt(bw))*independent['wy']/np.exp(y)
                vertical_ok = np.isfinite(quote_cz) & (quote_cz <= 1e-8) & (quote_cz >= -1-1e-8)
                admissible = supported & tvd['admissible'] & bstable & vertical_ok
                frame = pd.DataFrame({**labels, 'T': time, 'days': 365*time, 'side': side, 'y': y,
                    'K': model.forward(time)*np.exp(y), 'F': model.forward(time), 'D': model.discount_factor(time),
                    'ah_iv_status': iv_status, 'ah_iv': np.sqrt(aw/time), 'ah_w': aw,
                    'ah_actual_lv': np.sqrt(av), 'smoothed_actual_lv': np.sqrt(sv),
                    'ah_proxy_vol': np.sqrt(np.interp(y, raw.y, model.node_state(float(time), side)['proxy_variance'])),
                    'ah_wy': mapped['wy'], 'ah_wyy': mapped['wyy'], 'ah_wt': mapped['wt'],
                    'ah_density_z': mapped['density_z'], 'ah_density_K': mapped['density_z']/model.forward(time),
                    'ah_ct': mapped['ct'], 'ah_chain_dupire_variance': recovered,
                    'ah_g': ah_g,
                    'ah_chain_minus_actual_variance': recovered-av,
                    'ah_fd_variance_coarse': fd_coarse, 'ah_fd_variance_fine': fd_fine,
                    'ah_fd_change': fd_change, 'ah_fd_stable': fd_stable,
                    'quote_iv': np.sqrt(independent['w']/time), 'quote_w': independent['w'],
                    'quote_wy': independent['wy'], 'quote_wyy': independent['wyy'], 'quote_wt': independent['wt'],
                    'quote_g': tvd['g'], 'quote_density_z': tvd['density_z'],
                    'quote_density_K': tvd['density_z']/model.forward(time),
                    'quote_supported': supported, 'quote_price_supported': np.isfinite(independent['w']),
                    'quote_formula_admissible': supported & tvd['admissible'], 'quote_derivative_stable': bstable,
                    'quote_at_strike_knot': independent['at_strike_knot'], 'quote_admissible': admissible,
                    'quote_cz': quote_cz, 'quote_vertical_admissible': vertical_ok,
                    'quote_fd_change': bchange, 'quote_fd_minus_analytic_variance': bfd[1]-tvd['variance'],
                    'quote_dupire_variance': tvd['variance'],
                    'quote_local_vol': np.where(admissible, np.sqrt(tvd['variance']), np.nan),
                    'quote_minus_ah_variance': np.where(admissible, tvd['variance']-av, np.nan),
                    'quote_call_minus_ah_points': model.discount_factor(time)*model.forward(time)*(black_call(y, independent['w'])-(np.maximum(1-np.exp(y),0)+tv)),
                    'quote_strike_wyy_jump': independent['wyy']-independent_left['wyy']})
                rows.append(frame)
                shapes.append({**labels, 'days': 365*time, 'side': side, 'route': 'quote_total_variance',
                    'samples': len(y), 'supported': int(supported.sum()), 'admissible': int(admissible.sum()),
                    'negative_calendar_derivatives': int((supported & (independent['wt'] < -1e-10)).sum()),
                    'negative_density_samples': int((supported & (tvd['density_z'] < -1e-10)).sum()),
                    'ill_conditioned_denominators': int((supported & (tvd['g'] <= s.denominator_floor)).sum()),
                    'unstable_derivative_samples': int((supported & ~bstable).sum()),
                    'increasing_prices': int((supported & (quote_cz > 1e-8)).sum()),
                    'vertical_spread_violations': int((supported & (quote_cz < -1-1e-8)).sum())})
                numerical.append({**labels, 'days': 365*time, 'side': side,
                    'ah_chain_comparable': int(np.isfinite(recovered).sum()),
                    'max_ah_chain_variance_difference': finite_max(abs(recovered-av)),
                    'max_ah_fd_variance_difference': finite_max(abs(fd_fine-av)),
                    'max_ah_fd_width_change': finite_max(fd_change), 'ah_fd_stable_samples': int(fd_stable.sum()),
                    'quote_supported_samples': int(supported.sum()), 'quote_admissible_samples': int(admissible.sum()),
                    'max_quote_variance_difference': finite_max(abs(frame.quote_minus_ah_variance.to_numpy())),
                    'max_quote_price_difference': finite_max(abs(frame.quote_call_minus_ah_points.to_numpy()))})
            calls = state['calls']
            slopes = np.diff(calls)/np.diff(model.grid.z)
            shapes.append({**labels, 'days': 365*time, 'side': 'price', 'route': 'AH_native_full_domain',
                'samples': len(calls), 'increasing_prices': int((slopes > 1e-8).sum()),
                'vertical_spread_violations': int((slopes < -1-1e-8).sum()),
                'negative_butterflies': int((np.diff(slopes) < -1e-8).sum()),
                'intrinsic_shortfalls': int((calls < np.maximum(1-model.grid.z,0)-1e-10).sum()),
                'upper_bound_excesses': int((calls > 1+1e-10).sum()),
                'calendar_violations': int((calls < previous_calls-1e-10).sum())})
            previous_calls = calls.copy()
            density = state['curvature']
            dz = (model.grid.z[2:]-model.grid.z[:-2])/2
            window = abs(np.log(model.grid.z[1:-1])) <= s.half_width
            moments.append({**labels, 'days': 365*time,
                'finite_domain_nodal_mass': float(np.sum(density*dz)),
                'finite_domain_nodal_first_moment': float(np.sum(model.grid.z[1:-1]*density*dz)),
                'report_window_nodal_mass': float(np.sum(density[window]*dz[window])),
                'negative_nodal_densities': int((density < -1e-10).sum()),
                'zero_or_underflow_nodal_densities': int((density == 0).sum())})
            if len(sides) == 2:
                a, b = rows[-2], rows[-1]
                for route, column in [('AH_actual', 'ah_actual_lv'), ('quote_Dupire', 'quote_local_vol')]:
                    change = b[column].to_numpy()-a[column].to_numpy()
                    seams.append({**labels, 'days': 365*time, 'route': route,
                        'comparable_samples': int(np.isfinite(change).sum()),
                        'max_absolute_lv_jump': finite_max(abs(change))})
        observations = benchmark.observations.copy()
        observations['ah_call'] = np.nan
        for expiry, group in observations.groupby('expire_date', sort=False):
            time = float(c.loc[c.expire_date.eq(expiry), 'maturity_years'].iloc[0])
            observations.loc[group.index, 'ah_call'] = model.call_price(group.strike.to_numpy(), time)
        observations['ah_residual_half_spreads'] = (observations.ah_call-observations.call_mid)/observations.call_half_width
        observations['quote_reconstructed_call'] = observations.discount_factor*observations.forward*black_call(observations.y.to_numpy(), observations.observed_w.to_numpy())
        observations['quote_reconstruction_error_points'] = observations.quote_reconstructed_call-observations.call_mid
        return dict(surface_samples=pd.concat(rows, ignore_index=True), quote_inversions=observations,
                    shape_checks=pd.DataFrame(shapes), density_moments=pd.DataFrame(moments),
                    pillar_jumps=pd.DataFrame(seams), numerical_comparison=pd.DataFrame(numerical))


def finite_max(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    return float(values[valid].max()) if valid.any() else np.nan


def analytical_controls():
    """Spatial stencil refinement on flat and smoothly skewed Black price surfaces.

    These deterministic controls are not historical data. The skewed control
    checks price-route density against the variance-route formula, not a fit.
    """
    rows = []
    y = np.linspace(-0.08, 0.08, 41)
    for control, skew, curvature in [('flat_Black', 0.0, 0.0), ('smooth_skew', -0.1, 0.4)]:
        time = 0.2
        rate = 0.04 + skew*y + curvature*y*y
        w, wy, wyy = time*rate, time*(skew+2*curvature*y), np.full(y.shape, time*2*curvature)
        exact = dupire(y, w, wy, wyy, rate)
        for width in (0.004, 0.002, 0.001, 0.0005):
            z = np.exp(y)
            def price(zz):
                yy = np.log(zz)
                return black_call(yy, time*(0.04+skew*yy+curvature*yy*yy))
            dz = width*z
            density = (price(z+dz)-2*price(z)+price(z-dz))/dz**2
            dt = time*width
            ct = (black_call(y, (time+dt)*rate)-black_call(y, (time-dt)*rate))/(2*dt)
            recovered = 2*ct/(z*z*density)
            rows.append(dict(control=control, width=width,
                max_density_error=finite_max(abs(density-exact['density_z'])),
                max_variance_error=finite_max(abs(recovered-exact['variance'])),
                minimum_exact_g=float(exact['g'].min()),
                flat_expected_variance=0.04 if control == 'flat_Black' else np.nan))
    return pd.DataFrame(rows)


def plot_surface_evidence(samples, output_prefix):
    """Uncapped scientific views. NaNs are gaps; y is forward log-moneyness."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = samples.drop_duplicates(['T', 'y'], keep='first').sort_values(['T', 'y'])
    times, ys = np.sort(data['T'].unique()), np.sort(data.y.unique())
    title = f"{data.quote_date.iloc[0]} | {data.root.iloc[0]} | finite sampled support"
    fields = [('ah_iv', 'AH implied volatility'), ('ah_w', 'AH total variance'),
              ('ah_actual_lv', 'AH actual local volatility'),
              ('quote_iv', 'Quote PCHIP implied volatility'), ('quote_w', 'Quote PCHIP total variance'),
              ('quote_local_vol', 'Quote Dupire LV: admissible samples')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields):
        values = data.pivot(index='T', columns='y', values=field).to_numpy()
        shown = values if field.endswith('_w') else 100*values
        mesh = axis.pcolormesh(ys, times*365, np.ma.masked_invalid(shown), shading='nearest', cmap='viridis')
        axis.set(title=label, xlabel='y = log(K/F(T))', ylabel='Calendar days')
        fig.colorbar(mesh, ax=axis, label='variance' if field.endswith('_w') else '% per sqrt(year)')
    fig.suptitle(title+'\nBlank cells show missing support or failed admissibility. No volatility cap.')
    fig.savefig(str(output_prefix)+'_surfaces.png', dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    for i, (field, label) in enumerate(fields[:3], 1):
        axis = fig.add_subplot(1, 3, i, projection='3d')
        values = data.pivot(index='T', columns='y', values=field).to_numpy()
        x, t = np.meshgrid(ys, times*365)
        axis.plot_surface(x, t, values if field.endswith('_w') else 100*values, cmap='viridis', linewidth=0)
        axis.set(title=label, xlabel='log(K/F)', ylabel='Days', zlabel='w' if field.endswith('_w') else '%')
    fig.suptitle(title+' | original AH; values at pillars use the left side')
    fig.savefig(str(output_prefix)+'_3d.png', dpi=150)
    plt.close(fig)
    targets = times[[0, len(times)//2, -1]]
    fields = [('ah_wy', 'quote_wy', 'w_y'), ('ah_wyy', 'quote_wyy', 'w_yy'),
              ('ah_wt', 'quote_wt', 'w_T'), ('ah_g', 'quote_g', 'Dupire denominator g'),
              ('ah_density_K', 'quote_density_K', 'Density in physical K'),
              (None, 'quote_admissible', 'Quote admissibility mask')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (ah_field, quote_field, label) in zip(axes.flat, fields):
        for time in targets:
            part = data.loc[data['T'].eq(time)]
            line, = axis.plot(part.y, part[quote_field], '--', label=f'Quote {time*365:.3g}d')
            if ah_field:
                axis.plot(part.y, part[ah_field], color=line.get_color(), label=f'AH {time*365:.3g}d')
        axis.set(title=label, xlabel='log(K/F)')
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(title+'\nDiagnostics are signed; invalid cells are retained in the CSV tables.')
    fig.savefig(str(output_prefix)+'_diagnostics.png', dpi=150)
    plt.close(fig)
    plot_coefficient_evidence(samples, output_prefix)


def plot_coefficient_evidence(samples, output_prefix):
    """Plot one positive-time slice in increasing y, with an explicit pillar side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    earliest = samples['T'].min()
    early = samples.loc[samples['T'].eq(earliest)].copy()
    side = 'left' if early.side.eq('left').any() else 'right'
    early = early.loc[early.side.eq(side)].sort_values('y')
    if early.empty or early.y.duplicated().any() or not early.y.is_monotonic_increasing:
        raise ValueError('Coefficient figure requires one ordered positive-time slice.')
    title = f"{early.quote_date.iloc[0]} | {early.root.iloc[0]} | finite sampled support"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for field, label in [('ah_actual_lv','AH actual'), ('smoothed_actual_lv','Smoothed actual'), ('ah_proxy_vol','AH calibration proxy')]:
        axes[0].plot(early.y, 100*early[field], label=label)
    axes[0].set(title=f'Early coefficients: {early.days.iloc[0]:.3g} days', xlabel='log(state/F)', ylabel='% per sqrt(year)')
    axes[0].legend()
    axes[1].plot(early.y, early.ah_fd_variance_fine, label='AH w finite differences')
    axes[1].plot(early.y, early.ah_actual_lv**2, label='AH actual variance')
    axes[1].set(title='Same AH surface: numerical recovery', xlabel='log(K/F)', ylabel='Variance per year')
    axes[1].legend()
    fig.suptitle(title+'\nSmoothing changes the diffusion. AH IV/w/density panels refer to the original model.')
    fig.savefig(str(output_prefix)+'_coefficients.png', dpi=150)
    plt.close(fig)


def write_evidence_notebook(output):
    """Saved evidence only; editable date/root and no hidden model or PDE execution."""
    output = Path(output)
    cells = []
    def cell(kind, text):
        item = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '# Surface evidence\n\nOne surface per date and root. AH price diagnostics are discrete native-node quantities. '
         'Quote PCHIP is an independent diagnostic benchmark, not an arbitrage-free model certification. '
         'The smoothed coefficient is a changed diffusion; its prices and density are not the original AH prices and density. '
         'No coefficient is requested at zero.\n')
    cell('code', "from pathlib import Path\nimport json\nimport hashlib\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\n"
         "ROOT = Path.cwd()\n# If this notebook was opened from another working directory, set ROOT to its output folder.\n"
         "audit = json.loads((ROOT / 'audit.json').read_text())\nassert audit['status'] in ('completed', 'completed_with_failures')\n"
         "for name, digest in audit['output_sha256'].items():\n"
         "    if name.endswith('.ipynb'):  # Jupyter saves execution outputs into the notebook.\n"
         "        continue\n"
         "    path = (ROOT / name).resolve()\n"
         "    assert path.is_relative_to(ROOT.resolve()), 'Output path escapes evidence folder'\n"
         "    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Changed saved output: {name}'\n"
         "status = pd.read_csv(ROOT / 'evidence_status.csv', dtype={'quote_date':str, 'root':str})\n"
         "display(status)\ndisplay(pd.read_csv(ROOT / 'daily_summary.csv'))\n")
    cell('code', "# Change these values to inspect any saved date/root.\nDATE = str(status.loc[status.status.eq('completed'), 'quote_date'].iloc[0])\n"
         "ROOT_LABEL = str(status.loc[status.quote_date.eq(DATE) & status.status.eq('completed'), 'root'].iloc[0])\n"
         "record = status.loc[status.quote_date.eq(DATE) & status.root.eq(ROOT_LABEL)].iloc[0]\n"
         "samples = pd.read_csv(ROOT / record.samples_file)\n"
         "for suffix in ['surfaces', '3d', 'diagnostics', 'coefficients']:\n"
         "    display(Image(filename=str(ROOT / (record.plot_prefix + '_' + suffix + '.png'))))\n")
    cell('markdown', '## Rotate a saved 3D surface\n\nThe next cell can be rerun with a different FIELD. '
         'Use `%matplotlib widget` if your existing environment supports that backend; otherwise the static view works. '
         'No extra plotting package is required. Percent units apply to volatility, not total variance.\n')
    cell('code', "FIELD = 'ah_actual_lv'  # also ah_iv, ah_w, quote_iv, quote_w, quote_local_vol\n"
         "d = samples.drop_duplicates(['T','y']).sort_values(['T','y'])\n"
         "values = d.pivot(index='T',columns='y',values=FIELD)\n"
         "x, t = np.meshgrid(values.columns.to_numpy(), values.index.to_numpy()*365)\n"
         "fig = plt.figure(figsize=(10,6)); ax = fig.add_subplot(111,projection='3d')\n"
         "ax.plot_surface(x,t,values.to_numpy()*(1 if FIELD.endswith('_w') else 100),cmap='viridis',rcount=len(values.index),ccount=len(values.columns))\n"
         "ax.set(xlabel='log(K/F)',ylabel='Days',zlabel='w' if FIELD.endswith('_w') else '%',title=f'{DATE}: {FIELD}')\nplt.show()\n")
    cell('code', "for name in ['shape_checks','numerical_comparison','density_moments','pillar_jumps']:\n"
         "    f = pd.read_csv(ROOT / (name+'.csv'),dtype={'quote_date':str,'root':str})\n"
         "    print(name); display(f.loc[f.quote_date.eq(DATE) & f.root.eq(ROOT_LABEL)])\n"
         "display(pd.read_csv(ROOT / 'analytical_controls.csv'))\n")
    cell('markdown', '## Interpret the comparison\n\n'
         '* `AH chain rule` is a coordinate/formula consistency check on native price diagnostics. It is not an independent calibration or continuum proof.\n'
         '* `AH finite differences` differentiates the piecewise-linear price representation through its implied variance using stencils spanning several native cells. Width sensitivity is recorded.\n'
         '* `Quote Dupire` uses observed midpoint IVs, PCHIP in log-forward-moneyness and linear total variance between expiries. '
         'It is unsupported before the first pillar, after the last, outside contiguous valid quote support, or across an IV inversion gap. '
         'Local variance is shown only where time and density conditions and the derivative checks pass. '
         'All other signed diagnostics remain in the tables. PCHIP second derivatives can jump at strike knots, and time derivatives can jump at expiry pillars.\n'
         '* Native AH density sums are finite-domain quadrature diagnostics. Window mass is not rescaled to one. '
         'Passing sampled tests does not establish global tail conditions.\n'
         '* Quote reconstruction at its input knots is an inversion check, not evidence of predictive fit. '
         'An interpolator can reproduce observed midpoints and still have negative density or calendar derivatives.\n'
         '* These outputs assess surface construction. They do not establish hedge outperformance or validate every entry Greek.\n')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python',name='python3'),
        language_info=dict(name='python')), nbformat=4, nbformat_minor=4)
    (output/'surface_evidence.ipynb').write_text(json.dumps(notebook, indent=2)+'\n')


class SurfaceEvidenceInputs:
    """Pin only the required calibration/carry artifacts from an explicit run index."""

    def __init__(self, index_file, month, repository=None):
        self.repository = Path(repository or Path.cwd()).resolve()
        self.index_file = Path(index_file).resolve()
        self.inputs = {str(self.index_file): sha256(self.index_file)}
        index = json.loads(self.index_file.read_text())
        record = index['months'][month]
        self.folders, self.audits = {}, {}
        for stage in ('carry', 'calibration'):
            path = Path(record[stage])
            folder = (path if path.is_absolute() else self.repository/path).resolve()
            self.folders[stage] = folder
            self.audits[stage] = self.read_json(folder/'audit.json')
        self.manifest = self.table('calibration', 'model_manifest')
        self.quotes = self.table('carry', 'calibration_quotes')
        self.carry = self.table('carry', 'primary_carry')
        if self.manifest.duplicated(['quote_date', 'root']).any():
            raise ValueError('Duplicate model identities.')
        if self.manifest.empty or not self.manifest.quote_date.str.startswith(month+'-').all():
            raise ValueError('Calibration manifest dates disagree with the indexed month.')
        for name in ('calibration_quotes.csv', 'primary_carry.csv'):
            pinned = self.inputs[str(self.folders['carry']/name)]
            if pinned not in self.audits['calibration'].get('input_sha256', {}).values():
                raise ValueError('Calibration does not pin this carry producer.')
        self.models = {}
        for row in self.manifest.to_dict('records'):
            if row['status'] != 'fitted':
                continue
            path = (self.folders['calibration']/row['model_file']).resolve()
            if not path.is_relative_to(self.folders['calibration']):
                raise ValueError('Model path escapes the calibration folder.')
            digest = sha256(path)
            if digest != row['model_sha256'] or digest != self.audits['calibration'].get('output_sha256', {}).get(row['model_file']):
                raise ValueError(f'Model checksum mismatch: {path}')
            self.inputs[str(path)] = digest
            self.models[(row['quote_date'], row['root'])] = path

    def read_json(self, path):
        self.inputs[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def table(self, stage, name):
        path = self.folders[stage]/f'{name}.csv'
        actual = sha256(path)
        if self.audits[stage].get('output_sha256', {}).get(f'{name}.csv') != actual:
            raise ValueError(f'Missing or mismatched output pin: {path}')
        self.inputs[str(path)] = actual
        return pd.read_csv(path, dtype={'quote_date': str, 'root': str, 'expire_date': str})

    def verify_unchanged(self):
        for path, digest in self.inputs.items():
            if sha256(path) != digest:
                raise ValueError(f'Input changed during evidence generation: {path}')

    def safe_output(self, output):
        output = Path(output).resolve()
        for folder in (self.index_file.parent, *self.folders.values()):
            if output.is_relative_to(folder) or folder.is_relative_to(output):
                raise ValueError('Use a separate surface-evidence output folder.')
        return output
