"""Distinct smile conventions and a past-only Hull--White empirical correction.

AH PDE delta is copied from the saved panel, never relabelled minimum variance.
Surface movements are assumptions, not a consequence of static calibration.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from hedge_ledger import SelfFinancingHedgeLedger
from pilot_hedge_comparison import PilotHedgeComparison, PilotComparisonSettings, summarize
from surface_evidence import SurfaceEvidenceInputs, black_time_value, invert_time_value, sha256
from andreasen_huge import AndreasenHugeSurface


def black_greeks(spot, strike, maturity, forward, discount, sigma, kind):
    """Fixed IV spot delta; vega is per 1.0 decimal annual volatility."""
    values = np.asarray([spot, strike, maturity, forward, discount, sigma], float)
    if not np.isfinite(values).all() or (values <= 0).any() or kind not in ('call', 'put'):
        raise ValueError('Require positive finite inputs and call/put.')
    std = sigma * np.sqrt(maturity)
    d1 = np.log(forward / strike) / std + std / 2
    phi = np.exp(-d1*d1/2) / np.sqrt(2*np.pi)
    ratio = discount * forward / spot
    price = discount * forward * float(black_time_value(np.log(strike/forward), std*std))
    price += discount * max((forward-strike) * (1 if kind == 'call' else -1), 0)
    return dict(price=price, delta=ratio * (ndtr(d1) - (kind == 'put')),
                vega=discount * forward * phi * np.sqrt(maturity),
                forward_call_delta=ndtr(d1), d1=d1)


def smile_conventions(spot, strike, maturity, forward, discount, sigma, sigma_y, kind):
    """Chain rule for a frozen forward-delta smile, and the HW LV approximation.

    F/S and D held fixed; y=log(K/F). Sticky delta uses unadjusted normalized
    forward call delta N(d1), not spot/premium-adjusted FX delta. Its local
    representation requires decreasing, invertible delta versus strike.
    """
    if not np.isfinite(sigma_y):
        raise ValueError('Nonfinite smile slope.')
    b = black_greeks(spot, strike, maturity, forward, discount, sigma, kind)
    y = np.log(strike/forward)
    phi = np.exp(-b['d1']**2/2) / np.sqrt(2*np.pi)
    q_y = phi/(sigma*np.sqrt(maturity)) * (-1 + (y/sigma + sigma*maturity/2)*sigma_y)
    return dict(surface_iv=sigma, surface_sigma_y=sigma_y, surface_sigma_k=sigma_y/strike,
                surface_vega=b['vega'], surface_sticky_strike_delta=b['delta'],
                surface_sticky_delta_delta=b['delta'] - b['vega']*sigma_y/spot,
                surface_delta_coordinate_slope=q_y,
                surface_hw_lv_delta=b['delta'] + b['vega']*sigma_y/strike)


@dataclass(frozen=True)
class SmileSettings:
    fine_cells: int = 4
    delta_tolerance: float = 5e-4
    slope_absolute_tolerance: float = 0.01
    slope_relative_tolerance: float = 0.05

    def __post_init__(self):
        if (not isinstance(self.fine_cells, int) or self.fine_cells < 2
                or not np.isfinite([self.delta_tolerance, self.slope_absolute_tolerance,
                                    self.slope_relative_tolerance]).all()
                or min(self.delta_tolerance, self.slope_absolute_tolerance,
                       self.slope_relative_tolerance) <= 0):
            raise ValueError('Invalid smile stencil settings.')


class AHSmileHedges:
    """Original AH prices only; finite-width derivatives and honest support flags."""

    def __init__(self, settings=None):
        self.settings = settings or SmileSettings()

    def calculate(self, entries, model, calibration_quotes):
        rows = []
        s = self.settings
        native_h = float(np.diff(np.log(model.grid.z)).mean())
        h = s.fine_cells * native_h
        for maturity, group in entries.groupby('assumed_maturity_years', sort=True):
            maturity = float(maturity)
            f, d = model.forward(maturity), model.discount_factor(maturity)
            state = model.node_state(maturity)
            for row in group.to_dict('records'):
                out = dict(entry_id=row['entry_id'], smile_status='unresolved', smile_message='',
                           smile_fine_log_width=h, smile_coarse_log_width=2*h,
                           sticky_delta_status='unresolved', lv_smile_status='unresolved')
                try:
                    if (not np.isclose(row['underlying_last'], model.spot, rtol=0, atol=1e-8)
                            or not np.isclose(row['forward'], f, rtol=1e-12, atol=1e-8)
                            or not np.isclose(row['discount_factor'], d, rtol=1e-12, atol=1e-12)):
                        raise ValueError('Saved entry and original AH carry disagree.')
                    spot, strike = model.spot, float(row['strike'])
                    y = np.log(strike/f)
                    quotes = calibration_quotes.loc[calibration_quotes.expire_date.eq(row['expire_date'])]
                    observed = np.log(quotes.strike.to_numpy(float)/f)
                    points = y + h*np.array([-2., -1., -.5, 0., .5, 1., 2.])
                    if (not len(observed) or points[0] < observed.min() or points[-1] > observed.max()
                            or points[0] <= np.log(model.grid.z[0])
                            or points[-1] >= np.log(model.grid.z[-1])):
                        out['smile_status'] = 'outside_observed_stencil_support'
                        rows.append(out)
                        continue
                    tv = np.interp(np.exp(points), model.grid.z, state['time_value'])
                    w, status = invert_time_value(points, tv)
                    if not (status == 'ready').all():
                        out['smile_status'] = 'iv_stencil_unresolved'
                        rows.append(out)
                        continue
                    sigmas = np.sqrt(w/maturity)
                    fine = (sigmas[5]-sigmas[1])/(2*h)
                    coarse = (sigmas[6]-sigmas[0])/(4*h)
                    g = smile_conventions(spot, strike, maturity, f, d, sigmas[3], fine, row['kind'])
                    gc = smile_conventions(spot, strike, maturity, f, d, sigmas[3], coarse, row['kind'])
                    # Direct repricing under F'=F*S'/S, with the original normalized
                    # call-price curve frozen. The central two log-spot widths are independent checks.
                    def bump_delta(width):
                        spots = spot*np.exp(np.array([-width, width]))
                        yy = y - np.log(spots/spot)
                        calls = np.interp(np.exp(yy), model.grid.z, state['calls'])
                        prices = d*f*spots/spot*calls
                        if row['kind'] == 'put':
                            prices -= d*(f*spots/spot-strike)
                        return float(np.diff(prices)[0]/np.diff(spots)[0])
                    bumped = bump_delta(h/2)
                    bump_change = abs(bumped-bump_delta(h))
                    chain_error = abs(bumped-g['surface_sticky_delta_delta'])
                    width_delta = abs(g['surface_sticky_delta_delta']-gc['surface_sticky_delta_delta'])
                    density = float(np.interp(strike/f, model.grid.z[1:-1], state['curvature']))
                    out.update(g, smile_sigma_y_coarse=coarse, smile_slope_width_change=abs(fine-coarse),
                        smile_delta_width_change=width_delta, smile_bump_delta=bumped,
                        smile_bump_width_change=bump_change, smile_chain_minus_bump=chain_error,
                        smile_original_density_z=density)
                    slope_ok = abs(fine-coarse) <= s.slope_absolute_tolerance + s.slope_relative_tolerance*abs(fine)
                    if density < -1e-10:
                        out['smile_status'] = 'negative_original_density'
                    elif not slope_ok or max(width_delta, chain_error, bump_change) > s.delta_tolerance:
                        out['smile_status'] = 'stencil_disagreement'
                    else:
                        out['smile_status'] = 'ready'
                        out['sticky_delta_status'] = ('ready' if g['surface_delta_coordinate_slope'] < -1e-8
                                                      else 'delta_coordinate_not_invertible')
                        # HW's ATM-derived LV approximation uses the practitioner
                        # observed-IV Black delta/vega, not a recalibrated PDE delta.
                        if row['black_status'] == 'ready':
                            b = black_greeks(spot, strike, maturity, f, d, float(row['black_iv']), row['kind'])
                            out['lv_smile_delta'] = float(row['black_delta']) + b['vega']*fine/strike
                            out['lv_smile_status'] = 'ready'
                except (ValueError, ArithmeticError) as error:
                    out.update(smile_status='evaluation_failed', smile_message=f'{type(error).__name__}: {error}')
                rows.append(out)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class EmpiricalMVSettings:
    window_dates: int = 60
    minimum_dates: int = 10
    minimum_rows: int = 60
    condition_limit: float = 1e8

    def __post_init__(self):
        if (any(not isinstance(x, int) for x in (self.window_dates, self.minimum_dates, self.minimum_rows))
                or not 3 <= self.minimum_dates <= self.window_dates or self.minimum_rows < 6
                or not np.isfinite(self.condition_limit) or self.condition_limit <= 1):
            raise ValueError('Invalid fixed MV window or rank controls.')


class PastOnlyEmpiricalMV:
    """Hull--White quadratic correction, separate root/kind fits, equal row OLS.

    Minimise SSE of raw observed price changes, without an intercept. This is the
    conventional empirical MV specification, not an exact finite-horizon
    conditional-variance guarantee. Funding/cost P&L is evaluated separately.
    Only endpoints STRICTLY before the prediction timestamp are available.
    """

    def __init__(self, settings=None):
        self.settings = settings or EmpiricalMVSettings()

    @staticmethod
    def feature(row):
        b = black_greeks(float(row['underlying_last']), float(row['strike']),
            float(row['assumed_maturity_years']), float(row['forward']),
            float(row['discount_factor']), float(row['black_iv']), row['kind'])
        if not np.isclose(float(row['black_delta']), b['delta'], rtol=1e-10, atol=1e-8):
            raise ValueError('Saved observed-IV Black delta and carry disagree.')
        delta = float(row['black_delta'])
        scale = b['vega']/(row['underlying_last']*np.sqrt(row['assumed_maturity_years']))
        return scale*np.array([1., delta, delta*delta])

    def predict(self, entries, training):
        s = self.settings
        if training.entry_id.duplicated().any() or entries.entry_id.duplicated().any():
            raise ValueError('Duplicate training or prediction entry IDs.')
        data = training.copy(deep=True)
        for column in ('quote_timestamp_utc', 'end_timestamp'):
            for value in data[column].dropna():
                if pd.Timestamp(value).tzinfo is None:
                    raise ValueError('Training timestamps must be timezone-aware.')
        data['_entry'] = pd.to_datetime(data.quote_timestamp_utc, utc=True)
        data['_end'] = pd.to_datetime(data.end_timestamp, utc=True)
        predictions, fits, membership = [], [], []
        for (stamp, root, kind), group in entries.groupby(['quote_timestamp_utc', 'root', 'kind'], sort=True):
            now = pd.Timestamp(stamp)
            if now.tzinfo is None:
                raise ValueError('Prediction timestamp must be timezone-aware.')
            available = data.loc[data.root.eq(root) & data.kind.eq(kind)
                & data.end_status.eq('matched') & data.black_status.eq('ready')
                & data._entry.lt(now) & data._end.lt(now)].copy()
            if len(available) and not (available._entry < available._end).all():
                raise ValueError('Invalid training endpoint order.')
            dates = sorted(available.quote_date.unique())[-s.window_dates:]
            available = available.loc[available.quote_date.isin(dates)].sort_values(['quote_date', 'entry_id'])
            fit_id = hashlib.sha256(f'{now.isoformat()}|{root}|{kind}'.encode()).hexdigest()[:20]
            fit = dict(fit_id=fit_id, prediction_timestamp=now.isoformat(), quote_date=group.quote_date.iloc[0],
                root=root, kind=kind, status='warmup', available_dates=len(dates), training_rows=len(available),
                training_first_date=dates[0] if dates else '', training_last_date=dates[-1] if dates else '',
                maximum_training_endpoint=available._end.max().isoformat() if len(available) else '',
                message='')
            beta = None
            if len(dates) >= s.minimum_dates and len(available) >= s.minimum_rows:
                features = np.vstack([self.feature(row) for row in available.to_dict('records')])
                ds = (available.end_spot-available.underlying_last).to_numpy(float)
                target = (available.end_mid-available.mid-available.black_delta*ds).to_numpy(float)
                design = features*ds[:, None]
                if not np.isfinite(np.r_[design.ravel(), target]).all():
                    raise ValueError('Invalid available training observations.')
                # Scale columns solely for numerical conditioning; restore original coefficients.
                norms = np.linalg.norm(design, axis=0)
                if (norms <= 0).any():
                    fit['status'] = 'rank_deficient'
                else:
                    scaled = design/norms
                    singular = np.linalg.svd(scaled, compute_uv=False)
                    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else np.inf
                    rank = int(np.linalg.matrix_rank(scaled))
                    fit.update(design_rank=rank, scaled_condition_number=condition)
                    if rank < 3:
                        fit['status'] = 'rank_deficient'
                    elif condition > s.condition_limit:
                        fit['status'] = 'ill_conditioned'
                    else:
                        beta = np.linalg.lstsq(scaled, target, rcond=None)[0]/norms
                        error = target-design@beta
                        fit.update(status='ready', a=float(beta[0]), b=float(beta[1]), c=float(beta[2]),
                            training_black_sse=float(target@target), training_mv_sse=float(error@error),
                            training_mv_mean_error=float(error.mean()), training_mv_variance=float(error.var()),
                            training_delta_min=float(available.black_delta.min()),
                            training_delta_max=float(available.black_delta.max()))
            fits.append(fit)
            for row in available.to_dict('records'):
                membership.append(dict(fit_id=fit_id, training_entry_id=row['entry_id'],
                    training_date=row['quote_date'], training_endpoint=row['_end'].isoformat()))
            for row in group.to_dict('records'):
                out = dict(entry_id=row['entry_id'], empirical_mv_status=fit['status'], empirical_mv_fit_id=fit_id)
                if row['black_status'] != 'ready':
                    out['empirical_mv_status'] = 'black_not_ready'
                elif beta is not None:
                    correction = float(self.feature(row)@beta)
                    out.update(empirical_mv_correction=correction,
                        empirical_mv_delta=float(row['black_delta'])+correction,
                        empirical_mv_delta_extrapolated=not fit['training_delta_min'] <= row['black_delta'] <= fit['training_delta_max'])
                predictions.append(out)
        return (pd.DataFrame(predictions), pd.DataFrame(fits),
                pd.DataFrame(membership, columns=['fit_id', 'training_entry_id', 'training_date', 'training_endpoint']))


STRATEGIES = {
    'black': ('black_status', 'black_delta'),
    'ah_pde': ('ah_status', 'ah_delta'),
    'surface_sticky_strike': ('smile_status', 'surface_sticky_strike_delta'),
    'surface_sticky_delta': ('sticky_delta_status', 'surface_sticky_delta_delta'),
    'lv_smile': ('lv_smile_status', 'lv_smile_delta'),
    'empirical_mv': ('empirical_mv_status', 'empirical_mv_delta'),
}


def compare_hedges(panel, settings=None):
    """Independent coverage per strategy; paired Gain always uses common entries."""
    comparator = PilotHedgeComparison(settings or PilotComparisonSettings())
    panel = comparator.checked(panel)
    # Validate every matched mark path even when the old AH/Black pair failed.
    paths = panel.copy()
    paths['black_status'] = paths['ah_status'] = 'ready'
    paths['black_delta'] = paths['ah_delta'] = 0.
    comparator.checked(paths)
    trials, coverage = [], []
    for row in panel.to_dict('records'):
        meta = {name: row[name] for name in comparator.META}
        for scenario, ledger_settings in comparator.cases.items():
            for strategy, (status_column, delta_column) in STRATEGIES.items():
                status = row['end_status'] if row['end_status'] != 'matched' else row.get(status_column, 'unresolved')
                if status == 'ready':
                    delta = float(row[delta_column])
                    if not np.isfinite(delta):
                        raise ValueError('Nonfinite ready strategy delta.')
                    result = SelfFinancingHedgeLedger(ledger_settings).run(comparator.observations(row, delta))
                    direct = comparator.direct_price(row, delta, ledger_settings)
                    if abs(result.summary['net_pnl']-direct) > 1e-8:
                        raise ArithmeticError('Funded ledger and direct cash value disagree.')
                    raw_error = row['end_mid']-row['mid']-delta*(row['end_spot']-row['underlying_last'])
                    trials.append(dict(**meta, scenario=scenario, strategy=strategy, entry_delta=delta,
                        black_entry_delta=row['black_delta'], raw_mark_error=raw_error,
                        closed_form_error=result.summary['net_pnl']-direct, **result.summary))
                    status = 'compared'
                coverage.append(dict(**meta, scenario=scenario, strategy=strategy, status=status))
    trial_columns = list(comparator.META)+['scenario', 'strategy', 'entry_delta', 'black_entry_delta',
        'raw_mark_error', 'closed_form_error', 'net_pnl', 'direct_costs', 'total_interest',
        'total_hedge_turnover', 'max_reconciliation']
    trials = pd.DataFrame(trials) if trials else pd.DataFrame(columns=trial_columns)
    gains = paired_gains(trials)
    return trials, pd.DataFrame(coverage), gains


def paired_gains(trials):
    """1-SSE(strategy)/SSE(Black), pooled by rows, without demeaning."""
    results = []
    for (scenario, strategy), candidate in trials.groupby(['scenario', 'strategy'], sort=True):
        if strategy == 'black':
            continue
        black = trials.loc[trials.scenario.eq(scenario) & trials.strategy.eq('black'),
            ['entry_id', 'net_pnl', 'raw_mark_error']]
        paired = candidate.merge(black, on='entry_id', suffixes=('', '_black'), validate='one_to_one')
        if paired.empty:
            continue
        paired['maturity_bucket'] = pd.cut(paired.calendar_days, [0, 27, 39, np.inf],
            labels=['<=27d', '28-39d', '>=40d']).astype(str)
        paired['moneyness_bucket'] = np.select([paired.entry_spot_y < -.01, paired.entry_spot_y > .01],
            ['K_below_spot', 'K_above_spot'], default='near_spot')
        paired['delta_bucket'] = pd.cut(abs(paired.black_entry_delta), [0, .25, .5, .75, np.inf],
            labels=['0-.25', '.25-.50', '.50-.75', '>.75'], include_lowest=True).astype(str)
        groups = [('overall', 'all', paired)]
        for dimension in ('kind', 'maturity_bucket', 'moneyness_bucket', 'delta_bucket', 'quote_date'):
            groups += [(dimension, str(name), group) for name, group in paired.groupby(dimension, sort=True)]
        for dimension, name, group in groups:
            denominator = float(np.sum(group.net_pnl_black**2))
            raw_denominator = float(np.sum(group.raw_mark_error_black**2))
            results.append(dict(scenario=scenario, strategy=strategy, dimension=dimension, group=name,
                comparisons=len(group), entry_dates=group.quote_date.nunique(),
                black_sse=denominator, strategy_sse=float(np.sum(group.net_pnl**2)),
                gain=1-float(np.sum(group.net_pnl**2))/denominator if denominator > 1e-20 else np.nan,
                gain_status='ready' if denominator > 1e-20 else 'zero_black_sse',
                raw_black_sse=raw_denominator, raw_strategy_sse=float(np.sum(group.raw_mark_error**2)),
                raw_gain=1-float(np.sum(group.raw_mark_error**2))/raw_denominator if raw_denominator > 1e-20 else np.nan,
                mean_net_pnl=float(group.net_pnl.mean()), rms_net_pnl=float(np.sqrt(np.mean(group.net_pnl**2))),
                black_rms_net_pnl=float(np.sqrt(np.mean(group.net_pnl_black**2))),
                mae_improvement=float(np.mean(abs(group.net_pnl_black)-abs(group.net_pnl))),
                fraction_lower_abs_error=float(np.mean(abs(group.net_pnl) < abs(group.net_pnl_black)))))
    columns = ['scenario', 'strategy', 'dimension', 'group', 'comparisons', 'entry_dates',
        'black_sse', 'strategy_sse', 'gain', 'gain_status', 'raw_black_sse', 'raw_strategy_sse',
        'raw_gain', 'mean_net_pnl', 'rms_net_pnl', 'black_rms_net_pnl', 'mae_improvement', 'fraction_lower_abs_error']
    return pd.DataFrame(results, columns=columns)


class OptimalHedgeInputs(SurfaceEvidenceInputs):
    """Verify the original calibration, carry and panel chain without running it."""

    def __init__(self, index_file, month, repository=None):
        super().__init__(index_file, month, repository)
        index = json.loads(self.index_file.read_text())
        path = Path(index['months'][month]['panel'])
        folder = (path if path.is_absolute() else self.repository/path).resolve()
        self.folders['panel'] = folder
        self.audits['panel'] = self.read_json(folder/'audit.json')
        for stage in ('calibration', 'carry', 'panel'):
            status = self.audits[stage].get('status', 'completed')
            if status not in ('completed', 'completed_with_failures'):
                raise ValueError(f'Upstream {stage} is not complete.')
        self.panel = self.table('panel', 'delta_panel')
        pinned = self.audits['panel'].get('input_sha256', {}).values()
        for name in ('model_manifest.csv', 'audit.json'):
            if sha256(self.folders['calibration']/name) not in pinned:
                raise ValueError('Panel does not pin this calibration.')
        if sha256(self.folders['carry']/'primary_carry.csv') not in pinned:
            raise ValueError('Panel does not pin this carry.')
        required = {'entry_id', 'black_iv', 'black_delta', 'assumed_maturity_years',
                    'forward', 'discount_factor', 'quote_timestamp_utc', 'assumed_fixing_utc'}
        if self.panel.empty or not required <= set(self.panel) or self.panel.entry_id.duplicated().any():
            raise ValueError('Invalid or duplicate saved entries.')
        if not self.panel.quote_date.str.startswith(month+'-').all():
            raise ValueError('Saved entry dates disagree with indexed month.')
        PilotHedgeComparison().checked(self.panel)
        for row in self.panel.to_dict('records'):
            first, fixing = pd.Timestamp(row['quote_timestamp_utc']), pd.Timestamp(row['assumed_fixing_utc'])
            if (first.tzinfo is None or fixing.tzinfo is None
                    or not np.isclose((fixing-first).total_seconds()/(365*86400),
                                      row['assumed_maturity_years'], rtol=1e-12, atol=1e-12)):
                raise ValueError('Saved maturity and fixing timestamps disagree.')


def analytical_controls():
    """Known smiles with both signs; these controls are not historical data."""
    rows = []
    for slope in (0., -.3, .2):
        for kind in ('call', 'put'):
            spot, strike, t, f, d, sigma = 100., 101., .2, 102., .99, .25
            y = np.log(strike/f)
            values = smile_conventions(spot, strike, t, f, d, sigma, slope, kind)
            errors = []
            for width in (1e-3, 5e-4, 1e-4):
                spots = spot*np.exp(np.array([-width, width]))
                prices = [black_greeks(x, strike, t, f*x/spot, d,
                    sigma+slope*(np.log(strike/(f*x/spot))-y), kind)['price'] for x in spots]
                errors.append(abs((prices[1]-prices[0])/(spots[1]-spots[0])-values['surface_sticky_delta_delta']))
            rows.append(dict(control='analytic_linear_iv', kind=kind, sigma_y=slope,
                sticky_delta_delta=values['surface_sticky_delta_delta'],
                sticky_strike_delta=values['surface_sticky_strike_delta'],
                hw_lv_smile_delta=values['surface_hw_lv_delta'],
                bump_error_1e_3=errors[0], bump_error_5e_4=errors[1], bump_error_1e_4=errors[2]))
    result = pd.DataFrame(rows)
    if result.bump_error_1e_4.max() > 1e-6:
        raise ArithmeticError('Analytical smile control failed.')
    return result


def plot_optimal_hedges(output, panel, fits, gains):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, (status, delta) in STRATEGIES.items():
        if strategy == 'black' or delta not in panel:
            continue
        ready = panel[status].eq('ready') & panel.black_status.eq('ready')
        daily = (panel.loc[ready].assign(difference=lambda x: x[delta]-x.black_delta)
                 .groupby('quote_date').difference.mean())
        ax.plot(daily.index, daily.values, marker='.', label=strategy)
    ax.axhline(0, color='black', lw=.8)
    ax.set(ylabel='Mean delta minus observed-IV Black', title='Declared smile movements and past-only empirical correction')
    ax.tick_params(axis='x', rotation=60)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'01_delta_corrections.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for (root, kind), group in fits.loc[fits.status.eq('ready')].groupby(['root', 'kind']):
        for name in ('a', 'b', 'c'):
            ax.plot(group.quote_date, group[name], label=f'{root}/{kind}: {name}')
    if not fits.status.eq('ready').any():
        ax.text(.5, .5, 'No fit passed the fixed warm-up and rank controls', ha='center', transform=ax.transAxes)
    ax.set(title='Rolling quadratic coefficients; separate call/put fits', ylabel='Coefficient')
    ax.tick_params(axis='x', rotation=60)
    if fits.status.eq('ready').any(): ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'02_mv_coefficients.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    overall = gains.loc[gains.scenario.eq('mid') & gains.dimension.eq('overall')]
    ax.bar(overall.strategy, 100*overall.gain)
    ax.axhline(0, color='black', lw=.8)
    ax.set(ylabel='Gain (%) on each strategy\'s common Black entries', title='Midpoint funded hedge error; development evidence')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(output/'03_midpoint_gain.png', dpi=160); plt.close(fig)


def write_notebook(output):
    """Saved-result inspection only; no fitting or model execution in the notebook."""
    cells = []
    def add(kind, source):
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '# Smile conventions and empirical MV correction\n\nDevelopment research. AH PDE delta is not labelled MV. Read the assumptions in audit.json. The empirical strategy is the published quadratic SSE specification, not a guarantee of minimum future conditional variance.\n')
    add('code', "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nbase = Path.cwd()\nif not (base/'audit.json').exists():\n    base = Path("+repr(str(Path(output).resolve()))+")\naudit = json.loads((base/'audit.json').read_text())\nprint(audit['empirical_estimation'])\nprint('Measured seconds:', audit['elapsed_wall_seconds'])\n")
    add('code', "panel = pd.read_csv(base/'strategy_panel.csv')\nfits = pd.read_csv(base/'mv_fits.csv')\ngains = pd.read_csv(base/'gain_summary.csv')\ncoverage = pd.read_csv(base/'coverage.csv')\ndisplay(coverage.groupby(['scenario','strategy','status']).size().rename('entries'))\ndisplay(fits[['quote_date','root','kind','status','available_dates','training_rows','maximum_training_endpoint']])\n")
    add('markdown', 'Every Gain uses the same entries for that candidate and Black. MV warm-up failures remain in coverage. Sample sizes can differ across candidates; use common_strategy_gain.csv for the intersection of all ready strategies. Funding and transaction costs are separate from the raw-mark SSE training objective.\n')
    add('code', "display(gains.loc[gains.dimension.eq('overall')])\ndisplay(pd.read_csv(base/'common_strategy_gain.csv'))\ndisplay(pd.read_csv(base/'analytical_controls.csv'))\nfor name in ['01_delta_corrections.png','02_mv_coefficients.png','03_midpoint_gain.png']:\n    display(Image(filename=str(base/'plots'/name)))\n")
    add('code', "available_dates = sorted(panel.quote_date.unique())\nprint(available_dates)\nselected_date = available_dates[0]\ncolumns = ['quote_date','expire_date','strike','kind','black_delta','ah_delta','surface_sticky_strike_delta','surface_sticky_delta_delta','lv_smile_delta','empirical_mv_delta','smile_status','empirical_mv_status']\ndisplay(panel.loc[panel.quote_date.eq(selected_date), [c for c in columns if c in panel]])\n")
    Path(output, 'optimal_hedging.ipynb').write_text(json.dumps(dict(cells=cells,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5), indent=2)+'\n')
