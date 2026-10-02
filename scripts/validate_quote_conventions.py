"""Reproduce the price-convention audit without changing existing pipeline data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lv_project.preprocessing import parse_raw_option_file, build_clean_call_dataset
from lv_project.surface import prepare_surface_points_for_date
from lv_project.quote_conventions import (
    prepare_price_pairs, estimate_parity, forward_option_price, invert_forward_iv,
)


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    path.write_text(json.dumps(clean(value), indent=2)+'\n')


def preservation_snapshot(root, output):
    """Hash every existing data/output file, including earlier validation runs."""
    files = sorted({p for folder in ['data', 'outputs'] for p in (root/folder).rglob('*')
                    if p.is_file() and not p.resolve().is_relative_to(output.resolve())})
    result = []
    for path in files:
        stat = path.stat()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        result.append(dict(path=str(path.relative_to(root)), bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns, sha256=digest))
    return result


def verify_preservation(root, before, output):
    changed = []
    for row in before:
        p = root/row['path']
        if not p.exists():
            changed.append(row['path'])
            continue
        stat = p.stat()
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if (stat.st_size, stat.st_mtime_ns, digest) != (row['bytes'], row['mtime_ns'], row['sha256']):
            changed.append(row['path'])
    result = dict(files_checked=len(before), bytes_checked=sum(r['bytes'] for r in before),
                  changed_files=changed)
    write_json(output/'preservation_after.json', result)
    if changed:
        raise RuntimeError(f'Preservation failed: {changed}')
    return result


def metadata_table(q):
    rows = []
    for group, g in q.groupby('pair_group'):
        exp = pd.Timestamp(g.expire_date.iloc[0])
        third_friday = exp.weekday() == 4 and 15 <= exp.day <= 21
        utc = pd.to_datetime(g.expire_unix.iloc[0], unit='s', utc=True)
        rows.append(dict(pair_group=group, expiry=str(exp.date()), raw_rows=len(g),
                         quote_unixtime=g.quote_unixtime.iloc[0], expire_unix=g.expire_unix.iloc[0],
                         expiry_utc=utc.isoformat(), expiry_new_york=utc.tz_convert('America/New_York').isoformat(),
                         raw_dte=g.dte.iloc[0], elapsed_days=g.elapsed_days.iloc[0],
                         calendar_days=(exp-pd.Timestamp(g.quote_date.iloc[0])).days,
                         maturity=g.maturity.iloc[0], spot_min=g.underlying_last.min(),
                         spot_max=g.underlying_last.max(), matched_pairs=int(g.pair_usable.sum()),
                         call_price_usable=int(g.c_price_usable.sum()), put_price_usable=int(g.p_price_usable.sum()),
                         call_missing_vendor_iv=int(g.c_iv.isna().sum()), put_missing_vendor_iv=int(g.p_iv.isna().sum()),
                         possible_am_pm_ambiguity=third_friday, settlement_identified=False))
    return pd.DataFrame(rows)


def fit_conventions(q, metadata, min_days=7, max_days=365):
    estimates, sensitivity, heldout = [], [], []
    selections = [('primary', .10, .35, None), ('narrow_band', .05, .35, None),
                  ('wide_band', .20, .35, None), ('full_chain', np.inf, .35, None),
                  ('tight_spreads', .10, .10, None), ('medium_spreads', .10, .20, None),
                  ('positive_sizes', .10, .35, 'size'), ('positive_volume', .10, .35, 'volume')]
    q['calibration_selected'] = q.pair_usable & q.log_strike_spot.abs().le(.10)
    q['parity_fit'] = np.nan
    q['parity_residual'] = np.nan
    q['parity_interval_excess'] = np.nan
    for group, g in q.groupby('pair_group'):
        meta = metadata.loc[metadata.pair_group.eq(group)].iloc[0].to_dict()
        if not g.elapsed_days.between(min_days, max_days).all():
            estimates.append(dict(**meta, status='outside_maturity_window', reasons='outside_maturity_window',
                                  pair_count=0, forward=np.nan, discount=np.nan))
            continue
        base = g.loc[g.pair_usable]
        primary = base.loc[base.log_strike_spot.abs().le(.10)]
        fit = estimate_parity(primary)
        estimates.append({**meta, **fit})
        if np.isfinite(fit['forward']) and fit['discount'] > 0:
            pred = fit['discount']*(fit['forward']-g.strike)
            q.loc[g.index, 'parity_fit'] = pred
            q.loc[g.index, 'parity_residual'] = g.parity_mid-pred
            q.loc[g.index, 'parity_interval_excess'] = np.maximum(np.maximum(g.parity_lower-pred, pred-g.parity_upper), 0)
        for name, band, spread, extra in selections:
            sel = base.log_strike_spot.abs().le(band)
            sel &= base.c_relative_spread.le(spread) & base.p_relative_spread.le(spread)
            extra_available = True
            if extra:
                for side in ['c', 'p']:
                    columns = [side+'_bid_size', side+'_ask_size'] if extra == 'size' else [side+'_volume']
                    for c in columns:
                        if c not in base:
                            extra_available = False
                        else:
                            sel &= base[c].gt(0)
            sf = estimate_parity(base.loc[sel]) if extra_available else dict(
                status='not_run', reasons='missing_selection_metadata', pair_count=0, forward=np.nan, discount=np.nan)
            sensitivity.append(dict(pair_group=group, expiry=meta['expiry'], selection=name, **sf))
        # Interleaved strike folds avoid taking the same midpoint as target and input.
        ordered = primary.sort_values('strike')
        for fold in [0, 1]:
            training = ordered.iloc[fold::2]
            test = ordered.iloc[1-fold::2]
            hf = estimate_parity(training)
            row = dict(pair_group=group, expiry=meta['expiry'], fold=fold, training_pairs=len(training),
                       heldout_pairs=len(test), fit_status=hf['status'], forward=hf['forward'], discount=hf['discount'])
            if hf['discount'] > 0 and np.isfinite(hf['forward']) and len(test):
                pred = hf['discount']*(hf['forward']-test.strike)
                error = test.parity_mid-pred
                row.update(residual_rmse=float(np.sqrt(np.mean(error**2))),
                           interval_coverage=float(np.mean((pred >= test.parity_lower-1e-7)&(pred <= test.parity_upper+1e-7))))
            heldout.append(row)
    return pd.DataFrame(estimates), pd.DataFrame(sensitivity), pd.DataFrame(heldout)


def option_rows(q, fits, baseline_ids):
    """Keep individual usable sides, including sides not in a liquid pair."""
    rows = []
    fits = fits.set_index('pair_group')
    for side, option in [('c', 'call'), ('p', 'put')]:
        for row in q.loc[q[side+'_price_usable']].itertuples():
            fit = fits.loc[row.pair_group]
            vendor_iv = getattr(row, side+'_iv')
            bid, ask = getattr(row, side+'_bid'), getattr(row, side+'_ask')
            mid = (bid+ask)/2
            f, d, t = fit.forward, fit.discount, row.maturity
            record = dict(quote_id=row.quote_id, pair_group=row.pair_group, expiry=fit.expiry,
                          option=option, strike=row.strike, spot=row.underlying_last,
                          quote_unixtime=row.quote_unixtime, expire_unix=row.expire_unix,
                          maturity=t, raw_maturity=row.dte/365, forward=f, discount=d,
                          parity_status=fit.status, provisional_settlement=True,
                          possible_am_pm_ambiguity=fit.possible_am_pm_ambiguity,
                          bid=bid, ask=ask, mid=mid, vendor_iv=vendor_iv,
                          otm_or_atm=(row.strike >= f if option == 'call' else row.strike < f),
                          liquid_pair=row.pair_usable, calibration_selected=row.calibration_selected,
                          original_surface_input=option == 'call' and row.quote_id in baseline_ids)
            available = np.isfinite(vendor_iv) and vendor_iv > 0
            record['vendor_iv_usable'] = available
            if available:
                record['vendor_zero_carry_price'] = forward_option_price(row.underlying_last, row.strike, 1, row.dte/365, vendor_iv, option)
            if fit.status == 'identified':
                if available:
                    record['vendor_parity_price'] = forward_option_price(f, row.strike, d, t, vendor_iv, option)
                    record['vendor_parity_raw_dte_price'] = forward_option_price(f, row.strike, d, row.dte/365, vendor_iv, option)
                    calendar_t = (pd.Timestamp(row.expire_date)-pd.Timestamp(row.quote_date)).days/365
                    record['vendor_parity_calendar_price'] = forward_option_price(f, row.strike, d, calendar_t, vendor_iv, option)
                for target, price in [('bid', bid), ('mid', mid), ('ask', ask)]:
                    result = invert_forward_iv(price, f, row.strike, d, t, option)
                    record[target+'_iv'] = result.volatility
                    record[target+'_iv_status'] = result.status
                    record[target+'_repricing_error'] = result.repricing_error
                    if target == 'mid':
                        record.update(lower_price_bound=result.lower_bound, upper_price_bound=result.upper_bound,
                                      mid_vega=result.vega)
                if fit.possible_am_pm_ambiguity:
                    # Hypothetical 09:30 vs 16:00 convention only; SOQ has no fixed clock time.
                    am_t = t-6.5/(24*365)
                    am = invert_forward_iv(mid, f, row.strike, d, am_t, option)
                    record['hypothetical_am_mid_iv'] = am.volatility
                    record['hypothetical_am_iv_status'] = am.status
            else:
                for target in ['bid', 'mid', 'ask']:
                    record[target+'_iv_status'] = 'weak_parity_estimate'
            rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No usable price sides remain after metadata, maturity and liquidity filters')
    for convention, col in [('zero_carry', 'vendor_zero_carry_price'), ('parity', 'vendor_parity_price')]:
        if col not in result:
            result[col] = np.nan
        result[convention+'_mid_error'] = result[col]-result.mid
        result[convention+'_inside_bid_ask'] = (result[col] >= result.bid-1e-8)&(result[col] <= result.ask+1e-8)
        result[convention+'_bid_ask_excess'] = np.maximum(np.maximum(result.bid-result[col], result[col]-result.ask), 0)
    return result


def price_metrics(group):
    # Paired convention comparisons use identical rows in both denominators.
    g = group.loc[group.vendor_zero_carry_price.notna() & group.vendor_parity_price.notna()]
    result = dict(options=len(group), vendor_iv_missing_or_invalid=int((~group.vendor_iv_usable).sum()),
                  comparable_vendor_quotes=len(g))
    for convention in ['zero_carry', 'parity']:
        err = g[convention+'_mid_error'].to_numpy()
        result.update({convention+'_coverage': float(g[convention+'_inside_bid_ask'].mean()),
                       convention+'_median_abs_error': float(np.median(np.abs(err))) if len(err) else np.nan,
                       convention+'_rmse': float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
                       convention+'_max_abs_error': float(np.max(np.abs(err))) if len(err) else np.nan})
    result['mid_inversion_status'] = group.mid_iv_status.value_counts().to_dict()
    ok = group.loc[group.mid_iv_status.eq('ok')]
    result['roundtrip_max_abs_error'] = float(ok.mid_repricing_error.abs().max()) if len(ok) else np.nan
    return result


def make_plots(out, fits, sens, q, options):
    os.environ.setdefault('MPLCONFIGDIR', str(out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = fits.loc[fits.status.eq('identified')].sort_values('elapsed_days')
    if g.empty:
        write_json(out/'plot_status.json', dict(status='NOT_RUN', reason='No identified parity estimates'))
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, col, lower, upper, label in [
        (axes[0], 'forward', 'forward_lower', 'forward_upper', 'Forward (index points)'),
        (axes[1], 'discount', 'discount_lower', 'discount_upper', 'Discount factor')]:
        ax.fill_between(g.elapsed_days, g[lower], g[upper], alpha=.25, label='Bid/ask feasible range')
        ax.plot(g.elapsed_days, g[col], '.-', label='Weighted mid fit')
        ambiguous = g.loc[g.possible_am_pm_ambiguity]
        ax.scatter(ambiguous.elapsed_days, ambiguous[col], marker='x', color='red', label='AM/PM unidentified')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    axes[1].set_xlabel('Elapsed days, ACT/365F; vendor expiry timestamp')
    fig.tight_layout(); fig.savefig(out/'forward_discount.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for option, ax in zip(['call', 'put'], axes):
        x = options.loc[options.option.eq(option)]
        for convention in ['zero_carry', 'parity']:
            coverage = x.groupby('expiry').apply(lambda v: price_metrics(v)[convention+'_coverage'], include_groups=False)
            ax.plot(np.arange(len(coverage)), coverage*100, '.-', label=convention)
        ax.set(title=option.title()+' vendor-IV bid/ask coverage', xlabel='Expiry ordered from short to long', ylabel='Coverage (%)')
        ax.set_ylim(-2, 102); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'vendor_iv_coverage.png', dpi=160); plt.close(fig)
    worst = g.sort_values('residual_rmse', ascending=False).iloc[0]
    x = q.loc[q.pair_group.eq(worst.pair_group) & q.pair_usable].sort_values('strike')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x.strike, -x.parity_half_width, x.parity_half_width, alpha=.25, label='Observed parity half spread')
    ax.plot(x.strike, x.parity_residual, '.', label='Mid residual')
    ax.axvline(worst.spot*np.exp(-.10), color='grey', linestyle=':')
    ax.axvline(worst.spot*np.exp(.10), color='grey', linestyle=':')
    ax.set(title='Parity residuals: '+worst.expiry, xlabel='Strike', ylabel='Cmid − Pmid − D(F − K)')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'parity_residuals.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for exp in [g.expiry.iloc[0], g.expiry.iloc[-1]]:
        x = options.loc[options.expiry.eq(exp)&options.option.eq('call')&options.mid_iv_status.eq('ok')].sort_values('strike')
        axes[0].plot(x.strike, x.vendor_iv*100, '--', label=exp+' vendor')
        axes[0].plot(x.strike, x.mid_iv*100, '-', label=exp+' price inversion')
    axes[0].set(xlabel='Strike', ylabel='Call IV (%)', ylim=(0, 100), title='Raw quote IV; no surface fit')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    p = sens.merge(g[['pair_group', 'forward', 'discount', 'elapsed_days']], on='pair_group', suffixes=('', '_primary'))
    for name, x in p.groupby('selection'):
        axes[1].plot(x.elapsed_days, (x.discount-x.discount_primary)*1e4, '.', label=name)
    axes[1].set(xlabel='Elapsed days', ylabel='Δ discount factor × 10,000', title='Quote-selection sensitivity')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'iv_and_selection_sensitivity.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quote-date', default='2023-12-22')
    parser.add_argument('--raw-path', type=Path)
    parser.add_argument('--min-days', type=float, default=7)
    parser.add_argument('--max-days', type=float, default=365)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/diagnostics/validation/conventions')
    args = parser.parse_args()
    if not 0 < args.min_days <= args.max_days:
        parser.error('Require 0 < min-days <= max-days; expired quotes cannot be inverted')
    date = pd.Timestamp(args.quote_date)
    raw_path = args.raw_path or ROOT/f'data/raw/spx_eod_{date:%Y%m}.txt'
    if not raw_path.exists():
        parser.error(f'Required raw data unavailable: {raw_path.resolve()}')
    out = args.output_dir.resolve()
    if out.is_relative_to((ROOT/'data').resolve()) or not out.is_relative_to((ROOT/'outputs/diagnostics/validation').resolve()):
        parser.error('Output must be a dedicated directory under outputs/diagnostics/validation/')
    if out.exists() and any(out.iterdir()):
        parser.error(f'Output directory is nonempty; choose a fresh --output-dir: {out}')
    out.mkdir(parents=True, exist_ok=True)
    print('Hashing existing data and baseline outputs...', flush=True)
    before = preservation_snapshot(ROOT, out)
    write_json(out/'preservation_before.json', before)
    all_raw = parse_raw_option_file(raw_path)
    raw = all_raw.loc[all_raw.quote_date.eq(date)].copy()
    if raw.empty:
        raise ValueError(f'No quotes for {date.date()} in {raw_path}')
    q = prepare_price_pairs(raw, min_days=args.min_days, max_days=args.max_days)
    metadata = metadata_table(q)
    fits, sensitivity, heldout = fit_conventions(q, metadata, args.min_days, args.max_days)
    # Reproduce the existing raw->clean->input selection in memory; never write it.
    baseline, _ = prepare_surface_points_for_date(build_clean_call_dataset(raw), args.quote_date,
                                                  input_y_min=-.25, input_y_max=.20)
    baseline_keys = set(zip(baseline.expiry, baseline.strike))
    baseline_ids = set(q.loc[[key in baseline_keys for key in zip(q.expire_date, q.strike)], 'quote_id'])
    options = option_rows(q, fits, baseline_ids)
    for name, table in [('quote_audit', q), ('matched_pairs', q.loc[q.pair_usable]),
                        ('expiry_metadata', metadata), ('parity_estimates', fits),
                        ('quote_selection_sensitivity', sensitivity), ('parity_heldout', heldout),
                        ('repriced_and_inverted_quotes', options)]:
        table.to_csv(out/(name+'.csv'), index=False)
    metrics = []
    for (expiry, option), g in options.groupby(['expiry', 'option']):
        row = price_metrics(g)
        row.pop('mid_inversion_status')
        metrics.append(dict(expiry=expiry, option=option, **row))
    pd.DataFrame(metrics).to_csv(out/'repricing_by_expiry.csv', index=False)
    status_rows = []
    for (expiry, option), g in options.groupby(['expiry', 'option']):
        for target in ['bid', 'mid', 'ask']:
            for status, count in g[target+'_iv_status'].value_counts().items():
                status_rows.append(dict(expiry=expiry, option=option, target=target, status=status, count=count))
    pd.DataFrame(status_rows).to_csv(out/'inversion_status_by_expiry.csv', index=False)
    baseline_options = options.loc[options.original_surface_input]
    baseline_metrics = []
    for expiry, g in baseline_options.groupby('expiry'):
        row = price_metrics(g)
        row.pop('mid_inversion_status')
        baseline_metrics.append(dict(expiry=expiry, **row))
    pd.DataFrame(baseline_metrics).to_csv(out/'baseline_call_repricing_by_expiry.csv', index=False)
    failed = options.loc[options.mid_iv_status.ne('ok')].copy()
    if 'lower_price_bound' in failed:
        failed['lower_shortfall'] = failed.lower_price_bound-failed.mid
    failed.to_csv(out/'mid_inversion_failures.csv', index=False)
    comparison = []
    for sample, g in [('original_surface_calls', baseline_options),
                      ('all_usable_calls', options.loc[options.option.eq('call')]),
                      ('all_usable_puts', options.loc[options.option.eq('put')])]:
        row = price_metrics(g)
        row.pop('mid_inversion_status')
        comparison.append(dict(sample=sample, **row))
    pd.DataFrame(comparison).to_csv(out/'convention_comparison.csv', index=False)
    summary = dict(quote_date=args.quote_date, raw_path=str(raw_path.resolve()),
                   maturity_window_days=[args.min_days, args.max_days],
                   checkpoint=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   argv=sys.argv, python=platform.python_version(), numpy=np.__version__,
                   pandas=pd.__version__, scipy=scipy.__version__,
                   monthly_rows=len(all_raw), raw_rows=len(raw), matched_pairs=int(q.pair_usable.sum()),
                   matched_missing_call_iv=int(q.loc[q.pair_usable, 'c_iv'].isna().sum()),
                   matched_missing_put_iv=int(q.loc[q.pair_usable, 'p_iv'].isna().sum()),
                   baseline_surface_rows=len(baseline), baseline_option_rows=len(baseline_options),
                   calibration_pairs=int(q.calibration_selected.sum()),
                   parity_status=fits.status.value_counts().to_dict(),
                   all_options=price_metrics(options), baseline_calls=price_metrics(baseline_options),
                   calls=price_metrics(options.loc[options.option.eq('call')]),
                   puts=price_metrics(options.loc[options.option.eq('put')]),
                   raw_dte_max_error_days=float((q.dte-q.elapsed_days).abs().max()),
                   selected_bound_status=options.mid_iv_status.value_counts().to_dict())
    summary['otm_options'] = price_metrics(options.loc[options.otm_or_atm])
    make_plots(out, fits, sensitivity, q, options)
    summary['preservation'] = verify_preservation(ROOT, before, out)
    write_json(out/'conventions_summary.json', summary)
    expected_expiries = int(metadata.elapsed_days.between(args.min_days, args.max_days).sum())
    checks = [dict(check='Data/baseline preservation', status='PASS', detail=str(summary['preservation'])),
              dict(check='Parity identification', status='PASS' if fits.status.eq('identified').sum() == expected_expiries else 'FAIL',
                   detail=str(summary['parity_status'])),
              dict(check='Vendor-IV bid/ask consistency, original calls', status='PASS' if summary['baseline_calls']['parity_coverage'] == 1 else 'FAIL',
                   detail=str(summary['baseline_calls'])),
              dict(check='Every retained mid-price admits positive finite IV', status='PASS' if options.mid_iv_status.eq('ok').all() else 'FAIL',
                   detail=str(summary['selected_bound_status'])),
              dict(check='Contract settlement identity', status='NOT_RUN', detail='Raw file has no root/symbol or settlement flag'),
              dict(check='Quote synchrony at exchange', status='NOT_RUN', detail='Shared vendor snapshot; no side quote timestamps/ages'),
              dict(check='Constrained surface fitting / Dupire / hedging', status='NOT_RUN', detail='Outside scope')]
    pd.DataFrame(checks).to_csv(out/'check_status.csv', index=False)
    print(json.dumps({k: summary[k] for k in ['raw_rows', 'matched_pairs', 'calibration_pairs', 'parity_status', 'baseline_calls', 'preservation']}, indent=2))
    print(f'Artifacts: {out}')


if __name__ == '__main__':
    main()
