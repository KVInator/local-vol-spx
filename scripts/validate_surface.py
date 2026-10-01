"""Reproducible audit of the existing surface, without writing baseline paths.

Run with the volspx interpreter. Market failures are written as FAIL in the
check table; they do not raise an exception or trigger an unrequested repair.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lv_project import finite_diff
from lv_project.config import default_target_maturities_years
from lv_project.diagnostics import check_call_slice, reconstruct_call_prices
from lv_project.surface import (
    build_daily_surface, count_calendar_violations, interpolate_target_surface,
    surface_to_long_frame,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_date(path, date):
    """Read only the requested date into memory; do not alter the clean file."""
    frames = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        selected = chunk.loc[chunk.quote_date.eq(date)]
        if len(selected):
            frames.append(selected)
    if not frames:
        raise ValueError(f'No clean call quotes found for quote_date={date}')
    quotes = pd.concat(frames, ignore_index=True)
    for name in ['quote_date', 'expiry']:
        quotes[name] = pd.to_datetime(quotes[name])
    return quotes


def reproduce(args, out):
    """Run the existing script with its normal output paths in a disposable copy."""
    with tempfile.TemporaryDirectory(prefix='spx-validation-') as temp:
        stage = Path(temp)
        shutil.copytree(ROOT / 'src', stage / 'src')
        (stage / 'scripts').mkdir()
        shutil.copy2(ROOT / 'scripts/02_build_surface.py', stage / 'scripts/02_build_surface.py')
        (stage / 'data/processed').mkdir(parents=True)
        (stage / 'data/processed/spx_calls_clean.csv').symlink_to(args.clean_path.resolve())
        command = [sys.executable, 'scripts/02_build_surface.py', '--quote-date', args.quote_date,
                   '--y-min', str(args.y_min), '--y-max', str(args.y_max)]
        with (out / 'build_surface.log').open('w') as stream:
            process = subprocess.run(command, cwd=stage, stdout=stream, stderr=subprocess.STDOUT)
        write_json(out / 'reproduction_command.json', {
            'argv': command, 'cwd': str(stage), 'exit_code': process.returncode,
            'clean_file_symlink_target': str(args.clean_path.resolve()),
        })
        if process.returncode:
            raise RuntimeError(f'Surface build exited {process.returncode}; see {out / "build_surface.log"}')
        for directory in ['data/processed/surfaces', 'outputs/diagnostics', 'outputs/interactive']:
            shutil.copytree(stage / directory, out / 'reproduction' / directory, dirs_exist_ok=True)


def convergence(out, before_path):
    modules = {'current': finite_diff}
    if before_path.exists():
        spec = importlib.util.spec_from_file_location('finite_diff_before', before_path)
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
        modules['before'] = before
    rows, benchmarks = [], []
    for version, module in modules.items():
        for kind in ['uniform', 'smooth_nonuniform', 'alternating_nonuniform']:
            for n in [17, 33, 65, 129, 257]:
                u = np.linspace(0, 1, n)
                if kind == 'uniform':
                    x = u
                elif kind == 'smooth_nonuniform':
                    x = (u + u*u)/2
                else:
                    steps = np.resize([1., 2.], n-1)
                    x = np.r_[0., np.cumsum(steps)/steps.sum()]
                for order in [1, 2]:
                    operator = getattr(module, f'{"first" if order == 1 else "second"}_derivative_1d')
                    # exp(x) has the same exact first and second derivative.
                    error = np.abs(operator(x, np.exp(x)) - np.exp(x))
                    rows.append(dict(version=version, grid=kind, n=n, derivative=order,
                                     h_max=float(np.diff(x).max()), interior_error=float(error[1:-1].max()),
                                     left_error=float(error[0]), right_error=float(error[-1]),
                                     boundary_error=float(error[[0, -1]].max())))
        for kind, x in [('unequal_neighbors', np.array([0., 1., 3.])),
                        ('actual_maturities', default_target_maturities_years())]:
            for name, f, f1, f2 in [('quadratic', x*x, 2*x, np.full_like(x, 2)),
                                     ('cubic', x**3, 3*x*x, 6*x),
                                     ('exp', np.exp(x), np.exp(x), np.exp(x))]:
                for order, exact in [(1, f1), (2, f2)]:
                    operator = getattr(module, f'{"first" if order == 1 else "second"}_derivative_1d')
                    error = np.abs(operator(x, f) - exact)
                    benchmarks.append(dict(version=version, grid=kind, function=name, derivative=order,
                                           max_error=float(error.max()), interior_error=float(error[1:-1].max()),
                                           boundary_error=float(error[[0, -1]].max())))
    table = pd.DataFrame(rows)
    for col in ['interior_error', 'boundary_error']:
        table[col.replace('_error', '_order')] = table.groupby(['version', 'grid', 'derivative'])[col].transform(
            lambda v: np.log2(v.shift()/v))
    table.to_csv(out / 'finite_difference_convergence.csv', index=False)
    pd.DataFrame(benchmarks).to_csv(out / 'finite_difference_benchmarks.csv', index=False)
    return table


def support_masks(result):
    y = result.y_grid
    t = result.target_maturities
    pt = result.pillar_maturities
    coverage = result.expiry_coverage
    native = ((y[None, :] >= coverage.y_min_native.to_numpy()[:, None])
              & (y[None, :] <= coverage.y_max_native.to_numpy()[:, None]))
    # Track original pillars responsible for cumulative-max values, including ties.
    source = np.array([np.argmax(result.pillar_total_variance[:i+1], axis=0)
                       for i in range(len(pt))])
    repaired_native = native[source, np.arange(len(y))[None, :]]
    time_ok = (t >= pt[0]) & (t <= pt[-1])
    target_native = np.zeros((len(t), len(y)), dtype=bool)
    target_repaired_native = np.zeros_like(target_native)
    brackets = []
    for i, maturity in enumerate(t):
        if not time_ok[i]:
            brackets.append((-1, -1))
            continue
        right = int(np.searchsorted(pt, maturity, side='left'))
        left = right if pt[right] == maturity else right-1
        brackets.append((left, right))
        target_native[i] = native[left] & native[right]
        target_repaired_native[i] = repaired_native[left] & repaired_native[right]
    return native, repaired_native, time_ok, target_native, target_repaired_native, brackets


def call_tables(spot, t, y, w, native, stage):
    k, c, forward, discount = reconstruct_call_prices(spot, t, y, w)
    tables = []
    for i in range(len(t)):
        frame = check_call_slice(k[i], c[i], forward[i], discount[i])
        frame['stage'] = stage
        frame['time_to_expiry'] = t[i]
        frame['days_to_expiry'] = t[i]*365
        frame['log_moneyness'] = y
        frame['native_supported'] = native[i]
        frame['native_supported_triple'] = np.r_[False, native[i, :-2] & native[i, 1:-1] & native[i, 2:], False]
        tables.append(frame)
    return pd.concat(tables, ignore_index=True)


def summarize_prices(table):
    rows = []
    for (stage, t), frame in table.groupby(['stage', 'time_to_expiry'], sort=False):
        valid = frame.butterfly_valid
        rows.append(dict(stage=stage, time_to_expiry=t, days_to_expiry=t*365,
                         price_nodes=int(frame.price_valid.sum()), unchecked_nodes=int((~frame.price_valid).sum()),
                         strike_pairs=int(frame.right_pair_valid.sum()), butterfly_triples=int(valid.sum()),
                         bounds_violations=int(frame.bounds_violation.sum()),
                         monotonicity_violations=int(frame.monotonicity_violation.sum()),
                         vertical_spread_violations=int(frame.vertical_spread_violation.sum()),
                         butterfly_violations=int(frame.butterfly_violation.sum()),
                         native_butterfly_violations=int((frame.butterfly_violation & frame.native_supported_triple).sum()),
                         min_curvature=float(frame.loc[valid, 'strike_curvature'].min()) if valid.any() else None,
                         min_butterfly_cost=float(frame.loc[valid, 'butterfly_cost'].min()) if valid.any() else None))
    return pd.DataFrame(rows)


def tolerance_sensitivity(table, out):
    rows = []
    for stage, frame in table.groupby('stage', sort=False):
        for tolerance in [1e-10, 1e-8, 1e-6, 1e-4]:
            rows.append(dict(stage=stage, metric='negative_slope_change', tolerance=tolerance,
                             failures=int((frame.butterfly_valid & (frame.slope_change < -tolerance)).sum())))
            rows.append(dict(stage=stage, metric='positive_call_slope', tolerance=tolerance,
                             failures=int((frame.right_pair_valid & (frame.right_slope > tolerance)).sum())))
        for tolerance in [1e-8, 1e-6, 1e-3, .01, .1]:
            rows.append(dict(stage=stage, metric='negative_butterfly_cost', tolerance=tolerance,
                             failures=int((frame.butterfly_valid & (frame.butterfly_cost < -tolerance)).sum())))
    pd.DataFrame(rows).to_csv(out / 'price_tolerance_sensitivity.csv', index=False)


def repair_metrics(raw, repaired, t, y, out, prefix):
    delta = repaired-raw
    changed = delta > 1e-14
    relative = delta/raw
    iv_change = np.sqrt(repaired/t[:, None])-np.sqrt(raw/t[:, None])
    tt, yy = np.meshgrid(t, y, indexing='ij')
    frame = pd.DataFrame(dict(time_to_expiry=tt.ravel(), days_to_expiry=(tt*365).ravel(),
                              log_moneyness=yy.ravel(), original_w=raw.ravel(), repaired_w=repaired.ravel(),
                              delta_w=delta.ravel(), relative_delta_w=relative.ravel(),
                              delta_iv_bps=(iv_change*10000).ravel()))
    frame.to_csv(out / f'{prefix}_calendar_repairs.csv', index=False)
    pos = delta[changed]
    index = np.unravel_index(np.nanargmax(delta), delta.shape)
    return dict(changed_cells=int(changed.sum()), finite_cells=int(np.isfinite(raw).sum()),
                max_delta_w=float(np.nanmax(delta)), mean_delta_w_all=float(np.nanmean(delta)),
                mean_delta_w_changed=float(pos.mean()) if len(pos) else 0.,
                p50_delta_w_changed=float(np.median(pos)) if len(pos) else 0.,
                p95_delta_w_changed=float(np.quantile(pos, .95)) if len(pos) else 0.,
                max_relative_delta_w=float(np.nanmax(relative)),
                max_delta_iv_bps=float(np.nanmax(iv_change)*10000),
                max_repair_days=float(t[index[0]]*365), max_repair_y=float(y[index[1]]))


def baseline_comparison(result, out, date):
    baseline = ROOT / f'data/processed/surfaces/surface_{date}.csv'
    reproduction = out / f'reproduction/data/processed/surfaces/surface_{date}.csv'
    fresh = surface_to_long_frame(result)
    rows = []
    for label, path in [('existing_baseline', baseline), ('reproduction', reproduction)]:
        if not path.exists():
            rows.append(dict(source=label, path=str(path), status='NOT_RUN', reason='File missing'))
            continue
        saved = pd.read_csv(path)
        if len(saved) != len(fresh) or list(saved.columns) != list(fresh.columns):
            rows.append(dict(source=label, path=str(path), status='FAIL', reason='Schema/row mismatch'))
            continue
        row = dict(source=label, path=str(path), status='PASS')
        passed = saved.quote_date.eq(date).all()
        for col in ['time_to_expiry', 'log_moneyness', 'total_variance', 'implied_vol']:
            a = saved[col].to_numpy()
            b = fresh[col].to_numpy()
            row[f'max_abs_{col}_difference'] = float(np.nanmax(np.abs(a-b)))
            passed &= np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
        row['status'] = 'PASS' if passed else 'FAIL'
        rows.append(row)
    write_json(out / 'baseline_comparison.json', rows)
    return rows


def preservation_snapshot(out):
    """Snapshot current data/baseline files without depending on local notes.

    Every run starts a fresh manifest. All validation artifacts are excluded,
    including sibling runs when the requested output directory is nested.
    """
    path = out / 'preservation_before.json'
    validation_root = ROOT / 'outputs/diagnostics/validation'
    files = []
    for directory in ['data/raw', 'data/processed', 'outputs']:
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and not p.is_relative_to(validation_root))
    rows = []
    for p in sorted(files):
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stat = p.stat()
        rows.append(dict(path=str(p.relative_to(ROOT)), size_bytes=stat.st_size,
                         mtime_ns=stat.st_mtime_ns, sha256=digest))
    write_json(path, rows)
    return rows


def verify_preservation(manifest, out):
    rows = []
    for before in manifest:
        path = ROOT / before['path']
        if not path.exists():
            rows.append(dict(path=before['path'], unchanged=False, reason='Missing'))
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stat = path.stat()
        rows.append(dict(path=before['path'], unchanged=(digest == before['sha256']
                         and stat.st_mtime_ns == before['mtime_ns'] and stat.st_size == before['size_bytes']),
                         sha256=digest, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    write_json(out / 'preservation_after.json', rows)
    return dict(files_checked=len(rows), changed_files=[r['path'] for r in rows if not r['unchanged']])


def plots(out, result, native, repaired_native, convergence_table, price_table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    for data, name, title, cmap in [
        (result.repaired_pillar_total_variance-result.pillar_total_variance,
         'calendar_repairs.png', 'Calendar repair: added total variance', 'magma'),
        ((~native).astype(int) + (~repaired_native).astype(int),
         'pillar_edge_support.png', 'Edge-fill exposure: native + repair contributor',
         ListedColormap(['#e6f4ea', '#fbbc04', '#ea4335'])),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        mesh = ax.pcolormesh(result.y_grid, result.pillar_maturities*365, data, shading='nearest', cmap=cmap)
        ax.set(xlabel='Log-forward moneyness y', ylabel='Pillar maturity (days)', title=title)
        fig.colorbar(mesh, ax=ax)
        fig.tight_layout()
        fig.savefig(out / name, dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, order in zip(axes, [1, 2]):
        subset = convergence_table[convergence_table.derivative.eq(order)]
        for (version, kind), group in subset.groupby(['version', 'grid']):
            axis.loglog(group.h_max, group.boundary_error, marker='o',
                        linestyle='-' if version == 'current' else '--', label=f'{version}: {kind}')
        axis.set(xlabel='Maximum grid step', ylabel='Max endpoint error', title=f'exp(x), derivative {order}')
        axis.legend(fontsize=7)
        axis.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'finite_difference_boundary_convergence.png', dpi=170)
    plt.close(fig)


def price_plots(out, result, price_table):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for days in [7, 14, 30, 60, 180, 300]:
        i = np.abs(result.target_maturities*365-days).argmin()
        if np.isfinite(result.target_implied_vol[i]).any():
            ax.plot(result.y_grid, result.target_implied_vol[i], label=f'{days} days')
    ax.set(xlabel='Log-forward moneyness y', ylabel='Implied volatility', title='Reproduced SPX smile slices')
    ax.legend()
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'smile_slices.png', dpi=170)
    plt.close(fig)
    base = price_table.loc[price_table.stage.eq('target_repaired_n50')]
    curvature = base.pivot(index='days_to_expiry', columns='log_moneyness', values='strike_curvature')
    fig, ax = plt.subplots(figsize=(9, 5))
    limits = np.nanquantile(np.abs(curvature.to_numpy()), .95)
    mesh = ax.pcolormesh(curvature.columns, curvature.index, curvature.to_numpy(), shading='nearest',
                         cmap='RdBu', vmin=-limits, vmax=limits)
    ax.set(xlabel='Log-forward moneyness y', ylabel='Target maturity (days)',
           title='Call strike curvature: negative regions fail convexity\nColor scale clipped at 95th percentile')
    fig.colorbar(mesh, ax=ax, label='Second strike divided difference')
    fig.tight_layout()
    fig.savefig(out / 'call_strike_curvature.png', dpi=170)
    plt.close(fig)
    worst = base.loc[base.butterfly_cost.idxmin(), 'time_to_expiry']
    frame = base.loc[base.time_to_expiry.eq(worst)]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(frame.strike, frame.call_price, marker='.', label='Reconstructed call')
    failed = frame.butterfly_violation
    axes[0].scatter(frame.loc[failed, 'strike'], frame.loc[failed, 'call_price'], color='red', label='Butterfly failure')
    axes[0].set(ylabel='Discounted call (index points)', title=f'Worst baseline butterfly cost: {worst*365:g} day slice')
    axes[0].legend()
    axes[1].plot(frame.strike, frame.right_slope, marker='.')
    axes[1].set(xlabel='Strike K', ylabel='Adjacent strike slope')
    for ax in axes:
        ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'worst_call_slice.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quote-date', default='2023-12-22')
    parser.add_argument('--clean-path', type=Path, default=ROOT / 'data/processed/spx_calls_clean.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/diagnostics/validation')
    parser.add_argument('--y-min', type=float, default=-.15)
    parser.add_argument('--y-max', type=float, default=.10)
    parser.add_argument('--skip-reproduction', action='store_true', help='Reuse saved isolated-build artifacts.')
    parser.add_argument('--before-finite-diff', type=Path,
                        default=ROOT / 'outputs/diagnostics/validation/finite_diff_before.py')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / 'outputs/diagnostics/validation'):
        raise ValueError('Validation output must be under outputs/diagnostics/validation to protect baseline outputs.')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    if not args.clean_path.exists():
        raise FileNotFoundError(f'Clean dataset missing: {args.clean_path}')
    manifest = preservation_snapshot(out)
    if not args.skip_reproduction:
        reproduce(args, out)
    print('Loading date and auditing support/calendar/prices...', flush=True)
    quotes = load_date(args.clean_path, args.quote_date)
    y = np.linspace(args.y_min, args.y_max, 50)
    t = default_target_maturities_years()
    result = build_daily_surface(quotes, args.quote_date, y, t,
                                 input_y_min=args.y_min-.1, input_y_max=args.y_max+.1)
    native, repaired_native, time_ok, target_native, target_repaired_native, brackets = support_masks(result)
    target_raw = interpolate_target_surface(result.pillar_maturities, result.pillar_total_variance, t)
    pillars_delta = repair_metrics(result.pillar_total_variance, result.repaired_pillar_total_variance,
                                   result.pillar_maturities, y, out, 'pillar')
    target_delta = repair_metrics(target_raw, result.target_total_variance, t, y, out, 'target')
    coverage = result.expiry_coverage.copy()
    coverage['days_to_expiry'] = coverage.time_to_expiry*365
    coverage['left_edge_fills'] = (y[None, :] < coverage.y_min_native.to_numpy()[:, None]).sum(axis=1)
    coverage['right_edge_fills'] = (y[None, :] > coverage.y_max_native.to_numpy()[:, None]).sum(axis=1)
    coverage['repair_contributor_edge_cells'] = (~repaired_native).sum(axis=1)
    coverage.to_csv(out / 'pillar_support.csv', index=False)
    target = surface_to_long_frame(result)
    target['time_supported'] = np.repeat(time_ok, len(y))
    target['native_bracket_supported'] = target_native.ravel()
    target['repair_contributor_supported'] = target_repaired_native.ravel()
    target['conservative_supported'] = (target_native & target_repaired_native).ravel()
    target['left_pillar_days'] = np.repeat([result.pillar_maturities[a]*365 if a >= 0 else np.nan
                                           for a, b in brackets], len(y))
    target['right_pillar_days'] = np.repeat([result.pillar_maturities[b]*365 if b >= 0 else np.nan
                                            for a, b in brackets], len(y))
    target.to_csv(out / 'target_support.csv', index=False)
    edge_errors = []
    for i, expiry in enumerate(result.pillar_expiries):
        g = result.surface_points.loc[result.surface_points.expiry.eq(pd.Timestamp(expiry))]
        g = g.groupby('strike', as_index=False).total_variance.mean().sort_values('strike')
        left = y < coverage.y_min_native.iloc[i]
        right = y > coverage.y_max_native.iloc[i]
        edge_errors.extend(abs(result.pillar_total_variance[i, left]-g.total_variance.iloc[0]))
        edge_errors.extend(abs(result.pillar_total_variance[i, right]-g.total_variance.iloc[-1]))
    support = dict(pillar_min_days=float(result.pillar_maturities.min()*365),
                   pillar_max_days=float(result.pillar_maturities.max()*365),
                   target_cells=int(result.target_total_variance.size),
                   finite_target_cells=int(np.isfinite(result.target_total_variance).sum()),
                   nan_target_cells=int(np.isnan(result.target_total_variance).sum()),
                   unexpected_nonfinite_cells=int((~np.isfinite(result.target_total_variance)
                                                   & time_ok[:, None]).sum()),
                   unsupported_target_days=(t[~time_ok]*365).tolist(),
                   nonpositive_target_w_cells=int((result.target_total_variance <= 0).sum()),
                   pillar_edge_fill_cells=int((~native).sum()),
                   expiries_with_edge_fill=int((~native).any(axis=1).sum()),
                   target_native_bracket_cells=int(target_native.sum()),
                   target_repair_contributor_supported_cells=int(target_repaired_native.sum()),
                   target_conservative_supported_cells=int((target_native & target_repaired_native).sum()),
                   max_edge_fill_value_error=float(max(edge_errors, default=0)))
    calendar = dict(pillar_violations_before=count_calendar_violations(result.pillar_total_variance),
                    pillar_violations_after=count_calendar_violations(result.repaired_pillar_total_variance),
                    target_violations_before=count_calendar_violations(target_raw),
                    target_violations_after=count_calendar_violations(result.target_total_variance),
                    repaired_flat_pillar_intervals=int((np.abs(np.diff(result.repaired_pillar_total_variance, axis=0))
                                                        <= 1e-14).sum()),
                    pillar_intervals=int((len(result.pillar_maturities)-1)*len(y)),
                    pillar_repairs=pillars_delta, target_repairs=target_delta)
    tables = [
        call_tables(result.spot, result.pillar_maturities, y, result.pillar_total_variance,
                    native, 'pillar_raw_n50'),
        call_tables(result.spot, result.pillar_maturities, y, result.repaired_pillar_total_variance,
                    repaired_native, 'pillar_repaired_n50'),
        call_tables(result.spot, t, y, target_raw, target_native, 'target_raw_n50'),
        call_tables(result.spot, t, y, result.target_total_variance,
                    target_repaired_native, 'target_repaired_n50'),
    ]
    for n in [201, 1001]:
        dense_y = np.linspace(args.y_min, args.y_max, n)
        dense = build_daily_surface(quotes, args.quote_date, dense_y, t,
                                    input_y_min=args.y_min-.1, input_y_max=args.y_max+.1)
        dn, drn, dt, dtn, dtrn, db = support_masks(dense)
        tables.extend([
            call_tables(dense.spot, dense.pillar_maturities, dense_y, dense.repaired_pillar_total_variance,
                        drn, f'pillar_repaired_n{n}'),
            call_tables(dense.spot, t, dense_y, dense.target_total_variance,
                        dtrn, f'target_repaired_n{n}'),
        ])
    prices = pd.concat(tables, ignore_index=True)
    prices.to_csv(out / 'call_price_checks.csv', index=False)
    tolerance_sensitivity(prices, out)
    price_summary = summarize_prices(prices)
    price_summary.to_csv(out / 'call_checks_by_slice.csv', index=False)
    failed = prices[['bounds_violation', 'monotonicity_violation', 'vertical_spread_violation',
                      'butterfly_violation']].any(axis=1)
    prices.loc[failed].to_csv(out / 'call_violations.csv', index=False)
    totals = []
    for stage, group in price_summary.groupby('stage', sort=False):
        entry = dict(stage=stage)
        for col in ['price_nodes', 'unchecked_nodes', 'strike_pairs', 'butterfly_triples', 'bounds_violations',
                    'monotonicity_violations', 'vertical_spread_violations', 'butterfly_violations',
                    'native_butterfly_violations']:
            entry[col] = int(group[col].sum())
        entry['min_curvature'] = float(group.min_curvature.min())
        entry['min_butterfly_cost'] = float(group.min_butterfly_cost.min())
        totals.append(entry)
    from lv_project.black_scholes import vectorized_call_price
    p = result.surface_points.copy()
    p['reconstructed_call'] = vectorized_call_price(result.spot, p.strike, p.time_to_expiry, 0, 0, p.implied_vol)
    p['price_minus_mid'] = p.reconstructed_call-p.mid
    p['outside_bid_ask'] = (p.reconstructed_call < p.bid-1e-8) | (p.reconstructed_call > p.ask+1e-8)
    p.to_csv(out / 'vendor_iv_quote_repricing.csv', index=False)
    quote_repricing = dict(quotes=len(p), outside_bid_ask=int(p.outside_bid_ask.sum()),
                           median_abs_mid_error=float(p.price_minus_mid.abs().median()),
                           max_abs_mid_error=float(p.price_minus_mid.abs().max()),
                           rmse_mid_error=float(np.sqrt(np.mean(p.price_minus_mid**2))))
    fd = convergence(out, args.before_finite_diff)
    comparison = baseline_comparison(result, out, args.quote_date)
    plots(out, result, native, repaired_native, fd, prices)
    price_plots(out, result, prices)
    preservation = verify_preservation(manifest, out)
    import scipy, matplotlib, plotly
    summary = dict(quote_date=args.quote_date, conventions=dict(rate=0., dividend_yield=0.,
                   spot=result.spot, day_count='DTE/365', forward='S exp((r-q)T)', discount='exp(-rT)'),
                   python=sys.version, executable=sys.executable,
                   packages={m.__name__: m.__version__ for m in [np, pd, scipy, matplotlib, plotly]},
                   build=result.diagnostics, support=support, calendar=calendar, call_checks=totals,
                   quote_repricing=quote_repricing, baseline_comparison=comparison, preservation=preservation)
    write_json(out / 'validation_summary.json', summary)
    checks = []
    def check(name, passed, detail):
        checks.append(dict(check=name, status='PASS' if passed else 'FAIL', detail=detail))
    check('Baseline numeric reproduction', all(r['status'] == 'PASS' for r in comparison),
          'w/IV/T/y tolerance 1e-12, identical NaN masks; see baseline_comparison.json')
    check('Input/baseline preservation', not preservation['changed_files'], str(preservation))
    check('Support-aware NaN handling', support['unexpected_nonfinite_cells'] == 0,
          f'{support["nan_target_cells"]} NaNs; unsupported days {support["unsupported_target_days"]}')
    check('Full target contributor support', target_repaired_native.all(),
          f'{support["target_repair_contributor_supported_cells"]}/{support["target_cells"]} cells')
    check('Full native bracket support', target_native.all(),
          f'{support["target_native_bracket_cells"]}/{support["target_cells"]} cells')
    check('Constant edge-fill implementation', support['max_edge_fill_value_error'] < 1e-14,
          f'{support["pillar_edge_fill_cells"]} filled pillar cells')
    check('Calendar monotonicity after repair', calendar['pillar_violations_after'] == calendar['target_violations_after'] == 0,
          'Adjacent differences tested with tolerance 1e-12 at fixed y')
    for row in totals:
        for metric in ['bounds_violations', 'monotonicity_violations', 'vertical_spread_violations', 'butterfly_violations']:
            check(f'{row["stage"]}: {metric}', row[metric] == 0,
                  f'{row[metric]} failures; {row["price_nodes"]} finite nodes, {row["butterfly_triples"]} triples')
    check('Vendor IV repricing inside bid/ask under zero carry', quote_repricing['outside_bid_ask'] == 0,
          f'{quote_repricing["outside_bid_ask"]}/{quote_repricing["quotes"]} outside; convention/model limitation')
    pd.DataFrame(checks).to_csv(out / 'check_status.csv', index=False)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Artifacts saved to {out}. PASS/FAIL statuses are in check_status.csv.', flush=True)


if __name__ == '__main__':
    main()
