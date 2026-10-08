"""Consolidate saved daily AH surfaces and the quote total-variance/Dupire benchmark."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

import surface_evidence
from surface_evidence import (SurfaceEvidenceInputs, SurfaceEvidence, SurfaceEvidenceSettings,
    AndreasenHugeSurface, analytical_controls, finite_max, plot_surface_evidence,
    sha256, write_evidence_notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-index', type=Path, default=Path('outputs/research_pipeline/october_full_v1/run_index.json'))
    parser.add_argument('--month', default='2023-10')
    parser.add_argument('--dates', nargs='+', help='Default: every calibration date in this month, including recorded failures.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--half-width', type=float, default=0.09)
    parser.add_argument('--y-points', type=int, default=181)
    parser.add_argument('--time-subdivisions', type=int, default=2)
    parser.add_argument('--radius', type=float, default=0.0005)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    settings = SurfaceEvidenceSettings(half_width=args.half_width, y_points=args.y_points,
        time_subdivisions=args.time_subdivisions, radius=args.radius)
    run = SurfaceEvidenceInputs(args.run_index, args.month)
    manifest = run.manifest
    dates = sorted(set(args.dates or manifest.quote_date.unique()))
    if not set(dates) <= set(manifest.quote_date):
        raise ValueError('A requested date is absent from the indexed calibration manifest.')
    manifest = manifest.loc[manifest.quote_date.isin(dates)].copy()
    parent = run.safe_output(args.output or Path('outputs/surface_evidence')/args.month)
    if args.plan:
        print(json.dumps(dict(month=args.month, dates=dates, models=len(manifest),
            settings=asdict(settings), verified_input_files=len(run.inputs), output_parent=str(parent),
            models_refitted=False, pde_execution=False, hedge_execution=False), indent=2))
        print('No files written; no model loaded, surface evaluated or plot generated.')
        return 0
    sources = [Path(__file__).resolve(), Path(surface_evidence.__file__).resolve()]
    for module in ['andreasen_huge', 'ah_local_vol', 'daily_ah_validation', 'daily_ah']:
        sources.append(Path(__import__(module).__file__).resolve())
    source_pins = {str(path): sha256(path) for path in sources}
    stamp = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    output = parent/stamp
    output.mkdir(parents=True, exist_ok=False)
    (output/'dates').mkdir()
    (output/'plots').mkdir()
    audit_file = output/'audit.json'
    audit = dict(status='running', month=args.month, dates=dates, settings=asdict(settings),
        started_utc=started_utc, upstream_run_index=str(run.index_file))
    audit_file.write_text(json.dumps(audit, indent=2)+'\n')
    parts, statuses, summary = {}, [], []
    try:
        for record in manifest.to_dict('records'):
            date, root = record['quote_date'], record['root']
            clock = time.perf_counter()
            identity = dict(quote_date=date, root=root)
            status = dict(**identity, status='failed', producer_model_status=record['status'],
                          message='', samples_file='', plot_prefix='')
            print(f'{date} / {root}: surface views, support, derivatives and density...', flush=True)
            try:
                if record['status'] != 'fitted':
                    raise ValueError('Upstream model was not fitted; retained as a failure.')
                model = AndreasenHugeSurface.load(run.models[(date, root)])
                quotes = run.quotes.loc[run.quotes.quote_date.eq(date) & run.quotes.root.eq(root)]
                carry = run.carry.loc[run.carry.quote_date.eq(date) & run.carry.root.eq(root)]
                tables = SurfaceEvidence(settings).run(model, quotes, carry, identity)
                identifier = date+'_'+hashlib.sha256(root.encode()).hexdigest()[:12]
                samples_file = 'dates/'+identifier+'.csv'
                tables['surface_samples'].to_csv(output/samples_file, index=False)
                prefix = 'plots/'+identifier
                plot_surface_evidence(tables['surface_samples'], output/prefix)
                for name, frame in tables.items():
                    if name != 'surface_samples':
                        parts.setdefault(name, []).append(frame)
                sample, obs = tables['surface_samples'], tables['quote_inversions']
                supported = sample.quote_supported
                ready = obs.iv_status.eq('ready')
                summary.append(dict(**identity, quotes=len(obs), iv_ready=int(ready.sum()),
                    iv_unresolved=int((~ready).sum()), surface_samples=len(sample),
                    quote_supported_samples=int(supported.sum()), quote_admissible_samples=int(sample.quote_admissible.sum()),
                    quote_admissible_fraction_of_supported=float(sample.quote_admissible.sum()/supported.sum()) if supported.any() else np.nan,
                    negative_quote_density_samples=int((supported & sample.quote_density_z.lt(-1e-10)).sum()),
                    negative_quote_calendar_samples=int((supported & sample.quote_wt.lt(-1e-10)).sum()),
                    max_quote_inversion_error_points=finite_max(abs(obs.quote_reconstruction_error_points)),
                    original_ah_rms_half_spreads=float(np.sqrt(np.mean(obs.ah_residual_half_spreads**2))),
                    max_ah_chain_variance_difference=finite_max(abs(sample.ah_chain_minus_actual_variance)),
                    max_ah_fd_variance_difference=finite_max(abs(sample.ah_fd_variance_fine-sample.ah_actual_lv**2)),
                    max_quote_vs_ah_variance_difference=finite_max(abs(sample.quote_minus_ah_variance)),
                    max_quote_vs_ah_price_difference_points=finite_max(abs(sample.quote_call_minus_ah_points)),
                    max_ah_actual_lv_pct=finite_max(100*sample.ah_actual_lv),
                    max_smoothed_actual_lv_pct=finite_max(100*sample.smoothed_actual_lv),
                    samples_file=samples_file, plot_prefix=prefix))
                status.update(status='completed', samples_file=samples_file, plot_prefix=prefix)
            except Exception as error:
                status['message'] = f'{type(error).__name__}: {error}'
                print('  FAILED: '+status['message'], flush=True)
            status['elapsed_wall_seconds'] = time.perf_counter()-clock
            statuses.append(status)
            print(f"  {status['status']}; measured {status['elapsed_wall_seconds']:.3f} seconds", flush=True)
        pd.DataFrame(statuses).to_csv(output/'evidence_status.csv', index=False)
        pd.DataFrame(summary).to_csv(output/'daily_summary.csv', index=False)
        for name, frames in parts.items():
            pd.concat(frames, ignore_index=True).to_csv(output/f'{name}.csv', index=False)
        analytical_controls().to_csv(output/'analytical_controls.csv', index=False)
        if summary:
            write_evidence_notebook(output)
        run.verify_unchanged()
        for path, digest in source_pins.items():
            if sha256(path) != digest:
                raise ValueError(f'Source changed during evidence generation: {path}')
        failures = sum(item['status'] != 'completed' for item in statuses)
        import matplotlib
        audit.update(status='completed_with_failures' if failures else 'completed',
            successful_models=len(summary), failed_models=failures,
            input_sha256=run.inputs, source_sha256=source_pins,
            output_sha256={path.relative_to(output).as_posix(): sha256(path)
                for path in sorted(output.rglob('*')) if path.is_file() and path != audit_file},
            coordinates='T: ACT/365F elapsed time from the saved snapshot; y=log(K/F(T)); c=C/(D(T)*F(T)); w=T*IV^2.',
            carry='Pinned same-date conditional parity forward and assumed discount; log-linear interpolation between pillars, as in AH.',
            independent_route='Original equivalent-call midpoint IVs; contiguous PCHIP segments in y; linear w in T. Invalid IV observations retained and split support.',
            independent_short_end='Unsupported before the first observed expiry. No artificial initial variance curve or day-zero diffusion.',
            quote_admissibility='Common bracketing support; w>0, w_T>=0, g>denominator_floor; vertical-spread bounds; spatial stencil-width agreement and agreement with analytic derivatives; strike knots excluded.',
            independent_arbitrage_policy='Report signed calendar, density, vertical-spread and support diagnostics. No clipping, band expansion, isotonic repair or claim of arbitrage-free certification.',
            AH_derivatives='Native discrete curvature/time derivative and Black coordinate chain rule. These are not derivatives of the piecewise-linear price interpolant.',
            AH_finite_differences='Width 8 and 4 native log cells; positive one-sided time differences at pillars, central within each interval. A width check is not a continuum convergence proof.',
            density='density_z=c_zz; density_K=c_zz/F. AH density is nodal. Finite-domain mass/moment quadrature and unrescaled report-window mass are diagnostics, not global tail conditions.',
            derivative_sides='Both expiry sides saved except the last. PCHIP w_yy knot jumps and local-volatility pillar jumps reported. Surface figures use the left pillar side.',
            smoothing='The pinned-radius first-period Gaussian variance blend changes the diffusion. Only its coefficient is consolidated here; original AH IV, w and density do not describe its prices.',
            comparison_scope='Independent total-variance interpolation versus original AH on common admissible samples; different constructions, not a numerical-error estimate.',
            analytical_controls='Deterministic flat and smoothly skewed Black price surfaces; not historical observations or performance evidence.',
            evidence_scope='Finite reported grid and original AH native finite domain; no global arbitrage or tail guarantee.',
            models_refitted=False, optimizer_executed=False, pde_executed=False, hedge_ledger_executed=False,
            observations_modified=False, prices_clipped=False, actual_local_volatility_capped=False,
            candidate_promoted=False, global_wing_robustness_certified=False, all_entry_greeks_validated=False,
            contract_identity_verified=False, snapshot_provenance_verified=False, funding_curve_verified=False,
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__),
            elapsed_wall_seconds=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
            timing_scope='Read/pin inputs, reconstruct saved models, evaluate surfaces, diagnostics, figures, notebook and hashes; final audit write excluded.')
    except BaseException as error:
        audit.update(status='failed', message=f'{type(error).__name__}: {error}',
                     elapsed_wall_seconds=time.perf_counter()-started)
        audit_file.write_text(json.dumps(audit, indent=2)+'\n')
        raise
    audit_file.write_text(json.dumps(audit, indent=2)+'\n')
    report = pd.DataFrame(summary)
    if len(report):
        columns = ['quote_date','root','quotes','iv_unresolved','quote_admissible_fraction_of_supported',
                   'negative_quote_density_samples','negative_quote_calendar_samples','max_ah_chain_variance_difference']
        print('\nSurface evidence:\n'+report[columns].to_string(index=False))
    print(f"\nMeasured wall time: {audit['elapsed_wall_seconds']:.3f} seconds.\nEvidence: {output}")
    print('Four figures per successful model and a saved-evidence notebook. Signed failures remain in the tables.')
    print('No model promoted; a completed job does not certify the independent interpolator.')
    return 2 if audit['failed_models'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
