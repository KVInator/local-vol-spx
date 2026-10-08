"""Saved AH smile hedges and a past-only empirical MV correction; no backward-pricer rerun."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

import optimal_hedging
from optimal_hedging import (OptimalHedgeInputs, SmileSettings, AHSmileHedges,
    EmpiricalMVSettings, PastOnlyEmpiricalMV, STRATEGIES, compare_hedges,
    paired_gains, analytical_controls, plot_optimal_hedges, write_notebook,
    AndreasenHugeSurface, PilotComparisonSettings, sha256, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-index', type=Path, required=True)
    parser.add_argument('--month', required=True)
    parser.add_argument('--history-run-index', type=Path)
    parser.add_argument('--history-month')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window-dates', type=int, default=60)
    parser.add_argument('--minimum-dates', type=int, default=10)
    parser.add_argument('--minimum-rows', type=int, default=60)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    if bool(args.history_run_index) != bool(args.history_month):
        parser.error('Supply both --history-run-index and --history-month.')
    current = OptimalHedgeInputs(args.run_index, args.month)
    history = (OptimalHedgeInputs(args.history_run_index, args.history_month)
               if args.history_run_index else None)
    runs = [current] + ([history] if history else [])
    if history and history.panel.quote_date.max() >= current.panel.quote_date.min():
        raise ValueError('The declared history must precede the target entry dates.')
    mv_settings = EmpiricalMVSettings(args.window_dates, args.minimum_dates, args.minimum_rows)
    smile_settings = SmileSettings()
    parent = Path(args.output or Path('outputs/optimal_hedging')/args.month).resolve()
    for run in runs:
        run.safe_output(parent)
    # Pin the saved funding convention; do not silently change it for a new strategy.
    index = json.loads(current.index_file.read_text())
    comparison_folder = Path(index['months'][args.month]['comparison'])
    if not comparison_folder.is_absolute(): comparison_folder = current.repository/comparison_folder
    comparison_folder = comparison_folder.resolve()
    previous = current.read_json(comparison_folder/'audit.json')
    if sha256(current.folders['panel']/'delta_panel.csv') not in previous.get('input_sha256', {}).values():
        raise ValueError('Saved comparison does not pin this panel.')
    if parent.is_relative_to(comparison_folder) or comparison_folder.is_relative_to(parent):
        raise ValueError('Keep new output separate from the saved comparison.')
    funding = PilotComparisonSettings(**previous['settings'])
    if args.plan:
        print(json.dumps(dict(month=args.month, entries=len(current.panel),
            dates=sorted(current.panel.quote_date.unique()),
            declared_past_entries=len(history.panel) if history else 0,
            strategies=list(STRATEGIES), smile_settings=asdict(smile_settings),
            empirical_settings=asdict(mv_settings), funding=asdict(funding),
            verified_input_files=sum(len(r.inputs) for r in runs), output_parent=str(parent),
            models_refitted=False, pde_execution=False), indent=2))
        print('No files written; no model loaded, regression fitted or hedge ledger executed.')
        return 0
    output = parent/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    audit_file = output/'audit.json'
    sources = [Path(__file__).resolve(), Path(optimal_hedging.__file__).resolve()]
    for module in ('surface_evidence', 'andreasen_huge', 'pilot_hedge_panel', 'pilot_hedge_comparison', 'hedge_ledger'):
        sources.append(Path(__import__(module).__file__).resolve())
    source_pins = {str(p): sha256(p) for p in sources}
    audit = dict(status='running', month=args.month, started_utc=datetime.now(timezone.utc).isoformat(),
        smile_settings=asdict(smile_settings), empirical_settings=asdict(mv_settings), funding=asdict(funding))
    audit_file.write_text(json.dumps(audit, indent=2)+'\n')
    try:
        panel = current.panel.copy(deep=True)
        smiles, model_status = [], []
        for (date, root), group in panel.groupby(['quote_date', 'root'], sort=True):
            clock = time.perf_counter()
            status = dict(quote_date=date, root=root, status='completed', message='')
            print(f'{date} / {root}: original AH smile and derivative controls...', flush=True)
            try:
                path = current.models.get((date, root))
                if path is None: raise ValueError('Original AH model unavailable.')
                model = AndreasenHugeSurface.load(path)
                quotes = current.quotes.loc[current.quotes.quote_date.eq(date) & current.quotes.root.eq(root)]
                result = AHSmileHedges(smile_settings).calculate(group, model, quotes)
            except (ValueError, RuntimeError, ArithmeticError) as error:
                status.update(status='failed', message=f'{type(error).__name__}: {error}')
                result = pd.DataFrame(dict(entry_id=group.entry_id, smile_status='model_unavailable',
                    sticky_delta_status='model_unavailable', lv_smile_status='model_unavailable'))
            status['elapsed_wall_seconds'] = time.perf_counter()-clock
            model_status.append(status)
            smiles.append(result)
        panel = panel.merge(pd.concat(smiles, ignore_index=True), on='entry_id', validate='one_to_one')
        training = pd.concat([r.panel for r in runs], ignore_index=True)
        predictions, fits, membership = PastOnlyEmpiricalMV(mv_settings).predict(panel, training)
        panel = panel.merge(predictions, on='entry_id', validate='one_to_one')
        print('Past-only coefficients fitted; comparing observed endpoint marks...', flush=True)
        trials, coverage, gains = compare_hedges(panel, funding)
        if not trials.empty:
            counts = trials.groupby(['scenario', 'entry_id']).strategy.nunique()
            common_ids = counts.loc[counts.eq(len(STRATEGIES))].reset_index()[['scenario','entry_id']]
            common = trials.merge(common_ids, on=['scenario','entry_id'], validate='many_to_one')
        else:
            common = trials.copy()
        tables = dict(strategy_panel=panel, mv_fits=fits, mv_training_membership=membership,
            model_status=pd.DataFrame(model_status), strategy_results=trials, coverage=coverage,
            gain_summary=gains, common_strategy_gain=paired_gains(common),
            analytical_controls=analytical_controls())
        if not trials.empty:
            tables['strategy_summary'] = summarize(trials, ['scenario', 'strategy'])
        for name, frame in tables.items(): frame.to_csv(output/f'{name}.csv', index=False)
        plot_optimal_hedges(output/'plots', panel, fits, gains)
        for run in runs: run.verify_unchanged()
        for path, pin in source_pins.items():
            if sha256(path) != pin: raise ValueError('Source changed during this run.')
        audit.update(status='completed_with_failures' if any(r['status']=='failed' for r in model_status) else 'completed',
            entries=len(panel), entry_dates=panel.quote_date.nunique(),
            ready_empirical_predictions=int(panel.empirical_mv_status.eq('ready').sum()),
            empirical_estimation='Hull--White quadratic correction; equal-row OLS, no intercept, raw observed mark SSE. Separate root/kind; fixed rolling window; endpoint strictly earlier than prediction. Warm-up and rank failures retained; no fallback.',
            empirical_objective_scope='Conventional empirical MV specification. SSE is variance plus squared mean; no claim of exact future conditional-variance optimality. No funding/cost inputs in estimation.',
            empirical_equation='delta_HW = delta_Black + vega_Black/(S*sqrt(T))*(a+b*delta_Black+c*delta_Black^2). Vega per decimal volatility.',
            surface_input='Original calibrated AH price surface. The smoothed-PDE IV surface is not substituted; the failed independent PCHIP benchmark is not used.',
            sticky_strike='At fixed K,T, IV held fixed. F/S and D fixed under a spot bump; model-IV Black delta.',
            sticky_delta='Freeze IV versus unadjusted normalized forward call delta N(d1); locally equivalent to frozen y=log(K/F) when delta(y) is decreasing/invertible. Delta = surface Black delta - surface vega*sigma_y/S. Not premium-adjusted FX delta.',
            lv_smile='HW ATM-derived approximation applied to selected strikes: observed-IV Black delta + observed-IV Black vega*sigma_K; sigma_K from original AH IV. Not ordinary AH PDE delta and not an exact all-strike MV theorem.',
            AH_PDE='Saved radius-specific delta with physical coefficient and original carry fixed. Labelled ah_pde, not minimum variance.',
            derivatives='Original AH IV from stable OTM inversion; centred IV widths 4 and 8 native log cells, support checked against observed expiry strike range. Frozen-smile direct bump at widths 2 and 4 cells. Failures retained; no clipping.',
            gain='1 - candidate SSE / Black SSE on identical entries; zero-centred. Funded ledger Gain and unfunded raw-mark Gain both saved. Bucket results are descriptive.',
            coverage='Each candidate retains all entries. Pairwise Gain uses candidate/Black common entries; common_strategy_gain uses the intersection of all six ready strategies. Warm-up is not retrospectively removed from coverage.',
            hedge_scope='One-session short research option, unit multipliers, synthetic fractional index fills at vendor spot, zero dividend cash; funding and execution scenarios copied from saved comparison.',
            evidence_scope='Development evaluation; selected-contract historical observations, not independent rows or a held-out result.',
            elapsed_wall_seconds=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
            max_reconciliation=float(trials.max_reconciliation.max()) if len(trials) else None,
            input_sha256={p: h for run in runs for p, h in run.inputs.items()}, source_sha256=source_pins,
            models_refitted=False, diffusion_pricing_PDE_executed=False,
            native_AH_resolvent_evaluated=True, future_endpoints_used_in_estimation=False,
            delta_clipped=False, observations_modified=False, independent_quote_interpolator_promoted=False,
            confidence_intervals_computed=False, out_of_sample_claim=False,
            contract_identity_verified=False, snapshot_provenance_verified=False, funding_curve_verified=False,
            executable_hedge_returns_claimed=False,
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
        write_notebook(output)
        audit['output_sha256'] = {p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p != audit_file}
        audit_file.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
        print('\nStrategy coverage:')
        print(coverage.groupby(['scenario','strategy','status']).size().to_string())
        print('\nMidpoint pairwise Gain:')
        print(gains.loc[gains.scenario.eq('mid') & gains.dimension.eq('overall'),
            ['strategy','comparisons','entry_dates','gain','raw_gain','mae_improvement']].to_string(index=False))
        print(f'\nMeasured wall time: {audit["elapsed_wall_seconds"]:.3f} seconds.')
        print(f'Results: {output}\nThree plots and optimal_hedging.ipynb saved. Development research; no strategy promoted.')
        return 0
    except Exception as error:
        audit.update(status='failed', message=f'{type(error).__name__}: {error}',
                     elapsed_wall_seconds=time.perf_counter()-started)
        audit_file.write_text(json.dumps(audit, indent=2)+'\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
