# Integrated SPX research pipeline

The runner connects the existing preparation, assumed-rate carry, independent
daily AH calibration, optional daily validation, one-session delta panel,
self-financing hedge comparison, attribution and notebook stages. It does not
change pricing formulas, fit objectives or the accepted hedge convention.

## Install this update

Copy the delivered files into these repository paths:

| File | Repository path |
|---|---|
| `research_pipeline.py` | `src/research_pipeline.py` |
| `36_run_research_pipeline.py` | `scripts/36_run_research_pipeline.py` |
| `26_prepare_hedging_pilot.py` | `scripts/26_prepare_hedging_pilot.py` (replace) |
| `33_compare_pilot_hedges.py` | `scripts/33_compare_pilot_hedges.py` (replace) |
| `test_research_pipeline.py` | `tests/test_research_pipeline.py` |
| `research_pipeline_september.json` | `config/research_pipeline_september.json` |
| This guide | `docs/research_pipeline.md` |

No package structure or new runtime dependency is introduced. The runner uses
your current Python environment and existing flat `src` modules. The lock uses
`fcntl`, supported on macOS and Linux.

Script 26 now accepts a date range and several monthly files. Its original
single-file September command retains the same settings. Script 33 replaces
two September-specific descriptions with general development-sample labels;
its calculations are unchanged.

## First run: register the reviewed September checkpoint

The supplied configuration names your exact completed run folders. It checks
their output hashes, model paths, input chain, settings and selected entries.
It does not rerun their optimizers or hedge PDEs. The generated monthly notebook
uses the existing script-35 template and loads these saved results.

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_september.json --plan

PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_september.json
```

Expected September coverage: 20 calibration dates, 360 entries, 342 matched
comparisons. The remaining 18 entries keep their sample-end status. This
configuration deliberately reproduces that original observation window.

Outputs are under `outputs/research_pipeline/september_checkpoint_v1/`:

- `pipeline_state.json`: atomic stage progress, source/environment identity,
  checksums, checkpoint origins, attempt counts and durations.
- `stage_index.csv`: readable progress and timings.
- `run_index.json`: exact artifact folders for each month.
- `month_summary.csv`: dates, model failures, entries and matched comparisons.
- `2023-09/date_coverage.csv`: policy dates and current calibration availability,
  including dates without usable quotes.
- `2023-09/research_results.ipynb`: the saved-results notebook.

Open the notebook with your `volspx` kernel. It contains no optimization or hedge
PDE work. Its numerical-reference and research limitations remain explicit.

## Resume and status

Run the same command again to resume. Completed outputs are rechecked and
reused. Failed or interrupted process jobs run in a new numbered attempt
folder, preserving earlier logs and partial outputs. Fresh calibrations and
delta calculations are separate jobs by date, so completed dates are retained.

```bash
PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_september.json --status
```

Use `--stop-after carry`, `--stop-after calibration`, or another stage name to
pause at a checkpoint. Resume without that flag to finish the remaining stages.
Stopping after attribution does not build the notebook.

Changes to configuration, source code or scientific-library versions require
a new output folder. Changed inputs or corrupted saved outputs are rejected;
the runner does not silently overwrite a completed experiment. Explicitly
adopted old checkpoints preserve their recorded source hashes. Their reuse is
not a claim that current source versions generated those old outputs.

A calibration command can export an audited failed model and exit 1. The
runner retains that failure as `completed_with_failures` and proceeds using
the existing panel's missing-model policy. A process that fails without a valid
audit stops the run. Completed scientific failures are not automatically retried
until an improved experiment is configured in a new output folder. At the end,
`completed_with_failures` returns a nonzero exit code.

If a month has no matched entries with both deltas ready, its panel and coverage
remain saved, with `no_matched_ready_comparisons` in the month index. Comparison,
attribution and notebook generation are explicitly skipped rather than creating
an empty performance claim.

## Fresh reproduction and profiling

To deliberately rerun September rather than adopt checkpoints, use a separate
output directory:

```bash
PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_september.json \
  --ignore-checkpoints \
  --output outputs/research_pipeline/september_fresh_v1
```

This is expensive at the production grid settings. First use a short date
window in a separate configuration to measure calibration and PDE time.
Execution is sequential; no untested parallelism or automatic parameter search
is added. Full historical calibration and hedge PDE runs have not been performed
by this update.

## Another month and cross-month observations

Copy the JSON to a new configuration, change `start`, `end` and `output`, and
remove September's `checkpoints`. Keep model and hedge parameters frozen for
the next evaluation. `start`/`end` are inclusive entry dates, with work split
into monthly blocks.

With `include_next_session: true`, preparation also reads the next scheduled
reference session and its monthly file when necessary. Only dates inside the
entry window are calibrated and selected as entries. The additional quotes
are available for observed endpoint matching. Missing or excluded next
sessions are not bridged, and future quote availability does not select entries.

With `include_next_session: false`, every month ends at its configured boundary;
its last entries remain sample-end observations. Use `true` for the later
historical study when those next-session observations are available.

The supplied policy must cover the entry and observation dates and the full
60-day expiry/calendar horizon. Your current 2013–2023 policy cannot support
late-2023 maturity filtering without additional **calendar coverage**. The
runner reports that limitation rather than inventing calendar rows. It does
not require future option observations to establish future calendar sessions.

## Optional numerical validation

`validation_dates` defaults to `[]` in the supplied checkpoint configuration:
no new expensive validation study is requested. This does not certify every
September entry's Greek. Add explicit dates inside the entry range to invoke
script 30 for those dates, retaining its raw/smoothed, space/time/shift/domain
diagnostics. Its default 12,000/24,000 grids and 64/128 steps per day are kept.
Completion means diagnostics were produced, not that every sensitivity passed.

The hedge panel remains radius 0.0005, width 0.75, maturities near 21/35/45 days,
log-spot moneyness near -0.02/0/+0.02, calls and puts. The JSON exposes only the
grid/step choices currently supported by script 32; it cannot silently change
the fixed coefficient convention or entry basket.

## Evidence and remaining research

Pipeline completion is operational. Contract identity, snapshot provenance,
actual funding curves, wing robustness and all-entry Greek validation are not
certified by it. The existing 3/5/7 percent pricing rates, synthetic index fills,
zero dividend cash and assumed funding fees remain assumptions. No observed
mark is replaced with a model price, and no missing historical data is filled.

Month reports retain calendar, quote, fit and Greek failures. Only the existing
common-ready-entry rule supplies paired hedge comparisons; failures remain in
coverage. Absolute P&L and zero-centred RMS retain their existing meanings.
No portfolio wealth curve, annualization, IID significance test or out-of-sample
claim is added.

Next research stages remain surface-derived and empirical minimum-variance
hedges, discount-input improvements, held-out evaluation, broader history,
multi-session rebalancing and dependence-aware statistical analysis. This update
provides the execution and saved-result structure for those additions.
