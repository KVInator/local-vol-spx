# Fresh October research trial

This trial executes the existing pipeline on October 2 and 3, 2023, with
the reviewed September calibration, hedge and funding settings frozen.
October 4 is included for the final next-session exit observations. The dates
are fixed before inspecting their hedge outcomes. This is a small evaluation
and runtime trial, not evidence of broad hedge outperformance.

## Install

| Delivered file | Repository destination |
|---|---|
| `35_build_pilot_notebook.py` | `scripts/35_build_pilot_notebook.py` (replace) |
| `research_pipeline_october_trial.json` | `config/research_pipeline_october_trial.json` (new) |
| This guide | `docs/october_trial.md` (new) |

The notebook update selects the first successfully fitted date and root in
the current manifest. It removes the hardcoded September 1 explorer date.
Pricing, calibration, contract selection and accounting formulas are unchanged.

The trial reads `data/raw/spx_eod_202310.txt` and the existing provisional
2013-2023 session policy. That policy must cover the entry dates, next reference
session and 60-day expiry/calendar horizon. No raw quotes or calendar rows are
created or filled by this update.

The output directory is separate from the September checkpoint. The existing
September models, results and notebook are not modified. Because the runner
pins source hashes, installing the notebook update changes the identity of
future runs. To inspect the previous September run, use its `--status` command.
If re-registering that checkpoint under the new source, use a new output folder
with `--output`; do not overwrite the existing September state.

## Frozen experiment

| Item | Setting |
|---|---|
| Entry dates | October 2-3, 2023 |
| Additional observation date | Next reference session after October 3, expected October 4 |
| Daily calibration | Independent same-date AH fits, 8,000 intervals, width 1.0 |
| Calibration regularization / controls | Smoothing 1.0; 31 controls; 300 evaluations |
| Pricing carry | Primary assumed annual rate 5%; prepare existing 3/5/7% scenarios |
| Hedge diffusion | First-period Gaussian variance blend, radius 0.0005 |
| Hedge PDE | Width 0.75; 24,000 intervals; 128 steps/day |
| Entry basket | Calls and puts near 21/35/45 days and log(K/spot) -0.02/0/+0.02 |
| Holding period | Entry delta held to next reference session, then liquidation |
| Funding / hedge fee scenario | Existing assumed 5% lend/borrow; 1 bp hedge fee |
| Entry and exit option marks | Observed vendor quotes |
| Index hedge / dividend cash | Existing synthetic index convention; zero dividend cash |
| Saved September checkpoints | None adopted for October |

The additional observation date does not become an entry or calibration date.
The carry-preparation stage can produce same-date carry rows for observation
dates, but entry calculations use only the entry date's inputs. Missing or
excluded next-session quotes remain missing; the runner does not bridge gaps.

## Run

Use the existing `volspx` environment from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_october_trial.json --plan

PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_october_trial.json
```

The plan should show one month, entry start `2023-10-02`, entry end
`2023-10-03`, observation end `2023-10-04`, the October raw file, no checkpoints,
and both explicit validation dates. If actual policy coverage differs, review
the discrepancy before changing the dates or policy.

The full command executes preparation, carry, daily calibrations, daily
numerical studies, entry deltas, observed-mark hedge comparisons, attribution
and notebook generation. Fresh stages should show `RUN` and `origin=executed`.
Aggregate stages and notebook creation have their own origins.

The two validation studies are computationally substantial: each compares raw
and smoothed diffusions across five PDE cases and up to nine generated calls.
The command logs progress and preserves completed dates. Do not change grid
settings midway through the run.

To pause after calibration and profile that stage:

```bash
PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_october_trial.json --stop-after calibration
```

Resume with the full command. Completed jobs are reused; interrupted jobs get
a new attempt folder. To inspect progress without launching computations:

```bash
PYTHONPATH=src python scripts/36_run_research_pipeline.py \
  --config config/research_pipeline_october_trial.json --status
```

## Numerical validation

Both dates invoke existing script 30 with its unchanged defaults. It checks:

- Raw AH and radius-0.0005 smoothed diffusions separately.
- Spatial refinement from 12,000 to 24,000 intervals on width 0.75.
- Time refinement from 64 to 128 steps/day.
- A half-cell state shift and width 0.90 with unchanged spatial spacing.
- Forward and backward consistency on generated calls.
- Modified-diffusion quote fit, conditional bounds and sampled price shapes.

Contracts use the first pillar, a pillar near 35 days and the final pillar,
at forward log-strikes -0.02/0/+0.02. These generated calls are diagnostic
contracts and are not necessarily the exact selected hedge entries. Numerical
references are not analytical Greek truth. Native AH grid refinement and
economic wing stresses are not performed by script 30. The month summary's
`all_entry_greeks_independently_validated` therefore remains false even after
both numerical studies complete.

## Outputs to review

All new outputs sit under `outputs/research_pipeline/october_trial_v1/`:

| Output | Purpose |
|---|---|
| `stage_index.csv` | Status, origins, elapsed seconds, attempts and exact folders |
| `month_summary.csv` | Dates, fit failures, entries, matched comparisons and validation scope |
| `run_index.json` | Exact artifact locations for preparation through attribution |
| `2023-10/date_coverage.csv` | Entry-window policy and quote availability |
| `2023-10/calibration/model_manifest.csv` | Daily fit and shape diagnostics |
| `2023-10/research_results.ipynb` | October date explorer and hedge results |

Each validation row in `stage_index.csv` points to a dated results folder.
Inspect its `study_status.csv`, `quote_fit.csv`, `sensitivity.csv`,
`forward_backward.csv`, `forward_shapes.csv`, `conditioning.csv` and `audit.json`.
If a date fails, keep the failure and its process log. Do not replace failed
results with synthetic values or remove an unfavourable date.

For profiling, use the elapsed seconds of the executed per-date calibration,
validation and panel jobs. Verification, aggregation and notebook-generation
times are not estimates of historical computation cost. Two dates provide an
initial cost estimate; their expiry and quote counts may differ from later dates.

With the full entry basket, this trial can produce 36 entries and 36 matched
comparisons. Counts depend on current-date quote availability, successful fits
and deltas, and observed endpoints. Coverage tables are the authority.

Return the terminal output, stage and month summaries, daily calibration
manifest, both validation studies' main tables, and the notebook plots. The
review should separate calibration residuals, numerical sensitivity and hedge
performance. Examine midpoint RMS, MAE, signed bias, subgroup results and date
contributions together. Two dates cannot establish statistical superiority.

The next expansion is a separately configured full October evaluation after
reviewing this trial. Preserve the trial and the frozen baseline. Any parameter
change informed by October outcomes creates a development experiment and needs
a later untouched evaluation period.
