# Full October research run with measured runtime

This run uses all eligible October 2023 entry dates under the existing research
selection policy. The entry window is October 2-31; November 1 supplies the
final next-session exit observations. Missing or excluded observations remain
explicit failures. No missing option quotes, spots, marks or deltas are filled.

The October 2-3 trial remains separate. Full October uses a new output folder
and executes fresh work, including those two dates. Its checkpoint map is empty.
October has already been partly inspected, so this remains a development
evaluation. It is not an untouched out-of-sample experiment.

## Install

| File | Destination in `local-vol-spx` |
|---|---|
| `37_run_measured_pipeline.py` | `scripts/37_run_measured_pipeline.py` |
| `research_pipeline_october_full.json` | `config/research_pipeline_october_full.json` |
| `test_measured_pipeline.py` | `tests/test_measured_pipeline.py` |
| This guide | `docs/october_full.md` |

All four files are new. Keep the existing scripts 26-36, source modules and
October-capable notebook builder unchanged. This launcher calls script 36 with
the active Python interpreter and streams its output to the terminal and a log.
No packaging or additional market-data dependency is introduced.

The existing `exchange_calendars` dependency is used for calendar support,
following the same XNYS cash-session convention as the historical audit.

## Frozen settings and validation scope

| Item | Setting |
|---|---|
| Entry range | October 2-31, 2023 |
| Final observation date | November 1, as selected by the existing reference policy |
| Calibration | Independent daily AH; 8,000 intervals; width 1.0 |
| Regularization / controls / evaluation limit | 1.0 / 31 / 300 |
| Carry | Existing 3%, 5%, 7% scenarios; 5% primary pricing rate |
| First-period diffusion blend | Existing radius 0.0005 |
| Entry-hedge PDE | Width 0.75; 24,000 intervals; 128 steps/day |
| Entry basket | Calls and puts near 21/35/45 days and log(K/spot) -0.02/0/+0.02 |
| Funding and fee scenario | Existing 5% lend/borrow and 1 bp hedge fee |
| Marks and hedge | Observed option marks; existing synthetic index hedge |
| Validation dates | October 2, 3, 16, 23, 31 |

The three additional validation dates are declared before inspecting their
results. The first two reproduce the reviewed trial dates. Every eligible date
gets calibration and entry-hedge calculations; script 30's expensive numerical
study runs on the five declared dates. It keeps its existing spatial, temporal,
shifted-grid, domain, raw/smoothed and forward/backward checks.

These checks use generated diagnostic contracts and numerical references.
Native AH refinement, economic wing robustness and independent validation of
every actual hedge entry remain separate research tasks. The month summary's
`all_entry_greeks_independently_validated` remains false.

The actual maturity selected for a target basket can differ from 21/35/45 days.
Retain actual maturities, original quote bands, parity flags, opposite-side
fallback flags and failures when interpreting the results.

## Prepare the calendar support

The historical policy was built from sessions and observed dates through 2023.
Its last row can be December 29 even though the audit ended December 31. The
existing pipeline and preparation code require policy coverage through the
full 60-day horizon after the last observation. November 1 plus 60 days is
December 31. The current checks therefore need additional calendar coverage.

The launcher creates a separate derived policy. It preserves every historical
row and appends calendar-only dates through January 31, 2024, using XNYS session
and close information. Appended rows have `session_policy=calendar_only_not_audited`;
they have no observed row counts, quotes, spots or timestamps. They cannot
become entry observations under the existing preparation rule.

This extends calendar support, not the market-data sample. Cash-session dates
remain a provisional reference, not verified SPX contract or fixing conventions.
The original policy is never overwritten. Source/output checksums and the
calendar-provider version are saved in `session_policy_audit.json` beside the
derived policy. Repeating preparation reuses the verified file; changed inputs
or outputs are rejected.

From the repository root, with `volspx` active:

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python scripts/37_run_measured_pipeline.py \
  --config config/research_pipeline_october_full.json --prepare-calendar

PYTHONPATH=src python scripts/37_run_measured_pipeline.py \
  --config config/research_pipeline_october_full.json --plan
```

The plan should show entry end `2023-10-31`, observation end `2023-11-01`, both
`spx_eod_202310.txt` and `spx_eod_202311.txt`, no checkpoints, and the five declared
validation dates. The files must already be present in `data/raw`. Missing
files or insufficient calendar coverage cause a clear failure.

`--prepare-calendar` only prepares the derived policy. `--plan` and `--status`
delegate the existing read-only operations and create no timing reports.

## Execute and measure

```bash
PYTHONPATH=src python scripts/37_run_measured_pipeline.py \
  --config config/research_pipeline_october_full.json
```

The actual command runs preparation, carry, daily calibration, the five
validation studies, every eligible date's hedge panel, comparisons, attribution
and notebook creation. Completion is operational, not a certification of hedge
outperformance. Failure statuses and nonzero child exit codes are retained.

The launcher measures elapsed wall time with `time.perf_counter`. The timed
block covers registration of the invocation, child process startup, pipeline
work, streamed terminal output and child termination/cleanup. Final timing-file
saving and the launcher's final printed summary are outside the timed block.
The measurement includes verification/reuse overhead when resuming. It does
not sum stage durations and does not use a runtime estimate.

Each invocation creates:

- `outputs/research_pipeline/october_full_v1/runtime/run_.../timing.json`
- `outputs/research_pipeline/october_full_v1/runtime/run_.../terminal.log`
- An entry in `outputs/research_pipeline/october_full_v1/runtime_index.csv`

`elapsed_wall_seconds` is the measured duration; UTC start/end timestamps are
metadata. `command_returncode` records the child result, and `timing_status`
describes the invocation. A zero exit at `--stop-after calibration` is a pause,
not full completion; consult the pipeline state and stage index.

To pause at a boundary and resume, use the same configuration and output:

```bash
PYTHONPATH=src python scripts/37_run_measured_pipeline.py \
  --config config/research_pipeline_october_full.json --stop-after calibration

PYTHONPATH=src python scripts/37_run_measured_pipeline.py \
  --config config/research_pipeline_october_full.json
```

Each invocation gets its own measured duration. Previously completed jobs are
verified and reused by script 36. A resumed invocation's duration is not the
duration of the original full computation. To measure active command time
across pauses, add finalized invocation durations; idle time between commands
is excluded. Keep failed invocations in that history.

Ctrl-C or SIGTERM requests a controlled child shutdown and retains measured
elapsed time plus available logs. A forced kill or power loss can leave a report
without a finalized duration; it is not filled with an estimate.

For a read-only progress check from another terminal:

```bash
PYTHONPATH=src python scripts/37_run_measured_pipeline.py \
  --config config/research_pipeline_october_full.json --status
```

## Review the result

Open `outputs/research_pipeline/october_full_v1/2023-10/research_results.ipynb`
after the pipeline completes. The existing date explorer provides one surface
per fitted valuation date; the saved summaries cover the full October sample.
Running the notebook to render its saved-results analysis has its own runtime
and is not included in pipeline notebook-generation time.

Return the terminal summary and these reports:

1. `runtime_index.csv`, `stage_index.csv`, `month_summary.csv`.
2. The aggregate daily calibration `model_manifest.csv`.
3. The five validation runs' `quote_fit.csv`, `sensitivity.csv`,
   `forward_backward.csv` and `forward_shapes.csv`.
4. The comparison and attribution summaries, daily contributions,
   leave-one-date-out results and notebook plots.

Use `run_index.json` and stage-index folders to locate the exact reports. Keep
midpoint error results separate from spread/fee effects. Examine signed bias,
actual maturity mix, selected-side price residuals, date concentration and
coverage alongside pooled RMS/MAE. Do not retune parameters after an unfavourable
date. Assumed pricing/funding rates, unverified identity/snapshot provenance,
zero dividend cash and synthetic index execution remain explicitly documented.
