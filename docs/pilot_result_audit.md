# Saved-results audit

This stage reviews the full October research run without recalibrating a model,
solving a PDE, loading a model JSON or rerunning the hedge ledger. It keeps the
complete original comparison and creates a separate audit notebook.

Install the files in the existing flat repository:

| File | Destination |
|---|---|
| `pilot_result_audit.py` | `src/pilot_result_audit.py` |
| `38_audit_pilot_results.py` | `scripts/38_audit_pilot_results.py` |
| `test_pilot_result_audit.py` | `tests/test_pilot_result_audit.py` |
| `pilot_result_audit.md` | `docs/pilot_result_audit.md` |

From the repository root, run:

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python scripts/38_audit_pilot_results.py \
  --run-index outputs/research_pipeline/october_full_v1/run_index.json \
  --month 2023-10 --plan

PYTHONPATH=src python scripts/38_audit_pilot_results.py \
  --run-index outputs/research_pipeline/october_full_v1/run_index.json \
  --month 2023-10
```

The plan verifies the explicit input files and their producer checksums, but
writes nothing. Execution creates a timestamped folder beneath
`outputs/pilot_result_audit/october_2023/`. The terminal prints its exact path.
Open `result_audit.ipynb` there and run its cells. The notebook reads the saved
audit tables and displays the four figures. It does not perform model fitting
or pricing. The existing research notebook and run index remain unchanged.

Default calibration focus dates are October 19, 26 and 27. The basket focus
date is October 11. Override these with `--focus-dates` and
`--selection-dates` to inspect other completed research months. Focus choices
only control diagnostic views; they do not filter performance or entries.

## What the audit checks

- Prepared targets and saved residuals must agree on their original bands,
  midpoint, parity conversion, forward and discount factor.
- Current-date observed call/put pairs are compared with the pinned carry.
  Their parity interval is `[call bid - put ask, call ask - put bid]`. The
  fixed strike window is reused from the carry producer's settings.
- Both original AH and cached smoothed prices are tested against each selected
  contract's **own** observed bid/ask. Original put prices use the saved call
  price minus `D * (F - K)`. A training quote from the opposite option side is
  never substituted for that contract's entry quote.
- An original price is available only at an exact saved calibration strike.
  Missing prices remain missing. There is no interpolation or model-price
  recomputation. Non-ready smoothed entries retain their failure status.
- Selection buckets distinguish unique entries, repeated targets and missing
  option sides. Repeated buckets must point to a real saved entry.
- Saved paired P&L and attribution must agree. Entries without a comparison
  remain in coverage. Longer-maturity contracts are displayed, not excluded.
- Cached numerical validation jobs are read using their explicit stage-index
  paths. Their calibration pins and dates must match this run. Missing jobs
  remain visible. No all-entry Greek certification is inferred.

Calibration diagnostics use only the dates and roots in the model manifest.
November 1 observations used for October's last exit are not treated as an
additional October calibration date. The original input files are still
verified in full.

The legacy preparation stage may lack output hashes. In that specific case,
its quote file must match the saved carry stage's input checksum. Missing
checksums for other consumed producer tables cause an error. Inputs are
checked again after report generation.

## Main outputs

| File | Purpose |
|---|---|
| `daily_audit.csv` | Original quote fit, carry flags, selected-contract price fit and saved date contributions |
| `expiry_audit.csv` | Fit by expiry, optimizer diagnostics, carry flags and observed parity intersections |
| `quote_audit.csv` | Every saved fitted quote residual, including selected-strike indicators |
| `parity_pairs.csv` | Same-date observed pair bands and the fixed-carry diagnostic |
| `selected_contract_audit.csv` | Own-band price checks for every original entry |
| `selection_buckets.csv` | Exact target-to-contract mapping and original bucket statuses |
| `selection_daily.csv` | Daily unique-entry, duplicate-bucket and unavailable-side counts |
| `selection_target_summary.csv` | Bucket counts by target maturity and original status |
| `midpoint_entry_audit.csv` | Entry diagnostics joined to saved midpoint attribution; unmatched entries retained |
| `long_maturity_entries.csv` | Descriptive view of actual calendar maturities from 40 to 45 days |
| `validation_*.csv` | Coverage and already-produced numerical diagnostics |
| `plots/` | Four figures for calibration, contract price fit and selection coverage |
| `result_audit.ipynb` | Saved-results review notebook with an editable focus date |
| `audit.json` | Input/output/source hashes, assumptions, summary and measured audit runtime |

For the first review, share the terminal summary, `daily_audit.csv`,
`expiry_audit.csv`, `selected_contract_audit.csv`, `selection_buckets.csv` and
the four plots. The full quote and parity tables remain available for a
follow-up inspection.

## Interpretation

Original AH quote fit, the changed smoothed diffusion's quote fit, numerical
sensitivity and realised hedge P&L are separate measurements. Smoothed minus
original price includes both a changed diffusion and numerical error.

Calibration RMS pools squared quote residuals using quote counts. It is not
an average of daily RMS values. Performance columns in the daily audit remain
date-level values; if multiple roots exist, those columns repeat across roots
and must not be added again.

Associations between poor calibration and hedge outcomes do not establish
causation. Focus dates were chosen after reviewing October and are not a
backtest eligibility rule. No tuning, selection changes, date exclusions,
statistical significance claims or out-of-sample claims are introduced.

The existing unverified identity and snapshot provenance, assumed pricing
carry and funding, and synthetic fractional index hedge conventions remain.
Completing the audit is not certification of Greeks or economic wing robustness.

Measured audit time covers input verification, analysis, figure and notebook
generation, and output hashing. The final audit JSON write and subsequent
notebook cell execution are excluded. It is separate from October's already
measured pipeline invocation.
