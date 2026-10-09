# SPX local volatility and delta hedging

A surface can fit today's option prices and still give a poor hedge tomorrow.
This project compares those two questions using SPX end-of-day quotes from
2013–2023.

Andreasen–Huge is the main calibration. Its recovered local variance feeds a
backward PDE, whose delta is compared with Black delta, three smile conventions
and a correction fitted from past hedge errors. Each historical trade keeps the
same contract while its hedge is recalculated from the current day's information.

The September and October development studies gave mixed results. AH lowered RMS
hedge error in September and increased it in October. Those are useful findings;
the model is evaluated on its hedge errors rather than assumed to beat Black.
The full-history study is still being run in the original checkout.

## Start here

This is a separate source snapshot. Leave the checkout running the frozen study
alone. Open this folder separately to read the code or use its notebook.

[research.ipynb](notebooks/research.ipynb) brings the saved daily surfaces,
diagnostics, historical results and known-model experiment together. Set its
paths to the outputs in the original checkout. It reads those results directly;
there is no need to repeat the historical run.

For the explanation, read [methods](docs/methods.md),
[assumptions](docs/assumptions.md) and [results](docs/results.md).

## How the code fits together

| Responsibility | Main object | Source |
| --- | --- | --- |
| Prepare and cache quotes | `QuoteHistory` | `market_data.py` |
| Estimate carry and assemble calibration quotes | `CarryInputs` | `carry.py` |
| Select entry contracts | `ContractSelector` | `selection.py` |
| Calibrate and calculate daily hedges | `HedgeDecisionEngine` | `calibration.py` |
| Plan reference-session paths | `FixedContractPaths` | `paths.py` |
| Evaluate strategies along a path | `HedgePathEvaluator` | `paths.py` |
| Account for positions, cash and costs | `SelfFinancingHedgeLedger` | `ledger.py` |
| Accumulate paired statistics | `GainAccumulator` | `inference.py` |
| Coordinate a historical study | `HistoricalStudy` | `study.py` |
| Inspect saved results | `StudyResults` | `results.py` |

`pricing.py`, `surface.py` and `pde.py` contain the numerical models.
`diagnostics.py` and `validation.py` check surface derivatives and PDE refinement.
`simulation.py` contains the controlled Black–Scholes/CEV experiment.
Checksums and checkpoints have one implementation in `artifacts.py`.

The five scripts parse arguments and call these objects. Tests are grouped by
pricing, surfaces, PDEs, data, hedging, accounting, studies and analysis. They use
small artificial inputs; the numerical comparisons against the previous code
are recorded in [verification.json](verification.json).

The earlier spline, tail and numbered pilot investigations remain in the
original checkout. They are not copied into this main-workflow snapshot.

## Environment

Use the existing `volspx` environment, or Python 3.11/3.12 with
[requirements.txt](requirements.txt). The source is a flat `src` folder. There is
no package installation step.

From this folder:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Commands

| Command | Purpose |
| --- | --- |
| `scripts/audit_data.py` | Audit monthly files and prepare the session policy |
| `scripts/run_history.py` | Prepare, freeze, run or resume a historical study |
| `scripts/run_simulation.py` | Run the controlled replication experiment |
| `scripts/validate_model.py` | Refine one saved surface's PDE grids |
| `scripts/export_results.py` | Export saved hedge results and figures |

For a fresh study, put the monthly files in `data/raw` and run the audit. If the
audited policy already exists in another checkout, set `session_policy` in the
configuration to its absolute path instead. `raw_directory` accepts an absolute
path too, so the raw files do not need to be copied.

```bash
PYTHONPATH=src python scripts/audit_data.py
PYTHONPATH=src python scripts/run_history.py --prepare-calendar
PYTHONPATH=src python scripts/run_history.py --plan
PYTHONPATH=src python scripts/run_history.py --freeze
PYTHONPATH=src python scripts/run_history.py
```

The default output is `outputs/history/refactor_2013_2023`, separate from the
original study. Changing source, settings or inputs requires a fresh study
directory. Reading the original outputs with the notebook requires no new freeze.

Raw quotes, cached models and generated outputs are excluded from Git. The
historical study uses a synthetic fractional index hedge and assumed carry,
funding and fees. Its P&L measures hedge error under those conventions.
