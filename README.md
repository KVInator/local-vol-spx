# SPX local volatility and delta hedging

I built daily SPX volatility surfaces and a chronological fixed-contract hedge
study from 2013–2023 end-of-day option quotes. Andreasen–Huge supplies the main
price surface and recovered local variance; a backward PDE calculates prices
and deltas. Black, smile conventions and a past-only empirical correction share
the same observed contract paths and funded ledger.

The completed retrospective study selected 44,500 entries across 2,768 reference
dates. For five-session holdings, daily rebalancing and midpoint marks, AH
increased squared funded hedge error by 30.16% on entries shared by all methods.
The study does not support a stable AH hedge advantage over Black under its
conventions. [Results](docs/results.md) gives sample counts and uncertainty.

| Notebook | Purpose |
| --- | --- |
| [Surface explorer](notebooks/surface_explorer.ipynb) | Synchronized daily surfaces, quote fits, carry, saved deltas and numerical evidence |
| [Historical analysis](notebooks/historical_analysis.ipynb) | Completed history by schedule, execution, sample, grouping and block length |

Activate `volspx`, then launch from the repository:

```bash
conda activate volspx
PYTHONPATH="$PWD/src" MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" python -m jupyterlab
```

Both notebooks read the study selected in `config/notebooks.json`. Date browsing
loads saved models and performs surface diagnostics; it never recalibrates or
evaluates hedge paths. Data and runtime output remain in ignored `scratch/`.

See [data](docs/data.md), [methodology](docs/methods.md),
[assumptions](docs/assumptions.md), [validation](docs/validation.md) and
[reproducibility and migration](docs/reproducibility.md). The implementation
uses a flat `src/` core with thin scripts and notebook entry points.
