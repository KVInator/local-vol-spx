# Working with the Clean Version

The original `local-vol-spx` folder is still running the 2013–2023 historical
study. Keep the clean version separate until that run finishes. Placing
`local-vol-spx-clean` inside the original folder is fine as long as the original
source, configuration and data files remain in place.

## Reviewing the Project

Open the clean folder in a separate VS Code window and use the `volspx` kernel
for `notebooks/research.ipynb`.

The historical results are being saved to:

```text
/Users/pranayvij/GitHub/local-vol-spx/outputs/historical_evaluation/full_2013_2023_v1
```

The notebook's loading cell shows the latest saved daily date and the number of
dates covered by the hedge summary. Those counts can differ during the run:
daily calibration is completed first, followed by hedge-path evaluation and
summary generation. Rerun the loading cell to refresh the progress information.

The simulation section reads the completed Black–Scholes and CEV experiment.
Earlier monthly pilot results can be loaded by setting `PILOT_INDEX` to their
saved `run_index.json`.

## Work During the Historical Run

The next priority is to measure the cost of the daily hedge calculation. The
frozen settings use 24,000 PDE space intervals and 128 time steps per day, so a
small contract basket can still require substantial computation. The cached
surface allows that calculation to be profiled without repeating calibration.

From the clean folder, the following command profiles up to three unique
strikes at one expiry. It uses the saved study's numerical settings and reads
the original outputs. It does not write to the study or start a second full
historical run. `DATE` defaults to the latest saved date; choose another indexed
date if that date has no fitted `UNKNOWN` surface.

```bash
PYTHONPATH=src python - <<'PY'
import cProfile
import pstats
from results import StudyResults
from hedging import DailyDeltas
from selection import ContractSelector, HedgeSettings

saved = StudyResults(
    "/Users/pranayvij/GitHub/local-vol-spx/outputs/"
    "historical_evaluation/full_2013_2023_v1"
)
DATE = saved.dates[-1]
ROOT = "UNKNOWN"
model = saved.model(DATE, ROOT)
entries = ContractSelector.checked_quotes(saved.dated_table(DATE, "decisions"))
entries = entries.loc[entries.root.eq(ROOT)]
expiries = entries[["expire_date", "assumed_maturity_years"]].drop_duplicates()
nearest = (365 * expiries.assumed_maturity_years - 21).abs().argmin()
expiry = expiries.iloc[nearest].expire_date
entries = entries.loc[entries.expire_date.eq(expiry)].sort_values(["strike", "kind"])
entries = entries.loc[entries.strike.isin(entries.strike.unique()[:3])]
carry = saved.carry(DATE)
panel = saved.freeze["identity"]["config"]["settings"]["panel"]

print("Date:", DATE, "expiry:", expiry, "contracts:", len(entries))
print("PDE intervals:", panel["intervals"], "steps per day:", panel["steps_per_day"])
profile = cProfile.Profile()
decisions = profile.runcall(
    DailyDeltas(HedgeSettings(**panel)).calculate,
    entries, carry, model, lambda _: None,
)
print(decisions.ah_status.value_counts().to_string())
pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(15)
PY
```

This is an additional calculation on the same machine, so it can temporarily
compete with the active run for CPU time. One expiry is enough for the first
profile. Its result measures that hedge-pricing stage, not the complete
daily calibration or the remaining full-history runtime.

Further work can proceed in the clean version:

1. Review the measured bottleneck and compare candidate improvements on the
   same inputs, checking prices, Greeks and status coverage as well as runtime.
2. Extend numerical checks across a small representative set of already fitted
   surfaces, including short maturities and large local-volatility values.
3. Develop the constrained total-variance comparison and AH calibration-bump
   sensitivity study as separate experiments with recorded settings.
4. Review completed simulation and monthly pilot outputs, figures and
   interpretation before the final historical results are available.

The original study keeps its frozen source and settings throughout these
checks. Applying a speed improvement to a running process would require a
separate decision after its effect has been measured and validated.

## After the Historical Run

Once the final outputs have been checked, the clean files can replace the old
source, scripts, tests and documentation at the repository root. The data,
outputs and Git history can be retained in the same repository.

An original-source commit should preserve the code that produced the historical
results. The old files can then be removed from the working folder and the
cleanup committed to GitHub.
