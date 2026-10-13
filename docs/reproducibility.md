# Reproducibility and migration

Run commands from the repository root in `volspx`. Dependencies are declared
in `requirements.txt`; install them with `python -m pip install -r requirements.txt`
when preparing another environment. The source uses `PYTHONPATH=src`, and
notebooks launched through Jupyter use the absolute `PYTHONPATH="$PWD/src"`
so imports also work when a kernel starts in `notebooks/`. No notebook changes
`sys.path`. `config/notebooks.json` is the shared input configuration;
`SPX_NOTEBOOK_CONFIG` can select another file.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=scratch/cache/matplotlib \
  python -m unittest discover -s tests
PYTHONPATH="$PWD/src" MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  python -m jupyterlab
```

The notebooks are thin entry points to `SurfaceExplorer` and `HistoricalAnalysis`.
The explorer caches at most three recent selections and verifies their consumed
input pins on reuse. Missing dates and unavailable carry/models clear the
previous figures. Its date slider loads on release. Saved SSVI and numerical
diagnostics appear only on their recorded date and root; the validation
configuration supplies those producer folders. All cached paths resolve relative
to their relocated study or output folder, preserving historical audit paths.

## Retained inputs

| Material | Local location |
| --- | --- |
| Authoritative completed study | `scratch/runs/full_2013_2023_v1` |
| Raw quote files | `scratch/data/raw` |
| Original session policy, verified copy | `scratch/data/provisional_session_policy.csv` |
| Supplied market numerical evidence | `scratch/runs/market_validation` |
| Saved June SSVI comparison | `scratch/runs/ssvi_2015-06-19` |
| Scalar cache review | `scratch/runs/cache_review` |
| Original source/configuration and retired work | `scratch/archive/completion_2026-10-13` |
| Relocation journal and file checksums | `scratch/relocation.json` |

The study fingerprint is
`09c095fb8a89f2ab811b2bddb393ce0a9258a8d013095b8c23a9960dd6b4a232`.
Its `freeze.json` SHA-256 is
`b54c0bb421abb14fa64e1e223d796982312bbc34a7c02bb182829538a6207c7e`.
The notebook configuration checks both. Producer source, settings, stored
decisions and outcomes remain unchanged. The original environment recorded
Python 3.11.15, NumPy 2.4.3, pandas 2.3.3 and SciPy 1.17.1 in the freeze.

## Read-only cache diagnostics

Use a new output directory; the command rejects an existing populated directory.
It identifies the failed carry expiries and replays representative fixed-contract
ledgers at six specified entry dates. It reads cached observations and decisions.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python scripts/audit_history.py --study scratch/runs/full_2013_2023_v1 \
  --output scratch/runs/cache_review_repeat
```

Additional numerical experiments use one entry point, `scripts/validate_model.py`.
Methods are `ah`, `ssvi`, `greek-panel`, `calibration-vega` and `saved-decisions`.
`--workers` controls independent blend variants in the Greek workflow. These
commands perform explicit new numerical work; browsing never invokes them.

```bash
PYTHONPATH=src python scripts/validate_model.py \
  --study scratch/runs/full_2013_2023_v1 --date 2015-06-19 \
  --method greek-panel --workers 4 --output scratch/runs/june_panel_repeat
```

`scripts/run_history.py --plan` inspects a potential reproduction without running
it. `config/history_2013_2023.json` directs new work to a separate output scope.
Reading the original cache is supported; resuming its frozen run with reorganized
source is a different operation. No full-history reproduction was performed
during finalization.

## Reversibility

The migration checked active writers and locks, compared the reference with the
nested checkout, and recorded the initial Git status. Every core source file in
the nested checkout matched the supplied reference. Missing post-history helpers
were integrated into the retained validation workflow; the mixed notebook and
unused pilot reader were retired after checking callers.

All 45,131 completed-study files and 133 raw-directory files (132 quote files and
one metadata file) retained their file sizes
and SHA-256 checksums across relocation. The archived original producer source
matches every source pin in the freeze. Summary files and consumed caches retain
their original producer pins. The supplied reference base is
`3a797ae0db9aa9ab62439c8996bd632f3f966240`; the original root checkout began at
`3cc3f64`. The journal records individual old/new paths and verification.

To reverse relocation, first close readers, then move each journal destination
back to its original location in reverse order. Preserve the integrated source
in a separate sandbox directory before restoring an occupied source path.
The nested checkout, including its own `.git`, remains in the archive. The
repository's outer `.git` and existing unrelated sandbox contents were preserved.
No reset, history rewrite, deletion of unknown files or GitHub push is required.
