# Reproducibility and migration

Run commands from the repository root with Python 3.11. The local completed
study uses the external `volspx` environment. A fresh clone can use an environment
outside the repository or create one under the ignored sandbox:

```bash
python3.11 -m venv scratch/.venv
source scratch/.venv/bin/activate
mkdir -p scratch/cache/{pip,matplotlib,ipython,tmp} scratch/runs/verification
mkdir -p scratch/cache/jupyter/{config,data,runtime}
PIP_CACHE_DIR="$PWD/scratch/cache/pip" python -m pip install -r requirements.txt
```

`requirements.txt` declares supported dependency ranges. The saved freeze
records the exact versions used for the historical study; installing current
versions within these ranges does not recreate the frozen producer environment.
The source uses `PYTHONPATH=src`, and
notebooks launched through Jupyter use the absolute `PYTHONPATH="$PWD/src"`
so imports also work when a kernel starts in `notebooks/`. No notebook changes
`sys.path`. `config/notebooks.json` is the shared input configuration;
`SPX_NOTEBOOK_CONFIG` can select another file.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  IPYTHONDIR="$PWD/scratch/cache/ipython" TMPDIR="$PWD/scratch/cache/tmp" \
  python -m unittest discover -s tests
```

The data-free suite constructs synthetic quotes and temporary study fixtures.
It needs no private data or saved historical run. The focused
[GitHub Actions workflow](../.github/workflows/verification.yml) runs this suite
on Python 3.11 using the declared dependencies, with logs and runtime caches
under `scratch/`. It does not calibrate or rerun the historical study.

## Notebooks and private saved inputs

The public clone includes publication tables and figures, but excludes the raw
quote files and completed multi-year cache. Opening the notebooks in Jupyter is
supported on a fresh clone; executing their saved-study loading cells requires
the retained inputs below or an equivalent pinned configuration. A missing
required study raises an input error; no dataset is downloaded or manufactured.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  IPYTHONDIR="$PWD/scratch/cache/ipython" \
  JUPYTER_CONFIG_DIR="$PWD/scratch/cache/jupyter/config" \
  JUPYTER_DATA_DIR="$PWD/scratch/cache/jupyter/data" \
  JUPYTER_RUNTIME_DIR="$PWD/scratch/cache/jupyter/runtime" \
  TMPDIR="$PWD/scratch/cache/tmp" python -m jupyterlab \
  --FileCheckpoints.checkpoint_dir="$PWD/scratch/cache/notebook-checkpoints"
```

The notebooks are thin entry points to `SurfaceExplorer` and `HistoricalAnalysis`.
The explorer caches at most three recent selections and verifies their consumed
input pins on reuse. One requested-date state synchronizes its picker, slider
and navigation. Missing dates and unavailable carry/models clear the previous
figures; changing display controls cannot restore an earlier valid date.
Its date slider loads on release. Saved SSVI and numerical diagnostics appear
only on their recorded date and root. Missing optional producer folders are
reported as unavailable; existing producers with invalid audits, changed checksums
or a mismatched study identity raise errors. The validation configuration
supplies those producer folders. All cached paths resolve relative
to their relocated study or output folder, preserving historical audit paths.

## Render the standalone report

[docs/report.md](report.md) is the authoritative report source;
[docs/report.yaml](report.yaml) supplies its typesetting settings. The title,
author and date are defined in the Markdown front matter.
`docs/report.lua` assigns result-table widths and keeps execution-case tables
with their headings.
The tracked figures and tables suffice to typeset the report on a fresh clone.
Install native executables for [Pandoc](https://pandoc.org/installing.html) and
[Tectonic](https://tectonic-typesetting.github.io/book/latest/installation/)
on `PATH`. On macOS with Homebrew:

```bash
brew install pandoc tectonic
pandoc --version
tectonic --version
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  IPYTHONDIR="$PWD/scratch/cache/ipython" \
  python scripts/build_publication.py --report
```

The publication was rendered with Pandoc 3.12.1 and Tectonic 0.17.0.
Use binaries built for the host architecture; a stale incompatible executable
earlier on `PATH` can prevent rendering. Tectonic downloads its TeX bundle on
first use, requiring network access. `ReportRenderer` places that cache in
`scratch/cache/tectonic`, and generated TeX, logs and intermediate PDFs in
`scratch/build/report`. Only the completed selectable-text PDF is copied to
[docs/report.pdf](report.pdf). Inspect every rendered page after editing the
source, figures or typesetting settings, including equations, tables and links.
The renderer records consumed source/figure hashes, tool versions and the PDF
checksum in [report provenance](tables/report_provenance.json).

Regenerating the curated figures and summary tables requires the configured
completed study. It reads saved surfaces and aggregates through the existing
explorer and historical-analysis classes; it does not rerun calibration or history:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  IPYTHONDIR="$PWD/scratch/cache/ipython" \
  python scripts/build_publication.py --figures
```

This command intentionally replaces the curated publication assets under
`docs/figures` and `docs/tables`, and records consumed input pins, source hashes,
figure scope and output checksums in
[publication provenance](tables/publication_provenance.json). Rerender the report
after regenerating its figures. Run either command separately, or use
`--figures --report` when both inputs are available.

## Retained inputs

| Material | Local location |
| --- | --- |
| Authoritative completed study | `scratch/runs/full_2013_2023_v1` |
| Raw quote files | `scratch/data/raw` |
| Original session policy, verified copy | `scratch/data/provisional_session_policy.csv` |
| Supplied market numerical evidence | `scratch/runs/market_validation` |
| Saved June SSVI comparison | `scratch/runs/ssvi_2015-06-19` |
| Original scalar cache review | `scratch/runs/cache_review` |
| Corrected scenario-scoped publication review | `scratch/runs/publication_cache_review` |
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
