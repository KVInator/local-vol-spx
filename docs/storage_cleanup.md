# Local storage maintenance, November 2, 2026

The completed study was compressed and retired output trees moved into cold
storage. Calibration, hedge evaluation and the historical study were not rerun.
The clean `main` checkout was first fast-forwarded to its established upstream.
Process and open-file inspection found no study or notebook writers/readers.
Configuration, live readers, archived audits and producer source were inspected
before transforming study or evidence data.

## Summary exports and integrity

All sizes in this table are logical bytes. Gzip uses level 6 and zero modification
time; original audit pins describe decompressed content.

| Summary export | Original bytes | Gzip bytes | Parsed rows |
| --- | ---: | ---: | ---: |
| `strategy_results.csv` | 1,928,117,538 | 270,511,789 | 3,879,048 |
| `paired_results.csv` | 1,860,094,331 | 237,505,079 | 3,287,772 |
| `coverage.csv` | 846,683,649 | 20,921,756 | 4,672,500 |

Each original passed its existing `summary/audit.json` SHA-256 check before
compression. Each gzip stream was fully decompressed and hashed against that
same pin. The authoritative pinned reader then parsed every full compressed
table. All original and compressed hashes were rechecked before the three loose
CSV copies were removed.

`StudyResults.table` in `src/results.py` selects `.csv.gz` only when the original
`.csv` is absent. `PinnedInputs.read_pinned_csv` in `src/artifacts.py` validates
the decompressed bytes against the original pin and records the compressed
file's hash for later `verify_unchanged` checks. Corrupt gzip, changed CSV content,
missing pins and escaping symlinks are rejected. An existing invalid plain CSV
cannot be hidden by a valid gzip sibling. No dependency was added.

The freeze remains
`b54c0bb421abb14fa64e1e223d796982312bbc34a7c02bb182829538a6207c7e`;
the summary audit remains
`aba5e6ed9979a23bf977271b1ba9ac15dc0b976c8906ef2babe643e443c04fbb`.
The run index also retained its original bytes. Detailed container hashes,
original pins and verification results are recorded outside the frozen study in
`scratch/runs/storage_cleanup_2026-11-02/compression.json`.

## Retired evidence and disposable artifacts

The archived `development_outputs` and `processed_data` trees included unique
numerical diagnostics, old models and dependencies between pilot runs. Their
1,767 files were retained together in
`scratch/archive/completion_2026-10-13/retired_evidence.tar.gz`. Every archive
member's size and SHA-256 matched the pre-cleanup inventory; loose files were
rehashed again before removal. Their total logical size fell from 1,256,881,092
to 438,380,514 bytes. The archive SHA-256 is
`f6095c238288e71a743526cb069de5f0e998ba0c029b8f84e030f04f1cac605a`.
Live source, scripts, notebook configuration and notebooks do not read these
archived trees. Unique numerical evidence was preserved rather than inferred
obsolete from its age.

The following loose artifacts were removed:

- `scratch/github_checks`: 533 disposable GitHub rendering files, 128,517,007 bytes.
- `scratch/checkouts/publication`: 457 files, 56,358,074 bytes. It had no separate
  `.git` or registered worktree. Source differences were already reachable in
  retained Git history. Its two unpublished files, `docs/report.pdf` and
  `docs/tables/report_provenance.json`, were verified and retained in
  `scratch/archive/publication_2026-11-02/unpublished_renders.tar.gz`.
- `scratch/notebook_checks`: superseded notebook execution/render outputs;
  the repeated verification is retained in the storage journal's `notebooks/`.
- Old sandbox/publication bytecode and two Finder metadata files outside the
  sandbox.

**Retention exception:** the old publication runtime directory was removed
before its two notebook checkpoints were individually inspected. The
historical-analysis checkpoint was recovered byte-for-byte from Git into
`scratch/archive/publication_2026-11-02/retained_checkpoints`. The exact
3,166-byte `surface_explorer-checkpoint.ipynb` was not recovered; its recorded
SHA-256 was
`4365246774cb6a31e572f2b75a33fb3605b7271cf428cc44587983cae35957a8`.
Its contents were not verified before deletion. The maintained notebooks,
original frozen producer source and original producer configuration remain
intact. This exception prevents claiming that every historical source copy
was preserved.

## Verification and remaining storage

Post-cleanup hashes matched the baseline for all 45,130 unchanged study files,
134 data files, 50 current validation/cache-review files, 240 archived original
source/configuration files and 630 archived nested-checkout files. All 21 frozen
producer source pins match their archived originals, and the original historical
configuration equals the freeze configuration. Raw data, prepared monthly
quotes, per-date models/decisions and outcome caches remain available.

Eleven focused regression checks passed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  MPLCONFIGDIR=scratch/cache/matplotlib \
  python -m unittest test_compressed_results test_notebooks.AggregateChecks -v
```

Both maintained notebooks executed in actual `volspx` kernels, with study,
calibration and hedge-decision producers patched to fail if invoked. Checks
covered June 2015, March 2020 and September 2023, a carry-failure date, invalid
dates, navigation, date-matched optional evidence, domain/scale changes, historical
schedule/scenario/sample/grouping/block controls and March 2020 range views.
Representative surface and historical figures were visually inspected. The
repository hygiene audit reported zero errors and two review items for the
intentional maintained notebooks. The full numerical suite was not rerun.

Allocated usage and remaining directory sizes are recorded in
`scratch/runs/storage_cleanup_2026-11-02/space_report.json`. Savings are based on
`du` allocation measurements, which differ from logical gzip sizes and do not
measure filesystem snapshot retention. Net sandbox usage fell from 20.06 GB to
15.00 GB, recovering **5.06 GB (4.71 GiB)** after retaining verification outputs.

| Remaining material | Allocated GB |
| --- | ---: |
| Entire project, including Git | 15.10 |
| Completed study | 9.11 |
| Study cache within that study | 8.50 |
| Summary within that study | 0.61 |
| Raw quotes | 5.25 |
| All archives | 0.58 |
| Git history | 0.09 |

The main remaining usage is protected
raw data and study caches. Small handoff bundles and the old archived nested
Git checkout remain; these contain provenance/history and require a separate
uniqueness review before any further retirement.

## Restore a loose representation

Close readers first. These commands restore additional loose copies; retain
the archives until restored bytes have been verified.

```bash
gzip -dk scratch/runs/full_2013_2023_v1/summary/strategy_results.csv.gz
gzip -dk scratch/runs/full_2013_2023_v1/summary/paired_results.csv.gz
gzip -dk scratch/runs/full_2013_2023_v1/summary/coverage.csv.gz
tar -xzf scratch/archive/completion_2026-10-13/retired_evidence.tar.gz \
  -C scratch/archive/completion_2026-10-13
```

Verify restored summary SHA-256 values against the unchanged summary audit and
archive members against `retired_evidence.json` in the storage journal. Original
archived source/configuration stays loose and unchanged. The previous relocation
journal remains unchanged; consult this storage record when an old destination
now refers to compressed storage.
