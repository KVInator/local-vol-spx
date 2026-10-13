# Saved cache diagnostics

## Carry exclusions

The cache contains 30 excluded dates and 31 unavailable expiry
records: 20 dates in 2016 and 10 in 2022. Every bad expiry records
`free_fit_status=insufficient_pairs`, `conditional_fit_status=unavailable` and
`carry_status=unavailable`. Their retained parity-fit support is three to five
pairs. Other available quote pairs do not satisfy the fit's support selection.
May 27, 2022 has two affected expiries. No unavailable expiry was removed.

[The expiry table](tables/carry_exclusions.csv) records every date, expiry,
retained pair count, available pair count and status. These are recorded input
support failures before AH optimization, rather than failed AH optimizer fits.

## Fixed-contract tail paths

The diagnostic selects each requested date/method's contract with the largest
signed excess midpoint SSE over matched Black, for five-session holdings and
daily rebalancing. This is an outcome-selected explanation, not a new hedge
experiment. A largest excess can still be negative when the candidate improves
every matched contract on that date.

Twelve selected method/contract paths were replayed against Black under all
three execution scenarios: 36 paired scenario checks, or 72 scalar ledgers.
The maximum difference from cached P&L, costs, funding, turnover and component
P&L is 1.48e-12 index points. The paths use saved
observations and rebalance decisions; no model was recalibrated or repriced.

| Entry date | Method | Candidate P&L | Black P&L | Entry interval Δhedge P&L | Later intervals Δhedge P&L | Δfunding |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2020-03-02 | AH PDE | -5.34 | -67.31 | +15.92 | +45.54 | +0.50 |
| 2020-03-02 | Sticky delta | -132.73 | -78.92 | -13.07 | -40.27 | -0.46 |
| 2020-03-03 | AH PDE | -29.76 | -59.77 | -18.43 | +48.03 | +0.41 |
| 2020-03-03 | Sticky delta | -95.76 | -62.83 | +20.21 | -52.70 | -0.43 |
| 2020-03-16 | AH PDE | +112.08 | +89.46 | -19.31 | +41.61 | +0.32 |
| 2020-03-16 | Sticky delta | +43.00 | +56.18 | +14.51 | -27.47 | -0.22 |
| 2020-03-26 | AH PDE | +57.79 | +37.09 | +12.37 | +7.99 | +0.33 |
| 2020-03-26 | Sticky delta | +13.50 | +27.50 | -10.46 | -3.19 | -0.35 |
| 2023-09-19 | AH PDE | +11.81 | -9.80 | +5.47 | +15.59 | +0.55 |
| 2023-09-19 | Sticky delta | -28.25 | -12.70 | -4.17 | -11.02 | -0.35 |
| 2023-11-09 | AH PDE | -27.07 | -9.80 | -8.40 | -9.34 | +0.47 |
| 2023-11-09 | Sticky delta | +7.09 | -7.20 | +6.85 | +7.80 | -0.35 |

Values are index points for a unit short option. Candidate minus Black P&L
equals entry-interval plus later-interval hedge P&L differences, plus funding
difference, less cost difference. Observed option-mark P&L is identical within
each pair and cancels in this decomposition. Later contributions use each
saved chronological rebalance, rather than extrapolating the entry delta.
The sign and relative contribution change across dates; these checks do not
attribute entire multi-session errors to an entry-grid defect.

[The scenario table](tables/tail_ledger_summary.csv) retains contract identity,
entry delta differences, observed option-mark P&L, funding, costs, replay errors
and paired daily SSE/counts. `scratch/runs/cache_review/tail_intervals.csv`
records individual decisions and spot/mark changes; `tail_coverage.csv` records
completed and excluded paths for all methods at these entry dates. Observed
marks, exclusions and funding are kept separate from numerical entry-date
refinements described in [validation](validation.md).

The curated tables derive from `scratch/runs/cache_review/audit.json` and its
output pins. [Reproducibility](reproducibility.md) provides the read-only replay
command. Producer files and frozen historical outcomes remain unchanged.
