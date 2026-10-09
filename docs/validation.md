# Numerical Validation and Reproducibility

The project uses several checks because calibration fit, numerical accuracy and
historical hedge performance answer different questions. A close quote fit does
not establish that a hedge improves on Black, and a successful historical run
does not independently verify every Greek it produced.

This document describes the checks already completed and the limits of that
evidence. The market-data audit is covered in [Data](data.md); the empirical
findings are recorded in [Results and Interpretation](results.md).

## Pricing and Surface Checks

Black prices, implied-volatility inversion and finite-difference Greeks provide
basic pricing controls. Daily AH calibration records quote residuals,
conditioning information and price-shape checks.

Native AH checks examine monotonicity, vertical spreads, convexity, intrinsic
and upper bounds, and calendar ordering. The independent quote-total-variance
construction is checked separately for derivative support, calendar violations,
negative densities and ill-conditioned denominators.

An unsupported comparison is recorded as unavailable. It does not count as a
zero violation. The independent interpolation's remaining shape failures also
do not describe the native AH price surface.

## PDE Price and Greek Refinement

The saved-surface study varies space resolution, time resolution, domain width
and grid position. Spot-bump checks compare reported delta and gamma with
finite differences of prices. Comparisons also examine the effect of the
short-end variance blend.

These checks cover selected states and contracts. They do not establish
independent convergence of every historical delta and gamma. A wider
representative sample remains useful, particularly for short maturities and
states with large recovered local-volatility values.

### June 19, 2015 Short-Expiry Checks

The supplied June AH specification reproduces all 597 saved fitted prices to
within 5.1e-12 index points. Its largest seven-day coefficient was then checked
at strike 2,325. These checks keep the calibration parameters fixed; they do
not recalibrate on each grid.

| AH grid intervals | Recovered local volatility at strike 2,325 |
| --- | ---: |
| 4,000 | 288.6156% |
| 8,000, saved model | 289.2353% |
| 16,000 | 289.2572% |
| 32,000 | 289.2683% |

The peak persists under spatial refinement. The 32,000-interval reconstruction
changes the saved prices by at most 0.000411 points, with two prices still outside
their original bands. Direct nodal recovery and the logarithmic positive solver
agree to within 2.5e-12 relative variance on the first-expiry quoted domain.

At that strike, the coefficient changes from 289.24% immediately before the
seven-day pillar to 19.78% immediately after it. Prices are continuous there,
but the time derivative changes between maturity intervals. The quote at this
edge has no positive midpoint IV: the parity-converted call midpoint is below
zero, while its bid-ask band includes nonnegative prices. That leaves the small
wing time value weakly determined by the quote. It does not by itself explain
the whole peak or justify removing the observation.

The backward PDE check covers four seven-day calls: strikes at forward
log-moneyness -0.02, 0 and +0.02, and strike 2,325. It uses the saved 8,000-interval
AH model throughout. The existing validation cases compare 12,000 and 24,000
PDE intervals, 64 and 128 nominal steps per day, a half-cell shift, and log-domain
half-widths 0.75 and 0.90. Both the raw coefficient and the existing 0.0005
short-end blend are checked. Spot profiles span -1% to +1% in log spot.

Gamma sensitivity prompted one further 48,000-interval check at 128 nominal
steps per day, with and without a half-cell shift. For the blended coefficient,
the maximum changes at the original spot across the four contracts are:

| Comparison | Price change, points | Absolute delta change | Absolute gamma change |
| --- | ---: | ---: | ---: |
| 24,000 to 48,000 PDE intervals | 0.000139 | 0.00000336 | 0.000177 |
| Half-cell shift at 48,000 intervals | 0.0000575 | 0.00000111 | 0.000289 |

Prices and deltas move little in this sample. Gamma differences shrink with
refinement, but remain sensitive to grid position, bump width and the short-end
blend. The sampled profiles had no negative gamma or delta-bound violations.
This is evidence for selected short-expiry calculations, not independent
validation of all historical Greeks. Calibration uncertainty and AH
calibration-bump/PDE-vega sensitivities remain separate work.

## Constrained Total-Variance Checks

The SSVI comparison has been checked against flat Black prices and a known
skewed SSVI surface. Its analytical strike and time derivatives agree with
independent finite differences. Additional controls examine broad wings,
calendar ordering, positive density factors, expiry derivative sides, saved-model
round trips and preservation of the input quote tables.

Market checks used the original calibration-pinned quote and carry files for
1, 20 and 29 September 2023. All three optimizers converged. The sampled SSVI
surfaces had no calendar, density, price monotonicity, vertical-spread or convexity
violations. The analytic parameter constraints provide the static-arbitrage
guarantee; a finite sampling grid supplies a separate implementation check.

Quote adequacy remains unresolved. RMS residuals were 7.91, 5.25 and 5.23 original
half-spreads, with many fitted prices outside their original bands. The
[results document](results.md) records the counts. These outcomes do not support
using this SSVI fit as a validated replacement for the AH hedge diffusion.

To inspect one completed date from the historical cache, run from the clean
repository:

```bash
PYTHONPATH=src python scripts/validate_model.py \
  --method ssvi \
  --study ../outputs/historical_evaluation/full_2013_2023_v1 \
  --date 2015-06-19 \
  --output outputs/validation/ssvi_2015-06-19
```

This fits that date's saved quotes and writes its SSVI parameters, quote-fit
tables, shape checks and Dupire samples to the requested output folder. It does
not launch a hedge study. The audit retains a snapshot of the live study index
and checks the immutable consumed files, so a later indexed date can be added
by the running process without invalidating this comparison.

## Black–Scholes and CEV Controls

The controlled hedging experiment uses Black–Scholes and square-root CEV, with
analytical prices and Greeks and exact simulated transitions. It compares
analytical and PDE deltas across hedge frequencies, including a deliberately
misspecified volatility case.

The completed experiment used 20,000 paths per model and 4–256 nested rebalance
intervals. Reusing the nested paths makes frequency comparisons less sensitive
to differences in simulated shocks. The analysis reports error convergence,
mean P&L, funded fees and PDE checks.

Correct-delta RMS-error slopes were approximately -0.485 to -0.493. Their
proximity to -0.5 supports the expected replication pattern in these controls.
It does not establish that the same model fits or hedges historical SPX options
well.

## Hedge Accounting and Statistical Checks

The self-financing ledger tracks the option position, index hedge, cash funding,
traded notional, fees and final liquidation. Representative scalar paths are
compared with batch calculations, and reconciliation residuals are retained.

Historical statistics use matched candidate/Black entries. Coverage records
why entries are unavailable. The common-entry analysis checks the effect of
unequal strategy coverage, while date-block resampling addresses dependence
within entry dates and across overlapping holding periods.

## Clean-Code Verification

The original clean snapshot passed 262 unit tests, including checks on saved-output
integrity, copied study folders and directory symlinks. These tests use small
controlled inputs.

The comparison with the previous implementation covered an artificial
nine-reference-session study, a missing-session variant, prefix/resume behavior
and known-model controls. It found:

| Comparison | Outcome |
| --- | --- |
| Saved table comparisons | 115 exactly equal |
| Saved known-model arrays | 36 exactly equal |
| Daily AH model JSON | Exactly equal |
| Selected contract decisions | Exactly equal |

This establishes agreement on those controlled cases. The raw 2013–2023 SPX
history was not rerun through the clean implementation during that comparison.
The scope and hashes are recorded in [verification.json](../verification.json).

The notebook has also been checked against complete summaries, earlier
summaries, an active partial run and missing output folders. Those checks used
headless display capture. They do not verify IPython's notebook rendering.

## Remaining Validation Work

The full-history evaluation still needs final coverage, account reconciliation
and uncertainty review after it finishes. A constrained SSVI construction is
available, but its market fit needs improvement and its PDE prices and Greeks
have not been validated for historical hedging. AH calibration-bump and
PDE-vega sensitivities remain a separate unfinished study.

Those additions should be developed and checked separately from the running
historical study, with their inputs and settings recorded before comparison.
