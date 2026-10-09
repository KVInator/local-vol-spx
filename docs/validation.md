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

The clean snapshot passed 262 unit tests, including checks on saved-output
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
and uncertainty review after it finishes. The quote-total-variance comparison
also needs a constrained construction before it can serve as a validated
alternative diffusion. AH calibration-bump and PDE-vega sensitivities remain
a separate unfinished study.

Those additions should be developed and checked separately from the running
historical study, with their inputs and settings recorded before comparison.
