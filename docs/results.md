# Results and Interpretation

The completed studies cover the market-data audit, September and October 2023
hedge comparisons, and controlled Black–Scholes/CEV simulations. The 2013–2023
historical evaluation is still running. Final full-history Gain estimates and
uncertainty intervals are therefore not reported here.

The [methodology](methods.md) defines the comparisons, while
[validation](validation.md) records the numerical checks behind them.

## Market Data Coverage

The audit covers 132 monthly files and 14,750,241 raw rows. Of 2,768 reference
sessions, 2,744 have observations and 24 are missing. A further 21 observed
early-close dates are flagged for clock review.

These counts describe the input history, not the number of eligible hedge
entries. Contract identity remains provisional because the files do not
establish verified SPX/SPXW roots. Details are in [Data](data.md).

## September and October Hedge Comparisons

The monthly comparisons use one-session holding periods, observed midpoint
option marks and the synthetic index hedge. Errors are in option price units.

| Month | Matched entries | AH PDE RMS error | Black RMS error | Black MAE minus AH MAE |
| --- | ---: | ---: | ---: | ---: |
| September 2023 | 342 | 2.740419 | 3.130833 | +0.034033 |
| October 2023 | 390 | 3.549799 | 3.450136 | +0.046486 |

AH reduced RMS error in September and increased it in October. October's MAE
comparison was slightly favorable to AH despite its higher RMS error. The
result depends on the measure used: squared errors give more weight to larger
outcomes than absolute errors do.

Leave-one-date-out checks could change the sign of pilot findings. These short
samples therefore support a mixed assessment of AH hedging rather than a stable
claim of improvement over Black.

## Expanded October Strategy Comparison

The funded midpoint comparison adds smile-based and empirical hedge rules.
Gain measures the reduction in squared hedge error relative to Black on matched
entries.

| Strategy | Matched entries | Gain |
| --- | ---: | ---: |
| AH PDE | 390 | -5.86% |
| Empirical MV correction | 390 | +25.29% |
| LV smile approximation | 386 | +11.44% |
| Surface sticky strike | 386 | +0.13% |
| Surface sticky delta | 386 | -185.53% |

The empirical correction reduced squared error by 25.29% in this comparison.
The LV smile approximation also improved it, while sticky strike changed it
only slightly. AH PDE and sticky delta increased squared error.

Four entries failed the smile-stencil checks, leaving the smile methods with a
smaller matched sample. These are development results from October; they do
not establish the same strategy ranking across the full history.

## Surface Findings

All 22 October AH models fitted. The independent quote-PCHIP construction still
contained negative densities and some negative calendar derivatives, with gaps
in recovered local volatility where derivative conditions failed.

That construction remains a diagnostic comparison. A separate constrained SSVI
surface is now available, with analytic conditions for calendar and butterfly
arbitrage. Its first market checks used the original September quote and carry
inputs on three dates:

| Quote date | Quotes | Expiries | RMS half-spreads | Outside original bands |
| --- | ---: | ---: | ---: | ---: |
| 2023-09-01 | 2,442 | 24 | 7.9092 | 2,062 |
| 2023-09-20 | 2,432 | 26 | 5.2548 | 1,767 |
| 2023-09-29 | 2,627 | 25 | 5.2338 | 1,722 |

All three optimizers converged, and the sampled surfaces passed the calendar,
density and price-shape checks. Every checked source quote entered calibration;
prices were not clipped into their bands. The large residuals show that these
calibrated SSVI surfaces do not describe the quotes adequately. Optimizer
convergence does not establish that a global minimum was found. The current
fits are not accurate enough to replace the AH hedge model.

This completes a constrained construction, with quote adequacy and PDE hedge
validation still outstanding. The comparison has not changed the frozen
historical calculation. Its formulas and scope are described in
[Methods](methods.md) and [Validation](validation.md).

### June 19, 2015 Comparison

The notebook now compares AH and SSVI on the same 597 quotes across eight
expiries. Price residuals include all quotes, including the 31 midpoints that
do not have a positive implied volatility.

| Model | RMS original half-spreads | Outside original bands |
| --- | ---: | ---: |
| AH | 0.2536 | 2 of 597 |
| SSVI | 1.6935 | 329 of 597 |

AH reproduces these prices more closely. SSVI produces a smoother local-volatility
surface, but that smoothness comes with a substantially worse quote fit in this
comparison. Neither observation establishes which hedge will perform better.

The six-panel figure also shows why recovering local volatility from a quote
interpolation needs care. Quote-based Dupire values are available at about 54%
of its 10,695 plotting states and reach 2,397.58%. The largest value comes from
a small positive density denominator. Availability means the calculation passed
its support and sign checks; it does not certify derivative stability. Gaps
remain where those checks fail.

Recovered AH local volatility ranges from 5.80% to 289.54% on the same grid,
while SSVI ranges from 7.14% to 43.64%. The large AH value occurs in the
seven-day high-strike wing. At strike 2,325, within that expiry's quoted range,
it is 289.24%. This coefficient persists when the AH grid is refined with its
fitted parameters fixed. It is a property of this fitted model, rather than a
plotting error or a forecast of SPX volatility.

Selected seven-day PDE checks found small price and delta changes under further
refinement. Gamma remains more sensitive to grid position and the short-end
blend. [Validation](validation.md) records the contracts, settings and limits of
these checks. The June comparison is separate from the latest historical date
displayed elsewhere in the notebook.

## Controlled Simulation

The completed Black–Scholes and square-root CEV experiment used 20,000 paths
per model, with 4–256 nested rebalance intervals. Correct-delta RMS-error slopes
were approximately -0.485 to -0.493, close to the inverse-square-root reference
of -0.5.

Analytical and PDE calculations were compared under grid refinements, and
representative scalar ledgers were checked against batch P&L calculations.
Hedge fees had a clearer effect at higher rebalancing frequencies. The notebook
shows the frequency comparison alongside funded costs and mean P&L.

These controls support the numerical and accounting calculations in their
known-model setting. They do not establish a historical SPX hedging advantage.

## Full-History Evaluation

Daily models and broader hedge summaries are produced in separate stages.
During the active run, the latest saved daily date can be ahead of the
hedge-summary scope. A pending holding-period path is distinct from a failed
calibration or a missing market observation.

The final evaluation will need the following review:

- matched Gain, RMS error, MAE and mean P&L across holding schedules and scenarios
- coverage reasons and common-entry comparisons
- maturity, moneyness, delta and year breakdowns
- uncertainty estimates and sensitivity to date-block length
- hedge-account reconciliation and recorded study provenance

## Current Assessment

The completed market-data comparisons show mixed AH hedge performance. The
empirical and LV smile corrections were favorable in the expanded October
comparison, but their wider historical performance is still being evaluated.

Two model studies also remain unfinished: improving and validating the market
fit of the constrained independent surface, and an AH calibration-bump/PDE-vega comparison.
The conclusions above are limited to the calculations already completed and
the [assumptions](assumptions.md) under which they were produced.
