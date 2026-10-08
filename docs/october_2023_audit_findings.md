# October 2023 saved-results audit

Reviewed on 8 October 2026 from the supplied script-38 terminal output, two complete CSV-text attachments, notebook tables and all four figures.

## Scope and reproducibility

The local runner reported verification of 46 pinned input files and measured 3.255 seconds for the audit. The supplied test log reports 331 tests in 12.311 seconds without a failure report. The audit loaded saved results and ran no calibration, model pricing, PDE or hedge ledger. All 390 selected entries and 390 midpoint comparisons were retained. Every selected entry had an exact saved original AH calibration-strike price and a ready saved smoothed price.

Historical input identity and snapshot provenance remain unverified. Pricing discounting is the assumed rate_5pct scenario, and the hedge remains a synthetic fractional index position. This note does not certify executable returns or out-of-sample performance.

## Original calibration

There are 50,569 fitted equivalent-call targets across 22 dates and 515 expiry groups. Original calibration RMS is 0.341106 original half-spreads; 971 targets (1.920%) lie outside their original bands.

| Date | Quotes | RMS half-spreads | Outside original bands | Parity-incompatible expiry groups | Selected original / smoothed breaches |
|---|---:|---:|---:|---:|---:|
| 2023-10-19 | 2,374 | 0.932511 | 499 | 23 / 24 | 6 / 6 |
| 2023-10-26 | 2,399 | 0.455244 | 121 | 1 / 24 | 1 / 1 |
| 2023-10-27 | 2,432 | 0.786727 | 293 | 23 / 24 | 1 / 1 |

These three dates contain 14.248% of calibration targets, 94.027% of outside-band targets and 69.118% of squared calibration residuals. These concentration statistics are descriptive. The dates remain in every original performance comparison.

All 72 focus-date expiry fits report ftol termination, and none reports an active proxy bound. This is no evidence of simple evaluation-limit exhaustion or an active proxy cap on these cases; it does not prove global optimizer optimality.

## Parity under the assumed discounting

An expiry's near-spot observed call/put pairs imply forward intervals at the fixed discount factor. An empty intersection means that no single forward can satisfy every pair's original bid/ask bands at that discount factor. The complete expiry attachment confirms empty intersections in 23 groups on October 19, one group on October 26, and 23 groups on October 27.

For example, on October 19 the November 9 strike-4,365 contract has:

- Call bid/ask: 34.5 / 35.2 index points.
- Put bid/ask: 109.0 / 109.6 points.
- Admissible call-minus-put interval: [-75.1, -73.8].
- Pinned D(F-K): approximately -75.794245.

The pinned carry lies outside this interval. A model satisfying that parity cannot place both prices within those two observed bands simultaneously. The saved original prices, 35.517627 for the call and 111.311872 for the put, respect the pinned model parity and breach both observed bands.

This is a conditional inconsistency under the specified discounting and data. It does not identify its underlying cause, establish a vendor error, prove executable market arbitrage, or establish that another rate would resolve it.

October 19 also contains five calibration bands disjoint from the individual call-price bounds. October 26 has only one parity-incompatible group despite 121 calibration breaches. Joint shape constraints, regularization and other quote inconsistencies remain possible contributors; parity is not an explanation for every residual.

## Original versus smoothed selected-contract price fit

The own-side comparison uses each selected contract's actual call or put bid/ask. A price trained on the opposite option side is converted by the pinned parity before comparison. It is never tested against the training side's spread as a substitute.

Across all 390 entries:

| Measurement | Original AH | Saved smoothed diffusion |
|---|---:|---:|
| Own-entry RMS half-spreads | 0.670773 | 0.685489 |
| Own-entry outside-band contracts | 10 | 10 |

Outside-band membership changes for zero contracts. The largest absolute smoothed-minus-original price change is 0.047472 index points. In the focus dates, the respective maxima are 0.035999, 0.030448 and 0.022336 points.

The large selected-contract breaches already existed before smoothing. This supports separating original fit/carry inconsistency from the smoothing intervention. It does not establish that smoothing has no effect on hedge deltas or realized hedge P&L. Smoothed-minus-original price includes the changed diffusion and numerical error.

On the five numerical-validation dates, original AH and raw forward PDE each have four outside-band calibration targets among 11,220 targets; the smoothed PDE has 29. Smoothing has a measured quote-fit cost. Those are complete displayed quote-fit tables, not a validation of every October contract.

## October 11 basket

All 18 requested target buckets have records:

- The 21-day target selects November 1.
- The 35-day target selects November 15.
- The 45-day target also selects November 15, whose actual calendar maturity is 35 days.

The last six call/put and moneyness buckets point to the same contracts already selected for the 35-day target. Deduplication therefore produces 12 unique entries and six duplicate-bucket records. No option side is unavailable, and all 12 unique entries have ready deltas and matched endpoints.

Across October, 396 requested buckets produce 390 unique entries. This basket difference is explained and requires no selection-rule change. Target maturity labels must remain separate from actual maturity groups: the 45-day target produces actual calendar maturities from 35 to 45 days.

## Hedge findings

The saved full-October midpoint result remains:

- AH RMS: 3.549799 index points; Black RMS: 3.450136.
- MAE improvement, Black minus AH: +0.046486 points.
- MSE improvement, Black minus AH: -0.697632 points squared.
- AH lower absolute P&L in 50.8% of pairs.

Poor calibration and hedge performance do not have a one-to-one relationship. October 19 has the largest negative pooled MSE contribution (-2.668674); October 27 has poor calibration but a positive contribution (+0.305145). October 5 and 25 fit their calibration targets relatively well but contribute -1.311995 and -1.191130 respectively.

Actual maturity breakdown, midpoint case:

| Calendar maturity | Comparisons | AH RMS | Black RMS | MAE improvement | MSE improvement |
|---|---:|---:|---:|---:|---:|
| 14-27 days | 132 | 3.165756 | 3.179920 | +0.099409 | +0.089879 |
| 28-39 days | 204 | 3.406539 | 3.659651 | +0.357049 | +1.788537 |
| 40-45 days | 54 | 4.761325 | 3.264256 | -1.256120 | -12.014852 |

The longest group spans nine entry dates. Its MSE difference consists of +0.006164 variance improvement and -12.021016 squared-mean improvement. AH mean P&L is +3.471493 versus Black +0.173910. Its disadvantage is predominantly mean bias in this sample.

Its weighted contribution to pooled MSE improvement is approximately -1.663595, versus +0.030421 and +0.935542 for the first two maturity groups. These are descriptive overlapping-contract/date summaries, not independent observations or grounds for retrospectively removing the longest group.

## Numerical evidence and limits

The complete displayed sensitivity table contains 40 comparisons: five dates, two radii, and four checks. For the smoothed radius, maximum changes across those tested generated contracts are:

| Check | Price change, points | Delta change | Gamma change |
|---|---:|---:|---:|
| Space | 0.000731 | 0.000071707 | 0.000155016 |
| Time | 0.000333 | 0.000006984 | 0.000001948 |
| Grid shift | 0.000286 | 0.000021162 | 0.000232782 |
| Domain | 0.000010 | 0.000000595 | 0.000000029 |

These are sensitivity estimates for the sampled contracts and spot windows, not exact Greek error bounds or global economic wing robustness. The original raw radius retains substantially larger gamma sensitivity.

The focus outlier dates were not among the five requested numerical-validation dates. The uploaded forward-shape and forward/backward notebook views contain ellipses; those views cannot establish full-table totals or global maxima. The all-entry independent-Greek-validation flag correctly remains false.

## Closure and next dependency

This closes the saved-results audit's factual questions: calibration concentration, conditional parity incompatibility, own-side selected price breaches, smoothing's price-fit cost, and October 11 deduplication. Freeze the full October development baseline and keep the flags in its report.

The next scientific stage is the surface-derived smile hedge and empirical minimum-variance correction, followed by their prescribed Gain comparison. State the smile-motion convention, verify analytical corrections with numerical controls, train empirical parameters on prior observations only, and retain the frozen Black/AH baselines and complete coverage. Final evaluation must use declared chronological boundaries, with no parameter choices or exclusions selected to manufacture a positive Gain.

