# October 2023 optimal hedging findings

Review date: 2026-10-08. Stage 40 result: `outputs/optimal_hedging/2023-10/run_20261008T070649_350768Z`.

## Completion decision

Step 3 is complete for strategy implementation, analytical controls and the October development evaluation. Six distinct strategies are implemented: observed-IV Black, saved AH PDE delta, original-surface sticky strike, original-surface sticky delta, the Hull–White LV smile approximation and a past-only empirical quadratic correction. This completion does not establish future minimum-variance optimality or held-out superiority. Those claims require Step 5.

The user reports 387 passing unit tests in 12.757 seconds. The submitted stage audit records 22.309278 seconds of elapsed wall time, 390 entries over 22 dates and maximum ledger reconciliation of 1.1251e-12 index points. The reviewer examined the three submitted plots and all 495 rows of the supplied Gain summary. Every reported Gain, raw Gain and RMS agrees with its supplied SSE/count identity to at most 1.78e-15. The hashes of the delivered script 40 and `src/optimal_hedging.py` match the audit's source hashes. This review did not rerun the user's historical calculation or independently verify its raw input hashes.

## Main results

Gain is `1 - strategy SSE / Black SSE`, using zero-centred funded P&L on identical candidate/Black entries. A positive Gain means less total squared hedge error. It is not an investment return. Each row below uses that candidate's available entries, so 390-entry and 386-entry rows have different Black denominators.

| Strategy | Entries | Funded Gain | Raw-mark Gain | RMS P&L, points | MAE improvement, points |
| --- | ---: | ---: | ---: | ---: | ---: |
| ah_pde | 390 | -5.86% | -4.49% | 3.549799 | +0.046486 |
| empirical_mv | 390 | 25.29% | 25.70% | 2.982118 | +0.482846 |
| lv_smile | 386 | 11.44% | 12.79% | 3.249844 | +0.271299 |
| surface_sticky_delta | 386 | -185.53% | -186.14% | 5.835248 | -2.156587 |
| surface_sticky_strike | 386 | 0.13% | 0.16% | 3.451064 | +0.003248 |

Black midpoint RMS is 3.450136 points on 390 entries and 3.453283 on 386 entries. Empirical MV reduces midpoint RMS to 2.982118, a 13.565% RMS reduction; its 25.290% Gain is an SSE reduction. It improves absolute P&L in 57.692% of comparisons, with mean signed P&L +0.889693 points. This is not a claim of zero bias.

Empirical MV funded Gain remains positive with observed option spreads (18.275%) and with spreads plus the assumed 1 bp hedge fee (17.630%). Its raw-mark Gain is 25.702%. The close midpoint/raw comparison supports the descriptive finding that the midpoint improvement survives removal of the assumed funding treatment. It does not validate the assumed funding curve or execution costs.

The common-strategy table excerpt reports 386 common entries and Black SSE 4603.111938. The exact empirical row is omitted in that excerpt. Nevertheless, common-subset empirical SSE cannot exceed its full 390-entry SSE of 3468.281381 because the removed terms are squared errors. Consequently, empirical Gain on the common 386 entries is at least 24.654%; the exact value should be taken from `common_strategy_gain.csv` for the final report. The unequal counts cannot explain away its stronger pooled Gain than the LV smile approximation's 11.435% on that common subset.

## Coverage and numerical controls

Black, AH PDE and empirical MV compare all 390 entries in every execution scenario. Each surface strategy compares 386, retaining four failures. Date-level counts place the four additional surface failures on 2023-10-27: 14 usable surface comparisons versus 18 Black/AH/empirical comparisons. October 11 has 12 entries for all strategies because six selection buckets duplicated existing contracts; this is distinct from the derivative failures.

The supplied analytical controls cover calls and puts with flat, negative and positive linear IV slopes. Halving a direct spot bump from 1e-3 to 5e-4 cuts the error by approximately four. At 1e-4, the largest error is about 1.88e-8 delta units. These controls support the implemented derivative conventions. They do not independently certify every historical entry delta or the smoothed diffusion's wings.

## Interpretation of the three figures

The delta figure shows sticky strike near observed-IV Black; sticky delta increases the hedge by roughly 0.07–0.10 on average; the LV smile approximation decreases it by roughly 0.06–0.10; saved AH PDE decreases it further on many dates; empirical MV generally supplies a smaller negative correction. These are daily average signed delta differences, not option position weights or portfolio returns.

Under the declared frozen-smile convention, sticky delta uses `surface Black delta - surface vega * sigma_y / S`, with `y = log(K/F)`. The LV smile approximation uses `observed-IV Black delta + observed-IV Black vega * sigma_K`. Negative skew therefore produces corrections with opposite signs. The plots are consistent with this distinction. Ordinary AH PDE delta is a separate model sensitivity and is not labelled minimum variance.

Sticky strike's near-zero Gain is consistent with a good fit between original AH implied prices and observed-IV Black inputs at most selected contracts. Sticky delta has negative Gain in every reported maturity, moneyness and absolute-delta bucket. Its -185.532% midpoint Gain means SSE is 2.855 times the matched Black SSE, not a 185% trading loss. Neither observation proves a universal ranking outside this development sample.

The coefficient plot displays separate quadratic call/put fits. Changes in a, b and c individually do not establish structural market coefficients or overfitting: they act jointly through the predicted correction, and their uncertainty is not shown. Rank, condition number and actual training membership belong in the final estimation audit.

## Past-only estimation and evidence limits

The audited source selects matched, Black-ready training rows only when both entry and endpoint timestamps are strictly earlier than prediction. Funding and costs are not used in fitting. The fitted objective is equal-row raw mark SSE, with no intercept, separate root/kind regressions and three quadratic coefficients. It is an empirical MV specification; finite-sample SSE combines variance and squared mean and is not an exact guarantee about future conditional variance.

The implementation saves `mv_fits.csv` and `mv_training_membership.csv`. These were not included as full row-level attachments in this review, so actual membership/cutoffs, design conditioning and prediction extrapolation have not been independently checked here. The submitted audit reports no future endpoints used. The source hashes match the delivered implementation, whose timestamp selection enforces that rule.

The configured 60 completed-entry-date window is not filled by the September/October history supplied to this run. October is therefore still using expanding available history within the 60-date cap. Chronological prediction is past-only, but September and October remain development months; reviewing their performance does not turn October into a held-out test.

## Bucket results and concentration

Empirical midpoint Gain is positive for calls (15.929%) and puts (32.021%), and across each reported moneyness and absolute-delta bucket. By maturity it is 22.131% for <=27 days and 32.391% for 28–39 days, but -1.096% for >=40 days. That last group contains only 54 comparisons across nine entry dates. The supplied absolute-delta buckets span .25–.50 and .50–.75; results do not cover the whole delta range or all available contracts.

For the LV smile approximation, call Gain is -6.298% and put Gain +24.167%. Its >=40-day Gain is -57.318%, versus positive Gain in both shorter groups. Thus the approximation's pooled improvement is not uniform.

Date-omission diagnostics below were derived from the supplied date-level SSE totals. For each omitted date, all entries sharing that entry date are removed from the evaluation, leaving saved predictions unchanged. No coefficients are refitted and no entries are reselected. These are sensitivity diagnostics, not confidence intervals or a leave-one-date-out training experiment.

| Strategy | Dates with positive midpoint Gain | Minimum Gain after one date omission | Maximum Gain after one date omission |
| --- | ---: | ---: | ---: |
| ah_pde | 11/22 | -26.11% | 16.86% |
| empirical_mv | 13/22 | 13.57% | 36.05% |
| lv_smile | 11/22 | -4.04% | 30.08% |
| surface_sticky_delta | 4/22 | -215.81% | -167.58% |
| surface_sticky_strike | 11/22 | -0.05% | 0.19% |

For empirical MV, October 2 and October 17 contribute +14.075 and +7.119 percentage points respectively to its pooled Gain. Removing both leaves +5.477% Gain. October 19 is its largest adverse date, contributing -10.105 percentage points, followed by October 5 (-8.733) and October 25 (-4.011). The result remains favourable after a single-date omission, but is meaningfully concentrated. No significance or generalisation claim follows from these checks, especially because contracts share dates and may recur.

## Frozen findings for the report and next experiment

Do not tune the strategies to remove the adverse dates or repair the sticky-delta performance. Retain the failed derivative entries, the original observations and the separately named strategies. Carry the configured methods forward to the declared held-out evaluation, with failures and warm-up recorded and dependence-aware uncertainty reported.

Step 4 should use a known model, seeded common paths and nested rebalancing schedules to test terminal replication error, self-financing cash reconciliation and frequency effects against theory with measured sampling uncertainty. A Black–Scholes control with known parameters and analytical delta provides the first reference. Its correctly specified continuous-hedging theory concerns replication in that model; it does not require empirical market results to match a simulation or make LV outperform Black. Any extension to a known nonconstant local-volatility diffusion should separately document pricing, simulation and grid errors.

Observed market inputs remain distinct from the experimental assumptions: UNKNOWN contract labels, unverified snapshot/fixing provenance, conditional option-implied forwards, assumed 5% funding, synthetic fractional index fills and zero dividend cash. No executable-index-return claim, confidence interval, held-out result or all-history completion is established by this stage.
