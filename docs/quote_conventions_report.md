# SPX quote and pricing conventions

Audit date: **1 October 2026 (Asia/Kolkata)**. Quote date: **2023-12-22**.

**Price-derived forwards and discounts materially improve the existing call-IV repricing, but the vendor IVs remain inconsistent with observed prices. The data support a controlled constrained-fitting experiment using observed OTM prices and explicit uncertainty; they are not fully certified inputs.** Contract settlement identities, quote ages and vendor pricing assumptions remain unavailable. No surface fitting or repair was performed.

The starting checkpoint is `312975bb3eee34f14fb0ac4c62113eb87fd73620`. The existing call cleaner, Black–Scholes functions, surface implementation, tests and [surface validation report](validation_report.md) remain unchanged. That report's strike-arbitrage failures still apply to the existing surface. New work is isolated in `src/lv_project/quote_conventions.py`, `scripts/validate_quote_conventions.py`, this report and `tests/test_quote_conventions.py`. There was no dependency or raw-data blocker for the numerical checks.

## Inputs, timestamps and settlement

The input is `data/raw/spx_eod_202312.txt`: **177,774 monthly rows**, of which **9,158** belong to 2023-12-22, spanning **56 expiry dates**. Each wide row already contains call and put fields for one strike and shared expiry/snapshot metadata. Prices are measured in **SPX index points**, without applying the contract multiplier.

| Available fields | Observed meaning and treatment |
|---|---|
| `quote_date`, `quote_readtime`, `quote_unixtime`, `quote_time_hours` | All selected rows have date 2023-12-22, naive read time 16:00:00, UNIX time 1703278800 and hours 16. UNIX time converts to 21:00 UTC / 16:00 America/New_York. This supports a New York interpretation of the naive label; the file does not declare its timezone. |
| `underlying_last` | Exactly 4,755.11 on every selected row. Retained as the vendor snapshot spot; its source, observation timestamp and synchronization with option legs are unknown. |
| `strike`, `expire_date`, `expire_unix`, `dte` | Positive strike and dated expiry. Every expiry UNIX time maps to 16:00 New York, with 20:00 UTC in daylight saving time and 21:00 UTC otherwise. Expiry labels are dates, not settlement identities. |
| `c_bid`, `c_ask`, `p_bid`, `p_ask` | Present on all 9,158 rows. These prices, rather than last trades or vendor IV, drive matching, parity and inversion. |
| `c_size`, `p_size` | Text such as `1 x 1`, interpreted as displayed bid size × ask size. Parsed independently; this interpretation is not accompanied by a vendor field dictionary. |
| `c_volume`, `p_volume`, `c_last`, `p_last`, Greeks, IV | Retained. Volume is not quote age; last trades are not used as synchronized prices. Vendor IV is treated as decimal annual volatility, consistently with the existing code. Raw missing IV counts are 214 calls and 1,536 puts. |

There is **no option root, OCC symbol, AM/PM settlement flag, exercise-style field, separate call/put quote timestamp, exchange quote age, open interest, vendor rate/dividend curve or settlement cash-payment date**. The shared timestamp supports consistent vendor-row pairing, not proof of simultaneous executable exchange quotes.

The exact vendor/download provenance is not established by the repository. No vendor identity or undocumented pricing definition is inferred from the CSV layout.

Standard third-Friday SPX contracts can settle against a special opening quotation, whereas SPXW expiration trading commonly ends at the close. Cboe explains that the opening settlement quotation uses constituent opening prices and has no fixed calculation time. Consequently, a vendor 16:00 expiry timestamp does not establish PM settlement. The references postdate the quote sample and explain the contract distinction; they do not identify these particular records. See [Cboe SPX specifications](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications) and [Cboe's AM settlement description, July 2024](https://res-certification.cboe.com/resources/spx/Settlement_of_Standard_AM_Settled_SP_500_Index_Options.pdf).

Third-Friday dates are flagged as **possible AM/PM ambiguity**, not assigned a contract type. There are 20 such expiry dates in the raw sample and 12 among the 43 existing-pipeline expiries. All records retain `provisional_settlement=True`; non-third-Friday dates are not certified merely by their calendar pattern.

The primary maturity convention is elapsed **ACT/365F**:

```text
T = (expire_unix - quote_unixtime) / (365 × 86,400)
```

Raw DTE approximates elapsed days rounded to two decimals. Its maximum discrepancy is **0.001666667 days = 144 seconds**. For daylight-saving expiries, calendar-day subtraction is 1/24 day longer than exact elapsed time; 2,826 rows have this offset. The audit retains all three values: raw DTE/365, exact elapsed T and calendar days/365. Zero-DTE quotes cannot be inverted under a positive-maturity Black model and are excluded explicitly. Settlement-time, payoff-observation and cash-payment timing remain approximations where the vendor metadata cannot distinguish them.

## Matching, filtering and separate datasets

The primary audit covers the existing pipeline's **7–365 elapsed-day window**. A supplemental audit covers every positive-maturity expiry with a 1–2,500 day window, including the observed 4–6 day and 375–2,191 day expiries. This broad audit establishes which estimates are weak rather than silently omitting the longer contracts.

Matching uses the shared wide row. Its identity includes **strike, expiry date, expiry UNIX time and quote UNIX time**, plus `contract_root`, `contract_type` and `settlement_type` if supplied. Parity groups exclude strike but retain the other identity fields. Different snapshots or contract metadata cannot be pooled into one estimate. Separate side timestamps, when present, must equal the shared timestamp. Unknown contract identity in this file remains an assumption; the code cannot recover a symbol that is absent from the data.

Exact duplicate observations keep one representative. Conflicting duplicate identities exclude every member, with recorded reasons, rather than averaging prices. **Neither kind occurs on this quote date.** A price side is usable when metadata are finite/valid, K and S are positive, its bid is at least 0.05, ask ≥ bid, mid > 0 and `(ask−bid)/mid ≤ 0.35`, within the maturity window. A matched pair requires both usable sides. **No vendor-IV condition is imposed.** Each excluded raw row remains in `quote_audit.csv` with side eligibility and semicolon-separated exclusion reasons.

| Primary selection | Count |
|---|---:|
| Raw rows | 9,158 |
| Outside 7–365 days | 1,437 |
| Inside window | 7,721 |
| In-window put-only liquidity exclusions | 592 |
| In-window call-only liquidity exclusions | 315 |
| Matched price pairs | **6,814** |
| Matched sides missing vendor IV | **13 calls + 917 puts** |
| Individually usable price sides | **7,406 calls + 7,129 puts = 14,535** |
| Primary parity-calibration pairs | **4,011** |

The 592 put-side and 315 call-side exclusions are disjoint on this date. A usable call is retained for repricing even if its put side fails, and conversely. Primary calibration additionally requires **|log(K/S)| ≤ 0.10**, reducing the leverage of distant, wide ITM quotes while preserving strike slope information. Selection sensitivities test narrower/wider bands, the full liquid chain, tighter spreads, positive displayed sizes and positive volume.

Among matched pairs all displayed bid/ask sizes are positive, so that size condition changes no primary estimate. Call volume is zero on 210 pairs and put volume on 40; volume > 0 is a sensitivity, not a required live-quote filter. Displayed size and small spread cannot establish freshness without quote-age metadata.

The supplemental price-only dataset contains **7,665 pairs**, with **4,411 calibration pairs**, across all 55 positive-maturity expiries. The original cleaned CSV is never written. The original 5,760 surface-input calls are reconstructed in memory with the unchanged cleaner and surface-input selector, including vendor IV in (0.01, 2), raw DTE/365 and input log-moneyness [-0.25, 0.20]. Their zero-carry errors reproduce the earlier validation exactly.

## Forward and discount estimation

The parity model assumes the two prices represent European claims with the same strike, payoff settlement and payment convention:

```text
C − P = D(F − K) = A − D K,     A = D F
parity interval = [Cbid − Pask, Cask − Pbid]
```

The estimator fits mid-price parity by weighted least squares, with weight `1/max(h, 0.05)^2` and `h=(call spread + put spread)/2`. Strike is centered at its weighted mean and scaled by its weighted standard deviation. F and D are recovered from the intercept and strike slope. Neither is inferred from vendor IV; **there is no F=S, D=1 fallback**. Positive D above 1 is permitted because a negative effective rate is not inherently an implementation error.

Linear programs find the range of D and F satisfying every selected parity interval with D>0. F bounds use the equivalent linear-fractional transformation `t=1/D`. A numerical positive-discount floor of 1e-10 is used; it is inactive for every reported estimate. The solver also measures the smallest uniform extra interval half-width needed if the original intervals are inconsistent. These are **deterministic feasible ranges conditional on the observed quotes**, not statistical confidence intervals or an external discount curve. Marginal F and D extrema are generally different joint points; combinations of separately selected endpoints need not be feasible.

An estimate is weak if it has fewer than eight pairs or effective pairs, strike span <4% of S, nonpositive F/D, infeasible intervals, a mid fit outside those intervals, discount range wider than 0.05, or forward range wider than 0.5% of S. Effective count is `(sum weights)^2/sum(weights^2)`. These explicit screening thresholds identify numerical/input weakness; they do not certify settlement metadata. Weak estimates retain their diagnostic point estimates/ranges, but vendor repricing and IV inversion under them are skipped with `weak_parity_estimate` status.

**All 43 primary expiries are identified by this screen.** Each mid fit lies in every selected interval; minimum extra half-width is zero. The same point estimates also lie inside every parity interval across all 6,814 liquid pairs, including pairs outside the calibration band.

| Primary estimation diagnostic, across expiries | Measured result |
|---|---:|
| Calibration pairs per expiry | 34–185; median 101 |
| Effective pairs | 14.11–90.34; median 26.07 |
| Strike span | 665–945 index points |
| Condition number, raw `[1,K]` weighted design | 91,950–438,965 |
| Condition number, centered/scaled design | Approximately 1 |
| Mid-parity residual RMSE | 0.128244–0.529415; median 0.293305 points |
| Weighted residual RMSE | 0.049826–0.445424 points |
| Largest absolute selected residual | 2.814645 points |
| Feasible F range width | 0.509072–4.992263; median 1.206030 points |
| Feasible D range width | 0.003667–0.019931; median 0.009955 |
| Interleaved-strike held-out folds | 86; all held-out predictions inside their parity intervals |
| Held-out mid residual RMSE | 0.111155–0.669935 points |

Weighted centering/scaling explains the condition number near one; it does not eliminate economic uncertainty in the discount slope. Short-expiry feasible ranges can include D=1, and implied annualized rates can be poorly determined even when F is stable. Descriptive rates `r_eff=−log(D)/T`, `q_eff=r_eff−log(F/S)/T` range from 3.289% to 6.352% and −2.207% to 1.394%, respectively. They are inferred effective quantities, not independently verified financing/dividend curves. No smoothing or monotone discount-curve fitting was applied.

| Expiry | Pairs | F estimate [feasible range] | D estimate [feasible range] | Mid residual RMSE |
|---|---:|---|---|---:|
| 2023-12-29 | 132 | 4760.9388 [4760.6909, 4761.2000] | 0.998934 [0.993947, 1.003902] | 0.265876 |
| 2024-01-19 | 164 | 4774.0453 [4773.4688, 4774.7796] | 0.996108 [0.991500, 1.000286] | 0.293364 |
| 2024-03-15 | 155 | 4804.2386 [4803.4762, 4804.9721] | 0.987092 [0.984462, 0.990167] | 0.187085 |
| 2024-06-21 | 104 | 4855.4178 [4854.2589, 4856.4844] | 0.973198 [0.968645, 0.977600] | 0.235933 |
| 2024-12-20 | 38 | 4934.9357 [4932.5263, 4937.3781] | 0.952126 [0.944889, 0.957733] | 0.413744 |

All rows after the first in this example are settlement-ambiguous third Fridays. `parity_estimates.csv` supplies every expiry, including excluded/weak rows, residuals, conditioning and marginal feasible ranges.

| Alternative primary quote selection | Largest abs(ΔF), points | Largest abs(ΔD) | Identified / weak expiries |
|---|---:|---:|---|
| abs(log(K/S)) ≤ 0.05 | 0.302354 | 0.000996 | 43 / 0 |
| abs(log(K/S)) ≤ 0.20 | 0.154033 | 0.000560 | 43 / 0 |
| Full liquid chain | 0.228082 | 0.000641 | 43 / 0 |
| Relative spread ≤ 0.10 | 0.005729 | 0.000085 | 43 / 0 |
| Relative spread ≤ 0.20 | 0.003575 | 0.000050 | 43 / 0 |
| Positive displayed sizes | 0 | 0 | 43 / 0 |
| Positive call and put volume | 0.115598 | 0.000979 | 38 / 5 |

Differences use finite alternative estimates, including diagnostic estimates flagged weak; positive-volume selection can leave only one pair. No missing estimate was replaced by zero carry. The alternative bands and full chain give stable F, while their D differences should be assessed together with the substantially broader bid/ask feasible ranges.

The supplemental audit identifies **50 of 55 positive-maturity expiries**. The five weak expiries are:

| Expiry | Calibration pairs | Diagnostic F / D | Reason for rejection |
|---|---:|---|---|
| 2025-12-19 | 9 | 5059.2796 / 0.914483 | Effective pair count below 8 |
| 2026-12-18 | 9 | 5181.1336 / 0.881181 | Low effective count; wide F/D feasible ranges |
| 2027-12-17 | 6 | 5313.6531 / 0.846385 | Too few/effective pairs; wide F/D ranges |
| 2028-12-15 | 9 | 5451.6503 / 0.830637 | Wide F/D ranges |
| 2029-12-21 | 9 | 5614.7067 / 0.787358 | Low effective count; wide F/D ranges |

For example, 2029-12-21 permits F approximately **5373.06–6363.16** and D **0.38657–1.20229**. A point regression alone would conceal this uncertainty. **488 otherwise usable option sides** on these five expiries are retained with skipped convention-based inversion/repricing. The zero-maturity 2023-12-22 expiry is separately excluded.

## Vendor-IV repricing

The new convention prices calls and puts using the forward form of Black–Scholes:

```text
s = sigma sqrt(T); d1 = log(F/K)/s + s/2; d2 = d1 − s
C = D[F N(d1) − K N(d2)]
P = D[K N(−d2) − F N(−d1)]
```

Implementation evaluates the OTM leg and adds discounted intrinsic value for the ITM leg, reducing cancellation. For sigma=0 it returns discounted intrinsic. Baseline comparison uses **F=S, D=1, T=raw DTE/365**; corrected vendor-IV repricing uses **estimated F, D and exact elapsed T**. The put zero-carry reference is an analogous diagnostic, not an existing put surface. Both conventions are compared on the same rows with finite positive vendor IV and an identified parity estimate. Finite-positive vendor IV means numerically priceable, not quality-certified; there are 54 individually usable sides with vendor IV ≥2, up to 11.60571. They remain visible in the broad comparison. The original-call sample retains its existing IV limits.

| Identical-row comparison sample | Vendor-IV rows | Bid/ask coverage: zero → parity | Median absolute mid error: zero → parity | RMSE: zero → parity |
|---|---:|---:|---:|---:|
| Original surface-input calls | **5,760** | **5.97% → 60.35%** | **12.99449 → 2.73912** | **29.73297 → 2.88233** |
| All individually usable calls with vendor IV | 7,349 | 9.69% → 64.00% | 13.72710 → 2.97347 | 30.46907 → 4.48180 |
| All individually usable puts with vendor IV | 5,897 | **8.50% → 6.38%** | 2.64612 → 0.72836 | 14.72040 → 5.23524 |

All errors are index points. The baseline comparison's maximum absolute error falls from **92.38501 to 11.04167**. Broader call/put corrected maxima are **170.78474 / 66.53373** points, including deep ITM vendor-IV outliers. Per-expiry errors and coverage are saved for both the full side sample and the original 5,760 calls; aggregate improvement is not a universal improvement for every expiry or criterion.

| Original-call expiry | Rows | Zero / parity coverage | Zero / parity RMSE |
|---|---:|---:|---:|
| 2023-12-29 | 252 | 77.78% / 66.67% | 1.80193 / 2.77557 |
| 2024-01-19 | 273 | 0% / 72.89% | 10.22505 / 2.93889 |
| 2024-03-15 | 240 | 0% / 30.00% | 26.37881 / 3.11894 |
| 2024-06-21 | 162 | 0% / 42.59% | 50.55624 / 2.48529 |
| 2024-12-20 | 74 | 0% / 4.05% | 75.32626 / 6.40004 |

There are **2,284 original calls still outside bid/ask**. Call vendor repricing tends to exceed mids (broad median signed error +2.97347), while put vendor repricing falls below mids (median −0.72836); all numerically priceable put vendor prices are below their mids. Put coverage is zero on the earliest expiries despite some small absolute errors. Thus smaller RMSE does not imply adequate bid/ask coverage or recovered vendor assumptions.

At fixed estimated F/D, replacing exact T by rounded raw DTE/365 changes vendor prices by at most **0.001166 points**; using calendar days/365 changes them by at most **0.029155 points** over the primary sample. Rounding/DST differences alone cannot explain the remaining multi-point errors. Vendor spot, dividend/rate assumptions, reference price, IV calculation timestamp, settlement treatment and deep-ITM inversion behavior remain unestablished. No parameter adjustment was made merely to force vendor repricing agreement.

## Observed-price IV inversion and failures

Observed bid, mid and ask prices are inverted independently with the same F, D and exact T, while vendor IV remains alongside them. The lower/upper bounds are `D max(F−K,0)` / `DF` for calls and `D max(K−F,0)` / `DK` for puts. Bound tolerance is **1e-8 index points**. Brent inversion brackets volatility on [0,10], with root tolerances 1e-13 and up to 200 iterations. Statuses distinguish below/above bounds, zero intrinsic limit, infinite-volatility limit, volatility-cap failure, invalid inputs, solver failure and weak conventions. No observed price is clipped or silently substituted.

| Primary price sides, 14,535 total | Positive finite IV | Below lower price bound | Other failures |
|---|---:|---:|---:|
| Bid | 11,722 | 2,813 | 0 |
| Mid | **14,211** | **324** | **0** |
| Ask | **14,535** | **0** | **0** |

Mid failures comprise **232 calls and 92 puts**, across 27 expiries. Lower-bound shortfall ranges from **0.000383 to 5.689389 points**, median **0.251875**. Eight are original surface-input calls. All retained asks are within bounds and invert, so these midpoint/bid failures under the selected point estimates are **not by themselves executable arbitrage claims**. F/D uncertainty, broad ITM spreads and quote synchrony must be considered before rejecting a whole quote. A quote interval overlapping admissible prices can still be useful to a future constrained price fit even if its chosen mid cannot be inverted.

Of the 930 matched side observations missing vendor IV, **924 mid inversions succeed and six fail the lower bound**. Successful primary mid IVs range from 0.068486 to 4.083844. Fifty-five successful inversions have vega below 1 index point per unit annual volatility, so small price errors can produce large IV changes. A computed deep-ITM IV is not necessarily well identified or suitable for fitting.

The **6,814 usable OTM/ATM sides**, chosen as puts below F and calls at/above F, all invert successfully: 5,282 puts and 1,532 calls. There are 4,367 on non-third-Friday expiries and 2,447 on flagged third Fridays. This is a diagnostic candidate mask (`otm_or_atm`), not a newly fitted or written replacement clean dataset. Individual sides, their matched counterparts and uncertainty remain in the saved tables.

Maximum successful price→IV→price error is **3.86e-11 points**. This checks **numerical inversion only**; it does not establish strike convexity, calendar monotonicity, smooth derivatives or surface quality. The supplemental audit has 15,741 successful mids, 348 lower-bound failures and 488 skipped weak-convention sides. No cap, upper-bound or solver failures occur in either reported market audit.

For flagged primary third Fridays, a hypothetical 09:30 expiry convention, 6.5 hours earlier than the vendor 16:00 timestamp, raises recomputed IV by median **2.10 IV bps**, maximum **78.05 IV bps**, among successful comparisons. Here an IV bp is 0.0001 annual volatility. This holds estimated F/D fixed to isolate time sensitivity; it does not reconstruct the true AM settlement payoff or its forward, and 09:30 is not asserted to be the actual SOQ calculation time.

## Reproduction, evidence and status

All Python commands use the **volspx** conda environment: Python 3.11.15, NumPy 2.4.3, pandas 2.3.3 and SciPy 1.17.1. Matplotlib produces offline PNGs with its writable configuration directory inside the new audit directory. SciPy's HiGHS linear programs and Brent solver require no additional installation in this environment.

Commands executed from the repository root:

```bash
git status --short --branch
git diff --stat
conda run --no-capture-output -n volspx python -m unittest discover -s tests -p test_quote_conventions.py -v
conda run --no-capture-output -n volspx python -m unittest discover -s tests -v
conda run --no-capture-output -n volspx python scripts/00_smoke_test.py
conda run --no-capture-output -n volspx python scripts/validate_quote_conventions.py --quote-date 2023-12-22
conda run --no-capture-output -n volspx python scripts/validate_quote_conventions.py --quote-date 2023-12-22 --output-dir outputs/diagnostics/validation/conventions/reproduction
conda run --no-capture-output -n volspx python scripts/validate_quote_conventions.py --quote-date 2023-12-22 --min-days 1 --max-days 2500 --output-dir outputs/diagnostics/validation/conventions/all_positive_maturities
conda run --no-capture-output -n volspx python scripts/validate_quote_conventions.py --quote-date 2023-12-22 --min-days 1 --max-days 2500 --output-dir outputs/diagnostics/validation/conventions/all_positive_maturities_verified
git diff --check
```

The unit suite uses only synthetic fixtures; it does not require market data. **34 tests pass**, comprising the unchanged original 18 and 16 new tests. New tests recover known F/D, including nonzero/negative carry, compare forward pricing with the existing spot implementation, verify call/put parity and IV recovery across maturities, handle noisy/inconsistent intervals, reject unidentified slopes, check timestamp/type separation, preserve usable quotes without IV, handle duplicates and bounds, and prevent inversion/zero-carry substitution under weak estimates. Preservation tests explicitly monitor prior validation outputs. Existing smoke checks pass, including exact displayed BS/IV recovery.

Market reproduction requires the excluded raw monthly file locally and a **fresh** output directory:

```bash
conda run --no-capture-output -n volspx python -m unittest discover -s tests -v
conda run --no-capture-output -n volspx python scripts/validate_quote_conventions.py --quote-date 2023-12-22 --output-dir outputs/diagnostics/validation/conventions/repeat-primary
conda run --no-capture-output -n volspx python scripts/validate_quote_conventions.py --quote-date 2023-12-22 --min-days 1 --max-days 2500 --output-dir outputs/diagnostics/validation/conventions/repeat-all
```

`--raw-path /path/to/spx_eod_202312.txt` overrides the monthly input. A missing file produces the exact missing path. A missing pairing field produces its name. No synthetic market-data fallback is provided. Nonempty output directories are refused; generated diagnostics and raw/processed data remain Git-excluded. Investigation helpers/notes remain under the excluded `.local/` directory and are not required for reproduction.

Primary complete artifacts are under **`outputs/diagnostics/validation/conventions/reproduction/`**. Supplemental complete artifacts are under **`outputs/diagnostics/validation/conventions/all_positive_maturities_verified/`**. Earlier convention runs are retained in the parent and `all_positive_maturities/` directories.

Generated artifacts are not distributed in the source checkpoint. The plot links below resolve after local reproduction.

| Artifact in each run directory | Contents |
|---|---|
| `quote_audit.csv`, `matched_pairs.csv` | Raw fields, snapshot identity, vendor IV, price-side eligibility, duplicates/exclusions, parity intervals, selected-pair mask and residuals |
| `expiry_metadata.csv`, `parity_estimates.csv` | Every expiry, exact/raw/calendar time, spot ranges, settlement flags, pair counts, F/D, conditioning, feasible ranges and weakness reasons |
| `quote_selection_sensitivity.csv`, `parity_heldout.csv` | Selection alternatives and independently targeted strike folds |
| `repriced_and_inverted_quotes.csv` | Both conventions, vendor IV, recomputed bid/mid/ask IV, inversion/bound statuses, errors, vega, original-input and OTM candidate masks, hypothetical time sensitivity |
| `repricing_by_expiry.csv`, `baseline_call_repricing_by_expiry.csv`, `convention_comparison.csv` | Same-row coverage and errors by expiry/side/sample |
| `inversion_status_by_expiry.csv`, `mid_inversion_failures.csv` | Explicit failure counts and every failing/skipped midpoint with its bound shortfall where defined |
| `conventions_summary.json`, `check_status.csv` | Invocation, versions, checkpoint, measured summaries and separate market/check statuses |
| `preservation_before.json`, `preservation_after.json` | SHA-256, size and modification-time preservation checks for every pre-existing data/output file outside the current run |

The primary audit verified **222 existing files / 6,957,527,250 bytes**, with **zero changes**. The primary reproduction verified 241 files, including the first convention results; the initial broad audit verified 262 files, including both earlier runs. The verified broad audit checked **283 files / 7,009,795,746 bytes**, again with zero changes. The cleaned dataset, all raw monthly data, saved baseline surfaces and earlier validation results were preserved. All tracked files present at the starting checkpoint were unchanged during the audit.

Inspected primary figures: [forward/discount feasible ranges](../outputs/diagnostics/validation/conventions/reproduction/forward_discount.png), [vendor-IV bid/ask coverage](../outputs/diagnostics/validation/conventions/reproduction/vendor_iv_coverage.png), [parity residuals](../outputs/diagnostics/validation/conventions/reproduction/parity_residuals.png), and [IV/selection sensitivity](../outputs/diagnostics/validation/conventions/reproduction/iv_and_selection_sensitivity.png). The coverage plots corroborate the distinct call/put findings; the forward/discount plot shows discount uncertainty and highlights unresolved third-Friday identities. IV plots are raw-quote comparisons and apply a 0–100% display range; extreme IVs remain in the tables.

| Check | Result |
|---|---|
| Synthetic pricing/parity/inversion and preserved regression suite | **PASS: 34 tests** |
| Existing smoke checks | **PASS** |
| Primary parity identification and interval compatibility | **PASS: 43/43 expiries** |
| Identification on every positive-maturity raw expiry | **FAIL: five weak long expiries; 50/55 pass the screen** |
| Vendor-IV bid/ask consistency | **FAIL: substantial unresolved discrepancies; put coverage worsens in aggregate** |
| Positive finite IV at every retained primary mid | **FAIL: 324 lower-bound violations** |
| Successful numerical inversion round trips | **PASS: maximum error <4e-11 points, primary** |
| Original data and baseline-output preservation | **PASS: zero changed files** |
| Contract identity, actual settlement/payment time, exchange quote synchrony and vendor assumptions | **NOT RUN / unavailable metadata** |
| Surface repair, constrained fitting, new surface-arbitrage validation, Dupire and hedging | **NOT RUN: outside this task** |

Exit 0 means the audit and preservation checks completed, **not** that all input-quality checks passed. Early broad-run empty-sample RMSE calculations emitted warnings where weak-expiry repricing was deliberately skipped; empty comparison statistics now return NaN explicitly and the verified broad run checks that handling. Weak rows remain in tables rather than receiving fabricated estimates.

## Readiness and recommended next steps

The existing zero-carry/vendor-IV inputs are **not ready for acceptance as a constrained surface's target IVs**. Price-derived F/D and exact maturity are a materially better, reproducible convention, but vendor assumptions remain unresolved and the existing surface's previously documented arbitrage is unaffected.

The primary data are **conditionally ready for an experimental constrained price fit** using observed OTM prices, bid/ask intervals and explicit support/settlement masks. There are 6,814 OTM candidate observations with successful numerical inversion, including 4,367 on the 31 non-third-Friday primary expiries. That latter provisional subset is preferred for initial experiments; expiry pattern alone still does not certify identity. IV is a derived view of price uncertainty, not an exact observation or a surface-quality certificate.

Acceptance requires contract roots/settlement flags, vendor timezone/quote-age/pricing definitions, confirmed cash-payment timing and an external cross-check of discount information. The five weak long-expiry convention estimates should be excluded from accepted inputs until their uncertainty can be reduced. Price selection must account for jointly feasible F/D uncertainty; independently combined marginal interval endpoints or clipped ITM midpoint violations are not justified. An explicit OTM selection/splicing and liquidity policy is required, with both legs and rejection reasons retained.

A subsequent constrained fit should enforce price bounds, strike monotonicity, vertical-spread bounds, butterfly convexity and calendar consistency within quote uncertainty, retaining support masks and reporting deviations from quotes. Those checks must pass on native-supported coarse and dense grids before local-volatility extraction. Surface fitting, Dupire calculation and hedging implementation are outside the scope of this audit.
