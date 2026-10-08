# Smile hedges and empirical minimum-variance correction

This is Step 3 of the project closure work. It compares six explicitly named strategies on the saved one-session research panel. It does not recalibrate a surface or rerun the backward diffusion pricer. Loading an original AH model reconstructs its native implicit-grid prices; those native resolvent evaluations are recorded separately from a new diffusion-pricing study. Historical observations, failed entries, the existing AH diffusion radius and funding conventions are preserved.

## Strategies and what each assumes

Let T be remaining ACT/365F time, F the saved same-date forward, D the saved discount factor, y=log(K/F), and sigma the annual implied volatility in decimal units. Under a spot bump we hold F/S and D fixed. Vega is price sensitivity per 1.0 decimal volatility, not per volatility percentage point.

| Strategy | Implemented hedge | Meaning |
| --- | --- | --- |
| `black` | Observed-midpoint IV Black spot delta | Existing practitioner benchmark, fixed strike and IV. |
| `ah_pde` | Saved ordinary AH PDE spot delta | Physical local variance and carry fixed, with the saved first-period radius. It is not relabelled MV. |
| `surface_sticky_strike` | Original AH model-IV Black delta | Hold implied volatility at the fixed absolute strike constant. This need not equal observed-IV Black delta if the fitted price differs from the quote. |
| `surface_sticky_delta` | Surface Black delta − surface Black vega × sigma_y/S | Freeze the smile in the declared unadjusted forward-delta coordinate; equivalent locally to frozen forward moneyness under this carry convention. |
| `lv_smile` | Observed-IV Black delta + observed-IV Black vega × sigma_K | Hull–White's ATM-derived local-volatility smile approximation, with sigma_K extracted from original AH prices. Its extension to the selected non-ATM strikes is an assumption. |
| `empirical_mv` | Observed-IV Black delta + vega/(S sqrt(T)) × (a+b delta+c delta²) | Hull–White empirical quadratic correction fitted only to completed past observations, separately for each root and option kind. |

The signs of the sticky-delta and HW LV smile adjustments can differ. A static price surface does not determine its future response to spot. Neither convention certifies an optimal historical hedge. In particular, `lv_smile` is an approximation, not an identity for ordinary AH PDE delta or a proof of global minimum variance.

### Exact sticky-delta definition

The quote coordinate is q=N(d1), the **unadjusted normalized forward call delta**, with d1=−y/(sigma sqrt(T))+sigma sqrt(T)/2. Both calls and puts use this coordinate for the same smile. It is not premium-adjusted FX delta or the signed spot delta used for the hedge.

For a locally invertible decreasing q(y), freeze sigma as a function of q at this expiry. With fixed F/S, its implicit response gives d sigma/dS=−sigma_y/S. Thus the frozen q and frozen y descriptions agree locally. The implementation explicitly checks q_y<−1e−8. A noninvertible coordinate has a failure status; it is not silently accepted.

The sticky-strike and sticky-delta strategies use **original AH model-price IV**. The HW LV approximation uses **observed-price IV** for its base delta/vega and original AH IV for the strike slope. These inputs are recorded separately. None uses the independent quote PCHIP interpolator that failed the surface evidence checks. The smoothed diffusion's coefficient does not imply that its IV smile is the original AH smile.

### Surface derivatives and controls

At every selected expiry, invert the original AH OTM time value using the stable inversion introduced in Step 2. Compute sigma_y with centred widths of four and eight native log cells; sigma_K=sigma_y/K. The whole stencil must lie inside the same-expiry observed quote range and the finite AH grid. No wing extrapolation or clipping is added.

Save the fine/coarse slopes, delta changes, nodal density and direct spot-bump derivatives. Reprice the frozen normalized AH price curve at spot-bump widths of two and four cells. Require delta-width, bump-width and chain-rule disagreement <=0.0005, slope disagreement <=0.01+0.05 abs(sigma_y), and no negative interpolated nodal density below −1e−10. These are finite-width numerical checks, not a continuum convergence proof or a certificate of every entry's PDE Greeks.

Analytical controls check flat, negatively skewed and positively skewed IV curves, exact call/put delta parity, vega units, signs and implicit forward-delta repricing. The fixtures are explicitly synthetic and never become historical marks or missing-data fills.

## Empirical estimator and information timing

The published quadratic regression specification is used without an intercept. For each available fixed-contract observation:

    response = observed option midpoint change − entry Black delta × observed spot change
    regressors = observed spot change × entry vega/(entry spot sqrt(entry T)) × [1, delta, delta²]

Estimate a,b,c by equal-row least squares. Numerical column scaling is reversed before exporting coefficients; there is no penalty, parameter clipping or search over specifications. Call and put fits are separate, as are root labels. `UNKNOWN` remains an unverified vendor identity, not an inferred contract root.

Fixed development defaults: the last 60 completed entry dates, at least 10 dates and 60 rows **per root/kind**, scaled design rank three and condition number <=1e8. This is a pilot configuration, not a replication of Hull–White's 36-month research window. It must be frozen or replaced by a declared longer window before evaluating the agreed history. No setting is selected by maximising current-month Gain.

For a prediction at timestamp t, a training entry and its endpoint must both be **strictly earlier than t**. Even an observation ending at the same close is excluded. This conservative cutoff avoids relying on simultaneous vendor marks being available before a hedge decision. Current entry Greeks are known features; its endpoint mark is not used to fit its hedge. A separate history panel must precede all target entry dates. No September sample-end endpoint is fabricated to connect it to October.

Every fit saves its rank, condition, coefficients, dates, maximum endpoint, raw training SSE and mean error. `mv_training_membership.csv` records each used entry ID and endpoint. Warm-up, rank failure and ill-conditioning remain visible; they do not silently become a Black hedge. Prediction outside the fitted delta range is flagged. Correction deltas may be outside ordinary vanilla delta bounds; they are not clipped.

“Empirical MV” is the conventional name of this specification. The actual fitted objective is **zero-centred raw-mark SSE**. SSE combines variance and squared mean. It is not an exact finite-horizon conditional-variance optimum, and training SSE improvement is not out-of-sample hedge improvement. Funding, spread costs and hedge fees are not inserted into the published regression. They are evaluated separately through the already validated self-financing ledger.

## Comparison, Gain and coverage

The previous comparison audit pins the funding and execution conventions. We reuse midpoint, option-spread and option-spread-plus-assumed-fee scenarios. Both unit multipliers remain one, the option is short, the hedge is a synthetic fractional index position filled at vendor spot, and dividend cash remains zero. No executable SPX returns are claimed.

Each strategy keeps an independent coverage record for **every original entry**. Gain is:

    Gain = 1 − sum(candidate errors²)/sum(Black errors²)

Candidate and Black must have identical entry IDs. `gain_summary.csv` reports funded P&L Gain and raw midpoint-change Gain separately, overall and by kind, maturity, log-spot moneyness, absolute observed-IV Black delta and entry date. Gain is not demeaned or clipped. Zero benchmark SSE is reported explicitly. Raw Gain is unchanged across execution scenarios because its definition omits costs and funding.

Candidates may have different pairwise sample sizes because warm-up or surface support can fail. `common_strategy_gain.csv` instead uses the intersection where **all six strategies** have results. Both views retain counts and coverage. A favourable pairwise result on a smaller subset is not a ranking of every strategy over the full month.

Rows share dates and may reuse contracts. This stage provides descriptive results, not IID-row confidence intervals. Dependence-aware inference and a declared held-out period belong to Step 5. September and October 2023 have already been inspected and are development samples.

## Running the stage

Extract the update's `src/`, `scripts/`, `tests/` and `docs/` files into the repository root. It relies on the existing Step 2 `surface_evidence.py` and prior hedge modules; it does not replace them.

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python scripts/40_compare_optimal_hedges.py \
  --run-index outputs/research_pipeline/october_full_v1/run_index.json \
  --month 2023-10 \
  --history-run-index outputs/research_pipeline/september_checkpoint_v1/run_index.json \
  --history-month 2023-09 --plan

PYTHONPATH=src python scripts/40_compare_optimal_hedges.py \
  --run-index outputs/research_pipeline/october_full_v1/run_index.json \
  --month 2023-10 \
  --history-run-index outputs/research_pipeline/september_checkpoint_v1/run_index.json \
  --history-month 2023-09
```

`--plan` verifies required saved hashes and the producer chain without loading a model, fitting coefficients, writing files or running a ledger. The full run uses original daily models for smile extraction and uses saved endpoint marks for evaluation. It exports timing, source/input/output hashes, strategy panel, coverage, exact training membership, coefficients, Gain tables, three figures and `optimal_hedging.ipynb`. If no declared past panel is supplied, current-month observations become training data only after their endpoints are past; initial warm-up remains explicit.

Inspect `coverage.csv`, `mv_fits.csv`, `gain_summary.csv` and `common_strategy_gain.csv`, together with the audit and derivative flags. A completed execution is not a promotion or a certification of optimality.

## References

- Hull, J. and White, A. (2017), *Optimal delta hedging for options*, Journal of Banking & Finance 82, 180–190. DOI: https://doi.org/10.1016/j.jbankfin.2017.05.006 . Author copy: https://www-2.rotman.utoronto.ca/~hull/downloadablepublications/Optimal%20Delta%20Hedging.pdf . The author-copy URL was unavailable to open during development; indexed primary-source text supplied the ATM-derived LV formula. Do not treat that approximation as a general identity.
- Ruf, J. and Wang, W., *Hedging with linear regressions and neural networks*, https://arxiv.org/abs/2004.08891 . The quadratic delta/vega form is reproduced in their discussion of Hull–White.
- Alexander, C. and Nogueira, L. (2005), *Model-free hedge ratios and scale-invariant models*, https://www.maths.univ-evry.fr/pages_perso/crepey/Finance/AlexanderNogueiraOptimalHedgingHomogeneity.pdf . The frozen-moneyness chain rule has a different sign from the HW LV approximation. The present implementation derives and tests its declared forward-delta convention directly.
- Supplied CQF briefs, June 2025 and January 2026, local-volatility/optimal-hedging sections. No unsupported hedge-outperformance claim is required to satisfy a sound comparison study.
