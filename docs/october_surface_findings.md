# October 2023 surface evidence: findings and limits

Reviewed on 8 October 2026. Evidence run: `outputs/surface_evidence/2023-10/run_20261008T054238_553948Z`.

## Decision

The surface-evidence stage has produced the requested IV, total-variance, actual-local-volatility, derivative, density, support and comparison outputs for all 22 October calibration dates. The independent total-variance construction is a diagnostic benchmark with substantial failures. It is not accepted as a globally valid diffusion or a replacement hedging model.

Keep original AH and the separately identified radius-0.0005 changed diffusion distinct. Document the failures and the limits of each numerical comparison. One coefficient-figure drawing error has been corrected separately; it did not change saved numerical results. Whole-month native shape and density claims require the complete saved tables, not the truncated notebook excerpts reviewed here.

## Evidence reviewed

The review uses the supplied terminal output, full 22-date daily-summary values, producer audit JSON, five October 2 figures, and the notebook diagnostic extracts. The shape and numerical-comparison extracts omit interior rows. Density moments, pillar jumps and controls shown here cover October 2 and the deterministic controls, not all October dates.

The producer records 22 successful identities, zero failures and 132.9786365 seconds of measured wall time. This includes saved-model reconstruction, diagnostics, figures, notebook and hashes; it excludes the final audit write. The delivered source-file hashes match the source hashes in the supplied audit. Original observations and models were not changed by this stage.

The user reported 354 passing project tests in 10.952 seconds. That is evidence of the supplied test run, not proof of every numerical or economic property of the models.

## Aggregate counts

Counts below sum the daily summaries. Sample counts include separately reported expiry sides and repeated locations across dates. They are not independent trials or percentages of bad option contracts. Failure categories overlap.

| Item | October value | Interpretation |
| --- | ---: | --- |
| Original calibration quotes | 50,569 | Original equivalent-call midpoint inputs |
| IV-ready observations | 50,082 | 99.04% of inputs |
| IV-unresolved observations | 487 | 0.963%; retained, not assigned invented IVs |
| Surface samples | 283,627 | Finite positive-time grids and pillar sides |
| Supported quote derivative samples | 257,739 | 90.87% of surface samples |
| Locally admissible quote samples | 174,334 | 67.64% of supported samples; 61.47% of all samples |
| Negative quote-density samples | 55,257 | 21.44% of supported samples |
| Negative quote-calendar derivative samples | 2,511 | 0.974% of supported samples |
| Pooled original-AH quote RMS | 0.341106 half-spreads | Reproduces the earlier October calibration audit |

The largest quote-inversion round-trip error is approximately 1.194e-12 index points. This validates the IV inversion and reconstitution for resolvable inputs. It does not validate interpolated derivatives, out-of-sample prices, the carry assumption or the diffusion.

The IV-unresolved counts match the previously supplied daily counts of equivalent-call midpoints outside individual call bounds. This is consistent with price-bound failures under the assumed carry. The detailed `quote_inversions.csv` statuses are still needed to state the exact lower/upper-bound breakdown. No inference of vendor error, executable arbitrage or incorrect contract identity follows from this comparison.

## The independent quote surface

The benchmark interpolates original quote midpoint IVs as total variance: PCHIP in forward log-moneyness, linear interpolation in time, no extrapolation. Its IV and total-variance plots can resemble AH while its second derivatives and densities are very different. The October 2 derivative figure shows large positive and negative oscillations in the quote curvature, Dupire denominator and density.

Within smooth pieces and the stated deterministic-carry convention, local variance is `w_T/g`. The signed density has the sign of `g`; normalized calendar order requires nonnegative `w_T`. Smooth-looking IV and even monotone-looking variance are therefore insufficient. See Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*, Sections 2.1–2.2, https://arxiv.org/abs/1204.0646, and Ögetbil and Hientzsch, *Extensions of Dupire Formula*, deterministic-rates equation A.12, https://arxiv.org/abs/2005.05530.

White cells have several meanings: no quote support, unsupported short-end/time derivative, failed positivity or vertical-spread checks, instability at the selected stencil widths, or an excluded strike knot. They are not zero volatility and not all arbitrage failures. Unsupported-before-first-expiry values are expected because this benchmark does not impose an initial variance curve.

PCHIP interpolation of total variance does not enforce call-price convexity. Its continuous first derivative does not make its second derivative continuous at strike knots. Linear time interpolation does not force a nonnegative time slope. The observed failures therefore belong to this construction under these inputs and carry assumptions; they do not disprove the Dupire identity.

Even a sample passing the local mask is not certification of a connected globally admissible surface. The denominator floor 1e-8 and tolerance `1e-4 + 0.05*abs(variance)` are reporting checks, not economic plausibility limits.

## Three comparisons with different evidential strength

| Comparison | Evidence | Conclusion and limit |
| --- | --- | --- |
| Native AH derivatives, Black chain rule and Dupire coefficient | Maximum October discrepancy 4.0963e-10 annual variance | Strong algebraic/discrete consistency. Both routes use related AH native quantities; this is not independent continuum validation. |
| AH price-derived IV with finite-difference derivatives | Daily maximum variance errors approximately 0.006337 to 0.106723 | A separate numerical recovery check with finite-width and interpolation error. Two widths do not prove convergence; locations, scales and relative errors matter. |
| Independent quote total variance versus AH | Daily maximum admissible variance differences 10.4833 to 231.2183; maximum price difference 3.9251 points | Different surface constructions with radically different derivative behavior. These differences are not solver error against exact truth. |

The largest finite-difference discrepancy occurs on October 27. The largest quote-versus-AH variance discrepancy occurs on October 16. Both are units of annual variance, not volatility percentage points. A large local-variance discrepancy can coexist with a modest price discrepancy because Dupire differentiation amplifies price/interpolation differences, especially where density is small. The supplied maxima alone do not identify their physical strike, maturity or distance from observed quote support.

The flat-Black and smooth-skew analytical controls show errors decreasing by approximately four when stencil width halves. Finest reported maximum variance errors are 1.0554e-7 and 1.3118e-7 respectively. These support the formula and implementation on smooth known surfaces. They are synthetic deterministic controls, not historical market observations or hedging evidence.

## AH density, early coefficients and pillar behavior

The October 2 density table shows finite-domain nodal mass approximately one, normalized first moment approximately one and zero negative nodal densities at the displayed precision. These are useful checks on the finite discrete model and its quadrature. They do not certify global tails, off-grid curvature or the corresponding statistics for the smoothed diffusion.

October 2 mass inside the report window `abs(log(K/F)) <= 0.09` declines to 0.876069 at 59.041667 days. About 12.39% of the fitted nodal probability then lies outside that report window. That is not lost probability or an error requiring renormalization. It helps explain why wing assumptions can matter to central contracts.

Daily maximum original actual local volatility ranges from 188.21% to 996.47% per sqrt(year). October 12 and October 20 are close to 1,000%. These are extrema over the sampled state/time domain, not ATM implied volatility or forecasts of realized volatility. The October 2 figures place their displayed maximum in the early left wing. Exact locations for the other dates are not in the summary.

The calibration proxy, actual AH local volatility and smoothed coefficient are different objects. A bound on the fitted proxy does not bound actual local volatility. The radius-0.0005 blend does not act as a global volatility cap: its October maximum remains approximately 996.53%. The earlier improvement in gamma sensitivity near the original spot therefore cannot be generalized to all states or dates from this table.

The October 2 first-pillar maximum coefficient jump is 1.214327 in decimal volatility, or about 121.43 volatility percentage points. One-sided derivative changes at expiry pillars are visible in the local-volatility surface. Continuous prices can coexist with discontinuous coefficients. Both sides must be stated; left/right values must not be silently averaged to hide jumps.

October 19 and October 27 combine poor original price-fit diagnostics with the lowest quote admissibility fractions, 59.66% and 58.28%. This is a descriptive association, not evidence of a single cause. October 13 also has only 63.44% admissible support despite zero IV-unresolved observations: valid pointwise IV does not establish a valid derivative surface.

## Presentation correction

The original early-coefficient figure selected a time slice by sorting on time alone, then connected strike points in the resulting unsorted order. This produced crossing line segments. The numerical sample CSV, summary statistics, heatmaps and 3D surfaces are unaffected.

The replacement `src/surface_evidence.py` selects exactly the earliest positive time, an explicit derivative side and increasing log-moneyness. The companion script redraws only the coefficient figures from pinned saved CSVs into a new output folder. It leaves the original audited output files unchanged and creates a notebook for the corrected figures.

Extract the correction archive into the repository root, replacing `src/surface_evidence.py`, then run:

```bash
PYTHONPATH=src python scripts/39_replot_surface_coefficients.py \
  --evidence-run outputs/surface_evidence/2023-10/run_20261008T054238_553948Z
```

Correction verification: the 23 surface-evidence tests passed locally; the redraw script successfully read a pinned real September sample, verified its digest, saved a corrected figure and notebook, and preserved the original files. The figure was visually inspected. October redraw still runs on the user's local saved CSVs. No actual Jupyter kernel execution was performed here.

## Closure and retained limits

This completes the requested surface comparison as a diagnostic study, with the presentation correction supplied. Keep the quote interpolator explicitly unpromoted. Preserve original observations, failure masks, support holes and actual coefficient extrema.

The full native-AH shape counts for all October dates are not shown by the truncated excerpts. A final report should quote those whole-table totals only after reading `shape_checks.csv`; similarly use complete `density_moments.csv` and `pillar_jumps.csv` for whole-month extrema. Do not extrapolate October 2's clean visible rows to all 22 dates.

This stage does not validate all 390 entry Greeks, remove documented wing dependence, establish the smoothed diffusion's complete marginal distribution, add observed funding/execution data or demonstrate AH hedge superiority. Those claims need their separate evidence. A documented failed interpolation benchmark is a legitimate numerical-analysis result; it does not need to be cosmetically repaired or deployed as another hedge model.
