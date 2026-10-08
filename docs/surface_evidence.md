# Surface evidence and the AH / total-variance comparison

This stage produces one set of surfaces **per calibration date and root**. It reads the explicit saved pipeline run. It does not pool dates, change the observed quotes, refit the AH models, change hedge entries, or rerun a hedge experiment.

## Install and run

Place the four files from the archive in the corresponding `src/`, `scripts/`, `tests/`, and `docs/` folders at the repository root. These are additions; no existing module needs replacement. Use the existing `volspx` environment. NumPy, pandas, SciPy and Matplotlib are the existing dependencies; no additional plotting package is needed.

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python scripts/39_build_surface_evidence.py \
  --run-index outputs/research_pipeline/october_full_v1/run_index.json \
  --month 2023-10 --plan

PYTHONPATH=src python scripts/39_build_surface_evidence.py \
  --run-index outputs/research_pipeline/october_full_v1/run_index.json \
  --month 2023-10
```

The default processes every model identity in that month's manifest, retaining failed models as status records. To inspect particular dates, add `--dates 2023-10-19 2023-10-26 2023-10-27`. This is a reporting subset, not a new hedge selection rule. Each invocation writes a new output folder. `--plan` verifies hashes and lists the scope without evaluating models or writing files.

The terminal prints the output folder and measured wall time. `evidence_status.csv` records time separately for each date/root, including model reconstruction, diagnostics, tables and plots. No runtime estimate is substituted for a measurement. A model failure stays in the status table and gives a nonzero process return code; inspect its message.

Open `surface_evidence.ipynb` from the printed folder. Its cells read saved results. Change `DATE` and `ROOT_LABEL` to inspect another model. The editable 3D cell can display implied volatility, total variance or local volatility. The optional Matplotlib widget backend is only for an already configured environment; static 3D views also work. If Jupyter uses a different working directory, set `ROOT` to the printed evidence folder.

The notebook verifies the saved data/figure hashes before displaying them. It excludes its own file from that read-time check because Jupyter writes execution outputs and metadata into it. The audit still records the original generated notebook's hash.

## Distinct routes

| Route | Input / method | What it establishes |
| --- | --- | --- |
| Original AH | Existing fitted proxy controls and resolvent price surface | Original IV and total variance; discrete native density, time derivative and actual Dupire coefficient |
| Smoothed AH coefficient | Existing first-period Gaussian variance blend with log-state radius 0.0005 | The actual coefficient used by the research hedges; a changed diffusion |
| AH price → total variance → Dupire | Black chain rule on native AH price diagnostics, plus independent finite-difference evaluation of AH IVs | Coordinate/formula consistency and numerical sensitivity on the same AH representation |
| Observed quote → total variance → Dupire | Original equivalent-call midpoint IVs; PCHIP in log-forward-moneyness and linear total variance in maturity | A separate interpolation benchmark, its admissible support and its discrepancies from AH |

The smoothed coefficient panels do **not** reuse original AH prices or density as if they belonged to the smoothed diffusion. The existing script 30 validation supplies smoothed repricing/PDE evidence on its explicitly requested dates. This stage does not repeat that work or extend its validation claim to every entry.

The quote interpolation is deliberately transparent. It is not fitted to AH values and is not labelled arbitrage-free. Reproducing an input midpoint through Black inversion is not independent repricing evidence. Its negative density, negative time derivative or unstable derivative samples remain visible. This comparison can justify the use of the AH route while documenting why direct volatility interpolation requires care; it cannot certify a separate quote-based diffusion by masking failures.

## Conventions and formulas

Time `T` is positive elapsed ACT/365F time from the saved snapshot to the **assumed** fixing. The snapshot/fixing provenance remains unverified. The pinned same-date conditional-parity forwards and assumed discounts are retained. Between pillars, forward and discount use the same log-linear interpolation as AH. The pricing yield proxy is not observed dividend cash.

With deterministic carry:

\[
y=\log(K/F(T)),\qquad z=e^y,\qquad
c(z,T)=\frac{C(K,T)}{D(T)F(T)},\qquad w(y,T)=T\sigma_{\rm imp}(y,T)^2.
\]

Derivatives in time are at fixed **forward log-moneyness**, not fixed physical strike. The total-variance denominator and local variance are:

\[
g=\left(1-\frac{y w_y}{2w}\right)^2
-\frac{w_y^2}{4}\left(\frac1w+\frac14\right)
+\frac{w_{yy}}2,
\qquad a=\sigma_{\rm loc}^2=\frac{w_T}{g}.
\]

The equivalent price-coordinate expression is `a = 2 c_T / (z² c_zz)`. If `d2 = -y/sqrt(w) - sqrt(w)/2`, then the signed normalized-strike density is

\[
p_z(z,T)=\frac{\phi(d_2)}{z\sqrt w}\,g,
\qquad p_K(K,T)=\frac{p_z(K/F(T),T)}{F(T)}
=\frac{C_{KK}}{D(T)}.
\]

Positive `w_yy` alone is not the butterfly condition. Calendar order is tested at fixed `y`. Density positivity on a finite grid does not verify the required global tail conditions or total probability mass.

Formula references: Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*, Sections 2.1–2.2, [paper](https://arxiv.org/abs/1204.0646); Ögetbil and Hientzsch, *Extensions of Dupire Formula*, Appendix A, deterministic-rates formula A.12, [paper](https://arxiv.org/abs/2005.05530). The benchmark here uses PCHIP, not SVI calibration.

## Support and derivative policy

The report grid defaults to `y ∈ [-0.09, 0.09]` with 181 points. It includes positive early times, expiry pillars and interval midpoints. AH variance is never requested at time zero. Both AH expiry derivative sides are saved, except at the final pillar where only the left side is supported.

Each observed midpoint receives an IV status. Prices at/below intrinsic, at/above the upper bound, or unresolved inversions have no invented IV. Those rows remain in `quote_inversions.csv`, with their original prices and flags. Invalid IVs split each expiry's quote support into contiguous segments. Interpolation does not bridge these gaps or extrapolate into the wings.

The independent benchmark is unsupported before its first expiry and after its last. Between expiries, both bracketing slices must support the requested `y`. A first-pillar left time derivative and a last-pillar right time derivative are unsupported. A pillar price can still be supported when a time derivative is unavailable; both support masks are saved.

For a quote-based local-volatility value to appear in a plot, a sample must have common quote support, positive `w`, nonnegative `w_T`, `g > 1e-8`, acceptable normalized vertical-spread slope, and spatial derivative agreement. Its analytic PCHIP variance is compared with central stencils at one quarter and one eighth of the report-grid spacing. Both stencil-width change and disagreement with the analytic result must be below `1e-4 + 0.05 * abs(variance)`. These are explicit **diagnostic tolerances**, not economically calibrated thresholds. They affect reporting masks, not prices or coefficients.

PCHIP is C1 in strike, not C2. Its second derivative can jump at strike knots; both polynomial sides are evaluated, jumps are saved, and those knots are excluded from the displayed local-volatility benchmark. Linear total variance in time can also have derivative jumps at expiry pillars. Blank figure cells are unsupported or fail these checks. Signed derivatives and densities are retained even when a local-volatility cell is masked.

Native AH curvature and time derivative are discrete diagnostics. They are not the continuous second derivative of its piecewise-linear price interpolant. The Black chain rule maps those diagnostics into `w_y`, `w_yy` and `w_T`; the resulting Dupire coefficient is compared with the log-density AH adapter. In the ITM wing, intrinsic terms cancel symbolically to reduce floating-point cancellation. Separately, AH price-derived IVs are differentiated with stencils spanning eight and four native log cells. Time differences stay within an expiry interval and are one-sided at pillars. These checks expose sensitivity; they do not turn the piecewise-linear interpolant into a smooth continuum surface or certify convergence with two spatial widths.

The original native AH finite domain is checked for increasing calls, vertical-spread breaches, negative butterflies, intrinsic/upper-bound breaches and normalized calendar disorder. Its density mass and first moment use the native nonuniform-strike quadrature. The report-window mass is not renormalized. Underflow/zero native density counts are reported; there is no artificial density floor or volatility cap.

## Outputs and review order

| File | Review purpose |
| --- | --- |
| `audit.json` | Scope, conventions, measured runtime, input/model/source/output hashes and explicit limits |
| `evidence_status.csv` | Completed and failed date/root identities, messages, per-model runtime and figure paths |
| `daily_summary.csv` | IV failures, admissible support fractions, density/calendar failures and AH comparison errors |
| `quote_inversions.csv` | Every original calibration quote, inversion status, original AH fit and inversion round-trip error |
| `dates/*.csv` | Full signed sample grid, physical strikes/carry, distinct coefficients, derivatives and masks |
| `shape_checks.csv` | Original AH native arbitrage diagnostics and quote benchmark diagnostics, with counts/support denominators |
| `numerical_comparison.csv` | AH chain-rule and finite-difference discrepancies; quote comparison on common admissible samples |
| `density_moments.csv` | Native finite-domain mass/moment and unrescaled window mass |
| `pillar_jumps.csv` | Both-sided local-volatility jumps and counts of comparable samples |
| `analytical_controls.csv` | Deterministic Black and smooth-skew price/density/Dupire refinement controls |
| `plots/*` | Four figures per successful identity: six surface panels, three original AH 3D views, signed derivative/support views, coefficient comparison |
| `surface_evidence.ipynb` | Saved-result explorer and interpretation; no hidden calibration/PDE/hedge work |

Start with failures and support counts, then inspect the shapes, derivative checks, density moments and figures. Compare quote-Dupire with AH only where `quote_admissible` is true. A maximum price difference is a difference between constructions, not purely a numerical error. The independently reconstructed midpoint round-trip error is not an out-of-sample fit statistic.

The controls use synthetic **analytical price surfaces**, separately labelled and never mixed into market observations or hedge outcomes. Price-coordinate finite differences refine at approximately second order on those controls. The tests also cover invalid prices, quote-support holes, derivative sides, small/negative denominators, saved-carry disagreement, source preservation and tampered input/model hashes.

## Completion condition

This stage is reviewable when the requested dates have status records and their surface/derivative/density/arbitrage tables and views have been inspected, discrepancies have been explained, and any failed or inadmissible region is documented. A benchmark with failures remains a diagnostic comparison; it cannot be reported as a globally arbitrage-free calibrated diffusion.

There is no claim of global wing robustness, all-entry Greek certification, executable hedging returns, historical funding verification or hedge outperformance. Those conclusions must come from their separate evidence. The stage can satisfy the requested surface comparison without manufacturing a second day-zero model or changing October's observed results.
