# SPX implied-volatility surface validation

Validation date: **1 October 2026 (Asia/Kolkata)**. Quote date: **2023-12-22**.

The existing surface is reproducible, and the numerical coding defects identified below are fixed. **The reconstructed market surface fails strike-arbitrage checks and is not yet suitable for reliable Dupire extraction.** Passing smoke/unit tests and eliminating discrete calendar violations do not establish absence of static arbitrage.

The baseline implementation is repository commit `6ced5ec3c194471529f842d592c43e94760da897`. The active surface pipeline uses vendor call IV, linear interpolation in log-forward moneyness, constant wing fills, cumulative-maximum calendar repair, and linear interpolation in maturity. The results below validate that implementation against the locally available SPX quotes, independently of any later surface-repair work.

The original audit checked SHA-256, byte sizes and modification timestamps before and after on **148 existing files**, totaling **6,897,723,710 bytes**; **zero changed**. This included 132 monthly raw data files, the cleaned CSV, saved surfaces, diagnostics, tables, interactive plots, directory placeholders and one existing project note. The reusable runner monitors data and baseline outputs independently of local notes, creates a fresh preservation manifest for each run, and excludes the entire generated validation directory. See `preservation_before.json` and `preservation_after.json` in each run's artifact directory.

All Python commands used the **volspx** conda environment: Python 3.11.15, NumPy 2.4.3, pandas 2.3.3, SciPy 1.17.1, Matplotlib 3.10.8 and Plotly 5.24.1. Required data and dependencies were available for the reported measurements; there was **no data/dependency blocker**.

## Commands and execution status

Run the following commands from the repository root with the `volspx` environment installed. The tests use synthetic fixtures and require no market data. The market audit requires `data/processed/spx_calls_clean.csv`; raw data and generated files are excluded from version control and must be supplied locally. If the clean dataset is not available, prepare it from compatible `data/raw/spx_eod_*.txt` files with `conda run --no-capture-output -n volspx python scripts/01_prepare_quotes.py`, or provide an existing cleaned file through `--clean-path /path/to/spx_calls_clean.csv`.

```bash
conda run --no-capture-output -n volspx python scripts/00_smoke_test.py
conda run --no-capture-output -n volspx python -m unittest discover -s tests -p test_finite_diff.py -v
conda run --no-capture-output -n volspx python -m unittest discover -s tests -v
conda run --no-capture-output -n volspx python scripts/validate_surface.py
conda run --no-capture-output -n volspx python scripts/validate_surface.py --skip-reproduction
git diff --check
```

The audit runs the existing surface builder with the following arguments, using the same Python interpreter as the runner:

```bash
python scripts/02_build_surface.py --quote-date 2023-12-22 --y-min -0.15 --y-max 0.10
```

To preserve baseline files, the runner copies `src/` and the build script into a disposable temporary workspace. Its cleaned-data path is a symlink to the original input, which the builder only reads. The unchanged build script writes its normal output paths inside that workspace. All seven resulting files are copied to `outputs/diagnostics/validation/reproduction/`, retaining their relative directory layout. No original surface, HTML or diagnostics file is overwritten. The initial build exited 0 in approximately 13.15 seconds. `reproduction_command.json` records the interpreter, arguments, working directory and exit status; the original timing is retained in the local `baseline_commands.json` artifact.

| Execution/check | Status | Evidence |
|---|---|---|
| Existing smoke checks, before and after fixes | PASS | BS/IV error printed as 0.000000000000; first-derivative error 0.08000000; second-derivative error printed as 0.00000000; demo shape (14, 50) |
| Real-data build | PASS | Exit 0; headline results and saved grid reproduced |
| New finite-difference regression tests before fixes | FAIL, reproduced defects | 8 methods run, 8 assertion/subtest failure records; `tests_before.log` |
| Numerical, surface and support-provenance tests | PASS | Original 17 tests, exit 0; `tests_after.log` |
| Checkpoint regression suite | PASS | 18 tests, including preservation-manifest coverage; `python -m unittest discover -s tests -v` |
| Market audit, including support/tolerance checks | PASS execution | Exit 0; `validation_run.log`; market PASS/FAIL outcomes reported separately |
| `git diff --check` | PASS | No whitespace errors |
| Preservation of original audit inputs and existing artifacts | PASS | 148 hash/size/timestamp matches |

The checkpoint includes an additional regression test for preservation manifests in a checkout without local documentation, bringing the current suite to **18 tests**. It verifies detection of changed inputs, fresh manifests on subsequent runs, and exclusion of sibling validation outputs. No dependency installation or environment change was needed. The runner assigns `MPLCONFIGDIR` under the validation directory when it is unset, so plotting does not depend on a writable home cache.

A complete checkpoint re-audit under `outputs/diagnostics/validation/checkpoint/` reproduced the surface, support masks, calendar repairs, call-arbitrage counts and quote-repricing errors reported below. All **147 data and baseline-output files** monitored by the standalone runner matched their pre-run hashes, sizes and modification timestamps. This re-audit used no local documentation as an input.

For a complete rerun, use the two commands below. The audit writes explicit market failures to `check_status.csv`; exit 0 means the audit completed, not that every market criterion passed. Missing inputs produce an exact path/error instead of synthetic substitutions. An absent previously saved surface is recorded as `NOT_RUN` in `baseline_comparison.json`; the isolated reproduction still runs from the cleaned quotes. `--skip-reproduction` is appropriate only when the output directory already contains a successful reproduction for the same inputs and parameters.

```bash
conda run --no-capture-output -n volspx python -m unittest discover -s tests -v
conda run --no-capture-output -n volspx python scripts/validate_surface.py
```

Use `--output-dir outputs/diagnostics/validation/<run-name>` to retain an earlier audit's artifacts. The optional before-fix derivative comparison uses `--before-finite-diff`; it is skipped if the specified file does not exist. To recreate the original derivative source directly from the baseline commit:

```bash
mkdir -p outputs/diagnostics/validation
git show 6ced5ec3c194471529f842d592c43e94760da897:src/lv_project/finite_diff.py > outputs/diagnostics/validation/finite_diff_before.py
conda run --no-capture-output -n volspx python scripts/validate_surface.py --before-finite-diff outputs/diagnostics/validation/finite_diff_before.py
```

## Reproduced surface and support

| Quantity | Measured result | Status/interpretation |
|---|---:|---|
| Median spot | 4,755.1100 | Reproduced |
| Selected input quotes / retained expiries | 5,760 / 43 | Reproduced; quotes are not target cells |
| Target grid | 20 maturities × 50 y points | 1,000 cells |
| Target y / padded input y | [-0.15, 0.10] / [-0.25, 0.20] | Existing conventions |
| Pillar maturity range | 7–364 days | Every target maturity, 7–360 days, is in range |
| Finite target w and IV cells | 1,000 / 1,000 | PASS |
| NaNs / unexpected nonfinite cells / nonpositive target w | 0 / 0 / 0 | PASS on this date |
| Target IV range | 0.0960483069–0.5604201775 | Reproduced |
| Maximum saved-baseline versus fresh w difference | 9.986e-17 | PASS at absolute tolerance 1e-12 |
| Maximum saved-baseline versus fresh IV difference | 9.714e-17 | PASS at absolute tolerance 1e-12 |
| Pillar cells filled beyond native y support | 112 / 2,150 | 21 of 43 expiries; full native support FAIL |
| Error against the prescribed constant edge values | 0 | PASS implementation check |
| Target cells inside both native bracketing slices | 967 / 1,000 | Full native bracket support FAIL |
| Target cells whose calendar-repair contributors have native support | 976 / 1,000 | Full contributor support FAIL |
| Conservative intersection of those two target masks | 967 / 1,000 | 33 cells should be excluded from an admissible native region |

The audit records separate time, native-bracket and repair-contributor masks. At an intermediate maturity, native bracket support requires both adjacent input slices. A repaired value may inherit an earlier pillar's constant edge fill, even when the current slice has native quotes. Conversely, a supported earlier maximum can replace an unsupported later fill; nine target cells have contributor support but fail native bracket support on this date. The conservative mask requires both. For tied cumulative maxima, the earliest contributor is retained consistently.

The 112 wing fills are finite values, not missing observations. They must remain identifiable. `pillar_support.csv` gives left/right counts by expiry, and `target_support.csv` gives the masks and interpolation brackets. Synthetic tests confirm that target maturities outside the pillar range remain NaN, constant edge fills match the stated algorithm, sparse expiries are rejected and duplicate strikes are aggregated. NaN-containing derivative and price stencils invalidate the affected checks rather than bridging the gap. Absence of NaNs on this date does not remove the need for these masks.

## Differentiation: defects, fixes and convergence

For neighboring steps h_minus and h_plus, the old interior first derivative was the secant through the outer neighbors. For f(x)=x² on x=[0,1,3], it returned **3** at x=1 instead of **2**. Its error for a quadratic is h_plus−h_minus, so it also fails on the actual maturity grid.

The interior formula now differentiates the local quadratic:

```text
f'(x_i) ≈ [h_plus (f_i−f_(i−1))/h_minus
          +h_minus (f_(i+1)−f_i)/h_plus] / (h_minus+h_plus)
```

It reduces to the existing centered formula on uniform grids and matches `numpy.gradient(..., edge_order=2)` on independent irregular-grid examples. The three-point one-sided first derivatives were already correct and were retained. The unequal-spacing formula is independently documented in the [NumPy gradient notes](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).

The old second-derivative routine copied interior curvature to nonuniform endpoints. For x=[0,.13,.4,.9,1.4] and f=x³, its endpoint estimates were **1.06 and 5.4**, versus exact **0 and 8.4**. Four-point one-sided weights now differentiate the local cubic at each endpoint, with coordinates scaled before a small moment-system solve. The fix also avoids the old approximate-uniform `isclose` branch. Infinite grid coordinates are now rejected explicitly; previously [0,1,inf] passed grid validation and emitted invalid arithmetic.

| Manufactured-function check on the actual maturity grid | Before max absolute error | Current max absolute error |
|---|---:|---:|
| f=x², first derivative | 0.0410958904 | 2.22e-16 |
| f=x², second derivative | 3.77e-14 | 2.18e-14 |
| f=x³, first derivative | 0.0658660161 | 0.0135109777 |
| f=x³, second derivative, endpoints only | 0.1150684932 | 5.68e-14 |
| f=exp(x), first derivative | 0.0350534345 | 0.0056795004 |

The convergence study uses exp(x) on [0,1], whose first and second derivatives are both exp(x). Grids have 17, 33, 65, 129 and 257 nodes: uniform; smoothly graded x=(u+u²)/2; and alternating 1:2 steps normalized to [0,1]. Orders below are log2(previous error/current error) for the final 129→257 refinement, with interior and endpoint maximum errors separated.

| Grid | Current first-derivative interior / boundary order | Current second-derivative interior / boundary order |
|---|---:|---:|
| Uniform | 1.994 / 1.996 | 1.994 / 1.994 |
| Smooth nonuniform | 1.984 / 1.988 | 1.989 / 1.983 |
| Alternating nonuniform | 1.993 / 1.995 | **0.995** / 1.993 |

On alternating grids the old first derivative's interior order was 0.996; the corrected stencil restores approximately second order. On smoothly graded grids the old nonuniform second-derivative boundary order was 0.992; the corrected endpoint order is approximately two. All boundary/interior errors, sizes and orders are saved without rounding in `finite_difference_convergence.csv`.

**Remaining numerical limitation:** the unchanged three-point second derivative is only first order on arbitrary irregular interiors. Its leading error is `(h_plus−h_minus) f'''/3`. On the actual maturity grid, its cubic interior error remains 0.0821917808 at spacing transitions. With only three grid points, the returned second derivative is the quadratic interpolant's constant curvature; endpoints are generally only first order. No smoothing, clipping or NaN filling was introduced. None of these derivative fixes changes the surface build, which does not call these routines.

## Calendar repair

Calendar differences are checked at fixed y with the existing threshold −1e-12. Cumulative maximum leaves the first pillar unchanged, raises later values only, is idempotent, and produces monotone maturity interpolation. Those properties pass manufactured tests.

| Metric | Pillars, 43×50 | Target, 20×50 |
|---|---:|---:|
| Adjacent violations before repair | 128 | 4 |
| Adjacent violations after repair | 0 | 0 |
| Cells changed by more than 1e-14 | 145 / 2,150 | 47 / 1,000 |
| Maximum added total variance | 0.0003950590830 | 0.0002156828750 |
| Mean added variance, all cells | 6.699111254e-6 | 3.187771474e-6 |
| Mean added variance, changed cells | 9.933164963e-5 | 6.782492498e-5 |
| Median / 95th percentile addition, changed cells | 6.565259610e-5 / 0.0003062625469 | 5.920724356e-5 / 0.0001630513363 |
| Maximum relative addition to original w | **61.9419%** | **20.3734%** |
| Maximum IV increase | **301.7010 IV bps** | **110.2544 IV bps** |

An IV basis point here means 0.0001 absolute volatility: 301.7010 bps is 0.03017010, or 3.017010 percentage points. Maxima in different rows need not occur at the same cell. The largest absolute pillar variance repair occurs at 19 days, y=0.0897959184; the largest target variance repair occurs at 30 days, y=0.10. Changed pillars span 12–52 days. The repaired pillar grid has **145 flat maturity intervals out of 2,100**, using an absolute 1e-14 flatness threshold. These terraces can yield zero time slopes, even though calendar monotonicity passes. The audit reports size rather than imposing an unchosen acceptable-repair cutoff.

## Reconstructed calls and strike arbitrage

Prices are European discounted calls in **SPX index points**. The baseline's conventions are explicit: S=4,755.11, T=DTE/365, r=0, q=0, F(T)=S exp((r−q)T), D(T)=exp(−rT), and K=F(T) exp(y). Therefore F=S and D=1 for this run. For positive w:

```text
d1 = −y/sqrt(w) + sqrt(w)/2
d2 = d1 − sqrt(w)
C(K,T) = D [F N(d1) − K N(d2)]
```

Missing/negative w remains NaN; w=0 uses discounted intrinsic value. Independent tests compare this forward formula with the repository's spot Black–Scholes implementation under nonzero r and q. Flat-vol manufactured calls pass every price check.

The tests require `D max(F−K,0) <= C <= D F`, decreasing prices in strike, secant slopes in [−D,0], and nondecreasing secant slopes. **K is nonuniform even on a uniform y grid.** Butterfly curvature is computed from strike divided differences, not from second differences in y or convexity of w. At an interior strike, the butterfly cost is the strike-weighted interpolation of neighbor prices minus the central price; a negative cost fails convexity. Endpoints do not receive butterfly checks. Bounds use 1e-8 index-point tolerance; slope checks use 1e-10. The relation between strike convexity, butterfly arbitrage and calendar monotonicity is described in [Gatheral and Jacquier, Arbitrage-free SVI volatility surfaces](https://arxiv.org/abs/1204.0646).

| Stage | Finite nodes | Strike pairs / triples | Bounds failures | Increasing-strike failures | Slopes below −D | Butterfly failures |
|---|---:|---:|---:|---:|---:|---:|
| Raw pillars, 50 y | 2,150 | 2,107 / 2,064 | 0 | 0 | 63 | 499 |
| Repaired pillars, 50 y | 2,150 | 2,107 / 2,064 | 0 | 0 | 53 | 494 |
| Raw target, 50 y | 1,000 | 980 / 960 | 0 | 0 | 20 | 247 |
| Repaired target, 50 y | 1,000 | 980 / 960 | **0** | **0** | **15** | **248** |
| Repaired pillars, 201 y | 8,643 | 8,600 / 8,557 | 0 | 3 | 423 | 2,901 |
| Repaired target, 201 y | 4,020 | 4,000 / 3,980 | 0 | 1 | 133 | 1,361 |
| Repaired pillars, 1,001 y | 43,043 | 43,000 / 42,957 | 0 | 24 | 2,387 | 10,716 |
| Repaired target, 1,001 y | 20,020 | 20,000 / 19,980 | 0 | 10 | 797 | 3,857 |

**PASS:** individual call bounds on every tested grid. **PASS only at coarse resolution:** strike monotonicity on the 50-point target. **FAIL:** vertical-spread bounds and butterfly convexity. **FAIL at denser resolution:** strike monotonicity. Dense runs reevaluate the existing slice interpolation, calendar repair and maturity interpolation directly from the same quotes; they do not interpolate the saved 50-point CSV or introduce a new fitting method. Counts across resolutions are not convergence estimates for a smooth surface; nodes and the number/placement of stencils differ.

On the repaired 50-point target, **243 of 248** butterfly failures have native-supported repair contributors across the entire triple; wing exclusion alone cannot resolve the result. On the 1,001-point target, **3,838 of 3,857** failures satisfy that contributor test. The worst baseline target butterfly cost is **−0.8760123394** index points at **120 days**, **y=−0.0836734694**, **K=4,373.424584**. Its slope change is −0.078519; the most negative target strike curvature is −0.003745182325. The worst target secant slope is **−1.0159018213**, below the permitted −1.

These failures are substantial relative to roundoff. At slope tolerance 1e-6, all 248 coarse target butterfly failures persist. **229** have negative butterfly cost greater than 0.01 index points in magnitude, and **147** exceed 0.1. At 1,001 y points, all ten increasing-price segments survive a 1e-4 slope threshold, and 1,221 butterfly costs are below −0.01. See `price_tolerance_sensitivity.csv`. Calendar repair changes coarse target butterfly failures from 247 to 248; it does not enforce strike convexity.

Using each selected quote's unsmoothed vendor IV, the same median spot and zero carry, **5,416 / 5,760 (94.03%)** reconstructed calls are outside the original bid/ask interval. Median absolute mid-price error is **12.994493** index points; RMSE **29.732971**; maximum **92.385008**. This is a failed consistency check under the stated conventions, not evidence that the vendor quotes are themselves arbitrageable. Vendor IV carry/settlement inputs were not supplied or calibrated, and the existing path does not invert observed mid-prices. The errors are saved in `vendor_iv_quote_repricing.csv`.

## Changes, evidence and remaining work

Targeted production fixes are confined to `src/lv_project/finite_diff.py`: the correct unequal-spacing interior first derivative, endpoint second derivatives evaluated at the endpoints, finite-coordinate validation, and explicit accuracy/NaN documentation. The previously empty `src/lv_project/diagnostics.py` now contains discounted-call reconstruction and support-aware strike checks. `scripts/validate_surface.py` provides the reproducible market audit. Eighteen tests cover polynomial exactness, manufactured convergence including boundaries, actual maturity spacing, axis handling, NaNs, support, duplicates, sparse expiries, calendar repair, forward/discount conventions, known price violations and preservation manifests. The surface model, raw-data cleaner, existing baseline outputs, local-volatility stubs and hedging stubs were not changed.

The diagnostic artifacts are generated under `outputs/diagnostics/validation/` and remain excluded by `.gitignore`. They are not distributed in the repository; the paths and plot links below refer to files produced locally by the audit. Scientific results and reproduction instructions are retained in this report, while development history and investigation notes remain local.

| Artifacts | Purpose |
|---|---|
| `validation_summary.json`, `check_status.csv`, `baseline_comparison.json` | Machine-readable measurements and explicit PASS/FAIL outcomes |
| `reproduction_command.json`, `build_surface.log`, `validation_run.log` | Market audit execution evidence; the original local audit also retains `baseline_commands.json` and test logs |
| `preservation_before.json`, `preservation_after.json` | Data and baseline-output preservation evidence |
| `finite_diff_before.py`, `finite_difference_benchmarks.csv`, `finite_difference_convergence.csv` | Original routine and measured numerical comparisons |
| `pillar_support.csv`, `target_support.csv` | Native support, edge fills, contributor masks and maturity brackets |
| `pillar_calendar_repairs.csv`, `target_calendar_repairs.csv` | Full raw/repaired w, relative changes and IV effects |
| `call_price_checks.csv`, `call_checks_by_slice.csv`, `call_violations.csv`, `price_tolerance_sensitivity.csv` | Node/pair/triple checks, failure locations, magnitudes and tolerance robustness |
| `vendor_iv_quote_repricing.csv` | Raw vendor-IV repricing under the explicit zero-carry convention |
| `reproduction/` | All seven isolated-build outputs, including original-style interactive plots |

Six inspected plots provide complementary evidence: [smile slices](../outputs/diagnostics/validation/smile_slices.png), [repair magnitudes](../outputs/diagnostics/validation/calendar_repairs.png), [edge support](../outputs/diagnostics/validation/pillar_edge_support.png), [boundary convergence](../outputs/diagnostics/validation/finite_difference_boundary_convergence.png), [strike curvature](../outputs/diagnostics/validation/call_strike_curvature.png), and [worst baseline call slice](../outputs/diagnostics/validation/worst_call_slice.png). The smiles show the very high short-dated left wing and a right-wing bump; repairs concentrate at short maturities. Negative-curvature regions and oscillating secant slopes corroborate the price-table failures. Static plots work offline; the reproduced HTMLs retain the existing Plotly CDN dependency.

Remaining failures are **model/data-convention limitations**, not automatically correctable coding defects: incomplete native wing coverage, large repairs and maturity terraces, vendor-IV/zero-carry inconsistency, and strike arbitrage in the linearly interpolated variance surface. Piecewise-linear w has slope discontinuities; finite-difference refinement does not create a reliable continuous second derivative at those knots. The discrete checks cover the stated domain and resolutions; no global continuous no-arbitrage certificate or tail-density normalization was attempted.

The following checks were **NOT RUN**: vendor-convention repricing and parity-calibrated forwards/discounts (those conventions are unestablished; the active cleaned dataset retains calls only, although put fields exist in raw files), alternative interpolator or constrained-fit calibration, continuous wing/expiry extrapolation certification, Dupire extraction, PDE vanilla repricing, and hedging. The latter stages are deferred until surface validity is established. No smoke, numerical, support, calendar or reconstructed-price check in the reported audit was blocked by missing inputs.

Recommended next steps, in order:

1. Establish forwards, discounts, settlement/day-count inputs and quote-IV consistency before refitting. Recompute IV from cleaned prices using those conventions; evaluate OTM put/call splicing and deep-ITM quality with the available raw put fields.
2. Use the saved price violations to evaluate a strike-convex price fit or an explicitly constrained variance parameterization, with maturity constraints and support masks. Changing linear interpolation to a smoother interpolator alone does not guarantee no arbitrage.
3. Retain a maximum-repair budget and maturity-terrace diagnostics chosen from quote uncertainty. Require price bounds, vertical-spread bounds and butterfly checks to pass on native-supported coarse and dense grids before accepting a surface.
4. Test derivative stability on the chosen fitted surface and actual nonuniform maturities, with interior/endpoint/support masks separated. Choose any wider stencils or regularization against measured errors and noise, rather than assuming second order at maturity-spacing jumps.
5. Proceed to Dupire only after those surface criteria pass, then undertake pricing and hedging validation as separate stages.
