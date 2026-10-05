# Day-zero PDE repricing

This experiment loads the first-expiry smoothing-weight 0.1 candidate and
the other nine existing core models through its `tail_slices.json` manifest.
It recomputes the global calendar checks, interpolates normalized call
prices between expiries, and extends the first smile to day zero.

## Files

Add the following complete files at these repository paths:

| File | Role |
| --- | --- |
| `src/tailed_surface.py` | Load and interpolate the coupled tails; evaluate the short-end extension with stable density factors. |
| `scripts/15_pde_day_zero.py` | Run PDE repricing, refinement studies, quote comparisons and boundary sensitivity checks. |
| `tests/test_tailed_surface.py` | Eight tests using independent analytical Black price fixtures. |
| `docs/day_zero_repricing.md` | Explain this experiment and its interpretation. |

The existing `coupled_tails.py`, `short_end.py`, `forward_pde.py` and
primary calibrated model files are reused. This update does not change
the tail powers, refit the expiry cores or clip local volatility.

## Run

From the repository root, with the `volspx` environment active:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

Then run:

```bash
PYTHONPATH=src python scripts/15_pde_day_zero.py \
  --tails data/processed/surfaces/spx_2023-09-01_combined_central/first_expiry_conditioning/weight_0p1/tail_slices.json \
  --quotes-root data/processed/surfaces/spx_2023-09-01_combined_central \
  --spot 4516.02
```

The console prints progress as each solve starts and finishes. The first
preflight step recovers the first-expiry implied volatilities over the
widest numerical domain and can take longer than the later cached calls.

## Default experiment

- Start from the normalized intrinsic call payoff at day zero.
- Propagate to the last fitted maturity, 60 days.
- Align all coefficient changes with the ten fitted maturities.
- Use Crank-Nicolson with two initial Rannacher steps.
- Spatial study: 400, 800 and 1,600 intervals on log-moneyness [-0.5, 0.5],
  with at most 1/16 day per nominal time step.
- Time study: 8, 16 and 32 steps per day at 1,600 spatial intervals,
  compared with a 64-step-per-day reference on the same spatial grid.
- Domain study: widen to [-0.75, 0.75] and [-1, 1] with the same spacing.
- Boundary study: compare fitted-surface boundary values against the
  asymptotic approximation of intrinsic value on the left and zero on
  the right, using both the narrowest and widest domains.
- Report surface errors over log-moneyness [-0.09, 0.09] and compare every
  retained observed quote separately at its actual strike.

These domain bounds include assumed tails. They do not describe observed
market coverage. The shorter expiries retain their selected narrower
cores, while the longer expiry cores retain their wider coverage.

## Stable short-end calculation

The modelling assumption remains constant first-expiry implied volatility
at fixed forward moneyness, so w(y,T)=(T/T1)w(y,T1).

Write alpha=T/T1. The implementation evaluates the density factor as

```text
g(alpha) = (1-alpha) g0 + alpha g1
           + alpha (1-alpha) w_y^2 / 16.
```

The first-expiry factor g1 comes directly from the fitted normalized call
curvature. This avoids subtracting large derivative terms to recover a
small positive factor. It does not remove genuinely large model-implied
local volatility. The values remain visible in the preflight output and
audit.

The same formula also explains why nonnegative endpoint factors support
nonnegative density factors between them. The numerical preflight samples
the widest domain and points immediately adjacent to the tail joins. It
does not certify a strictly positive day-zero factor at every real strike.

## Outputs

Diagnostics are written under:

```text
outputs/pde_diagnostics/spx_2023-09-01_combined_central/day_zero_weight_0p1/
```

| Output | Interpretation |
| --- | --- |
| `space_convergence.csv` | Errors as spatial spacing decreases. |
| `time_convergence.csv` | Time error relative to a finer time grid, with space held fixed. |
| `domain_sensitivity.csv` | Changes in central prices as the domain widens. |
| `boundary_sensitivity.csv` | Sensitivity to the boundary policy. |
| `quote_repricing.csv` | PDE versus candidate prices and separate residuals versus original quote midpoints. |
| `maturity_errors.csv` | Errors at each evaluated maturity. |
| `all_cases.csv` | Every solve and its sampled central shape checks. |
| `day_zero_audit.json` | Input/code hashes, conventions, preflight statistics and reference-case results. |
| `day_zero_repricing.png` | Four panels showing the numerical comparisons. |

The saved PDE grid and loadable day-zero surface specification are under:

```text
data/processed/surfaces/spx_2023-09-01_combined_central/day_zero_weight_0p1/
```

The day-zero specification refers to the candidate manifest and its
existing NPZ core models using relative paths. Keep those files together
when moving a run.

## Interpretation

PDE-versus-candidate errors measure numerical reconstruction. The
candidate-versus-midpoint residual measures calibration movement; it
already contains the cost of convexity repair, smoothing and tails.
Comparing the two residuals prevents calibration error from being
mistaken for PDE error.

The fitted-surface boundaries are model-derived. The alternative boundary
and widening studies test sensitivity to those finite boundaries, but
they do not validate the extrapolated tails against market data.

Use decreasing errors, stability of observed-quote prices, boundary
sensitivity and sampled shape checks to assess numerical reliability.
There is no automatic promotion of the candidate and no arbitrary
volatility cap. A successful reconstruction alone does not establish
hedging improvement or economic correctness of the short-end assumption.

The first-expiry surface is piecewise smooth in strike and the
interpolated surface has one-sided maturity derivatives. Uniform
second-order convergence is therefore not assumed for this market test.

## Verification of the update

The update was checked in an isolated synthetic environment using the
uploaded coupled-tail, short-end and PDE implementations. The remaining
dependencies were independent analytical Black/Brent fixtures. All
eight new tests passed, and the complete runner produced its tables,
audit, loadable model and figure for a synthetic two-pillar surface.
The user's complete repository suite and SPX dataset were not run there.
