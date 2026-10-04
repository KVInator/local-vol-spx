# Forward PDE validation

## Method

The solver propagates normalized call prices:

$$
c(z,T)=\frac{C(F(T)z,T)}{D(T)F(T)}.
$$

With log-moneyness $y=\log z$ and $u(y,T)=c(e^y,T)$,
the forward equation is

$$
u_T=\frac{1}{2}v(e^y,T)(u_{yy}-u_y),
$$

where $v$ is local variance.

The implementation uses central spatial differences and
Crank–Nicolson time stepping with tridiagonal banded solves.
Rannacher startup steps are used for the analytical payoff benchmark.

Time grids align with expiry pillars. Coefficients use the right
side at the beginning of an interval and the left side at its end.

## Analytical validation

The test suite contains 65 passing tests, including PDE checks for:

- Constant volatility and spatial convergence.
- Time-dependent variance with known integrated variance.
- A variance discontinuity with explicit time-grid alignment.
- Time refinement from a smooth initial curve.
- Zero variance.

The Black convergence benchmark uses volatility 20%, maturity
0.5 years and exact analytical boundary prices.

Reported price errors use forward 100 and discount factor 0.98.

| Spatial intervals | Maximum price error |
| ---: | ---: |
| 100 | 0.01394044 |
| 200 | 0.00347840 |
| 400 | 0.00086960 |
| 800 | 0.00021746 |

Observed spatial orders are approximately 2.00.
Time refinement against a finer time solve on the same spatial
grid also gives orders approximately 2.00.

## SPX surface consistency

Reference surface: `spx_2023-09-01_regularized`.

The check propagates the fitted 21-day normalized call curve
through 60 days over forward log-moneyness [-0.09, 0.09].

Local variance comes from the fitted surface's Dupire derivatives.
Both spatial boundary prices are supplied by that surface.

The comparison covers 45 evaluation maturities. Aggregate error
statistics exclude the imposed initial curve and spatial boundaries.

| Spatial intervals | Maximum error, index points | RMS error, index points |
| ---: | ---: | ---: |
| 100 | 0.00535271 | 0.00227771 |
| 200 | 0.00110713 | 0.00056643 |
| 400 | 0.00039412 | 0.00015887 |
| 800 | 0.00008587 | 0.00003634 |

Across the full spatial refinement sequence, the observed order
is approximately 1.99.

The finest comparison grid contains 36,045 generated points.
Its largest error occurs at 39.525 days and log-moneyness 0.01665.

Sampled checks on the finest PDE grid report:

- Zero increasing-price intervals.
- Zero discounted vertical-spread bound violations.
- Zero negative butterflies.

Prices were not modified after solving. No extrapolation was used.

## Interpretation and scope

These results demonstrate numerical consistency between the
regularized call surface, its extracted local variance and forward
PDE propagation within the supported domain.

The market comparison uses fitted initial and boundary prices.
It does not validate propagation from time zero or a global
strike-tail extension.

Errors are measured against the fitted surface. They do not
measure agreement with original bid–ask quotes. The regularized
calibration retains 110 of 723 fitted prices outside those bands.

## Reproduction

From the repository root, with the `volspx` environment active:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v

PYTHONPATH=src python scripts/08_pde_black_benchmark.py

PYTHONPATH=src python scripts/09_pde_surface_consistency.py \
  --surface data/processed/surfaces/spx_2023-09-01_regularized/call_surface.json
```

Generated diagnostics are saved under `outputs/pde_diagnostics/`.
The market consistency audit records input and implementation hashes.