# Short-end maturity extension

The SPX surface for 2023-09-01 contains ten fitted expiry pillars
covering 4–60 calendar days. Its combined validated central
log-moneyness domain is [-0.01, 0.01].

## Construction

For maturities below the first fitted expiry T1, total variance is:

w(y, T) = (T / T1) * w(y, T1).

This holds the first expiry's implied-volatility smile constant
at fixed forward log-moneyness.

The forward is interpolated logarithmically from spot to the first
fitted forward. The discount factor is interpolated logarithmically
from 1 to the first fitted discount factor.

At maturity zero, call prices equal intrinsic payoff. Local variance
uses its right-hand limit. Strike and maturity derivatives of the
initial payoff are not evaluated at its kink.

## Validation

All 70 unit tests passed, including analytical Black benchmarks
and derivative checks using an independent quadratic call curve.

The market extension was evaluated at seven maturities and 201
moneyness points, producing 1,407 observations.

All sampled price-bound, derivative and local-variance checks passed.
The price join error at the first expiry was 4.179e-11 index points.

Day-zero local volatility ranged from 5.0601% to 8.4676%.
At day 4, its left-hand value ranged from 4.7313% to 7.4604%.

The maximum left/right local-volatility jump at day 4 was
7.021506 percentage points. This reflects the change in maturity
derivative between the extension and the fitted surface.

## Scope and assumptions

The interval before day 4 is a model extension without fitted
shorter-expiry observations.

Strike tails have not been constructed. The checks do not establish
global strike-domain validity or complete PDE repricing from time zero.

Data-provider identification remains unconfirmed, PM settlement is
inferred, and discounting uses the documented Treasury yield proxy.

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=src python -m unittest discover \
  -s tests -p 'test_*.py' -v

PYTHONPATH=src python scripts/10_validate_short_end.py \
  --surface data/processed/surfaces/spx_2023-09-01_combined_central/call_surface.json \
  --spot 4516.02
```