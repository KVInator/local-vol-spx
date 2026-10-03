# Single-expiry calibration findings

## Scope

Observation date: 2023-09-01.
Expiry: 2023-10-06.
Maturity: 35/365 years.

The study used 120 out-of-the-money option quotes spanning strikes
4110–5000. Puts were converted to equivalent call prices using put–call
parity, with forward 4534.174752 and discount factor 0.99479528.

Discounting used a Treasury-based proxy. Settlement conventions and
discount-curve construction require further work before extending the
analysis across expiries.

## Price consistency and fitting

The observed midpoint grid contained 43 negative butterflies. Two
adjacent strike triplets were incompatible with convexity even after
allowing prices to vary within their original bid–ask bands.

A linear programme found that a minimum half-spread multiplier of
1.43478261 was necessary for a feasible discrete call-price grid.

Ordinary natural cubic and PCHIP interpolation did not preserve
convexity between the repaired observations.

A constrained cubic B-spline provided analytic strike derivatives and
enforced price, slope and curvature restrictions throughout the fitted
strike interval. The 30-interval spline required a minimum multiplier
of 1.53813624.

## Sensitivity findings

At smoothing weight 1e-3, the 30- and 60-interval splines differed by
approximately 0.00656 index points on the comparison grid and 1.166%
in integrated curvature.

Smaller tested quote perturbations produced modest integrated
curvature responses. One larger perturbation produced a 27.396%
response under the fixed cap of 1.88865377.

A focused investigation showed that the feasibility margin contracted
and the number of binding quote caps increased from two to eight.
Curvature roughness rose to approximately 16.2 times its baseline value.

Using the previously studied wider cap of 3.03813724 reduced that
scenario's curvature response to 1.488%, relative to its own baseline.
It also permitted greater deviations at individual quotes.

The wider-cap scenario still changed curvature at strike 5000 by
approximately 46.4%. A small integrated response therefore does not
establish stability at every strike.

## Implementation consequences

- Treat shape feasibility, quote agreement and derivative sensitivity
  as separate diagnostics.
- Keep calibration settings explicit.
- Report infeasibility separately from numerical solver failure.
- Use analytic spline derivatives, with finite differences as checks.
- Assess boundary behaviour separately from aggregate error measures.
- Establish a supported domain before extracting local volatility.

The tested caps remain research benchmarks. Selecting a general
calibration policy requires evidence across further expiries and dates.