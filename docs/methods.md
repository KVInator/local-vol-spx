# Local Volatility and Hedge Methodology

The project compares SPX delta-hedging methods using daily option quotes and
fixed-contract historical trade paths. The main question is whether fitting a
local-volatility surface improves realized hedge errors relative to Black delta.

The calculation has three stages: estimate daily carry and calibrate the
surface, calculate hedge positions from that date's information, then account
for each position over its holding period. Data preparation is described in
[Market Data and Preparation](data.md).

## Andreasen–Huge Calibration

Andreasen–Huge (AH) is the main surface model. It fits normalized call prices
expiry by expiry on a finite strike grid. With normalized strike $z=K/F(T)$
and call price $c=C/(DF)$, the normalized Dupire equation gives local variance as

$$
a(z,T)=\frac{2c_T(z,T)}{z^2c_{zz}(z,T)}.
$$

The fit uses the original quote half-spreads and regularization. Calibration
residuals are reported in both option price units and half-spread units, so
pricing errors can be compared with the width of the observed quotes.

The proxy volatility parameters used by the optimizer are separate from the
local volatility recovered from price derivatives. The notebook shows the
recovered local volatility, together with implied volatility and total variance.

A Gaussian variance blend is used near the short end of the hedge diffusion.
This changes the diffusion relative to the native AH price surface. The
[validation study](validation.md) compares the resulting prices and Greeks
under numerical refinements.

## Independent Total-Variance Surface

The diagnostic comparison starts from quote-implied total variance,
$w(y,T)=\sigma_{\mathrm{imp}}(y,T)^2T$, where $y=\log(K/F)$.
PCHIP interpolates across log-moneyness and linear interpolation joins expiries.
Evaluation stays within contiguous valid quote segments and expiry support.

This construction examines the sensitivity of volatility derivatives to an
independent interpolation choice. It is unconstrained and can retain negative
calendar derivatives or densities. Invalid local-variance values remain
unavailable. It has not been accepted as an alternative hedge diffusion.

## Hedge Rules

| Method | Calculation |
| --- | --- |
| Black | Spot delta at market-implied IV, with that IV held fixed |
| AH PDE | Delta from backward pricing under the saved local-variance function |
| Surface sticky strike | Fitted IV held fixed at the strike |
| Surface sticky delta | Frozen normalized forward-call-delta smile |
| LV smile | Hull–White local-volatility smile approximation |
| Empirical MV | Past-data quadratic correction to Black delta |
| Unhedged | Zero index position |

The smile corrections use Black-formula vega evaluated at the relevant IV.
An AH calibration-pillar bump followed by PDE repricing is a separate sensitivity
calculation and remains unfinished.

The empirical minimum-variance model fits a quadratic Black-delta correction
without an intercept, separately by root and option kind. It uses a fixed
rolling window. A training observation is admitted only when its outcome
endpoint is strictly earlier than the prediction timestamp. Warm-up, rank and
conditioning failures remain visible in coverage.

## Historical Hedge Paths

The same option contract is held for 1, 5 or 10 reference sessions. The default
holding/rebalance pairs are 1/1, 5/1, 5/5, 10/1 and 10/5. Each rebalance uses
current-date quotes, carry and surface estimates.

The option position is short. The midpoint scenario uses midpoint entry and
exit marks. Spread scenarios enter at bid and close at ask; the hedge-fee
scenario also charges 1 bp of traded hedge notional. The account includes
funding, rebalancing and liquidation.

Daily models are produced before the aggregate hedge-path evaluation. During
an active run, a daily surface may therefore be available while the hedge
summary still covers an earlier part of the history.

## Performance Measures

Candidate and Black hedges are paired on the same option entries. Gain is

$$
\mathrm{Gain}=1-
\frac{\sum_i P_{i,\mathrm{candidate}}^2}
     {\sum_i P_{i,\mathrm{Black}}^2}.
$$

Positive Gain means lower squared hedge error than Black. These errors are
measured around zero, so mean error contributes alongside dispersion. RMS
error, MAE and mean P&L describe additional parts of the outcome.

Pairwise comparisons use the entries available to each candidate and Black.
The common-entry comparison restricts every method to the same intersection.
Results are also grouped by maturity, moneyness, delta and year.

## Uncertainty and Evaluation Scope

The bootstrap resamples whole entry dates, using mean block lengths of 10, 20
and 40 sessions. Contracts entered on the same date remain together, and date
blocks allow for dependence from overlapping holding periods.

September and October 2023 were examined during development. The full-history
study is therefore retrospective. Past-only empirical estimation prevents
future outcomes from entering an individual fit, but it does not make the
historical sample an untouched final test set.
