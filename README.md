# SPX Local Volatility and Delta Hedging

Daily SPX surface calibration and a retrospective comparison of delta-hedging
rules. Andreasen–Huge supplies the fitted price surface and Dupire local
variance; a backward PDE calculates prices and spot Greeks. Black delta,
smile-based rules and a rolling empirical minimum-variance correction are
compared on the same observed option contracts.

The research question is whether fitting today's European option prices leads
to smaller subsequent hedge errors. Calibration residuals, numerical convergence
and historical hedge errors are evaluated separately.

[Technical report (PDF)](docs/report.pdf) · [Report source](docs/report.md) ·
[Full results](docs/results.md) · [Reproduction](docs/reproducibility.md)

![SPX implied and local volatility surfaces](docs/figures/daily_surface.png)

Saved surfaces for 19 June 2015, with log forward-moneyness restricted to
−0.09 through +0.09. Volatility is annualized percent. Unsupported quote
interpolation and unstable local-variance estimates remain gaps.

## Historical results

| Study input or output | Scope |
| --- | --- |
| Raw data | 132 monthly files, 14,750,241 rows, 2013–2023 |
| Reference calendar | 2,768 sessions |
| Selected option entries | 44,500, targeting 21-, 35- and 45-day expiries |
| Holding / rebalancing intervals | 1/1, 5/1, 5/5, 10/1 and 10/5 sessions |
| Accounting scenarios | Midpoint marks; option bid–ask; option bid–ask plus hedge fee |
| Compared outcomes | 3,879,048 strategy/schedule/scenario records; zero pending paths |

An entry retains its original contract. The outcome count reuses entries across
methods, schedules and scenarios; it is not an independent sample size.

For terminal funded hedge errors $P_{i,m}$ on matched entries, the comparison is

$$
\mathrm{Gain}_m=1-\frac{\sum_i P_{i,m}^2}{\sum_i P_{i,\mathrm{Black}}^2}.
$$

Positive Gain means lower squared funded error than Black. Squaring around zero
includes both squared mean error and dispersion; Gain is distinct from demeaned
variance reduction and investment returns.

For **five-session holdings, daily rebalancing and midpoint marks**:

| Method | Pairwise Gain | Common-entry Gain | Common-entry 95% bootstrap interval |
| --- | ---: | ---: | ---: |
| AH PDE | −25.58% | −30.16% | [−55.50%, −7.57%] |
| Empirical MV | −8.40% | −7.92% | [−25.78%, +9.08%] |
| LV smile | −12.38% | −12.40% | [−30.03%, +4.53%] |
| Sticky delta | −136.71% | −136.74% | [−188.49%, −91.90%] |
| Sticky strike | +0.15% | +0.15% | [−0.04%, +0.37%] |

The common-entry sample contains **33,763 entries on 2,406 entry dates**. AH
therefore produced 30.16% more squared funded error than Black in this comparison.
Its reported interval and sticky delta's lie below zero. The remaining intervals
include zero and do not resolve the direction of the difference from Black or
establish equivalence.

![Common-entry historical hedge comparison](docs/figures/history_gain.png)

Intervals use 500 stationary entry-date block-bootstrap replicates with mean
block length 20 sessions; 10- and 40-session results are also saved. Resampling
keeps contracts from the same date together. The intervals are conditional on
the retained model estimates and selection, without refitting or simultaneous
adjustment across methods, schedules and scenarios.

The adverse AH result concerns this implementation, including its short-end
variance blend, under the stated conventions. It does not isolate the cause of
the difference or establish a general ranking of local-volatility hedges.
[Results](docs/results.md) includes completion counts, exclusions and range views.

### Financial and statistical scope

The history is retrospective: development included inspection of September and
October 2023. Empirical fits use only outcomes completed before each prediction,
but the broader sample is not an untouched holdout.

The raw files do not verify every SPX/SPXW root or official fixing. The research
key is `UNKNOWN`, with PM fixing assumed. Pricing and cash funding use 5%; the
pricing forward is parity-implied, while hedge-account dividend cash is zero.
The hedge is a fractional synthetic index position without observed executable
spreads or futures basis. Cost scenarios are sensitivities, not verified fills.
These conventions limit interpretation to a historical hedge-error comparison.
[Assumptions](docs/assumptions.md) gives the full specification.

## Local and implied volatility

Fix one valuation date as time zero. Let $S_t$ be the index level, $K$ the
strike, $t$ elapsed time and $T$ expiry, all times in ACT/365F years. Call prices
$C$ and strikes use index points. $r(t)$ is the deterministic pricing rate and
$q(t)$ the yield implied by pricing carry; $D(t)=\exp[-\int_0^t r(u)du]$ and
$F(t)=S_0\exp[\int_0^t(r(u)-q(u))du]$. $W^{\mathbb Q}$ denotes risk-neutral
Brownian motion; $\Phi$ and $\phi$ are the standard normal distribution and
density. Volatilities are annualized decimals. Pricing yield is distinct from
the zero dividend cash assumed in the hedge account.

Under deterministic rates and pricing yield, the risk-neutral index dynamics are

$$
\frac{dS_t}{S_t}=[r(t)-q(t)] dt+
\sigma_{\mathrm{LV}}(S_t,t) dW_t^{\mathbb Q}.
$$

Local volatility is a deterministic function of index level and time. A constant
coefficient recovers Black–Scholes; state and time dependence allow a smile and
term structure.

Implied volatility is a different quantity. For a European call with strike
$K$, time to expiry $T$, forward $F(T)$ and discount factor $D(T)$, it solves

$$
C_{\mathrm{market}}(K,T)
=D(T)\big[F(T)\Phi(d_1)-K\Phi(d_2)\big],
\qquad
 d_1=\frac{\log(F(T)/K)+\tfrac12\sigma_{\mathrm{imp}}^2T}
 {\sigma_{\mathrm{imp}}\sqrt{T}},
\quad d_2=d_1-\sigma_{\mathrm{imp}}\sqrt{T}.
$$

It is the constant Black volatility that reproduces one option price. A daily
implied-volatility surface collects those values across strikes and maturities.
Local volatility instead specifies the instantaneous diffusion used to price
and hedge within a model. Recovering it requires information about how option
prices change across both strike and maturity.

The surface coordinates are

$$
y=\log \left(\frac{K}{F(T)}\right),
\qquad w(y,T)=\sigma_{\mathrm{imp}}(y,T)^2T.
$$

Here $w$ is total implied variance.

## Dupire's equation

For a sufficiently regular European call-price surface, deterministic carry
and positive strike curvature, the forward pricing equation gives

$$
\sigma_{\mathrm{LV}}^2(K,T)=
\frac{2\big[C_T+q(T)C+(r(T)-q(T))K C_K\big]}
{K^2 C_{KK}}.
$$

All derivatives refer to options observed on one valuation date. The maturity
derivative moves along that day's term structure; it is not a difference
between two historical quote dates. The coefficient at state $K$ and future
time $T$ is inferred from the current price surface around that strike and
maturity.

Strike curvature is central to the calculation. In the regular deterministic-
rate setting, $C_{KK}$ is the discounted terminal risk-neutral density. Small
curvature can therefore make the recovered coefficient sensitive to small
price errors. Convexity, maturity consistency, interpolation support and
numerical derivatives matter before a local-volatility plot can be interpreted.

### Coordinates used in the implementation

The calibration removes deterministic carry by defining

$$
z=\frac{K}{F(T)},\qquad
c(z,T)=\frac{C(F(T)z,T)}{D(T)F(T)}.
$$

In these coordinates,

$$
\partial_T c(z,T)=\frac12 a(z,T)z^2\partial_{zz}c(z,T),
\qquad
 a(z,T)=\frac{2c_T(z,T)}{z^2c_{zz}(z,T)}.
$$

The time derivative is taken at fixed normalized strike $z$. The function
$a(z,T)$ supplies local variance at the physical index level $F(T)z$. The
initial normalized call payoff is $c(z,0)=(1-z)^+$.

The backward pricer uses the same carry normalization.

## Calibration and surface diagnostics

### Quotes and carry

The quote audit checks timestamps, contract fields, duplicate keys, index marks
and bid–ask validity. Of 2,768 reference sessions, 24 lack observations; 21
observed early-close dates are excluded pending clock review.

The primary parity fit uses $|\log(K/S)|\leq0.03$ and requires at least six
selected call-put pairs per expiry. The forward fit uses the assumed pricing
discount factor. Put-call parity is

$$
C(K,T)-P(K,T)=D(T)[F(T)-K].
$$

An out-of-the-money quote is chosen at each calibration strike and converted
to an equivalent call when necessary. Original quote half-spreads are retained
as calibration weights. Quotes without an invertible positive midpoint IV
remain visible in price-space diagnostics.

### Andreasen–Huge calibration

Each expiry is calibrated sequentially on a finite normalized-strike grid.
Within a maturity interval, a fully implicit step has the form

$$
\big[I-\Delta T_i A(\nu_i)\big]c_i=c_{i-1},
\qquad A(\nu_i)=\tfrac12\nu_i(z)^2z^2\partial_{zz}.
$$

The positive proxy volatility $\nu_i$ is parameterized through log-volatility
controls. Its parameters minimize equivalent-call price residuals in original
half-spread units, with a penalty on adjacent log-control differences:

$$
\min_{p_i}
\sum_j\left[
\frac{C_{\mathrm{model}}(K_j,T_i;p_i)-C_{\mathrm{mid},j}}
{h_j}\right]^2
+\lambda\Vert L p_i\Vert ^2,
\qquad h_j=\tfrac12(C_{\mathrm{ask},j}-C_{\mathrm{bid},j}).
$$

$L$ is the first-difference matrix for neighboring controls. Reported residuals
include both index-point errors and half-spread errors, with counts of prices
outside the original bid-ask bands.

**The fitted proxy volatility is not the recovered local volatility.** The
native price construction determines $c_T$ and $c_{zz}$; their ratio determines
$a$. The coefficient implementation uses positive density and derivative
calculations in logarithmic form to preserve small wing quantities. It does
not obtain curvature by differentiating a piecewise-linear price interpolant.

### Short-end diffusion blend

Write $a_y(y,t)=a(e^y,t)$. Before the first fitted maturity $T_1$, the
historical hedge diffusion blends
native variance with a Gaussian average in log normalized state:

$$
\widetilde a_y(y,t)=
[1-\omega(t)]a_y(y,t)+\omega(t)(G_\rho*a_y)(y,t),
\qquad \omega(t)=\left(1-\frac{t}{T_1}\right)^2,
\quad 0\lt t\lt T_1.
$$

The implementation uses a finite discrete Gaussian filter and retains raw
variance from $T_1$ onward. The historical radius is $\rho=0.0005$.
This blend changes the pricing diffusion relative to the native fitted price
surface. Its effect on prices, deltas and gamma is measured separately; a
smoother coefficient is not automatically a more accurate one.

### Independent surface diagnostics

A quote-based comparison interpolates total variance with PCHIP in $y$ and
linearly across expiries. It uses contiguous valid quote support. For a smooth
$w(y,T)$, the total-variance form of Dupire is

$$
a_y(y,T)=\frac{w_T}{g(y,T)},\qquad
 g=\left(1-\frac{y w_y}{2w}\right)^2
 -\frac{w_y^2}{4}\left(\frac1w+\frac14\right)+\frac{w_{yy}}2.
$$

Unsupported regions, negative calendar derivatives, invalid density factors
and unstable derivative stencils remain unavailable. This interpolation is a
diagnostic comparison and was not used as a historical hedge diffusion.

A constrained SSVI fit provides a second comparison, with sufficient static-
arbitrage restrictions on its shared-shape family. Its inspected quote fit is
less accurate than AH. Both SSVI and quote-PCHIP are diagnostics; neither supplies
the diffusion used in the historical hedge study. [Methods](docs/methods.md)
and the [report](docs/report.md#quote-pchip-and-constrained-ssvi-diagnostics)
give their specification.

![Observed quotes and the saved 35-day maturity slice](docs/figures/daily_slice.png)

19 June 2015, 24 July expiry: equivalent-call quote fits, original half-spread
residuals and local-volatility diagnostics.

## Backward pricing and Greeks

For a European payoff, the usual local-volatility backward equation is

$$
V_t+[r(t)-q(t)]S V_S+\tfrac12\sigma_{\mathrm{LV}}^2(S,t)S^2V_{SS}-r(t)V=0,
\qquad V(S,T)=(S-K)^+
$$

for a call, with the corresponding put payoff for a put. The implementation
works with $x=S/F(t)$ and $y=\log x$. For the normalized conditional payoff
$U(y,t)$, its generator gives

$$
U_t+\tfrac12\widetilde a_y(y,t)(U_{yy}-U_y)=0.
$$

The solver marches backward using midpoint Crank–Nicolson, implicit Rannacher
steps at the payoff, maturity breaks aligned with coefficient changes and a
time mesh graded toward zero. Coefficients are sampled at positive times within
supported domains. Grid settings are listed in [methods](docs/methods.md#numerical-settings).

For $k=K/F(T)$, the normalized terminal call payoff is $(x-k)^+$. Currency or
index-point prices are restored with $A(t,T)=D(T)F(T)/D(t)$:

$$
V(S,t)=A(t,T)U \left(\log\frac{S}{F(t)},t\right),\qquad
\Delta_{\mathrm{AH}}=\frac{A U_y}{S},\qquad
\Gamma_{\mathrm{AH}}=\frac{A(U_{yy}-U_y)}{S^2}.
$$

Spot derivatives hold the saved coefficient and deterministic carry fixed.
They do not recalibrate the surface. A new historical rebalance loads a newly
calibrated current-date model, which is a separate operation.

## Hedge rules

The observed-IV Black spot delta holds IV fixed and scales the forward with
spot at fixed $F/S$:

$$
\Delta_B=\frac{DF}{S}
\big[\Phi(d_1)-\mathbf 1_{\lbrace \mathrm{put}\rbrace }\big],
\qquad \mathcal V_B=DF\phi(d_1)\sqrt T.
$$

Vega $\mathcal V_B$ is per unit decimal annual volatility. Let
$\sigma_y=\partial\sigma_{\mathrm{surface}}/\partial\log(K/F)$.

| Method | Hedge convention |
| --- | --- |
| Black | Spot delta at observed midpoint IV |
| AH PDE | Spot derivative under the saved local-variance diffusion |
| Sticky strike | Black delta at fitted IV, with fitted IV fixed at strike |
| Sticky delta | $\Delta_B(\sigma_{\mathrm{surface}})-\mathcal V_B(\sigma_{\mathrm{surface}})\sigma_y/S$ |
| LV smile | $\Delta_B(\sigma_{\mathrm{obs}})+\mathcal V_B(\sigma_{\mathrm{obs}})\sigma_y/K$ |
| Empirical MV | Black delta plus a correction fitted from completed past outcomes |
| Unhedged | Zero index position |

Sticky delta uses an unadjusted normalized forward-call-delta smile, with
invertibility and derivative checks. The LV-smile rule is a smile approximation,
not the backward-PDE delta. These rules encode different assumptions about
smile movement, so their corrections are not interchangeable.

### Past-only empirical correction

The ideal one-period minimum-variance hedge motivates

$$
h^*=\frac{\mathrm{Cov}(\Delta C,\Delta S)}
{\mathrm{Var}(\Delta S)}.
$$

The implemented empirical rule uses the feature vector

$$
x_i=\frac{\mathcal V_{B,i}}{S_i\sqrt{T_i}}
\begin{bmatrix}1&\Delta_{B,i}&\Delta_{B,i}^2\end{bmatrix},
\qquad \Delta_{\mathrm{MV},i}=\Delta_{B,i}+x_i\widehat\beta.
$$

Ordinary least squares, without an intercept, fits
$\Delta C_i-\Delta_{B,i}\Delta S_i$ against $x_i\Delta S_i$. Fits are separate
by root and option kind, using up to 60 past entry dates. Every admitted outcome
must end strictly before the prediction timestamp. Warm-up, rank and
conditioning failures remain in coverage. The fitted objective uses raw price
changes; the funded multi-session hedge error is evaluated afterward.

## Fixed-contract hedge accounting

Current-date selection targets spot log-moneyness $\log(K/S)$ of −0.02, 0 and
+0.02, for calls and puts. Targets selecting the same contract are deduplicated.
Each path holds that contract for its declared reference-session horizon; a
missing mark leaves an incomplete path rather than extending the holding period.
Every rebalance loads the saved current-date decision.

The account is short one option. It starts with the option proceeds less the
hedge purchase and entry costs. Cash accrues between decisions, then funds hedge
trades and costs:

$$
B_{i+1}^{-}=B_i^{+}e^{r_f\Delta t_i},\qquad
B_{i+1}^{+}=B_{i+1}^{-}-(\delta_{i+1}-\delta_i)S_{i+1}
-\mathrm{cost}_{i+1}.
$$

At exit, the account liquidates the index hedge and closes the option. Midpoint
marks, option bid–ask and the additional 1 bp hedge-notional fee are separate
scenarios. Funding, turnover and option/index P&L components remain in the ledger.

![Historical path completion and exclusions](docs/figures/history_coverage.png)

Coverage for the five-session daily midpoint comparison retains all 44,500
entries per method. Pairwise comparisons additionally require a completed Black
path. Thirty dates lack AH surfaces because 31 expiry carry fits have too few
selected parity pairs; these failures precede optimization. Common-entry matching
aligns the retained samples but does not remove exclusion-related selection effects.

## Numerical validation

Analytical Black and square-root CEV controls check prices, Greeks, simulated
transitions and funded accounting. Market refinements vary space, time, domain,
grid position, blend radius and spot-bump width independently.

| Retained experiment | Measured result and scope |
| --- | --- |
| June 2015 Greek panel | 30 contracts × four blend radii × six PDE settings × three initial spots: 2,160 sampled states |
| Delta refinement | Maximum 24k-to-48k change of $3.36\times10^{-6}$ at the original spot and historical blend radius |
| Calibration vega | 18 contracts, seven calibrations and parallel quote-IV shocks of 1, 2 and 5 volatility basis points; maximum native/PDE relative difference 0.993% |
| Selected path replays | 72 scalar ledgers at six entry dates; component agreement within $1.49\times10^{-12}$ index points |

![Greek and short-end blend sensitivity](docs/figures/greek_blend_market.png)

Gamma is more sensitive to grid position and blending. Calibration-vega shocks
hold carry fixed and leave 31 noninvertible midpoint quotes unchanged; the check
covers one declared shock direction. The path replays are outcome-selected
explanations. These panels do not validate every historical Greek or independent
calibration pillar. [Validation](docs/validation.md) and
[saved-path diagnostics](docs/cache_checks.md) identify their inputs and limits.

## Run the project

Python 3.11 is the reference runtime. From the repository root, activate the
existing `volspx` environment or use an environment under the ignored sandbox:

```bash
python3.11 -m venv scratch/.venv
source scratch/.venv/bin/activate
PIP_CACHE_DIR="$PWD/scratch/cache/pip" python -m pip install -r requirements.txt
mkdir -p scratch/cache/{matplotlib,ipython}
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  IPYTHONDIR="$PWD/scratch/cache/ipython" \
  python -m unittest discover -s tests
```

The tests use synthetic fixtures and analytical references; private market data
is not required. The public clone also includes the report, figures and summary
tables. Daily browsing and full historical notebook controls require the retained
study and configured validation inputs, which are not distributed with the source.

```bash
PYTHONPATH="$PWD/src" MPLCONFIGDIR="$PWD/scratch/cache/matplotlib" \
  python -m jupyterlab
```

| Notebook | Purpose |
| --- | --- |
| [Surface explorer](notebooks/surface_explorer.ipynb) | Synchronized dates, surfaces, quote fits, carry, saved deltas and date-matched numerical evidence |
| [Historical analysis](notebooks/historical_analysis.ipynb) | Saved comparisons by schedule, cost scenario, sample, grouping and bootstrap block length |

Both read [config/notebooks.json](config/notebooks.json). Browsing loads saved
models and results; new calibration and PDE-risk experiments have separate
commands. Date ranges aggregate pairwise SSE and counts before calculating Gain,
with no range confidence intervals or inferred common-entry statistics.

[Reproduction instructions](docs/reproducibility.md) cover data locations,
report rendering and verification. The three large summary exports may be stored
as `.csv.gz`; the reader checks decompressed bytes against their original audit
hashes. [Storage maintenance](docs/storage_cleanup.md) records compression and
restoration without changing the frozen study.

## Implementation

The flat `src/` core owns the calculations; scripts and notebooks compose it.

| Responsibility | Source |
| --- | --- |
| Quotes, carry and selection | `market_data.py`, `carry.py`, `selection.py` |
| AH/SSVI calibration and surfaces | `calibration.py`, `surface.py` |
| Forward/backward PDEs and spot Greeks | `pde.py`, `pricing.py` |
| Hedge rules, paths and cash accounting | `hedging.py`, `paths.py`, `ledger.py` |
| Historical orchestration and inference | `study.py`, `inference.py` |
| Saved inputs and analysis interfaces | `artifacts.py`, `results.py`, `explorer.py`, `history_analysis.py` |
| Numerical experiments and publication | `simulation.py`, `validation.py`, `research_validation.py`, `publication.py` |

## References

1. Bruno Dupire (1994), *Pricing with a Smile*, Risk 7(1), 18–20.
   [Paper reprint](https://www.risk.net/sites/default/files/import_unmanaged/risk.net/data/Pay_per_view/risk/technical/1994/risk_0194_volatility.pdf).
2. Jesper Andreasen and Brian Huge (2011), *Volatility Interpolation*, Risk.
   [Publisher page](https://www.risk.net/derivatives/2029822/volatility-interpolation).
3. Jim Gatheral and Antoine Jacquier, *Arbitrage-free SVI volatility surfaces*.
   [Author manuscript](https://arxiv.org/abs/1204.0646).
4. John Hull and Alan White (2017), *Optimal Delta Hedging for Options*,
   Journal of Banking and Finance 82, 180–190 (first working version: 2015).
   [Author manuscript](https://www-2.rotman.utoronto.ca/~hull/downloadablepublications/Optimal%20Delta%20Hedging.pdf).

## Author

Kartik Vij
