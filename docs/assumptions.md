# Assumptions and Limitations

The historical experiment compares hedge rules under a common set of contract,
carry and execution conventions. These choices determine the P&L being measured
and the interpretation available from it.

## Contract and Observation Conventions

| Item | Treatment |
| --- | --- |
| Option observations | Vendor end-of-day bid-ask quotes |
| Contract root | `UNKNOWN` until SPX/SPXW identity is verified |
| Fixing | Assumed PM fixing at 16:00 New York |
| Time to expiry | Elapsed time on an ACT/365F basis |
| Missing observations | Affected paths remain incomplete with a recorded reason |
| Early closes | Flagged dates require separate clock review |

The available files do not establish the official settlement convention for
every contract. The PM fixing is a research assumption, and the provisional
contract key cannot resolve identities that the source data does not provide.
The [data documentation](data.md) describes the coverage and preparation checks.

## Pricing Carry and Hedge Funding

| Item | Default choice |
| --- | --- |
| Pricing rate | Constant 5%; earlier pilots also examined 3% and 7% |
| Expiry forward | Put-call parity estimate under the assumed discount factor |
| Hedge instrument | Fractional synthetic index position at the observed mark |
| Dividend cash | Zero in the hedge account |
| Cash funding | Constant 5% for lending and borrowing |
| Hedge fee | 1 bp of traded hedge notional in the fee scenario |

Parity-implied carry is used in pricing. It is separate from the dividend cash
credited to the hedge account. The synthetic index position also has no
observed executable spread or futures basis.

The resulting P&L measures hedge error under these conventions. It does not
establish the performance of an executable index-trading strategy. Alternative
carry or funding choices would define a different experiment.

## Calibration and Diffusion

Daily AH fits use quote-spread weighting and regularization. The recovered
local volatility is separate from the calibration proxy parameters, and its
reported values are not capped.

A Gaussian variance blend is applied near the short end of the hedge diffusion.
That diffusion differs from the native AH price surface. Its pricing and Greek
sensitivity therefore needs to be assessed alongside quote-fit quality.

The independent quote-total-variance construction is unconstrained and retains
calendar or density failures. It remains a diagnostic comparison. The current
smile corrections also use Black-formula vega rather than calibration-pillar
bumps and PDE repricing.

## Numerical and Statistical Scope

Grid refinements cover selected states and contracts. They do not independently
verify every Greek in the historical study. The controlled Black–Scholes and
CEV simulations isolate numerical and accounting behavior, but do not remove
market-model mismatch from the SPX experiment.

Gain compares each candidate with Black on matched entries. Different methods
can have different coverage, so pairwise rankings need to be read alongside
common-entry results. Date-block resampling addresses dependence; the available
number of dates and block-length sensitivity still limit interval precision.

## Retrospective Evaluation

September and October 2023 were used during development. The broader history
is therefore a retrospective evaluation, even though rolling empirical fits
use only outcomes available before each prediction.

The completed pilots provide development evidence. The full-history run is
needed to assess whether those findings persist across a wider sample, subject
to the same conventions and coverage limitations.
