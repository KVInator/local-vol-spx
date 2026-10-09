# Market Data and Preparation

The project uses monthly SPX end-of-day option files for 2013–2023. Each raw row
contains a strike and expiry, call and put bid-ask quotes, an index mark and
quote timestamps. The preparation stage converts these files into a dated
option panel for calibration and historical hedging.

## Dataset Coverage

| Metric | Value |
| --- | ---: |
| Study period | 2013-01-01 to 2023-12-31 |
| Monthly files | 132 |
| Raw rows | 14,750,241 |
| Reference trading sessions | 2,768 |
| Reference sessions with observations | 2,744 |
| Missing reference observations | 24 |
| Observed early-close dates requiring clock review | 21 |

Raw rows are not the number of selected hedge trades. Call and put observations
are separated during preparation, then filtered and selected for different
parts of the experiment.

## Quote and Timestamp Checks

Headers and dates are standardized before prices are used. The audit compares
the Unix quote timestamp with the vendor's readable timestamp in New York time.
It also checks the stated quote date, positive index marks, strike and expiry
fields, duplicate contract keys and consistency of the daily index mark.

Bid-ask checks distinguish crossed or invalid quotes, locked quotes and zero
bids. These conditions are recorded explicitly. Entry selection requires usable
two-sided quotes; eligible zero-bid observations can be retained separately for
endpoint bounds.

The standard observation clock is 16:00 New York. Early-close dates need
separate review because that clock does not describe every reference session.
The session policy carries those distinctions into the historical study.

## Contract Identity and Missing Sessions

The files do not establish a verified SPX/SPXW root for each contract. The
prepared data therefore uses `UNKNOWN`. A contract is tracked by root, expiry,
strike and option kind, with settlement treated under the assumption recorded
in [Assumptions and Limitations](assumptions.md).

Holding periods follow the reference trading calendar. A missing observation
is not replaced by the next available date or a carried-forward option mark.
Affected hedge paths retain an incomplete status and a reason in coverage.

## Calibration Quotes

Put-call pairs near the money are used to estimate an expiry forward under the
chosen discount factor. The preparation records parity-band inconsistencies
and whether the fitted forward lies outside those bands.

An out-of-the-money quote is selected at each calibration strike and converted
to an equivalent call using parity. Its original bid-ask spread is retained for
calibration weighting and residual checks. The calibration and hedge-entry
panels serve different purposes and need not contain the same contracts.

## Hedge-Entry Selection

The default entry basket targets 21, 35 and 45 calendar days to expiry and
log spot-moneyness values of -0.02, 0 and +0.02, for both calls and puts.
Eligible contracts lie within 14–45 days and the selection window. The nearest
available expiry and strike are chosen using current-date information, with
deterministic tie-breaking.

Different target buckets can select the same contract. Those selections are
deduplicated by entry identity rather than counted as additional trades.
Once entered, the historical path follows that fixed contract.

## Saved Inputs

The study records the configuration, source and input hashes when it is frozen.
Monthly preparation and daily decisions are cached separately. This allows a
compatible run to resume from completed work and lets the notebook inspect
saved outputs.

The [methodology](methods.md) describes how these inputs become hedge decisions;
[validation](validation.md) describes the checks on those calculations.
