# Numerical validation

Analytical Black and CEV controls test prices, payoff parity, Greeks and ledger
accounting. AH tests cover calibration invariants, price-shape checks, carry
consistency and the recovered coefficient. Quote interpolation retains
unsupported regions, negative calendar/density diagnostics and unstable
derivatives as gaps. The numerical model was unchanged during finalization.

## Retained market checks

The June 19, 2015 panel covers 30 contracts, four blend radii, six PDE settings
and three initial spots: 2,160 states. No evaluation failures or sampled
price/delta-bound or negative-gamma flags were recorded. At the original spot
and historical blend radius, the largest 24k-to-48k delta change is
3.36e-06. Gamma is more sensitive to grid position, blend radius and
spot-bump width.

![Greek blend sensitivity](figures/greek_blend_market.png)

The same-date calibration-vega panel covers 18 contracts and seven calibrations.
Symmetric 1, 2 and 5 volatility-basis-point shocks affect 566 of 597
quotes. The 31 noninvertible midpoints remain unchanged and participate
in calibration. The largest native/PDE relative vega difference is
0.993% for this parallel quote-IV shock direction, with carry
held fixed. This is a directional sensitivity, not a validation of every
individual calibration pillar.

![Calibration-vega comparison](figures/calibration_vega_market.png)

Selected calls at March 16, 2020 and November 9, 2023 were repriced on production,
finer and shifted grids. Production deltas reproduce saved values within
7.91e-13. These are entry-date numerical checks. The independent
scalar-ledger replay in [cache diagnostics](cache_checks.md) checks subsequent
decisions, observed marks, funding and costs without new PDE evaluation.

Seven-date smile replays cover 928 decisions, with zero recorded status mismatches. Empirical-fit membership
checks verify that training endpoints precede prediction timestamps. Their
producer audits and output hashes remain attached to
`scratch/runs/market_validation`; the explorer loads only date-matched rows.
The June saved constrained SSVI comparison is retained as a separate quote-fit
diagnostic, with its original producer pins.

## Verification scope

The local baseline passed 280 tests before integration. The integrated suite passed
290 tests in 59.050 seconds in `volspx`. Final verification
includes the retained numerical suite, date switching, missing/carry-failure
states, cache tampering, relocated cache reads, scalar-path replay, SSE/count
aggregation and the serial/parallel blend workflow. Both deliverable notebooks
are executed in actual `volspx` kernels, with rendered figures and widget changes
checked across June 2015, March 2020, September 2023 and a carry-failure date.
Execution and inspection evidence is retained under `scratch/notebook_checks`.

Broader gamma validation, every historical Greek, more flexible SSVI fitting
and deployable hedge execution are outside the completed evidence. The full
history was neither rerun nor revised. [Assumptions](assumptions.md) records
the financial and retrospective scope; [reproducibility](reproducibility.md)
lists commands and retained inputs.
