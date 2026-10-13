# Historical hedge results

The authoritative retrospective study covers 2013–2023. It selected 44,500
entries on 2,768 reference dates and completed 3,879,048
strategy/schedule/scenario outcomes, with 0 pending paths. Outcomes reuse
contracts and entry dates. The maximum recorded accounting reconciliation is
4.6e-12 index points. The original freeze and producer pins
identify the result scope; [reproducibility](reproducibility.md) gives their location.

## Five-session holdings

The comparison below uses daily rebalancing and observed midpoint marks. Each
pairwise sample matches a candidate with Black. The common-entry intersection
has 33,763 entries across 2,406 entry dates for every method.
Intervals use the saved entry-date bootstrap with a mean block of 20 sessions
and 500 replicates; the notebook also exposes 10- and 40-session blocks.

| Method | Paired entries | Pairwise Gain | Common-entry Gain | Common-entry 95% interval |
| --- | ---: | ---: | ---: | ---: |
| AH PDE | 39,252 | -25.58% | -30.16% | [-55.50%, -7.57%] |
| Empirical MV | 40,230 | -8.40% | -7.92% | [-25.78%, +9.08%] |
| LV smile | 33,887 | -12.38% | -12.40% | [-30.03%, +4.53%] |
| Sticky delta | 33,887 | -136.71% | -136.74% | [-188.49%, -91.90%] |
| Sticky strike | 33,887 | +0.15% | +0.15% | [-0.04%, +0.37%] |

Gain is one minus candidate squared funded error divided by matched Black
squared funded error, measured around zero. AH's common-entry Gain therefore
means 30.16% more squared error than Black. Empirical MV and LV smile have
intervals crossing zero, sticky delta is substantially worse, and sticky
strike's small positive estimate also has an interval crossing zero. These
findings do not support a stable AH hedge advantage under the [study conventions](assumptions.md).
The earlier general 8–12% variance-reduction claim is unsupported.

## Coverage

| Method | Completed paths | Incomplete paths |
| --- | ---: | ---: |
| Black | 40,362 | 4,138 |
| AH PDE | 39,316 | 5,184 |
| Empirical MV | 40,230 | 4,270 |
| LV smile | 33,887 | 10,613 |
| Sticky delta | 33,950 | 10,550 |
| Sticky strike | 33,950 | 10,550 |

Every variant retains the selected-entry population in coverage.
Completed counts describe each method's ready paths; pairwise counts also
require a completed Black path on the same entry. Recorded
incomplete-path reasons for this schedule include `session_excluded`, `unresolved`, `stencil_disagreement`, `model_unavailable`, `quote_unavailable_in_research_scope`, `sample_end`, `warmup`, `black_not_ready`, `no_finite_positive_iv`, `carry_unavailable`, `outside_observed_stencil_support`.
The notebook shows counts by method and reason, saved annual/maturity/moneyness
groupings and the largest paired entry-date SSE contributions. Missing marks
are never replaced with model prices or carried to a later date.

Date-range views sum saved squared errors and counts before computing Gain.
They describe the pairwise sample and carry no confidence interval. Common-entry
daily sufficient statistics were not saved, so arbitrary-range common-entry
results are unavailable. Full-history common-entry intervals remain attached
to their recorded scope.

Thirty dates lack AH surfaces because one or more expiry carry fits were
unavailable. [Cache diagnostics](cache_checks.md) identifies all 31 affected
expiry records and replays selected large-error paths. September/October
development displays and old November prefixes are archived; removing them
does not turn this retrospective study into a held-out experiment.
