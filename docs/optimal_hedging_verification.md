# Verification of the Step 3 implementation

The verification below was executed locally on 8 October 2026. It is not a claim that the user's full October strategy evaluation has run.

| Check | Measured result | Scope |
| --- | --- | --- |
| New unit tests | 33 passed in 3.642 seconds | Known smile derivatives, implicit sticky-delta coordinate, vega units, call/put parity, recoverable quadratic coefficients, rolling windows, strict past-only endpoints, future perturbations, rank/warm-up failures, paired Gain, ledger and checksum controls. |
| Existing hedge-panel/comparison regression controls | 47 passed in 2.437 seconds | Prior panel, comparison and attribution interfaces used by the new stage. |
| Existing ledger controls | 18 passed in 0.215 seconds | Self-financing accounting integration. |
| Complete saved-control stage | 12 retained entries; six matched endpoints and six sample-end entries | Deterministic pre-existing pipeline fixture; not historical performance evidence. All six strategies have coverage; empirical warm-up remains explicit because two dates cannot meet the declared minimum. |
| Control accounting | Maximum reconciliation 1.6570e−14 index points | Ledger versus independently computed funded cash value. |
| Original September-model smile compatibility | 20 models, 360 generated selected-grid contracts: 354 ready, six stencil disagreements | Original saved models; Black benchmark set to model-price IV for this compatibility check. This is not the actual observed-IV historical strategy comparison. |
| Largest compatibility stencil change | 0.000745906 in delta; largest chain/direct-bump disagreement 0.000193017 | The six failures stay failed under the fixed 0.0005 delta tolerance; the tolerance was not relaxed. |
| Compatibility runtime | 3.370 seconds | Native model and smile evaluation only, no backward diffusion repricing. |
| Notebook and plots | Code cells compile; three generated figures inspected | IPython was absent in this local environment, so an actual notebook-kernel execution is not claimed. The user's environment includes IPython/ipykernel. |

The coefficients in the synthetic recovery tests are known by construction. Their recovery establishes implementation correctness, not that a historical empirical hedge beats Black. The controlled stage's Gain values are likewise implementation checks, not project headline results.

Run the repository's full test discovery after copying the update, then execute the pinned September/October command in `optimal_hedging.md`. Only those outputs can close the historical evidence part of Step 3.
