# Oracle-Proposal Complete-Action Selector V1 Result

Date: 2026-06-16

Classification: **`oracle_proposal_selector_representation_insufficient`**

Selected architecture: `None`

Campaign pipeline passed: `True`

| Arm | Train recall | Train exact | Validation recall | Validation winner | Passed |
|---|---:|---:|---:|---:|---:|
| `wide-concat` | 39.71% | 0.18% | 34.93% | 78.33% | False |
| `screen-relative` | 41.03% | 0.36% | 34.55% | 78.75% | False |
| `factor-attention` | 40.47% | 0.36% | 34.65% | 79.17% | False |
| `pairwise-gated` | 41.90% | 0.36% | 33.71% | 75.83% | False |

This diagnostic used only the open oracle-factor proposal. It did not alter
ADR 0120, promote a selector, or open sealed gameplay.
