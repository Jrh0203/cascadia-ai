# ADR 0098 Raw Factor Construction Result

Classification: `raw_factor_construction_insufficient`

Selected construction: `none`

| Probe | Train recall | Train exact | Validation recall | Validation exact |
|---|---:|---:|---:|---:|
| complete-raw-flat | 0.302865 | 0.000000 | 0.251838 | 0.000000 |
| exact-local-relation | 0.378674 | 0.000000 | 0.212623 | 0.000000 |
| explicit-market-transition | 0.299376 | 0.000000 | 0.243505 | 0.000000 |
| fresh-entity-cross | 0.179478 | 0.000000 | 0.155637 | 0.000000 |

All four origin reports passed exact coverage, finite-score, memory, swap, source-identity, maximum-width, and ring-replay integrity checks.

Every arm missed the 80% train-recall and 25% exact-train-set gates. Another neural constructor, head, pool, width increase, or optimizer variation is therefore closed; the next experiment must audit target learnability and supervision structure.

Probe plus replay wall time was 2412.43 seconds, resolving 5.97 independent hypotheses per hour.

No derived feature cache was created. The sealed test split, gameplay, cloud, and external compute remained closed.
