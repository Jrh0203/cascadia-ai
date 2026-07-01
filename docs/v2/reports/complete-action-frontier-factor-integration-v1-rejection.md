# ADR 0097 Candidate-Factor Integration Result

Classification: `candidate_factor_inputs_insufficient`

| Probe | Train recall | Train exact | Validation recall | Validation exact |
|---|---:|---:|---:|---:|
| wide-concat | 0.300011 | 0.000000 | 0.250735 | 0.000000 |
| screen-relative | 0.308786 | 0.001786 | 0.254534 | 0.000000 |
| factor-attention | 0.293931 | 0.001786 | 0.249877 | 0.000000 |
| pairwise-gated | 0.306619 | 0.001786 | 0.246814 | 0.000000 |

All four origin reports passed coverage, finite-score, memory, swap, source-identity, and ring-replay integrity checks.

Probe plus replay wall time was 3944.72 seconds, resolving 3.65 independent hypotheses per hour.

The invalid allocator-default launch contributed no selection or classification data. The sealed test split, gameplay, cloud, and external compute remained closed.
