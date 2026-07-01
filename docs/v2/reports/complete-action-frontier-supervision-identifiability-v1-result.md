# ADR 0099 Frontier Supervision Identifiability Result

Classification: `uncertainty_aware_supervision_sufficient`

| Audit | Train gate | Validation gate |
|---|---:|---:|
| boundary signal | False | False |
| cross fidelity | False | False |
| teacher resampling | False | False |
| expected-rank ceiling | True | True |

Validation mechanism metrics:

- boundary-separated target slots: 10.38%; complete sets: 0.00%;
- R600 cohort coverage at width 64: 0.00%;
- 512-draw hard-target recall: 41.20%; exact-set reproduction: 2.50%;
- expected-rank nominal-target recall: 92.11%; exact target sets: 27.08%;
- expected-rank R4800 winner recall: 100.00%; confidence coverage: 100.00%; retained regret: 0.000000.

The finite-R1200 hard cutoff is statistically unstable, while uncertainty-aware expected-rank ordering preserves the complete open-validation R4800 decision signal. This authorizes one separately preregistered MLX pilot using ordinal expected-rank supervision.

All four origin reports covered every open group and candidate, used zero process swap, and matched their ring replays bit-for-bit.

Origin plus replay wall time was 21.79 seconds, resolving 660.90 independent hypotheses per hour.

The sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.
