# Perfect-Information Diverse Frontier V1

Status: structural candidate-recall hypothesis confirmed on 2026-06-11;
diagnostic only and not promoted.

## Question

Does the base K8+H6+B8 frontier omit useful species-preserving actions when
candidate values are measured with exact hidden-state full-game continuation?

## Protocol

Each of seeds 29100-29109 formed a paired four-seat block. Both policies
rotated one focal perfect-information oracle through all four seats against
three frozen pattern-aware opponents. The baseline used the exact K8+H6+B8+M4
frontier. The treatment changed only root candidate recall by adding up to two
distinct candidates per wildlife species.

Every candidate preserved the true hidden stack and bag and used identical
deterministic pattern-aware continuation. Hidden information was diagnostic
only and is not available to product play.

## Result

| Metric | Exact base frontier | Exact W2 frontier | Delta |
|---|---:|---:|---:|
| Mean base score | 92.625 | 93.975 | +1.350 |
| Habitat | 29.325 | 28.825 | -0.500 |
| Wildlife | 60.050 | 61.300 | +1.250 |
| Nature Tokens | 3.250 | 3.850 | +0.600 |

Paired 95% CI: `[+0.704,+1.996]`; record 9-0-1 over ten seed blocks. The
treatment P90 decision latency was 337.532 ms and its four-rotation block took
15.449 seconds per seed.

Wildlife deltas were Bear -0.050, Elk -0.450, Salmon +0.300, Hawk -0.325, and
Fox +1.775.

## Conclusion

The +1.350 gain exceeds the preregistered +0.50 materiality threshold with a
strictly positive interval, confirming that the base frontier omits useful
structural actions. The 93.975 treatment mean remains below the 97 diagnostic
boundary and 6.025 below target, so candidate breadth does not solve the
continuation ceiling.

Because Fox supplied the gain while the other four species summed to -0.525,
the next public-information experiment should isolate Fox candidate recall
under the promoted confidence-gated terminal operator.

Artifacts:

- `docs/archive/v2/reports/perfect-information-diverse-frontier-v1-runtime-smoke-1.json`
- `docs/archive/v2/reports/perfect-information-diverse-frontier-v1-pilot10.json`
