# Perfect-Information Focal Beam V1

Status: multi-turn continuation mechanism confirmed on 2026-06-11;
diagnostic only and not promoted.

## Question

Does jointly optimizing the focal player's final five decisions materially
improve the exact W2 one-step oracle?

## Protocol

Each of seeds 29600-29609 formed a four-seat block against three frozen
pattern-aware opponents. The baseline used exact-hidden-state W2 one-step
policy improvement. The treatment matched that policy before its final five
personal turns, then searched the same W2 frontier with an independent
width-16 beam per root action.

Beam branches preserved true hidden state, cloned common continuation
randomness, advanced opponents with pattern-aware play, pruned by the frozen
pattern heuristic, and selected by exact focal terminal base score.

## Result

| Metric | Exact W2 one-step | Exact W2 beam | Delta |
|---|---:|---:|---:|
| Mean base score | 92.900 | 93.650 | +0.750 |
| Habitat | 28.675 | 28.925 | +0.250 |
| Wildlife | 60.650 | 60.825 | +0.175 |
| Nature Tokens | 3.575 | 3.900 | +0.325 |

Paired 95% CI: `[+0.400,+1.100]`; record 9-1-0. Treatment P90 decision
latency was 3,252 ms and its four-rotation block averaged 89.306 seconds.

Wildlife deltas were Bear +0.400, Elk +0.675, Salmon -0.775, Hawk +0.325, and
Fox -0.450.

## Conclusion

The +0.750 gain exceeds the preregistered +0.50 threshold with a strictly
positive interval, confirming multi-turn continuation as an independent
lever. The 93.650 mean remains below the 97 diagnostic boundary and 6.350
below target. The result does not promote hidden-information play; it directs
the next public-state MLX work toward multi-turn counterfactual continuation
targets.

Artifacts:

- `docs/archive/v2/reports/perfect-information-focal-beam-v1-t5-b16-w2-runtime-smoke-1.json`
- `docs/archive/v2/reports/perfect-information-focal-beam-v1-t5-b16-w2-pilot10.json`
