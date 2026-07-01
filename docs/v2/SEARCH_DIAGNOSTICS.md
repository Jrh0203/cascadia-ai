# Search Diagnostics

The first promoted `determinized-lookahead-v2-k4-r4-d4` policy filtered legal
actions by exact immediate score. A controlled K4/K8 study measured whether
that filter discarded actions that the same four-sample, four-ply evaluator
preferred.

## One-Axis Pilot

All variants used seeds 20500-20509 and changed one parameter from K4/R4/D4.

| Variant | Mean | Paired delta | 95% CI | Record | Runtime |
|---|---:|---:|---:|---:|---:|
| K4/R4/D4 | 89.350 | baseline | - | - | - |
| K8/R4/D4 | 90.425 | +1.075 | [-0.939, 3.089] | 7-0-3 | 64.117 s |
| K4/R8/D4 | 89.600 | +0.250 | [-0.912, 1.412] | 6-0-4 | 58.372 s |
| K4/R4/D8 | 90.250 | +0.900 | [-0.613, 2.413] | 5-1-4 | 62.229 s |

The pilots are exploratory and individually inconclusive. K8 was selected for
diagnosis because it had the largest mean gain, the strongest win record, and
positive Elk and Salmon movement at comparable runtime.

## Candidate Recall

Five disjoint promoted-policy games, seeds 20600-20604, produced 400 decisions.
At every state, K8 used the same determinization schedule and evaluator as K4.

- selection coverage at K4: **83.25%**
- strict value recall at K4: **83.25%**
- strictly higher-valued action excluded: **67 of 400 decisions**
- mean estimated regret over all decisions: **0.076**
- mean estimated regret when K4 missed: **0.455**
- maximum estimated regret: **2.000**

| Phase | Decisions | Strict misses | Value recall | Mean regret |
|---|---:|---:|---:|---:|
| Early | 135 | 28 | 79.26% | 0.083 |
| Middle | 135 | 21 | 84.44% | 0.076 |
| Late | 130 | 18 | 86.15% | 0.069 |

The immediate rank selected by K8 was distributed as:
`[260, 38, 20, 15, 16, 13, 24, 14]` for ranks 1 through 8.

The result identified a real score-loss source: immediate-score pruning was too
aggressive, especially early in the game.

## Confirmation

On disjoint seeds 20700-20749, K8 scored 90.270 versus K4 at 89.305:

- paired delta: **+0.965**
- 95% CI: **[+0.418, +1.512]**
- record: 31-4-15

K8 cleared the pre-registered gate and replaced K4 as the promoted interactive
policy.

## Residual K8 Recall

A second diagnostic followed K8 on seeds 20800-20802 and evaluated K16 at all
240 decisions:

- value recall at K8: **89.17%**
- strictly higher-valued K16 action excluded: **26 decisions**
- mean estimated regret when missed: **0.510**
- early / middle / late recall: **85.19% / 90.12% / 92.31%**

K8 removes most of the original deficit but does not saturate breadth,
especially early in the game. K16 therefore advances to a registered paired
pilot.

The K16 pilot on seeds 20900-20909 scored 91.425 against K8 at 91.000, a
+0.425 paired delta with a 5-5 record and 95% CI [-1.079, 1.929]. It passed the
pre-registered advancement threshold but remains inconclusive, so it proceeds
to a disjoint 50-game confirmation without replacing K8.

The disjoint confirmation scored 91.555 for K16 versus 90.810 for K8:

- paired delta: +0.745
- 95% CI: [0.187, 1.303]
- record: 33-1-16

The effect is positive, but the lower confidence bound missed the
pre-registered +0.25 promotion gate. K16 is retained as a stronger research
teacher.

## Pattern-Aware Supersession

A later rules-derived policy evaluated immediate K8, habitat H6, and Bear B8
in one shared pass, adding the expected best exact one-token marginal from a
four-token future market. On a direct disjoint 50-game control it scored
91.890 versus K8 at 90.775. The K8-minus-pattern paired delta was -1.115 with
95% CI -1.696 to -0.534, and pattern-aware was 14.49x faster.

Pattern-aware therefore replaced K8 as the interactive product policy. K8 and
K16 remain research controls for fair hidden-state search.

## Pattern-Aware Rollout Policy

Using the stronger standalone policy inside H6 was tested as a separately
registered rollout intervention. Root K8+H6 candidates, R4 determinizations,
D4 horizon, and exact leaf scoring were fixed; only future actions changed
from greedy to pattern-aware.

The runtime smoke passed, but the ten-game pilot scored 90.525 against H6 at
91.075: -0.550 paired, 95% CI -1.796 to 0.696, and a 4-0-6 record. Aggregate
habitat was -0.050 and wildlife -0.400. The four-ply leaf gained Hawk but lost
Bear and Fox, indicating that pattern-aware setup is not consistently realized
inside this shallow exact-score horizon. No confirmation was run.
