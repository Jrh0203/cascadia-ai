# ADR 0071: Common-Random-Number Sequential Halving

Status: rejected on confirmation on 2026-06-13.

## Context

ADR 0070 closed residual learning on the fixed historical representation.
The follow-up identifiability audit found that only 18.359% of fresh
validation winners clear a 95% normal difference test, just 6.953% have
non-overlapping 95% intervals, and the mean 95% confidence set contains
10.140 actions. In opening turns, only 6.563% are distinguishable and the set
contains 16.309 actions.

The qualified K32/R600 teacher assigns independent random rollout seeds to
every candidate. Sequential halving is a fixed-budget best-arm method, but
independent streams leave the comparison variance equal to the sum of
candidate variances. Common random numbers can reduce variance of differences
when alternative outcomes are positively correlated under the shared random
input. Cascadia candidates begin from closely related afterstates, so this is
a plausible but unproven mechanism. CRN can fail when coupling is weak or
negative; no literature claim substitutes for local evidence.

## Decision

1. Add a typed experimental seed-coupling mode to the exact-MLX sequential
   halving implementation. The qualified independent mode remains unchanged.
2. In CRN mode, generate one ordered seed vector per halving round and give
   every alive candidate the prefix required by its frozen LMR allocation.
   Candidates with equal allocations receive identical seed sets.
3. Keep candidate generation, K32 prefilter, R600 total budget, LMR
   multipliers, exact MLX evaluator, rollout policy, opponent policy, integer
   scoring, elimination by accumulated mean, bridge, and tie ordering fixed.
4. Use independent warmed MLX services for baseline and treatment and run them
   sequentially to avoid resource contention.
5. Report exact requested and consumed rollout rows, candidate counts,
   selections, categories, runtime, latency, bridge diagnostics, service
   diagnostics, deterministic replay, and clean shutdown.
6. Do not train from CRN labels unless the search policy first passes gameplay.

## Implementation Gates

Before gameplay:

- unit tests prove independent mode retains its original seed schedule shape;
- unit tests prove equal-allocation CRN candidates receive identical ordered
  seeds and unequal allocations receive a common prefix;
- both modes consume the exact frozen per-round allocation;
- native and exact-MLX independent parity remains bit-exact;
- CRN is deterministic under repeated public-state search;
- all focused Rust tests and strict focused Clippy pass.

An implementation-only R32 trajectory may verify wiring and accounting. It is
not strength evidence and may not change the frozen protocol.

## Frozen Gameplay Protocol

- Rules: symmetric four-player AAAAA, base score without habitat bonuses.
- Baseline: exact-MLX K32/R600/LMR sequential halving with independent
  per-candidate rollout seeds.
- Treatment: identical search with within-round common random numbers.
- Model: immutable
  `artifacts/models/legacy-nnue-v4opp-mlx-v1`, manifest BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.
- Runtime smoke: seed 35,699.
- Pilot: seeds 35,700-35,702.
- Conditional confirmation: unauthorized until the pilot passes and a
  separate ADR freezes at least 20 fresh paired games.

The smoke passes only if both arms:

- complete 80 legal actions with zero bridge or neural fallback;
- consume the exact configured rollout allocation;
- produce finite fixed-width exact-MLX responses;
- complete in at most 220 seconds per game;
- shut down cleanly.

The three-game pilot advances only if:

- treatment-minus-baseline paired mean is at least +0.50;
- treatment absolute mean is at least 96.0;
- total wildlife and habitat deltas are each at least -0.50;
- Nature Token delta is at least -1.0;
- all smoke integrity and runtime gates remain true.

Three games cannot promote a policy. Passing authorizes only a separately
preregistered confirmation. Failure closes same-budget CRN as the next lever;
it does not authorize changing budget, frontier, confidence rule, or
elimination policy on the same seeds.

## Maximum Compute

One implementation-only R32 qualification, one seed-35,699 paired gameplay
smoke, and, only after a passing smoke, three paired games on
35,700-35,702. Local CPU and MLX GPU only. No training, architecture change,
budget increase, seed retry, threshold change, parameter sweep, test split, or
external compute.

## Result

All implementation gates passed. The independent path remained bit-exact
against native Rust over 80 R32 decisions and three R600 spots, including
candidate identity, selected actions, sample counts, and rollout means. CRN
replayed deterministically, both live services shut down cleanly, and the R32
qualification completed 160 legal actions without fallback.

The R600 smoke on seed 35,699 passed at 95.00 independent versus 96.25 CRN,
or +1.25. The preregistered three-game pilot then scored:

- independent mean: 95.9167;
- CRN mean: 97.0833;
- paired gain: +1.1667, 95% CI `[+0.5778,+1.7556]`;
- game record: 3-0-0;
- wildlife: +0.5833;
- habitat: -0.0833;
- Nature Tokens: +0.6667;
- runtime: 155.26 versus 155.69 seconds per game.

All 480 pilot actions were legal, both arms used zero bridge or policy
fallbacks, and both services shut down cleanly. Every frozen pilot gate passed.
Three games could not promote the policy. ADR 0072 froze the required 20-game
confirmation before any new seed was opened.

ADR 0072 rejected the mechanism: independent scored 95.775 versus 95.413 for
CRN, a paired -0.363 with 95% CI `[-1.129,+0.404]` and an 8-1-11 record.
Runtime, legality, rollout accounting, fallback, category, and shutdown gates
all passed. The pilot was a small-sample false positive. Same-budget CRN is
closed.

## References

- Z. Karnin, T. Koren, and O. Somekh, "Almost Optimal Exploration in
  Multi-Armed Bandits," ICML 2013:
  https://proceedings.mlr.press/v28/karnin13.html
- P. Glasserman and D. Yao, "Some Guidelines and Guarantees for Common Random
  Numbers," Management Science 38(6), 1992:
  https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf
- J. Veness et al., "Variance Reduction in Monte-Carlo Tree Search," NeurIPS
  2011: https://webdocs.cs.ualberta.ca/~bowling/papers/11nips-vrmcts.pdf
