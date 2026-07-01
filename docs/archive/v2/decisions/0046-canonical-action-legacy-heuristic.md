# ADR 0046: Canonical-Action Legacy Heuristic

Status: rejected at qualification on 2026-06-12.

## Context

ADR 0045 proved that root-action filtering is mechanically viable, then
rejected score-exact reuse of the historical evaluator. On compatibility seed
`32000`, canonical V2 scored a connected five-Elk layout as 14 while V1
returned 13.

The official rulebook requires connected Elk to use the interpretation that
produces the largest score:

<https://www.alderac.com/wp-content/uploads/2021/08/Cascadia-Rules.pdf>

The layout is a line of three plus a line of two, so the canonical result is
`9 + 5 = 14`. V1's Elk A implementation uses a longest-first greedy partition
and is only approximate. This invalidates V1 score parity, but it does not by
itself answer whether the historical NNUE and MCE remain useful as a heuristic
for choosing canonical V2 actions.

## Decision

Evaluate a separately named diagnostic policy:

1. translate the observable V2 public state and reconstructed unseen inventory
   exactly as in ADR 0045;
2. compare every translated board score;
3. permit only V1 Elk A undercount: habitat, Bear, Salmon, Hawk, Fox, Nature
   Tokens, and total-score accounting must otherwise match exactly, and V2 Elk
   must be greater than or equal to V1 Elk;
4. report the number, sum, and maximum of observed Elk undercounts;
5. discard every malformed root record through canonical V2 transition
   validation;
6. run the frozen diverse K32 NNUE prefilter and R600/LMR allocator on the
   surviving records;
7. revalidate the selected action and execute and score the game only in V2.

The treatment is explicitly a historical heuristic. Its internal value model
is not canonical and its weights are not eligible for production. A positive
result may authorize collection of canonical V2 action labels for a fresh
MLX-native model; it may not promote the historical network or binary.

## Frozen Compatibility Audit

- Trajectory: `pattern-aware-v1-k8-h6-b8-m4`.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Seeds: `32200-32203`.
- States: all 320 decisions.
- Local CPU only.

Every gate must pass:

- 100% structural public-state translation;
- repeated-translation and hidden-order invariance;
- all non-Elk score components match exactly;
- every score mismatch is a nonnegative V2-minus-V1 Elk difference;
- malformed source records are at most 10% of expanded records;
- at least one canonical candidate remains at every decision;
- every retained K32 candidate is canonical;
- no unregistered legacy-affecting environment variable is present.

## Frozen Strength Protocol

Baseline is promoted
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
Treatment is symmetric canonical-action K32/R600/LMR play with free-overflow
preparation and no paid wipes.

- Runtime smoke: seed `32299`.
- Qualification: seeds `32300-32302`, only after smoke.
- Exact 600 rollouts, sequential local CPU.
- Coordinate fallback at most 1%; every other bridge error aborts.

Smoke requires strict selected legality and at most 2,400 seconds per block.
Qualification authorizes a fresh MLX-native teacher experiment only if:

- treatment mean is at least 94.00;
- paired gain is at least +1.25;
- total wildlife delta is at least +0.25;
- habitat delta is at least -0.50;
- Nature Token delta is at least -1.00;
- fallback is at most 1%;
- runtime is at most 2,400 seconds per block.

Treatment mean below 92.75, gain below +0.50, malformed selected action, a
non-Elk score mismatch, V1 Elk overcount, or fallback above 5% rejects this
legacy-heuristic branch. No threshold, frontier, weight, rollout, allocator,
prelude, seed, fallback, or score-policy change is permitted.

## Commands

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- heuristic-compatibility \
  --games 4 --first-seed 32200 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-action-legacy-heuristic-v1-compatibility4.json
```

After a complete pass:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- heuristic-compare \
  --games 1 --first-seed 32299 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-action-legacy-heuristic-v1-r600-runtime-smoke-1.json
```

Only after the smoke:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- heuristic-compare \
  --games 3 --first-seed 32300 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-action-legacy-heuristic-v1-r600-qualification3.json
```

## Compatibility Result

The frozen `32200-32203` audit passed:

- all 320 states and 1,280 boards translated structurally;
- 12,876 expanded records were inspected;
- 248 malformed records were discarded, a 1.926% source-malformation rate;
- all 9,621 filtered K32 records were canonical;
- repeated translation and hidden-order redetermination were invariant;
- no score mismatch occurred on this seed suite.

Artifact:
`docs/v2/reports/canonical-action-legacy-heuristic-v1-compatibility4.json`,
BLAKE3
`74132c812b14e05e6b01733b8a6aeccb22c78f35fee5bb2eba600dd92936c80d`.

This authorizes the single registered R600 runtime smoke at seed `32299`.

## Runtime Smoke

Seed `32299` passed every smoke gate:

- treatment 98.25 versus strong at 91.00, a +7.25 paired gain;
- total wildlife +2.00, habitat +5.00, Nature Tokens +0.25;
- all 80 selected actions canonical, zero fallbacks;
- 100/3,272 malformed source records filtered before K32;
- zero score mismatches;
- 311.145 treatment seconds for the complete game block;
- 3.889-second mean and 6.774-second P90 decision latency.

Artifact:
`docs/v2/reports/canonical-action-legacy-heuristic-v1-r600-runtime-smoke-1.json`,
BLAKE3
`34816f5944795a70601bf84960e0b0cd051c120be0c79aeeb110c84a3ea751af`.

This authorizes the frozen qualification seeds `32300-32302`.

## Qualification Result

The three-game qualification was strong but failed one frozen guardrail:

- treatment 96.25 versus strong at 92.333, a +3.917 paired gain;
- all three seed blocks won, with deltas +4.25, +7.25, and +0.25;
- total wildlife +1.333 and habitat +3.667;
- Nature Tokens -1.083, missing the registered -1.00 floor by 0.083;
- all 240 selected actions canonical, zero fallbacks, zero score mismatches;
- 203/9,265 malformed source records filtered before K32;
- 157.531 treatment seconds per game.

Artifact:
`docs/v2/reports/canonical-action-legacy-heuristic-v1-r600-qualification3.json`,
BLAKE3
`472a999fc3e22880a1dd6704d78f1b4e9227e6ad642d62fd4a6831bcd3e82652`.

ADR 0046 is rejected and does not authorize MLX collection. The result is
nevertheless mechanistically important: total score already includes the lost
token points, so the policy converted approximately one token point into
roughly five non-token board points. A separately preregistered successor may
test whether that resource conversion remains positive and statistically
reliable on untouched seeds; this result cannot be retroactively re-gated.
