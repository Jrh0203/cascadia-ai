# ADR 0047: Productive Nature Token Confirmation

Status: qualified on 2026-06-12. MLX action-imitation work is authorized.

## Context

ADR 0046 rejected the canonical-action legacy heuristic because Nature Tokens
fell 1.083 against a frozen -1.00 floor. Every other qualification gate passed:
the treatment scored 96.25, gained 3.917 total, gained 1.333 wildlife and
3.667 habitat, won all three blocks, and executed all 240 actions canonically.

Nature Tokens are already worth one point each in the canonical total. The
observed +3.917 therefore remains after paying the full score cost of the
additional token expenditure. The implied non-token board gain was
`3.917 - (-1.083) = 5.000`, or approximately 4.62 habitat/wildlife points per
additional token point spent.

The prior guardrail correctly rejected ADR 0046 and will not be changed. The
new question is whether this resource conversion is positive, efficient, and
statistically reliable on a larger untouched suite.

## Decision

Run the exact unchanged ADR 0046 treatment:

- canonical V2 public-state translation;
- only measured V1 Elk undercount permitted;
- malformed root records discarded;
- diverse K32 NNUE prefilter;
- R600 sequential halving with LMR;
- free overflow replacement, no paid wildlife wipes;
- all final execution and scoring in V2.

No policy, feature, weight, seed derivation, fallback, rollout, allocator,
candidate, prelude, or threshold from ADR 0046 may change.

For each paired comparison define:

- `token_spend = max(0, -NatureTokenDelta)`;
- `non_token_score_delta = PairedTotalDelta - NatureTokenDelta`;
- `board_points_per_token = non_token_score_delta / token_spend` when
  `token_spend > 0`; otherwise the efficiency gate passes without division.

## Frozen Confirmation

- Baseline: promoted
  `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
- Treatment:
  `canonical-action-legacy-heuristic-v1-k32-r600-lmr-no-paid-prelude`.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Seeds: `32400-32409`.
- Games: 10, 40 scored seats per strategy.
- Exact 600 rollouts, sequential local CPU.
- Maximum runtime: 2,400 treatment seconds per game.

Every gate must pass:

- treatment mean at least 95.00;
- paired gain at least +1.50;
- paired 95% confidence lower bound greater than 0;
- non-token score delta at least +2.00;
- at least 2.00 board points per additional token point spent;
- total wildlife delta at least 0;
- habitat delta at least 0;
- Nature Token delta at least -2.00;
- all retained and selected actions canonical;
- fallback at most 1%;
- only permitted Elk undercount score differences;
- runtime at most 2,400 seconds per game.

A complete pass authorizes collection of canonical action-imitation data from
this teacher for a fresh MLX-native apprentice. The historical network remains
non-promotable. Any failed gate rejects this confirmation and opens no
distillation dataset.

## Command

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- productive-token-compare \
  --games 10 --first-seed 32400 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-action-legacy-productive-token-confirm10.json
```

## Result

Every frozen gate passed:

- treatment 96.350 versus strong at 91.975;
- +4.375 paired, 95% CI `[+2.938,+5.812]`;
- 10 wins, zero ties, zero losses;
- total wildlife +2.350 and habitat +2.250;
- Nature Tokens -0.225;
- non-token score +4.600;
- 20.444 board points per additional token point;
- all 800 selected actions canonical, zero fallbacks;
- all 24,069 retained K32 records canonical;
- 668/32,089 malformed source records removed, a 2.082% rate;
- zero score mismatches;
- 154.760 treatment seconds per game.

Artifact:
`docs/v2/reports/canonical-action-legacy-productive-token-confirm10.json`,
BLAKE3
`833848f809e938bd71611fb6df960e1a2569eb4d04a80022b5b12c08f3b1649f`.

This qualifies the teacher for fresh canonical action-imitation data and an
MLX-native apprentice. It does not promote the historical network itself.
