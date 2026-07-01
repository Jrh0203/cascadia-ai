# ADR 0043: Isolated Legacy Teacher Bridge

Status: rejected at the compatibility gate on 2026-06-12. Strength seeds were
not opened.

## Context

The independently reproduced v1 champion scored 95.895 over 50 games and 200
seat scores, while promoted v2 strong scored 91.915 on its disjoint
confirmation. The engines are not state-compatible, so this gap does not yet
show that the historical network or search policy can provide valid training
signal under canonical v2 rules.

V2 production crates must not import v1. The existing
`cascadia-differential` crate is the explicit test-only boundary for
cross-engine evidence and is the only permitted home for this adapter.
Historical weights remain ineligible as a final solution.

## Decision

Build a one-way public-state bridge from canonical v2 into an isolated v1
teacher state:

- expose an owned `PublicGameState` snapshot from `cascadia-game`;
- translate all visible boards, rotations, wildlife, Nature Tokens, market,
  current player, and remaining turns by enum name rather than discriminant;
- reconstruct public unseen tile and wildlife inventories from the official
  component catalog minus visible components;
- derive inventory order and search RNG only from a domain-separated hash of
  the public snapshot;
- reject coordinates outside the legacy 21 by 21 board;
- compare every translated board's complete AAAAA score breakdown with v2;
- map every legacy candidate back to canonical `TurnAction` and require v2
  transition validation before counting it as compatible.

The bridge may depend on both engines only inside `cascadia-differential`.
No v1 type, model, environment contract, or weight path may enter
`cascadia-game`, `cascadia-sim`, `cascadia-search`, the API, or the web app.

The teacher treatment freezes the reproduced champion's main move operator:

- `nnue_weights_v4opp_modal_iter3.bin`;
- `mid-features` plus `v4-opp`;
- expanded candidates;
- diverse top-32 NNUE prefilter;
- 600-rollout sequential halving with LMR;
- deterministic per-seat search RNG derived from the v2 game seed.

V2 applies its canonical free three-of-a-kind replacement before translation.
Paid wildlife wipes are disabled. This deliberately measures the legacy main
policy, not the full historical pre-move optimizer.

## Frozen Compatibility Audit

- Trajectory policy: frozen v2 `pattern-aware-v1-k8-h6-b8-m4`.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Seeds: `31600-31603`.
- States: all 320 decisions after the canonical free-overflow prelude.
- Candidate audit: complete expanded frontier and diverse NNUE top 32.
- Local CPU only.

Every gate must pass:

- 100% state translations succeed;
- every board has exact habitat, wildlife, Nature Token, and base-total parity;
- every state has at least one legacy candidate;
- 100% of expanded and prefiltered candidates map to v2-legal actions;
- mapped actions preserve market draft identity, coordinate, rotation, and
  optional wildlife placement;
- repeated translation is byte-deterministic and hidden-order invariant;
- no unregistered legacy-affecting environment variable is present.

Any failure rejects the bridge. Strength evaluation is forbidden until the
audit passes.

## Frozen Strength Protocol

Baseline is promoted
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
Treatment is symmetric isolated legacy main-policy play under canonical v2
rules. Both use identical game seeds.

- Runtime smoke: seed `31699`, one paired four-seat block.
- Qualification: seeds `31700-31702`, only after the smoke passes.
- Rollouts: exactly 600.
- Execution: sequential local CPU.
- Fallback: exact pattern-aware only if translation or mapped-action validation
  fails; every fallback is counted.

Smoke passes only if treatment completes within 2,400 seconds, every selected
action is legal, score parity remains exact at every decision, and fallback
rate is at most 1%.

Qualification is non-promotable. It authorizes a separately registered fresh
MLX teacher-distillation or policy-iteration experiment only if:

- treatment absolute mean is at least 94.50;
- paired gain over promoted strong is at least +1.50;
- total wildlife delta is at least +0.50;
- habitat delta is at least -0.50;
- Nature Token delta is at least -1.00;
- fallback rate remains at most 1%;
- runtime remains at most 2,400 seconds per block.

Treatment mean below 93.00, paired gain below +0.50, any score-parity failure,
or fallback above 5% rejects this teacher path. Intermediate results permit no
changes to weights, frontier, rollout count, allocator, prelude, seeds, or
thresholds.

## Commands

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- compatibility \
  --games 4 --first-seed 31600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/legacy-teacher-bridge-v1-compatibility4.json
```

After every compatibility gate passes:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- compare \
  --games 1 --first-seed 31699 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/legacy-teacher-bridge-v1-r600-runtime-smoke-1.json
```

Only after the smoke passes:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- compare \
  --games 3 --first-seed 31700 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/legacy-teacher-bridge-v1-r600-qualification3.json
```

## Interpretation

A pass would not promote a historical model. It would establish a strong,
rules-compatible cross-engine teacher from which fresh observable datasets can
be collected and new MLX-native models trained. A failure would show whether
the legacy advantage depends on incompatible rules, hidden state, pre-move
optimization, coordinate assumptions, or the old engine itself.

## Result

The audit stopped at decision 10 of seed `31600`, before any strength seed was
opened:

- 10 states and 40 boards translated with exact complete score parity;
- hidden-order and repeated-translation checks passed;
- 333 expanded candidates were inspected;
- 315 of those candidates mapped to legal v2 actions;
- all 261 candidates retained by the diverse top-32 prefilter up to the
  failure point were legal;
- maximum observed absolute coordinate was 2.

The first incompatible record was:

```text
ScoredMove {
  market_index: 2,
  tile_q: -2,
  tile_r: 2,
  rotation: 0,
  wildlife_q: Some(0),
  wildlife_r: Some(1),
  score: 17,
  eval: 517,
  wildlife_market_index: None
}
```

The drafted token was Bear, but the existing tile at `(0,1)` could not support
Bear. Canonical v2 rejected the action. The legacy `execute_move` path ignores
the return value of `place_wildlife`, advances the turn, and silently loses the
token. This is an actual move-representation and rules-execution mismatch, not
an enum, score, market, coordinate, or hidden-state translation error.

The frozen 100% expanded-frontier legality gate therefore failed. The report is
`reports/legacy-teacher-bridge-v1-compatibility4.json`, BLAKE3
`c9e7db3587f40d9e981a023a7529ad61d351428734e1e69f0a10ac5e300a4e25`.
No runtime smoke or qualification was permitted.

The result does not yet reject the actual K32/R600 teacher: every retained
candidate observed before the stop was canonical. A separately registered
successor may test whether malformed records are confined below the frozen
prefilter boundary. It must never execute, repair, or reinterpret an invalid
legacy record as a v2 action.
