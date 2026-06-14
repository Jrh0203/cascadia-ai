# ADR 0044: Retained-Frontier Legacy Teacher

Status: rejected at the compatibility gate on 2026-06-12. Strength seeds were
not opened.

## Context

ADR 0043 proved exact public-state and score translation, but rejected the
complete legacy expanded frontier because v1 contains malformed candidate
records. Its move executor silently ignores an unsupported wildlife placement
and advances the turn, behavior canonical v2 correctly forbids.

Before the failure, every candidate retained by the champion's diverse top-32
NNUE prefilter was canonical. The reproduced champion always applies that
prefilter when the expanded frontier exceeds 32 candidates. The remaining
question is whether malformed records are confined outside the actual R600
decision frontier.

## Decision

Reuse the isolated, public-only translation from ADR 0043. Preserve the legacy
expanded frontier and prefilter exactly, but distinguish telemetry from
authorization:

- inspect every expanded candidate and count strict v2 legality;
- do not execute, repair, canonicalize, or reinterpret malformed records;
- run the unchanged diverse NNUE prefilter over the original unmodified
  frontier;
- require every retained candidate to map strictly to a canonical v2 action;
- forbid strength evaluation if any retained candidate is malformed;
- require the final R600-selected candidate to pass v2 transition validation.

This keeps the teacher's real K32 ordering intact. Invalid low-ranked records
remain documented v1 debt and can never enter V2 state, datasets, or gameplay.

## Frozen Compatibility Audit

- Trajectory: v2 `pattern-aware-v1-k8-h6-b8-m4`.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Seeds: `31800-31803`.
- States: all 320 decisions after canonical free-overflow preparation.
- Weights/features/frontier: unchanged from ADR 0043.
- Local CPU only.

Every gate must pass:

- 100% state translation and complete score parity;
- repeated-translation and hidden-order invariance;
- at least one expanded and retained candidate per state;
- malformed expanded candidates are at most 10% of all expanded candidates;
- 100% of retained K32 candidates are canonical v2 actions;
- no malformed candidate is repaired or passed to R600;
- no unregistered legacy-affecting environment variable is present.

Any retained-candidate failure rejects the path. The expanded malformed rate
is descriptive legacy debt; exceeding 10% also rejects because the source
frontier would be too semantically noisy to trust.

## Frozen Strength Protocol

Baseline is promoted
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
Treatment uses the unchanged retained K32/R600/LMR main policy under v2 rules,
canonical free-overflow preparation, and no paid wipes.

- Runtime smoke: seed `31899`.
- Qualification: seeds `31900-31902`, only after smoke.
- Sequential local execution.
- Exact 600 rollouts.
- No fallback is allowed for a malformed retained or selected action.
  Coordinate-range translation failure may use pattern-aware fallback only for
  telemetry and must remain at most 1%.

Smoke requires strict selected-action legality, fallback at most 1%, and at
most 2,400 seconds per block.

Qualification authorizes a fresh MLX-native teacher experiment only if:

- treatment mean is at least 94.50;
- paired gain over strong is at least +1.50;
- total wildlife delta is at least +0.50;
- habitat delta is at least -0.50;
- Nature Token delta is at least -1.00;
- fallback remains at most 1%;
- runtime remains at most 2,400 seconds per block.

Treatment mean below 93.00, gain below +0.50, any retained/selected malformed
action, or fallback above 5% rejects this teacher path. No historical model is
eligible for production regardless of result.

## Commands

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- retained-compatibility \
  --games 4 --first-seed 31800 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/retained-legacy-teacher-v1-compatibility4.json
```

After a complete compatibility pass:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- retained-compare \
  --games 1 --first-seed 31899 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/retained-legacy-teacher-v1-r600-runtime-smoke-1.json
```

Only after the smoke:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- retained-compare \
  --games 3 --first-seed 31900 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/retained-legacy-teacher-v1-r600-qualification3.json
```

## Result

The successor also failed before strength evaluation. On seed `31800`, the
audit reached decision 19:

- 19 states and 76 boards translated with exact score parity;
- repeated translation and hidden-order invariance passed;
- 740 expanded records were inspected;
- 6 were malformed, a low 0.811% expanded-record rate;
- 575 records were retained across the top-32 frontiers;
- one retained record was malformed.

That retained candidate drafted Salmon and requested placement at `(0,1)`,
where the tile could not support Salmon. Six rotations of the same underlying
malformed placement appeared in the expanded set; at least one survived NNUE
prefiltering. The frozen 100% retained-frontier legality gate therefore failed.

The report is
`reports/retained-legacy-teacher-v1-compatibility4.json`, BLAKE3
`8d8975e13a72779f47912835146859d728ac8ac4db529a81d22448e50739bcdd`.
No R600 decision and no strength seed was opened.

The exact historical K32 policy is now closed as a canonical teacher. A final
bridge experiment may remove malformed root records before ranking, because
canonical legal-action filtering is an explicit V2 invariant. That successor
is a new policy, not a reproduction of the v1 champion, and must be evaluated
as such.
