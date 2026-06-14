# ADR 0045: Canonical-Filtered Legacy Teacher

Status: rejected at compatibility on 2026-06-12. No R600 or strength seed was
opened.

## Context

ADRs 0043 and 0044 independently established exact public-state and scoring
translation, then showed that malformed v1 wildlife-placement records can
reach the champion's retained K32 frontier. The exact historical policy is
therefore not canonical.

The malformed rate was only 0.811% before the retained failure. The remaining
bounded question is whether the useful historical value/search signal survives
when the root frontier is constrained by V2's canonical legal-action contract.

## Decision

Create a new diagnostic teacher policy:

1. translate the observable v2 state exactly as before;
2. generate the unmodified legacy expanded frontier;
3. map every record through canonical v2 transition validation;
4. permanently discard every record that fails validation;
5. run the unchanged diverse NNUE K32 prefilter and R600/LMR allocator on the
   surviving legacy records only;
6. require the selected action to validate again before V2 execution.

Malformed records are removed, never repaired, converted to wildlife skips, or
used as labels. The original and filtered candidate counts are reported at
every decision.

This is not the reproduced v1 champion. It is a fresh research policy using a
historical evaluator over a canonical root action set. Historical weights
remain ineligible for production and final validation.

## Frozen Compatibility Audit

- Trajectory: v2 `pattern-aware-v1-k8-h6-b8-m4`.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Seeds: `32000-32003`.
- States: all 320 decisions.
- Local CPU only.

Every gate must pass:

- 100% public-state translation and complete score parity;
- repeated-translation and hidden-order invariance;
- malformed source records at most 10% of expanded records;
- at least one canonical candidate remains at every decision;
- every retained K32 candidate is canonical;
- filtering is deterministic and selected-action mapping is replayable;
- no unregistered legacy-affecting environment variable is present.

## Frozen Strength Protocol

Baseline is promoted
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
Treatment is symmetric canonical-filtered K32/R600/LMR play with free-overflow
preparation and no paid wipes.

- Runtime smoke: seed `32099`.
- Qualification: seeds `32100-32102`, only after smoke.
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

Treatment mean below 92.75, gain below +0.50, malformed selected action, or
fallback above 5% rejects the legacy-teacher branch entirely. No threshold,
frontier, weight, rollout, or allocator change is permitted.

## Commands

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- filtered-compatibility \
  --games 4 --first-seed 32000 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-filtered-legacy-teacher-v1-compatibility4.json
```

After a complete pass:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- filtered-compare \
  --games 1 --first-seed 32099 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-filtered-legacy-teacher-v1-r600-runtime-smoke-1.json
```

Only after the smoke:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- filtered-compare \
  --games 3 --first-seed 32100 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-filtered-legacy-teacher-v1-r600-qualification3.json
```

## Result

The compatibility audit rejected at decision 49 of seed `32000`, before any
R600 search:

- 48 states translated and 192 boards matched through decision 48;
- 1,994 expanded records were inspected;
- 48 malformed records were removed, a 2.407% source-malformation rate;
- all 1,483 filtered K32 records inspected were canonical;
- player 3 then exposed a scoring mismatch: canonical V2 scored Elk A at 14
  while historical V1 scored the identical five-Elk layout at 13.

The layout contains a straight line of three Elk and an adjacent straight line
of two Elk. The official rulebook requires connected Elk to use the
interpretation yielding the largest score, so `9 + 5 = 14`. V1's
longest-first greedy partition is not exact and returned 13. The disagreement
is preserved by
`connected_elk_lines_use_the_highest_scoring_partition` in the trusted
differential fixtures.

Artifact:
`docs/v2/reports/canonical-filtered-legacy-teacher-v1-compatibility4.json`,
BLAKE3
`7bb0051910ab0daf14b6ae2c0218f15324e4ecf146e588e115a0fa6d32ad710c`.

This rejects score-exact reuse of the historical evaluator. Runtime seed
`32099` and qualification seeds `32100-32102` remain unopened. A successor may
study the legacy network/search as an explicitly approximate heuristic, but
must not describe it as canonical score parity.
