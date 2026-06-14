# ADR 0048: Legacy Teacher Pattern-Frontier Recall

Status: rejected on 2026-06-12.

## Context

ADR 0047 qualified a canonical V2 action teacher at 96.350. The hardened MLX
ranking pipeline scores candidates from
`pattern-aware-v1-k8-h6-b8-m4`. Distilling selected actions into that pipeline
is only valid when the teacher's selected action is present in the same
candidate frontier at inference.

Training on injected teacher actions that production cannot propose would
create an artificial offline result. Training only on recalled groups without
measuring the omissions would create selection bias.

## Decision

Measure exact action recall while replaying the unchanged qualified teacher:

1. select the canonical teacher action with the exact ADR 0047
   K32/R600/LMR policy;
2. independently generate the default V2 pattern frontier on the same
   pre-action state and market prelude;
3. compare complete typed `TurnAction` values;
4. record total candidates, exact recall, independent-draft recall, and
   early/mid/late recall;
5. preserve all existing legality, score-policy, fallback, runtime, and
   provenance checks.

Phases are based on completed global turns:

- early: `0-26`;
- middle: `27-53`;
- late: `54-79`.

No pattern candidate is injected, repaired, or expanded for this probe.

## Frozen Protocol

- Teacher: unchanged
  `canonical-action-legacy-heuristic-v1-k32-r600-lmr-no-paid-prelude`.
- Pattern frontier: default `pattern-aware-v1-k8-h6-b8-m4`.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Seeds: `32600-32601`.
- Games: 2, 160 decisions.
- Exact 600 rollouts, sequential local CPU.

The existing grouped ranker is authorized only if:

- overall exact recall is at least 80%;
- early, middle, and late recall are each at least 65%;
- all selected and retained actions are canonical;
- fallback is at most 1%;
- only permitted Elk undercount score differences occur;
- runtime is at most 2,400 seconds per game.

Below-threshold recall rejects pattern-frontier imitation and requires a
broader structured V2 action proposal model. The recall threshold, phase
boundaries, frontier, seeds, and teacher may not be changed.

## Command

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
  cargo run --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher -- frontier-recall-probe \
  --games 2 --first-seed 32600 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output docs/v2/reports/canonical-action-legacy-pattern-frontier-recall2.json
```

## Result

The existing pattern frontier is not a valid imitation action space:

- overall exact recall: **82/160 = 51.25%**;
- early recall: **26/54 = 48.15%**;
- middle recall: **24/54 = 44.44%**;
- late recall: **32/52 = 61.54%**;
- independent-draft recall: **11/18 = 61.11%**.

All 160 selected actions and all 4,753 retained legacy candidates were
canonical, fallback remained zero, and the malformed source rate was 2.91%.
The recall failure is therefore a real proposal mismatch, not bridge
corruption.

The report is
`docs/v2/reports/canonical-action-legacy-pattern-frontier-recall2.json`
(BLAKE3
`f3335bed4a1c45b2f5e71f305c6136010aa4f3aadac366aa2d2bd75034e0cd6a`).
No pattern-frontier imitation dataset is authorized. ADR 0049 replaces it
with an explicit-action learner whose production action space is the complete
canonical legal set.
