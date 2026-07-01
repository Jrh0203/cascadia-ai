# P1 Relational Selected-Prefix Pointer Pilot V1 Preregistration

Date: 2026-06-16

ADR: 0175

Experiment: `p1-relational-selected-prefix-pointer-pilot-v1`

## Question

Can exact selected-prefix pointers over the accepted sparse R2 state learn the
oracle-proven complete-action hierarchy that flattened pointwise factors could
not learn?

## Authorization

Production starts only after ADR 0174 produces
`p1_relational_pointer_foundation_passed` from crossed complete train and
validation audits and explicitly authorizes
`matched-mlx-selected-prefix-pointer-pilot`.

## Matched Contract

- Same immutable open train and validation targets as ADR 0115.
- Same draft-16, tile-32, wildlife-8 factorization.
- Same 20, 20, and 10 epoch budgets.
- Same 32, 32, and 256 query batch sizes.
- Same AdamW `3e-4` learning rate and `1e-4` weight decay.
- Same rank-regression, scale-16 listwise, and boundary objective.
- Same train-only checkpoint-selection order.
- Same action maps, champion-frontier anchors, top-64 selector, phase subsets,
  and R4800 labels.
- No sealed test or gameplay.

The treatment changes only the learned representation:

```text
draft object
-> exact frontier token plus rotation
-> exact none, new-tile, or occupied destination pointer
```

## Frozen Parent

The accepted ADR 0161 C0 exact-R2 parent is loaded by verified tensor name and
shape, then frozen. The pointer-specific heads are newly initialized from
preregistered deterministic seeds. Parent outputs are memoized by exact group
and D6 transform within each epoch.

## Stage Runs

The three stages are independent MLX jobs and may run simultaneously on three
cluster machines after authorization:

| Stage | Epochs | Batch | Seed |
|---|---:|---:|---:|
| Draft | 20 | 32 | 2026061675 |
| Tile | 20 | 32 | 2026061676 |
| Wildlife | 10 | 256 | 2026061677 |

Each job writes atomic checksummed checkpoints, optimizer state, an idempotent
JSONL metric history, a train-selected best pointer, and one final report.
The selected model is published at a fixed checksummed path, collected by the
coordinator, fanned out to a distinct host, and replayed over both complete
open splits. Integration is blocked until all three replays match exactly.

## Metrics

Per stage:

- target-factor recall;
- exact-query fraction;
- expected-rank mean absolute error;
- finite and exact query/item coverage;
- frozen-parent identity;
- parent-encoding memo hit rate;
- elapsed time and memory.

Integrated:

- target-positive recall;
- exact target-set fraction;
- R4800 winner retention;
- top-64 confidence-set coverage;
- retained R4800 regret;
- proposal-count distribution;
- early, middle, late, Nature Token, independent-draft, and action-family
  guardrails.

## Success

All validation gates are mandatory:

- proposal target recall greater than 98%;
- R4800 winner retention greater than 98%;
- mean proposals at most 1,024;
- top-64 confidence-set coverage at least 99%;
- mean retained R4800 regret below 0.15;
- no subset or action-family guardrail failure;
- complete finite coverage and unchanged parent identity.
- exact cross-host replay of all three selected stages.

The 512 mean-proposal target is a preferred efficiency threshold, not a reason
to reject an otherwise successful first pointer pilot below 1,024.

## Stop Rules

- A failed ADR 0174 classification blocks all production training.
- A nonfinite loss, parent mutation, cache mismatch, action-map mismatch, or
  incomplete coverage invalidates the affected run.
- Validation never selects a checkpoint or treatment.
- If conditional tile recall remains below 90%, stop before integration or
  gameplay and run one preregistered representation/gradient forensic pass.
- If integrated offline gates fail, no gameplay is authorized.
- If offline gates pass, gameplay still requires a separate paired,
  equal-budget preregistration.

## Claim Boundary

This pilot can establish that exact sparse pointers are a viable proposal
representation. It cannot establish gameplay improvement or attainment of the
100-point objective.
