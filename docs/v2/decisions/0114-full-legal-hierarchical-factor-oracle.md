# ADR 0114: Full-Legal Hierarchical Factor Oracle

Status: completed as `hierarchical_factor_oracle_sufficient`; one learned
hierarchical factor-retrieval pilot is authorized.

Date: 2026-06-16

Experiment ID: `full-legal-hierarchical-factor-oracle-v1`

## Context

ADR 0113 closed the monolithic 192-wide local adapter. Direct balanced
supervision fit the 324-candidate group exactly but failed on the three
2,975-4,368-candidate groups. Phase 2 requires a complete factorized proposal
whose authority scales with the legal action structure rather than forcing
thousands of actions through one shared scalar bottleneck.

Before training factor models, this audit measures the exact teacher ceiling
and candidate reduction of hierarchical retrieval.

## Frozen Evidence

- ADR 0113 combined report BLAKE3:
  `c2d373d58cb36fc3e876854736183c6580638460a98381dbd17240001d1d74d1`.
- ADR 0113 source bundle BLAKE3:
  `b113f69e8aef999fd6bdfbcd6994952579184fd93e71322c9df561b7bfd0a638`.
- Unchanged complete-action train and validation datasets, scale-16 expected
  ranks, target masks, champion-frontier flags, action hashes, and
  frontier-anchored width-64 selector.

No teacher compute, target, split, selector, action, or label may change.
Sealed test and gameplay remain closed.

## Exact Action Factorization

Every decoded 140-value complete action is partitioned into:

1. `draft_premove`: indices `0:34` plus `45:128`;
2. `tile_placement`: indices `34:42`; and
3. `wildlife_placement`: indices `42:45`.

Immediate score and score deltas at `128:140` are consequences, not action
choices, and are excluded from factor identity.

The concatenated exact float32 bytes of these three factors must be a bijection
with valid complete actions inside each group. Any collision invalidates the
audit.

Champion-frontier actions remain unconditionally retained. Retrieval applies
only to eligible nonfrontier actions with expected-rank supervision.

## Oracle Retrieval

For each group and arm:

1. Rank draft factors by the minimum expected rank of any supervised action
   containing that factor, with exact factor bytes as the tie break.
2. Retain the arm's top draft width.
3. For each retained draft, rank tile factors conditionally by the minimum
   expected rank of actions sharing that draft and retain the arm's tile
   width.
4. For each retained draft+tile prefix, rank wildlife factors conditionally
   and retain the arm's wildlife width.
5. Form the candidate set from actions passing all retained factors plus every
   champion-frontier action.
6. Apply the frozen expected-rank top-64 selector inside that candidate set and
   measure target recall, exact target sets, winner retention, and candidate
   count.

This is an oracle structural ceiling, not a deployable policy.

## Four Distinct Arms

- `conditional-compact`, widths `4 / 8 / 2`.
- `conditional-balanced`, widths `8 / 16 / 4`.
- `conditional-wide`, widths `16 / 32 / 8`.
- `independent-wide`, widths `16 / 32 / 8`, but tile and wildlife
  factors are ranked globally rather than conditionally.

Every arm audits all 560 train and 240 validation groups. A dynamic
one-process-per-host queue launches the four distinct origins across
john1-john4, then replays each arm on a different available host without a
fixed barrier.

## Gates

Every arm must:

- cover all 800 open groups and all valid candidates exactly once;
- prove complete-action factor bijection in every group;
- preserve every champion-frontier action;
- produce finite ranks and exact selector accounting;
- reproduce scientific metrics bit-for-bit on cross-host replay;
- use matching source and dataset identities across john1-john4;
- remain below 4 GiB RSS with zero process swaps and no attributable positive
  system-swap growth; and
- keep training, gradients, validation selection, sealed test, gameplay, new
  teacher compute, cloud, and external compute closed.

## Mechanical Classification

The hierarchical factor oracle is sufficient only when `conditional-wide`
achieves on both train and validation:

- target-positive recall at least 98%;
- exact target sets at least 90%;
- R4800 winner retention at least 99%; and
- mean retained candidates at most 2,048.

Classify:

1. `hierarchical_factor_oracle_invalid`
   - any identity, coverage, bijection, selector, replay, resource, or sealed
     gate fails.
2. `hierarchical_factor_oracle_sufficient`
   - every pipeline gate and all conditional-wide strength/size gates pass.
3. `hierarchical_factor_oracle_insufficient`
   - the pipeline passes but conditional-wide misses any gate.

Only the sufficient outcome authorizes one learned hierarchical factor
retrieval pilot. No result directly authorizes a full trainer, sealed test, or
gameplay.

## Maximum Compute

Four distinct full-open-data static oracle arms, four cross-host replays, source and
dataset checks, focused/full tests, one combine, and one report. No training,
gradient, optimizer, budget sweep, fifth arm, teacher rollout, full trainer,
sealed test, gameplay, cloud, Modal, or external compute.

## Result

All four arms audited all 560 train and 240 validation groups, preserved every
champion-frontier action, proved complete-action factor bijection, and
cross-host replayed bit-identically. Every source, dataset, resource, coverage,
selector, and sealed-domain gate passed.

| Arm | Train recall | Validation recall | Validation exact | Mean validation proposals |
|---|---:|---:|---:|---:|
| conditional compact | 56.69% | 58.03% | 0.00% | 80.7 |
| conditional balanced | 88.76% | 89.50% | 39.58% | 211.0 |
| **conditional wide** | **99.27%** | **99.18%** | **95.00%** | **482.4** |
| independent wide | 94.80% | 94.66% | 66.25% | 601.9 |

Conditional-wide also retained 100% of validation R4800 winners, with
validation p99 proposal count 724 and maximum 757. It passed every
preregistered strength and size gate. The 4.52-point recall gap between
conditional-wide and independent-wide proves that prefix conditioning is
material rather than cosmetic.

The mechanical classification is `hierarchical_factor_oracle_sufficient`.
This authorizes one learned hierarchical factor-retrieval pilot using the
frozen `16 / 32 / 8` conditional budget. It does not authorize a full trainer,
sealed test, or gameplay.

The campaign completed in 22.15 seconds, scheduled 61.91 process-seconds,
averaged 2.80 active processes, peaked at four, and recorded zero queued-work
idle.

Machine-readable result:
`artifacts/experiments/full-legal-hierarchical-factor-oracle-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/full-legal-hierarchical-factor-oracle-v1-result.md`.
