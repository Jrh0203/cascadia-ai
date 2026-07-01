# Full-Legal Screen Width Recovery Preregistration

Status: **locked before recovery collection**

Date: 2026-06-15

Experiment ID: `full-legal-screen-width-recovery-v1-20260615`

## Purpose

The frozen Full-Legal Decision Regret Audit has already rejected K64 action
retention. This experiment performs the required widened rerun. It changes
only the number of cheap-screen actions admitted to substantial public
evaluation; it does not weaken the rollout horizon, numerical path, model,
high-confidence budget, legal-action boundary, or public-information
constraint.

The original 13-game audit remains authoritative for paid-wipe and
realized-hidden-future diagnostics. This recovery run exists only to establish
a credible retained public action set.

## Width Selection

The recovery width was locked after 960 of the 1,040 original decisions were
available and before seed `61004` or any substantive recovery-corpus output
was opened.

Observed high-confidence-winner rank counts across those 960 decisions were:

| Width | Recalled | Recall |
|---:|---:|---:|
| 64 | 869 | 90.521% |
| 128 | 904 | 94.167% |
| 256 | 926 | 96.458% |
| 512 | 939 | 97.813% |
| 1,024 | 950 | 98.958% |

K512 is mathematically unable to satisfy the frozen 98% gate: even 80 of 80
hits on the unopened final game would yield only `1019 / 1040 = 97.981%`.
K1024 is therefore the smallest predeclared recovery width. If and only if the
fresh K1024 evaluation fails the gate, the complete corpus will be rerun at
K2048. No intermediate or post-hoc width may be introduced.

These rank counts select a width; they are not themselves recovery evidence.
Actions ranked 65 through 1,024 were not all substantially evaluated in the
original audit, so K1024 must be rerun over every decision.

## Frozen Domain

The recovery reuses the original champion trajectories and seed ownership:

- john1: `61000-61004`;
- john2: `61005-61008`;
- john3: `61009-61012`.

All 13 games and all 1,040 decisions are rerun. Reusing these seeds is
intentional because this is a paired action-coverage correction, not an online
strength estimate. Online oracle qualification must use a later disjoint
preregistered seed suite.

The rules remain four-player AAAAA with habitat bonuses disabled. The model is
`artifacts/models/legacy-nnue-v4opp-mlx-v1`. `MCE_LMR=1` and
`MCE_DIVERSE_PREFILTER=1` remain enabled.

## Evaluation Contract

Every decision:

1. applies the same canonical free prelude;
2. enumerates and cheaply scores every canonical legal post-prelude action;
3. admits the top 1,024 cheap-screen actions;
4. unions every champion-frontier action, the champion action, and 16
   deterministic rank-stratified sentinels;
5. evaluates that union with exact full-terminal R1200 sequential halving and
   common random numbers within each round;
6. re-evaluates the best eight substantial actions, the champion action, and
   the best champion-frontier finalist with exact full-terminal R4800.

The sequential-halving candidate floor remains active. Widening therefore
increases actual rollout work instead of diluting each round below one sample
per surviving action.

Paid-wipe and realized-hidden diagnostics are disabled in this rerun. They are
already complete in the original paired corpus, account for most of its wall
time, and cannot change deterministic post-prelude action recall. Their
original evidence must remain linked in the final report.

## Exactness Pairing

For every seed and turn, the recovery must match the original corpus on:

- terminal base scores and terminal state hash;
- public and staged public-state hashes;
- champion action identity;
- canonical legal-action count and identity;
- exact score deltas and resulting scores;
- MLX immediate value, remaining value, screen value, and screen rank.

Only retained-source labels, substantial/high-confidence estimates, corrected
frontier bookkeeping, timings, provenance timestamps, and intentionally
disabled diagnostics may differ.

Any fallback, bootstrapped terminal sample, illegal or duplicate action,
non-finite value, hidden-information dependency, identity mismatch, or
incomplete game rejects the shard.

## Performance Calibration

Before locking the full run, john2 and john3 independently evaluated the
performance-smoke seed `60999` at K1024. After removing host/timing metadata,
the complete behavioral artifacts were byte-identical:

- turn 12 digest:
  `522aece5cde817b832c9db398651f2006aec3226f2041521a1ff335ed3f2e591`;
- turn 39 digest:
  `1685d36524e8ab6e8634947b2252200f8bdfd18e3fcb8e5b79093bcef433a650`.

The selective screen-contract fingerprint also matched the original K64
reference exactly across widths:

- turn 12:
  `9b787d1430dcd1cb457979ab7f3469da2846663ccaed0e8ca2f9b3b7226c443a`;
- turn 39:
  `7659217362b7176afd78c54998dc0381a34bf22f2ecd2415fe5fb986482c18d0`.

| Turn | Legal actions | Substantial union | john2 substantial | john3 substantial |
|---:|---:|---:|---:|---:|
| 12 | 2,002 | 1,040 | 5.007s | 5.454s |
| 39 | 6,372 | 1,047 | 3.627s | 3.730s |

The complete audited-decision ladders took 6.77 to 7.86 seconds. This projects
to roughly 7-8 minutes per diagnostics-disabled game and supports a full
three-node collection without changing the evaluation contract.

The predeclared K2048 fallback was also sized at turn 39. Its 2,069-action
substantial union took 7.179 seconds on john2 and 7.462 seconds on john3. The
host-independent behavioral digest was
`ac75174c8b72398be252ef4204fc48ebccb5f332b33cc0738784bde7ac0bfd98`.

Raw calibration artifacts live under
`artifacts/performance/full-legal-audit-k1024-smoke-v1/`.

## Acceptance Gates

K1024 passes only if:

- all 13 games and all 1,040 decisions complete with clean shutdown;
- exact pairing with the original corpus passes every invariant above;
- every canonical action is cheaply screened;
- K1024 recall of the fresh high-confidence winner is at least 98%;
- retained-set mean high-confidence regret is at most 0.15 points;
- recall and regret are reported with game-block 95% confidence intervals;
- early, middle, late, token-bearing, paired-draft, and independent-draft
  slices are reported rather than hidden by the aggregate;
- results reproduce across john1, john2, and john3;
- checksums, model identity, binary identity, source identity, configuration,
  host utilization, and swap state are recorded.

Passing this experiment closes only the widened-screen substantive gate. Phase
1 still requires a disjoint online public-information oracle mean of at least
102 and a paired gain of at least six points with a positive 95% lower bound.
