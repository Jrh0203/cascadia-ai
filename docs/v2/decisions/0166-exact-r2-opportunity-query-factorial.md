# ADR 0166: Exact-R2 Opportunity Query Factorial

Status: accepted; implementation qualification in progress

Date: 2026-06-17

Experiment: `opportunity-cross-attention-mlx-tournament-v1`

Protocol: `exact-r2-opportunity-query-factorial-v1`

## Context

The representation audit identified a specific remaining blind spot: V2 already
stores exact semantic supply, exact occupied/frontier/component/motif objects,
and exact public action edits, but the relational ranker pools these facts
independently before candidate scoring. A candidate cannot explicitly ask which
scarce supply objects satisfy the opportunity created by that action, or which
frontier and motif objects become relevant because of that action.

The preceding compact-substrate tournament also established an important
boundary:

- the historical 441-cell dense footprint is unnecessary;
- a centered 121-cell hex disk does not exist;
- centered radii four, five, and six contain 61, 91, and 127 cells;
- exact R2 uses sparse occupied-plus-frontier objects rather than a dense
  441-cell tensor; and
- the compact quotient arms fail the frozen absolute serving limits, so the
  exact sparse R2 control is the only admissible warm-start substrate.

ADR 0166 therefore does not add another board census, dense crop, or wider MLP.
It tests whether candidate-conditioned retrieval from facts already present in
R2 and S1 improves action ranking.

## Decision

Warm-start one identical MLX adapter graph from the final authorized
`c0-exact-r2` checkpoint and freeze every inherited parameter. Train only the
104,256-parameter opportunity adapter.

The graph always contains:

1. an exact-S1 supply-token projection;
2. an exact-R2 board-memory projection;
3. one four-head supply cross-attention module;
4. one four-head frontier cross-attention module;
5. one shared context fusion trunk; and
6. a zero-initialized residual projection into the inherited candidate hidden
   state.

Arm identity changes only which query reads each memory:

| Arm | Supply query | Frontier query | Host |
|---|---|---|---|
| `c0-parent-conditioned` | parent | parent | john1 |
| `t1-supply-query` | candidate | parent | john2 |
| `t2-frontier-query` | parent | candidate | john3 |
| `t3-combined-query` | candidate | candidate | john4 |

The zero-initialized residual must make all four arms exactly prediction-equal
to the C0 warm start before optimization.

## Frozen Training

```text
Apple Silicon MLX GPU
2,000 AdamW steps
4 complete decision groups per step
maximum 512 retained actions per group
learning rate 1e-4
weight decay 1e-4
checkpoint every 250 steps
24-group live probe every 100 steps
no early stopping
no hidden or sealed data
```

Every arm receives the same exact-R2 parent surface, exact R2 candidate edit,
exact semantic supply tokens, labels, group order, candidate order, D6
transform, objective, and optimizer schedule.

## Proof Obligations

Before production:

- the final C0 report and checkpoint must be structurally valid and agree by
  BLAKE3;
- all inherited parameter tensors must equal C0;
- all four total and adapter parameter layouts must be identical;
- all four initial adapter tensors must be identical;
- the first scientific batch hash must be common;
- a common-arm smoke replay must agree across john1 through john4; and
- each host must pass a source, cache, MLX, warm-start, and zero-parity
  preflight.

During and after training:

- every batch trace is append-only and contiguous;
- all losses, predictions, and uncertainties are finite;
- every production report contains one stable record for each of the 240
  validation decisions;
- every collected report is re-bound to its exact final checkpoint manifest
  and model bytes;
- inherited parameter tensors remain byte-identical to C0;
- all validation decisions and candidates are scored once;
- R6 apply/undo parity has zero failures; and
- isolated serving is measured in a fresh process.

## Selection

Candidate-query treatments are compared against both controls:

1. `c0-parent-conditioned`, which controls for adapter capacity and global
   memory context; and
2. the unmodified final C0 checkpoint, which controls for whether adding the
   adapter helped at all.

A treatment can advance only if it:

- passes every structural and serving gate;
- improves full-validation top-64 R4800 winner recall against both controls;
- has positive paired evidence on the Elk, Salmon, and Hawk opportunity union;
- does not materially regress low-supply or independent-draft winner recall;
- is noninferior on R4800 RMSE and retained regret; and
- has no contradictory same-host or cross-host evidence.

Among qualifying treatments, selection is lexicographic:

1. top-64 winner recall;
2. primary strategic-opportunity recall;
3. top-64 retained regret;
4. R4800 RMSE;
5. complete-decision P99 latency; and
6. peak process RSS.

The two-by-two factorial supply effect, frontier effect, and interaction are
reported regardless of whether any arm advances.

## Serving Limits

```text
combined model plus R6 complete-decision P99 <= 250 ms
peak process RSS <= 4 GiB
system swap delta <= 0
R6 exact parity = true
```

These are absolute requirements, not relative wins.

## Cluster Schedule

1. Fan out the immutable source bundle and final C0 run.
2. Run the common-arm smoke on all four hosts concurrently.
3. Authorize and preflight all assigned arms.
4. Start `c0-parent-conditioned` on john1 as soon as S6 releases it.
5. Run C0 paired-control replays on john2 through john4 concurrently.
6. Start T1, T2, and T3 immediately as those hosts finish their paired replay.
7. Collect reports whole-tree, run the order-invariant paired classifier, and
   publish the dashboard record.

This ordering keeps all four machines useful while closing the prior
tournament and beginning the next causal test.

The installed scheduler graph is generated by
`tools/opportunity_cross_attention_mlx_queue.py`. It contains 20 checksum-bound
tasks: bundle fanout, four host smoke replicas, smoke collection and
comparison, authorization, launch-control fanout, four preflights, untouched
C0 panel construction, four unique production arms, checkpoint/report
collection, and one decision-terminal classifier. Production tasks depend on
all four preflights; each smoke task also waits for the registered host-local
ADR 0161 or S6 prerequisite.

The active scheduler graph is queue revision `v2`. Every john1 fanout and
collection operand is absolute because queue tasks execute with the immutable
bundle's `source/` directory as their working directory. The superseded
revision failed closed during the initial pre-scientific bundle fanout and is
retained in the queue as orchestration audit evidence.

The terminal classifier is
`tools/opportunity_cross_attention_mlx_report.py`. It performs the frozen
100,000-replicate complete-decision bootstrap, reports the two-by-two
factorial, checks protected slices and absolute serving, verifies collected
checkpoint bytes, and proves that forward and reverse report order produce
byte-identical scientific output.

ADR 0173 repairs two terminal-classifier bookkeeping defects discovered before
any final production report existed: protected-slice attribution is now
arm-local, and eligible-arm selection includes the preregistered RMSE and RSS
criteria. Training, thresholds, reports, and paired evidence are unchanged.

The same tool owns bundle construction:

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/opportunity_cross_attention_mlx_queue.py build-bundle
```

Its explicit source boundary includes the complete V2 MLX and Rust source
roots, this ADR and preregistration, the campaign, smoke, classifier, queue,
transfer, and bundle tools, plus the exact R6 replay binary. The generic bundle
builder rejects symlinks, hashes every regular file, content-addresses the
manifest, and seals the resulting tree read-only.

## Consequences

A passing arm authorizes gameplay qualification, not champion replacement.
Gameplay must first pass paired 4-player AAAAA, no-habitat-bonus evaluation
against the current 95.94 champion before any progress-to-100 claim.

A null result rejects candidate-query conditioning at this adapter capacity and
training regime. A parent-control-only win says additional global context
capacity helped, but candidate-specific retrieval did not. A serving failure
rejects the implementation regardless of offline quality.
