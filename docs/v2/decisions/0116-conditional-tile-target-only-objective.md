# ADR 0116: Conditional Tile Target-Only Objective Pilot

Status: complete; `target_only_tile_objective_insufficient`

Date: 2026-06-16

Experiment ID: `conditional-tile-target-only-objective-v1`

## Context

ADR 0115's frozen tile ranker is a valid but decisive proposal failure. Its
selected epoch-20 checkpoint reached 72.60% train factor recall and 66.57%
validation recall. With draft, wildlife, and final selection held
oracle-perfect, tile retrieval retained only 72.97% of validation target
actions and 92.08% of validation R4800 winners. No possible ADR 0115 wildlife
checkpoint can repair that proposal loss.

The post-selection mechanistic evidence isolates the first treatment:

- all 850,246 train and 348,069 validation tile model inputs are unique, with
  zero exact target or rank contradictions;
- on the eight widest supervised train queries, the top-32 boundary gradient
  and combined regression/listwise gradient have mean cosine `-0.738910`;
- boundary and rank-regression gradients have mean cosine `-0.879902`;
- the combined auxiliary gradient norm is `28.0464`, versus `24.3817` for the
  boundary term; and
- even an oracle reranker over the union of learned top 32 and screen-prior top
  32 reaches only 78.29% validation tile recall.

This is direct evidence of objective conflict, not exact representation
collision or a cheap score-blending opportunity.

## Frozen Evidence

- ADR 0115 tile weights BLAKE3:
  `b775d63def9c05495aa842aa9b71447ecb8a0c680ef1938167c46127170ee740`.
- Objective-gradient scientific BLAKE3:
  `e8bbe0b28c6a93f314266b12c72458a277e6dd7f0117141be79cc6eb63afbe04`.
- Train collision scientific BLAKE3:
  `0e5fa3f0d0feb2198c23024a34e879f4c714dca45eb66daa8c772e3b4bfd7d9f`.
- Validation collision scientific BLAKE3:
  `7f3f7257e29c39d6b8191785b4e10e9936d0b5b054e5bbfa07a14cb061bdff54`.
- Complementarity scientific BLAKE3:
  `346dc07b2d0babd86a5786aeaa0520d617f5f3dab24cae1aefd5653abb4d65a2`.
- Train cache payload BLAKE3:
  `1707fd84fac77dee0e4878165bf8f8b98869b6d4d206deb55db030321cc96ede`.
- Validation cache payload BLAKE3:
  `b128a3b5bf53e135febf39dba02d9c7486692245523516a5ee3031eea795229b`.

The sealed test, gameplay, new teacher compute, cloud, and external compute
remain closed.

## Frozen Treatment

Train exactly one conditional tile ranker from scratch with:

- the exact ADR 0115 parent state, draft-conditioned query, tile item features,
  set architecture, hidden width 256, and retrieval width 32;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- batch size 32, 20 epochs, seed `2026061648`;
- the exact class-balanced top-width membership BCE already present as ADR
  0115's boundary term; and
- no smooth-L1 rank regression and no scale-16 listwise term.

Queries whose item count is at most 32 are already retrieval-perfect and
produce no optimizer update. Wider queries receive equal query weight.
Checkpoint selection uses train target recall, then exact-query recovery. The
validation split is evaluated once after selection.

No architecture, feature, width, seed, optimizer, learning-rate, epoch,
initialization, warm-start, or loss-weight sweep is allowed.

## Evaluation

The selected checkpoint must:

1. replay bit-identically on a different host;
2. remain finite, below 4 GiB peak process RSS, and at zero process swaps;
3. be evaluated alone with draft, wildlife, and final selector held
   oracle-perfect; and
4. be evaluated in the frozen ADR 0115 hierarchy using the selected ADR 0115
   draft and wildlife checkpoints.

## Gates

Classify the treatment as `target_only_tile_objective_sufficient` only if:

- train tile factor recall exceeds 95%;
- validation tile factor recall exceeds 90%;
- oracle-other-stage validation target-action recall exceeds 98%;
- oracle-other-stage validation R4800 winner retention exceeds 98%; and
- the fully integrated learned proposal passes every ADR 0115 proposal gate.

Otherwise classify `target_only_tile_objective_insufficient`. A failure closes
boundary-only BCE and requires a new mechanistic audit of model capacity and
query-conditioned representation before another tile training run.

## Cluster Execution

One Mac trains the single frozen model. The other three Macs perform
nonduplicative work: ADR 0115 closeout, target-only implementation and
correctness tests, cross-host replay preparation, report tooling, and the next
independent research queue. Duplicate target-only training is prohibited.

The trainer launches only after ADR 0115's pipeline is mechanically classified.
If ADR 0115 is pipeline-invalid for a reason that undermines the shared cache
or tile evidence, this pilot is cancelled rather than interpreted.

## Maximum Compute

One 20-epoch tile origin, one cross-host selected-weight replay, one
oracle-other-stage ceiling evaluation, one frozen full-hierarchy integration,
focused and full tests, and one report. No sweep or confirmation seed.

## Result

Every pipeline gate passed. The selected john3 checkpoint remained finite,
used 3.00 GiB peak process RSS with zero process swaps, and replayed
bit-identically on john2. The immutable ADR 0115 caches and source pipeline
remained valid. Sealed test, gameplay, new teacher compute, cloud, and external
compute remained closed.

Target-only training improved conditional tile factor recall from 72.60% to
77.21% on train and from 66.57% to 70.59% on validation. That gain did not
survive the complete proposal:

- the tile-only, oracle-other-stage validation ceiling retained 72.34% of
  target actions and 89.58% of R4800 winners;
- the integrated learned proposal retained 71.83% of target actions and 89.58%
  of winners at 1,062.0 proposals on average; and
- the learned top 64 retained 11.83% of targets and 56.67% of winners, with
  0.172370 mean regret.

Relative to ADR 0115, validation tile factor recall improved by 4.02 points,
but integrated proposal recall fell by 0.65 points and winner retention fell
by 2.50 points. Removing the conflicting auxiliary losses therefore improved
stage fit without repairing end-to-end ranking.

The mechanical classification is
`target_only_tile_objective_insufficient`. Boundary-only BCE is closed. The
next experiment must audit capacity and query-conditioned representation
before another full tile training run.

Machine-readable combined scientific BLAKE3:
`0d4681df11a527ca1571008cca8b6e55800866380fb8a4cf7450def1ed54a4f6`.
