# ADR 0151: S4 Candidate-Relation Foundation

Status: accepted; corrected production census complete

Date: 2026-06-17

Experiment: `s4-candidate-relation-foundation-v1`

Research-plan item: S4

## Context

The R3 action-edit comparison isolates spatial candidate representation while
holding the parent encoder, exact global edits, supply semantics, capacity,
training schedule, and objective fixed. Its three compact arms completed
before the full-afterstate control. Their diagnostic-only failure atlases
already establish two facts that do not depend on the control:

1. radius one is better than radius two or three on winner recall, retained
   regret, and confidence-set coverage; and
2. residual failures concentrate in middle-game decisions and wide candidate
   sets.

That pattern makes additional local radius a weak next hypothesis. S4 instead
asks whether candidates need to be interpreted relative to sibling actions,
rank boundaries, and exact equivalence classes.

Earlier ADR 0096 work does not answer this question. It added context only
after a lossy 1,344-to-192 projection and correctly classified that projection
as insufficient. The present foundation operates on exact action records,
exact R3 afterstate identities, and the pre-output candidate surface.

## Decision

Run a source-frozen, open-data-only census before implementing a candidate-set
neural tournament.

For every retained train candidate and every complete validation candidate,
measure the exact relations:

- same public draft and prelude;
- same tile frontier;
- same tile pose;
- same wildlife destination;
- same draft plus tile pose, defining sibling placement plans; and
- byte-identical authoritative afterstate.

For observable screen-ranked anchor sets of 64, 128, and 256 candidates,
measure:

- selected-winner retention;
- R4800 confidence-set coverage;
- retained R4800 regret;
- relation coverage from every candidate to the anchors;
- winner and confidence-set linkage to anchors;
- relation edge counts, connected components, and isolated anchors; and
- dense-attention versus 8, 16, and 32 inducing-point pair-score budgets.

The census uses the ADR 0150 open-data authorization and exact R3 cache. It
does not train a model, read sealed test data, run gameplay, or alter the R3
classifier.

## Frozen Selection Rule

Choose the smallest anchor width satisfying all of:

- validation confidence-set coverage at least 99%;
- validation retained R4800 regret below 0.15;
- at least 98% of validation winners linked to another anchor by the union of
  exact relations;
- at least 95% of complete validation candidates linked to an anchor by the
  relation union; and
- no phase or action-width stratum with confidence-set coverage below 97%.

If 128 passes, S4 uses 128 anchors. Otherwise use 256 if it passes. If neither
passes, do not silently increase quadratic attention. Use adaptive anchors or
all-candidate inducing attention under a new decision.

The neural comparison following this foundation must retain an independent
candidate control and must distinguish candidate-set context from explicit
relation context.

## Cluster Execution

The 800 open groups are split by `row % 3` across john2, john3, and john4 while
john1 completes the R3 control. Every host processes disjoint rows from both
train and validation. The merge requires:

- exactly one report for every remainder;
- all 560 train and 240 validation rows exactly once;
- the same content-addressed cache and open-data proof;
- byte-identical forward and reverse merged reports; and
- no source outside the immutable bundle.

## Consequences

1. No S4 neural architecture is selected from intuition alone.
2. A top-128 or top-256 context is accepted only if the frozen oracle-retention
   and relation-coverage gates pass.
3. Equivalent-afterstate relations are exact hashes, not learned similarity.
4. Candidate context is introduced before the output heads and after exact
   candidate construction; ADR 0096's lossy pre-pool surface is not reused.
5. The R3 result remains independently classified under ADR 0150.

## Result

Launch one was invalidated because three non-semantic structured-action
padding bytes entered the exact `same_draft` and `same_sibling_plan` keys. Its
bundle and reports remain archived under `reports/invalid-launch-1`.

The corrected implementation canonicalizes every named action field in
zero-filled storage before forming byte keys. A regression test poisons the
padding bytes and proves they cannot affect relation equality. The corrected
bundle `69512ef62dd125d0231541a4bcf55cfeb861ef2eae79c75082cb90df72bdfc34`
was whole-tree verified on john2, john3, and john4. Its three shards covered
all 560 train and 240 validation groups exactly once, and forward and reverse
merges were byte-identical.

The 128-anchor surface preserved every validation confidence set at 0.05763
regret and linked every winner, but linked only 93.41% of complete candidates.
It therefore failed the frozen 95% complete-query linkage gate.

The 256-anchor surface achieved:

- 100.00% validation confidence-set coverage;
- 0.02874 mean retained R4800 regret;
- 100.00% winner linkage;
- 98.37% complete-candidate linkage; and
- 100.00% confidence coverage in every phase and action-width stratum.

The terminal classification is:

```text
s4_anchor_256_authorized
```

The first S4 neural comparison is bound to 256 anchors and 16 inducing
latents, with 128 retained as a serving-cost ablation. See
`docs/v2/reports/s4-candidate-relation-foundation-v1-result.md`.
