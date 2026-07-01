# ADR 0049: Full-Legal Explicit-Action Imitation

Status: rejected offline on 2026-06-12. The substantive train, validation,
and test splits were opened exactly once; no gameplay seed was opened.

## Context

ADR 0048 found that the production K8+H6+B8 frontier contains only 51.25% of
the qualified 96.35 teacher's selected actions. Injecting missing teacher
actions only during training would create an apprentice that cannot express
its target at inference. Expanding a hand-selected frontier until it fits two
games would preserve the same proposal bias.

V2 already has a canonical complete legal-action generator, a typed
Rust-to-MLX protocol, checksummed resumable datasets, grouped listwise
training, exact checkpoint resumption, and validation-only checkpoint
selection. The missing capability is direct selected-action imitation over a
broad sample with complete legal inference.

## Decision

Build `canonical-action-imitation-v1` as a shared-state/action MLX ranker:

1. Follow the unchanged qualified K32/R600 legacy teacher trajectory.
2. At every decision enumerate the complete canonical legal set under the
   same free-overflow, no-paid-wipe prelude.
3. Build a deterministic 64-action training group containing the exact
   selected action, the complete K8+H6+B8 pattern frontier, the top 16 exact
   immediate-score actions, and BLAKE3-ordered legal negatives.
4. Store the observable pre-action state once per decision, followed by
   compact explicit action rows. Each shard is checksummed, provenance-bound,
   one game, and resumable.
5. Encode the board, market, opponents, and global state once per group in
   MLX. Broadcast that state representation across lightweight action
   embeddings and train exact one-hot listwise cross-entropy.
6. At gameplay inference enumerate and score every canonical legal action in
   one shared-state request. Candidate recall is therefore 100% by
   construction.
7. Permit standalone promotion only after the untouched test gates pass.
   Promoted weights and manifests are integrity checked before inference.

Training groups are sampled for tractable teacher storage; production
inference is not restricted to those 64 actions. The selected action may not
be repaired, reinterpreted, or replaced.

## Disposable Prototypes

Game index 90,000 is reserved for implementation smoke tests and does not
open the substantive split.

The first afterstate-per-action prototype was rejected before substantive
collection. It re-encoded nearly identical boards for every legal action and
stored the state in every candidate record. A hidden-32 one-game smoke took
21.878 seconds per game, 273 ms per decision on average, and 699 ms maximum.

The replacement shared-state protocol scored every canonical legal action in
3.966 seconds per game, with 49.56 ms mean, 95.24 ms P90, and 376.50 ms
maximum decision latency. The tiny one-epoch GPU smoke improved validation
loss from 4.1596 to 4.0539, top-one accuracy from 2.5% to 16.25%, top-five
recall from 7.5% to 33.75%, and MRR to 0.2591. These are wiring measurements,
not strength evidence.

The final grouped shard stores one 880-byte state header and 68 bytes per
candidate. One 80-decision, 5,120-candidate game is 418,672 bytes rather than
4,874,352 bytes, an 11.64x reduction. Rust and Python independently validate
the layout and the framed inference protocol.

## Frozen First Experiment

- Teacher: unchanged ADR 0047 K32/R600/LMR policy and weights.
- Candidate group: 64, pattern frontier all, immediate top 16, deterministic
  hashed remainder.
- Train: 64 games, train split indices 50,000-50,063.
- Validation: 16 games, validation split indices 50,000-50,015.
- Untouched test: 16 games, test split indices 50,000-50,015.
- Shards: one game each for interruption-safe local collection.
- Model: `shared-state-action-imitation-v1`, hidden 96, four heads, two board
  blocks, one market block, feed-forward multiplier three.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Training: at most 20 epochs, batch 16 groups, patience five, seed 20260612.

Offline advancement requires:

- selected checkpoint improves validation listwise loss over initialization;
- untouched-test top-one accuracy at least 20%;
- untouched-test top-five recall at least 55%;
- untouched-test mean reciprocal rank at least 0.40;
- exact dataset, checkpoint, and promoted-artifact integrity.

Only after every offline gate passes may the promoted model play seeds
32700-32709 against promoted pattern-aware. A ten-game pilot advances only
with paired gain at least +0.25, no habitat or aggregate wildlife loss worse
than -0.50, and at most 10 seconds per game. A passing pilot authorizes a
separately registered 50-game confirmation on untouched seeds.

## Rejection Rules

- Any missing selected action, illegal record, source drift, checksum error,
  split overlap, non-MLX training path, or resume mismatch rejects the run.
- Missing any offline gate stops before promotion and gameplay.
- Missing any gameplay gate stops before confirmation.
- Narrowing production inference back to a sampled frontier is not an allowed
  optimization of this experiment.

## Result

Collection completed without repair, resume, source drift, or integrity
failure:

- Train: 64 games, 5,120 groups, 327,680 candidates, dataset
  `canonical-action-imitation-train-a0155b3613e51112`.
- Validation: 16 games, 1,280 groups, 81,920 candidates, dataset
  `canonical-action-imitation-validation-4929d2a8a2bb0a0d`.
- Test: 16 games, 1,280 groups, 81,920 candidates, dataset
  `canonical-action-imitation-test-e28d0c294d82e788`.
- Train plus validation collection took 11,713.6 seconds. Untouched-test
  collection took 2,364.6 seconds.
- Every split used source BLAKE3
  `f0481f6721ae9b392734df95c4f7b945c9facf390861abeefd6f6e1f9fec8775`,
  executable BLAKE3
  `ab7749c30496460664590276c470c6f2f61727ff91b7e57aff2107cb512e8174`,
  and teacher-weight BLAKE3
  `9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400`.

MLX training ran on the Apple GPU for eight epochs and stopped after five
non-improving validation epochs. The selected epoch-three checkpoint reduced
validation listwise loss from 4.158443 to 2.948818. On the untouched test
split it achieved:

- top-one accuracy 20.078%, passing the 20% gate;
- top-five recall 51.094%, failing the 55% gate;
- mean reciprocal rank 0.347386, failing the 0.40 gate;
- pairwise accuracy 88.084%;
- mean rank correlation 0.431597.

The experiment is rejected before promotion and gameplay, as preregistered.
The apprentice learned broad legal-action ordering and barely cleared exact
winner recall, but did not concentrate the selected teacher action near the
top reliably enough to replace search.

The v1 artifact records one-hot selected-action identity and immediate
ordering, but not per-candidate teacher rollout estimates, uncertainty, or
candidate-source tags. It therefore cannot determine whether top-rank misses
are decisive policy errors or Monte Carlo near-ties. That missing evidence is
a dataset-design constraint for the successor, not grounds to reinterpret
this result. The sealed test split is retained only as evidence for this
rejected protocol and may not be used to select its successor.
