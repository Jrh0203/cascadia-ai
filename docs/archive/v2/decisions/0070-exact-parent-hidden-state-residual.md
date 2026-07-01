# ADR 0070: Exact-Parent Hidden-State Candidate Residual

Status: rejected on fresh validation on 2026-06-12. Test and gameplay domains
remain unopened.

## Context

ADR 0069 preserved the exact MLX parent score and added complete candidate-set
context. It improved validation distributional loss, top-five recall, and
teacher-frontier coverage, but failed top-one, MRR, broad pairwise order,
value-difference correlation, conditional regret, and train-capacity gates.
Train top-one improved by only 2.70 percentage points, ruling out a
validation-only overfit explanation.

The exact parent computes a 64-dimensional non-negative second hidden layer
before projecting it to one remaining-value scalar. Candidate distinctions
orthogonal to that final projection are unavailable to ADR 0069. Its public
entity/action encoder therefore had to reconstruct a representation the
qualified parent already computed. The next isolated question is whether the
parent's exact internal candidate representation contains the missing
decision-local ranking signal.

## Decision

1. Add a versioned, checksummed hidden-state sidecar aligned by source dataset
   identity, shard range, group ID, candidate index/count, and canonical action
   hash.
2. Reuse ADR 0069's strict deterministic replay: reconstruct every compact
   action, require its JSON BLAKE3 to match the source, byte-compare the public
   position, and advance only with the recorded selected action.
3. Extend the qualified exact-MLX service with one typed operation returning,
   for every sparse afterstate, the exact 64-value `h2` activation followed by
   the exact remaining-value output. The scalar must be bit-identical to the
   existing exact operation.
4. Store exact legacy immediate score, exact MLX remaining value, and all 64
   float32 hidden values per candidate. Reject non-finite values, incomplete
   groups, source drift, model drift, and partial shard alignment.
5. Add architecture `exact-parent-hidden-set-residual-v5`:
   - no public board, market, opponent, or hand-built action encoder;
   - candidate input is LayerNorm(`h2`) plus parent immediate, remaining,
     total, standardized total, and reciprocal within-group rank;
   - one 128-wide candidate projection;
   - masked candidate mean, maximum, and candidate-minus-mean context;
   - a 256-to-128 residual trunk with a zero-initialized scalar output;
   - final score is masked-standardized exact parent total plus the residual,
     exactly preserving parent order at initialization.
6. Use the unchanged ADR 0053 uncertainty-aware pairwise target and selected
   listwise auxiliary target. Add no return loss, architecture sweep, public
   feature branch, or parent-coefficient tuning.
7. Treat this as an information-ablation bridge. A success may authorize a
   fresh V2 architecture or policy-iteration successor, but a model depending
   on historical hidden tensors cannot be the final V2 solution.

## Frozen Experiment

- Train MCE evidence: immutable
  `imitation-targets-5d855c4ef1ee2edb`, split `train`, indices
  51,000-51,063, 64 games.
- Train hidden sidecar: derive once from those exact 64 source games.
- Validation MCE evidence: collect split `validation`, indices
  51,048-51,063, 16 fresh games.
- Validation hidden sidecar: derive once from those exact 16 source games.
- Test domain: split `test`, indices 51,016-51,031; sealed unless every
  validation gate passes.
- Gameplay domain: seeds 35,499 and 35,500-35,502; sealed unless every offline
  test gate passes.
- Teacher: deterministic K32/R600/LMR MCE evidence teacher and candidate
  contract from ADR 0053.
- Parent:
  `artifacts/models/legacy-nnue-v4opp-mlx-v1/model.json`, manifest BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.
- Model: candidate width 128, context statistics mean/max/delta, residual
  widths 256 and 128.
- Optimizer: AdamW, learning rate `5e-5`, weight decay `1e-4`.
- Training: group batch eight, seed 20260623, at most 30 epochs, validation
  patience six, checkpoint every 500 steps.
- Checkpoint selection: validation distributional loss only.
- Warm start, augmentation, alternate validation seeds, hidden-dimension
  selection, architecture sweep, retry, and post-hoc blending are prohibited.

Before training, report the untouched exact-parent baseline on both splits.
The selected checkpoint advances only if every validation gate passes:

- all source, target, hidden, replay, model, checkpoint, and resume integrity
  checks pass;
- 100% action-hash and group alignment and 100% MCE estimate retention;
- the hidden operation's scalar exactly matches the existing exact operation;
- validation distributional loss improves over the exact-parent baseline;
- selected-action top-one improves by at least 3 percentage points;
- selected-action top-five improves by at least 5 percentage points;
- selected-action MRR improves by at least 0.04;
- scored-action pairwise accuracy improves by at least 2 percentage points;
- scored value-difference correlation does not regress;
- conditional mean regret improves by at least 0.15 point;
- predicted teacher-frontier coverage does not regress;
- train selected-action top-one improves by at least 5 percentage points.

Passing validation authorizes one fresh 16-game test collection and hidden
derivation. Test must preserve every directional validation gate and may not
regress parent conditional mean regret or pairwise accuracy. Passing test
authorizes one gameplay smoke and three paired games in which the residual
reranks the full canonical expanded root before unchanged K32/R600 search.
Gameplay advancement requires at least +0.50 paired mean, treatment mean at
least 96.0, wildlife at least -0.50, habitat at least -0.50, Nature Tokens at
least -1.0, zero fallback, and clean local MLX shutdown.

## Maximum Compute

One exact-MLX hidden derivation for the immutable train corpus, one fresh
16-game R600 validation collection, one validation hidden derivation, and one
Apple-GPU run of at most 30 epochs. Conditional test and gameplay work is
limited to the gates above. No external compute, parameter sweep, second
training run, validation retry, threshold change, or use of sealed domains
after a failed gate is authorized.

## Implementation Qualification

`make imitation-parent-hidden-smoke` exercised the complete pipeline on one R2
implementation-only game in each arm at index 90,011:

- both action/evidence datasets replayed all 80 public positions exactly;
- all 7,680 actions per split reconstructed, JSON-hash matched, and aligned;
- all 2,399 train and 2,352 validation teacher estimates were retained;
- exact hidden sidecars covered all 7,680 actions in each split;
- each record stored 64 finite hidden activations plus exact immediate and
  remaining values in the checksummed 312-byte schema;
- the exact service ran on `Device(gpu, 0)` and shut down cleanly;
- the zero-initialized model reproduced exact parent ordering;
- one Apple-GPU epoch completed 10 optimizer steps in 0.187 seconds;
- selected checkpoint, integrity manifests, metrics, and gate report loaded
  successfully.

The one-game smoke reduced validation distributional loss from 1.572486 to
1.563827 but failed seven substantive gates. That is expected implementation
evidence only: R2 targets at an implementation-only index neither open nor
prejudge the frozen R600 validation domain.

## Result

The single authorized full run completed all 30 epochs and 19,200 optimizer
steps on `Device(gpu, 0)` in 157.688 seconds. Epoch 30 was selected by
validation distributional loss.

The hidden-state residual reduced validation loss from 1.522383 to 1.417843,
but selected-action top-one moved only from 21.641% to 21.719%. Top-five
recall improved by 0.703 percentage point, MRR by 0.002335, pairwise accuracy
by 0.405 percentage point, and conditional mean regret by 0.001245 point.
Train selected-action top-one was exactly unchanged at 20.840%. It failed six
frozen gates: validation top-one, top-five, MRR, pairwise accuracy, regret,
and train top-one.

All 491,520 train and 122,880 validation hidden records aligned exactly.
Both checkpoint pointers reload with checksum verification. Test indices
51,016-51,031 and gameplay seeds 35,499-35,502 were not opened.

The scalar projection was not the important information bottleneck. Even the
parent's exact 64-dimensional candidate representation learned calibration
without learning the teacher's selected action. This closes the bounded
legacy-representation branch: a successor must first establish that its
teacher labels identify stable action preferences, then train a fresh V2
policy/search representation rather than adding another residual to this
fixed historical network.
