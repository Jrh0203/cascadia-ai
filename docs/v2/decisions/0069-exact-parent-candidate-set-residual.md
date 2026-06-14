# ADR 0069: Exact-Parent Candidate-Set Residual

Status: rejected on fresh validation on 2026-06-12. Test and gameplay domains
remain unopened.

## Context

The exact MLX K32/R600 policy is the only independently reproduced local
policy above 95 mean, at 95.800 over fresh paired gameplay. Increasing its
rollout budget, retaining generic K64, and adding H6 root candidates did not
produce a qualifying improvement. Fine-tuning its sparse value network on
terminal returns improved value error but regressed root ordering, while the
joint return/ranking successor increased exact selected-action recall at the
cost of broader pairwise accuracy and conditional regret.

The full-frontier MCE corpus exposes a different representation defect. Its
96 retained actions contain every K32 teacher estimate, the pattern frontier,
immediate anchors, and deterministic legal negatives. Existing imitation
models score each action independently after broadcasting one shared state
summary. They cannot condition an action correction on the alternatives
available in the same decision. The first distributional apprentice also
discarded the exact NNUE prior and reached only 17.46% train top-one, so its
failure was underfit rather than validation overfit.

The untested mechanism is to preserve the exact evaluator's ordering and learn
only a permutation-equivariant, decision-local correction from the complete
candidate set.

## Decision

1. Add a versioned, checksummed parent-prior sidecar aligned by source dataset
   identity, shard range, group ID, candidate index/count, and canonical action
   hash.
2. Replay each retained game from its deterministic split seed. Reconstruct
   every compact action, require its JSON BLAKE3 to match the stored hash,
   require each recorded public position to equal the replayed state, and
   advance only with the recorded selected action.
3. For every retained action, translate the staged public state through the
   isolated historical bridge, construct the sparse afterstate, and evaluate
   it through the qualified exact Rust-order MLX operation. Store exact legacy
   immediate score and MLX remaining value separately.
4. Reject paid-wipe records because the compact v1 action schema intentionally
   stores only aggregate wipe metadata. The frozen source teacher used no paid
   prelude, so every substantive row must have zero wipe count, mask, and
   slots.
5. Add architecture `exact-parent-candidate-set-residual-v4`:
   - public board, market, opponent, global, and explicit-action encoders;
   - candidate embeddings compared through masked mean and maximum set
     summaries plus candidate-minus-mean features;
   - exact parent total and within-group parent rank as inference inputs;
   - a zero-initialized residual head;
   - final ranking score equal to masked-standardized exact parent total plus
     the learned residual, making initialization exactly parent-order
     preserving.
6. Train with the ADR 0053 uncertainty-aware pairwise target and selected
   listwise auxiliary target. Add no absolute return loss.
7. Keep the exact MLX parent frozen. This experiment is a research bridge and
   cannot become the final V2 model while it depends on historical tensors.
   A successful result would authorize a separately registered self-play or
   distillation successor.

## Frozen Experiment

- Train MCE evidence: immutable
  `imitation-targets-5d855c4ef1ee2edb`, split `train`, indices
  51,000-51,063, 64 games.
- Train parent sidecar: derive once from those exact 64 source games.
- Validation MCE evidence: collect split `validation`, indices
  51,032-51,047, 16 fresh games.
- Validation parent sidecar: derive once from those exact 16 source games.
- Test domain: split `test`, indices 51,000-51,015; sealed unless every
  validation gate passes.
- Gameplay domain: seeds 35,299 and 35,300-35,302; sealed unless every
  offline test gate passes.
- Teacher: deterministic K32/R600/LMR MCE evidence teacher, exact candidate
  contract from ADR 0053.
- Parent:
  `artifacts/models/legacy-nnue-v4opp-mlx-v1/model.json`, manifest BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.
- Model: hidden 192, eight heads, three board blocks, one market block,
  feed-forward multiplier three.
- Optimizer: AdamW, learning rate `5e-5`, weight decay `1e-4`.
- Training: group batch eight, seed 20260622, at most 30 epochs, validation
  patience six, checkpoint every 500 steps.
- Checkpoint selection: validation distributional loss only.
- Warm start, augmentation, parent coefficient tuning, architecture sweep,
  alternate validation seeds, and post-hoc blending are prohibited.

Before training, report the untouched exact-parent baseline on both splits.
The selected checkpoint advances only if every validation gate passes:

- all source, target, prior, replay, model, checkpoint, and resume integrity
  checks pass;
- 100% action-hash and group alignment and 100% MCE estimate retention;
- validation distributional loss improves over the exact-parent baseline;
- selected-action top-one improves by at least 3 percentage points;
- selected-action top-five improves by at least 5 percentage points;
- selected-action MRR improves by at least 0.04;
- scored-action pairwise accuracy improves by at least 2 percentage points;
- scored value-difference correlation does not regress;
- conditional mean regret improves by at least 0.15 point;
- predicted teacher-frontier coverage does not regress;
- train selected-action top-one improves by at least 5 percentage points,
  guarding against another capacity-underfit result.

Passing validation authorizes one fresh 16-game test collection and prior
derivation. Test must preserve every directional validation gate and may not
regress parent conditional mean regret or pairwise accuracy. Passing test
authorizes one gameplay smoke and three paired games in which the residual
reranks the full canonical expanded root before the unchanged K32/R600
search. Gameplay advancement requires at least +0.50 paired mean, treatment
mean at least 96.0, wildlife at least -0.50, habitat at least -0.50, Nature
Tokens at least -1.0, zero fallback, and clean local MLX shutdown.

## Maximum Compute

One exact-MLX prior derivation for the immutable train corpus, one fresh
16-game R600 validation collection, one validation prior derivation, and one
Apple-GPU run of at most 30 epochs. Conditional test and gameplay work is
limited to the gates above. No external compute, parameter sweep, second
training run, validation retry, threshold change, or use of the sealed
domains after a failed gate is authorized.

## Implementation Qualification

`make imitation-parent-residual-smoke` exercised the complete pipeline on one
R2 implementation-only game in each arm at index 90,010:

- both action/evidence datasets replayed all 80 public positions exactly;
- all 7,680 actions per split reconstructed, JSON-hash matched, and aligned;
- all 2,396 train and 2,160 validation teacher estimates were retained;
- exact MLX parent priors covered all 7,680 actions in each split;
- the prior service started on `Device(gpu, 0)` and shut down cleanly;
- the zero-initialized model reproduced exact parent ordering before training;
- one Apple-GPU epoch completed 10 optimizer steps in 0.410 seconds;
- the selected checkpoint, integrity manifests, metrics, and gate report were
  written successfully.

The one-game smoke improved validation distributional loss from 1.557319 to
1.540530 and top-five recall from 0.3625 to 0.4125, but failed six substantive
gates. That rejection is expected implementation evidence only: the smoke
uses R2 targets and an implementation-only index, so it neither opens nor
prejudges the frozen R600 validation domain.

## Result

The immutable 64-game train sidecar contains 5,120 groups and 491,520 parent
priors. Fresh validation indices 51,032-51,047 produced 1,280 groups, 122,859
candidates, and 38,457/38,457 aligned R600 teacher estimates in 2,369.8
seconds. One late-game group contained 75 legal retained actions; 96 is the
frontier cap, not a fixed padding requirement. The sidecar preserved the
variable count exactly.

Training stopped after epoch 9 and 5,760 optimizer steps because six
consecutive epochs failed to improve validation loss. Epoch 3 was selected:

| Validation metric | Exact parent | Selected | Gate | Result |
|---|---:|---:|---:|---|
| Distributional loss | 1.528932 | 1.397396 | improve | pass |
| Selected top-one | 21.016% | 22.891% | +3 pp | fail |
| Selected top-five | 49.453% | 54.688% | +5 pp | pass |
| MRR | 0.347279 | 0.377889 | +0.04 | fail |
| Scored pairwise | 70.918% | 70.162% | +2 pp | fail |
| Value-difference correlation | 0.588722 | 0.573128 | non-regress | fail |
| Conditional regret | 0.747493 | 0.737381 | improve 0.15 | fail |
| Teacher coverage | 73.125% | 76.953% | non-regress | pass |
| Train selected top-one | 20.840% | 23.535% | +5 pp | fail |

The model concentrated more teacher winners into its top five but did not
improve broad value ordering or costly mistakes. Train top-one improved by
only 2.70 points, so this is representation underfit rather than
validation-only overfit. Candidate-set context plus scalar parent score/rank
is closed. A successor must add information, such as the exact parent's
internal candidate representation, rather than tune this architecture, loss,
seed, or threshold. No test collection, promotion, or gameplay run is
authorized.
