# ADR 0098: Frontier Raw Factor Construction

Status: complete; rejected as `raw_factor_construction_insufficient`.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-raw-factor-construction-v1`

## Context

ADR 0097 tested four integration mechanisms over the selected ranker's exact
seven pre-compression factors. Train target recall remained between 29.39% and
30.88%, with at most one of 560 target sets recovered exactly. The
classification `candidate_factor_inputs_insufficient` closes further heads,
pooling, width, and interactions over those projected factors.

The complete-action dataset still exposes substantially richer lossless public
state:

- four 23-cell boards with masks;
- the current market, public supply, and 96 global features;
- 140 action features and eight observable screen priors per candidate; and
- each candidate's staged market and public supply after its draft action.

Exact observable collisions were zero in ADR 0090, so the target is
information-theoretically identifiable from these inputs. The next question is
which raw candidate-state relation construction, if any, can materially fit
the open frontier target before the current 192-dimensional factors exist.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Dataset manifest BLAKE3 identities:
  - train:
    `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`;
  - validation:
    `302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31`.
- The unchanged ADR 0089 frontier target, group-balanced binary objective,
  frontier-anchored width-64 selector, stable action-hash tie breaking, and
  open metrics.
- Raw model inputs contain no rollout target, selected winner, source flag, or
  sealed-domain value. Observable screen priors remain allowed exactly as
  decoded by the dataset.

No feature cache is authorized. Every arm streams the existing 641 MiB grouped
dataset directly so the experiment does not duplicate ADR 0097's 15 GiB cache.
The sealed test split, gameplay, new teacher compute, cloud, and external
compute remain prohibited.

## Shared Set Scorer

Each arm constructs one 384-dimensional vector per legal candidate directly
from raw observables. The unchanged shared scorer concatenates:

1. the candidate vector;
2. complete-set mean;
3. complete-set maximum;
4. observable screen-top64 mean;
5. observable screen-top64 maximum;
6. candidate minus screen-top64 mean; and
7. candidate minus screen-top64 maximum.

It applies `2688 -> 768 -> 384 -> 1` with GELU and LayerNorm. Complete-set and
screen-top64 context are identical across arms so the experiment localizes
factor construction rather than testing another post-factor integration fork.
Padding never contributes to construction, pooling, loss, or selection.

## Frozen Arms

All arms train from scratch for exactly 20 epochs with AdamW, learning rate
`3e-4`, weight decay `1e-4`, the same group-balanced target loss, and no data
augmentation.

### Complete Raw Flat

- Host: john2.
- Seed: `2026061621`.
- Flatten the masked parent boards, board masks, current market and mask,
  globals, and public supply once per group.
- Concatenate that lossless parent row to each candidate's action, observable
  priors, staged market and mask, and staged public supply.
- Construction:
  `3504 -> 1024 -> 512 -> 384`, with GELU and LayerNorm.
- Question: can an unrestricted dense constructor recover the target when no
  hand-designed 192-dimensional factor is imposed?

### Exact Local Board Relation

- Host: john3.
- Seed: `2026061622`.
- Reuse the rotation-canonical 13-relation geometry primitive from ADR 0088:
  six neighbors around the placed tile, the wildlife target cell, and six
  neighbors around that cell.
- Concatenate all 390 exact local relation values with the candidate action,
  priors, staged market, staged mask, and staged supply.
- Independently project the complete raw parent. Combine candidate, parent,
  elementwise product, and absolute difference before producing 384 values.
- Question: does direct target-only learning from exact candidate-to-board
  relations succeed where ADR 0088's additive end-to-end correction did not?

### Explicit Market Transition

- Host: john4.
- Seed: `2026061623`.
- Align the four current and staged market slots and construct raw before,
  after, difference, and elementwise-product rows plus both masks.
- Construct analogous before, after, difference, and product public-supply
  rows, then append action and observable prior features.
- Combine the transition vector with a fresh complete-parent projection using
  candidate, parent, product, and absolute difference.
- Question: is the missing signal an explicit representation of what the draft
  removes, leaves, and exposes to opponents?

### Fresh Candidate-Conditioned Entity Cross-Attention

- Host: john1.
- Seed: `2026061624`.
- Train fresh 192-dimensional board and market entity encoders from the target;
  no ADR 0089 weights or factors are loaded.
- Build each query from raw action, priors, staged-market summary, staged
  supply, and a fresh parent summary.
- Cross-attend each query to all 92 masked board cells and its four staged
  market entities, then construct a 384-dimensional candidate vector from the
  query, both cross-attention outputs, and parent summary.
- Question: can target-supervised candidate-state attention learn relations
  that the old multi-objective factor encoder discarded?

## Classification Gates

Every arm must score every open action once per evaluation, produce finite
scores, stay below 6 GiB peak active MLX memory and 6 GiB peak process RSS, use
zero process swaps, clear retained MLX cache at phase boundaries, and keep all
sealed domains unopened.

An arm materially fits train only if:

- train target-positive recall is at least 80%; and
- train exact target-set recovery is at least 25%.

A fitting arm transfers only if:

- validation target-positive recall is at least 50%; and
- validation exact target-set recovery is at least 1%.

Classify as follows:

1. `raw_factor_construction_sufficient` if any arm passes both gates;
2. `raw_factor_construction_train_separable_not_generalized` if at least one
   arm passes train but none passes validation; or
3. `raw_factor_construction_insufficient` if every train gate fails.

If multiple arms pass, select by validation target recall, validation exact
sets, train target recall, lower measured inference memory, then fixed order:
market transition, local board relation, complete raw flat, fresh entity
cross-attention. A pass authorizes one end-to-end successor using only the
selected construction.

## Correctness Gates

- Full Python suite and Ruff pass before real training.
- Synthetic fixtures prove all four shapes, finite forward/backward execution,
  padding exclusion, target loss, save/load, and candidate-permutation
  equivariance.
- Exact local geometry and stable screen-top64 behavior retain their existing
  unit coverage.
- A true 10,854-action maximum-width forward/backward audit passes for all four
  architectures with the frozen 512 MiB MLX cache limit before launch.
- All four Macs use one byte-identical source bundle and matching dataset
  manifests.
- Every training artifact is reproduced bit-for-bit by the next host in the
  ring using saved weights and direct dataset evaluation.

## Cluster Execution

- john1: fresh entity cross-attention, expected critical-path arm.
- john2: complete raw flat.
- john3: exact local board relations.
- john4: explicit market transition.
- One distinct arm runs per Mac under the host lock and `caffeinate`.
- Ring replay is john1 to john2, john2 to john3, john3 to john4, and john4 to
  john1.
- Replays launch as soon as their incoming weights and destination host are
  available; they need not wait for the longest training arm.
- No same-arm replica, duplicate seed, sweep, or idle mirror training is
  authorized.

Reports include assigned and productive wall time, dependency-blocked idle,
idle with compatible work queued, candidates per second, peak memory, swaps,
and scientific digests. The governing metric is trustworthy factor-construction
hypotheses resolved per cluster wall-clock hour.

## Stop Rule

Run exactly 20 epochs per arm. Do not resize, continue, reseed, alter learning
rate, add features, or change an architecture after metrics are visible.
Classify only after all four reports and ring replays pass integrity checks.

## Maximum Compute

Four one-seed target probes, four deterministic ring replays, one maximum-width
audit per architecture on john1 plus a cross-host audit on john4, tests, source
identity checks, and reporting. No teacher rollout, new label, sealed test,
gameplay, cloud, external compute, extra epoch, or architecture sweep is
authorized.

## Consequences

- A passing construction authorizes one end-to-end complete-action ranker
  treatment with the selected raw relation path.
- Train-only success localizes the next problem to generalization or data
  coverage rather than representation.
- If every train gate fails, the next experiment must audit target
  learnability and supervision structure directly; another neural constructor,
  head, pool, width increase, or optimizer variation is prohibited.
- No result directly authorizes gameplay or champion promotion.

## Result

All four frozen arms completed exactly 20 epochs and failed the train-fit
gate:

| Construction | Train recall | Train exact | Validation recall | Validation exact |
|---|---:|---:|---:|---:|
| complete raw flat | 30.29% | 0.00% | 25.18% | 0.00% |
| exact local relation | 37.87% | 0.00% | 21.26% | 0.00% |
| explicit market transition | 29.94% | 0.00% | 24.35% | 0.00% |
| fresh entity cross-attention | 17.95% | 0.00% | 15.56% | 0.00% |

The exact-local-relation arm fit the open train target best, but it remained
42.13 percentage points below the 80% train-recall gate and recovered none of
560 train target sets exactly. No arm approached the 25% exact-train-set gate,
so validation transfer cannot authorize any construction.

Every arm scored all 2,135,111 train and 860,203 validation candidates exactly
once with finite values. All eight true 10,854-action maximum-width audits
passed on john1 and john4. Peak active MLX memory was 2.97 GB, peak process RSS
was 680 MB, allocator caches cleared to zero, and no process or system swap was
consumed. The 98-file MLX source bundle and both dataset identities matched
across all four Macs, and every ring replay reproduced its origin metrics
bit-for-bit.

The four distinct hypotheses plus their replays completed in 2,412.43 seconds
of cluster wall time, or 5.97 resolved hypotheses per hour, with zero duplicate
discovery compute. The sealed test, gameplay, new teacher compute, cloud, and
external compute remained closed.

The preregistered classification is
`raw_factor_construction_insufficient`. Complete raw flattening, explicit
local geometry, explicit market transitions, fresh entity cross-attention,
the seven prior projected factors, post-factor integration, output heads,
pooling variants, width increases, and optimizer-only changes are now closed
for this hard binary target. The next experiment must audit whether the
finite-R1200 top-64 membership labels are statistically stable and whether a
different supervision structure has a sufficient deterministic ceiling before
another neural training run.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-raw-factor-construction-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-raw-factor-construction-v1-rejection.md`.
