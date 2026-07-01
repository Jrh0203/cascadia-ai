# ADR 0094: Frontier Frozen-Embedding Separability

Status: active preregistered; sealed test and gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-embedding-separability-v1`

## Context

ADR 0089 underfit the target on train and validation. ADRs 0091, 0092, and
0093 then tested uniform set cross entropy, smooth extreme-value boundary
loss, and full rank-matched boundary loss. Every target is reachable inside
the model's ±12 score range, and both boundary objectives recover all audited
target sets when scores are free variables. Yet all three neural treatments
selected the untouched warm start and reduced deployed recall while their
training losses fell.

The unresolved question is whether the selected model's frozen candidate
representation contains the target signal. Another full-network loss or
learning-rate treatment is not justified until this is measured directly.

## Hypothesis

The exact 192-dimensional candidate vector immediately before the residual
and uncertainty heads contains enough information for a small probe to recover
the nonfrontier target set. Probe capacity identifies the smallest justified
successor:

- a linear fit authorizes output-head or optimizer-scope work;
- a nonlinear-only fit authorizes a richer output head with the trunk frozen;
- failure to fit train authorizes a trunk representation or capacity change.

## Frozen Caches

- Source model: exact selected ADR 0089 john2 checkpoint
  `step-000003592-epoch-0008-batch-000000`.
- Embedding point: output of `output_trunk`, before both learned heads.
- Dtype: float32; no quantization.
- Include every train and validation action exactly once.
- Preserve group offsets, target mask, source flags, action hashes, selected
  winner, and R4800 mean/mask metadata.
- john2 exports the 560-group train cache.
- john3 exports the 240-group validation cache.
- The two caches are exchanged once; no host repeats neural trunk inference.
- Cache manifests include all file hashes, dataset identity, checkpoint
  identity, group/action totals, RSS, swaps, and sealed-domain status.

The extraction refactor must pass the complete Python suite before cache
generation. john1 and john4 independently run the widest open group through
the original model and through both original heads applied to the exported
embedding. The residual and uncertainty outputs must be bit-identical, all
10,854 embeddings finite, RSS below 4 GiB, and process/system swap delta zero.

## Frozen Probes

Both probes use group-balanced binary cross entropy: mean positive loss plus
mean eligible-nontarget loss within each group, then equal weight across
groups. Probe logits rank eligible nonfrontier actions; the anchored frontier
and width-64 selector remain unchanged.

### Linear

- Host: john2.
- Seed: `2026061608`.
- Architecture: `192 -> 1`.
- AdamW, learning rate `1e-3`, weight decay `1e-4`.
- 20 epochs.

### Nonlinear

- Host: john3.
- Seed: `2026061609`.
- Architecture: `192 -> 128 -> 1` with GELU and LayerNorm.
- AdamW, learning rate `3e-4`, weight decay `1e-4`.
- 20 epochs.

For both probes, checkpoint selection uses train target recall, then train
exact target sets, then validation target recall. Validation is reported but
does not override a weaker train fit.

## Classification Gates

1. `linear_head_or_optimizer_scope_sufficient`
   - linear train target recall at least 60%; and
   - linear exact train sets at least 5%.
2. `nonlinear_head_capacity_sufficient`
   - linear gate fails;
   - nonlinear train target recall at least 80%;
   - nonlinear exact train sets at least 25%;
   - validation target recall at least 50%; and
   - validation exact target sets at least 1%.
3. `frozen_representation_train_separable_not_generalized`
   - nonlinear train gates pass but validation transfer gates fail.
4. `frozen_representation_insufficient`
   - nonlinear train gates fail.

All reports additionally require complete finite scoring, correct group/action
totals, RSS below 4 GiB, zero process swaps, and sealed test unopened.

## Cluster Use

- john1 owns protocol, source identity, aggregation, and final classification.
- john2 exports train embeddings, then trains the linear probe.
- john3 exports validation embeddings, then trains the nonlinear probe.
- john4 independently repeats the maximum-width embedding reconstruction,
  then cross-evaluates both saved probes on both caches and verifies
  bit-identical scientific payloads.

Cache generation, cache exchange, and the two probe jobs are staged to keep
independent useful work concurrent without duplicating trunk inference or
training.

## Maximum Compute

One exact cache per open split, one linear probe, one nonlinear probe, one
cross-host replay of each, and correctness/reporting work. No full-network
training, probe sweep, extra seed, new teacher compute, sealed test, gameplay,
cloud, or external compute.

## Result

Both preregistered probes failed their train-fit gates. The linear probe
reached 22.48% train target recall and 0% exact sets. The nonlinear probe
reached 24.67% train target recall and 0% exact sets. Validation reached
17.28% and 19.91% recall respectively, also with zero exact sets.

john1 and john4 reconstructed both original heads bit-for-bit on the
10,854-action maximum-width group. john4 then reproduced both saved probes'
train and validation metrics exactly from the transferred caches. All four
hosts used the same 94-file MLX source bundle; every action was scored once,
maximum RSS remained below 4 GiB, process swaps were zero, and the sealed
domains remained unopened.

The frozen classification is `frozen_representation_insufficient`. Output-head
and optimizer-scope work are closed. The next authorized mechanism is a
frozen-trunk raw-observable bypass separability audit that reuses these
embeddings and adds only a compact action/prior sidecar.
