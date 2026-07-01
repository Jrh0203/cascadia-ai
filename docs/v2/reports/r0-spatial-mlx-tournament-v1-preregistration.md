# R0 Spatial MLX Tournament V1 Preregistration

Date: 2026-06-17

ADR: 0142

Experiment: `r0-spatial-mlx-tournament-v1`

Status: complete - H0 retained; no compact arm selected

## Research Question

On the exact same 60,000 open `PositionRecord` rows, targets, D6 transforms,
model parameters, initialization seed, optimizer, batch schedule, and training
steps, which lossless spatial representation gives the best learned value
quality and local MLX throughput?

The purpose is to isolate representation. This experiment does not test a new
search algorithm, a larger model, a different target, or gameplay strength.

## Hypotheses

### H0

Once exact overflow is retained and architecture is controlled, the bounded
radius arms do not improve same-host MLX throughput enough to justify their
larger masked tensors relative to the 23-row exact entity control.

### H1

At least one of radius 6, radius 5, or radius 4 remains value-noninferior to the
exact control and passes either:

- 1.5 times same-host inference throughput; or
- 1.3 times same-host forward-plus-backward throughput.

### Diagnostic

Historical 441 quantifies the cost of retaining the old fixed square. It is not
eligible for selection, even if its value metrics happen to be favorable.

## Frozen Evidence Domain

Only the ADR 0138 open corpus is admissible:

```text
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-1
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-2
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-3
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-0
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-1
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-2
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-3
```

Verified before preregistration:

| Field | Frozen value |
|---|---|
| Train rows | 50,000 |
| Validation rows | 10,000 |
| Total rows | 60,000 |
| Train game interval | 200000 through 200624 |
| Validation game interval | 210000 through 210124 |
| Feature schema | `compact-entity-v2` |
| Target schema | `base-score-components-v1` |
| Strategy | `pattern-aware-v1-k8-h6-b8-m4` |
| Players | 4 |
| Wildlife cards | AAAAA |
| Habitat bonuses | false |
| Source V2 BLAKE3 | `78ec63415e342b4820b89ee5bc7acea32db39af652bf48ba43009e6e7489ae6b` |
| Corpus manifest-sequence BLAKE3 | `0f0891e5cb618bacb869123d882186af094de865d44e0a09283bbe48c8f05103` |
| Expected corpus lock ID | `6e6869d825b3a5ee4dda41f26245f40303174c2a215887ede4ebe153f20c6d43` |

The production freeze command must reproduce these values. Any drift blocks
authorization. Test and final data are prohibited.

## Arms

The canonical order is:

| Label | CLI ID | MLX rows per board | Role |
|---|---|---:|---|
| R0-A | `exact-entity-control` | 23 | scientific control |
| R0-B | `hex-radius-6-127` | 150 | compact candidate |
| R0-C | `hex-radius-5-91` | 114 | compact candidate |
| R0-D | `hex-radius-4-61` | 84 | aggressive compact candidate |
| R0-H | `historical-square-21x21-441` | 464 | diagnostic only |

The bounded shapes include a 23-row reserve for exact overflow entities.
Padding is masked, all zero, and verified before MLX arrays are created.

Every arm consumes identical nonspatial features and unchanged 11-component
targets. All semantic extraction and D6 transforms are produced by Rust.

## Frozen Model And Training

```text
architecture: r0-spatial-iso-set-value-v1
parameters: 74,635
hidden width: 32
attention heads: 4
attention blocks: 1
feed-forward multiplier: 2
seed: 2026061701
optimizer: AdamW
steps: 500
batch size: 32
learning rate: 0.0003
weight decay: 0.0001
checkpoint interval: 100
metric interval: 25
evaluation batch: 64
D6 sampling: uniform over Rust transform IDs 0..11 per sampled row
```

There are 16,000 sampled optimizer examples per arm. This is a small controlled
screen, not a final model training budget.

The training sample and transform at optimizer step `s` are derived from the
frozen seed and `s`. Interrupted runs resume the identical sequence.

## Measurements

### Integrity

- cache content address;
- corpus-lock equality;
- every file BLAKE3;
- Rust source semantic BLAKE3;
- Rust transformed semantic BLAKE3;
- target BLAKE3;
- exact and packed round trips;
- inverse D6 round trips;
- active, padding, and overflow row accounting;
- duplicate and out-of-range destination checks;
- nonzero padding checks; and
- confirmation that no test or final data was opened.

### Learning

For train and validation:

- loss;
- per-component MAE;
- per-component RMSE;
- per-component bias;
- mean component MAE;
- total-score MAE;
- total-score RMSE;
- total-score bias;
- total-score correlation;
- calibration slope; and
- calibration intercept.

Every train row and every validation row is evaluated exactly once with D6
identity transform 0.

### Performance

- first compiled invocation;
- warmup examples per second;
- steady examples per second;
- inference actions per second;
- P50, P90, and P99 latency;
- inference active, cache, and peak MLX memory;
- process peak RSS;
- cumulative optimizer examples per second;
- forward-plus-backward examples per second; and
- forward-plus-backward peak MLX memory.

Each arm also packs the same active rows into the 23-token exact-control shape
on the same host and repeats inference and gradient-step measurements. Only
these within-host ratios are eligible for the representation leverage gate.

## Stage 2 Gates

Structural completion requires exactly five reports and exact equality of:

- experiment, ADR, protocol, authorization, and corpus lock;
- complete MLX source digest;
- Python and MLX versions;
- Apple Silicon architecture and MLX GPU device kind;
- model configuration and parameter count;
- source semantic, D6 semantic, and target digests;
- optimizer constants and step count; and
- full train and validation evaluation coverage.

A compact arm is Stage 2 value-noninferior when:

```text
compact.validation.total_mae - exact.validation.total_mae <= 1.0
compact.validation.total_rmse - exact.validation.total_rmse <= 1.5
compact.validation.mean_component_mae
    - exact.validation.mean_component_mae <= 0.25
```

It passes the leverage gate when:

```text
same_host_inference_examples_per_second_ratio >= 1.5
or
same_host_gradient_examples_per_second_ratio >= 1.3
```

A Stage 2 candidate must pass both value noninferiority and leverage.

If multiple compact arms pass, selection order is:

1. fewer MLX rows per board;
2. lower validation total MAE;
3. higher same-host inference ratio; and
4. canonical arm ID.

Historical 441 is excluded before selection.

## Classification

Fail-closed precedence:

1. semantic failure;
2. structural incompleteness;
3. insufficient performance evidence;
4. complete.

Malformed JSON, missing arms, duplicate arms, nonfinite metrics, zero timing,
shape drift, runtime drift, protocol drift, source drift, and inconsistent
semantic hashes are not repaired or inferred.

The classifier is run in forward and reverse input order. Outputs must be
byte-identical.

## Cluster Allocation

Primary allocation:

| Host | Arm |
|---|---|
| john1 | exact control |
| john2 | radius 6 |
| john3 | radius 5 |
| john4 | radius 4 |

Historical 441 becomes ready only after all four preflights. It has lower
priority than every primary arm and is compatible with all four hosts. The
first released host backfills it.

This arrangement maximizes concurrent decision-changing work while keeping the
diagnostic from occupying a host that could start a primary arm.

## Immutable Bundle Inputs

The production bundle must include:

- the release `r0_spatial_mlx_export` binary;
- all workspace and lock files hashed by `cascadia-provenance`;
- all V2 and required legacy Rust source roots;
- the complete `python/cascadia_mlx` package;
- the R0 MLX campaign and classifier tools;
- the queue library; and
- the immutable bundle validator.

Bundle validation and whole-tree fanout occur before any host preflight.

## Production Entry Points

### 1. Build the exporter

```bash
cargo build --release -p cascadia-data --bin r0_spatial_mlx_export
```

### 2. Freeze the corpus

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/r0_spatial_mlx_campaign.py freeze-corpus \
  --output artifacts/experiments/r0-spatial-mlx-tournament-v1/control/corpus-lock.json
```

### 3. Build the immutable source and binary bundle

```bash
PYTHONPATH=python:tools .venv/bin/python tools/rust_experiment_bundle.py \
  --repository . \
  --experiment-id r0-spatial-mlx-tournament-v1 \
  --include CASCADIA_V2_GOAL.txt \
  --include Cargo.lock \
  --include Cargo.toml \
  --include Makefile \
  --include pyproject.toml \
  --include uv.lock \
  --include apps/web/src \
  --include crates/cascadia-api \
  --include crates/cascadia-cli-v2 \
  --include crates/cascadia-data \
  --include crates/cascadia-differential \
  --include crates/cascadia-eval \
  --include crates/cascadia-game \
  --include crates/cascadia-model \
  --include crates/cascadia-provenance \
  --include crates/cascadia-search \
  --include crates/cascadia-sim \
  --include legacy/crates/cascadia-ai \
  --include legacy/crates/cascadia-core \
  --include python/cascadia_mlx \
  --include tools/cluster_research_queue.py \
  --include tools/r0_spatial_mlx_campaign.py \
  --include tools/r0_spatial_mlx_report.py \
  --include tools/rust_experiment_bundle.py \
  --binary target/release/r0_spatial_mlx_export \
  --output-root artifacts/experiments/r0-spatial-mlx-tournament-v1/bundles
```

Capture the emitted `bundle_path`:

```bash
export R0_MLX_BUNDLE="artifacts/experiments/r0-spatial-mlx-tournament-v1/bundles/<bundle_id>"
```

### 4. Parent authorization

This is the first command that expresses approval to train:

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/r0_spatial_mlx_campaign.py authorize \
  --bundle "$R0_MLX_BUNDLE" \
  --corpus-lock \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/control/corpus-lock.json \
  --approved-by john \
  --output \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/control/authorization.json
```

### 5. Generate the inert queue specification

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/r0_spatial_mlx_campaign.py queue-spec \
  --repository . \
  --bundle "$R0_MLX_BUNDLE" \
  --corpus-lock \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/control/corpus-lock.json \
  --authorization \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/control/authorization.json \
  --queue artifacts/cluster/research-queue-v1.json \
  --experiment-root artifacts/experiments/r0-spatial-mlx-tournament-v1 \
  --output \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/queue-spec.json
```

This command does not mutate the queue.

### 6. Apply only after review

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/r0_spatial_mlx_campaign.py queue-spec \
  --repository . \
  --bundle "$R0_MLX_BUNDLE" \
  --corpus-lock \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/control/corpus-lock.json \
  --authorization \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/control/authorization.json \
  --queue artifacts/cluster/research-queue-v1.json \
  --experiment-root artifacts/experiments/r0-spatial-mlx-tournament-v1 \
  --output \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/queue-spec-applied.json \
  --apply
```

### 7. Run the four-host coordinator

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/cluster_research_queue.py \
  --queue artifacts/cluster/research-queue-v1.json \
  run-coordinator \
  --hosts john1 john2 john3 john4 \
  --lease-seconds 7200 \
  --poll-seconds 2 \
  --idle-timeout-seconds 300
```

The queue handles export, cache validation, training, evaluation, performance
calibration, dynamic collection, two classifier orders, and byte comparison.

### 8. Manual classifier replay

```bash
PYTHONPATH=python:tools .venv/bin/python \
  tools/r0_spatial_mlx_report.py \
  --collection \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/collection.json \
  --order forward \
  --output \
    artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/classification-replay.json
```

## Invalidations And Stop Rules

Stop immediately and preserve artifacts if:

- the corpus lock differs from the preregistered ID;
- any manifest or shard checksum changes;
- any host has a different bundle or authorization;
- MLX does not select the GPU;
- parameter count differs from 74,635;
- a cache semantic digest differs across arms;
- padding or overflow accounting fails;
- any metric is nonfinite;
- a run opens test or final data;
- a queue task runs on an incompatible host;
- historical 441 starts before every primary is claimed;
- the classifier lacks five reports; or
- forward and reverse classifications differ.

No failed or incomplete result is silently retried under a new scientific
identity. Queue retries retain attempt provenance. Source, protocol, corpus,
or authorization drift requires a new bundle and authorization.

## Remaining Promotion Blockers

Even if Stage 2 selects a candidate, promotion is blocked until the same
candidate passes the remaining R0 gates:

1. complete-action ranking on a frozen open-validation action corpus;
2. target recall within one percentage point of exact R0-A;
3. retained-regret increase no greater than 0.02;
4. realistic legal-set action latency and memory;
5. the preregistered paired 20-game smoke without futility-triggered
   degradation; and
6. a separate promotion decision.

This Stage 2 queue must not launch those later gates automatically.

## Launch State

As of 2026-06-17:

- exporter implementation: complete;
- MLX cache, model, and runner: complete;
- authorization and preflight gates: complete;
- four-host work-conserving queue: complete;
- deterministic classifier: complete;
- focused tests: complete;
- production bundle: sealed as
  `611bcebcad6d7dd94374ada8ee8022263363cbd53f67f027eebc0689d81487c6`;
- production authorization: issued as
  `0efdd13c4a693be30ec5e845a19f74f5996743c9b7850e6c84b41a1047cf2b4c`;
- production queue specification: applied with scientific task hash
  `425a4638005629ed047a92813c757acb098ab997125ba227486262c40cba68c9`;
- four-host bundle fanout and all four host preflights: passed; and
- production training: all four primary arms launched concurrently.

The first bundle/authorization/specification identity was invalidated before
queue application because adversarial preflight detected Python bytecode files
inside the bundle. The repaired entry points pass `-B`, the bundle tree is
read-only, and the complete incident is preserved in
`r0-spatial-mlx-tournament-v1-invalid-launch-1.md`.

## Result

Production execution completed on 2026-06-17 with five structurally complete
reports and byte-identical forward/reverse classifications.

- All three compact arms passed value noninferiority.
- No compact arm passed either same-host leverage threshold.
- Radius 4 was the fastest compact arm at 0.291x exact inference and 0.297x
  exact training throughput.
- The closest tested 121-ish shape was 114 rows. It was 8.67x faster than
  historical 441 in observed inference, but only 0.202x the exact control in
  same-host calibration.
- `selected_stage2_candidate` is null.
- Promotion and progress-to-100 claims remain false.

The full result and interpretation are recorded in
`r0-spatial-mlx-tournament-v1-result.md`.
