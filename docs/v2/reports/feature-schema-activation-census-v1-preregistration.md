# Feature Schema Activation Census V1 Preregistration

Date: 2026-06-16

Experiment ID: `feature-schema-activation-census-v1`

ADR 0129 preregisters F1 before any production census output is interpreted.
This experiment is permanent representation infrastructure, not a gameplay or
model-strength result.

## Hypothesis

Current representation surfaces contain silent dead, constant, rare, aliased,
colliding, perspective-ambiguous, or misdocumented channels that waste
capacity and can invalidate comparisons. A checksum-bound generated manifest
plus a million-row activation census will make every such status explicit.

## Frozen Implemented Schemas

The generated manifest covers:

- historical sparse V1 blocks and the exact 11,231-column
  `mid-features,v4-opp` champion layout;
- `compact-entity-v2` board, market, and global tensor blocks;
- complete-action graded-oracle action, observable prior, parent supply,
  staged market, and staged supply blocks;
- all seven frozen 192-dimensional candidate factors; and
- current hierarchical group-state, query-context, and item-feature arrays.

The proposed corrected NNUE tail and Relational Opportunity Graph are
manifest-only. They are frozen as unimplemented and unmeasurable.

## Frozen Open Domains

Production census inputs are:

```text
artifacts/datasets/complete-action-graded-oracle-v1-train
artifacts/datasets/complete-action-graded-oracle-v1-validation
artifacts/experiments/complete-action-frontier-factor-integration-v1/caches/train
artifacts/experiments/complete-action-frontier-factor-integration-v1/caches/validation
artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache/train
artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache/validation
```

The required legacy activation input is a separately manifested
`legacy-mid-v4opp-11231` sparse JSONL extraction. It must be generated from
the historical extractor without changing its opponent order or mid-tail
semantics. If that input is absent, the combined result is incomplete rather
than inferred from V2 data.

The graded train corpus contains 2,135,111 candidate rows and validation
contains 860,203. The combined candidate domain therefore exceeds the
1,000,000-row F1 minimum without opening test.

## Frozen Cohorts

Activation is reported by absolute focal seat `0..3` and:

| Phase | Personal turn |
|---|---:|
| Opening | 1 |
| Early | 2-5 |
| Middle | 6-13 |
| Late | 14-20 |

Missing cache metadata is reported as `unknown`; it is never reconstructed
from model values or neighboring artifacts.

## Frozen Analysis

For every measured block and channel, report:

- rows and scalar values;
- active-row and value-activation rates;
- minimum, maximum, and nonzero count;
- dead, constant, and rare channels;
- exact empirical aliases when retained cells are within the frozen budget;
- deterministic BLAKE3 alias candidates otherwise;
- declared structural aliases and known schema defects;
- bounded, byte-verified representation collisions;
- phase and focal-seat counts/rates; and
- manifest, payload, and source-code provenance.

Shard merge intersects channel equivalence classes and may preserve an exact
alias only when every contributing shard verified it exactly. Candidate-row
collision counts are byte-verified within source-evidence shards; cross-shard
row collisions are reported as unknown.

Rare means a nonzero activation rate below `1e-4`. Exact alias retention is
bounded at 2,000,000 scalar cells per block. Large candidate collision
analysis uses feature-fingerprint sampling modulus 64 and a maximum of 100,000
stored signatures. Sampling bounds and any truncation are part of the result.

## Integrity and Exclusions

Every selected source validates before its rows are interpreted. Validation
includes manifest schema, split, aggregate counts, payload size, payload
BLAKE3, binary header, record dimensions, and cache scientific identity.

Teacher labels are not features. The scanner excludes rollout means,
uncertainties, samples, expected ranks, target masks, selected labels, and
champion labels. Exact immediate public score deltas and observable screen
priors remain allowed because they are live-computable candidate inputs.

The following remain closed:

- sealed test;
- gameplay;
- new teacher or rollout compute;
- MLX training;
- cloud, Modal, or external compute; and
- the live research queue.

## Four-Machine Sharding

All machines receive the same frozen roots. Evidence ownership is
`BLAKE3(evidence_id) mod 4`.

```text
john1: --shard-index 0 --shard-count 4
john2: --shard-index 1 --shard-count 4
john3: --shard-index 2 --shard-count 4
john4: --shard-index 3 --shard-count 4
```

Each graded shard, legacy shard, factor-cache batch, and hierarchical shard
has one owner. The final merge requires all four shard indices and rejects any
duplicate evidence ID. This plan creates nonduplicative evidence and permits
mechanical rebalancing only by changing the preregistration.

### Launch Amendment

ADR 0132 was accepted before production launch because john2 was already
running the sole authorized long MLX dropout origin. Scientific ownership
remains the exact shard set `0/4`, `1/4`, `2/4`, and `3/4`, but the operational
mapping for this run is:

```text
john1: shard 0, then disjoint shard 1
john3: shard 2
john4: shard 3
john2: continue the existing dropout origin without interference
```

All shards use one immutable, checksum-verified scanner/source bundle. Host
identity remains descriptive and outside the scientific hash. No evidence ID
changes shard ownership and no payload is duplicated.

## Success Gates

The experiment succeeds only if:

1. every implemented active block is named and remains inside its boundary;
2. at least 1,000,000 open graded candidate rows are scanned;
3. train and validation both contribute;
4. all four focal seats and four phases are present where the source exposes
   them;
5. all manifests, headers, sizes, checksums, and cache payload identities pass;
6. historical 11,231-column extraction semantics remain exact;
7. dead, constant, rare, structural alias, empirical alias, collision, and
   unknown are separately reported;
8. no proposed schema receives fabricated measurements;
9. four unique shard reports merge with no overlap;
10. merge order does not change the scientific BLAKE3; and
11. every closed-domain flag remains false.

Failure on any gate yields
`feature_schema_activation_census_incomplete`. No model, schema correction, or
training run is authorized by an incomplete census.

## Reproduction Commands

Generate the frozen manifest:

```bash
uv run python tools/feature_schema_activation_census.py manifest \
  --output artifacts/experiments/feature-schema-activation-census-v1/manifest.json
```

One production shard uses:

```bash
uv run python tools/feature_schema_activation_census.py census \
  --train-root artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-root artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --factor-cache-root artifacts/experiments/complete-action-frontier-factor-integration-v1/caches/train \
  --factor-cache-root artifacts/experiments/complete-action-frontier-factor-integration-v1/caches/validation \
  --hierarchical-cache-root artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache/train \
  --hierarchical-cache-root artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache/validation \
  --legacy-root artifacts/datasets/legacy-mid-v4opp-activation-v1 \
  --shard-index 0 --shard-count 4 \
  --output artifacts/experiments/feature-schema-activation-census-v1/reports/john1.json \
  --details-jsonl artifacts/experiments/feature-schema-activation-census-v1/reports/john1-details.jsonl
```

The other hosts change only the shard index and output names. No command in
this preregistration launches the live queue.
