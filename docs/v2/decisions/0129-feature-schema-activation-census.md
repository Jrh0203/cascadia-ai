# ADR 0129: Feature Schema Manifest and Activation Census

Status: preregistered

Date: 2026-06-16

Experiment ID: `feature-schema-activation-census-v1`

## Context

F1 in the research implementation plan requires permanent infrastructure for
finding dead, constant, rare, aliased, perspective-wrong, colliding, or
misdocumented features before another model comparison is trusted.

The repository currently has five materially different representation
surfaces:

1. the historical sparse V1 and `mid-features,v4-opp` NNUE index spaces;
2. the fixed-width `compact-entity-v2` `PositionRecord`;
3. the lossless complete-action graded-oracle action, prior, staged-market, and
   staged-supply tensors;
4. the seven frozen 192-dimensional candidate-factor caches; and
5. the current draft/tile/wildlife hierarchical factor caches.

They do not share one indexing convention. Ownership may be an integer range,
a tensor slice, a token type, an NPZ array, or a future relation type. A valid
manifest must describe these directly rather than forcing them into a false
common dense-vector abstraction.

The historical NNUE also contains a confirmed schema defect. In the champion
`mid-features,v4-opp` build, indices `10561..10862` are not the documented
extended tile-supply tail. They are the first 301 columns of the full
per-cell adjacency block: three complete 78-column cells plus 67 columns of a
fourth cell. Indices `10862..11231` are the v4 opponent block. Existing
weights retain those exact semantics.

## Decision

Adopt `tools/feature_schema_activation_census.py` as the authoritative F1
manifest and census tool.

It provides three commands:

```text
manifest  emit the deterministic schema registry
census    validate and stream one or more open evidence roots
merge     combine disjoint shard reports
```

The checked-in generated registry is:

```text
artifacts/experiments/feature-schema-activation-census-v1/manifest.json
```

Every block entry records:

- schema name, version, and scientific BLAKE3;
- exact index range, tensor slice, token ownership, or NPZ array;
- semantic owner and value domain;
- expected D6 behavior;
- perspective convention;
- incremental dependencies;
- checkpoint/cache compatibility;
- activation and channel status when measured; and
- source evidence.

Future schemas may appear only with
`implementation_status=unimplemented` and
`measurement_status=unmeasurable`. The manifest must never synthesize
activation data for an implementation that does not exist.

## Frozen Phase and Perspective Contract

The four F1 phases are:

| Phase | Personal turn |
|---|---:|
| Opening | 1 |
| Early | 2-5 |
| Middle | 6-13 |
| Late | 14-20 |

`compact-entity-v2` and graded-oracle rows record the absolute focal seat.
Their board and player slots remain focal-relative. Cache formats that do not
retain absolute focal seat report seat `unknown`; they are not imputed.

The current hierarchical cache stores only its historical three-phase label.
Those rows retain the source labels `early`, `middle`, and `late`; the tool
does not fabricate an opening split from missing personal-turn metadata.

Legacy sparse streams must explicitly provide `focal_seat` and either the
frozen four-phase label or `personal_turn`. Historical v4 opponent ordering is
measured as emitted and is never silently repaired.

## Frozen Activation Definitions

For each block and channel:

- **active row:** at least one nonzero value in the block;
- **value activation rate:** nonzero scalar values divided by all scalar
  values;
- **dead:** zero in every observed row;
- **constant:** exact minimum equals exact maximum;
- **rare:** nonzero activation rate below `1e-4` and above zero;
- **structural alias:** forced by schema construction;
- **empirical alias:** byte-for-byte equal channel streams on the observed row
  domain;
- **sketch alias:** matching deterministic BLAKE3 streams when the exact cell
  budget was exceeded;
- **representation collision:** distinct stable row identities with
  byte-identical feature rows; and
- **unknown:** unavailable from the supplied evidence.

The default exact alias budget is 2,000,000 scalar cells per block. Larger
streams retain deterministic per-channel BLAKE3 sketches. Candidate-row
collision analysis uses a deterministic feature fingerprint sample,
cryptographic hashing, and byte-for-byte verification. The default sample
modulus is 64 and the bounded signature table contains at most 100,000
entries. Bounds and truncation are reported.

Merged activation reports intersect each shard's channel-stream equivalence
classes. They preserve exact aliases only when every contributing shard
verified the alias exactly; all other matches remain sketch candidates.
Candidate-row collisions remain byte-verified within each source-evidence
shard. The merged report marks cross-shard row collisions unknown rather than
claiming verification that did not occur.

## Input Integrity

No feature metric may be interpreted until its source manifest and selected
payload pass:

- schema/version checks;
- open split checks;
- declared aggregate counts;
- file size;
- BLAKE3 checksum;
- binary magic, header, record size, and feature/target hashes where
  applicable; and
- cache payload scientific identity.

Only `train` and `validation` are accepted. A test manifest is rejected.

The legacy input contract is a manifest-backed JSONL stream:

```json
{
  "schema_version": 1,
  "feature_schema": "legacy-mid-v4opp-11231",
  "feature_count": 11231,
  "split": "train",
  "rows": 1000000,
  "shards": [
    {
      "file": "part-000.jsonl",
      "row_count": 250000,
      "blake3": "..."
    }
  ]
}
```

Each JSONL row contains `features`, `focal_seat`, and `phase` or
`personal_turn`. Out-of-range indices invalidate the run.

## Teacher and Closed-Domain Policy

The census may read only live-computable inputs:

- public parent and staged state;
- lossless complete action;
- exact immediate score/component deltas; and
- observable screen priors.

It explicitly excludes:

- R600, R1200, and R4800 means;
- teacher standard deviations and sample counts;
- expected-rank labels;
- target masks; and
- selected/champion labels.

The sealed test split, gameplay, new teacher compute, cloud, Modal, and all
external compute remain closed.

## Streaming and Scientific Identity

Fixed-width datasets are memory-mapped. Graded candidates and `.npy` factor
rows are processed in bounded chunks. Hierarchical `.npz` shards are opened
one at a time. Parent-state blocks are counted with exact candidate
multiplicity without retaining candidate sets.

`--row-limit` exists only for tests and smoke checks. A scientific F1 result
must omit it.

The report's scientific BLAKE3 covers the manifest identity, configuration,
evidence identities/checksums, counts, statuses, and collision results. It
contains no timestamp, host timing, absolute input path, or output path.
Raw input-manifest file hashes remain in the top-level provenance object so
byte-for-byte auditability does not make the scientific identity machine- or
run-dependent.

## Merge Contract and Four-Machine Plan

Evidence assignment is:

```text
BLAKE3(evidence_id) mod 4
```

Run the identical frozen command on all four machines with:

| Host | Arguments |
|---|---|
| john1 | `--shard-index 0 --shard-count 4` |
| john2 | `--shard-index 1 --shard-count 4` |
| john3 | `--shard-index 2 --shard-count 4` |
| john4 | `--shard-index 3 --shard-count 4` |

The evidence ID includes the dataset/cache identity and source shard or batch.
No payload can enter two shard reports. The final merge requires exactly
`0/4`, `1/4`, `2/4`, and `3/4`, rejects duplicate evidence IDs, and sorts
inputs before hashing so collection order cannot change the result.

This produces unique evidence rather than four replicas. It also permits
factor-cache batches, graded shards, hierarchical shards, and manifested
legacy shards to distribute independently.

## Gates

Classify `feature_schema_activation_census_complete` only when:

- every implemented active block is named and bounded;
- at least 1,000,000 open complete-action candidate rows are scanned;
- train and validation are both represented;
- all four focal seats and all four frozen phases are reported where exposed;
- every selected manifest, header, file size, and checksum passes;
- no feature index crosses a schema boundary;
- the 11,231-column historical champion layout is reproduced exactly;
- dead, constant, rare, structural alias, empirical alias, collision, and
  unknown remain distinct statuses;
- all proposed-only schemas remain explicitly unimplemented/unmeasurable;
- all four unique shard reports merge without duplicate evidence;
- two independent merges produce the same scientific BLAKE3; and
- all closed-domain flags remain false.

Otherwise classify `feature_schema_activation_census_incomplete`. There is no
futility rule because this is permanent infrastructure.

## Consequences

New feature work must add or update a manifest block before model training.
Historical checkpoint columns are immutable. Correcting the NNUE mid tail
requires a new schema and checkpoint. Cache formats that omit phase or seat
metadata may remain measurable, but the missing cohort dimensions must stay
explicitly unknown.
