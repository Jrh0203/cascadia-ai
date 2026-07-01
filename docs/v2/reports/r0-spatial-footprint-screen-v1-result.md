# R0 Spatial-Footprint Extraction Screen V1 Result

Date: 2026-06-17

Experiment ID: `r0-spatial-footprint-screen-v1`

Dashboard ID: `r0-spatial-representation-extraction-v1`

Stage: R0 extraction, serialization, and exact round-trip screen

Verdict: **FAIL - lossless, but no state-build speedup**

Machine-readable verdict: `lossless_but_no_state_build_speedup`

## Executive Result

The user's central observation was correct in spirit: the historical 441-cell
state is mostly empty and should not be the default representation. A regular
centered hex disk does not contain 121 cells, however. Its exact size is
`1 + 3r(r + 1)`, so the tested compact supports were:

- radius 4: 61 cells;
- radius 5: 91 cells; and
- radius 6: 127 cells.

All three compact disks, when paired with exact overflow, preserved every
source semantic in the 60,000-position corpus. The 441-cell historical arm
also round-tripped exactly because this experiment prohibited clipping.

The performance hypothesis did not survive measurement. The direct
occupied-entity control was the fastest arm for extraction, serialization,
and deserialization. The best compact disk, radius 4 / 61 cells, reached only
`0.640x` the control's extraction throughput. Radius 6 / 127 reached `0.597x`.
The compact packed payload was only about 7.15% smaller because the same 51.5
occupied entities still had to be represented exactly.

The extraction-stage decision is therefore:

1. Reject dense 61-, 91-, and 127-cell materialization as a state-build
   optimization.
2. Retire the historical 441-cell shape as a new-work default.
3. Retain exact occupied entities as the R0 control and preferred data
   substrate.
4. Continue the preregistered iso-architecture MLX tournament because dense
   tensor layout may still improve accelerator throughput despite losing the
   CPU extraction screen.
5. Prioritize R2 sparse occupied-plus-frontier and relational
   representations. The measured board is sparse enough that this direction
   is now supported by direct evidence rather than intuition.

This result makes no gameplay-strength claim and does not select a learned
model.

## Frozen Experimental Contract

The accepted campaign used one immutable executable and source bundle on all
four Apple Silicon hosts:

```text
bundle BLAKE3:
c4e99c53462e9884c0d9bbbb2220fb70429ae71f6486c7769441e85f1a5750d9

V2 source BLAKE3:
78ec63415e342b4820b89ee5bc7acea32db39af652bf48ba43009e6e7489ae6b
```

The corpus contained:

| Split | Rows |
|---|---:|
| Train | 50,000 |
| Validation | 10,000 |
| Total | 60,000 |

The 60,000 rows were partitioned into four deterministic, nonoverlapping
record shards. Every shard was measured in three independent release-process
invocations. Loop iterations within a process were repeated measurements, not
replicates.

| Coverage item | Value |
|---|---:|
| Unique semantic rows | 60,000 |
| Record shards | 4 / 4 |
| Independent processes | 12 / 12 |
| Selected timed operations per arm | 3,000,000 |
| All replicate timed operations per arm | 9,000,000 |
| Required arms per process | 5 / 5 |

Each process measured all five arms against the same shard. Scientific
comparisons therefore use within-shard ratios before combining shards, which
prevents differences between Mac models from masquerading as representation
effects.

## Arms

| Arm | Spatial form | Exact overflow |
|---|---|---|
| `exact-entity-control` | Direct occupied entities with exact coordinates | Not needed |
| `hex-radius-4-61` | Dense radius-4 local index | Yes |
| `hex-radius-5-91` | Dense radius-5 local index | Yes |
| `hex-radius-6-127` | Dense radius-6 local index | Yes |
| `historical-square-21x21-441` | Historical 21x21 local index | Yes |

No arm was allowed to drop an entity, silently clip a coordinate, omit
overflow processing, change semantic channels, or receive a different corpus.

## Semantic Result

Every report passed:

- source and dataset identity;
- benchmark and packed-schema identity;
- exact nonoverlapping shard coverage;
- required-arm coverage;
- process-replicate identity;
- finite nonzero timing evidence;
- exact pack/unpack round-trip; and
- equality between each arm's reconstructed semantic digest and its source
  shard's semantic digest.

Across all 12 processes, all five arms reconstructed their source semantics
exactly. Merge order was also tested: forward and reverse report aggregation
produced byte-identical scientific payloads.

```text
aggregate scientific BLAKE3:
e7d80d4c281385ac6c541d8a874fec319073ccc3ca370c68d9a0555eeb9c76ad
```

The compact arms are therefore lossless under the tested exact-overflow
contract. This does not mean clipping is safe. F2 and F4 separately proved
that legal adversarial positions can exceed radius 6 and that clipped
radius-6 states can collide. Exact overflow remains mandatory.

## Performance Result

The combined absolute rates below use the selected process from each shard.
Promotion decisions use the combined within-shard geometric ratios in the
following section.

| Arm | Extract records/s | Serialize records/s | Deserialize records/s | Mean packed bytes |
|---|---:|---:|---:|---:|
| Exact entity | 312,847 | 961,499 | 414,073 | 552.000 |
| Radius 4 / 61 | 200,341 | 587,859 | 277,273 | 512.506 |
| Radius 5 / 91 | 193,136 | 562,182 | 263,842 | 512.500 |
| Radius 6 / 127 | 186,746 | 538,813 | 253,613 | 512.500 |
| Historical 441 | 257,932 | 844,122 | 379,632 | 556.000 |

### Within-shard ratios versus exact entity

Values above 1.0 would favor the treatment. Every throughput value below 1.0
is a regression.

| Arm | Extraction | Serialization | Deserialization | Packed-byte fraction |
|---|---:|---:|---:|---:|
| Exact entity | 1.000x | 1.000x | 1.000x | 1.000x |
| Radius 4 / 61 | 0.640x | 0.611x | 0.670x | 0.928x |
| Radius 5 / 91 | 0.617x | 0.585x | 0.637x | 0.928x |
| Radius 6 / 127 | 0.597x | 0.560x | 0.612x | 0.928x |
| Historical 441 | 0.825x | 0.878x | 0.917x | 1.007x |

The preregistered extraction screen required at least `1.5x` state-build
throughput. The fastest compact arm instead achieved `0.640x`. Stated in the
other direction, the exact occupied-entity control processed states about
`1.56x` faster than radius 4 and `1.68x` faster than radius 6.

The 441-cell arm was 17.5% slower than exact entity extraction and produced a
slightly larger packed payload. Its compatibility value is not a performance
argument for carrying it into new architectures.

## Sparsity And Overflow

The source positions contained a mean of 51.5 exact occupied entity rows.

| Arm | Local capacity rows | Mean active rows | Occupancy | Overflow position fraction |
|---|---:|---:|---:|---:|
| Radius 4 / 61 | 244 | 51.494 | 21.10% | 0.1967% |
| Radius 5 / 91 | 364 | 51.500 | 14.15% | 0.0000% |
| Radius 6 / 127 | 508 | 51.500 | 10.14% | 0.0000% |
| Historical 441 | 1,764 | 51.500 | 2.92% | 0.0000% |

Radius 4 used exact overflow in 118 of 60,000 positions. Its mean overflow
load was 0.00603 entity rows per position. Radius 5 and radius 6 did not
overflow in this measured corpus, but neither receives permission to remove
the overflow path: absence in this corpus is not a proof over the legal
domain.

## Interpretation

The measurements support the following explanation.

The direct control stores the information the board actually has: roughly
51.5 occupied entities. The disk arms first map those entities into a
centered local coordinate system, maintain a larger dense-capacity index, and
then preserve out-of-region entities separately. That work does not disappear
when the radius shrinks. The payload also cannot shrink in proportion to the
nominal cell count because exact reconstruction still needs the same entities
and attributes.

This is an inference from the measured operation rates, capacities, and byte
counts. A profiler may further divide the cost among center selection,
coordinate transforms, local-slot initialization, overflow branching, and
packing, but no such attribution is needed for the decision: the end-to-end
state-build contract already lost by a wide margin.

The result does not prove that dense disks are slower inside MLX. Accelerator
kernels can favor fixed shapes, regular memory, and larger batches. The
preregistration explicitly permits promotion through model throughput or
end-to-end leverage, so the controlled MLX tournament remains scientifically
necessary. It must not reuse this extraction result as a proxy for GPU
throughput.

## Gate Classification

| Gate | Result | Evidence |
|---|---|---|
| Exact semantic preservation | Pass | All arms round-trip all 60,000 rows in all 12 processes |
| Exact overflow retained | Pass | Every local arm used the exact overflow contract |
| D6 compatibility | Pass | F3 contract and focused representation tests |
| Four disjoint shards | Pass | 4 / 4 shards reconstruct the corpus |
| Three process replicas per shard | Pass | 12 / 12 accepted reports |
| Deterministic aggregation | Pass | Forward and reverse scientific payloads are byte-identical |
| At least 1.5x state-build throughput | **Fail** | Best compact arm is 0.640x exact |
| MLX model throughput | Not tested | Separate Stage 2 tournament |
| Validation target recall | Not tested | Requires matched learned models |
| Retained regret | Not tested | Requires matched learned models |
| Gameplay noninferiority | Not tested | Opens only after offline qualification |

This is a complete negative verdict for the extraction hypothesis, not an
invalid or inconclusive experiment.

## Research Decision

### Promoted

- Exact occupied-entity records remain the canonical R0 data and extraction
  control.
- Sparse occupied-plus-frontier and relational R2 work receives higher
  priority.
- Radius 4, 5, and 6 remain eligible only for the already-preregistered MLX
  layout tournament.

### Rejected

- The claim that reducing 441 cells to 127, 91, or 61 cells automatically
  provides a state-build speedup.
- Any plan to make a dense compact disk the canonical serialized state.
- Any new architecture that treats the historical 441-cell square as the
  default merely because old weights used it.

### Required next evidence

The R0 MLX stage must hold the corpus, semantic channels, hidden width,
parameter budget, optimizer, dtype, training steps, D6 schedule, and target
fixed while varying only the spatial layout. It must report:

- compile time;
- examples and legal actions per second;
- peak unified memory;
- realistic legal-set ranking latency;
- open-validation target recall and retained regret;
- tail-regret behavior; and
- paired gameplay only for offline-qualified arms.

In parallel, R2 should test native sparse occupied, frontier, action,
component, motif, and relation tokens without first materializing a dense
disk.

## Execution Audit

The first production graph was invalidated before timing because workers did
not share one complete source identity. Its incomplete tasks remain preserved
as cancelled audit history and contribute no evidence.

The accepted campaign used one source-frozen bundle on john1, john2, john3,
and john4. Collection intervals were rebalanced only before their execution,
without changing ordinals. One john1 timing process that overlapped unrelated
local verification was quarantined and replaced. The complete unstarted
shard-1 replicate group was moved together from john2 to john3 while john2 ran
the authorized MLX dropout experiment. These changes preserved source,
dataset, partition, arm, and replicate contracts.

Forty-one superseded tasks are recorded as cancellations. No accepted report
was overwritten, and no cancelled or quarantined report contributes to the
aggregate.

## Evidence

- Preregistration:
  `docs/v2/reports/r0-spatial-footprint-screen-preregistration.md`
- Representation contract:
  `docs/v2/decisions/0135-r0-lossless-spatial-representation-contract.md`
- Distributed classifier:
  `docs/v2/decisions/0136-r0-distributed-extraction-classifier.md`
- Work-conserving timing rebalance:
  `docs/v2/decisions/0141-r0-work-conserving-timing-host-rebalance.md`
- Forward aggregate:
  `artifacts/experiments/r0-spatial-footprint-screen-v1/reports/extraction-source-frozen-work-conserving-aggregate-forward.json`
- Reverse aggregate:
  `artifacts/experiments/r0-spatial-footprint-screen-v1/reports/extraction-source-frozen-work-conserving-aggregate-reverse.json`
- Whole-tree fanout proof:
  `artifacts/experiments/r0-spatial-footprint-screen-v1/reports/source-frozen-production-bundle-fanout.json`
- Immutable bundle manifest:
  `artifacts/experiments/r0-spatial-footprint-screen-v1/bundles/c4e99c53462e9884c0d9bbbb2220fb70429ae71f6486c7769441e85f1a5750d9/bundle.json`
