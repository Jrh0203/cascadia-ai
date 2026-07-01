# ADR 0154: R4 Exact Adaptive Multi-Resolution Foundation

Status: completed; compactness failed

Date: 2026-06-17

Research-plan item: R4

Experiment ID: `r4-adaptive-multires-foundation-v1`

Schema: `r4-focal-nearfield-topology-v1`

## Context

The state-footprint census established three binding facts:

1. the historical 21x21 / 441-cell lattice is oversized;
2. a complete centered hex disk has 61 cells at radius four, 91 cells at
   radius five, and 127 cells at radius six; and
3. no complete centered hex disk has 121 cells.

The 121-cell historical observation is therefore directionally correct but
cannot be implemented as a regular centered disk. A 91-cell radius-five disk
is the nearest smaller regular support, while a 127-cell radius-six disk is
the nearest larger support. Both require exact overflow for the legal rules
domain.

R0 showed that materializing a bounded dense disk for every player board was
slower than exact sparse entities. R2 established the accepted exact public
state substrate, and R3 showed that local action patches cannot replace exact
global component, motif, frontier, market, and supply changes. S4 then showed
that a large fixed candidate-context graph did not recover the quality or
serving envelope of the full R2 control.

R4 tests a narrower representation claim:

> Preserve exact geometry where the current action can modify the focal board,
> and expose exact-but-coarsened topology for the far field and opponents.

This foundation separates authoritative state reconstruction from the
model-visible view. Every arm retains an exact sidecar. An ablation may hide
that sidecar from the model view, but it may never delete, pool, or approximate
the state used for decoding and verification.

## Decision

Implement a standalone Rust package at:

`tools/r4_adaptive_multires_census`

It depends on the accepted R2 library by path and treats
`SparsePublicState` as the semantic authority. It does not copy or fork R2
component, frontier, motif, D6, or public-state logic.

The foundation has two regular near-field variants:

| Radius | Capacity | ID |
|---:|---:|---|
| 4 | 61 | `radius4-61` |
| 5 | 91 | `radius5-91` |

Radius six is the prior bounded control and remains available through R0. A
121-cell pseudo-disk is prohibited.

## Focal-Board Contract

The current relative seat from the exact R2 public state is the focal board.
Only that board receives a fixed near-field disk.

Every other active player board is represented through far-field topology in
the model-visible treatments. The exact sidecar still retains every opponent
tile and coordinate.

This is intentional. A candidate action directly edits the focal board, while
opponent boards affect value through public habitat structure, wildlife
structure, drafting demand, seat timing, and competition. Materializing 61 or
91 empty coordinate slots for every opponent would repeat the R0 cost without
testing the R4 hypothesis.

## Exact Frame And Near Field

The focal center is the exact F2/R0 minimax integer center over occupied focal
coordinates. Stable `(q, r)` tie-breaking is identical to ADR 0135.

D6 augmentation transforms the selected center with the state. It may not
rerun the tie-break after transforming a tied frame.

Every near-field index is present and distinguishes:

- outside the finite rules backing grid;
- empty nonfrontier;
- empty legal frontier; and
- occupied tile.

Occupied cells retain exact tile semantics, directed edge terrains, wildlife
eligibility, placed wildlife, and keystone status. Frontier cells retain exact
neighbor geometry, facing terrains, wildlife adjacency, habitat contacts,
merge sizes, bridge facts, repeated contacts, and optional supplied-tile
compatibility.

## Exact Sidecar And Decoder

The authoritative payload contains:

- exact public global, player, market, and supplied-tile metadata;
- one carried center per active board;
- local occupied entities indexed inside the selected disk; and
- individually addressed exact-coordinate overflow entities.

The focal near field is model-visible. Local/overflow partitioning for the
other boards exists only in the authoritative payload unless an explicit
exact-entity control enables it.

The canonical envelope is `CSR4AM1`. Decode reconstructs a target-free
`PositionRecord`, invokes R2 extraction, regenerates every derived layer, and
requires byte-identical encode after decode. Terminal targets are neither
serialized nor hashed.

No summary token participates in authoritative reconstruction.

## Far-Field Blocks

R4 defines three independently ablatable model-visible blocks.

### H: Habitat topology

One token is emitted for each habitat component with any member outside the
focal near field, and for every component on nonfocal boards. It carries exact:

- terrain and relative seat;
- total, near, and far member counts;
- matching internal, far-internal, and near/far crossing edge counts;
- open habitat-boundary and unique frontier-contact counts;
- degree histogram;
- radial-distance histogram;
- D6-covariant direction-sector bits and counts;
- local member indices; and
- local-to-far habitat portals.

The token omits exact far member coordinates. Those remain in the sidecar.

### W: Wildlife topology

The wildlife block contains exact same-species connected-component summaries
and a counted histogram of exact far-anchor neighborhood signatures. It
retains:

- species, relative seat, and component size;
- near/far counts and crossing edges;
- internal-edge and degree histograms;
- endpoint and branch counts;
- graph diameter;
- undirected edge-direction counts;
- maximum collinear run by the three hex axes;
- radial and direction-sector distributions;
- adjacent-species count vectors; and
- local-to-far wildlife portals.

This is an exact summary of the declared topology. It is not declared to be a
complete Card A wildlife quotient; S3 owns that stronger claim.

### F: Frontier affordances

Every far legal-frontier cell is assigned to a counted exact-signature bucket.
The signature retains:

- relative seat, radial distance, and D6-covariant direction-sector bits;
- occupied-neighbor count and circular-run count;
- opposite-neighbor-pair count;
- facing-terrain counts;
- adjacent-wildlife counts;
- sorted habitat-touch terrain and component-size facts;
- resulting habitat size by terrain;
- habitat-bridge and repeated-contact terrain bits; and
- near/far boundary contacts.

Coordinates are omitted from the model-visible bucket and retained exactly in
the sidecar.

## Frozen Ablation Lattice

For each radius, run the complete three-factor lattice:

| Arm | H | W | F |
|---|---:|---:|---:|
| `n0-near-only` | 0 | 0 | 0 |
| `h-habitat` | 1 | 0 | 0 |
| `w-wildlife` | 0 | 1 | 0 |
| `f-frontier` | 0 | 0 | 1 |
| `hw-habitat-wildlife` | 1 | 1 | 0 |
| `hf-habitat-frontier` | 1 | 0 | 1 |
| `wf-wildlife-frontier` | 0 | 1 | 1 |
| `hwf-all-topology` | 1 | 1 | 1 |

One additional `e-exact-far-control` arm exposes the complete exact R2 far
entity stream. It is an information ceiling, not a compact treatment.

Global metadata, player metadata, market, scoring objective, focal near field,
radius, and relative-seat ownership are identical across the nine arms.

## Adversarial Contract

The permanent Rust suite contains legal exact-state pairs for:

1. identical near field, different far habitat component;
2. identical near field, different long Salmon topology;
3. identical near field, different far Hawk conflict;
4. identical near field, different far Fox neighborhood diversity;
5. identical near field, different far legal frontier;
6. identical in-radius occupancy, different overflow consequence;
7. focal-equivalent state, different relative opponent board;
8. all twelve D6 transforms and inverses; and
9. target mutation with byte-identical representation.

`n0-near-only` must collide on the long-range pairs. This is a suite-sensitivity
requirement, not a failure of the runner.

The H, W, and F single-factor arms must each distinguish their corresponding
pair. `hwf-all-topology` and `e-exact-far-control` must distinguish every
required long-range pair.

## Frozen Corpus And Cluster Allocation

Use the accepted 60,000-position R0/R2 corpus with no new teacher labels:

- john1: train part 0 plus validation part 0;
- john2: train part 1 plus validation part 1;
- john3: train part 2 plus validation part 2; and
- john4: train part 3 plus validation part 3.

Each host processes unique source rows and evaluates both radii and all nine
model-visible arms for its rows. No host duplicates another host's corpus
partition.

The aggregate requires all eight frozen dataset identities exactly once,
merges integer histograms before computing quantiles, and is invariant to
input report order. Forward and reverse aggregate files must be byte-identical.

## Measurements

For each radius and ablation:

- near, habitat, wildlife, frontier, and exact-far token distributions;
- total model-visible token count;
- canonical feature-byte distribution;
- exact-state and feature-stream BLAKE3 identities;
- focal local/overflow occupied counts;
- opponent exact and summarized counts;
- authoritative packed bytes;
- extraction and view-construction throughput;
- codec round-trip counts;
- R2 semantic equality;
- D6 transform/inverse counts;
- target-independence counts; and
- adversarial collision/retention matrix.

Operational host timings and paths are excluded from scientific hashes.

## Promotion Gates

The foundation passes only if:

1. every source state, packed state, and decoded R2 state is exact;
2. every D6 transform/inverse and target-independence check passes;
3. no source entity is clipped, pooled, or omitted from the sidecar;
4. radius capacities are exactly 61 and 91 and no 121-cell claim appears;
5. the near-only collision controls fire;
6. H, W, and F each resolve their registered single-factor pairs;
7. HWF and the exact-far control resolve every long-range pair;
8. HWF P99 model-visible spatial tokens are at most 256 for radius four and
   at most 288 for radius five;
9. authoritative packed P99 is at most the 864-byte
   `compact-entity-v2` source record;
10. all 60,000 rows and eight frozen datasets are present exactly once; and
11. forward and reverse aggregate outputs are byte-identical.

Passing authorizes a matched MLX comparison. It does not establish learned
quality, latency superiority, search strength, gameplay strength, or progress
to the 100-point target.

## Learned Successor

If the foundation passes, the first learned comparison uses four independent
hosts:

- exact R2 control;
- best radius-four H/W/F treatment;
- best radius-five H/W/F treatment; and
- exact-far or next-best compact fallback.

Architecture, parameter count, optimizer, training rows, validation rows,
target, D6 schedule, batching, and serving measurement remain matched. A
compact R4 arm advances only if it is quality-noninferior to R2 and improves
realistic action-ranking latency or memory.

## Consequences

R4 can falsify individual far-field hypotheses without sacrificing an exact
decoder. A negative H, W, or F result closes that block cleanly. A positive
combination identifies which topology carries recovered information.

Any change to focal-seat selection, radius, center tie-break, sidecar
semantics, block definition, ablation IDs, adversarial pairs, corpus identity,
or promotion thresholds requires an ADR amendment before execution.

## Outcome

The four-host production census completed on 2026-06-17 and was classified
`r4_adaptive_multires_compactness_failed`.

All exactness, adversarial, corpus, packed-byte, cross-host parity, and
aggregate-order gates passed. HWF P99 was 271 tokens for `radius4-61` against
the frozen 256-token gate and 298 tokens for `radius5-91` against the frozen
288-token gate. MLX work under this ADR is not authorized.

The exact `CSR4AM1` envelope, carried centers, overflow sidecar, decoder, and
H/W/F extractors remain accepted infrastructure. The variable-cardinality
all-topology model view is rejected under the registered budgets. See
`../reports/r4-adaptive-multires-foundation-v1-result.md`.
