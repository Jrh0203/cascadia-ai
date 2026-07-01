# ADR 0155: R4 Bounded Far-Quotient Foundation

Status: completed; foundation passed

Date: 2026-06-17

Research-plan item: R4 successor / R5 bridge

Experiment ID: `r4-bounded-far-quotient-foundation-v1`

Schema: `r4-radius4-bounded-quotient-v1`

## Context

ADR 0154 established a clean mechanical result and a narrow compactness
failure:

- the exact `CSR4AM1` state, carried center, radius-four near field, overflow
  sidecar, decoder, D6 transforms, and target independence passed;
- every registered habitat, wildlife, frontier, overflow, and opponent
  distinction was retained;
- radius-four HWF P99 was 271 tokens against a frozen 256-token gate;
- radius-five HWF P99 was 298 against 288; and
- every single-factor and two-factor arm fit below the corresponding HWF
  budget.

The failed object was not the 61-cell near field or the exact state. It was the
simultaneous emission of one token per far habitat component, wildlife
component, wildlife signature bucket, and frontier signature bucket.

The next experiment must not relax the failed thresholds, delete exact state,
or hide cardinality inside unmeasured dense vectors. It must test explicit,
bounded quotient hypotheses.

## Decision

Add a new bounded-view implementation to
`tools/r4_adaptive_multires_census`, exposed through a separate
`r4-bounded-far-quotient-census` binary.

The new binary consumes the accepted R4 exact state and does not fork:

- R2 public-state extraction;
- radius-four center selection;
- near-field construction;
- authoritative local/overflow storage;
- habitat/wildlife/frontier topology extraction; or
- exact decode and validation.

Only the model-visible far projection is new.

## Shared Exact Substrate

Every arm uses:

- `radius4-61`;
- all 61 focal near-field cells;
- every R4 far habitat component token;
- every R4 far wildlife component token;
- exact global, player, market, supplied-tile, and relative-seat metadata; and
- the unchanged exact `CSR4AM1` sidecar.

Habitat and wildlife component tokens remain individually addressable because
their measured maxima were only 55 and 44. The unbounded wildlife-signature
and frontier-signature bucket streams are replaced or selectively retained by
the arms below.

No summary token participates in legality, transition, decode, scoring, or
authoritative hashing.

## Four Independent Arms

The frozen corpus is four-player, so all summary grids include zero-filled
tokens for four relative seats. Zero filling makes token count independent of
which species, sectors, or terrains happen to occur.

### Q1: Seat-marginal quotient

ID: `q1-seat-marginal`

- 20 wildlife summary tokens: `4 relative seats x 5 wildlife species`;
- 4 frontier summary tokens: one per relative seat.

Wildlife tokens retain counted distance, direction-sector, same-species
neighbor, occupied-neighbor, adjacency-diversity, and adjacent-species
distributions.

Frontier tokens retain counted distance, direction-sector, neighbor-shape,
facing-terrain, adjacent-wildlife, component-touch, resulting-size, bridge,
repeated-contact, and local-boundary-contact distributions.

Hard structural maximum:

```text
61 near + 55 habitat + 44 wildlife components + 20 + 4 = 184 tokens
```

### Q2: Directional frontier quotient

ID: `q2-directional`

- the same 20 wildlife seat/species tokens;
- 24 frontier tokens: `4 relative seats x 6 D6-covariant sectors`.

A frontier whose sector bitset contains a tie contributes to every tied sector
with an explicit multiplicity field. This preserves boundary directions
without pretending that a symmetric coordinate belongs to one arbitrary
sector.

Hard structural maximum:

```text
61 + 55 + 44 + 20 + 24 = 204 tokens
```

### Q3: Habitat-affordance quotient

ID: `q3-affordance`

- the same 20 wildlife seat/species tokens;
- 20 frontier tokens: `4 relative seats x 5 habitat terrains`.

A frontier contributes to a terrain token when it faces, touches, bridges,
repeats, or produces a nonzero resulting size for that terrain. A separate
untyped count inside every seat/terrain token preserves frontier mass that has
no contact with that terrain.

Hard structural maximum:

```text
61 + 55 + 44 + 20 + 20 = 200 tokens
```

### Q4: Selective exact plus residual quotient

ID: `q4-selective-exact`

- the Q1 20 wildlife and 4 frontier residual summary tokens;
- up to 16 exact wildlife motif-bucket tokens; and
- up to 24 exact frontier-bucket tokens.

Exact buckets are selected by a target-free, D6-compatible priority tuple.
The tuple uses invariant strategic magnitude first: count, local-boundary
contact, bridge/repeated-contact strength, resulting habitat size,
same-species connectivity, adjacency diversity, and distance. Canonical
orientation-invariant signature fields define remaining tie groups. Selection
admits or skips each complete tie group; it never cuts a group by raw sector,
edge, local-index, source order, or orientation-sensitive bytes. A group that
would cross the 16- or 24-token ceiling is skipped and the next lower-priority
group is considered. All unselected buckets are accounted for in the residual
Q1 summaries.

Hard structural maximum:

```text
61 + 55 + 44 + 20 + 4 + 16 + 24 = 224 tokens
```

Selection must be deterministic. Exact plus residual counts must equal the
source R4 bucket counts for every record.

## Fixed Summary Fields

Summary tokens use schema-owned integer fields only. They include:

- total represented anchors/cells and unique source-bucket count;
- exact count-weighted distance histograms over the legal coordinate range;
- exact D6 sector-bit and sector-multiplicity counts;
- exact bounded neighbor-count and circular-run histograms;
- adjacent wildlife sums, nonzero counts, and diversity histograms;
- facing-terrain and component-touch counts;
- resulting habitat-size sums and maxima;
- habitat bridge and repeated-contact counts; and
- local-boundary-contact counts.

Distance bins 0 through 14 are exact; bin 15 is the explicitly named
`distance >= 15` tail. Every distance summary also retains exact minimum,
maximum, total mass, and count-weighted distance sum. Every vector length and
bin meaning is a named constant. Variable-length maps, JSON objects, hashes as
features, learned pooling, target-derived priority, and silent overflow bins
are prohibited.

All counters use checked integer accumulation and fail closed on overflow.

## Honest Size Accounting

The binary writes a canonical little-endian feature envelope independent of
JSON. Every report measures:

- projected spatial token count;
- active primitive scalar count;
- padded primitive scalar slots implied by the type schema;
- canonical feature bytes;
- component, exact-bucket, and residual-summary counts; and
- source-bucket accounting equality.

The experiment fails if a small token count is achieved by embedding an
unbounded vector in a token.

## Adversarial Contract

An arm is information-passing only if it distinguishes all seven ADR 0154
long-range pairs:

1. far habitat component;
2. long Salmon topology;
3. far Hawk conflict;
4. far Fox diversity;
5. far legal frontier;
6. overflow consequence; and
7. relative opponent board.

The production suite also requires:

- exact source-bucket accounting;
- Q4 exact-plus-residual accounting;
- all twelve D6 transform/inverse checks;
- target mutation with identical bounded bytes;
- deterministic repeated construction; and
- malformed summary envelopes to fail closed.

All four arms are evaluated independently. The experiment may pass when at
least one arm passes every frozen gate; a failed arm is retained as negative
evidence. Passing the seven pairs is necessary, not proof that a quotient is
strategically sufficient.

## Frozen Corpus And Four-Host Allocation

Every host already holds all four accepted train and validation partitions.
The arms run concurrently over the identical 60,000 records:

| Host | Arm |
|---|---|
| john1 | `q1-seat-marginal` |
| john2 | `q2-directional` |
| john3 | `q3-affordance` |
| john4 | `q4-selective-exact` |

This is four independent hypotheses, not four replicas. Each source row is
processed once per distinct arm and never twice within an arm.

Before production, all four hosts run every arm on the adversarial suite and
must produce byte-identical scientific reports.

The aggregate requires:

- one complete 60,000-row report per arm;
- identical ordered source-stream identities across arms;
- all eight dataset identities in every arm;
- a checksum-bound collection receipt; and
- byte-identical forward/reverse classification output.

## Frozen Gates

An arm is foundation-passing only if all of the following hold:

1. exact R4 decode, R2 semantic parity, D6 inverse, and target independence
   pass for all 60,000 records;
2. all seven adversarial pairs are distinguished;
3. every source wildlife and frontier bucket is accounted for exactly once;
4. maximum projected token count is at most 224;
5. P99 projected token count is at most 192;
6. maximum active primitive scalar count is at most 16,384;
7. maximum padded primitive scalar slots are at most 24,576;
8. maximum canonical feature bytes are at most 65,536;
9. paired bounded-view construction and canonical encoding throughput is at
   least 0.90x the full R4 HWF view on the same records and host;
10. no counter overflow, malformed envelope, nondeterminism, or source drift
    occurs; and
11. aggregate order is byte-invariant.

The 224 maximum is a hard structural ceiling, not a percentile allowance.
The 192 P99 gate leaves useful headroom below ADR 0154's failed 256-token
radius-four threshold.

## Classification And Selection

Possible experiment classifications:

- `r4_bounded_quotient_foundation_passed`: at least one arm passes every gate;
- `r4_bounded_quotient_information_failed`: every mechanically valid arm loses
  a registered distinction or fails source-bucket accounting;
- `r4_bounded_quotient_size_failed`: information passes but no arm passes all
  token, scalar, byte, and throughput gates; or
- `r4_bounded_quotient_invalid`: source, exactness, corpus, parity, or
  aggregate evidence is malformed.

If multiple arms pass, record two successors:

- **minimal:** lowest maximum token count, then lowest P99 bytes, then highest
  paired throughput;
- **richest:** Q4 when passing, otherwise the passing arm with the highest
  information-preserving token budget.

This selection authorizes a matched MLX comparison. It does not promote a
representation or establish decision quality.

## Learned Successor Boundary

The first learned comparison must include the exact R2 control. It may compare
up to three passing bounded arms across the four Macs. Architecture,
parameters, optimizer, data, target, D6 schedule, serving batch, and validation
must be matched.

A bounded arm advances only if it is quality-noninferior to R2 on aggregate
and protected slices and improves realistic action-ranking latency or memory.

## Consequences

This experiment spends all four Macs on different representation hypotheses
while preserving a common exact authority. A passing arm creates a credible
compact substrate. A negative result closes deterministic fieldwise quotient
summaries and moves the primary lane to R5 component/motif graphs or R6 hybrid
sparse evaluation without weakening the evidence standard.

Any change to arm definitions, token budgets, summary bins, exact-bucket
limits, priority tuple, corpus, host ownership, or gates requires an ADR
amendment before production.

## Outcome

The four-host production campaign completed on 2026-06-17 and was classified
`r4_bounded_quotient_foundation_passed`.

Q1, Q2, and Q3 passed every exactness, adversarial, source-accounting, token,
scalar, byte, throughput, corpus, parity, and order gate. Their P99 token
counts were 166, 186, and 182 respectively. Q1 is the minimal successor and
Q2 is the richest passing successor.

Q4 retained exact-plus-residual source accounting and every adversarial
distinction, but reached P99 206 against the frozen 192-token limit. It is
classified `size_failed` and is not admitted to the learned comparison.

The exact R2 control plus Q1, Q2, and Q3 are authorized for one matched MLX
comparison. See
`../reports/r4-bounded-far-quotient-foundation-v1-result.md`.
