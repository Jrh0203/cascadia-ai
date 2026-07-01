# Public Action Equivalence Foundation V2 Result

Date: 2026-06-17

ADR: 0163

Experiment: `s7-public-action-equivalence-foundation-v2`

Protocol: `s7-exact-semantic-transition-v2`

Classification: `public_action_equivalence_proof_only_futile`

## Executive Result

The corrected S7 implementation is valid, complete, and scientifically
negative.

Across every one of the 2,995,314 complete actions in the open train and
validation domains, no two naturally occurring actions shared even the
broadest registered semantic-state-plus-supply successor key. Serving-safe,
exact-public, and exact-hidden successor classes were therefore also all
singletons.

S7 remains valuable proof infrastructure. It does not enter the serving path
and cannot improve iteration speed or player strength in the current action
representation.

## Corpus

| Split | Decisions | Complete actions |
|---|---:|---:|
| Train | 560 | 2,135,111 |
| Validation | 240 | 860,203 |
| **Total** | **800** | **2,995,314** |

The corpus was partitioned by row modulo three:

- john2: shard 0;
- john3: shard 1; and
- john4: shard 2.

All three hosts used immutable bundle
`dcbd9c2e3755c1c28745279a64978c6f928d03ff61812d367640d38bb59d3ce1`.
Collection verified every remote SHA-256 before aggregation.

## Validity

The aggregate passed:

- 2,400 exact parent-record checks;
- 2,400 public-state hash checks;
- 2,400 public-supply checks;
- 2,995,314 canonical action-hash checks;
- 2,995,314 grouped exact-R3 action matches;
- 2,995,314 exact R3 applies;
- the frozen three-state adversarial transition suite;
- the production-path duplicate-accounting witness; and
- forward/reverse shard-order byte identity.

Invariant failures were zero. Forward and reverse scientific BLAKE3 values
were both
`aed15aceab5114bd734c18623734ea8dec57e37c6fbac1ae037a5d162f1f57da`.

V1 remains invalid and contributed no production evidence.

## Compression Result

Every hierarchy level produced exactly zero natural collapses.

| Split and key | Candidates | Unique classes | Collapsed | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train semantic state+supply | 2,135,111 | 2,135,111 | 0 | 0% | 0% | 0% | 0% |
| Train serving-safe | 2,135,111 | 2,135,111 | 0 | 0% | 0% | 0% | 0% |
| Train exact public | 2,135,111 | 2,135,111 | 0 | 0% | 0% | 0% | 0% |
| Validation semantic state+supply | 860,203 | 860,203 | 0 | 0% | 0% | 0% | 0% |
| Validation serving-safe | 860,203 | 860,203 | 0 | 0% | 0% | 0% | 0% |
| Validation exact public | 860,203 | 860,203 | 0 | 0% | 0% | 0% | 0% |

No ordered trace rejected a broad semantic collision because no broad
semantic collision existed. No authoritative collision replay was needed in
production for the same reason.

The frozen promotion gate required either 2% validation median reduction, or
5% P90 reduction with at least 128 collapsed actions at P90. Both observed
values were zero.

## Interpretation

The current complete-action serialization is already injective over the
observed public semantic transition surface. Local similarities such as a
shared destination, draft object, or R3 afterstate fragment are not exact
transition equivalence and must not be used as lossless serving compression.

The positive duplicate smoke matters: a duplicated real action formed the
expected size-two serving-safe and exact-public class, produced one collapse,
reported zero beyond-exact collapse, and passed semantic plus hidden-successor
parity. The negative production result is therefore evidence about the game
corpus, not a dead code path.

## Decision

1. Retain the census, accounting helper, adversarial suite, and duplicate
   witness as regression infrastructure.
2. Reject an S7 serving-time quotient for the current action generator.
3. Do not train an action-equivalence model or add equivalence bookkeeping to
   search.
4. Reopen S7 only if a future hierarchical or symmetry-expanded generator can
   create multiple serialized actions for one exact transition.

This closes S7 without a speed or score claim. The active route to 100 remains
better state/action information, learned ranking, and exact search rather than
lossless action deduplication.

## Artifacts

- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/aggregate-forward.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/aggregate-reverse.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/order-proof.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/control/production-collection.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/control/adversarial.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/control/duplicate-smoke.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/shards/shard-0.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/shards/shard-1.json`
- `artifacts/experiments/s7-public-action-equivalence-foundation-v2/shards/shard-2.json`
