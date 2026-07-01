# R3 Exact Action Local-Patch Plus Global-Edit Foundation Result

Date: 2026-06-17

Experiment ID: `r3-action-edit-foundation-v1`

Protocol: `r3-action-edit-open-corpus-v1`

Contract: ADR 0148

Verdict: **PASS - authorize a matched R3 MLX prototype**

Scientific BLAKE3:
`9a3075bf4b9abb0ce05efad1856ce951163d04f41e619f83acdf77ee78130424`

## Executive Result

The exact R3 action-edit substrate passed every frozen mechanical,
compactness, corpus-coverage, and determinism gate.

Across 20 deterministic four-player games:

- 1,600 canonical decisions encoded exactly 1,600 reusable state trunks;
- 2,668,154 canonical complete actions were verified;
- 11,305 paid-wipe sentinel actions were verified;
- 2,679,459 action edits were decoded, applied, and compared with the
  authoritative public successor;
- all 2,679,459 semantic-supply, regenerated-global-edit, and codec checks
  passed; and
- all 19,200 D6 transform checks passed.

The action edit remained compact:

| Measure | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|
| Edit tokens | 55 | 59 | 62 | 70 |
| Packed edit bytes | 3,222 | 4,118 | 4,915 | 6,629 |

The frozen limits were median tokens <= 128, P99 tokens <= 256, maximum
tokens <= 384, and P99 bytes <= 8,192. Every gate passed with substantial
headroom.

This authorizes a matched MLX prototype that encodes one public state trunk
per decision and batches variable-length action edits. It does not yet prove
faster end-to-end action evaluation, better offline ranking, or stronger
gameplay.

## Immutable Identity

| Identity | BLAKE3 |
|---|---|
| Bundle | `24416fe767223fee6ca9e9cf2748fde45650bccd7526b78601f43c8490459b58` |
| Source | `5e3b2adf57ae80e85c5f90b78ae872becbe39501790409e146b2c9cf7bf65bfb` |
| Executable | `da8503711a0c5cabfe00ea75b564a95d428a213872ba8d182cea2c9c0a23dbb7` |
| Queue task specification | `cf1bacca83b971b82912b3aaf1a1a06b8d36f38bd70643e9340238f18aa0584f` |
| Aggregate scientific payload | `9a3075bf4b9abb0ce05efad1856ce951163d04f41e619f83acdf77ee78130424` |
| Aggregate order proof | `512669a511b3a3ddb0e4b536d8a4de9992131c278dc4eee1014111d9ba1de915` |

The immutable bundle contains 69 reviewed source files and the release
executable. Whole-tree fanout verification matched all 71 bundled files on
`john2`, `john3`, and `john4`.

## Frozen Corpus

| Split | Raw seed range | Games | Positions |
|---|---:|---:|---:|
| Train | `3,300,000..3,300,016` | 16 | 1,280 |
| Validation | `3,400,000..3,400,004` | 4 | 320 |
| **Total** | | **20** | **1,600** |

Every position used the authoritative engine to realize the feasible free
prelude, enumerate the complete canonical action screen, verify every action
edit, and select one deterministic trajectory action. Paid-wipe branches were
additional schema sentinels and did not alter the canonical action-count
distribution.

The complete action-screen distribution was:

| Legal actions per decision | Value |
|---|---:|
| Mean | 1,667.596 |
| Median | 984 |
| P90 | 3,875 |
| P99 | 10,527 |
| Maximum | 18,432 |

This validates the central serving concern behind R3: recomputing a full
afterstate representation thousands of times per decision is a meaningful
cost to attack.

## Exactness And Information Boundary

| Check | Count | Result |
|---|---:|---|
| Exact edit application | 2,679,459 | Pass |
| Authoritative normalized public successor | 2,679,459 | Pass |
| Exact semantic-supply delta | 2,679,459 | Pass |
| Regenerated global edit | 2,679,459 | Pass |
| Codec round trip | 2,679,459 | Pass |
| D6 canonical action view | 19,200 | Pass |
| State-trunk reuse | 1,600 / 1,600 decisions | Pass |
| Silent clipping or truncation | 0 | Pass |

The successor boundary is public and pre-refill. It includes the realized
visible prelude, selected public market objects, placed board objects, public
Nature Tokens, visible market removals, and exact semantic supply after the
action.

It excludes hidden tile-stack order, wildlife-bag order, excluded-tile
identity, wildlife return insertion position, future refill realization, RNG
seed, future actions, terminal targets, and learned labels.

## State-Trunk Shape

The state trunk is encoded once per canonical decision:

| Measure | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|
| Trunk tokens | 328 | 461 | 494 | 511 |
| Packed trunk bytes | 652 | 908 | 972 | 972 |

The exact equality of 1,600 state-trunk encodings and 1,600 canonical
decisions proves that the foundation does not rebuild the complete public
state for each candidate.

## Action-Edit Shape

| Measure | Minimum | Mean | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Edit tokens | 46 | 54.637 | 55 | 59 | 62 | 70 |
| Packed bytes | 1,689 | 3,219.405 | 3,222 | 4,118 | 4,915 | 6,629 |

The token distribution is unusually tight despite action screens ranging up
to 18,432 candidates. Variable-length paid wipes, touched components, frontier
updates, motif updates, and supply deltas did not create a long compactness
tail.

Packed bytes are larger than token count suggests because the mechanical
foundation serializes exact object content and canonical identities. The
matched MLX prototype should consume structured tensors rather than parse the
canonical binary envelope in its inner loop.

## Locality Result

The radius-only hypothesis is false. Exact global edits are necessary.

| Radius | Changed coordinates covered | Actions completely covered |
|---|---:|---:|
| 1 | 55.3977% | 12.5015% |
| 2 | 73.1348% | 42.5873% |
| 3 | 81.5576% | 58.2432% |

Even the complete 37-cell radius-3 patch fails to contain every direct changed
coordinate for 41.7568% of actions. This does not reject R3: the accepted R3
representation is the local patch **plus** exact component, motif, frontier,
market, supply, and metadata edits.

Consequences for the learned comparison:

1. radius 1, 2, and 3 may be compared only while retaining the same exact
   global edit objects;
2. no patch-only arm may be described as lossless;
3. exact global-object ablations must fail closed on the preserved long-range
   component and motif collision fixtures; and
4. the prototype should measure whether smaller local patches improve
   throughput after global-object cost is held constant.

## D6 Correction And Regression

The first pre-production smoke exposed a real representation bug at seed
`4,100,003`, turn zero, transform 2. Frontier equality compared transient R2
component numbers rather than semantic terrain-plus-membership identity.

The invalid run was stopped before production. The repair:

- normalizes component references by terrain and sorted exact membership;
- retains raw component numbers only where exact world application needs
  them;
- adds `d6_regression_seed_4100003_turn_zero_is_exact`; and
- reruns the complete Rust suite and immutable smoke process.

The repaired production corpus passed all 19,200 D6 checks.

## Determinism

### Cross-host and thread-count smoke

The same two-game smoke corpus ran as:

1. `john1`, `RAYON_NUM_THREADS=2`; and
2. `john4`, `RAYON_NUM_THREADS=1`.

Both emitted the same 106,274-byte report:

```text
SHA-256:
e841806d26078a98a81c8c2bb9661d066251baa193df9c260c37f29a6f72d520

scientific BLAKE3:
7d833b34f4a75732db6cef54f88c82a660188d02cf0ad9ae3d0af24b589f607d
```

### Aggregate order

Forward shard order `0,1,2,3` and reverse order `3,2,1,0` produced
byte-identical 146,654-byte aggregate files.

```text
aggregate SHA-256:
e2071681f6ee9c352ad3b5ea4e44aaedbe3e5e5656c1064bdfafad93e2f24a17

aggregate scientific BLAKE3:
9a3075bf4b9abb0ce05efad1856ce951163d04f41e619f83acdf77ee78130424
```

## Cluster Execution

The 13-task campaign completed without task failure:

- one immutable whole-tree fanout;
- four source/executable identity preflights;
- four nonoverlapping production shards;
- one checksum collection;
- two aggregate orders; and
- one terminal order proof.

Each shard owned four train games and one validation game and used five Rayon
workers. The four hosts therefore ran up to 20 independent games concurrently
without duplicated seeds.

Operational shard wall times from scheduler receipts were:

| Host | Shard | Wall seconds |
|---|---:|---:|
| `john1` | 0 | 307.738 |
| `john2` | 1 | 293.654 |
| `john3` | 2 | 267.285 |
| `john4` | 3 | 343.251 |

The full distributed census completed at the slowest shard in 5.72 minutes,
after which collection and both aggregates completed in under five seconds.
These operational timings are deliberately outside the scientific payload.

## Promotion Assessment

| Gate | Result |
|---|---|
| Authoritative public successor parity | Pass, 2,679,459 / 2,679,459 |
| Supply-delta parity | Pass, 2,679,459 / 2,679,459 |
| Regenerated global-edit parity | Pass, 2,679,459 / 2,679,459 |
| Canonical codec round trip | Pass, 2,679,459 / 2,679,459 |
| D6 canonical action-view parity | Pass, 19,200 / 19,200 |
| Silent truncation or clipping | None |
| State trunk encodings | Pass, exactly 1,600 / 1,600 decisions |
| Median edit tokens <= 128 | Pass, 55 |
| P99 edit tokens <= 256 | Pass, 62 |
| Maximum edit tokens <= 384 | Pass, 70 |
| P99 edit bytes <= 8,192 | Pass, 4,915 |
| Full frozen-corpus coverage | Pass |
| Aggregate order invariance | Pass, byte-identical |

## Authorized Next Experiment

The next experiment is a matched MLX action-ranking prototype with one shared
state trunk and batched action edits.

At minimum it must compare:

1. a full exact afterstate control using the accepted R2 public substrate;
2. R3 radius-3 local patch plus exact global edits;
3. R3 radius-2 local patch plus the same exact global edits; and
4. R3 radius-1 local patch plus the same exact global edits.

All arms must share:

- identical public positions and complete legal action sets;
- identical targets, split, D6 schedule, optimizer steps, and initialization;
- matched model capacity within a preregistered tolerance;
- one state-trunk encoding per decision;
- exact global objects unless that object is the explicit ablation;
- identical serving batch shapes where the representation permits; and
- evaluation of value error, top-K recall, retained regret, actions per second,
  P50/P95/P99 latency, RSS, active memory, and swap.

Promotion beyond the prototype requires equal or better held-out decision
quality than the full-afterstate control and a material throughput or memory
gain at realistic action counts. A representation-speed win that loses
decision strength does not advance.

## Evidence

- Aggregate:
  `artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-forward.json`
- Reverse aggregate:
  `artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-reverse.json`
- Order proof:
  `artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-order-proof.json`
- Collection receipt:
  `artifacts/experiments/r3-action-edit-foundation-v1/reports/collection.json`
- Prelaunch determinism proof:
  `artifacts/experiments/r3-action-edit-foundation-v1/reports/prelaunch-smoke-proof.json`
- Immutable bundle:
  `artifacts/experiments/r3-action-edit-foundation-v1/bundles/24416fe767223fee6ca9e9cf2748fde45650bccd7526b78601f43c8490459b58/bundle.json`
- Queue specification:
  `artifacts/experiments/r3-action-edit-foundation-v1/queue-spec.json`
- ADR:
  `docs/v2/decisions/0148-r3-exact-action-local-patch-global-edit-foundation.md`
- Preregistration:
  `docs/v2/reports/r3-action-edit-foundation-v1-preregistration.md`

## Scientific Limitations

- This is an exact representation census, not a learned comparison.
- Packed codec bytes are a mechanical persistence measure, not direct MLX
  tensor bytes or neural FLOPs.
- Radius coverage measures direct changed coordinates. Long-range component
  and motif consequences remain represented by exact global objects.
- The open corpus is sufficient for foundation gates but is not a sealed
  gameplay test.
- No score, search, or progress-to-100 claim follows from this result alone.
