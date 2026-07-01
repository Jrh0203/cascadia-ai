# Exact Semantic Supply V1 Result

Date: 2026-06-17

Experiment ID: `exact-semantic-supply-v1`

Classification: **`exact_semantic_supply_census_complete`**

## Result

The source-frozen four-host census validated exact public semantic supply on
500 complete four-player AAAAA games:

| Measurement | Result |
|---|---:|
| Train games / positions | 400 / 32,000 |
| Validation games / positions | 100 / 8,000 |
| Total positions | 40,000 |
| Unique exact supply states | 40,000 |
| Accepted shards | 8 / 8 |
| Canonical archetypes | 75 |
| Official physical tiles | 85 |
| Minimum / maximum unseen tiles | 2 / 81 |
| Minimum / maximum drawable tiles | 0 / 79 |
| Hidden setup exclusions | exactly 2 |
| Forward/reverse aggregate bytes | identical |

Scientific BLAKE3:

```text
44f2d8b6f6ab4d6f2f6920f2846ea4115693415f332b390a6f8f0cf4c45f589d
```

Catalog BLAKE3:

```text
362a1f090066f537fc29398fdc464f667b7e106889feff8a77607e35dd015c19
```

## Gates

Every accepted position passed:

- official physical and semantic inventory conservation;
- exact drawable and hidden-exclusion conservation;
- parity with all 30 legacy public supply marginals;
- canonical supply and refill serialization round trips;
- exact one- through four-slot refill identities and normalization;
- hidden redeterminization invariance;
- invariance under all 12 D6 transforms;
- market-to-archetype and rotation-reference validation; and
- explicit separation of the frozen legacy-marginal collision witness.

The frozen witness has identical legacy marginals for physical tile pools
`[0, 23]` and `[2, 20]`, but exact semantic archetype multisets `[26, 72]` and
`[24, 74]`.

## Provenance

- Immutable bundle:
  `e3f3481562a63c6ec4c688aa378a5877f1f98a43652ab780488505c4bba357c8`
- Source BLAKE3:
  `578752186be573f446cefa380e79bcc3967fd3d1d823d63bde3f84f3cba6b3fc`
- Executable BLAKE3:
  `09afc254d763e813b224e1ac8f4abdb4fababb6d2828eae1276de403f3e3f789`

The bundle was whole-tree verified on john1, john2, john3, and john4. Four
train and four validation partitions were disjoint, checksum-collected, and
merged in both directions.

## Decision

Promote exact semantic supply as factual V2 infrastructure. The learned S1
comparison must retain the legacy 30-value marginals as C0 and measure the
incremental value of the 75-archetype treatment independently.

No score gain is claimed by this foundation census.

## Artifacts

- `artifacts/experiments/exact-semantic-supply-v1/reports/source-frozen-aggregate-forward.json`
- `artifacts/experiments/exact-semantic-supply-v1/reports/source-frozen-aggregate-reverse.json`
- `artifacts/experiments/exact-semantic-supply-v1/reports/source-frozen-bundle-fanout.json`
- `artifacts/experiments/exact-semantic-supply-v1/queue-spec.json`
