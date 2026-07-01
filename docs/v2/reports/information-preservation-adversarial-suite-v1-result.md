# Information-Preservation and Adversarial Suite

- Classification: `information_preservation_suite_failed`
- Scientific BLAKE3: `f5575ea0c3780894fda656fefd8bab9bb96e2fcc814bceca933002c87a04e178`
- Families: 14/14
- Ready pairs: 14
- Dependency-blocked pairs: 0
- Frozen probes: 11/11

## Classification

- The suite is complete and non-blocked: every frozen family has executable evidence.
- `failed` means at least one required representation boundary loses information; it does not mean fixture generation or validation failed.
- The exact public-observable boundary retains all fourteen families.

## Boundaries

| Boundary | Applicable | Retained | Lost | Collisions | Eq violations |
|---|---:|---:|---:|---:|---:|
| public-observable-v1 | 14 | 14 | 0 | 0 | 0 |
| v2-position-record-v1 | 0 | 0 | 0 | 0 | 0 |
| current-dataset-tensors-v1 | 0 | 0 | 0 | 0 | 0 |
| graded-oracle-raw-v1 | 0 | 0 | 0 | 0 | 0 |
| graded-oracle-factors-v1 | 0 | 0 | 0 | 0 | 0 |
| declared-compact-projection-v1 | 1 | 1 | 0 | 0 | 0 |
| absolute-opponent-order-v1 | 1 | 0 | 1 | 1 | 0 |
| focal-board-market-only-v1 | 1 | 0 | 1 | 1 | 0 |
| marginal-factor-scores-v1 | 1 | 0 | 1 | 1 | 0 |
| mean-max-descendant-v1 | 1 | 0 | 1 | 1 | 0 |
| public-refill-distribution-v1 | 1 | 1 | 0 | 0 | 0 |
| public-supply-marginals-v1 | 1 | 0 | 1 | 1 | 0 |
| radius-6-in-radius-only-v1 | 1 | 0 | 1 | 1 | 0 |
| scalar-tile-id-v1 | 1 | 0 | 1 | 0 | 1 |

## Pair Families

- `semantic_tile_multiset`: complete; exact fixture
- `multiplicity_descendant_distribution`: complete; exact fixture
- `long_salmon_component_context`: complete; exact fixture
- `focal_relative_opponent_order`: complete; exact fixture
- `d6_transforms`: complete; exact fixture
- `tile_id_permutation`: complete; exact fixture
- `component_bridge`: complete; exact fixture
- `equal_immediate_different_future_conflict`: complete; exact fixture
- `opponent_demand_seat_timing`: complete; exact fixture
- `public_action_equivalence_refill_near_match`: complete; exact fixture
- `same_factor_scores_different_joint_completion`: complete; exact fixture
- `ambiguous_confidence_set_vs_distinguishable_winner`: complete; exact fixture
- `same_in_radius_different_overflow_consequence`: complete; exact fixture
- `same_compact_latent_different_legal_affordance`: complete; exact fixture

## Resolved F1/F2/F3 Evidence

- F1: all four formerly blocked pairs are executable from exact Rust component, motif, score, transition, and public-supply receipts.
- F1 classification scientific BLAKE3: `f7f8559431f53a461f9464e14ef4cee2119cf3ddcf0bf4e3dd9126ab8bdd91fb`.
- F1 merged-census scientific BLAKE3: `8906487b91aa0da25f388e2075d15150c8d1499a022cbf2d231987b37f182e65`.
- F3: 12 exact transforms over 540 legal rows; every legal map is bijective, every selected action round-trips, and every transition is equivariant.
- F3 contract scientific BLAKE3: `db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f`.
- F2: both Rust boards contain 23 legal tiles, retain the same 13 radius-6 occupied cells and the same in-radius frontier, and carry different exact `Board::frontier` legal masks.
- Radius-6 in-radius-only projection: exact collision; legal-mask retention `0.0`.
- Radius-6 / 127-cell compact projection with exact overflow sidecar: legal-mask retention `1.0`.
- F2 source scientific BLAKE3: `c6076545aa93e78902b739eefef1545a23b8f2dbe44770f427a30969511800e5`.
