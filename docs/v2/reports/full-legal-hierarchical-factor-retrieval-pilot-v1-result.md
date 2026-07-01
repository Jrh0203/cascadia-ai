# Full-Legal Hierarchical Factor Retrieval Pilot V1 Result

Date: 2026-06-16

Experiment ID: `full-legal-hierarchical-factor-retrieval-pilot-v1`

Classification: **`hierarchical_proposal_insufficient`**

## Stage Results

| Stage | Train factor recall | Validation factor recall | Best epoch | Runtime |
|---|---:|---:|---:|---:|
| draft | 93.33% | 92.84% | 16 | 53.6 s |
| tile | 72.60% | 66.57% | 20 | 670.8 s |
| wildlife | 100.00% | 100.00% | 6 | 1774.7 s |

## Integrated Result

| Metric | Train | Validation | Gate |
|---|---:|---:|---:|
| Learned proposal target recall | 86.18% | 72.48% | >98% |
| Learned proposal winner retention | 99.29% | 92.08% | >98% |
| Mean proposal count | 1098.0 | 1061.7 | <=2,048 |
| Learned top-64 target recall | 34.95% | 18.14% | >98% |
| Learned top-64 winner recall | 89.11% | 58.75% | >98% |
| Learned top-64 mean R4800 regret | 0.030352 | 0.140092 | <0.15 |

The oracle-inside-proposal diagnostic isolates learned retrieval. The learned
top-64 result uses only the summed draft, tile, and wildlife model scores.

## Integrity

- Cache audit reproduced ADR 0114 exactly on john1 and john4.
- All cache shards preserve factor bijection and prefix invariants.
- All three selected checkpoints replayed bit-identically on another host.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Failed Gates

- `proposal_train_recall_above_0_98`
- `proposal_validation_recall_above_0_98`
- `proposal_validation_winner_retention_above_0_98`
- `proposal_early_recall_at_least_0_97`
- `proposal_late_recall_at_least_0_97`
- `proposal_middle_recall_at_least_0_97`
- `proposal_independent_draft_winner_recall_at_least_0_95`
- `proposal_nature_token_available_recall_at_least_0_95`
- `selector_train_recall_above_0_98`
- `selector_validation_recall_above_0_98`
- `selector_train_winner_recall_above_0_98`
- `selector_validation_winner_recall_above_0_98`
- `selector_early_winner_recall_at_least_0_97`
- `selector_late_winner_recall_at_least_0_97`
- `selector_middle_winner_recall_at_least_0_97`
- `selector_independent_draft_winner_winner_recall_at_least_0_95`
- `selector_nature_token_available_winner_recall_at_least_0_95`
- `proposal_passed`
- `selector_passed`

## Mechanistic Audit

- Exact model-input target conflicts: train
  `0`,
  validation
  `0`.
- Objective classification:
  `objective_gradient_conflict`; boundary versus
  combined auxiliary gradient cosine
  `-0.738910`.
- Boundary gradient norm
  `24.3817` versus combined
  auxiliary norm
  `28.0464`.
- Learned/screen top-32 oracle-union validation recall:
  `78.29%`.
- Tile-only mixed-stage validation action recall:
  `72.97%`; winner retention:
  `92.08%`.

The tile failure is not an exact-label collision or simple prior-blending
problem. The rank-regression pressure directly conflicts with the top-32
membership boundary, authorizing ADR 0116's target-only objective pilot.

## Cluster Throughput

- Cache queue: 10 unique shards in
  `48.03` seconds wall and
  `159.88` process-seconds.
- Cache work by host: `john1` 3, `john2` 2, `john3` 2, `john4` 3.
- `john3`: trained `draft` for 53.6 seconds.
- `john1`: trained `tile` for 670.8 seconds.
- `john2`: trained `wildlife` for 1774.7 seconds.
- Duplicate discovery training fraction: `0%`. Cross-host replays were required
  validation work, not model-selection replicas.
- Dashboard journal coverage: 66 samples. Core-weighted host CPU averaged 10.3% and peaked at 19.0%. Per node: `john1` 26.1% mean/38.0% peak, `john2` 5.7% mean/12.5% peak, `john3` 4.9% mean/20.0% peak, `john4` 4.5% mean/21.6% peak. CPU excludes MLX GPU occupancy.

The long wildlife rank-calibration job was the critical path. Other hosts were
backfilled with cache audit, replay, integration tooling, exact collision,
objective-gradient, complementarity, correctness, and successor-preparation
work. Raw CPU remained low during MLX-heavy windows because the dashboard does
not measure Metal occupancy; the campaign optimized distinct decisions rather
than duplicating the wildlife trainer.

## Decision

The learned proposal itself misses the Phase 2 gate. The completed mechanistic audit selects ADR 0116's target-only conditional tile objective as the one authorized successor.
