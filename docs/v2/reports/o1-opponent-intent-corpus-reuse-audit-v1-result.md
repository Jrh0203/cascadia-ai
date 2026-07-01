# O1 Opponent-Intent Corpus Reuse Audit v1 Result

**Date:** 2026-06-17  
**Experiment:** `o1-opponent-intent-corpus-reuse-audit-v1`  
**Classification:** `exact_replay_foundation_reusable_policy_holdout_required`

## Verdict

The existing canonical action-imitation train and validation corpora are
authorized as an exact sequential foundation for O1 opponent-intent and
future-market-access research.

They are not authorized as final O1 training or promotion evidence. Both
splits use one teacher strategy and one checkpoint, so policy-held-out
generalization, checkpoint-shortcut resistance, and strategy-switch prediction
remain untested.

## Exact Replay

The john4 primary and john2 replay independently reconstructed:

| Measure | Result |
|---|---:|
| Complete games | 80 |
| Sequential positions | 6,400 |
| Candidate actions | 409,600 |
| Exact position-byte checks | 6,400 / 6,400 |
| Exact candidate-action hashes | 409,600 / 409,600 |
| Exact selected labels | 6,400 / 6,400 |
| Exact state transitions | 6,400 / 6,400 |
| Terminal games | 80 / 80 |
| Train/validation overlap checks | 0 on all four keys |

Both hosts produced scientific BLAKE3
`43e246ba50c4dac31b1d9f174acddd1ee4cbc70e14169ff7dbca254c24db0c4e`.
The terminal classification BLAKE3 is
`96e34990af5f2db805d9fa937377e9c5e46cb969e14f9845f8843cafe30a602d`.

## What The Sequence Recovers

Deterministic replay recovered all unique market tile IDs, every selected tile
and wildlife species, every post-action market, and complete public recent
draft history.

Semantic snapshots alone would not have been exact:

| Ambiguity | Positions | Ambiguous occurrences |
|---|---:|---:|
| Duplicate tile semantics | 82 | 164 |
| Duplicate tile + wildlife pair semantics | 20 | 40 |

This confirms that the engine-replayed physical tile identity is necessary for
honest survival labels.

## Future-Access Signal

The audit derived 6,080 focal post-action windows and 24,320 tile-level
survival labels:

| Target | Count | Rate |
|---|---:|---:|
| Exact tile survives three intervening opponents | 11,199 | 46.05% |
| Exact tile and public wildlife species survive | 8,892 | 36.56% |

The target is neither trivial nor vanishing. Roughly half of post-action tiles
remain available at the focal player's next access, while wildlife-pair
continuity is materially harder. This supports a compact O1 pilot that predicts
opponent consumption and future access from exact public state and recent
action history.

## Hypothesis Resolution

- **H1 passed:** split seed, game index, and compact selected actions exactly
  reproduce every state and candidate.
- **H2 passed:** replay recovers unique tile identity and exact survival labels
  through all three opponent turns.
- **H3 passed:** train and validation have zero overlap in group IDs, position
  records, public states, and initial hidden states.
- **H4 passed:** semantic-only snapshots are ambiguous in a measurable number
  of positions.
- **H5 passed:** same-policy foundation reuse is valid, while final O1 evidence
  still requires policy-held-out collection.

## Launch Integrity

Two fail-closed infrastructure repairs preceded the valid launch:

1. ADR 0183 separated portable immutable-dataset validation from
   collector-local weight-file validation.
2. ADR 0184 moved host-local dataset roots out of scientific identity and
   namespaced all launch artifacts by immutable bundle ID.

Neither invalid launch changed the frozen datasets, hypotheses, metrics, or
gates.

## Authorized Successor

The immediate successor is a multi-policy sequential corpus with:

- multiple opponent policy families and checkpoints;
- policy-family-held-out validation and test splits;
- exact tile identity, market transitions, next-pick labels, and survival
  windows;
- strategy-switch targets where a policy changes behavior;
- policy identity retained only as provenance, never as an observable feature.

The existing corpus may be used to build and test the exact derivation,
representation, and MLX training pipeline before the policy-held-out corpus is
opened.

## Artifacts

- Terminal classification:
  `artifacts/experiments/o1-opponent-intent-corpus-reuse-audit-v1/classification.json`
- Valid immutable launch:
  `artifacts/experiments/o1-opponent-intent-corpus-reuse-audit-v1/launches/9666a78a7110056558ec608f3346046c15157f2019f12580a8976946fd4bd6b7`
- Preregistration:
  `docs/v2/reports/o1-opponent-intent-corpus-reuse-audit-v1-preregistration.md`
- Decision:
  `docs/v2/decisions/0182-o1-opponent-intent-corpus-reuse-audit.md`
- Portability repair:
  `docs/v2/decisions/0183-portable-imitation-dataset-validation.md`
- Scientific-path repair:
  `docs/v2/decisions/0184-o1-cross-host-scientific-path-normalization.md`
