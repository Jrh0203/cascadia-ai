# Full-Legal Hierarchical Factor Oracle V1 Preregistration

Date: 2026-06-16

Experiment ID: `full-legal-hierarchical-factor-oracle-v1`

## Question

Can complete legal actions be recovered through hierarchical retrieval over
draft/premove, tile placement, and wildlife placement factors with at least
98% top-64 teacher recall and a materially smaller candidate set?

## Frozen Arms

| Arm | Draft | Tile | Wildlife | Conditioning |
|---|---:|---:|---:|---|
| `conditional-compact` | 4 | 8 | 2 | prefix-conditional |
| `conditional-balanced` | 8 | 16 | 4 | prefix-conditional |
| `conditional-wide` | 16 | 32 | 8 | prefix-conditional |
| `independent-wide` | 16 | 32 | 8 | global factors |

All arms audit all 560 train and 240 validation groups. Factor ranking uses
only frozen expected-rank teacher values to measure the structural ceiling.
Champion-frontier actions are always retained.

## Cluster Execution

One distinct arm per Mac, followed by dynamic cross-host replay. This
maximizes independent hypothesis throughput while preserving reproducibility
and avoiding fixed replay barriers.

## Decision Rule

`conditional-wide` must pass on train and validation:

- at least 98% target recall;
- at least 90% exact target sets;
- at least 99% winner retention; and
- at most 2,048 mean retained candidates.

Pipeline failure classifies `hierarchical_factor_oracle_invalid`; a valid miss
classifies `hierarchical_factor_oracle_insufficient`; all gates passing
classifies `hierarchical_factor_oracle_sufficient`.

No training, gradients, new teacher compute, sealed test, or gameplay.
