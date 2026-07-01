# ADR 0128: Exact State-Footprint Census

Status: preregistered

Date: 2026-06-16

Experiment ID: `state-footprint-census-v1`

## Context

The Cascadia research implementation plan makes compact state representation
the first decision of the next architecture program. The current and legacy
code contains three relevant facts:

- the historical dense lattice contains 441 indexed cells;
- the current V2 rules board uses a 49x49, 2,401-cell backing grid while
  separately tracking at most 23 occupied indices;
- complete centered radius-6 and radius-5 hex disks contain 127 and 91 cells;
- a legacy 50,000-state analysis reported 99.9% and 99.7% per-cell firing
  retention at those radii.

There is no complete centered hex disk containing exactly 121 cells. A
centered radius-`r` disk contains `1 + 3r(r + 1)` cells, producing 91 cells at
radius 5 and 127 at radius 6. Any historical 121-cell representation would
therefore have been an irregular crop, a padded layout, or a different
indexing convention. No current repository artifact establishes such a
representation as sufficient.

The legacy V6 result cannot settle the footprint question. It changed spatial
support, feature layout, feature content, model shape, and training path at
the same time. Its 1.80x wall-time improvement establishes leverage, while its
small strength regression establishes the need for a controlled tournament
with exact overflow semantics.

The later R0 control is therefore the exact, untruncated V2 coordinate/entity
representation, not the historical 441-cell square and not a 2,401-row dense
neural tensor. The historical 441 layout remains a diagnostic arm with exact
overflow handling.

Before implementing R0 models, the project needs one reproducible census of
occupied cells, frontiers, action destinations, feature firings, out-of-region
effects, sparse token counts, and adversarial boards in the current V2 rules
engine.

## Decision

Implement and run one deterministic Rust census with two disjoint evidence
arms:

1. a generated open-foundation corpus containing at least 50,000 pre-move
   four-player AAAAA states; and
2. every unique decision group in the open train and validation
   complete-action graded-oracle datasets.

The generated origin uses public state only during observation. Final scores
may label high- and low-score cohorts after a game finishes, but no future
tile order, wildlife order, or hidden refill state may enter any spatial
measurement.

For every generated state, inspect all four boards. For every graded-oracle
group, inspect each serialized board perspective once and every distinct
candidate tile destination without counting candidate rows as independent
states.

The census reports fixed-origin and best integer-recentered support. It must
not use recentering to hide a required public coordinate transform: the
original coordinates, selected center, and translation must remain exactly
recoverable.

## Frozen Generated Domain

- rules: four-player AAAAA, habitat bonuses disabled;
- first raw seed: `73000`;
- games: `625`;
- expected pre-move states: `50,000`;
- strategy: current V2 `pattern-aware`;
- all four absolute and focal-relative seats;
- opening, early, middle, and late phases;
- final-score cohorts attached only after simulation;
- no gameplay-strength claim.

The 625-game origin runs on john4. It is unique evidence, not a replica.

## Frozen Open-Corpus Domain

Read and validate:

- `artifacts/datasets/complete-action-graded-oracle-v1-train`; and
- `artifacts/datasets/complete-action-graded-oracle-v1-validation`.

The test split remains closed. The open-corpus arm runs on john1 and must
verify every manifest and shard checksum before interpretation.

## Required Measurements

For occupied cells, legal frontiers, selected tile destinations, and
complete-action candidate tile destinations:

- total events, maximum radius, and radius histogram;
- retained and overflow counts at radii 3 through 8;
- fraction retained and boards or groups with any overflow;
- fixed-origin and exact best integer-recentered radius;
- phase, seat, final-score, and wide-board cohorts where applicable.

For each radius, also report:

- centered disk capacity;
- wildlife firing retention;
- terrain-edge firing retention;
- allowed-wildlife firing retention;
- habitat components crossing the boundary;
- wildlife adjacencies crossing the boundary;
- legal-frontier effects outside the disk;
- dense cell and byte estimates;
- sparse occupied-plus-frontier token counts;
- canonical public-state serialization bytes;
- extraction time separately from simulation time.

Every radius-6-or-larger overflow state must be listed without truncation,
including seed, turn, current player, absolute seat, focal-relative seat,
public-state hash, maximum coordinates, and the overflowing coordinates.

The report must include legal elongated straight and bent board constructions
that deliberately exceed radius 6. These adversarial cases establish that an
overflow object is mandatory even if empirical retention is high.

## Geometry Contract

The implementation must prove:

- `cells(r) = 1 + 3r(r + 1)`;
- radius 4, 5, and 6 contain 61, 91, and 127 cells;
- no integer radius produces 121 cells;
- all 12 D6 transforms preserve hex radius;
- recentering is deterministic and exactly invertible;
- merging disjoint corpus accumulators is deterministic;
- the scientific hash excludes timestamps and output paths.

## Gates

Classify `state_footprint_census_complete` only when:

- generated pre-move states are at least 50,000;
- every generated state contributes all four boards;
- both open graded-oracle manifests and all shards validate;
- every unique open decision group is counted exactly once;
- radii 3 through 8 have complete measurement tables;
- no radius-6-or-larger outlier list is truncated;
- adversarial straight and bent boards both overflow radius 6;
- D6 radius and deterministic-merge tests pass;
- all counts are finite, internally consistent, and checksum-bound;
- source, executable, inputs, and report identities are recorded.

Otherwise classify `state_footprint_census_incomplete` and do not authorize R0
training.

## Interpretation

This census may authorize which supports enter R0, but it cannot promote a
representation. In particular:

- high empirical retention does not permit silent dropping;
- 127 is not privileged merely because the legacy V6 used it;
- 91 remains an arm only with exact overflow semantics;
- 61 remains an aggressive diagnostic arm;
- sparse occupied-plus-frontier remains mandatory;
- strength and decision quality are measured only in the later controlled R0
  tournament.

## Cluster Execution

- john4 runs the unique 625-game generated origin;
- john1 scans the open graded-oracle train and validation corpora;
- john1 collects the remote generated report and mechanically combines both
  arms;
- john2 and john3 continue the already running closure experiments until their
  dependencies release them.

The queue may move checksum, collection, or report work to another compatible
host, but it may not duplicate the generated origin.

## Maximum Compute

One release build, focused tests, one 625-game generated origin, one open
train/validation corpus scan, one checksum-verified collection, one combined
report, and one documentation update. No MLX training, teacher rollout,
sealed test, gameplay pilot, cloud, Modal, or external compute.
