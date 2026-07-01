# ADR 0131: Information-Preservation and Adversarial Suite

Status: preregistered

Date: 2026-06-16

Experiment ID: `information-preservation-adversarial-suite-v1`

## Context

F4 of the research implementation plan requires a permanent diagnostic that
can identify representation collisions before model training or gameplay.
The feature representation audit identifies the current failure modes:

- aggregate public supply aliases distinct semantic tile multisets;
- mean/max pooling aliases multiplicity and descendant distributions;
- radius-one features omit long components and wildlife motifs;
- perspective, D6, and arbitrary tile-ID behavior are not uniformly exact;
- staged markets, action edits, opponent demand, and legal affordances are
  compressed before final action comparison; and
- teacher ambiguity is not a top-one label.

A useful diagnostic cannot treat an unavailable exact fact as a negative or a
pass. Rust remains authoritative for game legality, public transitions,
components, motifs, D6 transforms, and compact-state overflow semantics.
Python may inspect serialized public records and existing open teacher
evidence, but it may not invent those missing facts.

## Decision

Implement one schema-versioned Python framework and CLI in
`python/cascadia_mlx/information_preservation_suite.py`.

Every adversarial pair contains:

- a stable pair and family identifier;
- two public-observable inputs;
- exact `equivalent` or `different` expectations by named concept;
- provenance and evidence-domain metadata;
- representation-boundary observations;
- an optional named dependency block with an executable completion contract;
- a canonical scientific BLAKE3.

The scientific digest uses canonical JSON and excludes filesystem paths,
timestamps, hostnames, output destinations, and the digest field itself.
Arrays are encoded with explicit dtype, shape, and values. NaN and infinity
are invalid.

The runner supports registered boundary adapters and frozen probes without
changing its core. Built-in adapters cover:

- exact public-observable mappings;
- current V2 position records and decoded dataset tensors;
- complete-action graded-oracle raw public inputs;
- graded-oracle typed factor tensors;
- declared compact projections; and
- future plugin boundaries registered by stable identifier.

The frozen probe registry always contains:

1. occupancy;
2. frontier;
3. component;
4. motif;
5. exact supply;
6. staged market;
7. action edit;
8. opponent demand;
9. D6 identity;
10. legal mask;
11. confidence-set membership.

An unavailable probe emits a blocked record with a named dependency. It is
never omitted and never counted as retained.

## Frozen Pair Families

The v1 suite contains exactly the 14 F4 families:

1. `semantic_tile_multiset`;
2. `multiplicity_descendant_distribution`;
3. `long_salmon_component_context`;
4. `focal_relative_opponent_order`;
5. `d6_transforms`;
6. `tile_id_permutation`;
7. `component_bridge`;
8. `equal_immediate_different_future_conflict`;
9. `opponent_demand_seat_timing`;
10. `public_action_equivalence_refill_near_match`;
11. `same_factor_scores_different_joint_completion`;
12. `ambiguous_confidence_set_vs_distinguishable_winner`;
13. `same_in_radius_different_overflow_consequence`;
14. `same_compact_latent_different_legal_affordance`.

Families 1, 2, 4, 6, 9, and 11 use exact deterministic public fixtures.
Family 12 uses existing open validation R4800 evidence selected by the frozen
rule in the preregistration. No new teacher computation is authorized.

The remaining families are dependency-blocked until their canonical
authorities exist:

- F1: exact component, motif, public-transition, and action-edit target
  exporters for families 3, 7, 8, and 10;
- F2: exact overflow-consequence and compact-arm fixtures for families 13 and
  14;
- F3: the shared Rust D6 transform and action-bijection fixture for family 5.

These records include executable contracts and cannot classify as passed.

## Evidence Boundary

Allowed evidence:

- deterministic synthetic public values whose claimed relation is arithmetic,
  set-theoretic, or a direct schema fact;
- current V2 position and graded-oracle public tensors;
- semantic tile identities already established by the catalog audit;
- open graded-oracle train or validation teacher estimates.

Forbidden evidence:

- sealed test records;
- hidden tile or wildlife order;
- future refill realization;
- terminal outcomes unavailable at the represented decision;
- newly generated rollout or teacher labels;
- gameplay, cloud, Modal, or external compute.

The validator recursively rejects hidden or future fields in public inputs.
Teacher evidence may appear only in a separate evidence object with open
dataset provenance and is never projected as model input.

## Classification

The suite classifications are:

1. `information_preservation_suite_passed`
   - all 14 families are present;
   - no required concept or probe is absent;
   - no pair is dependency-blocked;
   - every required boundary assertion passes.
2. `information_preservation_suite_dependency_blocked`
   - all 14 families are present;
   - every unavailable fact is tied to F1, F2, or F3 with an executable
     contract;
   - no blocked pair is reported as passed.
3. `information_preservation_suite_failed`
   - all families exist, but a ready exact assertion fails.
4. `information_preservation_suite_invalid`
   - a family, required concept, probe, schema field, provenance rule, or
     canonical hash is missing or invalid.

CLI exit codes are 0, 2, 3, and 4 respectively. A blocked or incomplete suite
therefore cannot produce a false-success process status.

This classification describes the diagnostic corpus and a selected boundary.
Known lossy boundaries are expected to fail individual concepts; that result
does not invalidate the framework.

## Outputs

The CLI emits:

- canonical JSON summary;
- Markdown summary;
- optional canonical JSONL pair details;
- boundary collision, separability, equivalence, concept-retention, legal-mask
  retention, and probe-status metrics;
- one deterministic suite scientific BLAKE3.

Output paths and wall-clock metadata do not affect the digest.

## CI And Machine Allocation

Focused CI validates the schema, hashes, fixtures, plugins, probes, and all 14
family IDs on every change.

Future unique evidence shards are allocated without duplicate origins:

- john1: fixture/schema validation and V2/raw boundary shard;
- john2: supply, pooling, hierarchy, and compact-projection shard;
- john3: F3 D6 identity and action-bijection shard;
- john4: component, motif, opponent-demand, legal-mask, and confidence-set
  shard.

One central deterministic merge compares shard scientific hashes. Replays are
allowed only after unique evidence is complete. This ADR does not enqueue or
launch any task.

## Promotion Rule

No representation advances to R0 or later model work after losing a required
exact concept. A dependency block authorizes only completion of its named
F1/F2/F3 prerequisite. It does not authorize a toy substitute, model training,
teacher rollout, or gameplay.
