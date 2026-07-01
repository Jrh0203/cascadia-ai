# Information-Preservation and Adversarial Suite V1 Preregistration

Date: 2026-06-16

Experiment ID: `information-preservation-adversarial-suite-v1`

ADR 0131 implements F4 as permanent production diagnostics. The suite measures
whether exact public concepts survive each representation boundary before any
new model is trained or evaluated in gameplay.

## Frozen Evidence Domains

The suite uses only:

- deterministic public fixture values;
- the current V2 position-record and decoded tensor contracts;
- complete-action graded-oracle raw public fields and typed factor tensors;
- semantic tile facts already documented by the catalog collision audit; and
- existing open graded-oracle validation R4800 estimates.

The sealed test split remains closed. No gameplay, teacher rollout, new label
generation, cloud, Modal, or external compute is permitted.

Public inputs may contain only facts observable at the represented decision.
Hidden bag order, hidden refill order, future realized market, future actions,
terminal outcomes, and rollout trajectories are rejected recursively. Existing
teacher estimates are evidence, not public model input, and must retain their
open dataset, group, candidate, sample-count, and action-hash provenance.

## Frozen Family Registry

The required family order is:

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

Each family has one immutable v1 fixture record. Additional fixtures may be
added only in a new fixture-set version; they may not replace or reinterpret a
v1 record.

## Exact Ready Fixtures

### Semantic tile multiset

Use the catalog-audit collision:

```text
{ID 1: Mountain/Hawk keystone, ID 24: River/Bear keystone}
{ID 2: Mountain/Bear keystone, ID 20: River/Hawk keystone}
```

The current 30-value aggregate supply signature is equivalent. The semantic
tile multiset and refill distribution are different.

### Multiplicity and descendant distribution

Use the exact distributions:

```text
{0.0, 0.0, 1.0, 1.0}
{0.0, 0.5, 0.5, 1.0}
```

Minimum, mean, and maximum are equivalent. Multiplicity and descendant
distribution are different.

### Focal-relative opponent order

Use one fixed absolute four-seat public opponent table under two focal seats.
The absolute table is equivalent; the clockwise focal-relative order is
different and is computed by `(absolute_seat - focal_seat) mod 4`.

### Tile-ID permutation

Use two equal semantic tile multisets with stable IDs permuted between semantic
rows. Tile semantics and legal semantic affordances are equivalent; scalar ID
content is different.

### Opponent demand and seat timing

Use equal focal board and market values with two exact public opponent-demand
tables and different turns-until-action vectors. Focal state and market are
equivalent; item pressure and survival order are different.

### Factor scores and joint completion

Use equal draft, tile, and wildlife marginal factor scores with different
public compatibility matrices. Marginal factor scores are equivalent; the
joint completion set is different.

### Teacher ambiguity

Read only
`artifacts/datasets/complete-action-graded-oracle-v1-validation`, verify its
manifest and shard checksums, and inspect groups in ascending `group_id`.

Using R4800-labeled candidates only:

- winner ordering is descending mean, then ascending action hash;
- standard error is `stddev / sqrt(samples)`;
- two candidates are in the same 95% confidence set when their difference from
  the winner is at most `1.959963984540054 * hypot(se_winner, se_candidate)`;
- a winner is distinguishable when its margin over the runner-up is strictly
  greater than the same threshold.

Select the first group with confidence-set size at least two and the first
later group with a distinguishable winner. Preserve only public input plus the
minimum open teacher evidence needed to prove those classifications. If either
group does not exist, family 12 is invalid, not synthetically replaced.

## Frozen Dependency Blocks

The following contracts remain present and blocked until exact upstream
artifacts exist.

| Family | Dependency | Executable release condition |
|---|---|---|
| Long Salmon/component context | F1 | Rust exports exact component and motif identities for a pair with equal radius-one public neighborhoods |
| D6 transforms | F3 | Shared Rust fixture exports all 12 transformed states/actions, inverse IDs, and bijective legal masks |
| Component bridge | F1 | Rust exports pre/post habitat component IDs and exact bridge merge targets |
| Equal immediate/different future conflict | F1 | Rust exports exact Hawk or Salmon conflict targets independent of future realization |
| Public-action equivalence/refill near-match | F1 | Rust public-transition canonicalizer proves an equivalent pair and a refill-divergent near-match without hidden refill order |
| In-radius/overflow consequence | F2 | State-footprint census exports equal in-radius states with exact different overflow concepts or legal effects |
| Compact latent/legal affordance | F2 | A declared compact arm exports latent target identity and exact Rust legal masks |

Each blocked fixture names the expected artifact schema and validation command.
The runner records the block and returns a nonzero status. No Python-only D6,
hand-written legality, fabricated motif, or toy latent may satisfy a contract.

## Boundaries And Probes

Required boundary adapters:

- public observable;
- V2 position record;
- current dataset tensors;
- graded-oracle raw public representation;
- graded-oracle factor representation;
- declared compact projection;
- plugin registry.

Required frozen probes:

- occupancy;
- frontier;
- component;
- motif;
- exact supply;
- staged market;
- action edit;
- opponent demand;
- D6 identity;
- legal mask;
- confidence-set membership.

Every probe emits `observed`, `blocked`, or `unsupported`. Missing output is
invalid. Blocked and unsupported outputs never count as retained.

## Metrics

For every boundary:

- exact projection collisions;
- exact-equivalence violations;
- exact-difference separability;
- concept retention by concept and family;
- legal-mask retention;
- pair separability;
- blocked and unsupported probe counts;
- deterministic projection and observation signatures.

The summary reports family completeness separately from boundary quality.

## Scientific Hash

Canonical scientific JSON:

- sorts object keys;
- preserves list order;
- normalizes dataclasses, arrays, numpy scalars, and finite floats;
- excludes keys named `canonical_hash`, `scientific_blake3`, `generated_at`,
  `created_at`, `updated_at`, `timestamp`, `hostname`, `host`, `path`,
  `output_path`, and `output_dir`;
- rejects NaN and infinity;
- hashes UTF-8 canonical JSON with BLAKE3.

Running the same suite with different output paths or timestamps must produce
the same digest.

## Mechanical Success

The F4 framework is complete only when:

- all 14 family IDs are present exactly once;
- every ready fixture validates and hashes reproducibly;
- every unavailable family is honestly blocked by F1, F2, or F3;
- all 11 probes are registered and explicit;
- missing families, concepts, probes, or required fixture evidence produce an
  invalid classification and nonzero exit;
- blocked dependencies cannot classify as passed;
- JSON, Markdown, and optional JSONL outputs are deterministic; and
- focused pytest and repository ruff checks pass.

The initial expected classification is
`information_preservation_suite_dependency_blocked`. That is a successful
implementation of the diagnostic corpus, not a claim that the blocked game
facts or any candidate representation have passed.

## Machine Allocation Plan

No task is launched by this preregistration.

When the unique fixture/probe shards are scheduled:

- john1 owns schema, V2/raw, and deterministic merge evidence;
- john2 owns supply, pooling, hierarchy, and compact projection evidence;
- john3 owns the unique F3 D6 fixture and action-bijection evidence;
- john4 owns component, motif, opponent-demand, legal-mask, and confidence-set
  evidence.

Origins are unique. Cross-host replay begins only after all unique evidence is
complete and is checksum comparison, not replacement evidence.
