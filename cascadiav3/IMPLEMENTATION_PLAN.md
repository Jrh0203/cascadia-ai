# Cascadia v3 Implementation Plan

## Executive Outcome

Build Cascadia v3 as a clean-slate, literature-first transformer game AI plan
centered on **CascadiaFormer-Zero**: sparse game-native object tokens,
**C-GAB** dynamic relation bias, a **legal-action query** policy head, a
multiplayer **value vector**, **score decomposition** heads, and
**chance-aware MCTS** that records full search root tables with visit counts and
**per-action Q** labels.

This package stops at CPU-only readiness. The intended completed state before
GPU access is:

- deterministic simulator contract specified;
- tokenizer and schema contracts specified;
- replay and search-root records specified;
- tiny CPU model smoke test specified for tensor shapes and legal logits;
- deterministic validation harness specified;
- performance budgets and regression gates documented;
- GPU handoff criteria explicit.

No simulator, model, search, training, inference server, or strength test is
implemented by this package.

## Source Hierarchy

Governing source:
`/Users/johnherrick/cascadia/docs/v3/LITERATURE_FIRST_TRANSFORMER_GAME_AI_PROPOSAL_2026-06-29.md`

Trust order inherited from the proposal:

1. Peer-reviewed or major-conference game AI papers and official paper pages.
2. Official project writeups from strong open engines.
3. Reproducible open-source implementations and benchmark datasets.
4. Cascadia repo logs and local experiment notes as hypotheses only.

Imported proposal decisions:

- Build **CascadiaFormer-Zero** rather than extending NNUE.
- Use sparse entity tokens rather than fixed image planes or text serialization.
- Use **C-GAB** relation templates for Cascadia geometry, market coupling, turn
  order, supply, and scoring context.
- Use a **legal-action query** head over exact simulator-enumerated compound
  actions.
- Predict a multiplayer **value vector**, score/rank distributions, and
  **score decomposition** auxiliaries.
- Use **chance-aware MCTS** with bounded chance expansion and saved search root
  tables.
- Train from self-play/search-root replay labels, including visit
  distributions, **per-action Q** labels, selected actions, chance samples, final
  score vectors, and score decomposition.
- Use score/rank/vector auxiliary heads and later action-value distillation once
  a search teacher exists.

## Architecture Summary

**CascadiaFormer-Zero** is a sparse entity transformer for Cascadia state,
action, and score structure. The model consumes game-native tokens:
`GameToken`, `PlayerToken`, placed and market `TileToken`, placed and market
`WildlifeToken`, `FrontierToken`, `SupplyToken`, `ScoreToken`, and one
`ActionToken` per fully legal compound action.

**C-GAB** is the Cascadia Geometric Attention Bias layer. It starts from
rule/topology relation templates such as same board, same market slot,
tile-wildlife pairing, adjacent hex direction, distance bucket, terrain
continuity, same species, action-draft slot, and action-target coordinates. A
small generator mixes those templates into additive attention biases per layer
or layer group.

The **legal-action query** policy head scores exact legal actions produced by the
simulator. It avoids a giant fixed action index and keeps the model aligned with
Cascadia's compound action space: cleanup, nature-token spending, draft slot,
tile placement, rotation, wildlife placement, and refill consequences.

The value stack predicts:

- own final score distribution;
- final score vector for all four seats;
- rank distribution;
- pairwise score differential distributions;
- wildlife, habitat, nature-token, and score-to-go decomposition heads;
- optional future market and opponent next-draft auxiliaries.

Search uses **chance-aware MCTS** with legal action priors, value bootstrap,
chance nodes for market refill and supply draws, and progressive widening for
stochastic afterstates. Every training root should export a search root table,
not just the selected action.

## Canonical Board Scope

Radius 6 is the canonical board fast path. A radius 6 hex board has 127 cells
per player board. With four player boards, the planned fast-path board coordinate
space is 508 board cells plus exact overflow records for any legal state outside
the radius 6 envelope.

The 127-cell radius 6 fast path is a default engineering target, not a measured
coverage claim. A future CPU coverage census must report observed overflow rates
before any expansion to a larger canonical radius.

## Pre-GPU Scope

The pre-GPU implementation phase should deliver only CPU-verifiable contracts
and dry runs.

### CPU Simulator Contract

Required contracts:

- deterministic legal action generator;
- exact scoring and score decomposition;
- chance transition API for tile/wildlife supply and market refill;
- canonical state serialization and stable state hash;
- four-seat value-vector orientation;
- symmetry transforms for hex rotations/reflections;
- golden tests for legal moves, scoring categories, supply mutation, and
  serialization round trips.

Exit gate: random and fixed-seed games are reproducible, legal, exactly
scoreable, and hash-stable under the documented serialization.

### Schema Contracts

Required contracts:

- radius 6 `CanonicalHexCoord` fast-path membership;
- exact overflow coordinate entity;
- all token schemas;
- action-token schema for exact compound legal actions;
- C-GAB template schema;
- search-root record schema;
- replay shard manifest schema;
- model config schema for S/M/L;
- validation gate registry.

Exit gate: schema fixtures serialize and deserialize without loss, and schema ids
are embedded in every replay/search-root artifact.

### Replay And Search-Root Dry Run

Required dry run:

- build a tiny fixed corpus of legal root states;
- enumerate legal actions on CPU;
- attach dummy priors, visits, **per-action Q** labels, selected action, chance
  samples, final score vector, and score decomposition;
- export root tables;
- load them back and verify checksums, record counts, schema id, and action
  alignment.

Exit gate: replay round trip is deterministic and rejects schema/action-count
mismatches.

### Tiny CPU Model Smoke

Required smoke specification:

- instantiate a minimal CascadiaFormer-S-compatible shape or mock backend on
  CPU;
- accept token tensors, action-token tensors, C-GAB template ids, and masks;
- produce one legal logit per legal action;
- produce value vector and score/rank/vector auxiliary outputs with expected
  dimensions;
- run forward-only shape checks and a tiny overfit-on-one-batch test only if the
  chosen backend can do so on CPU.

This is not a strength check. It verifies tensor plumbing, legal action logits,
masking, and output head dimensionality.

### Deterministic Validation Harness

Required harness:

- fixed seeds;
- fixed root fixtures;
- golden output files with schema versions and checksums;
- command registry for CPU-only gates;
- no network dependency;
- no GPU requirement flag on pre-GPU commands.

## Out-of-Scope GPU Gate

The following are explicitly outside this package and require the next approval
gate:

- MLX/Metal training or verification;
- RTX profiling;
- batched GPU inference serving;
- self-play strength testing;
- CascadiaFormer-M or CascadiaFormer-L training;
- large self-play/search generation.

## Milestone Map

| Stage | Scope | Exit Evidence | GPU Required |
|---|---|---:|---:|
| 0 | Formal plan/spec package | six `cascadiav3/` docs present | No |
| A | Simulator contract | deterministic legal/scoring/chance/golden test report | No |
| B | Tokenizer and schema | schema fixtures, C-GAB templates, radius 6 census | No |
| C | Tiny CPU model smoke | tensor shapes, masks, legal logits, value vector outputs | No |
| D | Search teacher dry run | tiny root tables and replay round trip | No |
| Handoff | GPU decision package | complete CPU artifact checklist | No |
| Later GPU | S/M/L training, serving, profiling, strength testing | new approval required | Yes |
