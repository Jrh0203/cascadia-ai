# CPU Pre-GPU Milestones

These milestones define the implementation runway up to the point immediately
before GPU work. They intentionally stop before GPU training, GPU verification,
large model fitting, inference serving, and strength testing.

## Milestone 0: Plan And Spec Package Complete

Deliverables:

- `README.md`
- `IMPLEMENTATION_PLAN.md`
- `SCHEMA_CONTRACTS.md`
- `CPU_PRE_GPU_MILESTONES.md`
- `PERFORMANCE_BUDGETS.md`
- `GPU_HANDOFF_GATE.md`

Exit gate:

- The package names the literature-first proposal as authority.
- CascadiaFormer-Zero, C-GAB, legal-action query head, value vector, score
  decomposition, chance-aware MCTS, search root tables, and per-action Q labels
  are imported.
- Radius 6 / 127 cells is canonical, with exact overflow and a future CPU
  coverage census requirement.
- GPU work is an explicit STOP gate.

## Milestone A: Clean Simulator Contract

Deliverables:

- deterministic legal action generator for full compound actions;
- exact scoring decomposition for wildlife, habitat, nature-token, total, rank,
  and score vector labels;
- chance API for market refill, wildlife bag draw, tile-stack reveal, and
  deterministic seed replay;
- canonical state serialization and stable state hash;
- symmetry transforms for hex rotations/reflections and player-seat
  orientation;
- golden tests for legal action enumeration, scoring, chance replay,
  serialization, and symmetry invariance.

CPU-only validation gates:

- fixed roots produce deterministic legal action lists in stable order;
- random games with fixed seeds are reproducible;
- score decomposition sums exactly to final score vector;
- chance transitions can be replayed from logged seeds/samples;
- symmetry transforms preserve legality and score labels.

Exit gate:

- The simulator contract is sufficient for tokenizer, replay, and search-root
  dry runs without referencing prior NNUE internals as governing design.

## Milestone B: Tokenizer And Schema Implementation

Deliverables:

- implementation of all token contracts from `SCHEMA_CONTRACTS.md`;
- exact action-token generation from simulator legal actions;
- C-GAB template construction for same board, same market slot,
  tile-wildlife pairing, adjacent direction, distance bucket, terrain
  continuity, same species, action-draft slot, and action-target coordinate
  relations;
- radius 6 coverage census command over representative CPU-generated or parsed
  games;
- exact overflow reporting for out-of-radius legal coordinates;
- schema fixtures and manifest round-trip tests.

CPU-only validation gates:

- 100 fixed roots tokenize deterministically;
- every `ActionToken` maps back to exactly one simulator legal action;
- C-GAB template ids are stable across repeated runs;
- radius 6 census reports total placements, in-radius placements, overflow
  placements, and example overflow coordinates;
- no legal state is clipped, projected, or dropped due to radius 6.

Exit gate:

- Tokenization, action schema, C-GAB relations, and overflow handling are ready
  for a tiny model smoke test.

## Milestone C: CPU-Only Model Smoke

Deliverables:

- minimal CascadiaFormer-S-compatible configuration or mock backend;
- tensor input contract for state tokens, action tokens, masks, and C-GAB
  relation templates;
- forward path on CPU that returns:
  - one legal-action query logit per legal action;
  - value vector output for four seats;
  - score/rank/vector auxiliary head tensors;
  - score decomposition head tensors;
- tiny deterministic fixture batch;
- optional one-batch CPU overfit check if the backend supports it locally.

CPU-only validation gates:

- shape checks pass for variable legal action counts;
- illegal/padded action positions are masked;
- output dimensions match `ModelConfig`;
- the smoke is documented as tensor plumbing only, not strength or training
  evidence.

Exit gate:

- Model input/output contracts are verified enough to hand to GPU work later.

## Milestone D: Chance-Aware Search Teacher Dry Run

Deliverables:

- chance-aware MCTS interface contract for tiny fixed roots;
- PUCT/Gumbel-root selection switch as a configuration field;
- bounded chance-node expansion policy in dry-run form;
- root table export containing legal actions, priors, visits, per-action Q,
  selected action, chance samples, final score vector, and score decomposition;
- replay shard manifest;
- replay load/round-trip validation.

CPU-only validation gates:

- tiny fixed roots export deterministic root tables under fixed seeds;
- every label array aligns one-to-one with legal actions;
- chance samples are logged and replayable;
- replay round trip preserves checksums and schema ids;
- root table JSONL/binary dry-run sizes are reported.

Exit gate:

- Search-root replay labels are ready for GPU-era model verification and later
  search-guided self-play.

## Stop: GPU Handoff Bundle

The CPU phase stops by producing:

- current schema docs;
- deterministic simulator test report placeholder or completed report;
- radius 6 coverage/overflow report placeholder or completed report;
- tiny corpus overfit plan;
- search-root replay contract and dry-run results;
- performance budget table and profiling notes;
- explicit approval question for the next GPU-only stage.

Do not proceed into MLX/Metal verification, RTX profiling, batched GPU inference
serving, self-play strength testing, or large model training without the next
approval gate.
