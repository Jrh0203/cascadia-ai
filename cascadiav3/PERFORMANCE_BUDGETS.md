# Cascadia v3 Performance Budgets

This document defines expected scale, complexity, and CPU acceptance budgets for
the pre-GPU implementation runway. It is a planning artifact; it does not assert
measured performance.

## Performance-Sensitive Paths

| Path | Why It Matters | Primary Risk | Pre-GPU Budget/Gate |
|---|---|---|---|
| Legal action enumeration | Every search root and action-token batch depends on exact legal action lists. | Variable legal action count can dominate latency. | 100 fixed roots enumerate deterministically with counts reported. |
| Tokenization | Converts simulator state into model input. | Rebuilding all objects naively can inflate CPU latency. | Schema/tokenization smoke on 100 fixed roots with wall-time and allocation report. |
| C-GAB relation matrix/template construction | Dense relation materialization can exceed memory before model work begins. | `O(tokens^2)` pair templates if built densely. | Report sparse edge count, dense-equivalent size, and peak memory. |
| Action-token cross-attention | The legal-action query head grows with legal action count. | Large legal action sets increase latency and memory. | Report min/median/max legal actions and padded action tensor size. |
| Chance-node expansion | Market refill and supply draws can branch heavily. | Unbounded chance-node children dominate search. | Dry-run bounded expansion counts and rejected overflow counts. |
| MCTS simulation loop | Future teacher cost is roughly simulations times expansion plus model eval. | Search can become latency-bound even with a fast model. | Tiny fixed-root dry run reports simulations, expansions, and per-root timing. |
| Replay serialization | Search root tables store all legal candidate action labels. | Root tables can be much larger than game records. | JSONL audit shards and compact tensor dry-run size estimates with checksums. |
| Future batched inference server | Search will require high throughput and low queue latency later. | Poor batching can waste GPU/CPU and stall simulations. | Pre-GPU phase only defines interface; no network/fleet traffic required. |

## Expected Scale

| Item | Planning Scale |
|---|---|
| Players | Four boards per game state. |
| Board fast path | Radius 6 canonical board, 127 cells per board. |
| Four-board fast path cells | 508 canonical cells, plus exact overflow entities. |
| Overflow | Legal out-of-radius states represented exactly and reported by census. |
| Legal actions | Variable count; exact simulator-owned list per root. |
| Model S | CascadiaFormer-Zero-S, 8 layers, `d_model` 384 or 512, 8 heads. |
| Model M | CascadiaFormer-Zero-M, 12 layers, `d_model` 768, 12 heads, roughly 80M-120M params. |
| Model L | CascadiaFormer-Zero-L, 15 layers, `d_model` 1024, 16 or 32 heads. |
| Search simulations | Future hypothesis ranges only; pre-GPU dry runs use tiny fixed roots. |

## Complexity Notes

- Dense transformer attention is `O(tokens^2 * layers)`. Token count control is
  a first-order design constraint.
- C-GAB construction can be sparse, but any dense matrix fallback has
  `tokens^2` memory behavior.
- Action-token scoring grows with legal action count and model depth.
- MCTS cost is roughly `simulations * (legal expansion + model eval + chance expansion)`.
- Chance-aware MCTS must bound stochastic afterstates; unbounded chance-node
  expansion is a correctness and latency risk.
- Search root replay cost grows with the number of legal action labels saved per
  root, not only with the number of game states.

## Memory And Network Implications

Root tables can be larger than ordinary game records because they retain all
legal candidate action labels, visit counts, priors, per-action Q labels, chance
samples, final score vector, and score decomposition. Replay shards therefore
need:

- schema ids;
- per-record or per-shard checksums;
- compact action ids where possible;
- JSONL dry-run format for inspection;
- compact tensor shards before large-scale generation;
- manifest-level record counts and scientific eligibility flags.

For greedy behavior-cloning pretraining, the default large-corpus format is now
Rust-native `float16` `.npz` tensor shards. The 2026-06-30 `john0` benchmark
measured a 1,024-game Rust-native deflated shard at 1:35.36 wall and 71,413,962
bytes, and a Rust-native stored shard at 18.28s wall and 1,248,396,657 bytes.
Stored `.npz` is therefore `5.22x` faster and `17.48x` larger on this workload.
Use stored shards for active local generation when disk is available; use
deflated shards for archival, transfer, or tighter disk budgets. Raw JSONL
remains the audit/debug format, not the persistent training format.

The pre-GPU phase has no network or fleet traffic requirement. Any future
distributed self-play, batched inference server, artifact upload, or evaluator
fleet belongs to the next approval gate.

## CPU Acceptance Budgets

| Gate | Required Report | Acceptance Rule |
|---|---|---|
| Schema/tokenization smoke | 100 fixed roots, token counts, action counts, elapsed time | Completes locally on CPU; no GPU required. |
| Replay round trip | Shard size, record count, checksum before/after | Loaded records match written records exactly. |
| C-GAB construction | Sparse relation edge counts, dense-equivalent memory, peak memory | Memory is reported and bounded for 100 fixed roots. |
| Radius 6 overflow census | Total placements, in-radius placements, overflow count, examples | Reports coverage/overflow without claiming prior measurement as fact. |
| Legal action enumeration | Min/median/max legal action count and timing | Deterministic order for fixed roots. |
| Chance-node dry run | Expansion count, sample count, cap behavior | Bounded chance-node expansion, no unbounded branching. |
| Search-root export | Root table size and alignment checks | Priors/visits/per-action Q arrays align with legal actions. |

## Regression Risks

| Risk | Failure Mode | Guardrail |
|---|---|---|
| Expanding hex radius | More cells increase token count, C-GAB pairs, memory, and latency. | Require CPU coverage census before changing the radius 6 default. |
| Dense candidate corpora | Materializing every candidate/action variant can dominate disk and memory. | Store exact legal actions per root with compact ids and checksums. |
| Unbounded chance nodes | Stochastic branching overwhelms MCTS simulation loop. | Progressive widening and explicit per-root chance-node caps. |
| Ignoring variable legal action count | Action-token tensors become padded to worst case and waste throughput. | Report legal action distributions and mask padded actions. |
| Dense C-GAB fallback | `O(tokens^2)` memory can dominate before model evaluation. | Prefer sparse templates; report dense-equivalent memory. |
| Replay without checksums | Silent corruption invalidates scientific conclusions. | Manifest and shard checksum gates are mandatory. |
| Hidden network dependency | CPU gates become flaky or environment-specific. | No network dependency in pre-GPU validation commands. |
