# GPU Handoff Gate

This is the explicit stop condition for the Cascadia v3 CPU planning and
readiness phase. Passing this gate means the repository is prepared to ask for
approval to move into GPU verification/training work; it does not authorize that
work by itself.

## Required CPU Artifacts Before GPU

| Artifact | Required State |
|---|---|
| Schema docs | `SCHEMA_CONTRACTS.md` current and referenced by implementation. |
| Deterministic simulator test report | Placeholder or completed CPU report covering legal actions, scoring decomposition, chance replay, serialization, and symmetry. |
| Radius 6 coverage/overflow report | Placeholder or completed CPU census reporting radius 6 membership, 127-cell fast path usage, exact overflow count, and examples. |
| Tiny corpus overfit plan | CPU-written plan for one-batch or tiny-corpus smoke using CascadiaFormer-Zero-S shape. |
| Search-root replay contract | Root table schema with legal actions, priors, visits, per-action Q, selected action, chance samples, final score vector, and score decomposition. |
| Replay dry-run report | Placeholder or completed CPU round-trip report with schema id, record count, and checksum. |
| Performance budget table | Current budgets for legal action enumeration, tokenization, C-GAB, chance-node expansion, MCTS loop, replay, memory, latency, throughput, network, and regression risks. |

## Blocked Until GPU Access And Approval

The following operations are blocked until the next approval gate:

- MLX/Metal training;
- MLX/Metal verification;
- RTX profile;
- batched GPU inference server;
- self-play strength testing;
- CascadiaFormer-M or CascadiaFormer-L training;
- large model training;
- large self-play/search generation intended to produce training data.

## Handoff Checklist

Before asking for GPU work approval, confirm:

- all CPU-only validation gates in `CPU_PRE_GPU_MILESTONES.md` are either passed
  or have explicit placeholders with owners and commands;
- radius 6 remains canonical, with exact overflow handling;
- no prior NNUE/radius-7 campaign artifact has become the governing design;
- replay/search-root schema can preserve full self-play/search-root replay
  labels;
- the tiny CPU model smoke has verified legal-action query logits, value vector
  outputs, and score/rank/vector auxiliary head dimensions;
- performance budgets include memory, latency, throughput, replay size, network
  assumptions, and regression criteria.

## Next Approval Question

After the CPU readiness bundle is complete, ask:

Should Cascadia v3 proceed from CPU readiness into GPU verification/training for
CascadiaFormer-Zero-S, beginning with tensor-shape verification, tiny-corpus
overfit, and a small search-root replay training smoke?
