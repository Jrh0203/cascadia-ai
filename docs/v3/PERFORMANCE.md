# V3 Performance Notes

These are engineering measurements and diagnostics, not promotion evidence.

## Tensor Export

Rust-native greedy tensor export on `john0`:

| Format | 1,024 games wall time | Size |
|---|---:|---:|
| Deflated `.npz` | 1:35.36 | 71,413,962 bytes |
| Stored `.npz` | 18.28 s | 1,248,396,657 bytes |

Stored shards were `5.22x` faster and `17.48x` larger in that benchmark. Use
stored shards when GPU input throughput matters and disk headroom exists. Use
deflated shards for archival or transport.

## Prefilter Findings

The p80x2 vanilla public-token transformer prefilter family produced a useful
serving signal but did not justify K16 promotion:

- dedicated fixed 3-seed vanilla public-token ensemble:
  - K16 recall `0.7672`;
  - oracle regret `0.1146`.
- 4 paired complete-game pilot:
  - K16 prefilter search `96.0625`;
  - full-32 sampled search `95.0625`;
  - shadow full-search winner retained on `77.8125%` of decisions.
- 20-seed non-shadow follow-up:
  - K16 prefilter search `95.4625`;
  - full-32 sampled search `96.3500`;
  - mean decision time `2.3558s` vs `4.4617s`;
  - `1.89x` speedup and `47.2%` time reduction.

Interpretation: the bridge and prefilter path work, but K16 is too lossy on
current evidence. Prefer K24 or a stronger retention/search-aware model.

## CascadiaFormer Baseline

The first relation-tail CascadiaFormer-S run completed, but no-search play was
below greedy:

| Menu | Policy | Q | Greedy |
|---|---:|---:|---:|
| K256 | 72.5125 | 57.4750 | 87.3375 |
| K32 | 82.7625 | 77.9250 | 87.3375 |

The corrected greedy-state K32 retention run was much closer:

- locked validation greedy top-1: `0.6780`;
- 100-game model mean: `86.7800`;
- greedy mean: `87.5875`;
- paired delta: `-0.8075`;
- exact greedy-action match: `67.3625%`.

Interpretation: CascadiaFormer can operate near the greedy policy surface, but
it had not yet surpassed greedy before EI-0.

## EI-0 Greedy Search Bootstrap

EI-0 is the first CascadiaFormer run with positive no-search gameplay evidence
against greedy.

Training run:

- source runbook:
  `docs/v3/EI0_GREEDY_SEARCH_BOOTSTRAP_RUNBOOK.md`;
- expert tensor mode: `greedy_search_bootstrap`;
- objective: `search-improved-greedy-retention`;
- filter: strict greedy-prefix K32;
- corpus: 20,000 train roots and 4,000 validation roots;
- search labels: 4 rollouts/action, rollout top-k 4;
- model: CascadiaFormer-S;
- training: 25,000 steps, batch size 192, LR 1e-4;
- selected checkpoint:
  `cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json`.

Measured throughput:

- generation: 1,569 s total;
- train generation: 1,282 s;
- validation generation: 287 s;
- training: 2,457 s;
- roots/s: `15.2964`;
- rollout evals/s: `1,957.9350`;
- train step seconds: `0.09828`.

Training health:

- train/validation tensor invariant status: `pass`;
- strict K32 selected-action drops: `0`;
- max absolute Q invariant error: `0.0`;
- guarded checkpoint step: `7,250`;
- guarded locked validation total: `5.8410`;
- guarded locked validation greedy top-1: `0.69375`;
- guarded locked validation mean greedy rank: `1.8860`;
- guarded locked validation teacher top-1: `0.13025`;
- guarded locked validation teacher advantage over greedy: `2.1672`.

100-game no-search complete-game benchmark:

| Strategy | Mean | P90 | Delta vs Greedy | Greedy Match |
|---|---:|---:|---:|---:|
| Greedy | 87.5575 | 92.0000 | - | - |
| CascadiaFormer policy | 87.7925 | 92.0000 | +0.2350 | 70.1125% |
| CascadiaFormer q | 89.6175 | 94.0000 | +2.0600 | 29.8125% |

Interpretation: the q serving head is now the useful no-search policy, and it
beats greedy by about two points on the first 100-game EI-0 gate. This is merit
evidence, not yet promotion evidence. The search-integrated 20-game gate must
still show that model-ranked retained sets improve or preserve sampled-search
quality without unacceptable decision-time overhead.
