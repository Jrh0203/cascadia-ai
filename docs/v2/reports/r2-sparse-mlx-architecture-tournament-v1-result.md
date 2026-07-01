# R2 Sparse MLX Architecture Tournament V1 Result

Date: 2026-06-17

ADR: 0146

Experiment: `r2-sparse-mlx-architecture-tournament-v1`

Protocol: `r2-sparse-mlx-matched-architecture-v1`

Status: completed

Classification: `r2_sparse_mlx_tournament_complete`

Selected architecture: `perceiver-fixed-latents`

Promotion authorized: false

## Executive Result

The complete sixteen-task, four-host R2 tournament passed every structural,
semantic, reproducibility, and order-invariance gate. The deterministic
classifier selected the Perceiver fixed-latent trunk as the only primary
architecture that remained value-noninferior to the frozen R0 exact-entity
control under all three preregistered error gates.

The selected Perceiver processed 10,178.87 public afterstates per second at
batch 64, with 6.213 ms P50 latency and 121.8 MB peak active inference memory.
It was 2.81 times faster than the Set Transformer and 7.57 times faster than
the directional graph arm in observed inference throughput.

This is a representation and value-screen result. It does not measure
complete-action ranking, retained regret, paired gameplay, or mean game score,
and it does not authorize model promotion or claim progress to the 100-point
goal.

## Immutable Production Identity

| Identity | Value |
|---|---|
| Bundle | `65613a7526f292af47740c1ef21bc3bca22efa5454f0a9b3948c037f2db2a962` |
| Authorization | `780455bfd22643c07ec8c9624d4e2c81342f7f3887a49087b7ea1bc759e5b6a3` |
| Corpus lock | `264315f193ea69232c699277f503c50aa1e1026eb1d27fecd7127503ff6ae0f7` |
| R0 control binding | `a314077de5cce28dabd7d25d17bd6520fb830c85d2c6f624688aa7875615e8a6` |
| Cache | `c97ce6b2de1beb4cc7d2d5e31e2fbed9213b28d3bde8a8ab4bdcc90b2edd85f8` |
| Cache binding | `70fa98f45854b7b1bd81f8e4dadf08ec098d1af5b575450014c4fb0d7dbb1982` |
| Cache manifest BLAKE3 | `739ca246f880bab2eeae4e87f97c9a0907b55a2108fa70f0fdc8e8738da67ab7` |
| Report collection | `f094fc73b2e4c870fbebcd5b2c2ef2064227dba82df34b72f0587af6d24e1e90` |
| Forward aggregate | `f707afa571519297be8ba03dfdadf87d7c465a15984b70a2e0f62bd984245b14` |
| Classification BLAKE3 | `79d6c5b611e41a20429434f1a3647e5b4dd3a5a072a438901fb211cae980836e` |

Forward and reverse classification outputs were byte-identical.

## Execution

All four Apple Silicon hosts passed the frozen source, corpus, authorization,
runtime, and MLX GPU preflight. One exact 60,000-position cache was exported
on john1 and fanned out byte-identically before any optimizer step.

| Host | Run role | Architecture | Result |
|---|---|---|---|
| john1 | `set-primary` | Padded Set Transformer | Complete; ineligible |
| john2 | `graph-primary` | Directional graph attention | Complete; ineligible |
| john3 | `perceiver-primary` | Perceiver fixed latents | Complete; selected |
| john4 | `set-replay` | Padded Set Transformer replay | Complete; exact replay |

Every run completed exactly 500 AdamW steps and consumed the same 16,000
optimizer examples, deterministic D6 schedule, target stream, seed, and cache.
The models contained 141,131 to 143,915 trainable parameters, a 1.9726%
spread under the frozen 3% gate.

## Validation Results

The frozen R0 exact-entity control had total MAE 2.650742, total RMSE
3.376045, and mean component MAE 2.707762.

| Architecture | Total MAE | Delta vs R0 | Total RMSE | Delta vs R0 | Mean component MAE | Delta vs R0 | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| Set Transformer | 2.606399 | -0.044343 | 3.332316 | -0.043729 | 3.160244 | +0.452482 | No |
| Directional graph | 2.652133 | +0.001391 | 3.362798 | -0.013247 | 3.149292 | +0.441530 | No |
| Perceiver latents | 2.942973 | +0.292230 | 3.716922 | +0.340877 | 2.869804 | +0.162042 | Yes |

The Set Transformer achieved the lowest total MAE, and the graph arm matched
the R0 total-score error closely. Both failed the stricter component-fidelity
gate, which allowed at most +0.25 mean component MAE. The Perceiver had worse
total-score error than those two arms but preserved score-component structure
well enough to pass every frozen R0 gate. Its validation total correlation was
0.171967, close to the R0 control's 0.181062 and materially above Set
Transformer at 0.035163 and graph attention at 0.069086.

## Reproducibility

The independent Set Transformer replay on john4 reproduced the john1 primary
exactly:

| Replay measurement | Observed absolute delta | Gate |
|---|---:|---:|
| Validation total MAE | 0.000000 | <= 0.10 |
| Validation total RMSE | 0.000000 | <= 0.15 |
| Validation mean component MAE | 0.000000 | <= 0.03 |
| First-256 maximum component prediction | 0.000000 | <= 0.10 |

The replay gate passed. The different checkpoint content identities are
expected because manifests include host-specific paths; scientific outputs
and fixed predictions were identical.

## MLX Performance

Cross-host throughput is operational evidence rather than a direct
same-hardware quality ratio, but the magnitude and memory ordering are clear.

| Architecture | Inference actions/s | P50 ms | P99 ms | Training examples/s | Gradient examples/s | Peak active inference memory |
|---|---:|---:|---:|---:|---:|---:|
| Set Transformer | 3,619.39 | 17.644 | 18.484 | 315.41 | 1,075.77 | 296.4 MB |
| Directional graph | 1,344.97 | 47.521 | 48.738 | 181.83 | 290.06 | 1,364.5 MB |
| Perceiver latents | 10,178.87 | 6.213 | 7.024 | 414.05 | 2,927.35 | 121.8 MB |
| Set replay | 3,867.94 | 16.466 | 16.906 | 342.47 | 1,170.73 | 296.5 MB |

The Perceiver is the strongest serving substrate from this screen. The graph
arm is not competitive at this tensor shape or implementation budget.

## Token-Ablation Evidence

The Perceiver masking deltas were:

| Masked token class | Total MAE delta | Total RMSE delta | Mean component MAE delta |
|---|---:|---:|---:|
| Occupied | +0.033931 | +0.039897 | -0.000435 |
| Frontier | -0.007642 | -0.009146 | +0.001006 |
| Habitat component | -0.023551 | -0.027676 | +0.000043 |
| Wildlife motif | -0.010878 | -0.012582 | -0.000154 |

Only occupied-token masking clearly damaged total-score prediction after 500
steps. Masking frontier, component, or motif tokens was neutral or slightly
beneficial on aggregate total error. This does not show that those exact
objects are useless. It shows that the current short value-only objective did
not learn to exploit them reliably.

The next R2-derived hypotheses should therefore preserve the fast Perceiver
substrate while testing direct token-type supervision, relational auxiliary
objectives, candidate-conditioned learning, or longer matched optimization.
Simply adding more explicit objects without an objective that requires their
use is unlikely to move gameplay strength.

## Classification

All preregistered gates resolved:

- structural completeness: passed;
- exact cache, D6, padding, ownership, and no-truncation integrity: passed;
- matched parameter and optimization controls: passed;
- independent replay: passed exactly;
- at least one R0-value-noninferior architecture: passed;
- serving latency and memory practicality: passed;
- forward/reverse order invariance: passed; and
- gameplay and promotion claims withheld: passed.

The deterministic verdict is:

```text
r2_sparse_mlx_tournament_complete
selected_architecture = perceiver-fixed-latents
selected_run_role = perceiver-primary
promotion_authorized = false
```

## Consequences

1. Use `perceiver-fixed-latents` as the R2 architecture baseline for future
   sparse value, action-conditioning, and auxiliary-objective experiments.
2. Do not advance the Set Transformer or directional graph trunks from this
   screen without a new preregistered objective or training-budget hypothesis.
3. Treat occupied tokens as the only token class with demonstrated aggregate
   predictive reliance under the 500-step value-only protocol.
4. Test whether frontier, habitat-component, and motif tokens become useful
   under objectives tied directly to legal actions, local edits, score
   decomposition, or future pattern completion.
5. Keep promotion blocked until complete-action ranking, retained-regret,
   realistic serving-latency, paired-gameplay, and mean-score gates pass.

