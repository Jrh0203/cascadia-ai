# R0 Spatial MLX Iso-Architecture Tournament V1 Result

Date: 2026-06-17

Experiment ID: `r0-spatial-mlx-tournament-v1`

Contract: ADR 0142

Stage: R0 learned-value and MLX shape screen

Verdict: **FAIL - compact dense tensors are value-noninferior, but slower than exact entities**

Machine-readable verdict:
`lossless_compact_dense_value_noninferior_but_slower_than_exact_entities`

Aggregate BLAKE3:
`c29e6b9ed2a6b816652e33394d978ed5b51be85e69f238bdf60c91ff5ee3ad9e`

Classification BLAKE3:
`f228014e4e9da73246b50c977f1fa2dbe2adfcbeec97ade0d1277c79c4ca255a`

## Executive Result

The user's prior 121-cell observation is verified in the important historical
sense: a compact hex tensor is dramatically faster than the old 441-cell
square and loses essentially no value accuracy in this controlled screen.
The closest tested shape to 121 rows was the radius-5 arm:

- 91 local hex cells;
- 23 exact overflow rows;
- 114 total MLX rows;
- 8.67x the historical arm's observed inference throughput; and
- 9.01x the historical arm's observed training throughput.

That is a large and real improvement over the historical representation.
It is not the winning representation, however. The exact entity control uses
only 23 rows. Under the preregistered same-host shape calibration, the
114-row arm reached only 0.202x exact inference throughput and 0.251x exact
forward-plus-backward throughput. Exact entities were therefore about 4.94x
faster for inference and 3.98x faster for training.

All three bounded compact arms passed the value-noninferiority gates by a wide
margin. None passed either leverage gate. The classifier selected no Stage 2
candidate.

The R0 decision is:

1. Retain exact occupied entities as the dense-screen winner and learned-value
   control.
2. Reject 61-, 91-, and 127-cell dense tensors as performance candidates when
   exact overflow is preserved.
3. Retire the historical 441-cell tensor from new representation work.
4. Do not open R1 dense-capacity reinvestment because R0 produced no qualifying
   compact arm.
5. Promote R2 sparse occupied, frontier, component, and motif tokens to the
   next matched MLX architecture tournament.

This experiment measures representation, learned value, and local MLX shape
cost. It does not measure complete-action ranking or gameplay strength and
does not authorize player promotion.

## Frozen Experimental Contract

Every arm consumed the same:

- 50,000 training positions;
- 10,000 validation positions;
- 12 Rust-authored D6 transforms;
- 11 nonnegative score-component targets;
- 74,635-parameter set-value architecture;
- initialization seed and deterministic batch stream;
- AdamW optimizer and learning-rate schedule;
- 500 optimizer steps and 16,000 sampled examples;
- full train and validation evaluation; and
- Apple Silicon MLX GPU runtime.

Only the spatial row layout changed.

The production identities were:

```text
corpus lock:
6e6869d825b3a5ee4dda41f26245f40303174c2a215887ede4ebe153f20c6d43

sealed bundle:
611bcebcad6d7dd94374ada8ee8022263363cbd53f67f027eebc0689d81487c6

authorization:
0efdd13c4a693be30ec5e845a19f74f5996743c9b7850e6c84b41a1047cf2b4c

queue scientific task hash:
425a4638005629ed047a92813c757acb098ab997125ba227486262c40cba68c9
```

All four host preflights passed with Python 3.12.13, MLX 0.31.2, arm64, and
the MLX GPU device. The four decision-changing arms ran concurrently on
john1 through john4. Historical 441 backfilled john1 after its exact control
completed.

## Semantic Integrity

Every report passed:

- source and corpus identity;
- cache content-address verification;
- exact Rust representation round trips;
- exact packed round trips;
- all D6 transform and inverse proofs;
- identical source semantic digests;
- identical transformed semantic digests;
- identical target digests;
- zero nonpadding data in padding rows;
- exact active-row and overflow accounting;
- model and parameter-count equality;
- optimizer-step and evaluation-coverage equality;
- finite timing and metric validation; and
- explicit nonpromotion claims.

Forward and reverse report-order classifications are byte-identical. The
final order proof records:

```text
byte_identical: true
promotion_authorized: false
classification:
r0_spatial_mlx_tournament_complete
```

The result is a complete negative decision, not an incomplete or invalid run.

## Value Result

All compact treatments were value-noninferior to exact entities under the
frozen Stage 2 thresholds.

| Arm | Rows | Validation total MAE | MAE delta | Total RMSE delta | Mean component MAE delta | Value gate |
|---|---:|---:|---:|---:|---:|---|
| Exact entities | 23 | 2.650742 | control | control | control | Pass |
| Radius 6 / 127 + overflow | 150 | 2.653647 | +0.002905 | -0.003232 | +0.043638 | Pass |
| Radius 5 / 91 + overflow | 114 | 2.653648 | +0.002905 | -0.003232 | +0.043638 | Pass |
| Radius 4 / 61 + overflow | 84 | 2.653805 | +0.003063 | -0.003124 | +0.043556 | Pass |
| Historical 441 + overflow | 464 | 2.650886 | diagnostic | diagnostic | diagnostic | Excluded |

The gates allowed:

- total MAE delta up to 1.0;
- total RMSE delta up to 1.5; and
- mean component MAE delta up to 0.25.

The compact arms missed none of these thresholds. Their validation differences
are small enough that this screen supplies no evidence of information loss
from the exact-overflow layouts.

The absolute validation correlation is low in every arm because this was a
500-step controlled representation screen, not a production value-model
training run. The scientific comparison remains valid because architecture,
targets, initialization, optimizer, sample stream, and budget were matched.

## Throughput Result

The preregistered leverage gate required either:

- at least 1.5x same-host inference throughput; or
- at least 1.3x same-host forward-plus-backward throughput.

No compact arm approached either threshold.

| Arm | Observed inference actions/s | Observed training examples/s | Same-host inference ratio | Same-host training ratio | Leverage |
|---|---:|---:|---:|---:|---|
| Exact entities | 29,717.3 | 6,002.5 | 1.009x | 1.007x | Control |
| Radius 6 / 127 + overflow | 3,988.9 | 1,313.3 | 0.140x | 0.143x | Fail |
| Radius 5 / 91 + overflow | 5,829.6 | 1,865.3 | 0.202x | 0.251x | Fail |
| Radius 4 / 61 + overflow | 8,388.7 | 2,607.2 | 0.291x | 0.297x | Fail |
| Historical 441 + overflow | 672.2 | 207.0 | 0.0227x | 0.0208x | Diagnostic |

Even the smallest dense treatment made exact entities:

- 3.44x faster in same-host inference calibration; and
- 3.37x faster in same-host training calibration.

The result is monotonic and mechanistically coherent: fewer masked rows
improve throughput, but materializing 84 rows is still substantially more
expensive than attending over the 23 exact entity rows that contain the
actual occupied board.

## Historical 441 Comparison

The compact-state intuition is strongly supported when the comparison target
is the old fixed square.

| Comparison with historical 441 | Radius 5 / 114 rows | Radius 4 / 84 rows | Exact / 23 rows |
|---|---:|---:|---:|
| Observed inference throughput | 8.67x | 12.48x | 44.21x |
| Observed training throughput | 9.01x | 12.59x | 29.00x |
| Same-host calibrated inference | 8.93x | 12.84x | 44.09x |
| Same-host calibrated training | 12.06x | 14.25x | 48.03x |

Thus the earlier compression finding was not wrong. It identified a serious
441-cell representation tax. The new experiment reveals that the better
answer is to remove dense empty-cell materialization entirely rather than
stop at a smaller dense disk.

## Unified-Memory Result

Masked dense rows also carried a large unified-memory cost.

| Arm | Inference peak active memory | Training peak active memory | Inference multiple vs exact | Training multiple vs exact |
|---|---:|---:|---:|---:|
| Exact entities | 21.18 MB | 25.94 MB | 1.00x | 1.00x |
| Radius 6 / 127 + overflow | 205.59 MB | 264.78 MB | 9.71x | 10.21x |
| Radius 5 / 91 + overflow | 139.89 MB | 177.12 MB | 6.61x | 6.83x |
| Radius 4 / 61 + overflow | 93.24 MB | 115.98 MB | 4.40x | 4.47x |
| Historical 441 + overflow | 1.231 GB | 1.679 GB | 58.12x | 64.74x |

The 441 diagnostic used roughly 1.68 GB of active MLX memory during the
training calibration for a 74,635-parameter model. Exact entities used about
25.9 MB. This is sufficient to prohibit 441 as a compatibility default in
future MLX architectures.

## Gate Classification

| Gate | Result | Evidence |
|---|---|---|
| Five structurally complete reports | Pass | Exact, radius 6, radius 5, radius 4, and historical 441 |
| Identical corpus and semantic targets | Pass | Shared corpus, source, D6, and target digests |
| Identical model and optimization | Pass | 74,635 parameters and 500 steps in every arm |
| Full train and validation evaluation | Pass | 50,000 and 10,000 rows per arm |
| Compact value noninferiority | Pass | All three compact arms pass every value delta gate |
| At least 1.5x same-host inference | **Fail** | Best compact arm is 0.291x |
| At least 1.3x same-host training | **Fail** | Best compact arm is 0.297x |
| Stage 2 candidate selected | **Fail** | `selected_stage2_candidate: null` |
| Deterministic classification | Pass | Forward and reverse outputs byte-identical |
| Gameplay or promotion claim | Not made | Explicitly false in every report and aggregate |

## Research Decision

### Selected

- Exact entity rows remain the R0 learned-value and throughput control.
- R2 sparse occupied, frontier, component, and motif state proceeds to a
  matched MLX architecture comparison.
- Action-centric R3 and hybrid sparse R6 remain live research directions.

### Rejected

- Dense radius 4, 5, or 6 as the selected R0 substrate.
- R1 reinvestment based on a dense R0 winner; no arm qualified to enter it.
- Historical 441 for any new model merely to preserve old tensor conventions.
- The assumption that a smaller nominal dense grid beats a variable-length
  exact set when the board contains at most 23 occupied tiles per player.

### Still Open

This result does not decide whether explicit frontier, component, motif,
semantic-supply, opponent, or action-edit tokens improve decision quality.
Those representations contain more first-class relational objects than the
23-row occupied-only control and require matched learned experiments.

## Execution Audit

The first proposed bundle identity was invalidated before queue application
and before any optimizer step. A local preflight created Python bytecode inside
the writable content-addressed tree. The invalid identities remain preserved:

```text
bundle:
2af77e4b8d2a9d60cded05e82fb68babc528b662f1c2fc9e112e9b1831ce8b0d

authorization:
6aef5e13081be8bdd527f3301e0301676d1fc509994bf2615de0cf36801718b5

inert queue specification:
402a2cd3b5cd57a0c44c5cb39e58678200cc31976686579c1365555a79a182e4
```

The permanent repair:

- runs every frozen Python entry point with `-B`;
- strips write permission from every sealed bundle file and directory;
- revalidates reused bundles; and
- tests both bytecode suppression and filesystem sealing.

The accepted production identity was then rebuilt, fanned out, preflighted,
and executed from scratch. The incident contributes no scientific data.

## Evidence

- ADR:
  `docs/v2/decisions/0142-r0-spatial-mlx-iso-architecture-tournament.md`
- Preregistration:
  `docs/v2/reports/r0-spatial-mlx-tournament-v1-preregistration.md`
- Invalid first launch:
  `docs/v2/reports/r0-spatial-mlx-tournament-v1-invalid-launch-1.md`
- Corpus lock:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/control/corpus-lock.json`
- Authorization:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/control/authorization.json`
- Report collection:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/collection.json`
- Forward classification:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/classification-forward.json`
- Reverse classification:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/classification-reverse.json`
- Order proof:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/reports/classification-order-proof.json`
- Dashboard ledger:
  `artifacts/experiments/r0-spatial-mlx-tournament-v1/ledger.json`

## Final Claim Boundary

R0 Stage 2 establishes that exact-overflow compact dense tensors can match the
small model's value error while massively outperforming historical 441.
It also establishes that all tested dense tensors are materially slower and
larger than a direct exact-entity sequence.

It does not establish that occupied entities alone are sufficient for a
100-point player. The next question is whether sparse first-class frontier,
component, motif, supply, opponent, and action-edit relations improve
decision quality while preserving the exact representation's serving
advantage.
