# O1 Opponent-Intent MLX Factorial v1 Result

**Completed:** 2026-06-17  
**Experiment:** `o1-opponent-intent-mlx-factorial-v1`  
**Classification:** `opponent_intent_policy_holdout_replication_passed`  
**Selected arm:** `a2-next-draft-auxiliary`

## Verdict

The O1 factorial passed.

Recent public action history improved prediction of which current market tile
would be consumed by each opponent or survive to the focal player's next
access. Training ordered next-draft auxiliary heads improved it further. The
selected A2 treatment reproduced on a different Mac and then passed the sealed
PatternPortfolio test.

This authorizes a separate high-regret draft-ranking integration experiment.
It does not establish gameplay strength, a higher Cascadia score, paid-wipe
intent, strategy switching, or transfer to the v1 champion.

## Matched Design

All four arms used:

- the same 374,171-parameter MLX graph;
- the same byte-identical initialization;
- the same 77,824 training windows;
- exactly 5,120 optimizer steps and 622,592 examples;
- AdamW at learning rate `3e-4` and weight decay `1e-4`;
- final-checkpoint-only validation;
- compact occupied-entity boards with no 441-cell tensor.

Only three gates differed: recent-history input, next-draft auxiliary loss, and
intent-to-survival routing.

Primary runs occupied john1 through john4 concurrently. Every arm then replayed
on a different host.

## Validation

PatternCompetition supplied 256 held-out games and 19,456 focal windows.

| Arm | Brier | Relative gain vs A0 | Paired 95% CI | NLL | ECE | Binary survival Brier | Auxiliary NLL gain | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 public state | 0.627112 | control | | 1.173192 | 0.023481 | 0.216346 | | |
| A1 recent history | 0.616246 | 1.733% | [-0.012530, -0.009182] | 1.152264 | 0.026309 | 0.213386 | not enabled | yes |
| **A2 next-draft auxiliary** | **0.614824** | **1.959%** | **[-0.014197, -0.010372]** | **1.149701** | **0.027525** | **0.212919** | **11.675%** | **selected** |
| A3 joint intent-survival | 0.616026 | 1.768% | [-0.012805, -0.009394] | 1.151216 | 0.029152 | 0.213372 | 10.371% | yes |

All three treatments passed every preregistered validation gate. A2 won the
frozen tie-break by lowest Brier, then lower NLL.

A2 beat A0 in 79.30% of held-out games. Its auxiliary heads also learned
substantial public behavioral structure:

- mean next-draft NLL improved 11.67% over train-frequency priors;
- drafted-wildlife NLL improved 17.67%;
- replace-three NLL improved 35.22%;
- tile-slot NLL improved 5.51%;
- wildlife-slot NLL improved 6.70%.

## Sealed Test

The test data stayed unopened until validation selected A2. PatternPortfolio
then supplied 256 games and 19,456 windows.

| Metric | A0 control | A2 selected | Result |
|---|---:|---:|---|
| Disposition Brier | 0.622901 | **0.611539** | **1.824% relative gain** |
| Paired game-bootstrap delta | | | **95% CI [-0.013081, -0.009649]** |
| Fraction of games improved | | | **76.56%** |
| Disposition NLL | 1.166911 | **1.144981** | improved |
| Top-label ECE | **0.019249** | 0.023799 | within +0.015 gate |
| Binary survival Brier | 0.216220 | **0.212972** | improved |
| Mean auxiliary NLL gain | | **11.640%** | passed |

Every sealed-test gate passed.

## Descriptive Stress

Random-policy stress was descriptive and could not create or reverse a pass.
Across 128 games:

- Brier improved from 0.640934 to 0.637156, a 0.589% relative gain;
- NLL improved from 1.201167 to 1.194825;
- binary survival Brier improved from 0.221417 to 0.220790;
- ECE increased from 0.039704 to 0.049015;
- auxiliary NLL improved 8.41% over the train-frequency prior.

The smaller gain and weaker calibration are useful warnings against claiming
universal opponent-policy transfer.

## Exactness

For all four arms, primary and rotated-host replay matched exactly on:

- final model bytes;
- final parameter tensor;
- validation prediction evidence;
- complete scientific identity.

The shared authorization, bundle, datasets, parameter layout, initialization,
and training priors also matched across all eight runs.

## Performance

| Arm | Training seconds | Examples/sec | Serving windows/sec | Batch latency | Training active memory |
|---|---:|---:|---:|---:|---:|
| A0 | 126.98 | 4,903 | 22,727 | 5.63 ms | 160.0 MB |
| A1 | 123.58 | 5,038 | 23,028 | 5.56 ms | 160.0 MB |
| A2 | 123.98 | 5,022 | 22,807 | 5.61 ms | 160.0 MB |
| A3 | 123.36 | 5,047 | 23,038 | 5.56 ms | 160.0 MB |

The four-host primary and replay waves each completed in roughly two minutes.
The compact representation and matched MLX implementation made the full
eight-run scientific campaign practical without external compute.

## Interpretation

Three conclusions survive the crossed-host and policy-held-out controls:

1. Observable action history contains predictive information absent from the
   instantaneous public state.
2. Auxiliary supervision for what opponents draft is the strongest tested
   mechanism for making that information useful to future-access prediction.
3. Explicitly routing intent tokens into the survival head did not compound
   the gain. The useful signal is present, but this particular cross-attention
   path is unnecessary at the tested capacity and corpus size.

The immediate successor should freeze A2 and test whether its calibrated
future-access predictions improve complete-action ranking specifically on
high-regret draft decisions. That experiment must compare a matched control,
static frequency features, A2 probabilities, and an oracle-survival ceiling
before any gameplay claim.

## Artifacts

- validation classification:
  `artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/validation-classification.json`;
- terminal classification:
  `artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/classification.json`;
- selected model:
  `artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/collected/a2-primary/training/final-model.safetensors`;
- all primary and replay runs:
  `artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/collected`;
- immutable source bundle:
  `artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/bundles/a00ed45260c5518c4471a89251fdf2324c7fc1829c9ddc84bc3584ca119fd562`.
