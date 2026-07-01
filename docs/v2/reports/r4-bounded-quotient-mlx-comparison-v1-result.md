# R4 Bounded Quotient MLX Comparison V1 Result

Date: 2026-06-17

ADR: 0156

Experiment: `r4-bounded-quotient-mlx-comparison-v1`

Protocol: `r4-bounded-parent-mlx-matched-comparison-v1`

Status: completed

Classification: `r4_bounded_parent_mlx_invalid`

Selected representation: none

Promotion authorized: false

## Executive Result

The four-host matched comparison completed all registered training and
validation work, but the formal evidence packet is invalid. The exact C0
same-host serving replay on john2 peaked at 4.1456 GiB process RSS, above the
prospectively frozen 4 GiB absolute limit. The deterministic classifier
therefore stopped before treatment selection.

The threshold was not changed after observing the result, and the replay was
not repeated until it happened to pass.

The other measurements make the practical outcome unambiguous even though
they are not a substitute for a valid classifier:

- C0 missed three of seven frozen quality sanity gates;
- every quotient missed multiple quality-noninferiority gates;
- no quotient reached the mandatory `0.80x` same-host parent-latency ratio;
- Q1 and Q3 also exceeded the 4 GiB absolute RSS limit in their own serving
  reports; and
- no quotient is eligible for promotion.

Exact sparse R2 remains the accepted learned substrate from ADR 0150. The R4
exact codecs and quotient extractors remain useful diagnostic machinery, but
Q1, Q2, and Q3 do not advance.

This is an offline representation result. It does not measure gameplay,
promote a player, or establish progress toward the 100-point mean target.

## Immutable Evidence

| Identity | Value |
|---|---|
| Production bundle | `5a922e36303fe382337bdb6b1c32028d2e7257c211d1be7c7d34391355cd723c` |
| Replay patch bundle | `8352e1a2efb26c5338cb17a65d831d266c3660ca37773b2b7b9d7e995f2c2dfa` |
| Parent sidecar | `ab038d49ac501b10e99cb9a038c594568bd16eb0403d39cda0c0e416f55ac2ce` |
| R3 cache | `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156` |
| S1 cache | `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15` |
| Authorization | `c4f8a054b18f31ba5b3102f7f415b7cac60589fa3c65af9c01cb0e5b9daad321` |
| Classification | `116240eb1ba58ec42027d358dddd60f66e6730904e75b8a9344af4d8cf68f001` |
| Order proof | `61d9893934592f5b0bd2fc5abb67b29c4edb2c91b02842237a4e02f6a6efb830` |

Forward and reverse report orders produced byte-identical scientific
identities.

## Completed Work

Every arm:

- consumed the same deterministic 3,000-step batch trace;
- used the same exact R3 candidate afterstate stream;
- used the same model graph, parameter layout, initialization, optimizer,
  labels, objective, and D6 schedule;
- scored exactly 240 validation decisions and 860,203 actions;
- produced finite scores and uncertainties;
- encoded each parent exactly once per decision; and
- completed without process swap or system swap growth.

The run therefore answered the registered comparison up to the frozen
evidence-validity gate. It did not fail because work was missing.

## Quality Results

| Arm | MAE | RMSE | Top-64 recall | Top-64 regret | Low-supply recall | Independent recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 exact R2 | 1.44703 | 1.91275 | 74.58% | 0.11617 | 85.96% | 76.19% | 97.92% |
| Q1 seat marginal | 1.42656 | 1.84418 | 70.00% | 0.15505 | 87.72% | 66.67% | 95.83% |
| Q2 directional | 1.42648 | 1.88981 | 72.50% | 0.14322 | 80.70% | 61.90% | 96.25% |
| Q3 affordance | 1.43197 | 1.88816 | 68.33% | 0.16910 | 78.95% | 57.14% | 94.17% |

C0 failed:

| Control gate | Limit | Observed | Passed |
|---|---:|---:|---|
| MAE | <= 1.42 | 1.44703 | no |
| RMSE | <= 1.85 | 1.91275 | no |
| Top-64 recall | >= 70% | 74.58% | yes |
| Top-64 regret | <= 0.12 | 0.11617 | yes |
| Low-supply recall | >= 88% | 85.96% | no |
| Independent recall | >= 76% | 76.19% | yes |
| Coverage | >= 97% | 97.92% | yes |

No treatment was quality-noninferior. Q1 preserved MAE, RMSE, and low-supply
recall but lost 4.58 percentage points of aggregate top-64 recall, added
0.03888 regret, lost 9.52 points on independent drafts, and missed the 99%
coverage floor. Q2 and Q3 lost still more protected-slice recall and also
missed aggregate ranking and coverage gates.

This is a coherent representation signal: the bounded parents retained broad
value calibration better than exact decision ranking. The missing information
is concentrated in market-conditioned competition and protected decision
slices, not merely global scalar value.

## Serving Results

Raw timing across unlike hosts is descriptive only:

| Arm | Host | Parent P50 | Fixed scores/s | Decision P99 | Peak active | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| C0 exact R2 | john1 | 1.765 ms | 73,819 | 181.29 ms | 0.442 GiB | 3.444 GiB |
| Q1 seat marginal | john2 | 1.097 ms | 78,706 | 175.18 ms | 0.442 GiB | 4.046 GiB |
| Q2 directional | john3 | 1.161 ms | 79,782 | 175.91 ms | 0.442 GiB | 3.986 GiB |
| Q3 affordance | john4 | 1.142 ms | 76,008 | 177.94 ms | 0.442 GiB | 4.267 GiB |

The registered promotion comparison uses exact C0 replays on each treatment
host:

| Treatment | Same-host C0 parent P50 | Parent ratio | Throughput ratio | P99 ratio | Active ratio | RSS ratio |
|---|---:|---:|---:|---:|---:|---:|
| Q1 / john2 | 1.349 ms | 0.813 | 1.180 | 1.010 | 1.000 | 0.976 |
| Q2 / john3 | 1.348 ms | 0.861 | 1.210 | 1.002 | 1.000 | 1.118 |
| Q3 / john4 | 1.353 ms | 0.845 | 1.145 | 1.032 | 0.999 | 1.103 |

All three quotients improved fixed-chunk throughput, but none passed the
mandatory parent P50 ratio of at most 0.80. None improved complete-decision
P99, active memory, or RSS enough to pass those end-to-end alternatives.

The john2 C0 replay measured:

```text
peak process RSS = 4,451,303,424 bytes = 4.1456 GiB
frozen maximum   = 4,294,967,296 bytes = 4.0000 GiB
```

That single absolute failure makes the classifier input invalid by design.

## Replay Correction

The first replay attempts on john2, john3, and john4 were rejected before an
accepted report because the replay command omitted the explicit production
row set and inherited the benchmark library's 20-row diagnostic default. The
verifier observed 20 decisions and 27,958 actions instead of 240 and 860,203.

The permanent correction:

1. derives rows `0..239` from the signed C0 report;
2. verifies complete certified coverage before launch;
3. passes the rows explicitly;
4. writes diagnostics outside the immutable C0 run tree; and
5. restores and verifies the checksum-fanned C0 tree before retrying.

The patch changed only the replay harness, benchmark artifact directory, ADR
reference, and replay amendment. It changed no arm, checkpoint, prediction,
quality metric, threshold, or classifier rule. Thirty-three focused R4 tests
passed before the corrected bundle was fanned to john2, john3, and john4.

## Formal Classification

The deterministic classifier resolved:

```text
classification = r4_bounded_parent_mlx_invalid
selected_arm = null
structural_error =
  host-paired C0 replay failed absolute serving on q1-seat-marginal-parent
```

The invalid classification withholds every scientific promotion claim. The
descriptive quality and efficiency failures above explain why no rerun is
scientifically justified merely to obtain a different label.

## Consequences

1. Select no R4 bounded parent representation.
2. Keep exact sparse R2 as the accepted learned candidate substrate.
3. Preserve Q1, Q2, and Q3 codecs for targeted diagnostics and ablations.
4. Do not modify the frozen 4 GiB limit or replay this run until it passes.
5. Carry the protected-slice losses into R5's legal-affordance and
   action-ranking controls.
6. Carry the throughput result into R6: compact token count alone is not
   enough; incremental updates must reduce actual parent and end-to-end work.
7. Continue to reject a 441-cell dense surface. The exact centered disk sizes
   remain 91 at radius five and 127 at radius six, with exact overflow where
   needed.
