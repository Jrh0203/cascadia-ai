# S4 Candidate-Context MLX Comparison V1 Result

Date: 2026-06-17

ADR: 0153

Experiment: `s4-candidate-context-mlx-comparison-v1`

Protocol: `s4-candidate-context-matched-comparison-v1`

Status: completed

Classification: `s4_candidate_context_mlx_all_treatments_degraded`

Selected arm: none

Compact representation rescued: false

Promotion authorized: false

## Executive Result

Exact whole-decision candidate context did not rescue the failed R3 radius-one
representation.

All four arms completed the same 3,000-step schedule, scored all 240 open
validation decisions and 860,203 legal actions exactly once, and produced
finite evidence. The independent compact control passed every absolute serving
gate. None of the three context treatments passed the frozen matched-control
quality contract, none recovered the full-R2 quality envelope, and none passed
the full-R2-relative latency, active-memory, or RSS gates.

The exact-relations arm produced the only material contextual signal:

- top-64 recall improved from 68.33% to 70.42%;
- low-supply recall improved from 82.46% to 87.72%;
- middle-phase recall improved from 56.79% to 61.73%;
- MAE improved by 0.01944; and
- retained regret improved by 0.01326.

That signal was not a rescue. Exact relations regressed independent-draft
winner recall from 80.95% to 76.19%, reached only 97.50% confidence coverage,
and remained substantially worse than the full-R2 control in value error,
protected-slice recall, confidence coverage, latency, active memory, and RSS.

The independent S4 control also degraded from the exact R3 radius-one warm
start after another 3,000 jointly fine-tuned steps. Its MAE worsened from
1.48856 to 1.62602, top-64 recall fell from 74.58% to 68.33%, and retained
regret rose from 0.10339 to 0.11308. This closes both the candidate-context
rescue and the matched additional-fine-tuning recipe on this substrate.

This is an offline representation result. It does not measure gameplay,
promote a model, or claim progress toward the 100-point mean-score target.

## Immutable Evidence

| Identity | Value |
|---|---|
| Source bundle | `8300dba3655dc5cce4d1ff8f6dc3f3964df163ea007029b6e23333639c89819c` |
| Source BLAKE3 | `a5b2c7bb0bd67bbf69f83010223c4b812171985f1d6a95dd3c5ee711c9c666bd` |
| Authorization | `99a0c9c2f12950fe927b56e7e05b7bb695b35c76d856be789c2e3c20962a9cc8` |
| Cross-host smoke | `07f83fd204fba1f5dfd1966120cb51033d95eb26fa3faf91a3b2904abc37d6bf` |
| R3 cache | `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156` |
| S1 cache | `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15` |
| S4 context cache | `fd3dcc8018cfe4b735a9a6514555e90e938fd142e746dc6d791f482e96463def` |
| Open-data proof | `a056aceadb7f53c01dc87c8a39d95a7866bac6df93b050c45cc860de2b8b87ea` |
| Classification | `d10caa780ad56ac9ab8e1e60e9f19d01b3630f86dc83b531a3da669c4ac04aaa` |
| Scientific classification | `48a0d5da7608c9dd06c54234890f7f6b01cf3649747421cdd305b625e4e862b4` |
| Order proof | `3af1d8de2086dbf32aac8db1aa77b3bf7ec4f8c81b70b1fcc38a607e05906ff7` |

Forward and reverse report orders produced byte-identical classification
files.

Arm report identities:

| Arm | Report ID |
|---|---|
| Independent | `fa56de849eb6f85d7a4dcb45c4b9f0effb74f8bc8db82a6088fd90355ea5b578` |
| Inducing 16 | `1bbe364094bd0dba7697421271e58e32741a08dbe6a341d391bb99755d7a567f` |
| Exact relations | `78f41763b238ec99dca5cfff2e78dc1460d4e53f9fd5db168fcbe5fba1a2f235` |
| Combined | `6879d0eab164add87dc51616afc0fe6f6c71506ec790dbddb4707d1a107d12ef` |

## Cluster Execution

The four production arms ran concurrently and nonduplicatively:

| Host | Arm |
|---|---|
| `john1` | `c0-independent` |
| `john2` | `t1-inducing-16` |
| `john3` | `t2-exact-relations` |
| `john4` | `t3-combined` |

All seven immutable input trees matched across the cluster before optimizer
execution. Every production arm depended on all four host-local preflights.
The four optimizer tasks were claimed within 65 milliseconds of one another.

All 19 queue tasks completed without retry or failure:

- seven whole-tree fan-outs;
- four host-local preflights;
- four independent production arms;
- one checksummed collection;
- two order-independent classifications; and
- one classification-order proof.

The frozen source tree passed 22 focused campaign, classifier, smoke, bundle,
and experiment-ledger tests. Ruff passed with no findings.

## Matched Offline Results

Every arm had 771,524 parameters, identical parameter layout, identical
initial parameter bytes, the same failed radius-one warm start, the same
deterministic batch and D6 schedules, and the same loss.

| Arm | MAE | RMSE | Top-64 recall | Top-64 regret | Confidence coverage |
|---|---:|---:|---:|---:|---:|
| Independent | 1.62602 | 2.10325 | 68.33% | 0.11308 | 97.50% |
| Inducing 16 | 1.64853 | 2.13031 | 68.33% | 0.11974 | 97.50% |
| Exact relations | 1.60658 | 2.08103 | 70.42% | 0.09982 | 97.50% |
| Combined | 1.61231 | 2.08202 | 68.75% | 0.10412 | 97.50% |

Matched deltas against the independent S4 control:

| Treatment | MAE | RMSE | Recall | Regret | Coverage |
|---|---:|---:|---:|---:|---:|
| Inducing 16 | +0.02251 | +0.02705 | +0.00 pp | +0.00666 | +0.00 pp |
| Exact relations | -0.01944 | -0.02223 | +2.08 pp | -0.01326 | +0.00 pp |
| Combined | -0.01371 | -0.02123 | +0.42 pp | -0.00896 | +0.00 pp |

Only exact relations met a material-effect gate, through its 2.08-point
top-64 recall gain. It still failed matched-control noninferiority because
independent-draft recall regressed by 4.76 points and confidence coverage was
below 99%.

## Protected Slices

| Arm | Low-supply recall | Independent-draft recall | Middle-phase recall |
|---|---:|---:|---:|
| Independent | 82.46% | 80.95% | 56.79% |
| Inducing 16 | 80.70% | 80.95% | 55.56% |
| Exact relations | 87.72% | 76.19% | 61.73% |
| Combined | 82.46% | 76.19% | 60.49% |

The slice pattern explains the aggregate result:

1. Inducing context added cost without a useful ranking change.
2. Exact relations helped low-supply and middle-game competition.
3. Both relation-enabled arms regressed independent-draft winners.
4. No arm improved confidence coverage.
5. Combining inducing and relation context weakened the relation-only recall
   gain rather than compounding it.

The exact-relation signal is therefore real but conditional. Candidate-set
relations help where actions compete through shared public resources and
middle-game structure, but the present aggregation distorts cases where the
best draft is more independent of its siblings.

## Failure To Rescue Full R2

The frozen full-R2 quality envelope was:

```text
MAE <= 1.37023
RMSE <= 1.79231
top-64 recall >= 72.00%
top-64 regret <= 0.10312
low-supply recall >= 90.23%
independent-draft recall >= 79.95%
confidence coverage >= 99%
```

No S4 arm passed it.

| Reference or best S4 arm | MAE | RMSE | Recall | Regret | Low supply | Independent | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full R2 reference | 1.32023 | 1.74231 | 72.50% | 0.09812 | 91.23% | 80.95% | 97.92% |
| R3 radius-one warm start | 1.48856 | 1.94580 | 74.58% | 0.10339 | 82.46% | 76.19% | 97.92% |
| S4 exact relations | 1.60658 | 2.08103 | 70.42% | 0.09982 | 87.72% | 76.19% | 97.50% |

Exact relations slightly beat full R2 on retained regret, but missed every
other rescue dimension. A single favorable aggregate metric cannot rescue a
representation under the frozen conjunctive contract.

## Additional Fine-Tuning Degraded The Substrate

The S4 independent arm is the matched control for context, but it is also a
direct measurement of another 3,000 steps of the shared S4 training recipe on
the failed radius-one warm start.

Relative to that warm start, the independent arm changed by:

| Metric | R3 radius one | S4 independent | Change |
|---|---:|---:|---:|
| MAE | 1.48856 | 1.62602 | +0.13746 |
| RMSE | 1.94580 | 2.10325 | +0.15745 |
| Top-64 recall | 74.58% | 68.33% | -6.25 pp |
| Top-64 regret | 0.10339 | 0.11308 | +0.00969 |
| Confidence coverage | 97.92% | 97.50% | -0.42 pp |
| Low-supply recall | 82.46% | 82.46% | +0.00 pp |
| Independent-draft recall | 76.19% | 80.95% | +4.76 pp |

Training loss continued to improve while full validation worsened. This is
not evidence that more epochs are needed. It closes the exact additional
joint-fine-tuning recipe for this substrate and loss.

The experiment does not isolate whether the degradation comes from the
560-group training distribution, the shared loss, the additional optimization
horizon, or interaction with the iso-graph context modules. Those are possible
forensic questions, not authorization for another radius-one rescue.

## Serving Performance

| Arm | Fixed scores/s | Complete P99 | Peak active | Peak RSS | Process swap |
|---|---:|---:|---:|---:|---:|
| Independent | 35,415 | 233.46 ms | 482.6 MB | 2,489.6 MB | 0 |
| Inducing 16 | 35,647 | 259.54 ms | 484.8 MB | 2,543.6 MB | 0 |
| Exact relations | 35,933 | 230.60 ms | 484.9 MB | 2,484.6 MB | 0 |
| Combined | 36,047 | 284.07 ms | 482.5 MB | 2,636.0 MB | 0 |

All arms passed complete coverage, finiteness, exact parent/anchor encode
counts, 20,000 scores/s, 4 GiB active memory, 4 GiB RSS, and zero process swap.
Inducing and combined failed the absolute 250 ms complete-decision P99 gate.

Against the full-R2 serving reference:

| Gate | Full-R2-derived bound | S4 range | Result |
|---|---:|---:|---|
| Throughput | at least 21,552 scores/s | 35,415-36,047 | pass |
| Complete P99 | at most 229.16 ms | 230.60-284.07 ms | fail |
| Peak active | at most 465.4 MB | 482.5-484.9 MB | fail |
| Peak RSS | at most 1,193.1 MB | 2,484.6-2,636.0 MB | fail |

The 256-anchor iso-graph cost is not a practical replacement for the existing
full-R2 path. It is approximately 2.1 times the allowed RSS bound and narrowly
misses even the relaxed relative latency and active-memory bounds in its best
case.

## Classification

The deterministic classifier resolved:

- independent-control absolute evidence and serving gates: passed;
- inducing matched-control noninferiority: failed;
- exact-relations matched-control noninferiority: failed;
- combined matched-control noninferiority: failed;
- exact-relations material context effect: passed on top-64 recall;
- every full-R2 quality rescue: failed;
- every full-R2 relative serving rescue except throughput: failed;
- forward/reverse order invariance: passed; and
- gameplay, promotion, and progress-to-100 claims withheld: passed.

The terminal verdict is:

```text
s4_candidate_context_mlx_all_treatments_degraded
selected_arm = null
compact_representation_rescued = false
promotion_authorized = false
```

## Consequences

1. Do not advance the independent, inducing, exact-relations, or combined S4
   checkpoint to gameplay.
2. Keep exact sparse full-R2 afterstates as the accepted independent-candidate
   quality and serving reference.
3. Close 256-anchor inducing context and the present bounded relation-segment
   architecture as a compact-rescue path.
4. Preserve the exact-relation signal as feature evidence: shared public
   relations can help low-supply and middle-game ranking, but must be expressed
   more selectively and without harming independent drafts.
5. Do not continue another 3,000-step joint fine-tune on the radius-one
   substrate under this loss and corpus.
6. Move the primary representation lane to R4 adaptive multi-resolution
   foundations: exact 61/91-cell near fields plus explicit far components,
   motifs, and overflow, compared against the accepted sparse full-R2 control.
7. Reuse relation facts only as bounded local or summary features in R4, S3,
   or S5 ablations. Do not carry the 256-anchor set model forward by default.
8. Keep 441 dense cells closed. A regular compact window must use 91 or 127
   cells plus exact overflow; 121 is not a centered hex-disk cardinality.
