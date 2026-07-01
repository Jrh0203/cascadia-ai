# Exact Semantic Supply Learned Comparison V1 Result

Date: 2026-06-17

ADR: 0147

Experiment: `exact-semantic-supply-learned-comparison-v1`

Protocol: `s1-exact-semantic-supply-mlx-comparison-v1`

Status: completed

Classification: `exact_supply_learned_comparison_control_failed`

Promotion authorized: false

## Executive Result

The complete 17-task, four-host S1 campaign finished with valid, byte-stable
evidence and a passing independent replay. The frozen classifier rejected the
experiment because the C0 benchmark consumed 206,905,016 bytes of system swap.
Even without that precedence gate, neither exact-supply arm cleared the full
promotion contract.

T2 relational supply produced the strongest aggregate prediction signal:
R4800 MAE improved by 0.13798, RMSE improved by 0.13297, top-64 winner recall
rose by 0.02083, and retained regret fell by 0.02271 versus C0. Those gains did
not generalize to the preregistered supply-sensitive slices. Low-supply recall
fell by 0.01754 and independent-draft-winner recall fell by 0.14286. T2 also
used 1.5735 times C0 process RSS, above the 1.50 limit, and its refill fidelity
was only 0.49739.

Exact semantic supply is therefore not promoted in this architecture. The
result supports a narrower future hypothesis: candidate-relational supply can
help aggregate value and shortlist quality, but it needs a better localized
objective and representation than a global refill-decoder auxiliary loss.

## Immutable Identity

| Identity | BLAKE3 |
|---|---|
| Bundle | `2baae4acb5a5375e056ae56e019180e57b98a5d032a7c5825357c93e6d2bf23c` |
| Authorization | `954d0bb2e1bb1d8dca32cf9109f4d21c2525c4664c3344ef51b4435f11e0afef` |
| Exact cache | `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15` |
| Collision witness | `b860814dfe1c16ca9f4c17f574b7d0040ab684ed1bfbcb1fe262395ec84af447` |
| Parameter layout | `f3d723afd7b938d01137b6587d98a3abf7b37217507ebc152b4d1d18413bbd2d` |
| Queue graph | `200cfb6fc1c241cb919a6b6ea01a3247724343e2db0a0c873d7fce96d7849e5d` |
| Aggregate | `e5464b60453fedf5805008d7adbdb75b17f4bf62100a6c62123ccaed0946172e` |
| Classification bytes | `db3e7a2d49f8b47910cc0490a9098c794a3ac9a89a9c9026baa776769a4d733e` |

Forward and reverse classification files were byte-identical.

## Execution

| Host | Role | Arm | Epochs | Stop |
|---|---|---|---:|---|
| john1 | Control training | C0 legacy marginals | 7 | Validation futility |
| john2 | Treatment training | T1 exact counts | 7 | Validation futility |
| john3 | Treatment training | T2 relational supply | 9 | Validation futility |
| john4 | Independent replay | Full corpus and D6 control | N/A | Passed |

All arms used 3,073,101 trainable parameters, identical parameter layouts,
identical seeded initial tensors, the same 560 train decisions, the same 240
validation decisions, and the same 2,995,314 complete legal actions.

John4 independently verified every public binding, all 12 D6 inverse round
trips, all parameter and initialization controls, and a clean hidden-information
boundary.

## Validation Results

| Arm | R4800 MAE | R4800 RMSE | Top-64 recall | Top-64 regret | Refill fidelity |
|---|---:|---:|---:|---:|---:|
| C0 legacy marginals | 1.57901 | 2.01861 | 0.73750 | 0.10240 | 0.40116 |
| T1 exact counts | 1.59921 | 2.08739 | 0.74583 | 0.09887 | 0.40912 |
| T2 relational supply | **1.44103** | **1.88565** | **0.75833** | **0.07969** | **0.49739** |

T1 failed value noninferiority because its RMSE increased by 0.06877, above
the allowed 0.05. T2 passed the aggregate value gates and the top-64 recall and
regret gates.

The decisive T2 slice results were:

| Slice | C0 recall | T2 recall | Delta | Required |
|---|---:|---:|---:|---:|
| Low supply | 0.89474 | 0.87719 | -0.01754 | >= +0.02 |
| Independent-draft winner | 0.76190 | 0.61905 | -0.14286 | >= +0.02 |

This is not a small miss. The representation improved aggregate shortlist
metrics while becoming materially worse on the two situations it was designed
to solve.

## Performance

| Arm | Action scores/s | P99 decision ms | Peak active memory | Peak RSS | System swap delta |
|---|---:|---:|---:|---:|---:|
| C0 | 124,425 | 118.03 | 567.5 MiB | 526.8 MiB | 197.3 MiB |
| T1 | 206,281 | 39.93 | 683.3 MiB | 829.6 MiB | 0 |
| T2 | 206,318 | 39.38 | 683.4 MiB | 828.9 MiB | 0 |

T2 retained 1.658 times C0 throughput and 1.204 times C0 active MLX memory, but
its process RSS multiplier was 1.5735 and failed the 1.50 cap. The C0 system
swap delta failed the control viability gate and therefore determined the
classifier precedence.

## Classification

The following gates passed:

- complete validation and action coverage;
- finite outputs and content-addressed reports;
- exact capacity, initialization, optimizer, and D6 fairness;
- independent john4 replay;
- T2 aggregate value noninferiority;
- T2 top-64 recall and retained-regret improvement;
- absolute throughput, latency, active-memory, RSS, and process-swap limits;
- T2 throughput and active-memory ratios; and
- forward/reverse order invariance.

The following gates failed:

- C0 system swap delta;
- T1 value noninferiority;
- T1 and T2 refill fidelity;
- T2 low-supply recall improvement;
- T2 independent-draft recall improvement; and
- T2 relative process RSS.

The deterministic verdict is:

```text
exact_supply_learned_comparison_control_failed
model_promotion_authorized = false
gameplay_strength_measured = false
progress_to_100_claimed = false
```

## Consequences

1. Do not promote C0, T1, or T2 into gameplay.
2. Do not repeat the same global exact-count or refill-decoder treatment.
3. Preserve the factual semantic-supply foundation and collision witness.
4. Carry only the supported signal forward: candidate-relational supply may
   improve aggregate value and shortlist quality.
5. Test that signal inside R3 exact local-patch/global-edit action tokens with
   objectives tied directly to low-supply and independent-draft decisions.
6. Isolate future performance gates from unrelated system swap activity.
   This operational correction cannot reinterpret or promote the frozen S1
   result.
7. Treat a valid negative scientific classification as a successful scheduler
   task. The current classifier now reserves nonzero command status for
   malformed evidence only.
