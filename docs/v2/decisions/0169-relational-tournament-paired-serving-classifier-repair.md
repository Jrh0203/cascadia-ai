# ADR 0169: Relational Tournament Paired-Serving Classifier Repair

- Status: accepted
- Date: 2026-06-17
- Experiment: `relational-substrate-mlx-tournament-v1`
- Scope: evidence classification only
- Does not change: training, model tensors, data, quality thresholds, absolute
  treatment thresholds, material-efficiency thresholds, or gameplay claims

## Context

ADR 0161 intentionally collected a full C0 replay on every treatment host so
material efficiency could be compared under the exact treatment-host R6 binary.
The original classifier nevertheless required all four arm reports to share one
global R6 binary hash. C0 ran on john1 while treatments ran on john2 through
john4, so the classifier rejected the evidence before using the paired controls.

After correcting that mismatch, a second issue appeared: the classifier
required each slow C0 replay, and the original C0 report, to satisfy the
treatment's absolute production-speed gate. That makes a representation-speed
tournament unclassifiable precisely when its baseline is slow.

Finally, the swap predicate accepted only zero or unknown delta. A negative
delta, meaning system swap usage decreased during measurement, was incorrectly
reported as failure.

## Decision

### Common training identity

Arm reports must remain identical on:

- R3, relational, and S1 cache IDs;
- complete 3,000-step scientific batch trace;
- protocol and seed;
- parameter count and layout;
- initial parameter tensor;
- source digest;
- authorization ID;
- open-data verification ID.

Global R6 equality is removed from this training-identity comparison.

### Host-paired serving identity

For every treatment, the classifier independently requires its signed C0 replay
to match:

- treatment host and treatment arm;
- treatment R6 binary BLAKE3;
- original C0 report ID;
- C0 checkpoint manifest and model BLAKE3;
- C0 global step;
- cache IDs;
- authorization and open-data verification IDs;
- all replay assertions, including full validation coverage, isolation, exact
  R6 apply/undo parity, and byte-identical C0 checkpoint transfer.

### Baseline integrity versus treatment qualification

C0 reports and paired replays must have:

- all 240 validation decisions and 860,203 actions;
- exact R6 parity and zero apply/undo failures;
- finite positive throughput and finite nonnegative latency/memory;
- zero process swaps;
- no system-swap growth.

Only treatments must satisfy the frozen absolute production thresholds:

- P99 at most 250 ms;
- fixed-chunk throughput at least 20,000 actions/s;
- peak active memory and RSS at most 4 GiB.

The original C0 quality sanity thresholds remain unchanged.

## Frozen Repair

Bundle:
`artifacts/experiments/relational-substrate-mlx-tournament-v1/repairs/bb9865c64365c3db8006c614ce89a02f5c9ad61c0ce07dea38bbbb729ac20b6e`

| File | BLAKE3 |
|---|---|
| `tools/relational_substrate_mlx_report.py` | `a98d5cce544d79dc1f186642db8356999ddc27a987ee4732d00cbad1585f7f44` |
| `tools/test_relational_substrate_mlx_report.py` | `a22a15d455a83fe16f9eeb850fd11d895df638b66ebe35281f3965266f7f7f3b` |

## Verification

- Ruff: pass.
- Classifier tests: 12 passed.
- Forward/reverse classification is byte-identical.
- Final classification:
  `relational_substrate_mlx_control_failed`.
- Aggregate ID:
  `f8a3e5d8420e02895516b6876c384f822966484a0a4c3de3054ae43e675dea7f`.
- Order-proof ID:
  `17bac09b9aa927a5d4fbcccf8b8396e08dfa8d80d2fc76de199d39f6336c4a4e`.

## Consequences

The evidence is valid, but no arm advances. D3 passes every relative quality
gate and every material-efficiency gate, yet all treatments fail absolute
serving. C0 also misses four frozen quality sanity gates. No gameplay
qualification, champion change, or progress-to-100 claim is authorized.
