# Complete-Action Frontier Target Curriculum V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-target-curriculum-v1`

The authoritative protocol is
`docs/v2/decisions/0091-frontier-target-only-curriculum.md`.

One john2 MLX pilot warm-starts ADR 0089's selected checkpoint and changes
only optimization: target-set cross entropy becomes the sole loss and
checkpoint selection follows target-slot miss rate. The architecture, data,
labels, score range, proposal width, and selector remain frozen.

The pilot must at least double train target recall to 60%, reach 5% exact train
sets, achieve 50% validation target recall and 1% exact validation sets, and
preserve winner, confidence, regret, memory, swap, and integrity gates.

john3 and john4 run independent reachability and trajectory diagnostics while
john2 trains. A second seed, sweep, sealed test, gameplay, new teacher compute,
cloud, and external compute are prohibited.
