# Opportunity Paired-Panel Group-ID Normalization Result

- Date: 2026-06-17
- Decision: ADR 0172
- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Verdict: repair passed; production relaunch unblocked

## Result

The signed-to-unsigned group-ID normalization passed its unit, replay,
cross-host, and full-corpus checks.

| Gate | Result |
|---|---|
| Signed and unsigned boundary tests | passed |
| Previously failing ID `-5482088856184735585` | replayed successfully |
| Focused opportunity suite | 30 tests passed |
| Fresh immutable bundle | `249819771886a11d235c8f91193bbea8b44143c7da54338522689f99628572dd` |
| Four-host smoke proof | `7062f18ac75793819f4504502bf7e43a350240ece2a7c7ee7609c04618df540f` |
| Production authorization | `9d11d2f80d9e4578c14f2135ea92b00eeea7764be1bf4630509099d039d9a56b` |
| Untouched C0 control | `0adf66a73208cd454cdfd512c62c8a87c8e2b283f74846f16b592a61bc2932f3` |
| Complete paired panel | `dc1b05cfe9e1105737ccd33c0916edaf3b9e376e1d9a379b501fd743da5bf5fa` |

The full untouched-C0 build completed in 382.1 seconds over all 240 validation
decisions and 860,203 legal actions. Every candidate and group was scored
exactly once, all scores and uncertainties were finite, and the report emitted
its canonical paired-panel identity without overflow.

## Scientific Effect

This repair changes no model input, label, parameter, optimizer step, score, or
ranking. Signed and unsigned JSON representations of the same opaque 64-bit
group hash map to the same `uint64` bit pattern before panel alignment and
identity construction.

The first production launch remains invalid and excluded. The repaired launch
uses fresh run directories and starts all four arms from the common exact-R2
warm start under one immutable source identity.

## Operational Effect

After the C0 control passed, the scheduler immediately launched
`c0-parent-conditioned` on john1. Together with the three treatment arms on
john2 through john4, all four hosts are now running distinct 2,000-step MLX
experiments. Each host has a separate exact-materialization qualification task
queued behind its arm.
