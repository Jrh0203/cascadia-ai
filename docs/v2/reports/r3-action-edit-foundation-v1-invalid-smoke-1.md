# R3 Action Edit Foundation Invalid Smoke 1

Status: invalid pre-production smoke; root cause repaired

Date: 2026-06-17

Experiment ID: `r3-action-edit-foundation-v1`

## Immutable Identity

| Field | BLAKE3 |
|---|---|
| Invalid bundle | `7c7dceb0c6c1273ec5986fa84df367f946fe01b9216c0cb37fb6bcdc5bbec68c` |
| Source bundle | `5047a1dd601a4922e95e98a09665f3982394e7223241948ea9492e96227689df` |
| Executable | `e67846bff186a7e545b0f677ece178215bff7062ef567fb7063e887c4331ed68` |

This bundle is permanently disqualified from production.

## Scope

The john4 run used a non-production smoke corpus:

- train first seed `4,100,000`;
- four train games;
- validation first seed `4,200,000`;
- four validation games;
- shard index `3` of `4`;
- paid-wipe sentinels enabled; and
- one all-D6 sentinel per position.

Modulo ownership reduced this to one train game and one validation game. No
production seed, sealed test, learned model, or gameplay benchmark was opened.

## Exact Failure Witness

The run failed closed at:

- raw seed `4,100,003`;
- game index `410,000,300`;
- completed turn `0`;
- paired market slot `1`;
- tile destination `(-1, 1)`;
- tile rotation `2`;
- no wildlife placement; and
- D6 transform `2`.

The source orientation reported eight frontier updates and 11 direct changed
coordinates. The transformed orientation reported seven frontier updates and
10 direct changed coordinates.

## Root Cause

`frontier_changes` compared raw R2 `FrontierToken` values. Those tokens contain
numeric habitat-component IDs assigned during traversal. D6 transformation can
renumber the same semantic component, so a raw token comparison can report a
change that does not exist in the public rules state.

The canonical output exposed the defect as a no-op update: canonical before and
after component content were equal even though their transient numbers
differed.

## Correction

Frontier equality now:

1. builds exact before and after habitat-component maps;
2. resolves every frontier component reference to terrain plus sorted member
   coordinates;
3. compares those semantic identities in the world frame; and
4. retains raw numeric IDs only in exact application edits.

The permanent regression
`d6_regression_seed_4100003_turn_zero_is_exact` freezes the complete witness.
The full Rust workspace tests and strict Clippy gate pass after the correction.

## Verdict

This smoke is invalid mechanical evidence and carries no positive or negative
scientific conclusion. It successfully prevented a malformed bundle from
entering the production queue. A new immutable bundle and fresh local and
john4 smoke are required before launch.
