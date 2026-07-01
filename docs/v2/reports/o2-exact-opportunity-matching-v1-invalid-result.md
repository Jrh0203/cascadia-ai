# O2 Exact Opportunity Matching v1 Invalid Result

**Completed:** 2026-06-17  
**Experiment:** `o2-exact-opportunity-matching-v1`  
**Protocol:** `o2-strict-train-top64-foundation-identifiability-v1`  
**Classification:** `o2_exact_foundation_invalid`  
**Result ID:** `40bf5b2fcfc8c8ef90413291c84066d5398131cf0288e5fd092489ac4987485c`

## Verdict

The O2 v1 foundation campaign is invalid because its frozen global
action-hash uniqueness gate used the wrong identity domain. No identifiability
or B1 treatment analysis is authorized from this run.

The production exporter completed all 560 group files and 35,840 candidate
rows, then correctly exited nonzero at terminal accounting. The
preregistration required all 35,840 action hashes to be globally unique. Only
21,009 were distinct; 14,831 rows reused an action hash seen in another
decision group.

This is not a cryptographic collision and not evidence corruption.
`canonical_complete_action_hash` hashes the serialized `TurnAction` under a
fixed domain and intentionally does not include public-state identity.
Identical action values in different public states must therefore share a
hash.

## Exact Accounting

| Measurement | Observed |
|---|---:|
| Group files | 560 |
| Candidate rows | 35,840 |
| Distinct action hashes globally | 21,009 |
| Rows beyond global uniqueness | 14,831 |
| Repeated hash values | 6,574 |
| Maximum cross-group multiplicity | 20 |
| Distinct `(group_id, action_hash)` pairs | 35,840 |
| Duplicate group-action pairs | 0 |
| Groups with within-group duplicates | 0 |

The 560 ordered group artifacts have aggregate BLAKE3
`afe30e9546f6b6080a6d48f5b5534cee75bd78408177d715f42d6f1aa9e8ed29`.
The terminal runner recorded exit code 1 and stderr:

```text
Error: "O2 candidate accounting or uniqueness failed"
```

## Why The Run Cannot Be Reclassified

The scientifically meaningful identity is the state-action pair, represented
here by `(group_id, action_hash)`. Those pairs are all unique, and every
within-group action set is unique. However, v1 explicitly froze global action
hash uniqueness as an integrity gate. Replacing that gate after seeing the
terminal result would be a post-hoc protocol change. The correct classification
is therefore invalid, not null or passed.

The group rows remain immutable invalid-run evidence. They are not used to fit
the residual probe, calculate F2, select B1 actions, inspect protected slices,
or authorize learned O2.

## Execution Recovery Evidence

The original PTY-owned process ended when its agent slot rotated after row
225. The exporter is resumable by construction: it replays the complete game
stream, loads each existing group, verifies schema, experiment, protocol,
cohort row, group ID, row count, invariant counts, and recomputed canonical
group-result ID, then continues. A tested typed host-local runner resumed the
same immutable source and inputs, reached all 560 groups, and recorded its
terminal exit status and progress.

The durable runner also demonstrated why a launchd wrapper was not used for
this external SSD: the launch-agent context could not open paths on the
removable exFAT volume. The successful recovery used a detached tmux session,
a supervisor and child PID, typed progress, dedicated SSD logs, and an atomic
terminal lifecycle record.

## Consequences

- O2 v1 does not pass its foundation gate.
- F2 identifiability and B1 deterministic treatment remain unrun.
- Learned O2 arms are not authorized by this experiment.
- No validation, sealed test, gameplay, confirmation, or cross-host replay is
  authorized.
- No score or progress-to-100 claim is made.
- A successor would require a new preregistration and mechanically correct
  state-action identity contract; v1 must not be silently repaired or rerun
  under relaxed gates.

## Artifacts

- failure evidence:
  `/Volumes/John_1/cascadia-cluster/john1/runs/o2-exact-opportunity-matching-v1/production-v1/failure-evidence-v1.json`;
- lifecycle status:
  `/Volumes/John_1/cascadia-cluster/john1/runs/o2-exact-opportunity-matching-v1/production-v1/process-status.json`;
- terminal manifest:
  `/Volumes/John_1/cascadia-cluster/john1/terminal/result-manifest.json`;
- frozen source manifest:
  `/Volumes/John_1/cascadia-cluster/john1/inputs/o2-source-manifest-v1.json`.
