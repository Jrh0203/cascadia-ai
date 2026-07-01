# Full-Legal Public Oracle V1 Invalid Launch

- Status: **invalid; excluded from every substantive gate**
- Date: 2026-06-15
- Attempted seeds: `62000-62011`
- Replacement seeds: `62020-62031`

## Cause

The online oracle reused the offline audit ladder. That ladder required the
R4800 cohort to contain an action retained from the complete-screen prefix so
it could report screen truncation regret. Online selection does not use that
diagnostic: it plays the highest-ranked R4800 action from the preregistered
union. On seed `62000`, the R4800 cohort contained champion-frontier and
sentinel actions but no retained screen action, so the run aborted.

The correction makes the diagnostic optional in the shared ladder and keeps
the strict requirement at the point where an offline audit record is built.
The online candidate set, rollout budgets, common-random-number coupling, and
winner selection are unchanged.

## Preserved Outcomes

| Host | Attempted seeds | Observed state | Disposition |
|---|---|---|---|
| john1 | `62000-62003` | Aborted after 25.13 seconds before a result artifact | Invalid implementation smoke |
| john2 | `62004-62007` | Four paired games completed before shutdown | Invalid and unblinded |
| john3 | `62008-62011` | One of four pairs reached progress output; no result artifact | Invalid partial execution |

The invalid john2 shard reported 95.0625 baseline, 99.8125 treatment, and a
+4.750 paired mean. Those numbers are retained for provenance only. They are
not a pilot result, are not combined with replacement data, and do not
authorize confirmation.

Because scores from the original domain became observable, the corrected
pilot moves to fresh seeds `62020-62031`. The preregistered thresholds and
sealed confirmation seeds `62100-62139` do not change.

Raw telemetry is preserved under
`artifacts/experiments/full-legal-public-oracle-v1/invalid-pre-fix/`.
The machine-readable ledger is
`docs/v2/reports/full-legal-public-oracle-v1-invalid-launch.json`.
