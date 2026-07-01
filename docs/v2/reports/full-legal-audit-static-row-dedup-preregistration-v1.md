# Full-Legal Static-Row Deduplication Preregistration

Status: **accepted**

Date: 2026-06-15

## Evidence

The accepted 4,096-row static cohort reduced request overhead, but the frozen
turn-16 qualification still submitted 3,664,806 physical sparse rows and
855,088,343 sparse features. Complete legal screens contain actions that can
produce byte-identical NNUE afterstate rows even when their canonical action
identity differs.

The accepted rollout pipeline already uses collision-checked exact sparse-row
deduplication. Complete static screens currently bypass that implementation.

## Treatment

For each accepted 4,096-row complete-prior cohort:

1. fingerprint each ordered sparse feature row;
2. confirm full row equality within every fingerprint bucket;
3. submit one copy of each byte-identical row to exact MLX;
4. scatter predictions back to the original logical row order.

Reuse the rollout pipeline's tested deduplication implementation. Keep
complete hidden-vector screens and the 96-state rollout pipeline unchanged in
this experiment. No action, immediate score, feature row, prediction, or
logical diagnostic may change.

Use one treatment-capable binary for the source screen. Remove the experiment
switch after acceptance or rejection.

## Frozen Screen

- seed `60999`, completed turn `16`;
- R600 trajectory;
- two-token paid-wipe hidden-invariance qualification;
- D8 root determinizations, D2 followup determinizations, width 3;
- all 15 first-wipe subsets;
- accepted static cohort of 4,096 rows;
- opposite balanced order on john2 and john3.

The frozen report SHA-256 is
`dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`.

## Gates

The treatment must:

1. reproduce the frozen qualification report byte for byte;
2. preserve every logical row, value, action, selected mask, and recursive
   paid-wipe count;
3. reduce physical service rows and sparse features by the exact duplicate
   count;
4. improve wall time on both john2 and john3 in balanced crossover;
5. avoid material memory or reliability regression;
6. improve the complete early/middle/late frozen audit.

Reject if hashing, equality checks, materialization, or scatter overhead erase
the end-to-end gain.

## Outcome

Exact deduplication removed 22.978% of physical rows in the frozen paid-wipe
screen, improved both workers, and reduced the promoted complete audit from
`217.302693625` to `212.191376166` seconds.

Acceptance report:
[`full-legal-audit-static-row-dedup-acceptance-v1.md`](full-legal-audit-static-row-dedup-acceptance-v1.md).
