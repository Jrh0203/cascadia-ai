# Full-Legal Paid-Screen Cache Preregistration

Status: **accepted**

Date: 2026-06-15

## Evidence

A two-token paid-wipe diagnostic performs 842 complete-screen requests:
the no-wipe value, 120 first-wipe chance branches, 720 followup planning
screens, and one realized recursive continuation. Every screen enumerates,
prepares, deduplicates, and evaluates its complete legal action set.

Chance branches can produce the same complete public state through different
hidden determinizations. The exact screen value depends only on that public
state: legal actions, board scores, visible market, nature tokens, public
supply reconstruction, sparse NNUE features, and tie-breaking are all public.

## Treatment

Create one bounded cache per paid-wipe diagnostic. For each complete-screen
request:

1. construct the complete `PublicGameState`;
2. use its canonical BLAKE3 only to select a bucket;
3. confirm full `PublicGameState` equality before a hit;
4. reuse the complete screen choice only for an exact equality match.

This is not a digest-only cache. A collision falls through to an independent
evaluation and a second entry in the bucket. The production representation
interns exact invariant contexts (`GameConfig`, boards, player, and completed
turns), stores the exact market per entry, and retains the selected action
hash, exact `f64` value derived from the original `f32` sum, and nature-token
return flag.

Do not share entries across audited decisions or games. Keep chance seeds,
branch order, recursive policy, logical counters, and output order unchanged.
Use one treatment-capable binary and remove its switch after the verdict.

## Frozen Screen

- seed `60999`, completed turn `16`;
- R600 trajectory;
- accepted 4,096-row static cohorts and exact row deduplication;
- D8 root determinizations, D2 followup determinizations, width 3;
- all 15 first-wipe subsets;
- opposite balanced order on john2 and john3.

The frozen paid-wipe report SHA-256 is
`dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`.

## Gates

The treatment must:

1. reproduce the frozen report byte for byte;
2. prove hidden reorderings hit the same public-state entry;
3. retain equality checks after canonical-hash bucket selection;
4. reduce actual complete-screen evaluations by at least 5%;
5. improve both workers and by at least 1% combined in balanced crossover;
6. reduce the complete early/middle/late frozen audit;
7. keep cache memory bounded to one paid-wipe diagnostic and preserve clean
   shutdown.

Reject immediately if the exact public-state hit rate is below 5%.

## Outcome

Accepted. The frozen qualification reused 440 of 842 requests and reduced
complete-screen evaluations by 52.257%. Opposite-order crossover improved
john2 by 36.979%, john3 by 34.925%, and the combined mean by 35.966%.

The compact production path reproduced both frozen qualification hashes and
the complete audit semantic payload exactly. Complete audit wall time fell
from 212.191376 to 177.686057 seconds, a 16.261% improvement. Full evidence is
in `full-legal-audit-paid-screen-cache-acceptance-v1.md`.
