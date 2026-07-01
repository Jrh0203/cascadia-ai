# Pattern Competition V1: First Rotation

Status: rejected on 2026-06-11.

## Question

Would exact public first-rotation wildlife refill and opponent consumption
preserve the phase-capped commitment policy's Bear gain while recovering its
non-Bear wildlife and habitat losses?

## Frozen Policy

`pattern-competition-v1-k8-h6-b8-m4-t2-first-rotation` retained the exact
K8+H6+B8 frontier, immediate base score, two-personal-turn phase cap, and
seeded ties. It replaced optimistic future supply with exact
without-replacement refill, free three-kind and automatic four-kind
replacement, and three opponent drafts chosen by exact one-token marginal
base-score gain.

Repeated four-kind paths were summed exactly by a generating-function dynamic
program. A small exhaustive reference test, probability-mass tests, hidden
redetermination invariance, legal seeded selection, and final-turn collapse
all passed.

## Runtime Gate

The initial exact implementation produced treatment mean 94.0 and delta
-0.25 on seed 25999 but required 5.206 seconds, missing the five-second gate.
A behavior-preserving arithmetic factorization reproduced those scores
exactly and reduced treatment runtime to 3.348 seconds, allowing the pilot.

## Pilot

Seeds 26000-26009:

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| Paired mean delta | +0.875 | >= +0.500 | Pass |
| Bear delta | +1.275 | >= +0.500 | Pass |
| Total wildlife delta | -0.250 | >= 0.000 | Fail |
| Non-Bear wildlife delta | -1.525 | >= -0.500 | Fail |
| Habitat delta | +0.400 | >= -0.500 | Pass |
| Runtime per treatment game | 10.316s | <= 5.000s | Fail |

The record was 6-0-4 and the paired 95% confidence interval was -0.151 to
+1.901. Nature Tokens improved by 0.725.

## Conclusion

Opponent-conditioned availability contains real score signal and repairs the
prior habitat loss, but expected-best-species continuation still shifts too
much value into Bear. The exact process also misses the parallel interactive
runtime budget. Three registered gates failed, so no 50-game confirmation was
run.

Artifacts:

- `docs/archive/v2/reports/pattern-competition-v1-runtime-smoke-1-initial-failed.json`
  (`7829cf6bdfc6b38fac29730429fe166ac03f034f5e07be716a2c07d0f16d636f`)
- `docs/archive/v2/reports/pattern-competition-v1-runtime-smoke-1.json`
  (`a833414b40a6a9e112564b472a82a75aa60592ab97e61d6f380702d38878f04e`)
- `docs/archive/v2/reports/pattern-competition-v1-pilot10.json`
  (`77412b5bdb8c37ba3aa0d9ab13100c8307e3bc5ce8faecf5f7ec001d34f69127`)
