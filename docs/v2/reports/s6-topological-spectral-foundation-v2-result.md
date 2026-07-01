# S6 Topological And Spectral Foundation V2 Result

Date: 2026-06-17

ADR: 0165

Experiment: `s6-topological-spectral-foundation-v2`

Protocol: `s6-exact-topology-activation-census-v2`

Classification: `s6_topological_spectral_foundation_v2_authorized`

Outcome: passed

## Executive Result

S6 V2 passed every registered foundation gate on four disjoint ten-game
shards. The exact topology, marked-path, deterministic random-walk, and
Laplacian families are active on natural Cascadia states, D6 invariant,
compact, and fast enough for matched learned ablations.

This result is deliberately narrower than a model result. No S6 family
separated the existing S3 collision pairs, and no ranking or gameplay target
was used. The foundation authorizes causal MLX tests; it does not claim score.

## Corpus

| Host | Seeds | Positions | Board encodings | Runtime |
|---|---:|---:|---:|---:|
| john1 | 5,610,000-5,610,009 | 800 | 3,200 | 137.882 s |
| john2 | 5,620,000-5,620,009 | 800 | 3,200 | 94.930 s |
| john3 | 5,630,000-5,630,009 | 800 | 3,200 | 88.182 s |
| john4 | 5,640,000-5,640,009 | 800 | 3,200 | 97.742 s |
| **Total** | **40 games** | **3,200** | **12,800** | |

The timing game immediately preceding each seed block was isolated and
excluded from production counts.

## Exactness

| Check | Total | Failures |
|---|---:|---:|
| Topology decoder | 140,800 | 0 |
| D6 invariance | 38,400 | 0 |
| Registered adversarial cases | 16 | 0 |

All four reports used immutable bundle
`53e6860babd2f5fa36a1e8eb95f792bd3d4b37734f73a2e9bd8bc2a0e7453c6a`
and executable BLAKE3
`d7fc615e0c7252bfce88b4296220d43aaff94813e3cf4a61581abe341dce072d`.

## Activation And Cost

| Host | Topology unique | Path unique | Walk unique | Spectral unique | Full unique | Long-range boards | Hole boards | Isolated P99 | Median bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| john1 | 795 | 40 | 768 | 735 | 795 | 924 | 400 | 1.927 ms | 1,682 |
| john2 | 794 | 40 | 763 | 725 | 794 | 897 | 288 | 1.160 ms | 1,683 |
| john3 | 793 | 51 | 762 | 731 | 793 | 836 | 328 | 1.122 ms | 1,686 |
| john4 | 795 | 35 | 764 | 729 | 795 | 955 | 303 | 1.125 ms | 1,686 |

Every host exceeded the preregistered variation floors:

```text
topology >= 128
marked paths >= 16
random walk >= 128
Laplacian >= 128
full encoding >= 256
long-range boards > 0
```

Worst-host isolated P99 was 1.927 ms against a 2.000 ms limit. Worst-host
median encoding was 1,686 bytes against a 4,096-byte limit.

John1's parallel-corpus extraction P99 was 8.630 ms while the isolated P99
remained below the gate. This confirms the V2 measurement correction:
parallel contention is a throughput diagnostic, not serving latency.

## Collision Diagnostic

The four shards contained 22,356 collision pairs under the existing S3
diagnostic signature. Topology, path, random-walk, spectral, and combined S6
encodings separated zero of those pairs.

This does not invalidate S6. V1 established that these collision groups are
repeated equivalent signatures and cannot measure the intended information
gain. The zero remains useful negative evidence: S6 should enter only a
matched target-bearing ablation, never be promoted from novelty counts alone.

## Decision

1. Authorize topology, marked-path, random-walk, and spectral families for
   matched learned ablations.
2. Preserve exact decoding, D6, compactness, and isolated-latency tests as
   permanent regression gates.
3. Add S6 families independently before testing combinations, so predictive
   gains and serving costs remain attributable.
4. Require full R4800, protected-slice, complete-decision latency, memory, and
   swap gates before any family enters a serving model.
5. Make no gameplay or progress-to-100 claim from this foundation result.

## Provenance

- aggregate:
  `artifacts/experiments/s6-topological-spectral-foundation-v2/aggregate-forward.json`
- reverse aggregate:
  `artifacts/experiments/s6-topological-spectral-foundation-v2/aggregate-reverse.json`
- order proof:
  `artifacts/experiments/s6-topological-spectral-foundation-v2/order-proof.json`
- collection receipt:
  `artifacts/experiments/s6-topological-spectral-foundation-v2/control/production-collection.json`
- aggregate scientific BLAKE3:
  `8c82103d1b771317013981913f35d370e52a8dec6721af1992f181856f712196`

## Claim Boundary

S6 V2 proves exact, active, compact, portable feature foundations. It does not
prove predictive value, action-ranking improvement, gameplay strength, or a
higher mean score.
