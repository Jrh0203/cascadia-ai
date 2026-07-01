# ADR 0068: Canonical Redetermination and Strong Requalification

Status: requalification failed and strong demoted on 2026-06-12.

## Context

ADR 0067's hidden-order invariance test found that the shared
`GameState::redeterminize_hidden` operator shuffled the existing concealed
vectors directly. Fisher-Yates still produced the correct uniform
distribution, but for a fixed public sample seed the concrete sampled world
depended on the source game's inaccessible hidden ordering.

That violates the deterministic public-policy contract. The engine now
canonicalizes the unseen tile multiset by tile ID and the wildlife multiset by
species before applying its domain-separated seeded shuffles. The same public
state and sample seed therefore reconstruct the identical sampled world from
every compatible hidden state.

The correction changes the finite deterministic samples consumed by the
promoted strong policy. Its historical confirmation remains evidence for the
algorithmic distribution, but it is not evidence for the corrected
deterministic implementation. Strong must be requalified before remaining a
promoted product tier.

## Decision

Run one fresh 50-block confirmation with no algorithmic changes:

- baseline: `pattern-aware-v1-k8-h6-b8-m4`;
- treatment:
  `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`;
- final-five cutoff;
- original K8+H6+B8 frontier;
- eight shared canonical public redeterminizations;
- frozen pattern-aware continuation;
- one-sided paired c90 with the existing R8 Student-t critical value;
- exact pattern-aware anchor fallback;
- canonical four-player AAAAA base score;
- sequential local CPU execution.

Seeds are the fresh contiguous suite `35100-35149`. No pilot or parameter
selection is permitted because this is a correction-only requalification of
an already frozen strategy.

## Gates

Strong remains promoted only if all original confirmation gates pass:

- paired 95% confidence-interval lower bound above zero;
- Bear delta at least 0.0;
- total wildlife delta at least 0.0;
- Elk+Salmon+Hawk+Fox delta at least 0.0;
- habitat delta at least -0.5;
- Nature Token delta at least -1.0;
- treatment runtime at most 12 seconds per block;
- treatment P90 decision latency at most 1.2 seconds;
- deterministic legal replay and complete provenance.

Failure demotes strong to research status and makes pattern-aware the product
strong tier until a newly confirmed strategy replaces it. The result does not
authorize tuning or a second correction suite.

## Pre-Run Evidence

- all 50 rules tests pass;
- all 60 search tests pass;
- both CLI tests pass;
- strict focused Clippy passes with `-D warnings`;
- the release binary is built from the corrected engine;
- direct redetermination tests prove identical complete sampled hashes after
  arbitrary prior hidden permutations;
- ADR 0067's corrected strong baseline completed ten legal replayable blocks
  at 3.371 seconds per block and 172.6 ms P90 decision latency.

## Command

```bash
target/release/cascadia-v2 \
  late-conservative-base-policy-improvement-compare \
  --games 50 --first-seed 35100 --terminal-turns 5 \
  --policy-candidates 8 --policy-habitat-candidates 6 \
  --policy-bear-candidates 8 --policy-market-draws 4 --sequential \
  --output docs/v2/reports/canonical-redetermination-strong-requalify50.json
```

## Result

Across fresh seeds `35100-35149`:

- pattern-aware mean: 91.580;
- corrected terminal-search mean: 92.100;
- paired delta: +0.520, 95% CI `[+0.260,+0.780]`;
- record: 31 wins, 10 ties, 9 losses;
- Bear delta: +0.460;
- total wildlife delta: +0.085;
- Elk+Salmon+Hawk+Fox delta: -0.375;
- habitat delta: +0.365;
- Nature Token delta: +0.070;
- treatment runtime: 3.063 seconds per block;
- treatment P90 decision latency: 172.6 ms.

The corrected strategy retained a statistically clear total-score gain and
passed Bear, total wildlife, habitat, token, runtime, latency, legality,
replay, and provenance gates. It failed the original non-Bear wildlife
guardrail. The gain is materially concentrated in Bear and no longer qualifies
as the balanced product promotion confirmed by ADR 0024.

Per the frozen decision, the strategy is demoted from the product `strong`
tier to an explicit `research` tier. Pattern-aware is now the strongest
promoted product policy. The terminal strategy and its result remain available
for research comparisons; no parameter retry is authorized.

The report is
`docs/v2/reports/canonical-redetermination-strong-requalify50.json` with
BLAKE3
`53bb18c705728533a154115d4972c2e45af1af3857d7b8f5d3cdd8085b03d47d`.
