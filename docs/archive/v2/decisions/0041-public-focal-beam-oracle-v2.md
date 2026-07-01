# ADR 0041: Public Focal-Beam Oracle v2

Status: rejected after qualification on 2026-06-12.

## Context

The exact-hidden final-five W2/B16 focal beam established a real continuation
signal, but it is diagnostic-only. The first deployable public treatment used
only R4/B4 and was null at -0.075. ADR 0038 later showed that public R8/B16
candidate values are highly repeatable across independent redetermination
batches, while ADRs 0039 and 0040 failed to distill those values into a fast
MLX decision policy.

Before changing the state representation or teacher again, measure whether the
full fair public operator is itself strong enough online. A weak online oracle
would close this target; a material gain would localize the remaining problem
to cost and amortization rather than target quality.

## Frozen Treatment

Compare promoted strong directly with:

- final five personal turns;
- K8+H6+B8+W2 root and future focal frontiers;
- eight shared public hidden-state redeterminizations;
- independent width-16 focal beam per root candidate and determinization;
- frozen pattern-aware opponents and scalar beam heuristic;
- paired candidate-minus-pattern-anchor outcomes;
- one-sided 90% Student-t lower bound with
  `t(0.90, 7) = 1.4149239276488585`;
- largest positive lower bound, otherwise the exact pattern-aware anchor.

Before the cutoff, treatment is exactly pattern-aware. The actual hidden stack
and bag order are never evaluated. Candidate and anchor evaluations share the
same public samples.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed `31199`.
- Qualification: seeds `31200-31202`, only after smoke passes.
- Final personal turns: 5.
- Determinizations: 8.
- Beam width: 16.
- Frontier: K8+H6+B8+W2+M4.
- Confidence rule: one-sided paired c90.
- Execution: sequential four-seat blocks on local CPU.

Smoke must complete within 720 seconds per block, with P90 searched-decision
latency at most 60 seconds.

Qualification is a non-promotable mechanism diagnostic. It advances to a
separately registered optimization or amortization experiment only if:

- paired treatment-minus-strong mean is at least +0.50;
- treatment absolute mean is at least 93.50;
- total wildlife delta is at least 0.0;
- aggregate Elk+Salmon+Hawk+Fox delta is at least -0.50;
- habitat delta is at least -0.50;
- Nature Token delta is at least -1.0;
- runtime remains at most 720 seconds per block.

Three blocks cannot promote a strategy regardless of result. A paired gain
below +0.25 or treatment mean below 92.50 rejects this online operator as the
next research lever. Intermediate results do not permit changes to sample
count, beam width, frontier, confidence level, cutoff, heuristic, continuation
policy, seeds, or thresholds.

## Required Implementation Evidence

- configuration accepts the registered R8 sample count while preserving R4;
- unsupported sample counts are rejected;
- c90 uses the registered R8 Student-t critical value;
- hidden-order invariance, determinism, legality, and replayability remain
  covered;
- CLI provenance identifies every frozen treatment parameter;
- strict tests and lint pass before the runtime smoke.

## Commands

```bash
target/release/cascadia-v2 public-focal-beam-compare \
  --games 1 --first-seed 31199 --terminal-turns 5 \
  --determinizations 8 --beam-width 16 --wildlife-candidates 2 \
  --sequential \
  --output docs/v2/reports/public-focal-beam-oracle-v2-t5-r8-b16-w2-c90-runtime-smoke-1.json
```

If the smoke passes:

```bash
target/release/cascadia-v2 public-focal-beam-compare \
  --games 3 --first-seed 31200 --terminal-turns 5 \
  --determinizations 8 --beam-width 16 --wildlife-candidates 2 \
  --sequential \
  --output docs/v2/reports/public-focal-beam-oracle-v2-t5-r8-b16-w2-c90-qualification3.json
```

## Runtime Smoke

The registered seed completed successfully:

- promoted strong: 92.000;
- public R8/B16 beam: 91.500;
- paired delta: -0.500;
- treatment runtime: 369.167 seconds per four-seat block;
- treatment mean decision latency: 4,614.6 ms;
- treatment P90 decision latency: 7,612.4 ms;
- treatment maximum decision latency: 72,929.8 ms.

Both frozen smoke gates passed: runtime was below 720 seconds and P90 latency
was below 60 seconds. The single-block score is not a strength stop condition,
so the exact registered three-block qualification proceeds unchanged.

## Qualification Result

Across seeds `31200-31202`:

- promoted strong mean: 92.500;
- public R8/B16 beam mean: 92.167;
- paired delta: -0.333, 95% CI `[-0.987,+0.320]`;
- record: 0 wins, 2 ties, 1 loss;
- treatment runtime: 200.143 seconds per block;
- treatment mean decision latency: 2,501.8 ms;
- treatment P90 decision latency: 8,292.0 ms.

The treatment lost 0.500 total wildlife and 1.167 aggregate
Elk+Salmon+Hawk+Fox, while habitat fell 0.333 and Nature Tokens gained 0.500.
It failed the +0.50 paired-score gate, the 93.50 absolute-score gate, and both
wildlife guardrails. It also fell below the preregistered 92.50 absolute
rejection floor. Runtime, habitat, and Nature Token gates passed.

The full fair public operator is therefore not a strong online oracle, even
though its candidate-value labels are statistically repeatable. Further
distillation, architecture tuning, or additional samples of this target are
closed. The next experiment must change the state, target, or planning
mechanism itself.
