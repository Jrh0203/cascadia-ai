# ADR 0147: Exact Semantic Supply Learned Comparison

Status: completed; control failed; no promotion

Date: 2026-06-17

Experiment: `exact-semantic-supply-learned-comparison-v1`

Protocol: `s1-exact-semantic-supply-mlx-comparison-v1`

Research-plan item: S1

## Context

ADR 0143 established a Rust-authoritative public semantic-supply foundation:
five wildlife counts, 75 canonical habitat-tile archetype counts, exact unseen
and drawable totals, and the still-hidden setup-exclusion count. Its production
census proved parity with the existing 30-value marginal representation,
hidden-order invariance, D6 invariance, and exact refill laws on 40,000 public
positions.

That foundation does not establish learned utility. A larger input can appear
better because it receives a larger projection, more trainable parameters, a
different optimizer stream, more examples, or candidate facts unavailable to
the control. S1 therefore requires an iso-capacity learned comparison in which
the legacy five-wildlife plus 25-tile marginals remain a real C0 control.

The factual non-identifiability is not hypothetical. The official tile pools
`[0, 23]` and `[2, 20]` have identical legacy marginals, while their exact
semantic archetype multisets are `[26, 72]` and `[24, 74]`. Their one-draw
refill laws differ. Any C0 prediction must alias this pair; an exact-supply arm
may separate it without reading hidden order.

## Decision

Implement one frozen MLX comparison over the existing open complete-action
graded-oracle train and validation rows. Production training, sealed/test data,
gameplay, promotion, and live queue or dashboard mutation remain prohibited
until explicit parent authorization.

### Arms

| Arm | Variable supply information | Candidate relation |
|---|---|---|
| `c0-legacy-marginals` | Five wildlife counts plus 25 legacy tile marginals | None |
| `t1-exact-counts` | Five wildlife counts, 75 exact archetype counts, unseen, drawable, excluded | None |
| `t2-relational-supply` | Same exact supply and refill law as T1 | Selected archetype and public frontier compatibility query exact supply |

C0 uses the already-normalized 30 values from the graded-oracle dataset. They
occupy the first 30 positions of a fixed 83-wide vector. Positions 30 through
82 are exactly zero. Its first 30 supply tokens contain those values; the
remaining tokens contain no state-dependent information.

T1 receives the exact 83-wide vector and 80 supply tokens: five wildlife tokens
and 75 semantic-archetype tokens. T1 does not receive selected archetype IDs or
frontier compatibility. Its supply attention query is parent-state-only.

T2 receives the same exact global supply as T1. It additionally receives the
public selected archetype and rotation-aware frontier compatibility facts
exported by Rust. Its complete-action query attends to the exact supply/refill
tokens. This is the sole relational treatment.

### Exact capacity match

Every arm instantiates the same module graph:

```text
architecture: s1-exact-supply-iso-complete-action-v1
hidden width: 128
attention heads: 4
board blocks: 2
market blocks: 1
supply blocks: 1
feed-forward multiplier: 3
supply vector projection width: 83
supply token count: 80
supply token feature width: 32
trainable scalar count: 3,073,101
```

The trainable parameter names, shapes, dtypes, and total count are hashed.
Authorization, preflight, control lock, arm reports, classifier, and independent
replay all require the three layout hashes to be identical and every count to
equal 3,073,101.

C0 does not receive a smaller projection. Its unavailable exact inputs are
zero or state-independent placeholders on the same 83-scalar, 80-token
surface. With seed `2026061707`, independent model construction must produce
byte-identical initial parameter tensors for all arms.

## Rust-Authoritative Cache

`tools/s1_exact_supply_mlx_exporter` is a standalone Rust crate. It imports the
accepted `cascadia-data` semantic-supply implementation without changing shared
source or the root workspace.

The exporter accepts only:

- `artifacts/datasets/complete-action-graded-oracle-v1-train`;
- `artifacts/datasets/complete-action-graded-oracle-v1-validation`; and
- optional bounded group counts that are marked smoke-only and incomplete.

For every decision group it reconstructs exact supply from the public
`PositionRecord`, proves all 30 legacy values match, checks physical/drawable
conservation, round-trips `CSSSUP1`, and binds the sidecar to the graded-oracle
public-state hash.

For every complete legal action it exports only:

- staged public wildlife counts;
- selected public tile archetype;
- six public neighbor-facing frontier requirements; and
- exact rotation-compatibility facts.

The candidate sidecar is bound to ordered graded-oracle action hashes. Python
rejects any group, candidate width, public-state hash, action hash, checksum,
shape, dtype, byte count, catalog hash, or content address that differs.

The cache records all of the following as false:

- hidden stack order read;
- hidden wildlife order read;
- excluded-tile identities read;
- future refills read;
- sealed/test data opened; and
- gameplay opened.

## Frozen Normalization

| Field | Divisor |
|---|---:|
| Legacy wildlife counts | 20 |
| Legacy tile marginals | 81 |
| Exact wildlife counts | 20 |
| Exact archetype counts | 2 |
| Exact unseen total | 81 |
| Exact drawable total | 79 |
| Exact exclusion count | 2 |

The refill target for archetype `i` is `count_i / unseen_total`. C0 uses the
same target as an information-decoding control even though factual collision
pairs make perfect C0 fidelity impossible. Unused C0 exact slots must remain
zero.

The cache embeds the ADR 0143 factual collision witness. Load-time validation
requires the exact physical IDs, archetype IDs, equal 30 marginals, unequal
refill numerators, and content-addressed witness identity.

## Shared Model And Objective

The existing complete-action graded-oracle board, market, global, action,
prior, staged-market, screen-value, and teacher fields are unchanged.

All arms use the same board and market set encoders, complete-action cross
attention, candidate-set pooling, bounded score residual, uncertainty head,
and 75-way refill decoder.

The scalar objective is identical:

```text
r1200 Huber
+ 4.0 * r4800 Huber
+ 0.5 * r1200 listwise cross entropy
+ 1.0 * r4800 winner cross entropy
+ 0.1 * standard-error calibration
+ 0.01 * screen-only residual regularization
+ 0.25 * exact refill cross entropy
```

No arm-specific loss coefficient, target, initialization, warm start, or
optimization budget is permitted.

## Frozen Training Protocol

| Variable | Value |
|---|---:|
| Seed | `2026061707` |
| Optimizer | AdamW |
| Epoch ceiling | 30 |
| Group batch size | 64 |
| Maximum padded actions per batch | 8,192 |
| Maximum actions in one group | 16,384 |
| Learning rate | `0.0001` |
| Weight decay | `0.0001` |
| Checkpoint interval | 250 optimizer steps |
| Validation patience | 6 epochs |
| Augmentation | Uniform full D6 per complete decision group |
| Initialization | Fresh, identical seeded tensors |
| Warm start | Prohibited |
| Additional data | Prohibited |

Every Python command in the frozen graph uses both `PYTHONDONTWRITEBYTECODE=1`
and `python -B`.

Training uses the generic atomic ranking checkpoint protocol. Resume is allowed
only from the same run directory after the control lock proves the same cache,
datasets, authorization, preflight, protocol, normalization, collision witness,
model configuration, and parameter budget.

## Evaluation

Every one of the 240 validation decisions and 860,203 complete legal actions
must be scored exactly once.

The report includes:

- R4800 MAE, RMSE, bias, correlation, calibration slope, and intercept;
- top-1, top-8, top-32, and top-64 R4800 winner recall;
- retained R4800 regret at the same widths;
- top-64 95% teacher-confidence-set coverage;
- exact refill total variation, cross entropy, probability MAE, mode accuracy,
  and fidelity;
- low-supply results where public unseen tiles are at most 20;
- decisions whose R4800 winner is an independent draft;
- complete-action throughput and decision latency;
- MLX peak active memory, process peak RSS, process swaps, and system swap
  delta; and
- full information-boundary and nonpromotion claims.

Stable score ties are broken by action hash.

## Frozen Gates

T1 and T2 must each remain value-calibrated relative to C0:

```text
MAE delta <= 0.05
RMSE delta <= 0.05
increase in abs(calibration slope - 1) <= 0.05
increase in abs(calibration intercept) <= 0.25
```

Both exact arms must reach refill fidelity at least `0.9999`.

T2 must additionally satisfy:

```text
top-64 R4800 winner recall delta versus C0 >= 0.02
top-64 retained-regret reduction versus C0 >= 0.01
top-64 95% confidence-set coverage >= 0.995
low-supply top-64 recall delta versus C0 >= 0.02
independent-draft-winner top-64 recall delta versus C0 >= 0.02
```

Every arm must satisfy:

```text
complete-action throughput >= 20,000 action scores/second
p99 decision latency <= 250 ms
MLX peak active memory <= 4 GiB
process peak RSS <= 4 GiB
process swaps == 0
system swap delta <= 0
```

T2 must retain at least 60% of C0 throughput and use no more than 1.5 times
C0 MLX active memory or process RSS.

## Deterministic Classification

Classification precedence is:

1. `exact_supply_learned_comparison_invalid_evidence`;
2. `exact_supply_learned_comparison_control_failed`;
3. `exact_supply_learned_comparison_exact_representation_failed`;
4. `exact_supply_learned_comparison_relational_success`; or
5. `exact_supply_learned_comparison_relational_null`.

Forward and reverse report orders must produce byte-identical aggregate JSON.
The classifier reconstructs each arm's scientific identity from its actual
metrics, model, normalization, collision, host, checkpoint, and information
boundary before accepting the report hash.

No classification authorizes gameplay, model promotion, queue mutation, or a
claim of progress toward 100 mean.

Valid negative classifications are completed scientific artifacts, not command
failures. The CLI exits nonzero only for malformed evidence; control failure,
exact-representation failure, and relational null all exit successfully after
writing their deterministic classification.

## Immutable Authorization And Preflight

One parent-created authorization must pin:

- experiment, protocol, and ADR;
- immutable bundle and source IDs;
- exporter executable BLAKE3;
- content-addressed cache and catalog;
- both open dataset manifests;
- complete optimizer and D6 protocol;
- normalization and all arm input contracts;
- collision witness ID;
- exact cross-arm parameter counts and layout hashes;
- all three authorized arms;
- independent replay role;
- approver; and
- approval timestamp.

Each host preflight verifies immutable source, authorization, cache, datasets,
Apple Silicon, MLX GPU, assigned host, `python -B`, runtime identity, parameter
fairness, and that production training has not started.

## Inert Four-Host Graph

The generated graph contains 17 tasks and cannot install itself:

1. fan out bundle;
2. fan out cache;
3. fan out train data;
4. fan out validation data;
5. fan out authorization;
6. preflight john1;
7. preflight john2;
8. preflight john3;
9. preflight john4;
10. train C0 on john1;
11. train T1 on john2;
12. train T2 on john3;
13. independently replay all public bindings, D6 transforms, capacity, and
    seeded initialization on john4;
14. checksum-collect three arm reports and one replay;
15. classify forward;
16. classify reverse; and
17. prove byte-identical classification order.

## Result

The campaign completed on all four hosts. John4 replay passed every public
binding, D6, capacity, initialization, and hidden-information check. Forward
and reverse classifications were byte-identical with classification BLAKE3
`db3e7a2d49f8b47910cc0490a9098c794a3ac9a89a9c9026baa776769a4d733e`.

The frozen verdict is
`exact_supply_learned_comparison_control_failed`. C0 consumed 206,905,016 bytes
of system swap during its benchmark. T2 nevertheless showed descriptive
aggregate gains versus C0: R4800 MAE `-0.13798`, RMSE `-0.13297`, top-64 recall
`+0.02083`, and retained-regret reduction `+0.02271`. It failed the intended
low-supply and independent-draft slices, refill fidelity, and the 1.50 process
RSS ratio.

No arm is promoted. Future work may reuse the exact semantic-supply foundation
only under a newly preregistered candidate-local objective, preferably through
the R3 local-patch/global-edit representation. Full evidence is recorded in
`docs/v2/reports/exact-semantic-supply-learned-comparison-v1-result.md`.

The three training arms are nonduplicative. John4 performs an independent
control rather than a fourth training copy. The queue specification sets
`applied: false`, exposes no live queue path, and contains no apply operation.

## Verification Completed Without Production Training

Implementation verification completed on 2026-06-17:

- 19 focused Python tests passed;
- 5 standalone Rust exporter tests passed;
- Ruff passed on every new Python and tool surface;
- strict Clippy passed with `-D warnings`;
- a bounded Rust smoke exported one train group with 306 actions and one
  validation group with 648 actions;
- Python verified the cache content address, tensor checksums, collision
  witness, and unsigned 64-bit group IDs; and
- all three arms rebound the same 306 authoritative action hashes and produced
  finite `[1, 306]` scores with 80 supply tokens.
- one real AdamW smoke update per arm produced the same 175 gradient tensors,
  finite losses, and finite gradients;
- the one-group model-only smoke measured approximately 81,000-86,000 action
  scores per second and 39 MiB MLX peak active memory on john1; and
- a final 192-source-file immutable bundle ran its own bounded `export-cache`
  entry point successfully, after which production authorization rejected the
  incomplete cache exactly as required.

The bounded smoke cache is explicitly incomplete and cannot pass production
authorization or preflight.

The reviewed pre-authorization export surface is:

```bash
PYTHONPATH=python:tools .venv/bin/python -B \
  tools/s1_exact_supply_mlx_campaign.py export-cache \
  --host john1 \
  --repository /path/to/bundle/source \
  --bundle /path/to/bundle \
  --train-dataset artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-dataset artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --output-root artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache \
  --receipt artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache-receipt.json \
  --output artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache-export.json
```

The campaign refuses authorization unless the cache manifest records the exact
exporter executable BLAKE3 from that immutable bundle.

## Production Authorization

John Herrick authorized the complete four-host S1 campaign through
`CASCADIA_V2_GOAL.txt` on 2026-06-17. The production cache covers all 560
open train decisions, 240 open validation decisions, and 2,995,314 complete
legal actions. It was generated by the immutable bundle's release exporter and
retains every frozen public-information boundary.

| Identity | Value |
|---|---|
| Bundle | `2baae4acb5a5375e056ae56e019180e57b98a5d032a7c5825357c93e6d2bf23c` |
| Authorization | `954d0bb2e1bb1d8dca32cf9109f4d21c2525c4664c3344ef51b4435f11e0afef` |
| Cache | `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15` |
| Cache manifest BLAKE3 | `a99d1aad79be950eb030fc56a3340205031a7ed0ebabe9980d7b49e3584b16c1` |
| Exporter BLAKE3 | `b1cb47bd848632597414b6636b05c6697291b3db19f2df923d26484d57476a84` |
| Collision witness | `b860814dfe1c16ca9f4c17f574b7d0040ab684ed1bfbcb1fe262395ec84af447` |
| Parameter layout | `f3d723afd7b938d01137b6587d98a3abf7b37217507ebc152b4d1d18413bbd2d` |
| Queue task graph BLAKE3 | `200cfb6fc1c241cb919a6b6ea01a3247724343e2db0a0c873d7fce96d7849e5d` |

All three arms contain exactly 3,073,101 trainable scalars and have the same
parameter-layout hash. The live campaign contains exactly seventeen tasks:
five immutable fanouts, four host preflights, three nonduplicative training
arms, one independent replay/control, report collection, forward and reverse
classification, and the byte-identical order proof.

## Consequences

This experiment can determine whether exact public supply is useful by itself,
whether candidate-to-supply relations are the missing inductive bias, and what
throughput and memory they cost. It cannot attribute a win to extra trainable
capacity, different examples, a larger input projection, hidden order, or host
duplication.

The parent review, immutable bundle, complete cache, explicit authorization,
and inert queue review are complete. Live execution remains fail-closed on all
four host preflights and the exact queue dependencies.
