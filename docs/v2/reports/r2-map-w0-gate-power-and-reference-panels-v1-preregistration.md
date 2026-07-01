# R2-MAP W0 Gate Power and Reference Panels v1 Preregistration

Date: 2026-06-18

Campaign: `r2-map-expert-iteration-v1`

Status: frozen before the first R2-MAP candidate outcome is opened; paired-SD
calibration is provisional until the first scientifically eligible focal-seat
incumbent-versus-control pilot is supplied explicitly

## Decision

The development gate remains exactly:

1. one 20-pair integrity/runtime smoke whose strength outputs are blinded;
2. one fixed 250-pair focal-seat candidate/control comparison;
3. promotion only when the paired mean is positive, its two-sided 95% interval
   excludes zero, and all preregistered integrity, guardrail, resource, memory,
   and zero-swap gates pass; and
4. no outcome-driven extension, optional stopping, sample-size search, or reuse
   of smoke strength.

The machine-readable calculation is
[`r2-map-paired-gate-power-v1.json`](r2-map-paired-gate-power-v1.json), with
internal analysis SHA-256
`484c1a9e14f5ada613d79b06085425e5ca84f6f3a9750721917cd41434d2b9ab`.
The tool is [`tools/r2_map_paired_gate_power.py`](../../../tools/r2_map_paired_gate_power.py).
It has no repository-discovery path: it accepts either one explicitly named
JSON file containing raw paired deltas or explicit sensitivity SD values.

## Pilot compatibility audit

The strongest accepted historical paired report located for this calibration
audit was
[`canonical-redetermination-strong-requalify50.json`](canonical-redetermination-strong-requalify50.json):
50 Card-A/no-habitat-bonus seeds, paired SD 0.938, file SHA-256
`08f8501d235fb89ea68f3fb3457e5b1e99adff995c60a0a4cabc3a3723fa7f32`.

It is not a compatible calibration pilot. Each historical pair is the
difference between the four-seat mean from an all-treatment game and the
four-seat mean from an all-baseline game. The R2-MAP gate instead measures one
candidate focal-seat score minus one incumbent focal-seat score while the same
three frozen historical opponents occupy the other seats. Averaging four
correlated seats changes the estimand and suppresses variance. Importing 0.938
would therefore overstate the new gate's power.

Accordingly, no observed compatible pilot SD is claimed:

| Field | Frozen value |
|---|---:|
| Eligible pilot pairs | 0 |
| Observed paired mean | unavailable |
| Observed paired SD | unavailable |
| Calibration status | provisional sensitivity |

This is intentionally conservative. The first compatible pilot may calibrate a
future planning report only when its raw focal paired deltas are supplied by an
explicit path and immutable source identity. It cannot change the already
frozen 20+250 rule for the current campaign.

## Fixed-250 sensitivity

The calculation uses a paired-mean normal approximation with two-sided
alpha 0.05 and 80% target power. MDE is the alternative mean delta giving that
planning power at exactly 250 pairs.

| Assumed paired SD | MDE at 250 pairs | N for +0.25 | N for +0.50 | N for +0.75 | N for +1.00 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.354 | 503 | 126 | 56 | 32 |
| 3 | 0.532 | 1,131 | 283 | 126 | 71 |
| 4 | 0.709 | 2,010 | 503 | 224 | 126 |
| 5 | 0.886 | 3,140 | 785 | 349 | 197 |
| 6 | 1.063 | 4,521 | 1,131 | 503 | 283 |

The table is sensitivity analysis, not a search over sample sizes and not a
strength claim. The gate still runs 250 pairs once. At SD 4, for example, it is
planned to detect about +0.71, while a true +0.50 effect would often remain
inconclusive. An inconclusive result retains the incumbent.

Assumptions are independent registered pair deltas, stable variance, an effect
chosen before outcome opening, and adequate normal approximation at N=250.
Limitations are that component/tail guardrails and systems gates can still
block promotion, the calculation has no finite-sample Student-t correction,
and training examples or D6 transforms do not increase benchmark N.

## Frozen reference panels

The machine-readable manifest is
[`r2-map-w0-reference-panel-manifest-v1.json`](r2-map-w0-reference-panel-manifest-v1.json).
Its canonical manifest SHA-256 is
`5d88e296810eb5f8c5abc67ebc317ce987a2edb11d97b0c4e55ea873d96e5a65`
(the formatted file SHA-256 is
`12555a92ab337eca8d299210e19f5c4bb52298822e82f688ad967ceeaed1f7ec`).

| Panel | Frozen panel SHA-256 | Contract |
|---|---|---|
| Maximum-width service | `a6b1f2c24eda1024a3be13c10b058b143effe2186c066a0069712d028756da16` | 6,372 actions, exact one-for-one scoring, stable order, no truncation |
| D6/public-only | `e95128e384d60a04fa575c86d4d8d2a498ec73dd8d11415d28ec0bf51085c153` | all 12 transforms, exact round trip and prediction symmetry, forbidden metadata excluded |
| Replay/Pinecone | `2f5bdc1f7147dd8b4b28443480181cc48ae32f11582967e7def97223a32bda6a` | sealed AAAAA replay, 20/80 focal/bootstrap decisions, score and earned/spent/remaining reconciliation |
| Checkpoint/resume | `7a1fe86b720f84b60463b0b19fb799b187c4504a9e189d21c78d593451fda8ca` | schema v2 model/optimizer/RNG/cursor/sampler/loss-head and next-batch exactness |
| Open 100-game performance | `850f3849397147b975f65a246598684c1923d209c22ded26c17216dcdf3c7019` | 100 unique domain-separated open seeds, balanced focal seat, no strength claim |

Each panel hash binds its canonical definition and the SHA-256 of every source
and regression-test file implementing the contract. Any implementation drift
causes `tools/r2_map_reference_panels.py verify` to fail until the change is
reviewed and a new manifest version is frozen.

The open 100-game performance panel is explicitly for reference/optimized
correctness and throughput comparisons. Its seeds are visible by design. It is
not a candidate gate, promotion panel, or final strength domain.

## Protected domains remain unopened

The manifest contains only public descriptors and commitments for:

| Domain | Count | Descriptor commitment SHA-256 |
|---|---:|---|
| Strength-blinded smoke | 20 pairs | `34a2dcc12a114549c2ba2e5f28d67b53dca52b8a402086c0a88a008fbe7cc610` |
| Fixed development gate | 250 pairs | `a58f04c60289261ec9eadb6d4fefcfa41011730d768ec775bf35845f34f1abfe` |
| Final strength | 1,000 games | `78cc76adb2dfd495579c8c34dc31c4babafb1683ab82b0bac64aca176a306923` |

No protected seed values, first seed, derivation secret, or seed commitment is
present. The manifest tool cannot accept or derive those values. A separate
sealed workflow may provision them only after the corresponding registered
phase barrier. This preregistration did not open any protected or final seed.

## Reproduction

Storage amendment (2026-06-18): the original external-volume reproduction
recipe is superseded and is not executable authority. ADR 0195 makes John2's
internal APFS campaign root the only writable data plane. The commands below
describe the historical container payload only. They must be submitted through
the signed container-job wrapper after D0; they must never be invoked as native
John2 `python3` or `uv run` commands. John1 must not run them or materialize
their cache, temporary, or result files.

```bash
export R2_ROOT=/campaign
export TMPDIR="$R2_ROOT/tmp/w0-prereg"

python3 tools/r2_map_reference_panels.py \
  verify docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.json

python3 tools/r2_map_paired_gate_power.py \
  --effect 0.25 --effect 0.50 --effect 0.75 --effect 1.00 \
  --sensitivity-sd 2 --sensitivity-sd 3 --sensitivity-sd 4 \
  --sensitivity-sd 5 --sensitivity-sd 6
```

Historical container payload for the focused tests:

```bash
UV_CACHE_DIR="$R2_ROOT/cache/uv" \
TMPDIR="$R2_ROOT/tmp/pytest-w0-prereg" \
uv run pytest -q \
  tools/test_r2_map_paired_gate_power.py \
  tools/test_r2_map_reference_panels.py
```

## W0 disposition

The power calculation and open reference panels are frozen. Statistical
calibration remains explicitly provisional, which is scientifically preferable
to binding an incompatible historical SD. Full training remains subject to all
other W0 schema and review gates in the campaign plan.
