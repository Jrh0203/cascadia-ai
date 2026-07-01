# R4 Adaptive Multi-Resolution Foundation V1 Result

Date: 2026-06-17

ADR: 0154

Experiment: `r4-adaptive-multires-foundation-v1`

Schema: `r4-focal-nearfield-topology-v1`

Status: completed

Classification: `r4_adaptive_multires_compactness_failed`

MLX comparison authorized: false

## Executive Result

R4 proved that an exact 61- or 91-cell focal near field with an exact
overflow sidecar is mechanically sound, deterministic across the four Macs,
and substantially smaller than the historical 441-cell lattice. It did not
prove that the registered all-topology model view is compact enough for the
matched MLX comparison.

Every exactness and information-preservation gate passed:

- all 60,000 frozen positions and eight dataset identities were present
  exactly once;
- both radii completed 60,000 codec round trips, 60,000 R2 semantic equality
  checks, 720,000 D6 inverse checks, and 60,000 target-independence checks;
- all four host preflights produced the same adversarial scientific identity;
- every registered habitat, wildlife, frontier, overflow, and opponent
  distinction was retained by HWF and the exact-far control;
- authoritative packed P99 was 765 bytes for both radii, below the frozen
  864-byte ceiling; and
- forward and reverse aggregation were byte-identical.

The compactness gates failed:

| Radius | HWF P99 | Gate | Delta | Result |
|---|---:|---:|---:|---|
| `radius4-61` | 271 tokens | at most 256 | +15 / +5.86% | fail |
| `radius5-91` | 298 tokens | at most 288 | +10 / +3.47% | fail |

The deterministic classifier therefore returned `compactness_failed`.
Per ADR 0154, no R4 MLX training is authorized.

This is a narrow negative result. The exact codec, framing, overflow contract,
and topology extractors passed. The failed object is the variable-cardinality
H+W+F model view under the preregistered token budgets.

## The 441 And 121 Claims

The result confirms the user's direction while correcting the geometry:

- 441 dense cells are unnecessary for the observed focal boards;
- a complete centered radius-four disk has 61 cells;
- a complete centered radius-five disk has 91 cells;
- a complete centered radius-six disk has 127 cells; and
- 121 is not the cardinality of a complete centered hex disk.

The 91-cell radius-five representation contained every focal occupied tile in
the 60,000-position corpus. The 61-cell radius-four representation required
only 59 overflow entities in total, with P99 zero and maximum five. Legal
adversarial overflow still exists, so neither empirical result permits
deleting the exact sidecar.

## Immutable Evidence

| Identity | Value |
|---|---|
| Bundle | `35c50bf9122a0a0ea3ab31a95f789a67a76c47471d0936bca166fa135391a6eb` |
| Release binary SHA256 | `a5d7cbbcddef9ed9e4f73d67ce86143d2329a4c08346cd9f144031512ba627b3` |
| Source tree SHA256 | `963df63b96db31df40d52ad7b74da61961e4024de49fd41bf5659469a02972ac` |
| Aggregate scientific BLAKE3 | `c2e0578d992c19df3d7254b16a86982d5b2deb31c485fbcbf94a9ca6b6d88a49` |
| Adversarial suite BLAKE3 | `65f3eba56662b71d5c8b5cdd0dd8029871d7d68c977eed0048f3b61779aaa0a2` |
| Cross-host parity BLAKE3 | `83173588b984ac7a4eb29518f676f43bd75d606ff80367824c95d699d36b974d` |
| Order-proof BLAKE3 | `40357c252416885b14d49e0048fce28c8427c5c41c6f489dfaad673c14cb27718` |

The final bundle contained 61 files and matched byte-for-byte on john1,
john2, john3, and john4 before production.

## Cluster Execution

Each host processed a unique train/validation pair. No scientific row was
duplicated:

| Host | Part | Rows | Extractor seconds | Rows/s | Scientific BLAKE3 |
|---|---:|---:|---:|---:|---|
| john1 | 0 | 15,120 | 84.979 | 177.93 | `90151d181edcc9e339f4a0ad395a664f27c596c7e22c4485f57783d10487df74` |
| john2 | 1 | 14,960 | 51.512 | 290.42 | `763b6dd6327af27ebb2b83e012507474b776e4fa151a1bd85206e0952da79916` |
| john3 | 2 | 14,960 | 52.205 | 286.57 | `3d43783a79fb75bdb5d5b8325bea9699723183ff76c2223682fee5d9de9313b8` |
| john4 | 3 | 14,960 | 51.861 | 288.46 | `6b267212c64a3a67df1144e4f1bc36110971f9a860e0d4b3f165db177a8fa9de` |

The four census tasks began within 67 milliseconds of one another. Cluster
wall time for the unique-row census was about 85 seconds, versus about 241
seconds of summed extractor time.

The first coordinator-only collection attempt used repository-relative
destination paths while its working directory was the immutable bundle
source. It failed before collecting or changing scientific evidence. The
attempt was preserved and cancelled, the campaign generator was repaired to
emit absolute coordinator paths, and only collection, parity, and aggregation
were rerun as `*-pathfix1` tasks. No preflight or corpus shard was rerun.

## Exact State Measurements

| Measurement | `radius4-61` | `radius5-91` |
|---|---:|---:|
| Capacity | 61 | 91 |
| Packed bytes mean | 488.506 | 488.500 |
| Packed bytes median | 485 | 485 |
| Packed bytes P99 | 765 | 765 |
| Packed bytes max | 772 | 765 |
| Focal occupied mean | 12.499 | 12.500 |
| Focal occupied P99 | 22 | 22 |
| Overflow entities total | 59 | 0 |
| Overflow entities P99 | 0 | 0 |
| Overflow entities max | 5 | 0 |

Packed P99 is 99 bytes below the 864-byte ceiling. Radius four and radius five
have essentially identical packed-state cost because the canonical codec
stores occupied local indices rather than materializing every empty disk cell.
This validates the sparse exact envelope and explains why a larger fixed
near-field capacity is not itself a packed-state penalty.

## Model-Visible Token Census

### Radius Four

| Arm | Mean | Median | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| N0 near only | 61.00 | 61 | 61 | 61 | 61 |
| H habitat | 89.40 | 89 | 102 | 108 | 116 |
| W wildlife | 109.53 | 110 | 146 | 156 | 166 |
| F frontier | 111.26 | 112 | 127 | 136 | 150 |
| HW | 137.94 | 138 | 187 | 200 | 215 |
| HF | 139.67 | 140 | 167 | 179 | 197 |
| WF | 159.80 | 161 | 212 | 227 | 248 |
| HWF | 188.20 | 188 | 252 | 271 | 297 |
| Exact far control | 210.89 | 211 | 287 | 309 | 340 |

### Radius Five

| Arm | Mean | Median | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| N0 near only | 91.00 | 91 | 91 | 91 | 91 |
| H habitat | 119.40 | 119 | 132 | 138 | 146 |
| W wildlife | 139.53 | 140 | 176 | 186 | 194 |
| F frontier | 140.95 | 142 | 156 | 161 | 171 |
| HW | 167.94 | 168 | 217 | 230 | 243 |
| HF | 169.36 | 170 | 196 | 206 | 219 |
| WF | 189.49 | 191 | 241 | 254 | 267 |
| HWF | 217.89 | 218 | 281 | 298 | 314 |
| Exact far control | 240.58 | 241 | 316 | 336 | 350 |

Every single-factor and two-factor treatment fits below its corresponding HWF
budget. Only the complete HWF combination fails. The all-topology summary
still removes 38 P99 tokens from the exact-far control at both radii, but not
enough to cross the frozen threshold.

## Block Anatomy

| Variable-cardinality family | Radius-four mean/P99/max | Radius-five mean/P99/max |
|---|---:|---:|
| Habitat component tokens | 28.40 / 47 / 55 | 28.40 / 47 / 55 |
| Wildlife component tokens | 18.63 / 36 / 44 | 18.63 / 36 / 44 |
| Wildlife signature buckets | 29.91 / 59 / 65 | 29.91 / 59 / 60 |
| Frontier signature buckets | 50.26 / 75 / 89 | 49.95 / 70 / 80 |

The near field is fixed and the exact sidecar is compact. The tail comes from
combining several variable-cardinality far summaries. Frontier buckets are
the largest individual family by mean and tail, while wildlife components and
signature buckets together are similarly substantial. The evidence does not
support deleting any one family: each passed its registered adversarial role.
It supports replacing per-signature token emission with a bounded,
field-aware quotient or residual summary.

## Adversarial And Determinism Results

The production extractor passed all 14 pair/radius cases:

- far habitat component;
- long Salmon topology;
- far Hawk conflict;
- far Fox neighborhood diversity;
- far legal frontier;
- overflow consequence; and
- relative opponent state.

The N0 controls collided where registered. H, W, and F each retained their
registered single-factor distinction. HWF and E retained every required
long-range distinction. Four independent host reports had identical
scientific bytes.

Forward shard order `[0, 1, 2, 3]` and reverse order `[3, 2, 1, 0]` produced
the same aggregate scientific BLAKE3 and byte-identical documents.

## Gate Resolution

| Gate | Observation | Result |
|---|---|---|
| Exact mechanics | 120,000 codec and R2 checks passed | pass |
| D6 inverse | 1,440,000 transform/inverse checks passed | pass |
| Target independence | 120,000 checks passed | pass |
| Adversarial suite | all 14 pair/radius cases passed on all hosts | pass |
| Corpus completeness | 60,000 rows and eight datasets exactly once | pass |
| Packed P99 | 765 bytes against 864 | pass |
| Radius-four HWF P99 | 271 against 256 | fail |
| Radius-five HWF P99 | 298 against 288 | fail |
| Aggregate order | byte-identical forward/reverse output | pass |

The deterministic outcome is:

```text
classification = compactness_failed
authorize_mlx = false
```

## Consequences

1. Do not train the ADR 0154 matched MLX comparison.
2. Keep `CSR4AM1`, carried F2 centers, exact local indexing, exact-coordinate
   overflow, strict decode/re-encode, and target-independent hashing as
   accepted infrastructure.
3. Keep 441 dense cells closed. Do not describe 121 cells as a regular
   centered hex disk.
4. Prefer radius four as the compact near-field basis for the next bounded
   view. It has the lower HWF distribution and required only 59 exact overflow
   entities across the full corpus.
5. Do not raise the 256/288 gates after seeing the result.
6. Do not drop habitat, wildlife, frontier, overflow, or opponent information
   merely to meet a token count. Every family owns a demonstrated distinction.
7. The immediate successor must bound the combined far summaries themselves:
   retain exact component objects where cheap, replace per-signature wildlife
   and frontier bucket tokens with deterministic field-aware quotient
   summaries, and retain the exact sidecar for mechanics and audit.
8. That successor must gate both token count and numeric feature width so a
   nominally small token set cannot hide an oversized dense vector.
9. R5 quotient-state and R6 hybrid sparse work may proceed as independent
   lanes. Any learned comparison remains blocked until a new exact foundation
   passes its own preregistered information and compactness gates.

## Claim Boundary

This result establishes exact mechanics, exact declared topology,
cross-host determinism, corpus distributions, and failure of the frozen HWF
compactness gates.

It does not establish learned quality, MLX latency, search strength, gameplay
strength, or progress toward the 100-point mean target.
