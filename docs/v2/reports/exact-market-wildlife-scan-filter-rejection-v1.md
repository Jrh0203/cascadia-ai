# Exact Market Wildlife Scan Filter Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The candidate generator computed the best existing placement for all five
wildlife categories even though its normal and independent draft combinations
can use only wildlife represented in the current public market. The treatment
retained the original category order and skipped only categories absent from
every combination.

Candidate construction, habitat scans, required wildlife scans, strict ties,
shared outcomes, potential arithmetic, candidate order, sparse rows, MLX
requests, random streams, search allocation, and terminal scoring were
unchanged.

## Exactness

A test-only all-five-category oracle matched the filtered implementation
across four complete seeded four-player AAAAA games. Each state was checked
both as played and with a Nature Token forced onto the acting board. The suite
covered markets with duplicate wildlife and states with and without Nature
Tokens.

The existing independent per-combination outcome-cache oracle also remained
green. The complete gates passed:

- default workspace libraries: `cascadia-ai` 86, `cascadia-core` 125,
  `cascadia-search` 61, and every other workspace library;
- `mid-features,v4-opp`: `cascadia-ai` 87;
- focused Python exact client/service tests: 15 passed;
- formatting and patch-integrity checks.

Every diagnostic and formal source run reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

The paired diagnostic runs showed the intended work reduction:

| Host | All-five scan | Market-only scan | Reduction |
|---|---:|---:|---:|
| john2 template preparation | 4,532.693 ms | 4,518.542 ms | 0.312% |
| john3 template preparation | 4,517.007 ms | 4,482.154 ms | 0.772% |

No candidate or downstream diagnostic changed.

## Source Screen

One treatment-capable non-PGO binary, SHA-256
`51fdaa480a5817815ce3234f7b7bf1098da6f05565cb0635d32111f8bea29438`,
was crossed in opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

| Host | Control mean | Treatment mean | Treatment result |
|---|---:|---:|---:|
| john2 | 14.763773 s | 14.560019 s | 1.380% faster |
| john3 | 14.391788 s | 14.406529 s | 0.102% slower |
| Combined | **14.577781 s** | **14.483274 s** | **0.648% faster** |

Mean maximum RSS fell 0.032%. Mean allocator peak footprint rose from
58,978,736 to 61,465,008 bytes, a 4.216% increase.

## Verdict

Reject before PGO. The combined signal and targeted stage reduction are real,
but the preregistration required a positive end-to-end result on both workers.
John3 regressed, and the allocator-footprint increase also violated the memory
gate. Advancing only the favorable aggregate would invalidate the experiment's
registered decision rule.

The environment switch, dual release monomorphizations, filter, and temporary
oracle were removed. The accepted bounded-slice PGO champion remains unchanged
at 14.16305453125 seconds, or 9.957x versus the 141.027296-second reference.
The Phase 0 gap remains 0.06032493125 seconds.

Machine-readable evidence:
`docs/v2/reports/exact-market-wildlife-scan-filter-rejection-v1.json`.

Raw evidence is archived under
`artifacts/performance/exact-market-wildlife-scan-filter-v1/`.
