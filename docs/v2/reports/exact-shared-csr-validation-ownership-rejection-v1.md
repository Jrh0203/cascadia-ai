# Exact Shared CSR Validation Ownership Rejection

Status: **rejected**

Date: 2026-06-15

## Treatment

The experiment assigned canonical shared-memory CSR validation to the Rust
producer. Rust retained batch-size, row-width, and feature-range validation
and continued to encode canonical zero-based monotonic offsets. The private
Python child retained protocol, request-id, total-feature, mapped-capacity,
response-offset, output, and shutdown checks, but skipped duplicate offset,
width, and feature-range scans for shared requests only.

Ordinary sparse and pipe-carried exact CSR requests were unchanged. The same
accepted bounded-slice PGO Rust binary was used in both modes.

## Correctness

The treatment preserved every frozen exact diagnostic in all twelve
diagnostic and formal runs:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

The treatment-capable service passed 18 focused Python tests, including new
coverage for the retained shared-memory header, total-feature, and mapping
capacity checks. After rejection, all treatment code and temporary tests were
removed and the original 15 focused tests passed.

## Mechanism

The intended duplicate work was removed consistently:

| Host | Control decode/validate | Treatment decode/validate | Reduction |
|---|---:|---:|---:|
| john2 | 313.867380 ms | 139.295181 ms | 55.620% |
| john3 | 322.401333 ms | 138.482135 ms | 57.047% |

Total service request time fell 79.614 ms on john2 and 44.781 ms on john3.
However, measured MLX evaluation time rose by 83.294 ms and 111.969 ms in the
same diagnostic runs. The diagnostic wall result was mixed by host and only
0.116% faster combined, below the preregistered 0.30% gate.

## Decisive Screen

The accepted Rust binary and treatment-capable Python service were crossed
with two timer-off measurements per mode per host in opposite balanced
orders:

| Host | Control | Treatment | Treatment result |
|---|---:|---:|---:|
| john2 | 14.384470 s | 14.324836 s | 0.415% faster |
| john3 | 13.977943 s | 14.202196 s | 1.604% slower |
| Combined | **14.181207 s** | **14.263516 s** | **0.580% slower** |

Mean maximum RSS fell 0.799% and allocator peak footprint fell 5.984%, so
memory was not the failure. The treatment failed because the isolated
validation saving did not survive end-to-end CPU/GPU scheduling on john3 or
in the combined result.

## Verdict

Reject. The duplicated validation is measurable, but removing it does not
produce a stable fleet wall-time gain and regresses the decisive combined
metric. No production validation contract changed, the environment switch
and treatment branch were removed, and the accepted Phase 0 baseline remains
14.163055 seconds or 9.957x.

Machine-readable evidence:
`docs/v2/reports/exact-shared-csr-validation-ownership-rejection-v1.json`.

The complete local evidence archive is preserved under
`artifacts/performance/exact-shared-csr-validation-ownership-v1/`.
