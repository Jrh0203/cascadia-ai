# ADR 0097 Invalid Launch 1

Date: 2026-06-16

Experiment: `complete-action-frontier-factor-integration-v1`

Disposition: invalid operational launch; no scientific result

## What Happened

The first four-arm probe launch used MLX's default free-buffer cache limit.
On john1, the default was 16,320,875,724 bytes. Variable candidate widths
left freed Metal buffers in that cache across hundreds of batches. The
wide-concat process reached approximately 11.3 GiB physical footprint while
host swap rose to approximately 6.7 GiB. Its first epoch took 703.43 seconds,
far outside the cache-generation and remote-arm throughput envelope.

All four jobs were interrupted together after approximately 870-873 seconds.
No arm completed the frozen 20 epochs:

| Arm | Host | Completed epochs | Last partial train recall | Last partial validation recall |
|---|---|---:|---:|---:|
| wide-concat | john1 | 1 | 0.262529 | 0.234926 |
| screen-relative | john2 | 7 | 0.290283 | 0.250490 |
| factor-attention | john3 | 3 | 0.278547 | 0.256985 |
| pairwise-gated | john4 | 2 | 0.266917 | 0.225000 |

These partial metrics are archived for auditability but are prohibited from
selection, classification, architecture changes, or future priors.

## Root Cause

The preregistered maximum-width audit exercised inference only. It therefore
did not test the larger temporary allocation graph created by
backpropagation. The model architectures, factor cache, labels, objective,
seeds, optimizer, and scientific gates were not at fault.

## Permanent Correction

The unchanged experiment now:

- sets MLX's reusable free-buffer cache limit to 512 MiB;
- clears the cache at training and evaluation phase boundaries;
- records active, cached, and peak allocator memory before and after clearing;
- preserves allocator telemetry outside cross-host scientific hashes; and
- runs forward plus backward for every architecture on the 10,854-candidate
  maximum-width group before launch.

A deterministic unit test proves the bounded-cache path produces bit-exact
loss and gradient tensors relative to the default path.

The corrected maximum-width audits passed on john1 and john4. Both measured
the same per-architecture peak active memory:

| Arm | Peak active memory |
|---|---:|
| wide-concat | 0.94 GB |
| screen-relative | 1.42 GB |
| factor-attention | 4.33 GB |
| pairwise-gated | 1.78 GB |

Every arm stayed below the frozen 6 GiB gate, retained zero cached bytes after
the phase clear, used zero process swaps, and caused no positive system-swap
delta.

## Integrity

- Local complete suite: 260 passed.
- john2, john3, john4 focused suite: 12 passed on every host.
- Ruff: passed locally and on every remote host.
- Corrected source bundle: 97 files, SHA-256
  `a9b30f3f468ad41cee66aa997d4ae28000a2be19244601499d64379f14b7bb6e`
  on all four hosts.
- Existing factor-cache payloads remain valid because the correction changes
  runtime memory retention only; factor values, metadata, and portable payload
  identities are unchanged.
- Sealed test, gameplay, cloud, and external compute remained unopened.

The real launch must reuse the original four architectures, seeds, 20-epoch
stop rule, host assignments, factor caches, and classification gates.
