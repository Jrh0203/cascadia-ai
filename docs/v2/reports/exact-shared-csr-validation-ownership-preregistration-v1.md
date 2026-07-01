# Exact Shared CSR Validation Ownership Preregistration

Status: **completed - rejected**

Date: 2026-06-15

## Evidence

The accepted bounded-slice PGO path measures 14.163055 seconds against the
14.102730-second Phase 0 threshold. Only 0.060325 seconds, or 0.426%, remains.

Every exact shared-memory NNUE request is validated twice:

1. the Rust `ModelProcess` validates non-empty batch size, maximum row width,
   and every feature index before encoding;
2. its private CSR encoder deterministically writes zero-based monotonic
   offsets and the validated feature payload;
3. the child Python service validates the shared header, request identity,
   mapped bounds, offset endpoints, offset monotonicity, row widths, and every
   feature index again before constructing MLX arrays.

The shared file and control pipe are private to the parent and child process.
The parent sends the request header only after validation and encoding
succeed. The child already retains the checks required to prevent an invalid
mapping access: protocol magic/version, request identity, batch size, total
feature bound, response offset, and mapped capacity.

Final accepted diagnostics report 7,709 requests and 1,415,654,768 encoded
features. Python attributes 319-326 ms to repeated decode, structural
validation, range validation, and MLX array construction. Assigning canonical
CSR validation to the sole producer may remove enough duplicate work to clear
the remaining Phase 0 gap without changing a feature or prediction.

## Treatment

For `MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED` only:

1. retain Rust validation unchanged;
2. retain canonical Rust CSR encoding unchanged;
3. retain Python protocol, request-id, batch-size, total-feature, response
   offset, and mapped-capacity checks;
4. construct NumPy and MLX views directly from the producer-owned canonical
   offsets and features;
5. omit only the duplicate Python endpoint, monotonicity, per-row-width, and
   feature-range scans.

Pipe-carried exact CSR and ordinary sparse requests retain all current Python
validation. The treatment must not alter numerical dtypes, MLX graph
construction, output validation, response framing, or shutdown behavior.

An environment switch may select control or treatment during the source
screen. Acceptance removes the switch and makes single-owner validation the
documented shared-transport contract. Rejection removes all treatment code.

## Frozen Contract

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Seed: `34400`
- Four treatment seats
- Candidate budget: K32
- Rollouts: R600 sequential halving
- `MCE_LMR=1`
- `MCE_DIVERSE_PREFILTER=1`
- Full terminal rollouts
- Pipeline chunk states: 96
- Rust binary SHA-256:
  `0023c3d8bea978082b835fd933281b97d6800047afc32fcb3a33112fc1586cdd`
- Weights: `nnue_weights_v4opp_modal_iter3.bin`
- Model: `legacy-nnue-v4opp-mlx-v1`

Every diagnostic and timed run must reproduce:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Correctness Gates

Before timing, the treatment must:

1. preserve all existing service tests for ordinary and pipe-carried exact
   requests, including malformed offset and out-of-range rejection;
2. add explicit tests documenting that shared CSR is producer-validated and
   still rejects bad shared magic, version, request identity, total-feature
   bounds, and mapped-capacity violations;
3. pass the complete focused legacy NNUE Python suites;
4. pass the complete `cascadia-model` Rust suite;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any feature, prediction, selected action, score, sample, fallback,
random-stream, protocol, or shutdown mismatch rejects the treatment.

## Performance Gate

Use the accepted bounded-slice PGO Rust binary for both modes. Cross the
Python service switch in opposite balanced orders with two measurements per
mode per host:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Accept only if:

- both hosts improve;
- combined treatment time improves by more than 0.30%;
- repeated Python decode/validation time falls materially on both hosts;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

Because the treatment changes only Python service validation ownership, no
Rust profile regeneration is required. On acceptance, rerun the crossed
production service without the switch and clear Phase 0 only if the combined
accepted time is at or below 14.1027296 seconds.

## Result

Rejected and removed. Duplicate Python decode/validation time fell from
313.867 to 139.295 ms on john2 and from 322.401 to 138.482 ms on john3, but
the timer-off crossed result was not stable across hosts. John2 improved
0.415%, john3 regressed 1.604%, and combined wall time regressed 0.580%.

Full evidence:
[`exact-shared-csr-validation-ownership-rejection-v1.md`](exact-shared-csr-validation-ownership-rejection-v1.md).
