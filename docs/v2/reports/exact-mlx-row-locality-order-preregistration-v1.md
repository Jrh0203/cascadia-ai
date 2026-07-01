# Exact MLX Row Locality Order Preregistration

Status: **completed: rejected and removed**

Date: 2026-06-15

## Outcome

The treatment preserved every exact service, rollout-wave, score, and frozen
diagnostic result, and increased largest-request adjacent trie reduction from
4.93% to 35.61%. It nevertheless regressed the crossed source benchmark by
6.69% on john2 and 7.06% on john3, or 6.87% combined. Host sorting and
prediction scatter increased Rust-side neural evaluation by more than 10% on
both workers while MLX evaluation was flat on john2 and only 1.26% faster on
john3.

The source gate failed, so no PGO run was authorized. All treatment code and
the experimental switch were removed. The rebuilt release is byte-for-byte
identical to the retained control at SHA-256
`786351ea84e4b2674e81f2ade87d0596e47a8a3b21be2f336dc9e6ff62c4cd94`.

Final evidence:
[`exact-mlx-row-locality-order-rejection-v1.md`](exact-mlx-row-locality-order-rejection-v1.md).

## Evidence

The accepted parent-afterstate PGO path spends about 7.6-8.0 seconds inside
MLX evaluation across 7,709 service requests and 5,062,306 physical rows.
Those rows contain 1,415,654,768 ordered sparse feature indices.

The accepted service diagnostic retained the largest 1,298-row request. In
its canonical search order, adjacent rows represented 258,189 distinct trie
edges, only a 4.93% reduction from the request's 271,588 feature occurrences.
Lexicographically ordering the same rows reduced that count to 174,862 edges,
a 35.61% reduction. Exact pair and group-of-four prefix kernels proved that
this locality can accelerate isolated H1 work, but their explicit prefix
planning, metadata transport, and larger kernels erased the gain end to end.

The current exact H1 kernel processes adjacent rows in the same Metal
threadgroup and reads the same immutable first-layer weight matrix. Improving
adjacent row locality may therefore increase ordinary device-cache reuse
without changing the kernel, arithmetic, protocol, or search.

## Mechanism

For each already-deduplicated sparse MLX request:

1. build a permutation of physical row indices;
2. sort that permutation by the complete ordered sparse feature vector;
3. encode rows in the resulting locality order into the existing shared CSR
   mapping;
4. run the unchanged exact three-kernel MLX evaluator;
5. scatter returned values back to the request's original physical-row order.

The feature order within every row must remain byte-identical. The treatment
must not alter row deduplication, logical-to-physical mappings, predictions,
candidate ordering, selected actions, random streams, search allocation,
protocol framing, model parameters, or Metal arithmetic.

The implementation may reuse permutation and scatter buffers across requests.
It must add no new wire metadata and no Python-side planner. The experimental
path may use a gated switch while measured; acceptance removes the switch and
makes the proven ordering the single production path. Rejection removes all
treatment code.

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
- Weights: `nnue_weights_v4opp_modal_iter3.bin`
- Model: `legacy-nnue-v4opp-mlx-v1`

The exact diagnostic vector is:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Correctness Gates

Before timing, the treatment must:

1. preserve every row's feature bytes, order, and multiplicity;
2. return predictions in original request order for empty, duplicate, equal,
   prefix-related, and arbitrary rows;
3. match the control evaluator bit for bit across at least 200 varied service
   requests on john2 and john3;
4. preserve complete rollout-wave predictions and selected actions;
5. pass the complete default and `mid-features,v4-opp` library suites plus the
   focused Python service/client tests;
6. reproduce the frozen score and diagnostic vector on john2 and john3.

Any prediction, ordering, action, score, or diagnostic mismatch rejects the
treatment before performance measurement.

## Performance Gates

Matched non-PGO release binaries will be crossed on john2 and john3 with two
measurements per binary per host. The treatment advances to fresh race-free
PGO only if:

- combined end-to-end treatment time improves by more than `1.00%`;
- both hosts improve;
- MLX service evaluation time falls on both hosts;
- row-ordering and scatter CPU cost does not erase the service gain;
- maximum resident set size and allocator peak footprint do not materially
  regress;
- every timed run preserves the frozen exact diagnostic vector.

PGO profiles must be collected once per host with `RAYON_NUM_THREADS=1`, then
merged. The final candidate will be crossed against the accepted
parent-afterstate PGO binary.

## Acceptance

Accept only if the fresh PGO treatment is reproducibly faster on both workers,
remains bit-exact, and improves the accepted 15.018871-second result toward
the 14.102730-second Phase 0 threshold without an operational regression.
Otherwise remove it and retain a machine-readable rejection report.
