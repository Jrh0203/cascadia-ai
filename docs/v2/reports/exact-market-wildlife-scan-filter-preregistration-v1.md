# Exact Market Wildlife Scan Filter Preregistration

Status: **completed - rejected after source screen**

Date: 2026-06-15

## Evidence

The accepted bounded-slice PGO path measures 14.16305453125 seconds against
the 14.1027296-second Phase 0 threshold. The remaining gap is 0.06032493125
seconds, or 0.426%.

Fresh native samples of that exact production binary identify
`candidate_move_set` as the largest native CPU stack on both workers:

- john2: 8,302 top-of-stack samples;
- john3: 8,441 top-of-stack samples.

`score_wildlife_after_placement` contributes another 1,151 and 1,191
top-of-stack samples respectively. Source inspection found one unnecessary
complete-board scoring loop in every candidate-template request:
`candidate_move_set_impl` computes the best existing placement for all five
wildlife categories even though every normal and independent draft
combination can use only wildlife currently present in the public market.

The rollout-opponent greedy path already applies the same logical filter. The
hotter candidate-template path does not.

## Hypothesis

Restricting the candidate generator's best-existing-wildlife scan to the
distinct wildlife categories represented by its already-built draft
combinations will remove at least one of five category scans for every
ordinary four-slot market, and more when the market contains duplicate
wildlife, without changing any reachable candidate or tie.

## Mechanism

Add an experimental exact path inside the original candidate-generation
boundary:

1. derive a five-entry required-wildlife mask from the ordered draft
   combinations already consumed by the candidate generator;
2. retain the original `Wildlife::ALL` iteration order;
3. skip only categories whose mask entry is false;
4. leave every scan for a required category unchanged, including tile order,
   scoring-card dispatch, strict tie comparison, coordinates, and mutation
   history;
5. preserve combination construction, habitat placements, shared outcomes,
   potential arithmetic, candidate ordering, sparse rows, MLX requests,
   random streams, and terminal scoring exactly.

A same-binary environment switch may disable the filter during the source
screen. Acceptance removes the switch and makes the exact filter
unconditional. Rejection removes all treatment code.

## Exactness Argument

Every `Combo.wildlife` value comes from one of the currently available market
pairs. Normal drafts use the wildlife paired with their tile. Independent
drafts combine a tile from one available pair with wildlife from another
available pair. Therefore a wildlife category absent from all combinations
cannot index `best_existing_wildlife` later in the function and its computed
value is dead.

The treatment does not move work across a tile place/undo boundary and does
not alter union-find path-compression history. It only avoids hypothetical
wildlife placements for unreachable categories before the first candidate
tile mutation.

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

Every diagnostic and timed run must reproduce:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Correctness Gates

Before timing, the treatment must:

1. compare the filtered candidate set with an all-five-category reference
   across complete seeded four-player AAAAA games, including markets with and
   without duplicate wildlife and boards with and without Nature Tokens;
2. preserve the existing per-combination outcome-cache reference parity;
3. pass the complete default and `mid-features,v4-opp` library suites;
4. pass the focused Python exact-service/client suites;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any candidate, base move, row, prediction, selected action, score, sample,
fallback, random-stream, or shutdown mismatch rejects the treatment before
performance measurement.

## Source Performance Gate

Use one matched non-PGO treatment-capable binary on both workers, with two
measurements per mode per host and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance to production and fresh PGO only if:

- both hosts improve;
- combined end-to-end time improves by more than 0.25%;
- candidate-template preparation falls on both hosts in diagnostic runs;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

## Production And PGO Gate

On source-screen success:

1. make the required-wildlife filter unconditional;
2. remove the experiment switch, all-five release branch, and dead code;
3. retain a test-only all-five reference implementation;
4. rerun complete default and `mid-features,v4-opp` library suites plus the
   focused Python exact suites;
5. reproduce source parity on both workers;
6. collect one complete R600 profile per host with
   `RAYON_NUM_THREADS=1`;
7. merge only those two profiles;
8. cross the fresh production PGO binary against the accepted bounded-slice
   PGO champion in the same opposite balanced order.

## Acceptance

Accept only if the fresh production PGO binary:

- is faster on both workers;
- preserves the complete frozen exact contract;
- has no material memory, reliability, shutdown, or latency regression; and
- has a crossed treatment mean at or below 14.1027296 seconds.

Only that final condition clears the mandatory 10.0x Phase 0 gate versus the
141.027296-second reference.

## Result

Rejected before PGO. The filter preserved the complete frozen diagnostic
vector and reduced the targeted template-preparation stage on both workers.
The formal source screen improved john2 by 1.380% and improved the combined
mean by 0.648%, but john3 regressed by 0.102%. Mean allocator peak footprint
also rose 4.216%.

The treatment therefore failed the preregistered requirements that both hosts
improve and memory not materially regress. The switch and release treatment
were removed. Full evidence:
[`exact-market-wildlife-scan-filter-rejection-v1.md`](exact-market-wildlife-scan-filter-rejection-v1.md).
