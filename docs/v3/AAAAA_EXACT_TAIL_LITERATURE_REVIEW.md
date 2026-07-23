# AAAAA exact-tail literature review and implementation verdict

Date: 2026-07-23

## Outcome

The online review supports the architecture already responsible for all seven
specialized AAAAA certificates: layered arithmetic filtering, canonical
finite shape enumeration, local set packing, and independent certificate
decoding. It does **not** support spending more time on the monolithic
20-coordinate CP-SAT model.

Three literature-derived implementations were built and measured:

1. a layered single-anchor/local-packing relaxation;
2. exact radius-two fox-neighborhood relation tables; and
3. canonical per-species Fox-A witnesses with common-witness ring tables.

All preserve legal boards, have focused containment/structure tests, and fail
closed. None passed its preregistered strength gate, so none was sent over the
98 unresolved vectors and none changed an optimum claim. The measurements
identify the next representation precisely: enumerate complete interacting
fox/motif components outside the coordinate solver and factorize remote
components, rather than award them abstract coverage or express local
canonicality as redundant tables inside the full model.

## Why this tail is the relevant problem

The certified union covers 728 of 826 cap-six count vectors. All 98 unresolved
vectors have five or six foxes; 81 have six, and 96 contain all five species.
The count-only upper independently awards every fox every present species.
The proof must instead show that those observations cannot coexist with
isolated Bear-A pairs, straight Elk-A groups, unbranched Salmon-A components,
and isolated Hawk-A tokens on the same hex cells.

This is an exclusion problem. Strong boards are already available. The hard
question is whether any board can score one point more, which leaves many
symmetric near-solutions and makes an `UNKNOWN` timeout non-evidence.

## Primary sources reviewed

### Layered filtering rather than a monolithic solve

Kovař and Zhang's March 2026 preprint,
[A SAT-based Filtering Framework for Exact Coverings of K33 by Cliques of Order 3, 4 or 5](https://arxiv.org/abs/2603.29548),
is the closest methodological match. Their successful exact pipeline combines
arithmetic profiles, local signatures, symmetry reduction, geometric filters,
SAT realization only on reduced instances, and final decoding. They report it
as substantially more effective than direct full-instance ILP, DLX, or SAT.

That ordering maps directly to AAAAA:

```text
score deficit/profile
    -> canonical fox/motif signature
    -> local geometric set packing
    -> reduced exact realization
    -> independent board/certificate decoder
```

The accepted AAAAA motif, zero-hawk, Hawk, and joint-salmon certificates
already follow this pattern. The review increases confidence that the pattern
should be generalized, not replaced.

### Canonical component codes

Peng and Solnon,
[BFS-Based Canonical Codes for Generating Graphs with Constraint Programming](https://doi.org/10.4230/LIPIcs.CP.2025.32),
generate connected graphs through BFS-based canonical codes and a dedicated
global constraint, outperforming prior CP and SMT approaches on their graph
family. The applicable lesson is to enumerate each interacting fox component
once under a canonical traversal/code before it reaches the realization
solver.

Anders, Brenner, and Rattan's
[satsuma: Structure-based Symmetry Breaking in SAT](https://arxiv.org/abs/2406.13557)
similarly finds structured row, row-column, and Johnson symmetries rather than
treating symmetry as an undifferentiated permutation group. For AAAAA, the
corresponding structures are equal-size scoring groups, interchangeable foxes,
and repeated local component signatures.

The caution is equally relevant. Anders et al.,
[The Complexity of Symmetry Breaking Beyond Lex-Leader](https://doi.org/10.4230/LIPIcs.CP.2024.3),
show a natural hardness barrier for complete symmetry-breaking predicates in
graph and matrix-style models. The project should use cheap, provably sound
canonical component generation, not wait for a universal complete static
symmetry break.

### Specialized propagators and forbidden local patterns

Zhang and Szeider,
[The 3-Decomposition Conjecture: A SAT-Based Approach with Specialized Propagators](https://doi.org/10.4230/LIPIcs.CP.2025.39),
combine SAT modulo symmetries with domain-specific propagators and dynamically
forbidden substructures. Their experiments show that theoretical properties
of a minimal counterexample can prune much more effectively than generic SAT
alone.

Szeider's 2025
[SAT Modulo Symmetries survey](https://ceur-ws.org/Vol-4116/invited1.pdf)
describes the broader pattern: test canonicity during search and learn a clause
when a partial object cannot extend to a canonical one.

For AAAAA, target-score arithmetic supplies the "minimal counterexample"
properties: only a small score-loss budget is available, which forces a short
list of card-score profiles and Fox-A observation deficits. The useful
propagators should operate on those explicit component/profile branches.

### Counterexample-guided refinement

Ohashi et al.,
[SAT-Based CEGAR Method for the Hamiltonian Cycle Problem Enhanced by Cut-Set Constraints](https://doi.org/10.4230/LIPIcs.SAT.2025.24),
replace weak one-counterexample blocking with cut-set refinements that exclude
larger counterexample families. Their best variant solves 937 of 1,001
instances within 1,800 seconds, versus at most 666 for the compared eager
encodings.

The AAAAA analogue is:

1. let remote fox/motif components be abstract;
2. decode a relaxed witness;
3. identify the abstract component or cross-component motif that makes the
   witness unrealizable; and
4. refine the whole signature family, not just that coordinate assignment.

This is useful only after a reduced relaxation produces a witness. It is not a
reason to wrap the current full coordinate model in lazy connectivity cuts:
the difficult disconnected models usually time out without finding any
counterexample to refine.

Surynek et al.'s object-packing work,
[Object Packing and Scheduling for Sequential 3D Printing](https://arxiv.org/abs/2503.05071),
also reports CEGAR-inspired refinement as the key efficiency change. A March
2026 follow-up,
[Portfolio of Solving Strategies in CEGAR-based Object Packing and Scheduling](https://arxiv.org/abs/2603.12224),
parallelizes independent high-level refinement strategies. The latter is a
preprint, but its portfolio structure fits the Mac mini fleet: shard canonical
profile/component branches, not identical long monolithic timeouts.

### Current OR-Tools capabilities

The project already uses the current OR-Tools 9.15.6755 release. The official
[9.15 release notes](https://github.com/google/or-tools/releases/tag/v9.15)
report improved clause sharing, linear encodings, Python model construction,
and extensive LRAT proof work. The official
[`SatParameters` contract](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto)
also documents the important limit: LRAT/DRAT output currently applies only to
pure SAT under restrictive presolve, linearization, symmetry, and worker
settings. The present integer/table CP-SAT model cannot simply switch on LRAT
and obtain independently checkable proofs.

## Implementations and measured verdicts

### 1. Layered single-anchor bound

Source: `tools/aaaaa_wildlife_single_anchor_bound.py`

For a species with exactly one token, every observing fox lies in its six-cell
ring. The engine:

- exhausts near-target Bear/Elk/Salmon/Hawk score structures;
- enumerates ring fox subsets modulo all twelve dihedral transforms;
- locally packs every scoring group or leftover token covering an explicit
  fox; and
- places non-covering groups and non-anchor foxes optimistically.

The positive-containment calibration passed on four already-proven hard
vectors. The plus-one strength calibration failed: all four relaxations were
feasible after one local solve. Three witnesses used one explicit fox and one
used two; optimistic remote fox coverage was the exact looseness.

Verdict: retain as a fast first-stage filter, but it cannot certify the tail
without explicit remote fox components.

### 2. Radius-two fox-neighborhood tables

Source: `tools/aaaaa_wildlife_fox_neighborhood_exact.py`

The engine classifies each fox pair as distance one, distance two, or farther.
It enumerates every realizable connected relation graph through five foxes and
adds exact five-fox tables (six overlapping tables when there are six foxes).
This captures whether foxes can share adjacent animal witnesses, not merely
whether foxes touch.

On the fixed already-certified `(3,6,6,0,5) >= 62` case it remained
`UNKNOWN` after 60.011 seconds: 287,656 branches and 60,345 conflicts. That is
some pruning versus the earlier direct fox-graph table, but no exact result.

Verdict: correct but not selected.

### 3. Canonical Fox-A witnesses and ring triples

Source: `tools/aaaaa_wildlife_fox_witness_exact.py`

Every positive Fox-A observation is deterministically assigned to its
lowest-index adjacent target. Foxes sharing a target must be within distance
two, and every shared-target triple must match three distinct positions in
the target's six-cell ring.

The same calibration remained `UNKNOWN` after 60.015 seconds: 312,024
branches and 65,484 conflicts. The extra Boolean/witness structure cost more
than it pruned.

Verdict: correct but rejected. Static local tables inside the full coordinate
model are not the right abstraction boundary.

## Next exact architecture

The next solver should be a generalized external branch generator, not
another coordinate wrapper:

1. Enumerate every arithmetic score profile capable of beating the stored
   incumbent, including the allocation of Fox-A observation losses.
2. Choose the forced high-coverage species/motif for that profile.
3. Generate each interacting fox/motif component once using a canonical
   BFS/dihedral code.
4. Enumerate every relative separation at which a Bear pair, length-four Elk
   line, Salmon component, Hawk token, or Fox observation can cross component
   boundaries.
5. Replace all farther separations with a proved factorized case.
6. Solve local cell-set packing for each canonical signature.
7. If a relaxed signature is feasible, decode it and add a family-level
   refinement for the missing component interaction.
8. Shard independent profile/signature branches over john1–john4.
9. Accept a vector only when every registered branch is exact and the retained
   incumbent independently rescored.

This generalizes the successful seven-certificate machinery. It also gives
the fleet fine-grained work units, durable branch ledgers, deterministic
resume points, and exact per-branch completion rather than another collection
of opaque whole-vector timeouts.

## Evidence

- `docs/v3/evidence/aaaaa_single_anchor_containment_calibration_2026-07-23.json`
- `docs/v3/evidence/aaaaa_single_anchor_strength_calibration_2026-07-23.json`
- `docs/v3/evidence/aaaaa_fox_neighborhood_calibration_2026-07-23.json`
- `docs/v3/evidence/aaaaa_fox_witness_calibration_2026-07-23.json`
- chronological preregistrations, hashes, and decisions:
  `cascadiav3/EXPERIMENT_LOG.md`
