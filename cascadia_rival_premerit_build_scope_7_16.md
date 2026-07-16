# Cascadia Rival: pre-merit build scope

**Date:** 2026-07-16 (v2 — corrected per the builder audit of the same date:
P2a decoupled to independent worlds, P2b coupling work priced as WI-4c,
T0/T1/T2 renumbered to the serialized schema, kill-bar witness/bound
asymmetry fixed, MEASURED-0 dependencies completed, CPU-1 arithmetic fixed)
**Purpose:** exact scope of the engineering required to reach the MEASURED-0
merit verdict for Cascadia Rival. Everything in this document is buildable
now or at the D1 boundary; everything the merit verdict does NOT need is
explicitly out of scope. Parent documents:
[implementation plan](cascadia_rival_implementation_plan_7_16.md) (authoritative
for walls/permits), [architecture](cascadia_rival_final_architecture_proposal_7_16.md),
[critique](architecture_proposal_critiques_7_16.md), and the delivered CPU
machinery on branch `feat/rival-cpu-machinery`
(status ledger: `docs/v3/RIVAL_CPU_MACHINERY.md`).

**Merit question this scope serves.** Four measurements decide whether Rival
deserves its 100–200 day build:
M1 selfish headroom (tomography) · M2 full-incumbent terminal-pair cost (P2a)
· M3 low/high correlation ρ (P2b) · M4 the power-envelope verdict.
MEASURED-0 consumes **Gate 0 (delta_b) + M1 + M2 + M3** — all four, not just
cost and correlation. This document scopes exactly the code those
measurements need and nothing else.

---

## Standing constraints (apply to every work item)

1. **Branch:** all implementation lands on `feat/rival-cpu-machinery` (or a
   child branch merged into it). Nothing merges to `main` before the durable
   D1 boundary.
2. **CPU-only:** no CUDA/MPS import, no device query, no GPU code path. The
   existing `cpu_test_guard` conventions apply; new entry points that could
   ever touch a device must call `require_cpu_test_device`.
3. **D1 source closure:** work items marked **[D1-WALL]** touch files the
   live D1 chain depends on (exporter, trainer, bridge, `cascadia-game`).
   They may be *prepared* (schemas, fixtures, worktree) now but their
   source-closure edits land only after the D1 boundary is durable.
4. **No science:** no scientific seeds, no experiment artifacts, no john0
   jobs, no partial-D1 reads. Deterministic local fixtures only.
5. **Fail closed:** follow the branch's established conventions — schema
   validation, content hashes, exact-key checks, error-on-unknown, and
   hermetic test fixtures.
6. **Selfish-only:** every estimand is own-seat score. No table utility,
   donation, cross-seat plans, or shared memory anywhere, including in
   diagnostics (John's objective ruling, 07-16).

---

## Work item 1 — CPU-1 release-scale battery

**Goal:** claim the plan's CPU-1 gate: the canonical CPU reference harness is
trusted at release scale.

**Current state:** `crates/cascadia-rival/tests/cpu_reference_battery.rs`
runs 125 games / 10,000 transition checks with hardcoded constants. The
branch ledger says: "Not run; CPU-1 remains unclaimed."

**Build:**
- Parameterize the battery scale via environment (e.g.
  `RIVAL_BATTERY_GAMES`, default the current PR scale; release scale
  **12,500 games / 1,000,000 transition checks** — four-player games yield 80
  transitions each, so the plan's original "10,000 games AND 1,000,000
  transitions" was internally inconsistent; honor the stronger number). Keep
  the PR-scale run in the
  ordinary test suite; the release run is invoked explicitly (ignored test +
  `--ignored`, or a dedicated bin) so CI time is unaffected.
- Extend coverage per the plan's CPU-1 row: random legal play across all
  four seats, overflow-prone states (long Salmon runs, component merges,
  near-empty bag), score-decomposition identity at every terminal, ledger
  replay + hash verification, zero leakage/replay mismatches.
- Emit a small JSON report (games, transitions, wall seconds, zero-mismatch
  attestation, source digest) to `cascadiav3/reports/` so CPU-1 is a durable
  artifact, not a console line.

**Done when:** release run completes clean on local hardware; report artifact
committed; ledger in `RIVAL_CPU_MACHINERY.md` flips CPU-1 to claimed with the
report path.

**Estimated effort:** ~0.5 day engineering + hours of unattended CPU.

---

## Work item 2 — T1/T2 selfish tomography optimizers (measurement M1)

**Goal:** turn the existing tomography *result types*
(`crates/cascadia-rival/src/tomography.rs`: `TomographyKind`,
`TomographyEvidence`, `TomographyResult` with witness/bound discipline) into
actual measurements of unilateral recoverable headroom.

**Current state:** types and validation only. No repacking search, no
hindsight replay optimizer exists.

**Build two optimizers, both CPU-only, both in `cascadia-rival`:**

### 2a. T0 — static own-seat repacking

(Numbering follows the serialized tomography schema: static repacking is
**T0**; chronology-preserving replay is T1/T2's information-boundary family.
The scope previously mislabeled these.)

- Input: one terminal board from a completed game (via `TrajectoryLedger`).
- Freeze that seat's realized multiset: tile identities, wildlife multiset,
  starter tile, terminal Nature-token count.
- Search for a better *legal terminal arrangement* of the same multiset
  (exact scoring via the canonical engine / dense compiler).
- Output: a `TomographyResult` whose evidence is a **feasible witness lower
  bound** on the repacked optimum (a materialized board), explicitly labeled
  optimistic/non-policy per the parent §12.1. Optionally a certified upper
  bound where a relaxation is provable; a heuristic best is NEVER an upper
  bound.
- Search implementation is the builder's choice (beam / simulated annealing /
  exact for small boards), but determinism is required: fixed seed → fixed
  witness. Every emitted witness board must round-trip legality and exact
  score through the canonical engine.

### 2b. T1 — chronology-preserving replay

- Input: one seat's complete realized chronology from a `TrajectoryLedger`:
  the exact 20 draft pairs in order, free-refresh decisions/reveals, paid
  wipes, Nature transactions.
- Hold the chronology fixed; re-optimize only the legal *placement* decisions
  (tile coordinate/rotation, wildlife placement/decline) through real
  `GameState` transitions in original order.
- Output: certified hindsight-placement witness (still future-knowledge, still
  not a policy — label accordingly via `InformationBoundary`).

### Harness

- A CPU driver that consumes a directory of completed-game ledgers, runs
  T1/T2 per seat, and emits one JSON summary: per-seat realized score,
  witness score, delta; distribution stats; source digests. Deterministic
  given the input set and seed.
- Input ledgers come from the battery's random-play games initially
  (mechanism validation); the *scientific* run after the D1 boundary feeds it
  incumbent games from the fresh baseline battery. The harness must record
  which policy produced its inputs and refuse mixed populations.

Additionally: the current `TomographyEvidenceDomain` is permanently
`CpuProxy`. Scientific M1 on incumbent games requires a distinct, validated
evidence-domain variant (schema + validation + tests) so proxy-population
diagnostics can never be presented as incumbent measurements.

**Done when:** on random-play fixture games, the optimizers find (obviously large)
headroom with valid witnesses, all witnesses re-verify through the canonical
engine, results are deterministic, and the JSON summary validates against a
new schema. Unit tests: witness legality, score identity, chronology
violation rejection, determinism, information-boundary labeling.

**Kill bar it enables (preregistered separately, not in code):** a witness
is a LOWER bound — it can fund, never kill. Closing Rival on "insufficient
headroom" additionally requires either (a) a certified upper bound from an
admissible relaxation whose feasible set provably contains the exact
problem, or (b) a preregistered search-sufficiency bar that the optimizer
demonstrably cleared. Absent both, a small witness is inconclusive, not a
kill.

**Estimated effort:** 3–5 days. This is the largest new-code item.

---

## Work item 3 — exporter extraction under golden traces **[D1-WALL]**

**Goal:** make the *real serving incumbent* (transformer + Gumbel + exact-K1
+ refresh policy, byte-exact) callable as a library policy for terminal
continuations. This is the plan's Slice 11 and the single prerequisite for
admissible P2a/P2b numbers.

**Prepare now (allowed):**
- Golden-trace schema: a serialized record of one serving decision — public
  state hash, prelude, menu, search config, chosen action id, and every
  bridge request/response digest — sufficient to prove behavioral identity.
- Synthetic CPU fixtures exercising the trace comparator.
- An isolated worktree with the planned crate boundary
  (`cascadia-v3-policy` per the plan) stubbed.

**After the D1 boundary (source-closure edits):**
- Capture production golden traces from the resolved incumbent on a handful
  of seeds (CPU bridge acceptable for traces; config pinned).
- Extract the policy-critical paths from
  `cascadiav3/real-root-exporter/src/main.rs` (+`gumbel.rs` callers) into the
  `cascadia-v3-policy` library crate, behavior-identically. The exporter then
  depends on the library. No parallel reimplementation — the same code moves.
- Re-run: captured traces byte-identical through the library path; the FULL
  existing exporter test suite (68) plus workspace suites green; one focused
  commit updating both `Cargo.toml`/`Cargo.lock` files.

**Done when:** the library exposes `decide(state, config) -> action` used by
both the exporter and the Rival terminal harness, golden traces prove
identity, and all pre-existing suites pass.

**Estimated effort:** 1–2 days prep now; 2–3 days at the boundary.

---

## Work item 4 — P2a/P2b probe harnesses (measurements M2, M3)

**Goal:** the two bounded probes whose outputs feed the power calculator.
Harness code is CPU-testable now; GPU execution happens only under Permit A
after the D1 boundary.

### 4a. P2a — full-incumbent terminal-pair cost probe

- Given a root cohort (S-cohort machinery already exists in
  `rival/cohorts.py`), one frozen challenger per root, and the extracted
  incumbent policy: complete terminal continuations for incumbent-action and
  challenger-action on **INDEPENDENT physical worlds with `beta_cv = 0`**
  (the `estimate_high_fidelity_only` path). Cross-action physical coupling
  is NOT claimed in P2a: valid coupling requires the WI-4c chance-oracle
  proofs, and the branch correctly denies production coupling until they
  exist. Independent worlds make P2a's cost and variance numbers
  conservative, which is what a cost probe should be.
- Measure and ledger: wall seconds per completed paired terminal (by phase
  stratum and remaining turns), completion/timeout rates, memory, and total
  incumbent decisions executed. Every row lands in the existing
  `appeal_journal` machinery.
- Output feeds `rival/power.py`'s currently-`UNRESOLVED` cost fields.

### 4b. P2b — coupled covariance probe (requires WI-4c first)

- Same roots; additionally run the cheap continuation
  (v1 low-fidelity policy = the existing typed proxy continuations; RivalNet
  explicitly does NOT exist yet and is not required for the merit verdict) to
  produce paired (D_H, D_L) rows.
- Compute stratified correlation/covariance via the existing
  `multifidelity.py` calibration machinery (disjoint calibration cohort
  discipline enforced by the existing `CoefficientBinding`).
- Output feeds the calculator's covariance fields; per-stratum ρ is the M3
  number.

### 4c. Chance-coupling validity machinery (new; prerequisite for P2b)

The plan's CPU-3b gate. Valid paired (D_H, D_L) rows require sharing physical
scenarios across fidelities, which today's sampler cannot legally do:

- a canonical **ChanceOracle**: semantic-event random domains for every
  chance event (bag draws, wipes, refresh reveals, dynamic urn returns);
- **stable physical token identities** so "the same world" is well-defined
  across diverging action prefixes; and
- **marginal-equality proofs**: exhaustive small-bag enumerations plus
  property tests demonstrating that coupling never changes any single-arm
  chance marginal (the branch's stated reason for denying production
  coupling). Coupling that fails any proof falls back to independent worlds.

Estimated effort: 2–4 days. Without WI-4c, P2b produces no admissible
covariance and Rival-MF's merit cannot be measured — only P2a's
independent-world cost bound would exist.

**Build now:** the runners, ledgers, stratification, fixture-driven CPU tests
(proxy policies both sides), and the two preregistration templates (sample
sizes, strata, error budgets, kill bars left as frozen-at-launch fields).
**Run later:** on john0 under an explicit Permit A instruction.

**Done when:** end-to-end CPU dry runs on synthetic fixtures produce validated
ledgers and a power-calculator input file; the GPU path is exactly the same
code with a different (permitted) policy/backend binding.

**Estimated effort:** 2–3 days.

---

## Work item 5 — Gate 0 baseline preregistration (campaign-level, tiny)

Not Rival code, but merit math needs it: the fresh canonical baseline battery
under `..._rules_2026_07_16` (existing benchmark machinery, champion tier,
~100 games) that defines `b` and `delta_b = 100.10 − b`. Deliverable here is
only the preregistration text + runner env (the battery scripts already
exist). Runs on john0 after the D1 boundary, before or alongside P2a.

**Estimated effort:** hours.

---

## Explicitly OUT of scope (do not build)

- **Slice 9:** incremental/component compilers, D6 reflection transforms in
  `cascadia-game` — held for P3, only funded if MEASURED-0 passes.
- **P4:** any accelerator-resident engine or backend.
- **P5:** RivalNet — any training, any model. The merit verdict deliberately
  uses proxy continuations for ρ; a low ρ with proxies is itself decision
  information.
- **P8:** trainer integration (`--rival-preference-training` stays
  fail-closed).
- Serving wrappers, overrides, or any change to live serving behavior.
- Any GPU execution, seed allocation, or scientific run (permits required).

---

## Sequence and dependency graph

```text
NOW (parallel):
  WI-1 CPU-1 battery          [no deps]
  WI-2 tomography optimizers  [no deps]
  WI-3 prep: trace schema     [no deps]
  WI-4 probe harnesses (CPU)  [uses existing cohorts/multifidelity]
  WI-5 Gate 0 prereg          [no deps]

AT D1 DURABLE BOUNDARY:
  resolve B_0 -> Gate 0 battery (b, delta_b)
  WI-3 extraction lands       [gates P2a/P2b admissibility]
  tomography SCIENTIFIC run on incumbent games (M1)

UNDER PERMIT A (explicit instruction):
  P2a probe (M2) -> P2b probe (M3)

FREE:
  MEASURED-0: power calculator consumes M2+M3 -> fund P3+ / narrow / close
```

Total new engineering to the merit verdict: **~10–16 focused days** (the
earlier 7–12 omitted WI-4c's coupling-validity machinery), most of it
parallelizable and startable immediately; only WI-3's landing and the probe
executions wait on the D1 boundary and Permit A. D1 is NOT yet a durable
boundary at this writing — generation is live; nothing D1-gated may start.

## Acceptance ledger (update as items complete)

| Item | State |
| --- | --- |
| WI-1 CPU-1 release battery | NOT STARTED |
| WI-2 T1/T2 tomography optimizers + harness | NOT STARTED |
| WI-3 exporter extraction (prep / landing) | NOT STARTED / HELD AT D1 WALL |
| WI-4a/b P2a/P2b harnesses + preregistration templates | NOT STARTED |
| WI-4c chance-coupling validity (ChanceOracle + proofs) | NOT STARTED |
| WI-5 Gate 0 preregistration | NOT STARTED |
| M1 headroom measured | BLOCKED (WI-2 + D1 boundary) |
| M2 cost measured | BLOCKED (WI-3/4 + Permit A) |
| M3 rho measured | BLOCKED (WI-3/4 + Permit A) |
| M4 MEASURED-0 verdict | BLOCKED (Gate 0 + M1 + M2 + M3) |
