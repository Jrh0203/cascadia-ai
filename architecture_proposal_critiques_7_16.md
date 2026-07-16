# Critical evaluation of the July 16 architecture proposals

**Date:** 2026-07-16
**Author:** Claude (Fable 5), at John's request
**Scope:** critical evaluation and ranking of the three post-D1 architecture
proposals, judged against the campaign's measured constraints. This document
ranks research bets; it is not strength evidence and it does not reorder the
authorized D1 chain.

Proposals under review:

1. [Cascadia-NX](stochastic_board_game_ai_architecture_research_7_16.md) —
   structured incremental afterstate evaluator + GPU-resident exact-rules
   search.
2. [Cascadia-Anchor](incumbent_anchored_gpu_rollout_policy_improvement_7_16.md)
   — incumbent-anchored terminal-rollout override wrapper around the current
   serving policy.
3. [Cascadia Foundry](cascadia_foundry_original_architecture_proposal_7_16.md)
   — score-contract program synthesis; terminal witnesses, reactive controller
   programs, cooperative table objective.

## Ranking

**Anchor > NX > Foundry** — with one carve-out: Foundry's §12 ceiling
tomography is severable from its architecture and should be extracted and run
FIRST, ahead of any of the three builds.

All three documents share unusually good evidence discipline (labeled claims,
explicit kill tests, D1 primacy respected). The ranking is therefore about the
bets themselves: evidence behind the mechanism × ceiling × cost to first
falsifier × downside if wrong.

---

## 1. Cascadia-Anchor — most promising

### For

- **The only proposal with causal evidence in this repository.** The v2
  `LateConservativeBasePolicyImprovementStrategy` was the same mechanism —
  incumbent anchor, paired terminal continuations, one-sided confidence guard,
  literal fallback — and measured **+0.420 (95% CI [+0.179, +0.661])**,
  **+0.520 ([+0.260, +0.780])** on redetermination. It was demoted for a
  preregistered wildlife guardrail, not for lack of points. Old rules and old
  policy, so not current strength evidence — but neither competitor has ANY
  gameplay result behind its core mechanism.
- **Attacks the measured failure mode directly.** At median decision SNR
  ~1.06 with ~46% of decisions noise-flippable, it asks the question the
  campaign's most-replicated finding ("exactness beats estimation") says to
  ask: does this deviation actually FINISH with more points? Terminal score is
  the exact objective; a terminal comparison is the most exact estimator
  available at serving time.
- **Bounded downside by construction.** The fallback is the incumbent action;
  override requires a statistically valid lower bound clearing a preregistered
  margin. The failure mode is wasted compute, not regression. After a campaign
  running roughly 0-for-10 on strength interventions, asymmetric downside is
  worth a lot.
- **Cheap staged falsifiers.** Phases A–C (contract freeze, v2-harness
  requalification as a CPU control, minimal GPU exact engine with parity kill
  tests) are mostly CPU work that reuses existing machinery.

### Against

- **The nested-cost problem is severe** (and honestly flagged as the kill
  test): a truly incumbent-faithful continuation runs the full
  transformer+Gumbel policy at every simulated future decision — candidates ×
  outer worlds × remaining decisions × full search cost. If unaffordable, the
  proxy continuations (A-DIRECT/A-DISTILL) reintroduce exactly the estimator
  error the wrapper exists to escape.
- **Headroom overlaps D1's.** Both harvest the same pool of wrong-argmax
  decisions — D1 at training time, Anchor at serving time. If D1 gates
  positive, Anchor's ceiling shrinks. Realistically a +0.3–0.8 idea, not a
  1.7 idea.
- **GPU-simulator assumptions are systems claims.** The Pgx-style 10–100x
  numbers come from a different stack; the 07-16 topology incident (24 CUDA
  contexts at "100% util"/149W completing zero seeds in 70 minutes) is a fresh
  reminder that GPU-residence claims on this WSL2 box need parity + throughput
  proof before anything downstream of them is believed.

### Verdict

The correct first post-D1 serving experiment. It is also the natural pivot for
EITHER D1 outcome: if D1 succeeds, Anchor probes the residual serving-time
errors; if D1 fails, Anchor pursues the same corrected-decision prize without
touching training at all.

---

## 2. Cascadia-NX — promising diagnosis, contested payoff route

### For

- **Sharpest diagnosis of our data.** When comparison noise at the decision
  boundary binds, per-evaluation cost is the lever; Cascadia's factorability
  (local patches, exact score deltas, monotone placement, D6 symmetry, finite
  chance model) is real and unexploited by a dense 88M trunk that re-derives
  it at every call.
- **Coherent literature line.** The best published stochastic-game systems of
  this shape won with cheap symmetry-shared afterstate evaluators buying more
  exact/chance-aware search: 2048 n-tuple + expectimax, the Azul NNUE thesis
  (94.07% of 10,218 games vs the strongest handcrafted heuristic), Stockfish
  NNUE economics (+92.77 Elo), TD-Gammon.
- **Best engineering discipline of the three documents.** It confronts the
  repo's own failed accumulator (`d64d32b6`: ~2.5x faster, ~−3 points,
  reverted) with dependency-complete deltas, property tests against full
  recompute, and — critically — a **pre-training economics kill test** on a
  frozen root bank: if p95 dependency closure scales with the whole occupied
  board, or the full-menu delta path loses to vectorized recompute, the thesis
  closes before any training spend.
- Inherits v3's exact contracts (menu enumeration, afterstate grounding,
  Gumbel allocation) — an evaluator/engine swap, not a whole-stack rewrite.

### Against

- **Its payoff route runs substantially through an axis we measured as
  saturated.** "Spend the saved budget on more comparisons" is, mechanically,
  more simulations and worlds — and n4096/d16 bought +0.21 ns; d32 was ns on
  strength; deep own-turn planning (R3.2) failed; chance-node leaf expectimax
  failed. For NX to pay, the cheap evaluator must enable a QUALITATIVELY
  different search (exact chance expansion, true multi-ply max^n), and our
  only probes of deeper search were nulls.
- **The accuracy crux.** A tiny factor network must roughly match an 88M
  transformer's RANKING accuracy exactly at the contested boundary. Our own
  smaller-transformer experiments already failed the accuracy-for-throughput
  trade at mere 2–3x shrinks. The Azul NNUE beat a handcrafted heuristic, not
  a strong learned model; 2048 is a single-agent 4x4 domain.
- **Months-scale build to first gameplay evidence** (state compiler for five
  wildlife-card semantics, habitat topology deltas, GPU-resident search),
  with the same systems risk noted for Anchor.

### Verdict

A legitimate challenger whose cheap offline kill tests should run as a
background workstream — they cost little and would falsify or fund the whole
thesis. The central open question its document cannot close: whether the
throughput-to-strength conversion beats our measured saturation curve.

---

## 3. Cascadia Foundry — deepest thinking, weakest bet

### For

- **The best first-principles content in any of the three documents.**
  Cascadia as monotone construction under shared supply; exact terminal
  witnesses; and the mathematically correct observation that the scoreboard IS
  a cooperative team MDP (mean seat = table/4), making the cooperative lever
  the only one whose ceiling covers the entire remaining gap — consistent with
  the campaign's own R1.1c assessment.
- **Exemplary methodological hygiene:** witness-vs-bound discipline, explicit
  premise gates, a confidence number designed to be falsified rather than to
  decorate.
- **§12 ceiling tomography is genuinely valuable and severable** (see
  carve-out below).

### Against

- **A conjunction of six novel mechanisms, each with zero precedent** of
  beating learned evaluation in any comparable game: score contracts,
  completion genomes, scarcity-price exchange, scenario braiding,
  archive-support action selection, commitment-collapse refinement. Program
  synthesis / population planning has a long history of losing to learned
  evaluators in complex games; Foundry needs to win six times simultaneously.
- **Discards the strongest asset we own** — a 98.3 policy and its calibrated
  search — rather than building on it. Longest time-to-first-evidence,
  highest opportunity cost.
- **The 76% forecast should be discounted heavily.** It is conditional on four
  premises, two of which (≥2.5 mean-seat points of nonanticipatively
  recoverable headroom; the GPU throughput bar) are themselves the hard open
  questions. The unconditional confidence is far lower; to the authors'
  credit, the document nearly says so.
- **Its load-bearing cooperative objective requires John's explicit ruling** —
  and once that ruling is granted, R1.1c table-native values reach the same
  lever with a fraction of the machinery. Foundry is not the cheapest claim on
  its own best idea.

### Verdict

Rank last as an architecture. Extract and run its diagnostic layer first.

---

## The carve-out: run ceiling tomography first

Foundry §12 (same-resource repacking, chronology-preserving hindsight replay,
four-board reallocation bounds, known-world diagnostics) is the most valuable
single experiment proposed in any of these documents, and it does not require
believing in Foundry:

- It answers, from inside the game, the question the July-16 external
  calibration could not: **how many points are physically recoverable under
  this exact rules identity** — i.e., whether 100 sits comfortably below the
  achievable ceiling.
- It is CPU-friendly and severable.
- It doubles as Foundry's own premise gate, as objective evidence for the
  R1.1c cooperative-values program, and as a sanity check on the campaign
  target itself.

## Recommended sequencing

1. **D1 to verdict** (running; nothing changes).
2. **Ceiling tomography** (Foundry §12, extracted) — cheap, informs
   everything.
3. **Anchor Phases A–C** — bounded, evidence-backed, the natural pivot for
   either D1 outcome.
4. **NX economics kill test** in parallel — offline, falsifies or funds the
   incremental-evaluator thesis before real spend.
5. **Foundry as a system** only if tomography exposes large recoverable
   headroom AND John rules for the cooperative objective — and even then,
   R1.1c is the cheaper first claim on that lever.

## Meta-observation

All three proposals independently converge on the campaign's two hardest-won
empirical findings: evaluation noise at the decision boundary is the enemy,
and exactness is the weapon. That convergence raises confidence in the
DIRECTION. Skepticism should be allocated in proportion to how much
unvalidated novelty each proposal stacks on top of those two facts — which is
precisely the ordering above.
