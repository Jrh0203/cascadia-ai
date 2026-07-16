# Critical evaluation of the July 16 architecture proposals

**Date:** 2026-07-16
**Author:** Claude (Fable 5), at John's request
**Scope:** critical evaluation and ranking of the three post-D1 architecture
proposals, judged against the campaign's measured constraints. This document
ranks research bets; it is not strength evidence and it does not reorder the
authorized D1 chain.

**Resolution (John, 2026-07-16):** the cooperative carve-out is rejected. The
allowed policy class is four isolated agents maximizing their own raw terminal
score. Foundry-Commons, table utility, donation, shared cross-seat
plans/prices/memory, joint four-board genomes, and the associated conditional
76% forecast are withdrawn. The finalized synthesis is
[Cascadia Rival](cascadia_rival_final_architecture_proposal_7_16.md), which
accepts this document's Anchor > NX > Foundry ranking but retains only
unilateral, seat-local Foundry tomography and contracts.

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

After the objective ruling, “tomography” here means unilateral own-seat
repacking, chronology-correct public replay, late-game best response, and
explicit information/resource relaxations. Four-board cooperative resource
reallocation is not part of the funded measurement.

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
  witnesses; and the arithmetic identity that mean seat equals table total
  divided by four. **Resolution:** that evaluation identity does not authorize
  the larger policy class of a centralized cooperative controller. John
  rejected that controller class, so the original “cooperative team MDP”
  interpretation and ceiling argument are not current recommendations.
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
- **Its load-bearing cooperative objective was rejected by John's ruling.**
  The original critique observed that, even if accepted, R1.1c table-native
  values would reach the same lever with a fraction of the machinery. That
  route is now outside the allowed policy class rather than an open fallback.

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
- Its unilateral and explicitly relaxed variants remain premise evidence for
  seat-local Rival, and a sanity check on the campaign target itself. They are
  not evidence for or authorization of the closed cooperative-values program.

## Recommended sequencing

1. **D1 to verdict** (running; nothing changes).
2. **Ceiling tomography** (Foundry §12, extracted) — cheap, informs
   everything.
3. **Anchor Phases A–C** — bounded, evidence-backed, the natural pivot for
   either D1 outcome.
4. **NX economics kill test** in parallel — offline, falsifies or funds the
   incremental-evaluator thesis before real spend.
5. **Cascadia Rival synthesis** only if the preceding selfish-headroom and
   economics gates pass. Foundry-Commons is withdrawn; its seat-local
   contracts and chronology audits may enter only as measured optional
   proposal/diagnostic modules.

## Meta-observation

All three proposals independently converge on the campaign's two hardest-won
empirical findings: evaluation noise at the decision boundary is the enemy,
and exactness is the weapon. That convergence raises confidence in the
DIRECTION. Skepticism should be allocated in proportion to how much
unvalidated novelty each proposal stacks on top of those two facts — which is
precisely the ordering above.

---

## Addendum (2026-07-16 evening): evaluation of Cascadia Rival

[Cascadia Rival](cascadia_rival_final_architecture_proposal_7_16.md) is the
finalized synthesis produced after John's objective ruling. Assessment: **the
strongest of the four documents; adopt it as the ranked-first post-D1
program, superseding the standalone Anchor recommendation above.**

### What it gets right

1. **It solves Anchor's central flaw with the correct statistical tool.**
   Rival-MF makes the cheap continuation a control variate whose measured
   correlation with paired full-incumbent continuations buys variance
   reduction while the high-fidelity one-deviation estimand stays unbiased —
   neither "exact but absurd" (A-EXACT alone; the doc's own §19.3 arithmetic:
   ~15 serial hours per root) nor "cheap but wrong estimand" (A-DIRECT). The
   variance algebra is honest, and rho is treated as the load-bearing unknown
   rather than assumed. The single best design idea in the four documents.
2. **The offline-first reframe makes Rival D1-shaped.** V1 is a labeling
   instrument: terminally confirmed one-deviation preferences become a
   hash-pinned relabel tranche -> ordinary retrain -> fresh paired gate ->
   promotion only by John's ruling. Structurally identical to the running D1
   pipeline, so its machinery (masked training views, matched-control retrain
   path, exposure audits, model-vs-model gates, relabel tooling) is directly
   reusable. Composition risk is deferred to an ordinary frozen policy that
   standard gates can judge.
3. **Best epistemics in the set.** A genuinely hostile red-team section
   (nested-cost arithmetic; "unbiased yet useless" control variates;
   winner's-curse cohort discipline; "terminal is not synonymous with
   precise"; D1-overlap with effect addition forbidden), and a 25–35% headline
   forecast with a conditional ladder keyed to the measured baseline gap —
   the exact opposite of the withdrawn 76%.
4. **It respects the objective ruling cleanly:** selfish-only,
   Foundry-Commons categorically rejected, Foundry reduced to seat-local
   contracts and unilateral tomography — while retaining this document's
   tomography carve-out in its legal form.

### Where the skepticism belongs

- **The affordability–power tension is the crux.** Even with MF, the
  full-incumbent panel n_H must be tiny for affordability, and the guard's
  power scales as sigma_H/sqrt(n_H) against a terminal variance the doc
  concedes is high. The likeliest failure is not bias or systems collapse but
  a statistically sound wrapper that admits almost no overrides at affordable
  budgets — exactly what the equal-wall control will expose. The program's
  value therefore concentrates in one early number: **measured low/high
  action-difference correlation rho on paired panels.** rho >= ~0.7 in the
  strata where appeals matter and the economics work; below ~0.5 the online
  form should close fast. Elevate the rho calibration to THE decisive early
  experiment, ahead of most of the Layer 2–4 build.
- **Two owned priors bear on rho and belong in the calibration design:**
  ghost continuations (a crude opponent proxy) were score-noninferior at
  serving — weak positive evidence that cheap continuations preserve
  decision-relevant ordering; and the D1 repeat/audit data bounds how noisy
  action differences are under the exact policy itself — an upper bound on
  any continuation correlation.
- **Scope discipline:** despite "NX must earn its role," Layers 2–4 are most
  of NX's engineering bill; the evidence ladder stages it correctly, but the
  Layer-2/3 build must not run ahead of the cheap falsifiers. The 3,000
  post-D1 GPU-hour cap is honest and material (~four months of john0 at
  recent utilization).

### Revised sequencing

1. D1 to verdict (running).
2. Gate 0: fresh July-16 canonical baseline.
3. Selfish ceiling tomography (T1/T2 legal forms).
4. **rho calibration on paired low/high continuation panels** — the decisive
   scientific unknown; promote it ahead of the full systems build.
5. Exact-GPU parity + throughput gates; bounded W_k shadow instrument.
6. First Rival relabel iteration through the standard screen/gate ladder.
