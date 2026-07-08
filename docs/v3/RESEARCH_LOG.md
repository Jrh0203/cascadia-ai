# Cascadia v3 Research Log — the road to 100

**Deliverable doc.** Every research direction tried in the Gumbel self-play
campaign: what it was, why we tried it, what we measured, and the verdict.
Updated continuously; the freshest entries are in the "Active program" section.

Goal (the gate): **mean seat score ≥ 100 over 1,000 games of 4-player
self-play.** Honest measured optimum as of 2026-07-08 morning: **98.28**
(100 games, champion serving config), gap **−1.72**.

---

## 1. Architecture

### 1.1 Model: CascadiaFormer

Transformer over tokenized game state (M ≈ 100M params; L ≈ 207M tested).
Inputs: public-state tokens + per-action feature tokens with pairwise
relation-bias attention (CGAB). Heads:

- **policy logits** (per action)
- **q / score-to-go** (per action, scalar): predicted final own score minus
  exact afterstate score — serving always grounds Q as
  `exact_afterstate_score + score_to_go`
- **value_vector** (4 seats, absolute seat order): predicted final score of
  every seat
- **rank / score-decomposition / uncertainty / aux heads** (training-time
  regularizers)

### 1.2 Search: Gumbel AlphaZero over determinized worlds

Rust exporter (`real-root-exporter/src/gumbel.rs`):

- Root: batched model eval → Gumbel top-m over the full legal menu →
  sequential halving on `g + logits + sigma(completed_q)`.
- Hidden information: every simulation **redeterminizes** the hidden
  tile/bag order (strict no-peek); `d` distinct determinized worlds are
  cycled across each action's simulations (common random numbers across
  actions).
- Interior plies: every seat advances by argmax of its own derived final Q
  (max^n multiplayer), menus capped at k_interior.
- Leaf value: `w·(model bootstrap) + (1−w)·(sampled-greedy rollout)`;
  w=0.5 at serving, w=1.0 acceptable for training labels.
- Completed-Q: visited actions = mean simulation value; unvisited = model
  derived Q. Improved policy = softmax(logits + sigma(completed_q)).

**Champion serving config: n=1024 simulations, d=16 worlds, w=0.5 → 98.28**
(10.6 s/decision). Scores 11/100 individual games ≥ 100.

### 1.3 Training: expert iteration (EI)

Self-play games at search strength → every visited root exports
(improved_policy, completed_q, real final outcome vectors) as training
targets → objective `gumbel-selfplay` (soft-target policy CE + Q + value +
aux) → gate new checkpoint vs incumbent with paired-seed batteries, 95% CI
excluding zero required for promotion.

---

## 2. Scaling laws & levers measured (all closed)

| Lever | Result | Verdict |
|---|---|---|
| Simulations n (64→2048) | +gains flatten past 1024 | closed, peak n1024 |
| Worlds d (4→32) at ~64 sims/world | **+0.9/doubling** 4→8→16; REVERSES at 32 (CI−) | closed, peak d16 |
| Leaf blend w at serving | w=0.5 best; w=1.0 −2.94 CI− | closed |
| Oracle peek (true hidden state) | **LOSES** to honest multi-world search | key science, see §3 |
| Model capacity (M→L 207M) | flat | closed |
| More data (3×), fresh-from-scratch M | flat (3 independent replications) | closed — EI saturated |
| Better labels (n512/d8-taught cycle-6) | flat | closed |
| Checkpoint ensembles (SWA, lineage, cross-arch M+L) | never CI+; weak members actively hurt (−0.78 CI−) | closed — shared model bias |
| bf16 serving | 26% action agreement w/ fp32 (label-unsafe), ~3% faster | rejected |
| Generation env knobs (gather/row-cap) | flat even at saturation — per-ply lockstep is latency-bound | closed; structural fix queued |
| Trainer data path | **5.5× step-time win** (shard mmap; 1.26→0.23 s/step) | shipped |

## 3. Key scientific finding

**Determinization gains are ensemble variance-reduction, not
hidden-information reasoning.** An oracle that peeks at the true hidden
tile order (upper bound on the value of hidden info) *loses* to honest
search over 16 determinized worlds. Therefore the binding constraint on
playing strength is **noise in the value estimates**, not hidden
information, capacity, or data. Every direction in the active program
attacks value noise or sidesteps the competitive objective.

---

## 4. Active program (2026-07-08, ranked by expected value)

### 4.1 Table-total search objective — IN PROGRESS

**Hypothesis.** The gate metric is the *table mean* (all four seats are
ours). Max^n competitive search spends points on denial moves that lower
the table mean. Retargeting search values from own-seat to
table-total converts destroyed value into measured points.

**Design.** `--gumbel-table-total`: terminal & rollout values become table
sums; leaf bootstrap = own exact-grounded Q + Σ other seats'
`value_vector` estimates; unvisited-action fallbacks shifted onto the same
scale (additive shift preserves ranking). Interior plies remain selfish
argmax (approximation; noted).

**Experiment.** 100-game candidate arm at n256/d4 w=0.5, paired seeds
2026995000+, verdict vs the existing own-seat n256/d4 baseline (96.95).
If CI+ → confirm at champion n1024/d16.

**Status.** Implementing. Results TBD.

### 4.2 Softened leaf bootstrap (max-bias correction) — IN PROGRESS

**Hypothesis.** The leaf value bootstrap takes the **max** over the leaf
menu's Q estimates. The max of N noisy estimates is upward-biased and
high-variance — and eval noise is the proven binding constraint. A
softmax(q/τ)-weighted mean lowers both bias and variance at the cost of a
slightly pessimistic policy value.

**Design.** `--gumbel-leaf-softmix <tau>`; interior advance stays argmax;
τ→0 recovers max. Implemented + unit-tested (monotone in τ, bounded by
max and mean, changes search values end-to-end).

**Experiment.** 100g at n256/d4 w=0.5 τ∈{2,4} vs the 96.95 baseline,
after the table-total probe (john0 sequential). Results TBD.

### 4.3 Reanalyze value targets — QUEUED

**Hypothesis.** EI saturation was measured on the *policy* prior. The
value head still trains on single noisy game outcomes. Training it toward
search root values (completed-Q at n1024/d16 — a far lower-variance
estimator) is a different label family; the saturation evidence does not
cover it. Lower value noise → better leaf bootstraps → the same +0.9/
doubling mechanism that made worlds pay.

### 4.4 Distributional value head — QUEUED

**Hypothesis.** Reduce per-eval variance at the source with a
quantile/categorical value head; serving uses the mean (later:
variance-aware world weighting). Attacks the proven constraint directly.

### 4.5 Market-refill chance-node expectimax — BACKLOG

Model the refill chance node explicitly instead of averaging it through
determinized worlds; surgical variance reduction where randomness enters.

### 4.6 Multi-bridge generation throughput — BACKLOG (enabler)

~2× generation wall-clock via worker partitioning across bridge processes.
No points directly; halves the cost of every probe and EI cycle.

---

## 5. Historical record (campaign to date, condensed)

- **Baselines:** greedy 87.6 → no-search q-head 89.6 → rollout search
  96.97 (later found hidden-info-leaky; honest rebaseline lower) →
  Gumbel n256/d4 96.95 → n512/d8 97.845 → **n1024/d16 98.28**.
- **EI cycles:** EI-0/EI-1 (rollout teacher era, superseded) → cycles 3–4
  (Gumbel selfplay, produced champion M) → cycle-5 CI− (fleet label
  poisoning at weight 0.75, n128/d4 MPS labels; nofleet ablation
  exonerated w=1.0 labels) → cycle-6 (d8-taught) flat → fresh-M solo flat.
  **EI is saturated at M capacity: replicated 3×.**
- **Fleet (john1-4 M4 minis, MPS):** training data only, never gates.
  n256/d4 labels at fold weight 0.25 verified safe (no regression);
  currently no customer while EI is saturated.
- **Ops lessons:** batteries TF32 OFF / generation TF32 ON; fp32 serving
  batch-invariant, bf16 not; paired verdicts via `paired_delta_stats`
  (t_ci_low/t_ci_high), promotion = CI excluding zero at ≥100 games;
  john0 jobs strictly sequential.

Full chronological detail: `cascadiav3/EXPERIMENT_LOG.md`.
Resume state / decision history: `docs/v3/CAMPAIGN_STATE.md`.
