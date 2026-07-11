# Breaking 100: A Max-Effort Research Portfolio

*Written 2026-07-10 by Claude (Fable 5), at John's request, after reading the full campaign record
(README / CAMPAIGN_STATE / RESEARCH_LOG / all 5,226 lines of EXPERIMENT_LOG), a code-level audit of
`gumbel.rs` + the feature/model stack, and a literature sweep (2011–2026). Literature claims below
were extracted from primary sources; ~20 passed adversarial verification before the verify pass was
cut for cost — unverified extractions are marked (unv.). Everything here respects the corrected
rules contract (`..._rules_2026_07_09`) and the standing promotion methodology (≥100 paired seeds,
95% CI excluding zero).*

**State:** champion cycle4 scalar n1024/d16 = **98.2975**; distq-k8 = 98.3850 (tied, flat in every
category). Gap to gate ≈ **1.6–1.7 points**. Deficit is diffuse: 0/396 seats below 90, 31.3% of
seats already ≥ 100, p10/p50/p90 = 95/98/102.

---

## TL;DR — the ten bets I'd make, in order

1. **R0.1 Calibrate the Gumbel sigma stack** — c_visit/c_scale are hardcoded Go defaults over a
   min-max-normalized Q; the Gumbel paper itself says shrink c_scale when Q is noisy. Days of work,
   plausibly the cheapest real points available.
2. **R0.2+R0.4 Variance-engineer the root decision** (paired rollouts via CRN, bias-corrected
   unvisited Q, LCB/top-2 racing final selection) — attacks the measured SNR≈1 mechanism directly
   with textbook, Elo-proven tools the stack doesn't use yet.
3. **R1.2 Ghost opponents** — your instinct is right and the code confirms the prize: 3 of every 4
   model evals in a simulation are spent animating opponents. Marginalize them (greedy ghosts →
   learned market transition → distribution tokens) and reinvest 3–4x eval budget into worlds/sims.
4. **R1.1 Play the gate, not the seat** — the metric is table-mean in self-play; max^n search
   spends points on denial. Training-side table-native Q was parked, never refuted. Cheapest test
   (persona-diverse seats) costs zero training.
5. **R0.6 Fix refresh-node economics** — a root refresh costs 9–10 full searches (55s vs 6s
   decisions). Exact hypergeometric enumeration or reduced-n sampling frees ~30–40% of wall time to
   reinvest everywhere.
6. **R1.4 Densify the training signal** — EI "saturation" was measured with sparse targets.
   KataGo's ownership/score-distribution auxiliaries were worth 1.65x alone and PCZero's path
   consistency halves value noise; per-hex targets are the strongest literature-backed challenge to
   a "closed" direction.
7. **R1.3 Factor the compound action** (your terrain+animal idea) — theory says additive Q is safe
   when interactions are explicitly encoded, and the only hard coupling in a Cascadia turn is
   "token onto the just-placed tile." Use factorization for enumeration/coverage first (it kills
   the greedy-256 truncation and the 86% recall ceiling), policy heads second.
8. **R3.2 Deep own-turn planning** — at depth_rounds=1 the search literally never sees our second
   own move; every multi-turn plan lives in the value head. Depth was closed *at 4-evals-per-ply
   prices*; ghost opponents change the price. Retest depth 2–3 own turns after R1.2.
9. **R2.1 Puzzle-bank offline gate** — a one-night investment that turns every future serving idea
   into a 45-minute screen instead of a 3-hour gate. Velocity compounds.
10. **R3.6 Measure the ceiling** — before spending weeks, spend one day estimating whether selfish
    self-play at this ruleset even *has* 100 in reach (mega-budget probe + scaling-law
    extrapolation). If it doesn't, the cooperative reframe (R1.1/R3.1) is not optional.

---

## 1. Why the last 1.6 points are hard: three structural diagnoses

### D1. The noise wall (proven)
Median top1–top2 completed-Q gap 0.049 vs median pairwise SE 0.051 → SNR ≈ 1; ~46% of decisions
are noise-flippable. The campaign's central finding — determinization gains are ensemble variance
reduction, oracle peek *loses* — means strength now comes from **more accurate comparisons**, not
more information. Two independent literature anchors agree:
- Veness et al. (NeurIPS 2011): in stochastic games, the error of the *return difference between
  actions* is variance-dominated below ~1024 simulations, and classical variance reduction (CRN +
  control variates) bought the equivalent of 25–60% more simulations (unv., primary source).
- Gedda et al.'s Kingdomino study (CIG 2018) found flat MCE beat UCT in a 4-player drafting game
  *because* chance-event noise makes per-move estimates unrepresentative — the same diagnosis, in
  the nearest published cousin of Cascadia (unv.).

Crucially, the Gumbel policy-improvement guarantee is **conditional on correctly evaluated
Q-values** (Danihelka 2022, ICLR spotlight; thesis 2023). At SNR≈1 the premise of the serving
algorithm's optimality is exactly what's broken — which licenses modifying the root machinery.

### D2. A selfish agent graded on a cooperative metric (unmeasured)
The gate is *mean seat score across all four seats of self-play* = table-total/4. All four seats
are ours. Max^n search maximizes own-seat score, which includes denial value — points that
*subtract from the measured metric*. The serving-side table-total experiments lost (−1.65, −1.05
CI−), but the forensics attributed the loss to a **noise multiplier at leaves** (~4x outcome
variance), not to absence of cooperative signal; the training-side variant (labels average noise
away) was designed (§4.3) and parked when distq owned the GPU. Nobody has ever measured how many
points per game seats currently destroy through contention. If the selfish self-play equilibrium
of this ruleset sits at ~99, no amount of noise reduction reaches 100 — only changing the joint
policy does.

### D3. Structural compute taxes (confirmed in code this session)
Where the 46 s/decision at n1024/d16 actually goes:
- **Opponents eat ~75% of model evals.** One simulation at depth_rounds=1 = 4 batched evals: 3 to
  animate opponent plies by argmax-Q over greedy-capped menus, 1 for our re-entry leaf
  (`gumbel.rs:459-658`). The ensemble that produces all measured scaling gains is paying a 4x tax
  to simulate teammates-modeled-as-adversaries.
- **A root refresh decision costs 9–10 full searches** (8 hidden-replacement sample searches +
  decline search + post-accept draft search; `gumbel.rs:907-997`). Refresh decisions average 54.9s
  vs 5.9s ordinary — with 9.5 opportunities/game that's roughly a third of game wall time.
- **The root menu is a greedy-ranked truncation at 256** (`rank_greedy_actions`, despite the
  docstring), out of legal menus reaching 9,204–10,512. Model-policy top-16 covers the
  completed-Q-best action only 86.1% of the time; whether the *greedy*-256 cut itself drops
  winners has never been audited.
- **Rollout noise is deliberately decorrelated across actions**: worlds are CRN
  (`det_seed` independent of action) but `rollout_rng` seeds include `action_index`
  (`gumbel.rs:759-769`) — the rollout half of every leaf value adds independent noise to exactly
  the pairwise comparison that is SNR-bound.
- **Sigma constants are unexamined defaults**: `c_visit=50, c_scale=1.0`, hardcoded, over
  min-max-normalized Q (`gumbel.rs:129-130, 339-357`). The Gumbel paper reports c_scale is a
  first-order knob under noisy/partially-observable evaluation and used 0.1 for Atari.

### A budget for the missing ~1.7 points
Not additive (mechanisms overlap), but plausible ranges given measured anchors:

| Source | Anchor | Plausible yield |
|---|---|---:|
| Root-decision variance engineering (R0.x) | 46% flippable; Veness 25–60% effective-sim equivalent; KataGo uncertainty stack +75 Elo | +0.2 – +0.8 |
| Reclaimed compute reinvested (R1.2, R0.6) at measured scaling (+1.1–1.2 per 4x budget, decelerating) | 4x eval tax; ~35% wall in refresh | +0.3 – +1.0 |
| Contention recovery / cooperative play (R1.1, R3.1) | unmeasured; mechanism sound; equilibrium unknown | 0 – +2.0 |
| Training-signal densification un-saturating EI (R1.4) | KataGo 1.65x factor; PCZero 2x value-noise | 0 – +1.0 |
| Candidate coverage to ~100% (R1.3) | mean candidate-oracle regret 0.075/root measured | +0.05 – +0.3 |
| Commitment/bimodality (R1.5) | bear triage already near-free (−0.08 residual) in *this* ruleset | 0 – +0.2 |

---

## 2. What the literature adds (salvaged sweep, 2011–2026)

**Search under noise.**
- *Variance Reduction in MCTS* (Veness et al. 2011): CRN should target **pairwise differences
  between sibling actions**; control variates were the most robust technique and stack with CRN;
  antithetic variates failed in deck games (draw goodness is state-dependent) — skip antithetic,
  do CRN + control variates. (unv.)
- *BAI-MCTS* (Kaufmann & Koolen, NeurIPS 2017): LUCB/UGapE-style **top-two racing** at the root
  (sample whichever of best-guess/challenger is most uncertain, stop on CI separation) used ~7–15x
  fewer leaf evaluations than elimination schedules at equal confidence. (unv.)
- *Gumbel AlphaZero/MuZero* (Danihelka et al. 2022 + thesis): SH was chosen over UCB-E etc. for
  hyperparameter-freeness, not dominance; **smaller c_scale is the published mitigation for noisy
  Q**; deterministic non-root selection is itself a variance-reduction measure; completed-Q is
  load-bearing. (unv., primary)
- *Anytime SH* (2024) and *Batch SH* (RLC 2024): allocator swaps alone bought little in
  perfect-information games; batched SH is provably lossless under budget conditions — so don't
  expect points from exotic allocators per se; expect them from **variance-aware final selection
  and adaptive budgets**, where KataGo's measured gains live. (unv.)
- *KataGo methods* (living doc): uncertainty-weighted playouts + dynamic cPUCT ≈ **+75 Elo**
  (+50 vs tuned constant); **subtree value-bias correction +30–60 Elo** (targets *correlated*
  eval error — precisely our "shared model bias" ensemble finding); optimistic-policy head
  +40–90 Elo; LCB final move selection is standard at evaluation. (unv., firsthand engineering doc)

**Stochasticity & hidden information.**
- *Stochastic MuZero* (ICLR 2022) [verified]: afterstate + discrete chance-code planning matches
  perfect-simulator AlphaZero on 2048 and exceeds GNUbg past ~1000 sims in backgammon. We have an
  exact simulator, so the transferable content is the **afterstate/chance-node structure** and the
  backgammon recipe: **autoregressive micro-action policy + sampled search** for compound moves.
- *AlphaZe\*\** (Frontiers 2023) [verified]: PIMC-style AlphaZero is surprisingly strong when
  hidden information is non-adversarial shuffle randomness (our case), and identifies **"strategy
  hopping"** — instability caused by resampling worlds between moves without reuse — as a real
  pathology. Our exporter reseeds determinization streams per decision; world persistence across a
  turn is worth testing.
- *MAPLE* (arXiv 2605.24139, IEEE CoG 2026) [verified]: aggregating evaluations from multiple
  sampled world states inside one tree, with a Siamese network choosing the most *informative*
  worlds, beat PIMC-AlphaZero by +291/+136 Elo (Phantom Go / Dark Hex). We already aggregate
  worlds in one tree; the novel transplant is **informative (non-uniform) world selection**.
- *Kingdomino MCE* (2018): player-greedy playouts (opponents random) ≈ fully-greedy playouts in a
  4p drafting game — direct precedent for marginalizing opponents. Score-based backups beat
  win/loss backups when the metric is score. (unv.)

**Opponents in multiplayer search.**
- *Guiding multiplayer MCTS by focusing on yourself* (CoG 2020) and *MCTS-OMA-PW* (CoG 2020):
  in n-player games your own actions dominate your score; **reaching your own second move matters
  more than modeling the opponents in between** (the win-rate jump concentrates exactly at depth
  n+1); opponent-move abstraction = biased-but-low-variance estimates blended with decaying
  weight. (unv.) — This is your "opponent modeling is 3x compute for small benefit" intuition,
  published, twice.
- *KataGo*: the cheap counterpoint — an auxiliary head predicting the opponent's reply was a
  **1.30x training-efficiency factor** with zero serving cost. Model opponents in the loss
  function, not in the tree. (unv.)

**Action-space factorization.**
- *Sampled MuZero* (ICML 2021): planning over K sampled actions approximates full-menu
  improvement with variance σ²/K; K=50 of 362 nearly matched full search in Go; the β̂/β prior
  correction is the principled way to search a sampled compound menu. (unv.)
- *Multiagent Gumbel MuZero* (AAAI 2024): Gumbel-Top-k + sequential halving survives
  **autoregressive factorization** of a combinatorial action via stochastic beam search, with an
  importance-weighted policy-improvement loss for the sampled subset; robust at tiny budgets.
  Caveat: experiments used the simplest factorization on small tasks. (unv.)
- *Tang et al.* (NeurIPS 2022): linear Q-decomposition over sub-actions is **provably zero-bias
  iff transitions/rewards/policy factor**; when sub-actions interact, encode the interaction as an
  explicit term or the bias binds *precisely in the data-rich regime*. For Cascadia: factor for
  enumeration and candidate generation; keep a joint head for final ranking. (unv.)
- *AlphaStar Unplugged* (2023): the engineering template for autoregressive compound-action heads;
  also two cautions — value heads over factored actions were unstable, and **return-conditioned
  behavior cloning failed in a stochastic multi-agent setting** (condition on realized strategy
  features, not returns). (unv.)

**Training-side sample efficiency.**
- *KataGo paper* (2019): auxiliary **ownership + score-distribution targets were the single
  largest measured factor (1.65x)**; playout-cap randomization 1.37x; opponent-reply aux 1.30x;
  all stack to ~an order of magnitude. (unv.)
- *PCZero* (ICML 2022): path-consistency regularizer → ~2x reduction in value-estimate std along
  game paths, ~1.5x self-play sample efficiency, 84.3%→94.1% vs MoHex on Hex; explicitly argued
  to transfer to stochastic games via a Cauchy–Schwarz bound. Cheap to add. (unv.)
- *MuZero Reanalyse* (NeurIPS 2021) / *ReZero* (2024): with an exact simulator, reanalyze is just
  re-searching stored states for fresh targets; backward-view reuse makes it 2–4x cheaper. Use for
  target *changes* (new heads, table-native labels), not for more-of-the-same EI. (unv.)
- *Go-Exploit* (AAMAS 2023): start self-play from an archive of visited states — sample the states
  where improvement is informative (for us: high-variance/commitment states, refresh decisions).
  (unv.)

**Euro-game precedents.** No published strong Cascadia agent exists. Kingdomino (above) and PyTAG
(2024; MCTS still dominates PPO on Sushi Go!) calibrate the field: our 98.3 vs a 100 target is
already deep in unpublished territory — expect wins from mechanisms, not from copying a recipe.

---

## 3. The portfolio

Conventions: **Cost** = wall time on john0's 5090 unless noted (fleet = the four M4 minis;
"gate" = 100 paired seeds vs champion under corrected rules, CI-gated; n256 gate ≈ 2.5–3h/arm,
n1024 gate ≈ 10h/arm, 25g probe ≈ 45min). Every direction lists a preregisterable kill test.
Tier 0 items compose into a single "serving-v2" candidate after individual ablations.

### Tier 0 — Root-decision engineering (days; serving-side; no training)

**R0.1 Sigma calibration & final-selection rules.**
*Mechanism.* `sigma(q̂) = (c_visit + max_visits) · c_scale · minmax(q̂)` with c_visit=50,
c_scale=1.0 hardcoded. At n1024 the sigma range (~60–180) dwarfs logit differences, i.e. we sit at
the "trust noisy Q" extreme the Gumbel paper warns about; min-max normalization also lets one
terrible candidate compress the top-2 gap. Final action = argmax(gumbel+logit+sigma), not a
variance-aware choice.
*Do.* Expose `--gumbel-c-visit/c-scale` + normalization scheme (minmax | fixed-score-scale |
z-score | top-k-range). Sweep c_scale ∈ {0.05, 0.1, 0.25, 1.0} × 2 normalizations at 25g, confirm
best at 100g n256, then n1024.
*Kill test.* Preregister: any arm CI+ vs champion at n256 → n1024 confirm; all ns → close with the
sweep table as the artifact. Cost ~1 day. *Confidence: medium-high (published knob, measured
miscalibration risk, zero downside).*

**R0.2 Paired rollouts — CRN the other half of the leaf.**
*Mechanism.* Leaf = 0.5·bootstrap + 0.5·rollout. Worlds are CRN across actions; rollouts are
deliberately not (`action_index` in the RNG seed). Veness: apply CRN to *pairwise differences*.
Sharing rollout streams across actions at equal (visit, world) index converts independent rollout
noise into common-mode noise that cancels in comparisons.
*Do.* Seed `rollout_rng` from (det_stream, visit_index) only. ~5-line change + determinism tests.
*Kill test.* Offline first: on ~200 stored roots, re-run search both ways × many seeds; measure
variance of top1–top2 completed-Q gap and root-action flip rate across repeats. If paired variance
< unpaired by ≥20%, run the n256 gate. Cost: hours + one gate. *Confidence: medium-high.*

**R0.3 Unvisited-Q bias correction + shrinkage completed-Q.**
*Mechanism.* Completed-Q mixes sim-backed means (visited) with raw model Q (unvisited). The
structured-Q pilot measured the incumbent model Q running **+1.02 hot** vs teacher completed-Q.
Halving comparisons and improved-policy targets therefore systematically favor unexplored actions.
KataGo's subtree-bias correction (+30–60 Elo) attacks the same class of correlated error.
*Do.* Per root, after each phase: offset = mean(sim_value − model_q) over visited candidates;
apply to unvisited fallbacks (and to the completed-Q used for improved-policy export). Optionally
also shrink low-visit means toward corrected model Q with inverse-variance weights (the exported
per-action variance already exists).
*Kill test.* n256 gate; also check improved-policy entropy shift. Cost: ~1 day. *Confidence:
medium-high (mechanism measured, fix textbook).*

**R0.4 Variance-aware final selection: LCB + top-2 racing.**
*Mechanism.* 46% of decisions are coin flips; the final pick ignores variance. KataGo ships LCB
selection at evaluation; BAI-MCTS shows top-two racing dominates elimination at equal budget.
Because worlds are CRN, per-world value pairs for the last two survivors form a *paired* sample —
a paired t-statistic on ≤ d=16 world pairs is far tighter than unpaired comparison.
*Do.* (a) Final selection by LCB (mean − c·SE) among sufficiently-visited candidates; (b) reserve
~10% of n for a verification phase racing the final two survivors on matched worlds/rollouts,
pick by paired sign. The uncertainty head the model already emits (and Rust ignores,
`model_bridge.rs:388-414`) can prior-load the SE estimate after an offline calibration audit.
*Kill test.* n256 gate per variant; keep the better; n1024 confirm. Cost: 2–3 days. *Confidence:
medium-high.*

**R0.5 Adaptive per-decision budgets (spend sims where SNR is low).**
*Mechanism.* Fixed n per ply wastes budget on decisions with gap ≫ SE (p75 gap = 0.146) and
starves near-ties. Metareasoning literature + KataGo playout-cap results say allocation is where
compute-strength lives. Wall-neutral reallocation: early-stop halving when the leader's paired
lead is decisive; bank the surplus; grant banked sims to plies whose phase-1 top-2 gap is inside
noise.
*Do.* Implement stop rule (paired CI on matched worlds) + per-game sim bank. Preregister
wall-neutrality (±5%) so the gate measures pure allocation gain.
*Kill test.* n1024-wall-equivalent gate vs champion. Cost: ~3 days. *Confidence: medium.*

**R0.6 Refresh-node economics.**
*Mechanism.* Root refresh = 8 full accept-sample searches + decline + draft search. Three
independent savings, cumulative: (i) accept/decline needs less precision than move choice — run
sample searches at n/4; (ii) CRN across the 8 samples (shared determinization streams) tightens
the accept-vs-decline difference at equal samples; (iii) the replacement triple's distribution is
an exactly enumerable multivariate hypergeometric over ≤35 species-multisets from public bag
counts — replace Monte Carlo over hidden orders with exact weights on distinct multisets, valued
at reduced budget.
*Do.* (i) is a flag; (iii) is a day of enumeration code + tests. Reinvest saved wall into n or d
globally (that's the point — don't pocket latency, buy points with it).
*Kill test.* Wall-matched gate: candidate = cheap-refresh + bigger n/d at equal mean wall vs
champion. Cost: 2–3 days. *Confidence: medium-high (pure reallocation along a measured scaling
curve).*

**R0.7 World persistence across plies (anti–strategy-hopping).**
*Mechanism.* AlphaZe\*\* names the pathology: resampling hidden-world sets every decision makes
consecutive decisions optimize against different futures — plans wobble. Our det streams reseed
per decision.
*Do.* Freeze the d world seeds for a full own-turn (or a whole game phase), refreshing only on
real information reveals. Cheap flag.
*Kill test.* n256 gate + a trajectory-consistency metric (how often consecutive own decisions
flip strategic cluster). Cost: ~1 day. *Confidence: low-medium (cheap enough to try anyway).*

**R0.8 Control variates from the exact simulator.**
*Mechanism.* Veness's most robust technique, unused here. With exact chance distributions we can
subtract β·(1[event] − P[event]) from rollout returns for events whose conditional probability is
known (e.g., "next refill contains species s", hypergeometric from bag counts; "own next draw
completes a pair"). Removes chance-realization variance from leaf values without touching bias.
*Do.* Instrument rollouts to log candidate indicator events; fit β offline on stored rollouts;
implement the 2–3 best. *Kill test.* Offline variance reduction ≥15% on leaf values → n256 gate.
Cost: ~3 days. *Confidence: medium.*

### Tier 1 — Structural programs (1–3 weeks each)

**R1.1 Play the gate: cooperative table optimization** *(the highest-variance, highest-ceiling bet)*
*Mechanism.* D2 above. Three escalating stages, each independently informative:
- **(a) Contention audit (CPU/minis, ~1 day, no GPU).** Replay stored n1024 ledgers; for each of
  our decisions, evaluate the chosen action and the runner-up under *table* value (sum of
  value_vector) instead of own-Q. Measure: how often argmax flips, and the implied table-points
  delta. This bounds the prize before any training. Also measure market-contention events
  directly (we take a pair another seat's Q ranked much higher than ours).
- **(b) Persona-diverse table probe (zero training, ~1 gate).** The distq bridge already exposes
  q25/q50/q75 serving modes with provenance; they're score-flat individually but genuinely
  trajectory-diverse. Assign different personas per seat (needs a small per-seat mode map in the
  exporter) and gate the *table mean* vs 4×identical. If diversity alone is CI+, contention is
  real and large; if flat, temper (a)'s estimate.
- **(c) Table-native Q training (the §4.3 design, unparked).** Generate labels with
  `--gumbel-table-total` on the fleet (labels average leaf noise — the mechanism that killed the
  serving-side variant doesn't apply), train a table-Q head (warm start, ~25min–1.6h on the 5090),
  serve with `--gumbel-table-native-q` (flag exists; interior plies become cooperative argmax
  table-Q — no value-vector shift, no leaf noise multiplier). Gate at n256 then n1024.
*Decision John owns:* whether a champion whose seats cooperate (or are persona-assigned) is
in-spirit for the gate. It is literally what "mean seat score in self-play" measures; but it
changes what the number *means*. I'd rule: table-native value = legitimate (it's just the right
objective); explicit seat-asymmetric collusion conventions = separate scoreboard line.
*Kill tests.* (a) prize < 0.3 pts → deprioritize; (b) CI− → skip personas; (c) standard gate.
Cost: (a) 1 day, (b) 1 gate, (c) ~1 week including fleet generation. *Confidence: medium, ceiling
highest of any direction.*

**R1.2 Ghost opponents: marginalize, don't simulate** *(your market-distribution idea, staged)*
*Mechanism.* D3: 3/4 of per-sim evals animate opponents; the user-hypothesis and three published
results (Kingdomino playouts, self-focused MCTS, OMA) all say opponents in low-interaction
drafting games ≈ a stochastic market operator.
- **Stage A — greedy ghosts (days).** Interior opponent plies advance by the *existing CPU greedy
  policy* instead of model argmax: zero GPU cost for opponents, exact simulator dynamics kept.
  This is the same approximation the leaf rollout already makes, one ply earlier. Reinvest: at
  equal wall, n×~3 or d×~3 (from the +1.1–1.2/4x scaling anchor, worth ~+0.5–0.9 if the ghost
  bias is small).
- **Stage B — learned own-turn transition (1–2 weeks).** Train a market/state transition
  ("what does the world look like at my next turn"): inputs = afterstate + public supply; outputs
  = distribution over next-own-turn market compositions (and refill events). Self-play transitions
  are free training data (millions of rows in existing corpora). Search then samples s' directly:
  1 eval/simulation instead of 4. Opponent tendencies (your salmon-collector effect) are learned
  *as depletion conditioned on visible boards* — exactly your framing of opponent modeling as
  distribution-shaping.
- **Stage C — distribution-token leaves (2 weeks, composes with B).** Your "don't even have
  specific markets" idea: teach the model to evaluate states whose market slots are replaced by
  supply-distribution tokens (bag counts are already input features — `wildlife_bag[5]`,
  per-species capacity). Training needs **no new generation**: mask market tokens in existing
  shards as augmentation. At serving, leaves beyond the next own turn use one distribution-token
  eval instead of d concrete-market evals.
*The honest physics:* the d-world ensemble works by decorrelating model error via input
perturbation; a single marginal eval loses that averaging. So Stage C must beat the ensemble on
*accuracy per eval*, which is an offline-measurable question — **preflight before any gate**: on
held-out roots with known realized outcomes, compare RMSE(1 distribution-token eval) vs
RMSE(mean of d concrete evals) vs cost. Stage A/B keep sampled concrete futures (perturbation
preserved) and only remove the opponent-eval tax, so they're the safe first moves.
*Kill tests.* A: wall-matched gate (ghost + reinvested budget vs champion) — CI− on ghost-only at
equal n would also be informative (bias too big). B: transition-model held-out log-loss + A-style
wall-matched gate. C: the RMSE preflight gates everything downstream.
*Confidence: A medium-high; B medium; C genuinely uncertain but cheap to preflight.*

**R1.3 Factored compound actions** *(your terrain+animal decomposition, made exact where possible)*
*Mechanism.* A turn = draft (≤20 incl. nature-token splits) × tile placement (coord×rotation) ×
wildlife cell (or none). Monolithic enumeration hits 10k actions; the 256 root cap is a greedy
truncation; policy top-16 misses the best action 14% of the time. The coupling structure is
mercifully thin: given a draft, tile-placement value and token-placement value interact mainly
through (i) "token onto the just-placed tile" (c = p case) and (ii) local adjacency (token next to
the new tile). Tang et al.: additive Q is zero-bias iff factorization holds; encode interactions
explicitly where it doesn't.
- **(a) Coverage first (audit + prefilter, days).** Offline audit: on stored full-menu v4 roots,
  compute how often the completed-Q-best compound action would be dropped by the greedy-256 root
  cap (unknown today!). Then build the factored scorer as a *candidate generator*: score all
  placements f(p,r|draft) and all cells g(c|draft) with two cheap head passes off the shared
  trunk, compose top-K compounds by f+g (+ explicit h(p) for the c=p case), verified against
  exact full-menu model scoring on a validation shard. Target: ≥99% best-action coverage at
  ≤ current eval cost. This dissolves both the 256 cap and the 86% ceiling *without touching the
  final ranking head* (final ranking stays the joint Q — the Tang caveat honored).
- **(b) Autoregressive policy head (1–2 weeks).** Draft → placement → cell, each conditioned on
  the previous (AlphaStar/backgammon-Stochastic-MuZero template). MA-Gumbel-MuZero shows Gumbel +
  SH + importance-weighted improvement survives factorization. Payoff: legal-menu-size-independent
  policy cost, better priors on huge menus, and a training-cost drop for every future cycle.
- **(c) Factored exactness (with R3.3).** Exact K1 currently enumerates thousands of compound
  afterstates; factored enumeration makes full-menu exact ops cheap enough to push exactness
  earlier (last-2-own-turns) and to run exact top-k retrieval at *every* root.
*Kill tests.* (a) coverage audit numbers are the gate — if greedy-256 basically never drops the
winner and factored coverage can't beat 86% at cost parity, close (b) too. (b) offline: policy
top-16 coverage vs monolithic head on held-out full menus; then n256 gate. Cost: (a) ~3 days,
(b) 1–2 weeks. *Confidence: (a) high value-of-information; (b) medium.*

**R1.4 Densify the training signal (the anti-saturation program).**
*Mechanism.* "EI is saturated at M" was established with sparse targets (policy CE + scalar Q +
value vector + light aux). The largest single measured factor in KataGo was dense
ownership/score-distribution auxiliaries (1.65x); PCZero's path consistency halves value-estimate
noise along trajectories — and **value noise is our binding constraint**, so a lower-variance
value head is points even at fixed policy quality.
*Do (bundle, one retrain each, ~1.6h/train — trivially cheap; corpus regen on fleet where new
labels are needed):*
1. **Per-hex ownership analog:** predict final per-cell (species-scored?, habitat-in-largest-area?,
   contributes-k-points) maps. Terminal boards are in the raw games; needs an export field +
   corpus regen on fleet (days, parallel to everything).
2. **Score-distribution head** over final own score (KataGo-style buckets) — doubles as a
   calibrated uncertainty source for R0.4.
3. **Path-consistency loss** on stored trajectories (no new data needed).
4. **Opponent-reply aux** (predict the next seat's chosen action) — 1.30x in KataGo, and it *is*
   your cheap opponent modeling.
5. **Short-horizon value heads** (score at +1/+2 own turns) — lower-variance feedback.
6. **Input fixes while we're in there:** feed the already-exported-but-unread supply arrays
   (`unseen_keystones_by_terrain`, `unseen_dual_terrain_pairs`, full per-species capacity instead
   of 2 sums), and explicit turns-remaining. (Model currently infers progress from tile counts.)
*Kill test.* Fixed recipe, one variable at a time where cheap (3 is data-free; 2/4/5 reuse
existing shards; 1 needs regen): held-out value RMSE + decision-SNR proxy first, n256 gate for
any that clears offline. If the bundle moves nothing, EI saturation survives a much stronger
challenge and we stop resourcing training-side ideas. *Confidence: medium; high information value
either way.*

**R1.5 Commitment & bimodality** *(your bear problem)*
*Honest calibration first:* in the current ruleset the engine's bear triage is already near-free —
zero-bear seats recover ~11.2 of the ~11.5 forfeited points (residual ~0.08). So the measured
upside *here* is small; the direction matters mainly as insurance and for richer rulesets where
nonlinear payoffs need 5+ setup turns.
*Do (in order of cheapness):*
1. **Straddle forensics (CPU, free):** from stored ledgers, cluster mid-game decisions by
   strategic direction (species mix of scored actions); measure how often the policy oscillates
   across clusters within a game and whether oscillation correlates with sub-100 seats.
2. **Oracle-bound features:** per-species *max-achievable-remaining* via greedy knapsack over
   remaining supply/board (CPU-cheap, exact-ish upper bound — the old queued idea, revived as an
   *input feature*). Gives the value head the "is 8-bears even reachable" signal that a 1-ply Q
   cannot represent. Needs corpus regen (fleet) + retrain.
3. **Strategy-conditioned value with hindsight relabeling:** condition the value/Q heads on a
   realized-strategy summary (final per-species count buckets), trained by relabeling existing
   corpora (free data). At serving, a periodic meta-decision (every ~5 plies) evaluates the root
   under the 3–5 plausible strategy tokens and commits to the argmax, hysteresis-gated.
   *AlphaStar caution honored:* condition on realized structural features, never on returns.
4. **Deep own-turn search (R3.2)** is the search-side fix for the same problem.
*Kill test.* (1) decides whether (3) is worth building: if straddling doesn't correlate with bad
outcomes, skip to (2) only. *Confidence: low-medium in this ruleset; higher for future rulesets.*

**R1.6 Targeted data: hard-root reanalyze + commitment-state seeding.**
*Mechanism.* Go-Exploit + Reanalyse: with an exact simulator, minting better labels for *chosen*
states is trivial. Two uses: (i) relabel measured-hard roots (top-2 gap < SE) with mega-search
(n4096/d32) on the fleet — precision exactly where SNR<1; (ii) seed self-play from stored
mid/late-game states (especially refresh decisions and commitment forks) so new corpora
oversample the decisions that matter.
*Kill test.* One EI-style cycle with 30% hard-root-relabeled mix vs the known-flat control recipe;
n256 gate. Cost: fleet-days + one train + one gate. *Confidence: low-medium (EI saturation looms),
pairs naturally with R1.4.*

### Tier 2 — Velocity multipliers (make every other bet cheaper)

**R2.1 Puzzle-bank offline gate.** One overnight GPU run resolves ~800–1,500 stratified roots
(across phases, SNR bands, refresh/non-refresh) with mega-search (n4096/d32) → frozen
ground-truth-ish action values. Every serving candidate then gets a 45-minute screen (run its
search on the bank, score regret/flip-rate vs resolved values) before any 3-hour gate. Validate
the screen once by correlating against the 6+ historical gate verdicts we can replay. *This is
the single biggest research-velocity item.*

**R2.2 Group-sequential gates.** Preregister O'Brien-Fleming-style interim looks (e.g., at 33/66
seeds) with alpha spending; stop early on decisive CI. Saves ~30–50% GPU on clear results with
controlled error rates — the campaign runs many gates; this compounds.

**R2.3 Covariate-adjusted verdicts (CUPED).** Regress per-seed deltas on per-seed covariates
(greedy floor score, refresh-opportunity count) already in the ledgers; adjusted CI shrinks at
zero GPU cost. Validate on stored ledgers first (does it shrink the rebaseline CIs by ≥10%?).

**R2.4 Bridge throughput.** The deferred `torch.compile` fix (triton needs a C compiler on WSL —
zig cc adapter exists), CUDA graphs for the fixed-shape eval path, and the queued jobs12/16/24
probe. Raw eval throughput is fungible with every serving direction (more n/d at equal wall).

**R2.5 GPU-native rollouts.** The open structural item from the logs: move terminal greedy
rollouts onto the GPU (batched across sims) or replace with a distilled CPU-cheap policy
(N-tuple/pattern table distilled from M's policy — v1 heritage, reborn as a rollout engine).
Rollout quality/cost feeds every leaf.

### Tier 3 — Radical reformulations

**R3.1 One brain, four hands.** The logical endpoint of R1.1: formally treat the table as a
single agent choosing all four seats' actions to maximize table total; chance = tile/token draws
only. Max^n disappears; the equilibrium question disappears; search depth in *decisions* triples
for free (opponent plies become our plies). This is the cleanest formulation of "the gate" — and
the most philosophically aggressive. Needs John's ruling before any build. Cheap prototype:
Stage-A ghost machinery + table-native leaf values already approximate it.

**R3.2 Deep own-turn planning.** Today a simulation ends at our re-entry menu: the search never
compares *sequences* of our own moves — every multi-turn plan is delegated to the value head.
Depth was closed at 4-evals/ply prices (`depth2` flat at 1.8x cost); R1.2 changes the price to
1–2 evals per own-turn. Retest depth_rounds 2–3 (own turns) under ghost/marginalized transitions,
ideally with R0.5's adaptive budgets concentrating depth on commitment-heavy midgame plies. This
is the search-side answer to your "5 setup turns before the payoff" concern, and the direction
most likely to matter for richer rulesets.

**R3.3 Exactness expansion.** K1 was pure profit (score-neutral, 29x). The frontier: (i) exact
top-k compound retrieval at every root via factored bounds (with R1.3c); (ii) last-2-own-turns
semi-exact endgame — expectimax over marginalized opponents + hypergeometric refills, factored
menus making the tree tractable; (iii) exact late-game *labels* for training (the old queued
idea): replace noisy search labels with near-exact values for the final 3–4 own turns, where
habitat completions concentrate (recall: high-budget scaling bought habitat).

**R3.4 Learned compute allocator.** Train a tiny head to predict per-root flip-probability
(features: phase-1 gap/SE, menu entropy, refresh flag); allocate n, d, and depth per decision by
predicted marginal value of compute. Supervision comes free from R2.1's puzzle bank (which roots
did more search actually flip?). The learned version of R0.5.

**R3.5 Smarter worlds, not more worlds.** d16 is the measured peak of *uniform* world sampling.
Two upgrades: (i) stratified determinization — partition hidden orders by decision-relevant
features (e.g., which species appear in the next two refills) and sample one world per stratum
(QMC-style; Veness warns antithetic fails in deck games, but stratification is the robust
sibling); (ii) MAPLE-style informative world selection — score candidate worlds by predicted
decision-relevance and spend sims on the discriminating ones. Both attack ensemble variance at
fixed d.

**R3.6 Measure the ceiling before climbing it.** One day of GPU: (a) extrapolate the measured
budget curve (97.07 → 98.30 for 4x; fit the decelerating curve to estimate the n→∞ asymptote of
*this* policy family); (b) run a 25–50g mega-budget probe (n4096/d16, ~2x champion wall via R0.6
savings) to test the extrapolation; (c) compare with the R1.1(a) contention audit. Outcome: an
evidence-based split of the 1.7-point gap into "reachable by noise reduction" vs "requires
changing the objective/policy" — which is *the* portfolio-allocation question.

---

## 4. Direct answers to your hypotheses

**"Opponent modeling is ~3x compute for a small benefit."** Agreed, and it's stronger than that —
the code shows opponents cost 3/4 of all per-sim evals (4x), and two published lines (Kingdomino
playouts; self-focused/OMA multiplayer MCTS) plus our own oracle-peek result all point the same
way. The portfolio operationalizes your instinct as R1.2 (remove the tax) rather than as better
adversary prediction. Two twists worth keeping: (i) the *cheap* form of opponent modeling —
auxiliary next-seat-action prediction in the loss — bought KataGo 1.30x training efficiency at
zero serving cost (R1.4.4); (ii) under the gate's metric the other seats aren't adversaries at
all, which is R1.1/R3.1 — the one place where "opponent" reasoning might be worth a lot, by
deleting it.

**"Model the market as probability distributions; collapse on observation."** This is
Stochastic-MuZero-shaped and I think it's right as a *destination* — R1.2 stages it so each step
is separately falsifiable (greedy ghosts → learned transition → distribution tokens), because the
physics cuts both ways: marginal evals lose the input-perturbation decorrelation that makes the
d16 ensemble work, so distribution tokens must win on accuracy-per-eval, which we can measure
offline for a few GPU-hours before betting a gate on it. Meanwhile the model already sees exact
bag counts; R1.4.6 feeds it the deck detail it's currently denied.

**"Terrain + animal instead of terrain × animal."** Right, with one theory-guided amendment (Tang
et al.): additive decompositions are safe for *enumeration and candidate generation*, risky as the
*final ranking function* when interactions matter — and Cascadia's turn has exactly one hard
interaction (token onto the drafted tile) plus soft adjacency ones. So R1.3 factors the
combinatorics (fixing the greedy-256 truncation and the 86% recall ceiling, and making exactness
cheap) while the joint Q head keeps the last word. Best of both.

**"Bimodal strategies; the model straddles the middle."** The bear data says the engine already
handles the canonical case shockingly well (~0.08 residual cost) — so I've sized this as
insurance here (R1.5.1 forensics first) and as a *first-class* concern for richer rulesets, where
the fixes are strategy-conditioned values via hindsight relabeling (on realized features, not
returns — AlphaStar's failure mode), oracle-bound reachability features, and R3.2's deeper
own-turn search, which is the only mechanism that lets *search* (not just the value prior)
discover that turn 3 of a 5-turn setup is worth it.

**"Take everything written down with a grain of salt."** The Tier-0/Tier-2 items are designed so
most verdicts arrive in hours (offline variance measurements, puzzle-bank screens, 25g probes)
before any 100-seed spend, and R3.6 exists specifically to check whether the whole selfish program
can get there at all.

**"More complex rulesets, nonlinear scoring, 5+ setup turns."** The directions that transfer
as-is: factored actions (R1.3), distribution abstraction (R1.2), dense targets (R1.4),
strategy conditioning (R1.5), deep own-turn planning (R3.2), and the velocity stack (Tier 2).
The ones that don't: anything tuned to card-A scoring specifics. Worth keeping in mind when
choosing between two otherwise-equal bets.

---

## 5. Suggested fortnight (respecting the live GPU queue)

The worlds screen + jobs probe (+possible 20h det16/det32 confirm) own john0 first. Everything
below slots around them; fleet and CPU work start immediately.

**Days 1–2 (CPU/minis, zero GPU):** R1.1a contention audit; R1.5.1 straddle forensics; R1.3a
greedy-256 coverage audit (offline on stored v4 full-menu roots); R2.3 CUPED validation on stored
ledgers; uncertainty-head calibration audit (for R0.4/R0.8).
**Days 2–5 (implementation, gates as GPU frees):** R0.1 sigma flags + sweep; R0.2 paired
rollouts (offline variance check → gate); R0.3 bias correction; R0.6(i) reduced-n refresh.
**Nights:** R2.1 puzzle bank generation (one night); R3.6 mega-budget ceiling probe (one night).
**Days 5–10:** R0.4 LCB/racing; R0.5 adaptive budgets; compose surviving Tier-0 pieces into one
"serving-v2" candidate → full n1024 gate. R1.1b persona-table probe. R1.2A greedy ghosts +
wall-matched reinvestment gate. Fleet: R1.4.1 ownership-label corpus regen + R1.1c table-native
label generation (they can share generation runs).
**Days 10–14:** R1.4 retrain bundle (path consistency first — data-free) + gates; R1.1c
table-native train + gate; decide R1.3b / R3.2 based on the audits.

Rough GPU arithmetic: Tier-0 ablations ≈ 6–8 n256 gates (~20h) + 2 n1024 confirms (~20h) + 2
overnight banks (~10h) — comfortably one week of 5090-nights alongside the queued campaign work.

## 6. How we'll know we're wrong (portfolio-level)

- If the Tier-0 bundle (R0.1–R0.8) moves nothing at n256, the noise-wall theory of the residual
  gap is weaker than measured SNR suggests — reweight hard toward R1.1/R3.6 (the objective, not
  the noise, is the wall).
- If R3.6 says the selfish asymptote is ≥100.5, deprioritize cooperative work; if ≤99.5, promote
  R1.1/R3.1 to the only game in town.
- If R1.2A's greedy ghosts are CI− *at equal n* (before reinvestment), opponent fidelity matters
  more than the marginalization thesis allows — cap R1.2 at Stage A and revisit.
- If R1.4's densified targets don't move held-out value RMSE, the saturation verdict extends to
  rich targets and training-side work stops competing for GPU.
- If the R1.3a audit shows greedy-256 never drops winners *and* factored candidates can't beat
  86% coverage at parity cost, the action-space program shrinks to the (still-valuable) exactness
  uses in R3.3.

*— end of document —*
