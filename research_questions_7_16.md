# Cascadia v3 external research brief — 2026-07-16

Status: **scope and context frozen before external research**  
Repository snapshot: `27dd9888e018f12b649297a550de10a25408a321`  
Live-state snapshot: 2026-07-16 01:27 EDT  
Requested output: a source-grounded answer to all ten questions below, with
explicit recommendations for the decisions each question gates.

## 1. What we are trying to accomplish

The project is building a superhuman four-player Cascadia agent using the v3
CascadiaFormer transformer and Gumbel search stack. The formal gate is:

> Mean seat score **at least 100 over 1,000 complete four-player self-play
> games**.

The current corrected-rules champion is the cycle-4 scalar CascadiaFormer-M
served at `n=1024`, `top_m=16`, `d=16`, blend `0.5`. Its canonical 100-game
mean seat score is **98.2975**. The corrected-rules distributional-Q model is
statistically tied at champion budget: **98.3850**, paired delta `+0.0875`,
95% CI `[-0.2411,+0.4161]`. The nominal gap to the target is therefore about
**1.7 points per seat**, but the target has never been externally calibrated
against elite human or prior-agent play.

This research is meant to decide whether the campaign is optimizing the
right target, using the right learning signal, and spending the next large
block of GPU time on the right mechanism. It is not a generic board-game-AI
survey. Every answer must end in a concrete implication for the current
Cascadia program.

The ten questions fall into three tiers:

- **Tier 1:** could redirect the campaign before the next approximately 100
  GPU-hours;
- **Tier 2:** could change the search/training machinery already built;
- **Tier 3:** cheap to answer and occasionally decisive.

If research capacity is constrained, answer questions **1, 2, and 3 first**.
They can reveal that the project is aimed at the wrong score, the wrong
training signal, or the wrong relabeling recipe.

## 2. Source-of-truth order inside the repository

Read current evidence in this order. Newer live evidence outranks dated plans.

1. [V3 source-of-truth README](docs/v3/README.md) — canonical document map and
   standing status.
2. [Campaign state](docs/v3/CAMPAIGN_STATE.md) — operational resume state,
   with the warning that its header can lag the newest experiment-log entry.
3. [Experiment log](cascadiav3/EXPERIMENT_LOG.md) — chronological evidence;
   use the newest entry for the latest recorded transition.
4. [Research log](docs/v3/RESEARCH_LOG.md), especially section 7 — consolidated
   verdicts and scientific lessons.
5. [Research agenda](docs/v3/RESEARCH_AGENDA.md) — prioritized living queue and
   decision rules.
6. [R1.4 densification design](docs/v3/R1_4_DENSIFICATION_DESIGN.md) — code-level
   audit of current targets and the D1 relabeling program. Its early status and
   staging text is historical where later experiment-log entries supersede it.
7. [Rules contract](docs/v3/RULES_CONTRACT.md) — official turn semantics and
   the July 9 compatibility break.
8. [Architecture](docs/v3/ARCHITECTURE.md) and
   [Gumbel self-play campaign](docs/v3/GUMBEL_SELFPLAY_CAMPAIGN.md) — model,
   search, and expert-iteration design.
9. [Infrastructure](docs/v3/INFRASTRUCTURE.md) — host roles, seed registry,
   gate methodology, and operational contracts.
10. [Original July 10 portfolio](claude_max_research_ideas.md) and
    [radical directions](docs/v3/RADICAL_DIRECTIONS.md) — hypothesis-generating
    documents, not current status. Claims marked unverified in the portfolio
    must be rechecked against primary sources.

Relevant implementation entry points:

- [Gumbel search](cascadiav3/real-root-exporter/src/gumbel.rs)
- [Exporter and rules identity](cascadiav3/real-root-exporter/src/main.rs)
- [CascadiaFormer model](cascadiav3/src/cascadiav3/torch_cascadiaformer.py)
- [Training objective](cascadiav3/src/cascadiav3/torch_train_cascadiaformer.py)
- [Game rules/configuration](crates/cascadia-game/src/game.rs)
- [Scoring-card definitions](crates/cascadia-game/src/types.rs)

## 3. Exact evaluation target and comparability warning

The campaign does **not** use an ordinary off-the-shelf BoardGameArena setup.
The enforced research identity is:

`cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`

Its material properties are:

- four players, 20 turns per seat;
- wildlife scoring card A for Bear, Elk, Salmon, Hawk, and Fox
  (`ScoringCards::AAAAA`);
- **habitat majority/size bonuses disabled** by
  `GameConfig::research_aaaaa`;
- corrected optional three-of-a-kind market refresh: decide from public
  information, then reveal the replacement draw;
- unplaced drafted wildlife returns before the market refill;
- score metric is the arithmetic mean of all four seat scores in self-play,
  not win rate, rank, or margin over opponents.

Question 1 therefore has a mandatory normalization step. A raw human or BGA
mean that includes standard habitat bonuses is not directly comparable with
98.2975 or the target of 100. Research must either find a matching no-habitat-
bonus population, reconstruct/subtract bonuses from game records, or state
that only a bounded/qualitative comparison is possible. It must also match
the all-A wildlife-card setup or quantify the card-set effect.

### Rules/provenance issue discovered on July 16

Commit `45fb5072` fixed a rare consecutive-four-of-a-kind bug by returning
each wiped set to the bag after that individual overpopulation resolution.
That change is covered by a deterministic regression test and affects only
previously buggy consecutive-wipe trajectories. However, at this snapshot:

- `RULES_CONTRACT.md` still says all set-aside tokens return only after the
  market is stable;
- the exporter still carries the July 9 rules identity rather than a new
  identity for the July 16 engine fix.

This is a documentation/provenance gap, not permission to blend artifacts.
Research conclusions must name the source revision and rules identity they
assume. Any eventual experiment design must resolve this contract mismatch
before treating pre-fix and post-fix results as one evidence population.

## 4. Current scientific state the research must respect

### 4.1 What is established

- **Evaluation noise is binding.** On 20,000 sampled roots, the median top-2
  completed-Q gap was `0.049` points and the median pairwise standard error
  was `0.051`; median decision SNR was about 1.06. Roughly 46% of serving
  decisions and 54.6% of corpus roots are noise-flippable.
- **Determinization behaves primarily as variance reduction.** Honest
  multi-world search beat an oracle that peeked at the true hidden order.
  More hidden information was not the missing strength mechanism.
- **The selfish fixed-model scaling axis is decelerating.** A 4x simulation
  probe at n4096/d16 gained only `+0.21` points with CI
  `[-0.59,+1.01]`, about one third of the prior log-linear prediction.
- **Validation loss is not a strength proxy.** Multiple 5–15% locked-
  validation improvements failed to improve puzzle-bank decisions or
  gameplay. Screens must inspect decisions; gates must inspect paired games.
- **Exactness helps when it is rigorous.** Exact final-personal-turn K1 is
  score-neutral and about 29x faster on that frontier. It is an adopted speed
  default, not a strength result.
- **The remaining score gap is diffuse.** The corrected champion has no
  catastrophic low-seat mode; mechanism attribution did not identify one
  scoring category that explains the whole gap.

### 4.2 Adopted economics/methodology

- exact K1;
- refresh search at one-quarter of the ordinary simulation budget;
- ghost opponents plus `d=32` as a score-noninferior serving-speed default;
- the accepted frozen puzzle bank for rapid offline ranking;
- preregistered Lan-DeMets/O'Brien-Fleming group-sequential gates with fixed
  looks at 40/60/80/100 pairs and planned final N at least 100;
- optional CUPED using the preregistered baseline per-seed score covariate.

The descriptive adopted-default battery scored `98.3925`, but it is not a
promotion verdict. The canonical champion identity remains cycle-4 scalar M
at n1024/d16 with mean `98.2975`.

### 4.3 Closed directions that must not be silently reopened

- static Gumbel sigma calibration at n256;
- rollout-level common random numbers as a standalone variance fix;
- serving-side Q-bias correction and LCB selection;
- leaf softmix, symmetry TTA at inference, checkpoint ensembles, and
  quantile-risk serving modes;
- smaller-model/larger-search on john0 CUDA;
- action-conditioned structured-Q head-only training;
- pairwise comparator/Borda serving and small-data policy-recall tuning;
- serving-side table-total values;
- simple root-menu widening from 256 to 512;
- depth-2 own-turn planning at the measured price;
- Stage-1 V1b/V2/C1/T0 densification arms and the flagless ctrl-SWA lead;
- bridge micro-optimizations below their preregistered throughput bars.

A new result may reopen one only if it changes the mechanism or supplies
materially new evidence. For example, static sigma tuning is closed, while a
root-variance-scaled sigma rule is a distinct hypothesis; rollout-level CRN
failed, while determinization-level paired sampling remains unanswered.

### 4.4 Current funded path: D1 targeted relabeling

D1 asks whether generation-grade n256/d4 targets are wrong specifically on
contested roots and whether high-budget relabeling can train a better policy.

Measured evidence:

- pilot: high-budget n2048/d16 x2 relabeling changed the argmax on **43.2%**
  of repeat-stable roots, versus a preregistered 20% continuation bar;
- full ledger: **43.6%** stable-label movement over 7,600 non-exact roots;
- mean regret on moved roots: approximately **0.36 points**, so the movement
  is not merely equal-value tie churn;
- movement is largest in opening/midgame roots and lowest late;
- uniform better-label scalar EI had previously been flat, so D1 is targeted
  and is planned with a distributional-Q head rather than another saturated
  scalar head.

At the live-state snapshot, Stage A generation attempt 3 is alive on john0:

- revision `45fb5072`;
- seeds `2026794000..2026795249`;
- ghost opponents off;
- corrected wildlife-bag engine;
- 24 owned model-bridge sessions, matching the successful cycle-4 generation
  topology;
- replayable decision and hard-root sidecars enabled.

The intended next sequence is generation -> hard-root harvest -> high-budget
relabel -> training-record emission -> distributional-Q fold retrain -> bank
screen -> paired gate. The approximately 15,000-root, n2048/d16 x2 relabel
tranche was estimated near 26 GPU-hours and still requires John's decision.

The external literature answer to question 3 is needed **before** locking its
tranche size, teacher budget, fold weight, sampling distribution, and whether
reanalyze becomes a standing pipeline stage.

## 5. Definitions needed for the questions

- **Mean seat score:** total points across every seat and game divided by the
  number of seats; in self-play every seat uses the same policy family.
- **Champion:** a model/configuration John has approved for the canonical
  scoreboard. A fast noninferior default is not automatically a new champion.
- **n:** root simulations per decision.
- **d:** number of distinct hidden-order determinizations/worlds cycled across
  simulations.
- **Completed-Q:** simulation mean for visited actions plus a model-derived
  fallback for unvisited actions.
- **Served action value:**
  `exact_afterstate_score_active + predicted_score_to_go`.
- **Hard root:** operationally, a root whose leading action comparison is
  uncertainty-sized (for the corpus census, top-2 gap below pairwise SE).
- **Repeat-stable relabel:** two independent mega-search repeats select the
  same teacher argmax; D1's primary movement rate is measured only on these
  roots.
- **Screen:** an offline ranking instrument, never promotion evidence.
- **Gate:** a preregistered paired-game experiment on a fresh registered seed
  block, with the allowed sequential looks and repeated CI.
- **Oracle guiding:** privileged information available during training but
  removed or annealed away before evaluation. This must not be confused with
  the campaign's failed oracle-peek serving experiment.
- **Luck correction:** a target or auxiliary that conditions out exogenous
  draw variance without leaking information unavailable at decision time.
- **Distributional-Q:** K quantile predictions of score-to-go. The currently
  served default is their mean; quantile-specific risk modes were screened
  separately and did not justify a gate.

## 6. Research questions

## Tier 1 — could redirect the campaign

### 1. External calibration of the achievable score ceiling

What mean seat scores do the strongest humans and bots actually achieve in
four-player Cascadia under a setup equivalent to all five wildlife card-A
scorers, especially on BoardGameArena? We selected 100 as the goal without
external calibration of the achievable ceiling. If elite comparable play
averages about 97, the remaining 1.7 points may mostly reflect luck variance
or a miscalibrated target. If elite comparable play averages 105 or higher,
there are likely strategic points still being left on the table.

Also determine whether any prior Cascadia AI, solver, benchmark, dataset, or
academic game-playing work exists.

Required analysis:

- establish score-rule comparability before quoting means;
- separate per-seat mean, winning-player score, table mean, and best-game
  anecdotes;
- report sample sizes, player-count mix, card/scenario settings, expansions,
  and rating/skill-selection method;
- estimate uncertainty and selection bias where raw game records permit;
- explain whether habitat bonuses can be removed or bounded;
- search academic literature, theses, repositories, competitions, and public
  implementations for prior Cascadia agents;
- do not infer a human ceiling from a leaderboard rating alone.

**Decision gated:** continue treating 100 as a hard strength target, change
the target or evaluation setup, or formally bring the calibration question
to John for a methodology ruling.

### 2. Transfer from superhuman stochastic imperfect-information agents

How did superhuman four-player, stochastic, imperfect-information agents —
especially Suphx for Mahjong — solve the closest published version of our
problem shape: multiple seats, hidden information, chance, score
maximization, and high outcome variance?

Suphx reportedly used oracle guiding (train with privileged information and
anneal it away), a global reward predictor to de-noise learning against luck,
and run-time policy adaptation. Determine:

- the exact algorithms and target definitions;
- what information was privileged and how it was removed;
- whether the reward predictor estimated expected return, corrected luck,
  modeled opponents, or served another role;
- the quantitative ablations for each component;
- which elements transferred to other games and which were Mahjong-specific;
- whether later Mahjong systems confirm, replace, or contradict these
  mechanisms;
- how to transplant an idea without violating Cascadia's public-information
  rules or leaking hidden order into evaluation labels.

**Decision gated:** add an oracle-guided training arm and/or a luck-corrected
reward target to the queue. Both are distinct from giving the serving policy
hidden information and must be judged against the measured SNR-about-1 root
problem.

### 3. Quantitative recipe for targeted relabeling / Reanalyze

What does the literature say quantitatively about relabeling states with a
stronger or higher-budget teacher?

Required questions:

- how much teacher-budget amplification paid;
- what fraction of replay data was relabeled;
- whether relabeling was uniform, recent-state-biased, surprise-weighted,
  uncertainty-weighted, or restricted to hard states;
- whether targeted hard-state relabeling has been validated head-to-head
  against uniform relabeling;
- how reanalyzed targets were mixed or weighted against original targets;
- whether originals were retained, replaced, or interpolated;
- the diminishing-returns curve from teacher search budget to student
  decision improvement;
- how often relabeling ran and whether it became a permanent pipeline stage;
- how off-policy distribution shift and stale targets were controlled;
- which published effects were decision/gameplay improvements rather than
  training-loss improvements.

Map every quantitative result onto D1's choices: approximately 15,000 hard
roots, n2048/d16 x2 teacher search, distributional-Q retraining, fold weight,
bank-screen threshold, and eventual paired gate. State clearly where the
literature does not identify a transferable numerical optimum.

**Decision gated:** D1 tranche size, teacher budget and repeats, fold weight,
hard-only versus mixed sampling, and whether reanalyze should become a
standing stage after one successful cycle.

## Tier 2 — would tune the machinery we already have

### 4. Best-arm identification under a fixed noisy-comparison budget

The root decision is a fixed-budget best-arm-identification problem: choose
the best of roughly 16 candidates while about 46% of comparisons are
noise-flippable. Current allocation uses Gumbel top-k plus sequential halving.

Survey fixed-budget BAI and simulation-optimization results relevant to:

- sequential halving;
- successive rejects;
- LUCB, lil'UCB, top-two racing, and related fixed-confidence methods when
  adapted to a hard budget;
- ranking-and-selection/optimal computing budget allocation;
- correlated/common-random-number observations;
- non-Gaussian or heteroscedastic outcomes;
- model bias plus Monte Carlo noise, rather than independent stationary
  bandit rewards;
- batched allocation constraints.

Compare finite-budget error/simple-regret guarantees, robustness to this
noise structure, and empirical evidence. Do not treat the closed LCB final-
selection experiment as a test of a different allocation algorithm.

**Decision gated:** modify the root allocation algorithm itself, or retain
sequential halving and limit changes to its inputs/constants.

### 5. Gumbel AlphaZero constants under noisy Q

Find published guidance and sensitivity studies for `c_visit`, `c_scale`, and
Q normalization in stochastic or high-variance domains such as 2048 and
backgammon-like games. The current implementation uses:

`sigma(q) = (c_visit + max_visits) * c_scale * normalize(q)`

with `c_visit=50`, `c_scale=1.0`, and min-max normalization.

The campaign already swept static lower scales/normalizations at n256: a
shared-baseline screen looked positive, but the disjoint 100-seed confirm for
`c_scale=0.25`, top-k-range normalization was `-0.2325`, CI
`[-0.5440,+0.0790]`. Static calibration is therefore closed at this regime.

Research must distinguish:

- published static hyperparameter choices;
- domain sensitivity/ablation evidence;
- scale invariance assumptions;
- variance- or uncertainty-adaptive sigma/cPUCT rules;
- tuning under learned biased Q versus unbiased rollout returns.

**Decision gated:** parameters and form of the open L3 variance-scaled-sigma
experiment, not a repeat of the failed static sweep.

### 6. Value targets for multiplayer score-maximization games

What did successful multiplayer Go, Mahjong, Catan, Blokus, drafting-game,
and related RL systems train their value functions against:

- own absolute score;
- score margin over the field;
- win probability or rank utility;
- full table score/value vector;
- team/table total;
- multi-head combinations of the above?

Find head-to-head ablations where possible. Separate the training target from
the search backup/objective and from the evaluation metric.

Cascadia currently trains own-seat score-to-go for the load-bearing Q head
and a four-seat outcome vector as an auxiliary. Serving-side table-total
search lost twice because it multiplied leaf noise; a root contention audit
found no cheap own-Q-parity points. A table-native target trained from table
outcomes is a distinct, still-open proposal. It also changes the meaning of
the agent from selfish competitive play toward optimizing the self-play table
mean, which John must rule on.

**Decision gated:** the exact target definition for the first R1.1c/table-
native retrain and whether it is scientifically legitimate for the 100-point
scoreboard.

### 7. Variance reduction across determinizations in imperfect-information MCTS

The search averages over 4–32 sampled hidden worlds. Rollout-level common
random numbers failed its preregistered variance bar, but the campaign has
not answered whether candidates should share paired/stratified world samples
at the determinization level.

Research information-set MCTS, PIMC, stochastic MCTS, and simulation-
optimization work on:

- common random numbers across root actions/candidates;
- determinization reuse versus resampling;
- stratified, quasi-Monte Carlo, antithetic, or control-variate world
  sampling;
- variance reduction on action *differences*, not marginal returns;
- bias from strategy fusion, nonlocality, or world determinization;
- correlation effects on selection guarantees;
- empirical results in card, tile, or hidden-state games.

**Decision gated:** whether a determinization-level pairing/stratification
experiment should replace the failed rollout-level CRN experiment.

## Tier 3 — cheap to answer, occasionally decisive

### 8. Hex-symmetry data augmentation in board-game training

What measured gains have rotation/reflection augmentation produced in
hexagonal or square board-game agents? Distinguish effective-data gains,
sample efficiency, final playing strength, and inference-time ensembling.

Cascadia's inference-time three-rotation TTA was flat at 3x cost, plausibly
because relation-bias attention is already geometric/invariant. That does not
by itself test free training-time augmentation.

**Decision gated:** add rotation/reflection augmentation to the trainer as a
cheap data-bound experiment, or close it because the architecture/data
already supplies the relevant invariance.

### 9. KataGo-style adversarial blind-spot discovery

How were KataGo and related game-playing systems' systematic blind spots
found, reproduced, trained against, and tested for generalization? What made
an adversarial probe productive rather than a narrow exploit generator?

The Cascadia champion may have self-play blind spots in rare states. A rare
deep ghost trajectory already exposed the consecutive-overpopulation bag
bug, demonstrating that ordinary self-play coverage misses important paths.

**Decision gated:** allocate a targeted adversarial/probe-generation arm and
define its acceptance tests, diversity constraints, and non-overfitting
evaluation.

### 10. Risk-sensitive action selection for a mean-score objective

When the evaluation objective is expected score under high variance, do
distributional/quantile planning heads outperform scalar expected-value heads
in decision quality, and under what aggregation rule? Find ablations that
connect quantile heads to decisions or gameplay, not merely lower loss.

Cascadia's K=8 distributional-Q head was CI-positive at low budget under the
legacy rules, tied the scalar head at corrected-rules champion budget, and
showed no useful standalone signal from q25/q50/q75 risk modes. D1 plans to
use a distributional head as a less-saturated learner, but the serving rule
still needs justification.

Research must distinguish:

- distributional training as regularization for a better conditional mean;
- risk-sensitive objectives such as lower quantiles, CVaR, or optimistic
  quantiles;
- risk-neutral expectation, which is theoretically aligned with mean score;
- aleatoric luck variance versus epistemic/model uncertainty;
- where a distributional prediction should enter completed-Q and Gumbel
  improvement without double-counting search variance.

**Decision gated:** how D1's distributional-Q output is reduced into the
completed-Q value at serving, and whether any risk-sensitive mode deserves a
new experiment.

## 7. Evidence and sourcing standard for the answers

The final research report must:

1. Prefer primary sources: papers, official technical reports, authors'
   repositories, official game/rules/statistics pages, and original datasets.
2. Use secondary sources only to locate primary evidence or when no primary
   source exists; label that limitation.
3. Give a direct link for every material factual claim.
4. Record exact experimental conditions, sample sizes, metrics, baselines,
   and ablation deltas rather than saying a method merely “worked.”
5. Separate:
   - published fact;
   - inference from published evidence;
   - transfer hypothesis for Cascadia;
   - recommendation under uncertainty.
6. Avoid importing Elo, win rate, raw score, or sample-efficiency factors as
   commensurate quantities without an explicit mapping.
7. Treat negative and null evidence as first-class results.
8. Check whether an apparently relevant mechanism is already implemented or
   already closed in this campaign.
9. Never use pre-July-9 Cascadia scores as corrected-rules promotion evidence.
10. State when the literature cannot support a precise numerical recipe.

## 8. Required shape of the final answer

Produce a companion `research_answers_7_16.md` that keeps this brief stable
as the scope contract. The answer document should contain:

1. **Executive verdict:** what changes now, what does not, and why.
2. **One section per question**, each with:
   - short answer;
   - best available evidence table;
   - applicability and failure modes for Cascadia;
   - direct answer to the gated decision;
   - confidence level and unresolved unknowns.
3. **Decision matrix:** `adopt`, `run a bounded experiment`, `defer`, or
   `close`, including estimated implementation/GPU cost where repo evidence
   supports it.
4. **D1 prescription:** tranche size, sampling distribution, teacher budget,
   repeats, fold weight, target replacement/interpolation, screen, gate, and
   standing-pipeline rule — with every parameter labeled as evidence-based,
   repo-derived, or judgment.
5. **External calibration verdict:** whether 100 remains defensible under a
   genuinely comparable scoring setup.
6. **Source ledger:** primary references with the exact claim each supports.
7. **Repo follow-ups:** specific documentation, preregistration, or code
   changes implied by the research. Do not launch or promote an experiment
   solely from literature evidence.

The work is successful only if it changes or validates concrete campaign
decisions. A broad literature summary without those mappings is incomplete.
