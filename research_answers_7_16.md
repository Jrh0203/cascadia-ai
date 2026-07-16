# Cascadia v3 external research answers — 2026-07-16

- Status: **complete research synthesis**
- Scope contract: [research_questions_7_16.md](research_questions_7_16.md)
- Repository baseline: `28dbad8b` (the scope brief); live Stage A engine
  revision: `45fb5072`
- Evidence cutoff: 2026-07-16

Operational update, reverified 2026-07-16 02:08 EDT: `campaign_status.sh` found Stage A
attempt 3 PID `204702` dead and john0 idle. The wrapper ended at “generating,”
the run log and both sidecars were zero bytes, and no tensor/manifest or
COMPLETE/FAILED record existed. Read-only diagnosis found john0/WSL had rebooted
at 01:32, terminating the run. This report does not restart it; the exact
failure record is in the [experiment log](cascadiav3/EXPERIMENT_LOG.md).
A second `campaign_status.sh` check at 02:36 confirmed the state was unchanged:
john0 remained idle and no Stage A job or waiter was live.

## 1. Executive verdict

The research supports the current funded direction, but it corrects three
important premises in the question set.

1. **No production Board Game Arena Cascadia population was available at the
   2026-07-16 cutoff.** There is therefore no BGA elite score distribution to
   query.
   The best public human proxy is World Boardgaming Championships play. Its
   standard-rules scores are not directly comparable with the repository's
   all-A, no-habitat-bonus identity. The 2025 WBC final's recorded habitat
   bonuses can be stripped exactly, yielding a recorded mean of `96.75`, but
   the players optimized with those bonuses present and used random wildlife
   cards. This result does not falsify `100`; it cannot calibrate the target's
   difficulty or establish a human ceiling.
2. **Suphx did not use a luck-corrected reward.** Its global reward predictor
   distributed a final *rank* reward across a Mahjong match's many rounds. Its
   oracle guiding exposed opponents' hands and the wall during training, then
   annealed those features away. Neither mechanism directly attacks Cascadia's
   measured public-state root-estimation problem. VLOG and AlphaStar provide
   stronger evidence for training-only privileged value/posterior mechanisms
   than for raw policy feature dropout.
3. **Determinization-level common random numbers already exist in this search.**
   Every action cycles through the same `d` hidden-world seeds. R0.2 tested the
   still-unpaired rollout stream and failed. The open question is narrower:
   stratified or otherwise better *world construction*, not basic pairing.

The resulting campaign decisions are:

| Question | Decision now | Why |
|---|---|---|
| Q1 external ceiling | **Keep 100 as an internal engineering gate; commission a separate exact-rules human calibration before making a human-superhuman claim** | Public data do not identify the comparable ceiling. |
| Q2 Suphx transfer | **Do not queue GRP, raw oracle dropout, pMCPA, or a purported luck-corrected target** | The premise is mismatched; D1 is the closer public-information analogue. |
| Q3 reanalyze | **With John's restart permission, rerun Stage A; then proceed with the 15k hard-root D1 design subject to his already-required relabel-tranche approval** | Literature supports current-teacher reanalysis and soft replacement; the pilot's hard-root movement persisted in the full 7,600-root ledger. The complete masked-fold/student/gameplay recipe is untested. |
| Q4 best-arm allocation | **Keep sequential halving; defer a frozen-bank allocator bakeoff until after D1** | No theorem covers the current correlated, biased, batched estimator, and the incumbent is a strong fixed-budget baseline. |
| Q5 Gumbel constants | **Keep `50/1.0/minmax`; defer L3 until an offline reliability calibration supplies its functional form** | Published constants do not identify a stochastic-domain variance rule; the static repo sweep already failed confirmation. |
| Q6 multiplayer target | **If John authorizes R1.1c, generate table-native per-action Q and policy targets; preserve a four-seat action-value decomposition where practical** | Cascadia already has a four-seat root value head. R1.1c must change action supervision under table utility, not merely residualize that existing head. |
| Q7 determinization variance | **Close “add basic pairing” as already implemented; defer a stratification test** | The premise is obsolete and direct evidence for permutation-stratified hidden worlds is weak. |
| Q8 symmetry augmentation | **Do not prioritize a standalone trainer arm; keep only a cost-neutral shadow option** | Inference TTA is closed under its tested configuration. Random D6 training augmentation remains unmeasured, and the located literature does not isolate its gameplay effect. |
| Q9 adversarial probes | **Run a bounded diagnostic probe bank after D1 and the rules-identity repair; do not open full adversarial training yet** | KataGo shows blind spots can survive huge search and naive adversarial training; probes need transfer and diversity gates. |
| Q10 quantile serving | **Serve the uniform quantile mean; no CVaR/quantile-risk mode** | The formal objective is expected score, and both published and repo evidence reject a universal risk distortion. |

No literature result authorizes a promotion, a rules change, or an experiment
launch by itself. The most valuable immediate non-research action is to repair
the July 16 rules-contract/rules-ID mismatch before the D1 artifacts are
allowed to join a scientific evidence population.

## 2. How to read the evidence

Each section distinguishes four kinds of statement:

- **Published fact:** directly reported by a primary paper, official rules
  source, official event report, or authors' implementation.
- **Repo fact:** measured or implemented in this repository and linked to its
  durable record.
- **Inference:** a conclusion that follows from facts but was not itself tested.
- **Recommendation:** a campaign choice under uncertainty.

An absence claim means that targeted searches of the named primary-source
venues and public implementations found no qualifying result by the evidence
cutoff. It is not a proof that private data or an obscure implementation does
not exist.

## 3. Q1 — External calibration of the achievable score ceiling

### Short answer

No public dataset measures strong humans or strong external bots under the
exact repository identity: four players, all five wildlife cards A, no habitat
majority bonuses, and 20 turns per seat. Cascadia is not live on BGA. WBC is
the best public tournament proxy, but it uses random wildlife cards and
standard habitat-majority scoring.

The defensible conclusion is therefore:

> `100` remains a legitimate frozen internal engineering target. It is not a
> demonstrated human ceiling, and `98.2975` is not yet a defensible
> human-superhuman claim.

### Best available human evidence

| Source | Exact observation | Comparability and use |
|---|---|---|
| [Official base-game rulebook](https://www.alderac.com/wp-content/uploads/2025/02/Cascadia-Rulebook.pdf) | Standard play has 20 turns per player, one wildlife scoring card per species, and habitat-majority bonuses. Card A is a recommended introductory set, not the mandatory competitive set. | Establishes why ordinary scores cannot simply be compared with `research_aaaaa`. |
| [WBC 2025 event conditions](https://www.boardgamers.org/wbc25/Previews/cas.html) | Base game, no promos, four-player games preferred, wildlife cards drawn randomly; Hawk C may be excluded by agreement. | Confirms that WBC is neither all-A nor no-bonus. |
| [WBC 2023 report](https://www.boardgamers.org/yearbook23/cas.html) | 90 unique players, 175 player-starts, 44 games; reported mean `93`, maximum `111`; final `97/95/92/87`. | Tournament context only. Stripping the table's unknown 15–20 majority bonuses bounds the final's bonus-stripped recorded mean at `87.75–89.00`. |
| [WBC 2024 report](https://www.boardgamers.org/yearbook24/cas.html) | 111 unique players, 250 starts, 63 games; reported mean `93`, maximum `114`; final `103/100/96/95`. | Tournament context only. The final's bounded bonus-stripped recorded mean is `93.50–94.75`. |
| [WBC 2025 report](https://www.boardgamers.org/yearbook25/cas.html) | 159 unique players, 357 starts, 90 games; reported mean `95`, maximum `114`; final `107/101/99/99`. The report gives final habitat bonuses `7/6/2/4`. | Exact bonus stripping gives recorded totals `100/95/97/95`, mean **`96.75`**, for one elite random-card table. |
| [2025 WSBG Cascadia Ring Championship broadcast](https://youtu.be/EeFo6DlVOR8) | One public final ended `101/98/96/94`, mean `97.25`. | Another standard-rules selected table, not exact-rules calibration; no machine-readable replay was published. |
| [BGA game catalog](https://en.boardgamearena.com/gamelist) and [direct Cascadia panel](https://en.boardgamearena.com/gamepanel?game=cascadia) | Checked 2026-07-16: Cascadia is absent; the direct panel returns “Game not found.” | There is no production BGA corpus, rating distribution, or elite mean to estimate. |
| [Cascadia Digital announcement](https://news.direwolfdigital.com/cascadia-digital-is-now-available-in-steam-early-access/) | The cited settings description documents selectable cards and the no-majority toggle only for Family scoring. It reports no bot-strength benchmark or public logs; targeted searches found none. | Confirms an opaque external bot exists, not its strength or whether a newer undocumented setting can reproduce the identity. |
| [Steam global achievements](https://steamcommunity.com/stats/2438970/achievements) | On the check date, 39.8% had ever scored 100+, 22.5% 105+, and 10.1% 110+. | Only proves that high single games occur. It has no mode, card, player-count, attempt, or selection denominator and cannot estimate a mean. |

Across the 2023–2025 finals, standard three-/four-player majority rules imply
15–20 habitat-majority bonus points per table. Stripping that bounded total
gives a 12-seat, three-table recorded mean of **`92.67–93.50`**. These are
bonus-stripped recorded totals, not estimates of what the same players would
score if majority scoring had been disabled from the start. Treating them as
an all-A human-policy estimate is invalid. The 2025 final's exact `96.75` is
the cleanest stripped proxy, but it is one selected table under a different
card distribution and different incentives.

The current `98.2975` is descriptively 1.55 points above that table mean and
roughly 4.8–5.6 above the pooled finals bound. Shared-market interaction,
random card mix, opponents, seed distribution, selection, and objective all
differ: WBC players optimize tournament placement and may block rivals, while
the repo measures own mean score in homogeneous self-play. Those deltas are
**not** promotion evidence.

Luck also cannot explain a gap between two expected means. Chance widens the
sampling interval around a fixed policy mean; it does not turn an unknown
external ceiling into `97` or `105`.

### Prior Cascadia AI, solver, and dataset search

| Project | What exists | Strength evidence |
|---|---|---|
| [davidxmoody/cascadia-ai](https://github.com/davidxmoody/cascadia-ai/tree/a75b0bc4) | PyTorch/PyG RL scaffold with all-A scoring. Its README says greedy play may account for about 90% of score and proposes Deep Q-learning for the remainder; the algorithm section is still `TODO`. | No released reproducible four-player exact-rules mean or trained benchmark artifact. |
| [TMahls/Cascadia](https://github.com/TMahls/Cascadia/tree/ef331947) | MATLAB rules shell with 1–4 players, all-A/easy rules, and habitat-bonus toggle. | Closest configurable shell, but autoplay is a placeholder and no agent result is reported. |
| [MarcelEindhoven/bga-cascadia](https://github.com/MarcelEindhoven/bga-cascadia/tree/240271ff5a1e2a012558fe638cad6bafb1bbe166) | BGA Studio prototype whose final commit says “Ready to play family version”; Default/Solo scaffolding also exists. `stats.json` is empty and the AI uses random market/position choices. | Never became a production BGA population; no useful strength result. |
| [awweide/cascadia](https://github.com/awweide/cascadia/tree/e66ec8a8) | Cascadia-inspired solo engine with random and short-horizon examples. | Incompatible short example setup and no competitive result. |
| [kononenkodaniil/cascadia-ai](https://github.com/kononenkodaniil/cascadia-ai/tree/1914b68a) | State simulation, parser, and augmentation pipeline with a tiny example log. | No packaged trained agent or benchmark with sufficient provenance. |
| [Sanat14/CascadiaBoardGame](https://github.com/Sanat14/CascadiaBoardGame/tree/61ec1b2e) | Java implementation with an immediate-score, one-turn programmed agent. | No reported four-player mean and not an exact solver. |
| [phyrwork/cascadai](https://github.com/phyrwork/cascadai/tree/78f1c5b7), [jaetill/cascadia_ai](https://github.com/jaetill/cascadia_ai/tree/528669af) | Partial engines, scoring, or roadmaps. | No reproducible strong agent or benchmark. |

Targeted arXiv, OpenAlex, GitHub, tournament, and publisher searches found no
peer-reviewed Cascadia gameplay-AI paper, exact solver, expert replay dataset,
or public external-bot benchmark. Commercial AI and private play data may
exist, but they cannot calibrate the campaign without disclosed records.

Reproducible query families used at the cutoff included `"Cascadia board
game" AI`, `"Cascadia" reinforcement learning`, `"Cascadia" solver`,
`"Cascadia" MCTS`, `"Cascadia" dataset`, and `"Cascadia" bot`, with
site-scoped variants for arXiv, OpenAlex, GitHub, BGA, WBC, WSBG, Dire Wolf,
and BoardGameGeek. This records the absence search without claiming proof of
nonexistence.

### Gated decision

**Retain `mean seat score >= 100 over 1,000 games` as the internal gate.** Do
not lower it from the available evidence, and do not describe it as a known
human ceiling.

Create a separate calibration protocol before a human-superhuman claim:

1. Freeze the exact all-A/no-bonus rules identity and score definition.
2. Recruit demonstrably strong players, ideally repeat WBC/WSBG finalists.
3. Run a blinded pilot, estimate table-level variance, then preregister sample
   size from a target CI half-width (a one-point half-width is a suggested
   **judgment**, not a published standard). `30–50` tables is only an initial
   planning heuristic; use the measured variance for the real N.
4. Rotate seats, preregister seeds, and retain complete market/action/card and
   category-score records. Define whether the estimand is the fixed recruited
   panel or an elite-player population; repeated players require player/table
   random effects rather than treating tables as independent population draws.
5. Run the agent in a separate standard-rules bridge battery only as
   model-specific sensitivity analysis. It cannot translate WBC human scores
   into exact-rules human scores.
6. Request anonymized replays from Dire Wolf, WBC, and WSBG. No complete
   machine-readable action/card/category replay corpus was public at cutoff.

**Confidence:** high that no *public* exact-comparison dataset or BGA
population was available; moderate in the completeness of the long-tail prior
AI inventory; low on the actual human ceiling because the necessary experiment
has not been run.

## 4. Q2 — Transfer from superhuman stochastic imperfect-information agents

### Short answer

Suphx is valuable as a design analogy, but not for the reasons in the initial
question. Its global reward predictor (GRP) was a final-rank credit-assignment
mechanism across 8–12 Mahjong rounds, not a luck correction. Oracle guiding
used privileged opponents' hands and wall tiles during training and annealed
them away with feature dropout. Cascadia has neither Suphx's round-score versus
match-rank mismatch nor hidden strategic hands that can be inferred from public
play.

**Do not queue literal GRP, Suphx oracle dropout, or runtime pMCPA.** D1's
higher-budget, public-information search-Q relabeling attacks the measured
SNR-about-1 failure much more directly. It is still a biased/noisy search
estimate, not proven ground-truth expected Q.

### What Suphx actually did

Primary source: [Li et al., “Suphx: Mastering Mahjong with Deep Reinforcement
Learning”](https://arxiv.org/abs/2003.13590).

| Component | Exact mechanism | Quantitative evidence and limitation |
|---|---|---|
| Global reward prediction | A two-layer GRU plus two fully connected layers predicts the same final rank-based match reward from every round prefix. Per-round RL reward is the telescoping difference `Phi(prefix_k)-Phi(prefix_{k-1})`. Inputs include round score, accumulated scores, dealer/repeat-dealer state, and riichi bets. | Figure 8 moves from approximately stable rank `8.06` for raw-round-score RL to `8.24` after GRP, but this is a sequential system comparison, not a factorial ablation. Predictor error and variance reduction are not reported. The paper explicitly leaves initial-hand/luck adjustment using perfect information as future work. |
| Oracle guiding | Normal observations are augmented with all three opponents' hands and remaining wall. Each privileged feature is Bernoulli-masked with keep probability `gamma_t`, annealed from 1 to 0. After `gamma_t` reaches zero, training continues at one-tenth learning rate and rejects samples over an unpublished importance-weight threshold. | Figure 8 moves approximately `8.24 -> 8.31`; exact schedule, cutoff, and independent-seed uncertainty are unpublished. Direct oracle-to-public distillation reportedly failed without numbers. |
| pMCPA runtime adaptation | At each round start, fix the legal own hand, sample opponents/wall, roll out under the offline policy, importance-weight policy updates, play the adapted round, then reset. | 100k adaptation and 10k test trajectories per hand over only “hundreds” of hands; adapted versus unadapted win `66%`, with no CI. It was too slow and was not part of online Suphx. |
| Full online system | RL-2 trained about 2.5M games; the deployed system omitted pMCPA. | Stable rank `8.74` over 5,760 games. The result does not isolate GRP or oracle guiding. |

Each reported RL variant used 1.5M training games, 44 GPUs for two days, and a
one-million-game evaluation against three SL-weak opponents. The displayed
uncertainty came from repeated 800k-game subsamples of that evaluation, not
independent training runs. The approximate GRP and oracle increments should
therefore be treated as suggestive, not causal estimates.

### Stronger later evidence

| Source | Finding | Cascadia relevance |
|---|---|---|
| [Variational Oracle Guiding (VLOG), ICLR 2022](https://openreview.net/forum?id=pjqqxepwoMy) and [pinned official code](https://github.com/Agony5757/mahjong/tree/db1e72e792fbae175bc9c3abb38f7dd92ee832b0) | Learns a public prior and privileged posterior over a latent representation with a shared Q/value decoder. In 100k four-player games per comparison, VLOG versus the trained baseline had average payoff `233+-13` and match win `55.7+-0.4%`; Suphx-style dropout had `59+-13`, `51.2+-0.4%`; VLOG without oracle input had `61+-13`, `52.0+-0.4%`. | Privileged information can help a value/Q learner, but raw feature dropout is not the best-established method and some gain comes from latent regularization. VLOG is not itself an actor-critic experiment. |
| [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) | Used opponent observations only in the training-time value function; the legal policy did not need them at evaluation. A simplified fixed-opponent best-response study reported `82%` win with opponent information versus `22%` without. | Supports a privileged **critic**, not hidden-information policy/Q labels at serving. StarCraft's hidden state is strategic rather than an exogenous future shuffle. |
| [Pluribus](https://doi.org/10.1126/science.aay2400) and [AIVAT](https://ojs.aaai.org/index.php/AAAI/article/view/11481) | Six-player poker used self-play plus depth-limited imperfect-information search, not oracle guiding or a luck-corrected training reward. AIVAT is an unbiased evaluation control variate; the AIVAT paper reports about 85% SD reduction, equivalent to roughly 44x fewer games in its poker tests. | If luck correction is wanted, evaluation variance reduction is better grounded than changing the target. |
| [Posterior Value Functions](https://proceedings.mlr.press/v139/nota21a.html) | Hindsight-inferred chance/hidden variables define an unbiased policy-gradient baseline whose variance is no greater than an ordinary state-value baseline. | A gradient control variate, not a replacement score/Q label. |
| [DouZero](https://proceedings.mlr.press/v139/zha21a.html) | Strong three-player imperfect-information play from parallel deep Monte Carlo Q learning on terminal rewards, without GRP or oracle guiding. | Negative evidence against treating Suphx's mechanisms as necessary. |

The open-source [Mortal descendant implementation](https://github.com/Equim-chan/Mortal/tree/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2)
also predicts final-rank permutations and differences between successive
expected rank points across rounds. It began as a Suphx reproduction and
therefore illustrates GRP's *role* as rank-credit shaping; it is not independent
confirmation and supplies no isolated causal ablation.

### Applicability and failure modes

- **Repo fact:** the true-hidden-order oracle already lost to honest
  multi-world search; see [RESEARCH_LOG §3.2](docs/v3/RESEARCH_LOG.md).
- **Repo fact:** the future tile/bag order is not a legal model input; see
  [ARCHITECTURE](docs/v3/ARCHITECTURE.md).
- **Repo fact:** serving depends on policy and completed-Q, while the trajectory
  outcome/value vector is auxiliary; see
  [R1.4 design](docs/v3/R1_4_DENSIFICATION_DESIGN.md).
- **Inference:** a realized future order is exogenous and generally not
  inferable from public history. A one-realization oracle label adds
  realization-specific noise and can encourage spurious finite-data
  correlations rather than estimating the public conditional action value.
- **Failure mode:** subtracting “luck” from a score target can change the
  expected-score objective unless the correction has zero **conditional** mean
  given the legal public state/action and is incorporated so expected action
  values and rankings remain unchanged.

### Gated decision

- **Reject a Suphx GRP arm.** Cascadia has no multi-round rank-credit mismatch.
- **Reject a “luck-corrected outcome” target.** Use paired/CUPED/AIVAT-like
  evaluation controls where legally constructible; do not silently redefine
  reward.
- **Reject Suphx oracle-feature dropout and pMCPA.** The direct bottleneck and
  runtime stack make them poor fits.
- **Optional later falsifier only:** train a public-prior/privileged-posterior
  *critic*. For every public root/action, average the privileged critic over
  hidden orders drawn from the engine's exact conditional chance distribution;
  only the resulting Monte Carlo estimate of public expected value may
  supervise Q. Require held-out target RMSE/variance improvement and puzzle-bank
  decision movement before a game gate. Run this only after D1.

**Confidence:** high on the Suphx mechanism and negative decision; moderate on
the transfer value of VLOG because its Mahjong experiment is offline and the
hidden state is strategically informative.

## 5. Q3 — Quantitative recipe for targeted relabeling / Reanalyze

### Short answer

The literature strongly supports refreshing stale search targets with the
current teacher, replacing rather than interpolating contradictory search
policies, retaining a soft search distribution, favoring recent/on-distribution
states, and controlling replay bias. It does **not** publish a transferable
optimum for D1's hard fraction, teacher budget, repeats, phase mix, or fold
weight.

No located primary study holds teacher compute fixed and compares uncertain-
state MCTS relabeling head-to-head with uniform relabeling. D1's mechanism is
externally supported; its exact numbers must be justified by the repo's direct
measurements and a preregistered dose curve.

### Best available quantitative evidence

| Source | Exact design/result | What transfers—and what does not |
|---|---|---|
| [MuZero](https://arxiv.org/abs/1911.08265), Appendix H | The published Reanalyze result was Atari: MCTS with current parameters revisited old timesteps and supplied `80%` of policy updates, with no separately reported higher reanalysis budget. Base search used 50 simulations in Atari and 800 in board games, but no board-game Reanalyze result was reported. Other Atari changes were confounded: samples/state `0.1 -> 2.0`, value-loss weight `1 -> 0.25`, TD horizon `10 -> 5`. | Supports current-teacher policy replacement and frequent refresh. `80%` is not a D1 fold-weight recipe. |
| [MuZero Reanalyse / MuZero Unplugged](https://arxiv.org/abs/2104.06294) | Samples from the `N` most recent interactions. At fixed total learner/search compute: 50% reanalysis with 2,000M frames scored median/mean `1331.7/4094.4%`; 95% with 200M `1006.4/2856.2%`; 99.5% with 20M `126.6/450.6%`. | Demonstrates a compute/data tradeoff and recent-state refresh, not an optimal fraction or hard-state result. Authors warn reward-sorted replay may bias stochastic tasks. |
| [EfficientZero](https://arxiv.org/abs/2111.00210) and [pinned official code](https://github.com/YeWR/EfficientZero/tree/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e) | Reanalyzed policy ratio `0.99`, value ratio `1.00`, 50 simulations for both collection and reanalysis; dynamic TD horizon and fresh root value correct stale trajectory targets. Removing off-policy correction reduced Atari mean/median normalized score `1.943/1.090 -> 1.475/0.836`. | Strong warning against multiplying stale behavior-trajectory value labels. Ratios do not transfer to a targeted shard. |
| [Expert Iteration](https://arxiv.org/abs/1705.08439) | Initial 100k-state construction used 1,000-simulation exploration and 10,000-simulation labels; later DAgger batches used the apprentice to visit states and 10k MCTS to label them. One state was sampled per game. Soft tree-policy versus chosen-action targets had top-1 errors `47.7%` versus `47.0%`, yet the soft-policy apprentice was `50+-13` Elo stronger. | A 10x precedent and direct support for low within-game correlation/soft targets. Teacher budget was not ablated, so the paper does not show that 10x caused the gain. |
| [KataGo](https://arxiv.org/abs/1902.10565) | Mixed 600/100 then 1,000/200 full/fast searches; full-search turns sampled with probability `.25` and only those were stored. Removing playout-cap randomization reduced final reported Elo `1329 -> 1242`. | Supports mixed search grades and broad recent coverage, not relabeling. |
| [KataGo policy-surprise weighting](https://github.com/lightvector/KataGo/blob/4c1a18216b15aaf7990dc8723a67641c4768dd72/docs/KataGoMethods.md#policy-surprise-weighting) | Roughly half the sample mass remains uniform and half is allocated by root-prior/search-policy KL. The author reports a large practical improvement but no isolated number. | Supports a uniform coverage floor plus surprise emphasis; not an exact D1 fraction. |
| [ReZero](https://arxiv.org/abs/2404.16364) and [pinned LightZero code](https://github.com/opendilab/LightZero/tree/676df2d9e454838e7f2118a8c3421cf8526cc3b4) | Periodic whole-buffer sweeps; frequency 0, 1/3, 1, 2 ablated in Ms. Pac-Man. With 50 simulations and reanalysis ratio 1, ReZero used 2–4x less wall time on many tasks while retaining comparable returns. Weighted-subset selection is explicitly future work. | Supports periodic decoupled refresh. It contains no hard-root recipe. |
| [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | Standard proportional setup `alpha=.6`, importance correction `beta=.4 -> 1`; warns that noise, stochastic outliers, stale priorities, and lost diversity can cause overconcentration. | Supports stochastic stratification and a coverage floor. Gradient replay priority is not proof of where teacher compute should go. |
| [Go-Exploit](https://arxiv.org/abs/2302.12359) | Uniformly restarts self-play from archived visited/search states and improves sample efficiency in Connect Four and 9x9 Go. Weighted sampling and stochastic/imperfect-information games are left to future work. | Supports revisit/recent coverage, not uncertainty-targeted labels. |

### Gated decision

Attempt 3's reboot failure and zero-artifact outcome are now durably logged.
Proceed with the existing Stage A hypothesis only after the rules-contract/new-
ID repair and, with John's restart permission, a successful rerun. Use
[§15](#15-d1-prescription) as **proposed
preregistration amendments**. The July 15 record already fixed the 15k cap,
opening/mid emphasis, n2048/d16 x2 without ghosts, K=8 distq, `>=0.010` bank
bar, fresh sequential CUPED n256 gate, and final-n256-null close rule. New
sampling, aggregation, masking, fold, dose, look, and standing-stage details
below must be frozen at the appropriate pre-data boundary. The key decisions
are:

1. use the 15,000 hard-root tranche, phase-stratified and spread across games;
2. keep n2048/d16 with two independent repeats;
3. aggregate repeat Q and visit statistics into one soft search target;
4. replace cheap policy/Q/search fields at selected roots in a provenance-
   pinned training view; retain the immutable raw corpus;
5. do not duplicate the old behavior trajectory's realized value target in the
   D1 row;
6. use D1 raw weight 1 on the existing `4,2,1` scale, yielding 12.5% of draws
   and four expected passes per D1 root;
7. make reanalysis a standing stage only after a positive game gate and a
   second fresh-cycle replication.

**Confidence:** high that reanalysis and soft target replacement are sound;
high that no published hard-versus-uniform numerical optimum was found;
moderate in the proposed Cascadia dose because it is intentionally a
preregistered design judgment rather than borrowed authority.

## 6. Q4 — Best-arm identification under a fixed noisy-comparison budget

### Short answer

Sequential halving remains the right production baseline. Fixed-budget
best-arm identification (BAI) supplies useful alternatives, but none has a
finite-budget optimality result for Cascadia's combination of:

- correlated returns from shared determinizations;
- heteroscedastic outcomes;
- biased/noisy learned leaf and bootstrap returns;
- Gumbel/prior offsets in the ranking score;
- batched bridge execution; and
- a hard wall-clock budget rather than one-sample-at-a-time access.

The repo's closed R0.4 result changed only final selection with an LCB. It did
**not** test a variance-adaptive allocator and must not be cited as one.

### Best available evidence

| Method/source | Guarantee or empirical result | Fit to Cascadia |
|---|---|---|
| [Sequential Halving](https://proceedings.mlr.press/v28/karnin13.html) | For independent bounded rewards, `P(error) <= 3 log2(K) exp[-T/(8 H2 log2(K))]`; the paper establishes near-optimality over a broad instance family, not equality with the general `H` lower bound. | Strong incumbent for about 16 arms and a fixed budget; naturally stage-batched. Its independence/stationarity assumptions do not model shared worlds or learned-value bias. |
| [Successive Rejects](https://www.learningtheory.org/colt2010/papers/59Audibert.pdf) | Fixed-budget and parameter-free with respect to unknown instance hardness; allocates increasingly many pulls as arms are eliminated. Its bounded-reward error exponent still carries a logarithmic-in-`K` factor. | A credible comparator, but its many elimination rounds and uneven allocation may batch less cleanly. |
| [LUCB](https://icml.cc/2012/papers/359.pdf) | Fixed-confidence: samples the empirical leader and highest-confidence challenger until their intervals separate. | Useful top-two intuition, but truncating it at a hard budget discards its stopping guarantee. Repo R0.4 was not LUCB. |
| [lil'UCB](https://proceedings.mlr.press/v35/jamieson14.html) | Fixed-confidence: samples the arm with the largest law-of-the-iterated-logarithm upper bound and uses a pull-count stopping rule. | Not the same two-arm procedure as LUCB; hard-budget conversion again loses its guarantee. |
| [Top-two probability sampling](https://arxiv.org/abs/1602.08448) | For fixed `beta`, obtains the best asymptotic exponent among allocations assigning fraction `beta` to the true best arm. Global optimality needs the instance-dependent `beta*` or consistent adaptation; `beta=1/2` is generally a factor-two guarantee. | Suggests top-two concentration, but requires a defensible posterior and gives no finite-budget guarantee for biased search values. |
| [SHVar/SHAdaVar](https://proceedings.mlr.press/v216/lalitha23a.html) | Heteroscedastic sequential-halving variants: SHVar assumes known variances; SHAdaVar estimates them. The analysis uses i.i.d. Gaussian arm observations. | Closest bounded candidate for a bakeoff, but shared-world covariance, bounded/discrete returns, and model bias lie outside its guarantee. |
| [OCBA under correlated sampling](https://informs-sim.org/wsc04papers/072.pdf), extending [classic OCBA](https://doi.org/10.1016/S1569-190X%2802%2900095-3) | Allocates a fixed simulation budget to maximize an approximation to probability of correct selection using estimated means, variances, and—in the correlated extension—correlations with the apparent best design. The correlated paper gives an exact two-design allocation and an approximation for more than two under jointly normal samples. | Directly names the simulation-optimization analogue in the question, but its plug-in apparent-best estimates can be brittle at Cascadia's small per-root budget; learned-value bias, changing survivor normalization, and bridge batching remain outside the model. Keep it out of the first bakeoff unless replay tapes show stable mean/covariance estimation. |
| [Fixed-budget lower bounds](https://proceedings.mlr.press/v49/carpentier16.html) and [open problem on instance-optimal fixed-budget BAI](https://proceedings.mlr.press/v178/open-problem-qin22a.html) | Characterize limits and explicitly document that generally optimal fixed-budget instance adaptation remains unresolved. | Prevents claiming any published allocator is universally correct here. |
| [Ranking-and-selection with common random numbers](https://arxiv.org/abs/1410.6782) | Studies selection procedures under CRN-dependent, jointly multivariate-normal observations. Positive covariance can lower variance of differences; negative covariance can hurt. | Confirms that marginal arm variance is insufficient while leaving a distributional-assumption gap to Cascadia. |

### Applicability and failure modes

The chosen root action emerges during sequential halving from
`gumbel + logit + sigma(normalize(current survivor mean values))`; normalization
is recomputed on the survivor set. Full-menu `completed_q` and the improved-
policy target are constructed only after selection. An allocator can reduce
Monte Carlo variance while preserving or amplifying biased/noisy learned leaf
and bootstrap values. With shared worlds, pairwise precision depends on
`Var(Xa-Xb)=Var(Xa)+Var(Xb)-2*Cov(Xa,Xb)`, so marginal variance is insufficient.

Sequential halving also maps cleanly to the model bridge: evaluate a batch of
survivors, eliminate, and repeat. LUCB/top-two methods may require fine-grained
adaptive synchronization that loses enough throughput to erase statistical
gains. Any comparison must therefore be at equal wall time as well as equal
nominal simulations.

### Gated decision

**Retain sequential halving now.** If and when D1 produces a model that
materially changes the Q regime, a bounded frozen-root allocator bakeoff is
justified:

1. freeze a preregistered calibration/holdout split of replayable roots;
2. compare incumbent sequential halving, successive rejects, and one
   variance-adaptive sequential-halving implementation;
3. make equal measured wall time the primary budget; retain equal nominal
   simulations only as a secondary diagnostic;
4. preserve CRN in a replayable action x world x replicate outcome tape;
5. on an independent split, freeze the reference action pair, reference
   budget/repeats, qtransform coefficient, and tie/ambiguous-best handling; do
   not import the high-budget `max_visits` coefficient into the production-
   budget target;
6. primary endpoints: wrong-best rate and simple regret against that reference;
   secondary endpoints: top-two difference variance, bridge utilization, and
   nominal simulations;
7. require a material holdout reduction (recommended bar: at least 20% in
   simple regret or wrong-best decisions) with no measured-wall regression
   before spending a paired game gate.

That `20%` is a campaign decision bar, not a published constant. Do not run the
bakeoff ahead of D1: the teacher/relabel path has much stronger direct evidence.

**Confidence:** high that sequential halving should remain the baseline;
moderate that variance-adaptive halving is the best challenger; low that any
standard BAI theorem predicts the in-game result.

## 7. Q5 — Gumbel AlphaZero constants under noisy Q

### Short answer

Published Gumbel AlphaZero work gives domain constants, not a validated
root-variance rule for stochastic score games. The selected repo candidate
(`c_scale=.25`, top-k:8) failed its disjoint confirmation, closing that
preregistered static sweep—not every possible lower-`c_scale` rule.

Keep `c_visit=50`, `c_scale=1.0`, and min-max normalization. **Do not launch L3
from literature alone.** First estimate how much low-budget completed-Q should
be trusted as a function of measured root noise; only then preregister a
reliability-shrink rule.

### Best available evidence

| Source | Exact evidence | Interpretation |
|---|---|---|
| [Gumbel AlphaZero, ICLR 2022](https://openreview.net/forum?id=bERaNdoegnO) | Go/chess experiments use `c_visit=50`, `c_scale=1.0`. Atari uses min-max-normalized Q with `c_visit=50`, `c_scale=0.1`. The Go sensitivity plot is broadly stable for `c_visit >= 50`; a Beam Rider `c_scale` sweep spans `.01,.1,1,10,100` over 10 seeds. | Values are domain choices, not a stochastic-board-game calibration law. Exact policy-improvement theory assumes correct Q; the paper separately studies stochastic bandits with empirical means and suggests normalization/clipping for noisy Q, without the same finite-sample guarantee. |
| [Pinned DeepMind MCTX qtransform](https://github.com/google-deepmind/mctx/blob/450fbf7656b88dd1d8ca5b2db3a2f9464cb322f2/mctx/_src/qtransforms.py) | Official sequential-halving qtransform defaults at the pinned revision include `value_scale=0.1`, `maxvisit_init=50`, and rescaling. | Confirms those implementation defaults at the cutoff; it does not establish that Cascadia's current score normalization should copy Atari's scale. |
| [Gumbel MuZero for 2048, TAAI 2022](https://scholar.nycu.edu.tw/en/publications/gumbel-muzero-for-the-game-of-2048/) and the authors' [full dissertation account](https://arxiv.org/abs/2212.11087) | The stochastic 2048 system combined Gumbel and Stochastic MuZero. The fuller account tested sampled actions `m in {2,3,4}` and training simulations `n in {m,16,50}`; `n=m` was best for every `m`. The `m=n=3` model averaged `359,721` when served at three simulations and `394,645` at 50, over 100 test games; no CI or independent-training-seed count is reported. | Direct stochastic-domain evidence that *more training search was not always better* and that “394,645 with three simulations” refers to training, not serving. It is a simulation-budget result, not a verified `c_visit`/`c_scale` sensitivity study. |
| [Stochastic versus deterministic EWN, TAAI 2023](https://scholar.nycu.edu.tw/en/publications/an-empirical-analysis-of-gumbel-muzero-on-stochastic-and-determin/) | Stochastic/fixed-die variants compare `m=6,n=6` against `n=50`; no-die variants use `m=18,n=18` against `n=50`. The low-budget arm won `55%` against high-budget in stochastic EWN, versus `38%`, `45%`, `40%`, and `0%` across four deterministic variants. Random-board estimates use both seat orders over 720 symmetric starts (`1,440` games); fixed-board estimates use 100 alternating-first games. | Stronger evidence that search budget and stochastic training can interact counterintuitively; still no isolated qtransform-constant ablation or reusable `c_scale(noise)` law. The cross-variant association is not a general proof of causation. |
| [Pinned KataGo methods](https://github.com/lightvector/KataGo/blob/4c1a18216b15aaf7990dc8723a67641c4768dd72/docs/KataGoMethods.md) | KataGo scales exploration using empirical utility variance plus a prior and reports substantial combined gains when its dynamic exploration/uncertainty machinery was introduced. The documentation does not isolate a Gumbel-compatible sigma rule. | Evidence that uncertainty-aware exploration can matter, but KataGo's PUCT coefficient is not the same object as Gumbel's completed-Q transform. |
| [Repo R0.1 verdict](docs/v3/RESEARCH_LOG.md) | n256 screen favored every lower static scale, but fresh 100-seed confirmation for `c_scale=.25`, top-k normalization was `-0.2325`, CI `[-0.5440,+0.0790]`. | Direct evidence outranks the suggestive screen and closes another static sweep in this regime. |

The directly relevant 2048/EWN results make the negative answer more precise:
stochasticity can reverse the usual “more training simulations helps” intuition,
but neither accessible peer-reviewed account supplies the qtransform tuning
rule the question asks about. PUCT tuning papers likewise cannot be imported as
though PUCT and Gumbel qtransforms were identical.

One absence caveat matters. The catalog for Chih-Yu Kao's [2022 master's
thesis](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22110NYCU5394144%22.&searchmode=basic)
lists a section titled “Parameters in the Monotonically Increasing
Transformation,” but the relevant full-text page is CAPTCHA/license-gated.
The accessible peer-reviewed metadata and Hung Guei's full reproduced 2048
account contain no named `c_visit`/`c_scale` grid. This report therefore claims
that no **verifiable transferable stochastic-domain rule** was found; it does
not pretend the gated thesis page was inspected or that no private tuning
result exists.

### A defensible form for any future L3

High sampling variance should usually make the system trust Q **less**, not
automatically multiply Q's influence. A calibration-first rule would be:

1. On a frozen calibration bank, compute low-budget centered action values
   `q_low(a)-mean(q_low)` and an independent high-budget reference.
2. Bin or smoothly regress by root SNR/pairwise SE and estimate a reliability
   slope `rho(r)` between low- and high-budget centered values.
3. Clip the preregistered slope to `[0,1]` and apply it **after** Q
   normalization at every sequential-halving phase:
   `sigma_cal(a)=rho(r)*(c_visit+maxN)*c_scale*minmax(q)_a`, equivalently
   `c_scale_eff=rho(r)*c_scale`. A positive affine shrink before min-max would
   cancel exactly and cannot change decisions.
4. Freeze the fit, evaluate on a disjoint root bank, and compare at equal wall
   time. Only a holdout decision improvement may unlock a game gate.

This is an inference from measurement-error calibration, not a published
Gumbel formula. It also overlaps conceptually with the closed R0.3 Q-bias
class, so it should proceed only if the post-D1 Q/noise relationship supplies
materially new evidence. K-head spread must not be substituted for `rho`: the
current heads are not calibrated as either return-risk or model-error measures.

### Gated decision

- retain the existing constants and normalization;
- mark L3 **deferred pending offline calibration**, not funded;
- do not repeat the closed preregistered static scale/normalization sweep;
- reopen only after D1 or another model change alters the measured Q-noise
  regime and a disjoint root-bank calibration predicts a monotone benefit.

**Confidence:** high on retaining defaults now; high that no verifiable
stochastic variance-to-scale recipe was found in the accessible primary
sources; moderate on completeness because the thesis parameter section is
gated; moderate on reliability shrink as the best falsifiable form.

## 8. Q6 — Value targets for multiplayer score-maximization games

### Short answer

The cited multiplayer systems predict the utility that their policy and search
actually optimize: rank reward, win probability, or a vector of player
utilities. The literature does **not** show that margin, rank, or table total
beats own absolute score when the evaluation objective is own expected score.

Cascadia already trains `value_vector`, a state-level four-seat final-score
head. Residualizing that head would not test R1.1c because it cannot rank root
actions. If John authorizes R1.1c, the new intervention must generate
**table-native per-action Q and improved-policy targets** under cooperative
table-sum continuation. A four-seat per-action action-value vector preserves
the most information, but its sum—not the existing root value vector—is the
served cooperative utility.

### Best available evidence

| System/source | Training value and search use | Evidence limit |
|---|---|---|
| [Multiplayer AlphaZero](https://arxiv.org/abs/1910.13012) | Terminal return is a vector `z` of each player's utility. The value head predicts vector `v`; MCTS backs up the component for the player to act; training uses `(state, search policy, z)`. | Demonstrated on simple three-player games with `-1/0/+1` utilities, not stochastic score maximization. |
| [Deep Catan](https://www.lamsade.dauphine.fr/~cazenave/papers/DeepCatan.pdf) | Four-way softmax of winning probabilities estimated from root MCTS outcomes; search uses win-oriented values. A reported 400-game evaluation gives 58% team win for a later network in 2v2, while a local-value UCTNet comparison reports 60%. | Small, partially team-structured experiment; trading was omitted and network inputs appear to expose all players' resource/development cards. It is target-shape evidence, not a clean public-information Catan precedent. |
| [BlokusZero](https://ipsj.ixsq.nii.ac.jp/record/204042/files/IPSJ-GI20043004.pdf) | Although four-player Blokus is score-based, the modified AlphaZero target is a four-component terminal vector: each seat gets `+1` for a win and `-1` for a loss. MAXN search stores/backs up every player's component and selects the component for the player to act. | Direct four-player evidence for vector backup, but it discards score magnitude and reports no score-versus-rank/win target ablation. |
| [Suphx](https://arxiv.org/abs/2003.13590) | Final rank-based match reward, redistributed across Mahjong rounds by GRP. | Optimizes placement/rank, not raw tile points or table total. |
| [LOCM drafting via RL](https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2020.pdf) | PPO drafting episodes end with one downstream battle by a fixed battle agent; reward is `+1/0/-1` for win/draw/loss, discount `1`, and the value function estimates that expected terminal reward. | A genuine drafting application, but still a two-player win target. It does not compare own score, score margin, rank, or table-vector targets. |
| [Score versus win targets in Go](https://arxiv.org/abs/2201.13176) | A score-trained 9x9 Go agent lost 14–1 to a strong human. Against a 32-network calibration panel, it converged roughly 1,500 Elo below a comparably trained outcome-target SAI run, interpreted by the authors as less than a 1-in-5,000 win probability. | Strong warning evidence, but still two-player zero-sum Go rather than multiplayer expected-score play. |

[Pluribus](https://doi.org/10.1126/science.aay2400) and
[DouZero](https://proceedings.mlr.press/v139/zha21a.html) likewise optimize
competitive poker/DouDizhu utilities, not cooperative table totals. No
credible head-to-head study was found where a margin/rank target beats
own-score training under a metric defined as own mean score in a four-player
non-zero-sum game. The located Blokus and drafting systems therefore do not
fill the requested ablation gap: both reduce their scored downstream game to a
win/loss utility before learning.

### Exact Cascadia target if R1.1c is authorized

For public state `s`, legal action `a`, final scores `F_j`, and a continuation
policy that is itself trained/searching for table sum, the informative target
is:

```text
q_j(s,a) = E[F_j | public s, a, cooperative continuation]
           - exact_score_j(after(s,a)),  j in {0,1,2,3}

Q_table(s,a) = sum_j(exact_score_j(after(s,a)) + q_j(s,a))
```

The scalar equivalent is:

```text
q_table(s,a) = E[sum_j F_j | public s, a, cooperative continuation]
               - sum_j exact_score_j(after(s,a))
```

The improved-policy target must be generated from `Q_table`, with table utility
used at root and interior plies. Merely summing the existing state value vector
at a leaf does not create table-native action supervision. Prefer the per-action
four-vector because it preserves seat attribution and permits both own-seat and
table diagnostics; the scalar target is scientifically equivalent for the
table objective and is simpler if exporter cost dominates.

Keep own-seat per-action Q as a retention auxiliary in the first arm. The
table-derived improved policy is the load-bearing new label. Target/loss weights
have no published optimum and must be frozen on an offline instrument, not
selected from validation loss alone.

Mathematically, table total divided by four equals the formal self-play mean-
seat scoreboard. Methodologically, optimizing table total allows the active
seat to sacrifice its own score to improve other seats and therefore changes
the game from selfish competitive Cascadia to a cooperative four-seat planner.
It may be valid for the internal benchmark if John explicitly adopts that
objective; it cannot be mixed into a human-superhuman claim without saying so.

### Gated decision

- **Do not launch R1.1c before John's methodology ruling.**
- If authorized, generate table-native per-action Q **and** improved-policy
  labels; preserve the four-seat action-value decomposition if practical and
  derive the table scalar by summation.
- Do not substitute rank/margin without changing the stated metric.
- Treat the earlier failures precisely: they served untrained table totals with
  selfish interior plies and increased terminal/rollout variance. They are
  negative evidence for that serving shortcut, not a test of table-native
  trained action Q/policy labels.

**Confidence:** high on objective alignment and the need for per-action labels;
moderate that a vector is preferable to the scalar equivalent; low that
cooperative table serving will improve the current score without a new
target-trained model and direct gate.

## 9. Q7 — Variance reduction across determinizations

### Short answer

The requested “determinization-level pairing” is already the incumbent. In
[gumbel.rs](cascadiav3/real-root-exporter/src/gumbel.rs), every action's visit
index maps to `det_index = visit_index % d`, and the determinization seed does
not contain the action index. Thus corresponding visits across surviving
actions use the same synthetic hidden world. The rollout RNG still contains
the action index; pairing that stream was R0.2 and failed.

The remaining open idea is stratified/antithetic/quasi-random generation of the
`d` legal hidden permutations. Published imperfect-information search work
does not provide a plug-in method or a reliable effect size for this setting.

### Best available evidence

| Source | Result | Transfer |
|---|---|---|
| [Veness, Lanctot, and Bowling, “Variance Reduction in Monte-Carlo Tree Search”](https://proceedings.neurips.cc/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html) | Applies CRN recursively to action differences. `Var(g-h)` falls by `2*Cov(g,h)` when covariance is positive and can rise when negative. The best tested variance-reduction approach/combination was roughly equivalent to 50–60% more simulations in Can't Stop and 25–40% in Dominion, with Pig also improved; those figures are not an isolated CRN effect. | Direct support for sharing randomness across candidates, but not proof of positive covariance for Cascadia's rollout stream. |
| [Information Set MCTS](https://edpowley.com/academic/papers/tciaig_ismcts.pdf) | Samples determinizations into a shared information-set tree and documents strategy fusion, nonlocality, and duplicated-budget failures of naive determinization/PIMC. | Warns that lower variance does not cure determinization bias. Cascadia's shared-world root comparison is narrower than full ISMCTS. |
| [Ensemble Determinization in Magic](https://eprints.whiterose.ac.uk/id/eprint/75050/1/EnsDetMagic.pdf) | At a fixed 10k simulation budget, around 40 determinizations x250 simulations was competitive; at 100k, the best balance shifted toward more determinizations. | Supports a budget-dependent worlds-versus-depth tradeoff, not a universal `d` or stratification scheme. |
| [Repo search description](docs/v3/RESEARCH_LOG.md) | Explicitly records `d` worlds cycled across each action with common random numbers. | Current source of truth; corrects the question premise. |
| [Repo R0.2 verdict](docs/v3/RESEARCH_LOG.md) | Pairing the rollout stream changed pooled gap variance `0.020538 -> 0.021438`: a **4.4% increase**, equivalently `-4.4%` reduction, versus the preregistered `>=20%` reduction floor. | Directly closes rollout-level CRN as a standalone fix, not world pairing. |

### Applicability and failure modes

Stratifying permutations can silently bias expectation if strata do not carry
correct combinatorial weights. “Antithetic” hidden orders also require a
defined negative-dependence construction; reversing a bag is not automatically
antithetic for downstream score. Quasi-Monte Carlo theory over continuous
hypercubes does not transfer automatically to constrained multisets and a
discontinuous policy/search map.

Any experiment must preserve legality, pairing across actions, and the engine's
exact conditional hidden-order target—either through target-distributed draws
or predeclared importance weights. Freeze the compared action pair on a
selection split independent of the reference/evaluation split, and pin the
high-budget reference's budget and repeats. Primary endpoints are variance and
bias of top-two **differences**, not marginal Q variance.

### Gated decision

- close “replace R0.2 with determinization pairing” because that replacement
  is already implemented;
- do not queue a stratification arm ahead of D1;
- if revisited, compare incumbent target-distributed paired worlds with a
  formally weighted stratified sampler at equal **measured wall time**;
- preregister at least 20% reduction in top-two difference variance, lower
  decision regret, and a bias-difference 95% CI wholly inside a fixed practical-
  equivalence margin before a game gate.

**Confidence:** high on the implementation correction and negative scheduling
decision; moderate on the general value of stratification; low on any specific
permutation construction until it is mathematically specified.

## 10. Q8 — Hex-symmetry data augmentation

### Short answer

Rotation/reflection augmentation is standard in Go systems, but the primary
sources located for this review do not isolate a reliable final-strength
multiplier for training-time augmentation, and the Hex literature does not
supply the requested gameplay ablation. Cascadia's three-rotation inference
TTA was flat at roughly three times model-evaluation cost. That directly
rejects the tested inference ensemble; it is only indirect evidence about
orientation invariance and does not measure reflection or training-time
augmentation.

**Do not prioritize a standalone trainer-augmentation arm now.** Inference TTA
is closed; random D6 trainer augmentation remains unmeasured and is eligible
only as a cost-neutral shadow arm in an already-funded retrain.

### Best available evidence

| Source | Measured evidence | Limitation |
|---|---|---|
| [AlphaGo](https://www.nature.com/articles/nature16961) and [Extended Data Table 3](https://www.nature.com/articles/nature16961/tables/3) | Training examples were expanded through all eight board rotations/reflections. The inference comparison reports raw policy accuracy about `55.9% -> 57.0%` for an eight-symmetry ensemble, while evaluation latency rose roughly `7.1 -> 55.3 ms`. | No no-augmentation training arm. Inference ensembling is not free data augmentation and loses search throughput. |
| [AlphaGo Zero](https://www.nature.com/articles/nature24270) | Uses random rotations/reflections during self-play training. | Again standard practice, not an isolated augmentation effect. |
| [Expert Iteration on Hex](https://arxiv.org/abs/1705.08439) | Strong Hex apprentice/expert results and DAgger-style data growth. | Does not report a symmetry-augmentation ablation. |
| [HexaConv](https://arxiv.org/abs/1803.02108) | Constructs convolution/equivariance machinery for hexagonal lattices and improves hex-grid vision tasks. | Not a board-game decision or playing-strength experiment. |
| [Repo TTA verdict](docs/v3/RESEARCH_LOG.md) | Three-rotation inference TTA scored `96.91` versus `96.95`, delta about `-0.04` with CI spanning zero, at 3x model evaluation; cost-matched extra worlds were numerically better and cheaper, without a demonstrated significant advantage. | Directly closes that inference ensemble; it is only indirect evidence about orientation invariance and says nothing direct about reflection or trainer augmentation. |

### Gated decision

- do not prioritize a standalone trainer augmentation experiment;
- preserve and expand the existing rotation-equivariance tests; implement
  reflection transforms and exact legality/action/scoring/target-remapping
  tests before any D6 shadow arm;
- if a future already-funded retrain changes architecture or data regime, a
  preregistered shadow arm may add random D6 transforms while holding unique
  base states, optimizer steps, and wall time fixed;
- require decision-bank movement, not lower augmented validation loss.

**Confidence:** high that inference TTA is closed under the tested
configuration; moderate in deprioritizing a dedicated trainer arm; low on any
universal “2–5x effective data” claim because no qualifying measured source
supports it.

## 11. Q9 — KataGo-style adversarial blind-spot discovery

### Short answer

Adversarial probing is worth a bounded diagnostic arm. KataGo's case shows
that a policy can be superhuman in ordinary play yet catastrophically weak on
adversarially constructed states, but it does not establish that those states
occur naturally or materially affect Cascadia mean score. Iterated defenses
consumed substantial compute and still failed to generalize to new attack
families.

The right Cascadia first step is a diverse, legality-checked **probe bank**, not
an adversarial opponent-training program. A probe must transfer across model
seeds/checkpoints and be confirmed by an independent high-budget teacher before
it is admitted as a real decision weakness.

### Best available evidence

| Source | Exact result | Lesson |
|---|---|---|
| [Wang et al., “Adversarial Policies Beat Superhuman Go AIs,” ICML 2023](https://proceedings.mlr.press/v202/wang23g.html) | The cyclic adversary used less than 14% of KataGo's training compute. It won `95.7%` of 1,052 games against a defended victim at 4,096 visits, and still won `82%` of 50 at one million visits and `72%` of 50 at ten million visits. Despite its exploit success, it was not generally strong: a human amateur could beat it. | More search does not necessarily fix an out-of-distribution strategic blind spot; attack success need not imply general game strength. |
| [“Can Go AIs Be Adversarially Robust?”, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34980) | The paper separately evaluated nine rounds of iterated adversarial training and a transformer-based victim architecture; newly trained attacks persisted in both settings. Against the ninth adversarially trained victim, the final cyclic adversary won `42%` at 65,536 visits using about `26%` of the victim's training compute. A fresh atari-style adversary reached `81%` at 512 visits with under 5% victim compute but fell to `4%` at 4,096 after a long attack-training run. | Patch success must be tested on held-out attack families and multiple search budgets. Architecture change alone is not immunity. |
| [Repo bag-bug verdict](docs/v3/RESEARCH_LOG.md) | A rare deep ghost trajectory exposed the consecutive-four-of-a-kind transient bag drain, then a deterministic regression reproduced it. | Rare-state generation is already valuable for rules correctness, independently of strength. |

### Cascadia probe-bank design

Generate or mine candidate public states using several independent objectives:

1. policy versus completed-Q disagreement;
2. incumbent versus D1 teacher action disagreement/regret;
3. disagreement across at least three independently trained checkpoints or
   seeds;
4. rare public features: market overpopulation chains, low-bag regimes,
   constrained placements, fragmented habitat geometry, and unusual nature-
   token choices;
5. search-instability and high top-two SE, while keeping this family separate
   from ordinary D1 hard roots.

Admission requirements:

- legal under the repaired, hash-pinned rules identity and replayable from a
  durable seed/action trace;
- independent high-budget repeats agree that the incumbent action has material
  regret, not merely a near tie;
- weakness transfers across at least three model checkpoints/seeds;
- descriptor clustering and per-family caps prevent one exploit template from
  occupying the bank;
- report natural self-play frequency and estimated score impact separately;
- reserve entire generator/objective families as holdouts;
- evaluate fixes on the natural frozen bank, the known-probe bank, and unseen
  probe families to detect regression and overfitting.

Rules-engine failures found by the same generator must enter deterministic
rules tests and receive a new rules identity where required; they are not model
strength examples.

### Gated decision

**Run a bounded diagnostic bank after D1 and after the July 16 rules identity is
repaired.** Do not start full adversarial training unless the bank finds a
repeatable, diverse class with measurable natural-score impact. The first arm
is analysis/data generation and must not displace the current GPU-funded line.

**Confidence:** moderate that probes will expose transferable model blind
spots; high that they can expose rules/pathology failures; high that naive
patch-and-retest is inadequate; low-to-moderate that discovered weaknesses
will materially affect natural Cascadia mean score.

## 12. Q10 — Risk-sensitive action selection for a mean-score objective

### Short answer

For a mean-score objective, the decision statistic is the conditional expected
score. Distributional learning can improve representation and optimization,
but choosing a lower/upper quantile or CVaR changes the objective. Published
distributional RL results do not identify a universally helpful risk
distortion. Small Cascadia engineering screens of q25/q50/q75 produced too
little action movement to justify a powered gate.

**D1 should retain the existing arithmetic mean of its K midpoint heads plus
exact afterstate score.** This is the preregistered mean-serving projection,
not evidence that the heads are calibrated final-return quantiles. Do not use a
risk mode or head spread as search uncertainty without separate calibration.

### Best available evidence

| Source | Exact result | Interpretation |
|---|---|---|
| [C51](https://proceedings.mlr.press/v70/bellemare17a.html) | Learns a categorical return distribution but selects actions by expected value. On Atari it reported mean/median human-normalized score `1010%/178%` versus DQN `228%/79%`; C51 beat the fully trained DQN on 45/57 games at 50M frames. | Strong evidence for distributional representation under **risk-neutral expectation**, not quantile action selection. Cross-paper Atari aggregates and limited seeds are not a board-game effect size. |
| [Quantile Regression DQN](https://arxiv.org/abs/1710.10044) | Replaces fixed categorical atoms with learned quantiles while retaining expected-return control. | Supports quantile heads as a way to learn a return distribution, not a reason to optimize tails. |
| [Implicit Quantile Networks](https://proceedings.mlr.press/v80/dabney18a.html) | Full 57-game evaluation uses the risk-neutral mean and reports mean/median `1019%/218%` (five seeds). Risk distortions were shown on only six games: risk aversion helped some and hurt others; CVaR `.1` hurt Q*bert and Space Invaders, and risk seeking was worse on three of six. | No universal risk mode. The paper explicitly distinguishes return uncertainty from parameter uncertainty. |
| [Repo distq verdicts](docs/v3/RESEARCH_LOG.md) | K=8 distributional Q was positive at low budget under legacy rules, tied scalar at corrected champion budget (`+0.0875`, CI `[-0.2411,+0.4161]`), and q25/q50/q75 modes produced little action movement and no useful signal. | Supports retaining distq as the incumbent representation while mean-serving; it does not establish a positive corrected-rules strength gain. |

### Exact serving rule

For midpoint quantiles `q_k(s,a)`, use:

```text
predicted_score_to_go(s,a) = (1/K) * sum_k q_k(s,a)
served_Q(s,a) = exact_afterstate_score_active(s,a)
                + predicted_score_to_go(s,a)
```

If `q_k` are calibrated midpoint quantiles of a common conditional
distribution, equal weighting is a midpoint quadrature approximation to its
expectation. For nonuniform levels, use a preregistered integration rule and
corresponding weights.

These heads are trained quantile-wise on scalar search-Q labels, not calibrated
samples of Cascadia's conditional final-return distribution. Without separate
calibration, their spread cannot be interpreted reliably as either return risk
or epistemic model error. It therefore must not enter sigma, LCB, or search
allocation.

### Gated decision

- serve the risk-neutral quantile mean in D1;
- retain all K heads for training diagnostics and calibration;
- close the tested q25/q50/q75 modes; do not queue untested CVaR or other
  tail-risk serving because it misaligns with the current mean-score objective
  absent a deliberate objective change;
- require an independently calibrated model-error target before any
  distributional statistic affects sigma, LCB, or search allocation.

**Confidence:** high on retaining the mean-serving projection; high that no
universal risk distortion is supported; low that the distributional head itself
improves corrected-rules D1 strength until a paired gate demonstrates it.

## 13. Decision matrix and ordering

| Rank | Program | Disposition | Preconditions | Estimated incremental cost from repo evidence |
|---:|---|---|---|---|
| 0 | July 16 rules contract + new rules identity | **Adopt / blocking repair** | Reconcile per-wipe return semantics, update ID, fail closed on mixed artifacts | Small engineering/docs change; no scientific GPU run until identity is clean |
| 1 | D1 15k targeted reanalyze | **Adopt after recovery, pending John's restart and relabel-tranche authorizations** | Repair rules contract/new ID; complete Stage A; implement versioned masked targets and per-repeat emission; freeze amendments before their data boundaries | Outstanding generation was priced ~7–10 john0 GPU-hours, then ~26h for 15k x2 n2048/d16, plus D1 and matched K=8 no-D1 retrains/screens and the game gate; sentinel/optional dose arms cost extra |
| 2 | Exact-rules external human calibration | **Adopt as separate study** | Human recruitment, frozen estimand/rules/seeds/logging, blinded variance pilot | No john0 training cost; choose N from preregistered precision, not an unsupported fixed table count |
| 3 | Adversarial diagnostic probe bank | **Bounded experiment after D1** | Repaired rules ID; diversity/transfer/teacher-confirmation gates | Engineering and analysis first; GPU cost must be priced before launch |
| 4 | BAI allocator bakeoff | **Defer** | Post-D1 Q regime; frozen calibration/holdout roots; wall-parity harness | Bounded offline search cost; no game gate unless >=20% decision-regret improvement |
| 5 | R1.1c table-native action target | **Defer to John's methodology ruling** | Decide selfish versus cooperative objective; generate table-Q and table-derived improved-policy labels | One matched generation/retrain plus screens/gate; not yet durably priced |
| 6 | L3 variance/reliability sigma | **Defer** | Disjoint offline reliability calibration after Q regime changes | Calibration first; no static sweep |
| 7 | Determinization stratification | **Defer / low rank** | Unbiased weighted permutation sampler and frozen-root test | Unknown until sampler specified; pairing itself costs zero and already exists |
| 8 | Privileged posterior critic | **Defer / falsifier only** | D1 complete; hidden-order marginalization; public-only serving | Small offline arm before any gate; not priced |
| 9 | Symmetry trainer arm | **Defer standalone / shadow-only** | Existing rotation tests; add reflection transforms/tests before a cost-neutral D6 arm in an already-funded retrain | Zero dedicated GPU budget |
| 10 | Suphx GRP, oracle dropout, pMCPA, risk serving | **Close** | Materially new mechanism/evidence required | Zero |

This ordering keeps the one line whose pilot **hard-root label movement**
persisted in the full 7,600-root ledger—D1—ahead of literature-inspired
alternatives. The full
15k/d16x2/12.5%-masked-fold student and gameplay effect remains untested.

## 14. External calibration verdict

`100` is defensible as a **frozen internal engineering target**, not as an
externally calibrated estimate of achievable strength:

1. the champion produced a `98.2975` mean-seat estimate on a 100-game canonical
   battery, exactly replicated on a fresh 100-game block, leaving an observed
   1.7025-point sample gap—not a known expectation or a passed 1,000-game gate;
2. elite standard-rules individual recorded totals can reach 100 after their
   recorded majority bonus is stripped, but this does **not** show that a 100
   mean is attainable under the research identity;
3. no located solver or comparable bot result falsifies 100, but that absence
   is not affirmative evidence of achievability; and
4. preserving a frozen target maintains scientific continuity while the
   separate calibration study measures what it means externally.

It is **not** yet defensible as a human-superhuman threshold. The closest
external proxy differs in scoring cards, incentives, opponent/game
distribution, and objective, and no production BGA corpus was available at the
cutoff. The formal claim should remain:

> “Mean seat score at least 100 over 1,000 games under the pinned Cascadia v3
> research identity.”

An exact-identity elite-human claim requires direct exact-rules human
calibration. A matched standard-rules bridge can support only a separate
standard-rules model comparison; it cannot substitute for exact-rules human
data.

## 15. D1 prescription

This section answers the operational choices requested in the scope brief. A
label identifies the basis of every numeric choice:

- **[Published-direct]** measured or specified in primary literature;
- **[Published-analogical]** a mechanism in a materially different domain;
- **[Repo-direct]** measured or already preregistered in Cascadia;
- **[Judgment]** a falsifiable design choice not numerically fixed by evidence.

### 15.1 Corpus and tranche

| Parameter | Prescription | Basis |
|---|---|---|
| Base corpus | A successfully completed 1,250-seed x80-ply Stage A rerun, after the rules-contract/new-ID repair and John's restart permission, with durable decision/hard-root sidecars | **[Repo-direct]**: attempt 3's failure is logged; the 01:32 WSL reboot left no usable artifact |
| Eligibility | `hard == true` under the generation-time top-two-gap versus pairwise-SE criterion; exclude exact K1 rows | **[Repo-direct]** |
| Tranche | **15,000 roots** | **[Repo-direct/Judgment]**: inside the preregistered 10–20k band and current ~26h price; no published optimum |
| Phase allocation | **6,000 opening / 6,000 mid / 3,000 late** using the repository's phase definitions | **[Repo-direct premise/Judgment]**: stable movement was 50.3% / 49.5% / 36.9%, while 20% late preserves coverage |
| Hardness coverage | Stratify each phase across fixed gap/SE deciles, sample uniformly within cells | **[Published-analogical/Judgment]**: maintains diversity rather than selecting only extreme/noisy outliers |
| Correlation control | Target 12 roots/game in the first pass, spread across seats/phases; deterministic phase-stratified top-up may exceed 12 but never 16 | **[Published-analogical/Judgment]**: Expert Iteration sampled one state/game; freeze the shortage/top-up rule before teacher output |
| Repeat stability | **Do not prefilter** on repeat agreement; it is unknown before labeling. Keep agreement as an audit field and average repeats to denoise search-estimator instability. | **[Judgment]**: repeat disagreement is estimator noise, not a return-distribution target |

If the rerun yields all 100k planned roots and reproduces the prior 54.6% hard
rate, the 15k tranche is 15% of the corpus and approximately 27.5% of the hard
pool. These are planned/expected fractions, not known denominators or claimed
optima. Publish actual roots, skips, hard fraction, and selected coverage from
the completed manifest.

An optional **1,500-root phase-matched non-hard sentinel** could measure label
movement per GPU-hour outside the hard pool, but include or omit it before any
n2048 teacher output. It remains descriptive, outside the first training arm,
and adds about 10% to relabel work (roughly 2.6h on the current 26h estimate).
A causal hard-versus-uniform claim would require
equal 15k tranches, equal teacher compute, matched retrains, and matched gates;
the literature does not remove that requirement.

### 15.2 Teacher search and aggregation

| Parameter | Prescription | Basis |
|---|---|---|
| Teacher | Pinned current legal public-information champion/search stack | **[Published-direct/Repo-direct]**: current-teacher reanalysis is the common mechanism |
| Budget | **n2048/d16** | **[Repo-direct]**: 8x generation simulations and 4x distinct worlds; 43.6% stable movement with 0.361 moved-root regret over the full 7,600-root ledger (pilot 43.2%; not an independent sample) |
| Repeats | **Two independent search seeds, frozen and registered before relabeling and disjoint from generation and game-gate seeds** | **[Repo-direct/Judgment]**: 42.6% repeat instability is material; literature gives no optimal count |
| Search information | Never use the real hidden tile/bag order; independently sample worlds from the engine's exact conditional chance distribution | **[Repo contract]** |
| Aggregate Q | For action `a`, omit invalid repeats and set `N=sum_i n_i`; `Q_D1=sum_i(n_i*Q_i)/N` when `N>0`; `valid_Q=(N>0)` | **[Judgment]**: do not supervise from an unvisited repeat's model fallback |
| Aggregate policy | `pi_D1(a)=(pi1(a)+pi2(a))/2`; renormalize; select argmax with deterministic action-ID tie-break | **[Published-direct/Judgment]**: Expert Iteration supports soft targets, not this exact averaging rule |
| Aggregate value/variance | `V_D1=(V1+V2)/2`. If repeat `i` reports population variance `v_i`, use `var_D1=sum_i n_i*(v_i+(Q_i-Q_D1)^2)/N`, omitting invalid repeats | **[Judgment]**: exact population-variance pooling contract; define/test the one-valid-repeat and `N=1` cases |
| Metadata | Retain every per-repeat Q/visit/variance array, improved policy, root value, agreement, source hash, seed, ply, rules ID, and aggregate derivation | **[Repo contract/Judgment]** |

One repeat is 8x the n256 generation simulation count; two repeats consume 16x
the simulations per selected root. `d16` changes how those simulations cover
worlds; it is not an additional 4x multiplier on total simulations. The 5–10x
full/fast ratios in Expert Iteration and KataGo are useful analogies, but the
full-ledger Cascadia label-movement result is the real reason to keep n2048. The
existing puzzle-bank output does **not** retain enough per-repeat fields to
construct these aggregates; bank-mode training emission must add them before
relabeling. Summing visits and rerunning the Gumbel improvement transform is
invalid because doubling `max_visits` would artificially sharpen the policy.

### 15.3 Target semantics: replace stale search fields

The raw generated corpus remains immutable. Build a hash-pinned training view
with a versioned per-head masking schema. At minimum add `policy_valid`,
`outcome_valid` (or separate `value_valid`, `score_valid`, and `rank_valid`),
per-action Q validity/count/variance, and search-value validity. Defaults must
be `true` for legacy rows so old recipes remain bit-identical.

**For the selected root in the ordinary/base view:**

- retain state, legal actions, exact afterstate score, realized terminal
  scores, and provenance;
- mask the cheap `per_action_Q`, improved policy, search-root value, and their
  confidence/count fields.

**For the D1 view of that root:**

- enable the aggregated high-budget per-action Q, soft improved policy,
  search-root value, counts, and uncertainty fields;
- mask every duplicated behavior-outcome loss—value, score, rank, and any
  outcome-derived Q decomposition—unless a new valid current-policy outcome is
  generated.

This gives selected states one unambiguous fresh search target. Duplicating the
cheap and expensive policies would make the fold weight an accidental
interpolation of contradictory teachers. Duplicating the same old terminal
outcome would wrongly pretend the high-budget search had generated another
on-policy continuation. EfficientZero's stale-target correction is the closest
published warning; the exact masks are a Cascadia design requirement.

This requires schema, loader, batching, and loss-assembly changes, not only an
exporter emission flag. The current trainer averages policy loss over every row
and always trains value/score/rank; its per-action Q validity is insufficient
for source-local replacement.

The D1 row should train all eight quantile outputs through the existing pinball
objective, but serving reduces them to their uniform mean as specified in Q10.
The two teacher repeats are search-estimator repeats, not samples of the final
return distribution.

### 15.4 Fold weight and exposure

Use raw source weights:

```text
recent base : previous : older : D1
      4     :    2     :   1   : 1
```

The normalized mix is `50% / 25% / 12.5% / 12.5%`. The D1 source is therefore
0.25 relative to the newest base source.

Why this exact weight:

```text
2,500 steps * batch 192                 = 480,000 draws
480,000 * 12.5%                         =  60,000 D1 draws
60,000 draws / 15,000 unique D1 roots   =       4 expected passes/root
```

This is **[Repo-direct/Judgment]**: it targets four expected D1 passes. Add a
fail-closed weighted-source exposure audit because the current global guard
uses concatenated corpus length and does not enforce per-source exposure. The
ghost-label trial established non-regression for n256/d4 ghost labels at the
same relative exposure; it did not establish safety or efficacy for hard,
high-budget D1 labels. D1's own screen/gate is the test. The weight is not
borrowed from MuZero's 80% or EfficientZero's 99%, which mean nearly continuous
refresh of ordinary replay, not a small targeted shard.

### 15.5 Diminishing-returns measurement

Label all 15k roots before training. Freeze a deterministic nested ordering
using only pre-teacher phase, game, and hardness strata. The 15k model remains
the **primary arm in the proposed preregistration amendment**; optional 5k and
10k retrains are secondary descriptives for future pipeline sizing and cannot
replace it at the game gate. Freeze whether those secondary arms will run, and
their nested ordering/weights, before any retraining outcome is available.

Provided the base raw weights still sum to seven, training completes exactly
2,500 steps at global batch 192, and source sampling is unchanged, hold expected
exposure at four passes per root:

| D1 roots | Raw D1 weight with base sum 7 | Normalized D1 share | Expected D1 draws | Passes/root |
|---:|---:|---:|---:|---:|
| 5,000 | `7/23 = 0.3043` | 4.167% | 20,000 | 4 |
| 10,000 | `7/11 = 0.6364` | 8.333% | 40,000 | 4 |
| 15,000 | `1.0000` | 12.5% | 60,000 | 4 |

Draw counts are expectations; record actual cumulative source draws in the
training manifest. This curve isolates labeled **coverage** from repeated
exposure. It does not
measure the n512/n1024/n2048 teacher-budget curve. No primary source provides
that curve, and the current full-ledger n2048 label movement is sufficient to avoid
adding another GPU experiment before the primary D1 verdict.

### 15.6 Retrain, screen, and gate

1. **Preflight/fail closed:** verify all 15k roots, two repeats, action sets,
   rules ID, source revision, teacher hash, seed/ply identity, and SHA-256
   manifests. Reject partial/mismatched records.
2. **Retrain:** exact champion invocation, including every performance/numeric
   environment knob recorded after the recipe-fidelity incident; K=8 distq
   head; D1 weight above. Train a matched K=8 **no-D1 control** from the same
   initialization with the same base corpora, steps, numeric/runtime flags, and
   frozen training seed. This is the causal control; the historical scalar
   ctrl-SWA is only an absolute continuity anchor.
3. **Target audit:** confirm masks, normalized source frequencies, four-pass
   expectation, no duplicated selected-root search target, finite quantiles,
   and exact-afterstate reconstruction.
4. **Locked bank screen:** require both (a) D1 mean regret at least `0.010`
   better than the matched K=8 no-D1 control and (b) the already-preregistered
   absolute continuity bar versus historical ctrl-SWA `+0.2470`, i.e. D1 at
   most `+0.2370`, on the disjoint instrument. Report the matched-control
   difference and both absolute values. Training/validation loss is not a
   continuation signal.
5. **Fresh n256 game gate:** only after the screen. The comparator is the pinned
   current champion, with candidate and champion served under the identical
   n256/d4 search configuration—not the fresh no-D1 training control. As a
   proposed gate-launch preregistration, use paired fresh seeds and Lan-DeMets
   O'Brien-Fleming looks at 40/60/80/100 with planned final N=100. If CUPED is
   used, freeze its covariate and coefficient-estimation procedure before any
   gate outcome; otherwise use the unadjusted sequential gate. No other partial
   read.
6. **Champion-budget confirmation:** a positive screen/n256 gate earns a fresh
   corrected-rules champion-tier test. Promotion still requires at least 100
   paired games with the repeated 95% CI excluding zero and John's decision.
7. **Kill rule:** final n256 non-significance closes D1 and R1.4's current
   training-side label-margin hypothesis, per the existing preregistration.

### 15.7 Standing-pipeline rule

Do **not** make reanalysis permanent after an offline win. Promote it to a
standing stage only after:

1. the first D1 cycle passes its paired gameplay gate; and
2. one fresh generation/relabel/retrain cycle independently replicates the
   benefit.

Thereafter, once per new cycle:

- snapshot the current champion and newest recent corpus;
- recompute the hard census under the same audited definition;
- relabel a budgeted, stratified recent tranche;
- use only the latest D1 shard, or preregister explicit decay for older shards;
- never repeatedly re-label one stale buffer indefinitely;
- retain a uniform recent-data floor and monitor phase/hardness coverage;
- require the same screen and gate ladder for changes to the recipe.

MuZero supports continuous current-teacher refresh and ReZero supports periodic
sweeps. The two-positive-Cascadia-cycle rule is **[Judgment]** that prevents an
analogy from becoming permanent infrastructure after one noisy success.

## 16. Primary-source ledger

### Q1 — calibration and prior work

| Primary reference | Exact claim supported |
|---|---|
| [AEG official Cascadia rulebook](https://www.alderac.com/wp-content/uploads/2025/02/Cascadia-Rulebook.pdf) | Standard 20-turn setup, wildlife cards, the `3/1` majority schedule for four players, and tie handling used to derive the 15–20 table-bonus bound. |
| [WBC 2025 preview](https://www.boardgamers.org/wbc25/Previews/cas.html) | Random wildlife goals and event conditions. |
| [WBC 2023 report](https://www.boardgamers.org/yearbook23/cas.html) | Participation, game count, mean/max, and final scores. |
| [WBC 2024 report](https://www.boardgamers.org/yearbook24/cas.html) | Participation, game count, mean/max, and final scores. |
| [WBC 2025 report](https://www.boardgamers.org/yearbook25/cas.html) | Participation, mean/max, final scores, and final habitat bonuses. |
| [2025 WSBG final broadcast](https://youtu.be/EeFo6DlVOR8) | Public competitive final and its `101/98/96/94` recorded totals. |
| [BGA catalog](https://en.boardgamearena.com/gamelist) / [Cascadia panel](https://en.boardgamearena.com/gamepanel?game=cascadia) | Cascadia absent from the production catalog at the cutoff. |
| [Dire Wolf Cascadia Digital announcement](https://news.direwolfdigital.com/cascadia-digital-is-now-available-in-steam-early-access/) | Published modes, participant/AI support, and custom-card features. |
| [Dire Wolf Daily Trek description](https://news.direwolfdigital.com/cascadia-digitals-daily-trek-mode-a-new-puzzle-every-day/) | Daily leaderboard is a changing solo puzzle with three attempts, not four-player calibration. |
| [Steam global achievements](https://steamcommunity.com/stats/2438970/achievements) | Cutoff-date prevalence of the 100/105/110 single-game achievements, with no mode or attempt denominator. |
| [Public RL scaffold](https://github.com/davidxmoody/cascadia-ai/tree/a75b0bc4), [MATLAB shell](https://github.com/TMahls/Cascadia/tree/ef331947), and [BGA prototype](https://github.com/MarcelEindhoven/bga-cascadia/tree/240271ff5a1e2a012558fe638cad6bafb1bbe166) | Scope, incompleteness, and absence of a reproducible strong benchmark in the three closest public implementations. |
| [Additional public implementation inventory](#prior-cascadia-ai-solver-and-dataset-search) | Pinned source revisions and exact limitations for the five other located Cascadia engines/scaffolds. |
| [Recorded absence-search queries](#prior-cascadia-ai-solver-and-dataset-search) | Reproducible arXiv/OpenAlex/GitHub/platform query families and the cutoff-bound meaning of the negative search. |

### Q2 — imperfect information and privileged training

| Primary reference | Exact claim supported |
|---|---|
| [Suphx](https://arxiv.org/abs/2003.13590) | GRP target/equation, oracle features/dropout, pMCPA, training/evaluation scale, and full-system ranks. |
| [VLOG](https://openreview.net/forum?id=pjqqxepwoMy) / [pinned official code](https://github.com/Agony5757/mahjong/tree/db1e72e792fbae175bc9c3abb38f7dd92ee832b0) | Public-prior/privileged-posterior method and four-player Mahjong ablations. |
| [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) | Training-only opponent information in the value function and fixed-opponent ablation. |
| [Pluribus](https://doi.org/10.1126/science.aay2400) | Six-player superhuman poker architecture/evaluation without a luck-corrected training reward. |
| [AIVAT](https://ojs.aaai.org/index.php/AAAI/article/view/11481) | Unbiased poker evaluation variance reduction. |
| [Posterior Value Functions](https://proceedings.mlr.press/v139/nota21a.html) | Unbiased chance-conditioned policy-gradient baseline and variance guarantee. |
| [DouZero](https://proceedings.mlr.press/v139/zha21a.html) | Strong three-player imperfect-information learning from terminal Monte Carlo returns. |
| [Mortal](https://github.com/Equim-chan/Mortal/tree/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2) | Open-source Suphx-descendant implementation of rank-credit shaping; no independent causal ablation. |

### Q3 — reanalysis and targeted data

| Primary reference | Exact claim supported |
|---|---|
| [MuZero](https://arxiv.org/abs/1911.08265) | 80% fresh-policy reanalysis and its confounded hyperparameter changes. |
| [MuZero Reanalyse](https://arxiv.org/abs/2104.06294) | Reanalysis fraction, recent-state sampling, fixed-compute Atari scaling, and stochastic-bias warning. |
| [EfficientZero](https://arxiv.org/abs/2111.00210) / [pinned official code](https://github.com/YeWR/EfficientZero/tree/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e) | Near-total fresh policy/value reanalysis and stale off-policy value correction. |
| [Expert Iteration](https://arxiv.org/abs/1705.08439) | Initial 100k-state construction used 1k exploration and 10k labeling; later DAgger states were visited by the apprentice and labeled at 10k; one state/game and the soft-policy Elo result. |
| [KataGo training paper](https://arxiv.org/abs/1902.10565) | Fast/full search mixtures, recorded-turn sampling, and playout-cap-randomization ablation. |
| [KataGo policy-surprise weighting](https://github.com/lightvector/KataGo/blob/4c1a18216b15aaf7990dc8723a67641c4768dd72/docs/KataGoMethods.md#policy-surprise-weighting) | Uniform floor plus KL-based sample emphasis and the absence of an isolated numeric result. |
| [ReZero](https://arxiv.org/abs/2404.16364) | Periodic whole-buffer reanalysis, frequency study, and wall-time results. |
| [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | Priority/importance parameters, gains, and diversity/noise failure modes. |
| [Go-Exploit](https://arxiv.org/abs/2302.12359) | Archive restart mechanism and the boundary of its published experiments. |

### Q4–Q7 — search allocation, Gumbel values, multiplayer targets, worlds

| Primary reference | Exact claim supported |
|---|---|
| [Sequential Halving](https://proceedings.mlr.press/v28/karnin13.html) | Independent bounded-reward fixed-budget algorithm, its `H2` error bound, and qualified near-optimality scope. |
| [Successive Rejects](https://www.learningtheory.org/colt2010/papers/59Audibert.pdf) | Fixed-budget elimination that is parameter-free with respect to unknown hardness, under bounded-reward assumptions. |
| [LUCB](https://icml.cc/2012/papers/359.pdf) / [lil'UCB](https://proceedings.mlr.press/v35/jamieson14.html) | Distinct fixed-confidence leader/challenger and LIL-index sampling/stopping rules. |
| [Top-two probability sampling](https://arxiv.org/abs/1602.08448) | Fixed-`beta`, `beta*`, and factor-two asymptotic exponent qualifications. |
| [Variance-aware sequential halving](https://proceedings.mlr.press/v216/lalitha23a.html) | Known-variance SHVar and estimated-variance SHAdaVar under i.i.d. Gaussian arms. |
| [OCBA under correlated sampling](https://informs-sim.org/wsc04papers/072.pdf) / [classic OCBA](https://doi.org/10.1016/S1569-190X%2802%2900095-3) | Fixed-budget PCS objective, plug-in allocation ingredients, joint-normal/correlation assumptions, and the exact-two/approximate-many distinction. |
| [Fixed-budget lower bounds](https://proceedings.mlr.press/v49/carpentier16.html) / [open problem](https://proceedings.mlr.press/v178/open-problem-qin22a.html) | Limits and lack of general instance-optimal fixed-budget solution. |
| [CRN ranking and selection](https://arxiv.org/abs/1410.6782) | Selection with jointly multivariate-normal dependent observations under common random numbers. |
| [Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO) / [pinned MCTX](https://github.com/google-deepmind/mctx/blob/450fbf7656b88dd1d8ca5b2db3a2f9464cb322f2/mctx/_src/qtransforms.py) | Published constants, normalization choices, sensitivity scope, and qtransform defaults at the pinned revision. |
| [Gumbel MuZero for 2048](https://scholar.nycu.edu.tw/en/publications/gumbel-muzero-for-the-game-of-2048/) / [full experimental account](https://arxiv.org/abs/2212.11087) | Stochastic-domain performance at three training simulations and the `m`/`n` simulation-budget grid; no isolated `c_visit`/`c_scale` tuning result. |
| [Stochastic versus deterministic EWN](https://scholar.nycu.edu.tw/en/publications/an-empirical-analysis-of-gumbel-muzero-on-stochastic-and-determin/) | Replication of the low-training-simulation phenomenon in stochastic EWN and its absence from four deterministic variants. |
| [Kao 2048 master's-thesis catalog](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22110NYCU5394144%22.&searchmode=basic) | Documents the existence of a gated parameter section; used to bound, rather than overstate, the constants-search absence claim. |
| [Pinned KataGo methods](https://github.com/lightvector/KataGo/blob/4c1a18216b15aaf7990dc8723a67641c4768dd72/docs/KataGoMethods.md) | Dynamic utility-variance exploration as an adjacent, non-Gumbel mechanism. |
| [Multiplayer AlphaZero](https://arxiv.org/abs/1910.13012) | Player-value vector target and component-wise multiplayer backup. |
| [Deep Catan](https://www.lamsade.dauphine.fr/~cazenave/papers/DeepCatan.pdf) | Four-player win-probability value target and reported evaluation. |
| [BlokusZero](https://ipsj.ixsq.nii.ac.jp/record/204042/files/IPSJ-GI20043004.pdf) | Four-component win/loss target and MAXN component-wise backup in a four-player score-based game, with no target-shape ablation. |
| [LOCM drafting via RL](https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2020.pdf) | Terminal downstream battle outcome used as the PPO drafting reward/value target, with no own-score/margin/rank comparison. |
| [Score versus win in Go](https://arxiv.org/abs/2201.13176) | Direct warning that score targets do not automatically improve aligned playing strength. |
| [Variance reduction in MCTS](https://proceedings.neurips.cc/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html) | CRN variance identity and empirical game results. |
| [ISMCTS](https://edpowley.com/academic/papers/tciaig_ismcts.pdf) | Determinization pathologies and shared information-set search. |
| [Ensemble determinization](https://eprints.whiterose.ac.uk/id/eprint/75050/1/EnsDetMagic.pdf) | Budget-dependent number-of-worlds versus depth evidence in Magic. |

### Q8–Q10 — symmetry, blind spots, and distributions

| Primary reference | Exact claim supported |
|---|---|
| [AlphaGo](https://www.nature.com/articles/nature16961) / [Extended Data Table 3](https://www.nature.com/articles/nature16961/tables/3) | Dihedral training augmentation and the numerical inference-ensemble cost/accuracy comparison. |
| [AlphaGo Zero](https://www.nature.com/articles/nature24270) | Random rotation/reflection training augmentation, without a no-augmentation ablation. |
| [Expert Iteration on Hex](https://arxiv.org/abs/1705.08439) | Strong Hex apprentice/expert evidence, with no reported symmetry-augmentation ablation. |
| [HexaConv](https://arxiv.org/abs/1803.02108) | Hex-lattice equivariant convolution evidence outside gameplay. |
| [Adversarial policies beat Go AIs](https://proceedings.mlr.press/v202/wang23g.html) | Cyclic-attack compute and win rates at high victim search. |
| [Can Go AIs Be Adversarially Robust?](https://ojs.aaai.org/index.php/AAAI/article/view/34980) | Separate iterated-adversarial-training and architecture-defense experiments; persistence of newly trained attacks across defenses and search budgets. |
| [C51](https://proceedings.mlr.press/v70/bellemare17a.html) | Distributional representation with expectation-based control and Atari results. |
| [QR-DQN](https://arxiv.org/abs/1710.10044) | Quantile representation with expected-return control. |
| [IQN](https://proceedings.mlr.press/v80/dabney18a.html) | Risk-neutral full benchmark, limited risk-distortion ablations, and the distinction between intrinsic return uncertainty and parametric uncertainty; this does not classify Cascadia search-label-head spread. |

### Repository evidence used across the answers

| Repository record | Exact claim supported |
|---|---|
| [README status](docs/v3/README.md) / [RESEARCH_LOG baseline and §7](docs/v3/RESEARCH_LOG.md) | July-9 historical scoreboard, `98.2975` champion estimate/fresh-block reproduction, rules boundary, and current funded ordering. |
| [EXPERIMENT_LOG 2026-07-10/11](cascadiav3/EXPERIMENT_LOG.md) / [RESEARCH_LOG §4.10 and §7](docs/v3/RESEARCH_LOG.md) | R0.1's eight-arm sigma screen plus disjoint confirm-null and R0.2's rollout-pairing variance `0.020538 -> 0.021438`. |
| [Gumbel implementation](cascadiav3/real-root-exporter/src/gumbel.rs) / [R1.4 design](docs/v3/R1_4_DENSIFICATION_DESIGN.md) | Current `50/1.0` qtransform, per-action determinization cycling/CRN seed contract, four-seat state value vector, and the distinction between served Q and auxiliary table value. |
| [EXPERIMENT_LOG 2026-07-15/16](cascadiav3/EXPERIMENT_LOG.md) / [RESEARCH_LOG §7–8](docs/v3/RESEARCH_LOG.md) | D1 pilot/full-run movement, Stage A preregistration, attempt-3 reboot failure, zero artifacts, and the bounded interpretation of what remains untested. |
| [RESEARCH_LOG §4.5b](docs/v3/RESEARCH_LOG.md) / [EXPERIMENT_LOG 2026-07-08 16:10](cascadiav3/EXPERIMENT_LOG.md) | Three-rotation inference TTA score, confidence interval, and approximately 3x model-evaluation cost. |
| [RESEARCH_LOG §7](docs/v3/RESEARCH_LOG.md) / [EXPERIMENT_LOG 2026-07-16 00:30](cascadiav3/EXPERIMENT_LOG.md) | Ghost trajectory's consecutive-wipe bag failure, deterministic reproduction, and engine fix. |
| [RESEARCH_LOG distq/risk verdicts](docs/v3/RESEARCH_LOG.md) / [EXPERIMENT_LOG 2026-07-09 entries](cascadiav3/EXPERIMENT_LOG.md) | Corrected-rules K=8 high-budget tie and the small q25/q50/q75 engineering screens. |

## 17. Repository follow-ups implied by the research

### Blocking before D1 artifacts can be scientific evidence

1. Reconcile [RULES_CONTRACT.md](docs/v3/RULES_CONTRACT.md) with commit
   `45fb5072`: each automatic four-of-a-kind wipe must return its set before a
   subsequent automatic wipe, rather than holding all wiped tokens until the
   whole chain is stable.
2. Assign a new rules/config identity in
   [real-root-exporter/src/main.rs](cascadiav3/real-root-exporter/src/main.rs),
   propagate it through manifests/reports, and fail closed on July 9/July 16
   mixtures.
3. The attempt-3 reboot/zero-artifact experiment-log record is complete. Add a
   prominent rules-contract/new-ID record and rerun the full rules/exporter
   test set required by `AGENTS.md`. Existing deterministic bag regression
   coverage is necessary but not sufficient for provenance.

### D1 implementation and preregistration

1. Implement the versioned supervision schema, bank-mode emitter, loader,
   collation, and masked loss reductions for policy, search value, value,
   score, rank, and outcome-derived Q targets.
2. Add tests proving exact repeat aggregation (including one valid repeat and
   `N=1`), legal-action alignment, soft-policy normalization, selected-root
   replacement, legacy defaults, and no duplicate value/score/rank or
   outcome-derived Q loss.
3. Emit a harvest manifest containing the 6k/6k/3k quotas, hardness cells,
   per-game caps, deterministic ordering, and SHA-256 hashes before teacher
   output exists.
4. Freeze and register the independent teacher-repeat seed pair before
   relabeling; keep it disjoint from generation and game-gate blocks.
5. Extend the existing Stage A preregistration with the exact target semantics,
   weight arithmetic, primary 15k arm, whether the sentinel/secondary 5k/10k
   descriptives will run, screen, fresh seed block, and sequential gate rule.
6. Launch nothing from this report alone. The 15k relabel tranche remains
   reserved to John's decision under the existing autonomy boundary.

### Durable research state

1. **Completed:** linked this report and its frozen scope brief from the v3
   README/document map.
2. **Completed:** consolidated Q1–Q10 into RESEARCH_LOG and reprioritized L1–L8
   in RESEARCH_AGENDA without rewriting older evidence.
3. **Completed:** recorded the cutoff-bound BGA correction in the durable
   research state.
4. Keep all absence claims date-stamped and revisit them only when a new public
   dataset, paper, or production platform appears.

## 18. Final campaign answer

The research does **not** reveal that the campaign is aimed at the wrong score
or needs a Suphx-style detour. It validates the current order:

1. repair the rules/provenance identity;
2. with John's restart permission and separate relabel-tranche approval,
   complete and gate D1 under the then-frozen prescription, retaining K=8 and
   its mean-serving projection;
3. keep 100 as the internal target while separately building a comparable
   human calibration;
4. pursue bounded adversarial diagnostics next;
5. leave allocator, table-native, reliability-sigma, and stratified-world work
   behind direct offline evidence and the current funded line.

The permanent lesson is that the literature supports mechanisms more reliably
than constants. D1's mechanism—current-teacher, public-information search
targets on states where cheap decisions are demonstrably unstable—is strongly
supported, and the Cascadia pilot's n2048/d16 hard-root label movement persisted
over the full 7,600-root ledger. The full
15k/n2048/d16x2/12.5% masked-fold recipe is costed and preregisterable but
untested; it explicitly awaits student and gameplay verdicts.
