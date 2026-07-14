# R1.4 — Densify the Training Signal (Design)

Status: **DESIGN — nothing launched, nothing preregistered yet.** This memo
grounds the R1.4 program (`claude_max_research_ideas.md` §3 Tier 1) in the
actual trainer/exporter code and prices its stages against measured anchors.
Line references are to the working tree as of 2026-07-13. Companion memos:
`BRIDGE_THROUGHPUT.md` (R2.4, closed), `CAMPAIGN_STATE.md`,
`cascadiav3/EXPERIMENT_LOG.md`.

Campaign context: champion = cycle4 scalar M served at n1024/d16, **98.30
mean seat** (replicated 98.2975 on 07-13); gap to the 100-gate ~1.7 points,
diffuse (zero seats <90, 31.3% of seats already ≥100 — EXPERIMENT_LOG 07-10
11:00). The central measured finding is that **evaluation noise is binding**:
median top1–top2 completed-Q gap 0.049 vs pairwise SE 0.051 (decision SNR
≈ 1, ~46% of decisions noise-flippable), and oracle peek *loses* to honest
determinization averaging (07-07 15:04). Every serving-side estimation fix
(R0.1 sigma, R0.2 CRN, R0.3 bias, R0.4 LCB, R3.6 raw budget) returned null
or decelerating.

R1.4's stated hypothesis: training targets are sparse (one scalar outcome
per position; policy target only on the chosen action), so eval variance is
data-limited and the "EI saturation" verdict — measured with sparse targets
— understates the ceiling. Precedents: KataGo ~1.65x sample efficiency from
dense auxiliaries; PCZero ~2x from value-noise reduction.

**Honest headline first: the hypothesis as stated is about half false.**
The code audit below shows the pipeline already trains on soft full-menu
policy distributions, all-visited-action Q regression with SE-based
confidence weighting, per-seat/per-category outcome decompositions, a rank
head, an uncertainty head, and (optionally, already gated CI+ at n256) a
distributional Q head. Two of the five candidate targets named in the R1.4
brief are already implemented, and a third (action-conditioned category
decomposition) already **failed** its preregistered pilot on 07-10. What
genuinely remains sparse is narrower and is itemized in §2. The program is
still worth running — but smaller, cheaper, and with revised priors.

## 1. What the training signal actually is today (code audit)

Data path: the Rust exporter plays Gumbel self-play
(`--gumbel-selfplay-tensor-corpus`), and every visited root becomes one
schema-v4 record (`real-root-exporter/src/main.rs:3343-3407`, outcome
labels backfilled at game end by `backfill_final_outcome`,
`main.rs:3410-3434`). Packed shards feed
`torch_train_cascadiaformer.py` under the `gumbel-selfplay` objective
(weights at `torch_train_cascadiaformer.py:103-117`: policy 1.0, q 0.5,
value 0.5, score 0.05, rank 0.02, uncertainty 0.01).

Per record, the exporter emits (`main.rs:3343-3407`; struct
`GumbelSearchResult`, `gumbel.rs:222-237`):

- `priors` — model root priors over the full retained menu;
- `visits` — per-action visit counts (`gumbel.rs:230`);
- `per_action_Q` — completed Q: simulation-averaged value for visited
  actions, model derived-final-Q for unvisited (`gumbel.rs:1075-1083`);
- `per_action_Q_variance` / `per_action_Q_count` — population variance and
  count of simulation values per action (`gumbel.rs:1084-1095`);
- `per_action_Q_valid` — visits > 0;
- `improved_policy` — softmax(logits + sigma(normalized completed-Q)) over
  the **full retained menu** (`gumbel.rs:1097-1110`; sigma at
  `gumbel.rs:396-397`);
- `search_root_value` — improved-policy-weighted mean completed-Q
  (`gumbel.rs:1110-1113`, emitted at `main.rs:3374`);
- `exact_afterstate_score_active` plus, in v4, exact per-action
  wildlife/habitat/Nature afterstate components (`main.rs:188-195`);
- `final_score_vector`, `score_decomposition` (3 categories × 4 seats),
  `rank_vector` — **real terminal outcomes**, backfilled;
- `exact_endgame` provenance, active seat, full search/teacher contracts.

The trainer consumes (loss assembly `_loss_components`,
`torch_train_cascadiaformer.py:513-700`):

| Head (`torch_cascadiaformer.py:220-233`) | Target | Loss | Density today |
|---|---|---|---|
| policy logits | `improved_policy` (soft, full retained menu) | soft-target CE (`:541-549`) | dense over menu; information-bearing only on the ~top_m visited actions (§2.3) |
| `q_head` (score-to-go, optionally K quantiles) | `per_action_Q − exact_afterstate` for **every q-valid action** | smooth-L1 (`:586-592`) or pinball (`:574-584`), SE-confidence-weighted (`:571-572`) | ~top_m of ≤256 retained actions per root |
| `value_head` (4 seats) | `final_score_vector` (one realized outcome) | MSE (`:596`) | **1 sample/position — the sparse one** |
| `score_head` (3×4) | realized category decomposition (`:243-254`, `:597`) | MSE, weight 0.05 | 1 sample/position |
| `rank_head` | realized rank vector | CE (`:598`) | 1 sample/position |
| `uncertainty_head` | teacher SE = sqrt(var/count) (`:599-605`) | L1 | per q-valid action |
| `q_component_head` (v4 structured Q, optional) | selected-action realized category residuals (`:606-640`) | smooth-L1/pinball | **pilot FAILED 07-10, closed** |

Two facts that reshape R1.4:

1. **`search_root_value` is exported, required by the v2+ shard contract
   (`expert_tensor_shards.py:220-223`), collated into every training batch
   (`torch_train_cascadiaformer.py:319-321`) — and never used by any loss
   term.** A one-search-per-position low-noise value estimate is already
   sitting in every batch, unread. This is the cheapest lever in the
   program (§4 candidate V1).
2. The distributional Q head (`--q-quantiles 8`) already exists and was the
   **only CI+ training-side result since saturation** (+0.4275 at n256/d4,
   07-08 12:15) — but it compressed to +0.12 ns at n1024/d16 and, under
   corrected rules, distq ties scalar at the champion config (+0.0875 ns,
   07-10 03:30). Densification and search-time world-averaging are
   overlapping variance reducers; expect champion-tier shrinkage and
   preregister accordingly (§5, §8).

Generation grade for the corpus that produced all of the above: cycles run
at n=256–512 sims, top_m 16, depth_rounds 1, 4–8 determinizations, blend
0.5 (`scripts/run_gumbel_selfplay_cycle.sh:31-37`), ~1,250 seeds × 80 plies
≈ 100k roots/cycle (`:40-45`). Before training, shards are filtered to
top-64 actions by Q with the selected action always retained
(`FILTER_TOP_K=64`, `FILTER_MODE=top-q-with-selected`, `:50-51`); retained
improved-policy slices are renormalized (TRAINING_PIPELINE.md, Data
Formats).

## 2. Where sparsity genuinely remains

- **2.1 Value/score/rank labels are one Monte-Carlo sample.** Every
  outcome-labeled head sees a single realized terminal per position, whose
  variance includes ~60 plies of downstream decision and chance noise. This
  is the PCZero-shaped gap, and it is real. Caveat that bounds the prize:
  the value head is **not load-bearing at serving** — own-seat search reads
  only q/score-to-go (`EvalOut::value_vector` is "used only by table-total
  search", `gumbel.rs:212-214`; confirmed operationally in the table-total
  v1 postmortem, EXPERIMENT_LOG 07-08 10:15). Value-target densification
  pays only through trunk shaping and any future value-consuming serving
  mode, not directly through decision SNR.
- **2.2 Q labels carry generation-grade search noise.** Completed-Q labels
  come from n256–512/d4–8 searches: per visited action, mean of ~2–32
  noisy simulation values (SE mitigated but not removed by the confidence
  weight `1/(0.25+var/count)` clamped [0.25,4]). The serving config that
  defines the champion is n1024/d16 — labels are systematically *coarser*
  than the play they are meant to improve. Important measured damper:
  uniformly better labels (n256→n512→n512/d8-taught cycle-6) were **flat**
  through the scalar-head EI loop (07-07 08:26); "better labels
  everywhere" is a measured null. What was never tried is *targeted*
  precision (§4 D1) and better labels through a *non-saturated* head.
- **2.3 Policy improvement information covers only top_m ≈ 16 actions.**
  `improved_policy` is dense over the retained menu, but for unvisited
  actions its logit adjustment uses the model's *own* derived Q
  (`gumbel.rs:1075-1079` fallback feeding `:1097-1110`) — self-distillation
  with zero new information. With top_m=16 visited out of ≤256 retained
  (median full legal menu 1258; greedy-256 cap drops the true best action
  1.5% of the time at +0.30 regret each — R1.3a, 07-12 10:55), the
  fraction of the policy target that is search-verified is ~6% of the
  retained menu. The R0.3 unvisited-Q bias correction was structurally
  null at serving but its value explicitly "moves to improved-policy
  training targets (R1.4 program)" (07-12 01:35).
- **2.4 No spatial/dense-per-cell targets exist at all.** Nothing in the
  target set is per-hex. This is the one KataGo-style axis (ownership maps,
  their largest single factor) entirely absent from the pipeline.
- **2.5 No trajectory-linked targets, and packed shards cannot express
  them.** Packed v4 arrays (`expert_tensor_shards.py:659-689`) carry no
  per-record game/seed/ply fields — `root_seed_u64`/`ply_index` exist only
  in the per-record JSON metadata that packing discards (the v2 format
  note admits generation provenance "cannot be reconstructed reliably
  after packing", TRAINING_PIPELINE.md). Path consistency, short-horizon
  value targets, and opponent-reply auxiliaries all need record linkage:
  a v5 schema with two int32 arrays (`game_index`, `ply_index`) — small,
  but a schema rev with its validator/audit tail (§4 T1).
- **2.6 Market-refresh chance values are computed and discarded.** Refresh
  decisions are valued over `market_decision_samples=8` hidden-order
  samples, each a model evaluation (`gumbel.rs:687-698`); only the counts
  survive into artifacts (`GumbelTurnDecision`, `gumbel.rs:243-250`;
  `market_branches_searched`/`market_chance_samples` in record metadata).
  A distributional chance-node target would need new emission.

## 3. Already recoverable vs needs new emission

| Signal | Where it lives today | Usable for | Status |
|---|---|---|---|
| Full visit distributions, per-action root Q/variance/count, priors, improved policy | every v4 shard (`main.rs:3343-3374`) | (a), (b) | **already trained on** (§1) |
| `search_root_value` | every v2+ shard, already in every batch (`torch_train_cascadiaformer.py:319-321`) | low-noise value target (V1) | **recoverable now, zero regen** |
| Per-seat category decomposition of realized outcome | every shard (`score_decomposition`) | (c) state-level aux | **already trained on** (weight 0.05) |
| Exact per-action category afterstates | v4 shards | (c) action-conditioned | **exists; structured-Q pilot FAILED, closed pending new evidence** |
| n4096/d16 mega-budget root labels | R2.1 puzzle bank (727 roots ×2 repeats, ACCEPTED 07-12 01:15) | (d) reanalyze; offline screening instrument for all R1.4 members | machinery exists; training-scale relabel needs new runs |
| Champion decision ledgers + fail-closed ledger replayer | reports on john0; `--search-stability-probe`/`--puzzle-bank` modes | root harvesting for reanalyze/hard-root mining | exists |
| Trajectory linkage (game/ply per record) | JSON metadata only; **absent from packed arrays** (`expert_tensor_shards.py:659-689`) | path consistency, short-horizon heads, opponent-reply aux | **needs v5 emission** (cheap fields, real schema tail) |
| Per-hex terminal ownership/contribution maps | reconstructable from seed+replay, but not exported | (per-cell aux) | **needs new emission + corpus regen** |
| Market-refresh per-sample values | computed, discarded (`gumbel.rs:687-698`) | chance-node distributional target | **needs new emission** |
| Unread supply arrays (`unseen_keystones_by_terrain`, dual-pair counts), explicit turns-remaining | already exported in public tokens, unread by the feature stack | input-side fix (rider) | feature-stack change, no regen |

## 4. Candidate menu, ranked

Ranking = (expected variance reduction on a target that can move decisions)
× (prior survival odds given §1-§2) ÷ cost. Costs: S = env/flag/loss-term
change + one retrain; M = schema/emission change or corpus regen; L = both
plus new model surface.

| # | Candidate | Mechanism | Loss | Why variance falls | Cost | Risk |
|---|---|---|---|---|---|---|
| V1 | **Search-value value target** (use the unread `search_root_value`) | value target ← λ·search_root_value + (1−λ)·realized outcome for the active seat (λ∈{0.25,0.5}); keep other seats on outcomes | same MSE (`:596`), retargeted | replaces a 1-sample outcome (full downstream-play variance) with an n256–512-search posterior mean; PCZero's mechanism, exactly | **S** (trainer-only, zero regen) | search value is biased (optimism of max-flavored search; exploration temperature); value head isn't load-bearing at serving (§2.1) so gameplay payoff is indirect |
| P1 | **Improved-policy target completion** (label-side R0.3 + label-side sigma work) | at *generation*, apply `--gumbel-q-bias-correction` (exists, default off) so unvisited-action logit adjustments aren't self-distillation; optionally emit visit-masked targets so the trainer can down-weight unvisited mass | same soft CE (`:541-549`) | removes a systematic self-confirmation term from the densest loss in the objective (weight 1.0); the serving screen already showed the correction is structurally a *label* fix (07-12 01:35) | **S–M** (flag exists; needs a generation run to matter) | effect only via the next cycle's corpus → gated behind Stage 3; label-distribution shift vs replay-window shards |
| V2 | **Distributional value head** (KataGo-style buckets / quantiles over own final score) | new K-quantile `value_head` variant; pinball loss (pattern exists for q, `:574-584`) | richer gradient per sample from the same label; doubles as calibrated uncertainty for any future LCB/racing serving mode | **S** (head + loss, zero regen; distq machinery is the template) | distq precedent says champion-tier gain compresses where d16 already denoises; value head not load-bearing (§2.1) |
| D1 | **Targeted reanalyze of hard roots** (puzzle-bank-grade labels where SNR<1) | mine stored corpus/ledger roots with top-2 gap < SE (~46%); relabel per-action Q (and search value) at n2048–4096/d16 via the `--puzzle-bank` machinery; fold in as a weighted shard | existing Q loss; confidence weights rise automatically (var/count from the mega-search) | precision goes exactly where decisions flip; avoids the measured null of *uniform* label upgrades (§2.2) | **M** (harvest script + relabel runs + admission audit) | EI-saturation looms — cycle-6 says the scalar head can't absorb better labels; run only with a non-saturated head (distq) or after V1/V2 shows offline movement |
| T1 | **v5 linkage fields → trajectory losses** (path consistency; short-horizon value; opponent-reply aux) | add `game_index`/`ply_index` int32 arrays to the shard schema; then (i) PC-style consistency between value at t and search value at the same seat's next root, (ii) score-at-+1/+2-own-turns heads (labels computable at pack time from exact afterstates), (iii) predict next seat's chosen action | L2 consistency + MSE + CE | trajectory-averaged targets carry ~1/k of single-outcome variance; opponent-reply is KataGo's 1.30x aux and is free opponent modeling | **M** (schema v5 + validators + audit + regen or repack) | schema-rev tail is the real cost (v3→v4 precedent: a full day of contract work); benefits stack only through retrains |
| O1 | **Per-hex ownership analog** | per-cell final-contribution map head (does this hex score? member of largest habitat area? points contributed), labels from terminal boards at generation | per-cell BCE/regression over hex tokens | KataGo's biggest single densification factor; Cascadia's score *is* a spatial decomposition, so the analog is faithful | **L** (new emission + regen + new head over hex token positions) | most speculative transfer; competes for capacity at M scale; only candidate needing model-surface work |
| C1 | Aux weight sweep (score/rank/uncertainty 0.05/0.02/0.01 → ×4–×10) | reweight existing dense-ish auxiliaries | unchanged | weights were never swept; nearly-free control arm for the bundle | **S** | prior is weak; mostly valuable as a cheap comparator inside Stage 2 |
| X1 | Input rider: feed unread supply arrays + explicit turns-remaining | feature-stack change (not a target change) | — | not densification; listed because it shares retrain slots | S | scope creep — keep out of R1.4 verdicts, run as its own preregistered arm if at all |

Explicitly resolved against the original R1.4 brief:

- **(a) all-visited-action Q targets: already implemented** (`:518`,
  `:586-592`). No experiment to run; the residual is coverage (P1, D1).
- **(b) full-visit-distribution policy targets: already implemented**
  (`improved_policy`, `:541-549`). The residual is target *quality* on
  unvisited actions (P1), not target *shape*.
- **(c) category-decomposition auxiliaries: 2/3 already exist** —
  state-level `score_head` since v1 (weight 0.05, C1 sweeps it);
  action-conditioned structured-Q **failed its preregistered pilot**
  (selected-final RMSE −17.04% vs +10% bar; 07-10 10:20) and is closed
  pending materially new evidence. R1.4 must not relitigate it through the
  back door; only the per-hex analog (O1) is genuinely new.
- **(d) reanalyze: viable only as targeted relabeling** (D1). Full-corpus
  n4096 relabeling is priced out (§6), and uniform label upgrades are a
  measured null.
- **(e) distributional value head: the Q head is already distributional**
  (distq, CI+ at n256, champion-tie at n1024); V2 extends the idea to the
  value head, with expectations damped by that same precedent.

## 5. Staged kill-test plan (cheapest falsifier first)

Idiom: every stage preregisters its bar in EXPERIMENT_LOG before launch;
screens rank, never promote; gates are paired on fresh registered seed
blocks with group-sequential looks (40/60/80/100, OBF — adopted 07-12); a
null at any stage is a valid closed route, not a failure.

**Stage 0 — zero-GPU label-noise audit (CPU, ~1 day, blocks nothing).**
Over existing v4 shards + the champion ledger:
1. Density census: mean q-valid fraction per root, visit distribution over
   top_m, improved-policy mass resting on unvisited actions (quantifies
   §2.3 for the record).
2. **The V1 falsifier:** per position, compare |realized outcome −
   search_root_value| vs |realized outcome − model value prediction| vs
   outcome variance; measure search-value bias (mean signed error) by game
   phase. *Preregistered continuation bar:* search_root_value must cut
   value-target RMSE ≥20% vs the raw outcome at |bias| ≤ 0.5 points
   overall (phase-stratified read reported). Below that, V1's mechanism is
   absent — close V1 without a retrain and down-rank T1(i).
3. Hard-root census for D1: fraction of corpus roots with top-2 gap < SE
   (predicts ~46%), by phase.
Artifacts: one analyzer + JSON/MD report, same shape as
`analyze_menu_coverage`.

**Stage 1 — data-free retrains (V1, V2, C1): one variable each on the
known-flat control recipe.** The distq playbook, exactly: clone the
cycle-6/champion recipe (same corpora, same guard-clamped ~6,250 steps,
warm start incumbent, `--init-skip-mismatched` where heads change) so the
target change is the only variable. Selection on locked validation only.
- *Offline bar (per arm, preregistered):* locked-val value RMSE −10% (V1,
  V2) without q-regret degradation >0.05 (the structured-Q gate's
  convention); C1 must not degrade locked-val policy/q at all.
- *Screen:* puzzle-bank regret screen (~35 min/arm) — accepted instrument,
  never promotion evidence.
- *Gate (only for arms clearing both):* n256/d4 100-pair sequential gate
  vs incumbent on a fresh block. **A null here looks like:** offline RMSE
  moves but bank regret and the paired gate don't — i.e. the value trunk
  learned calmer numbers that decisions never consult (§2.1 realized).
  That outcome closes V1/V2 *and* is publishable evidence that value-head
  quality is not the binding constraint.
- Cost: ~3h/retrain + ~35min/screen + ~5h/gate (both arms) — §6.

**Stage 2 — conditional confirm at champion tier.** Any Stage-1 CI+ arm
gets an n1024-tier sequential confirmation on a fresh block (the distq
lesson: n256 wins can vanish at d16). Ghost+d32 serving (1.68x faster,
noninferiority gate live as of 07-13) halves this cost if adopted as the
speed default. **Preregister the expectation of shrinkage:** the confirm
bar is CI+ at champion tier, not replication of the n256 delta.

**Stage 3 — corpus-touching members (P1, D1), only if Stage 1 produced at
least one offline-positive arm.** Rationale: both are label-side bets, and
§2.2's uniform-upgrade null says labels only pay through a head that can
still learn — Stage 1 tells us whether one exists.
- P1: regenerate one cycle-sized corpus (~100k roots, ~10h john0 overnight)
  with `--gumbel-q-bias-correction` on; retrain the best Stage-1 recipe on
  new+replay-window shards; same screen→gate ladder.
- D1: harvest hard roots (Stage 0 census), relabel ~10–20k of them at
  n2048/d16 via puzzle-bank machinery (~30–60h, or fleet), fold in at
  weight ≤0.5 with the admission audit; retrain; screen→gate. **A null
  looks like:** flat gate despite confidence weights confirming much
  lower label SE on the folded shard — extending the saturation verdict
  from "more/better data" to "precision-targeted data", which would be
  strong evidence to stop resourcing training-side work entirely.

**Stage 4 — schema/emission members (T1, then O1), only on a Stage 1–3
CI+.** v5 fields + one regen unlock the trajectory losses; O1 remains the
long-pole bet with its own design note if we get here. Do not build v5
speculatively — a schema rev without a live consumer is pure tail risk.

**Stage 5 — the compounding question (densified EI cycle).** The only lane
that raises the policy family itself: if any target change is CI+ in play,
run one EI cycle generating labels *with* the improved model and retrain —
does the gain compound or is it one-shot? (Open since the distq EI-1 was
invalidated by the 07-08 rules correction; never re-run.) This is the
program's actual payoff thesis; everything before it is qualification.

Portfolio-level kill rule (restating `claude_max_research_ideas.md` §6):
if the Stage 1–3 bundle moves neither held-out value RMSE nor any paired
gate, **EI saturation survives a much stronger challenge and training-side
directions stop being resourced.** R1.4 is deliberately structured so that
verdict costs ≤ ~2 GPU-days before any corpus is regenerated.

## 6. Compute pricing (measured anchors, john0 RTX 5090, jobs12)

Anchors: champion serving n1024/d16 ≈ **35 s/decision**; ghost n1024/d32 ≈
**21 s/decision** (1.68x, score-noninferiority gate pending as of 07-13);
100-game arm ≈ **7–8h** champion shape / **~4.5h** ghosted; n256/d4 ≈ 85
s/game → 100-game arm ≈ **2.4h**; trainer ≈ 1.69 s/step (M, measured) →
guard-clamped ~6,250-step retrain ≈ **~3h**; cycle generation (1,250×80
roots at n512/d8) ≈ **~10h** overnight; puzzle-bank resolution ≈ **~11
s/root** at n4096/d16 (727 roots ×2 ≈ 4.5h); bank screen ≈ **~35
min/candidate**.

| Stage | GPU cost | Wall (queue-exclusive) |
|---|---|---|
| 0 audit | zero (CPU) | ~1 day orchestrator-side, parallel to any gate |
| 1 (V1+V2+C1) | 3 retrains ≈ 9h + 3 screens ≈ 2h + ≤3 n256 gates ≈ ≤15h | ~1–2 nights |
| 2 confirm | 1 × n1024-tier sequential gate ≈ 14–16h (or ~9h ghosted) | 1 day |
| 3 P1 | ~10h regen + 3h retrain + screen/gate ≈ ~20h | 2 nights |
| 3 D1 | 10–20k roots × ~11 s ≈ 30–60h relabel (john0) — or fleet-days on john1-4 at n256-grade (rejected: defeats the purpose; hard roots need mega-budget) + 3h retrain + gates | the expensive one; requires Stage-1 evidence first |
| 4 T1 | schema work (CPU) + repack/regen ≈ 10h + retrains | only on CI+ |
| 5 EI cycle | ~10h gen + 3h train + gates ≈ ~20h | only on CI+ |
| Priced out | full-corpus n4096 relabel: 100k roots × 11 s ≈ **~305h/pass** | not viable — hence D1's targeting |

## 7. Integration constraints

- **One scientific job at a time on john0.** Everything here enters the
  standing queue (`run_experiment_queue.sh` JSONL configs; HOLD pause
  files; done-marker resume; failure-tolerant stages). The live ghost
  speed-default noninferiority gate (PID 4110505) finishes first.
- **Preregistration discipline:** every bar in §5 is written to
  EXPERIMENT_LOG before its stage launches; screens are selection, never
  verdicts; disjoint fresh seed blocks from the registry, touched once;
  no partial reads — verdicts computed on-box by the runner.
- **Champion promotion** needs paired gates on registered seed blocks
  (group-sequential OBF schedule), n1024-tier confirmation, treatment/
  control timing ratio ≤1.20 unless pre-approved, and **John rules alone**.
  Offline RMSE, bank regret, and n256 wins are never promotion evidence.
- **Data admission:** corrected-rules (`cascadia-base-official-2026-07-09`)
  v4 shards only; raw shards pass `audit_structured_q_shards` with
  SHA-pinned NPZ/sidecars and globally disjoint seed intervals; D1 folded
  shards enter at explicit `TRAIN_SOURCE_WEIGHTS` with a safety trial
  (fleet-fold precedent: weak n128 labels poisoned cycle-5; upgraded ones
  were safe but customer-less).
- **Trainer guardrails preserved:** `--max-example-passes 4`,
  locked-validation selection, checkpoint contract, `--resume` mismatch
  refusal. New loss terms must be default-off flags with bit-identical
  defaults (the R0.x pattern) so every recipe stays replayable.
- TF32 stays off for batteries, on for generation (INFRASTRUCTURE
  precedent); bf16 stays banned for anything label-bearing.

## 8. Falsifiers and honest unknowns

- **The hypothesis's strongest form is already contradicted** (§1): policy
  and Q targets are dense; the pipeline sits closer to KataGo-dense than
  the portfolio assumed, which mechanically lowers the expected share of
  KataGo's 1.65x still available. The defensible residual claim is about
  *value-label variance*, *coverage beyond top_m*, *spatial targets*, and
  *trajectory structure* — nothing else.
- **Value head is not consumed by own-seat search** (§2.1). If Stage 1
  moves RMSE but no gate, that's the explanation, and it also predicts
  Stage 3–4 futility for value-flavored members — stop there.
- **Champion-tier compression is the expected failure mode**, not a
  surprise: distq (+0.43 at n256 → tie at n1024) is the measured template.
  R1.4's realistic wins may be (i) equal strength at much cheaper serving
  (distq was 97.38 at 2.8 s/dec), which compounds through every future
  gate and generation run, and (ii) compounding through EI (Stage 5) —
  not a direct champion jump.
- **Unknown until Stage 0:** search_root_value's bias structure;
  improved-policy unvisited mass; whether packed record order preserves
  trajectory adjacency (if it does, a zero-schema-change path-consistency
  prototype becomes possible ahead of v5 — verify, don't assume).
- **Unknown until Stage 3:** whether ghost-generated labels (interior
  opponents via CPU greedy) are training-safe teachers; ghosting halves
  label cost but changes the behavior policy of the corpus — needs its own
  low-weight safety fold before any densified cycle generates ghosted.
- If everything nulls: the saturation verdict survives its strongest
  training-side challenge; the residual ~1.7 points then live in serving
  reformulations (R3.x) or a redefinition of done — and R1.4 will have
  bought that certainty for about two GPU-days before touching a corpus.
