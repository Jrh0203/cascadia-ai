# Research Agenda — Break 100

Living document: the prioritized experiment queue, every program's current
status, and the standing decision rules. Updated at every verdict; the
blow-by-blow evidence lives in
[`cascadiav3/EXPERIMENT_LOG.md`](../../cascadiav3/EXPERIMENT_LOG.md), live
PIDs and resume state in [`CAMPAIGN_STATE.md`](CAMPAIGN_STATE.md), and the
original tiered portfolio (rationale, mechanisms, literature) in
[`claude_max_research_ideas.md`](../../claude_max_research_ideas.md).
The complete July 16 external-research scope and verdicts are in
[`research_questions_7_16.md`](../../research_questions_7_16.md) and
[`research_answers_7_16.md`](../../research_answers_7_16.md).

**Goal:** mean seat score ≥ 100 over 1,000 games under the pinned four-player
all-A/no-habitat-bonus rules identity current at the time of the gate.
**Historical scoreboard identity:**
`cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`.
Commit `45fb5072` corrected an additional rules bug, so this identity is now a
historical evidence boundary; a successor identity is required before new
scientific runs.
**Historical champion under the July-9 identity:** cycle4 scalar M at
n1024/d16, **98.30** (98.2975 canonical; reproduced as 98.2975 on a fresh block
07-13). The historical observed gap is approximately 1.7 points and diffuse.
**Priority order ruled by John 2026-07-13** ("align the research queue as
you see fit to maximize our chance of breaking past 100").

## Standing methodology (all adopted)

- Screens rank (puzzle bank, ~6 min), gates decide (paired, fresh
  registered seed blocks, touch-once). Preregister before peeking; John
  alone rules champion promotion and rules-design changes.
- **Group-sequential gates** (07-12): looks 40/60/80/100, Lan-DeMets OBF,
  repeated CIs; the RCI at a stop is the evidence. First live early stop
  07-13 (60/100 pairs).
- **CUPED** (07-13): opt-in `SEQ_CUPED=1`; covariate fixed = baseline
  per-seed seat score; interval narrows ~10-25%, point estimate untouched.
- **Adopted serving speed defaults** (score-noninferior, cheaper):
  exact-K1 (07-10) → refresh-div4 (07-12, 1.24x) → **ghost+d32 (07-13,
  0.688x wall)**. Serving/gate/benchmark default is now
  `--gumbel-ghost-opponents --gumbel-determinizations 32
  --gumbel-exact-endgame-turns 1 --gumbel-refresh-sample-divisor 4` at
  n1024. Gate cost is ~3-4x below the 07-12 fixed-N baseline.
- **Ghost-generated labels CLEARED as teachers at 0.25-fold weight**
  (07-15 safety fold: both preregistered legs passed). Caveats: cleared
  at n256/d4 generation grade; higher fold weights need their own
  trial. Ghost pricing is a SERVING-side win only — generation with
  ghosts measured ~2× slower in all-seats selfplay.

## Active queue (07-16)

| # | Item | State | Decision rule (preregistered) |
|---|---|---|---|
| 1 | Canonical battery of adopted default (rebaseline block) | **DONE 07-14**: 98.3925 (descriptive; pre-ghost config read 98.2975 on the same block) | descriptive reference + fresh serving-default ledgers; never evidence |
| 2 | R1.3b menu-widening gate (`--gumbel-root-menu 512`, champion tier) | **CLOSED 07-14**: final look ns, delta −0.03, RCI [−0.27, +0.21] | menu widening is a measured null; R3.3 exact top-k is the surviving route to coverage |
| 3 | R1.4 Stage 1 retrains: **V1b**, **V2**, **C1**, **T0** (+ctrl) | **FULLY CLOSED 07-15**: all flag effects = continued training (ctrl −6.2% beat every arm); ctrl-SWA lead died on the bank screen (+0.2470 vs 0.2351) | lesson: locked-val loss deltas of 5-15% carry zero decision signal — only bank regret + gates screen training candidates |
| 4 | R1.4 D1 pilot / Stage A | **PILOT PASSED 07-15; Stage A attempt 3 FAILED 07-16 (john0/WSL reboot 01:32). Rules-identity repair DONE 03:50 (`..._rules_2026_07_16`, commit 31fc2c30); john0 idle, AWAITING JOHN's restart go.** | D1 remains funded; rerun stamps the 07-16 identity; the 15k n2048/d16x2 relabel remains a separate John decision |
| 5 | D1 relabel/retrain/screen/gate | conditional on a complete Stage A corpus | freeze the proposed sampling, repeat aggregation, per-head masks, 12.5% draw share, matched K=8 no-D1 control, and gate-launch rule before their data boundaries; bank screen requires ≥0.010 improvement over that matched control plus the historical ≤0.237 continuity bar, and the fresh n256 gate compares D1 with the pinned champion |
| 6 | Ghost-label safety fold | **CLEARED 07-15**: both preregistered legs passed (regret Δ0.0046 vs ±0.015; q-regret better than ctrl) | ghost labels are teacher-safe at 0.25-fold; ghost pricing is serving-only (generation measured ~2× slower) |

Current ordering supersedes the earlier “adaptive/table/exactness next” text:
repair rules identity → authorized Stage A rerun → separately authorized D1
relabel/retrain/screen/gate; exact-rules human calibration may proceed in
parallel. A bounded adversarial diagnostic follows D1. Adaptive allocation,
table-native values, reliability-sigma, stratified worlds, and exactness
expansion remain behind direct offline evidence and the funded line.

## Literature-inspired candidates (07-16 full Q1–Q10 synthesis)

Mapped from the primary-source synthesis in
[`research_answers_7_16.md`](../../research_answers_7_16.md) onto the measured
campaign constraints. Literature supports mechanisms much more strongly than
transferable constants. None jump the repaired Stage A/D1 line.

| # | Candidate | Maps to | Sketch |
|---|---|---|---|
| L1 | Continuous reanalyze (MuZero Reanalyze / ReZero) | label ceiling | D1 is the offline pilot. Do **not** make it standing infrastructure after one win: require a positive paired game gate and one independent fresh-cycle replication. Published `80–99.5%` replay ratios are not fold-weight recipes for this targeted shard. |
| L2 | Phase-keyed value-bias correction | decision noise | Retain as a low-ranked offline calibration question; the July 16 synthesis did not validate a transferable KataGo effect size or search-time correction. It does not displace D1. |
| L3 | Reliability-scaled Gumbel Q | decision noise | **DEFER.** Keep `c_visit=50`, `c_scale=1.0`, min-max. Any future rule must first estimate a disjoint low- versus high-budget reliability curve; do not launch another static sweep. |
| L4 | Mixed-grade generation | label ceiling / data volume | KataGo supports the mechanism, not a Cascadia ratio. Consider only in a later corpus recipe after D1 establishes whether high-budget label correction transfers to play. |
| L5 | Targeted relabel weighting | label ceiling | Fold into D1 rather than open a second arm: preserve a uniform base-data floor, use a 15k phase/hardness-stratified shard, and audit actual weighted draws. Repeat disagreement is estimator noise to average/audit, not a return target. |
| L6 | Adversarial diagnostic probe bank | blind spots / rules | **BOUNDED AFTER D1 and the rules-ID repair.** Require high-budget confirmation, cross-checkpoint transfer, diversity caps, and natural-frequency estimates before any adversarial training. |
| L7 | Cooperative table values | multi-seat | Existing `value_vector` already predicts four final scores. The real R1.1c intervention is table-native **per-action Q plus table-derived improved-policy labels**, requiring John's explicit selfish-versus-cooperative objective ruling. |
| L8 | League/exploiter populations / Suphx oracle detours | fixed point | **DEPRIORITIZED/CLOSED for the current objective.** Do not queue GRP, raw oracle dropout, pMCPA, or luck-corrected rewards. Revisit a privileged posterior critic only after D1 and only through exact public-state marginalization. |

Reassuring negative result from the same pass: Gumbel-with-few-sims,
sequential halving, and stochastic-game determinization — our existing
stack — match current published practice; nothing suggests the search
scaffold itself is the bottleneck.

## Program scoreboard

### Open

- **R1.4 Densify the training signal** — design:
  [`R1_4_DENSIFICATION_DESIGN.md`](R1_4_DENSIFICATION_DESIGN.md).
  Stage 0 (07-13): V1 closed, V1b born, adjacency confirmed, hard-root
  fraction 54.6%. **Stage 1 (07-14): V1b/V2/T0 measured NULL (value
  RMSE −2..−5% vs −10% bar), C1 comparator flat — trainer-only value
  densification does not clear its bar at this corpus/recipe.**
  Survivors: D1 (pilot passed; Stage A corpus still absent after the 07-16
  reboot), P1 (needs a generation run), and the
  ctrl-SWA q-loss lead. Kill rule intact: if D1/P1 also fail, EI
  saturation survives and training-side work stops.
- ~~**R1.3b/c menu coverage**~~ — **CLOSED 07-14**: root-menu 512 gate
  final-look ns (delta −0.03, RCI [−0.27, +0.21]); the R1.3a ceiling
  (+0.37) is not capturable by wider greedy menus. Bank screens proven
  VOID for menu candidates (frozen menus). Coverage survives only via
  R3.3 exact top-k retrieval.
- **R0.5/R3.4 adaptive compute allocation** — 46-55% of decisions are
  noise-flippable; puzzle bank + stability probes supervise for free.
  Queued behind the R1.4 slate.
- **R1.1c/R3.1 cooperative table values** — highest ceiling (0 to +2.0),
  only idea whose ceiling covers the whole gap; sequenced after R1.4's
  training infra. R1.1a (root-level re-ranking with selfish values) found
  nothing — the surviving version requires table-outcome-trained values.
- **R3.3 exactness expansion** — exact-K1 precedent was pure profit;
  frontier: factored bounds for exact top-k retrieval, last-2-own-turns.

### Adopted (velocity/economics, score-neutral)

- Ghost+d32 serving default (07-13) · refresh-div4 (07-12) · exact-K1
  (07-10) · puzzle-bank screens (07-12) · group-sequential gates (07-12)
  · CUPED (07-13).

### Closed — strength programs (verdicts, not opinions)

- **R1.2 ghost opponents as a STRENGTH lever** (07-13): CI+ +0.545 at
  n256-tier; ns at champion tier reinvested as sims (−0.08) and as worlds
  (+0.18, RCI straddles). Survives as the adopted speed default and for
  data generation (post safety fold).
- **R3.2 deep own-turn planning** (07-13): screen +0.0586 regret vs
  ≤+0.020 bar — sims diverted to our second turn starve the root.
- **R0.1 sigma calibration** (07-11): confirm null. **R0.2 CRN paired
  rollouts** (07-11): −4.4% vs −20% floor. **R0.3 q-bias at serving**
  (07-12): structurally null (label-side value moved into R1.4-P1).
  **R0.4 LCB** (07-12): flat. **R3.6 ceiling probe** (07-11): selfish
  scaling decelerating (+0.21 ns at 4x) — reshaped the portfolio.
- **R1.1a contention audit** (07-11): no cheap cooperative points at the
  root at own-Q parity.
- **R2.4 bridge throughput** (07-13): every lever below bar (pipelining
  +4.2% bit-identical, CHUNK_ROWS bound +3.9%, compile +0.5%, bucket
  negative); serving is within ~5% of the architectural ceiling —
  [`BRIDGE_THROUGHPUT.md`](BRIDGE_THROUGHPUT.md). Knobs landed,
  default-off.
- **Structured-Q (action-conditioned category heads)** (07-10): failed
  its preregistered pilot (−17% vs +10% bar); do not relitigate without
  materially new evidence.
- **Worlds det16→det32 pure confirmation** (07-10): PAUSED by ruling
  (wall-adverse); superseded in practice by the ghost+d32 adoption,
  block `2027071600..99` still reserved.

### Deprioritized (ruled 07-13)

R3.5 smarter worlds (family returns small effects; d32 ns on strength) ·
R0.7 world persistence, R0.8 control variates (root-estimation class is
0-for-4) · R1.2B/C learned ghost stages (need a fresh screened case).

## Central scientific findings

1. **Evaluation noise is binding** (decision SNR ≈ 1; 46% of serving
   decisions and 54.6% of corpus roots noise-flippable). Exactness beats
   estimation; selection-rule fixes at the root are a measured dead end.
2. **The selfish sim-scaling axis is saturated** at champion tier;
   reclaimed compute only pays if reinvested into non-saturated axes.
3. **Search values carry a phase-monotone calibration drift** (−7 at
   opening → +0.5 at endgame vs realized outcomes) — both a training
   target opportunity (V1b) and a caution for any search change relying
   on absolute Q calibration.
4. **Economics compound**: the 07-10→07-13 velocity stack (K1, div4,
   ghost+d32, bank screens, sequential stops, CUPED) cut the cost of a
   decisive experiment by roughly an order of magnitude combined.
