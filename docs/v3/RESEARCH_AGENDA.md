# Research Agenda — Break 100

Living document: the prioritized experiment queue, every program's current
status, and the standing decision rules. Updated at every verdict; the
blow-by-blow evidence lives in
[`cascadiav3/EXPERIMENT_LOG.md`](../../cascadiav3/EXPERIMENT_LOG.md), live
PIDs and resume state in [`CAMPAIGN_STATE.md`](CAMPAIGN_STATE.md), and the
original tiered portfolio (rationale, mechanisms, literature) in
[`claude_max_research_ideas.md`](../../claude_max_research_ideas.md).

**Goal:** mean seat score ≥ 100 over 1,000 games under
`cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`.
**Champion:** cycle4 scalar M at n1024/d16, **98.30** (98.2975 canonical;
replicated 98.2975 on a fresh block 07-13). Gap ≈ 1.7 points, diffuse.
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
- Ghost-generated labels are NOT yet cleared as training teachers
  (safety fold pending — see queue item 6).

## Active queue (07-14)

| # | Item | State | Decision rule (preregistered) |
|---|---|---|---|
| 1 | Canonical battery of adopted default (rebaseline block) | **DONE 07-14**: 98.3925 (descriptive; pre-ghost config read 98.2975 on the same block) | descriptive reference + fresh serving-default ledgers; never evidence |
| 2 | R1.3b menu-widening gate (`--gumbel-root-menu 512`, champion tier) | **CLOSED 07-14**: final look ns, delta −0.03, RCI [−0.27, +0.21] | menu widening is a measured null; R3.3 exact top-k is the surviving route to coverage |
| 3 | R1.4 Stage 1 retrains: **V1b**, **V2**, **C1**, **T0** | **VERDICT 07-14: ALL FAIL the value bar** (−5.3/−3.4/−3.7/−2.1% vs −10%); no gates | ctrl arm (running) attributes the residual drift; if ctrl SWA reproduces the shared q −14%, screen ctrl SWA |
| 4 | R1.4 D1 pilot | **PASSED 07-15: 43.2% stable movement (bar 20%), 0.40 pts mean stake — D1 FUNDED** | next: full-ledger relabel running (~13h); D1 Stage A (corpus hard-root harvest + distq-head retrain, 30-60h) to be preregistered in daylight |
| 5 | Survivor gates from #3 | conditional | per-arm preregistration at launch |
| 6 | Ghost-label safety-fold corpus (~20k ghosted roots) | backstop filler | generation only; labels quarantined until the fold retrain clears locked-val |

Next up after this slate: R0.5/R3.4 adaptive per-root budgets (supervision
free from the puzzle bank + tonight's ledgers), R1.1c/R3.1 cooperative
table values (after R1.4 infrastructure), R3.3 exactness expansion.

## Program scoreboard

### Open

- **R1.4 Densify the training signal** — design:
  [`R1_4_DENSIFICATION_DESIGN.md`](R1_4_DENSIFICATION_DESIGN.md).
  Stage 0 (07-13): V1 closed, V1b born, adjacency confirmed, hard-root
  fraction 54.6%. **Stage 1 (07-14): V1b/V2/T0 measured NULL (value
  RMSE −2..−5% vs −10% bar), C1 comparator flat — trainer-only value
  densification does not clear its bar at this corpus/recipe.**
  Survivors: D1 (pilot in queue), P1 (needs a generation run), and the
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
