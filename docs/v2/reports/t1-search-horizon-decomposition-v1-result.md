# T1 Search-Horizon Decomposition v1 Result

**Completed:** 2026-06-17
**Experiment:** `t1-search-horizon-decomposition-v1`
**Protocol:** `t1-strict-train-horizon-decomposition-v1`
**Classification:** `t1_search_horizon_decomposition_development_null`
**Aggregate ID:** `817c5d469c59b830b5f7530712ceeacf105f19b2bb0d91850f9606d4e0559d13`

## Verdict

The T1 horizon decomposition is a clean open-train development null. The
frozen direct exact-R2 ranker selected better complete actions than immediate
qualified-leaf rescoring and every tested stochastic opponent horizon.

No arm is eligible. H0 was materially worse than direct ranking, and adding
one, two, or three opponent turns did not recover the loss. Validation, sealed
test, and gameplay remained closed. This result does not establish a new
gameplay score or progress toward 100.

## Frozen Development Result

All arms used the same 560 unfiltered open-train decisions, strict exact-R2
top-64 roots, exact root-first chance boundary, qualified MLX leaf, and R4800
reference labels.

| Arm | Mean R4800 regret | Median regret | Top-1 recall | R1200 pairwise |
|---|---:|---:|---:|---:|
| Direct exact R2 | **0.562608** | **0.369878** | **0.223214** | **0.588746** |
| H0 root leaf | 0.636628 | 0.471578 | 0.150000 | 0.573601 |
| H1 one opponent | 0.646858 | 0.487730 | 0.153571 | 0.572253 |
| H2 two opponents | 0.640652 | 0.476847 | 0.157143 | 0.572918 |
| H3 full rotation | 0.635184 | 0.468815 | 0.164286 | 0.572296 |

Positive treatment-minus-reference regret is worse:

| Comparison | Mean difference | 95% game-bootstrap interval | One-sided p |
|---|---:|---:|---:|
| H0 minus direct | +0.074021 | [+0.044333, +0.110573] | 1.000000 |
| H1 minus direct | +0.084251 | [+0.065579, +0.105223] | 1.000000 |
| H2 minus direct | +0.078045 | [+0.049532, +0.104611] | 1.000000 |
| H3 minus direct | +0.072577 | [+0.038559, +0.104745] | 1.000000 |
| H1 minus H0 | +0.010230 | [-0.012370, +0.033312] | 0.808510 |
| H2 minus H0 | +0.004024 | [-0.022241, +0.026829] | 0.638718 |
| H3 minus H0 | -0.001444 | [-0.025228, +0.021434] | 0.463877 |

The best searched mean, H3, was only 0.001444 better than H0 and remained
0.072577 worse than direct ranking. No searched-horizon comparison survived
the frozen Holm-Bonferroni family.

## Gate Accounting

All three searched horizons failed every decision-quality gate:

- no horizon improved direct regret by at least 0.05;
- no horizon improved H0 regret by at least 0.03;
- no direct or H0 superiority comparison survived Holm correction;
- no horizon preserved direct/H0 top-1 recall;
- no horizon remained within 0.005 of the better direct/H0 R1200 pairwise
  accuracy.

The leaf-only alternative also failed. H0 regressed direct ranking by 0.074021,
and its paired interval was wholly on the wrong side of zero.

## Exact Replication And Integrity

Every primary reproduced exactly on a different host:

| Arm | Primary | Replay | Scientific result ID |
|---|---|---|---|
| H0 | john1 | john2 | `c3820194b4194a36f2b8bf83f41153c9722c5f0a254ae2e87563d1e09f9d0771` |
| H1 | john2 | john3 | `cd249d07fd0b3df80f1d5b0c32f40289ac877ebce899c551dbef3ab2465856e8` |
| H2 | john3 | john4 | `02dd2ac564dd861f2fa6672f831859509230e900d8353a8393851f8e75491c11` |
| H3 | john4 | john1 | `9047313853b845f54c03bc5f0d7bfd8a2974fb914cbe56b11f03c5c6a8b103cf` |

The terminal aggregate proves:

- all four arms shared the frozen roots;
- every primary/replay pair matched exactly;
- every report was complete and fully accounted;
- all 560 groups participated;
- H0 consumed 35,840 root leaves;
- each searched horizon consumed exactly 358,400 trajectories;
- validation, sealed test, and gameplay were not opened.

Primary wall times over the complete 560-group cohort were:

- H0: 5.14 seconds;
- H1: 45.67 seconds;
- H2: 78.91 seconds;
- H3: 111.51 seconds.

## Interpretation

The tested legacy-compatible qualified leaf is weaker for immediate complete-
action selection than the frozen exact-R2 ranker. Searching one complete
opponent rotation around that leaf does not recover the lost ordering quality.

The experiment does not isolate one universal cause among leaf error,
opponent-policy error, and search allocation. It does establish the decision
needed by the roadmap: this exact leaf/search stack is not eligible for
validation or gameplay, and making it faster through T2 tree reuse would only
accelerate a failed decision mechanism.

The reusable public-belief engine, root-first chance contract, deterministic
prefix domains, resumable runner, and exact accounting tests remain valid
infrastructure. They are not promoted into the player.

## Consequences

1. Mark T1 complete with a development-null verdict.
2. Block T2 cross-turn tree reuse until a different search mechanism first
   demonstrates decision quality.
3. Keep the exact sparse R2 direct ranker as the selected serving and research
   substrate.
4. Do not rerun T1 with more depth, more trajectories, or another generic
   allocator without a diagnosed residual and new preregistration.
5. Move the primary research frontier to O2 demand-supply matching and the
   independent O3 plan-slot campaign.
6. Redesign the compact learned leaf before any future search successor uses
   this infrastructure.

## Claim Boundary

This was an open-train offline mechanism experiment. It cannot establish:

- open-validation or sealed-test generalization;
- gameplay improvement;
- champion promotion;
- a score increase or progress toward 100;
- tree-reuse value;
- superiority of a dense 441-cell representation.

The search state remained typed sparse v2 `GameState`, and the learned root
representation remained exact sparse R2. Historical 441-coordinate indices
appeared only inside the frozen legacy-compatible leaf.

## Artifacts

- terminal aggregate:
  `artifacts/experiments/t1-search-horizon-decomposition-v1/aggregate-v1.json`;
- authorization:
  `artifacts/experiments/t1-search-horizon-decomposition-v1/control/authorization-package-v1/authorization.json`;
- immutable bundle:
  `artifacts/experiments/t1-search-horizon-decomposition-v1/bundles/00a54a0c4a28353975257d01f17ddfd6b3c6257b9210970422e82b3270730334`;
- collected reports:
  `artifacts/experiments/t1-search-horizon-decomposition-v1/collected-v1/reports`;
- queue specification:
  `artifacts/experiments/t1-search-horizon-decomposition-v1/queue-spec-v1.json`.
