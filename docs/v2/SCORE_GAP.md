# Current Score Gap

The strongest promoted v2 product policy is
`pattern-aware-v1-k8-h6-b8-m4`. It scored 91.525 on its original confirmation,
91.890 in the direct K8 product control, and 91.580 on ADR 0068's fresh
canonical-redetermination suite. H6 remains a confirmed research teacher at
91.760 on its own suite.

The final-five R8 c90 terminal policy remains a stronger score-only research
control at 92.100, +0.520 over pattern-aware with 95% CI
`[+0.260,+0.780]`. ADR 0068 demoted it from product promotion because its
corrected run shifted +0.460 into Bear while losing 0.375 aggregate
Elk+Salmon+Hawk+Fox, failing the original balanced-allocation guardrail.

The independent v1 reproduction is complete over 50 games and 200 scored
seats. The engines are not rules-identical, so this is a research diagnostic
rather than a canonical head-to-head result.

| Component | V2 H6, 50 games | V1 reference, 50 games | V2 - V1 |
|---|---:|---:|---:|
| Habitat | 29.730 | 30.985 | -1.255 |
| Bear | 6.545 | 11.615 | -5.070 |
| Elk | 12.040 | 11.050 | +0.990 |
| Salmon | 12.455 | 12.580 | -0.125 |
| Hawk | 12.510 | 11.110 | +1.400 |
| Fox | 15.180 | 14.720 | +0.460 |
| Nature Tokens | 3.300 | 3.835 | -0.535 |
| Base total | 91.760 | 95.895 | -4.135 |

The current gap is not a uniform wildlife failure. H6 repaired 1.08 habitat
points relative to the earlier K8 profile and added roughly two Bear points,
but Bear pair construction remains the dominant category deficit. Habitat is
still second. H6 already exceeds v1 in Elk, Hawk, and Fox; Salmon is effectively
level at this sample size.

This evidence prioritizes:

1. strong-policy MLX value learning that can price Bear setup without
   distorting the other species;
2. learned leaf or rollout evaluation on top of H6;
3. local self-play iteration with paired regression gates.

Pattern-aware independently recovered much of the same missing mechanism:
against greedy it added 3.875 Bear, 1.295 habitat, 1.730 aggregate wildlife,
and 1.865 Nature Tokens while improving every one of 50 games. Against K8 it
retained 2.805 more Bear points and 0.480 more total wildlife. The remaining
gap is now a rollout and policy-iteration problem rather than a root candidate
recall problem.

Fair Nature Token refresh search was tested after this diagnosis. It improved
Bear by 2.25 points in a five-game pilot but paid for 60 wipes, lost 1.30
Nature Token points, and regressed by 0.50 overall. The mechanism remains
research-only until its exercise value can be calibrated from held-out
outcomes.

Bear-specific candidate recall and MLX ranking both increased Bear while
materially reducing the four other wildlife cards. Habitat H6 is the first
specialized candidate intervention to confirm a positive total-score gain
without aggregate wildlife collapse. Extending H6 rollouts from four to eight
plies was null, so further depth with the same greedy policy is closed.

The direct K8+H6+B8 union repeated the tradeoff against the stronger H6
baseline: +2.075 Bear, -2.500 aggregate non-Bear wildlife, and -0.300 total.
That closes brute-force Bear candidate injection as a route to the gap.

Replacing H6's greedy future plies with the confirmed pattern-aware policy was
also negative: -0.550 paired over ten disjoint games, with 95% CI -1.796 to
0.696. A four-ply exact-score leaf does not reliably realize pattern-aware
setup value. The next iteration should use learned search targets or longer
term value estimates, not another heuristic rollout substitution.

A complete apprentice-distribution policy-iteration round also produced only a
small ranking gain: held-out apprentice top-one regret improved by 0.0178
without forgetting the original H6 distribution, short of the registered
0.03 gate. No gameplay was run. This weakens the hypothesis that state
distribution shift alone explains the learned policy gap and points more
strongly toward missing cross-turn, market, and opponent representation.

An exact two-personal-turn wildlife opportunity model initially regressed
because it valued impossible late-game setup. After phase capping, its
registered pilot improved by 0.650 and added 1.900 Bear with an 8-0-2 record,
confirming that cross-turn commitment contains useful signal. It still lost
0.950 across Elk, Salmon, Hawk, and Fox plus 0.650 habitat, so it failed the
mechanism guardrails and did not advance. The next model must price setup
realization under market competition and preserve global allocation rather
than maximizing an optimistic species opportunity in isolation.

A fair R2 terminal policy-improvement oracle was the first Bear intervention
to improve both total wildlife (+1.167) and habitat (+0.917), but it gained
only 0.250 total across three games, lost 1.833 Nature Tokens, and had
per-seed deltas from -2.75 to +5.00. Two full-game samples are too noisy to
qualify training labels. This leaves higher-sample terminal evaluation as a
specific variance question, not evidence that terminal targets are already
ready for MLX distillation.

R8 answered that variance question positively on its registered qualification
suite: +1.333 paired, +1.750 Bear, +0.333 total wildlife, +1.417 habitat, and
only -0.417 Nature Tokens. Its 94.833 mean is the closest fresh v2 policy yet
to the 95.895 v1 reference, but its 185.878-second runtime and three-game
sample make it a data teacher rather than a champion. The immediate task is
to distill these terminal action values into MLX and test whether held-out
ranking quality survives fast gameplay.

The first distillation collection was invalidated before training: candidate
records encoded the real post-draft refill from hidden stack order, while the
teacher target averaged redetermined futures. That was information leakage and
target-feature mismatch, not evidence about model quality. `compact-entity-v2`
now encodes the deterministic board result and depleted public market before
refill; corrected terminal data must be recollected from scratch.

That corrected experiment is complete and negative before gameplay. The model
learned broad ordering (0.680 pairwise accuracy, 0.508 value-difference
correlation) but missed best-action fidelity at 0.968 mean regret and 0.276
tie-aware recall. It only slightly beat immediate rank 1 at 0.999 regret. The
full-afterstate encoder has no action identity or newly placed marker, forcing
it to rediscover a tiny candidate delta inside four nearly identical boards.
The next neural experiment should encode that delta directly and retain the
correct pre-refill public boundary.

That successor is now fully implemented and frozen before substantive test
collection. `action-delta-ranker-v1` reconstructs the same terminal labels but
adds exact draft identity, changed-tile and changed-wildlife markers, placement
coordinates, market-prelude costs, and eleven immediate category deltas. The
pipeline hash-matches actions and byte-compares observable afterstates against
the corrected source, then trains a compact MLX ranker under validation-only
checkpoint selection.

This experiment has not earned a gameplay claim yet. Sixteen untouched R8 test
games control advancement through four fixed ranking gates. Only a complete
pass may unlock the ten-game gameplay pilot; promotion cannot be forced by a
good validation checkpoint alone.

The frozen result was negative. On untouched test, action-delta ranking passed
pairwise accuracy (0.670) and value-difference correlation (0.495) but failed
mean top-one regret (0.968) and tie-aware recall (0.273). No model was promoted
and no gameplay was run. Explicit action identity therefore does not explain
the remaining teacher-distillation gap.

The next independent mechanism is opponent-conditioned future market
availability. The existing pattern planner estimates wildlife opportunity from
optimistic shared supply, but in four-player play three opponents draft before
the acting player returns. A useful successor must price which species are
likely to survive that competition and preserve global habitat/species
allocation, rather than attempt another encoding of the same terminal labels.

That experiment is now complete and rejected. Exact first-rotation competition
improved the ten-game paired mean by 0.875, Bear by 1.275, habitat by 0.400,
and Nature Tokens by 0.725. It did not solve allocation: total wildlife fell
0.250 and Elk+Salmon+Hawk+Fox fell 1.525. Its parallel runtime was also 10.316
seconds per game against a five-second gate. Market competition is therefore
real signal, but expected-best-species continuation is the wrong scalar
objective. The next rules-derived policy should value a multi-species
portfolio directly and use a cheaper availability correction.

An anchored follow-up kept the promoted one-turn opportunity and added only
the exact conditioned second-turn premium. It eliminated the allocation
collapse: non-Bear wildlife moved from -1.525 to +0.500, habitat was +0.025,
and runtime passed at 2.866 seconds per game. Strength disappeared with it:
paired score was +0.025, Bear was -0.575, and total wildlife was -0.075.
Together the two pilots close hand-authored scalar interpolation as the next
route. The stronger independent signal remains terminal policy improvement,
which should now receive the optimization effort.

Behavior-preserving profiling reduced the promoted pattern policy from 0.506
to 0.066 seconds per game on the exact ten-seed reference, and reduced the R8
reference seed from 273.884 to roughly 80-85 seconds without changing any
score. Applying R8 only during each player's final four turns made the search
interactive: 5.895 seconds per complete game and 315 ms P90 move latency.
That hybrid gained 0.475 with a tightly positive 95% interval
`[+0.197,+0.753]`, but missed its frozen +0.500 gate by 0.025 and Bear fell
0.100. The result confirms late terminal-conversion value while indicating
that four remaining turns begin too late to recover the full teacher's Bear
gain.

Starting the same operator with five personal turns remaining passed its pilot
at +1.000 and confirmed a smaller but statistically positive +0.425 gain over
50 disjoint games, with 95% CI `[+0.198,+0.652]`. Wildlife gained 0.145,
habitat gained 0.340, and runtime held at 7.530 seconds per game with 382 ms
P90 decision latency. Bear gained 0.200, narrowly below the frozen +0.250
confirmation guardrail, so the strategy was correctly rejected. The score
signal is now replicated; the unresolved question is candidate information
and multi-species allocation inside those terminal decisions, not whether to
sweep another cutoff.

A bounded wildlife-diverse frontier tested that candidate-information
hypothesis directly. It gained 0.550 with an 8-0-2 pilot record, 1.625 Bear,
and 0.500 habitat, but total wildlife rose only 0.100 because
Elk+Salmon+Hawk+Fox fell 1.525. The wider frontier therefore reproduced the
allocation failure rather than fixing it. Since terminal selection maximizes
eight-sample candidate means, adding actions also increases finite-sample
winner's-curse exposure. Conservative paired policy improvement is the next
specific variance-control question.

That variance-control test was constructive but incomplete. A one-sided 90%
paired lower-bound gate increased the W2 pilot to +0.825 with a positive
interval and cut the non-Bear loss from 1.525 to 0.800. Total wildlife and
habitat both improved, so finite-sample maximization was genuinely part of the
failure. The non-Bear guardrail still failed. The remaining controlled
question is whether the same fixed confidence rule on the original
K8+H6+B8 frontier keeps the gain while removing the W2 allocation pressure.

That final controlled corner passed under the original finite sampler. On the
original frontier, c90 gained
0.500 in pilot while improving Bear, total wildlife, non-Bear wildlife, and
habitat. The frozen 50-game confirmation retained +0.420 with a strictly
positive interval and preserved every component guardrail: Bear +0.080,
non-Bear wildlife +0.035, total wildlife +0.115, habitat +0.365, and Nature
Tokens -0.060. ADR 0068 later superseded the promotion after canonicalizing
redetermination: the score gain remained positive, but non-Bear wildlife fell
0.375. Late terminal policy improvement is useful as a score-only research
control, while product promotion remains with pattern-aware.

The v1 reference mean is 95.895 with a game-block 95% confidence interval of
95.480-96.310. The strongest promoted product policy is 91.580 on the current
fresh suite, leaving an observed 4.315-point gap to the v1 reference and
8.420 points to the 100 target. The research terminal control is 92.100,
leaving 7.900 points to target. Those gaps are not explained by benchmark
sampling noise.

The first MLX attempt to learn strong's exact confidence-gated decision is now
also closed. Direct c90 lower-bound regression generalized numerically
(validation MSE 0.967 versus 4.789 for zero and correlation 0.761) but recovered
only 0.7% of selected challengers. Even ignoring the zero threshold, its exact
challenger rank ceiling was 16.8%. The failure is therefore groupwise policy
identification under a sparse positive class, not simple scalar calibration.
The registered successor trains that decision directly with balanced
anchor-versus-challenger cross-entropy and keeps regression only as an
auxiliary signal.

That successor also failed before test access. Its groupwise policy head chose
the anchor in every validation group and achieved only 15.4% exact challenger
argmax recall without the threshold. The shared failure is now explained by
the teacher protocol: R8 sample seeds depend on the hidden game seed, so exact
sample-level choices include noise absent from the public model input. More
loss engineering on the same labels is closed. The next controlled question
is whether R32 c90 is stronger than promoted R8 and therefore worth distilling
as a lower-variance expected-advantage teacher.

R32 was not stronger. It scored -0.125 against R8 over the frozen pilot, with
a 4-0-6 record and interval spanning -0.616 to +0.366, while costing 28.217
seconds per game. Higher-sample terminal labels are therefore closed as the
next route.

The next MLX value experiment returns to a distinct measured failure. The H6
final-score model reached low MAE but only 0.212 rank correlation because all
positions in a game shared one narrow final target. ADR 0028 freezes signed
score-to-go components and reconstructed-final evaluation on the identical H6
seed domains. This tests target semantics without changing the trajectory or
encoder architecture. The versioned signed dataset, residual head, exact target
identity checks, atomic one-game resume path, and Apple MLX smoke have passed;
the frozen 256/64 run is now complete.

Score-to-go was rejected before gameplay. It improved reconstructed-final
correlation from 0.212 to 0.397 and learned residual total almost perfectly
(`r=0.992`), but the best selected checkpoint still missed the 0.50 gate and
slightly worsened total MAE from 2.538 to 2.569. Adding current score back
exposes the same narrow game-outcome signal after the model learns the much
larger phase-dependent workload. Future value work must create
counterfactual, decision-local signal rather than relabeling the same
trajectory outcomes.

ADR 0029 now asks a more basic bounded question before further learning: if
the true hidden future is exposed only for diagnosis, how strong is exact
full-game policy improvement over the existing K8+H6+B8 frontier? A result at
or above 100 would place the bottleneck in uncertainty and value
identification. A result below 97 would show that the current frontier or
frozen pattern continuation is itself a material ceiling.

The corrected focal-seat result was 93.150, +1.775 over pattern-aware with a
strictly positive interval, but below the frozen 97-point boundary. Perfect
future knowledge moved 11.325 points into Bear while losing 9.025 across Elk
and Hawk. This is not a global mathematical upper bound on the frontier; it is
one exact policy-improvement step. It rejects an uncertainty-only explanation
and shows that the frozen continuation's global species allocation is a
material limit. A perfect-information wildlife-diverse frontier control can
now test whether missing structural candidates contribute independently,
without Monte Carlo winner's curse.

That control confirmed an independent candidate-recall gap. Adding two
distinct candidates per wildlife species under exact full-game evaluation
improved the focal mean from 92.625 to 93.975: +1.350 with 95% CI
`[+0.704,+1.996]` and a 9-0-1 record. The gain was almost entirely Fox
(+1.775); Bear, Elk, and Hawk declined slightly, Salmon gained 0.300, habitat
fell 0.500, and Nature Tokens gained 0.600. The result clears the structural
materiality threshold but remains below 97 and 6.025 points short of 100.
Candidate breadth and continuation quality are therefore both active limits.
The narrow next test is Fox-only recall under the promoted public-information
confidence gate, not another unrestricted all-species expansion.

That public transfer test was a complete null. Strong and strong plus Fox F2
tied all ten pilot blocks at 92.150, with Fox only +0.050, total wildlife
-0.025, non-Fox wildlife -0.075, flat habitat, and passing runtime. The
focused channel changed some category allocations but produced no net score.
Together with the exact +1.350 W2 result, this isolates the failure: candidate
recall contains real value, while the current public-state R8 c90 evaluator
cannot identify it reliably on strong trajectories. More frontier breadth and
more samples of the same terminal estimator are both closed. The next learned
target must be counterfactual and decision-local rather than another encoding
of narrow trajectory outcomes or noisy sample-specific choices.

A multi-turn exact diagnostic then separated continuation from one-step value
identification. Replacing the final five one-step focal decisions with a
width-16 beam improved exact W2 from 92.900 to 93.650: +0.750 with 95% CI
`[+0.400,+1.100]` and a 9-1-0 record. Habitat, total wildlife, and Nature
Tokens all improved. The mechanism is real, but the absolute mean remains
below 97 and 6.350 short of 100. This closes the claim that one-step exact
values are sufficient and identifies multi-turn counterfactual continuation
as the next neural target. A public model must learn that signal from
redetermined futures; hidden state remains diagnostic-only.

The first public recovery attempt failed before neural collection. A
four-redetermination, width-four final-five focal beam scored -0.075 against
strong with an interval spanning -0.565 to +0.415. Bear improved 0.475, but
non-Bear wildlife fell 0.500 and total wildlife fell 0.025. Runtime passed at
114.312 seconds per game. The exact beam mechanism is therefore not directly
recoverable by this bounded public sampler, and it is not a qualified MLX
teacher. The exact beam's own Salmon and Fox losses point to the next bounded
question: preserve multiple wildlife portfolios during beam pruning rather
than retaining states by one scalar heuristic.

That pruning hypothesis is now closed. Reserving width-16 capacity across
scalar value, habitat, all five wildlife species, and Nature Tokens changed
only one of ten seed blocks. Portfolio retention scored 94.075 versus scalar
at 94.025, +0.050 with 95% CI `[-0.048,+0.148]`; habitat and tokens gained
0.050 each while wildlife fell 0.050. The treatment remained below 97 and
5.925 points short of target. The next exact diagnostic should change the
searched action set, not reshuffle the same evaluated children: compare W2
against a wider wildlife-diverse frontier under the otherwise frozen focal
beam.

W4 at every focal layer also failed. A +3.250 smoke disappeared over the
frozen pilot: W4 scored 93.075 versus W2 at 94.000, -0.925 with 95% CI
`[-2.426,+0.576]`. Fox gained 1.425, confirming that the extra actions are
structurally different, but Bear fell 1.250, total wildlife 0.550, and habitat
0.400. The fixed beam is harmed when wider root recall and wider future
branching are coupled. The next exact isolation is W4 only at the root with
W2 retained for future focal layers; that tests candidate recall without
letting W4 crowd every beam expansion.

That isolation was null. Root-only W4 gained 0.075 with 95% CI
`[-0.030,+0.180]`; eight of ten blocks tied, habitat and wildlife were flat,
and treatment remained at 94.625. Candidate breadth is closed as the next
exact lever. The unresolved continuation limit is beam capacity or state
evaluation, not W2 root recall.

Beam capacity is now closed too. B32 changed one of ten blocks and gained
only 0.025 with 95% CI `[-0.024,+0.074]`. The exact W2 beam is not losing
material value merely because it retains 16 rather than 32 states. With
candidate breadth, category retention, and raw capacity eliminated, the
remaining search bottleneck is continuation-state evaluation. The next MLX
target should be decision-local beam-state value, trained from counterfactual
terminal outcomes rather than narrow trajectory labels.

That target has now passed its observability prerequisite. Two disjoint R8
batches agreed at 0.9914 on raw candidate value and 0.9365 on centered
advantage across 586 candidates. Top actions agreed 65.625% of the time, while
choosing one batch's winner under the other cost only 0.1133 points on average.
The mean within-group signal range was 3.3965 points, over eleven times the
0.3018 centered batch discrepancy. Unlike prior hidden-seed labels, this is a
repeatable public counterfactual target. ADR 0039 now tests whether MLX can
compress it into a fast final-five policy and convert that signal into score.

ADR 0039 is now closed before test access. The independent scalar afterstate
model reached 0.673 centered correlation, confirming that some relative signal
is learnable, but exact top agreement was only 0.141 and mean regret was 0.728.
Terminal MAE was 2.768 and raw correlation 0.583. The failure is therefore
decision identification and calibration, not target observability. The next
learned experiment must compare the complete legal candidate set jointly and
optimize centered decision advantage; another isolated scalar scorer on the
same inputs is closed.

The joint candidate-set successor improved exactly the diagnosed weakness but
did not cross its frozen gates. Centered correlation rose to 0.789, regret
fell from the immediate baseline's 0.487 to 0.373, and tie-aware recall rose
from 0.297 to 0.352. The required thresholds were 0.35 regret and 0.40 recall.
No sealed-test or gameplay result exists. Since both independent and joint
architectures now fail on the same immutable public target, further network
shape or loss tuning on this corpus is closed. The next experiment must add
information, alter the target, or change the search process.

The decisive online check is also negative. Full public R8/B16 beam scored
92.167 against strong at 92.500 over the frozen three-block qualification:
-0.333 with 95% CI `[-0.987,+0.320]`. It lost 0.500 total wildlife and 1.167
non-Bear wildlife, despite passing its 200.143-second runtime and latency
gates. The treatment fell below the preregistered 92.50 rejection floor.
This means the distillation failures were not hiding a strong deployable
teacher: the repeatable public target itself is too weak online. Further model
or sample tuning on this target is closed. Progress now requires a planning
objective with materially better gameplay strength, not a more faithful
approximation of the same beam.

A complete additive structural-policy search is also closed. Across a frozen
125-point grid and 32 train blocks per point, the best phase-decayed policy
added 0.75 points of Bear-ready credit and no habitat credit. It scored only
+0.125 over production, far below the +0.40 gate. Bear rose 1.023, but Elk,
Salmon, Hawk, Fox, and habitat all declined. The result is useful because it
exhausts the simple scalar explanation: the policy does not merely need a
better fixed coefficient on setup. It needs stateful allocation or a target
that represents competing multi-turn plans rather than summing local
potentials.

The research frontier has now moved materially beyond the promoted product
policy. A
feature-gated historical NNUE/MCE adapter, constrained to canonical V2 root
actions and scored entirely by V2, qualified at 96.350 over ten untouched
games. It beat the then-promoted terminal control by +4.375 with 95% CI
`[+2.938,+5.812]`, won all ten
blocks, and improved both wildlife (+2.350) and habitat (+2.250). Nature
Tokens fell only 0.225, while non-token board score gained 4.600.

This teacher is not product-eligible: its neural evaluator is historical,
non-MLX, and contains a known approximate Elk A scorer. Its value is the
action-selection signal. All 800 final actions were revalidated and executed
canonically in V2, and malformed source records were removed before K32.

The qualified teacher leaves a 3.650-point observed gap to 100. The immediate
question is no longer whether a local 95+ policy exists; it does. The question
is whether its selected actions are representable by the current V2 pattern
frontier. That recall measurement determines whether the next MLX experiment
can reuse the hardened grouped ranker or requires a broader structured action
proposal model.

That representation question is now answered. The production K8+H6+B8
frontier recalled only 51.25% of qualified teacher selections, while the
first 64-action full-legal corpus retained only 25.38% of the teacher's
per-candidate rollout estimates in a measured complete game. Winner-only
imitation therefore hid both the required action breadth and the strength of
preference among near alternatives.

ADR 0053 changes the information, not merely the network shape. A fresh
96-action corpus retains every K32 teacher estimate, its uncertainty and
sample allocation, the complete pattern frontier, and deterministic legal
negatives. The MLX learner optimizes uncertainty-aware pairwise preferences
plus complete-set selected-action recall. This is the next registered attempt
to preserve the 96.35 teacher's actionable signal at interactive inference
cost without inheriting its non-MLX evaluator.

That first distributional attempt is now rejected. It learned meaningful
rollout-value structure, reaching 0.444 scored value-difference correlation,
but its selected top-one was only 13.75% and it left the teacher-scored
frontier 28.59% of the time. The exact immediate baseline itself reached
20.63% top-one, proving that the unanchored scalar discarded useful known
score while learning a noisy continuation correction.

The next target is therefore point-scale continuation residual, not another
attention ablation. Exact immediate score remains a fixed part of the final
prediction; MLX learns only the remaining expected score from the observable
state and action. The full R600 means and uncertainty support this target
directly.

That absolute residual target is now rejected. It reduced fresh-validation
anchored loss from 4.984 to 0.985, but exact selected top-one regressed from
18.91% to 17.19%, value-difference correlation fell from 0.567 to 0.381, and
conditional regret rose from 1.007 to 1.157. No test or gameplay domain was
opened.

The failure is unusually clear: only 0.456% of continuation-residual variance
on the fresh split was within action groups. Almost all absolute target
variance described how much game remained, not which action was better. The
next MLX target must subtract that groupwise nuisance constant and learn
decision-local continuation advantage while preserving exact immediate score.
This is a target-semantic correction, not another architecture or coefficient
ablation.

A development-only screen on the already-open split removed that constant but
still regressed exact top-one to 17.50% and regret to 1.1874. It improved
teacher-frontier coverage and value geometry, so the correction was
mathematically real, but it did not solve action identification. No fresh R600
domain was opened.

Direct model reuse has passed its bounded prerequisite. The qualified
teacher's 11,231-feature, 512-64-1 NNUE is now a checksummed MLX artifact.
Across 260 synthetic inputs and all 80 states of a canonical Rust trajectory,
maximum absolute error was 0.00004197 points; repeated calls were
bit-identical. Batch-32 Apple-GPU throughput was 40,569 evaluations per second.

This removes approximation error from the neural migration question. The next
bounded risk is systems integration: whether sparse features can be collected
and served in sufficiently large batches through a framed Rust/MLX boundary
without changing search order, rollout allocation, canonical V2 action
validation, or the teacher's selected action. Only after end-to-end service
parity and IPC throughput pass should a fresh gameplay domain test whether the
96.35 action signal survives MLX-native execution.

That boundary now passes and the exact MLX teacher reproduced 95.800 mean in
fresh gameplay. Increasing rollout budget, generic root breadth, semantic H6
injection, trajectory-return fine-tuning, joint return/ranking fine-tuning,
and a public open-loop tree all failed their frozen gates. ADR 0069 therefore
tests a distinct remaining representation question: whether a model that sees
the complete candidate set can make decision-local corrections while
preserving the exact parent's useful ordering at initialization. Its
end-to-end R2 smoke passes; the one authorized full R600 validation run is
next.

That run is now closed and negative. The candidate-set residual moved the
teacher winner into its top five more often, but selected top-one improved only
1.875 points, pairwise accuracy fell 0.756 points, value-difference
correlation fell 0.0156, and conditional regret improved only 0.0101. Even
train top-one gained just 2.695 points. The model therefore did not merely
overfit validation; public entity/action features plus scalar parent score and
rank were insufficient to recover the teacher's decision-local correction.
The next bounded representation question was whether the exact parent's
64-dimensional hidden candidate state contained useful distinctions discarded
by its scalar output.

ADR 0070 answered no at the required decision quality. The exact hidden-state
residual reduced fresh-validation loss from 1.522383 to 1.417843, but
selected-action top-one moved only 0.078 percentage point and train top-one
did not move at all. Pairwise accuracy improved 0.405 percentage point and
conditional regret improved only 0.001245 point. Test and gameplay remained
sealed.

The legacy parent branch is therefore closed. The next uncertainty is upstream
of architecture: whether K32/R600 estimates identify stable action winners at
all. A teacher identifiability audit will quantify winner margins relative to
standard errors, selected-action confidence, parent-rank coverage, and
phase-wise noise. Its result will choose between better local teacher
estimation and a fresh MLX-native V2 policy/search representation.

That audit identified the estimator, not the model representation, as the next
bottleneck. Same-budget common random numbers changed only how stochastic
futures are paired across root candidates. The exact MLX R600 smoke gained
1.25, and the three-game pilot gained 1.167 with 95% CI
`[+0.578,+1.756]`, lifting the teacher mean from 95.917 to 97.083. Wildlife
rose 0.583, habitat fell only 0.083, Nature Tokens rose 0.667, and runtime was
flat.

The 20-game confirmation overturned that pilot. Independent scored 95.775 and
CRN scored 95.413, a paired -0.363 with 95% CI `[-1.129,+0.404]` and an
8-1-11 record. Wildlife fell 0.100 and habitat fell 0.350, while Nature Tokens
rose 0.088. Runtime and all integrity gates passed.

The small pilot was a false positive, so same-budget seed coupling is closed.
The strongest confirmed local teacher remains the independent exact-MLX
K32/R600 search at roughly 95.8, leaving about 4.2 points to 100. The next
material lever must change the learned policy/search representation or target,
not resample the same root comparisons differently.

ADR 0073 then isolated representation from target semantics. A fresh
edge-aware graph model reconstructed exact axial adjacency and oriented
matching-terrain edges for all four public boards, but on 64 fresh H6 games it
regressed final-score correlation from 0.393 to 0.342 and MAE from 2.541 to
2.798. Its pairwise log loss improved from 0.763 to 0.730, while pairwise
accuracy moved only from 64.74% to 65.39%.

This closes geometry alone on single realized trajectory outcomes. The useful
next signal is counterfactual expected return: multiple public-information
continuations per state, explicit remaining supply, and opponent demand. That
target should expose decision-local value while preserving uncertainty,
rather than asking a larger encoder to explain stochastic market evolution
from one terminal sample.

ADR 0074 measured that target directly. Sixteen full H6 continuations from
each of 160 fresh public states showed that R8 is already a strong estimator:
0.487-point MAE to R16, 91.14% same-round ordering accuracy, and a projected
13.86-hour cost for 256 games. One factual trajectory was much noisier at
1.335-point MAE to the R16 mean.

Absolute state value still failed for a different reason. The R16 expected
totals had only 1.945 points of standard deviation, below the frozen 2.0
signal-width gate. Averaging removes random market noise but leaves the same
narrow game-level target. The next useful quantity is centered candidate
advantage within one decision: evaluate several observable afterstates with
shared R8 continuations, subtract the group baseline, and learn the local
choice rather than the compressed absolute total.

ADR 0075 tested that quantity on 32 fresh decisions. Shared R8 continuations
were accurate enough for training labels: 0.274 centered MAE to R16, 0.855
correlation, 89.58% pairwise accuracy, 81.25% winner agreement, and only 0.057
points of mean regret. The shallow H6 choice matched the R16 winner only
56.25% of the time, showing that complete continuation contains corrective
decision signal.

The selected action plus its three nearest ranked alternatives were still too
compressed. Their mean R16 range was 1.367 points, below the frozen 1.50 gate,
so no corpus or model was authorized. The next target must preserve the stable
R8 estimator while selecting rank-stratified alternatives from across H6's
existing frontier. That asks whether broad legal contrasts supply enough
learning signal without changing the search teacher or revisiting hidden-state
assumptions.

ADR 0076 answered the width question positively. Selecting the H6 choice plus
the highest, median, and lowest remaining ranked actions increased mean R16
range to 2.803 points. R8 preserved 0.353 centered MAE, 0.931 correlation,
85.42% pairwise accuracy, and only 0.145 points of mean winner regret, with a
projected 7.10-hour 160-game collection.

The exact winner gate still failed by one decision: 20 of 32 R8 winners
matched R16, or 62.50%, against 65% required. That rejects R8 for this broader
candidate set without undoing the central finding that the target now has
adequate width. The next isolated question is sample budget: qualify R12 on
fresh validation games, preserving the candidate selection, continuation
policy, thresholds, and public-information boundary.

ADR 0077 resolved that estimator question on fresh games. R12 reached 0.204
centered MAE, 0.968 correlation, 92.19% pairwise accuracy, 78.13% exact winner
agreement, and 0.037 mean regret. On the same groups, R8 reached 56.25% exact
agreement and 0.283 regret. Mean R16 range remained healthy at 2.469 points,
and a 160-game R12 corpus projects to 10.33 local hours.

For the first time in this research sequence, the public-information target,
candidate contrast, estimator stability, uncertainty, and local collection
cost all pass together. The next evidence-bearing step is no longer another
target audit: it is the authorized local corpus and an MLX ranker trained to
select among the complete four-candidate contrast set.

ADR 0078 now freezes and implements that experiment. The ranker consumes each
explicit observable action afterstate plus the exact public supply snapshot,
compares all four candidates jointly, and begins exactly at immediate score
through a zero-initialized bounded residual. Its implementation-only R12 smoke
completed 384 real continuations and proved Apple GPU train, checkpoint,
resume, and deterministic gate evaluation end to end. The four-group smoke is
not strength evidence; the active evidence-bearing step is the fresh
128-game/32-game corpus and its single preregistered validation run.

The first substantive collection was invalidated before training when a rare
mandatory four-of-a-kind chain produced no legal stabilized market in one
sampled continuation. This was a sampling-semantics defect, not a score
result. The archived 128 train games and 19 partial validation games cannot be
used. H6 and the R12 collector now implement ADR 0018's existing conditional
chance semantics through deterministic rejection of only impossible market
trajectories, and every corrected manifest declares that contract. The exact
failing game 70,019 completed at full R12 after the correction, so the active
evidence step remains one clean recollection followed by the single frozen
MLX run.

Before that validation result was available, ADR 0079 reserved fresh test
indices 71,000-71,031 and froze the same absolute and H6-relative fidelity
gates for the exact selected checkpoint. The test remains sealed unless every
ADR 0078 validation gate passes. This prevents a favorable validation result
from selecting the confirmation corpus, threshold, or acceptance rule. The
conditional handoff is already implemented: authorization must prove the test
path was absent on all three nodes, and the external evaluator must preserve
checkpoint identity and replay validation bit-exactly.

ADR 0078 is now complete and rejected. Its corrected 128-game train and
32-game validation corpora passed every schema, checksum, provenance,
finite-market, public-supply, and shared-seed integrity check. The one frozen
MLX run stopped after five non-improving epochs. The final trained epoch
substantially improved broad ordering, moving pairwise accuracy from 52.48% to
74.64% and centered correlation from 0.750 to 0.788, but it worsened the
registered decision objective, mean regret, and exact top selection.

Because the initial exact-immediate checkpoint remained best, substantive
validation top-value recall was only 44.92% versus H6's 48.24%, and regret was
0.493 versus H6's 0.439. Six frozen gates failed. No retry is authorized, and
ADR 0079 and ADR 0080 closed without opening a test record, inference
service, or gameplay seed.

The result separates target quality from policy learnability. Shared R12
returns are stable and decision-local, and the model can learn much of their
pairwise geometry. That is still insufficient: the learned correction does
not concentrate probability on the best action or beat the shallow H6 choice.
This closes the complete four-candidate public-supply-aware set ranker as the
final route for v2.

The strongest qualified canonical-engine policy remains the exact MLX
K32/R600 historical evaluator at roughly 95.8. It is below the 100 target and
is a research reference rather than a newly trained v2 model. The final
evidence step is therefore the sealed 1,000-game final-domain benchmark,
distributed across all three local Macs, with an explicit unmet-target verdict
unless that held-out mean reaches 100.
