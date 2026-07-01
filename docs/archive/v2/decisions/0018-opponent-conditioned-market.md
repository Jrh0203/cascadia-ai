# ADR 0018: Opponent-Conditioned Wildlife Market

Status: rejected on 2026-06-11 after the registered pilot.

## Context

The phase-capped two-turn commitment policy found real cross-turn signal:
+0.650 total and +1.900 Bear over ten games. It failed because it lost 0.950
non-Bear wildlife and 0.650 habitat. Its opportunity model treats the acting
player's next market as four draws from the complete unplaced supply, even
though three opponents draft first.

The corrected entity and action-delta terminal rankers both learned broad
ordering but failed untouched best-action fidelity. Another encoder change is
not justified. The next controlled question is whether the commitment signal
becomes globally useful when first-rotation market competition is priced
directly.

## Decision

Keep the exact K8+H6+B8 frontier, exact immediate base score, two-personal-turn
phase cap, and unweighted scoring. Replace only the first future-turn market
availability calculation.

After each candidate:

1. Build the observable post-placement, pre-refill public state.
2. Infer wildlife-bag species counts from the fixed 20-per-species supply,
   all public boards, and visible remaining market tokens.
3. Enumerate the exact without-replacement refill distribution.
4. Resolve automatic four-of-a-kind replacement exactly.
5. Before each modeled turn, apply one free three-of-a-kind replacement when
   available, matching the production policy.
6. Simulate the three intervening opponents in seat order. An opponent drafts
   a visible species maximizing its exact one-token marginal base-score gain
   on that opponent's board. Equal-valued species are selected in proportion
   to visible token multiplicity.
7. Refill and stabilize the wildlife market after each opponent draft.
8. At the acting player's next turn, take the expected maximum of the frozen
   per-species continuation values over the resulting market.

For a two-turn continuation, each species value is its exact best immediate
placement plus the existing without-replacement opportunity for the final
future personal turn. Thus the experiment corrects competition for the first
full table rotation only. It deliberately does not model habitat-tile pairing,
opponent tile placement, paid wipes, or a second opponent rotation.

The complete expectation is deterministic, public-information-only, and
independent of hidden bag order. There are no learned weights or
species-specific constants.

Repeated automatic four-of-a-kind replacements are summed with a finite
combinatorial dynamic program. Rejected monochrome groups are returned only
after stabilization, so the final bag depends on the final stable market but
not on rejection order; grouping paths by per-species rejection count removes
the exponential ordered tree without approximating its probability.

The canonical engine can fail only on a path that sets aside enough successive
four-of-a-kind groups to leave fewer than four tokens available for refill.
Such paths have no legal terminal market. The expectation tracks their missing
probability mass explicitly and conditions on successful canonical
stabilization. Production-scale balanced supplies conserve mass to within
`1e-12`; the edge case is documented rather than assigned an invented score.

## Required Tests

- exact draw probabilities conserve mass;
- the closed-form four-kind calculation matches exhaustive ordered
  enumeration on a small reference bag;
- free three-of-a-kind and automatic four-of-a-kind replacement preserve all
  wildlife tokens;
- an opponent with uniquely highest Bear marginal removes Bear when present;
- hidden wildlife-bag redetermination leaves the complete candidate ranking
  byte-for-byte unchanged;
- selected actions are legal and seeded reproducibly;
- zero remaining future turns produces zero opportunity.

## Experiment

Strategy ID:
`pattern-competition-v1-k8-h6-b8-m4-t2-first-rotation`.

The mandatory runtime smoke uses seed 25999 and must finish within five
treatment seconds. A passing implementation runs seeds 26000-26009 and
requires:

- paired mean delta at least +0.5;
- Bear delta at least +0.5;
- total wildlife delta at least 0.0;
- aggregate non-Bear wildlife delta at least -0.5;
- habitat delta at least -0.5;
- treatment runtime at most five seconds per game.

Only a passing pilot may run the frozen 50-game confirmation on seeds
26100-26149. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same category and runtime guardrails.

## Result

The first exact implementation scored the same seed-25999 treatment result as
the optimized implementation but required 5.206 seconds. Factoring the
four-kind generating-function arithmetic once per bag preserved the 94.0
treatment mean and -0.25 delta exactly while reducing runtime to 3.348 seconds,
so the behavior-preserving smoke passed.

The frozen ten-game pilot then scored 92.350 versus 91.475:

- paired delta +0.875, 95% CI -0.151 to +1.901, record 6-0-4;
- Bear +1.275;
- habitat +0.400;
- Nature Tokens +0.725;
- total wildlife -0.250;
- aggregate Elk, Salmon, Hawk, and Fox -1.525;
- treatment runtime 10.316 seconds per game.

The experiment failed the total-wildlife, non-Bear, and runtime gates. The
positive score signal confirms that first-rotation competition matters, but
an expected-best-species continuation still reallocates too much value into
Bear and is too expensive under parallel evaluation. No confirmation was run.
