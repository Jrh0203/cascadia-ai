# Full-Legal Public Decision Cache Preregistration

Status: **closed - accepted**

Date: 2026-06-15

## Question

At each realized-hidden checkpoint, the diagnostic follows every
high-confidence finalist to the terminal state. The branch beginning with the
champion action deterministically reproduces the public-state path that the
outer audit then plays again. Later champion branches also revisit suffixes
already solved by the earliest diagnostic.

Can one exact audit-scoped public-decision cache remove those repeated R600
searches without changing any action, score, terminal state, search contract,
or hidden-information boundary?

## Frozen Treatment

Add one cache with game lifetime inside the full-legal audit collector:

- key by the canonical public-state BLAKE3 digest only to select a bucket;
- require exact `PublicGameState` equality before every hit;
- store the selected canonical `TurnAction`;
- insert decisions produced by ordinary champion play, audited champion play,
  and only the realized-hidden continuation rooted at the outer champion
  action;
- evaluate and count every non-champion finalist continuation normally, but do
  not retain its decisions because those counterfactual states cannot appear
  on the outer trajectory;
- permit ordinary `select_action` calls to reuse an exact cached action;
- continue evaluating `select_action_with_estimates` in full because audited
  decisions require the complete R600 frontier;
- if an evaluated action disagrees with an existing exact-state entry, fail
  the audit;
- keep the cache local to one game and never persist it across seeds;
- omit hidden tile-stack and wildlife-bag order because the qualified teacher
  consumes only public state and is already required to be hidden invariant.

The treatment changes no model, weights, candidates, rollout budget, random
stream, action ordering, public-information contract, or rules behavior.

The temporary qualification switch is
`--disable-public-decision-cache`. If accepted, remove the switch and uncached
production branch.

## Correctness Gates

1. Dedicated tests prove exact-state equality is required even under a forced
   digest collision and that hidden reorderings of one public state reuse the
   same entry.
2. Every cached lookup returns the exact action produced by a fresh teacher
   evaluation for the same public state.
3. The frozen seed-60999 report preserves all game records after timing fields
   are removed, including every champion action, finalist result, score
   breakdown, realized winner, and terminal-state digest.
4. The final scores remain `[96,99,92,102]`.
5. All evaluated searches remain exact R600/R1200/R4800 with zero bootstrap
   samples, zero policy fallbacks, and clean shutdown.

## Performance Gates

Advance only if:

1. the frozen full seed records at least 50 exact cache hits;
2. evaluated ordinary policy decisions fall by at least 10%;
3. complete uncontended wall time improves by at least 5% on both john2 and
   john3 in opposite-order confirmation;
4. combined wall time improves by at least 7.5%;
5. maximum RSS and allocator peak footprint do not regress by more than 10%;
6. the result remains positive in a final switch-free production run.

If the mechanism passes but complete improvement is below 7.5%, reject it as
insufficient leverage for the remaining 7.329x teacher gap.

## Interpretation

Acceptance would remove exact duplicate outer searches. It would not solve
the remaining independent finalist branches; those still require coordinated
multi-trajectory batching and neural/native work elimination.

Intermediate measurements authorize no large audit collection.

Final evidence:
[`full-legal-audit-public-decision-cache-acceptance-v1.md`](full-legal-audit-public-decision-cache-acceptance-v1.md).

## Qualification Amendment

The first complete local qualification run preserved exact semantics and
recorded 118 hits, but retaining every finalist continuation left 920 entries
resident even though only the outer-champion branch was ever reused. A compact
canonical-byte representation reduced that footprint but still exceeded the
preregistered maximum-RSS gate.

Before any cross-host confirmation, retention is therefore narrowed to the
single realized-hidden branch whose root action equals the audited outer
champion action. All finalists remain fully evaluated and included in request
and evaluation counts. The correctness and performance gates above are
unchanged.

The initial remote preflight also showed that an environment-variable switch
would correctly be rejected by the legacy teacher's frozen-environment
validator. Before any timed confirmation completed, the qualification control
was moved to the explicit `--disable-public-decision-cache` audit CLI flag so
the teacher environment remains unchanged.
