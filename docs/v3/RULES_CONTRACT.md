# Cascadia Base-Game Rules Contract

This document is the authoritative rules boundary for Cascadia v3. AI,
simulation, search, training data, the API, and the web game must all use the
same transactional implementation in `crates/cascadia-game`.

Rules semantics ID: `cascadia-base-official-2026-07-16`.
(Supersedes `cascadia-base-official-2026-07-09`; see the compatibility
breaks at the bottom.)

Primary source: the official AEG
[Cascadia rulebook](https://www.alderac.com/wp-content/uploads/2025/02/Cascadia-Rulebook.pdf),
especially the turn summary and placement rules on pages 6–7.

## Wildlife return at the end of a turn

After drafting, a player may decline to place the Wildlife Token or may be
unable to place it legally. The token returns to the Cloth Bag before the
empty market pair is replenished. The refill therefore may draw the same token
again, exactly as it can when playing with the physical bag.

Engine contract:

1. Take the selected tile and wildlife from the market.
2. Place the tile and optionally place the wildlife.
3. If the wildlife was not placed, return it to the bag.
4. Refill the empty market pair.
5. Resolve any automatic four-of-a-kind overpopulation.
6. Complete the turn and pass play clockwise.

`GameState::finish_turn` owns this ordering. The regression test
`unplaced_drafted_wildlife_returns_before_end_of_turn_refill` empties the bag
before a turn and proves that the declined token is available for the refill.

## Wildlife overpopulation

- Four matching market wildlife are automatically wiped. Each resolution is
  completed in full before observing the new market: set the four aside, draw
  replacements, then return the set-aside tokens to the bag. If the refill
  produces another four-of-a-kind, repeat the whole procedure — the previous
  wipe's tokens are already back in the bag.
- Three matching market wildlife are optional. Before drafting, the active
  player may decline or may wipe exactly those three once. The wiped tokens
  are set aside while replacements are drawn and return to the bag as soon as
  that refill completes, before any automatic overpopulation is resolved.

Per-resolution return is a conservation invariant (John's ruling,
2026-07-16): with 100 tokens, at most 80 ever on boards, and 4 in the
market, the bag holds at least 16 tokens at any stable point, and no wipe
sequence may transiently drain it — the cloth bag can never be unexpectedly
empty. The regression test
`consecutive_overpopulation_wipes_near_exhaustion_do_not_drain_the_bag`
proves a near-exhaustion double wipe resolves without error. (Before
2026-07-16 the engine returned set-aside tokens only after overpopulation
was stable, draining the bag 4 tokens per consecutive wipe; a deep
self-play line emptied it and faulted.)
- A Nature Token may still be spent to wipe any selected non-empty subset
  under the existing paid-wipe rules.

The free three-token choice is a policy action. The engine exposes both legal
branches through `GameState::free_three_of_a_kind_choices`, in deterministic
order: decline, then accept. It never chooses a branch for the caller.

Policy-facing consumers must decide from public information before the
replacement draw:

- random chooses the branch before enumerating its draft; greedy and pattern
  policies compare accept/decline values over common public-hash-derived
  hidden-order samples, then rank drafts in the real revealed market;
- rollout and lookahead policies inherit the same explicit market-decision
  boundary before their normal action search;
- CascadiaFormer/Gumbel searches decline once and values accept over common
  hidden-order samples. If accept wins, it then reveals the real replacement
  market and runs a separate downstream draft search;
- Gumbel interior plies enforce the same decision → chance → draft ordering;
- the API and web UI retain explicit human accept/decline control.

Separate model rows are mandatory for sampled chance outcomes and the final
revealed market. The accept/decline choice itself must be invariant to the
actual hidden bag order; only the downstream draft may change with the market
revealed after acceptance.

## Scientific compatibility breaks

**2026-07-16 (`cascadia-base-official-2026-07-16`, engine rev `45fb5072`):**
overpopulation wipes now return set-aside tokens per resolution instead of
after the loop stabilizes. Trajectories are bit-identical unless a game
contains consecutive four-of-a-kind wipes (or an automatic wipe following a
voluntary wipe), so the practical divergence is rare — but any divergence
breaks paired seed comparability, so artifacts stamped with the 2026-07-09
identity are a closed historical evidence boundary. New scientific runs must
stamp `..._rules_2026_07_16`; do not mix identities in a paired confidence
interval. The historical champion number (98.2975) belongs to the 07-09
identity and needs a fresh canonical battery under 07-16 before use as a
paired control.

**2026-07-09 (`cascadia-base-official-2026-07-09`):**
All reports, checkpoints, corpora, and score baselines produced before this
rules contract used a policy stack that automatically accepted a free
three-of-a-kind refresh. They are legacy evidence and are not valid controls
for promotion under the corrected action space.

New artifacts must record the updated ruleset/config identity. Rebaseline
greedy, no-search model play, and every promoted Gumbel serving configuration
before making strength claims. Do not mix pre-fix and post-fix games in a
paired confidence interval. Benchmark reports must retain per-decision
accept/decline telemetry and summarize the acceptance rate so a score claim is
auditable against the corrected policy boundary.
