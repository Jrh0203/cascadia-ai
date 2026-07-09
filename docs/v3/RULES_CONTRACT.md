# Cascadia Base-Game Rules Contract

This document is the authoritative rules boundary for Cascadia v3. AI,
simulation, search, training data, the API, and the web game must all use the
same transactional implementation in `crates/cascadia-game`.

Rules semantics ID: `cascadia-base-official-2026-07-09`.

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

- Four matching market wildlife are automatically wiped. Set all four aside,
  draw replacements one at a time, repeat if another four-of-a-kind appears,
  and return all set-aside tokens only after overpopulation is stable.
- Three matching market wildlife are optional. Before drafting, the active
  player may decline or may wipe exactly those three once. The wiped tokens are
  set aside while replacements are drawn and return after overpopulation is
  stable.
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

## Scientific compatibility break

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
