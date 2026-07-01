# ADR 0196: Public-universal R2-MAP market legality

Status: accepted

Date: 2026-06-18

Protocol note: ADR 0197 leaves these legality/action semantics unchanged and
supersedes only the live request envelope to
`r2-map-public-market-decision-request-v3` with 4x139 parent tensors. The v2
request hash below remains predecessor evidence.

## Context

An R2-MAP turn has two public market stages before the exhaustive draft:

1. keep or replace a visible three-of-a-kind; and
2. stop or spend one nature token to wipe any non-empty subset of the four
   visible wildlife tokens, repeated after each refill.

The wildlife bag's species counts are public, but its order is hidden. A
choice that happens to complete under the simulator's current hidden order is
not necessarily legal at the corresponding public information set. Near bag
exhaustion, another hidden permutation of the same public multiset can require
an additional automatic four-of-a-kind replacement that the bag cannot fill.
Advertising that action would leak hidden order through the legal-action mask.
It could also make the model choose an action that fails when committed after
redeterminization.

ADR 0018 and ADR 0078 allow exact replay and offline teacher/search machinery
to condition on already observed staged public contingencies. That does not
authorize a live public policy to condition the legality of its next choice on
an unrevealed refill or to resample hidden order until a chosen action works.

## Decision

Free replacement and every paid-wipe subset use
`public-universal-hidden-order-intersection-v1` legality. An action is exposed
if and only if its refill is guaranteed to produce a stable four-token market
for every hidden ordering consistent with the public wildlife-count vector.

`KeepThreeOfAKind` is always the first free-stage action.
`ReplaceThreeOfAKind` follows only when the universal proof succeeds.
`StopWiping` is always the first paid-stage action. Safe non-empty wipe masks
then appear in ascending numeric order from `0x01` through `0x0f`. A paid mask
is considered only when the active player has a nature token. A committed
wipe spends exactly one token, reveals exactly one refill transition, and
starts a new paid decision point with a new public parent identity.

The proof operates only on:

- the five public bag species counts;
- the four visible market species; and
- the candidate replacement mask.

After retaining unmasked wildlife, production applies the equivalent
constant-space O(5) theorem:

- with at least two retained species, a four-of-a-kind is impossible and every
  feasible refill stabilizes;
- with one retained species, only the all-matching completion can reject; after
  subtracting it, the empty-market theorem applies; and
- an empty market with public counts `c` and total `T` is universally safe
  exactly when `T >= 4` and `sum(c[i] // 4) < T // 4`.

Non-monochrome outcomes are immediately stable. Insufficient tokens for the
current refill fail the proof. Tokens in an automatically rejected cohort stay
out of the remaining bag until the market stabilizes, matching the canonical
simulator. The recursive multiset implementation remains test-only as an
independent oracle; bounded exhaustive equivalence and ordered hidden-bag cases
guard the closed form used by every service request.

There is no conditional resampling, rejection sampling, hidden-order peek, or
fallback from a failed commit. Teacher/search code may evaluate a frozen,
already-observed public contingency under its own versioned protocol, but that
conditioning is not used to widen live inference legality.

## Protocol binding

The cross-language fixture is
`tests/fixtures/r2_map/public-market-decision-protocol-v2.json`. It binds:

- request schema `r2-map-public-market-decision-request-v2`, BLAKE3
  `cd9d05b92c6fb173119487fa6ae9aeec5cdf6d7d96caf58a5ef7884b94d1ae48`;
- response schema `r2-map-public-market-decision-response-v2`, BLAKE3
  `d0d00527f75b7bbcb868433da7cb9f2cd415d0f9fb4e7591a966e51728c708c5`;
- action schema BLAKE3
  `e9ab2382f20f4ea440591adba6021d85f8be83ff4e483513b05d46e5f285cd38`;
  and
- legality identifier `public-universal-hidden-order-intersection-v1`.

Rust independently reconstructs the complete ordered action surface from the
public counts and market in each request. It rejects omitted, extra, reordered,
duplicated, stale-schema, or resource-inconsistent candidates before model
scoring. A serving bundle freezes the collector, source, and serving protocol
identities; a runner refuses a request from a stale source protocol.

## Verification

The release gate requires:

- exhaustive agreement with a brute-force multiset enumerator over bounded
  count cubes;
- identical legal masks for adversarial hidden permutations sharing one public
  state;
- successful commit of every advertised action under every such permutation;
- explicit short-bag cases where only a candidate-specific subset is safe;
- repeated four-of-a-kind exhaustion where free replacement disappears;
- Rust/Python fixture byte and schema-hash parity;
- sequential transition and bundled `TurnAction` replay parity; and
- stale serving-bundle/source rejection.

## Consequences

- The legal-action mask is a function of public information only.
- Short bags can expose fewer than all fifteen paid subsets, including safe
  masks smaller than four slots; the old total-count shortcut is removed.
- Some actions that would succeed under the current hidden order are
  intentionally unavailable because they are not valid across the public
  information set.
- Each refill remains a causal boundary: later choices may use the revealed
  public market, while earlier choices cannot observe it.
- The canonical simulator, dataset exporter, Rust runner, Python service, and
  replay validator share one explicit ordered action identity.
