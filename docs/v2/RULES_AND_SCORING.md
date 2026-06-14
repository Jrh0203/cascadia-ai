# Rules And Scoring

Primary source: the [official 2021 AEG rulebook][rules].

[rules]: https://www.alderac.com/wp-content/uploads/2021/08/Cascadia-Rules.pdf
[salmon-guidance]: https://news.direwolfdigital.com/cascadia-animal-spotlight-chinook-salmon/

## Canonical Turn

V2 represents a complete turn as one transactional `TurnAction`:

1. Optionally wipe exactly three matching wildlife tokens, once.
2. Spend zero or more Nature Tokens to wipe any nonempty wildlife subsets.
3. Draft a paired tile and token, or spend one Nature Token to take any tile
   and any token. A same-slot independent draft is legal, though the equivalent
   paired draft produces the same selection without spending the token.
4. Place the tile adjacent to the environment at one of six rotations.
5. Place the wildlife on any compatible empty tile, or return it to the bag.
6. Refill only the vacated tile and wildlife positions.

All four matching wildlife tokens are wiped automatically, repeatedly if
necessary. Wiped tokens are set aside during replacement, then returned to a
random position in the bag. Invalid actions leave the original state unchanged.
If an optional free three-of-a-kind replacement would enter a repeated
four-of-a-kind chain with too few drawable tokens to produce a stable market,
the optional replacement is declined. The original market remains legal and
unchanged; the engine does not invent replacement tokens or score an
impossible terminal market.

Stochastic search and training teachers condition on legal stabilized markets.
If a sampled hidden order reaches an impossible mandatory replacement chain,
that complete hidden trajectory is rejected and deterministically resampled;
it is never assigned a fabricated terminal score. Dataset manifests version
this behavior explicitly.

Tile and wildlife display slots are stored separately. This is essential for
independent drafting: the unchosen token beside the selected tile and the
unchosen tile beside the selected token remain where they are.

## Setup

| Players | Mode | Habitat tiles selected | Turns |
|---:|---|---:|---:|
| 1 | Solo | 43 | 20 |
| 2 | Standard | 43 | 40 |
| 3 | Standard | 63 | 60 |
| 4 | Standard | 83 | 80 |

Solo follows the two-player tile setup. After each draft, the furthest remaining
tile and wildlife are discarded, the other two slide away from the draw stack,
and two new tiles and wildlife are drawn.

## Scoring

All A-D cards for bear, elk, salmon, hawk, and fox are implemented. Card values
were transcribed from page 11 of the official rulebook and encoded in tests.

Notable independently identified corrections to v1:

- Bear D scores 13, not 14, for a group of four.
- Hawk B requires a hawk to be nonadjacent to every other hawk and to have line
  of sight to another hawk.
- Elk B's four-elk shape is the compact pictured diamond.
- Elk A uses exact maximum-score partitioning rather than greedy longest-line
  assignment.

Salmon D's three-token minimum was cross-checked against
[Dire Wolf Digital's official Cascadia rules guidance][salmon-guidance] because
the printed rulebook text is terse.

`ScoreBreakdown.base_total` includes wildlife, largest habitat corridors, and
unused Nature Tokens. Habitat majority bonuses are reported separately and are
excluded from the primary research metric.

## State Guarantees

Every stable game state validates:

- 85 standard habitat tiles are accounted for across stack, market, boards,
  excluded tiles, and solo discards.
- 100 wildlife tokens are accounted for across bag, market, boards, and solo
  discards.
- no automatic four-token overpopulation remains unresolved;
- board occupancy and placement indexes agree;
- player, turn, and schema bounds are valid.

The v2 grid supports the full 20-turn maximum-distance chain that the v1
21-by-21 grid could not represent.
