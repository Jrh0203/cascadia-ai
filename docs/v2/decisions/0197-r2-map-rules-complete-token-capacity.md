# ADR 0197: Rules-Complete Live R2-MAP Token Capacity

Status: accepted before W0 source freeze

## Decision

Keep the historical sparse-foundation MLX cache frozen at 92 padded tokens per
board, but version the live R2-MAP Rust/Python/service boundary to 139 tokens
per board (4x139 per public state). No token is truncated, saturated, reused,
or omitted, and no legal action is pruned.

The grouped and public-market service frame version is 3. Their request
schemas are respectively `r2-map-grouped-exhaustive-request-v3` and
`r2-map-public-market-decision-request-v3`. A v3 service rejects the obsolete
4x92 live shape. Their canonical BLAKE3 identities explicitly bind 4 boards,
139 rows, and 60 token features and are respectively
`bce9b1e6701dd86debc7a0fae496e6e55d72acac554eb572dcdcbf5356b6b8fa`
and
`68ca4d115e6a2ca5981a75d8c979752efbd8b1d466fb25afa278e9de103e9082`.

## Why 139 is complete

Let `n` be the number of occupied tiles on one legal player board.

- `n <= 23`: three starter tiles plus twenty turns.
- occupied tokens: at most `n`.
- frontier tokens: at most `2n + 4`. A one-cell connected polyhex has six
  frontier cells. Adding a connected hex removes its chosen frontier cell and
  can add at most three previously unseen frontier cells, so each addition
  increases the distinct frontier by at most two.
- habitat-component tokens: at most `2n`, because a tile contains at most two
  habitat terrains and therefore introduces at most two components.
- wildlife-motif tokens: at most `n - 3`, because starter tiles contain no
  placed wildlife and each of the twenty turns places at most one token.

Thus the live total is at most

`n + (2n + 4) + 2n + (n - 3) = 6n + 1 <= 6*23 + 1 = 139`.

Rust validates each layer against these bounds before encoding. Python derives
the same constants independently in `r2_map_tensor_contract.py`. Boundary
tests fill slot 138 under all twelve D6 transforms and reject the first
impossible slot 139 rather than truncating it.

## Failure that exposed the defect

The v34 P1 full-100 corpus stopped on a legal wildlife sibling with
`wildlife sibling motif suffix exceeds board capacity`. The old 92 value was
the maximum observed in the 60,000-position R2 sparse-foundation corpus, not a
rules upper bound. The canonical full encoder would have rejected the same
state, so this was a global schema defect rather than an incremental-encoder
bug.

The immutable v36 diagnostic reproduced the first failure at open-corpus game
offset 27, game index `5778487003071774747`, turn 78, D6 transform 9, and legal
width 1,176. The candidate board layers were exactly 23 occupied, 30 frontier,
20 habitat-component, and 20 wildlife-motif tokens: 93 total. Its literal seed
was
`[166,98,109,28,252,10,196,39,3,24,208,213,77,179,118,9,99,239,41,118,105,190,150,216,218,161,205,235,121,222,142,230]`.

The captured 78-turn replay is permanently stored at
`tests/fixtures/r2_map/p1-v34-first-93-token-replay.json` (SHA-256
`3891b6982d55437ad7f3b6f51eb4b4717268f4d455b64eb7aabbd29791762f1f`).
The regression reconstructs that exact pre-decision state, confirms all 1,176
actions remain legal, reaches the 93-token active board without truncation,
and proves byte-for-byte equality between the independent canonical full
encoder and incremental materializer under all twelve D6 transforms.

The immutable diagnostic evidence is rooted at
`reports/runs/run-rust-p1-diag-1631ea6f-v36-full100` on john2. Its source tar
SHA-256 is
`1631ea6fe9c27dd5b2d8bb7166adb5ee799110bc861f7af7a8fbea408e1eb7b6`,
stderr SHA-256 is
`7a7ab26a592c0d2e6bdef61b8bea039b2b4c62c7f041b226a8d150ccd5b92a43`,
and run receipt SHA-256 is
`704149954626a8ffdae6ddb9c49ddabd004c33667c3b722b11d981c74ef064f3`.

## Compatibility and resource gates

Sequence length does not change R2-MAP parameter tensors, so the v1.1 model
parameter count remains unchanged. Archived 4x92 foundation-cache identities
remain reproducible and are not relabeled as live-safe. New live source,
protocol fixtures, Rust request serialization, Python validation, dataset
padding, packing, and maximum-width service tests all use 4x139.

Before W0 freezes, the repaired source must pass targeted Rust/Python protocol
tests, a three-game calibrated P1 gate, a prefix beyond the original failure,
and the full 100-game P1 gate below 4 GiB RSS with zero process swaps and zero
system swap delta.
