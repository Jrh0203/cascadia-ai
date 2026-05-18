# v6-bounded: shrink per-cell feature blocks to active play region

**Status**: design doc, not implemented.
**Date**: 2026-04-19
**Author**: Claude session

## Motivation

Empirical analysis of 50,000 game states from `nnue_weights_v5sh_iter40.bin` selfplay
shows the 21×21 hex grid (441 cells) is wildly oversized for actual play:

| Hex distance from grid center (0,0) | # cells | % of all per-cell firings |
|---|---|---|
| ≤ 3 | 37 | 92.3% |
| ≤ 4 | 61 | 98.3% |
| ≤ 5 | 91 | 99.7% |
| **≤ 6** | **127** | **99.9%** ← chosen radius |
| > 6 | 314 | 0.1% |

**63% of cells (278/441) are NEVER used** in 50K game states. Training continues
to allocate weights to those rows of `w1` which see no gradient signal — pure
waste of parameters and memory bandwidth.

Games are naturally centered at hex (0,0) because all 5 starter clusters in
`game.rs:24-70` use fixed coordinates (0,0), (0,1), (1,0). No translation
normalization is needed; just shrink the active region.

## Goal

Reduce per-cell feature blocks by ~71% while preserving 99.9% of feature
information. Total feature count: **17,115 → ~9,500 (-44%)**.

Secondary win: opens room for HabitatBucket conditioning (P1 from earlier
proposal) at full scale: 39,690 → 11,400 features, finally feasible at
current data scale.

## Design

### Constants

```rust
pub const LOCAL_RADIUS: i8 = 6;
pub const LOCAL_CELLS: usize = 127;  // 1 + 6 + 12 + 18 + 24 + 30 + 36

pub const FEATURES_PER_CELL: usize = 11;
pub const CELL_FEATURES: usize = LOCAL_CELLS * FEATURES_PER_CELL;  // 1397

pub const ALLOWED_WL_PER_CELL: usize = 5;
pub const ALLOWED_WL_FEATURES: usize = LOCAL_CELLS * 5;  // 635
pub const SEC_TERRAIN_FEATURES: usize = LOCAL_CELLS * 5;  // 635
```

### Lookup tables

Build at init time (lazy_static or const-fn):

```rust
/// Maps global hex grid index (0..441) → local index (0..127) or -1 if outside.
pub static GLOBAL_TO_LOCAL: LazyLock<[i16; 441]> = LazyLock::new(|| {
    let mut tbl = [-1i16; 441];
    let mut next = 0u16;
    // Hex distance from center: max(|q|, |r|, |q+r|)
    // Iterate cells in spiral order so local indices have predictable layout
    for d in 0..=LOCAL_RADIUS {
        for col in 0..GRID_SIZE {
            for row in 0..GRID_SIZE {
                let q = col as i32 - GRID_CENTER as i32;
                let r = row as i32 - GRID_CENTER as i32;
                let dist = q.abs().max(r.abs()).max((q + r).abs());
                if dist == d as i32 {
                    let global = col * GRID_SIZE + row;
                    tbl[global] = next as i16;
                    next += 1;
                }
            }
        }
    }
    tbl
});

/// Inverse: local index (0..127) → global hex grid index (0..441).
pub static LOCAL_TO_GLOBAL: LazyLock<[u16; LOCAL_CELLS]> = ...;
```

### Per-cell feature extraction

Wrap every `cell_idx * FEATURES_PER_CELL` in a translation step:

```rust
#[inline]
fn local_cell_base(global_idx: usize) -> Option<usize> {
    let local = GLOBAL_TO_LOCAL[global_idx];
    if local < 0 { return None; }  // outside active region
    Some(local as usize * FEATURES_PER_CELL)
}

pub fn cell_features(board: &Board, idx: usize) -> ArrayVec<u16, 2> {
    let mut features = ArrayVec::new();
    let cell = board.grid.get(idx);
    if !cell.is_present() { return features; }
    let base = match local_cell_base(idx) {
        Some(b) => b,
        None => return features,  // edge case: tile placed outside radius
    };
    // ... emit features at `base + offset` (unchanged logic)
}
```

Same pattern for:
- `cell_features_before_wildlife`
- `compute_placement_diff` (per-cell + pairwise updates)
- `extract_features_with_bag` (every per-cell loop)
- `extract_v5_frontier_features` (frontier emission)
- All v2/v3 per-cell blocks

### Edge case handling

If a tile lands at hex distance > LOCAL_RADIUS:
- **Per-cell feature**: silently drop (no `base` to emit at)
- **Pairwise adjacency feature**: still fires (relative direction encoding doesn't care about absolute position)
- **Patterns/bag/opponent features**: unaffected (don't depend on per-cell indexing)

Net loss for the 0.1% of out-of-region tiles: their wildlife/terrain identity
isn't directly observable to the value function, but adjacency to in-region tiles
still informs the patterns. Acceptable.

### Feature block layout (new)

```
[0 .. 1397)       Per-cell core (127 × 11)              [-71%]
[1397 .. 1408)    Phase features (110, unchanged)
[1408 .. 1555)    Pairwise wildlife adj (147, unchanged)
[1555 .. 1644)    Wildlife pattern (89, unchanged)
[1644 .. 1699)    Bag remaining (55, unchanged)
[1699 .. 1754)    Opp habitat (55, unchanged)
[1754 .. 2389)    Allowed wildlife per cell (127 × 5)   [-71%]
[2389 .. 2439)    Wildlife count ext (50, unchanged)
[2439 .. 2547)    Terrain pairwise (108, unchanged)
[2547 .. 3182)    Sec terrain per cell (127 × 5)        [-71%]
[3182 .. ...)     v2 blocks (mostly unchanged)
... (similarly shrunk per-cell pieces)
```

Total: ~9,500 features (was 17,115).

## Cargo feature gating

```toml
# In crates/cascadia-ai/Cargo.toml
v6-bounded = []  # Shrinks per-cell blocks to LOCAL_CELLS=127. Breaks v5 backward
                 # compatibility. New weights, new file format magic.
```

The Rust code uses `#[cfg(feature = "v6-bounded")]` to swap between full-grid and
bounded constants. This avoids breaking v5sh weights — they continue to work with
the existing binary.

A v6 binary would build at `target-mid-v6/release/cascadia-cli` to keep separate
from the v5 champion binary.

## Training pipeline changes

1. **Selfplay shards**: existing MCV4 selfplay shards from v5sh_iter*.bin **CANNOT** be
   reused. Feature indices changed. Need fresh selfplay from a v6-binary worker.

2. **PyTorch trainer (`train_pytorch.py`)**: needs `--num-features` flag updated
   per the new total. Also the augmentation remap tables (`build_cell_remap`)
   need to use `LOCAL_TO_GLOBAL` mapping. ~30 lines change.

3. **Init weights**: cannot zero-pad from v5sh (incompatible feature layout).
   Must train from scratch.

4. **Modal images**: new image build with `--features mid-features,v4-opp,v6-bounded`.
   ~5 min cargo rebuild.

## Validation plan

1. **Unit tests for lookup tables**:
   - `GLOBAL_TO_LOCAL[GRID_CENTER * GRID_SIZE + GRID_CENTER]` should return 0 (origin = local 0).
   - All cells with hex distance ≤ 6 from center should have valid local idx.
   - Cells at distance > 6 should map to -1.
   - Round-trip: `LOCAL_TO_GLOBAL[GLOBAL_TO_LOCAL[i]] == i` for in-region cells.

2. **Smoke test**: build v6 binary, run 10 selfplay games, verify MCV4 shards
   produced and have expected feature count.

3. **From-scratch training**: 5 iters × 30K games (Modal). RMSE trajectory should
   converge similarly to v5sh's first 5 iters (which hit ~5.0).

4. **HH validation**: v6sh_iter5 vs v5sh_iter40 (current champion).
   - If competitive: continue to iter 10/20+
   - If much worse: investigate (likely indexing bug)

5. **Compare per-iter RMSE convergence rate** between v5 and v6:
   - v5sh trained on 17K features, hit 4.32 RMSE at iter 20
   - v6 trained on ~9.5K features, should converge faster per iter (fewer params)
     but may plateau higher (less capacity for edge cases)

## Risks / open questions

| Risk | Severity | Mitigation |
|---|---|---|
| Off-by-one in GLOBAL_TO_LOCAL | High (silent corruption) | Comprehensive unit tests + assertion in extract_features |
| Indexing changes break translation augmentation | Medium | Update `build_cell_remap` in train_pytorch.py to use local indexing |
| Edge case tile loss hurts play (<0.1%) | Low | Document; can fall back to global indexing for outliers if needed |
| PyTorch trainer hard-coded NUM_FEATURES break | Low | Pass --num-features explicitly |
| Modal image cache invalidation | Low (just slow) | Accept ~5 min build cost |
| Convergence might plateau higher than v5 | Medium | If so, the saved budget can fund HabitatBucket addition (next step) |

## Implementation order (when given green light)

1. Add `v6-bounded` cargo feature to both crates
2. Implement lookup tables + unit tests in `crates/cascadia-ai/src/nnue.rs`
3. Refactor every per-cell extraction call site
4. Update Rust unit tests for feature extraction
5. Build v6 binary at `target-mid-v6/`
6. Smoke test: 10-game selfplay, confirm MCV4 written, feature count correct
7. Update `train_pytorch.py` (NUM_FEATURES default, augmentation tables)
8. Update Modal image (`overnight/selfplay_fsp_v6_modal.py`)
9. Run from-scratch training: 5 iters × 30K games on Modal
10. HH validate v6_iter5 vs v5sh_iter40
11. If competitive, run another 10-15 iters with LR decay (mirror v5sh recipe)
12. Final HH validation at iter 20

**Total effort estimate**: ~1 day Rust + Python implementation + ~12 hours Modal compute (~$15) for full convergence + HH.

## Followup work (not in v6 scope)

- **Add HabitatBucket conditioning** (P1) on top of v6 since the saved feature
  budget makes it feasible: 6,615 features added, total stays ~16K (still
  smaller than current v5).
- **Bigger NNUE (HIDDEN1=1024)**: same total parameters as v5sh due to feature
  reduction, but more representational capacity per input.
- **Per-cell distance-to-wildlife features** (P9 from earlier proposal):
  127 cells × 5 wildlife × 4 bins = 2,540 features. Now affordable.

## Open questions for the user

1. **LOCAL_RADIUS**: 6 (127 cells, 99.9% coverage) or 5 (91 cells, 99.7% coverage)
   to be even more aggressive? At R=5 we'd save another ~28% on per-cell blocks.

2. **Edge case tile**: silently drop vs fall back to global indexing? Drop is
   simpler; fallback adds complexity for 0.1% of cases.

3. **Train v6 from scratch (50K-game iters like v5sh) or go straight to a richer
   architecture (HabitatBucket + bigger NNUE)?** The latter is higher-EV but
   slower to validate.

4. **When**: schedule v6 training for "after M3 Ultra arrives" (faster locally) or
   continue to use Modal for parallel scale?
