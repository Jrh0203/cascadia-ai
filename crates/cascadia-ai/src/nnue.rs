//! NNUE (Efficiently Updatable Neural Network) for Cascadia board evaluation.
//!
//! Architecture: sparse binary inputs → 512 ReLU → 64 ReLU → 1 scalar
//!
//! Feature layout:
//!   [0 .. 4851)       Per-cell: 441 cells × 11 features (wildlife/terrain primary)
//!   [4851 .. 4872)    Turn number one-hot (21, 0-20)
//!   [4872 .. 4881)    Nature tokens one-hot (9, 0-8)
//!   [4881 .. 4911)    Wildlife count per type, legacy (6 bins × 5 types, counts 0-5)
//!   [4911 .. 4961)    Largest habitat per terrain (10 bins × 5 terrains, 0-9)
//!   [4961 .. 5108)    Pairwise adjacency: 3 dirs × 49 wildlife pair states
//!   [5108 .. 5197)    Wildlife pattern features (bear pairs, elk lines, etc.)
//!   [5197 .. 5252)    Bag remaining: 5 types × 11 bins (0-10+)
//!   [5252 .. 5307)    Opponent habitat: 5 terrains × 11 bins (max opponent 0-10+)
//!   [5307 .. 7512)    Allowed wildlife per cell: 441 cells × 5 flags
//!   [7512 .. 7562)    Extended wildlife count: 10 bins × 5 types (0-9)
//!   [7562 .. 7670)    Terrain pairwise: 3 dirs × 36 terrain pair states
//!
//!   ─── v2 features (appended for backward compat) ───
//!   [7670 .. 9875)    SECONDARY terrain per cell: 441 × 5 = 2205 (fixes dual-terrain blindness)
//!   [9875 .. 9945)    Habitat extended: 5 × 14 = 70  (0-13+)
//!   [9945 .. 10000)   Wildlife count extended2: 5 × 11 = 55  (0-10+)
//!   [10000 .. 10040)  Pairwise extension capacity: 5 × 8 = 40  (0-7+)
//!   [10040 .. 10088)  Smart pattern v2 features = 48
//!   [10088 .. 10193)  Bag remaining extended: 5 × 21 = 105  (0-20)
//!   [10193 .. 10263)  Opponent habitat extended: 5 × 14 = 70  (0-13+)
//!   [10263 .. 10351)  Market visibility: 4 slots × 22 = 88
//!   [10351 .. 10456)  Tile bag terrain distribution: 5 × 21 = 105  (0-20+)
//!   [10456 .. 10561)  Tile bag wildlife capacity: 5 × 21 = 105  (0-20+)
//!
//!   ─── v3 features (per-cell adjacency + extended bag + overflow) ───
//!   [10561 .. 44959)  Per-cell adjacency: 441 × 6 dirs × 13 states = 34398
//!   [44959 .. 45109)  Tile bag terrain extended: 5 × 30 = 150  (0-29)
//!   [45109 .. 45259)  Tile bag wildlife extended: 5 × 30 = 150  (0-29)
//!   [45259 .. 45260)  Overflow refresh used this turn: 1 bit

use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, ADJACENCY, GRID_SIZE};
use cascadia_core::types::Wildlife;

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

// Per-cell: 11 core features each (backward-compatible indices)
// 0-4: wildlife type placed (Bear, Elk, Salmon, Hawk, Fox)
// 5: tile present, no wildlife
// 6-10: terrain type (Forest, Prairie, Wetland, Mountain, River)
pub const FEATURES_PER_CELL: usize = 11;
pub const CELL_FEATURES: usize = GRID_SIZE * FEATURES_PER_CELL; // 4851

const TURN_FEATURES: usize = 21;     // turns 0-20
const TOKEN_FEATURES: usize = 9;     // tokens 0-8
const WL_COUNT_FEATURES: usize = 30; // 6 bins × 5 types (counts 0-5) — legacy size, backward compat

// Extended wildlife count: 10 bins × 5 types (0-9), appended after all legacy features
const WL_COUNT_EXT_BINS: usize = 10;
pub const WL_COUNT_EXT_FEATURES: usize = WL_COUNT_EXT_BINS * 5; // 50

pub const ALLOWED_WL_PER_CELL: usize = 5;
pub const ALLOWED_WL_FEATURES: usize = GRID_SIZE * ALLOWED_WL_PER_CELL; // 2205
const HAB_SIZE_FEATURES: usize = 50; // 10 bins × 5 terrains (sizes 0-9)
pub const PHASE_FEATURES: usize = TURN_FEATURES + TOKEN_FEATURES + WL_COUNT_FEATURES + HAB_SIZE_FEATURES; // 110

const PAIR_DIRS: usize = 3;
const PAIR_STATES: usize = 7 * 7;    // 49
pub const PAIR_FEATURES: usize = PAIR_DIRS * PAIR_STATES; // 147

const BEAR_PAIR_FEATURES: usize = 5;
const ELK_LINE_FEATURES: usize = 20;
const SALMON_RUN_FEATURES: usize = 24;
const HAWK_ISO_FEATURES: usize = 9;
const FOX_DIV_FEATURES: usize = 6;
const EMPTY_SLOTS_FEATURES: usize = 25;
pub const PATTERN_FEATURES: usize = BEAR_PAIR_FEATURES + ELK_LINE_FEATURES
    + SALMON_RUN_FEATURES + HAWK_ISO_FEATURES + FOX_DIV_FEATURES + EMPTY_SLOTS_FEATURES; // 89

const BAG_BINS: usize = 11;
pub const BAG_FEATURES: usize = 5 * BAG_BINS; // 55

const OPP_HAB_BINS: usize = 11;
pub const OPP_HAB_FEATURES: usize = 5 * OPP_HAB_BINS; // 55

const TERRAIN_PAIR_STATES: usize = 6 * 6; // 36
pub const TERRAIN_PAIR_FEATURES: usize = PAIR_DIRS * TERRAIN_PAIR_STATES; // 108

// ── v2 feature blocks (appended for backward compat) ──
/// Per-cell SECONDARY terrain: 441 cells × 5 terrains. Fires only on dual-terrain tiles.
pub const SEC_TERRAIN_FEATURES: usize = GRID_SIZE * 5; // 2205
/// Habitat size extended: 5 terrains × 14 bins (0-13+). Higher resolution than legacy.
pub const HAB_EXT_BINS: usize = 14;
pub const HAB_EXT_FEATURES: usize = 5 * HAB_EXT_BINS; // 70
/// Wildlife count extended-v2: 5 types × 11 bins (0-10+).
pub const WL_COUNT_EXT2_BINS: usize = 11;
pub const WL_COUNT_EXT2_FEATURES: usize = 5 * WL_COUNT_EXT2_BINS; // 55
/// Pairwise extension capacity: per-wildlife count of placed-W tiles adjacent to empty-W-allowed cells.
/// 5 wildlife × 8 bins (0-7+).
pub const EXT_CAP_FEATURES: usize = 5 * 8; // 40
/// Smart pattern v2: extendable lines/runs, bear waste, at-risk hawks, max-div foxes, forced slots.
const PAT_EXT_ELK_LINES: usize = 4;     // 0/1/2/3+ extendable elk lines
const PAT_EXT_SALMON_RUNS: usize = 4;
const PAT_BEAR_EXT_SINGLES: usize = 4;
const PAT_BEAR_WASTE: usize = 4;
const PAT_HAWK_AT_RISK: usize = 4;
const PAT_FORCED_ALLOC: usize = 5 * 4;  // 5 wildlife × 4 bins
const PAT_MAX_DIV_FOX: usize = 4;
const PAT_KEYSTONE_OPEN: usize = 4;
pub const PATTERN_V2_FEATURES: usize = PAT_EXT_ELK_LINES + PAT_EXT_SALMON_RUNS
    + PAT_BEAR_EXT_SINGLES + PAT_BEAR_WASTE + PAT_HAWK_AT_RISK
    + PAT_FORCED_ALLOC + PAT_MAX_DIV_FOX + PAT_KEYSTONE_OPEN; // 4+4+4+4+4+20+4+4 = 48
/// Bag remaining extended: 5 types × 21 bins (0-20).
pub const BAG_EXT_BINS: usize = 21;
pub const BAG_EXT_FEATURES: usize = 5 * BAG_EXT_BINS; // 105
/// Opponent habitat extended: 5 terrains × 14 bins (0-13+).
pub const OPP_HAB_EXT_BINS: usize = 14;
pub const OPP_HAB_EXT_FEATURES: usize = 5 * OPP_HAB_EXT_BINS; // 70
/// Market visibility: 4 slots × 22 features each.
/// Per slot: 5 t1 (one-hot terrain1) + 6 t2 (one-hot terrain2 OR none) + 5 allowed wildlife mask
///          + 1 keystone bit + 5 wildlife token (one-hot)
pub const MARKET_PER_SLOT: usize = 5 + 6 + 5 + 1 + 5; // 22
pub const MARKET_FEATURES: usize = 4 * MARKET_PER_SLOT; // 88
/// Tile bag terrain distribution: 5 terrains × 21 bins (0-20+).
/// Counts tiles in tile bag with each terrain (primary OR secondary).
pub const TBAG_TERRAIN_FEATURES: usize = 5 * BAG_EXT_BINS; // 105
/// Tile bag wildlife capacity: 5 wildlife × 21 bins (0-20+).
/// Counts tiles in tile bag whose allowed mask includes each wildlife.
pub const TBAG_WL_FEATURES: usize = 5 * BAG_EXT_BINS; // 105

// ── v3 feature blocks (appended for backward compat) ──
/// Per-cell adjacency: for each placed tile, 6 directions × (7 wildlife + 6 terrain) states.
/// Encodes what each neighbor looks like from each cell's perspective (position-dependent).
pub const ADJ_WILDLIFE_STATES: usize = 7;  // 0=no tile, 1-5=wildlife, 6=tile-no-wildlife
pub const ADJ_TERRAIN_STATES: usize = 6;   // 0=no tile, 1-5=terrain
pub const ADJ_STATES_PER_DIR: usize = ADJ_WILDLIFE_STATES + ADJ_TERRAIN_STATES; // 13
pub const ADJ_DIRS: usize = 6;
pub const ADJ_FEATURES_PER_CELL: usize = ADJ_DIRS * ADJ_STATES_PER_DIR; // 78
pub const CELL_ADJ_FEATURES: usize = GRID_SIZE * ADJ_FEATURES_PER_CELL; // 34398
/// Extended tile bag terrain: 5 terrains × 30 bins (0-29). Covers full range (29 tiles/terrain).
pub const TBAG_EXT_BINS: usize = 30;
pub const TBAG_TERRAIN_EXT_FEATURES: usize = 5 * TBAG_EXT_BINS; // 150
/// Extended tile bag wildlife capacity: 5 wildlife × 30 bins (0-29).
pub const TBAG_WL_EXT_FEATURES: usize = 5 * TBAG_EXT_BINS; // 150
/// Whether the 3-of-a-kind overflow refresh has been used this turn.
pub const OVERFLOW_FEATURES: usize = 1;

// ── v4-opp: per-opponent detail block ────────────────────────────────
// For each of 3 opponents (ordered by relative seat from current player),
// encode wildlife counts, habitat sizes, nature tokens, and pattern signals.
// Total: 3 × 123 = 369 features appended at the end.
pub const OPP_DET_WL_BINS: usize = 11;  // 0, 1, ..., 9, 10+
pub const OPP_DET_HAB_BINS: usize = 11; // 0, 1, ..., 9, 10+
pub const OPP_DET_TOK_BINS: usize = 9;  // 0, 1, ..., 7, 8+
pub const OPP_DET_PATTERN_BITS: usize = 4; // bear-singleton, elk-line-3+, salmon-run-4+, hawk-iso-5+
pub const OPP_DET_PER_OPP: usize = 5 * OPP_DET_WL_BINS
    + 5 * OPP_DET_HAB_BINS
    + OPP_DET_TOK_BINS
    + OPP_DET_PATTERN_BITS; // 55 + 55 + 9 + 4 = 123
pub const NUM_OPP_SLOTS: usize = 3;
pub const OPP_DETAILED_FEATURES: usize = NUM_OPP_SLOTS * OPP_DET_PER_OPP; // 369

/// Feature count of the original architecture (for backward-compatible weight loading)
/// Old: 441×11 + 110 + 147 + 89 = 5197 (no bag/opponent/allowed features)
pub const NUM_FEATURES_LEGACY: usize = 5197;
/// v1 architecture (= what iter1-20 weights were trained with)
pub const NUM_FEATURES_V1: usize = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES + PATTERN_FEATURES
    + BAG_FEATURES + OPP_HAB_FEATURES + ALLOWED_WL_FEATURES + WL_COUNT_EXT_FEATURES
    + TERRAIN_PAIR_FEATURES; // 7670
/// v2 architecture — appends new feature blocks for richer info
/// 7670 (v1) + 2205 (sec terrain) + 70 (hab ext) + 55 (wl ext2) + 40 (ext cap)
///          + 48 (pat v2) + 105 (bag ext) + 70 (opp hab ext) + 88 (market)
///          + 105 (tbag terrain) + 105 (tbag wl) = 10561
pub const NUM_FEATURES_V2: usize = NUM_FEATURES_V1
    + SEC_TERRAIN_FEATURES + HAB_EXT_FEATURES + WL_COUNT_EXT2_FEATURES
    + EXT_CAP_FEATURES + PATTERN_V2_FEATURES
    + BAG_EXT_FEATURES + OPP_HAB_EXT_FEATURES
    + MARKET_FEATURES + TBAG_TERRAIN_FEATURES + TBAG_WL_FEATURES; // 10561
/// v3 architecture (current) — per-cell adjacency + extended bag + overflow
/// 10561 (v2) + 34398 (cell adj) + 150 (tbag terr ext) + 150 (tbag wl ext) + 1 (overflow) = 45260
pub const NUM_FEATURES_V3: usize = NUM_FEATURES_V2
    + CELL_ADJ_FEATURES + TBAG_TERRAIN_EXT_FEATURES + TBAG_WL_EXT_FEATURES
    + OVERFLOW_FEATURES; // 45260

/// Feature set selection via cargo features:
/// - `legacy-features`: original 5,197 (mce93's set, ~10MB weights, 2.7M params)
/// - `mid-features`: all v3 MINUS per-cell adjacency = 10,862 (~22MB, 5.6M params)
/// - default: full v3 = 45,260 (~89MB, 23M params)
/// - `v4-opp` (additive with above): appends 369 per-opponent detail features.
pub const NUM_FEATURES_MID: usize = NUM_FEATURES_V2
    + TBAG_TERRAIN_EXT_FEATURES + TBAG_WL_EXT_FEATURES + OVERFLOW_FEATURES; // 10862

pub const NUM_FEATURES_V3_V4: usize = NUM_FEATURES_V3 + OPP_DETAILED_FEATURES; // 45629
pub const NUM_FEATURES_MID_V4: usize = NUM_FEATURES_MID + OPP_DETAILED_FEATURES; // 11231

/// Base index where the v4-opp block starts in the feature index space.
/// Crucially, opp-detail indices don't care about whether cell-adj etc fire;
/// they just append after the last non-v4 block (mid or v3), so a v4 weights
/// file trained with mid-features has its opp-detail columns at indices
/// NUM_FEATURES_MID..NUM_FEATURES_MID_V4 and plays nicely with mid weights
/// by zero-padding the v4 block on load.
pub const OPP_DETAILED_BASE: usize = if cfg!(feature = "legacy-features") { NUM_FEATURES_LEGACY }
                                     else if cfg!(feature = "mid-features") { NUM_FEATURES_MID }
                                     else { NUM_FEATURES_V3 };

pub const NUM_FEATURES: usize = if cfg!(feature = "legacy-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_LEGACY + OPP_DETAILED_FEATURES
} else if cfg!(feature = "legacy-features") { NUM_FEATURES_LEGACY }
  else if cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") { NUM_FEATURES_MID_V4 }
  else if cfg!(feature = "mid-features") { NUM_FEATURES_MID }
  else if cfg!(feature = "v4-opp") { NUM_FEATURES_V3_V4 }
  else { NUM_FEATURES_V3 };

/// Training data for one position's policy candidates (delta-feature approach).
pub struct PositionPolicyData {
    pub base_features: Vec<u16>,
    pub candidates: Vec<(Vec<u16>, f32)>,  // (afterstate_features, mce_score)
}

// Build with `--features small-net` to get mce93's lean architecture (256→32,
// ~10MB weight files). Use for experiments where we want to test whether
// smaller capacity generalizes better with limited training data.
// `--features large-net` inflates to 1024→128. Default (no feature) is 512→64.
pub const HIDDEN1: usize = if cfg!(feature = "small-net") { 256 }
                           else if cfg!(feature = "large-net") { 1024 }
                           else { 512 };
pub const HIDDEN2: usize = if cfg!(feature = "small-net") { 32 }
                           else if cfg!(feature = "large-net") { 128 }
                           else { 64 };

// ─────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────

/// Encode terrain on a specific edge of a cell: 0=empty, 1-5=terrain type
/// For dual-terrain tiles, the terrain depends on rotation and edge direction.
#[inline(always)]
fn terrain_code_on_edge(board: &Board, idx: usize, direction: usize) -> u8 {
    let cell = board.grid.get(idx);
    if !cell.is_present() { return 0; }
    let rotation = board.rotations[idx];
    match cascadia_core::board::terrain_on_edge(cell, rotation, direction) {
        Some(t) => (t as u8) + 1,
        None => 0,
    }
}

/// Encode wildlife state for a cell: 0=empty, 1-5=wildlife, 6=tile_no_wildlife
#[inline(always)]
fn wildlife_code(board: &Board, idx: usize) -> u8 {
    let cell = board.grid.get(idx);
    if !cell.is_present() { return 0; }
    match cell.placed_wildlife() {
        Some(w) => (w as u8) + 1,
        None => 6,
    }
}

/// Pre-computed first-layer activation values. Maintained incrementally
/// as tiles/wildlife are placed/undone to avoid recomputing from scratch.
#[derive(Clone)]
pub struct Accumulator {
    pub values: [f32; HIDDEN1],
}

impl Accumulator {
    /// Build a fresh accumulator from a board + NNUE weights.
    /// Includes per-cell + phase + pairwise + pattern features (not bag/opponent).
    pub fn from_board(board: &Board, net: &NNUENetwork) -> Self {
        let features = extract_features(board);
        let mut values = [0.0f32; HIDDEN1];
        values.copy_from_slice(&net.b1);
        for &fi in &features {
            let base = fi as usize * HIDDEN1;
            let col = &net.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 {
                values[j] += col[j];
            }
        }
        Accumulator { values }
    }

    /// Rebuild the accumulator from scratch. Used to reset after
    /// accumulation drift or after complex board changes.
    pub fn rebuild(&mut self, board: &Board, net: &NNUENetwork) {
        *self = Self::from_board(board, net);
    }
}

/// Compute per-cell feature indices for a given cell index on a board.
/// Returns core features (wildlife/no-wildlife + terrain). Does NOT include
/// allowed-wildlife features (those are in a separate block at the end).
#[inline]
pub fn cell_features(board: &Board, idx: usize) -> arrayvec::ArrayVec<u16, 2> {
    let mut features = arrayvec::ArrayVec::new();
    let cell = board.grid.get(idx);
    if !cell.is_present() { return features; }
    let base = idx * FEATURES_PER_CELL;
    if let Some(w) = cell.placed_wildlife() {
        features.push((base + w as usize) as u16);
    } else {
        features.push((base + 5) as u16); // tile_no_wildlife
    }
    if let Some(t) = cell.primary_terrain() {
        features.push((base + 6 + t as usize) as u16);
    }
    features
}

/// Compute what a cell's features were BEFORE wildlife was placed on it.
/// (It had "tile_no_wildlife" + terrain, instead of the wildlife type + terrain.)
#[inline]
pub fn cell_features_before_wildlife(board: &Board, idx: usize, wildlife: Wildlife) -> arrayvec::ArrayVec<u16, 2> {
    let mut features = arrayvec::ArrayVec::new();
    let cell = board.grid.get(idx);
    if !cell.is_present() { return features; }
    let base = idx * FEATURES_PER_CELL;
    // Before wildlife was placed, this cell had "tile_no_wildlife"
    features.push((base + 5) as u16);
    if let Some(t) = cell.primary_terrain() {
        features.push((base + 6 + t as usize) as u16);
    }
    features
}

/// Compute the feature diff caused by placing a tile (and optionally wildlife) at idx.
/// Must be called AFTER the placement (board reflects the new state).
/// Returns (removed_features, added_features).
pub fn compute_placement_diff(
    board: &Board,
    idx: usize,
    wildlife_idx: Option<usize>,
    wildlife: Option<Wildlife>,
) -> (arrayvec::ArrayVec<u16, 64>, arrayvec::ArrayVec<u16, 64>) {
    let mut removed = arrayvec::ArrayVec::new();
    let mut added = arrayvec::ArrayVec::new();
    let adj = &*ADJACENCY;

    // 1. Per-cell features for the placed tile cell
    // Before: cell was empty (no features). After: has terrain + tile_no_wildlife (or wildlife if placed here)
    let cell = board.grid.get(idx);
    let base = idx * FEATURES_PER_CELL;
    if let Some(w) = cell.placed_wildlife() {
        added.push((base + w as usize) as u16);
    } else {
        added.push((base + 5) as u16); // tile_no_wildlife
    }
    if let Some(t) = cell.primary_terrain() {
        added.push((base + 6 + t as usize) as u16);
    }

    // 2. Per-cell features for the wildlife cell (if different from tile cell)
    if let (Some(widx), Some(wl)) = (wildlife_idx, wildlife) {
        if widx != idx {
            let wbase = widx * FEATURES_PER_CELL;
            // Before: had tile_no_wildlife. After: has wildlife type
            removed.push((wbase + 5) as u16); // remove tile_no_wildlife
            added.push((wbase + wl as usize) as u16); // add wildlife
        }
    }

    // 3. Pairwise adjacency features: the placed cell and all its neighbors
    // Every (tile_cell, neighbor) pair in all 3 line directions changes
    let pair_base = CELL_FEATURES + PHASE_FEATURES;
    let my_wl = wildlife_code(board, idx);

    for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
        let coord = HexCoord::from_index(idx);
        let neighbor = HexCoord::new(coord.q + dq, coord.r + dr);
        if let Some(nidx) = neighbor.to_index() {
            let n_wl = wildlife_code(board, nidx);
            if n_wl > 0 {
                // Before: pair was (0, n_wl) since our cell was empty
                let old_pair = dir * PAIR_STATES + 0 * 7 + n_wl as usize;
                removed.push((pair_base + old_pair) as u16);
            }
            if my_wl > 0 || n_wl > 0 {
                // After: pair is (my_wl, n_wl)
                let new_pair = dir * PAIR_STATES + my_wl as usize * 7 + n_wl as usize;
                added.push((pair_base + new_pair) as u16);
            }
        }

        // Also check reverse direction (neighbor → us) for neighbors that have tiles
        let rev_neighbor = HexCoord::new(coord.q - dq, coord.r - dr);
        if let Some(rnidx) = rev_neighbor.to_index() {
            let rn_wl = wildlife_code(board, rnidx);
            if rn_wl > 0 {
                // Before: pair was (rn_wl, 0) since our cell was empty
                let old_pair = dir * PAIR_STATES + rn_wl as usize * 7 + 0;
                removed.push((pair_base + old_pair) as u16);
                // After: pair is (rn_wl, my_wl)
                let new_pair = dir * PAIR_STATES + rn_wl as usize * 7 + my_wl as usize;
                added.push((pair_base + new_pair) as u16);
            }
        }
    }

    // 4. Wildlife placement on a different cell also changes pairwise features for that cell
    if let (Some(widx), Some(_wl)) = (wildlife_idx, wildlife) {
        if widx != idx {
            let wcoord = HexCoord::from_index(widx);
            let new_wl = wildlife_code(board, widx); // now has wildlife
            // old_wl was 6 (tile_no_wildlife)
            let old_wl: u8 = 6;
            for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
                let neighbor = HexCoord::new(wcoord.q + dq, wcoord.r + dr);
                if let Some(nidx) = neighbor.to_index() {
                    let n_wl = wildlife_code(board, nidx);
                    if n_wl > 0 || old_wl > 0 {
                        removed.push((pair_base + dir * PAIR_STATES + old_wl as usize * 7 + n_wl as usize) as u16);
                    }
                    if n_wl > 0 || new_wl > 0 {
                        added.push((pair_base + dir * PAIR_STATES + new_wl as usize * 7 + n_wl as usize) as u16);
                    }
                }
                let rev = HexCoord::new(wcoord.q - dq, wcoord.r - dr);
                if let Some(rnidx) = rev.to_index() {
                    let rn_wl = wildlife_code(board, rnidx);
                    if rn_wl > 0 {
                        removed.push((pair_base + dir * PAIR_STATES + rn_wl as usize * 7 + old_wl as usize) as u16);
                        added.push((pair_base + dir * PAIR_STATES + rn_wl as usize * 7 + new_wl as usize) as u16);
                    }
                }
            }
        }
    }

    // 5. Phase + pattern features: these are global aggregates, too complex for incremental.
    // We skip them — caller should handle via full recompute of phase/pattern block.

    (removed, added)
}

/// Extract only phase + pattern features (no per-cell, no pairwise).
/// Used by incremental accumulator to recompute global features cheaply.
pub fn extract_phase_pattern_features(board: &Board, _cards: &cascadia_core::types::ScoringCards) -> Vec<u16> {
    let mut features = Vec::with_capacity(30);

    // Phase features (same as in extract_features)
    let phase_base = CELL_FEATURES;
    let turn = (board.tile_count as usize).saturating_sub(3).min(20);
    features.push((phase_base + turn) as u16);
    let tokens = (board.nature_tokens as usize).min(8);
    features.push((phase_base + TURN_FEATURES + tokens) as u16);
    let wl_base = phase_base + TURN_FEATURES + TOKEN_FEATURES;
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(5);
        features.push((wl_base + wtype * 6 + count) as u16);
    }
    let hab_base = wl_base + WL_COUNT_FEATURES;
    for terrain in 0..5 {
        let size = (board.largest_group[terrain] as usize).min(9);
        features.push((hab_base + terrain * 10 + size) as u16);
    }

    // Pattern features
    let pat_base = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES;
    extract_pattern_features(board, &mut features, pat_base);

    features
}

/// Extract only bag + opponent habitat features.
pub fn extract_bag_features(board: &Board, bag: &BagInfo) -> Vec<u16> {
    let mut features = Vec::with_capacity(10);
    let bag_base = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES + PATTERN_FEATURES;
    for wtype in 0..5 {
        let count = (bag.remaining[wtype] as usize).min(BAG_BINS - 1);
        features.push((bag_base + wtype * BAG_BINS + count) as u16);
    }
    let opp_base = bag_base + BAG_FEATURES;
    for terrain in 0..5 {
        let size = (bag.max_opponent_habitat[terrain] as usize).min(OPP_HAB_BINS - 1);
        features.push((opp_base + terrain * OPP_HAB_BINS + size) as u16);
    }
    features
}

/// A market slot's tile + wildlife token (mirror of `cascadia_core::market::MarketPair`
/// kept here so we can pass it through `BagInfo` without depending on game state).
#[derive(Clone, Copy, Default)]
pub struct MarketSlotInfo {
    /// 0 = empty slot; otherwise (terrain1 as u8 + 1)
    pub terrain1: u8,
    /// 0 = no second terrain; otherwise (terrain2 as u8 + 1)
    pub terrain2: u8,
    /// 5-bit wildlife mask (which animals can go on this tile)
    pub allowed_mask: u8,
    /// True if this is a single-terrain (keystone) tile
    pub keystone: bool,
    /// 0 = no token; otherwise (wildlife as u8 + 1)
    pub wildlife_token: u8,
}

/// Per-opponent summary for v4-opp feature block.
/// Captures visible opponent state that informs threat assessment.
#[derive(Clone, Copy, Default, Debug)]
pub struct OpponentDetail {
    /// Count of placed wildlife per type [Bear, Elk, Salmon, Hawk, Fox].
    pub wildlife_counts: [u8; 5],
    /// Largest habitat group per terrain [Forest, Prairie, Wetland, Mountain, River].
    pub largest_group: [u8; 5],
    /// Nature tokens held by this opponent (raw count, binned on emit).
    pub nature_tokens: u8,
    /// Opponent has at least one isolated (singleton) bear — one more bear finishes a pair.
    pub has_bear_singleton: bool,
    /// Opponent has an elk line of length 3 with room to extend.
    pub has_elk_line_3plus: bool,
    /// Opponent has a salmon run of length 4+ with an extendable endpoint.
    pub has_salmon_run_4plus: bool,
    /// Opponent's isolated-hawk count (5 or 6 = threat; bin at emit).
    pub isolated_hawk_count: u8,
}

/// Game-level information visible to the AI beyond the player's own board:
/// bag composition and opponent habitat sizes.
#[derive(Clone, Default)]
pub struct BagInfo {
    /// Remaining drawable count per wildlife type [Bear, Elk, Salmon, Hawk, Fox]
    pub remaining: [u8; 5],
    /// Max opponent habitat size per terrain [Forest, Prairie, Wetland, Mountain, River]
    pub max_opponent_habitat: [u8; 5],
    /// The 4 market slots (tile + wildlife token), or empty
    pub market: [MarketSlotInfo; 4],
    /// Count of remaining tile-bag tiles having each terrain (counts both primary AND secondary).
    /// Indexed by Terrain (Forest, Prairie, Wetland, Mountain, River).
    pub tbag_terrain: [u8; 5],
    /// Count of remaining tile-bag tiles whose allowed mask includes each wildlife.
    /// Indexed by Wildlife (Bear, Elk, Salmon, Hawk, Fox).
    pub tbag_wildlife: [u8; 5],
    /// Whether the 3-of-a-kind overflow refresh has been used this turn.
    pub overflow_used: bool,
    /// Per-opponent detail for the v4-opp feature block.
    /// Entries ordered by relative seat offset from the observing player (1, 2, 3).
    /// For games with fewer than 4 players, unused slots are default (all zero).
    pub opp_detail: [OpponentDetail; NUM_OPP_SLOTS],
}

impl BagInfo {
    /// Compute from full game state for the given player.
    pub fn from_game(game: &cascadia_core::game::GameState) -> Self {
        Self::from_game_for_player(game, game.current_player)
    }

    /// Compute from full game state for a specific player.
    pub fn from_game_for_player(game: &cascadia_core::game::GameState, player: usize) -> Self {
        // Bag remaining: count what's on all boards + in market
        let mut placed = [0u8; 5];
        for board in &game.boards {
            for wtype in 0..5 {
                placed[wtype] += board.wildlife_positions[wtype].len() as u8;
            }
        }
        for pair in game.market.pairs.iter().flatten() {
            placed[pair.wildlife as usize] += 1;
        }
        let mut remaining = [0u8; 5];
        for i in 0..5 {
            remaining[i] = 20u8.saturating_sub(placed[i]);
        }

        // Max opponent habitat size per terrain
        let mut max_opponent_habitat = [0u8; 5];
        for (i, board) in game.boards.iter().enumerate() {
            if i == player { continue; }
            for t in 0..5 {
                let size = board.largest_group[t] as u8;
                if size > max_opponent_habitat[t] {
                    max_opponent_habitat[t] = size;
                }
            }
        }

        // Market slot info
        let mut market: [MarketSlotInfo; 4] = Default::default();
        for (i, slot) in game.market.pairs.iter().enumerate() {
            if let Some(pair) = slot {
                let cell = pair.tile.to_cell();
                let t1 = cell.primary_terrain().map(|t| (t as u8) + 1).unwrap_or(0);
                let t2 = cell.secondary_terrain().map(|t| (t as u8) + 1).unwrap_or(0);
                market[i] = MarketSlotInfo {
                    terrain1: t1,
                    terrain2: t2,
                    allowed_mask: pair.tile.allowed.0,
                    keystone: pair.tile.keystone,
                    wildlife_token: (pair.wildlife as u8) + 1,
                };
            }
        }

        // Tile bag distributions: counts over remaining tiles
        let (tbag_terrain, tbag_wildlife) = game.tile_bag.feature_distributions();

        // Per-opponent detail (relative seat order from the observing player).
        let mut opp_detail: [OpponentDetail; NUM_OPP_SLOTS] = Default::default();
        let mut slot = 0;
        for i in 0..game.num_players {
            if i == player || slot >= NUM_OPP_SLOTS { continue; }
            opp_detail[slot] = compute_opponent_detail(&game.boards[i]);
            slot += 1;
        }

        BagInfo {
            remaining,
            max_opponent_habitat,
            market,
            tbag_terrain,
            tbag_wildlife,
            overflow_used: game.overflow_used_this_turn,
            opp_detail,
        }
    }
}

/// Compute a single opponent's detail from its board. Read-only.
fn compute_opponent_detail(board: &Board) -> OpponentDetail {
    let adj = &*ADJACENCY;
    let wildlife_counts: [u8; 5] =
        std::array::from_fn(|i| board.wildlife_positions[i].len() as u8);
    let largest_group: [u8; 5] =
        std::array::from_fn(|i| board.largest_group[i] as u8);

    // Bear-singleton: any bear whose bear neighbours count is zero.
    let mut has_bear_singleton = false;
    for &p in board.wildlife_positions[Wildlife::Bear as usize].iter() {
        let pos = p as usize;
        let bear_neigh = adj.neighbors_of(pos)
            .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
            .count();
        if bear_neigh == 0 { has_bear_singleton = true; break; }
    }

    // Longest elk line along any of 3 hex line directions.
    let mut has_elk_line_3plus = false;
    'elk: for &p in board.wildlife_positions[Wildlife::Elk as usize].iter() {
        let coord = cascadia_core::hex::HexCoord::from_index(p as usize);
        for &(dq, dr) in &cascadia_core::hex::HexCoord::LINE_DIRECTIONS {
            let mut len = 1u16;
            let mut c = cascadia_core::hex::HexCoord::new(coord.q + dq, coord.r + dr);
            while let Some(i) = c.to_index() {
                if board.grid.get(i).placed_wildlife() == Some(Wildlife::Elk) {
                    len += 1;
                    c = cascadia_core::hex::HexCoord::new(c.q + dq, c.r + dr);
                } else { break; }
            }
            let mut c = cascadia_core::hex::HexCoord::new(coord.q - dq, coord.r - dr);
            while let Some(i) = c.to_index() {
                if board.grid.get(i).placed_wildlife() == Some(Wildlife::Elk) {
                    len += 1;
                    c = cascadia_core::hex::HexCoord::new(c.q - dq, c.r - dr);
                } else { break; }
            }
            if len >= 3 { has_elk_line_3plus = true; break 'elk; }
        }
    }

    // Salmon: largest connected run length (BFS from each salmon, dedupe via visited).
    let mut has_salmon_run_4plus = false;
    {
        let mut visited = [false; 441];
        for &p in board.wildlife_positions[Wildlife::Salmon as usize].iter() {
            let start = p as usize;
            if visited[start] { continue; }
            // BFS
            let mut stack: arrayvec::ArrayVec<usize, 32> = arrayvec::ArrayVec::new();
            stack.push(start); visited[start] = true;
            let mut len = 0u16;
            while let Some(cur) = stack.pop() {
                len += 1;
                for n in adj.neighbors_of(cur) {
                    if !visited[n]
                        && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                        visited[n] = true;
                        if !stack.is_full() { stack.push(n); }
                    }
                }
            }
            if len >= 4 { has_salmon_run_4plus = true; break; }
        }
    }

    // Isolated hawk count.
    let mut isolated_hawks = 0u8;
    for &p in board.wildlife_positions[Wildlife::Hawk as usize].iter() {
        let pos = p as usize;
        let has_hawk_neigh = adj.neighbors_of(pos)
            .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
        if !has_hawk_neigh { isolated_hawks += 1; }
    }

    OpponentDetail {
        wildlife_counts,
        largest_group,
        nature_tokens: board.nature_tokens,
        has_bear_singleton,
        has_elk_line_3plus,
        has_salmon_run_4plus,
        isolated_hawk_count: isolated_hawks,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tile-token feature extraction (for transformer / GNN experiments)
// ─────────────────────────────────────────────────────────────────────

/// Per-tile token for transformer/GNN input.
#[derive(Clone, Debug)]
pub struct TileToken {
    /// Terrain on each of 6 hex edges (0-4 = Forest..River), rotation-aware.
    pub terrain_triangles: [u8; 6],
    /// Wildlife placed (0=none, 1-5=Bear..Fox).
    pub wildlife: u8,
    /// Allowed wildlife bitmask (5 bits).
    pub allowed_mask: u8,
    /// Keystone tile flag.
    pub keystone: bool,
    /// Whether wildlife is placed on this tile.
    pub has_wildlife: bool,
    /// Hex axial coordinates.
    pub q: i8,
    pub r: i8,
}

/// Global (non-spatial) features for transformer/GNN.
#[derive(Clone, Debug)]
pub struct GlobalFeatures {
    pub turn: u8,
    pub nature_tokens: u8,
    pub wildlife_counts: [u8; 5],
    pub largest_habitat: [u8; 5],
    pub bag_remaining: [u8; 5],
    pub opp_habitat: [u8; 5],
    pub market_terrain1: [u8; 4],
    pub market_terrain2: [u8; 4],
    pub market_wildlife: [u8; 4],
    pub tbag_terrain: [u8; 5],
    pub tbag_wildlife: [u8; 5],
    pub overflow_used: bool,
}

/// Extract tile tokens + global features from a board state.
pub fn extract_tile_tokens(board: &Board, bag: Option<&BagInfo>) -> (Vec<TileToken>, GlobalFeatures) {
    let mut tokens = Vec::with_capacity(23);

    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        if !cell.is_present() { continue; }

        let rotation = board.rotations[idx];
        let mut terrain_triangles = [0u8; 6];
        for dir in 0..6 {
            if let Some(t) = cascadia_core::board::terrain_on_edge(cell, rotation, dir) {
                terrain_triangles[dir] = t as u8;
            }
        }

        let wildlife = match cell.placed_wildlife() {
            Some(w) => (w as u8) + 1,
            None => 0,
        };

        let coord = HexCoord::from_index(idx);
        tokens.push(TileToken {
            terrain_triangles,
            wildlife,
            allowed_mask: cell.allowed_wildlife().0,
            keystone: cell.is_keystone(),
            has_wildlife: cell.has_wildlife(),
            q: coord.q,
            r: coord.r,
        });
    }

    let global = GlobalFeatures {
        turn: (board.tile_count as u8).saturating_sub(3).min(20),
        nature_tokens: board.nature_tokens,
        wildlife_counts: std::array::from_fn(|i| board.wildlife_positions[i].len() as u8),
        largest_habitat: std::array::from_fn(|i| board.largest_group[i] as u8),
        bag_remaining: bag.map(|b| b.remaining).unwrap_or([0; 5]),
        opp_habitat: bag.map(|b| b.max_opponent_habitat).unwrap_or([0; 5]),
        market_terrain1: bag.map(|b| std::array::from_fn(|i| b.market[i].terrain1)).unwrap_or([0; 4]),
        market_terrain2: bag.map(|b| std::array::from_fn(|i| b.market[i].terrain2)).unwrap_or([0; 4]),
        market_wildlife: bag.map(|b| std::array::from_fn(|i| b.market[i].wildlife_token)).unwrap_or([0; 4]),
        tbag_terrain: bag.map(|b| b.tbag_terrain).unwrap_or([0; 5]),
        tbag_wildlife: bag.map(|b| b.tbag_wildlife).unwrap_or([0; 5]),
        overflow_used: bag.map(|b| b.overflow_used).unwrap_or(false),
    };

    (tokens, global)
}

/// Rich per-tile token with per-cell adjacency info (NNUE-v3-level richness per tile).
/// Used by the rich-feature transformer (TIL2 format).
#[derive(Clone, Debug)]
pub struct RichTileToken {
    /// Terrain on each of 6 hex edges (0-4 = Forest..River), rotation-aware.
    pub terrain_triangles: [u8; 6],
    /// Wildlife placed (0=none, 1-5=Bear..Fox).
    pub wildlife: u8,
    /// Allowed wildlife bitmask (5 bits).
    pub allowed_mask: u8,
    pub keystone: bool,
    pub has_wildlife: bool,
    pub q: i8,
    pub r: i8,
    /// For each of 6 hex directions: neighbor's wildlife code (0=no tile,
    /// 1-5=wildlife, 6=tile-no-wildlife). Matches NNUE v3 adjacency encoding.
    pub neighbor_wildlife: [u8; 6],
    /// For each of 6 hex directions: neighbor's terrain on the shared edge,
    /// looking "back" toward this cell (rotation-aware). 0=no tile, 1-5=terrain.
    pub neighbor_terrain: [u8; 6],
}

/// Extract rich tile tokens including per-cell adjacency, for the richer transformer.
pub fn extract_rich_tile_tokens(
    board: &Board,
    bag: Option<&BagInfo>,
) -> (Vec<RichTileToken>, GlobalFeatures) {
    let adj = &*ADJACENCY;
    let mut tokens = Vec::with_capacity(23);

    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        if !cell.is_present() { continue; }

        let rotation = board.rotations[idx];
        let mut terrain_triangles = [0u8; 6];
        for dir in 0..6 {
            if let Some(t) = cascadia_core::board::terrain_on_edge(cell, rotation, dir) {
                terrain_triangles[dir] = t as u8;
            }
        }

        let wildlife = match cell.placed_wildlife() {
            Some(w) => (w as u8) + 1,
            None => 0,
        };

        // Per-direction adjacency encoding (matches NNUE v3 Block K semantics).
        let mut neighbor_wildlife = [0u8; 6];
        let mut neighbor_terrain = [0u8; 6];
        for dir in 0..6 {
            let nidx_val = adj.neighbors[idx][dir];
            if nidx_val == u16::MAX {
                continue;  // OOB: leave as 0 (= "no tile")
            }
            let nidx = nidx_val as usize;
            let n_cell = board.grid.get(nidx);
            if !n_cell.is_present() {
                continue;  // empty neighbor: wildlife=0 (no tile), terrain=0 (no tile)
            }
            // Wildlife code: 0=no tile (handled above), 1-5=type, 6=tile-no-wildlife
            neighbor_wildlife[dir] = match n_cell.placed_wildlife() {
                Some(w) => (w as u8) + 1,
                None => 6,
            };
            // Terrain looking back toward us = direction (dir+3)%6 on the neighbor
            let nrot = board.rotations[nidx];
            if let Some(t) = cascadia_core::board::terrain_on_edge(n_cell, nrot, (dir + 3) % 6) {
                neighbor_terrain[dir] = (t as u8) + 1;  // shift so 0=no tile, 1-5=terrain
            }
        }

        let coord = HexCoord::from_index(idx);
        tokens.push(RichTileToken {
            terrain_triangles,
            wildlife,
            allowed_mask: cell.allowed_wildlife().0,
            keystone: cell.is_keystone(),
            has_wildlife: cell.has_wildlife(),
            q: coord.q,
            r: coord.r,
            neighbor_wildlife,
            neighbor_terrain,
        });
    }

    let global = GlobalFeatures {
        turn: (board.tile_count as u8).saturating_sub(3).min(20),
        nature_tokens: board.nature_tokens,
        wildlife_counts: std::array::from_fn(|i| board.wildlife_positions[i].len() as u8),
        largest_habitat: std::array::from_fn(|i| board.largest_group[i] as u8),
        bag_remaining: bag.map(|b| b.remaining).unwrap_or([0; 5]),
        opp_habitat: bag.map(|b| b.max_opponent_habitat).unwrap_or([0; 5]),
        market_terrain1: bag.map(|b| std::array::from_fn(|i| b.market[i].terrain1)).unwrap_or([0; 4]),
        market_terrain2: bag.map(|b| std::array::from_fn(|i| b.market[i].terrain2)).unwrap_or([0; 4]),
        market_wildlife: bag.map(|b| std::array::from_fn(|i| b.market[i].wildlife_token)).unwrap_or([0; 4]),
        tbag_terrain: bag.map(|b| b.tbag_terrain).unwrap_or([0; 5]),
        tbag_wildlife: bag.map(|b| b.tbag_wildlife).unwrap_or([0; 5]),
        overflow_used: bag.map(|b| b.overflow_used).unwrap_or(false),
    };

    (tokens, global)
}

/// Write rich tile-token format samples to a binary file.
/// Format: "TIL2" magic, then per sample:
///   u8 num_tiles,
///   [rich_tile × num_tiles] (23 bytes each),
///   45 global bytes,
///   f32 target
///
/// Per-tile layout (23 bytes):
///   u8[6] terrain_triangles
///   u8 wildlife, u8 allowed_mask, u8 flags (bit0=keystone, bit1=has_wildlife)
///   i8 q, i8 r
///   u8[6] neighbor_wildlife (0=no tile, 1-5=wildlife, 6=tile-no-wildlife)
///   u8[6] neighbor_terrain (0=no tile, 1-5=terrain)
pub fn write_rich_tile_token_samples(
    path: &str,
    samples: &[(Vec<RichTileToken>, GlobalFeatures, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"TIL2")?;
    for (tokens, global, target) in samples {
        file.write_all(&[tokens.len() as u8])?;
        for t in tokens {
            file.write_all(&t.terrain_triangles)?;                 // 6 bytes
            file.write_all(&[t.wildlife, t.allowed_mask])?;        // 2 bytes
            let flags = (t.keystone as u8) | ((t.has_wildlife as u8) << 1);
            file.write_all(&[flags, t.q as u8, t.r as u8])?;       // 3 bytes
            file.write_all(&t.neighbor_wildlife)?;                 // 6 bytes
            file.write_all(&t.neighbor_terrain)?;                  // 6 bytes
        }
        file.write_all(&[global.turn, global.nature_tokens])?;
        file.write_all(&global.wildlife_counts)?;
        file.write_all(&global.largest_habitat)?;
        file.write_all(&global.bag_remaining)?;
        file.write_all(&global.opp_habitat)?;
        file.write_all(&global.market_terrain1)?;
        file.write_all(&global.market_terrain2)?;
        file.write_all(&global.market_wildlife)?;
        file.write_all(&global.tbag_terrain)?;
        file.write_all(&global.tbag_wildlife)?;
        file.write_all(&[global.overflow_used as u8])?;
        file.write_all(&target.to_le_bytes())?;
    }
    Ok(())
}

/// Write tile-token format samples to a binary file.
/// Format: "TILE" magic, then per sample:
///   u8 num_tiles, [TileToken × num_tiles], global features, f32 target
pub fn write_tile_token_samples(
    path: &str,
    samples: &[(Vec<TileToken>, GlobalFeatures, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"TILE")?;
    for (tokens, global, target) in samples {
        file.write_all(&[tokens.len() as u8])?;
        for t in tokens {
            file.write_all(&t.terrain_triangles)?;
            file.write_all(&[t.wildlife, t.allowed_mask])?;
            let flags = (t.keystone as u8) | ((t.has_wildlife as u8) << 1);
            file.write_all(&[flags, t.q as u8, t.r as u8])?;
        }
        // Global features as raw bytes
        let g = global;
        file.write_all(&[g.turn, g.nature_tokens])?;
        file.write_all(&g.wildlife_counts)?;
        file.write_all(&g.largest_habitat)?;
        file.write_all(&g.bag_remaining)?;
        file.write_all(&g.opp_habitat)?;
        file.write_all(&g.market_terrain1)?;
        file.write_all(&g.market_terrain2)?;
        file.write_all(&g.market_wildlife)?;
        file.write_all(&g.tbag_terrain)?;
        file.write_all(&g.tbag_wildlife)?;
        file.write_all(&[g.overflow_used as u8])?;
        file.write_all(&target.to_le_bytes())?;
    }
    Ok(())
}

/// Extract active feature indices from a board state (without bag info).
pub fn extract_features(board: &Board) -> Vec<u16> {
    extract_features_with_bag(board, None)
}

/// Extract active feature indices from a board state, optionally with bag composition.
pub fn extract_features_with_bag(board: &Board, bag: Option<&BagInfo>) -> Vec<u16> {
    let mut features = Vec::with_capacity(450);

    // ── Per-cell features ──
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        let base = idx * FEATURES_PER_CELL;

        if let Some(w) = cell.placed_wildlife() {
            features.push((base + w as usize) as u16);
        } else {
            features.push((base + 5) as u16); // tile_no_wildlife
        }

        if let Some(t) = cell.primary_terrain() {
            features.push((base + 6 + t as usize) as u16);
        }
    }

    // ── Game-phase features ──
    let phase_base = CELL_FEATURES;

    // Turn number (tiles placed minus 3 starting tiles, clamped 0-20)
    let turn = (board.tile_count as usize).saturating_sub(3).min(20);
    features.push((phase_base + turn) as u16);

    // Nature tokens (clamped 0-8)
    let tokens = (board.nature_tokens as usize).min(8);
    features.push((phase_base + TURN_FEATURES + tokens) as u16);

    // Wildlife count per type — legacy bins (clamped 0-5)
    let wl_base = phase_base + TURN_FEATURES + TOKEN_FEATURES;
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(5);
        features.push((wl_base + wtype * 6 + count) as u16);
    }

    // Largest habitat group per terrain (clamped 0-9)
    let hab_base = wl_base + WL_COUNT_FEATURES;
    for terrain in 0..5 {
        let size = (board.largest_group[terrain] as usize).min(9);
        features.push((hab_base + terrain * 10 + size) as u16);
    }

    // ── Pairwise adjacency features ──
    let pair_base = CELL_FEATURES + PHASE_FEATURES;

    for &tile_idx in board.placed_tiles.iter() {
        let start = HexCoord::from_index(tile_idx as usize);
        let my_wl = wildlife_code(board, tile_idx as usize);

        for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
            let neighbor = HexCoord::new(start.q + dq, start.r + dr);
            if let Some(nidx) = neighbor.to_index() {
                let n_wl = wildlife_code(board, nidx);
                // Only emit if at least one cell has a tile
                if my_wl > 0 || n_wl > 0 {
                    let pair_idx = dir * PAIR_STATES + my_wl as usize * 7 + n_wl as usize;
                    features.push((pair_base + pair_idx) as u16);
                }
            }
        }
    }

    // ── Wildlife pattern features ──
    let pat_base = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES;
    extract_pattern_features(board, &mut features, pat_base);

    // ── Bag remaining features ──
    if let Some(bag) = bag {
        let bag_base = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES + PATTERN_FEATURES;
        for wtype in 0..5 {
            let count = (bag.remaining[wtype] as usize).min(BAG_BINS - 1);
            features.push((bag_base + wtype * BAG_BINS + count) as u16);
        }

        // ── Opponent habitat features ──
        let opp_base = bag_base + BAG_FEATURES;
        for terrain in 0..5 {
            let size = (bag.max_opponent_habitat[terrain] as usize).min(OPP_HAB_BINS - 1);
            features.push((opp_base + terrain * OPP_HAB_BINS + size) as u16);
        }
    }

    // ── Allowed wildlife per cell ──
    let allowed_base = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES + PATTERN_FEATURES + BAG_FEATURES + OPP_HAB_FEATURES;
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        // Only emit for open slots (tile present, no wildlife)
        if cell.is_present() && !cell.has_wildlife() {
            let mask = cell.allowed_wildlife();
            for w in Wildlife::ALL {
                if mask.contains(w) {
                    features.push((allowed_base + idx * ALLOWED_WL_PER_CELL + w as usize) as u16);
                }
            }
        }
    }

    // ── Extended wildlife count (10 bins, 0-9) ──
    let ext_wl_base = allowed_base + ALLOWED_WL_FEATURES;
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(9);
        features.push((ext_wl_base + wtype * WL_COUNT_EXT_BINS + count) as u16);
    }

    // ── Terrain pairwise adjacency ──
    // For each placed tile, encode terrain-terrain pairs with neighbors in 3 line directions.
    // Uses edge-aware terrain (accounts for dual-terrain rotation).
    let terrain_pair_base = ext_wl_base + WL_COUNT_EXT_FEATURES;
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let coord = HexCoord::from_index(idx);

        for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
            let neighbor = HexCoord::new(coord.q + dq, coord.r + dr);
            if let Some(nidx) = neighbor.to_index() {
                if !board.grid.get(nidx).is_present() { continue; }
                // Get terrain on the shared edge: my terrain facing direction `dir`,
                // neighbor's terrain facing direction `(dir+3)%6`
                let my_terrain = terrain_code_on_edge(board, idx, dir);
                let n_terrain = terrain_code_on_edge(board, nidx, (dir + 3) % 6);
                if my_terrain > 0 && n_terrain > 0 {
                    let pair_idx = dir * TERRAIN_PAIR_STATES + my_terrain as usize * 6 + n_terrain as usize;
                    features.push((terrain_pair_base + pair_idx) as u16);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // v2 feature blocks (appended at NUM_FEATURES_V1 = 7670)
    // ─────────────────────────────────────────────────────────────────
    extract_v2_features(board, bag, &mut features);

    // When compiled with legacy-features, drop any indices beyond the
    // legacy range. This lets the same extract_features code run safely
    // with the smaller first-layer weight matrix.
    // Note: the retain filter uses the "no-v4" boundary so that v4-opp
    // features (emitted AFTER this retain) survive.
    let non_v4_cap = if cfg!(feature = "legacy-features") { NUM_FEATURES_LEGACY }
                      else if cfg!(feature = "mid-features") { NUM_FEATURES_MID }
                      else { NUM_FEATURES_V3 };
    if non_v4_cap < NUM_FEATURES_V3 {
        features.retain(|&f| (f as usize) < non_v4_cap);
    }

    // ─────────────────────────────────────────────────────────────────
    // v4-opp feature block (appended at OPP_DETAILED_BASE)
    // ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "v4-opp")]
    if let Some(b) = bag {
        extract_opp_detailed_features(b, &mut features, OPP_DETAILED_BASE);
    }

    features
}

/// Emit per-opponent detail features. `base` is the absolute starting index
/// in the feature index space (one of NUM_FEATURES_LEGACY, NUM_FEATURES_MID,
/// or NUM_FEATURES_V3 depending on cargo-feature combination).
#[cfg(feature = "v4-opp")]
fn extract_opp_detailed_features(bag: &BagInfo, features: &mut Vec<u16>, base: usize) {
    for (slot, opp) in bag.opp_detail.iter().enumerate() {
        let slot_base = base + slot * OPP_DET_PER_OPP;

        // Wildlife counts (5 × 11 bins: 0..=9, 10+)
        let wl_base = slot_base;
        for w in 0..5 {
            let bin = (opp.wildlife_counts[w] as usize).min(OPP_DET_WL_BINS - 1);
            features.push((wl_base + w * OPP_DET_WL_BINS + bin) as u16);
        }

        // Habitat sizes (5 × 11 bins)
        let hab_base = wl_base + 5 * OPP_DET_WL_BINS;
        for t in 0..5 {
            let bin = (opp.largest_group[t] as usize).min(OPP_DET_HAB_BINS - 1);
            features.push((hab_base + t * OPP_DET_HAB_BINS + bin) as u16);
        }

        // Nature tokens (9 bins)
        let tok_base = hab_base + 5 * OPP_DET_HAB_BINS;
        let tok_bin = (opp.nature_tokens as usize).min(OPP_DET_TOK_BINS - 1);
        features.push((tok_base + tok_bin) as u16);

        // Pattern signal bits (4 separate binary flags; emit when set)
        let pat_base = tok_base + OPP_DET_TOK_BINS;
        if opp.has_bear_singleton         { features.push((pat_base + 0) as u16); }
        if opp.has_elk_line_3plus         { features.push((pat_base + 1) as u16); }
        if opp.has_salmon_run_4plus       { features.push((pat_base + 2) as u16); }
        if opp.isolated_hawk_count >= 5   { features.push((pat_base + 3) as u16); }
    }
}

/// Extract all v2 feature blocks, appending to `features`.
/// All v2 indices are relative to NUM_FEATURES_V1 = 7670.
fn extract_v2_features(board: &Board, bag: Option<&BagInfo>, features: &mut Vec<u16>) {
    let v2_base = NUM_FEATURES_V1;

    // ── Block A: per-cell SECONDARY terrain (441 × 5 = 2205) ──
    // Fires only on dual-terrain placed tiles.
    let sec_base = v2_base;
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        if let Some(t) = cell.secondary_terrain() {
            features.push((sec_base + idx * 5 + t as usize) as u16);
        }
    }

    // ── Block B: habitat extended 0-13 (5 × 14 = 70) ──
    let hab_ext_base = sec_base + SEC_TERRAIN_FEATURES;
    for terrain in 0..5 {
        let size = (board.largest_group[terrain] as usize).min(HAB_EXT_BINS - 1);
        features.push((hab_ext_base + terrain * HAB_EXT_BINS + size) as u16);
    }

    // ── Block C: wildlife count extended 0-10 (5 × 11 = 55) ──
    let wl_ext2_base = hab_ext_base + HAB_EXT_FEATURES;
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(WL_COUNT_EXT2_BINS - 1);
        features.push((wl_ext2_base + wtype * WL_COUNT_EXT2_BINS + count) as u16);
    }

    // ── Block D: pairwise extension capacity (5 × 8 = 40) ──
    // For each wildlife type W, count placed-W tiles having ≥1 empty neighbor that allows W.
    let ext_cap_base = wl_ext2_base + WL_COUNT_EXT2_FEATURES;
    let adj = &*ADJACENCY;
    let mut ext_cap_counts = [0u32; 5];
    for wtype in 0..5 {
        let positions = &board.wildlife_positions[wtype];
        let target_wildlife = Wildlife::from_u8(wtype as u8).unwrap();
        for &pos in positions.iter() {
            let mut has_extension = false;
            for nidx in adj.neighbors_of(pos as usize) {
                let cell = board.grid.get(nidx);
                if cell.is_present() && !cell.has_wildlife()
                   && cell.allowed_wildlife().contains(target_wildlife) {
                    has_extension = true;
                    break;
                }
            }
            if has_extension {
                ext_cap_counts[wtype] += 1;
            }
        }
    }
    for wtype in 0..5 {
        let count = (ext_cap_counts[wtype] as usize).min(7);
        features.push((ext_cap_base + wtype * 8 + count) as u16);
    }

    // ── Block E: smart pattern v2 (48 features) ──
    let pat_v2_base = ext_cap_base + EXT_CAP_FEATURES;
    extract_pattern_v2_features(board, features, pat_v2_base);

    // ── Block F: bag remaining extended 0-20 (5 × 21 = 105) ──
    if let Some(b) = bag {
        let bag_ext_base = pat_v2_base + PATTERN_V2_FEATURES;
        for wtype in 0..5 {
            let count = (b.remaining[wtype] as usize).min(BAG_EXT_BINS - 1);
            features.push((bag_ext_base + wtype * BAG_EXT_BINS + count) as u16);
        }

        // ── Block G: opponent habitat extended 0-13 (5 × 14 = 70) ──
        let opp_ext_base = bag_ext_base + BAG_EXT_FEATURES;
        for terrain in 0..5 {
            let size = (b.max_opponent_habitat[terrain] as usize).min(OPP_HAB_EXT_BINS - 1);
            features.push((opp_ext_base + terrain * OPP_HAB_EXT_BINS + size) as u16);
        }

        // ── Block H: market visibility (4 × 22 = 88) ──
        // Per slot layout: [t1: 5][t2: 6][allowed_mask: 5][keystone: 1][wildlife_token: 5]
        let market_base = opp_ext_base + OPP_HAB_EXT_FEATURES;
        for (i, slot) in b.market.iter().enumerate() {
            let slot_base = market_base + i * MARKET_PER_SLOT;
            // terrain1: 0=empty slot (no fire); 1-5 = terrain
            if slot.terrain1 > 0 {
                let t = (slot.terrain1 - 1) as usize;
                features.push((slot_base + t) as u16);
            }
            // terrain2: 0 = none, 1-5 = terrain (we use 6 bits: 0=none, 1-5=terrain)
            // Encoding: bit position (slot_base + 5 + slot.terrain2 as usize)
            let t2_off = slot_base + 5;
            // Always emit a "t2" bit: 0..5 covers (none,F,P,W,M,R)
            features.push((t2_off + slot.terrain2 as usize) as u16);
            // Allowed wildlife mask: 5 bits
            let allowed_off = slot_base + 5 + 6;
            let mask = slot.allowed_mask;
            for w in 0..5 {
                if mask & (1 << w) != 0 {
                    features.push((allowed_off + w) as u16);
                }
            }
            // Keystone bit
            let key_off = slot_base + 5 + 6 + 5;
            if slot.keystone {
                features.push(key_off as u16);
            }
            // Wildlife token: 0=none, 1-5 = wildlife
            let wt_off = slot_base + 5 + 6 + 5 + 1;
            if slot.wildlife_token > 0 {
                let w = (slot.wildlife_token - 1) as usize;
                features.push((wt_off + w) as u16);
            }
        }

        // ── Block I: tile bag terrain distribution (5 × 21 = 105) ──
        let tbag_terr_base = market_base + MARKET_FEATURES;
        for t in 0..5 {
            let count = (b.tbag_terrain[t] as usize).min(BAG_EXT_BINS - 1);
            features.push((tbag_terr_base + t * BAG_EXT_BINS + count) as u16);
        }

        // ── Block J: tile bag wildlife capacity (5 × 21 = 105) ──
        let tbag_wl_base = tbag_terr_base + TBAG_TERRAIN_FEATURES;
        for w in 0..5 {
            let count = (b.tbag_wildlife[w] as usize).min(BAG_EXT_BINS - 1);
            features.push((tbag_wl_base + w * BAG_EXT_BINS + count) as u16);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // v3 feature blocks (appended at NUM_FEATURES_V2 = 10561)
    // ─────────────────────────────────────────────────────────────────

    // ── Block K: per-cell adjacency (441 × 6 × 13 = 34398) ──
    // For each placed tile, encode each of 6 neighbors' wildlife + terrain state.
    let adj_base = NUM_FEATURES_V2;
    let adj = &*ADJACENCY;
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell_base = adj_base + idx * ADJ_FEATURES_PER_CELL;
        for dir in 0..6 {
            let dir_base = cell_base + dir * ADJ_STATES_PER_DIR;
            let nidx_val = adj.neighbors[idx][dir];
            if nidx_val == u16::MAX {
                // Out of bounds: fire state 0 for both wildlife and terrain
                features.push(dir_base as u16);
                features.push((dir_base + ADJ_WILDLIFE_STATES) as u16);
            } else {
                let nidx = nidx_val as usize;
                // Wildlife state of neighbor
                let wl_state = wildlife_code(board, nidx) as usize;
                features.push((dir_base + wl_state) as u16);
                // Terrain state: neighbor's terrain on the edge facing back toward us
                let terr_state = terrain_code_on_edge(board, nidx, (dir + 3) % 6) as usize;
                features.push((dir_base + ADJ_WILDLIFE_STATES + terr_state) as u16);
            }
        }
    }

    // Bag-dependent v3 blocks
    if let Some(b) = bag {
        // ── Block L: tile bag terrain extended 0-29 (5 × 30 = 150) ──
        let tbag_terr_ext_base = adj_base + CELL_ADJ_FEATURES;
        for t in 0..5 {
            let count = (b.tbag_terrain[t] as usize).min(TBAG_EXT_BINS - 1);
            features.push((tbag_terr_ext_base + t * TBAG_EXT_BINS + count) as u16);
        }

        // ── Block M: tile bag wildlife extended 0-29 (5 × 30 = 150) ──
        let tbag_wl_ext_base = tbag_terr_ext_base + TBAG_TERRAIN_EXT_FEATURES;
        for w in 0..5 {
            let count = (b.tbag_wildlife[w] as usize).min(TBAG_EXT_BINS - 1);
            features.push((tbag_wl_ext_base + w * TBAG_EXT_BINS + count) as u16);
        }

        // ── Block N: overflow refresh used (1 bit) ──
        let overflow_base = tbag_wl_ext_base + TBAG_WL_EXT_FEATURES;
        if b.overflow_used {
            features.push(overflow_base as u16);
        }
    }
}

/// Extract Block E (smart pattern v2) features. Indices are relative to `base`.
fn extract_pattern_v2_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let adj = &*ADJACENCY;

    // ── Extendable elk lines (4 bins: 0/1/2/3+) ──
    // An elk line is "extendable" if at least one end has an empty cell allowing elk.
    // Quick approximation: count elk components where any constituent has an elk-allowed empty neighbor.
    let elk_positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let mut elk_extendable_components = 0usize;
    {
        let mut visited = [false; 441];
        for &pos in elk_positions.iter() {
            let idx = pos as usize;
            if visited[idx] { continue; }
            // BFS the elk component
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(pos);
            visited[idx] = true;
            let mut has_extension = false;
            while let Some(cur) = queue.pop() {
                for nidx in adj.neighbors_of(cur as usize) {
                    let cell = board.grid.get(nidx);
                    if cell.placed_wildlife() == Some(Wildlife::Elk) {
                        if !visited[nidx] {
                            visited[nidx] = true;
                            queue.push(nidx as u16);
                        }
                    } else if cell.is_present() && !cell.has_wildlife()
                              && cell.allowed_wildlife().contains(Wildlife::Elk) {
                        has_extension = true;
                    }
                }
            }
            if has_extension {
                elk_extendable_components += 1;
            }
        }
    }
    features.push((base + elk_extendable_components.min(3)) as u16);

    // ── Extendable salmon runs (4 bins) ──
    let pat2_off1 = base + PAT_EXT_ELK_LINES;
    let salmon_positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let mut salmon_extendable_components = 0usize;
    {
        let mut visited = [false; 441];
        for &pos in salmon_positions.iter() {
            let idx = pos as usize;
            if visited[idx] { continue; }
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(pos);
            visited[idx] = true;
            let mut has_extension = false;
            while let Some(cur) = queue.pop() {
                for nidx in adj.neighbors_of(cur as usize) {
                    let cell = board.grid.get(nidx);
                    if cell.placed_wildlife() == Some(Wildlife::Salmon) {
                        if !visited[nidx] {
                            visited[nidx] = true;
                            queue.push(nidx as u16);
                        }
                    } else if cell.is_present() && !cell.has_wildlife()
                              && cell.allowed_wildlife().contains(Wildlife::Salmon) {
                        has_extension = true;
                    }
                }
            }
            if has_extension {
                salmon_extendable_components += 1;
            }
        }
    }
    features.push((pat2_off1 + salmon_extendable_components.min(3)) as u16);

    // ── Bear singletons with extension (4 bins) ──
    // Lone bears (component size 1) that have at least one bear-allowed empty neighbor.
    let pat2_off2 = pat2_off1 + PAT_EXT_SALMON_RUNS;
    let bear_positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut bear_extendable_singles = 0usize;
    {
        // Count component sizes per bear
        let mut visited = [false; 441];
        for &pos in bear_positions.iter() {
            let idx = pos as usize;
            if visited[idx] { continue; }
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(pos);
            visited[idx] = true;
            let mut size = 0usize;
            let mut roots = arrayvec::ArrayVec::<u16, 24>::new();
            while let Some(cur) = queue.pop() {
                size += 1;
                roots.push(cur);
                for nidx in adj.neighbors_of(cur as usize) {
                    if !visited[nidx]
                       && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear) {
                        visited[nidx] = true;
                        queue.push(nidx as u16);
                    }
                }
            }
            if size == 1 {
                let r = roots[0] as usize;
                let extends = adj.neighbors_of(r).any(|nidx| {
                    let c = board.grid.get(nidx);
                    c.is_present() && !c.has_wildlife()
                        && c.allowed_wildlife().contains(Wildlife::Bear)
                });
                if extends {
                    bear_extendable_singles += 1;
                }
            }
        }
    }
    features.push((pat2_off2 + bear_extendable_singles.min(3)) as u16);

    // ── Bear waste (4 bins): bears in components of size ≥3 (worth 0 in Card A) ──
    let pat2_off3 = pat2_off2 + PAT_BEAR_EXT_SINGLES;
    let mut bear_waste = 0usize;
    {
        let mut visited = [false; 441];
        for &pos in bear_positions.iter() {
            let idx = pos as usize;
            if visited[idx] { continue; }
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(pos);
            visited[idx] = true;
            let mut size = 0usize;
            while let Some(cur) = queue.pop() {
                size += 1;
                for nidx in adj.neighbors_of(cur as usize) {
                    if !visited[nidx]
                       && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear) {
                        visited[nidx] = true;
                        queue.push(nidx as u16);
                    }
                }
            }
            if size >= 3 {
                bear_waste += size;
            }
        }
    }
    features.push((pat2_off3 + bear_waste.min(3)) as u16);

    // ── At-risk isolated hawks (4 bins) ──
    // Hawks currently isolated but with a hawk-allowed empty neighbor (could lose isolation).
    let pat2_off4 = pat2_off3 + PAT_BEAR_WASTE;
    let hawk_positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let mut hawks_at_risk = 0usize;
    for &pos in hawk_positions.iter() {
        let idx = pos as usize;
        let mut isolated = true;
        let mut has_hawk_slot = false;
        for nidx in adj.neighbors_of(idx) {
            let cell = board.grid.get(nidx);
            if cell.placed_wildlife() == Some(Wildlife::Hawk) {
                isolated = false;
                break;
            }
            if cell.is_present() && !cell.has_wildlife()
               && cell.allowed_wildlife().contains(Wildlife::Hawk) {
                has_hawk_slot = true;
            }
        }
        if isolated && has_hawk_slot {
            hawks_at_risk += 1;
        }
    }
    features.push((pat2_off4 + hawks_at_risk.min(3)) as u16);

    // ── Forced single-allocation slots per wildlife (5 × 4 = 20) ──
    // Empty placed cells where exactly one wildlife type is allowed.
    let pat2_off5 = pat2_off4 + PAT_HAWK_AT_RISK;
    let mut forced_counts = [0usize; 5];
    for &tile_idx in board.placed_tiles.iter() {
        let cell = board.grid.get(tile_idx as usize);
        if !cell.is_present() || cell.has_wildlife() { continue; }
        let mask = cell.allowed_wildlife();
        let bits = mask.0.count_ones();
        if bits == 1 {
            // Find the single allowed wildlife
            for w in 0..5 {
                if mask.0 & (1 << w) != 0 {
                    forced_counts[w] += 1;
                    break;
                }
            }
        }
    }
    for wtype in 0..5 {
        let bin = forced_counts[wtype].min(3);
        features.push((pat2_off5 + wtype * 4 + bin) as u16);
    }

    // ── Max-diversity foxes (4 bins): foxes adjacent to ≥4 distinct species ──
    let pat2_off6 = pat2_off5 + PAT_FORCED_ALLOC;
    let fox_positions = &board.wildlife_positions[Wildlife::Fox as usize];
    let mut max_div_foxes = 0usize;
    for &pos in fox_positions.iter() {
        let mut mask = 0u8;
        for nidx in adj.neighbors_of(pos as usize) {
            if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                mask |= 1 << (w as u8);
            }
        }
        if mask.count_ones() >= 4 {
            max_div_foxes += 1;
        }
    }
    features.push((pat2_off6 + max_div_foxes.min(3)) as u16);

    // ── Open keystone slots (4 bins) ──
    // Empty placed cells whose tile is a keystone (single-terrain). Approximation:
    // we treat single-terrain (no secondary) cells as keystone-equivalent.
    let pat2_off7 = pat2_off6 + PAT_MAX_DIV_FOX;
    let mut open_keystone = 0usize;
    for &tile_idx in board.placed_tiles.iter() {
        let cell = board.grid.get(tile_idx as usize);
        if !cell.is_present() || cell.has_wildlife() { continue; }
        if cell.secondary_terrain().is_none() {
            open_keystone += 1;
        }
    }
    features.push((pat2_off7 + open_keystone.min(3)) as u16);
}

/// Extract wildlife pattern features: bear pairs, elk lines, salmon runs,
/// isolated hawks, fox diversity, empty wildlife slots.
fn extract_pattern_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let adj = &*ADJACENCY;

    // ── Bear pairs (connected components of exactly size 2, isolated) ──
    let bear_positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut visited = [false; 441];
    let mut bear_pairs = 0usize;
    for &pos in bear_positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        if component.len() == 2 { bear_pairs += 1; }
    }
    features.push((base + bear_pairs.min(4)) as u16);

    // ── Elk lines (find all maximal lines, greedily assign, count by length) ──
    let elk_base = base + BEAR_PAIR_FEATURES;
    let elk_positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let mut is_elk = [false; 441];
    for &pos in elk_positions.iter() { is_elk[pos as usize] = true; }

    // Find all maximal lines in 3 directions
    let mut all_lines: Vec<usize> = Vec::new(); // lengths of assigned lines
    let mut used = [false; 441];
    let mut in_line = [[false; 441]; 3];

    for dir in 0..3 {
        let (dq, dr) = HexCoord::LINE_DIRECTIONS[dir];
        // Collect maximal lines for this direction
        let mut dir_lines: Vec<(usize, Vec<u16>)> = Vec::new(); // (length, positions)
        for &pos in elk_positions.iter() {
            if in_line[dir][pos as usize] { continue; }
            let coord = HexCoord::from_index(pos as usize);
            // Walk backward to find start
            let mut start = coord;
            loop {
                let prev = HexCoord::new(start.q - dq, start.r - dr);
                if let Some(pidx) = prev.to_index() {
                    if is_elk[pidx] { start = prev; continue; }
                }
                break;
            }
            // Walk forward to build line
            let mut line = Vec::new();
            let mut current = start;
            loop {
                if let Some(cidx) = current.to_index() {
                    if is_elk[cidx] {
                        line.push(cidx as u16);
                        in_line[dir][cidx] = true;
                        current = HexCoord::new(current.q + dq, current.r + dr);
                        continue;
                    }
                }
                break;
            }
            if line.len() >= 2 {
                dir_lines.push((line.len(), line));
            }
        }
        dir_lines.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, line) in dir_lines {
            // Find longest contiguous unused run
            let mut best_run = 0;
            let mut run = 0;
            for &p in &line {
                if !used[p as usize] { run += 1; best_run = best_run.max(run); }
                else { run = 0; }
            }
            if best_run >= 1 {
                // Mark used
                run = 0;
                let mut marking = false;
                let mut marked = 0;
                for &p in &line {
                    if !used[p as usize] {
                        run += 1;
                        if run >= best_run && !marking {
                            // Go back and mark
                            marking = true;
                        }
                    } else {
                        run = 0;
                    }
                }
                // Simpler: just mark the first best_run unused elk
                let mut count = 0;
                for &p in &line {
                    if !used[p as usize] && count < best_run {
                        used[p as usize] = true;
                        count += 1;
                    }
                    if count >= best_run { break; }
                }
                all_lines.push(best_run);
            }
        }
    }
    // Remaining unused elk count as lines of 1
    for &pos in elk_positions.iter() {
        if !used[pos as usize] { all_lines.push(1); }
    }
    all_lines.sort_by(|a, b| b.cmp(a));

    // Encode: up to 4 lines, length bins 0(none)/1/2/3/4+
    for line_slot in 0..4 {
        let len = all_lines.get(line_slot).copied().unwrap_or(0).min(4);
        features.push((elk_base + line_slot * 5 + len) as u16);
    }

    // ── Salmon runs (connected components with degree ≤ 2) ──
    let salmon_base = elk_base + ELK_LINE_FEATURES;
    let salmon_positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let mut visited = [false; 441];
    let mut run_lengths: Vec<usize> = Vec::new();
    for &pos in salmon_positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        // Check valid run (all degrees ≤ 2)
        let is_valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count() <= 2
        });
        if is_valid {
            run_lengths.push(component.len());
        }
    }
    run_lengths.sort_by(|a, b| b.cmp(a));

    // Encode: up to 3 runs, length bins 0(none)/1/2/3/4/5/6/7+
    for run_slot in 0..3 {
        let len = run_lengths.get(run_slot).copied().unwrap_or(0).min(7);
        features.push((salmon_base + run_slot * 8 + len) as u16);
    }

    // ── Isolated hawks ──
    let hawk_base = salmon_base + SALMON_RUN_FEATURES;
    let hawk_positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let mut isolated_hawks = 0usize;
    for &pos in hawk_positions.iter() {
        let has_neighbor = adj.neighbors_of(pos as usize)
            .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
        if !has_neighbor { isolated_hawks += 1; }
    }
    features.push((hawk_base + isolated_hawks.min(8)) as u16);

    // ── Fox diversity (average unique adjacent types across all foxes) ──
    let fox_base = hawk_base + HAWK_ISO_FEATURES;
    let fox_positions = &board.wildlife_positions[Wildlife::Fox as usize];
    let mut total_div = 0usize;
    for &pos in fox_positions.iter() {
        let mut mask = 0u8;
        for nidx in adj.neighbors_of(pos as usize) {
            if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                mask |= 1 << (w as u8);
            }
        }
        total_div += mask.count_ones() as usize;
    }
    let avg_div = if fox_positions.is_empty() { 0 } else {
        (total_div + fox_positions.len() / 2) / fox_positions.len() // rounded avg
    };
    features.push((fox_base + avg_div.min(5)) as u16);

    // ── Empty wildlife slots per type ──
    let slot_base = fox_base + FOX_DIV_FEATURES;
    let mut empty_slots = [0u16; 5];
    for &tile_idx in board.placed_tiles.iter() {
        let cell = board.grid.get(tile_idx as usize);
        if cell.has_wildlife() { continue; }
        for w in Wildlife::ALL {
            if cell.allowed_wildlife().contains(w) {
                empty_slots[w as usize] += 1;
            }
        }
    }
    for wtype in 0..5 {
        let count = (empty_slots[wtype] as usize).min(4); // bins: 0,1,2,3,4+
        features.push((slot_base + wtype * 5 + count) as u16);
    }
}

// ─────────────────────────────────────────────────────────────────────
// NNUE Network
// ─────────────────────────────────────────────────────────────────────

/// The NNUE network weights. Flat contiguous layout for cache-friendly access.
#[derive(Clone)]
pub struct NNUENetwork {
    /// First layer weights: flat [NUM_FEATURES * HIDDEN1], row-major: w1[feature * HIDDEN1 + neuron]
    pub w1: Vec<f32>,      // [NUM_FEATURES * HIDDEN1]
    pub b1: Vec<f32>,      // [HIDDEN1]
    /// Second layer weights: flat [HIDDEN1 * HIDDEN2], row-major: w2[i * HIDDEN2 + j]
    pub w2: Vec<f32>,      // [HIDDEN1 * HIDDEN2]
    pub b2: Vec<f32>,      // [HIDDEN2]
    /// Output layer weights: HIDDEN2 → 1 (legacy total value head)
    pub w3: Vec<f32>,      // [HIDDEN2]
    pub b3: f32,
    /// Policy head weights: HIDDEN2 → 1 (scores candidate afterstates for move ranking)
    pub w3_policy: Vec<f32>,  // [HIDDEN2]
    pub b3_policy: f32,
    /// Split value heads (v5 architecture). When `has_split_value_heads` is true,
    /// `forward()` returns `wildlife_pred + habitat_pred` (fixed 1:1 sum, no variable
    /// blending). When false, falls back to the legacy w3/b3 total value head.
    ///
    /// `w3_wildlife`/`b3_wildlife` predicts the wildlife-only remaining score.
    /// `w3_habitat`/`b3_habitat` predicts (habitat + nature_tokens + habitat_bonus)
    /// remaining score.
    pub has_split_value_heads: bool,
    pub w3_wildlife: Vec<f32>,  // [HIDDEN2]
    pub b3_wildlife: f32,
    pub w3_habitat: Vec<f32>,   // [HIDDEN2]
    pub b3_habitat: f32,
}

impl NNUENetwork {
    /// Create a new network with Xavier initialization.
    pub fn new() -> Self {
        use std::f32::consts::PI;
        let mut seed: u64 = 12345;
        let mut rand_f32 = move || -> f32 {
            // Simple xorshift64 + Box-Muller for normal distribution
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u1 = (seed as f32) / (u64::MAX as f32);
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u2 = (seed as f32) / (u64::MAX as f32);
            let u1 = u1.max(1e-10);
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };

        // Xavier initialization: scale = sqrt(2 / fan_in)
        let scale1 = (2.0 / 46.0_f32).sqrt(); // ~46 active features on average
        let scale2 = (2.0 / HIDDEN1 as f32).sqrt();
        let scale3 = (2.0 / HIDDEN2 as f32).sqrt();

        let w1: Vec<f32> = (0..NUM_FEATURES * HIDDEN1)
            .map(|_| rand_f32() * scale1)
            .collect();
        let b1 = vec![0.0; HIDDEN1];

        let w2: Vec<f32> = (0..HIDDEN1 * HIDDEN2)
            .map(|_| rand_f32() * scale2)
            .collect();
        let b2 = vec![0.0; HIDDEN2];

        let w3: Vec<f32> = (0..HIDDEN2).map(|_| rand_f32() * scale3).collect();
        let b3 = 0.0;

        // Policy head initialized to zero (no effect until trained)
        let w3_policy = vec![0.0; HIDDEN2];
        let b3_policy = 0.0;

        // Split value heads off by default — legacy total value head is used.
        let w3_wildlife = vec![0.0; HIDDEN2];
        let b3_wildlife = 0.0;
        let w3_habitat = vec![0.0; HIDDEN2];
        let b3_habitat = 0.0;

        NNUENetwork {
            w1, b1, w2, b2, w3, b3,
            w3_policy, b3_policy,
            has_split_value_heads: false,
            w3_wildlife, b3_wildlife, w3_habitat, b3_habitat,
        }
    }

    /// Average weights from multiple trained copies (for parallel training).
    pub fn average_from(&mut self, others: &[NNUENetwork]) {
        let n = others.len() as f32;
        // W1 (flat)
        for idx in 0..NUM_FEATURES * HIDDEN1 {
            let sum: f32 = others.iter().map(|o| o.w1[idx]).sum();
            self.w1[idx] = sum / n;
        }
        // b1
        for j in 0..HIDDEN1 {
            let sum: f32 = others.iter().map(|o| o.b1[j]).sum();
            self.b1[j] = sum / n;
        }
        // W2 (flat)
        for idx in 0..HIDDEN1 * HIDDEN2 {
            let sum: f32 = others.iter().map(|o| o.w2[idx]).sum();
            self.w2[idx] = sum / n;
        }
        // b2
        for j in 0..HIDDEN2 {
            let sum: f32 = others.iter().map(|o| o.b2[j]).sum();
            self.b2[j] = sum / n;
        }
        // W3
        for j in 0..HIDDEN2 {
            let sum: f32 = others.iter().map(|o| o.w3[j]).sum();
            self.w3[j] = sum / n;
        }
        self.b3 = others.iter().map(|o| o.b3).sum::<f32>() / n;
        // Policy head
        for j in 0..HIDDEN2 {
            let sum: f32 = others.iter().map(|o| o.w3_policy[j]).sum();
            self.w3_policy[j] = sum / n;
        }
        self.b3_policy = others.iter().map(|o| o.b3_policy).sum::<f32>() / n;

        // Split value heads (only meaningful if all inputs have them set)
        let all_split = others.iter().all(|o| o.has_split_value_heads);
        self.has_split_value_heads = all_split;
        if all_split {
            for j in 0..HIDDEN2 {
                let sw: f32 = others.iter().map(|o| o.w3_wildlife[j]).sum();
                self.w3_wildlife[j] = sw / n;
                let sh: f32 = others.iter().map(|o| o.w3_habitat[j]).sum();
                self.w3_habitat[j] = sh / n;
            }
            self.b3_wildlife = others.iter().map(|o| o.b3_wildlife).sum::<f32>() / n;
            self.b3_habitat = others.iter().map(|o| o.b3_habitat).sum::<f32>() / n;
        }
    }

    /// Full forward pass from sparse features. Returns predicted value.
    /// Uses stack-allocated buffers and flat contiguous weight access.
    pub fn forward(&self, features: &[u16]) -> f32 {
        // Layer 1: accumulate active feature columns + bias (stack-allocated)
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            let col = &self.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 {
                h1[j] += col[j];
            }
        }
        // ReLU
        for v in h1.iter_mut() {
            *v = v.max(0.0);
        }

        // Layer 2 (stack-allocated)
        let mut h2 = [0.0f32; HIDDEN2];
        h2.copy_from_slice(&self.b2);
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                let row = &self.w2[base..base + HIDDEN2];
                for j in 0..HIDDEN2 {
                    h2[j] += h1[i] * row[j];
                }
            }
        }
        // ReLU
        for v in h2.iter_mut() {
            *v = v.max(0.0);
        }

        // Output (value head): split heads if enabled, legacy total head otherwise.
        if self.has_split_value_heads {
            let mut wildlife = self.b3_wildlife;
            let mut habitat = self.b3_habitat;
            for j in 0..HIDDEN2 {
                wildlife += h2[j] * self.w3_wildlife[j];
                habitat += h2[j] * self.w3_habitat[j];
            }
            // 1:1 sum — wildlife + habitat components (no variable blending)
            return wildlife + habitat;
        }

        let mut out = self.b3;
        for j in 0..HIDDEN2 {
            out += h2[j] * self.w3[j];
        }

        out
    }

    /// Forward pass returning both value and policy logit.
    /// Policy logit is a raw scalar — softmax over candidates externally.
    pub fn forward_dual(&self, features: &[u16]) -> (f32, f32) {
        // Layer 1
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            let col = &self.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 {
                h1[j] += col[j];
            }
        }
        for v in h1.iter_mut() { *v = v.max(0.0); }

        // Layer 2
        let mut h2 = [0.0f32; HIDDEN2];
        h2.copy_from_slice(&self.b2);
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                let row = &self.w2[base..base + HIDDEN2];
                for j in 0..HIDDEN2 { h2[j] += h1[i] * row[j]; }
            }
        }
        for v in h2.iter_mut() { *v = v.max(0.0); }

        // Value head: split heads if enabled, legacy total head otherwise
        let value = if self.has_split_value_heads {
            let mut wildlife = self.b3_wildlife;
            let mut habitat = self.b3_habitat;
            for j in 0..HIDDEN2 {
                wildlife += h2[j] * self.w3_wildlife[j];
                habitat += h2[j] * self.w3_habitat[j];
            }
            wildlife + habitat
        } else {
            let mut v = self.b3;
            for j in 0..HIDDEN2 { v += h2[j] * self.w3[j]; }
            v
        };

        // Policy head
        let mut policy = self.b3_policy;
        for j in 0..HIDDEN2 { policy += h2[j] * self.w3_policy[j]; }

        (value, policy)
    }

    /// Evaluate a board position directly (without bag info).
    pub fn evaluate(&self, board: &Board) -> f32 {
        let features = extract_features(board);
        self.forward(&features)
    }

    /// Evaluate a board position with bag composition info.
    pub fn evaluate_with_bag(&self, board: &Board, bag: &BagInfo) -> f32 {
        let features = extract_features_with_bag(board, Some(bag));
        self.forward(&features)
    }

    /// Fast forward pass using a pre-computed accumulator for layer 1.
    /// Skips the sparse feature accumulation entirely — just applies ReLU + layers 2-3.
    pub fn forward_from_accumulator(&self, acc: &Accumulator, extra_features: &[u16]) -> f32 {
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&acc.values);
        // Add any extra features (bag, opponent habitat) not tracked in accumulator
        for &fi in extra_features {
            let base = fi as usize * HIDDEN1;
            let col = &self.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 {
                h1[j] += col[j];
            }
        }
        // ReLU
        for v in h1.iter_mut() { *v = v.max(0.0); }
        // Layer 2
        let mut h2 = [0.0f32; HIDDEN2];
        h2.copy_from_slice(&self.b2);
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                let row = &self.w2[base..base + HIDDEN2];
                for j in 0..HIDDEN2 { h2[j] += h1[i] * row[j]; }
            }
        }
        for v in h2.iter_mut() { *v = v.max(0.0); }
        // Output: split heads if enabled, legacy total head otherwise.
        if self.has_split_value_heads {
            let mut wildlife = self.b3_wildlife;
            let mut habitat = self.b3_habitat;
            for j in 0..HIDDEN2 {
                wildlife += h2[j] * self.w3_wildlife[j];
                habitat += h2[j] * self.w3_habitat[j];
            }
            return wildlife + habitat;
        }
        let mut out = self.b3;
        for j in 0..HIDDEN2 { out += h2[j] * self.w3[j]; }
        out
    }

    /// Add a feature to the first-layer accumulator.
    #[inline]
    pub fn accumulator_add(&self, acc: &mut Accumulator, feature: u16) {
        let base = feature as usize * HIDDEN1;
        let col = &self.w1[base..base + HIDDEN1];
        for j in 0..HIDDEN1 {
            acc.values[j] += col[j];
        }
    }

    /// Remove a feature from the first-layer accumulator.
    #[inline]
    pub fn accumulator_sub(&self, acc: &mut Accumulator, feature: u16) {
        let base = feature as usize * HIDDEN1;
        let col = &self.w1[base..base + HIDDEN1];
        for j in 0..HIDDEN1 {
            acc.values[j] -= col[j];
        }
    }

    /// Train on a single sample. Returns the loss (MSE).
    /// Uses backpropagation with SGD.
    pub fn train_sample(&mut self, features: &[u16], target: f32, lr: f32) -> f32 {
        // ─── Forward pass (save intermediates, stack-allocated) ───
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            let col = &self.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 {
                h1[j] += col[j];
            }
        }
        let mut h1_pre = [0.0f32; HIDDEN1];
        h1_pre.copy_from_slice(&h1);
        for v in h1.iter_mut() {
            *v = v.max(0.0);
        }

        let mut h2 = [0.0f32; HIDDEN2];
        h2.copy_from_slice(&self.b2);
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                let row = &self.w2[base..base + HIDDEN2];
                for j in 0..HIDDEN2 {
                    h2[j] += h1[i] * row[j];
                }
            }
        }
        let mut h2_pre = [0.0f32; HIDDEN2];
        h2_pre.copy_from_slice(&h2);
        for v in h2.iter_mut() {
            *v = v.max(0.0);
        }

        let mut out = self.b3;
        for j in 0..HIDDEN2 {
            out += h2[j] * self.w3[j];
        }

        // ─── Loss ───
        let error = out - target;
        let loss = error * error;

        // ─── Backward pass ───
        let d_out = 2.0 * error * lr;

        // Output layer gradients
        for j in 0..HIDDEN2 {
            self.w3[j] -= d_out * h2[j];
        }
        self.b3 -= d_out;

        // d_h2 = d_out * w3 * relu'(h2_pre)
        let mut d_h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            if h2_pre[j] > 0.0 {
                d_h2[j] = d_out * self.w3[j];
            }
        }

        // Layer 2 gradients
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                for j in 0..HIDDEN2 {
                    self.w2[base + j] -= d_h2[j] * h1[i];
                }
            }
        }
        for j in 0..HIDDEN2 {
            self.b2[j] -= d_h2[j];
        }

        // d_h1 = W2^T @ d_h2 * relu'(h1_pre)
        let mut d_h1 = [0.0f32; HIDDEN1];
        for i in 0..HIDDEN1 {
            if h1_pre[i] > 0.0 {
                let base = i * HIDDEN2;
                for j in 0..HIDDEN2 {
                    d_h1[i] += self.w2[base + j] * d_h2[j];
                }
            }
        }

        // Layer 1 gradients (only active features)
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            for j in 0..HIDDEN1 {
                self.w1[base + j] -= d_h1[j];
            }
        }
        for j in 0..HIDDEN1 {
            self.b1[j] -= d_h1[j];
        }

        loss
    }

    /// Train on a single sample, but only update weights for features >= freeze_below.
    /// Layers 2, 3, and biases are also frozen. This lets new features learn
    /// while preserving existing representations.
    pub fn train_sample_frozen(&mut self, features: &[u16], target: f32, lr: f32, freeze_below: usize) -> f32 {
        // Forward pass (same as train_sample)
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            let col = &self.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 { h1[j] += col[j]; }
        }
        let mut h1_pre = [0.0f32; HIDDEN1];
        h1_pre.copy_from_slice(&h1);
        for v in h1.iter_mut() { *v = v.max(0.0); }

        let mut h2 = [0.0f32; HIDDEN2];
        h2.copy_from_slice(&self.b2);
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                let row = &self.w2[base..base + HIDDEN2];
                for j in 0..HIDDEN2 { h2[j] += h1[i] * row[j]; }
            }
        }
        let h2_pre = h2;
        for v in h2.iter_mut() { *v = v.max(0.0); }

        let mut out = self.b3;
        for j in 0..HIDDEN2 { out += h2[j] * self.w3[j]; }

        let error = out - target;
        let loss = error * error;
        let d_out = 2.0 * error * lr;

        // Layers 2, 3, biases: FROZEN (no updates)

        // Backprop through frozen layers to get d_h1
        let mut d_h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            if h2_pre[j] > 0.0 { d_h2[j] = d_out * self.w3[j]; }
        }
        let mut d_h1 = [0.0f32; HIDDEN1];
        for i in 0..HIDDEN1 {
            if h1_pre[i] > 0.0 {
                let base = i * HIDDEN2;
                for j in 0..HIDDEN2 { d_h1[i] += self.w2[base + j] * d_h2[j]; }
            }
        }

        // Layer 1: only update NEW features (>= freeze_below)
        for &fi in features {
            if (fi as usize) >= freeze_below {
                let base = fi as usize * HIDDEN1;
                for j in 0..HIDDEN1 {
                    self.w1[base + j] -= d_h1[j];
                }
            }
        }
        // Biases frozen too

        loss
    }

    /// Save weights to a binary file.
    /// Train on a RANKING batch: given a position with N candidate afterstates and
    /// their MCE scores, update ALL weights via cross-entropy on softmax(MCE/τ).
    ///
    /// This trains the ENTIRE network as a ranking model — every layer learns to
    /// produce logits that rank candidates correctly, not predict absolute scores.
    pub fn train_ranking_batch(
        &mut self,
        position: &PositionPolicyData,
        lr: f32,
        temperature: f32,
    ) -> (f32, bool) {  // (loss, top1_correct)
        let n = position.candidates.len();
        if n < 2 { return (0.0, false); }

        // Forward all candidates, save intermediates for backprop
        let mut h1s: Vec<[f32; HIDDEN1]> = Vec::with_capacity(n);
        let mut h1_pres: Vec<[f32; HIDDEN1]> = Vec::with_capacity(n);
        let mut h2s: Vec<[f32; HIDDEN2]> = Vec::with_capacity(n);
        let mut h2_pres: Vec<[f32; HIDDEN2]> = Vec::with_capacity(n);
        let mut logits: Vec<f32> = Vec::with_capacity(n);

        for (feats, _) in &position.candidates {
            let mut h1 = [0.0f32; HIDDEN1];
            h1.copy_from_slice(&self.b1);
            for &fi in feats {
                let base = fi as usize * HIDDEN1;
                if base + HIDDEN1 > self.w1.len() { continue; }
                let col = &self.w1[base..base + HIDDEN1];
                for j in 0..HIDDEN1 { h1[j] += col[j]; }
            }
            let mut h1_pre = [0.0f32; HIDDEN1];
            h1_pre.copy_from_slice(&h1);
            for v in h1.iter_mut() { *v = v.max(0.0); }

            let mut h2 = [0.0f32; HIDDEN2];
            h2.copy_from_slice(&self.b2);
            for i in 0..HIDDEN1 {
                if h1[i] > 0.0 {
                    let base = i * HIDDEN2;
                    let row = &self.w2[base..base + HIDDEN2];
                    for j in 0..HIDDEN2 { h2[j] += h1[i] * row[j]; }
                }
            }
            let mut h2_pre = [0.0f32; HIDDEN2];
            h2_pre.copy_from_slice(&h2);
            for v in h2.iter_mut() { *v = v.max(0.0); }

            // Use value head output as ranking logit
            let mut out = self.b3;
            for j in 0..HIDDEN2 { out += h2[j] * self.w3[j]; }
            logits.push(out);

            h1s.push(h1);
            h1_pres.push(h1_pre);
            h2s.push(h2);
            h2_pres.push(h2_pre);
        }

        // Target: softmax(MCE_scores / temperature)
        let mce_scores: Vec<f32> = position.candidates.iter().map(|(_, s)| *s).collect();
        let max_mce = mce_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let target: Vec<f32> = {
            let exps: Vec<f32> = mce_scores.iter().map(|s| ((s - max_mce) / temperature).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum.max(1e-8)).collect()
        };

        // Predicted: softmax(logits)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pred: Vec<f32> = {
            let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum.max(1e-8)).collect()
        };

        // Loss
        let loss: f32 = target.iter().zip(pred.iter())
            .map(|(t, p)| -t * p.max(1e-8).ln()).sum();

        // Top-1 check
        let mce_best = mce_scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let pred_best = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);

        // Backprop for each candidate
        let scaled_lr = lr / n as f32;
        for ci in 0..n {
            let d_out = (pred[ci] - target[ci]) * scaled_lr;

            // Output layer
            for j in 0..HIDDEN2 {
                self.w3[j] -= d_out * h2s[ci][j];
            }
            self.b3 -= d_out;

            // d_h2
            let mut d_h2 = [0.0f32; HIDDEN2];
            for j in 0..HIDDEN2 {
                if h2_pres[ci][j] > 0.0 { d_h2[j] = d_out * self.w3[j]; }
            }

            // Layer 2
            for i in 0..HIDDEN1 {
                if h1s[ci][i] > 0.0 {
                    let base = i * HIDDEN2;
                    for j in 0..HIDDEN2 { self.w2[base + j] -= d_h2[j] * h1s[ci][i]; }
                }
            }
            for j in 0..HIDDEN2 { self.b2[j] -= d_h2[j]; }

            // d_h1
            let mut d_h1 = [0.0f32; HIDDEN1];
            for i in 0..HIDDEN1 {
                if h1_pres[ci][i] > 0.0 {
                    let base = i * HIDDEN2;
                    for j in 0..HIDDEN2 { d_h1[i] += d_h2[j] * self.w2[base + j]; }
                }
            }

            // Layer 1 (sparse update)
            let feats = &position.candidates[ci].0;
            for &fi in feats {
                let base = fi as usize * HIDDEN1;
                if base + HIDDEN1 > self.w1.len() { continue; }
                for j in 0..HIDDEN1 { self.w1[base + j] -= d_h1[j]; }
            }
            for j in 0..HIDDEN1 { self.b1[j] -= d_h1[j]; }
        }

        (loss, mce_best == pred_best)
    }

    /// Forward pass returning the hidden layer 2 activations (post-ReLU).
    /// Used for policy head training data collection.
    pub fn forward_hidden(&self, features: &[u16]) -> [f32; HIDDEN2] {
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            if base + HIDDEN1 > self.w1.len() { continue; }
            let col = &self.w1[base..base + HIDDEN1];
            for j in 0..HIDDEN1 { h1[j] += col[j]; }
        }
        for v in h1.iter_mut() { *v = v.max(0.0); }

        let mut h2 = [0.0f32; HIDDEN2];
        h2.copy_from_slice(&self.b2);
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                let row = &self.w2[base..base + HIDDEN2];
                for j in 0..HIDDEN2 { h2[j] += h1[i] * row[j]; }
            }
        }
        for v in h2.iter_mut() { *v = v.max(0.0); }
        h2
    }

    /// Train the policy head on a batch of positions using FEATURE DELTAS.
    ///
    /// Each position has a base feature set (board before move) and N candidates,
    /// each with (afterstate_features, mce_score). The policy scores the DELTA
    /// between afterstate and base features using a separate linear model stored
    /// in w1's first HIDDEN1 columns (reusing existing weight storage).
    ///
    /// Actually, we use a standalone delta-weight vector stored in w3_policy
    /// (repurposed: now [NUM_FEATURES] instead of [HIDDEN2]). This requires
    /// resizing w3_policy on first call.
    ///
    /// Returns (avg_loss, top1_agreement_pct).
    pub fn train_policy_delta(
        &mut self,
        positions: &[PositionPolicyData],
        lr: f32,
        temperature: f32,
        delta_weights: &mut Vec<f32>,  // [NUM_FEATURES] — caller owns, persists across epochs
    ) -> (f32, f32) {
        let mut total_loss = 0.0f64;
        let mut agree = 0usize;
        let mut n_positions = 0usize;

        for pos in positions {
            if pos.candidates.len() < 2 { continue; }
            n_positions += 1;

            // Compute delta features for each candidate: features in afterstate but not in base
            // (and vice versa). Since features are sparse binary, delta = symmetric difference.
            let base_set: std::collections::HashSet<u16> = pos.base_features.iter().copied().collect();

            let deltas: Vec<Vec<(u16, f32)>> = pos.candidates.iter().map(|(feats, _)| {
                let after_set: std::collections::HashSet<u16> = feats.iter().copied().collect();
                let mut delta = Vec::new();
                // Features added by the move (+1)
                for &f in feats {
                    if !base_set.contains(&f) && (f as usize) < NUM_FEATURES {
                        delta.push((f, 1.0));
                    }
                }
                // Features removed by the move (-1)
                for &f in &pos.base_features {
                    if !after_set.contains(&f) && (f as usize) < NUM_FEATURES {
                        delta.push((f, -1.0));
                    }
                }
                delta
            }).collect();

            // Policy logits: dot(delta_weights, delta_features) for each candidate
            let policy_logits: Vec<f32> = deltas.iter().map(|delta| {
                let mut logit = 0.0f32;
                for &(fi, sign) in delta {
                    logit += delta_weights[fi as usize] * sign;
                }
                logit
            }).collect();

            // Target: softmax(mce_scores / temperature)
            let mce_scores: Vec<f32> = pos.candidates.iter().map(|(_, s)| *s).collect();
            let max_mce = mce_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let target: Vec<f32> = {
                let exps: Vec<f32> = mce_scores.iter()
                    .map(|s| ((s - max_mce) / temperature).exp())
                    .collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum.max(1e-8)).collect()
            };

            // Predicted: softmax(policy_logits)
            let max_logit = policy_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let pred: Vec<f32> = {
                let exps: Vec<f32> = policy_logits.iter()
                    .map(|l| (l - max_logit).exp())
                    .collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum.max(1e-8)).collect()
            };

            // Cross-entropy loss
            let loss: f32 = target.iter().zip(pred.iter())
                .map(|(t, p)| -t * p.max(1e-8).ln())
                .sum();
            total_loss += loss as f64;

            // Top-1 agreement
            let mce_best = mce_scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            let policy_best = policy_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            if mce_best == policy_best { agree += 1; }

            // Gradient: d_logit_i = pred_i - target_i
            let n = pos.candidates.len() as f32;
            for (ci, delta) in deltas.iter().enumerate() {
                let d = (pred[ci] - target[ci]) * lr / n;
                for &(fi, sign) in delta {
                    delta_weights[fi as usize] -= d * sign;
                }
            }
        }

        let avg_loss = if n_positions > 0 { (total_loss / n_positions as f64) as f32 } else { 0.0 };
        let agree_pct = if n_positions > 0 { 100.0 * agree as f32 / n_positions as f32 } else { 0.0 };
        (avg_loss, agree_pct)
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // version=1: legacy single value head + policy head
        // version=2: adds split value heads (wildlife + habitat) appended at end
        let version: u32 = if self.has_split_value_heads { 2 } else { 1 };
        file.write_all(b"NNUE")?;
        file.write_all(&version.to_le_bytes())?;

        // W1: flat [NUM_FEATURES * HIDDEN1]
        for &v in &self.w1 {
            file.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.b1 {
            file.write_all(&v.to_le_bytes())?;
        }

        // W2: flat [HIDDEN1 * HIDDEN2]
        for &v in &self.w2 {
            file.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.b2 {
            file.write_all(&v.to_le_bytes())?;
        }

        // W3 + b3 (legacy value head, kept for back-compat even with split heads)
        for &v in &self.w3 {
            file.write_all(&v.to_le_bytes())?;
        }
        file.write_all(&self.b3.to_le_bytes())?;

        // Policy head: w3_policy + b3_policy
        for &v in &self.w3_policy {
            file.write_all(&v.to_le_bytes())?;
        }
        file.write_all(&self.b3_policy.to_le_bytes())?;

        // Split value heads (v2 only): w3_wildlife + b3_wildlife + w3_habitat + b3_habitat
        if self.has_split_value_heads {
            for &v in &self.w3_wildlife {
                file.write_all(&v.to_le_bytes())?;
            }
            file.write_all(&self.b3_wildlife.to_le_bytes())?;
            for &v in &self.w3_habitat {
                file.write_all(&v.to_le_bytes())?;
            }
            file.write_all(&self.b3_habitat.to_le_bytes())?;
        }

        Ok(())
    }

    /// Load weights from a binary file. Supports version 1 (single value head)
    /// and version 2 (split value heads appended after policy head).
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"NNUE" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut ver_buf = [0u8; 4];
        file.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);

        let mut buf = [0u8; 4];
        let mut read_f32 = |f: &mut std::fs::File| -> std::io::Result<f32> {
            f.read_exact(&mut buf)?;
            Ok(f32::from_le_bytes(buf))
        };

        // Layout (from start): header(8) + w1 + b1 + w2 + b2 + w3 + b3 + w3_policy + b3_policy + [split heads]
        // We don't know w1_features upfront (legacy files had smaller NUM_FEATURES). Compute it by
        // subtracting everything else from the file size.
        let file_size = file.metadata()?.len();
        let header_size = 8u64;
        let fixed_after_w1 = ((HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 + HIDDEN2 + 1) as u64) * 4;
        let policy_head_size = (HIDDEN2 + 1) as u64 * 4;
        let split_head_size = (2 * (HIDDEN2 + 1)) as u64 * 4; // w3_wildlife + b + w3_habitat + b
        // The v1 file has no split heads appended, v2 does.
        let trailing_size = policy_head_size + if version >= 2 { split_head_size } else { 0 };
        // Compute w1 features from remaining bytes. If the file is v1 without a policy head
        // (truly legacy), fall back by assuming zero trailing. That's best-effort.
        let mut w1_bytes = file_size.saturating_sub(header_size + fixed_after_w1 + trailing_size);
        let mut trailing_has_policy = true;
        let mut trailing_has_split = version >= 2;
        let mut computed_features = w1_bytes / (HIDDEN1 as u64 * 4);
        if computed_features > NUM_FEATURES as u64 {
            // Stored in a newer file with more features than we know — cap and warn.
            computed_features = NUM_FEATURES as u64;
        }
        // If this arithmetic looks insane (e.g. v1 file without policy head), retry without policy.
        if w1_bytes == 0 || (w1_bytes % (HIDDEN1 as u64 * 4)) != 0 {
            let w1_bytes_no_trailing = file_size.saturating_sub(header_size + fixed_after_w1);
            if w1_bytes_no_trailing > 0 && (w1_bytes_no_trailing % (HIDDEN1 as u64 * 4)) == 0 {
                w1_bytes = w1_bytes_no_trailing;
                trailing_has_policy = false;
                trailing_has_split = false;
                computed_features = (w1_bytes / (HIDDEN1 as u64 * 4)).min(NUM_FEATURES as u64);
            }
        }
        let w1_features = computed_features as usize;
        let w1_features = w1_features.min(NUM_FEATURES);

        let mut w1 = Vec::with_capacity(NUM_FEATURES * HIDDEN1);
        for _ in 0..w1_features * HIDDEN1 {
            w1.push(read_f32(&mut file)?);
        }
        w1.resize(NUM_FEATURES * HIDDEN1, 0.0);
        let mut b1 = Vec::with_capacity(HIDDEN1);
        for _ in 0..HIDDEN1 {
            b1.push(read_f32(&mut file)?);
        }

        let mut w2 = Vec::with_capacity(HIDDEN1 * HIDDEN2);
        for _ in 0..HIDDEN1 * HIDDEN2 {
            w2.push(read_f32(&mut file)?);
        }
        let mut b2 = Vec::with_capacity(HIDDEN2);
        for _ in 0..HIDDEN2 {
            b2.push(read_f32(&mut file)?);
        }

        let mut w3 = Vec::with_capacity(HIDDEN2);
        for _ in 0..HIDDEN2 {
            w3.push(read_f32(&mut file)?);
        }
        let b3 = read_f32(&mut file)?;

        // Policy head (optional — backward compatible with pre-policy-head files)
        let (w3_policy, b3_policy) = if trailing_has_policy {
            let mut wp = Vec::with_capacity(HIDDEN2);
            for _ in 0..HIDDEN2 {
                wp.push(read_f32(&mut file)?);
            }
            let bp = read_f32(&mut file)?;
            (wp, bp)
        } else {
            (vec![0.0; HIDDEN2], 0.0)
        };

        // Split value heads (v2 only)
        let (has_split, w3_wildlife, b3_wildlife, w3_habitat, b3_habitat) = if trailing_has_split {
            let mut ww = Vec::with_capacity(HIDDEN2);
            for _ in 0..HIDDEN2 {
                ww.push(read_f32(&mut file)?);
            }
            let bw = read_f32(&mut file)?;
            let mut wh = Vec::with_capacity(HIDDEN2);
            for _ in 0..HIDDEN2 {
                wh.push(read_f32(&mut file)?);
            }
            let bh = read_f32(&mut file)?;
            (true, ww, bw, wh, bh)
        } else {
            (false, vec![0.0; HIDDEN2], 0.0, vec![0.0; HIDDEN2], 0.0)
        };

        Ok(NNUENetwork {
            w1, b1, w2, b2, w3, b3,
            w3_policy, b3_policy,
            has_split_value_heads: has_split,
            w3_wildlife, b3_wildlife, w3_habitat, b3_habitat,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Standalone Policy Network (separate from value NNUE)
// ─────────────────────────────────────────────────────────────────────

/// Separate policy network for candidate ranking.
/// Independent architecture from the value NNUE — can have different hidden sizes.
pub struct PolicyNetwork {
    pub num_features: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub w1: Vec<f32>,  // [num_features * hidden1]
    pub b1: Vec<f32>,  // [hidden1]
    pub w2: Vec<f32>,  // [hidden1 * hidden2]
    pub b2: Vec<f32>,  // [hidden2]
    pub w3: Vec<f32>,  // [hidden2]
    pub b3: f32,
}

impl PolicyNetwork {
    /// Forward pass — returns a single policy logit for this afterstate.
    pub fn forward(&self, features: &[u16]) -> f32 {
        let mut h1 = vec![0.0f32; self.hidden1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * self.hidden1;
            if base + self.hidden1 > self.w1.len() { continue; }
            let col = &self.w1[base..base + self.hidden1];
            for j in 0..self.hidden1 {
                h1[j] += col[j];
            }
        }
        for v in h1.iter_mut() { *v = v.max(0.0); }

        let mut h2 = vec![0.0f32; self.hidden2];
        h2.copy_from_slice(&self.b2);
        for i in 0..self.hidden1 {
            if h1[i] > 0.0 {
                let base = i * self.hidden2;
                let row = &self.w2[base..base + self.hidden2];
                for j in 0..self.hidden2 { h2[j] += h1[i] * row[j]; }
            }
        }
        for v in h2.iter_mut() { *v = v.max(0.0); }

        let mut out = self.b3;
        for j in 0..self.hidden2 { out += h2[j] * self.w3[j]; }
        out
    }

    /// Score multiple candidates and return softmax probabilities.
    pub fn rank_candidates(&self, candidate_features: &[Vec<u16>]) -> Vec<f32> {
        let logits: Vec<f32> = candidate_features.iter()
            .map(|f| self.forward(f))
            .collect();
        // Softmax
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Load from PLCY binary format.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];

        file.read_exact(&mut buf4)?;
        if &buf4 != b"PLCY" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        file.read_exact(&mut buf4)?; // version

        let mut read_u32 = |f: &mut std::fs::File| -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let mut read_f32 = |f: &mut std::fs::File| -> std::io::Result<f32> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        };

        let num_features = read_u32(&mut file)? as usize;
        let hidden1 = read_u32(&mut file)? as usize;
        let hidden2 = read_u32(&mut file)? as usize;

        let mut w1 = vec![0.0f32; num_features * hidden1];
        for v in w1.iter_mut() { *v = read_f32(&mut file)?; }
        let mut b1 = vec![0.0f32; hidden1];
        for v in b1.iter_mut() { *v = read_f32(&mut file)?; }

        let mut w2 = vec![0.0f32; hidden1 * hidden2];
        for v in w2.iter_mut() { *v = read_f32(&mut file)?; }
        let mut b2 = vec![0.0f32; hidden2];
        for v in b2.iter_mut() { *v = read_f32(&mut file)?; }

        let mut w3 = vec![0.0f32; hidden2];
        for v in w3.iter_mut() { *v = read_f32(&mut file)?; }
        let b3 = read_f32(&mut file)?;

        eprintln!("Loaded PolicyNetwork: {}→{}→{}→1", num_features, hidden1, hidden2);

        Ok(PolicyNetwork { num_features, hidden1, hidden2, w1, b1, w2, b2, w3, b3 })
    }
}
