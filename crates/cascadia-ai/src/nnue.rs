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

// ── v5-feat: frontier + richer opp patterns + bonus distance + habitat structure
//             + tile-bag joint distribution ──────────────────────────────
//
// Block layout (all gated on v5-feat, appended after v4-opp block):
//   V5_FRONTIER_BASE       : 441 cells × 1 bit = 441 (is on frontier)
//   V5_FRONT_WL_BASE       : 441 cells × 5 bits = 2205 (this frontier cell has wildlife W in any neighbor)
//   V5_FRONT_TERR_BASE     : 441 cells × 5 bits = 2205 (this frontier cell touches a terrain T on a shared edge)
//   V5_OPP_PAT_BASE        : 3 opps × 36 = 108 (richer P4 pattern histograms)
//   V5_BONUS_DIST_BASE     : 5 terr × 3 opps × 21 bins = 315 (signed (my - opp) hab diff)
//   V5_HAB_STRUCT_BASE     : 5 terr × (6 cluster-count bins + 11 second-largest bins) = 85
//   V5_TBAG_JOINT_BASE     : 5 terr × 5 wildlife × 21 bins = 525 (joint count of bag tiles with terrain T AND allowed wildlife W)
//
// Total v5-feat additions: 5,884 features.
pub const V5_FRONTIER_FLAG_FEATURES: usize = GRID_SIZE; // 441
pub const V5_FRONT_WL_FEATURES: usize = GRID_SIZE * 5;  // 2205
pub const V5_FRONT_TERR_FEATURES: usize = GRID_SIZE * 5; // 2205

// Per-opp pattern histograms (P4): bear singletons, longest elk, longest salmon, isolated hawks
pub const V5_OPP_BEAR_SING_BINS: usize = 8;   // 0..7+
pub const V5_OPP_ELK_LINE_BINS: usize = 8;    // 0..7+
pub const V5_OPP_SALMON_RUN_BINS: usize = 10; // 0..9+
pub const V5_OPP_ISOL_HAWK_BINS: usize = 10;  // 0..9+
pub const V5_OPP_PAT_PER_OPP: usize = V5_OPP_BEAR_SING_BINS + V5_OPP_ELK_LINE_BINS
    + V5_OPP_SALMON_RUN_BINS + V5_OPP_ISOL_HAWK_BINS; // 36
pub const V5_OPP_PAT_FEATURES: usize = NUM_OPP_SLOTS * V5_OPP_PAT_PER_OPP; // 108

// Bonus-threshold-distance (P5): per terrain × opp, signed diff bins -10..-1, 0, 1..10+
pub const V5_BONUS_DIFF_BINS: usize = 21; // -10..=-1, 0, 1..=10+
pub const V5_BONUS_DIST_FEATURES: usize = 5 * NUM_OPP_SLOTS * V5_BONUS_DIFF_BINS; // 315

// Habitat structure (P6): per terrain, count of distinct clusters + 2nd largest size
pub const V5_HAB_CLUSTER_COUNT_BINS: usize = 6; // 0..5+
pub const V5_HAB_SECOND_BINS: usize = 11;       // 0..10+
pub const V5_HAB_STRUCT_PER_TERR: usize = V5_HAB_CLUSTER_COUNT_BINS + V5_HAB_SECOND_BINS; // 17
pub const V5_HAB_STRUCT_FEATURES: usize = 5 * V5_HAB_STRUCT_PER_TERR; // 85

// Tile-bag joint terrain × wildlife distribution (P7)
pub const V5_TBAG_JOINT_BINS: usize = 21; // 0..20+
pub const V5_TBAG_JOINT_FEATURES: usize = 5 * 5 * V5_TBAG_JOINT_BINS; // 525

pub const V5_FEAT_FEATURES: usize = V5_FRONTIER_FLAG_FEATURES
    + V5_FRONT_WL_FEATURES + V5_FRONT_TERR_FEATURES
    + V5_OPP_PAT_FEATURES + V5_BONUS_DIST_FEATURES
    + V5_HAB_STRUCT_FEATURES + V5_TBAG_JOINT_FEATURES; // 5884

pub const NUM_FEATURES_MID_V4_V5: usize = NUM_FEATURES_MID_V4 + V5_FEAT_FEATURES; // 17115
pub const NUM_FEATURES_V3_V4_V5: usize = NUM_FEATURES_V3_V4 + V5_FEAT_FEATURES;   // 51513

// ──────────────────────────────────────────────────────────────────────
// cards-alt feature block (Bear C, Elk B, Salmon D, Hawk D, Fox B)
// ──────────────────────────────────────────────────────────────────────
//
// Player-side (117 features): replaces the network's blind spots under the
// alt scoring set. The existing PATTERN_FEATURES / PATTERN_V2_FEATURES blocks
// remain in the index space (the network just learns to under-weight them
// under alt scoring) — these new features supply the missing alt-card signal.
//
// Bear C (21):    counts of components at sizes 1/2/3/4+ (5 bins each = 20)
//                 + all-three-sizes-present bit (1)
// Elk B (25):     counts of shapes singleton/pair/triangle/rhombus/blob (5 each)
// Salmon D (21):  for top-3 runs, qualifying-bit (1) + adj-non-salmon types
//                 (6 bins) = 7 per run × 3 = 21
// Hawk D (25):    count-by-intervening-types histogram, 5 type-classes ×
//                 5 count-bins = 25 (binned LOS-pair census)
// Fox B (25):     count-by-pair-types histogram, 5 pair-type-classes ×
//                 5 count-bins = 25 (binned per-fox pair-type census)
//
// Opponent-side (75 features): per-opponent alt-card threat summary, 25 each.
// Per opp: best bear size class (5), densest elk shape (5), best salmon
// adj-animal count (5), best hawk LOS-pair-types (5), best fox pair-types (5).

pub const ALT_BEAR_SIZE_BINS: usize = 5;       // 0/1/2/3/4+ count
pub const ALT_BEAR_FEATURES: usize = 4 * ALT_BEAR_SIZE_BINS + 1; // 4 sizes × 5 bins + all-3-sizes bit = 21

pub const ALT_ELK_SHAPE_BINS: usize = 5;       // 0/1/2/3/4+ count of each shape
pub const ALT_ELK_FEATURES: usize = 5 * ALT_ELK_SHAPE_BINS; // 5 shape kinds × 5 = 25

pub const ALT_SALMON_TOP_RUNS: usize = 3;
pub const ALT_SALMON_ADJ_BINS: usize = 6;      // 0/1/2/3/4/5+ unique adj types
pub const ALT_SALMON_PER_RUN: usize = 1 + ALT_SALMON_ADJ_BINS; // qualifies-bit + adj-bins = 7
pub const ALT_SALMON_FEATURES: usize = ALT_SALMON_TOP_RUNS * ALT_SALMON_PER_RUN; // 21

pub const ALT_HAWK_PT_CLASSES: usize = 5;      // 0/1/2/3/4+ unique types between
pub const ALT_HAWK_COUNT_BINS: usize = 5;      // 0/1/2/3/4+ pairs
pub const ALT_HAWK_FEATURES: usize = ALT_HAWK_PT_CLASSES * ALT_HAWK_COUNT_BINS; // 25

pub const ALT_FOX_PT_CLASSES: usize = 5;       // 0/1/2/3/4+ pair-types per fox
pub const ALT_FOX_COUNT_BINS: usize = 5;       // 0/1/2/3/4+ foxes
pub const ALT_FOX_FEATURES: usize = ALT_FOX_PT_CLASSES * ALT_FOX_COUNT_BINS; // 25

pub const ALT_PLAYER_FEATURES: usize = ALT_BEAR_FEATURES + ALT_ELK_FEATURES
    + ALT_SALMON_FEATURES + ALT_HAWK_FEATURES + ALT_FOX_FEATURES; // 117

pub const ALT_OPP_BEAR_BINS: usize = 5;        // best size class (no/single/pair/triple/quad+)
pub const ALT_OPP_ELK_SHAPE_BINS: usize = 5;   // best shape (no/single/pair/triangle/rhombus+)
pub const ALT_OPP_SALMON_BINS: usize = 5;      // 0/1/2/3/4+ best D-pts equivalent
pub const ALT_OPP_HAWK_BINS: usize = 5;        // best LOS-pair-types (0/1/2/3/4+)
pub const ALT_OPP_FOX_BINS: usize = 5;         // best per-fox pair-types (0/1/2/3+)
pub const ALT_OPP_PER_OPP: usize = ALT_OPP_BEAR_BINS + ALT_OPP_ELK_SHAPE_BINS
    + ALT_OPP_SALMON_BINS + ALT_OPP_HAWK_BINS + ALT_OPP_FOX_BINS; // 25
pub const ALT_OPP_FEATURES: usize = NUM_OPP_SLOTS * ALT_OPP_PER_OPP; // 75

pub const CARDS_ALT_FEATURES: usize = ALT_PLAYER_FEATURES + ALT_OPP_FEATURES; // 192

pub const NUM_FEATURES_MID_V4_ALT: usize = NUM_FEATURES_MID_V4 + CARDS_ALT_FEATURES; // 11423

// ──────────────────────────────────────────────────────────────────────
// cards-alt-v2: per-piece relational features
// ──────────────────────────────────────────────────────────────────────
//
// Replaces the cards-alt histogram approach with PER-CELL × pattern-class
// features. The 21×21 grid (441 cells) hosts at most 20 wildlife placements;
// per-piece features fire only for cells containing the relevant wildlife,
// giving direct credit assignment and preserving positional information that
// the histogram approach destroyed.
//
// Each block emits ONE feature per qualifying cell:
//   ALT2_HAWK_LOS:    127 cells × 6 dirs × 5 LOS-classes  = 3,810
//   ALT2_SALMON_CTX:  127 cells × 16 (4 len-classes × 4 adj-classes) = 2,032
//   ALT2_FOX_CTX:     127 cells × 16 (4 PT-classes × 4 density-classes) = 2,032
//   ALT2_BEAR_CTX:    127 cells × 12 (4 size-classes × 3 ext-classes) = 1,524
//   ALT2_ELK_CTX:     127 cells × 7 shape-roles  = 889
//
// Total: ~10,287 features. Only emitted when `cards-alt-v2` feature is on.
// Backward-compatible: appended after the existing cards-alt block, so a
// cards-alt-v15 weights file loads with zero-padding for the new columns.
// (V6_LOCAL_CELLS=127 reused as the bounded play region — empirically 99.9% of
// real placements; same trick v6-peak used.)
//
// Note: per-piece blocks are gated to fire ONLY on the player's own board (not
// per opponent — opponents are still summarized via the existing `alt_*_class`
// fields in OpponentDetail). This is consistent with how NNUE evaluates the
// player's afterstate.

pub const ALT2_HAWK_LOS_DIRS: usize = 6;
pub const ALT2_HAWK_LOS_CLASSES: usize = 5; // 0=no_partner, 1=adj_partner, 2=partner_0_types, 3=partner_1_type, 4=partner_2plus_types
pub const ALT2_HAWK_LOS_PER_CELL: usize = ALT2_HAWK_LOS_DIRS * ALT2_HAWK_LOS_CLASSES; // 30
pub const ALT2_HAWK_LOS_FEATURES: usize = V6_LOCAL_CELLS * ALT2_HAWK_LOS_PER_CELL; // 3810

pub const ALT2_SALMON_LEN_CLASSES: usize = 4; // 0=invalid_or_singleton, 1=len2, 2=len3-4, 3=len5+
pub const ALT2_SALMON_ADJ_CLASSES: usize = 4; // 0=0_types, 1=1_type, 2=2_types, 3=3+_types
pub const ALT2_SALMON_PER_CELL: usize = ALT2_SALMON_LEN_CLASSES * ALT2_SALMON_ADJ_CLASSES; // 16
pub const ALT2_SALMON_FEATURES: usize = V6_LOCAL_CELLS * ALT2_SALMON_PER_CELL; // 2032

pub const ALT2_FOX_PT_CLASSES: usize = 4; // 0=0_pair_types, 1=1, 2=2, 3=3+
pub const ALT2_FOX_DENSITY_CLASSES: usize = 4; // 0=0-1_adj, 1=2-3, 2=4-5, 3=6_adj
pub const ALT2_FOX_PER_CELL: usize = ALT2_FOX_PT_CLASSES * ALT2_FOX_DENSITY_CLASSES; // 16
pub const ALT2_FOX_FEATURES: usize = V6_LOCAL_CELLS * ALT2_FOX_PER_CELL; // 2032

pub const ALT2_BEAR_SIZE_CLASSES: usize = 4; // 0=size1, 1=size2, 2=size3, 3=size4+
pub const ALT2_BEAR_EXT_CLASSES: usize = 3; // 0=no_ext, 1=1-2_slots, 2=3+_slots
pub const ALT2_BEAR_PER_CELL: usize = ALT2_BEAR_SIZE_CLASSES * ALT2_BEAR_EXT_CLASSES; // 12
pub const ALT2_BEAR_FEATURES: usize = V6_LOCAL_CELLS * ALT2_BEAR_PER_CELL; // 1524

pub const ALT2_ELK_SHAPE_CLASSES: usize = 7; // 0=single, 1=pair-end, 2=triangle-vertex, 3=rhombus-vertex, 4=line-of-3, 5=line-of-4+, 6=blob/other
pub const ALT2_ELK_FEATURES: usize = V6_LOCAL_CELLS * ALT2_ELK_SHAPE_CLASSES; // 889

pub const CARDS_ALT_V2_FEATURES: usize =
    ALT2_HAWK_LOS_FEATURES
    + ALT2_SALMON_FEATURES
    + ALT2_FOX_FEATURES
    + ALT2_BEAR_FEATURES
    + ALT2_ELK_FEATURES; // 10287

pub const NUM_FEATURES_MID_V4_ALT_V2: usize = NUM_FEATURES_MID_V4_ALT + CARDS_ALT_V2_FEATURES; // 21710

#[cfg(feature = "cards-alt-v2")]
pub const CARDS_ALT_V2_BASE: usize = NUM_FEATURES_MID_V4_ALT;

/// Base index where the v5-feat block starts. Always appended after the v4-opp block.
pub const V5_FEAT_BASE: usize = if cfg!(feature = "legacy-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_LEGACY + OPP_DETAILED_FEATURES
} else if cfg!(feature = "legacy-features") {
    NUM_FEATURES_LEGACY
} else if cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_MID_V4
} else if cfg!(feature = "mid-features") {
    NUM_FEATURES_MID
} else if cfg!(feature = "v4-opp") {
    NUM_FEATURES_V3_V4
} else {
    NUM_FEATURES_V3
};

/// Base index where the v4-opp block starts in the feature index space.
/// Crucially, opp-detail indices don't care about whether cell-adj etc fire;
/// they just append after the last non-v4 block (mid or v3), so a v4 weights
/// file trained with mid-features has its opp-detail columns at indices
/// NUM_FEATURES_MID..NUM_FEATURES_MID_V4 and plays nicely with mid weights
/// by zero-padding the v4 block on load.
pub const OPP_DETAILED_BASE: usize = if cfg!(feature = "legacy-features") { NUM_FEATURES_LEGACY }
                                     else if cfg!(feature = "mid-features") { NUM_FEATURES_MID }
                                     else { NUM_FEATURES_V3 };

// ── v6-peak: bounded play region + 6-dir adjacency + HabitatBucket ────
//
// Complete feature redesign that addresses three known bottlenecks:
// 1. Play region bound: empirical analysis of 50K v5sh game states showed only
//    37% of the 21×21 grid is ever used. Active region is hex-distance ≤ 6 from
//    grid center = 127 cells (captures 99.9% of placements). Per-cell blocks
//    shrink ~71%.
// 2. 3-vs-6 direction gap: pre-v6 only encoded 3 line directions of pairwise
//    wildlife adjacency. But Cascadia scoring uses 6-direction adjacency for
//    bear pairs, salmon runs, hawk isolation, fox diversity, and habitat union.
//    v6 adds full per-cell 6-direction wildlife AND terrain adjacency at the
//    affordable v6-bounded scale.
// 3. Habitat cluster opacity: pre-v6 forced the network to infer cluster
//    membership from pattern-feature aggregates. v6 adds Stockfish-style
//    cluster-role conditioning per cell (in-largest / in-2nd / in-isolated).
//
// v6-peak feature layout (total ~17,608 — comparable size to v5 but with
// fundamentally better coverage):
//
//   [0..1397)         Per-cell core (127 × 11)
//   [1397..1507)      Phase (110)
//   [1507..1654)      Pairwise wl adj (147, 3 line dirs — kept for elk lines)
//   [1654..1743)      Patterns v1 (89)
//   [1743..1798)      Bag remaining (55)
//   [1798..1853)      Opp habitat (55)
//   [1853..2488)      Allowed wl per cell (127 × 5 = 635)
//   [2488..2538)      WL count ext (50)
//   [2538..2646)      Terrain pairwise (108)
//   [2646..3281)      Sec terrain per cell (127 × 5 = 635)
//   [3281..3351)      Hab ext (70)
//   [3351..3406)      WL count ext2 (55)
//   [3406..3446)      Pair ext capacity (40)
//   [3446..3494)      Pattern v2 (48)
//   [3494..3599)      Bag ext (105)
//   [3599..3669)      Opp hab ext (70)
//   [3669..3757)      Market (88)
//   [3757..3862)      Tbag terrain (105)
//   [3862..3967)      Tbag wildlife (105)
//   [3967..4117)      Tbag terrain ext (150)
//   [4117..4267)      Tbag wildlife ext (150)
//   [4267..4268)      Overflow (1)
//   [4268..4637)      v4-opp block (369)
//   [4637..4764)      v5-feat frontier flag (127, shrunk from 441)
//   [4764..4872)      v5-feat opp pattern detail (108)
//   [4872..5187)      v5-feat bonus distance (315)
//   [5187..5272)      v5-feat hab structure (85)
//   [5272..5797)      v5-feat tbag joint (525)
//   [5797..11131)     ⭐ NEW per-cell 6-dir wildlife adj (127 × 6 × 7 = 5,334)
//   [11131..15703)    ⭐ NEW per-cell 6-dir terrain edge (127 × 6 × 6 = 4,572)
//   [15703..17608)    ⭐ NEW HabitatBucket smaller (127 × 5 × 3 = 1,905)
//
// IMPORTANT: v6-peak is mutually exclusive with v5-feat. Use one or the other.

pub const V6_LOCAL_RADIUS: i32 = 6;
pub const V6_LOCAL_CELLS: usize = 127; // 1 + 6 + 12 + 18 + 24 + 30 + 36
pub const V6_FPC: usize = 11;          // FEATURES_PER_CELL_V6
pub const V6_CELL_FEATURES: usize = V6_LOCAL_CELLS * V6_FPC; // 1397
pub const V6_ALLOWED_WL_FEATURES: usize = V6_LOCAL_CELLS * 5; // 635
pub const V6_SEC_TERRAIN_FEATURES: usize = V6_LOCAL_CELLS * 5; // 635
pub const V6_FRONTIER_FLAG_FEATURES: usize = V6_LOCAL_CELLS;  // 127
// 6-direction per-cell adjacency
pub const V6_ADJ_DIRS: usize = 6;
pub const V6_ADJ_WL_STATES: usize = 7;       // 0=no tile, 1-5=wildlife, 6=tile-no-wildlife
pub const V6_ADJ_TERR_STATES: usize = 6;     // 0=no tile, 1-5=terrain
pub const V6_CELL_ADJ_WL_FEATURES: usize = V6_LOCAL_CELLS * V6_ADJ_DIRS * V6_ADJ_WL_STATES; // 5334
pub const V6_CELL_ADJ_TERR_FEATURES: usize = V6_LOCAL_CELLS * V6_ADJ_DIRS * V6_ADJ_TERR_STATES; // 4572
// HabitatBucket smaller: per-cell (terrain × cluster role)
pub const V6_HAB_BUCKET_ROLES: usize = 3;    // 0=largest, 1=2nd-largest, 2=isolated
pub const V6_HAB_BUCKET_FEATURES: usize = V6_LOCAL_CELLS * 5 * V6_HAB_BUCKET_ROLES; // 1905

pub const NUM_FEATURES_V6_PEAK: usize =
    V6_CELL_FEATURES                  // 1397
    + PHASE_FEATURES                  // 110
    + PAIR_FEATURES                   // 147
    + PATTERN_FEATURES                // 89
    + BAG_FEATURES                    // 55
    + OPP_HAB_FEATURES                // 55
    + V6_ALLOWED_WL_FEATURES          // 635
    + WL_COUNT_EXT_FEATURES           // 50
    + TERRAIN_PAIR_FEATURES           // 108
    + V6_SEC_TERRAIN_FEATURES         // 635
    + HAB_EXT_FEATURES                // 70
    + WL_COUNT_EXT2_FEATURES          // 55
    + EXT_CAP_FEATURES                // 40
    + PATTERN_V2_FEATURES             // 48
    + BAG_EXT_FEATURES                // 105
    + OPP_HAB_EXT_FEATURES            // 70
    + MARKET_FEATURES                 // 88
    + TBAG_TERRAIN_FEATURES           // 105
    + TBAG_WL_FEATURES                // 105
    + TBAG_TERRAIN_EXT_FEATURES       // 150
    + TBAG_WL_EXT_FEATURES            // 150
    + OVERFLOW_FEATURES               // 1
    + OPP_DETAILED_FEATURES           // 369 (v4-opp)
    + V6_FRONTIER_FLAG_FEATURES       // 127
    + V5_OPP_PAT_FEATURES             // 108
    + V5_BONUS_DIST_FEATURES          // 315
    + V5_HAB_STRUCT_FEATURES          // 85
    + V5_TBAG_JOINT_FEATURES          // 525
    + V6_CELL_ADJ_WL_FEATURES         // 5334
    + V6_CELL_ADJ_TERR_FEATURES       // 4572
    + V6_HAB_BUCKET_FEATURES;         // 1905
// = 17608

pub const NUM_FEATURES: usize = if cfg!(feature = "v6-peak") {
    NUM_FEATURES_V6_PEAK
} else if cfg!(feature = "cards-alt-v2") && cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_MID_V4_ALT_V2
} else if cfg!(feature = "cards-alt") && cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_MID_V4_ALT
} else if cfg!(feature = "v5-feat") && cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_MID_V4_V5
} else if cfg!(feature = "v5-feat") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_V3_V4_V5
} else if cfg!(feature = "v5-feat") && cfg!(feature = "mid-features") {
    NUM_FEATURES_MID + V5_FEAT_FEATURES
} else if cfg!(feature = "v5-feat") {
    NUM_FEATURES_V3 + V5_FEAT_FEATURES
} else if cfg!(feature = "legacy-features") && cfg!(feature = "v4-opp") {
    NUM_FEATURES_LEGACY + OPP_DETAILED_FEATURES
} else if cfg!(feature = "legacy-features") { NUM_FEATURES_LEGACY }
  else if cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") { NUM_FEATURES_MID_V4 }
  else if cfg!(feature = "mid-features") { NUM_FEATURES_MID }
  else if cfg!(feature = "v4-opp") { NUM_FEATURES_V3_V4 }
  else { NUM_FEATURES_V3 };

/// Base index where the cards-alt block starts (mutually exclusive with v5/v6).
/// Always appended after the v4-opp block when `cards-alt` is enabled.
#[cfg(feature = "cards-alt")]
pub const CARDS_ALT_BASE: usize = NUM_FEATURES_MID_V4;

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

// ── v5-feat: split-value-heads architecture (P3) ─────────────────────
//
// 11 heads (KataGo-style decomposed value). When `has_split11_heads` is true on
// an NNUENetwork, `forward()` returns the sum of all head outputs.
//
// Head index → subscore:
//   0..5: per-wildlife remaining (bear, elk, salmon, hawk, fox)
//   5..10: per-terrain hab+bonus remaining (forest, prairie, wetland, mountain, river)
//   10: nature tokens remaining
//
// Sum of head outputs = total remaining points (= final_score - current_score
// including habitat majority bonuses, attributed per-terrain). The decomposition
// gives finer credit-assignment signal during training and lets each head
// specialize on a distinct sub-game without interference from other subscores.
pub const NUM_HEADS: usize = 11;
pub const HEAD_BEAR: usize = 0;
pub const HEAD_ELK: usize = 1;
pub const HEAD_SALMON: usize = 2;
pub const HEAD_HAWK: usize = 3;
pub const HEAD_FOX: usize = 4;
pub const HEAD_HAB_FOREST: usize = 5;
pub const HEAD_HAB_PRAIRIE: usize = 6;
pub const HEAD_HAB_WETLAND: usize = 7;
pub const HEAD_HAB_MOUNTAIN: usize = 8;
pub const HEAD_HAB_RIVER: usize = 9;
pub const HEAD_TOKENS: usize = 10;

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
    // ── v5-feat richer pattern detail (P4) ──
    /// Number of bear singletons (component size 1).
    pub bear_singleton_count: u8,
    /// Length of the longest elk line in any of 3 line directions.
    pub longest_elk_line: u8,
    /// Length of the longest valid salmon run (≤2 neighbors per cell).
    pub longest_salmon_run: u8,
    // ── cards-alt opponent threat metrics ──
    // Gated on cards-alt feature so non-alt builds (v4opp, v5, v6) don't pay
    // the per-BagInfo cost of 5 extra BFS-style scans per opponent.
    /// Best bear-component class observed (0=none, 1=single, 2=pair, 3=triple, 4=quad+).
    #[cfg(feature = "cards-alt")]
    pub alt_bear_class: u8,
    /// Densest elk shape achieved (0=none, 1=single, 2=pair, 3=triangle, 4=rhombus+).
    #[cfg(feature = "cards-alt")]
    pub alt_elk_shape: u8,
    /// Best Card-D salmon run "score-equivalent" class (0=no qualifying, 1=3, 2=4, 3=5+, 4=5+with adj animals).
    #[cfg(feature = "cards-alt")]
    pub alt_salmon_class: u8,
    /// Best Card-D hawk LOS-pair intervening-types count (0..4+).
    #[cfg(feature = "cards-alt")]
    pub alt_hawk_pair_types: u8,
    /// Best Card-B fox pair-types count for any single fox (0..4+).
    #[cfg(feature = "cards-alt")]
    pub alt_fox_pair_types: u8,
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
    // ── v5-feat additions ──
    /// Joint count of remaining tile-bag tiles per (terrain, allowed_wildlife) pair.
    /// joint[t][w] = number of bag tiles whose terrain (primary OR secondary) == t
    /// AND whose allowed mask includes wildlife w. Used by the P7 NNUE block.
    pub tbag_joint: [[u8; 5]; 5],
    /// Observing player's largest habitat per terrain (used by P5 bonus-distance
    /// features to compute my_size - opp_size). Mirrors `Board.largest_group`.
    pub my_largest_group: [u8; 5],
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

        // v5-feat: joint terrain × wildlife distribution from tile bag.
        let tbag_joint = game.tile_bag.joint_distribution();
        // v5-feat: observing player's own largest habitat per terrain (for bonus-distance feature).
        let mut my_largest_group = [0u8; 5];
        for t in 0..5 {
            my_largest_group[t] = game.boards[player].largest_group[t] as u8;
        }

        BagInfo {
            remaining,
            max_opponent_habitat,
            market,
            tbag_terrain,
            tbag_wildlife,
            overflow_used: game.overflow_used_this_turn,
            opp_detail,
            tbag_joint,
            my_largest_group,
        }
    }
}

/// Compute a single opponent's detail from its board. Read-only.
/// Populates v4-opp fields (binary pattern flags) AND v5-feat richer pattern
/// counts (bear_singleton_count, longest_elk_line, longest_salmon_run).
/// All fields are computed unconditionally — the cost is dominated by BFS over
/// small wildlife sets, and gating extraction (not population) keeps the data
/// path simple.
fn compute_opponent_detail(board: &Board) -> OpponentDetail {
    let adj = &*ADJACENCY;
    let wildlife_counts: [u8; 5] =
        std::array::from_fn(|i| board.wildlife_positions[i].len() as u8);
    let largest_group: [u8; 5] =
        std::array::from_fn(|i| board.largest_group[i] as u8);

    // Bear singletons: count components of size exactly 1.
    let mut bear_singleton_count: u8 = 0;
    {
        let mut visited = [false; 441];
        for &p in board.wildlife_positions[Wildlife::Bear as usize].iter() {
            let idx = p as usize;
            if visited[idx] { continue; }
            let mut size = 0u16;
            let mut queue: arrayvec::ArrayVec<u16, 24> = arrayvec::ArrayVec::new();
            queue.push(p); visited[idx] = true;
            while let Some(cur) = queue.pop() {
                size += 1;
                for n in adj.neighbors_of(cur as usize) {
                    if !visited[n]
                       && board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear) {
                        visited[n] = true;
                        if !queue.is_full() { queue.push(n as u16); }
                    }
                }
            }
            if size == 1 { bear_singleton_count = bear_singleton_count.saturating_add(1); }
        }
    }
    let has_bear_singleton = bear_singleton_count > 0;

    // Longest elk line: walk all 3 directions for each elk, keep max.
    let mut longest_elk_line: u16 = 0;
    for &p in board.wildlife_positions[Wildlife::Elk as usize].iter() {
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
            if len > longest_elk_line { longest_elk_line = len; }
        }
    }
    let has_elk_line_3plus = longest_elk_line >= 3;

    // Salmon: longest valid run (component where each cell has ≤2 salmon neighbors).
    // BFS from each salmon, dedupe via visited, track max valid run length.
    let mut longest_salmon_run: u16 = 0;
    {
        let mut visited = [false; 441];
        for &p in board.wildlife_positions[Wildlife::Salmon as usize].iter() {
            let start = p as usize;
            if visited[start] { continue; }
            let mut component: arrayvec::ArrayVec<u16, 32> = arrayvec::ArrayVec::new();
            let mut stack: arrayvec::ArrayVec<u16, 32> = arrayvec::ArrayVec::new();
            stack.push(p); visited[start] = true;
            while let Some(cur) = stack.pop() {
                component.push(cur);
                for n in adj.neighbors_of(cur as usize) {
                    if !visited[n]
                        && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                        visited[n] = true;
                        if !stack.is_full() { stack.push(n as u16); }
                    }
                }
            }
            // Validate: each cell in component has ≤2 salmon neighbors.
            let valid = component.iter().all(|&c| {
                adj.neighbors_of(c as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count() <= 2
            });
            if valid && (component.len() as u16) > longest_salmon_run {
                longest_salmon_run = component.len() as u16;
            }
        }
    }
    let has_salmon_run_4plus = longest_salmon_run >= 4;

    // Isolated hawk count.
    let mut isolated_hawks = 0u8;
    for &p in board.wildlife_positions[Wildlife::Hawk as usize].iter() {
        let pos = p as usize;
        let has_hawk_neigh = adj.neighbors_of(pos)
            .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
        if !has_hawk_neigh { isolated_hawks += 1; }
    }

    // ── cards-alt opponent threat metrics ──
    // Computed only when cards-alt is enabled. TODO (perf): for hot benches
    // these can share scan work with the existing bear/elk/salmon BFS above
    // (currently independent passes); also worth caching opp_detail across
    // BagInfo::from_game calls inside one outer move (~3750 redundant calls
    // per move under MCE-750 rollouts because rollout-level BagInfo never
    // changes for the opp side until an opponent simulated turn fires).
    #[cfg(feature = "cards-alt")]
    let alt_bear_class = compute_alt_bear_class(board);
    #[cfg(feature = "cards-alt")]
    let alt_elk_shape = compute_alt_elk_shape(board);
    #[cfg(feature = "cards-alt")]
    let alt_salmon_class = compute_alt_salmon_class(board);
    #[cfg(feature = "cards-alt")]
    let alt_hawk_pair_types = compute_alt_hawk_pair_types(board);
    #[cfg(feature = "cards-alt")]
    let alt_fox_pair_types = compute_alt_fox_pair_types(board);

    OpponentDetail {
        wildlife_counts,
        largest_group,
        nature_tokens: board.nature_tokens,
        has_bear_singleton,
        has_elk_line_3plus,
        has_salmon_run_4plus,
        isolated_hawk_count: isolated_hawks,
        bear_singleton_count,
        longest_elk_line: longest_elk_line.min(255) as u8,
        longest_salmon_run: longest_salmon_run.min(255) as u8,
        #[cfg(feature = "cards-alt")]
        alt_bear_class,
        #[cfg(feature = "cards-alt")]
        alt_elk_shape,
        #[cfg(feature = "cards-alt")]
        alt_salmon_class,
        #[cfg(feature = "cards-alt")]
        alt_hawk_pair_types,
        #[cfg(feature = "cards-alt")]
        alt_fox_pair_types,
    }
}

// ─────────────────────────────────────────────────────────────────────
// cards-alt feature computation helpers (always compiled, gated on emit)
// ─────────────────────────────────────────────────────────────────────

/// Per-component sizes for a wildlife type (BFS, hex adjacency).
#[cfg(feature = "cards-alt")]
fn alt_components(board: &Board, w: Wildlife) -> arrayvec::ArrayVec<u16, 32> {
    let positions = &board.wildlife_positions[w as usize];
    let adj = &*ADJACENCY;
    let mut out = arrayvec::ArrayVec::<u16, 32>::new();
    let mut visited = [false; 441];
    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut size = 0u16;
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            size += 1;
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(w) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let _ = out.try_push(size);
    }
    out
}

/// Best bear size class observed: 0=none, 1=single, 2=pair, 3=triple, 4=quad+.
#[cfg(feature = "cards-alt")]
fn compute_alt_bear_class(board: &Board) -> u8 {
    let sizes = alt_components(board, Wildlife::Bear);
    let max = sizes.iter().copied().max().unwrap_or(0);
    match max {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 4,
    }
}

/// Densest elk shape: 0=none, 1=single, 2=pair, 3=triangle (3 mutually adjacent),
/// 4=rhombus (triangle + ≥1 adjacent extra).
#[cfg(feature = "cards-alt")]
fn compute_alt_elk_shape(board: &Board) -> u8 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() { return 0; }
    let adj = &*ADJACENCY;
    // Count mutually-adjacent triads. If any 4-component has a triangle subshape, shape=4.
    // Cheap check: for each elk, neighbors & elk_set; pair of those mutually adjacent → triangle.
    let mut elk = [false; 441];
    for &p in positions.iter() { elk[p as usize] = true; }
    let mut max_shape: u8 = 1; // any elk = singleton
    // Check pair (≥2 adjacent elk anywhere).
    let mut has_pair = false;
    for &p in positions.iter() {
        for n in adj.neighbors_of(p as usize) {
            if elk[n] { has_pair = true; break; }
        }
        if has_pair { break; }
    }
    if has_pair && max_shape < 2 { max_shape = 2; }
    // Check triangle: any 3 mutually-adjacent.
    let mut triangle_anchors: arrayvec::ArrayVec<u16, 32> = arrayvec::ArrayVec::new();
    'outer: for &p in positions.iter() {
        let nbrs: arrayvec::ArrayVec<u16, 6> = adj.neighbors_of(p as usize)
            .filter(|&n| elk[n])
            .map(|n| n as u16)
            .collect();
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                let a = nbrs[i] as usize;
                let b = nbrs[j] as usize;
                if adj.neighbors_of(a).any(|n| n == b) {
                    if max_shape < 3 { max_shape = 3; }
                    let _ = triangle_anchors.try_push(p);
                    continue 'outer;
                }
            }
        }
    }
    // Check rhombus: any cell in a triangle that has a 4th elk-neighbor outside the triad.
    if max_shape >= 3 {
        for &t in triangle_anchors.iter() {
            let elk_n: usize = adj.neighbors_of(t as usize).filter(|&n| elk[n]).count();
            // If center has ≥3 elk neighbors, rhombus exists.
            if elk_n >= 3 { max_shape = 4; break; }
        }
        // Or any vertex of the triangle has an elk neighbor outside the triad.
        if max_shape < 4 {
            for &t in triangle_anchors.iter() {
                let mut tri_neigh = arrayvec::ArrayVec::<u16, 6>::new();
                for n in adj.neighbors_of(t as usize) {
                    if elk[n] { let _ = tri_neigh.try_push(n as u16); }
                }
                // For each pair forming the triangle with t, check if any other elk
                // is adjacent to either of those vertices.
                for i in 0..tri_neigh.len() {
                    for j in (i + 1)..tri_neigh.len() {
                        let a = tri_neigh[i] as usize;
                        let b = tri_neigh[j] as usize;
                        if !adj.neighbors_of(a).any(|n| n == b) { continue; }
                        // a and b are part of a triangle with t. Check 4th elk adjacent to a or b.
                        for v in [a, b] {
                            for n in adj.neighbors_of(v) {
                                if elk[n] && n != t as usize && n != a && n != b {
                                    max_shape = 4;
                                    break;
                                }
                            }
                            if max_shape == 4 { break; }
                        }
                        if max_shape == 4 { break; }
                    }
                    if max_shape == 4 { break; }
                }
                if max_shape == 4 { break; }
            }
        }
    }
    max_shape
}

/// Best Card-D salmon "score class": 0=no qualifying run, 1=qualifying min,
/// 2=length 4, 3=length 5+, 4=length 5+ with ≥3 adjacent non-salmon types.
#[cfg(feature = "cards-alt")]
fn compute_alt_salmon_class(board: &Board) -> u8 {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() { return 0; }
    let adj = &*ADJACENCY;
    let mut visited = [false; 441];
    let mut best: u8 = 0;
    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let valid = comp.iter().all(|&c| {
            adj.neighbors_of(c as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count() <= 2
        });
        if !valid { continue; }
        let len = comp.len();
        if len < 3 { continue; }
        let mut seen = [false; 441];
        let mut adj_types = 0u32;
        for &c in &comp {
            for n in adj.neighbors_of(c as usize) {
                if seen[n] { continue; }
                seen[n] = true;
                if let Some(w) = board.grid.get(n).placed_wildlife() {
                    if w != Wildlife::Salmon { adj_types += 1; }
                }
            }
        }
        let class: u8 = if len == 3 { 1 }
                        else if len == 4 { 2 }
                        else if adj_types >= 3 { 4 }
                        else { 3 };
        if class > best { best = class; }
    }
    best
}

/// Best hawk LOS-pair intervening unique non-hawk types observed (0..4+).
#[cfg(feature = "cards-alt")]
fn compute_alt_hawk_pair_types(board: &Board) -> u8 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.len() < 2 { return 0; }
    let mut hawk_set = [false; 441];
    for &p in positions.iter() { hawk_set[p as usize] = true; }
    let mut best: u32 = 0;
    for &p in positions.iter() {
        let coord = cascadia_core::hex::HexCoord::from_index(p as usize);
        for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
            let mut cur = cascadia_core::hex::HexCoord::new(coord.q + dq, coord.r + dr);
            let mut steps = 1u32;
            let mut types_mask = 0u8;
            loop {
                match cur.to_index() {
                    Some(idx) => {
                        if hawk_set[idx] {
                            if steps >= 2 {
                                let unique = (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones();
                                if unique > best { best = unique; }
                            }
                            break;
                        }
                        if let Some(w) = board.grid.get(idx).placed_wildlife() {
                            types_mask |= 1 << (w as u8);
                        }
                    }
                    None => break,
                }
                cur = cascadia_core::hex::HexCoord::new(cur.q + dq, cur.r + dr);
                steps += 1;
            }
        }
    }
    best.min(4) as u8
}

/// Best per-fox pair-type count (0..4+) — count of non-fox types with ≥2 in adjacent cells.
#[cfg(feature = "cards-alt")]
fn compute_alt_fox_pair_types(board: &Board) -> u8 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() { return 0; }
    let adj = &*ADJACENCY;
    let mut best: u32 = 0;
    for &p in positions.iter() {
        let mut counts = [0u8; 5];
        for n in adj.neighbors_of(p as usize) {
            if let Some(w) = board.grid.get(n).placed_wildlife() {
                if w != Wildlife::Fox { counts[w as usize] += 1; }
            }
        }
        let pair_types = counts.iter().filter(|&&c| c >= 2).count() as u32;
        if pair_types > best { best = pair_types; }
    }
    best.min(4) as u8
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
    // v6-peak uses a completely different feature layout (bounded play region +
    // 6-dir adjacency + HabitatBucket). Dispatch to the dedicated v6 extractor.
    #[cfg(feature = "v6-peak")]
    {
        return extract_features_v6_peak(board, bag);
    }

    #[allow(unreachable_code)]
    {
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

    // ─────────────────────────────────────────────────────────────────
    // cards-alt block (appended at CARDS_ALT_BASE = NUM_FEATURES_MID_V4)
    // Player-side alt-card pattern features + per-opp alt threat metrics.
    // ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "cards-alt")]
    {
        let mut alt_off = CARDS_ALT_BASE;
        extract_alt_bear_features(board, &mut features, alt_off);
        alt_off += ALT_BEAR_FEATURES;
        extract_alt_elk_features(board, &mut features, alt_off);
        alt_off += ALT_ELK_FEATURES;
        extract_alt_salmon_features(board, &mut features, alt_off);
        alt_off += ALT_SALMON_FEATURES;
        extract_alt_hawk_features(board, &mut features, alt_off);
        alt_off += ALT_HAWK_FEATURES;
        extract_alt_fox_features(board, &mut features, alt_off);
        alt_off += ALT_FOX_FEATURES;
        if let Some(b) = bag {
            extract_alt_opp_features(b, &mut features, alt_off);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // cards-alt-v2 block (appended at CARDS_ALT_V2_BASE = NUM_FEATURES_MID_V4_ALT)
    // Per-piece relational features: each placed wildlife emits a single feature
    // index encoding its alt-card-relevant context. Sparse, position-aware.
    // ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "cards-alt-v2")]
    {
        let mut v2_off = CARDS_ALT_V2_BASE;
        extract_alt2_hawk_los_features(board, &mut features, v2_off);
        v2_off += ALT2_HAWK_LOS_FEATURES;
        extract_alt2_salmon_features(board, &mut features, v2_off);
        v2_off += ALT2_SALMON_FEATURES;
        extract_alt2_fox_features(board, &mut features, v2_off);
        v2_off += ALT2_FOX_FEATURES;
        extract_alt2_bear_features(board, &mut features, v2_off);
        v2_off += ALT2_BEAR_FEATURES;
        extract_alt2_elk_features(board, &mut features, v2_off);
    }

    // ─────────────────────────────────────────────────────────────────
    // v5-feat block (appended at V5_FEAT_BASE)
    // Frontier (P2), richer per-opp pattern (P4), bonus distance (P5),
    // habitat structure (P6), tile-bag joint distribution (P7).
    // ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "v5-feat")]
    {
        extract_v5_frontier_features(board, &mut features, V5_FEAT_BASE);
        if let Some(b) = bag {
            let opp_pat_base = V5_FEAT_BASE
                + V5_FRONTIER_FLAG_FEATURES
                + V5_FRONT_WL_FEATURES
                + V5_FRONT_TERR_FEATURES;
            extract_v5_opp_pattern_features(b, &mut features, opp_pat_base);
            let bonus_base = opp_pat_base + V5_OPP_PAT_FEATURES;
            extract_v5_bonus_distance_features(b, &mut features, bonus_base);
            let tbag_base = bonus_base + V5_BONUS_DIST_FEATURES + V5_HAB_STRUCT_FEATURES;
            extract_v5_tbag_joint_features(b, &mut features, tbag_base);
        }
        let hab_struct_base = V5_FEAT_BASE
            + V5_FRONTIER_FLAG_FEATURES + V5_FRONT_WL_FEATURES + V5_FRONT_TERR_FEATURES
            + V5_OPP_PAT_FEATURES + V5_BONUS_DIST_FEATURES;
        extract_v5_hab_structure_features(board, &mut features, hab_struct_base);
    }

    features
    } // end allow(unreachable_code) block (v6-peak returns earlier)
}

// ─────────────────────────────────────────────────────────────────────
// v5-feat extraction helpers
// ─────────────────────────────────────────────────────────────────────

/// P2 frontier features:
/// - 441 cells: 1 bit per cell "is on frontier" (empty cell with at least one placed neighbor)
/// - 441 × 5 cells: 1 bit per (cell, wildlife) "this frontier cell has wildlife W in any neighbor"
/// - 441 × 5 cells: 1 bit per (cell, terrain) "this frontier cell touches terrain T on a shared edge"
/// Sparsity: ~10-20 frontier cells × ~5 fires each = ~50-200 active features.
#[cfg(any(feature = "v5-feat", feature = "v6-peak"))]
fn extract_v5_frontier_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let adj = &*ADJACENCY;
    let frontier = board.frontier();
    let flag_base = base;
    let wl_base = base + V5_FRONTIER_FLAG_FEATURES;
    let terr_base = wl_base + V5_FRONT_WL_FEATURES;

    for &fpos in frontier.iter() {
        let idx = fpos as usize;
        // Frontier flag.
        features.push((flag_base + idx) as u16);

        // For each direction, look at the neighbor (which must be a placed tile
        // for this cell to be on the frontier). Encode the wildlife code +
        // the terrain that faces this frontier cell.
        let mut wl_seen = [false; 5];
        let mut terr_seen = [false; 5];
        for dir in 0..6 {
            let nidx_val = adj.neighbors[idx][dir];
            if nidx_val == u16::MAX { continue; }
            let nidx = nidx_val as usize;
            let n_cell = board.grid.get(nidx);
            if !n_cell.is_present() { continue; }
            // Wildlife on neighbor (only if placed).
            if let Some(w) = n_cell.placed_wildlife() {
                let wi = w as usize;
                if wi < 5 && !wl_seen[wi] {
                    wl_seen[wi] = true;
                    features.push((wl_base + idx * 5 + wi) as u16);
                }
            }
            // Terrain on the neighbor's edge that faces this cell (shared edge).
            // The shared edge from neighbor's POV is direction (dir + 3) % 6.
            let nrot = board.rotations[nidx];
            if let Some(t) = cascadia_core::board::terrain_on_edge(n_cell, nrot, (dir + 3) % 6) {
                let ti = t as usize;
                if ti < 5 && !terr_seen[ti] {
                    terr_seen[ti] = true;
                    features.push((terr_base + idx * 5 + ti) as u16);
                }
            }
        }
    }
}

/// P4 richer per-opp pattern histograms (replaces v4-opp's 4 binary flags with
/// 4 small histograms: bear singletons, longest elk line, longest salmon run,
/// isolated hawk count). Per opp: 36 features. 3 opps × 36 = 108 total.
#[cfg(any(feature = "v5-feat", feature = "v6-peak"))]
fn extract_v5_opp_pattern_features(bag: &BagInfo, features: &mut Vec<u16>, base: usize) {
    for (slot, opp) in bag.opp_detail.iter().enumerate() {
        let slot_base = base + slot * V5_OPP_PAT_PER_OPP;
        let bear_base = slot_base;
        let elk_base = bear_base + V5_OPP_BEAR_SING_BINS;
        let salmon_base = elk_base + V5_OPP_ELK_LINE_BINS;
        let hawk_base = salmon_base + V5_OPP_SALMON_RUN_BINS;

        let bb = (opp.bear_singleton_count as usize).min(V5_OPP_BEAR_SING_BINS - 1);
        features.push((bear_base + bb) as u16);
        let eb = (opp.longest_elk_line as usize).min(V5_OPP_ELK_LINE_BINS - 1);
        features.push((elk_base + eb) as u16);
        let sb = (opp.longest_salmon_run as usize).min(V5_OPP_SALMON_RUN_BINS - 1);
        features.push((salmon_base + sb) as u16);
        let hb = (opp.isolated_hawk_count as usize).min(V5_OPP_ISOL_HAWK_BINS - 1);
        features.push((hawk_base + hb) as u16);
    }
}

/// P5 bonus-threshold-distance features: for each (terrain, opponent), the
/// signed difference (my_largest_group[t] - opp.largest_group[t]) binned to
/// 21 bins (-10..=-1, 0, 1..=10+). 5 terrains × 3 opps × 21 = 315 features.
/// Sparsity: 15 active features per position (one bin per terrain×opp pair).
#[cfg(any(feature = "v5-feat", feature = "v6-peak"))]
fn extract_v5_bonus_distance_features(bag: &BagInfo, features: &mut Vec<u16>, base: usize) {
    for t in 0..5 {
        for (slot, opp) in bag.opp_detail.iter().enumerate() {
            let mine = bag.my_largest_group[t] as i32;
            let theirs = opp.largest_group[t] as i32;
            let diff = (mine - theirs).clamp(-10, 10);
            // bin index: -10 → 0, ... 0 → 10, ... 10 → 20
            let bin = (diff + 10) as usize;
            let block_base = base + (t * NUM_OPP_SLOTS + slot) * V5_BONUS_DIFF_BINS;
            features.push((block_base + bin) as u16);
        }
    }
}

/// P6 habitat structure: per terrain, count of distinct connected groups +
/// 2nd-largest group size. 5 × (6 + 11) = 85 features. Sparsity: 10 active.
/// Implementation: BFS over all cells of each terrain to enumerate components.
#[cfg(any(feature = "v5-feat", feature = "v6-peak"))]
fn extract_v5_hab_structure_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let adj = &*ADJACENCY;
    for t in 0..5 {
        let terrain_t = match cascadia_core::types::Terrain::from_u8(t as u8) {
            Some(tt) => tt,
            None => continue,
        };
        // Mark cells that have a tile facing terrain_t on at least one edge.
        // This is the same predicate the union-find uses; we re-derive via cells.
        let mut visited = [false; 441];
        let mut sizes: arrayvec::ArrayVec<u16, 64> = arrayvec::ArrayVec::new();
        for &tile_idx in board.placed_tiles.iter() {
            let idx = tile_idx as usize;
            if visited[idx] { continue; }
            let cell = board.grid.get(idx);
            if !cell.is_present() { continue; }
            // Cell qualifies if it has terrain_t on some edge.
            let has_t = (0..6).any(|d| {
                cascadia_core::board::terrain_on_edge(cell, board.rotations[idx], d)
                    == Some(terrain_t)
            });
            if !has_t { continue; }
            // BFS the component connected via shared-terrain edges.
            let mut queue: arrayvec::ArrayVec<u16, 128> = arrayvec::ArrayVec::new();
            queue.push(tile_idx);
            visited[idx] = true;
            let mut size = 0u16;
            while let Some(cur) = queue.pop() {
                size += 1;
                let cur_idx = cur as usize;
                let cur_cell = board.grid.get(cur_idx);
                let cur_rot = board.rotations[cur_idx];
                for d in 0..6 {
                    let n_val = adj.neighbors[cur_idx][d];
                    if n_val == u16::MAX { continue; }
                    let n_idx = n_val as usize;
                    if visited[n_idx] { continue; }
                    let n_cell = board.grid.get(n_idx);
                    if !n_cell.is_present() { continue; }
                    // Both cells must show terrain_t on the shared edge.
                    let my_terr = cascadia_core::board::terrain_on_edge(cur_cell, cur_rot, d);
                    let n_terr = cascadia_core::board::terrain_on_edge(
                        n_cell, board.rotations[n_idx], (d + 3) % 6);
                    if my_terr == Some(terrain_t) && n_terr == Some(terrain_t) {
                        visited[n_idx] = true;
                        if !queue.is_full() { queue.push(n_val); }
                    }
                }
            }
            if size > 0 && !sizes.is_full() { sizes.push(size); }
        }
        // Sort descending: largest first, second-largest at index 1.
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let cluster_count = sizes.len().min(V5_HAB_CLUSTER_COUNT_BINS - 1);
        let second_size = if sizes.len() >= 2 { sizes[1] as usize } else { 0 };
        let second_bin = second_size.min(V5_HAB_SECOND_BINS - 1);
        let count_base = base + t * V5_HAB_STRUCT_PER_TERR;
        features.push((count_base + cluster_count) as u16);
        let second_base = count_base + V5_HAB_CLUSTER_COUNT_BINS;
        features.push((second_base + second_bin) as u16);
    }
}

/// P7 tile-bag joint terrain × wildlife distribution. 5×5 cells × 21 bins each.
/// Per (t, w) cell: how many remaining bag tiles have terrain T (primary or
/// secondary) AND allow wildlife W. Always exactly 25 features active.
#[cfg(any(feature = "v5-feat", feature = "v6-peak"))]
fn extract_v5_tbag_joint_features(bag: &BagInfo, features: &mut Vec<u16>, base: usize) {
    for t in 0..5 {
        for w in 0..5 {
            let count = (bag.tbag_joint[t][w] as usize).min(V5_TBAG_JOINT_BINS - 1);
            let cell_base = base + (t * 5 + w) * V5_TBAG_JOINT_BINS;
            features.push((cell_base + count) as u16);
        }
    }
}

/// Emit per-opponent detail features. `base` is the absolute starting index
/// in the feature index space (one of NUM_FEATURES_LEGACY, NUM_FEATURES_MID,
/// or NUM_FEATURES_V3 depending on cargo-feature combination).
#[cfg(any(feature = "v4-opp", feature = "v5-feat", feature = "v6-peak"))]
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
    // GATED OFF under cards-alt builds: under Bear C/D/B, components of size 3 (or 4)
    // are HIGH-VALUE, not waste. The Card-A "waste" signal would actively mislead.
    // Feature SLOT remains in the index space (so older weights load) — we just
    // don't emit, leaving the network's existing weight on the dead bin to do nothing.
    let pat2_off3 = pat2_off2 + PAT_BEAR_EXT_SINGLES;
    #[cfg(not(feature = "cards-alt"))]
    {
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
    }

    // ── At-risk isolated hawks (4 bins) ──
    // GATED OFF under cards-alt builds: under Hawk B/C/D, "at risk of losing
    // isolation" is the OPPOSITE incentive — losing isolation creates LOS pairs
    // which is what those cards reward.
    let pat2_off4 = pat2_off3 + PAT_BEAR_WASTE;
    #[cfg(not(feature = "cards-alt"))]
    {
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
    }

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
    // GATED OFF under cards-alt builds: Card-A diversity is the wrong target for
    // Fox B (pair-types), Fox C (single-type max), Fox D (pair-pair-types). The
    // network has dedicated alt-aware fox features (ALT2_FOX_CTX) for those.
    let pat2_off6 = pat2_off5 + PAT_FORCED_ALLOC;
    #[cfg(not(feature = "cards-alt"))]
    {
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
    }

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
    /// 11-head split value architecture (v5-feat / P3). Layout:
    ///   w3_heads: row-major [NUM_HEADS × HIDDEN2]; head h's weights live at
    ///             w3_heads[h*HIDDEN2 .. (h+1)*HIDDEN2].
    ///   b3_heads: [NUM_HEADS]; per-head bias.
    /// When `has_split11_heads` is true, forward() returns sum of head outputs.
    /// When false, falls back to has_split_value_heads (2-head) or legacy w3/b3.
    pub has_split11_heads: bool,
    pub w3_heads: Vec<f32>,     // [NUM_HEADS * HIDDEN2]
    pub b3_heads: Vec<f32>,     // [NUM_HEADS]
    /// Heteroscedastic-NLL head (v7 / Exp #3). Predicts log-variance σ² of the
    /// rollout-distribution-conditional-on-position. Used by the loss
    /// 0.5·(y−μ)²/σ² + 0.5·log σ² (Kendall & Gal 2017) so that the network
    /// down-weights gradients on intrinsically noisy positions instead of
    /// blowing capacity trying to fit unfittable variance. The trained σ²
    /// is also exposed via `evaluate_logvar` for use by the search-time
    /// variance-adaptive halving allocator (SeqHalvingHetero).
    /// When `has_heteroscedastic` is false, w3_var/b3_var are zero (loaded
    /// weights without the head produce log_var ≈ 0 → σ ≈ 1, which is harmless
    /// for inference but defeats the loss benefit at training time).
    pub has_heteroscedastic: bool,
    pub w3_var: Vec<f32>,       // [HIDDEN2]
    pub b3_var: f32,
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

        // 11-head split (v5) — initialized to zero; only meaningful when
        // has_split11_heads is true (set by training pipeline / load).
        let w3_heads = vec![0.0; NUM_HEADS * HIDDEN2];
        let b3_heads = vec![0.0; NUM_HEADS];

        NNUENetwork {
            w1, b1, w2, b2, w3, b3,
            w3_policy, b3_policy,
            has_split_value_heads: false,
            w3_wildlife, b3_wildlife, w3_habitat, b3_habitat,
            has_split11_heads: false,
            w3_heads, b3_heads,
            has_heteroscedastic: false,
            w3_var: vec![0.0; HIDDEN2],
            b3_var: 0.0,
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

        // 11-head split (v5)
        let all_11 = others.iter().all(|o| o.has_split11_heads);
        self.has_split11_heads = all_11;
        if all_11 {
            for idx in 0..NUM_HEADS * HIDDEN2 {
                let s: f32 = others.iter().map(|o| o.w3_heads[idx]).sum();
                self.w3_heads[idx] = s / n;
            }
            for h in 0..NUM_HEADS {
                let s: f32 = others.iter().map(|o| o.b3_heads[h]).sum();
                self.b3_heads[h] = s / n;
            }
        }

        // Heteroscedastic head (v7)
        let all_het = others.iter().all(|o| o.has_heteroscedastic);
        self.has_heteroscedastic = all_het;
        if all_het {
            for j in 0..HIDDEN2 {
                let s: f32 = others.iter().map(|o| o.w3_var[j]).sum();
                self.w3_var[j] = s / n;
            }
            self.b3_var = others.iter().map(|o| o.b3_var).sum::<f32>() / n;
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

        // Output (value head). Priority: 11-head split (v5) > 2-head split > legacy.
        if self.has_split11_heads {
            let mut sum = 0.0f32;
            for h in 0..NUM_HEADS {
                let mut head = self.b3_heads[h];
                let row = &self.w3_heads[h * HIDDEN2..(h + 1) * HIDDEN2];
                for j in 0..HIDDEN2 {
                    head += h2[j] * row[j];
                }
                sum += head;
            }
            return sum;
        }
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

    /// Forward pass returning per-head outputs (only meaningful if has_split11_heads).
    /// Used for diagnostics and split-head training validation.
    pub fn forward_heads(&self, features: &[u16]) -> [f32; NUM_HEADS] {
        let mut h1 = [0.0f32; HIDDEN1];
        h1.copy_from_slice(&self.b1);
        for &fi in features {
            let base = fi as usize * HIDDEN1;
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
        let mut heads = [0.0f32; NUM_HEADS];
        for h in 0..NUM_HEADS {
            heads[h] = self.b3_heads[h];
            let row = &self.w3_heads[h * HIDDEN2..(h + 1) * HIDDEN2];
            for j in 0..HIDDEN2 { heads[h] += h2[j] * row[j]; }
        }
        heads
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

        // Value head: 11-head split (v5) > 2-head split > legacy total head.
        let value = if self.has_split11_heads {
            let mut sum = 0.0f32;
            for h in 0..NUM_HEADS {
                let mut head = self.b3_heads[h];
                let row = &self.w3_heads[h * HIDDEN2..(h + 1) * HIDDEN2];
                for j in 0..HIDDEN2 { head += h2[j] * row[j]; }
                sum += head;
            }
            sum
        } else if self.has_split_value_heads {
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
        // Output: 11-head split (v5) > 2-head split > legacy total head.
        if self.has_split11_heads {
            let mut sum = 0.0f32;
            for h in 0..NUM_HEADS {
                let mut head = self.b3_heads[h];
                let row = &self.w3_heads[h * HIDDEN2..(h + 1) * HIDDEN2];
                for j in 0..HIDDEN2 { head += h2[j] * row[j]; }
                sum += head;
            }
            return sum;
        }
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
        // version=3: adds 11-head split block appended at very end
        //            (NUM_HEADS × HIDDEN2 weights + NUM_HEADS biases)
        // version=4: adds heteroscedastic w3_var + b3_var at the very end
        //            (HIDDEN2 + 1 floats). Implies v3 11-head block precedes
        //            (we always write split + 11-head sections when v3+; if
        //            those heads aren't trained, zeros are written).
        let version: u32 = if self.has_heteroscedastic { 4 }
                           else if self.has_split11_heads { 3 }
                           else if self.has_split_value_heads { 2 }
                           else { 1 };
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

        // Split value heads (v2 and v3+): w3_wildlife + b3_wildlife + w3_habitat + b3_habitat
        // When v4 (heteroscedastic) is enabled, also write split + 11-head blocks (zeros if
        // not trained) so layout is unambiguous.
        if self.has_split_value_heads || self.has_split11_heads || self.has_heteroscedastic {
            for &v in &self.w3_wildlife {
                file.write_all(&v.to_le_bytes())?;
            }
            file.write_all(&self.b3_wildlife.to_le_bytes())?;
            for &v in &self.w3_habitat {
                file.write_all(&v.to_le_bytes())?;
            }
            file.write_all(&self.b3_habitat.to_le_bytes())?;
        }

        // 11-head split (v3 and v4): NUM_HEADS × HIDDEN2 weights + NUM_HEADS biases
        if self.has_split11_heads || self.has_heteroscedastic {
            for &v in &self.w3_heads {
                file.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.b3_heads {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        // Heteroscedastic head (v4 only): w3_var [HIDDEN2] + b3_var
        if self.has_heteroscedastic {
            for &v in &self.w3_var {
                file.write_all(&v.to_le_bytes())?;
            }
            file.write_all(&self.b3_var.to_le_bytes())?;
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

        // Layout (from start): header(8) + w1 + b1 + w2 + b2 + w3 + b3 + w3_policy + b3_policy + [split heads (v2+)] + [11-head block (v3)]
        // We don't know w1_features upfront (legacy files had smaller NUM_FEATURES). Compute it by
        // subtracting everything else from the file size.
        let file_size = file.metadata()?.len();
        let header_size = 8u64;
        let fixed_after_w1 = ((HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 + HIDDEN2 + 1) as u64) * 4;
        let policy_head_size = (HIDDEN2 + 1) as u64 * 4;
        let split_head_size = (2 * (HIDDEN2 + 1)) as u64 * 4; // w3_wildlife + b + w3_habitat + b
        let split11_head_size = ((NUM_HEADS * HIDDEN2 + NUM_HEADS) as u64) * 4;
        let het_head_size = ((HIDDEN2 + 1) as u64) * 4; // w3_var + b3_var
        // The v1 file has no split heads appended, v2 has 2-head, v3 has both 2-head + 11-head,
        // v4 adds heteroscedastic head at the very end.
        let trailing_size = policy_head_size
            + if version >= 2 { split_head_size } else { 0 }
            + if version >= 3 { split11_head_size } else { 0 }
            + if version >= 4 { het_head_size } else { 0 };
        // Compute w1 features from remaining bytes. If the file is v1 without a policy head
        // (truly legacy), fall back by assuming zero trailing. That's best-effort.
        let mut w1_bytes = file_size.saturating_sub(header_size + fixed_after_w1 + trailing_size);
        let mut trailing_has_policy = true;
        let mut trailing_has_split = version >= 2;
        let mut trailing_has_11 = version >= 3;
        let trailing_has_het = version >= 4;
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
                trailing_has_11 = false;
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

        // Split value heads (v2 and v3 both write this block)
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

        // 11-head split (v3 and v4): NUM_HEADS × HIDDEN2 weights + NUM_HEADS biases
        let (has_split11, w3_heads, b3_heads) = if trailing_has_11 {
            let mut wh = Vec::with_capacity(NUM_HEADS * HIDDEN2);
            for _ in 0..NUM_HEADS * HIDDEN2 {
                wh.push(read_f32(&mut file)?);
            }
            let mut bh = Vec::with_capacity(NUM_HEADS);
            for _ in 0..NUM_HEADS {
                bh.push(read_f32(&mut file)?);
            }
            (true, wh, bh)
        } else {
            (false, vec![0.0; NUM_HEADS * HIDDEN2], vec![0.0; NUM_HEADS])
        };

        // Heteroscedastic head (v4): w3_var [HIDDEN2] + b3_var
        let (has_het, w3_var, b3_var) = if trailing_has_het {
            let mut wv = Vec::with_capacity(HIDDEN2);
            for _ in 0..HIDDEN2 {
                wv.push(read_f32(&mut file)?);
            }
            let bv = read_f32(&mut file)?;
            (true, wv, bv)
        } else {
            (false, vec![0.0; HIDDEN2], 0.0)
        };

        Ok(NNUENetwork {
            w1, b1, w2, b2, w3, b3,
            w3_policy, b3_policy,
            has_split_value_heads: has_split,
            w3_wildlife, b3_wildlife, w3_habitat, b3_habitat,
            has_split11_heads: has_split11,
            w3_heads, b3_heads,
            has_heteroscedastic: has_het,
            w3_var, b3_var,
        })
    }

    /// Forward pass returning (mean_pred, log_var_pred). Only meaningful when
    /// `has_heteroscedastic` is true; otherwise log_var = 0 (σ² = 1).
    /// Used by the heteroscedastic NLL loss + by the variance-adaptive halving
    /// allocator at search time.
    pub fn forward_with_logvar(&self, features: &[u16]) -> (f32, f32) {
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
        let mut mean = self.b3;
        for j in 0..HIDDEN2 { mean += h2[j] * self.w3[j]; }
        let mut log_var = self.b3_var;
        for j in 0..HIDDEN2 { log_var += h2[j] * self.w3_var[j]; }
        // Clamp log_var to prevent numerical issues (σ in [exp(-3), exp(6)] = [0.05, 403])
        let log_var = log_var.clamp(-3.0, 6.0);
        (mean, log_var)
    }

    /// Heteroscedastic NLL training step (Kendall & Gal 2017).
    /// Loss = 0.5·(target − μ)²/exp(log_var) + 0.5·log_var.
    /// Updates w1, b1, w2, b2, w3, b3 (mean head), AND w3_var, b3_var (var head).
    /// Returns the squared error (target − μ)² for tracking RMSE comparable to
    /// the legacy MSE training loop.
    pub fn train_sample_heteroscedastic(&mut self, features: &[u16], target: f32, lr: f32) -> f32 {
        // ─── Forward (with intermediates) ───
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
        let mut h2_pre = [0.0f32; HIDDEN2];
        h2_pre.copy_from_slice(&h2);
        for v in h2.iter_mut() { *v = v.max(0.0); }

        let mut mean = self.b3;
        for j in 0..HIDDEN2 { mean += h2[j] * self.w3[j]; }
        let mut log_var = self.b3_var;
        for j in 0..HIDDEN2 { log_var += h2[j] * self.w3_var[j]; }
        let log_var = log_var.clamp(-3.0, 6.0);
        let inv_var = (-log_var).exp();
        let err = mean - target;
        let sq_err = err * err;
        // Loss = 0.5·err²·inv_var + 0.5·log_var. We don't return the NLL itself
        // (caller wants RMSE-comparable signal); return sq_err so old logging works.
        // Gradients (per-sample, scaled by lr externally):
        //   d_loss/d_mean = err · inv_var
        //   d_loss/d_log_var = 0.5 · (1 − err² · inv_var)
        let d_mean = err * inv_var * lr;
        let d_logvar = 0.5 * (1.0 - sq_err * inv_var) * lr;

        // ─── Output layer gradients ───
        for j in 0..HIDDEN2 {
            self.w3[j] -= d_mean * h2[j];
            self.w3_var[j] -= d_logvar * h2[j];
        }
        self.b3 -= d_mean;
        self.b3_var -= d_logvar;

        // d_h2 = (d_mean·w3 + d_logvar·w3_var) · relu'(h2_pre)
        let mut d_h2 = [0.0f32; HIDDEN2];
        for j in 0..HIDDEN2 {
            if h2_pre[j] > 0.0 {
                d_h2[j] = d_mean * self.w3[j] + d_logvar * self.w3_var[j];
            }
        }

        // Layer 2 gradients
        for i in 0..HIDDEN1 {
            if h1[i] > 0.0 {
                let base = i * HIDDEN2;
                for j in 0..HIDDEN2 { self.w2[base + j] -= d_h2[j] * h1[i]; }
            }
        }
        for j in 0..HIDDEN2 { self.b2[j] -= d_h2[j]; }

        // d_h1 = W2^T @ d_h2 * relu'(h1_pre)
        let mut d_h1 = [0.0f32; HIDDEN1];
        for i in 0..HIDDEN1 {
            if h1_pre[i] > 0.0 {
                let base = i * HIDDEN2;
                for j in 0..HIDDEN2 { d_h1[i] += self.w2[base + j] * d_h2[j]; }
            }
        }

        // Layer 1 gradients (only active features)
        for &fi in features {
            let base = fi as usize * HIDDEN1;
            for j in 0..HIDDEN1 { self.w1[base + j] -= d_h1[j]; }
        }
        for j in 0..HIDDEN1 { self.b1[j] -= d_h1[j]; }

        sq_err
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

// ─────────────────────────────────────────────────────────────────────
// v6-peak: bounded play region + 6-dir adjacency + HabitatBucket
// ─────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "v6-peak", feature = "cards-alt-v2"))]
mod v6_peak {
    use super::*;
    use std::sync::OnceLock;

    /// Hex distance from cell at (col, row) to grid center.
    /// Note: cascadia_core::hex::GRID_SIZE is the TOTAL cell count (441);
    /// the linear dimension is GRID_DIM = 21. Center axial coords are (0, 0)
    /// at array (col=10, row=10).
    #[inline]
    fn hex_dist_from_center(col: usize, row: usize) -> i32 {
        let q = col as i32 - 10;
        let r = row as i32 - 10;
        q.abs().max(r.abs()).max((q + r).abs())
    }
    const GRID_DIM: usize = 21;

    /// Build the 441 → 127 lookup. Cells within hex distance ≤ V6_LOCAL_RADIUS
    /// from grid center get a local index assigned in spiral order (innermost ring
    /// first). Out-of-region cells map to -1.
    fn build_global_to_local() -> [i16; 441] {
        let mut tbl = [-1i16; 441];
        let mut next = 0u16;
        for d in 0..=V6_LOCAL_RADIUS {
            for col in 0..GRID_DIM {
                for row in 0..GRID_DIM {
                    if hex_dist_from_center(col, row) == d {
                        let global = col * GRID_DIM + row;
                        tbl[global] = next as i16;
                        next += 1;
                    }
                }
            }
        }
        debug_assert_eq!(next as usize, V6_LOCAL_CELLS,
            "spiral build produced {} cells, expected {}", next, V6_LOCAL_CELLS);
        tbl
    }

    fn build_local_to_global(g2l: &[i16; 441]) -> [u16; V6_LOCAL_CELLS] {
        let mut tbl = [0u16; V6_LOCAL_CELLS];
        for (g, &l) in g2l.iter().enumerate() {
            if l >= 0 { tbl[l as usize] = g as u16; }
        }
        tbl
    }

    static GLOBAL_TO_LOCAL: OnceLock<[i16; 441]> = OnceLock::new();
    static LOCAL_TO_GLOBAL: OnceLock<[u16; V6_LOCAL_CELLS]> = OnceLock::new();

    #[inline]
    pub fn global_to_local(global_idx: usize) -> i16 {
        GLOBAL_TO_LOCAL.get_or_init(build_global_to_local)[global_idx]
    }

    #[inline]
    pub fn local_to_global(local_idx: usize) -> u16 {
        let g2l = GLOBAL_TO_LOCAL.get_or_init(build_global_to_local);
        LOCAL_TO_GLOBAL.get_or_init(|| build_local_to_global(g2l))[local_idx]
    }

    /// For each terrain T, compute every cell's "habitat cluster role":
    ///   0 = part of largest connected T-cluster
    ///   1 = part of 2nd-largest T-cluster
    ///   2 = part of any other cluster (3rd+, singletons, etc.)
    ///   None = cell does not show terrain T on any edge (skip emitting role)
    ///
    /// Returns: roles[cell_idx][terrain_idx] = Option<u8>
    pub fn compute_habitat_roles(board: &Board) -> [[Option<u8>; 5]; 441] {
        let adj = &*ADJACENCY;
        let mut roles: [[Option<u8>; 5]; 441] = [[None; 5]; 441];

        for t in 0..5 {
            let terrain_t = match cascadia_core::types::Terrain::from_u8(t as u8) {
                Some(tt) => tt,
                None => continue,
            };

            // BFS each connected component (sharing terrain_t on shared edges).
            let mut visited = [false; 441];
            let mut clusters: Vec<Vec<u16>> = Vec::new();

            for &tile_idx in board.placed_tiles.iter() {
                let idx = tile_idx as usize;
                if visited[idx] { continue; }
                let cell = board.grid.get(idx);
                if !cell.is_present() { continue; }
                // Cell qualifies for terrain T if it has T on any edge.
                let has_t = (0..6).any(|d| {
                    cascadia_core::board::terrain_on_edge(cell, board.rotations[idx], d)
                        == Some(terrain_t)
                });
                if !has_t { continue; }
                // BFS this cluster.
                let mut comp: Vec<u16> = Vec::new();
                let mut queue: Vec<u16> = vec![tile_idx];
                visited[idx] = true;
                while let Some(cur) = queue.pop() {
                    comp.push(cur);
                    let cur_idx = cur as usize;
                    let cur_cell = board.grid.get(cur_idx);
                    let cur_rot = board.rotations[cur_idx];
                    for d in 0..6 {
                        let n_val = adj.neighbors[cur_idx][d];
                        if n_val == u16::MAX { continue; }
                        let n_idx = n_val as usize;
                        if visited[n_idx] { continue; }
                        let n_cell = board.grid.get(n_idx);
                        if !n_cell.is_present() { continue; }
                        // Both cells must show terrain_t on the shared edge.
                        let my_terr = cascadia_core::board::terrain_on_edge(cur_cell, cur_rot, d);
                        let n_terr = cascadia_core::board::terrain_on_edge(
                            n_cell, board.rotations[n_idx], (d + 3) % 6);
                        if my_terr == Some(terrain_t) && n_terr == Some(terrain_t) {
                            visited[n_idx] = true;
                            queue.push(n_val);
                        }
                    }
                }
                clusters.push(comp);
            }

            // Sort clusters by size desc, assign roles.
            let mut indexed: Vec<(usize, &Vec<u16>)> = clusters.iter().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.len().cmp(&a.1.len()));
            for (rank, (_, comp)) in indexed.iter().enumerate() {
                let role: u8 = if rank == 0 { 0 } else if rank == 1 { 1 } else { 2 };
                for &cell in comp.iter() {
                    roles[cell as usize][t] = Some(role);
                }
            }
        }

        roles
    }
}

#[cfg(feature = "v6-peak")]
pub use v6_peak::{global_to_local as v6_global_to_local, local_to_global as v6_local_to_global};

/// v6-peak feature extraction. Self-contained; does not share code with the
/// existing extract_features_with_bag for the v5 layout — uses the v6 layout
/// documented in the V6_PEAK_LAYOUT comment block above.
///
/// All per-cell blocks operate over LOCAL_CELLS=127 cells (hex distance ≤ 6
/// from grid center). Tiles placed outside this region (~0.1% of cases) silently
/// do not contribute per-cell features but still influence pairwise/pattern/
/// adjacency-style features unaffected by per-cell indexing.
#[cfg(feature = "v6-peak")]
pub fn extract_features_v6_peak(board: &Board, bag: Option<&BagInfo>) -> Vec<u16> {
    let mut features = Vec::with_capacity(800);
    let adj = &*ADJACENCY;

    // Block bases — laid out in the order documented in V6_PEAK_LAYOUT.
    let mut base = 0usize;
    let cell_core_base = base; base += V6_CELL_FEATURES;             // 1397
    let phase_base = base;     base += PHASE_FEATURES;                // 110
    let pair_wl_base = base;   base += PAIR_FEATURES;                 // 147
    let pattern_base = base;   base += PATTERN_FEATURES;              // 89
    let bag_base = base;       base += BAG_FEATURES;                  // 55
    let opp_hab_base = base;   base += OPP_HAB_FEATURES;              // 55
    let allowed_base = base;   base += V6_ALLOWED_WL_FEATURES;        // 635
    let wl_ext_base = base;    base += WL_COUNT_EXT_FEATURES;         // 50
    let terr_pair_base = base; base += TERRAIN_PAIR_FEATURES;         // 108
    let sec_terr_base = base;  base += V6_SEC_TERRAIN_FEATURES;       // 635
    let hab_ext_base = base;   base += HAB_EXT_FEATURES;              // 70
    let wl_ext2_base = base;   base += WL_COUNT_EXT2_FEATURES;        // 55
    let ext_cap_base = base;   base += EXT_CAP_FEATURES;              // 40
    let pat2_base = base;      base += PATTERN_V2_FEATURES;           // 48
    let bag_ext_base = base;   base += BAG_EXT_FEATURES;              // 105
    let opp_ext_base = base;   base += OPP_HAB_EXT_FEATURES;          // 70
    let market_base = base;    base += MARKET_FEATURES;               // 88
    let tbag_t_base = base;    base += TBAG_TERRAIN_FEATURES;         // 105
    let tbag_w_base = base;    base += TBAG_WL_FEATURES;              // 105
    let tbag_te_base = base;   base += TBAG_TERRAIN_EXT_FEATURES;     // 150
    let tbag_we_base = base;   base += TBAG_WL_EXT_FEATURES;          // 150
    let overflow_base = base;  base += OVERFLOW_FEATURES;             // 1
    let v4opp_base = base;     base += OPP_DETAILED_FEATURES;         // 369
    let frontier_base = base;  base += V6_FRONTIER_FLAG_FEATURES;     // 127
    let v5opp_base = base;     base += V5_OPP_PAT_FEATURES;           // 108
    let bonus_base = base;     base += V5_BONUS_DIST_FEATURES;        // 315
    let hab_str_base = base;   base += V5_HAB_STRUCT_FEATURES;        // 85
    let tbag_j_base = base;    base += V5_TBAG_JOINT_FEATURES;        // 525
    let cell_adj_wl_base = base;   base += V6_CELL_ADJ_WL_FEATURES;   // 5334
    let cell_adj_terr_base = base; base += V6_CELL_ADJ_TERR_FEATURES; // 4572
    let hab_bucket_base = base;    base += V6_HAB_BUCKET_FEATURES;    // 1905
    debug_assert_eq!(base, NUM_FEATURES_V6_PEAK);

    // Compute habitat roles once (used by HabitatBucket block).
    let hab_roles = v6_peak::compute_habitat_roles(board);

    // ── Per-cell core (1397) ──
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let local = v6_peak::global_to_local(idx);
        if local < 0 { continue; }
        let cell = board.grid.get(idx);
        let cb = cell_core_base + local as usize * V6_FPC;
        if let Some(w) = cell.placed_wildlife() {
            features.push((cb + w as usize) as u16);
        } else {
            features.push((cb + 5) as u16); // tile_no_wildlife
        }
        if let Some(t) = cell.primary_terrain() {
            features.push((cb + 6 + t as usize) as u16);
        }
    }

    // ── Phase (110) ──
    let turn = (board.tile_count as usize).saturating_sub(3).min(20);
    features.push((phase_base + turn) as u16);
    let tokens = (board.nature_tokens as usize).min(8);
    features.push((phase_base + TURN_FEATURES + tokens) as u16);
    let wl_block = phase_base + TURN_FEATURES + TOKEN_FEATURES;
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(5);
        features.push((wl_block + wtype * 6 + count) as u16);
    }
    let hab_block = wl_block + WL_COUNT_FEATURES;
    for terrain in 0..5 {
        let size = (board.largest_group[terrain] as usize).min(9);
        features.push((hab_block + terrain * 10 + size) as u16);
    }

    // ── Pairwise wildlife adj (147, 3 line dirs — kept as-is) ──
    for &tile_idx in board.placed_tiles.iter() {
        let start = HexCoord::from_index(tile_idx as usize);
        let my_wl = wildlife_code(board, tile_idx as usize);
        for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
            let neighbor = HexCoord::new(start.q + dq, start.r + dr);
            if let Some(nidx) = neighbor.to_index() {
                let n_wl = wildlife_code(board, nidx);
                if my_wl > 0 || n_wl > 0 {
                    let pair_idx = dir * PAIR_STATES + my_wl as usize * 7 + n_wl as usize;
                    features.push((pair_wl_base + pair_idx) as u16);
                }
            }
        }
    }

    // ── Patterns v1 (89) ──
    extract_pattern_features(board, &mut features, pattern_base);

    // ── Bag remaining + opp hab (55 + 55) ──
    if let Some(b) = bag {
        for wtype in 0..5 {
            let count = (b.remaining[wtype] as usize).min(BAG_BINS - 1);
            features.push((bag_base + wtype * BAG_BINS + count) as u16);
        }
        for terrain in 0..5 {
            let size = (b.max_opponent_habitat[terrain] as usize).min(OPP_HAB_BINS - 1);
            features.push((opp_hab_base + terrain * OPP_HAB_BINS + size) as u16);
        }
    }

    // ── Allowed wl per cell (635) ──
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let local = v6_peak::global_to_local(idx);
        if local < 0 { continue; }
        let cell = board.grid.get(idx);
        if cell.is_present() && !cell.has_wildlife() {
            let mask = cell.allowed_wildlife();
            for w in Wildlife::ALL {
                if mask.contains(w) {
                    features.push((allowed_base + local as usize * 5 + w as usize) as u16);
                }
            }
        }
    }

    // ── Wildlife count ext (50) ──
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(9);
        features.push((wl_ext_base + wtype * 10 + count) as u16);
    }

    // ── Terrain pairwise (108) ──
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let coord = HexCoord::from_index(idx);
        for (dir, &(dq, dr)) in HexCoord::LINE_DIRECTIONS.iter().enumerate() {
            let neighbor = HexCoord::new(coord.q + dq, coord.r + dr);
            if let Some(nidx) = neighbor.to_index() {
                if !board.grid.get(nidx).is_present() { continue; }
                let my_terrain = terrain_code_on_edge(board, idx, dir);
                let n_terrain = terrain_code_on_edge(board, nidx, (dir + 3) % 6);
                if my_terrain > 0 && n_terrain > 0 {
                    let pair_idx = dir * TERRAIN_PAIR_STATES + my_terrain as usize * 6 + n_terrain as usize;
                    features.push((terr_pair_base + pair_idx) as u16);
                }
            }
        }
    }

    // ── Sec terrain per cell (635) ──
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let local = v6_peak::global_to_local(idx);
        if local < 0 { continue; }
        let cell = board.grid.get(idx);
        if let Some(t) = cell.secondary_terrain() {
            features.push((sec_terr_base + local as usize * 5 + t as usize) as u16);
        }
    }

    // ── Hab ext (70), wl_ext2 (55) ──
    for terrain in 0..5 {
        let size = (board.largest_group[terrain] as usize).min(HAB_EXT_BINS - 1);
        features.push((hab_ext_base + terrain * HAB_EXT_BINS + size) as u16);
    }
    for wtype in 0..5 {
        let count = board.wildlife_positions[wtype].len().min(WL_COUNT_EXT2_BINS - 1);
        features.push((wl_ext2_base + wtype * WL_COUNT_EXT2_BINS + count) as u16);
    }

    // ── Pair extension capacity (40) ──
    {
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
                if has_extension { ext_cap_counts[wtype] += 1; }
            }
        }
        for wtype in 0..5 {
            let count = (ext_cap_counts[wtype] as usize).min(7);
            features.push((ext_cap_base + wtype * 8 + count) as u16);
        }
    }

    // ── Pattern v2 (48) ──
    extract_pattern_v2_features(board, &mut features, pat2_base);

    // ── Bag ext, opp hab ext, market, tbag terrain/wl/ext, overflow ──
    if let Some(b) = bag {
        for wtype in 0..5 {
            let count = (b.remaining[wtype] as usize).min(BAG_EXT_BINS - 1);
            features.push((bag_ext_base + wtype * BAG_EXT_BINS + count) as u16);
        }
        for terrain in 0..5 {
            let size = (b.max_opponent_habitat[terrain] as usize).min(OPP_HAB_EXT_BINS - 1);
            features.push((opp_ext_base + terrain * OPP_HAB_EXT_BINS + size) as u16);
        }
        for (i, slot) in b.market.iter().enumerate() {
            let slot_base = market_base + i * MARKET_PER_SLOT;
            if slot.terrain1 > 0 {
                features.push((slot_base + (slot.terrain1 - 1) as usize) as u16);
            }
            features.push((slot_base + 5 + slot.terrain2 as usize) as u16);
            let allowed_off = slot_base + 5 + 6;
            for w in 0..5 {
                if slot.allowed_mask & (1 << w) != 0 {
                    features.push((allowed_off + w) as u16);
                }
            }
            if slot.keystone {
                features.push((slot_base + 5 + 6 + 5) as u16);
            }
            if slot.wildlife_token > 0 {
                features.push((slot_base + 5 + 6 + 5 + 1 + (slot.wildlife_token - 1) as usize) as u16);
            }
        }
        for t in 0..5 {
            let count = (b.tbag_terrain[t] as usize).min(BAG_EXT_BINS - 1);
            features.push((tbag_t_base + t * BAG_EXT_BINS + count) as u16);
            let count2 = (b.tbag_terrain[t] as usize).min(TBAG_EXT_BINS - 1);
            features.push((tbag_te_base + t * TBAG_EXT_BINS + count2) as u16);
        }
        for w in 0..5 {
            let count = (b.tbag_wildlife[w] as usize).min(BAG_EXT_BINS - 1);
            features.push((tbag_w_base + w * BAG_EXT_BINS + count) as u16);
            let count2 = (b.tbag_wildlife[w] as usize).min(TBAG_EXT_BINS - 1);
            features.push((tbag_we_base + w * TBAG_EXT_BINS + count2) as u16);
        }
        if b.overflow_used {
            features.push(overflow_base as u16);
        }

        // ── v4-opp block (369) ──
        extract_opp_detailed_features(b, &mut features, v4opp_base);

        // ── v5-feat opp pattern detail (108) ──
        extract_v5_opp_pattern_features(b, &mut features, v5opp_base);

        // ── v5-feat bonus distance (315) ──
        extract_v5_bonus_distance_features(b, &mut features, bonus_base);

        // ── v5-feat tbag joint (525) ──
        extract_v5_tbag_joint_features(b, &mut features, tbag_j_base);
    }

    // ── v5-feat hab structure (85) — board-only ──
    extract_v5_hab_structure_features(board, &mut features, hab_str_base);

    // ── v5-feat frontier flag v6 (127) ──
    let frontier = board.frontier();
    for &fpos in frontier.iter() {
        let local = v6_peak::global_to_local(fpos as usize);
        if local < 0 { continue; }
        features.push((frontier_base + local as usize) as u16);
    }

    // ── Per-cell 6-dir wildlife adjacency (5334) ⭐ NEW ──
    // For each placed tile within v6 region, encode each of 6 neighbors' wildlife state.
    // States: 0=no tile (out-of-bounds OR empty cell), 1-5=wildlife type, 6=tile-no-wildlife.
    // Indexing: cell_adj_wl_base + local_idx * 6 * 7 + dir * 7 + wl_state
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let local = v6_peak::global_to_local(idx);
        if local < 0 { continue; }
        let lbase = cell_adj_wl_base + local as usize * V6_ADJ_DIRS * V6_ADJ_WL_STATES;
        for dir in 0..6 {
            let n_val = adj.neighbors[idx][dir];
            let wl_state = if n_val == u16::MAX {
                0u8 // OOB treated as no tile
            } else {
                wildlife_code(board, n_val as usize)
            };
            features.push((lbase + dir * V6_ADJ_WL_STATES + wl_state as usize) as u16);
        }
    }

    // ── Per-cell 6-dir terrain-on-edge (4572) ⭐ NEW ──
    // For each placed tile within v6 region, encode each neighbor's terrain on the
    // shared edge facing back toward us.
    // States: 0=no tile, 1-5=terrain type.
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let local = v6_peak::global_to_local(idx);
        if local < 0 { continue; }
        let lbase = cell_adj_terr_base + local as usize * V6_ADJ_DIRS * V6_ADJ_TERR_STATES;
        for dir in 0..6 {
            let n_val = adj.neighbors[idx][dir];
            let terr_state = if n_val == u16::MAX {
                0u8
            } else {
                terrain_code_on_edge(board, n_val as usize, (dir + 3) % 6)
            };
            features.push((lbase + dir * V6_ADJ_TERR_STATES + terr_state as usize) as u16);
        }
    }

    // ── HabitatBucket smaller (1905) ⭐ NEW ──
    // For each placed cell × each terrain T it touches, emit (terrain × cluster role).
    // Indexing: hab_bucket_base + local_idx * 5 * 3 + t * 3 + role
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let local = v6_peak::global_to_local(idx);
        if local < 0 { continue; }
        let lbase = hab_bucket_base + local as usize * 5 * V6_HAB_BUCKET_ROLES;
        for t in 0..5 {
            if let Some(role) = hab_roles[idx][t] {
                features.push((lbase + t * V6_HAB_BUCKET_ROLES + role as usize) as u16);
            }
        }
    }

    features
}

// ─────────────────────────────────────────────────────────────────────
// cards-alt feature extraction
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "cards-alt")]
fn extract_alt_bear_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let sizes = alt_components(board, Wildlife::Bear);
    let mut counts = [0u32; 4]; // [size-1, size-2, size-3, size-4+]
    for &s in sizes.iter() {
        match s { 1 => counts[0] += 1, 2 => counts[1] += 1, 3 => counts[2] += 1, _ => counts[3] += 1, }
    }
    for k in 0..4 {
        let bin = (counts[k] as usize).min(ALT_BEAR_SIZE_BINS - 1);
        features.push((base + k * ALT_BEAR_SIZE_BINS + bin) as u16);
    }
    let all_three = counts[0] >= 1 && counts[1] >= 1 && counts[2] >= 1;
    if all_three {
        features.push((base + 4 * ALT_BEAR_SIZE_BINS) as u16);
    }
}

#[cfg(feature = "cards-alt")]
fn extract_alt_elk_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        // Emit zero-bin for each shape kind.
        for k in 0..5 { features.push((base + k * ALT_ELK_SHAPE_BINS) as u16); }
        return;
    }
    let adj = &*ADJACENCY;
    let mut elk = [false; 441];
    for &p in positions.iter() { elk[p as usize] = true; }

    // Component scan: for each component, classify into:
    //   single | pair | triangle | rhombus | blob (anything else, incl. lines)
    let mut visited = [false; 441];
    let mut counts = [0u32; 5]; // [single, pair, triangle, rhombus, blob]
    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && elk[n] {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let n = comp.len();
        let kind = match n {
            1 => 0, // single
            2 => 1, // pair
            _ => {
                // Detect triangle (any 3 mutually adjacent)
                let mut has_triangle = false;
                for i in 0..comp.len() {
                    let a = comp[i] as usize;
                    let mut tri_nbrs: arrayvec::ArrayVec<u16, 6> = arrayvec::ArrayVec::new();
                    for nn in adj.neighbors_of(a) {
                        if elk[nn] && comp.contains(&(nn as u16)) {
                            let _ = tri_nbrs.try_push(nn as u16);
                        }
                    }
                    for ii in 0..tri_nbrs.len() {
                        for jj in (ii + 1)..tri_nbrs.len() {
                            let b = tri_nbrs[ii] as usize;
                            let cc = tri_nbrs[jj] as usize;
                            if adj.neighbors_of(b).any(|nn| nn == cc) {
                                has_triangle = true;
                                break;
                            }
                        }
                        if has_triangle { break; }
                    }
                    if has_triangle { break; }
                }
                if n == 3 && has_triangle { 2 }      // triangle
                else if n == 4 && has_triangle { 3 } // rhombus = 4-cluster with triangle subshape
                else { 4 }                            // blob (line, Y-shape, larger cluster)
            }
        };
        counts[kind] += 1;
    }
    for k in 0..5 {
        let bin = (counts[k] as usize).min(ALT_ELK_SHAPE_BINS - 1);
        features.push((base + k * ALT_ELK_SHAPE_BINS + bin) as u16);
    }
}

#[cfg(feature = "cards-alt")]
fn extract_alt_salmon_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let adj = &*ADJACENCY;
    // Find all valid runs with (length, adj_unique_non_salmon_types).
    let mut runs: arrayvec::ArrayVec<(usize, u32), 32> = arrayvec::ArrayVec::new();
    let mut visited = [false; 441];
    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let valid = comp.iter().all(|&c| {
            adj.neighbors_of(c as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count() <= 2
        });
        if !valid { continue; }
        let mut seen = [false; 441];
        let mut adj_types = 0u32;
        for &c in &comp {
            for n in adj.neighbors_of(c as usize) {
                if seen[n] { continue; }
                seen[n] = true;
                if let Some(w) = board.grid.get(n).placed_wildlife() {
                    if w != Wildlife::Salmon { adj_types += 1; }
                }
            }
        }
        let _ = runs.try_push((comp.len(), adj_types));
    }
    // Sort by run length descending; emit top-3 (qualifying-bit + adj-types-bin).
    runs.sort_by(|a, b| b.0.cmp(&a.0));
    for slot in 0..ALT_SALMON_TOP_RUNS {
        let off = base + slot * ALT_SALMON_PER_RUN;
        if let Some(&(len, adj_types)) = runs.get(slot) {
            if len >= 3 {
                features.push(off as u16); // qualifying bit
            }
            let adj_bin = (adj_types as usize).min(ALT_SALMON_ADJ_BINS - 1);
            features.push((off + 1 + adj_bin) as u16);
        } else {
            // Empty slot: emit adj-types=0 bin so the column still fires consistently.
            features.push((off + 1) as u16);
        }
    }
}

#[cfg(feature = "cards-alt")]
fn extract_alt_hawk_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    use cascadia_core::hex::HexCoord;
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let mut counts = [0u32; ALT_HAWK_PT_CLASSES]; // bins by intervening unique non-hawk types
    if positions.len() >= 2 {
        let mut hawk_set = [false; 441];
        let mut pos_to_idx = [u8::MAX; 441];
        for (i, &p) in positions.iter().enumerate() {
            hawk_set[p as usize] = true;
            pos_to_idx[p as usize] = i as u8;
        }
        for (i, &p) in positions.iter().enumerate() {
            let coord = HexCoord::from_index(p as usize);
            for &(dq, dr) in &HexCoord::DIRECTIONS {
                let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
                let mut steps = 1u32;
                let mut types_mask = 0u8;
                loop {
                    match cur.to_index() {
                        Some(idx) => {
                            if hawk_set[idx] {
                                let j = pos_to_idx[idx];
                                if (i as u8) < j && steps >= 2 {
                                    let unique = (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones() as usize;
                                    let cls = unique.min(ALT_HAWK_PT_CLASSES - 1);
                                    counts[cls] += 1;
                                }
                                break;
                            }
                            if let Some(w) = board.grid.get(idx).placed_wildlife() {
                                types_mask |= 1 << (w as u8);
                            }
                        }
                        None => break,
                    }
                    cur = HexCoord::new(cur.q + dq, cur.r + dr);
                    steps += 1;
                }
            }
        }
    }
    for k in 0..ALT_HAWK_PT_CLASSES {
        let bin = (counts[k] as usize).min(ALT_HAWK_COUNT_BINS - 1);
        features.push((base + k * ALT_HAWK_COUNT_BINS + bin) as u16);
    }
}

#[cfg(feature = "cards-alt")]
fn extract_alt_fox_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    let adj = &*ADJACENCY;
    let mut counts = [0u32; ALT_FOX_PT_CLASSES]; // foxes by # of pair-types
    for &p in positions.iter() {
        let mut tcounts = [0u8; 5];
        for n in adj.neighbors_of(p as usize) {
            if let Some(w) = board.grid.get(n).placed_wildlife() {
                if w != Wildlife::Fox { tcounts[w as usize] += 1; }
            }
        }
        let pair_types = tcounts.iter().filter(|&&c| c >= 2).count();
        let cls = pair_types.min(ALT_FOX_PT_CLASSES - 1);
        counts[cls] += 1;
    }
    for k in 0..ALT_FOX_PT_CLASSES {
        let bin = (counts[k] as usize).min(ALT_FOX_COUNT_BINS - 1);
        features.push((base + k * ALT_FOX_COUNT_BINS + bin) as u16);
    }
}

#[cfg(feature = "cards-alt")]
fn extract_alt_opp_features(bag: &BagInfo, features: &mut Vec<u16>, base: usize) {
    for (slot, opp) in bag.opp_detail.iter().enumerate() {
        let slot_base = base + slot * ALT_OPP_PER_OPP;
        let mut off = slot_base;
        // Bear class (5 bins)
        features.push((off + (opp.alt_bear_class as usize).min(ALT_OPP_BEAR_BINS - 1)) as u16);
        off += ALT_OPP_BEAR_BINS;
        // Elk shape (5 bins)
        features.push((off + (opp.alt_elk_shape as usize).min(ALT_OPP_ELK_SHAPE_BINS - 1)) as u16);
        off += ALT_OPP_ELK_SHAPE_BINS;
        // Salmon class (5 bins)
        features.push((off + (opp.alt_salmon_class as usize).min(ALT_OPP_SALMON_BINS - 1)) as u16);
        off += ALT_OPP_SALMON_BINS;
        // Hawk pair-types (5 bins)
        features.push((off + (opp.alt_hawk_pair_types as usize).min(ALT_OPP_HAWK_BINS - 1)) as u16);
        off += ALT_OPP_HAWK_BINS;
        // Fox pair-types (5 bins)
        features.push((off + (opp.alt_fox_pair_types as usize).min(ALT_OPP_FOX_BINS - 1)) as u16);
    }
}

// ─────────────────────────────────────────────────────────────────────
// cards-alt-v2: per-piece relational feature extractors
// ─────────────────────────────────────────────────────────────────────
//
// Helpers reuse v6_peak::global_to_local for the bounded play region (127
// cells, 99.9% coverage of real placements). Off-region cells are silently
// skipped — they won't fire for placed wildlife in practice.

#[cfg(feature = "cards-alt-v2")]
fn alt2_local_idx(global_idx: usize) -> Option<usize> {
    let l = v6_peak::global_to_local(global_idx);
    if l < 0 { None } else { Some(l as usize) }
}

/// Block A: per-hawk × direction × LOS-class.
/// 5 classes per direction:
///   0 = no other hawk on this hex axis (or off-grid)
///   1 = adjacent partner hawk (LOS pair scores 0 under D, but not "no partner")
///   2 = non-adjacent partner with 0 intervening non-hawk wildlife types
///   3 = non-adjacent partner with 1 intervening type
///   4 = non-adjacent partner with 2+ intervening types
#[cfg(feature = "cards-alt-v2")]
fn extract_alt2_hawk_los_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    use cascadia_core::hex::HexCoord;
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.is_empty() { return; }
    let mut hawk_set = [false; 441];
    for &p in positions.iter() { hawk_set[p as usize] = true; }

    for &p in positions.iter() {
        let local = match alt2_local_idx(p as usize) { Some(l) => l, None => continue };
        let coord = HexCoord::from_index(p as usize);
        let cell_base = base + local * ALT2_HAWK_LOS_PER_CELL;
        for (dir_i, &(dq, dr)) in HexCoord::DIRECTIONS.iter().enumerate() {
            let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
            let mut steps = 1u16;
            let mut types_mask = 0u8;
            let mut class: u8 = 0; // default no_partner
            loop {
                match cur.to_index() {
                    Some(idx) => {
                        if hawk_set[idx] {
                            class = if steps == 1 { 1 } // adj_partner
                            else {
                                let unique = (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones();
                                match unique {
                                    0 => 2,
                                    1 => 3,
                                    _ => 4,
                                }
                            };
                            break;
                        }
                        if let Some(w) = board.grid.get(idx).placed_wildlife() {
                            types_mask |= 1 << (w as u8);
                        }
                    }
                    None => break,
                }
                cur = HexCoord::new(cur.q + dq, cur.r + dr);
                steps += 1;
            }
            features.push((cell_base + dir_i * ALT2_HAWK_LOS_CLASSES + class as usize) as u16);
        }
    }
}

/// Block B: per-salmon × (length-class × adj-class).
/// Each salmon belongs to one component (run if valid). Encode the run's
/// length class plus the count of unique non-salmon adjacent types from the
/// salmon's perspective (its 6 neighbors).
#[cfg(feature = "cards-alt-v2")]
fn extract_alt2_salmon_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() { return; }
    let adj = &*ADJACENCY;

    // First pass: build per-salmon (component_size, valid).
    let mut visited = [false; 441];
    let mut salmon_meta: [(u8, bool); 441] = [(0, false); 441];

    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let valid = comp.iter().all(|&c| {
            adj.neighbors_of(c as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count() <= 2
        });
        let len = comp.len() as u8;
        for &c in &comp {
            salmon_meta[c as usize] = (len, valid);
        }
    }

    for &p in positions.iter() {
        let local = match alt2_local_idx(p as usize) { Some(l) => l, None => continue };
        let (len, valid) = salmon_meta[p as usize];
        // Length class (4 bins): invalid_or_singleton, len2, len3-4, len5+
        let len_class: usize = if !valid || len <= 1 { 0 }
                                else if len == 2 { 1 }
                                else if len <= 4 { 2 }
                                else { 3 };
        // Adj-class: count unique non-salmon adjacent TYPES around this salmon
        let mut types_mask = 0u8;
        for n in adj.neighbors_of(p as usize) {
            if let Some(w) = board.grid.get(n).placed_wildlife() {
                if w != Wildlife::Salmon { types_mask |= 1 << (w as u8); }
            }
        }
        let adj_count = types_mask.count_ones() as usize;
        let adj_class = adj_count.min(ALT2_SALMON_ADJ_CLASSES - 1);

        let class = len_class * ALT2_SALMON_ADJ_CLASSES + adj_class;
        features.push((base + local * ALT2_SALMON_PER_CELL + class) as u16);
    }
}

/// Block C: per-fox × (pair-type-class × density-class).
#[cfg(feature = "cards-alt-v2")]
fn extract_alt2_fox_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() { return; }
    let adj = &*ADJACENCY;

    for &p in positions.iter() {
        let local = match alt2_local_idx(p as usize) { Some(l) => l, None => continue };
        let mut tcounts = [0u8; 5];
        let mut occupied = 0u8;
        for n in adj.neighbors_of(p as usize) {
            if let Some(w) = board.grid.get(n).placed_wildlife() {
                if w != Wildlife::Fox { tcounts[w as usize] += 1; }
                occupied += 1;
            }
        }
        let pair_types = tcounts.iter().filter(|&&c| c >= 2).count();
        let pt_class = pair_types.min(ALT2_FOX_PT_CLASSES - 1);
        // Density class: 0=0-1, 1=2-3, 2=4-5, 3=6
        let density_class: usize = if occupied <= 1 { 0 }
                                   else if occupied <= 3 { 1 }
                                   else if occupied <= 5 { 2 }
                                   else { 3 };
        let class = pt_class * ALT2_FOX_DENSITY_CLASSES + density_class;
        features.push((base + local * ALT2_FOX_PER_CELL + class) as u16);
    }
}

/// Block D: per-bear × (size-class × extension-class).
#[cfg(feature = "cards-alt-v2")]
fn extract_alt2_bear_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    if positions.is_empty() { return; }
    let adj = &*ADJACENCY;

    // Component scan to compute (size, ext_slots) per bear.
    let mut visited = [false; 441];
    let mut bear_meta: [(u8, u8); 441] = [(0, 0); 441];
    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let size = comp.len() as u8;
        // Ext slots: count unique empty cells adjacent to ANY component member that allow bear.
        let mut seen = [false; 441];
        let mut ext = 0u8;
        for &c in &comp {
            for n in adj.neighbors_of(c as usize) {
                if seen[n] { continue; }
                seen[n] = true;
                let cell = board.grid.get(n);
                if cell.is_present() && !cell.has_wildlife() && cell.can_place_wildlife(Wildlife::Bear) {
                    ext += 1;
                }
            }
        }
        for &c in &comp {
            bear_meta[c as usize] = (size, ext);
        }
    }

    for &p in positions.iter() {
        let local = match alt2_local_idx(p as usize) { Some(l) => l, None => continue };
        let (size, ext) = bear_meta[p as usize];
        // Size class (4): single, pair, triple, quad+
        let size_class: usize = match size { 1 => 0, 2 => 1, 3 => 2, _ => 3 };
        // Ext class (3): 0=no_ext, 1=1-2_slots, 2=3+_slots
        let ext_class: usize = if ext == 0 { 0 }
                                else if ext <= 2 { 1 }
                                else { 2 };
        let class = size_class * ALT2_BEAR_EXT_CLASSES + ext_class;
        features.push((base + local * ALT2_BEAR_PER_CELL + class) as u16);
    }
}

/// Block E: per-elk × shape-role.
/// Roles: 0=single, 1=pair-end, 2=triangle-vertex (3 mutually adjacent),
///        3=rhombus-vertex (in a 4-cluster containing a triangle),
///        4=line-of-3, 5=line-of-4+, 6=blob/other (size 5+ non-line, etc.)
#[cfg(feature = "cards-alt-v2")]
fn extract_alt2_elk_features(board: &Board, features: &mut Vec<u16>, base: usize) {
    use cascadia_core::hex::HexCoord;
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() { return; }
    let adj = &*ADJACENCY;
    let mut elk = [false; 441];
    for &p in positions.iter() { elk[p as usize] = true; }

    // BFS per component, classify shape, assign role to each member.
    let mut visited = [false; 441];
    let mut elk_role: [u8; 441] = [0; 441];

    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && elk[n] {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let n = comp.len();
        // Shape detection
        let mut has_triangle = false;
        for i in 0..comp.len() {
            let a = comp[i] as usize;
            let mut tri_nbrs: arrayvec::ArrayVec<u16, 6> = arrayvec::ArrayVec::new();
            for nn in adj.neighbors_of(a) {
                if elk[nn] && comp.contains(&(nn as u16)) {
                    let _ = tri_nbrs.try_push(nn as u16);
                }
            }
            for ii in 0..tri_nbrs.len() {
                for jj in (ii + 1)..tri_nbrs.len() {
                    let b = tri_nbrs[ii] as usize;
                    let cc = tri_nbrs[jj] as usize;
                    if adj.neighbors_of(b).any(|nn| nn == cc) {
                        has_triangle = true;
                        break;
                    }
                }
                if has_triangle { break; }
            }
            if has_triangle { break; }
        }
        // Line detection: if no triangle, check if ALL elk are colinear in some axis.
        let is_line = !has_triangle && {
            // Pick first elk, scan all 3 directions; if all elk fit on one line through any axis, it's a line.
            let mut found_line = false;
            if comp.len() >= 2 {
                let coord0 = HexCoord::from_index(comp[0] as usize);
                'lines: for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
                    // Walk back from coord0
                    let mut start = coord0;
                    loop {
                        let prev = HexCoord::new(start.q - dq, start.r - dr);
                        if let Some(pidx) = prev.to_index() {
                            if elk[pidx] { start = prev; continue; }
                        }
                        break;
                    }
                    let mut line_set = [false; 441];
                    let mut line_count = 0usize;
                    let mut cur = start;
                    loop {
                        if let Some(idx) = cur.to_index() {
                            if elk[idx] {
                                line_set[idx] = true;
                                line_count += 1;
                                cur = HexCoord::new(cur.q + dq, cur.r + dr);
                                continue;
                            }
                        }
                        break;
                    }
                    if line_count == comp.len() && comp.iter().all(|&c| line_set[c as usize]) {
                        found_line = true;
                        break 'lines;
                    }
                }
            }
            found_line
        };
        let role: u8 = if n == 1 { 0 }
                       else if n == 2 { 1 }
                       else if n == 3 && has_triangle { 2 }
                       else if n == 4 && has_triangle { 3 }
                       else if is_line && n == 3 { 4 }
                       else if is_line && n >= 4 { 5 }
                       else { 6 };
        for &c in &comp {
            elk_role[c as usize] = role;
        }
    }

    for &p in positions.iter() {
        let local = match alt2_local_idx(p as usize) { Some(l) => l, None => continue };
        let role = elk_role[p as usize] as usize;
        features.push((base + local * ALT2_ELK_SHAPE_CLASSES + role) as u16);
    }
}
