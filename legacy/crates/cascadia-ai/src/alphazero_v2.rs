//! AlphaZero v2 — compact 127-cell hex disk + Set-Transformer entity stream +
//! cross-attention + multi-head value with phase gate.
//!
//! Trained from scratch on AAAAA with-bonus. Forward-only in Rust; training
//! happens in MLX (see `train_alphazero_mlx_v2.py`).
//!
//! Reuses the empirically-validated 441→127 hex-disk lookup from `nnue::v6_peak`
//! (overnight/v6_bounded_design.md: 99.9% of placements lie within radius 6).
//!
//! Sizing:
//!   trunk      : 96ch × 6 ResHexBlocks over 128 cells (127 disk + 1 pad)
//!   entities   : 8 tokens × 64 dim (4 market + 3 opp + 1 bag)
//!   attention  : 2 SAB blocks (4 heads) on entities; 2-head cross-attn board←ent
//!   value      : MLP 96→128→16 sub-heads, 3-way phase gate, convex blend → [0,1]
//!   policy     : factorized (tile-cell, wildlife-cell, market, wl-market, skip)
//!
//! All weights are flat f32 vectors; row-major layout documented at each site.

#![cfg(feature = "az-v2")]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::sync::OnceLock;

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::{HexCoord, ADJACENCY, GRID_SIZE};
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::{ScoringCards, Wildlife};

use crate::eval::ScoredMove;
use crate::nnue::v6_peak::{global_to_local, local_to_global};
use crate::search::{execute_scored_move, greedy_move};

// ─────────────────────────────────────────────────────────────────────
// Architectural constants
// ─────────────────────────────────────────────────────────────────────

/// Hex distance ≤ 6 disk = 127 cells (99.9% empirical coverage).
pub const AZ_LOCAL_CELLS: usize = 127;
/// Pad cell index appended after the 127 real cells. Out-of-disk moves and
/// out-of-disk hex neighbors route through this index — every plane is 0 on
/// the pad cell except the bias plane (plane 0), which is 1 on real cells and
/// 0 on pad. That single difference is enough to mark "real vs pad" for the
/// trunk; no dedicated is-pad plane is needed.
pub const AZ_PAD_INDEX: usize = 127;
pub const AZ_CELLS_PADDED: usize = 128;

/// Input planes (Phase 0.5 layout, no broadcast scalars):
///   0       bias (1.0 on real cells, 0 on pad)
///   1       tile present
///   2       keystone
///   3       free wildlife slot (present + can-place + empty)
///   4       is-frontier (empty real cell adjacent to ≥1 placed)
///   5–9     placed wildlife one-hot (Bear/Elk/Salmon/Hawk/Fox)
///  10–14    allowed wildlife mask (5)
///  15–44    per-direction × per-terrain edge bits (6 dirs × 5 terrains = 30)
///  45–59    cluster role per terrain (5 terrains × {largest, 2nd, other} = 15)
///  60–65    per-direction same-wildlife adjacency (6)
///  66       distance-to-frontier (Chebyshev / 10, 0 on pad)
///  67       placement recency ((stack_pos + 1) / placed_tiles_len, 0 empty/pad)
pub const AZ_INPUT_CHANNELS_V2: usize = 68;

pub const AZ_TRUNK_CHANNELS: usize = 96;
pub const AZ_TRUNK_BLOCKS: usize = 6;

/// Shared opponent trunk: 32 channels × 3 ResHexBlocks. Applied once per
/// opponent (3× total) per evaluation, weights shared. Pooled output → 32-dim
/// vector → drops into entity tokens 4..7 at forward time. The 14-scalar
/// summary that used to live there is gone; race-state token still carries
/// the bonus-race rank signal.
pub const AZ_OPP_TRUNK_CHANNELS: usize = 32;
pub const AZ_OPP_TRUNK_BLOCKS: usize = 3;
pub const AZ_MAX_OPPONENTS: usize = 3;

/// Entity token layout (Phase 0.5):
///   0–3 : market slots (4)
///   4–6 : opponents (3, 14-scalar summary by design)
///   7   : bag belief
///   8   : globals (phase one-hot, seat one-hot, num-players one-hot, etc.)
///   9   : race state (per-terrain rank one-hot)
pub const AZ_ENTITY_TOKENS: usize = 10;
pub const AZ_ENTITY_RAW_DIM: usize = 32;
pub const AZ_ENTITY_DIM: usize = 64;
pub const AZ_ATTN_HEADS: usize = 4;
pub const AZ_ATTN_HEAD_DIM: usize = AZ_ENTITY_DIM / AZ_ATTN_HEADS; // 16
pub const AZ_SAB_BLOCKS: usize = 2;
pub const AZ_SAB_FFN_DIM: usize = 128;

pub const AZ_CROSS_HEADS: usize = 2;
pub const AZ_CROSS_HEAD_DIM: usize = AZ_ENTITY_DIM / AZ_CROSS_HEADS; // 32

pub const AZ_VALUE_HIDDEN: usize = 128;
pub const AZ_VALUE_SUBHEADS: usize = 16; // 5 wl + 5 hab + 1 token + 5 bonus
pub const AZ_VALUE_PHASES: usize = 3;

pub const AZ_VALUE_SCALE: f32 = 120.0;

pub const AZ_MAGIC_V2: &[u8; 4] = b"AZR2";
pub const AZ_DATA_MAGIC_V2: &[u8; 4] = b"AZD2";

/// Aux-value normalization factors per sub-head, calibrated to empirical
/// AAAAA-with-bonus maxes. Each head sees a roughly uniform [0,1] target
/// range so the MSE per head is evenly weighted.
///   0–4  wildlife per type — Bear/Elk: 4-pair max; Salmon/Hawk: long runs;
///        Fox: full diversity
///   5–9  habitat per terrain — largest connected cluster
///   10   nature tokens — typical 0–5, hard cap 8
///   11–15 habitat bonus per terrain — max 3 in 4-player
pub const AZ_AUX_SCALES: [f32; AZ_VALUE_SUBHEADS] = [
    13.0, 13.0, 25.0, 25.0, 12.0, // wildlife: Bear, Elk, Salmon, Hawk, Fox
    15.0, 15.0, 15.0, 15.0, 15.0, // habitat per terrain
    5.0,  // nature tokens
    3.0, 3.0, 3.0, 3.0, 3.0, // habitat bonus per terrain
];

const LN_EPS: f32 = 1e-5;

/// Pre-built plane-0 (bias) layout: 1.0 on real cells (0..AZ_LOCAL_CELLS),
/// 0.0 on the pad cell. Used by `encode_board_planes` via a single
/// `copy_from_slice` instead of a 127-iter write loop.
static BIAS_PLANE: std::sync::LazyLock<[f32; AZ_CELLS_PADDED]> = std::sync::LazyLock::new(|| {
    let mut p = [0.0f32; AZ_CELLS_PADDED];
    for li in 0..AZ_LOCAL_CELLS {
        p[li] = 1.0;
    }
    p
});

// ─────────────────────────────────────────────────────────────────────
// Hex-neighbor lookup over the 128 padded cells.
// neighbors[i] = [self, n_E, n_NE, n_NW, n_W, n_SW, n_SE]
// Out-of-disk neighbors → AZ_PAD_INDEX (127).
// The pad cell at index 127 has all 7 entries = AZ_PAD_INDEX so it stays inert.
// ─────────────────────────────────────────────────────────────────────

static HEX_NEIGHBORS_LOCAL: OnceLock<Vec<[usize; 7]>> = OnceLock::new();

fn build_hex_neighbors_local() -> Vec<[usize; 7]> {
    let mut out = vec![[AZ_PAD_INDEX; 7]; AZ_CELLS_PADDED];
    for li in 0..AZ_LOCAL_CELLS {
        let gi = local_to_global(li) as usize;
        let coord = HexCoord::from_index(gi);
        let mut row = [AZ_PAD_INDEX; 7];
        row[0] = li; // self
        for (d, &(dq, dr)) in HexCoord::DIRECTIONS.iter().enumerate() {
            let nq = coord.q + dq;
            let nr = coord.r + dr;
            row[1 + d] = match HexCoord::new(nq, nr).to_index() {
                Some(gn) => {
                    let ln = global_to_local(gn);
                    if ln >= 0 {
                        ln as usize
                    } else {
                        AZ_PAD_INDEX
                    }
                }
                None => AZ_PAD_INDEX,
            };
        }
        out[li] = row;
    }
    out
}

#[inline]
pub fn hex_neighbors_local() -> &'static [[usize; 7]] {
    HEX_NEIGHBORS_LOCAL.get_or_init(build_hex_neighbors_local)
}

// ─────────────────────────────────────────────────────────────────────
// Public configuration + sample types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct AlphaZeroV2Config {
    pub channels: usize,
    pub blocks: usize,
    pub entity_dim: usize,
    pub sab_blocks: usize,
    pub heads: usize,
    pub value_hidden: usize,
    pub value_subheads: usize,
    pub max_candidates: usize,
    pub c_puct: f32,
}

impl Default for AlphaZeroV2Config {
    fn default() -> Self {
        AlphaZeroV2Config {
            channels: AZ_TRUNK_CHANNELS,
            blocks: AZ_TRUNK_BLOCKS,
            entity_dim: AZ_ENTITY_DIM,
            sab_blocks: AZ_SAB_BLOCKS,
            heads: AZ_ATTN_HEADS,
            value_hidden: AZ_VALUE_HIDDEN,
            value_subheads: AZ_VALUE_SUBHEADS,
            max_candidates: 24,
            c_puct: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AzSampleV2 {
    /// Own-board input planes, channel-major row layout: input[c * 128 + cell].
    pub input: Vec<f32>,
    /// Opponent-board input planes (3 boards in seat-rotation order), each with
    /// the same 68×128 channel-major layout as `input`. Zero-padded for
    /// missing opponents in <4-player games.
    pub opp_inputs: Vec<Vec<f32>>,
    /// Entity tokens, row-major: entities[t * 32 + d]. Opponent slots (4..7)
    /// are intentionally zero at sample time; the shared opp trunk fills them
    /// during the network forward pass from `opp_inputs`.
    pub entities: Vec<f32>,
    pub candidates: Vec<ScoredMove>,
    pub policy: Vec<f32>,
    /// Scalar final-score target normalized to [0,1] by AZ_VALUE_SCALE.
    pub value: f32,
    /// 16 sub-head targets, each in [0,1] after scaling by AZ_AUX_SCALES.
    pub aux_values: [f32; AZ_VALUE_SUBHEADS],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AzDataV2Summary {
    pub samples: usize,
    pub input_channels: usize,
    pub local_cells: usize,
    pub cells_padded: usize,
    pub entity_tokens: usize,
    pub entity_raw_dim: usize,
    pub value_subheads: usize,
    pub max_opponents: usize,
    pub max_candidates: usize,
}

// ─────────────────────────────────────────────────────────────────────
// Encoder (Phase 0.5): 68 planes × 128 cells, no broadcast scalars.
// All global / categorical signal lives in the entity stream.
// ─────────────────────────────────────────────────────────────────────

#[inline]
fn set_plane(input: &mut [f32], plane: usize, cell: usize, v: f32) {
    input[plane * AZ_CELLS_PADDED + cell] = v;
}

/// Axial hex distance: max(|dq|, |dr|, |dq + dr|).
#[inline]
fn hex_distance(a: HexCoord, b: HexCoord) -> usize {
    let dq = (a.q as i32) - (b.q as i32);
    let dr = (a.r as i32) - (b.r as i32);
    let dqr = dq + dr;
    dq.unsigned_abs()
        .max(dr.unsigned_abs())
        .max(dqr.unsigned_abs()) as usize
}

/// Map current turn / total turns → 3-way phase index (0=early, 1=mid, 2=late).
/// Identical threshold logic to the value head's `phase_one_hot()`.
#[inline]
fn phase_index(game: &GameState) -> usize {
    let total = (20 * game.num_players).max(1) as f32;
    let played = (total - game.turns_remaining as f32).clamp(0.0, total);
    let frac = played / total;
    if frac < 1.0 / 3.0 {
        0
    } else if frac < 2.0 / 3.0 {
        1
    } else {
        2
    }
}

/// Rank category against opponents per the habitat-bonus rules from
/// `scoring/mod.rs::compute_with_bonuses`:
///   0 sole-largest, 1 tied-largest, 2 sole-2nd, 3 behind.
fn habitat_rank(game: &GameState, player: usize, terrain_idx: usize) -> usize {
    let my_size = game.boards[player].largest_group[terrain_idx];
    let mut larger = 0usize;
    let mut tied = 0usize;
    for (i, b) in game.boards.iter().enumerate() {
        if i == player {
            continue;
        }
        let their = b.largest_group[terrain_idx];
        if their > my_size {
            larger += 1;
        } else if their == my_size {
            tied += 1;
        }
    }
    if larger == 0 && tied == 0 {
        0 // sole largest
    } else if larger == 0 {
        1 // tied largest
    } else if larger == 1 && tied == 0 {
        2 // sole 2nd
    } else {
        3 // behind
    }
}

/// Encode a single board's spatial state into the 68×128 plane layout.
///
/// Plane layout (all per-cell, no broadcast scalars, no game-level state):
///   0     bias (1.0 on real cells, 0 on pad)
///   1     tile present
///   2     keystone
///   3     free wildlife slot (present + can-place + empty)
///   4     is-frontier (empty real cell adjacent to ≥1 placed)
///   5–9   placed wildlife one-hot
///  10–14  allowed wildlife mask
///  15–44  per-direction × per-terrain edge bits (6 dirs × 5 terrains = 30)
///  45–59  cluster role per terrain (5 terrains × {largest/2nd/other} = 15)
///  60–65  per-direction same-wildlife adjacency (6)
///  66     distance-to-frontier (hex dist / 10, 0 on pad)
///  67     placement recency ((stack_pos + 1) / placed_tiles_len, 0 if empty)
///
/// Used both for the *own* board (input to the main trunk) and for each
/// opponent's board (input to the shared opponent trunk).
pub fn encode_board_planes(board: &Board) -> Vec<f32> {
    let mut input = vec![0.0f32; AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED];

    // Plane 0: bias. Pre-built static written via one memcpy instead of 127
    // indexed stores.
    input[0..AZ_CELLS_PADDED].copy_from_slice(&*BIAS_PLANE);

    // Frontier set (empty cells adjacent to ≥1 placed). Used for both the
    // is-frontier plane (4) and the distance-to-frontier plane (66).
    let frontier_global = board.frontier();
    let mut frontier_coords: Vec<HexCoord> = frontier_global
        .iter()
        .map(|&gi| HexCoord::from_index(gi as usize))
        .collect();

    // Per-cell structural planes.
    for li in 0..AZ_LOCAL_CELLS {
        let gi = local_to_global(li) as usize;
        let cell = board.grid.get(gi);
        if cell.is_present() {
            set_plane(&mut input, 1, li, 1.0);
            if cell.is_keystone() {
                set_plane(&mut input, 2, li, 1.0);
            }
            // free wildlife slot: present + no wildlife placed yet + at least
            // one wildlife type allowed.
            if cell.placed_wildlife().is_none() {
                let any_allowed = Wildlife::ALL.iter().any(|&w| cell.can_place_wildlife(w));
                if any_allowed {
                    set_plane(&mut input, 3, li, 1.0);
                }
            }
            if let Some(w) = cell.placed_wildlife() {
                set_plane(&mut input, 5 + w as usize, li, 1.0);
            }
            for w in Wildlife::ALL {
                if cell.can_place_wildlife(w) {
                    set_plane(&mut input, 10 + w as usize, li, 1.0);
                }
            }
        }
    }

    // Plane 4: is-frontier flag on the empty-but-adjacent-to-placed cells.
    for &gi_u in frontier_global.iter() {
        let gi = gi_u as usize;
        let li = global_to_local(gi);
        if li >= 0 {
            set_plane(&mut input, 4, li as usize, 1.0);
        }
    }

    // Planes 15–44: per-direction × per-terrain edge bits.
    // For each placed cell, for each of 6 hex directions, if the tile's edge
    // in that direction has terrain T, plane index 15 + d*5 + T fires.
    for li in 0..AZ_LOCAL_CELLS {
        let gi = local_to_global(li) as usize;
        let cell = board.grid.get(gi);
        if !cell.is_present() {
            continue;
        }
        let rot = board.rotations[gi];
        for d in 0..6 {
            if let Some(t) = cascadia_core::board::terrain_on_edge(cell, rot, d) {
                set_plane(&mut input, 15 + d * 5 + t as usize, li, 1.0);
            }
        }
    }

    // Planes 45–59: cluster role per terrain via proper BFS.
    let roles = crate::nnue::v6_peak::compute_habitat_roles(board);
    for li in 0..AZ_LOCAL_CELLS {
        let gi = local_to_global(li) as usize;
        for ti in 0..5 {
            if let Some(role) = roles[gi][ti] {
                set_plane(&mut input, 45 + ti * 3 + role as usize, li, 1.0);
            }
        }
    }

    // Planes 60–65: per-direction same-wildlife adjacency.
    let adj = &*ADJACENCY;
    for li in 0..AZ_LOCAL_CELLS {
        let gi = local_to_global(li) as usize;
        let cell = board.grid.get(gi);
        let own_wl = match cell.placed_wildlife() {
            Some(w) => w,
            None => continue,
        };
        for d in 0..6 {
            let n_u16 = adj.neighbors[gi][d];
            if n_u16 == u16::MAX {
                continue;
            }
            let ncell = board.grid.get(n_u16 as usize);
            if let Some(nwl) = ncell.placed_wildlife() {
                if nwl == own_wl {
                    set_plane(&mut input, 60 + d, li, 1.0);
                }
            }
        }
    }

    // Plane 66: distance-to-frontier (hex distance / 10, clipped at 10).
    if frontier_coords.is_empty() {
        for li in 0..AZ_LOCAL_CELLS {
            set_plane(&mut input, 66, li, 1.0);
        }
    } else {
        frontier_coords.sort_by_key(|c| (c.q, c.r));
        frontier_coords.dedup();
        for li in 0..AZ_LOCAL_CELLS {
            let gi = local_to_global(li) as usize;
            let coord = HexCoord::from_index(gi);
            let dist = frontier_coords
                .iter()
                .map(|&fc| hex_distance(coord, fc))
                .min()
                .unwrap_or(10);
            let v = (dist.min(10) as f32) / 10.0;
            set_plane(&mut input, 66, li, v);
        }
    }

    // Plane 67: placement recency.
    let n_placed = board.placed_tiles.len();
    if n_placed > 0 {
        let denom = n_placed as f32;
        for (k, &gi_u) in board.placed_tiles.iter().enumerate() {
            let gi = gi_u as usize;
            let li = global_to_local(gi);
            if li >= 0 {
                let recency = (k as f32 + 1.0) / denom;
                set_plane(&mut input, 67, li as usize, recency);
            }
        }
    }

    input
}

/// Build an empty (all-zero) 68×128 board tensor — used to pad opponent slots
/// in <4-player games so the opp trunk always sees `AZ_MAX_OPPONENTS` inputs.
/// The pad opp's bias plane (plane 0) is also zero, distinguishing it from a
/// real but empty opponent board (which would have bias=1 on real cells).
pub fn zero_board_planes() -> Vec<f32> {
    vec![0.0f32; AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED]
}

/// Encode the current game's perspective:
///   - `own_input` : 68×128 plane tensor for the player whose turn it is
///   - `opp_inputs`: 3 boards in seat-rotation order (+1, +2, +3 ahead of own).
///                   Missing opponents (in <4-player games) are zero-padded.
///   - `entities`  : 10×32 entity stream. Opponent slots 4..7 are intentionally
///                   zeroed; their content is filled by the shared opponent
///                   trunk at forward time.
pub fn encode_game_local(game: &GameState) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let player = game.current_player;
    let own_input = encode_board_planes(&game.boards[player]);

    // Walk distinct opponents in seat-rotation order (offset 1..num_players);
    // pad remaining slots up to AZ_MAX_OPPONENTS with zero boards.
    //
    // The prior `offset in 1..=AZ_MAX_OPPONENTS` scheme would duplicate the
    // same opponent in 2-player games (offsets 1 and 3 both land on seat 1),
    // because the modulus wrapped past `player` and back to a real seat. Not a
    // bug in our 4-P AAAAA pipeline, but worth fixing so the encoder is right
    // for any player count.
    let mut opp_inputs: Vec<Vec<f32>> = Vec::with_capacity(AZ_MAX_OPPONENTS);
    for offset in 1..game.num_players {
        if opp_inputs.len() >= AZ_MAX_OPPONENTS {
            break;
        }
        let opp_seat = (player + offset) % game.num_players;
        opp_inputs.push(encode_board_planes(&game.boards[opp_seat]));
    }
    while opp_inputs.len() < AZ_MAX_OPPONENTS {
        opp_inputs.push(zero_board_planes());
    }
    debug_assert_eq!(opp_inputs.len(), AZ_MAX_OPPONENTS);

    let entities = build_entity_tokens(game);
    (own_input, opp_inputs, entities)
}

/// Build 10 entity tokens × 32 dims (row-major). Phase 0.5 layout:
///   tokens[0..4]   : market slots (4)
///   tokens[4..7]   : opponents (3; padded with zeros if fewer players)
///   tokens[7]      : bag belief
///   tokens[8]      : globals (phase, seat, num-players, etc.)
///   tokens[9]      : race state (per-terrain rank one-hot)
pub fn build_entity_tokens(game: &GameState) -> Vec<f32> {
    let player = game.current_player;
    let mut ent = vec![0.0f32; AZ_ENTITY_TOKENS * AZ_ENTITY_RAW_DIM];

    let market_three = game.market.has_3_of_kind();
    let overflow_wildlife = game.can_replace_overflow();
    let joint = game.tile_bag.joint_distribution();

    // ── Tokens 0..4: market slots ──
    let avail: Vec<_> = game.market.available().collect();
    for (slot, (_, pair)) in avail.iter().take(4).enumerate() {
        let base = slot * AZ_ENTITY_RAW_DIM;
        // 0..5: primary terrain one-hot
        ent[base + pair.tile.terrain1 as usize] = 1.0;
        // 5..10: secondary terrain one-hot (zero if single tile)
        if let Some(t2) = pair.tile.terrain2 {
            ent[base + 5 + t2 as usize] = 1.0;
        }
        // 10: dual-tile flag
        ent[base + 10] = if pair.tile.terrain2.is_some() {
            1.0
        } else {
            0.0
        };
        // 11..16: wildlife one-hot
        ent[base + 11 + pair.wildlife as usize] = 1.0;
        // 16..21: wildlife allowed mask
        for w in Wildlife::ALL {
            if pair.tile.allowed.contains(w) {
                ent[base + 16 + w as usize] = 1.0;
            }
        }
        // 21: keystone flag
        ent[base + 21] = if pair.tile.keystone { 1.0 } else { 0.0 };
        // 22..26: slot position one-hot
        ent[base + 22 + slot] = 1.0;
        // 26: 3-of-a-kind alert (this slot's wildlife matches the trio)
        if market_three == Some(pair.wildlife) {
            ent[base + 26] = 1.0;
        }
        // 27: overflow-eligible (this slot's wildlife is the overflow type)
        if overflow_wildlife == Some(pair.wildlife) {
            ent[base + 27] = 1.0;
        }
        // 28: joint-rarity — remaining tiles with this terrain AND wildlife / 20
        let t1 = pair.tile.terrain1 as usize;
        let w = pair.wildlife as usize;
        let joint_count = joint[t1][w];
        ent[base + 28] = (joint_count as f32 / 20.0).min(1.0);
        // 29..32: reserved zeros
    }

    // ── Tokens 4..7: opponents — left ZERO at encode time. ──
    //
    // The shared opponent trunk fills these slots from the opp_inputs spatial
    // tensors during the network forward pass. Don't write anything here; the
    // forward() pass would overwrite it anyway, and zeroing keeps the AZD2
    // payload smaller (dense-zero compresses well at the OS level).

    // ── Token 7: bag belief ──
    let bag_base = 7 * AZ_ENTITY_RAW_DIM;
    let (tile_dist, _) = game.tile_bag.feature_distributions();
    let wl_in_bag = game.wildlife_bag.counts_per_type();
    let tile_remaining = game.tile_bag.remaining();
    let wl_remaining = game.wildlife_bag.remaining();
    // 0..5: tile-bag terrain counts / 25 (per-terrain tile counts top out near 25)
    for t in 0..5 {
        ent[bag_base + t] = (tile_dist[t] as f32 / 25.0).min(1.0);
    }
    // 5..10: wildlife-bag counts per type / 20 (FIX: bag has 20 of each species)
    for w in 0..5 {
        ent[bag_base + 5 + w] = (wl_in_bag[w] as f32 / 20.0).min(1.0);
    }
    // 10: dual-tile fraction in remaining tile bag.
    // sum(tile_dist) = singles*1 + duals*2 = remaining + duals  =>  duals = sum - remaining.
    let dual_count =
        (tile_dist.iter().map(|&x| x as usize).sum::<usize>()).saturating_sub(tile_remaining);
    ent[bag_base + 10] = if tile_remaining > 0 {
        (dual_count as f32 / tile_remaining as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // 11: tile bag size / 85
    ent[bag_base + 11] = (tile_remaining as f32 / 85.0).min(1.0);
    // 12: wildlife bag size / 100
    ent[bag_base + 12] = (wl_remaining as f32 / 100.0).min(1.0);
    // 13..18: probability of drawing each wildlife in next 4 draws
    //   ≈ min(1, 4 * count_w / wl_remaining)
    for w in 0..5 {
        let p = if wl_remaining > 0 {
            4.0 * (wl_in_bag[w] as f32) / (wl_remaining as f32)
        } else {
            0.0
        };
        ent[bag_base + 13 + w] = p.clamp(0.0, 1.0);
    }
    // 18: turns until tile-bag exhaustion / 80 (each turn consumes one tile)
    ent[bag_base + 18] = (tile_remaining as f32 / 80.0).clamp(0.0, 1.0);
    // 19: is-bag flag
    ent[bag_base + 19] = 1.0;
    // 20..32: reserved

    // ── Token 8: globals ──
    let glb_base = 8 * AZ_ENTITY_RAW_DIM;
    // 0..3: phase one-hot
    ent[glb_base + phase_index(game)] = 1.0;
    // 3..7: own seat one-hot (4 seats max)
    if player < 4 {
        ent[glb_base + 3 + player] = 1.0;
    }
    // 7..10: num-players one-hot for {2, 3, 4}
    let n = game.num_players;
    if (2..=4).contains(&n) {
        ent[glb_base + 7 + (n - 2)] = 1.0;
    }
    // 10..14: position in current draft round one-hot.
    //   Best-effort: GameState doesn't track round-start seat; leave zero.
    //   Follow-up: add `round_start_player` to GameState and one-hot it here.
    // 14: turn index / 80
    let total_turns = (20 * game.num_players).max(1);
    let played = total_turns.saturating_sub(game.turns_remaining as usize);
    ent[glb_base + 14] = (played as f32 / 80.0).clamp(0.0, 1.0);
    // 15: turns remaining / 80
    ent[glb_base + 15] = (game.turns_remaining as f32 / 80.0).clamp(0.0, 1.0);
    // 16..20: own nature tokens bucketed one-hot {0, 1, 2, 3+}
    let bucket = (game.boards[player].nature_tokens as usize).min(3);
    ent[glb_base + 16 + bucket] = 1.0;
    // 20: turns until next overflow / 20.
    //   Best-effort: if an overflow already exists, value = 0. Otherwise the
    //   exact wait depends on future market draws; leave 0 for now and add a
    //   forecast helper later if useful.
    if overflow_wildlife.is_some() {
        ent[glb_base + 20] = 0.0;
    }
    // 21: is-globals flag
    ent[glb_base + 21] = 1.0;
    // 22..32: reserved

    // ── Token 9: race state ──
    let race_base = 9 * AZ_ENTITY_RAW_DIM;
    for ti in 0..5 {
        let rank = habitat_rank(game, player, ti);
        ent[race_base + ti * 4 + rank] = 1.0;
    }
    // 20..25: per-terrain (my_size − best_opp_size) clamped to [-10, 10], normalized.
    for ti in 0..5 {
        let my = game.boards[player].largest_group[ti] as i32;
        let best_opp = game
            .boards
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != player)
            .map(|(_, b)| b.largest_group[ti] as i32)
            .max()
            .unwrap_or(0);
        let diff = (my - best_opp).clamp(-10, 10);
        ent[race_base + 20 + ti] = (diff + 10) as f32 / 20.0;
    }
    // 25: is-race-state flag
    ent[race_base + 25] = 1.0;
    // 26..32: reserved

    ent
}

// ─────────────────────────────────────────────────────────────────────
// Tensor utilities
// ─────────────────────────────────────────────────────────────────────

fn rand_vec(rng: &mut StdRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Softmax in-place on a slice. Mirrors `softmax()` but avoids the
/// allocation for hot-path callers (cross-attention iterates softmax over
/// hundreds of cell×head rows per forward).
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_l = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_l).exp();
        sum += *v;
    }
    if sum > 0.0 && sum.is_finite() {
        for v in x.iter_mut() {
            *v /= sum;
        }
    } else {
        let n = x.len() as f32;
        for v in x.iter_mut() {
            *v = 1.0 / n;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Layer-norm a token of dim `d` in-place (mean+variance over the dim).
fn layer_norm_inplace(x: &mut [f32], scale: &[f32], bias: &[f32]) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let denom = (var + LN_EPS).sqrt();
    for i in 0..n {
        x[i] = (x[i] - mean) / denom * scale[i] + bias[i];
    }
}

// ─────────────────────────────────────────────────────────────────────
// HexConv: 7-tap (self + 6 hex neighbors) forward pass on 128 cells.
// Weights: w[out_c, in_c, 7] flat; bias: b[out_c].
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HexConv {
    in_c: usize,
    out_c: usize,
    w: Vec<f32>, // out_c * in_c * 7
    b: Vec<f32>, // out_c
}

impl HexConv {
    fn new(in_c: usize, out_c: usize, rng: &mut StdRng) -> Self {
        let fan = (in_c * 7).max(1) as f32;
        let scale = (2.0 / fan).sqrt();
        HexConv {
            in_c,
            out_c,
            w: rand_vec(rng, out_c * in_c * 7, scale),
            b: vec![0.0; out_c],
        }
    }

    /// Forward: input layout x[ic * 128 + cell], output y[oc * 128 + cell].
    ///
    /// Phase 0.8.B: im2col + single SGEMM. Build the `(in_c·7, 128)` gather
    /// buffer from `hex_neighbors_local()`, then call `sgemm_rm` once with
    /// weights reshaped as `(out_c, in_c·7)` (already the native flat
    /// layout — no copy or transpose). Add bias as a broadcast pass.
    ///
    /// The scalar / matmul-portable / accelerate backends are interchangeable
    /// at the `sgemm_rm` façade; behavior is identical modulo f32 fma
    /// accumulation reorder (~5e-5 max-abs-diff against the prior nested-loop
    /// implementation).
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let neighbors = hex_neighbors_local();
        let in_c = self.in_c;
        let out_c = self.out_c;
        let cells = AZ_CELLS_PADDED;

        // im2col: cols[(ic*7 + k) * 128 + cell] = input[ic * 128 + neighbors[cell][k]]
        let k_dim = in_c * 7;
        let mut cols = vec![0.0f32; k_dim * cells];
        for ic in 0..in_c {
            let ib = ic * cells;
            for k in 0..7 {
                let row_base = (ic * 7 + k) * cells;
                for cell in 0..cells {
                    cols[row_base + cell] = input[ib + neighbors[cell][k]];
                }
            }
        }

        // out = W @ cols  where W is (out_c, in_c*7) row-major, cols is (in_c*7, 128).
        let mut out = vec![0.0f32; out_c * cells];
        crate::sgemm::sgemm_rm(out_c, cells, k_dim, 1.0, &self.w, &cols, 0.0, &mut out);

        // Broadcast-add bias per output channel.
        for oc in 0..out_c {
            let bias = self.b[oc];
            let ob = oc * cells;
            for cell in 0..cells {
                out[ob + cell] += bias;
            }
        }

        out
    }
}

#[derive(Clone)]
struct ResHexBlock {
    c1: HexConv,
    c2: HexConv,
}

impl ResHexBlock {
    fn new(channels: usize, rng: &mut StdRng) -> Self {
        ResHexBlock {
            c1: HexConv::new(channels, channels, rng),
            c2: HexConv::new(channels, channels, rng),
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut z1 = self.c1.forward(input);
        relu_inplace(&mut z1);
        let z2 = self.c2.forward(&z1);
        let mut out = vec![0.0f32; z2.len()];
        for i in 0..out.len() {
            out[i] = (z2[i] + input[i]).max(0.0);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────
// Set-Transformer SAB block (pre-norm, multi-head self-attention + FFN).
// Tokens: [N, d_model]. Heads: H, head_dim = d_model / H.
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Sab {
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    // Pre-norm 1 params (LayerNorm).
    ln1_scale: Vec<f32>,
    ln1_bias: Vec<f32>,
    // QKV projection: [3 * d, d] row-major.
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    // Output projection: [d, d].
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    // Pre-norm 2 params.
    ln2_scale: Vec<f32>,
    ln2_bias: Vec<f32>,
    // FFN: [ffn_dim, d] + [d, ffn_dim].
    ffn1_w: Vec<f32>,
    ffn1_b: Vec<f32>,
    ffn2_w: Vec<f32>,
    ffn2_b: Vec<f32>,
}

impl Sab {
    fn new(d: usize, heads: usize, ffn_dim: usize, rng: &mut StdRng) -> Self {
        let head_dim = d / heads;
        let s_attn = (2.0 / d as f32).sqrt();
        let s_ffn = (2.0 / d as f32).sqrt();
        Sab {
            d,
            heads,
            head_dim,
            ffn_dim,
            ln1_scale: vec![1.0; d],
            ln1_bias: vec![0.0; d],
            qkv_w: rand_vec(rng, 3 * d * d, s_attn),
            qkv_b: vec![0.0; 3 * d],
            out_w: rand_vec(rng, d * d, s_attn),
            out_b: vec![0.0; d],
            ln2_scale: vec![1.0; d],
            ln2_bias: vec![0.0; d],
            ffn1_w: rand_vec(rng, ffn_dim * d, s_ffn),
            ffn1_b: vec![0.0; ffn_dim],
            ffn2_w: rand_vec(rng, d * ffn_dim, (2.0 / ffn_dim as f32).sqrt()),
            ffn2_b: vec![0.0; d],
        }
    }

    /// Forward: x[N*d] in row-major order. Returns y[N*d].
    ///
    /// Phase 0.8.B: per-token weight projections (QKV, output, FFN1, FFN2)
    /// are batched through `sgemm_rm_nt` (B-transposed SGEMM). Per-head
    /// attention stays scalar since N=10 tokens makes the matmul tiny and
    /// the SGEMM call overhead would dominate.
    fn forward(&self, x: &[f32], n_tokens: usize) -> Vec<f32> {
        let d = self.d;
        // Step 1: LayerNorm copy of x.
        let mut x_ln = x.to_vec();
        for t in 0..n_tokens {
            let s = t * d;
            layer_norm_inplace(&mut x_ln[s..s + d], &self.ln1_scale, &self.ln1_bias);
        }
        // Step 2: QKV projection. qkv[t, 3d] = qkv_w @ x_ln[t] + qkv_b.
        //   X_ln is (n_tokens, d); qkv_w is (3d, d). Output (n_tokens, 3d).
        //   sgemm computes X_ln @ qkv_w^T (the natural `out = X @ W^T` form).
        let mut qkv = vec![0.0f32; n_tokens * 3 * d];
        crate::sgemm::sgemm_rm_nt(n_tokens, 3 * d, d, 1.0, &x_ln, &self.qkv_w, 0.0, &mut qkv);
        // Bias broadcast.
        for t in 0..n_tokens {
            let qb = t * 3 * d;
            for o in 0..3 * d {
                qkv[qb + o] += self.qkv_b[o];
            }
        }
        // Step 3: Multi-head attention. For each head h:
        //   Q_h = qkv[:, 0*d + h*head_dim : 0*d + (h+1)*head_dim]
        //   K_h = qkv[:, 1*d + ...]
        //   V_h = qkv[:, 2*d + ...]
        //   scores[i, j] = (Q_h[i] · K_h[j]) / sqrt(head_dim)
        //   attn[i, j] = softmax_j(scores)
        //   ctx_h[i] = sum_j attn[i, j] * V_h[j]
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; n_tokens * d];
        for h in 0..self.heads {
            let q_off = 0 * d + h * self.head_dim;
            let k_off = 1 * d + h * self.head_dim;
            let v_off = 2 * d + h * self.head_dim;
            for i in 0..n_tokens {
                // scores for token i against all tokens j.
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..n_tokens {
                    let qb = i * 3 * d + q_off;
                    let kb = j * 3 * d + k_off;
                    let mut s = 0.0;
                    for k in 0..self.head_dim {
                        s += qkv[qb + k] * qkv[kb + k];
                    }
                    scores[j] = s * scale;
                }
                let attn = softmax(&scores);
                let ob = i * d + h * self.head_dim;
                for k in 0..self.head_dim {
                    let mut s = 0.0;
                    for j in 0..n_tokens {
                        let vb = j * 3 * d + v_off;
                        s += attn[j] * qkv[vb + k];
                    }
                    attn_out[ob + k] = s;
                }
            }
        }
        // Step 4: Output projection + residual.
        //   after_attn[t, d] = attn_out[t, d] @ out_w^T[d, d] + out_b + x (residual on x, not x_ln)
        let mut after_attn = vec![0.0f32; n_tokens * d];
        crate::sgemm::sgemm_rm_nt(
            n_tokens,
            d,
            d,
            1.0,
            &attn_out,
            &self.out_w,
            0.0,
            &mut after_attn,
        );
        for t in 0..n_tokens {
            let ob = t * d;
            for o in 0..d {
                after_attn[ob + o] += self.out_b[o] + x[ob + o];
            }
        }
        // Step 5: Pre-norm 2 + FFN + residual.
        let mut x2_ln = after_attn.clone();
        for t in 0..n_tokens {
            let s = t * d;
            layer_norm_inplace(&mut x2_ln[s..s + d], &self.ln2_scale, &self.ln2_bias);
        }
        // FFN1: hidden = relu(x2_ln @ ffn1_w^T + ffn1_b)
        let mut hidden = vec![0.0f32; n_tokens * self.ffn_dim];
        crate::sgemm::sgemm_rm_nt(
            n_tokens,
            self.ffn_dim,
            d,
            1.0,
            &x2_ln,
            &self.ffn1_w,
            0.0,
            &mut hidden,
        );
        for t in 0..n_tokens {
            let hb = t * self.ffn_dim;
            for o in 0..self.ffn_dim {
                hidden[hb + o] = (hidden[hb + o] + self.ffn1_b[o]).max(0.0);
            }
        }
        // FFN2: out = hidden @ ffn2_w^T + ffn2_b + after_attn (residual)
        let mut out = vec![0.0f32; n_tokens * d];
        crate::sgemm::sgemm_rm_nt(
            n_tokens,
            d,
            self.ffn_dim,
            1.0,
            &hidden,
            &self.ffn2_w,
            0.0,
            &mut out,
        );
        for t in 0..n_tokens {
            let ob = t * d;
            for o in 0..d {
                out[ob + o] += self.ffn2_b[o] + after_attn[ob + o];
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cross-Attention: Q from per-cell trunk features → K/V from entity tokens.
// Q dim = entity_dim (after projection from trunk channels).
// Output added back as a per-cell residual into the trunk feature map.
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct CrossAttn {
    d_ent: usize,
    trunk_c: usize,
    heads: usize,
    head_dim: usize,
    // Q projection: trunk channels → entity_dim, per cell. [d_ent, trunk_c]
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    // K, V projections from entity tokens. [d_ent, d_ent] each
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    // Output projection: entity_dim → trunk channels. [trunk_c, d_ent]
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl CrossAttn {
    fn new(trunk_c: usize, d_ent: usize, heads: usize, rng: &mut StdRng) -> Self {
        let head_dim = d_ent / heads;
        let s_q = (2.0 / trunk_c as f32).sqrt();
        let s_kv = (2.0 / d_ent as f32).sqrt();
        let s_out = (2.0 / d_ent as f32).sqrt();
        CrossAttn {
            d_ent,
            trunk_c,
            heads,
            head_dim,
            q_w: rand_vec(rng, d_ent * trunk_c, s_q),
            q_b: vec![0.0; d_ent],
            k_w: rand_vec(rng, d_ent * d_ent, s_kv),
            k_b: vec![0.0; d_ent],
            v_w: rand_vec(rng, d_ent * d_ent, s_kv),
            v_b: vec![0.0; d_ent],
            out_w: rand_vec(rng, trunk_c * d_ent, s_out),
            out_b: vec![0.0; trunk_c],
        }
    }

    /// Forward: trunk[trunk_c * 128] (channel-major), entities[N_tokens * d_ent]
    /// (token-major). Returns updated trunk[trunk_c * 128] = trunk + cross_out.
    ///
    /// Phase 0.8.B: all per-cell scalar loops are replaced by batched SGEMM
    /// over `(128 cells × …)` blocks. Per-head attention is computed for all
    /// cells in one `sgemm_rm_nt` (scores) + `sgemm_rm` (ctx) pair per head.
    fn forward(&self, trunk: &[f32], entities: &[f32], n_tokens: usize) -> Vec<f32> {
        let d = self.d_ent;
        let c = self.trunk_c;
        let cells = AZ_CELLS_PADDED;
        let head_dim = self.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // K projection: k_tok = entities @ k_w^T + k_b, shape (n_tokens, d_ent).
        let mut k_tok = vec![0.0f32; n_tokens * d];
        crate::sgemm::sgemm_rm_nt(n_tokens, d, d, 1.0, entities, &self.k_w, 0.0, &mut k_tok);
        for t in 0..n_tokens {
            let eb = t * d;
            for o in 0..d {
                k_tok[eb + o] += self.k_b[o];
            }
        }
        // V projection: same shape.
        let mut v_tok = vec![0.0f32; n_tokens * d];
        crate::sgemm::sgemm_rm_nt(n_tokens, d, d, 1.0, entities, &self.v_w, 0.0, &mut v_tok);
        for t in 0..n_tokens {
            let eb = t * d;
            for o in 0..d {
                v_tok[eb + o] += self.v_b[o];
            }
        }

        // Transpose trunk to cell-major: trunk_t[cell * c + ic] = trunk[ic * cells + cell].
        // This lets the Q projection, attention, and output proj run on
        // contiguous (cell, channel) rows.
        let mut trunk_t = vec![0.0f32; cells * c];
        for ic in 0..c {
            let ib = ic * cells;
            for cell in 0..cells {
                trunk_t[cell * c + ic] = trunk[ib + cell];
            }
        }

        // Q projection across ALL cells: q_cell = trunk_t @ q_w^T + q_b,
        //   shape (cells, d_ent).
        let mut q_cell = vec![0.0f32; cells * d];
        crate::sgemm::sgemm_rm_nt(cells, d, c, 1.0, &trunk_t, &self.q_w, 0.0, &mut q_cell);
        for cell in 0..cells {
            let cb = cell * d;
            for o in 0..d {
                q_cell[cb + o] += self.q_b[o];
            }
        }

        // Per-head attention, batched across cells. For each head h:
        //   Q_h: (cells, head_dim) slice of q_cell
        //   K_h: (n_tokens, head_dim) slice of k_tok
        //   V_h: (n_tokens, head_dim) slice of v_tok
        //   scores = (Q_h @ K_h^T) * scale → (cells, n_tokens), softmax row-wise
        //   ctx_h = scores @ V_h → (cells, head_dim), written into ctx[cell, h_off:]
        let mut ctx = vec![0.0f32; cells * d];
        let mut q_h = vec![0.0f32; cells * head_dim];
        let mut k_h = vec![0.0f32; n_tokens * head_dim];
        let mut v_h = vec![0.0f32; n_tokens * head_dim];
        let mut scores = vec![0.0f32; cells * n_tokens];
        let mut ctx_h = vec![0.0f32; cells * head_dim];
        for h in 0..self.heads {
            let h_off = h * head_dim;
            // Gather per-head slices into contiguous buffers.
            for cell in 0..cells {
                let src = cell * d + h_off;
                let dst = cell * head_dim;
                q_h[dst..dst + head_dim].copy_from_slice(&q_cell[src..src + head_dim]);
            }
            for j in 0..n_tokens {
                let src = j * d + h_off;
                let dst = j * head_dim;
                k_h[dst..dst + head_dim].copy_from_slice(&k_tok[src..src + head_dim]);
                v_h[dst..dst + head_dim].copy_from_slice(&v_tok[src..src + head_dim]);
            }
            // scores = scale * Q_h @ K_h^T.
            crate::sgemm::sgemm_rm_nt(
                cells,
                n_tokens,
                head_dim,
                scale,
                &q_h,
                &k_h,
                0.0,
                &mut scores,
            );
            // Softmax each row in-place.
            for cell in 0..cells {
                let row = &mut scores[cell * n_tokens..(cell + 1) * n_tokens];
                softmax_inplace(row);
            }
            // ctx_h = scores @ V_h (no transpose).
            crate::sgemm::sgemm_rm(
                cells, head_dim, n_tokens, 1.0, &scores, &v_h, 0.0, &mut ctx_h,
            );
            // Scatter ctx_h back into the per-head slice of ctx.
            for cell in 0..cells {
                let src = cell * head_dim;
                let dst = cell * d + h_off;
                ctx[dst..dst + head_dim].copy_from_slice(&ctx_h[src..src + head_dim]);
            }
        }

        // Output projection: delta_cellmajor = ctx @ out_w^T + out_b
        //   shape (cells, trunk_c).
        let mut delta_cellmajor = vec![0.0f32; cells * c];
        crate::sgemm::sgemm_rm_nt(
            cells,
            c,
            d,
            1.0,
            &ctx,
            &self.out_w,
            0.0,
            &mut delta_cellmajor,
        );
        for cell in 0..cells {
            let cb = cell * c;
            for oc in 0..c {
                delta_cellmajor[cb + oc] += self.out_b[oc];
            }
        }

        // Transpose delta back to channel-major, add to trunk, ReLU.
        let mut out = vec![0.0f32; trunk.len()];
        for ic in 0..c {
            let ob = ic * cells;
            for cell in 0..cells {
                let delta = delta_cellmajor[cell * c + ic];
                out[ob + cell] = (trunk[ob + cell] + delta).max(0.0);
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────
// Multi-Head Value with phase gate.
//
// Pool trunk over cells → MLP 96 → 128 → 16 sub-heads → sigmoid each.
// Phase one-hot [3] × blend_matrix [3 × 16] → softmax over 16 → blend weights.
// v̂ = sum(blend_weights × subhead_predictions) → already in [0,1].
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct MultiHeadValue {
    channels: usize,
    hidden: usize,
    subheads: usize,
    w1: Vec<f32>, // hidden × channels
    b1: Vec<f32>, // hidden
    w2: Vec<f32>, // subheads × hidden
    b2: Vec<f32>, // subheads
    // Blend matrix: 3 phases × subheads. Softmaxed across subheads per phase.
    blend: Vec<f32>, // 3 × subheads
}

impl MultiHeadValue {
    fn new(channels: usize, hidden: usize, subheads: usize, rng: &mut StdRng) -> Self {
        let s1 = (2.0 / channels as f32).sqrt();
        let s2 = (2.0 / hidden as f32).sqrt();
        MultiHeadValue {
            channels,
            hidden,
            subheads,
            w1: rand_vec(rng, hidden * channels, s1),
            b1: vec![0.0; hidden],
            w2: rand_vec(rng, subheads * hidden, s2),
            b2: vec![0.0; subheads],
            // Initial blend: uniform — softmax(0) = 1/16 per phase.
            blend: vec![0.0; AZ_VALUE_PHASES * subheads],
        }
    }

    /// Compute (scalar_value, sub_predictions[16]) from pooled trunk + phase one-hot.
    fn forward(&self, pooled: &[f32], phase: [f32; AZ_VALUE_PHASES]) -> (f32, Vec<f32>) {
        // Hidden layer.
        let mut h = vec![0.0f32; self.hidden];
        for o in 0..self.hidden {
            let mut s = self.b1[o];
            let wb = o * self.channels;
            for i in 0..self.channels {
                s += self.w1[wb + i] * pooled[i];
            }
            h[o] = s.max(0.0);
        }
        // 16 sub-head logits → sigmoid.
        let mut subs = vec![0.0f32; self.subheads];
        for o in 0..self.subheads {
            let mut s = self.b2[o];
            let wb = o * self.hidden;
            for i in 0..self.hidden {
                s += self.w2[wb + i] * h[i];
            }
            subs[o] = sigmoid(s);
        }
        // Blend weights: phase[3] @ blend[3,16] = logits[16], softmax over subheads,
        // then weighted by phase mass. With a one-hot phase this reduces to
        // softmax(blend[phase, :]). For soft phases we do the linear combo.
        let mut weights = vec![0.0f32; self.subheads];
        for p in 0..AZ_VALUE_PHASES {
            if phase[p] == 0.0 {
                continue;
            }
            let pb = p * self.subheads;
            let phase_logits = &self.blend[pb..pb + self.subheads];
            let phase_weights = softmax(phase_logits);
            for o in 0..self.subheads {
                weights[o] += phase[p] * phase_weights[o];
            }
        }
        // Normalize blend weights so they sum to 1.
        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            for w in weights.iter_mut() {
                *w /= total;
            }
        } else {
            for w in weights.iter_mut() {
                *w = 1.0 / self.subheads as f32;
            }
        }
        // Scalar value = weighted sum of sub predictions (each in [0,1]).
        let mut v = 0.0;
        for o in 0..self.subheads {
            v += weights[o] * subs[o];
        }
        (v.clamp(0.0, 1.0), subs)
    }
}

/// Map current turn / total turns → 3-way phase distribution (early/mid/late).
/// Hard one-hot; MLX side does the same partitioning.
pub fn phase_one_hot(game: &GameState) -> [f32; AZ_VALUE_PHASES] {
    let total = (20 * game.num_players).max(1) as f32;
    let played = (total - game.turns_remaining as f32).clamp(0.0, total);
    let frac = played / total;
    let mut out = [0.0; AZ_VALUE_PHASES];
    if frac < 1.0 / 3.0 {
        out[0] = 1.0;
    } else if frac < 2.0 / 3.0 {
        out[1] = 1.0;
    } else {
        out[2] = 1.0;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// AlphaZeroNetV2: end-to-end network.
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AlphaZeroNetV2 {
    cfg: AlphaZeroV2Config,
    // Main board trunk
    stem: HexConv,
    blocks: Vec<ResHexBlock>,
    // Shared opponent trunk — applied to each of 3 opp boards with the same
    // weights. Pooled output (32 dims) replaces opp entity tokens 4..7 before
    // the entity up-projection.
    opp_stem: HexConv,
    opp_blocks: Vec<ResHexBlock>,
    // Entity stream
    ent_up_w: Vec<f32>, // [entity_dim, raw_dim]
    ent_up_b: Vec<f32>, // [entity_dim]
    sabs: Vec<Sab>,
    // Cross-attention fusion
    cross: CrossAttn,
    // Policy heads
    policy_tile_w: Vec<f32>, // [channels]
    policy_tile_b: f32,
    policy_wildlife_w: Vec<f32>, // [channels]
    policy_wildlife_b: f32,
    policy_market_w: Vec<f32>, // [4, channels + entity_dim]
    policy_market_b: [f32; 4],
    policy_wildlife_market_w: Vec<f32>, // [4, channels + entity_dim]
    policy_wildlife_market_b: [f32; 4],
    policy_skip_w: Vec<f32>, // [channels]
    policy_skip_b: f32,
    // Value head
    value: MultiHeadValue,
}

impl AlphaZeroNetV2 {
    pub fn new(cfg: AlphaZeroV2Config, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = cfg.channels;
        let d = cfg.entity_dim;
        let stem = HexConv::new(AZ_INPUT_CHANNELS_V2, c, &mut rng);
        let blocks = (0..cfg.blocks)
            .map(|_| ResHexBlock::new(c, &mut rng))
            .collect();
        // Shared opp trunk. Output channels must equal AZ_ENTITY_RAW_DIM so the
        // pooled vector slots directly into the entity stream's opp tokens.
        assert_eq!(
            AZ_OPP_TRUNK_CHANNELS, AZ_ENTITY_RAW_DIM,
            "opp trunk output channels must match entity raw dim ({} vs {})",
            AZ_OPP_TRUNK_CHANNELS, AZ_ENTITY_RAW_DIM
        );
        let opp_stem = HexConv::new(AZ_INPUT_CHANNELS_V2, AZ_OPP_TRUNK_CHANNELS, &mut rng);
        let opp_blocks = (0..AZ_OPP_TRUNK_BLOCKS)
            .map(|_| ResHexBlock::new(AZ_OPP_TRUNK_CHANNELS, &mut rng))
            .collect();
        let ent_up_scale = (2.0 / AZ_ENTITY_RAW_DIM as f32).sqrt();
        let sabs = (0..cfg.sab_blocks)
            .map(|_| Sab::new(d, cfg.heads, AZ_SAB_FFN_DIM, &mut rng))
            .collect();
        let cross = CrossAttn::new(c, d, AZ_CROSS_HEADS, &mut rng);
        let head_scale = (2.0 / c as f32).sqrt();
        let market_scale = (2.0 / (c + d) as f32).sqrt();
        AlphaZeroNetV2 {
            cfg,
            stem,
            blocks,
            opp_stem,
            opp_blocks,
            ent_up_w: rand_vec(&mut rng, d * AZ_ENTITY_RAW_DIM, ent_up_scale),
            ent_up_b: vec![0.0; d],
            sabs,
            cross,
            policy_tile_w: rand_vec(&mut rng, c, head_scale),
            policy_tile_b: 0.0,
            policy_wildlife_w: rand_vec(&mut rng, c, head_scale),
            policy_wildlife_b: 0.0,
            policy_market_w: rand_vec(&mut rng, 4 * (c + d), market_scale),
            policy_market_b: [0.0; 4],
            policy_wildlife_market_w: rand_vec(&mut rng, 4 * (c + d), market_scale),
            policy_wildlife_market_b: [0.0; 4],
            policy_skip_w: rand_vec(&mut rng, c, head_scale),
            policy_skip_b: 0.0,
            value: MultiHeadValue::new(c, cfg.value_hidden, cfg.value_subheads, &mut rng),
        }
    }

    /// Run the shared opp trunk on one opponent board, then average-pool over
    /// real cells (excluding pad) to a 32-dim vector.
    fn encode_opp(&self, opp_input: &[f32]) -> Vec<f32> {
        let stem_z = self.opp_stem.forward(opp_input);
        let mut x = stem_z;
        relu_inplace(&mut x);
        for blk in &self.opp_blocks {
            x = blk.forward(&x);
        }
        // Average-pool over the 127 real cells. Pad cell (index 127) excluded.
        let mut pooled = vec![0.0f32; AZ_OPP_TRUNK_CHANNELS];
        for ch in 0..AZ_OPP_TRUNK_CHANNELS {
            let mut s = 0.0;
            for cell in 0..AZ_LOCAL_CELLS {
                s += x[ch * AZ_CELLS_PADDED + cell];
            }
            pooled[ch] = s / AZ_LOCAL_CELLS as f32;
        }
        pooled
    }

    pub fn config(&self) -> AlphaZeroV2Config {
        self.cfg
    }

    /// Project raw entity tokens to entity_dim.
    fn project_entities(&self, raw: &[f32]) -> Vec<f32> {
        let d = self.cfg.entity_dim;
        let mut out = vec![0.0f32; AZ_ENTITY_TOKENS * d];
        for t in 0..AZ_ENTITY_TOKENS {
            let rb = t * AZ_ENTITY_RAW_DIM;
            let ob = t * d;
            for o in 0..d {
                let mut s = self.ent_up_b[o];
                let wb = o * AZ_ENTITY_RAW_DIM;
                for i in 0..AZ_ENTITY_RAW_DIM {
                    s += self.ent_up_w[wb + i] * raw[rb + i];
                }
                out[ob + o] = s;
            }
        }
        out
    }

    /// Full forward returning a ForwardCacheV2 with everything needed for
    /// policy and value extraction.
    ///
    /// `opp_inputs` must have exactly `AZ_MAX_OPPONENTS` entries, each of
    /// length `AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED`. Zero-padded boards
    /// for missing opponents are handled by the caller (`encode_game_local`).
    pub fn forward(
        &self,
        input: &[f32],
        opp_inputs: &[Vec<f32>],
        entities_raw: &[f32],
        phase: [f32; AZ_VALUE_PHASES],
    ) -> ForwardCacheV2 {
        debug_assert_eq!(opp_inputs.len(), AZ_MAX_OPPONENTS);
        // Main trunk.
        let stem_z = self.stem.forward(input);
        let mut x = stem_z.clone();
        relu_inplace(&mut x);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        // Shared opp trunk: produce a 32-dim vector per opponent.
        let mut opp_pooled: [Vec<f32>; AZ_MAX_OPPONENTS] = Default::default();
        for (i, opp) in opp_inputs.iter().enumerate() {
            opp_pooled[i] = self.encode_opp(opp);
        }
        // Build the raw entity tensor with opp slots overwritten by trunk output.
        let mut entities_with_opps = entities_raw.to_vec();
        for i in 0..AZ_MAX_OPPONENTS {
            let token_idx = 4 + i;
            let base = token_idx * AZ_ENTITY_RAW_DIM;
            // First 32 dims of each opp token come from the pooled trunk
            // vector. Remaining dims (if any) stay at the raw value (which is
            // zero from the encoder).
            for k in 0..AZ_OPP_TRUNK_CHANNELS {
                entities_with_opps[base + k] = opp_pooled[i][k];
            }
        }
        // Entity stream.
        let mut ent = self.project_entities(&entities_with_opps);
        for sab in &self.sabs {
            ent = sab.forward(&ent, AZ_ENTITY_TOKENS);
        }
        // Cross-attention fusion.
        let fused = self.cross.forward(&x, &ent, AZ_ENTITY_TOKENS);
        // Pool over real cells (exclude pad).
        let c = self.cfg.channels;
        let mut pooled = vec![0.0f32; c];
        for ch in 0..c {
            let mut s = 0.0;
            for cell in 0..AZ_LOCAL_CELLS {
                s += fused[ch * AZ_CELLS_PADDED + cell];
            }
            pooled[ch] = s / AZ_LOCAL_CELLS as f32;
        }
        // Value head.
        let (value, subs) = self.value.forward(&pooled, phase);
        // Policy logits per cell + market/skip from pooled.
        let mut tile_logits = vec![0.0f32; AZ_CELLS_PADDED];
        let mut wildlife_logits = vec![0.0f32; AZ_CELLS_PADDED];
        for cell in 0..AZ_CELLS_PADDED {
            let mut t = self.policy_tile_b;
            let mut w = self.policy_wildlife_b;
            for ch in 0..c {
                let v = fused[ch * AZ_CELLS_PADDED + cell];
                t += self.policy_tile_w[ch] * v;
                w += self.policy_wildlife_w[ch] * v;
            }
            tile_logits[cell] = t;
            wildlife_logits[cell] = w;
        }
        // Market and wildlife-market logits use pooled + per-slot market token concat.
        // The 4 market tokens are indices 0..4 in the entity stream (post-SAB).
        let d = self.cfg.entity_dim;
        let mut market_logits = [0.0f32; 4];
        let mut wildlife_market_logits = [0.0f32; 4];
        for slot in 0..4 {
            let mut concat = vec![0.0f32; c + d];
            concat[..c].copy_from_slice(&pooled);
            let eb = slot * d;
            concat[c..].copy_from_slice(&ent[eb..eb + d]);
            let mut m = self.policy_market_b[slot];
            let mut wm = self.policy_wildlife_market_b[slot];
            let wb = slot * (c + d);
            for i in 0..(c + d) {
                m += self.policy_market_w[wb + i] * concat[i];
                wm += self.policy_wildlife_market_w[wb + i] * concat[i];
            }
            market_logits[slot] = m;
            wildlife_market_logits[slot] = wm;
        }
        // Skip logit.
        let mut skip_logit = self.policy_skip_b;
        for ch in 0..c {
            skip_logit += self.policy_skip_w[ch] * pooled[ch];
        }
        ForwardCacheV2 {
            value,
            sub_predictions: subs,
            tile_logits,
            wildlife_logits,
            market_logits,
            wildlife_market_logits,
            skip_logit,
        }
    }

    /// Forward K independent positions. Stage 0.8.C entrypoint for
    /// batched-leaf MCTS. This stub calls `forward` K times; a follow-up
    /// (Stage 0.8.C.2) replaces this with batched SGEMMs across the trunk
    /// and entity stream so the per-leaf cost amortizes over K.
    ///
    /// The shape contract (Vec<ForwardCacheV2> of length K, same order as
    /// `inputs`) is the stable surface so the SGEMM follow-up is drop-in.
    pub fn forward_batch(
        &self,
        inputs: &[Vec<f32>],
        opp_inputs_batch: &[Vec<Vec<f32>>],
        ents: &[Vec<f32>],
        phases: &[[f32; AZ_VALUE_PHASES]],
    ) -> Vec<ForwardCacheV2> {
        debug_assert_eq!(inputs.len(), opp_inputs_batch.len());
        debug_assert_eq!(inputs.len(), ents.len());
        debug_assert_eq!(inputs.len(), phases.len());
        inputs
            .iter()
            .zip(opp_inputs_batch.iter())
            .zip(ents.iter())
            .zip(phases.iter())
            .map(|(((input, opp), ent), phase)| self.forward(input, opp, ent, *phase))
            .collect()
    }

    pub fn evaluate(&self, game: &GameState, candidates: &[ScoredMove]) -> (f32, Vec<f32>) {
        let (input, opp_inputs, ent) = encode_game_local(game);
        let phase = phase_one_hot(game);
        let cache = self.forward(&input, &opp_inputs, &ent, phase);
        let logits = candidate_logits_v2(&cache, candidates);
        (cache.value, softmax(&logits))
    }

    /// AZR2 serialization.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(AZ_MAGIC_V2)?;
        // Header.
        write_u32(&mut f, self.cfg.channels as u32)?;
        write_u32(&mut f, self.cfg.blocks as u32)?;
        write_u32(&mut f, self.cfg.entity_dim as u32)?;
        write_u32(&mut f, self.cfg.sab_blocks as u32)?;
        write_u32(&mut f, self.cfg.heads as u32)?;
        write_u32(&mut f, self.cfg.value_hidden as u32)?;
        write_u32(&mut f, self.cfg.value_subheads as u32)?;
        write_u32(&mut f, self.cfg.max_candidates as u32)?;
        write_f32(&mut f, self.cfg.c_puct)?;
        // Opp-trunk hyperparameters (constants for now, but recorded explicitly
        // so a future load can reject mismatched checkpoints).
        write_u32(&mut f, AZ_OPP_TRUNK_CHANNELS as u32)?;
        write_u32(&mut f, AZ_OPP_TRUNK_BLOCKS as u32)?;
        write_u32(&mut f, AZ_MAX_OPPONENTS as u32)?;
        // Main trunk.
        write_conv(&mut f, &self.stem)?;
        for b in &self.blocks {
            write_conv(&mut f, &b.c1)?;
            write_conv(&mut f, &b.c2)?;
        }
        // Shared opponent trunk.
        write_conv(&mut f, &self.opp_stem)?;
        for b in &self.opp_blocks {
            write_conv(&mut f, &b.c1)?;
            write_conv(&mut f, &b.c2)?;
        }
        // Entity up-projection.
        write_vec(&mut f, &self.ent_up_w)?;
        write_vec(&mut f, &self.ent_up_b)?;
        // SAB blocks.
        for sab in &self.sabs {
            write_sab(&mut f, sab)?;
        }
        // Cross-attention.
        write_cross(&mut f, &self.cross)?;
        // Policy heads.
        write_vec(&mut f, &self.policy_tile_w)?;
        write_f32(&mut f, self.policy_tile_b)?;
        write_vec(&mut f, &self.policy_wildlife_w)?;
        write_f32(&mut f, self.policy_wildlife_b)?;
        write_vec(&mut f, &self.policy_market_w)?;
        for &v in &self.policy_market_b {
            write_f32(&mut f, v)?;
        }
        write_vec(&mut f, &self.policy_wildlife_market_w)?;
        for &v in &self.policy_wildlife_market_b {
            write_f32(&mut f, v)?;
        }
        write_vec(&mut f, &self.policy_skip_w)?;
        write_f32(&mut f, self.policy_skip_b)?;
        // Value head.
        write_vec(&mut f, &self.value.w1)?;
        write_vec(&mut f, &self.value.b1)?;
        write_vec(&mut f, &self.value.w2)?;
        write_vec(&mut f, &self.value.b2)?;
        write_vec(&mut f, &self.value.blend)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != AZ_MAGIC_V2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad AZR2 magic",
            ));
        }
        let cfg = AlphaZeroV2Config {
            channels: read_u32(&mut f)? as usize,
            blocks: read_u32(&mut f)? as usize,
            entity_dim: read_u32(&mut f)? as usize,
            sab_blocks: read_u32(&mut f)? as usize,
            heads: read_u32(&mut f)? as usize,
            value_hidden: read_u32(&mut f)? as usize,
            value_subheads: read_u32(&mut f)? as usize,
            max_candidates: read_u32(&mut f)? as usize,
            c_puct: read_f32(&mut f)?,
        };
        let opp_channels = read_u32(&mut f)? as usize;
        let opp_blocks_n = read_u32(&mut f)? as usize;
        let max_opp = read_u32(&mut f)? as usize;
        if opp_channels != AZ_OPP_TRUNK_CHANNELS
            || opp_blocks_n != AZ_OPP_TRUNK_BLOCKS
            || max_opp != AZ_MAX_OPPONENTS
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "AZR2 opp-trunk header mismatch: got channels={} blocks={} max_opp={}, \
                     expected {}/{}/{}",
                    opp_channels,
                    opp_blocks_n,
                    max_opp,
                    AZ_OPP_TRUNK_CHANNELS,
                    AZ_OPP_TRUNK_BLOCKS,
                    AZ_MAX_OPPONENTS,
                ),
            ));
        }
        let stem = read_conv(&mut f, AZ_INPUT_CHANNELS_V2, cfg.channels)?;
        let mut blocks = Vec::with_capacity(cfg.blocks);
        for _ in 0..cfg.blocks {
            blocks.push(ResHexBlock {
                c1: read_conv(&mut f, cfg.channels, cfg.channels)?,
                c2: read_conv(&mut f, cfg.channels, cfg.channels)?,
            });
        }
        let opp_stem = read_conv(&mut f, AZ_INPUT_CHANNELS_V2, AZ_OPP_TRUNK_CHANNELS)?;
        let mut opp_blocks = Vec::with_capacity(AZ_OPP_TRUNK_BLOCKS);
        for _ in 0..AZ_OPP_TRUNK_BLOCKS {
            opp_blocks.push(ResHexBlock {
                c1: read_conv(&mut f, AZ_OPP_TRUNK_CHANNELS, AZ_OPP_TRUNK_CHANNELS)?,
                c2: read_conv(&mut f, AZ_OPP_TRUNK_CHANNELS, AZ_OPP_TRUNK_CHANNELS)?,
            });
        }
        let ent_up_w = read_vec(&mut f)?;
        let ent_up_b = read_vec(&mut f)?;
        let mut sabs = Vec::with_capacity(cfg.sab_blocks);
        for _ in 0..cfg.sab_blocks {
            sabs.push(read_sab(&mut f, cfg.entity_dim, cfg.heads, AZ_SAB_FFN_DIM)?);
        }
        let cross = read_cross(&mut f, cfg.channels, cfg.entity_dim, AZ_CROSS_HEADS)?;
        let policy_tile_w = read_vec(&mut f)?;
        let policy_tile_b = read_f32(&mut f)?;
        let policy_wildlife_w = read_vec(&mut f)?;
        let policy_wildlife_b = read_f32(&mut f)?;
        let policy_market_w = read_vec(&mut f)?;
        let mut policy_market_b = [0.0; 4];
        for v in policy_market_b.iter_mut() {
            *v = read_f32(&mut f)?;
        }
        let policy_wildlife_market_w = read_vec(&mut f)?;
        let mut policy_wildlife_market_b = [0.0; 4];
        for v in policy_wildlife_market_b.iter_mut() {
            *v = read_f32(&mut f)?;
        }
        let policy_skip_w = read_vec(&mut f)?;
        let policy_skip_b = read_f32(&mut f)?;
        let value_w1 = read_vec(&mut f)?;
        let value_b1 = read_vec(&mut f)?;
        let value_w2 = read_vec(&mut f)?;
        let value_b2 = read_vec(&mut f)?;
        let value_blend = read_vec(&mut f)?;
        let value = MultiHeadValue {
            channels: cfg.channels,
            hidden: cfg.value_hidden,
            subheads: cfg.value_subheads,
            w1: value_w1,
            b1: value_b1,
            w2: value_w2,
            b2: value_b2,
            blend: value_blend,
        };
        Ok(AlphaZeroNetV2 {
            cfg,
            stem,
            blocks,
            opp_stem,
            opp_blocks,
            ent_up_w,
            ent_up_b,
            sabs,
            cross,
            policy_tile_w,
            policy_tile_b,
            policy_wildlife_w,
            policy_wildlife_b,
            policy_market_w,
            policy_market_b,
            policy_wildlife_market_w,
            policy_wildlife_market_b,
            policy_skip_w,
            policy_skip_b,
            value,
        })
    }
}

pub struct ForwardCacheV2 {
    pub value: f32,
    pub sub_predictions: Vec<f32>,
    pub tile_logits: Vec<f32>,     // 128
    pub wildlife_logits: Vec<f32>, // 128
    pub market_logits: [f32; 4],
    pub wildlife_market_logits: [f32; 4],
    pub skip_logit: f32,
}

// ─────────────────────────────────────────────────────────────────────
// Candidate logit assembly (factorized policy).
// ─────────────────────────────────────────────────────────────────────

#[inline]
fn move_tile_index_local(mv: &ScoredMove) -> usize {
    match HexCoord::new(mv.tile_q, mv.tile_r).to_index() {
        Some(gi) => {
            let li = global_to_local(gi);
            if li >= 0 {
                li as usize
            } else {
                AZ_PAD_INDEX
            }
        }
        None => AZ_PAD_INDEX,
    }
}

#[inline]
fn move_wildlife_index_local(mv: &ScoredMove) -> Option<usize> {
    match (mv.wildlife_q, mv.wildlife_r) {
        (Some(q), Some(r)) => match HexCoord::new(q, r).to_index() {
            Some(gi) => {
                let li = global_to_local(gi);
                if li >= 0 {
                    Some(li as usize)
                } else {
                    Some(AZ_PAD_INDEX)
                }
            }
            None => Some(AZ_PAD_INDEX),
        },
        _ => None,
    }
}

pub fn candidate_logits_v2(cache: &ForwardCacheV2, candidates: &[ScoredMove]) -> Vec<f32> {
    candidates
        .iter()
        .map(|mv| {
            let mut logit = 0.0;
            logit += cache.tile_logits[move_tile_index_local(mv)];
            match move_wildlife_index_local(mv) {
                Some(li) => logit += cache.wildlife_logits[li],
                None => logit += cache.skip_logit,
            }
            if mv.market_index < 4 {
                logit += cache.market_logits[mv.market_index];
            }
            let wslot = mv.wildlife_market_index.unwrap_or(mv.market_index);
            if wslot < 4 {
                logit += cache.wildlife_market_logits[wslot];
            }
            logit
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// PUCT search (mirror of v1)
// ─────────────────────────────────────────────────────────────────────

pub fn candidate_moves_v2(game: &GameState, max_candidates: usize) -> Vec<ScoredMove> {
    let mut cands = crate::mce::default_greedy_mce_candidates(game);
    let k = max_candidates.max(1);
    if cands.len() > k {
        cands.select_nth_unstable_by(k, candidate_rank_cmp);
        cands.truncate(k);
    }
    cands.sort_by(candidate_rank_cmp);
    cands
}

fn candidate_rank_cmp(a: &ScoredMove, b: &ScoredMove) -> Ordering {
    b.eval
        .cmp(&a.eval)
        .then_with(|| a.market_index.cmp(&b.market_index))
        .then_with(|| a.wildlife_market_index.cmp(&b.wildlife_market_index))
        .then_with(|| a.tile_q.cmp(&b.tile_q))
        .then_with(|| a.tile_r.cmp(&b.tile_r))
        .then_with(|| a.rotation.cmp(&b.rotation))
        .then_with(|| a.wildlife_q.cmp(&b.wildlife_q))
        .then_with(|| a.wildlife_r.cmp(&b.wildlife_r))
}

#[derive(Clone)]
pub struct SearchResultV2 {
    pub selected: ScoredMove,
    pub candidates: Vec<ScoredMove>,
    pub visit_policy: Vec<f32>,
    pub visits: Vec<u32>,
}

struct Edge {
    action: ScoredMove,
    prior: f32,
    visits: u32,
    value_sum: f32,
    // Stage 0.8.C: virtual-loss counter for batched-leaf descent.
    // Decremented to 0 on backup; serial path leaves this at 0 always so
    // `select_puct_v2` math is bit-identical to the pre-Stage-C formulation.
    virtual_loss: u32,
    child: Option<Box<Node>>,
}

struct Node {
    expanded: bool,
    visits: u32,
    edges: Vec<Edge>,
}

impl Node {
    fn new() -> Self {
        Node {
            expanded: false,
            visits: 0,
            edges: Vec::new(),
        }
    }
}

pub fn az_search_v2(
    game: &GameState,
    net: &AlphaZeroNetV2,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResultV2> {
    if should_use_root_parallel_v2(simulations) {
        let workers = az_parallel_workers_v2(simulations);
        return az_search_v2_root_parallel(game, net, simulations, workers, temperature, rng);
    }
    let batch_k = az_batch_k();
    if batch_k > 1 && simulations >= 2 {
        return az_search_v2_batched(game, net, simulations, batch_k, temperature, rng);
    }
    az_search_v2_serial(game, net, simulations, temperature, rng)
}

fn az_search_v2_serial(
    game: &GameState,
    net: &AlphaZeroNetV2,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResultV2> {
    let root_player = game.current_player;
    let mut root = Node::new();
    for _ in 0..simulations.max(1) {
        let mut g = game.clone();
        g.shuffle_bags(rng);
        simulate_puct_v2(&mut root, &g, root_player, net);
    }
    if root.edges.is_empty() {
        return None;
    }
    let visits: Vec<u32> = root.edges.iter().map(|e| e.visits).collect();
    let total: u32 = visits.iter().sum();
    let visit_policy: Vec<f32> = if total == 0 {
        vec![1.0 / visits.len() as f32; visits.len()]
    } else {
        visits.iter().map(|&v| v as f32 / total as f32).collect()
    };
    let selected_idx = select_from_visits(&visits, temperature, rng);
    Some(SearchResultV2 {
        selected: root.edges[selected_idx].action,
        candidates: root.edges.iter().map(|e| e.action).collect(),
        visit_policy,
        visits,
    })
}

/// Whether to dispatch to root-parallel. Gated by `CASCADIA_AZ_PARALLEL=1`,
/// matching the v1 env-var contract so existing CLI flags keep working.
fn should_use_root_parallel_v2(simulations: usize) -> bool {
    if simulations < 2 {
        return false;
    }
    std::env::var("CASCADIA_AZ_PARALLEL")
        .ok()
        .map(|s| !s.is_empty() && s != "0" && s.to_ascii_lowercase() != "false")
        .unwrap_or(false)
}

fn az_parallel_workers_v2(simulations: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let requested = std::env::var("CASCADIA_AZ_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(available);
    let min_sims_per_worker = std::env::var("CASCADIA_AZ_MIN_SIMS_PER_THREAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize)
        .max(1);
    let useful_workers = (simulations / min_sims_per_worker).max(1);
    requested
        .max(1)
        .min(available)
        .min(useful_workers)
        .min(simulations.max(1))
}

/// Root-parallel PUCT for v2. Each worker runs an independent tree on a clone
/// of the game state; visits are aggregated at the root. Net is shared via
/// scoped thread borrow — no deep clone of the ~985K-param model per worker.
///
/// `workers` is parameterized (rather than read directly from env) so tests can
/// dispatch deterministically. The public `az_search_v2` entry computes it via
/// `az_parallel_workers_v2`.
fn az_search_v2_root_parallel(
    game: &GameState,
    net: &AlphaZeroNetV2,
    simulations: usize,
    workers: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResultV2> {
    if workers <= 1 {
        return az_search_v2_serial(game, net, simulations, temperature, rng);
    }

    let root_candidates = candidate_moves_v2(game, net.config().max_candidates);
    if root_candidates.is_empty() {
        return None;
    }
    let base = simulations / workers;
    let rem = simulations % workers;
    let mut seeds = Vec::with_capacity(workers);
    for _ in 0..workers {
        seeds.push(rng.gen::<u64>());
    }

    // Scoped threads let workers borrow `net` directly — no clone, no Arc.
    let visits: Vec<u32> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for (worker, seed) in seeds.into_iter().enumerate() {
            let sims = base + usize::from(worker < rem);
            if sims == 0 {
                continue;
            }
            let g = game.clone();
            // `net` is borrowed for the lifetime of the scope; no clone.
            handles.push(s.spawn(move || {
                let mut wrng = StdRng::seed_from_u64(seed);
                // Temperature is 0 for workers — we sample the final pick at
                // the merged-visits stage below using the caller's `rng`.
                az_search_v2_serial(&g, net, sims, 0.0, &mut wrng)
            }));
        }

        let mut visits = vec![0u32; root_candidates.len()];
        for handle in handles {
            if let Ok(Some(result)) = handle.join() {
                // Common case: the worker saw the same root-candidate ordering.
                // Otherwise fall back to per-action lookup.
                if result.candidates.len() == root_candidates.len()
                    && result
                        .candidates
                        .iter()
                        .zip(root_candidates.iter())
                        .all(|(a, b)| same_move(a, b))
                {
                    for (dst, src) in visits.iter_mut().zip(result.visits.iter()) {
                        *dst += *src;
                    }
                } else {
                    for (candidate, src) in result.candidates.iter().zip(result.visits.iter()) {
                        if let Some(idx) = root_candidates
                            .iter()
                            .position(|root| same_move(root, candidate))
                        {
                            visits[idx] += *src;
                        }
                    }
                }
            }
        }
        visits
    });

    if visits.iter().all(|&v| v == 0) {
        return az_search_v2_serial(game, net, simulations, temperature, rng);
    }

    let total: u32 = visits.iter().sum();
    let visit_policy = visits.iter().map(|&v| v as f32 / total as f32).collect();
    let selected_idx = select_from_visits(&visits, temperature, rng);
    Some(SearchResultV2 {
        selected: root_candidates[selected_idx],
        candidates: root_candidates,
        visit_policy,
        visits,
    })
}

fn simulate_puct_v2(
    node: &mut Node,
    game: &GameState,
    root_player: usize,
    net: &AlphaZeroNetV2,
) -> f32 {
    if game.is_game_over() {
        node.visits += 1;
        return score_with_bonus(game, root_player) / AZ_VALUE_SCALE;
    }
    if !node.expanded {
        let candidates = candidate_moves_v2(game, net.cfg.max_candidates);
        let (input, opp_inputs, ent) = encode_game_local(game);
        let phase = phase_one_hot(game);
        let cache = net.forward(&input, &opp_inputs, &ent, phase);
        if candidates.is_empty() {
            node.expanded = true;
            node.visits += 1;
            return cache.value;
        }
        let priors = softmax(&candidate_logits_v2(&cache, &candidates));
        node.edges = candidates
            .into_iter()
            .zip(priors.into_iter())
            .map(|(action, prior)| Edge {
                action,
                prior,
                visits: 0,
                value_sum: 0.0,
                virtual_loss: 0,
                child: None,
            })
            .collect();
        node.expanded = true;
        node.visits += 1;
        return cache.value;
    }
    let edge_idx = select_puct_v2(node, net.cfg.c_puct);
    let mut next = game.clone();
    if !execute_scored_move(&mut next, &node.edges[edge_idx].action) {
        node.visits += 1;
        return 0.0;
    }
    advance_to_player_greedy(&mut next, root_player);
    let child = node.edges[edge_idx]
        .child
        .get_or_insert_with(|| Box::new(Node::new()));
    let value = simulate_puct_v2(child, &next, root_player, net);
    let edge = &mut node.edges[edge_idx];
    edge.visits += 1;
    edge.value_sum += value;
    node.visits += 1;
    value
}

fn select_puct_v2(node: &Node, c_puct: f32) -> usize {
    // Effective-visits PUCT (Stage 0.8.C). When every edge has
    // `virtual_loss == 0` (the serial path's invariant), this is bit-identical
    // to the pre-Stage-C formulation: `eff_visits == visits`,
    // `eff_value == value_sum`, `parent_eff == node.visits.max(1)`.
    let total_vloss: u32 = node.edges.iter().map(|e| e.virtual_loss).sum();
    let parent_eff = (node.visits + total_vloss).max(1) as f32;
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, edge) in node.edges.iter().enumerate() {
        let eff_visits = edge.visits + edge.virtual_loss;
        let q = if eff_visits == 0 {
            0.5
        } else {
            // VLOSS = 1.0 in [0,1] value space: each pending visit pretends
            // the child returned 0 (a loss), discouraging sibling-paths from
            // re-selecting this edge until the real value backs up.
            let eff_value = edge.value_sum - edge.virtual_loss as f32;
            eff_value / eff_visits as f32
        };
        let u = c_puct * edge.prior * parent_eff.sqrt() / (1.0 + eff_visits as f32);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

// ─────────────────────────────────────────────────────────────────────
// Stage 0.8.C — Batched-leaf MCTS with virtual loss.
//
// Algorithm: collect up to K leaves with PUCT-with-virtual-loss descents,
// forward all K in one call, expand + backup all K. K=1 is bit-identical
// to `az_search_v2_serial`. K>1 amortizes the network call across leaves;
// virtual-loss bookkeeping discourages sibling descents from racing to the
// same edge.
//
// The unsafe raw-pointer descent mirrors the proven pattern in
// `crates/cascadia-ai/src/ol_mcts.rs:229-360`. Rust's borrow checker can't
// see through tree mutation; we sequentialize the K descents within
// `az_search_v2_batched` so each pointer is unique at any given moment.
// ─────────────────────────────────────────────────────────────────────

enum DescentResult {
    /// Hit an unexpanded node — needs `forward` to build priors.
    NeedsForward { path: Vec<usize>, game: GameState },
    /// Hit a terminal (game over), an empty-candidates node, or an aborted
    /// (illegal-move) descent. No forward needed; backup with `value`.
    BackupOnly { path: Vec<usize>, value: f32 },
}

fn descend_one(
    root: &mut Node,
    base_game: &GameState,
    root_player: usize,
    rng: &mut StdRng,
    c_puct: f32,
) -> DescentResult {
    let mut g = base_game.clone();
    g.shuffle_bags(rng);
    let mut path: Vec<usize> = Vec::new();
    let mut node: *mut Node = root as *mut Node;
    // SAFETY: `node` is reachable via `root.edges[path[0]].child...child` and
    // the path keeps every traversed Box alive. Within one `descend_one` call
    // there are no concurrent borrows of `root`.
    loop {
        let n: &mut Node = unsafe { &mut *node };
        if g.is_game_over() {
            let v = score_with_bonus(&g, root_player) / AZ_VALUE_SCALE;
            return DescentResult::BackupOnly { path, value: v };
        }
        if !n.expanded {
            return DescentResult::NeedsForward { path, game: g };
        }
        if n.edges.is_empty() {
            // Expanded with zero candidates is the "candidates empty"
            // branch in `simulate_puct_v2` — returns `cache.value` which
            // we don't carry around. In production this is unreachable
            // (candidate_moves_v2 always returns at least the skip move
            // on terminal-ish states), so back up zero to keep totals
            // consistent. Asserts in debug builds catch regressions.
            debug_assert!(false, "az_v2: expanded node has zero edges");
            return DescentResult::BackupOnly { path, value: 0.0 };
        }
        let idx = select_puct_v2(n, c_puct);
        // Mark virtual loss BEFORE the exec — if exec succeeds, the loss
        // stays until backup decrements it; if exec fails, undo it before
        // returning so siblings see the correct state.
        n.edges[idx].virtual_loss += 1;
        let action = n.edges[idx].action;
        if !execute_scored_move(&mut g, &action) {
            // Mirror serial-PUCT aborted path: at this node, increment
            // visits and return 0. No edge backup at the failing index.
            n.edges[idx].virtual_loss -= 1;
            // `path` doesn't include `idx` (we push only after exec_ok).
            // Backup walks ancestors; this node gets +1 in `backup_path`.
            return DescentResult::BackupOnly { path, value: 0.0 };
        }
        path.push(idx);
        advance_to_player_greedy(&mut g, root_player);
        let child_ref = n.edges[idx]
            .child
            .get_or_insert_with(|| Box::new(Node::new()));
        node = (&mut **child_ref) as *mut Node;
    }
}

fn backup_path(root: &mut Node, path: &[usize], value: f32) {
    let mut node: *mut Node = root as *mut Node;
    // SAFETY: `path` was produced by `descend_one`; every step's child still
    // exists because we never deallocate during search.
    for &idx in path {
        let n: &mut Node = unsafe { &mut *node };
        n.visits += 1;
        let edge = &mut n.edges[idx];
        edge.visits += 1;
        edge.value_sum += value;
        edge.virtual_loss -= 1;
        let child_ref = edge.child.as_mut().expect("child after descent");
        node = (&mut **child_ref) as *mut Node;
    }
    // Final node corresponds to `simulate_puct_v2`'s `node.visits += 1`
    // line for the BackupOnly / NeedsForward cases.
    unsafe {
        (*node).visits += 1;
    }
}

fn expand_and_backup_leaf(
    root: &mut Node,
    path: &[usize],
    leaf_game: &GameState,
    cache: &ForwardCacheV2,
    max_candidates: usize,
) {
    let mut node: *mut Node = root as *mut Node;
    for &idx in path {
        let n: &mut Node = unsafe { &mut *node };
        n.visits += 1;
        let edge = &mut n.edges[idx];
        edge.visits += 1;
        edge.value_sum += cache.value;
        edge.virtual_loss -= 1;
        let child_ref = edge.child.as_mut().expect("child after descent");
        node = (&mut **child_ref) as *mut Node;
    }
    // `node` is the unexpanded leaf. Build edges and mark expanded —
    // same order as `simulate_puct_v2`.
    let leaf: &mut Node = unsafe { &mut *node };
    let candidates = candidate_moves_v2(leaf_game, max_candidates);
    if !candidates.is_empty() {
        let priors = softmax(&candidate_logits_v2(cache, &candidates));
        leaf.edges = candidates
            .into_iter()
            .zip(priors.into_iter())
            .map(|(action, prior)| Edge {
                action,
                prior,
                visits: 0,
                value_sum: 0.0,
                virtual_loss: 0,
                child: None,
            })
            .collect();
    }
    leaf.expanded = true;
    leaf.visits += 1;
}

/// `CASCADIA_AZ_BATCH_K` — leaf-batch size for `az_search_v2_batched`.
/// Default 1, which makes `az_search_v2` dispatch to the serial path.
fn az_batch_k() -> usize {
    std::env::var("CASCADIA_AZ_BATCH_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize)
        .max(1)
}

fn az_search_v2_batched(
    game: &GameState,
    net: &AlphaZeroNetV2,
    simulations: usize,
    batch_size: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<SearchResultV2> {
    let root_player = game.current_player;
    let mut root = Node::new();
    let total = simulations.max(1);
    let k_cap = batch_size.max(1);
    let c_puct = net.cfg.c_puct;
    let max_candidates = net.cfg.max_candidates;

    let mut done = 0usize;
    while done < total {
        let k = k_cap.min(total - done);
        let mut pending: Vec<(Vec<usize>, GameState)> = Vec::with_capacity(k);
        for _ in 0..k {
            match descend_one(&mut root, game, root_player, rng, c_puct) {
                DescentResult::NeedsForward { path, game: g } => {
                    pending.push((path, g));
                }
                DescentResult::BackupOnly { path, value } => {
                    backup_path(&mut root, &path, value);
                }
            }
            done += 1;
        }
        if !pending.is_empty() {
            let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
            let mut opps: Vec<Vec<Vec<f32>>> = Vec::with_capacity(pending.len());
            let mut ents: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
            let mut phases: Vec<[f32; AZ_VALUE_PHASES]> = Vec::with_capacity(pending.len());
            for (_, g) in &pending {
                let (input, opp, ent) = encode_game_local(g);
                inputs.push(input);
                opps.push(opp);
                ents.push(ent);
                phases.push(phase_one_hot(g));
            }
            let caches = net.forward_batch(&inputs, &opps, &ents, &phases);
            for ((path, g), cache) in pending.iter().zip(caches.iter()) {
                expand_and_backup_leaf(&mut root, path, g, cache, max_candidates);
            }
        }
    }

    if root.edges.is_empty() {
        return None;
    }
    let visits: Vec<u32> = root.edges.iter().map(|e| e.visits).collect();
    let total_v: u32 = visits.iter().sum();
    let visit_policy: Vec<f32> = if total_v == 0 {
        vec![1.0 / visits.len() as f32; visits.len()]
    } else {
        visits.iter().map(|&v| v as f32 / total_v as f32).collect()
    };
    let selected_idx = select_from_visits(&visits, temperature, rng);
    Some(SearchResultV2 {
        selected: root.edges[selected_idx].action,
        candidates: root.edges.iter().map(|e| e.action).collect(),
        visit_policy,
        visits,
    })
}

fn select_from_visits(visits: &[u32], temperature: f32, rng: &mut StdRng) -> usize {
    if temperature <= 0.01 {
        return visits
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    let weights: Vec<f32> = visits
        .iter()
        .map(|&v| (v.max(1) as f32).powf(1.0 / temperature))
        .collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.gen_range(0.0..total.max(1e-9));
    for (i, w) in weights.iter().enumerate() {
        if r <= *w {
            return i;
        }
        r -= *w;
    }
    weights.len().saturating_sub(1)
}

fn advance_to_player_greedy(game: &mut GameState, player: usize) {
    while !game.is_game_over() && game.current_player != player {
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        match greedy_move(game) {
            Some(mv) => {
                if !execute_scored_move(game, &mv) {
                    break;
                }
            }
            None => break,
        }
    }
}

fn score_with_bonus(game: &GameState, player: usize) -> f32 {
    let mut boards = game.boards.clone();
    ScoreBreakdown::compute_with_bonuses(&mut boards, &game.scoring_cards, player).total as f32
}

fn same_move(a: &ScoredMove, b: &ScoredMove) -> bool {
    a.market_index == b.market_index
        && a.wildlife_market_index == b.wildlife_market_index
        && a.tile_q == b.tile_q
        && a.tile_r == b.tile_r
        && a.rotation == b.rotation
        && a.wildlife_q == b.wildlife_q
        && a.wildlife_r == b.wildlife_r
}

// ─────────────────────────────────────────────────────────────────────
// Data collection: greedy bootstrap + self-play.
// ─────────────────────────────────────────────────────────────────────

/// Compute 16 aux-value targets from the final ScoreBreakdown for `player`.
fn aux_targets(breakdown: &ScoreBreakdown) -> [f32; AZ_VALUE_SUBHEADS] {
    let mut out = [0.0f32; AZ_VALUE_SUBHEADS];
    for i in 0..5 {
        out[i] = (breakdown.wildlife[i] as f32 / AZ_AUX_SCALES[i]).clamp(0.0, 1.0);
    }
    for i in 0..5 {
        out[5 + i] = (breakdown.habitat[i] as f32 / AZ_AUX_SCALES[5 + i]).clamp(0.0, 1.0);
    }
    out[10] = (breakdown.nature_tokens as f32 / AZ_AUX_SCALES[10]).clamp(0.0, 1.0);
    for i in 0..5 {
        out[11 + i] = (breakdown.habitat_bonus[i] as f32 / AZ_AUX_SCALES[11 + i]).clamp(0.0, 1.0);
    }
    out
}

fn finalize_v2(game: &GameState, pending: Vec<(AzSampleV2, usize)>, out: &mut Vec<AzSampleV2>) {
    for (mut sample, player) in pending {
        let mut boards = game.boards.clone();
        let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &game.scoring_cards, player);
        sample.value = (bd.total as f32 / AZ_VALUE_SCALE).clamp(0.0, 1.0);
        sample.aux_values = aux_targets(&bd);
        out.push(sample);
    }
}

/// Run one greedy-bootstrap game with a dedicated RNG and return its samples.
/// Pulled out as a free function so `collect_bootstrap_v2` can `par_iter` it.
fn run_one_bootstrap_game(seed: u64) -> Vec<AzSampleV2> {
    let mut rng = StdRng::seed_from_u64(seed);
    let cards = ScoringCards::all_a();
    let mut game = GameState::new(4, cards, &mut rng);
    let mut pending: Vec<(AzSampleV2, usize)> = Vec::new();
    while !game.is_game_over() {
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        let player = game.current_player;
        let candidates = candidate_moves_v2(&game, AlphaZeroV2Config::default().max_candidates);
        if candidates.is_empty() {
            break;
        }
        let greedy = greedy_move(&game).unwrap_or(candidates[0]);
        let target_idx = candidates
            .iter()
            .position(|mv| same_move(mv, &greedy))
            .unwrap_or(0);
        let mut policy = vec![0.0; candidates.len()];
        policy[target_idx] = 1.0;
        let (input, opp_inputs, entities) = encode_game_local(&game);
        pending.push((
            AzSampleV2 {
                input,
                opp_inputs,
                entities,
                candidates: candidates.clone(),
                policy,
                value: 0.0,
                aux_values: [0.0; AZ_VALUE_SUBHEADS],
            },
            player,
        ));
        if !execute_scored_move(&mut game, &greedy) {
            break;
        }
    }
    let mut out = Vec::new();
    finalize_v2(&game, pending, &mut out);
    out
}

pub fn collect_bootstrap_v2(num_games: usize, rng: &mut StdRng) -> Vec<AzSampleV2> {
    // Phase 0.8.E: each game is independent (no shared state, no net), so
    // games-in-parallel via rayon is a clean linear speedup with cores.
    // Deterministic: pre-generate per-game seeds from the master `rng` and
    // collect ordered output via `into_par_iter().collect()`.
    let seeds: Vec<u64> = (0..num_games).map(|_| rng.gen()).collect();
    use rayon::prelude::*;
    seeds
        .into_par_iter()
        .map(run_one_bootstrap_game)
        .flatten()
        .collect()
}

/// Run one self-play game with a dedicated RNG; uses `az_search_v2_serial` so
/// within-game search stays single-threaded — the per-game parallelism is
/// supplied by the outer `par_iter`. Mixing both saturates 8-core Mac mini.
fn run_one_selfplay_game(
    net: &AlphaZeroNetV2,
    simulations: usize,
    temperature: f32,
    seed: u64,
) -> Vec<AzSampleV2> {
    let mut rng = StdRng::seed_from_u64(seed);
    let cards = ScoringCards::all_a();
    let mut game = GameState::new(4, cards, &mut rng);
    let mut pending: Vec<(AzSampleV2, usize)> = Vec::new();
    while !game.is_game_over() {
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        let player = game.current_player;
        let (input, opp_inputs, entities) = encode_game_local(&game);
        let Some(search) = az_search_v2_serial(&game, net, simulations, temperature, &mut rng)
        else {
            break;
        };
        pending.push((
            AzSampleV2 {
                input,
                opp_inputs,
                entities,
                candidates: search.candidates.clone(),
                policy: search.visit_policy.clone(),
                value: 0.0,
                aux_values: [0.0; AZ_VALUE_SUBHEADS],
            },
            player,
        ));
        if !execute_scored_move(&mut game, &search.selected) {
            break;
        }
    }
    let mut out = Vec::new();
    finalize_v2(&game, pending, &mut out);
    out
}

pub fn collect_selfplay_v2(
    net: &AlphaZeroNetV2,
    num_games: usize,
    simulations: usize,
    temperature: f32,
    rng: &mut StdRng,
) -> Vec<AzSampleV2> {
    // Phase 0.8.E: games-in-parallel via rayon. `net` is shared via borrow
    // (rayon scoped tasks can capture &AlphaZeroNetV2). Within each game we
    // force serial PUCT so cores don't get oversubscribed.
    let seeds: Vec<u64> = (0..num_games).map(|_| rng.gen()).collect();
    use rayon::prelude::*;
    seeds
        .into_par_iter()
        .map(|seed| run_one_selfplay_game(net, simulations, temperature, seed))
        .flatten()
        .collect()
}

pub fn best_move_v2(
    game: &GameState,
    net: &AlphaZeroNetV2,
    simulations: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    az_search_v2(game, net, simulations, 0.0, rng).map(|r| r.selected)
}

pub fn benchmark_v2(
    net: &AlphaZeroNetV2,
    games: usize,
    simulations: usize,
    rng: &mut StdRng,
) -> (f32, f32) {
    let mut base = 0.0f32;
    let mut bonus = 0.0f32;
    for _ in 0..games {
        let cards = ScoringCards::all_a();
        let mut game = GameState::new(4, cards, rng);
        while !game.is_game_over() {
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let mv = if game.current_player == 0 {
                best_move_v2(&game, net, simulations, rng).or_else(|| greedy_move(&game))
            } else {
                greedy_move(&game)
            };
            match mv {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) {
                        break;
                    }
                }
                None => break,
            }
        }
        base +=
            ScoreBreakdown::compute(&mut game.boards[0].clone(), &game.scoring_cards).total as f32;
        bonus += score_with_bonus(&game, 0);
    }
    let denom = games.max(1) as f32;
    (base / denom, bonus / denom)
}

// ─────────────────────────────────────────────────────────────────────
// AZD2 sample file I/O
// ─────────────────────────────────────────────────────────────────────

pub fn save_samples_v2(path: &std::path::Path, samples: &[AzSampleV2]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(AZ_DATA_MAGIC_V2)?;
    write_u32(&mut f, AZ_INPUT_CHANNELS_V2 as u32)?;
    write_u32(&mut f, AZ_LOCAL_CELLS as u32)?;
    write_u32(&mut f, AZ_CELLS_PADDED as u32)?;
    write_u32(&mut f, AZ_ENTITY_TOKENS as u32)?;
    write_u32(&mut f, AZ_ENTITY_RAW_DIM as u32)?;
    write_u32(&mut f, AZ_VALUE_SUBHEADS as u32)?;
    write_u32(&mut f, AZ_MAX_OPPONENTS as u32)?;
    write_u32(&mut f, samples.len() as u32)?;
    let board_len = AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED;
    for s in samples {
        if s.input.len() != board_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AZD2 sample wrong input length",
            ));
        }
        if s.opp_inputs.len() != AZ_MAX_OPPONENTS
            || s.opp_inputs.iter().any(|b| b.len() != board_len)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AZD2 sample wrong opp_inputs shape",
            ));
        }
        if s.entities.len() != AZ_ENTITY_TOKENS * AZ_ENTITY_RAW_DIM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AZD2 sample wrong entity length",
            ));
        }
        if s.policy.len() != s.candidates.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AZD2 sample policy/candidate mismatch",
            ));
        }
        write_f32(&mut f, s.value)?;
        for &v in &s.aux_values {
            write_f32(&mut f, v)?;
        }
        write_u32(&mut f, s.candidates.len() as u32)?;
        for &x in &s.input {
            write_f32(&mut f, x)?;
        }
        for opp in &s.opp_inputs {
            for &x in opp {
                write_f32(&mut f, x)?;
            }
        }
        for &x in &s.entities {
            write_f32(&mut f, x)?;
        }
        for mv in &s.candidates {
            write_i32(&mut f, move_tile_index_local(mv) as i32)?;
            write_i32(
                &mut f,
                match move_wildlife_index_local(mv) {
                    Some(li) => li as i32,
                    None => -1,
                },
            )?;
            write_i32(&mut f, mv.market_index as i32)?;
            write_i32(
                &mut f,
                mv.wildlife_market_index.map(|i| i as i32).unwrap_or(-1),
            )?;
        }
        for &p in &s.policy {
            write_f32(&mut f, p)?;
        }
    }
    Ok(())
}

pub fn inspect_samples_v2(path: &std::path::Path) -> std::io::Result<AzDataV2Summary> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != AZ_DATA_MAGIC_V2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad AZD2 magic",
        ));
    }
    let input_channels = read_u32(&mut f)? as usize;
    let local_cells = read_u32(&mut f)? as usize;
    let cells_padded = read_u32(&mut f)? as usize;
    let entity_tokens = read_u32(&mut f)? as usize;
    let entity_raw_dim = read_u32(&mut f)? as usize;
    let value_subheads = read_u32(&mut f)? as usize;
    let max_opponents = read_u32(&mut f)? as usize;
    let samples = read_u32(&mut f)? as usize;
    let input_floats = input_channels * cells_padded;
    let entity_floats = entity_tokens * entity_raw_dim;
    let opp_floats = max_opponents * input_channels * cells_padded;
    let mut max_candidates = 0usize;
    let mut value_buf = [0u8; 4];
    for _ in 0..samples {
        f.read_exact(&mut value_buf)?; // value
        skip_bytes(&mut f, value_subheads * 4)?; // aux
        let n = read_u32(&mut f)? as usize;
        max_candidates = max_candidates.max(n);
        skip_bytes(&mut f, input_floats * 4)?;
        skip_bytes(&mut f, opp_floats * 4)?;
        skip_bytes(&mut f, entity_floats * 4)?;
        skip_bytes(&mut f, n * 4 * 4)?; // candidates (4 i32 each)
        skip_bytes(&mut f, n * 4)?; // policy
    }
    let mut trailing = [0u8; 1];
    if f.read(&mut trailing)? != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trailing AZD2 bytes",
        ));
    }
    Ok(AzDataV2Summary {
        samples,
        input_channels,
        local_cells,
        cells_padded,
        entity_tokens,
        entity_raw_dim,
        value_subheads,
        max_opponents,
        max_candidates,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Low-level read/write helpers
// ─────────────────────────────────────────────────────────────────────

fn write_u32(f: &mut std::fs::File, v: u32) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&v.to_le_bytes())
}

fn write_f32(f: &mut std::fs::File, v: f32) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&v.to_le_bytes())
}

fn write_i32(f: &mut std::fs::File, v: i32) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&v.to_le_bytes())
}

fn write_vec(f: &mut std::fs::File, v: &[f32]) -> std::io::Result<()> {
    write_u32(f, v.len() as u32)?;
    for &x in v {
        write_f32(f, x)?;
    }
    Ok(())
}

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(f: &mut std::fs::File) -> std::io::Result<f32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_vec(f: &mut std::fs::File) -> std::io::Result<Vec<f32>> {
    let len = read_u32(f)? as usize;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_f32(f)?);
    }
    Ok(out)
}

fn skip_bytes(f: &mut std::fs::File, n: usize) -> std::io::Result<()> {
    use std::io::Read;
    let mut remaining = n;
    let mut buf = [0u8; 8192];
    while remaining > 0 {
        let take = remaining.min(buf.len());
        f.read_exact(&mut buf[..take])?;
        remaining -= take;
    }
    Ok(())
}

fn write_conv(f: &mut std::fs::File, conv: &HexConv) -> std::io::Result<()> {
    write_u32(f, conv.in_c as u32)?;
    write_u32(f, conv.out_c as u32)?;
    write_vec(f, &conv.w)?;
    write_vec(f, &conv.b)
}

fn read_conv(
    f: &mut std::fs::File,
    expect_in: usize,
    expect_out: usize,
) -> std::io::Result<HexConv> {
    let in_c = read_u32(f)? as usize;
    let out_c = read_u32(f)? as usize;
    if in_c != expect_in || out_c != expect_out {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "AZR2 conv shape mismatch: got {}x{} expected {}x{}",
                in_c, out_c, expect_in, expect_out
            ),
        ));
    }
    let w = read_vec(f)?;
    let b = read_vec(f)?;
    Ok(HexConv { in_c, out_c, w, b })
}

fn write_sab(f: &mut std::fs::File, sab: &Sab) -> std::io::Result<()> {
    write_vec(f, &sab.ln1_scale)?;
    write_vec(f, &sab.ln1_bias)?;
    write_vec(f, &sab.qkv_w)?;
    write_vec(f, &sab.qkv_b)?;
    write_vec(f, &sab.out_w)?;
    write_vec(f, &sab.out_b)?;
    write_vec(f, &sab.ln2_scale)?;
    write_vec(f, &sab.ln2_bias)?;
    write_vec(f, &sab.ffn1_w)?;
    write_vec(f, &sab.ffn1_b)?;
    write_vec(f, &sab.ffn2_w)?;
    write_vec(f, &sab.ffn2_b)
}

fn read_sab(f: &mut std::fs::File, d: usize, heads: usize, ffn_dim: usize) -> std::io::Result<Sab> {
    let ln1_scale = read_vec(f)?;
    let ln1_bias = read_vec(f)?;
    let qkv_w = read_vec(f)?;
    let qkv_b = read_vec(f)?;
    let out_w = read_vec(f)?;
    let out_b = read_vec(f)?;
    let ln2_scale = read_vec(f)?;
    let ln2_bias = read_vec(f)?;
    let ffn1_w = read_vec(f)?;
    let ffn1_b = read_vec(f)?;
    let ffn2_w = read_vec(f)?;
    let ffn2_b = read_vec(f)?;
    Ok(Sab {
        d,
        heads,
        head_dim: d / heads,
        ffn_dim,
        ln1_scale,
        ln1_bias,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        ln2_scale,
        ln2_bias,
        ffn1_w,
        ffn1_b,
        ffn2_w,
        ffn2_b,
    })
}

fn write_cross(f: &mut std::fs::File, c: &CrossAttn) -> std::io::Result<()> {
    write_vec(f, &c.q_w)?;
    write_vec(f, &c.q_b)?;
    write_vec(f, &c.k_w)?;
    write_vec(f, &c.k_b)?;
    write_vec(f, &c.v_w)?;
    write_vec(f, &c.v_b)?;
    write_vec(f, &c.out_w)?;
    write_vec(f, &c.out_b)
}

fn read_cross(
    f: &mut std::fs::File,
    trunk_c: usize,
    d_ent: usize,
    heads: usize,
) -> std::io::Result<CrossAttn> {
    let q_w = read_vec(f)?;
    let q_b = read_vec(f)?;
    let k_w = read_vec(f)?;
    let k_b = read_vec(f)?;
    let v_w = read_vec(f)?;
    let v_b = read_vec(f)?;
    let out_w = read_vec(f)?;
    let out_b = read_vec(f)?;
    Ok(CrossAttn {
        d_ent,
        trunk_c,
        heads,
        head_dim: d_ent / heads,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        out_w,
        out_b,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> AlphaZeroV2Config {
        AlphaZeroV2Config {
            channels: 8,
            blocks: 1,
            entity_dim: 8,
            sab_blocks: 1,
            heads: 2,
            value_hidden: 8,
            value_subheads: AZ_VALUE_SUBHEADS,
            max_candidates: 8,
            c_puct: 2.0,
        }
    }

    #[test]
    fn encode_game_local_shape() {
        let mut rng = StdRng::seed_from_u64(1);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let (input, opp_inputs, entities) = encode_game_local(&game);
        assert_eq!(input.len(), AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED);
        assert_eq!(opp_inputs.len(), AZ_MAX_OPPONENTS);
        for opp in &opp_inputs {
            assert_eq!(opp.len(), AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED);
        }
        assert_eq!(entities.len(), AZ_ENTITY_TOKENS * AZ_ENTITY_RAW_DIM);
        assert!(input.iter().all(|v| v.is_finite()));
        assert!(opp_inputs.iter().all(|b| b.iter().all(|v| v.is_finite())));
        assert!(entities.iter().all(|v| v.is_finite()));
        // Bias plane (0) is 1.0 on real cells and 0.0 on the pad cell. This is
        // the sole "real vs pad" discriminator (no dedicated is-pad plane).
        for li in 0..AZ_LOCAL_CELLS {
            assert_eq!(input[0 * AZ_CELLS_PADDED + li], 1.0);
        }
        assert_eq!(input[0 * AZ_CELLS_PADDED + AZ_PAD_INDEX], 0.0);
        // Each opponent board (in a 4P game) also has bias=1 on real cells.
        for opp in &opp_inputs {
            for li in 0..AZ_LOCAL_CELLS {
                assert_eq!(opp[0 * AZ_CELLS_PADDED + li], 1.0);
            }
            assert_eq!(opp[0 * AZ_CELLS_PADDED + AZ_PAD_INDEX], 0.0);
        }
    }

    #[test]
    fn entity_tokens_shape() {
        let mut rng = StdRng::seed_from_u64(2);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let entities = build_entity_tokens(&game);
        assert_eq!(entities.len(), AZ_ENTITY_TOKENS * AZ_ENTITY_RAW_DIM);
        // Market tokens have slot-position one-hot at dims 22..26.
        for slot in 0..4 {
            let off = slot * AZ_ENTITY_RAW_DIM + 22 + slot;
            assert_eq!(
                entities[off], 1.0,
                "market slot {} missing slot-position one-hot",
                slot
            );
        }
        // Opponent tokens are intentionally zero at encode time — the shared
        // opp trunk fills them at forward() time from opp_inputs.
        for opp in 0..AZ_MAX_OPPONENTS {
            let base = (4 + opp) * AZ_ENTITY_RAW_DIM;
            for d in 0..AZ_ENTITY_RAW_DIM {
                assert_eq!(
                    entities[base + d],
                    0.0,
                    "opp slot {} dim {} should be zero (filled by trunk later)",
                    opp,
                    d
                );
            }
        }
        // Bag token is-bag flag is at dim 19.
        assert_eq!(entities[7 * AZ_ENTITY_RAW_DIM + 19], 1.0);
        // Globals token (idx 8) is-globals flag at dim 21.
        assert_eq!(entities[8 * AZ_ENTITY_RAW_DIM + 21], 1.0);
        // Race-state token (idx 9) is-race-state flag at dim 25.
        assert_eq!(entities[9 * AZ_ENTITY_RAW_DIM + 25], 1.0);

        // Phase one-hot at globals dims 0..3 sums to exactly 1.0.
        let phase_sum: f32 = (0..3).map(|d| entities[8 * AZ_ENTITY_RAW_DIM + d]).sum();
        assert!(
            (phase_sum - 1.0).abs() < 1e-6,
            "phase one-hot sum = {}",
            phase_sum
        );

        // Own-seat one-hot at globals dims 3..7 sums to 1.0.
        let seat_sum: f32 = (3..7).map(|d| entities[8 * AZ_ENTITY_RAW_DIM + d]).sum();
        assert!(
            (seat_sum - 1.0).abs() < 1e-6,
            "seat one-hot sum = {}",
            seat_sum
        );

        // Num-players one-hot at globals dims 7..10 sums to 1.0 (game is 4-P).
        let np_sum: f32 = (7..10).map(|d| entities[8 * AZ_ENTITY_RAW_DIM + d]).sum();
        assert!(
            (np_sum - 1.0).abs() < 1e-6,
            "num-players one-hot sum = {}",
            np_sum
        );

        // Race-state: each of 5 terrains gets a 4-way rank one-hot in dims
        // 0..20. Each 4-block sums to exactly 1.0.
        for ti in 0..5 {
            let rank_sum: f32 = (0..4)
                .map(|r| entities[9 * AZ_ENTITY_RAW_DIM + ti * 4 + r])
                .sum();
            assert!(
                (rank_sum - 1.0).abs() < 1e-6,
                "race-state terrain {} rank one-hot sum = {}",
                ti,
                rank_sum
            );
        }
    }

    #[test]
    fn hex_conv_neighbors_pad() {
        let n = hex_neighbors_local();
        assert_eq!(n.len(), AZ_CELLS_PADDED);
        // Pad cell routes all 7 entries to itself (inert).
        for k in 0..7 {
            assert_eq!(n[AZ_PAD_INDEX][k], AZ_PAD_INDEX);
        }
        // Local index 0 = center cell — all 6 neighbors are in-disk (within radius 6).
        for k in 1..7 {
            assert_ne!(
                n[0][k], AZ_PAD_INDEX,
                "center cell neighbor {} should be in-disk",
                k
            );
        }
        // Some outer-ring cell must have at least one out-of-disk neighbor.
        let mut found_pad_neighbor = false;
        for li in 0..AZ_LOCAL_CELLS {
            for k in 1..7 {
                if n[li][k] == AZ_PAD_INDEX {
                    found_pad_neighbor = true;
                }
            }
        }
        assert!(
            found_pad_neighbor,
            "expected outer-ring cells to route to pad"
        );
    }

    #[test]
    fn move_index_local_handles_in_and_out_of_disk() {
        // In-disk: origin (0,0)
        let mv_in = ScoredMove {
            market_index: 0,
            wildlife_market_index: None,
            tile_q: 0,
            tile_r: 0,
            rotation: 0,
            wildlife_q: None,
            wildlife_r: None,
            score: 0,
            eval: 0,
        };
        let li_in = move_tile_index_local(&mv_in);
        assert!(
            li_in < AZ_LOCAL_CELLS,
            "(0,0) should be in disk, got {}",
            li_in
        );
        // Out-of-disk: far corner (10,-10)
        let mv_out = ScoredMove {
            market_index: 0,
            wildlife_market_index: None,
            tile_q: 10,
            tile_r: -10,
            rotation: 0,
            wildlife_q: None,
            wildlife_r: None,
            score: 0,
            eval: 0,
        };
        let li_out = move_tile_index_local(&mv_out);
        assert_eq!(li_out, AZ_PAD_INDEX, "far corner should clamp to pad");
    }

    #[test]
    fn multi_head_blend_outputs_in_unit_interval() {
        let mut rng = StdRng::seed_from_u64(3);
        let mhv = MultiHeadValue::new(8, 8, AZ_VALUE_SUBHEADS, &mut rng);
        let pooled = vec![0.5f32; 8];
        for phase_idx in 0..AZ_VALUE_PHASES {
            let mut phase = [0.0; AZ_VALUE_PHASES];
            phase[phase_idx] = 1.0;
            let (v, subs) = mhv.forward(&pooled, phase);
            assert!((0.0..=1.0).contains(&v), "phase {} v={}", phase_idx, v);
            assert_eq!(subs.len(), AZ_VALUE_SUBHEADS);
            assert!(subs.iter().all(|s| (0.0..=1.0).contains(s)));
        }
    }

    #[test]
    fn network_policy_is_distribution_over_legal_candidates() {
        let mut rng = StdRng::seed_from_u64(4);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let net = AlphaZeroNetV2::new(small_config(), 5);
        let cands = candidate_moves_v2(&game, 8);
        let (v, probs) = net.evaluate(&game, &cands);
        assert!((0.0..=1.0).contains(&v));
        assert_eq!(probs.len(), cands.len());
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy sum={}", sum);
        assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
    }

    #[test]
    fn azr2_save_load_roundtrip() {
        let mut rng = StdRng::seed_from_u64(6);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let net = AlphaZeroNetV2::new(small_config(), 7);
        let path = std::env::temp_dir().join("cascadia_azr2_roundtrip.azr");
        net.save(&path).unwrap();
        let loaded = AlphaZeroNetV2::load(&path).unwrap();
        let cands = candidate_moves_v2(&game, 8);
        let (v1, p1) = net.evaluate(&game, &cands);
        let (v2, p2) = loaded.evaluate(&game, &cands);
        assert!((v1 - v2).abs() < 1e-5, "v1={} v2={}", v1, v2);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn puct_legality_v2() {
        let mut rng = StdRng::seed_from_u64(8);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let net = AlphaZeroNetV2::new(small_config(), 9);
        let result = az_search_v2(&game, &net, 4, 0.0, &mut rng).unwrap();
        assert!(!result.candidates.is_empty());
        assert_eq!(result.visit_policy.len(), result.candidates.len());
        assert!(result
            .candidates
            .iter()
            .any(|m| same_move(m, &result.selected)));
    }

    #[test]
    fn azd2_replay_roundtrip() {
        let mut rng = StdRng::seed_from_u64(10);
        let samples = collect_bootstrap_v2(1, &mut rng);
        assert!(!samples.is_empty());
        let path = std::env::temp_dir().join("cascadia_azd2_roundtrip.azd");
        save_samples_v2(&path, &samples).unwrap();
        let summary = inspect_samples_v2(&path).unwrap();
        assert_eq!(summary.samples, samples.len());
        assert_eq!(summary.input_channels, AZ_INPUT_CHANNELS_V2);
        assert_eq!(summary.local_cells, AZ_LOCAL_CELLS);
        assert_eq!(summary.cells_padded, AZ_CELLS_PADDED);
        assert_eq!(summary.entity_tokens, AZ_ENTITY_TOKENS);
        assert_eq!(summary.entity_raw_dim, AZ_ENTITY_RAW_DIM);
        assert_eq!(summary.value_subheads, AZ_VALUE_SUBHEADS);
        assert_eq!(summary.max_opponents, AZ_MAX_OPPONENTS);
        assert!(summary.max_candidates > 0);
        // Validate label structure.
        for s in samples.iter().take(3) {
            assert_eq!(s.candidates.len(), s.policy.len());
            let psum: f32 = s.policy.iter().sum();
            assert!((psum - 1.0).abs() < 1e-6, "policy sum={}", psum);
            assert!((0.0..=1.0).contains(&s.value));
            assert!(s.aux_values.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn greedy_bootstrap_v2_labels_are_valid() {
        let mut rng = StdRng::seed_from_u64(11);
        let samples = collect_bootstrap_v2(1, &mut rng);
        assert!(!samples.is_empty());
        for s in samples.iter().take(5) {
            assert_eq!(s.input.len(), AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED);
            assert_eq!(s.entities.len(), AZ_ENTITY_TOKENS * AZ_ENTITY_RAW_DIM);
            assert_eq!(s.candidates.len(), s.policy.len());
            let sum: f32 = s.policy.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            assert!((0.0..=1.0).contains(&s.value));
            assert!(s.aux_values.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    /// Verify that every (cell, terrain) pair flagged by
    /// `compute_habitat_roles` matches the corresponding plane bit, and that
    /// no extra bits are set. Cross-checks the encoder against the
    /// source-of-truth BFS, ensuring the size-threshold heuristic is gone.
    #[test]
    fn cluster_roles_correctness() {
        let mut rng = StdRng::seed_from_u64(20);
        let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        // Play a few greedy moves so the board has non-trivial clusters.
        use crate::search::{execute_scored_move, greedy_move};
        for _ in 0..12 {
            if game.is_game_over() {
                break;
            }
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let mv = greedy_move(&game).expect("greedy returns Some");
            assert!(execute_scored_move(&mut game, &mv));
        }
        let player = game.current_player;
        let board = &game.boards[player];
        let (input, _, _) = encode_game_local(&game);
        let roles = crate::nnue::v6_peak::compute_habitat_roles(board);
        let mut any_role_seen = false;
        for li in 0..AZ_LOCAL_CELLS {
            let gi = local_to_global(li) as usize;
            for ti in 0..5 {
                for role in 0..3u8 {
                    let plane_idx = 45 + ti * 3 + role as usize;
                    let bit = input[plane_idx * AZ_CELLS_PADDED + li];
                    let expected = roles[gi][ti] == Some(role);
                    if expected {
                        any_role_seen = true;
                        assert_eq!(
                            bit, 1.0,
                            "cluster role mismatch at cell {} terrain {} role {}: \
                             expected 1 (BFS says yes), got {}",
                            li, ti, role, bit
                        );
                    } else {
                        assert_eq!(
                            bit, 0.0,
                            "cluster role mismatch at cell {} terrain {} role {}: \
                             expected 0 (BFS says no), got {}",
                            li, ti, role, bit
                        );
                    }
                }
            }
        }
        assert!(
            any_role_seen,
            "no cluster role bits ever set after 12 moves"
        );
    }

    /// Place a starter cluster (3 tiles) and verify the per-direction edge
    /// terrain planes exactly match `board::terrain_on_edge`.
    #[test]
    fn per_dir_edge_terrain_probe() {
        let mut rng = StdRng::seed_from_u64(21);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let player = game.current_player;
        let board = &game.boards[player];
        let (input, _, _) = encode_game_local(&game);
        let mut any_edge_seen = false;
        for li in 0..AZ_LOCAL_CELLS {
            let gi = local_to_global(li) as usize;
            let cell = board.grid.get(gi);
            if !cell.is_present() {
                continue;
            }
            let rot = board.rotations[gi];
            for d in 0..6 {
                let expected = cascadia_core::board::terrain_on_edge(cell, rot, d);
                for ti in 0..5 {
                    let plane_idx = 15 + d * 5 + ti;
                    let bit = input[plane_idx * AZ_CELLS_PADDED + li];
                    let want = expected == cascadia_core::types::Terrain::from_u8(ti as u8);
                    if want {
                        any_edge_seen = true;
                        assert_eq!(
                            bit, 1.0,
                            "edge plane mismatch cell={} dir={} terrain={}: \
                             expected 1 (terrain_on_edge agrees), got {}",
                            li, d, ti, bit
                        );
                    } else {
                        assert_eq!(
                            bit, 0.0,
                            "edge plane mismatch cell={} dir={} terrain={}: \
                             expected 0, got {}",
                            li, d, ti, bit
                        );
                    }
                }
            }
        }
        assert!(any_edge_seen, "no edge-terrain bits set on starter cluster");
    }

    /// After several placements, the recency plane should:
    ///   1. Have non-zero values on all placed cells and zero elsewhere.
    ///   2. Assign higher values to more-recently placed tiles.
    #[test]
    fn recency_monotone() {
        let mut rng = StdRng::seed_from_u64(22);
        let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        use crate::search::{execute_scored_move, greedy_move};
        for _ in 0..10 {
            if game.is_game_over() {
                break;
            }
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let mv = greedy_move(&game).expect("greedy returns Some");
            assert!(execute_scored_move(&mut game, &mv));
        }
        let player = game.current_player;
        let board = &game.boards[player];
        let (input, _, _) = encode_game_local(&game);
        // Walk placed_tiles in stack order. Recency must be (k+1)/N at each
        // placed cell, monotonically non-decreasing.
        let n = board.placed_tiles.len() as f32;
        assert!(n > 3.0, "expected >3 placements after greedy moves");
        for (k, &gi_u) in board.placed_tiles.iter().enumerate() {
            let gi = gi_u as usize;
            let li = global_to_local(gi);
            if li < 0 {
                continue; // out-of-disk (rare)
            }
            let v = input[67 * AZ_CELLS_PADDED + li as usize];
            let expected = (k as f32 + 1.0) / n;
            assert!(
                (v - expected).abs() < 1e-5,
                "recency mismatch at stack pos {} cell {}: got {}, expected {}",
                k,
                li,
                v,
                expected
            );
        }
        // Empty cells (within disk) have recency 0.
        for li in 0..AZ_LOCAL_CELLS {
            let gi = local_to_global(li) as usize;
            if !board.grid.get(gi).is_present() {
                assert_eq!(
                    input[67 * AZ_CELLS_PADDED + li],
                    0.0,
                    "empty cell {} should have recency 0",
                    li
                );
            }
        }
    }

    /// At game start the wildlife bag has 100 - 4 = 96 tokens (4 drawn by the
    /// initial market). Bag-token dims 5..10 record per-type counts / 20 — so
    /// each bin should be ≤ 1.0, and the sum should match (96/20 = 4.8).
    #[test]
    fn bag_token_scales() {
        let mut rng = StdRng::seed_from_u64(23);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let entities = build_entity_tokens(&game);
        let bag = &entities[7 * AZ_ENTITY_RAW_DIM..(7 + 1) * AZ_ENTITY_RAW_DIM];
        // Per-type counts at dims 5..10. Each in [0, 1].
        for d in 5..10 {
            assert!(
                (0.0..=1.0).contains(&bag[d]),
                "bag dim {} = {} out of [0,1]",
                d,
                bag[d]
            );
        }
        // Sum of /20 normalized counts ≈ total_in_bag / 20.
        let sum: f32 = bag[5..10].iter().sum();
        let bag_size = game.wildlife_bag.remaining() as f32;
        let expected = bag_size / 20.0;
        assert!(
            (sum - expected).abs() < 1e-4,
            "bag-token wildlife sum = {}, expected ≈ {} (bag_size = {})",
            sum,
            expected,
            bag_size
        );
    }

    /// The shared opp trunk fills entity slots 4..7 from spatial inputs. Verify
    /// the pooled vectors are non-trivial and that running with zero
    /// opp_inputs vs real opp_inputs measurably changes the network's value
    /// output (= proof the trunk is actually wired into the forward path).
    #[test]
    fn opp_trunk_pooled_shape() {
        let mut rng = StdRng::seed_from_u64(40);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let (input, opp_inputs, ent_raw) = encode_game_local(&game);
        let net = AlphaZeroNetV2::new(small_config(), 41);

        let mut total_nonzero = 0usize;
        for opp in &opp_inputs {
            let pooled = net.encode_opp(opp);
            assert_eq!(pooled.len(), AZ_OPP_TRUNK_CHANNELS);
            assert!(pooled.iter().all(|v| v.is_finite()));
            total_nonzero += pooled.iter().filter(|v| v.abs() > 1e-6).count();
        }
        assert!(
            total_nonzero > 0,
            "opp trunk pooled vectors are all zero — check forward path"
        );

        let phase = phase_one_hot(&game);
        let cache_real = net.forward(&input, &opp_inputs, &ent_raw, phase);
        let zero_opps: Vec<Vec<f32>> = (0..AZ_MAX_OPPONENTS).map(|_| zero_board_planes()).collect();
        let cache_zero = net.forward(&input, &zero_opps, &ent_raw, phase);
        assert!(
            (cache_real.value - cache_zero.value).abs() > 1e-8,
            "real-opps and zero-opps produce identical values; opp trunk \
             likely disconnected from the entity stream"
        );
    }

    /// Sample collection populates `opp_inputs` with 3 boards, each 68×128.
    #[test]
    fn opp_inputs_collected() {
        let mut rng = StdRng::seed_from_u64(42);
        let samples = collect_bootstrap_v2(1, &mut rng);
        assert!(!samples.is_empty());
        for s in samples.iter().take(5) {
            assert_eq!(s.opp_inputs.len(), AZ_MAX_OPPONENTS);
            for opp in &s.opp_inputs {
                assert_eq!(opp.len(), AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED);
                assert!(opp.iter().all(|v| v.is_finite()));
                // In a 4P AAAAA game every opp is real → bias plane non-zero.
                let bias_sum: f32 = (0..AZ_LOCAL_CELLS)
                    .map(|li| opp[0 * AZ_CELLS_PADDED + li])
                    .sum();
                assert!(
                    bias_sum > 0.0,
                    "opp board bias plane is all zero — opponent should be real"
                );
            }
        }
    }

    // ── Phase 0.8.B per-layer parity tests ──────────────────────────────
    //
    // These tests preserve the *exact* pre-SGEMM scalar implementation as a
    // reference and assert that the new SGEMM-backed versions produce
    // outputs within 5e-4 max-abs-diff. Confirms the math is identical
    // modulo f32 FMA accumulation reorder (the standard BLAS tolerance).

    /// Pre-SGEMM HexConv body, copy-pasted from the prior implementation.
    /// Used only in tests as a reference. If the SGEMM-backed forward
    /// matches this within 5e-4, the refactor is mathematically equivalent.
    fn hexconv_scalar_reference(
        in_c: usize,
        out_c: usize,
        w: &[f32],
        b: &[f32],
        input: &[f32],
    ) -> Vec<f32> {
        let neighbors = hex_neighbors_local();
        let mut out = vec![0.0f32; out_c * AZ_CELLS_PADDED];
        for oc in 0..out_c {
            let wb = oc * in_c * 7;
            let ob = oc * AZ_CELLS_PADDED;
            for cell in 0..AZ_CELLS_PADDED {
                let nb = &neighbors[cell];
                let mut sum = b[oc];
                for ic in 0..in_c {
                    let ib = ic * AZ_CELLS_PADDED;
                    let wic = wb + ic * 7;
                    for k in 0..7 {
                        sum += w[wic + k] * input[ib + nb[k]];
                    }
                }
                out[ob + cell] = sum;
            }
        }
        out
    }

    #[test]
    fn parity_b_hexconv_matches_scalar_reference() {
        // Stem shape (most common): in_c=72 (AZ_INPUT_CHANNELS_V2), out_c=96.
        let mut rng = StdRng::seed_from_u64(0xB101);
        let conv = HexConv::new(AZ_INPUT_CHANNELS_V2, AZ_TRUNK_CHANNELS, &mut rng);
        let input: Vec<f32> = (0..AZ_INPUT_CHANNELS_V2 * AZ_CELLS_PADDED)
            .map(|i| ((i as i64 * 7919) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let actual = conv.forward(&input);
        let expected = hexconv_scalar_reference(
            AZ_INPUT_CHANNELS_V2,
            AZ_TRUNK_CHANNELS,
            &conv.w,
            &conv.b,
            &input,
        );
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs < 5e-4,
            "HexConv SGEMM vs scalar-reference max-abs-diff = {max_abs} (limit 5e-4)"
        );
    }

    #[test]
    fn parity_b_hexconv_resblock_inner_matches_scalar_reference() {
        // Inner ResBlock conv: 96 → 96.
        let mut rng = StdRng::seed_from_u64(0xB201);
        let conv = HexConv::new(AZ_TRUNK_CHANNELS, AZ_TRUNK_CHANNELS, &mut rng);
        let input: Vec<f32> = (0..AZ_TRUNK_CHANNELS * AZ_CELLS_PADDED)
            .map(|i| ((i as i64 * 6151 + 11) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let actual = conv.forward(&input);
        let expected = hexconv_scalar_reference(
            AZ_TRUNK_CHANNELS,
            AZ_TRUNK_CHANNELS,
            &conv.w,
            &conv.b,
            &input,
        );
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs < 5e-4,
            "HexConv (96→96) SGEMM vs scalar-reference max-abs-diff = {max_abs} (limit 5e-4)"
        );
    }

    /// Pre-SGEMM Sab forward body, preserved verbatim as a reference.
    fn sab_scalar_reference(sab: &Sab, x: &[f32], n_tokens: usize) -> Vec<f32> {
        let d = sab.d;
        let mut x_ln = x.to_vec();
        for t in 0..n_tokens {
            let s = t * d;
            layer_norm_inplace(&mut x_ln[s..s + d], &sab.ln1_scale, &sab.ln1_bias);
        }
        let mut qkv = vec![0.0f32; n_tokens * 3 * d];
        for t in 0..n_tokens {
            let xb = t * d;
            let qb = t * 3 * d;
            for o in 0..3 * d {
                let mut sum = sab.qkv_b[o];
                let wb = o * d;
                for i in 0..d {
                    sum += sab.qkv_w[wb + i] * x_ln[xb + i];
                }
                qkv[qb + o] = sum;
            }
        }
        let scale = 1.0 / (sab.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; n_tokens * d];
        for h in 0..sab.heads {
            let q_off = h * sab.head_dim;
            let k_off = d + h * sab.head_dim;
            let v_off = 2 * d + h * sab.head_dim;
            for i in 0..n_tokens {
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..n_tokens {
                    let qb = i * 3 * d + q_off;
                    let kb = j * 3 * d + k_off;
                    let mut s = 0.0;
                    for k in 0..sab.head_dim {
                        s += qkv[qb + k] * qkv[kb + k];
                    }
                    scores[j] = s * scale;
                }
                let attn = softmax(&scores);
                let ob = i * d + h * sab.head_dim;
                for k in 0..sab.head_dim {
                    let mut s = 0.0;
                    for j in 0..n_tokens {
                        let vb = j * 3 * d + v_off;
                        s += attn[j] * qkv[vb + k];
                    }
                    attn_out[ob + k] = s;
                }
            }
        }
        let mut after_attn = vec![0.0f32; n_tokens * d];
        for t in 0..n_tokens {
            let ob = t * d;
            for o in 0..d {
                let mut sum = sab.out_b[o];
                let wb = o * d;
                for i in 0..d {
                    sum += sab.out_w[wb + i] * attn_out[ob + i];
                }
                after_attn[ob + o] = sum + x[ob + o];
            }
        }
        let mut x2_ln = after_attn.clone();
        for t in 0..n_tokens {
            let s = t * d;
            layer_norm_inplace(&mut x2_ln[s..s + d], &sab.ln2_scale, &sab.ln2_bias);
        }
        let mut hidden = vec![0.0f32; n_tokens * sab.ffn_dim];
        for t in 0..n_tokens {
            let xb = t * d;
            let hb = t * sab.ffn_dim;
            for o in 0..sab.ffn_dim {
                let mut sum = sab.ffn1_b[o];
                let wb = o * d;
                for i in 0..d {
                    sum += sab.ffn1_w[wb + i] * x2_ln[xb + i];
                }
                hidden[hb + o] = sum.max(0.0);
            }
        }
        let mut out = vec![0.0f32; n_tokens * d];
        for t in 0..n_tokens {
            let hb = t * sab.ffn_dim;
            let ob = t * d;
            for o in 0..d {
                let mut sum = sab.ffn2_b[o];
                let wb = o * sab.ffn_dim;
                for i in 0..sab.ffn_dim {
                    sum += sab.ffn2_w[wb + i] * hidden[hb + i];
                }
                out[ob + o] = sum + after_attn[ob + o];
            }
        }
        out
    }

    #[test]
    fn parity_b_sab_matches_scalar_reference() {
        let mut rng = StdRng::seed_from_u64(0xB301);
        let sab = Sab::new(AZ_ENTITY_DIM, AZ_ATTN_HEADS, AZ_SAB_FFN_DIM, &mut rng);
        let x: Vec<f32> = (0..AZ_ENTITY_TOKENS * AZ_ENTITY_DIM)
            .map(|i| ((i as i64 * 4391 + 17) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let actual = sab.forward(&x, AZ_ENTITY_TOKENS);
        let expected = sab_scalar_reference(&sab, &x, AZ_ENTITY_TOKENS);
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs < 5e-4,
            "SAB SGEMM vs scalar-reference max-abs-diff = {max_abs} (limit 5e-4)"
        );
    }

    /// Pre-SGEMM CrossAttn forward body, preserved as a reference.
    fn crossattn_scalar_reference(
        cross: &CrossAttn,
        trunk: &[f32],
        entities: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let d = cross.d_ent;
        let mut k_tok = vec![0.0f32; n_tokens * d];
        let mut v_tok = vec![0.0f32; n_tokens * d];
        for t in 0..n_tokens {
            let eb = t * d;
            for o in 0..d {
                let mut sk = cross.k_b[o];
                let mut sv = cross.v_b[o];
                let wb = o * d;
                for i in 0..d {
                    sk += cross.k_w[wb + i] * entities[eb + i];
                    sv += cross.v_w[wb + i] * entities[eb + i];
                }
                k_tok[eb + o] = sk;
                v_tok[eb + o] = sv;
            }
        }
        let scale = 1.0 / (cross.head_dim as f32).sqrt();
        let mut delta = vec![0.0f32; cross.trunk_c * AZ_CELLS_PADDED];
        let mut q_cell = vec![0.0f32; d];
        let mut ctx = vec![0.0f32; d];
        for cell in 0..AZ_CELLS_PADDED {
            for o in 0..d {
                let mut s = cross.q_b[o];
                let wb = o * cross.trunk_c;
                for ic in 0..cross.trunk_c {
                    s += cross.q_w[wb + ic] * trunk[ic * AZ_CELLS_PADDED + cell];
                }
                q_cell[o] = s;
            }
            for x in ctx.iter_mut() {
                *x = 0.0;
            }
            for h in 0..cross.heads {
                let h_off = h * cross.head_dim;
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..n_tokens {
                    let kb = j * d + h_off;
                    let mut s = 0.0;
                    for k in 0..cross.head_dim {
                        s += q_cell[h_off + k] * k_tok[kb + k];
                    }
                    scores[j] = s * scale;
                }
                let attn = softmax(&scores);
                for k in 0..cross.head_dim {
                    let mut s = 0.0;
                    for j in 0..n_tokens {
                        let vb = j * d + h_off;
                        s += attn[j] * v_tok[vb + k];
                    }
                    ctx[h_off + k] = s;
                }
            }
            for oc in 0..cross.trunk_c {
                let mut s = cross.out_b[oc];
                let wb = oc * d;
                for i in 0..d {
                    s += cross.out_w[wb + i] * ctx[i];
                }
                delta[oc * AZ_CELLS_PADDED + cell] = s;
            }
        }
        let mut out = vec![0.0f32; trunk.len()];
        for i in 0..trunk.len() {
            out[i] = (trunk[i] + delta[i]).max(0.0);
        }
        out
    }

    #[test]
    fn parity_b_crossattn_matches_scalar_reference() {
        let mut rng = StdRng::seed_from_u64(0xB401);
        let cross = CrossAttn::new(AZ_TRUNK_CHANNELS, AZ_ENTITY_DIM, AZ_CROSS_HEADS, &mut rng);
        let trunk: Vec<f32> = (0..AZ_TRUNK_CHANNELS * AZ_CELLS_PADDED)
            .map(|i| ((i as i64 * 3119 + 23) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let entities: Vec<f32> = (0..AZ_ENTITY_TOKENS * AZ_ENTITY_DIM)
            .map(|i| ((i as i64 * 2293 + 41) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let actual = cross.forward(&trunk, &entities, AZ_ENTITY_TOKENS);
        let expected = crossattn_scalar_reference(&cross, &trunk, &entities, AZ_ENTITY_TOKENS);
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs < 5e-4,
            "CrossAttn SGEMM vs scalar-reference max-abs-diff = {max_abs} (limit 5e-4)"
        );
    }

    // ── Phase 0.8.E parity tests ────────────────────────────────────────

    /// `collect_bootstrap_v2` with rayon's parallel iterator produces the same
    /// samples as a serial loop over the same per-game seeds. Proves that the
    /// rayon refactor is observationally identical to the prior serial code —
    /// no race conditions, no determinism loss.
    #[test]
    fn parity_e_bootstrap_collect_par_equals_serial() {
        // Hand-seed N games, run each through the per-game function serially,
        // and compare to the parallel collect with the SAME seeds. The
        // parallel path goes through `into_par_iter().map(run_one_bootstrap_game).flatten().collect()`,
        // which (per rayon docs) preserves input order on `collect`.
        let n_games = 4usize;
        let seeds: Vec<u64> = (0..n_games).map(|i| 0xE0_0000 + i as u64).collect();

        // Serial reference: same per-game function called sequentially.
        let serial: Vec<AzSampleV2> = seeds
            .iter()
            .flat_map(|&s| run_one_bootstrap_game(s))
            .collect();

        // Parallel via rayon (production path).
        use rayon::prelude::*;
        let parallel: Vec<AzSampleV2> = seeds
            .clone()
            .into_par_iter()
            .map(run_one_bootstrap_game)
            .flatten()
            .collect();

        assert_eq!(
            serial.len(),
            parallel.len(),
            "sample count mismatch: serial={}, parallel={}",
            serial.len(),
            parallel.len()
        );
        for (idx, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(
                a.value.to_bits(),
                b.value.to_bits(),
                "sample {idx}: value drift {} vs {}",
                a.value,
                b.value
            );
            assert_eq!(a.input, b.input, "sample {idx}: input mismatch");
            assert_eq!(
                a.opp_inputs, b.opp_inputs,
                "sample {idx}: opp_inputs mismatch"
            );
            assert_eq!(a.entities, b.entities, "sample {idx}: entities mismatch");
            assert_eq!(a.policy, b.policy, "sample {idx}: policy mismatch");
        }
    }

    /// Same parity check for `collect_selfplay_v2`. The PUCT search inside
    /// each game uses a per-game `StdRng`, so determinism per-seed must hold.
    #[test]
    fn parity_e_selfplay_collect_par_equals_serial() {
        let net = AlphaZeroNetV2::new(small_config(), 0xE0_0010);
        let n_games = 3usize;
        let seeds: Vec<u64> = (0..n_games).map(|i| 0xE1_0000 + i as u64).collect();

        let serial: Vec<AzSampleV2> = seeds
            .iter()
            .flat_map(|&s| run_one_selfplay_game(&net, 4, 1.0, s))
            .collect();

        use rayon::prelude::*;
        let parallel: Vec<AzSampleV2> = seeds
            .clone()
            .into_par_iter()
            .map(|s| run_one_selfplay_game(&net, 4, 1.0, s))
            .flatten()
            .collect();

        assert_eq!(serial.len(), parallel.len());
        for (idx, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(
                a.value.to_bits(),
                b.value.to_bits(),
                "selfplay sample {idx} value drift"
            );
            assert_eq!(a.policy, b.policy, "selfplay sample {idx} policy mismatch");
            assert_eq!(
                a.candidates.len(),
                b.candidates.len(),
                "selfplay sample {idx} candidates len mismatch"
            );
        }
    }

    /// Same `AlphaZeroNetV2` evaluated twice on the same game must produce
    /// bit-identical results (proves the forward path itself is deterministic;
    /// any drift is from cross-build / cross-backend differences, not
    /// nondeterminism inside a single build).
    #[test]
    fn parity_b_forward_is_deterministic() {
        let mut rng = StdRng::seed_from_u64(0xB501);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let net = AlphaZeroNetV2::new(small_config(), 0xB502);
        let cands = candidate_moves_v2(&game, 8);
        let (v1, p1) = net.evaluate(&game, &cands);
        let (v2, p2) = net.evaluate(&game, &cands);
        assert_eq!(v1.to_bits(), v2.to_bits(), "value should be bit-identical");
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "policy probs should be bit-identical"
            );
        }
    }

    // ── Phase 0.8.A parity tests ────────────────────────────────────────

    /// The static `BIAS_PLANE` matches the loop-based construction it
    /// replaced: 1.0 on real cells (0..127), 0.0 on the pad cell.
    #[test]
    fn parity_a_bias_plane_static() {
        for li in 0..AZ_LOCAL_CELLS {
            assert_eq!(BIAS_PLANE[li], 1.0, "real cell {} should be 1.0", li);
        }
        assert_eq!(BIAS_PLANE[AZ_PAD_INDEX], 0.0, "pad cell should be 0.0");
    }

    /// The wrapper `az_search_v2` (default dispatch) returns identical
    /// visits to `az_search_v2_serial` when the env var is unset. Locks
    /// in that the new dispatch wrapper is a no-op in the default path.
    #[test]
    fn parity_a_search_default_dispatch_is_serial() {
        // `should_use_root_parallel_v2` reads `CASCADIA_AZ_PARALLEL`; unset
        // means the wrapper takes the serial branch. We verify by calling
        // `az_search_v2_serial` directly with the same seeded RNG and
        // confirming it produces the same visits[] / selected move.
        let mut rng1 = StdRng::seed_from_u64(0x4001);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng1);
        let net = AlphaZeroNetV2::new(small_config(), 0x4002);

        let r_serial = {
            let mut rng = StdRng::seed_from_u64(0x4003);
            az_search_v2_serial(&game, &net, 8, 0.0, &mut rng).unwrap()
        };
        let r_wrapper = {
            let mut rng = StdRng::seed_from_u64(0x4003);
            az_search_v2(&game, &net, 8, 0.0, &mut rng).unwrap()
        };
        assert_eq!(r_serial.visits, r_wrapper.visits);
        assert!(same_move(&r_serial.selected, &r_wrapper.selected));
    }

    /// `az_search_v2_root_parallel(workers=1)` falls through to the serial
    /// path and produces a bit-identical result. Sanity-checks the no-op
    /// parallelism case.
    #[test]
    fn parity_a_root_parallel_workers1_equals_serial() {
        let mut rng1 = StdRng::seed_from_u64(0x4101);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng1);
        let net = AlphaZeroNetV2::new(small_config(), 0x4102);

        let r_serial = {
            let mut rng = StdRng::seed_from_u64(0x4103);
            az_search_v2_serial(&game, &net, 8, 0.0, &mut rng).unwrap()
        };
        let r_par = {
            let mut rng = StdRng::seed_from_u64(0x4103);
            az_search_v2_root_parallel(&game, &net, 8, 1, 0.0, &mut rng).unwrap()
        };
        assert_eq!(r_serial.visits, r_par.visits);
        assert!(same_move(&r_serial.selected, &r_par.selected));
    }

    /// Stage 0.8.C parity: `az_search_v2_batched` with K=1 (batch size 1)
    /// must produce bit-identical visits[] / selected move to the serial
    /// path. Each iteration: descend → forward → expand+backup. With
    /// virtual_loss always cleared inside one iteration before the next
    /// descent begins, the math collapses to the serial PUCT formulation.
    #[test]
    fn parity_c_batched_k1_equals_serial() {
        let mut rng1 = StdRng::seed_from_u64(0x5001);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng1);
        let net = AlphaZeroNetV2::new(small_config(), 0x5002);

        let r_serial = {
            let mut rng = StdRng::seed_from_u64(0x5003);
            az_search_v2_serial(&game, &net, 16, 0.0, &mut rng).unwrap()
        };
        let r_batched = {
            let mut rng = StdRng::seed_from_u64(0x5003);
            az_search_v2_batched(&game, &net, 16, 1, 0.0, &mut rng).unwrap()
        };
        assert_eq!(
            r_serial.visits, r_batched.visits,
            "K=1 visits must match serial"
        );
        assert!(same_move(&r_serial.selected, &r_batched.selected));
    }

    /// Stage 0.8.C invariant: after `az_search_v2_batched` finishes, every
    /// edge's `virtual_loss` must be zero. Any non-zero leak indicates a
    /// descent that wasn't paired with a backup — would corrupt PUCT
    /// selection on subsequent searches sharing the tree.
    ///
    /// Walks the tree recursively counting non-zero vloss after a K=4
    /// batched search.
    #[test]
    fn parity_c_virtual_loss_zero_after_search() {
        let mut rng1 = StdRng::seed_from_u64(0x5101);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng1);
        let net = AlphaZeroNetV2::new(small_config(), 0x5102);

        // We need to inspect the internal tree. The public search APIs
        // throw away the tree, so this test instead runs a custom mini-loop
        // that exposes the root for inspection.
        let root_player = game.current_player;
        let mut root = Node::new();
        let mut rng = StdRng::seed_from_u64(0x5103);
        // 32 sims, batch K=4 → 8 batches. Exercises terminal/abort/forward
        // mixes plus normal expansion.
        let total = 32usize;
        let k_cap = 4usize;
        let c_puct = net.cfg.c_puct;
        let max_candidates = net.cfg.max_candidates;
        let mut done = 0usize;
        while done < total {
            let k = k_cap.min(total - done);
            let mut pending: Vec<(Vec<usize>, GameState)> = Vec::with_capacity(k);
            for _ in 0..k {
                match descend_one(&mut root, &game, root_player, &mut rng, c_puct) {
                    DescentResult::NeedsForward { path, game: g } => pending.push((path, g)),
                    DescentResult::BackupOnly { path, value } => {
                        backup_path(&mut root, &path, value);
                    }
                }
                done += 1;
            }
            if !pending.is_empty() {
                let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
                let mut opps: Vec<Vec<Vec<f32>>> = Vec::with_capacity(pending.len());
                let mut ents: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
                let mut phases: Vec<[f32; AZ_VALUE_PHASES]> = Vec::with_capacity(pending.len());
                for (_, g) in &pending {
                    let (input, opp, ent) = encode_game_local(g);
                    inputs.push(input);
                    opps.push(opp);
                    ents.push(ent);
                    phases.push(phase_one_hot(g));
                }
                let caches = net.forward_batch(&inputs, &opps, &ents, &phases);
                for ((path, g), cache) in pending.iter().zip(caches.iter()) {
                    expand_and_backup_leaf(&mut root, path, g, cache, max_candidates);
                }
            }
        }

        // Walk the tree; assert all virtual_loss == 0.
        fn walk(node: &Node, leaks: &mut usize) {
            for edge in &node.edges {
                if edge.virtual_loss != 0 {
                    *leaks += 1;
                }
                if let Some(child) = &edge.child {
                    walk(child, leaks);
                }
            }
        }
        let mut leaks = 0usize;
        walk(&root, &mut leaks);
        assert_eq!(
            leaks, 0,
            "virtual_loss leaked on {} edges after batched search",
            leaks
        );
    }

    /// Stage 0.8.C dispatch: with `CASCADIA_AZ_BATCH_K` unset (default 1),
    /// the public `az_search_v2` wrapper goes through the serial path and
    /// matches `az_search_v2_serial` bit-for-bit. With env set to K=1
    /// explicitly we *still* take the serial branch (the dispatch guards
    /// on `batch_k > 1`), so this also verifies the K=1 explicit-no-op path.
    #[test]
    fn parity_c_dispatch_default_is_serial() {
        // SAFETY: clears any leftover env from prior tests in the same binary.
        std::env::remove_var("CASCADIA_AZ_BATCH_K");
        std::env::remove_var("CASCADIA_AZ_PARALLEL");
        let mut rng1 = StdRng::seed_from_u64(0x5201);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng1);
        let net = AlphaZeroNetV2::new(small_config(), 0x5202);

        let r_serial = {
            let mut rng = StdRng::seed_from_u64(0x5203);
            az_search_v2_serial(&game, &net, 8, 0.0, &mut rng).unwrap()
        };
        let r_wrapper = {
            let mut rng = StdRng::seed_from_u64(0x5203);
            az_search_v2(&game, &net, 8, 0.0, &mut rng).unwrap()
        };
        assert_eq!(r_serial.visits, r_wrapper.visits);
        assert!(same_move(&r_serial.selected, &r_wrapper.selected));
    }

    /// Stage 0.8.C behavior: `forward_batch` over K independent positions
    /// returns the same K caches as K independent `forward` calls. Locks in
    /// the contract before the SGEMM-batched follow-up rewrites the body.
    #[test]
    fn parity_c_forward_batch_equals_serial_forward() {
        let mut rng1 = StdRng::seed_from_u64(0x5301);
        let net = AlphaZeroNetV2::new(small_config(), 0x5302);

        // Build 3 distinct games (just different RNG seeds → different
        // initial markets / bags).
        let games: Vec<GameState> = (0..3)
            .map(|i| {
                let mut r = StdRng::seed_from_u64(0x5400 + i);
                GameState::new(4, ScoringCards::all_a(), &mut r)
            })
            .collect();
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut opps: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut ents: Vec<Vec<f32>> = Vec::new();
        let mut phases: Vec<[f32; AZ_VALUE_PHASES]> = Vec::new();
        for g in &games {
            let (input, opp, ent) = encode_game_local(g);
            inputs.push(input);
            opps.push(opp);
            ents.push(ent);
            phases.push(phase_one_hot(g));
        }
        let batched = net.forward_batch(&inputs, &opps, &ents, &phases);
        let serial: Vec<ForwardCacheV2> = (0..games.len())
            .map(|i| net.forward(&inputs[i], &opps[i], &ents[i], phases[i]))
            .collect();
        assert_eq!(batched.len(), serial.len());
        for (b, s) in batched.iter().zip(serial.iter()) {
            assert!(
                (b.value - s.value).abs() < 1e-6,
                "value: {} vs {}",
                b.value,
                s.value
            );
            assert_eq!(b.tile_logits.len(), s.tile_logits.len());
            for (a, c) in b.tile_logits.iter().zip(s.tile_logits.iter()) {
                assert!((a - c).abs() < 1e-6);
            }
            for (a, c) in b.wildlife_logits.iter().zip(s.wildlife_logits.iter()) {
                assert!((a - c).abs() < 1e-6);
            }
            for (a, c) in b.market_logits.iter().zip(s.market_logits.iter()) {
                assert!((a - c).abs() < 1e-6);
            }
            for (a, c) in b
                .wildlife_market_logits
                .iter()
                .zip(s.wildlife_market_logits.iter())
            {
                assert!((a - c).abs() < 1e-6);
            }
            assert!((b.skip_logit - s.skip_logit).abs() < 1e-6);
        }
    }

    /// Root-parallel with multiple workers produces a valid SearchResult:
    /// total visits equal `sims − workers` (each worker's first sim
    /// expands its root without incrementing any edge), the selected
    /// move is in the candidate set. (Distribution differs from serial
    /// because each worker explores its own tree — that's expected.)
    #[test]
    fn behavior_a_root_parallel_aggregates_visits() {
        let mut rng1 = StdRng::seed_from_u64(0x4201);
        let game = GameState::new(4, ScoringCards::all_a(), &mut rng1);
        let net = AlphaZeroNetV2::new(small_config(), 0x4202);

        let sims = 16usize;
        let workers = 4usize;
        let mut rng = StdRng::seed_from_u64(0x4203);
        let result = az_search_v2_root_parallel(&game, &net, sims, workers, 0.0, &mut rng).unwrap();
        let total: u32 = result.visits.iter().sum();
        // Per-worker: each `sims_per_worker` budget produces `sims_per_worker − 1`
        // root-edge visits (the first sim expands the root without touching
        // any edge). Summed over workers: `sims − workers`.
        let expected = sims - workers;
        assert_eq!(total, expected as u32);
        assert!(!result.candidates.is_empty());
        assert!(result
            .candidates
            .iter()
            .any(|m| same_move(m, &result.selected)));
        // No NaN / negative visits / out-of-range policy.
        assert!(result
            .visit_policy
            .iter()
            .all(|&p| p.is_finite() && (0.0..=1.0).contains(&p)));
    }

    /// Lock the per-head aux-value scales so the calibration doesn't silently
    /// regress to the old uniform `[30×10, 8, 10×5]` array.
    #[test]
    fn aux_scales_calibrated() {
        // Wildlife
        assert_eq!(AZ_AUX_SCALES[0], 13.0, "bear aux scale");
        assert_eq!(AZ_AUX_SCALES[2], 25.0, "salmon aux scale");
        assert_eq!(AZ_AUX_SCALES[3], 25.0, "hawk aux scale");
        assert_eq!(AZ_AUX_SCALES[4], 12.0, "fox aux scale");
        // Habitat per terrain
        for i in 5..10 {
            assert_eq!(AZ_AUX_SCALES[i], 15.0, "habitat[{}] aux scale", i - 5);
        }
        // Nature tokens
        assert_eq!(AZ_AUX_SCALES[10], 5.0, "nature tokens aux scale");
        // Habitat bonus per terrain
        for i in 11..16 {
            assert_eq!(AZ_AUX_SCALES[i], 3.0, "habitat_bonus[{}] aux scale", i - 11);
        }
    }
}
