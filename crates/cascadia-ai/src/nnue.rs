//! NNUE (Efficiently Updatable Neural Network) for Cascadia board evaluation.
//!
//! Architecture: sparse binary inputs → 512 ReLU → 64 ReLU → 1 scalar
//!
//! Feature layout:
//!   [0 .. 4851)       Per-cell: 441 cells × 11 features (wildlife/terrain)
//!   [4851 .. 4872)    Game phase: turn number one-hot (21 features, 0-20)
//!   [4872 .. 4881)    Nature tokens one-hot (9 features, 0-8)
//!   [4881 .. 4911)    Wildlife count per type, one-hot (6 × 5 types, counts 0-5)
//!   [4911 .. 4961)    Largest habitat group per terrain, one-hot (10 × 5 terrains, 0-9)
//!   [4961 .. 5108)    Pairwise adjacency: 3 directions × 49 wildlife pair states
//!   [5108 .. 5197)    Wildlife pattern features (bear pairs, elk lines, etc.)
//!   [5197 .. 5252)    Bag remaining: 5 types × 11 bins (0-10+ available to draw)
//!   [5252 .. 5307)    Opponent habitat: 5 terrains × 11 bins (max opponent habitat 0-10+)

use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, ADJACENCY, GRID_SIZE};
use cascadia_core::types::Wildlife;

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

const FEATURES_PER_CELL: usize = 11;
const CELL_FEATURES: usize = GRID_SIZE * FEATURES_PER_CELL; // 4851

const TURN_FEATURES: usize = 21;     // turns 0-20
const TOKEN_FEATURES: usize = 9;     // tokens 0-8
const WL_COUNT_FEATURES: usize = 30; // 6 bins × 5 types (counts 0-5)
const HAB_SIZE_FEATURES: usize = 50; // 10 bins × 5 terrains (sizes 0-9)
const PHASE_FEATURES: usize = TURN_FEATURES + TOKEN_FEATURES + WL_COUNT_FEATURES + HAB_SIZE_FEATURES; // 110

const PAIR_DIRS: usize = 3;
const PAIR_STATES: usize = 7 * 7;    // 49 (my_wildlife × neighbor_wildlife)
const PAIR_FEATURES: usize = PAIR_DIRS * PAIR_STATES; // 147

// Wildlife pattern features (directly expose scoring-relevant info)
const BEAR_PAIR_FEATURES: usize = 5;     // bear pair count one-hot (0-4)
const ELK_LINE_FEATURES: usize = 20;     // elk lines: 5 length bins × 4 max lines
const SALMON_RUN_FEATURES: usize = 24;   // salmon runs: 8 length bins × 3 max runs
const HAWK_ISO_FEATURES: usize = 9;      // isolated hawk count one-hot (0-8)
const FOX_DIV_FEATURES: usize = 6;       // avg fox diversity one-hot (0-5)
const EMPTY_SLOTS_FEATURES: usize = 25;  // empty wildlife slots per type: 5 bins × 5 types
const PATTERN_FEATURES: usize = BEAR_PAIR_FEATURES + ELK_LINE_FEATURES
    + SALMON_RUN_FEATURES + HAWK_ISO_FEATURES + FOX_DIV_FEATURES + EMPTY_SLOTS_FEATURES; // 89

// Bag remaining features: how many of each wildlife type are available to draw
// 5 types × 11 bins (0,1,2,...,9,10+) = 55 features
// "Available" = bag + deferred returns (tokens that will come back at end of turn)
const BAG_BINS: usize = 11;
const BAG_FEATURES: usize = 5 * BAG_BINS; // 55

// Opponent habitat features: max opponent habitat size per terrain
// 5 terrains × 11 bins (0-10+) = 55 features
// Tells the NNUE what it needs to beat for habitat majority bonuses
const OPP_HAB_BINS: usize = 11;
const OPP_HAB_FEATURES: usize = 5 * OPP_HAB_BINS; // 55

/// Feature count without bag/opponent features (for backward-compatible weight loading)
pub const NUM_FEATURES_LEGACY: usize = CELL_FEATURES + PHASE_FEATURES + PAIR_FEATURES + PATTERN_FEATURES;
pub const NUM_FEATURES: usize = NUM_FEATURES_LEGACY + BAG_FEATURES + OPP_HAB_FEATURES;
pub const HIDDEN1: usize = 512;
pub const HIDDEN2: usize = 64;

// ─────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────

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

/// Game-level information visible to the AI beyond the player's own board:
/// bag composition and opponent habitat sizes.
#[derive(Clone, Default)]
pub struct BagInfo {
    /// Remaining drawable count per wildlife type [Bear, Elk, Salmon, Hawk, Fox]
    pub remaining: [u8; 5],
    /// Max opponent habitat size per terrain [Forest, Prairie, Wetland, Mountain, River]
    pub max_opponent_habitat: [u8; 5],
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

        BagInfo { remaining, max_opponent_habitat }
    }
}

/// Extract active feature indices from a board state (without bag info).
pub fn extract_features(board: &Board) -> Vec<u16> {
    extract_features_with_bag(board, None)
}

/// Extract active feature indices from a board state, optionally with bag composition.
pub fn extract_features_with_bag(board: &Board, bag: Option<&BagInfo>) -> Vec<u16> {
    let mut features = Vec::with_capacity(120);

    // ── Per-cell features ──
    for &tile_idx in board.placed_tiles.iter() {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        let base = idx * FEATURES_PER_CELL;

        if let Some(w) = cell.placed_wildlife() {
            features.push((base + w as usize) as u16);
        } else {
            features.push((base + 5) as u16);
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

    // Wildlife count per type (clamped 0-5)
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

    features
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
    /// Output layer weights: HIDDEN2 → 1
    pub w3: Vec<f32>,      // [HIDDEN2]
    pub b3: f32,
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

        NNUENetwork { w1, b1, w2, b2, w3, b3 }
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

        // Output
        let mut out = self.b3;
        for j in 0..HIDDEN2 {
            out += h2[j] * self.w3[j];
        }

        out
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

    /// Save weights to a binary file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        file.write_all(b"NNUE")?;
        file.write_all(&1u32.to_le_bytes())?;

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

        // W3 + b3
        for &v in &self.w3 {
            file.write_all(&v.to_le_bytes())?;
        }
        file.write_all(&self.b3.to_le_bytes())?;

        Ok(())
    }

    /// Load weights from a binary file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"NNUE" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;

        let mut buf = [0u8; 4];
        let mut read_f32 = |f: &mut std::fs::File| -> std::io::Result<f32> {
            f.read_exact(&mut buf)?;
            Ok(f32::from_le_bytes(buf))
        };

        // Detect file size to handle legacy weights (without bag features)
        let file_size = file.metadata()?.len();
        let header_size = 8u64; // magic + version
        let legacy_w1_size = (NUM_FEATURES_LEGACY * HIDDEN1) as u64 * 4;
        let new_w1_size = (NUM_FEATURES * HIDDEN1) as u64 * 4;
        let rest_size = ((HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 + HIDDEN2 + 1) as u64) * 4;
        let is_legacy = file_size < header_size + new_w1_size + rest_size;
        let w1_features = if is_legacy { NUM_FEATURES_LEGACY } else { NUM_FEATURES };

        let mut w1 = Vec::with_capacity(NUM_FEATURES * HIDDEN1);
        for _ in 0..w1_features * HIDDEN1 {
            w1.push(read_f32(&mut file)?);
        }
        // Pad new features with zeros if loading legacy weights
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

        Ok(NNUENetwork { w1, b1, w2, b2, w3, b3 })
    }
}
