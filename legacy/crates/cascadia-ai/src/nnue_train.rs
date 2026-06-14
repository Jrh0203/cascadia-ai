//! NNUE training: generate self-play data, train network with mini-batch SGD.

use std::sync::Arc;
use std::thread;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

use crate::nnue::{extract_features, extract_phase_pattern_features, NNUENetwork};
use crate::score_target::{is_all_a, ScoreTarget};
use crate::search::{execute_scored_move, greedy_move};

/// A training sample: board features + target score(s).
///
/// `target` is the main value target (final_score - current_score), i.e. the TOTAL
/// remaining points to gain from this afterstate to game end. For MCV3 format,
/// this is the base score (no habitat majority bonuses). For MCV4 (v5 split-head
/// training), this is the bonus-INCLUDED total: the sum of `subscore_targets`.
///
/// `target_wildlife` is the WILDLIFE-only remaining points (final_wildlife - current_wildlife).
/// The habitat+tokens remaining is derived as `target - target_wildlife`. Stored as a
/// separate field to avoid recomputation during multi-head training. v5 2-head split
/// uses this for the wildlife value head target.
///
/// `subscore_targets` (v5 / MCV4): per-subscore remaining deltas for 11-head split
/// training. Layout matches `nnue::HEAD_*` constants:
///   [0..5]:  per-wildlife (bear, elk, salmon, hawk, fox)
///   [5..10]: per-terrain hab + final-bonus (forest, prairie, wetland, mountain, river)
///   [10]:    nature tokens
/// Sum equals `target` for MCV4 data. For MCV3-loaded data, all zero.
///
/// `aux_bear` and `aux_salmon` are auxiliary regression targets used by
/// multi-head training (v4 architecture). They are the FINAL bear pair count
/// and FINAL longest salmon chain length for the game this sample came from.
#[derive(Clone)]
pub struct Sample {
    pub features: Vec<u16>,
    pub target: f32,
    pub aux_bear: f32,
    pub aux_salmon: f32,
    pub target_wildlife: f32,
    pub subscore_targets: [f32; crate::nnue::NUM_HEADS],
}

/// Count isolated bear pairs (connected components of size exactly 2).
pub fn count_bear_pairs(board: &cascadia_core::board::Board) -> usize {
    use cascadia_core::hex::ADJACENCY;
    use cascadia_core::types::Wildlife;
    let adj = &*ADJACENCY;
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut visited = [false; 441];
    let mut pairs = 0;
    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut size = 0;
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(cur) = queue.pop() {
            size += 1;
            for nidx in adj.neighbors_of(cur as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        if size == 2 {
            pairs += 1;
        }
    }
    pairs
}

/// Find the longest valid salmon chain (component where each cell has ≤2 salmon neighbors).
pub fn longest_salmon_chain(board: &cascadia_core::board::Board) -> usize {
    use cascadia_core::hex::ADJACENCY;
    use cascadia_core::types::Wildlife;
    let adj = &*ADJACENCY;
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let mut visited = [false; 441];
    let mut max_len = 0;
    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(cur) = queue.pop() {
            component.push(cur);
            for nidx in adj.neighbors_of(cur as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        let valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count()
                <= 2
        });
        if valid && component.len() > max_len {
            max_len = component.len();
        }
    }
    max_len
}

/// A completed game's training data: samples + final score.
pub struct GameResult {
    pub samples: Vec<Sample>,
    pub final_score: u16,
}

/// Generate self-play games, returning per-game results (for top-% filtering).
/// Uses ε-greedy sampling. For SA softmax sampling, use `generate_games_with_mode`.
pub fn generate_games(
    num_games: usize,
    seed: u64,
    net: Option<&NNUENetwork>,
    epsilon: f32,
    num_players: usize,
) -> Vec<GameResult> {
    generate_games_with_mode(
        num_games,
        seed,
        net,
        SamplingMode::EpsilonGreedy(epsilon),
        num_players,
    )
}

pub fn generate_games_with_mode(
    num_games: usize,
    seed: u64,
    net: Option<&NNUENetwork>,
    mode: SamplingMode,
    num_players: usize,
) -> Vec<GameResult> {
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let games_per_thread = (num_games + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let n = if t < num_threads - 1 {
                games_per_thread.min(num_games.saturating_sub(t * games_per_thread))
            } else {
                num_games.saturating_sub(t * games_per_thread)
            };
            let thread_seed = seed.wrapping_add(t as u64 * 1000000);
            let net_clone = net.cloned();
            let mode = mode;
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(thread_seed);
                let mut results = Vec::with_capacity(n);
                for _ in 0..n {
                    let game_seed = rng.gen::<u64>();
                    results.push(generate_single_game(
                        game_seed,
                        net_clone.as_ref(),
                        mode,
                        num_players,
                    ));
                }
                results
            })
        })
        .collect();

    let mut all_results = Vec::with_capacity(num_games);
    for handle in handles {
        all_results.extend(handle.join().unwrap());
    }
    all_results
}

/// Per-afterstate intermediate state captured during play. Captures everything
/// needed to compute per-subscore (final - current) deltas at game end.
struct Afterstate {
    features: Vec<u16>,
    /// Single-player base score (no habitat bonus); legacy `target` field source.
    current_total: u16,
    /// Wildlife-only sum (legacy target_wildlife field source).
    current_wildlife: u16,
    /// Per-wildlife sub-scores at this afterstate.
    current_wildlife_per: [u16; 5],
    /// Per-terrain habitat sizes (= per-terrain habitat sub-score; bonus is 0 mid-game).
    current_hab_per: [u16; 5],
    /// Nature tokens held by player 0.
    current_tokens: u16,
}

/// Parse "A,B,C,A,D" → ScoringCards (Bear, Elk, Salmon, Hawk, Fox order).
fn parse_train_cards(s: &str) -> Option<ScoringCards> {
    use cascadia_core::types::ScoringCardVariant;
    let parts: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
    if parts.len() != 5 {
        return None;
    }
    let mut cards = [ScoringCardVariant::A; 5];
    for (i, p) in parts.iter().enumerate() {
        cards[i] = match p.to_ascii_uppercase().as_str() {
            "A" => ScoringCardVariant::A,
            "B" => ScoringCardVariant::B,
            "C" => ScoringCardVariant::C,
            "D" => ScoringCardVariant::D,
            _ => return None,
        };
    }
    Some(ScoringCards { cards })
}

/// Generate one game, return its samples and final score.
fn generate_single_game(
    seed: u64,
    net: Option<&NNUENetwork>,
    mode: SamplingMode,
    num_players: usize,
) -> GameResult {
    use cascadia_core::types::Terrain;
    let mut samples = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    // Allow CASCADIA_SCORING_CARDS env override so training data can be
    // produced with alt scoring (e.g. for cards-alt builds).
    let cards = std::env::var("CASCADIA_SCORING_CARDS")
        .ok()
        .and_then(|s| parse_train_cards(&s))
        .unwrap_or_else(ScoringCards::all_a);
    let mut game = GameState::new(num_players, cards, &mut rng);
    let mut afterstates: Vec<Afterstate> = Vec::with_capacity(20);

    while !game.is_game_over() {
        if game.current_player != 0 {
            // Opponents ALWAYS take the free 3-of-a-kind replacement if available.
            // Must match the inference-time opponent behavior (CLI bench +
            // cascadia-web) so training data reflects realistic opponent play.
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            match greedy_move(&game) {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) {
                        break;
                    }
                }
                None => break,
            }
            continue;
        }
        greedy_pre_move(&mut game, &mut rng);
        let mv = match mode {
            SamplingMode::EpsilonGreedy(epsilon) => {
                if epsilon > 0.0 && rng.gen::<f32>() < epsilon {
                    pick_random_move(&game, &mut rng)
                } else {
                    match net {
                        Some(n) => pick_best_move_nnue(&game, n),
                        None => greedy_move(&game),
                    }
                }
            }
            SamplingMode::Softmax(temperature) => match net {
                Some(n) => pick_softmax_move_nnue(&game, n, temperature, &mut rng),
                None => greedy_move(&game),
            },
        };
        match mv {
            Some(mv) => {
                if !execute_scored_move(&mut game, &mv) {
                    break;
                }
            }
            None => break,
        }
        let bd = ScoreBreakdown::compute(&mut game.boards[0], &game.scoring_cards);
        // Use player 0's POV for BagInfo so opp_detail ordering matches the
        // features extracted from boards[0]. from_game() would use whoever
        // current_player happens to be (usually opponents post-move).
        let bag_info = crate::nnue::BagInfo::from_game_for_player(&game, 0);
        let mut hab_per = [0u16; 5];
        for t in Terrain::ALL {
            hab_per[t as usize] = bd.habitat[t as usize];
        }
        afterstates.push(Afterstate {
            features: crate::nnue::extract_features_with_bag(&game.boards[0], Some(&bag_info)),
            current_total: bd.total,
            current_wildlife: bd.wildlife_total(),
            current_wildlife_per: bd.wildlife,
            current_hab_per: hab_per,
            current_tokens: bd.nature_tokens,
        });
    }

    // Compute per-subscore final values WITH habitat majority bonus baked into
    // the per-terrain head (the bonus is rank-based vs. opponents and only
    // determinable at game end). Sum of per-subscore deltas equals
    // `final_total_with_bonus - current_total_no_bonus`.
    let final_bd_with_bonus = if num_players >= 2 {
        ScoreBreakdown::compute_with_bonuses(&mut game.boards, &game.scoring_cards, 0)
    } else {
        ScoreBreakdown::compute(&mut game.boards[0], &game.scoring_cards)
    };
    let final_score_base = ScoreBreakdown::compute(&mut game.boards[0], &game.scoring_cards).total;
    let final_score_with_bonus = final_bd_with_bonus.total;
    let final_wildlife = final_bd_with_bonus.wildlife_total();
    let final_wildlife_per = final_bd_with_bonus.wildlife;
    let mut final_hab_plus_bonus = [0u16; 5];
    for t in Terrain::ALL {
        let ti = t as usize;
        final_hab_plus_bonus[ti] =
            final_bd_with_bonus.habitat[ti] + final_bd_with_bonus.habitat_bonus[ti] as u16;
    }
    let final_tokens = final_bd_with_bonus.nature_tokens;
    let final_bear_pairs = count_bear_pairs(&game.boards[0]) as f32;
    let final_salmon_chain = longest_salmon_chain(&game.boards[0]) as f32;
    for st in afterstates {
        // Legacy `target` = base score remaining (no bonus) for back-compat with
        // MCV3-trained models loading via target field. This MAY differ from the
        // sum of subscore_targets (which IS bonus-included).
        let remaining_base = final_score_base.saturating_sub(st.current_total) as f32;
        let remaining_wildlife = final_wildlife.saturating_sub(st.current_wildlife) as f32;
        let mut subscore: [f32; crate::nnue::NUM_HEADS] = [0.0; crate::nnue::NUM_HEADS];
        // Heads 0..5: per-wildlife
        for w in 0..5 {
            subscore[w] = (final_wildlife_per[w] as i32 - st.current_wildlife_per[w] as i32) as f32;
        }
        // Heads 5..10: per-terrain (hab + bonus)
        for t in 0..5 {
            subscore[5 + t] =
                (final_hab_plus_bonus[t] as i32 - st.current_hab_per[t] as i32) as f32;
        }
        // Head 10: tokens
        subscore[10] = (final_tokens as i32 - st.current_tokens as i32) as f32;
        // Sanity: target_with_bonus = sum of subscores
        let target_with_bonus = (final_score_with_bonus as i32 - st.current_total as i32) as f32;
        // We intentionally use bonus-included target for MCV4 — the sum of subscores.
        // The legacy `remaining_base` is only emitted in MCV3 paths (older callers).
        // Both kept for backward-compat readability; downstream chooses.
        let _ = remaining_base; // keep computation but use target_with_bonus below
        samples.push(Sample {
            features: st.features,
            target: target_with_bonus,
            aux_bear: final_bear_pairs,
            aux_salmon: final_salmon_chain,
            target_wildlife: remaining_wildlife,
            subscore_targets: subscore,
        });
    }

    GameResult {
        samples,
        final_score: final_score_base,
    }
}

/// Sampling mode for self-play move selection.
#[derive(Clone, Copy)]
pub enum SamplingMode {
    /// ε-greedy: with probability ε pick a random move, otherwise argmax.
    EpsilonGreedy(f32),
    /// Simulated annealing: softmax(score / temperature), sample from the distribution.
    /// T → 0 recovers argmax; T → ∞ approaches uniform sampling.
    Softmax(f32),
}

/// Generate training data from self-play games (flat, all games included).
/// `num_players`: 1 for pre-training (AI gets all turns), 4 for realistic play.
pub fn generate_samples(
    num_games: usize,
    seed: u64,
    net: Option<&NNUENetwork>,
    epsilon: f32,
    num_players: usize,
) -> Vec<Sample> {
    generate_samples_with_mode(
        num_games,
        seed,
        net,
        SamplingMode::EpsilonGreedy(epsilon),
        num_players,
    )
}

/// A single opponent kind in the training opponent pool.
#[derive(Clone)]
pub enum OpponentKind {
    /// Greedy play (the original default).
    Greedy,
    /// Frozen NNUE opponent (per-path weights, strong).
    Nnue(NNUENetwork),
    /// Draft uniformly at random from the market, then place greedily.
    Random,
    /// Draft the market slot whose wildlife is most scarce in the bag
    /// (measured from public info: tokens placed on any board).
    Scarcity,
    /// Draft weighted by a per-game random preference distribution over
    /// the 5 wildlife types. Uniform case reduces to Random.
    Preference,
}

/// Parse the CASCADIA_TRAIN_OPP_POOL env var into a Vec of opponent kinds.
/// Format: comma-separated list. Each entry is either a path to a `.bin`
/// weights file (interpreted as an Nnue opponent) or one of the tags
/// `greedy`, `random`, `scarcity`, `preference`.
fn parse_opp_pool(spec: &str) -> Vec<OpponentKind> {
    let mut pool = Vec::new();
    for raw in spec.split(',') {
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        let kind = match s.to_lowercase().as_str() {
            "greedy" => Some(OpponentKind::Greedy),
            "random" => Some(OpponentKind::Random),
            "scarcity" => Some(OpponentKind::Scarcity),
            "preference" => Some(OpponentKind::Preference),
            _ => None,
        };
        if let Some(k) = kind {
            pool.push(k);
        } else {
            // Treat as a path to NNUE weights
            match NNUENetwork::load(std::path::Path::new(s)) {
                Ok(n) => {
                    eprintln!("[train] Pool: loaded NNUE opponent from {}", s);
                    pool.push(OpponentKind::Nnue(n));
                }
                Err(e) => eprintln!("[train] Pool: failed to load {}: {} (skipped)", s, e),
            }
        }
    }
    pool
}

pub fn generate_samples_with_mode(
    num_games: usize,
    seed: u64,
    net: Option<&NNUENetwork>,
    mode: SamplingMode,
    num_players: usize,
) -> Vec<Sample> {
    // CASCADIA_TRAIN_PLAYER_MCE=<N> — have player 0 select moves via MCE(N)
    // during self-play data generation instead of single-ply NNUE argmax.
    // Aligns training state distribution with eval state distribution (where
    // player 0 also plays MCE). Costs ~3× sample-gen time per rollout budget.
    let player_mce: Option<usize> = std::env::var("CASCADIA_TRAIN_PLAYER_MCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0);
    if let Some(r) = player_mce {
        eprintln!("[train] Player 0 uses MCE({}) instead of NNUE-direct", r);
    }

    // Resolve the opponent pool in priority order:
    //   1. CASCADIA_TRAIN_OPP_POOL (multi-opponent, per-game sampling)
    //   2. CASCADIA_TRAIN_OPP_WEIGHTS (single frozen NNUE — legacy)
    //   3. Default: greedy opponent (original behavior)
    let pool: Vec<OpponentKind> = if let Ok(spec) = std::env::var("CASCADIA_TRAIN_OPP_POOL") {
        if !spec.is_empty() {
            let p = parse_opp_pool(&spec);
            if p.is_empty() {
                eprintln!("[train] OPP_POOL parsed empty, falling back to greedy");
                vec![OpponentKind::Greedy]
            } else {
                eprintln!(
                    "[train] Opponent pool has {} entries (per-seat independent sampling)",
                    p.len()
                );
                if std::env::var("CASCADIA_TRAIN_OPP_WEIGHTS").is_ok() {
                    eprintln!("[train] NOTE: OPP_WEIGHTS is also set but OPP_POOL wins");
                }
                p
            }
        } else {
            vec![OpponentKind::Greedy]
        }
    } else if let Ok(p) = std::env::var("CASCADIA_TRAIN_OPP_WEIGHTS") {
        if p.is_empty() {
            vec![OpponentKind::Greedy]
        } else {
            match NNUENetwork::load(std::path::Path::new(&p)) {
                Ok(n) => {
                    eprintln!("[train] Loaded frozen opponent net from {}", p);
                    vec![OpponentKind::Nnue(n)]
                }
                Err(e) => {
                    eprintln!(
                        "[train] Failed to load opponent weights {}: {} (using greedy)",
                        p, e
                    );
                    vec![OpponentKind::Greedy]
                }
            }
        }
    } else {
        vec![OpponentKind::Greedy]
    };

    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let games_per_thread = (num_games + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let n = if t < num_threads - 1 {
                games_per_thread.min(num_games.saturating_sub(t * games_per_thread))
            } else {
                num_games.saturating_sub(t * games_per_thread)
            };
            let thread_seed = seed.wrapping_add(t as u64 * 1000000);
            let net_clone = net.cloned();
            let pool_clone = pool.clone();
            let mode = mode;
            let num_players = num_players;
            let player_mce_t = player_mce;
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(thread_seed);
                let mut samples = Vec::with_capacity(n * 20);

                for _ in 0..n {
                    let game_seed = rng.gen::<u64>();
                    // Per-SEAT independent sampling: each of (num_players - 1) opponent
                    // seats picks its opponent kind uniformly from the pool. Produces
                    // heterogeneous market conditions within a single game — sometimes
                    // all 3 opponents happen to be random, sometimes all strong, often
                    // mixed. This is strictly more opponent variety than per-game
                    // sampling and the diversity the learner actually needs.
                    let seat_opps: Vec<OpponentKind> = (0..num_players.saturating_sub(1))
                        .map(|_| pool_clone[rng.gen_range(0..pool_clone.len())].clone())
                        .collect();
                    generate_game_samples(
                        &mut samples,
                        game_seed,
                        net_clone.as_ref(),
                        &seat_opps,
                        mode,
                        num_players,
                        player_mce_t,
                    );
                }

                samples
            })
        })
        .collect();

    let mut all_samples = Vec::with_capacity(num_games * 20);
    for handle in handles {
        all_samples.extend(handle.join().unwrap());
    }
    all_samples
}

/// Play one game, record all AI afterstates, label with final score.
///
/// Exact-late-labels (K=1): for the LAST AI turn, the AI's epsilon-random move
/// can produce sub-optimal labels for sample N-2 (the state with 1 move remaining).
/// We compute the GREEDY-optimal move 20 separately and use its resulting score
/// as the label for sample N-2 instead of the AI's actual (sometimes random) move.
/// This is sound because move 20 is a leaf — no future to worry about, so greedy
/// IS optimal. K=1 is essentially free (one extra greedy_move call per game).
fn generate_game_samples(
    samples: &mut Vec<Sample>,
    seed: u64,
    net: Option<&NNUENetwork>,
    seat_opps: &[OpponentKind],
    mode: SamplingMode,
    num_players: usize,
    player_mce: Option<usize>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let cards = std::env::var("CASCADIA_SCORING_CARDS")
        .ok()
        .and_then(|s| parse_train_cards(&s))
        .unwrap_or_else(ScoringCards::all_a);
    let mut game = GameState::new(num_players, cards, &mut rng);

    // CASCADIA_TRAIN_OPP_NNUE=1 — legacy flag. Honored only if EVERY seat in the
    // pool pick is Greedy (OPP_POOL / specific kinds otherwise dictate behavior).
    let opp_nnue: bool = seat_opps.iter().all(|o| matches!(o, OpponentKind::Greedy))
        && std::env::var("CASCADIA_TRAIN_OPP_NNUE")
            .ok()
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);

    // Per-seat preference vector for any Preference opponent. Sampled once at
    // game start and held constant — each preference seat has a consistent
    // drafting "character" across the 20 turns.
    let mut seat_prefs: Vec<Option<[f32; 5]>> = Vec::with_capacity(seat_opps.len());
    for opp in seat_opps {
        seat_prefs.push(if matches!(opp, OpponentKind::Preference) {
            Some(crate::draft_opponents::sample_preferences(&mut rng))
        } else {
            None
        });
    }

    // Collect afterstate features + per-subscore current values during the game.
    // Captures everything generate_game_samples needs to produce both legacy MCV3
    // targets (target, target_wildlife) AND v5 MCV4 per-subscore deltas.
    let mut afterstates: Vec<Afterstate> = Vec::with_capacity(20);
    // Saved state right before the AI's LAST move (for K=1 exact label)
    let mut state_before_last_ai_move: Option<GameState> = None;

    while !game.is_game_over() {
        if game.current_player != 0 {
            // Always take the free 3-of-a-kind replacement before drafting.
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            // seat_opps is indexed by (current_player - 1): seat 0 is the AI.
            let seat_idx = game.current_player - 1;
            let opp = &seat_opps[seat_idx.min(seat_opps.len() - 1)];
            let opp_mv = match opp {
                OpponentKind::Greedy => {
                    if opp_nnue {
                        match net {
                            Some(n) => pick_best_move_nnue(&game, n).or_else(|| greedy_move(&game)),
                            None => greedy_move(&game),
                        }
                    } else {
                        greedy_move(&game)
                    }
                }
                OpponentKind::Nnue(n) => {
                    pick_best_move_nnue(&game, n).or_else(|| greedy_move(&game))
                }
                OpponentKind::Random => crate::draft_opponents::random_draft_move(&game, &mut rng)
                    .or_else(|| greedy_move(&game)),
                OpponentKind::Scarcity => {
                    crate::draft_opponents::scarcity_draft_move(&game, &mut rng)
                        .or_else(|| greedy_move(&game))
                }
                OpponentKind::Preference => crate::draft_opponents::preference_draft_move(
                    &game,
                    seat_prefs[seat_idx.min(seat_prefs.len() - 1)]
                        .as_ref()
                        .unwrap(),
                    &mut rng,
                )
                .or_else(|| greedy_move(&game)),
            };
            match opp_mv {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) {
                        break;
                    }
                }
                None => break,
            }
            continue;
        }

        // Pre-move: simple greedy mulligan logic for training data generation
        greedy_pre_move(&mut game, &mut rng);

        // K=1 exact-label hook: if this is the AI's LAST turn (afterstates.len() == 19,
        // meaning 19 AI moves recorded so far, this is move 20), save the state.
        if afterstates.len() == 19 {
            state_before_last_ai_move = Some(game.clone());
        }

        // Select move according to sampling mode. When CASCADIA_TRAIN_PLAYER_MCE
        // is set and a net is available, use MCE(N) instead of NNUE-direct for
        // player 0's greedy pick — aligns training state distribution with the
        // distribution seen at eval time (where player 0 also plays MCE).
        let mv = match mode {
            SamplingMode::EpsilonGreedy(epsilon) => {
                if epsilon > 0.0 && rng.gen::<f32>() < epsilon {
                    pick_random_move(&game, &mut rng)
                } else if let (Some(rollouts), Some(n)) = (player_mce, net) {
                    let mut cands = crate::mce::expanded_candidates(&game);
                    if cands.len() > 8 {
                        cands = crate::mce::nnue_prefilter_candidates(&game, n, cands, 8);
                    }
                    crate::mce::best_move_nnue_rollout_mce(
                        &game,
                        n,
                        rollouts,
                        crate::mce::GreedyMceAlloc::SeqHalving,
                        cands,
                        &mut rng,
                    )
                } else {
                    match net {
                        Some(n) => pick_best_move_nnue(&game, n),
                        None => greedy_move(&game),
                    }
                }
            }
            SamplingMode::Softmax(temperature) => {
                match net {
                    Some(n) => pick_softmax_move_nnue(&game, n, temperature, &mut rng),
                    None => greedy_move(&game), // no net → fall back to greedy
                }
            }
        };
        match mv {
            Some(mv) => {
                if !execute_scored_move(&mut game, &mv) {
                    break;
                }
            }
            None => break,
        }

        // Record afterstate features + per-subscore current values for v5 split-head
        // training. BagInfo from player 0's POV so opp_detail ordering matches features.
        use cascadia_core::types::Terrain;
        let bd = ScoreBreakdown::compute(&mut game.boards[0], &game.scoring_cards);
        let bag_info = crate::nnue::BagInfo::from_game_for_player(&game, 0);
        let mut hab_per = [0u16; 5];
        for t in Terrain::ALL {
            hab_per[t as usize] = bd.habitat[t as usize];
        }
        afterstates.push(Afterstate {
            features: crate::nnue::extract_features_with_bag(&game.boards[0], Some(&bag_info)),
            current_total: bd.total,
            current_wildlife: bd.wildlife_total(),
            current_wildlife_per: bd.wildlife,
            current_hab_per: hab_per,
            current_tokens: bd.nature_tokens,
        });
    }

    // Final scores (with bonus included for split-head per-terrain target).
    use cascadia_core::types::Terrain;
    let final_bd_with_bonus = if num_players >= 2 {
        ScoreBreakdown::compute_with_bonuses(&mut game.boards, &game.scoring_cards, 0)
    } else {
        ScoreBreakdown::compute(&mut game.boards[0], &game.scoring_cards)
    };
    let final_bd = ScoreBreakdown::compute(&mut game.boards[0], &game.scoring_cards);
    let final_score = final_bd.total;
    let final_wildlife = final_bd.wildlife_total();
    let final_score_with_bonus = final_bd_with_bonus.total;
    let final_wildlife_per = final_bd_with_bonus.wildlife;
    let mut final_hab_plus_bonus = [0u16; 5];
    for t in Terrain::ALL {
        let ti = t as usize;
        final_hab_plus_bonus[ti] =
            final_bd_with_bonus.habitat[ti] + final_bd_with_bonus.habitat_bonus[ti] as u16;
    }
    let final_tokens = final_bd_with_bonus.nature_tokens;

    // Auxiliary targets: count final bear pairs and longest salmon chain.
    // All afterstates from this game share the same aux targets — they predict
    // the FINAL outcome of the game from any earlier state.
    let final_bear_pairs = count_bear_pairs(&game.boards[0]) as f32;
    let final_salmon_chain = longest_salmon_chain(&game.boards[0]) as f32;

    // K=1: compute the OPTIMAL final score by replaying the last move greedily
    // from the saved state. This may differ from the AI's actual final_score if
    // the AI played a random epsilon-exploration move on its last turn.
    // We compute total + wildlife separately so the split-head target stays consistent.
    let (optimal_final_score, optimal_final_wildlife) =
        if let Some(mut g) = state_before_last_ai_move.clone() {
            if let Some(greedy_mv) = greedy_move(&g) {
                if execute_scored_move(&mut g, &greedy_mv) {
                    let b = ScoreBreakdown::compute(&mut g.boards[0], &g.scoring_cards);
                    (b.total, b.wildlife_total())
                } else {
                    (final_score, final_wildlife)
                }
            } else {
                (final_score, final_wildlife)
            }
        } else {
            (final_score, final_wildlife)
        };

    // Delta labels: remaining points to gain.
    // For sample N-2 (state with 1 AI move remaining = afterstates[len-2]), use the
    // optimal final score. For all other samples, use the actual final score.
    // For v5 11-head training, target = bonus-INCLUDED total (sum of subscore_targets).
    // Per-subscore deltas are computed from the bonus-aware final breakdown.
    let n = afterstates.len();
    for (i, st) in afterstates.iter().enumerate() {
        let (label_final, label_final_wildlife) = if i == n.saturating_sub(2) {
            (
                optimal_final_score.max(final_score),
                optimal_final_wildlife.max(final_wildlife),
            )
        } else {
            (final_score, final_wildlife)
        };
        let remaining_wildlife = label_final_wildlife.saturating_sub(st.current_wildlife) as f32;
        let _ = label_final; // legacy base-only target (kept for read-side back-compat)
                             // Target = bonus-included remaining (matches sum of subscore_targets).
        let target_with_bonus = (final_score_with_bonus as i32 - st.current_total as i32) as f32;
        let mut subscore: [f32; crate::nnue::NUM_HEADS] = [0.0; crate::nnue::NUM_HEADS];
        for w in 0..5 {
            subscore[w] = (final_wildlife_per[w] as i32 - st.current_wildlife_per[w] as i32) as f32;
        }
        for t in 0..5 {
            subscore[5 + t] =
                (final_hab_plus_bonus[t] as i32 - st.current_hab_per[t] as i32) as f32;
        }
        subscore[10] = (final_tokens as i32 - st.current_tokens as i32) as f32;
        samples.push(Sample {
            features: st.features.clone(),
            target: target_with_bonus,
            aux_bear: final_bear_pairs,
            aux_salmon: final_salmon_chain,
            target_wildlife: remaining_wildlife,
            subscore_targets: subscore,
        });
    }

    // Cache high-scoring games (use global mutex for thread-safe writes)
    if final_score >= 90 {
        use std::sync::{Mutex, OnceLock};
        static CACHE_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        let mutex = CACHE_MUTEX.get_or_init(|| Mutex::new(()));
        let _guard = mutex.lock().unwrap();

        let cache_path = std::path::Path::new("training_cache_90plus.bin");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cache_path)
        {
            use std::io::Write;
            // Build the full record in memory first, then write atomically
            let mut buf: Vec<u8> = Vec::with_capacity(1024);
            buf.extend_from_slice(&(afterstates.len() as u16).to_le_bytes());
            buf.extend_from_slice(&final_score.to_le_bytes());
            for st in &afterstates {
                buf.extend_from_slice(&(st.features.len() as u16).to_le_bytes());
                for &f in &st.features {
                    buf.extend_from_slice(&(f as u16).to_le_bytes());
                }
                buf.extend_from_slice(&st.current_total.to_le_bytes());
            }
            let _ = file.write_all(&buf);
        }
    }
}

/// Load samples from the high-score game cache.
/// Each game in the cache scored 90+ during training. Labels use the delta
/// scheme: target = final_score - current_score (remaining points to gain).
pub fn load_cache_samples(cache_path: &std::path::Path) -> std::io::Result<Vec<Sample>> {
    use std::io::Read;
    let mut file = std::fs::File::open(cache_path)?;
    let mut samples = Vec::new();
    let mut buf2 = [0u8; 2];

    // Read all bytes first — easier to handle truncated files
    let mut all_bytes = Vec::new();
    file.read_to_end(&mut all_bytes)?;
    let mut pos = 0usize;

    let mut read_u16 = |bytes: &[u8], pos: &mut usize| -> Option<u16> {
        if *pos + 2 > bytes.len() {
            return None;
        }
        let v = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
        *pos += 2;
        Some(v)
    };

    let mut games_loaded = 0;
    let mut games_skipped = 0;

    while pos < all_bytes.len() {
        let game_start = pos;
        // Read num_positions
        let n = match read_u16(&all_bytes, &mut pos) {
            Some(v) => v as usize,
            None => break,
        };
        let final_score = match read_u16(&all_bytes, &mut pos) {
            Some(v) => v,
            None => break,
        };

        // Validate: num_positions should be reasonable (0..25)
        if n > 25 {
            // Looks corrupted at this offset
            pos = game_start + 2;
            games_skipped += 1;
            if games_skipped > 100 {
                break;
            }
            continue;
        }

        let mut game_samples: Vec<Sample> = Vec::with_capacity(n);
        let mut game_ok = true;

        for _ in 0..n {
            let nf = match read_u16(&all_bytes, &mut pos) {
                Some(v) => v as usize,
                None => {
                    game_ok = false;
                    break;
                }
            };
            if nf > 300 {
                game_ok = false;
                break;
            }

            let mut features = Vec::with_capacity(nf);
            for _ in 0..nf {
                match read_u16(&all_bytes, &mut pos) {
                    Some(v) => features.push(v),
                    None => {
                        game_ok = false;
                        break;
                    }
                }
            }
            if !game_ok {
                break;
            }

            let current_score = match read_u16(&all_bytes, &mut pos) {
                Some(v) => v,
                None => {
                    game_ok = false;
                    break;
                }
            };
            let target = final_score.saturating_sub(current_score) as f32;
            game_samples.push(Sample {
                features,
                target,
                aux_bear: 0.0,
                aux_salmon: 0.0,
                target_wildlife: 0.0,
                subscore_targets: [0.0; crate::nnue::NUM_HEADS],
            });
        }

        if game_ok {
            samples.extend(game_samples);
            games_loaded += 1;
        } else {
            // Try to resync: advance 2 bytes and retry
            pos = game_start + 2;
            games_skipped += 1;
            if games_skipped > 100 {
                break;
            } // too corrupted
        }
    }

    eprintln!(
        "  [loaded {} games, skipped {} corrupted]",
        games_loaded, games_skipped
    );
    let _ = buf2; // silence unused
    Ok(samples)
}

// ── MCE Policy Samples: flat file format ──
// Magic: 4 bytes b"MCEP" (v1), b"MCV2" (v2 with aux), b"MCV3" (v3 adds target_wildlife),
//        b"MCV4" (v4 adds NUM_HEADS subscore deltas for v5 split-head training)
// v1: u16 nf, nf × u16 features, f32 target
// v2: v1 + f32 aux_bear, f32 aux_salmon
// v3: v2 + f32 target_wildlife
// v4: v3 + NUM_HEADS × f32 subscore_targets
//     (target field is bonus-included total; sum of subscore_targets equals target.)
const MCE_POLICY_MAGIC: &[u8; 4] = b"MCEP";
const MCE_POLICY_MAGIC_V2: &[u8; 4] = b"MCV2";
const MCE_POLICY_MAGIC_V3: &[u8; 4] = b"MCV3";
const MCE_POLICY_MAGIC_V4: &[u8; 4] = b"MCV4";

/// Append MCE-labeled samples to a file in legacy MCEP (v1) format.
/// Prefer `append_mce_samples_v3` for all new work — MCEP loses aux + wildlife targets.
pub fn append_mce_samples(
    path: &std::path::Path,
    samples: &[(Vec<u16>, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(samples.len() * 64);
    if is_new {
        buf.extend_from_slice(MCE_POLICY_MAGIC);
    }
    for (features, target) in samples {
        buf.extend_from_slice(&(features.len() as u16).to_le_bytes());
        for &f in features {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&target.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Load all MCE policy samples from a file (auto-detects v1 MCEP, v2 MCV2, v3 MCV3, or v4 MCV4).
/// For v1/v2 samples, `target_wildlife` is set to 0.0; for v1-v3 samples,
/// `subscore_targets` is all-zero (not trainable for 11-head split heads).
pub fn load_mce_samples(path: &std::path::Path) -> std::io::Result<Vec<Sample>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let mut pos = 0usize;
    if bytes.len() < 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too short",
        ));
    }
    let is_v4 = &bytes[..4] == MCE_POLICY_MAGIC_V4;
    let is_v3 = &bytes[..4] == MCE_POLICY_MAGIC_V3;
    let is_v2 = &bytes[..4] == MCE_POLICY_MAGIC_V2;
    let is_v1 = &bytes[..4] == MCE_POLICY_MAGIC;
    if !is_v1 && !is_v2 && !is_v3 && !is_v4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    pos += 4;
    // Per-sample extra bytes beyond `target`:
    //   v1: 0, v2: 8, v3: 12, v4: 12 + NUM_HEADS*4
    let extra_per_sample: usize = if is_v4 {
        12 + crate::nnue::NUM_HEADS * 4
    } else if is_v3 {
        12
    } else if is_v2 {
        8
    } else {
        0
    };
    let mut samples = Vec::new();
    while pos + 2 <= bytes.len() {
        let nf = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;
        if nf > 1024 || pos + nf * 2 + 4 + extra_per_sample > bytes.len() {
            break;
        }
        let mut features = Vec::with_capacity(nf);
        for _ in 0..nf {
            features.push(u16::from_le_bytes([bytes[pos], bytes[pos + 1]]));
            pos += 2;
        }
        let target =
            f32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;
        let (aux_bear, aux_salmon) = if is_v2 || is_v3 || is_v4 {
            let b =
                f32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
            pos += 4;
            let s =
                f32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
            pos += 4;
            (b, s)
        } else {
            (0.0, 0.0)
        };
        let target_wildlife = if is_v3 || is_v4 {
            let w =
                f32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
            pos += 4;
            w
        } else {
            0.0
        };
        let mut subscore_targets = [0.0f32; crate::nnue::NUM_HEADS];
        if is_v4 {
            for st in subscore_targets.iter_mut() {
                *st = f32::from_le_bytes([
                    bytes[pos],
                    bytes[pos + 1],
                    bytes[pos + 2],
                    bytes[pos + 3],
                ]);
                pos += 4;
            }
        }
        samples.push(Sample {
            features,
            target,
            aux_bear,
            aux_salmon,
            target_wildlife,
            subscore_targets,
        });
    }
    Ok(samples)
}

/// Append samples in v2 format (features + target + aux_bear + aux_salmon).
/// Kept for backward compat; new work should prefer `append_mce_samples_v3`.
pub fn append_mce_samples_v2(path: &std::path::Path, samples: &[Sample]) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(samples.len() * 64);
    if is_new {
        buf.extend_from_slice(MCE_POLICY_MAGIC_V2);
    }
    for sample in samples {
        buf.extend_from_slice(&(sample.features.len() as u16).to_le_bytes());
        for &f in &sample.features {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&sample.target.to_le_bytes());
        buf.extend_from_slice(&sample.aux_bear.to_le_bytes());
        buf.extend_from_slice(&sample.aux_salmon.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Append samples in v3 format: v2 fields + target_wildlife.
/// This is the default format for all new caches and training data — the v5 split value
/// head architecture (wildlife + habitat+tokens) depends on target_wildlife being present.
pub fn append_mce_samples_v3(path: &std::path::Path, samples: &[Sample]) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(samples.len() * 68);
    if is_new {
        buf.extend_from_slice(MCE_POLICY_MAGIC_V3);
    }
    for sample in samples {
        buf.extend_from_slice(&(sample.features.len() as u16).to_le_bytes());
        for &f in &sample.features {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&sample.target.to_le_bytes());
        buf.extend_from_slice(&sample.aux_bear.to_le_bytes());
        buf.extend_from_slice(&sample.aux_salmon.to_le_bytes());
        buf.extend_from_slice(&sample.target_wildlife.to_le_bytes());
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Append samples in v4 format: v3 fields + NUM_HEADS subscore_targets.
/// Required for v5 split-value-head training (11 per-subscore deltas per sample).
/// `target` field is the bonus-INCLUDED total (sum of subscore_targets) — differs
/// from MCV3 where `target` is base-only.
pub fn append_mce_samples_v4(path: &std::path::Path, samples: &[Sample]) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let bytes_per_sample = 2 + 4 + 4 + 4 + 4 + crate::nnue::NUM_HEADS * 4 + 64; // ~120 bytes
    let mut buf: Vec<u8> = Vec::with_capacity(samples.len() * bytes_per_sample);
    if is_new {
        buf.extend_from_slice(MCE_POLICY_MAGIC_V4);
    }
    for sample in samples {
        buf.extend_from_slice(&(sample.features.len() as u16).to_le_bytes());
        for &f in &sample.features {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&sample.target.to_le_bytes());
        buf.extend_from_slice(&sample.aux_bear.to_le_bytes());
        buf.extend_from_slice(&sample.aux_salmon.to_le_bytes());
        buf.extend_from_slice(&sample.target_wildlife.to_le_bytes());
        for &st in &sample.subscore_targets {
            buf.extend_from_slice(&st.to_le_bytes());
        }
    }
    file.write_all(&buf)?;
    Ok(())
}

// ── Policy Training Data (MCP2 format) ──
// Magic: 4 bytes b"MCP2"
// Per position group:
//   u16 num_candidates
//   f32 value_target (final_score - current_score)
//   Per candidate:
//     u16 num_features
//     num_features × u16 feature indices
//     f32 candidate_score (expectimax evaluation)
const MCP2_MAGIC: &[u8; 4] = b"MCP2";

/// A position group for policy training: K candidates with scores + value target.
pub struct PolicyGroup {
    pub candidates: Vec<(Vec<u16>, f32)>, // (features, expectimax_score)
    pub value_target: f32,                // final_score - current_score
}

#[derive(Clone)]
pub struct CZeroCandidate {
    pub features: Vec<u16>,
    pub teacher_score: f32,
    pub current_score: f32,
}

#[derive(Clone)]
pub struct CZeroGroup {
    pub candidates: Vec<CZeroCandidate>,
    pub value_target: f32,
    pub subscore_targets: [f32; crate::nnue::NUM_HEADS],
}

pub struct CZeroData {
    pub feature_set: String,
    pub git_rev: String,
    pub groups: Vec<CZeroGroup>,
}

/// Write policy training data to a file.
pub fn save_policy_data(path: &std::path::Path, groups: &[PolicyGroup]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(groups.len() * 512);
    buf.extend_from_slice(MCP2_MAGIC);
    for group in groups {
        buf.extend_from_slice(&(group.candidates.len() as u16).to_le_bytes());
        buf.extend_from_slice(&group.value_target.to_le_bytes());
        for (features, score) in &group.candidates {
            buf.extend_from_slice(&(features.len() as u16).to_le_bytes());
            for &f in features {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf.extend_from_slice(&score.to_le_bytes());
        }
    }
    file.write_all(&buf)?;
    Ok(())
}

/// Load policy training data from a file.
pub fn load_policy_data(path: &std::path::Path) -> std::io::Result<Vec<PolicyGroup>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() < 4 || &bytes[..4] != MCP2_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    let mut pos = 4usize;
    let mut groups = Vec::new();
    while pos + 6 <= bytes.len() {
        let k = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;
        let value_target =
            f32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;
        let mut candidates = Vec::with_capacity(k);
        for _ in 0..k {
            if pos + 2 > bytes.len() {
                break;
            }
            let nf = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;
            if nf > 1024 || pos + nf * 2 + 4 > bytes.len() {
                break;
            }
            let mut features = Vec::with_capacity(nf);
            for _ in 0..nf {
                features.push(u16::from_le_bytes([bytes[pos], bytes[pos + 1]]));
                pos += 2;
            }
            let score =
                f32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
            pos += 4;
            candidates.push((features, score));
        }
        groups.push(PolicyGroup {
            candidates,
            value_target,
        });
    }
    Ok(groups)
}

const CZR1_MAGIC: &[u8; 4] = b"CZR1";

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u16).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn read_str(bytes: &[u8], pos: &mut usize) -> std::io::Result<String> {
    if *pos + 2 > bytes.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "string length",
        ));
    }
    let len = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]) as usize;
    *pos += 2;
    if *pos + len > bytes.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "string bytes",
        ));
    }
    let s = std::str::from_utf8(&bytes[*pos..*pos + len])
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "utf8 string"))?
        .to_string();
    *pos += len;
    Ok(s)
}

/// Save CascadiaZero grouped policy/value records.
pub fn save_czero_data(
    path: &std::path::Path,
    groups: &[CZeroGroup],
    feature_set: &str,
    git_rev: &str,
    cards: &ScoringCards,
    score_target: ScoreTarget,
) -> std::io::Result<()> {
    if !is_all_a(cards) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "CZR1 requires AAAAA",
        ));
    }
    if score_target != ScoreTarget::WithHabitatBonus {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "CZR1 requires with-bonus score target",
        ));
    }

    use std::io::Write;
    let mut buf = Vec::with_capacity(groups.len() * 1024);
    buf.extend_from_slice(CZR1_MAGIC);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&(crate::nnue::NUM_FEATURES as u32).to_le_bytes());
    buf.push(score_target.as_u8());
    buf.extend_from_slice(&[0u8; 5]); // A,A,A,A,A encoded as 0
    buf.extend_from_slice(&(crate::nnue::HIDDEN1 as u32).to_le_bytes());
    buf.extend_from_slice(&(crate::nnue::HIDDEN2 as u32).to_le_bytes());
    write_str(&mut buf, feature_set);
    write_str(&mut buf, git_rev);
    buf.extend_from_slice(&(groups.len() as u32).to_le_bytes());

    for group in groups {
        buf.extend_from_slice(&(group.candidates.len() as u16).to_le_bytes());
        buf.extend_from_slice(&group.value_target.to_le_bytes());
        for &v in &group.subscore_targets {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for cand in &group.candidates {
            buf.extend_from_slice(&(cand.features.len() as u16).to_le_bytes());
            for &f in &cand.features {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf.extend_from_slice(&cand.teacher_score.to_le_bytes());
            buf.extend_from_slice(&cand.current_score.to_le_bytes());
        }
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;
    Ok(())
}

pub fn load_czero_data(path: &std::path::Path) -> std::io::Result<CZeroData> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 4 || &bytes[..4] != CZR1_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad CZR1 magic",
        ));
    }
    let mut pos = 4usize;
    let read_u32 = |bytes: &[u8], pos: &mut usize| -> std::io::Result<u32> {
        if *pos + 4 > bytes.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "u32",
            ));
        }
        let v = u32::from_le_bytes([
            bytes[*pos],
            bytes[*pos + 1],
            bytes[*pos + 2],
            bytes[*pos + 3],
        ]);
        *pos += 4;
        Ok(v)
    };
    let read_f32 = |bytes: &[u8], pos: &mut usize| -> std::io::Result<f32> {
        if *pos + 4 > bytes.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "f32",
            ));
        }
        let v = f32::from_le_bytes([
            bytes[*pos],
            bytes[*pos + 1],
            bytes[*pos + 2],
            bytes[*pos + 3],
        ]);
        *pos += 4;
        Ok(v)
    };

    let version = read_u32(&bytes, &mut pos)?;
    if version != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported CZR version",
        ));
    }
    let num_features = read_u32(&bytes, &mut pos)? as usize;
    if num_features != crate::nnue::NUM_FEATURES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "feature count mismatch: file={} build={}",
                num_features,
                crate::nnue::NUM_FEATURES
            ),
        ));
    }
    if pos >= bytes.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "score target",
        ));
    }
    let target = ScoreTarget::from_u8(bytes[pos])
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad score target"))?;
    pos += 1;
    if target != ScoreTarget::WithHabitatBonus {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "CZR1 is not with-bonus",
        ));
    }
    if pos + 5 > bytes.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "cards",
        ));
    }
    if bytes[pos..pos + 5] != [0u8; 5] {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "CZR1 is not AAAAA",
        ));
    }
    pos += 5;
    let hidden1 = read_u32(&bytes, &mut pos)? as usize;
    let hidden2 = read_u32(&bytes, &mut pos)? as usize;
    if hidden1 != crate::nnue::HIDDEN1 || hidden2 != crate::nnue::HIDDEN2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "network shape mismatch: file={}x{} build={}x{}",
                hidden1,
                hidden2,
                crate::nnue::HIDDEN1,
                crate::nnue::HIDDEN2
            ),
        ));
    }
    let feature_set = read_str(&bytes, &mut pos)?;
    let git_rev = read_str(&bytes, &mut pos)?;
    let n_groups = read_u32(&bytes, &mut pos)? as usize;
    let mut groups = Vec::with_capacity(n_groups);

    for _ in 0..n_groups {
        if pos + 2 > bytes.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "candidate count",
            ));
        }
        let k = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;
        let value_target = read_f32(&bytes, &mut pos)?;
        let mut subscore_targets = [0.0f32; crate::nnue::NUM_HEADS];
        for v in subscore_targets.iter_mut() {
            *v = read_f32(&bytes, &mut pos)?;
        }
        let mut candidates = Vec::with_capacity(k);
        for _ in 0..k {
            if pos + 2 > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "features len",
                ));
            }
            let nf = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;
            if nf > 4096 || pos + nf * 2 + 8 > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "bad candidate record",
                ));
            }
            let mut features = Vec::with_capacity(nf);
            for _ in 0..nf {
                let f = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
                if f as usize >= crate::nnue::NUM_FEATURES {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "feature out of range",
                    ));
                }
                features.push(f);
                pos += 2;
            }
            let teacher_score = read_f32(&bytes, &mut pos)?;
            let current_score = read_f32(&bytes, &mut pos)?;
            candidates.push(CZeroCandidate {
                features,
                teacher_score,
                current_score,
            });
        }
        groups.push(CZeroGroup {
            candidates,
            value_target,
            subscore_targets,
        });
    }
    if pos != bytes.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trailing CZR1 bytes",
        ));
    }
    Ok(CZeroData {
        feature_set,
        git_rev,
        groups,
    })
}

pub fn train_czero_value(
    net: &mut NNUENetwork,
    data: &CZeroData,
    epochs: usize,
    lr: f32,
) -> TrainStats {
    let mut stats = TrainStats::default();
    let mut samples: Vec<(Vec<u16>, f32, [f32; crate::nnue::NUM_HEADS])> = data
        .groups
        .iter()
        .filter_map(|g| {
            let best = g.candidates.iter().max_by(|a, b| {
                a.teacher_score
                    .partial_cmp(&b.teacher_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            let target = (best.teacher_score - best.current_score).max(0.0);
            Some((best.features.clone(), target, g.subscore_targets))
        })
        .collect();
    stats.num_samples = samples.len();
    if samples.is_empty() {
        return stats;
    }

    net.enable_split11_from_current_value();
    let mut rng = StdRng::seed_from_u64(0xC2E0_0001);
    let batch_size = 256;
    for epoch in 0..epochs {
        samples.shuffle(&mut rng);
        let mut loss = 0.0f64;
        let mut count = 0usize;
        for batch_start in (0..samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(samples.len());
            let batch_lr = lr / (batch_end - batch_start) as f32;
            for (features, target, heads) in &samples[batch_start..batch_end] {
                loss += net.train_sample_split11(features, *target, heads, batch_lr) as f64;
                count += 1;
            }
        }
        let rmse = (loss / count.max(1) as f64).sqrt();
        stats.final_rmse = rmse;
        eprint!(
            "\r  CZero value epoch {}/{}: RMSE={:.2}    ",
            epoch + 1,
            epochs,
            rmse
        );
    }
    eprintln!();
    stats
}

// ── Hex rotation augmentation ──
// 120° CW in axial coords: (q, r) → (-q-r, q)
// 240° CW in axial coords: (q, r) → (r, -q-r)
// Pairwise line directions cycle: 0→1→2→0 under 120° CW.

const GRID_DIM: usize = 21;
const GRID_CENTER: i8 = 10;

/// Build a cell index rotation table. Returns None entries for cells that rotate out of bounds.
fn build_rotation_table(rot: usize) -> [Option<usize>; 441] {
    let mut table = [None; 441];
    for idx in 0..441 {
        let q = (idx / GRID_DIM) as i8 - GRID_CENTER;
        let r = (idx % GRID_DIM) as i8 - GRID_CENTER;
        let (q2, r2) = match rot {
            1 => (-q - r, q), // 120° CW
            2 => (r, -q - r), // 240° CW
            _ => (q, r),
        };
        let col = q2 as i16 + GRID_CENTER as i16;
        let row = r2 as i16 + GRID_CENTER as i16;
        if col >= 0 && col < GRID_DIM as i16 && row >= 0 && row < GRID_DIM as i16 {
            table[idx] = Some(col as usize * GRID_DIM + row as usize);
        }
    }
    table
}

/// Whether a pairwise pair should swap (my, neighbor) order when rotating.
/// Under 120° CW: E→SW(reverse), NE→SE(reverse), NW→E(forward).
/// Under 240° CW: E→NW(forward), NE→W(reverse), NW→SW(reverse).
/// Swap when the rotated direction is a REVERSE direction (raw dir >= 3).
const PAIR_SWAP: [[bool; 3]; 3] = [
    [false, false, false], // dir_shift=0: identity, no swap
    [true, true, false],   // dir_shift=1 (120° CW): dirs 0,1 swap; dir 2 doesn't
    [false, true, true],   // dir_shift=2 (240° CW): dirs 1,2 swap; dir 0 doesn't
];

/// Swap a wildlife pairwise pair_state: (my*7 + n) → (n*7 + my)
#[inline]
fn swap_wl_pair(pair_state: usize) -> usize {
    let my = pair_state / 7;
    let n = pair_state % 7;
    n * 7 + my
}

/// Swap a terrain pairwise pair_state: (my*6 + n) → (n*6 + my)
#[inline]
fn swap_terrain_pair(pair_state: usize) -> usize {
    let my = pair_state / 6;
    let n = pair_state % 6;
    n * 6 + my
}

/// Rotate a sparse feature vector. Returns None if any active cell rotates out of bounds.
fn rotate_features(
    features: &[u16],
    rotation_table: &[Option<usize>; 441],
    dir_shift: usize,
) -> Option<Vec<u16>> {
    use crate::nnue;

    // Feature block boundaries (must match nnue.rs layout)
    const FPC: usize = 11; // FEATURES_PER_CELL
    const CELL_END: usize = 441 * FPC; // 4851
    const PHASE_END: usize = CELL_END + 110; // 4961
    const WL_PAIR_STATES: usize = 49;
    const WL_PAIR_END: usize = PHASE_END + 3 * WL_PAIR_STATES; // 5108
    const PATTERN_END: usize = WL_PAIR_END + 89; // 5197
    const BAG_END: usize = PATTERN_END + 55; // 5252
    const OPP_HAB_END: usize = BAG_END + 55; // 5307
                                             // Allowed wildlife: 441 cells × 5 flags
    const ALLOWED_WL_PC: usize = 5;
    const ALLOWED_END: usize = OPP_HAB_END + 441 * ALLOWED_WL_PC; // 7512
    const EXT_WL_END: usize = ALLOWED_END + 50; // 7562
                                                // Terrain pairwise: 3 dirs × 36 states
    const TERRAIN_PAIR_STATES: usize = 36;
    const TERRAIN_PAIR_END: usize = EXT_WL_END + 3 * TERRAIN_PAIR_STATES; // 7670

    // v3 block boundaries
    const V2_END: usize = 10561; // NUM_FEATURES_V2
    const ADJ_FPC: usize = 6 * 13; // ADJ_FEATURES_PER_CELL = 78
    const ADJ_SPD: usize = 13; // ADJ_STATES_PER_DIR
    const ADJ_END: usize = V2_END + 441 * ADJ_FPC; // 44959

    let mut rotated = Vec::with_capacity(features.len());
    for &f in features {
        let fi = f as usize;
        if fi < CELL_END {
            let cell_idx = fi / FPC;
            let offset = fi % FPC;
            let new_cell = rotation_table[cell_idx]?;
            rotated.push((new_cell * FPC + offset) as u16);
        } else if fi < PHASE_END {
            rotated.push(f);
        } else if fi < WL_PAIR_END {
            let rel = fi - PHASE_END;
            let dir = rel / WL_PAIR_STATES;
            let mut pair_state = rel % WL_PAIR_STATES;
            let new_dir = (dir + dir_shift) % 3;
            if PAIR_SWAP[dir_shift][dir] {
                pair_state = swap_wl_pair(pair_state);
            }
            rotated.push((PHASE_END + new_dir * WL_PAIR_STATES + pair_state) as u16);
        } else if fi < PATTERN_END {
            rotated.push(f);
        } else if fi < OPP_HAB_END {
            rotated.push(f);
        } else if fi < ALLOWED_END {
            let rel = fi - OPP_HAB_END;
            let cell_idx = rel / ALLOWED_WL_PC;
            let offset = rel % ALLOWED_WL_PC;
            let new_cell = rotation_table[cell_idx]?;
            rotated.push((OPP_HAB_END + new_cell * ALLOWED_WL_PC + offset) as u16);
        } else if fi < EXT_WL_END {
            rotated.push(f);
        } else if fi < TERRAIN_PAIR_END {
            let rel = fi - EXT_WL_END;
            let dir = rel / TERRAIN_PAIR_STATES;
            let mut pair_state = rel % TERRAIN_PAIR_STATES;
            let new_dir = (dir + dir_shift) % 3;
            if PAIR_SWAP[dir_shift][dir] {
                pair_state = swap_terrain_pair(pair_state);
            }
            rotated.push((EXT_WL_END + new_dir * TERRAIN_PAIR_STATES + pair_state) as u16);
        } else if fi < V2_END {
            // v2 blocks between terrain pairwise and v3 — mostly pass through.
            // Secondary terrain per cell (7670..9875) should remap cell index.
            let sec_base = TERRAIN_PAIR_END;
            let sec_end = sec_base + 441 * 5;
            if fi < sec_end {
                let rel = fi - sec_base;
                let cell_idx = rel / 5;
                let offset = rel % 5;
                let new_cell = rotation_table[cell_idx]?;
                rotated.push((sec_base + new_cell * 5 + offset) as u16);
            } else {
                rotated.push(f); // hab ext, wl ext, ext cap, pat v2, bag, opp, market, tbag
            }
        } else if fi < ADJ_END {
            // v3 Block K: per-cell adjacency — remap cell + rotate direction
            let rel = fi - V2_END;
            let cell_idx = rel / ADJ_FPC;
            let within_cell = rel % ADJ_FPC;
            let dir = within_cell / ADJ_SPD;
            let state = within_cell % ADJ_SPD;
            let new_cell = rotation_table[cell_idx]?;
            // 120° rotation = 2 steps in 6-direction space
            let new_dir = (dir + 2 * dir_shift) % 6;
            rotated.push((V2_END + new_cell * ADJ_FPC + new_dir * ADJ_SPD + state) as u16);
        } else {
            // v3 Blocks L/M/N (tbag ext, overflow) — global, no remap
            rotated.push(f);
        }
    }
    Some(rotated)
}

/// Build a cell index translation table for shifting by (dq, dr).
fn build_translation_table(dq: i8, dr: i8) -> [Option<usize>; 441] {
    let mut table = [None; 441];
    for idx in 0..441 {
        let q = (idx / GRID_DIM) as i8 - GRID_CENTER;
        let r = (idx % GRID_DIM) as i8 - GRID_CENTER;
        let q2 = q + dq;
        let r2 = r + dr;
        let col = q2 as i16 + GRID_CENTER as i16;
        let row = r2 as i16 + GRID_CENTER as i16;
        if col >= 0 && col < GRID_DIM as i16 && row >= 0 && row < GRID_DIM as i16 {
            table[idx] = Some(col as usize * GRID_DIM + row as usize);
        }
    }
    table
}

/// Translate a sparse feature vector (shift all cell indices, no direction change).
fn translate_features(features: &[u16], table: &[Option<usize>; 441]) -> Option<Vec<u16>> {
    // Translation is rotation with dir_shift=0 (no direction change)
    rotate_features(features, table, 0)
}

/// Augment samples with rotations (3×) and translations (up to 25×).
/// Combined: up to 75× data augmentation.
/// Public wrapper for augmentation (used by --export-pytorch)
pub fn augment_samples_pub(samples: &[Sample]) -> Vec<Sample> {
    augment_with_rotations(samples)
}

fn augment_with_rotations(samples: &[Sample]) -> Vec<Sample> {
    let table_120 = build_rotation_table(1);
    let table_240 = build_rotation_table(2);

    // Translation offsets: ±2 in q and r = 5×5 = 25 offsets (including (0,0) = identity)
    let mut translation_tables: Vec<(i8, i8, [Option<usize>; 441])> = Vec::new();
    for dq in -2i8..=2 {
        for dr in -2i8..=2 {
            if dq == 0 && dr == 0 {
                continue;
            } // skip identity
            translation_tables.push((dq, dr, build_translation_table(dq, dr)));
        }
    }

    // Total: 1 original + 2 rotations + 24 translations + 48 (translations × 2 rotations)
    let max_factor = 1 + 2 + 24 + 48; // 75
    let mut augmented = Vec::with_capacity(samples.len() * max_factor);
    let mut skipped = 0usize;

    for sample in samples {
        let aux_b = sample.aux_bear;
        let aux_s = sample.aux_salmon;
        let tw = sample.target_wildlife;
        // Original
        augmented.push(sample.clone());

        // 2 rotations of original
        if let Some(rot) = rotate_features(&sample.features, &table_120, 1) {
            augmented.push(Sample {
                features: rot,
                target: sample.target,
                aux_bear: aux_b,
                aux_salmon: aux_s,
                target_wildlife: tw,
                subscore_targets: sample.subscore_targets,
            });
        } else {
            skipped += 1;
        }
        if let Some(rot) = rotate_features(&sample.features, &table_240, 2) {
            augmented.push(Sample {
                features: rot,
                target: sample.target,
                aux_bear: aux_b,
                aux_salmon: aux_s,
                target_wildlife: tw,
                subscore_targets: sample.subscore_targets,
            });
        } else {
            skipped += 1;
        }

        // 24 translations
        for &(dq, dr, ref table) in &translation_tables {
            if let Some(trans) = translate_features(&sample.features, table) {
                // 2 rotations of each translation
                if let Some(rot) = rotate_features(&trans, &table_120, 1) {
                    augmented.push(Sample {
                        features: rot,
                        target: sample.target,
                        aux_bear: aux_b,
                        aux_salmon: aux_s,
                        target_wildlife: tw,
                        subscore_targets: sample.subscore_targets,
                    });
                } else {
                    skipped += 1;
                }
                if let Some(rot) = rotate_features(&trans, &table_240, 2) {
                    augmented.push(Sample {
                        features: rot,
                        target: sample.target,
                        aux_bear: aux_b,
                        aux_salmon: aux_s,
                        target_wildlife: tw,
                        subscore_targets: sample.subscore_targets,
                    });
                } else {
                    skipped += 1;
                }

                // The translation itself (after rotations so we still have `trans`)
                augmented.push(Sample {
                    features: trans,
                    target: sample.target,
                    aux_bear: aux_b,
                    aux_salmon: aux_s,
                    target_wildlife: tw,
                    subscore_targets: sample.subscore_targets,
                });
            } else {
                skipped += 1;
            }
        }
    }

    if skipped > 0 {
        eprintln!(
            "  [augmentation: skipped {} out-of-bounds transforms]",
            skipped
        );
    }
    augmented
}

/// Train NNUE from MCE policy samples (imitation of MCE via regression on rollout averages).
/// If checkpoint_path is provided, saves weights after every epoch.
pub fn train_from_mce_samples(
    net: &mut NNUENetwork,
    samples_path: &std::path::Path,
    epochs: usize,
    lr: f32,
) -> std::io::Result<TrainStats> {
    train_from_mce_samples_with_checkpoint(net, samples_path, epochs, lr, None, 0)
}

pub fn train_from_mce_samples_with_checkpoint(
    net: &mut NNUENetwork,
    samples_path: &std::path::Path,
    epochs: usize,
    lr: f32,
    checkpoint_path: Option<&std::path::Path>,
    freeze_below: usize, // 0 = train all, >0 = only train features >= this index
) -> std::io::Result<TrainStats> {
    let mut stats = TrainStats::default();
    eprint!("  Loading MCE samples from {:?}...", samples_path);
    let start = std::time::Instant::now();
    let raw_samples = load_mce_samples(samples_path)?;
    eprintln!(" {} samples in {:.1?}", raw_samples.len(), start.elapsed());
    if raw_samples.is_empty() {
        return Ok(stats);
    }

    // Augment with 120° and 240° hex rotations (3× data)
    eprint!("  Augmenting with hex rotations...");
    let aug_start = std::time::Instant::now();
    let mut samples = augment_with_rotations(&raw_samples);
    eprintln!(
        " {} → {} samples in {:.1?}",
        raw_samples.len(),
        samples.len(),
        aug_start.elapsed()
    );
    stats.num_samples = samples.len();

    let mut rng = StdRng::seed_from_u64(42);
    let batch_size = 256;

    let num_threads: usize = std::env::var("CASCADIA_TRAIN_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Learning rate schedule: warmup for first 3 epochs, then cosine decay
    let warmup_epochs = 3.min(epochs);
    let lr_schedule = |epoch: usize| -> f32 {
        if epoch < warmup_epochs {
            // Linear warmup: 0.1*lr → lr
            let t = (epoch + 1) as f32 / warmup_epochs as f32;
            lr * (0.1 + 0.9 * t)
        } else {
            // Cosine decay: lr → 0.01*lr
            let t = (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs).max(1) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            lr * (0.01 + 0.99 * cosine)
        }
    };

    for epoch in 0..epochs {
        let epoch_lr = lr_schedule(epoch);
        samples.shuffle(&mut rng);

        let (loss, count) = if num_threads > 1 {
            // Parallel training: split samples across threads, each trains
            // a local copy, then average weights back.
            let chunk_size = (samples.len() + num_threads - 1) / num_threads;
            let net_arc = std::sync::Arc::new(net.clone());
            let samples_arc = std::sync::Arc::new(samples.clone());

            let handles: Vec<_> = (0..num_threads)
                .map(|t| {
                    let net_copy = (*net_arc).clone();
                    let samples_ref = std::sync::Arc::clone(&samples_arc);
                    let start = t * chunk_size;
                    let end = ((t + 1) * chunk_size).min(samples_ref.len());
                    let lr = epoch_lr;
                    let freeze_below = freeze_below;
                    let batch_size = batch_size;

                    thread::spawn(move || {
                        let mut local_net = net_copy;
                        let mut loss = 0.0f64;
                        let mut count = 0usize;
                        for batch_start in (start..end).step_by(batch_size) {
                            let batch_end = (batch_start + batch_size).min(end);
                            let batch_lr = lr / (batch_end - batch_start) as f32;
                            for sample in &samples_ref[batch_start..batch_end] {
                                let l = if freeze_below > 0 {
                                    local_net.train_sample_frozen(
                                        &sample.features,
                                        sample.target,
                                        batch_lr,
                                        freeze_below,
                                    )
                                } else {
                                    local_net.train_sample(
                                        &sample.features,
                                        sample.target,
                                        batch_lr,
                                    )
                                };
                                loss += l as f64;
                                count += 1;
                            }
                        }
                        (local_net, loss, count)
                    })
                })
                .collect();

            let mut total_loss = 0.0f64;
            let mut total_count = 0usize;
            let mut trained_nets: Vec<NNUENetwork> = Vec::with_capacity(num_threads);
            for handle in handles {
                let (local_net, loss, count) = handle.join().unwrap();
                total_loss += loss;
                total_count += count;
                trained_nets.push(local_net);
            }

            // Average all thread-local networks back into master
            net.average_from(&trained_nets);

            (total_loss, total_count)
        } else {
            // Single-threaded (original path)
            let mut loss = 0.0f64;
            let mut count = 0usize;
            for batch_start in (0..samples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(samples.len());
                let batch_lr = epoch_lr / (batch_end - batch_start) as f32;
                for sample in &samples[batch_start..batch_end] {
                    let l = if freeze_below > 0 {
                        net.train_sample_frozen(
                            &sample.features,
                            sample.target,
                            batch_lr,
                            freeze_below,
                        )
                    } else {
                        net.train_sample(&sample.features, sample.target, batch_lr)
                    };
                    loss += l as f64;
                    count += 1;
                }
            }
            (loss, count)
        };

        let rmse = (loss / count as f64).sqrt();
        let thread_str = if num_threads > 1 {
            format!(" [{}T]", num_threads)
        } else {
            String::new()
        };
        eprint!(
            "\r  Epoch {}/{}: RMSE={:.2} lr={:.6}{}{}    ",
            epoch + 1,
            epochs,
            rmse,
            epoch_lr,
            if freeze_below > 0 {
                format!(" [frozen<{}]", freeze_below)
            } else {
                String::new()
            },
            thread_str
        );
        stats.final_rmse = rmse;

        // Save checkpoint after every epoch
        if let Some(path) = checkpoint_path {
            let _ = net.save(path);
        }
    }
    eprintln!();
    Ok(stats)
}

/// Train NNUE from the high-score cache file (expert imitation learning).
/// This trains on ~1000+ games that scored 90+, labeled with delta targets.
/// Train value head with pairwise ranking loss (Exp #4).
/// Loads MCP2 grouped data (one group per game state, K candidates each scored by MCE).
/// For each group, applies pairwise sigmoid loss `−log σ(score_winner − score_loser)`
/// over all C(K,2) candidate pairs (skipping pairs with MCE-score-difference < margin
/// to avoid noise). Backprops through entire network. Optionally blends with MSE
/// loss on the per-candidate MCE score (alpha=pairwise_weight).
///
/// Theory: pairwise loss is invariant to per-position score offsets, so it doesn't
/// dominate gradients on noisy positions. Tests the hypothesis that ranking-only
/// signal helps where MSE struggles.
pub fn train_from_mcp2_pairwise(
    net: &mut NNUENetwork,
    groups_path: &std::path::Path,
    epochs: usize,
    lr: f32,
    pairwise_weight: f32, // 0.0 = pure MSE, 1.0 = pure pairwise
    margin: f32,          // skip pairs whose MCE-score-diff < margin (default 1.0)
) -> std::io::Result<TrainStats> {
    let mut stats = TrainStats::default();
    eprint!("  Loading MCP2 groups from {:?}...", groups_path);
    let start = std::time::Instant::now();
    let groups = load_policy_data(groups_path)?;
    eprintln!(" {} groups in {:.1?}", groups.len(), start.elapsed());
    if groups.is_empty() {
        return Ok(stats);
    }

    // Filter out groups with < 2 candidates (no pairwise to do)
    let groups: Vec<PolicyGroup> = groups
        .into_iter()
        .filter(|g| g.candidates.len() >= 2)
        .collect();
    stats.num_samples = groups.iter().map(|g| g.candidates.len()).sum();

    let mut rng = StdRng::seed_from_u64(42);
    let mut group_indices: Vec<usize> = (0..groups.len()).collect();

    eprintln!(
        "  Mode: pairwise ranking (alpha={}, margin={})",
        pairwise_weight, margin
    );

    for epoch in 0..epochs {
        group_indices.shuffle(&mut rng);
        let mut total_pair_loss = 0.0f64;
        let mut total_mse_loss = 0.0f64;
        let mut pair_count = 0usize;
        let mut sample_count = 0usize;

        for &gi in &group_indices {
            let group = &groups[gi];
            let n_cands = group.candidates.len();

            // Forward pass: compute value pred for each candidate
            // We re-run net.forward per candidate (no shared backbone optimization here —
            // this is fine-tuning so cost is dominated by data loading anyway)
            let preds: Vec<f32> = group
                .candidates
                .iter()
                .map(|(feats, _)| net.forward(feats))
                .collect();
            let mce_scores: Vec<f32> = group.candidates.iter().map(|(_, s)| *s).collect();

            // ── Pairwise loss + gradients ──
            // For each pair (i, j) with mce_i > mce_j + margin:
            //   target: pred_i should be > pred_j
            //   loss: -log sigmoid(pred_i - pred_j) = log(1 + exp(-(pred_i - pred_j)))
            //   d_loss/d_pred_i = -sigmoid(-(pred_i - pred_j)) = -(1 - σ(diff)) = σ(diff) - 1
            //   d_loss/d_pred_j = +sigmoid(-(pred_i - pred_j)) = 1 - σ(diff)
            //
            // Apply gradients per-candidate by accumulating per-candidate d_pred,
            // then run a target-style update: net.train_sample(features, pred + d_pred, lr)
            // doesn't quite work because train_sample uses (out - target) gradient.
            // Workaround: we set target = pred - d_pred, so (pred - target) = d_pred (same gradient).
            let mut d_pred = vec![0.0f32; n_cands];
            if pairwise_weight > 0.0 {
                for i in 0..n_cands {
                    for j in 0..n_cands {
                        if i == j {
                            continue;
                        }
                        let mce_diff = mce_scores[i] - mce_scores[j];
                        if mce_diff < margin {
                            continue;
                        } // i not clearly better than j
                        let pred_diff = preds[i] - preds[j];
                        let s = 1.0 / (1.0 + (-pred_diff).exp());
                        let loss = -((s.max(1e-9)).ln());
                        total_pair_loss += loss as f64;
                        pair_count += 1;
                        // d_loss / d(pred_i) = -(1 - s) = s - 1
                        // d_loss / d(pred_j) = +(1 - s)
                        let g = pairwise_weight * (s - 1.0);
                        d_pred[i] += g;
                        d_pred[j] -= g;
                    }
                }
            }
            // ── MSE component ──
            if pairwise_weight < 1.0 {
                let alpha_mse = 1.0 - pairwise_weight;
                for i in 0..n_cands {
                    let mse_grad = preds[i] - mce_scores[i];
                    d_pred[i] += alpha_mse * mse_grad;
                    total_mse_loss += (mse_grad * mse_grad) as f64;
                    sample_count += 1;
                }
            }

            // Apply per-candidate gradient via target-style update.
            // Normalize d_pred by approx pair count per candidate (each candidate
            // participates in ~n_cands-1 pairs) to keep effective gradient bounded.
            // Also clip target deviation to ±5 for stability.
            let batch_lr = lr / (n_cands as f32);
            let pair_norm = (n_cands as f32 - 1.0).max(1.0);
            for i in 0..n_cands {
                let dp = (d_pred[i] / pair_norm).clamp(-5.0, 5.0);
                let target = preds[i] - dp;
                let _ = net.train_sample(&group.candidates[i].0, target, batch_lr);
            }
        }
        let avg_pair = if pair_count > 0 {
            total_pair_loss / pair_count as f64
        } else {
            0.0
        };
        let mse_rmse = if sample_count > 0 {
            (total_mse_loss / sample_count as f64).sqrt()
        } else {
            0.0
        };
        eprint!(
            "\r  Epoch {}/{}: pair_loss={:.4}, mse_rmse={:.2}, pairs={}, samples={}    ",
            epoch + 1,
            epochs,
            avg_pair,
            mse_rmse,
            pair_count,
            sample_count
        );
        stats.final_rmse = mse_rmse;
    }
    eprintln!();
    Ok(stats)
}

pub fn train_from_cache(
    net: &mut NNUENetwork,
    cache_path: &std::path::Path,
    epochs: usize,
    lr: f32,
) -> std::io::Result<TrainStats> {
    let mut stats = TrainStats::default();
    eprint!("  Loading cache from {:?}...", cache_path);
    let start = std::time::Instant::now();

    // Detect format: MCV3/MCV2/MCEP start with 4-byte magic. Legacy game-oriented
    // cache (training_cache_90plus.bin) has no magic — starts with u16 num_positions.
    let magic = {
        use std::io::Read;
        let mut f = std::fs::File::open(cache_path)?;
        let mut m = [0u8; 4];
        f.read_exact(&mut m).ok();
        m
    };
    let mut samples = if &magic == b"MCV3" || &magic == b"MCV2" || &magic == b"MCEP" {
        load_mce_samples(cache_path)?
    } else {
        load_cache_samples(cache_path)?
    };
    eprintln!(" {} samples in {:.1?}", samples.len(), start.elapsed());
    stats.num_samples = samples.len();

    // Heteroscedastic NLL mode: CASCADIA_TRAIN_HETEROSCEDASTIC=1 → use
    // train_sample_heteroscedastic (Kendall & Gal 2017). Sets has_heteroscedastic
    // on the net so the v4 save format writes the variance head. Only relevant
    // when fine-tuning from v1/v2/v3 weights (will initialize w3_var to zero
    // which means initial log_var = 0 → σ = 1 → identical-to-MSE first step,
    // then variance head learns).
    let use_heteroscedastic: bool = std::env::var("CASCADIA_TRAIN_HETEROSCEDASTIC")
        .ok()
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    if use_heteroscedastic {
        net.has_heteroscedastic = true;
        eprintln!("  Mode: heteroscedastic NLL (Kendall & Gal 2017)");
    }

    let mut rng = StdRng::seed_from_u64(42);
    let batch_size = 256;

    for epoch in 0..epochs {
        samples.shuffle(&mut rng);
        let mut loss = 0.0f64;
        let mut count = 0usize;
        for batch_start in (0..samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(samples.len());
            let batch_lr = lr / (batch_end - batch_start) as f32;
            for sample in &samples[batch_start..batch_end] {
                let l = if use_heteroscedastic {
                    net.train_sample_heteroscedastic(&sample.features, sample.target, batch_lr)
                } else {
                    net.train_sample(&sample.features, sample.target, batch_lr)
                };
                loss += l as f64;
                count += 1;
            }
        }
        let rmse = (loss / count as f64).sqrt();
        eprint!("\r  Epoch {}/{}: RMSE={:.2}    ", epoch + 1, epochs, rmse);
        stats.final_rmse = rmse;
    }
    eprintln!();

    Ok(stats)
}

/// Train the NNUE network with optional self-play iterations.
/// iterations=1: train on greedy data only (default).
/// iterations>1: first iteration uses greedy, subsequent use NNUE-guided self-play.
pub fn train_nnue(
    net: &mut NNUENetwork,
    num_games: usize,
    epochs: usize,
    lr: f32,
    seed: u64,
) -> TrainStats {
    let iterations: usize = std::env::args()
        .position(|a| a == "--iterations")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let epsilon: f32 = std::env::args()
        .position(|a| a == "--epsilon")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let pretrain: bool = std::env::args().any(|a| a == "--pretrain");

    let mut stats = TrainStats::default();
    let mut rng = StdRng::seed_from_u64(seed + 999);

    // Phase 0: 1-player pre-training (if --pretrain)
    // Teaches the network what high-scoring boards look like
    if pretrain {
        let pretrain_iters = 3;
        for iter in 0..pretrain_iters {
            let use_net = if iter == 0 { None } else { Some(&*net) };
            let iter_epsilon = if iter == 0 { 0.0 } else { epsilon.max(0.05) };
            let label = if use_net.is_some() {
                "1p self-play"
            } else {
                "1p greedy"
            };

            eprint!(
                "  Pre-train {}/{} ({}): generating {} games...",
                iter + 1,
                pretrain_iters,
                label,
                num_games
            );
            let start = std::time::Instant::now();
            let mut samples = generate_samples(
                num_games,
                seed + iter as u64 * 99999,
                use_net,
                iter_epsilon,
                1,
            );
            let gen_time = start.elapsed();
            eprintln!(" {} samples in {:.1?}", samples.len(), gen_time);

            let batch_size = 256;
            for epoch in 0..epochs {
                samples.shuffle(&mut rng);
                let mut epoch_loss = 0.0f64;
                let mut epoch_count = 0usize;
                for batch_start in (0..samples.len()).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(samples.len());
                    let batch_lr = lr / (batch_end - batch_start) as f32;
                    for sample in &samples[batch_start..batch_end] {
                        let loss = net.train_sample(&sample.features, sample.target, batch_lr);
                        epoch_loss += loss as f64;
                        epoch_count += 1;
                    }
                }
                let rmse = (epoch_loss / epoch_count as f64).sqrt();
                eprint!(
                    "\r  Pre {}, Epoch {}/{}: RMSE={:.2}    ",
                    iter + 1,
                    epoch + 1,
                    epochs,
                    rmse
                );
                stats.final_rmse = rmse;
            }
            eprintln!();
        }
        eprintln!("  Pre-training complete. Fine-tuning on 4p...");
    }

    // CASCADIA_TRAIN_LR_DECAY=<end_lr> enables linear LR decay from `lr` (epoch 0)
    // down to <end_lr> (final epoch) within each iteration. If unset, LR stays flat.
    let end_lr: f32 = std::env::var("CASCADIA_TRAIN_LR_DECAY")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(lr);
    if end_lr != lr {
        eprintln!(
            "  [LR decay enabled: {} → {} linearly across {} epochs per iter]",
            lr, end_lr, epochs
        );
    }

    // CASCADIA_TRAIN_TEMPERATURE=<τ> switches player 0's move selection from
    // ε-greedy to softmax sampling over NNUE scores with temperature τ.
    // Higher τ = more exploration (τ=1 is natural; τ=0.5 is sharper/greedier;
    // τ=2+ is very exploratory). This is AlphaZero-style exploration.
    // If set, epsilon is ignored.
    let sample_temperature: Option<f32> = std::env::var("CASCADIA_TRAIN_TEMPERATURE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|&t| t > 0.0);
    if let Some(t) = sample_temperature {
        eprintln!("  [Softmax sampling enabled: τ={} (overrides ε-greedy)]", t);
    }

    // Main training: 4-player iterations
    for iter in 0..iterations {
        let use_net = if iter == 0 && !pretrain {
            None
        } else {
            Some(&*net)
        };
        let iter_epsilon = if iter == 0 && !pretrain { 0.0 } else { epsilon };
        let iter_label = if use_net.is_some() {
            if sample_temperature.is_some() {
                "4p self-play+softmax"
            } else if iter_epsilon > 0.0 {
                "4p self-play+explore"
            } else {
                "4p self-play"
            }
        } else {
            "4p greedy"
        };

        let mode = match sample_temperature {
            Some(t) if use_net.is_some() => SamplingMode::Softmax(t),
            _ => SamplingMode::EpsilonGreedy(iter_epsilon),
        };

        eprint!(
            "  Iteration {}/{} ({}): generating {} games...",
            iter + 1,
            iterations,
            iter_label,
            num_games
        );
        let start = std::time::Instant::now();
        let mut samples =
            generate_samples_with_mode(num_games, seed + iter as u64 * 12345, use_net, mode, 4);
        let gen_time = start.elapsed();
        eprintln!(" {} samples in {:.1?}", samples.len(), gen_time);

        stats.num_samples = samples.len();
        let batch_size = 256;
        let mut last_good_rmse: f64 = f64::INFINITY;
        let mut diverged = false;

        for epoch in 0..epochs {
            samples.shuffle(&mut rng);

            // Linear LR decay within this iteration.
            let epoch_lr = if epochs <= 1 {
                lr
            } else {
                let t = epoch as f32 / (epochs - 1) as f32;
                lr + (end_lr - lr) * t
            };

            let mut epoch_loss = 0.0f64;
            let mut epoch_count = 0usize;

            for batch_start in (0..samples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(samples.len());
                let batch_lr = epoch_lr / (batch_end - batch_start) as f32;

                for sample in &samples[batch_start..batch_end] {
                    let loss = net.train_sample(&sample.features, sample.target, batch_lr);
                    epoch_loss += loss as f64;
                    epoch_count += 1;
                }
            }

            let avg_loss = epoch_loss / epoch_count as f64;
            let rmse = avg_loss.sqrt();
            eprint!(
                "\r  Iter {}, Epoch {}/{}: RMSE={:.2} lr={:.6}    ",
                iter + 1,
                epoch + 1,
                epochs,
                rmse,
                epoch_lr
            );
            stats.final_rmse = rmse;
            if rmse.is_finite() {
                last_good_rmse = rmse;
            } else {
                eprintln!(
                    "\n  [DIVERGED at epoch {} — RMSE={}, halting this iter, not saving]",
                    epoch + 1,
                    rmse
                );
                // Keep stats.final_rmse = NaN (from the assignment above) so the
                // outer CLI layer can detect divergence and refuse to save.
                let _ = last_good_rmse;
                diverged = true;
                break;
            }
        }
        if diverged {
            eprintln!(
                "  [iter {} diverged — existing weights file left untouched]",
                iter + 1
            );
            break;
        }
        // Save weights after each iteration (only if we didn't diverge)
        let weights_path = std::env::args()
            .position(|a| a == "--weights")
            .and_then(|i| std::env::args().nth(i + 1))
            .unwrap_or_else(|| "nnue_weights.bin".to_string());
        let _ = net.save(std::path::Path::new(&weights_path));
        eprintln!("  [saved to {}]", weights_path);
    }

    stats
}

/// Compute the marginal value of each AI-placed tile in the final board.
/// For each tile: how much would the score drop if this tile weren't there?
/// Wildlife: analytically compute per-token contribution to pattern scores.
/// Habitat: each tile contributes 1 per terrain (simplified, no group splitting).
/// Returns marginals in placement order (index 0 = first tile placed).
fn compute_tile_marginals(board: &Board, cards: &ScoringCards) -> Vec<f32> {
    let adj = &*cascadia_core::hex::ADJACENCY;
    let mut marginals = Vec::with_capacity(board.placed_tiles.len());

    // Pre-compute pattern info for wildlife marginals
    let bear_pairs = count_bear_pairs_list(board, adj);
    let bear_pair_count = bear_pairs.len();
    let elk_lines = compute_elk_line_lengths(board);
    let salmon_runs = compute_salmon_run_lengths(board, adj);
    let hawk_isolated = count_isolated_hawks(board, adj);

    // Skip first 3 tiles (starter tiles, not AI-placed)
    let ai_start = 3.min(board.placed_tiles.len());

    for i in 0..board.placed_tiles.len() {
        if i < ai_start {
            // Starter tiles — not counted
            continue;
        }
        let idx = board.placed_tiles[i] as usize;
        let cell = board.grid.get(idx);
        let mut marginal = 0.0f32;

        // Habitat marginal: 1 per terrain on this tile
        if cell.primary_terrain().is_some() {
            marginal += 1.0;
        }
        if cell.secondary_terrain().is_some() {
            marginal += 1.0;
        }

        // Wildlife marginal
        if let Some(w) = cell.placed_wildlife() {
            let variant = cards.variant_for(w);
            marginal += wildlife_marginal(
                board,
                idx,
                w,
                variant,
                adj,
                bear_pair_count,
                &elk_lines,
                &salmon_runs,
                hawk_isolated,
            );

            // Nature token from keystone
            if cell.is_keystone() {
                marginal += 1.0;
            }
        }

        marginals.push(marginal);
    }

    marginals
}

/// Compute marginal value of a specific wildlife token at `pos`.
fn wildlife_marginal(
    board: &Board,
    pos: usize,
    w: cascadia_core::types::Wildlife,
    _variant: cascadia_core::types::ScoringCardVariant,
    adj: &cascadia_core::hex::AdjacencyTable,
    bear_pair_count: usize,
    elk_lines: &[(usize, usize)],   // (position, line_length)
    salmon_runs: &[(usize, usize)], // (position, run_length)
    hawk_isolated: usize,
) -> f32 {
    use cascadia_core::types::Wildlife;

    match w {
        Wildlife::Bear => {
            // Check if this bear is part of a valid pair
            let bear_neighbors: usize = adj
                .neighbors_of(pos)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
                .count();
            if bear_neighbors == 1 {
                // Part of a pair — marginal = half the pair's marginal value
                // Going from N pairs to N-1: score table [0,4,11,19,27]
                let pair_scores = [0.0, 4.0, 11.0, 19.0, 27.0];
                let n = bear_pair_count.min(4);
                let with = pair_scores[n];
                let without = if n > 0 { pair_scores[n - 1] } else { 0.0 };
                (with - without) / 2.0 // split credit between both bears
            } else {
                0.0 // isolated or in cluster — no scoring contribution
            }
        }
        Wildlife::Elk => {
            // Find the line this elk belongs to
            if let Some(&(_, line_len)) = elk_lines.iter().find(|&&(p, _)| p == pos) {
                let line_scores = [0.0, 2.0, 5.0, 9.0, 13.0];
                let len = line_len.min(4);
                let with = line_scores[len];
                let without = if len > 0 { line_scores[len - 1] } else { 0.0 };
                with - without // marginal of this elk extending the line by 1
            } else {
                2.0 // single elk = 2 points
            }
        }
        Wildlife::Salmon => {
            // Find the run this salmon belongs to
            if let Some(&(_, run_len)) = salmon_runs.iter().find(|&&(p, _)| p == pos) {
                let run_scores = [0.0, 2.0, 4.0, 7.0, 11.0, 15.0, 20.0, 26.0];
                let len = run_len.min(7);
                let with = run_scores[len];
                let without = if len > 0 { run_scores[len - 1] } else { 0.0 };
                with - without
            } else {
                2.0
            }
        }
        Wildlife::Hawk => {
            let has_hawk_neighbor = adj
                .neighbors_of(pos)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
            if !has_hawk_neighbor {
                // Isolated — marginal of Nth isolated hawk
                let hawk_scores = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 18.0, 22.0, 28.0];
                let n = hawk_isolated.min(8);
                let with = hawk_scores[n];
                let without = if n > 0 { hawk_scores[n - 1] } else { 0.0 };
                with - without
            } else {
                0.0
            }
        }
        Wildlife::Fox => {
            // Individual fox score = unique adjacent wildlife types
            let mut mask = 0u8;
            for nidx in adj.neighbors_of(pos) {
                if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                    mask |= 1 << (w as u8);
                }
            }
            mask.count_ones() as f32
        }
    }
}

// Helper: list all positions that are part of bear pairs
fn count_bear_pairs_list(
    board: &Board,
    adj: &cascadia_core::hex::AdjacencyTable,
) -> Vec<(usize, usize)> {
    use cascadia_core::types::Wildlife;
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut visited = [false; 441];
    let mut pairs = Vec::new();
    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        if component.len() == 2 {
            pairs.push((component[0] as usize, component[1] as usize));
        }
    }
    pairs
}

// Helper: for each elk, find the line it belongs to and the line length
fn compute_elk_line_lengths(board: &Board) -> Vec<(usize, usize)> {
    use cascadia_core::types::Wildlife;
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let mut results = Vec::new();
    let mut is_elk = [false; 441];
    for &pos in positions.iter() {
        is_elk[pos as usize] = true;
    }

    // For each elk, find the longest line through it in any direction
    for &pos in positions.iter() {
        let coord = HexCoord::from_index(pos as usize);
        let mut best_len = 1;
        for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
            let mut len = 1;
            let mut c = HexCoord::new(coord.q + dq, coord.r + dr);
            while let Some(idx) = c.to_index() {
                if is_elk[idx] {
                    len += 1;
                    c = HexCoord::new(c.q + dq, c.r + dr);
                } else {
                    break;
                }
            }
            c = HexCoord::new(coord.q - dq, coord.r - dr);
            while let Some(idx) = c.to_index() {
                if is_elk[idx] {
                    len += 1;
                    c = HexCoord::new(c.q - dq, c.r - dr);
                } else {
                    break;
                }
            }
            best_len = best_len.max(len);
        }
        results.push((pos as usize, best_len));
    }
    results
}

// Helper: for each salmon, find the run it belongs to and run length
fn compute_salmon_run_lengths(
    board: &Board,
    adj: &cascadia_core::hex::AdjacencyTable,
) -> Vec<(usize, usize)> {
    use cascadia_core::types::Wildlife;
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let mut visited = [false; 441];
    let mut results = Vec::new();

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        let is_valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count()
                <= 2
        });
        let len = if is_valid { component.len() } else { 0 };
        for &p in &component {
            results.push((p as usize, len));
        }
    }
    results
}

// Helper: count isolated hawks
fn count_isolated_hawks(board: &Board, adj: &cascadia_core::hex::AdjacencyTable) -> usize {
    use cascadia_core::types::Wildlife;
    board.wildlife_positions[Wildlife::Hawk as usize]
        .iter()
        .filter(|&&pos| {
            !adj.neighbors_of(pos as usize)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk))
        })
        .count()
}

/// Simple pre-move optimization for training data generation.
/// Checks whether replacing 3-of-a-kind or mulliganing improves the greedy score.
fn greedy_pre_move(game: &mut GameState, _rng: &mut StdRng) {
    const MAX_MULLIGANS: usize = 3;
    let player = game.current_player;
    let mut mulligans_used = 0;

    loop {
        let baseline = greedy_score(game);

        // Option 1: free 3-of-a-kind replacement
        if game.can_replace_overflow().is_some() {
            let mut test = game.clone();
            test.replace_overflow();
            if greedy_score(&test) > baseline {
                game.replace_overflow();
                continue;
            }
        }

        // Option 2: paid mulligan (only if significantly better, to offset token cost)
        if mulligans_used < MAX_MULLIGANS && game.boards[player].nature_tokens > 0 {
            let mut test = game.clone();
            if test.mulligan_wildlife() {
                // Use greedy eval on actual post-mulligan state (no sampling for speed)
                let new_score = greedy_score(&test);
                if new_score > baseline + 2 {
                    game.mulligan_wildlife();
                    mulligans_used += 1;
                    continue;
                }
            }
        }
        break;
    }
}

fn greedy_score(game: &GameState) -> u16 {
    greedy_move(game).map(|m| m.score).unwrap_or(0)
}

/// Pick a random valid move (for epsilon-greedy exploration).
fn pick_random_move(game: &GameState, rng: &mut StdRng) -> Option<crate::eval::ScoredMove> {
    use crate::eval::ScoredMove;
    use cascadia_core::hex::HexCoord;

    let mp: Vec<_> = game
        .market
        .available()
        .map(|(i, p)| (i, p.tile, p.wildlife))
        .collect();
    if mp.is_empty() {
        return None;
    }

    let board = &game.boards[game.current_player];
    let frontier = board.frontier();
    if frontier.is_empty() {
        return None;
    }

    // Pick random market pair
    let &(idx, tile, wildlife) = &mp[rng.gen_range(0..mp.len())];

    // Pick random frontier cell
    let fi = frontier[rng.gen_range(0..frontier.len())] as usize;
    let coord = HexCoord::from_index(fi);

    // Pick random rotation
    let max_rot: u8 = if tile.terrain2.is_none() { 1 } else { 6 };
    let rot = rng.gen_range(0..max_rot);

    // Try to place tile; if invalid, fall back to greedy
    let mut board_clone = board.clone();
    if board_clone.place_tile(coord, tile, rot).is_none() {
        return greedy_move(game);
    }

    // Pick random wildlife placement (or skip with 20% chance)
    let valid_positions: Vec<u16> = board_clone
        .placed_tiles
        .iter()
        .copied()
        .filter(|&ti| {
            board_clone
                .grid
                .get(ti as usize)
                .can_place_wildlife(wildlife)
        })
        .collect();

    let (wq, wr) = if !valid_positions.is_empty() && rng.gen::<f32>() > 0.2 {
        let ti = valid_positions[rng.gen_range(0..valid_positions.len())];
        let wc = HexCoord::from_index(ti as usize);
        (Some(wc.q), Some(wc.r))
    } else {
        (None, None)
    };

    Some(ScoredMove {
        market_index: idx,
        tile_q: coord.q,
        tile_r: coord.r,
        rotation: rot,
        wildlife_q: wq,
        wildlife_r: wr,
        score: 0,
        eval: 0,
        wildlife_market_index: None,
    })
}

#[derive(Debug, Clone)]
pub struct PreparedNnueMoveCandidate {
    pub movement: crate::eval::ScoredMove,
    pub actual_score: f32,
    pub features: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct PreparedNnueMove {
    pub fallback: Option<crate::eval::ScoredMove>,
    pub candidates: Vec<PreparedNnueMoveCandidate>,
}

/// Build the exact candidate afterstates consumed by the historical NNUE
/// rollout policy without evaluating the network.
pub fn prepare_nnue_move(game: &GameState) -> PreparedNnueMove {
    use crate::eval::ScoredMove;
    use cascadia_core::hex::HexCoord;

    let mp: Vec<_> = game
        .market
        .available()
        .map(|(i, p)| (i, p.tile, p.wildlife))
        .collect();
    if mp.is_empty() {
        return PreparedNnueMove {
            fallback: None,
            candidates: Vec::new(),
        };
    }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let player = game.current_player;
    let mut board = game.boards[player].clone();
    let base_move = crate::eval::best_move_with_potential(&mut board, &mp, &cards, turns);

    // Use fast greedy candidates + NNUE re-ranking (faster than full NNUE candidates)
    let mut candidates: Vec<ScoredMove> = crate::search::candidate_moves_pub(game);
    if let Some(ref bm) = base_move {
        if !candidates.iter().any(|c| {
            c.tile_q == bm.tile_q
                && c.tile_r == bm.tile_r
                && c.rotation == bm.rotation
                && c.wildlife_q == bm.wildlife_q
                && c.wildlife_r == bm.wildlife_r
        }) {
            candidates.push(*bm);
        }
    }
    candidates.truncate(15);

    if candidates.is_empty() {
        return PreparedNnueMove {
            fallback: base_move,
            candidates: Vec::new(),
        };
    }

    let bag_info = crate::nnue::BagInfo::from_game_for_player(game, player);

    // Optimization: use place/undo on the single outer `board` rather than
    // cloning it per candidate. Saves 15 × ~15KB board copies per decision
    // (each rollout calls pick_best_move_nnue ~2 times, so ~450KB/rollout).
    //
    // Safety: place_tile/place_wildlife return UndoAction handles that exactly
    // reverse their effects (including keystone nature_tokens). ScoreBreakdown::compute
    // takes &mut but only reads state. Candidates where tile placement fails
    // leave `board` untouched (place_tile returns None without mutating).
    let mut prepared = Vec::with_capacity(candidates.len());
    for mv in &candidates {
        let coord = HexCoord::new(mv.tile_q, mv.tile_r);
        let tile = match mp.iter().find(|&&(i, _, _)| i == mv.market_index) {
            Some(&(_, tile, _)) => tile,
            None => continue,
        };
        let wildlife = match mp
            .iter()
            .find(|&&(i, _, _)| i == mv.wildlife_market_index.unwrap_or(mv.market_index))
        {
            Some(&(_, _, wl)) => wl,
            None => continue,
        };

        let tile_act = match board.place_tile(coord, tile, mv.rotation) {
            Some(a) => a,
            None => continue,
        };
        let wl_act = if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
            HexCoord::new(wq, wr)
                .to_index()
                .and_then(|idx| board.place_wildlife(idx, wildlife))
        } else {
            None
        };
        let nt_saved = if mv.wildlife_market_index.is_some() {
            let saved = board.nature_tokens;
            board.nature_tokens = board.nature_tokens.saturating_sub(1);
            Some(saved)
        } else {
            None
        };

        let actual =
            cascadia_core::scoring::ScoreBreakdown::compute(&mut board, &cards).total as f32;
        let features = crate::nnue::extract_features_with_bag(&board, Some(&bag_info));
        prepared.push(PreparedNnueMoveCandidate {
            movement: *mv,
            actual_score: actual,
            features,
        });

        // Undo in reverse order of application.
        if let Some(saved) = nt_saved {
            board.nature_tokens = saved;
        }
        if let Some(wa) = wl_act {
            board.undo(wa);
        }
        board.undo(tile_act);
    }

    PreparedNnueMove {
        fallback: base_move,
        candidates: prepared,
    }
}

pub fn select_prepared_nnue_move(
    prepared: &PreparedNnueMove,
    remaining_values: &[f32],
) -> Option<crate::eval::ScoredMove> {
    select_prepared_nnue_candidate_index(prepared, remaining_values)
        .map(|index| prepared.candidates[index].movement)
        .or(prepared.fallback)
}

pub fn select_prepared_nnue_candidate_index(
    prepared: &PreparedNnueMove,
    remaining_values: &[f32],
) -> Option<usize> {
    if prepared.candidates.len() != remaining_values.len() {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for (index, (candidate, &remaining)) in
        prepared.candidates.iter().zip(remaining_values).enumerate()
    {
        let estimated_final = candidate.actual_score + remaining;
        if best.is_none() || estimated_final > best.as_ref().unwrap().1 {
            best = Some((index, estimated_final));
        }
    }
    best.map(|(index, _)| index)
}

/// Pick best move: get greedy top-K candidates, re-rank by NNUE afterstate value.
pub fn pick_best_move_nnue(game: &GameState, net: &NNUENetwork) -> Option<crate::eval::ScoredMove> {
    let prepared = prepare_nnue_move(game);
    let remaining_values = prepared
        .candidates
        .iter()
        .map(|candidate| net.forward(&candidate.features))
        .collect::<Vec<_>>();
    select_prepared_nnue_move(&prepared, &remaining_values)
}

/// Simulated-annealing move selection: scores all candidate moves with NNUE,
/// applies softmax(score / temperature), then samples from that distribution.
///
/// Higher temperature → broader exploration (approaches uniform as T → ∞).
/// Lower temperature → greedier (approaches argmax as T → 0).
///
/// Replaces ε-greedy in self-play when you want smooth exploration control.
/// Typical schedule: T_start=2.0 (broad exploration in iter1), T_end=0.1 (near-argmax at iter_final).
pub fn pick_softmax_move_nnue(
    game: &GameState,
    net: &NNUENetwork,
    temperature: f32,
    rng: &mut StdRng,
) -> Option<crate::eval::ScoredMove> {
    use crate::eval::ScoredMove;
    use cascadia_core::hex::HexCoord;
    use rand::Rng;

    let mp: Vec<_> = game
        .market
        .available()
        .map(|(i, p)| (i, p.tile, p.wildlife))
        .collect();
    if mp.is_empty() {
        return None;
    }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let player = game.current_player;
    let mut board = game.boards[player].clone();
    let base_move = crate::eval::best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates: Vec<ScoredMove> = crate::search::candidate_moves_pub(game);
    if let Some(ref bm) = base_move {
        if !candidates.iter().any(|c| {
            c.tile_q == bm.tile_q
                && c.tile_r == bm.tile_r
                && c.rotation == bm.rotation
                && c.wildlife_q == bm.wildlife_q
                && c.wildlife_r == bm.wildlife_r
        }) {
            candidates.push(*bm);
        }
    }
    candidates.truncate(15);

    if candidates.is_empty() {
        return base_move;
    }

    let bag_info = crate::nnue::BagInfo::from_game_for_player(game, player);

    // Score each candidate
    let mut scored: Vec<(ScoredMove, f32)> = Vec::with_capacity(candidates.len());
    for mv in &candidates {
        let coord = HexCoord::new(mv.tile_q, mv.tile_r);
        let tile = match mp.iter().find(|&&(i, _, _)| i == mv.market_index) {
            Some(&(_, tile, _)) => tile,
            None => continue,
        };
        let wildlife = match mp
            .iter()
            .find(|&&(i, _, _)| i == mv.wildlife_market_index.unwrap_or(mv.market_index))
        {
            Some(&(_, _, wl)) => wl,
            None => continue,
        };

        let mut eval_board = board.clone();
        if eval_board.place_tile(coord, tile, mv.rotation).is_none() {
            continue;
        }
        if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
            let wcoord = HexCoord::new(wq, wr);
            if let Some(idx) = wcoord.to_index() {
                eval_board.place_wildlife(idx, wildlife);
            }
        }
        if mv.wildlife_market_index.is_some() {
            eval_board.nature_tokens = eval_board.nature_tokens.saturating_sub(1);
        }

        let actual =
            cascadia_core::scoring::ScoreBreakdown::compute(&mut eval_board, &cards).total as f32;
        let remaining = net.evaluate_with_bag(&eval_board, &bag_info);
        let estimated_final = actual + remaining;
        scored.push((*mv, estimated_final));
    }

    if scored.is_empty() {
        return None;
    }

    // Degenerate temperature: fall back to argmax
    if temperature <= 1e-6 {
        return scored
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mv, _)| mv);
    }

    // Softmax with numerical stability: subtract max before exp
    let max_score = scored
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = scored
        .iter()
        .map(|(_, s)| ((s - max_score) / temperature).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // Numerical blowup — fall back to argmax
        return scored
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mv, _)| mv);
    }
    for w in weights.iter_mut() {
        *w /= sum;
    }

    // Sample from the distribution
    let r: f32 = rng.gen::<f32>();
    let mut cum = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cum += w;
        if r <= cum {
            return Some(scored[i].0);
        }
    }
    Some(scored.last().unwrap().0)
}

/// Enumerate ALL legal moves and score each afterstate with NNUE.
/// No pre-filtering — every (market, frontier, rotation, wildlife_placement) combo is evaluated.
pub fn pick_best_move_nnue_full(
    game: &GameState,
    net: &NNUENetwork,
) -> Option<crate::eval::ScoredMove> {
    use crate::eval::ScoredMove;
    use cascadia_core::scoring::ScoreBreakdown;

    let player = game.current_player;
    let board = &game.boards[player];
    let cards = game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return None;
    }

    let market_pairs: Vec<_> = game
        .market
        .available()
        .map(|(i, p)| (i, p.tile, p.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return None;
    }

    let mut board_clone = board.clone();
    let mut best: Option<(ScoredMove, f32)> = None;

    for &(mi, tile, wildlife) in &market_pairs {
        let max_rot: u8 = if tile.terrain2.is_none() { 1 } else { 6 };

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let tile_action = match board_clone.place_tile(coord, tile, rot) {
                    Some(a) => a,
                    None => continue,
                };

                // Option 1: skip wildlife placement
                let actual = ScoreBreakdown::compute(&mut board_clone, &cards).total as f32;
                let remaining = net.evaluate(&board_clone);
                let score_skip = actual + remaining;

                let skip_mv = ScoredMove {
                    market_index: mi,
                    tile_q: coord.q,
                    tile_r: coord.r,
                    rotation: rot,
                    wildlife_q: None,
                    wildlife_r: None,
                    score: actual as u16,
                    eval: 0,
                    wildlife_market_index: None,
                };
                if best.is_none() || score_skip > best.as_ref().unwrap().1 {
                    best = Some((skip_mv, score_skip));
                }

                // Option 2: try every valid wildlife placement
                let placed: arrayvec::ArrayVec<u16, 64> =
                    board_clone.placed_tiles.iter().copied().collect();
                for &ti in placed.iter() {
                    if !board_clone
                        .grid
                        .get(ti as usize)
                        .can_place_wildlife(wildlife)
                    {
                        continue;
                    }
                    let wl_action = match board_clone.place_wildlife(ti as usize, wildlife) {
                        Some(a) => a,
                        None => continue,
                    };

                    let actual_w = ScoreBreakdown::compute(&mut board_clone, &cards).total as f32;
                    let remaining_w = net.evaluate(&board_clone);
                    let score_w = actual_w + remaining_w;

                    board_clone.undo(wl_action);

                    if score_w > best.as_ref().map(|b| b.1).unwrap_or(f32::NEG_INFINITY) {
                        let wc = HexCoord::from_index(ti as usize);
                        best = Some((
                            ScoredMove {
                                market_index: mi,
                                tile_q: coord.q,
                                tile_r: coord.r,
                                rotation: rot,
                                wildlife_q: Some(wc.q),
                                wildlife_r: Some(wc.r),
                                score: actual_w as u16,
                                eval: 0,
                                wildlife_market_index: None,
                            },
                            score_w,
                        ));
                    }
                }

                board_clone.undo(tile_action);
            }
        }
    }

    best.map(|(mv, _)| mv)
}

#[derive(Default)]
pub struct TrainStats {
    pub num_samples: usize,
    pub final_rmse: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scored_move(market_index: usize) -> crate::eval::ScoredMove {
        crate::eval::ScoredMove {
            market_index,
            tile_q: market_index as i8,
            tile_r: 0,
            rotation: 0,
            wildlife_q: None,
            wildlife_r: None,
            score: 0,
            eval: 0,
            wildlife_market_index: None,
        }
    }

    #[test]
    fn prepared_nnue_selection_uses_final_value_and_stable_ties() {
        let prepared = PreparedNnueMove {
            fallback: Some(test_scored_move(9)),
            candidates: vec![
                PreparedNnueMoveCandidate {
                    movement: test_scored_move(0),
                    actual_score: 10.0,
                    features: vec![1],
                },
                PreparedNnueMoveCandidate {
                    movement: test_scored_move(1),
                    actual_score: 8.0,
                    features: vec![2],
                },
                PreparedNnueMoveCandidate {
                    movement: test_scored_move(2),
                    actual_score: 11.0,
                    features: vec![3],
                },
            ],
        };

        assert_eq!(
            select_prepared_nnue_candidate_index(&prepared, &[2.0, 5.0, 2.0]),
            Some(1)
        );
        assert_eq!(
            select_prepared_nnue_candidate_index(&prepared, &[3.0, 5.0, 2.0]),
            Some(0),
            "equal estimated finals keep the first candidate"
        );
        assert_eq!(
            select_prepared_nnue_move(&prepared, &[3.0, 5.0, 2.0])
                .map(|movement| movement.market_index),
            Some(0)
        );
    }

    #[test]
    fn prepared_nnue_selection_falls_back_on_prediction_width_mismatch() {
        let fallback = test_scored_move(7);
        let prepared = PreparedNnueMove {
            fallback: Some(fallback),
            candidates: vec![PreparedNnueMoveCandidate {
                movement: test_scored_move(0),
                actual_score: 10.0,
                features: vec![1],
            }],
        };

        assert_eq!(select_prepared_nnue_candidate_index(&prepared, &[]), None);
        assert_eq!(
            select_prepared_nnue_move(&prepared, &[]).map(|movement| movement.market_index),
            Some(fallback.market_index)
        );
    }

    #[test]
    fn czero_roundtrip_and_rejects_bad_target() {
        let path = std::env::temp_dir().join(format!(
            "czero_roundtrip_{}_{}.czr",
            std::process::id(),
            crate::nnue::NUM_FEATURES
        ));
        let groups = vec![CZeroGroup {
            candidates: vec![CZeroCandidate {
                features: vec![0, (crate::nnue::NUM_FEATURES - 1) as u16],
                teacher_score: 101.0,
                current_score: 90.0,
            }],
            value_target: 11.0,
            subscore_targets: [1.0; crate::nnue::NUM_HEADS],
        }];

        save_czero_data(
            &path,
            &groups,
            "test-features",
            "test-git",
            &ScoringCards::all_a(),
            ScoreTarget::WithHabitatBonus,
        )
        .unwrap();
        let loaded = load_czero_data(&path).unwrap();
        assert_eq!(loaded.groups.len(), 1);
        assert_eq!(loaded.groups[0].candidates[0].teacher_score, 101.0);

        let mut bytes = std::fs::read(&path).unwrap();
        // magic(4) + version(4) + num_features(4) = score_target byte
        bytes[12] = ScoreTarget::Base.as_u8();
        let bad_path = path.with_extension("bad.czr");
        std::fs::write(&bad_path, bytes).unwrap();
        assert!(load_czero_data(&bad_path).is_err());

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(bad_path);
    }

    #[test]
    fn czp1_header_rejects_base_target_and_wrong_feature_count() {
        fn write_fake(path: &std::path::Path, num_features: u32, target: ScoreTarget) {
            use std::io::Write;
            let mut f = std::fs::File::create(path).unwrap();
            f.write_all(b"CZP1").unwrap();
            f.write_all(&2u32.to_le_bytes()).unwrap();
            f.write_all(&num_features.to_le_bytes()).unwrap();
            f.write_all(&(512u32).to_le_bytes()).unwrap();
            f.write_all(&(256u32).to_le_bytes()).unwrap();
            f.write_all(&[target.as_u8()]).unwrap();
            f.write_all(&[0u8; 5]).unwrap();
            f.write_all(&0u16.to_le_bytes()).unwrap(); // feature set
            f.write_all(&0u16.to_le_bytes()).unwrap(); // git rev
        }

        let base_path =
            std::env::temp_dir().join(format!("czp1_base_{}.policy", std::process::id()));
        write_fake(
            &base_path,
            crate::nnue::NUM_FEATURES as u32,
            ScoreTarget::Base,
        );
        assert!(crate::nnue::PolicyNetwork::load(&base_path).is_err());

        let feature_path =
            std::env::temp_dir().join(format!("czp1_features_{}.policy", std::process::id()));
        write_fake(
            &feature_path,
            crate::nnue::NUM_FEATURES.saturating_add(1) as u32,
            ScoreTarget::WithHabitatBonus,
        );
        assert!(crate::nnue::PolicyNetwork::load(&feature_path).is_err());

        let _ = std::fs::remove_file(base_path);
        let _ = std::fs::remove_file(feature_path);
    }

    #[test]
    fn czero_feature_count_is_append_only_when_enabled() {
        if cfg!(feature = "czero-feat")
            && cfg!(feature = "v5-feat")
            && cfg!(feature = "v4-opp")
            && !cfg!(feature = "mid-features")
            && !cfg!(feature = "legacy-features")
        {
            assert_eq!(
                crate::nnue::CZERO_FEAT_BASE,
                crate::nnue::NUM_FEATURES_V3_V4_V5
            );
            assert_eq!(
                crate::nnue::NUM_FEATURES,
                crate::nnue::NUM_FEATURES_V3_V4_V5 + crate::nnue::CZERO_FEAT_FEATURES
            );
        }
    }

    #[test]
    fn test_swap_wl_pair() {
        // bear(1) looking at salmon(3) → pair_state = 1*7+3 = 10
        // swapped: salmon(3) looking at bear(1) → 3*7+1 = 22
        assert_eq!(swap_wl_pair(10), 22);
        assert_eq!(swap_wl_pair(22), 10);
        // identity: bear(1) looking at bear(1) → 1*7+1 = 8
        assert_eq!(swap_wl_pair(8), 8);
        // empty(0) looking at hawk(4) → 0*7+4 = 4, swapped → 4*7+0 = 28
        assert_eq!(swap_wl_pair(4), 28);
        assert_eq!(swap_wl_pair(28), 4);
    }

    #[test]
    fn test_swap_terrain_pair() {
        // forest(1) next to river(5) → 1*6+5 = 11
        // swapped: river(5) next to forest(1) → 5*6+1 = 31
        assert_eq!(swap_terrain_pair(11), 31);
        assert_eq!(swap_terrain_pair(31), 11);
        // same terrain: mountain(4) next to mountain(4) → 4*6+4 = 28
        assert_eq!(swap_terrain_pair(28), 28);
    }

    #[test]
    fn test_pair_swap_table_consistency() {
        // dir_shift=0 should never swap (identity)
        for dir in 0..3 {
            assert!(!PAIR_SWAP[0][dir]);
        }
        // Rotation 1 (120° CW): dirs 0,1 swap, dir 2 doesn't
        assert!(PAIR_SWAP[1][0]);
        assert!(PAIR_SWAP[1][1]);
        assert!(!PAIR_SWAP[1][2]);
        // Rotation 2 (240° CW): dirs 1,2 swap, dir 0 doesn't
        assert!(!PAIR_SWAP[2][0]);
        assert!(PAIR_SWAP[2][1]);
        assert!(PAIR_SWAP[2][2]);
    }

    #[test]
    fn test_rotate_pairwise_feature_swap() {
        // Create a feature: bear(1) looking at salmon(3) in direction E (dir 0)
        // pair_state = 1*7+3 = 10, feature = PHASE_END + 0*49 + 10 = 4961 + 10 = 4971
        let feature = 4971u16;

        let table_120 = build_rotation_table(1);

        // Rotate 120° CW: dir 0 → dir 1, and pair should SWAP
        // Swapped pair_state = 3*7+1 = 22
        // Expected: PHASE_END + 1*49 + 22 = 4961 + 49 + 22 = 5032
        let rotated = rotate_features(&[feature], &table_120, 1).unwrap();
        assert_eq!(rotated[0], 5032);

        // Without swap it would be 4961 + 49 + 10 = 5020 (wrong)
        assert_ne!(rotated[0], 5020);
    }

    #[test]
    fn test_rotate_pairwise_no_swap_when_forward() {
        // Direction 2 (NW) with rotation 1 → dir 0. NW→E is forward, NO swap.
        // bear(1) looking at elk(2) in dir NW: pair_state = 1*7+2 = 9
        // feature = PHASE_END + 2*49 + 9 = 4961 + 98 + 9 = 5068
        let feature = 5068u16;

        let table_120 = build_rotation_table(1);
        let rotated = rotate_features(&[feature], &table_120, 1).unwrap();

        // dir 2 → dir 0, pair_state stays 9 (no swap)
        // Expected: PHASE_END + 0*49 + 9 = 4961 + 9 = 4970
        assert_eq!(rotated[0], 4970);
    }

    #[test]
    fn test_rotation_120_then_240_is_identity_for_pairwise() {
        // Rotating 120° then 240° should give back the original feature
        let table_120 = build_rotation_table(1);
        let table_240 = build_rotation_table(2);

        // Test several pairwise features
        for dir in 0..3 {
            for my in 0..7 {
                for n in 0..7 {
                    if my == 0 && n == 0 {
                        continue;
                    }
                    let pair_state = my * 7 + n;
                    let fi = (4961 + dir * 49 + pair_state) as u16;
                    let rot1 = rotate_features(&[fi], &table_120, 1).unwrap();
                    let rot2 = rotate_features(&rot1, &table_240, 2).unwrap();
                    assert_eq!(
                        rot2[0], fi,
                        "120+240 not identity for dir={}, my={}, n={}: {} → {} → {}",
                        dir, my, n, fi, rot1[0], rot2[0]
                    );
                }
            }
        }
    }

    #[test]
    fn test_rotation_120_three_times_is_identity() {
        let table_120 = build_rotation_table(1);

        // Per-cell feature at center (should always be in bounds)
        let center = 10 * 21 + 10; // cell (0,0) = index 220
        let fi = (center * 11 + 3) as u16; // salmon at center
        let rot1 = rotate_features(&[fi], &table_120, 1).unwrap();
        let rot2 = rotate_features(&rot1, &table_120, 1).unwrap();
        let rot3 = rotate_features(&rot2, &table_120, 1).unwrap();
        assert_eq!(
            rot3[0], fi,
            "3x 120° rotation should be identity for cell features"
        );

        // Pairwise feature
        for dir in 0..3 {
            for ps in 0..49 {
                let fi = (4961 + dir * 49 + ps) as u16;
                let r1 = rotate_features(&[fi], &table_120, 1).unwrap();
                let r2 = rotate_features(&r1, &table_120, 1).unwrap();
                let r3 = rotate_features(&r2, &table_120, 1).unwrap();
                assert_eq!(
                    r3[0], fi,
                    "3x 120° not identity for pairwise dir={}, ps={}",
                    dir, ps
                );
            }
        }
    }

    #[test]
    fn test_feature_block_boundaries() {
        // Verify the constants match between here and nnue.rs
        let expected = if cfg!(feature = "v6-peak") {
            crate::nnue::NUM_FEATURES_V6_PEAK
        } else if cfg!(feature = "czero-feat") {
            crate::nnue::CZERO_FEAT_BASE + crate::nnue::CZERO_FEAT_FEATURES
        } else if cfg!(feature = "v5-feat")
            && cfg!(feature = "mid-features")
            && cfg!(feature = "v4-opp")
        {
            crate::nnue::NUM_FEATURES_MID_V4_V5
        } else if cfg!(feature = "v5-feat") && cfg!(feature = "v4-opp") {
            crate::nnue::NUM_FEATURES_V3_V4_V5
        } else if cfg!(feature = "mid-features") && cfg!(feature = "v4-opp") {
            crate::nnue::NUM_FEATURES_MID_V4
        } else if cfg!(feature = "mid-features") {
            crate::nnue::NUM_FEATURES_MID
        } else if cfg!(feature = "v4-opp") {
            crate::nnue::NUM_FEATURES_V3_V4
        } else {
            crate::nnue::NUM_FEATURES_V3
        };
        assert_eq!(crate::nnue::NUM_FEATURES, expected);
        assert_eq!(crate::nnue::NUM_FEATURES_V2, 10561);
        assert_eq!(crate::nnue::CELL_FEATURES, 4851);
        assert_eq!(crate::nnue::PHASE_FEATURES, 110);
        assert_eq!(crate::nnue::PAIR_FEATURES, 147);
        assert_eq!(crate::nnue::PATTERN_FEATURES, 89);
        assert_eq!(crate::nnue::BAG_FEATURES, 55);
        assert_eq!(crate::nnue::OPP_HAB_FEATURES, 55);
        assert_eq!(crate::nnue::ALLOWED_WL_FEATURES, 2205);
        assert_eq!(crate::nnue::WL_COUNT_EXT_FEATURES, 50);
        assert_eq!(crate::nnue::TERRAIN_PAIR_FEATURES, 108);
        // v3 blocks
        assert_eq!(crate::nnue::CELL_ADJ_FEATURES, 34398);
        assert_eq!(crate::nnue::TBAG_TERRAIN_EXT_FEATURES, 150);
        assert_eq!(crate::nnue::TBAG_WL_EXT_FEATURES, 150);
        assert_eq!(crate::nnue::OVERFLOW_FEATURES, 1);
    }

    #[test]
    fn test_translation_preserves_pairwise_order() {
        // Translation (dir_shift=0) should NEVER swap pairwise pairs
        let table = build_translation_table(1, 0);

        for dir in 0..3 {
            let pair_state = 1 * 7 + 3; // bear-salmon
            let fi = (4961 + dir * 49 + pair_state) as u16;
            // translate_features uses rotate_features with dir_shift=0
            if let Some(trans) = rotate_features(&[fi], &table, 0) {
                let rel = trans[0] as usize - 4961;
                let new_ps = rel % 49;
                assert_eq!(
                    new_ps, pair_state,
                    "Translation should not swap pairwise pair for dir {}",
                    dir
                );
            }
        }
    }
}
