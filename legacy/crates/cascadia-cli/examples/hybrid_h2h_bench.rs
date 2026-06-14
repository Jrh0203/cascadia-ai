//! Value-function H2H bench for HybridNet.
//!
//! For each α in the sweep, plays `--games` games where all 4 seats use
//! NNUE-style argmax move selection but the value function is HybridNetwork
//! (NNUE + α·Δ). Compares against a pure-NNUE baseline (α=0, which the
//! Rust fast-path proves is bit-identical to NNUE alone).
//!
//! This is the cheap, fast bench — no MCE rollouts. It isolates the
//! contribution of Δ to *value-function quality alone*. A later H2H with
//! MCE in the pipeline measures whether the Δ signal survives search.
//!
//! Usage:
//!   cargo run --release -p cascadia-cli --features mid-features,v4-opp,accelerate \
//!     --example hybrid_h2h_bench -- \
//!     --weights nnue_weights_v4opp_modal_iter3.bin \
//!     --azr3    /tmp/hybr_smoke.azr3 \
//!     --alphas  0.0,0.1,0.2,0.3,0.5 \
//!     --games   50

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

use cascadia_ai::endgame::{solve_endgame, EndgameConfig};
use cascadia_ai::hybrid::HybridNetwork;
use cascadia_ai::nnue::{extract_features_with_bag, BagInfo, NNUENetwork};
use cascadia_ai::search::{candidate_moves_pub, execute_scored_move};
use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

fn parse_arg<T: std::str::FromStr>(args: &[String], key: &str) -> Option<T> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_str(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Pick the candidate that maximizes `current_score_after_move +
/// value_estimate`. `value_estimate` comes from `evaluate(board, features)`
/// supplied as a closure so the same harness works for NNUE-only and
/// Hybrid.
fn pick_best_move_with_value<F>(
    game: &GameState,
    extract: &F,
) -> Option<cascadia_ai::eval::ScoredMove>
where
    F: Fn(&cascadia_core::board::Board, &[u16]) -> f32,
{
    let candidates: Vec<_> = candidate_moves_pub(game).into_iter().take(15).collect();
    if candidates.is_empty() {
        return None;
    }
    let player = game.current_player;
    let mut best: Option<(cascadia_ai::eval::ScoredMove, f32)> = None;
    for mv in &candidates {
        let mut g = game.clone();
        if !execute_scored_move(&mut g, mv) {
            continue;
        }
        let board = &g.boards[player];
        let bag = BagInfo::from_game_for_player(&g, player);
        let features = extract_features_with_bag(board, Some(&bag));
        let mut boards = g.boards.clone();
        let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, player);
        let current = bd.total as f32;
        let value = extract(board, &features);
        let total = current + value;
        match best {
            None => best = Some((*mv, total)),
            Some((_, t)) if total > t => best = Some((*mv, total)),
            _ => {}
        }
    }
    best.map(|(m, _)| m)
}

#[derive(Clone, Copy)]
enum Strategy<'a> {
    Nnue(&'a NNUENetwork),
    Hybrid(&'a HybridNetwork),
    /// NNUE-argmax for `turns_remaining > endgame.max_depth`, exact endgame
    /// solver below that threshold. Tests Lever 2 in isolation.
    NnueWithEndgame(&'a NNUENetwork, EndgameConfig),
}

fn play_game(strategies: &[Strategy<'_>; 4], seed: u64) -> [f32; 4] {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
    let mut steps = 0usize;
    while !g.is_game_over() {
        let player = g.current_player;
        let mv = match strategies[player] {
            Strategy::Nnue(net) => pick_best_move_with_value(&g, &|_b, f| net.forward(f)),
            Strategy::Hybrid(h) => pick_best_move_with_value(&g, &|b, f| h.evaluate(b, f)),
            Strategy::NnueWithEndgame(net, cfg) => {
                if let Some((m, _)) = solve_endgame(&g, net, cfg) {
                    Some(m)
                } else {
                    pick_best_move_with_value(&g, &|_b, f| net.forward(f))
                }
            }
        };
        let mv = match mv {
            Some(m) => m,
            None => break,
        };
        if !execute_scored_move(&mut g, &mv) {
            break;
        }
        steps += 1;
        if steps > 800 {
            break;
        }
    }
    let mut final_scores = [0.0f32; 4];
    for p in 0..4 {
        let mut boards = g.boards.clone();
        let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, p);
        final_scores[p] = bd.total as f32;
    }
    final_scores
}

fn run_strategy_bench(label: &str, strategies: [Strategy<'_>; 4], games: usize, seed_base: u64) {
    let totals = Mutex::new(Vec::<f32>::new());
    let game_indices: Vec<usize> = (0..games).collect();
    game_indices.into_par_iter().for_each(|gi| {
        let scores = play_game(&strategies, seed_base.wrapping_add(gi as u64));
        let mut lock = totals.lock().unwrap();
        for s in &scores {
            lock.push(*s);
        }
    });
    let scores = totals.into_inner().unwrap();
    let n = scores.len() as f32;
    let mean = scores.iter().sum::<f32>() / n;
    let var = scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let stderr = (var / n).sqrt();
    let p10 = percentile(&scores, 0.10);
    let p50 = percentile(&scores, 0.50);
    let p90 = percentile(&scores, 0.90);
    println!(
        "  {:30}  N={:4}  mean={:6.2} ± {:.2}  P10={:.0}  P50={:.0}  P90={:.0}",
        label, n as usize, mean, stderr, p10, p50, p90
    );
}

fn percentile(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f32 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Run an alternating-seat H2H. Each game has 2 seats with strategy A
/// and 2 seats with strategy B (interleaved 0,1,0,1 or 1,0,1,0). Across
/// the games we ensure each interleaving is played equally often so any
/// per-seat bias cancels. Returns (means_a, means_b) over all seat-games.
fn run_h2h_alternating(
    label: &str,
    a: Strategy<'_>,
    b: Strategy<'_>,
    games: usize,
    seed_base: u64,
) {
    // For each game, also play the mirror — flips the seat assignment.
    let a_scores = Mutex::new(Vec::<f32>::new());
    let b_scores = Mutex::new(Vec::<f32>::new());
    let game_indices: Vec<usize> = (0..games).collect();
    game_indices.into_par_iter().for_each(|gi| {
        // First seat assignment: A on seats 0,2; B on seats 1,3.
        let s1: [Strategy<'_>; 4] = [a, b, a, b];
        let r1 = play_game(&s1, seed_base.wrapping_add(gi as u64 * 2));
        // Mirror: A on seats 1,3; B on seats 0,2. Same RNG seed +1.
        let s2: [Strategy<'_>; 4] = [b, a, b, a];
        let r2 = play_game(&s2, seed_base.wrapping_add(gi as u64 * 2 + 1));
        let mut la = a_scores.lock().unwrap();
        la.push(r1[0]);
        la.push(r1[2]);
        la.push(r2[1]);
        la.push(r2[3]);
        let mut lb = b_scores.lock().unwrap();
        lb.push(r1[1]);
        lb.push(r1[3]);
        lb.push(r2[0]);
        lb.push(r2[2]);
    });
    let a_scores = a_scores.into_inner().unwrap();
    let b_scores = b_scores.into_inner().unwrap();
    let na = a_scores.len() as f32;
    let nb = b_scores.len() as f32;
    let ma = a_scores.iter().sum::<f32>() / na;
    let mb = b_scores.iter().sum::<f32>() / nb;
    let va = a_scores.iter().map(|x| (x - ma).powi(2)).sum::<f32>() / na;
    let vb = b_scores.iter().map(|x| (x - mb).powi(2)).sum::<f32>() / nb;
    let sea = (va / na).sqrt();
    let seb = (vb / nb).sqrt();
    let delta = mb - ma;
    let se_delta = (va / na + vb / nb).sqrt();
    // Per-game pairwise win-rate (counts seat-pairs A vs B within the
    // SAME seat layout — A on seat 0 vs B on seat 1 etc).
    let mut wins_a = 0;
    let mut wins_b = 0;
    let mut ties = 0;
    for (a_score, b_score) in a_scores.iter().zip(b_scores.iter()) {
        if a_score > b_score {
            wins_a += 1;
        } else if a_score < b_score {
            wins_b += 1;
        } else {
            ties += 1;
        }
    }
    let n = a_scores.len() as f32;
    println!(
        "  {:30}  A={:6.2}±{:.2}  B={:6.2}±{:.2}  ΔB-A={:+.2}±{:.2}  win B={:.1}%  tie={:.1}%",
        label,
        ma,
        sea,
        mb,
        seb,
        delta,
        se_delta,
        100.0 * wins_b as f32 / n,
        100.0 * ties as f32 / n,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weights = parse_str(&args, "--weights")
        .unwrap_or_else(|| "nnue_weights_v4opp_modal_iter3.bin".to_string());
    let azr3 = parse_str(&args, "--azr3");
    let alphas_str = parse_str(&args, "--alphas").unwrap_or_else(|| "0.0,0.2,0.3".to_string());
    let games: usize = parse_arg(&args, "--games").unwrap_or(50);
    let seed: u64 = parse_arg(&args, "--seed").unwrap_or(0xBE_E_F);
    let do_h2h: bool = args.iter().any(|a| a == "--h2h");
    let do_endgame: bool = args.iter().any(|a| a == "--endgame");
    let endgame_depth: u8 = parse_arg(&args, "--endgame-depth").unwrap_or(4);
    let endgame_k: usize = parse_arg(&args, "--endgame-k").unwrap_or(8);

    let alphas: Vec<f32> = alphas_str
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();

    eprintln!(
        "hybrid_h2h_bench: weights={} azr3={:?} alphas={:?} games={} seed=0x{:x}",
        weights, azr3, alphas, games, seed
    );

    let nnue = NNUENetwork::load(Path::new(&weights)).expect("load nnue");
    let nnue_arc = Arc::new(nnue.clone());

    // Baseline: pure NNUE in all 4 seats.
    println!("\n== Baseline (pure NNUE, all 4 seats) ==");
    let n_ref: &NNUENetwork = &nnue_arc;
    run_strategy_bench(
        "NNUE (all 4 seats)",
        [Strategy::Nnue(n_ref); 4],
        games,
        seed,
    );

    // Endgame block first — runs independent of --azr3.
    if do_endgame {
        let cfg = EndgameConfig {
            max_depth: endgame_depth,
            prefilter_k: endgame_k,
        };
        println!(
            "\n== Endgame solver bench (depth={}, prefilter_k={}) ==",
            cfg.max_depth, cfg.prefilter_k
        );
        let n_ref: &NNUENetwork = &nnue_arc;
        run_strategy_bench(
            "NNUE + Endgame (all 4 seats)",
            [Strategy::NnueWithEndgame(n_ref, cfg); 4],
            games,
            seed,
        );
        if do_h2h {
            println!("\n== Alternating-seat H2H (A = NNUE only, B = NNUE+Endgame) ==");
            let label = format!(
                "NNUE+Endgame(d={},k={}) vs NNUE",
                cfg.max_depth, cfg.prefilter_k
            );
            run_h2h_alternating(
                &label,
                Strategy::Nnue(n_ref),
                Strategy::NnueWithEndgame(n_ref, cfg),
                games,
                seed,
            );
        }
    }

    // Hybrid sweep — requires --azr3.
    let Some(ref azr3_path) = azr3 else {
        if !do_endgame {
            eprintln!("\n(no --azr3 given; baseline-only run)");
        }
        return;
    };

    let hybrids: Vec<HybridNetwork> = alphas
        .iter()
        .map(|&alpha| {
            let mut h = HybridNetwork::load_with_nnue(Path::new(&azr3_path), nnue.clone())
                .expect("load hybrid");
            h.alpha = alpha;
            h
        })
        .collect();

    println!("\n== Hybrid sweep (NNUE + α·Δ, all 4 seats) ==");
    for (alpha, h) in alphas.iter().zip(hybrids.iter()) {
        let label = format!("α = {:.2}", alpha);
        let h_ref: &HybridNetwork = h;
        run_strategy_bench(&label, [Strategy::Hybrid(h_ref); 4], games, seed);
    }

    if do_h2h && azr3.is_some() {
        println!("\n== Alternating-seat H2H (A = NNUE, B = Hybrid at α) ==");
        let n_ref: &NNUENetwork = &nnue_arc;
        for (alpha, h) in alphas.iter().zip(hybrids.iter()) {
            if *alpha == 0.0 {
                continue; // α=0 is identical to NNUE — H2H would be trivial
            }
            let label = format!("Hybrid α={:.2} vs NNUE", alpha);
            let h_ref: &HybridNetwork = h;
            run_h2h_alternating(
                &label,
                Strategy::Nnue(n_ref),
                Strategy::Hybrid(h_ref),
                games,
                seed,
            );
        }
    }
}
