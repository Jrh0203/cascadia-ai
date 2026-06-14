//! MCE performance + correctness bench.
//!
//! Plays N games with champion-strength MCE (mce_wide_v1 settings) and records:
//!   - per-game final scores (for correctness comparison)
//!   - wall-clock per game / per decision
//!   - NNUE forward count (if `MCE_PERF_COUNT=1` is set — see counters in nnue.rs)
//!
//! The first run with `--save-baseline <file>` records final scores; subsequent
//! runs with `--check-baseline <file>` (same seeds) verify exact tie-out.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

use cascadia_ai::mce::{
    best_move_nnue_rollout_mce, default_greedy_mce_candidates, expanded_candidates,
    nnue_prefilter_candidates, GreedyMceAlloc,
};
use cascadia_ai::nnue::NNUENetwork;
use cascadia_ai::search::{execute_scored_move, greedy_move};
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

fn champion_mce_pick(
    game: &GameState,
    net: &NNUENetwork,
    rollouts: usize,
    prefilter_k: usize,
    rng: &mut StdRng,
) -> Option<cascadia_ai::eval::ScoredMove> {
    let mut cands = expanded_candidates(game);
    if cands.len() > prefilter_k {
        cands = nnue_prefilter_candidates(game, net, cands, prefilter_k);
    }
    best_move_nnue_rollout_mce(game, net, rollouts, GreedyMceAlloc::SeqHalving, cands, rng).or_else(
        || {
            let cands = default_greedy_mce_candidates(game);
            if cands.is_empty() {
                greedy_move(game)
            } else {
                Some(cands[0])
            }
        },
    )
}

#[derive(Default, Clone)]
struct GameResult {
    seed: u64,
    final_scores: [u16; 4],
    elapsed_ms: u64,
    decisions: u32,
}

fn play_one(seed: u64, net: &NNUENetwork, rollouts: usize, prefilter_k: usize) -> GameResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
    let mut decisions = 0u32;
    let start = Instant::now();
    while !g.is_game_over() {
        let mv = match champion_mce_pick(&g, net, rollouts, prefilter_k, &mut rng) {
            Some(m) => m,
            None => break,
        };
        if !execute_scored_move(&mut g, &mv) {
            break;
        }
        decisions += 1;
        if decisions > 800 {
            break;
        }
    }
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let mut final_scores = [0u16; 4];
    let score_mode = std::env::var("BENCH_SCORE_MODE").unwrap_or_else(|_| "with-bonus".into());
    for p in 0..4 {
        let total = if score_mode == "base" {
            // Strip the end-of-game habitat-bonus rank to remove the
            // opponent-frozen-board inflation when comparing the decoupled
            // path against the baseline.
            let bd = ScoreBreakdown::compute(&mut g.boards[p].clone(), &g.scoring_cards);
            bd.total
        } else {
            let mut boards = g.boards.clone();
            let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, p);
            bd.total
        };
        final_scores[p] = total;
    }
    GameResult {
        seed,
        final_scores,
        elapsed_ms,
        decisions,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weights = parse_str(&args, "--weights")
        .unwrap_or_else(|| "nnue_weights_v4opp_modal_iter3.bin".to_string());
    let games: usize = parse_arg(&args, "--games").unwrap_or(10);
    let rollouts: usize = parse_arg(&args, "--rollouts").unwrap_or(200);
    let prefilter_k: usize = parse_arg(&args, "--prefilter-k").unwrap_or(8);
    let seed_base: u64 = parse_arg(&args, "--seed").unwrap_or(0xC57AD1A);
    let label = parse_str(&args, "--label").unwrap_or_else(|| "default".to_string());
    let save_baseline = parse_str(&args, "--save-baseline");
    let check_baseline = parse_str(&args, "--check-baseline");
    let serial = args.iter().any(|a| a == "--serial");

    eprintln!(
        "mce_perf_bench[{}]: weights={} games={} rollouts={} prefilter_k={} seed_base=0x{:x} {}",
        label,
        weights,
        games,
        rollouts,
        prefilter_k,
        seed_base,
        if serial {
            "(serial games)"
        } else {
            "(parallel games)"
        },
    );

    let net = NNUENetwork::load(Path::new(&weights)).unwrap_or_else(|e| {
        eprintln!("Failed to load weights: {}", e);
        std::process::exit(1);
    });
    let net = Arc::new(net);

    let seeds: Vec<u64> = (0..games)
        .map(|i| seed_base.wrapping_add(i as u64))
        .collect();
    let wall_start = Instant::now();

    let results: Vec<GameResult> = if serial {
        seeds
            .iter()
            .map(|&s| play_one(s, &net, rollouts, prefilter_k))
            .collect()
    } else {
        seeds
            .par_iter()
            .map(|&s| play_one(s, &net, rollouts, prefilter_k))
            .collect()
    };
    let wall_ms = wall_start.elapsed().as_millis() as u64;

    // Verify against baseline if provided.
    if let Some(baseline_path) = check_baseline.as_ref() {
        let baseline_text = std::fs::read_to_string(baseline_path).unwrap_or_else(|e| {
            eprintln!("Failed to read baseline {}: {}", baseline_path, e);
            std::process::exit(1);
        });
        let mut mismatches = 0;
        for (i, line) in baseline_text.lines().enumerate() {
            if i >= results.len() {
                break;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 5 {
                continue;
            }
            let exp_seed: u64 = parts[0].parse().unwrap_or(0);
            let exp_scores: Vec<u16> = parts[1..5]
                .iter()
                .map(|s| s.parse::<u16>().unwrap_or(0))
                .collect();
            if exp_seed != results[i].seed || exp_scores != results[i].final_scores.to_vec() {
                eprintln!(
                    "BASELINE MISMATCH game {}: exp seed={} scores={:?}, got seed={} scores={:?}",
                    i, exp_seed, exp_scores, results[i].seed, results[i].final_scores
                );
                mismatches += 1;
            }
        }
        if mismatches == 0 {
            eprintln!(
                "  ✓ baseline match: all {} games tied out exactly",
                results.len()
            );
        } else {
            eprintln!("  ✗ baseline mismatch on {} games", mismatches);
            std::process::exit(2);
        }
    }

    if let Some(baseline_path) = save_baseline.as_ref() {
        let mut text = String::new();
        for r in &results {
            text.push_str(&format!(
                "{},{},{},{},{}\n",
                r.seed, r.final_scores[0], r.final_scores[1], r.final_scores[2], r.final_scores[3]
            ));
        }
        std::fs::write(baseline_path, text).unwrap_or_else(|e| {
            eprintln!("Failed to write baseline: {}", e);
            std::process::exit(1);
        });
        eprintln!("  wrote baseline to {}", baseline_path);
    }

    // Summary.
    let total_decisions: u32 = results.iter().map(|r| r.decisions).sum();
    let total_per_game_ms: u64 = results.iter().map(|r| r.elapsed_ms).sum();
    let avg_per_game_ms = total_per_game_ms as f64 / games.max(1) as f64;
    let avg_per_decision_ms = total_per_game_ms as f64 / total_decisions.max(1) as f64;
    let mut all_scores: Vec<u16> = Vec::new();
    for r in &results {
        for s in &r.final_scores {
            all_scores.push(*s);
        }
    }
    let mean = all_scores.iter().map(|&s| s as f64).sum::<f64>() / all_scores.len() as f64;

    eprintln!(
        "\n[{}]  wall_total={:.2}s  per_game={:.0}ms  per_decision={:.1}ms  mean_score={:.2}  decisions={}",
        label,
        wall_ms as f64 / 1000.0,
        avg_per_game_ms,
        avg_per_decision_ms,
        mean,
        total_decisions,
    );

    // Also emit one machine-readable line for grep'ing.
    println!(
        "RESULT label={} wall_ms={} per_game_ms={:.1} per_decision_ms={:.2} mean={:.3} games={} decisions={} rollouts={}",
        label, wall_ms, avg_per_game_ms, avg_per_decision_ms, mean, games, total_decisions, rollouts
    );
}
