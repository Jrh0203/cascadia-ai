use std::sync::Arc;
use std::thread;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use cascadia_core::game::GameState;
use cascadia_core::types::ScoringCards;
use cascadia_ai::eval::{best_move_with_potential, best_move_lookahead};
use cascadia_ai::ntuple::NTupleNetwork;
use cascadia_ai::search::{best_move_beam, best_move_mcts, execute_scored_move, greedy_move};

#[derive(Clone)]
enum Strategy {
    Greedy,
    Lookahead1,
    Beam { width: usize, depth: usize },
    MonteCarlo { rollouts: usize },
    NTuple { net: Arc<NTupleNetwork> },
    NNUE { net: Arc<cascadia_ai::nnue::NNUENetwork> },
    MCE { net: Arc<cascadia_ai::nnue::NNUENetwork>, rollouts: usize },
    Expectimax { net: Arc<cascadia_ai::nnue::NNUENetwork>, samples: usize, depth: usize, branching: usize },
    ExactExpectimax { net: Arc<cascadia_ai::nnue::NNUENetwork> },
    Hybrid { net: Arc<cascadia_ai::nnue::NNUENetwork>, rollouts: usize, top_k: usize },
    MCTS { net: Arc<cascadia_ai::nnue::NNUENetwork>, simulations: usize },
    PolicyMCE { net: Arc<cascadia_ai::nnue::NNUENetwork>, policy: Arc<cascadia_ai::nnue::PolicyNetwork>, rollouts: usize, top_k: usize },
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Strategy::Greedy => write!(f, "greedy"),
            Strategy::Lookahead1 => write!(f, "lookahead-1"),
            Strategy::Beam { width, depth } => write!(f, "beam(w={},d={})", width, depth),
            Strategy::MonteCarlo { rollouts } => write!(f, "monte-carlo(n={})", rollouts),
            Strategy::NTuple { .. } => write!(f, "ntuple"),
            Strategy::NNUE { .. } => write!(f, "nnue"),
            Strategy::MCE { rollouts, .. } => write!(f, "mce(n={})", rollouts),
            Strategy::Expectimax { samples, depth, branching, .. } => write!(f, "expectimax(k={},d={},b={})", samples, depth, branching),
            Strategy::ExactExpectimax { .. } => write!(f, "exact-expectimax"),
            Strategy::Hybrid { rollouts, top_k, .. } => write!(f, "hybrid(k={},n={})", top_k, rollouts),
            Strategy::MCTS { simulations, .. } => write!(f, "mcts(n={})", simulations),
            Strategy::PolicyMCE { rollouts, top_k, .. } => write!(f, "policy-mce(k={},n={})", top_k, rollouts),
        }
    }
}

fn pick_move(
    game: &GameState,
    strategy: &Strategy,
    cards: &ScoringCards,
    search_rng: &mut StdRng,
) -> Option<cascadia_ai::eval::ScoredMove> {
    match strategy {
        Strategy::Greedy => {
            let mp: Vec<_> = game.market.available()
                .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
            let turns = game.turns_remaining;
            let mut board = game.boards[game.current_player].clone();
            best_move_with_potential(&mut board, &mp, cards, turns)
        }
        Strategy::Lookahead1 => best_move_lookahead(game),
        Strategy::Beam { width, depth } => best_move_beam(game, *width, *depth),
        Strategy::MonteCarlo { rollouts } => best_move_mcts(game, *rollouts, search_rng),
        Strategy::NTuple { net } => cascadia_ai::train::pick_best_move_ntuple(game, net),
        Strategy::NNUE { net } => cascadia_ai::nnue_train::pick_best_move_nnue(game, net),
        Strategy::MCE { net, rollouts } => cascadia_ai::mce::best_move_mce(game, net, *rollouts, search_rng),
        Strategy::Expectimax { net, samples, depth, branching } => {
            if *depth <= 1 {
                cascadia_ai::expectimax::best_move_expectimax(game, net, *samples, search_rng)
            } else {
                cascadia_ai::expectimax::best_move_expectimax_deep(game, net, *samples, *depth, *branching, search_rng)
            }
        }
        Strategy::ExactExpectimax { net } => {
            let depth: usize = std::env::var("EXPECTIMAX_DEPTH")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            cascadia_ai::mce::best_move_expectimax_nply(game, net, depth)
        }
        Strategy::Hybrid { net, rollouts, top_k } => {
            cascadia_ai::mce::best_move_hybrid(game, net, *rollouts, *top_k, search_rng)
        }
        Strategy::MCTS { net, simulations } => {
            cascadia_ai::mcts::best_move_mcts(game, net, *simulations)
        }
        Strategy::PolicyMCE { net, policy, rollouts, top_k } => {
            cascadia_ai::mce::best_move_mce_with_policy(game, net, policy, *rollouts, *top_k, search_rng)
        }
    }
}

/// Fast greedy evaluation of a game state (for pre-move decisions).
/// Using greedy instead of the full strategy keeps pre-move logic fast.
fn greedy_eval(game: &GameState, cards: &ScoringCards) -> f32 {
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    let turns = game.turns_remaining;
    let mut board = game.boards[game.current_player].clone();
    best_move_with_potential(&mut board, &mp, cards, turns)
        .map(|m| m.score as f32)
        .unwrap_or(0.0)
}

/// Pre-move optimization: decide whether to replace 3-of-a-kind or mulligan.
/// Uses enumerated mulligan analysis with NNUE when available (exact EV over
/// all 625 possible draws), falls back to greedy sampling otherwise.
fn pre_move_optimize(
    game: &mut GameState,
    strategy: &Strategy,
    cards: &ScoringCards,
    search_rng: &mut StdRng,
) {
    // Extract NNUE net if available
    let net = match strategy {
        Strategy::NNUE { ref net } | Strategy::MCE { ref net, .. }
            | Strategy::Hybrid { ref net, .. } | Strategy::ExactExpectimax { ref net }
            | Strategy::MCTS { ref net, .. }
            | Strategy::PolicyMCE { ref net, .. } => Some(net.clone()),
        _ => None,
    };

    const MAX_MULLIGANS: usize = 5;
    let mut mulligans_used = 0;

    loop {
        // Use enumerated analysis when NNUE is available
        if let Some(ref net) = net {
            let analysis = cascadia_ai::mce::analyze_mulligan_fast(game, net);

            // Option 1: Replace 3-of-a-kind (free) — only if it improves
            if game.can_replace_overflow().is_some() {
                // Check if replacing improves: compare current_best with post-replace best
                let mut test = game.clone();
                test.replace_overflow();
                let post_analysis = cascadia_ai::mce::analyze_mulligan_fast(&test, net);
                if post_analysis.current_best > analysis.current_best {
                    game.replace_overflow();
                    continue;
                }
            }

            // Option 2: Enumerated mulligan (exact EV)
            if mulligans_used < MAX_MULLIGANS && analysis.should_mulligan {
                if game.mulligan_wildlife() {
                    mulligans_used += 1;
                    continue;
                }
            }

            // Option 3: Mulligan + pinecone (exact EV, costs 2 tokens)
            if mulligans_used < MAX_MULLIGANS && analysis.should_mulligan_pinecone {
                if game.mulligan_wildlife() {
                    mulligans_used += 1;
                    continue;
                }
            }

            break;
        }

        // Fallback: greedy evaluation (no NNUE)
        let baseline = greedy_eval(game, cards);

        if game.can_replace_overflow().is_some() {
            let mut test = game.clone();
            test.replace_overflow();
            if greedy_eval(&test, cards) > baseline + 0.5 {
                game.replace_overflow();
                continue;
            }
        }

        if mulligans_used < MAX_MULLIGANS && game.boards[game.current_player].nature_tokens > 0 {
            let mut total = 0.0f32;
            let mut samples = 0;
            for _ in 0..3 {
                let mut test = game.clone();
                test.shuffle_bags(search_rng);
                if test.mulligan_wildlife() {
                    total += greedy_eval(&test, cards);
                    samples += 1;
                }
            }
            if samples > 0 && total / samples as f32 > baseline + 1.5 {
                if game.mulligan_wildlife() {
                    mulligans_used += 1;
                    continue;
                }
            }
        }

        break;
    }
}

/// Slow pre-move optimization using the full strategy (MCE) for evaluation.
fn pre_move_optimize_slow(
    game: &mut GameState,
    strategy: &Strategy,
    cards: &ScoringCards,
    search_rng: &mut StdRng,
) {
    const MULLIGAN_SAMPLES: usize = 3;
    const MAX_MULLIGANS: usize = 5;

    let eval_with_strategy = |g: &GameState, rng: &mut StdRng| -> f32 {
        pick_move(g, strategy, cards, rng)
            .map(|m| m.score as f32)
            .unwrap_or(0.0)
    };

    let mut mulligans_used = 0;
    loop {
        let baseline = eval_with_strategy(game, search_rng);

        if game.can_replace_overflow().is_some() {
            let mut test = game.clone();
            test.replace_overflow();
            if eval_with_strategy(&test, search_rng) > baseline + 0.5 {
                game.replace_overflow();
                continue;
            }
        }

        if mulligans_used < MAX_MULLIGANS && game.boards[game.current_player].nature_tokens > 0 {
            let mut total = 0.0f32;
            let mut samples = 0;
            for _ in 0..MULLIGAN_SAMPLES {
                let mut test = game.clone();
                test.shuffle_bags(search_rng);
                if test.mulligan_wildlife() {
                    total += eval_with_strategy(&test, search_rng);
                    samples += 1;
                }
            }
            if samples > 0 {
                let expected = total / samples as f32;
                if expected > baseline + 1.5 {
                    if game.mulligan_wildlife() {
                        mulligans_used += 1;
                        continue;
                    }
                }
            }
        }
        break;
    }
}

fn simulate_game(rng: &mut StdRng, strategy: &Strategy) -> (cascadia_core::scoring::ScoreBreakdown, cascadia_core::scoring::ScoreBreakdown) {
    simulate_game_inner(rng, strategy, None)
}

fn simulate_game_inner(
    rng: &mut StdRng,
    strategy: &Strategy,
    mut sample_sink: Option<&mut Vec<(Vec<u16>, f32)>>,
) -> (cascadia_core::scoring::ScoreBreakdown, cascadia_core::scoring::ScoreBreakdown) {
    let cards = ScoringCards::all_a();
    let mut game = GameState::new(4, cards, rng);
    let mut search_rng = StdRng::seed_from_u64(rng.gen());

    while !game.is_game_over() {
        // Player 0 is the AI; players 1-3 use NNUE if available, otherwise greedy
        if game.current_player != 0 {
            let opp_mv = match strategy {
                Strategy::NNUE { ref net } | Strategy::MCE { ref net, .. }
                    | Strategy::Hybrid { ref net, .. } | Strategy::ExactExpectimax { ref net }
                    | Strategy::MCTS { ref net, .. }
                    | Strategy::PolicyMCE { ref net, .. } => {
                    cascadia_ai::nnue_train::pick_best_move_nnue(&game, net)
                        .or_else(|| greedy_move(&game))
                }
                _ => greedy_move(&game),
            };
            match opp_mv {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) { break; }
                }
                None => break,
            }
            continue;
        }

        // Pre-move: decide whether to replace 3-of-a-kind (free) or mulligan (costs token).
        // Iterative: keep applying pre-move actions as long as expected value improves.
        pre_move_optimize(&mut game, strategy, &cards, &mut search_rng);

        // If MCE strategy and sample_sink provided, collect training samples
        // from the same MCE run (avoids running the pipeline twice).
        let mv = if sample_sink.is_some() {
            if let Strategy::MCE { ref net, rollouts } = strategy {
                let tops = cascadia_ai::mce::top_moves_mce(&game, net, *rollouts, &mut search_rng, 15);
                // Collect afterstate samples from all evaluated candidates
                for (mv, avg) in &tops {
                    let mut g = game.clone();
                    if cascadia_ai::search::execute_scored_move(&mut g, mv) {
                        let current = cascadia_core::scoring::ScoreBreakdown::compute(
                            &mut g.boards[game.current_player], &g.scoring_cards,
                        ).total as f32;
                        let target = (*avg as f32 - current).max(0.0);
                        let features = cascadia_ai::nnue::extract_features(&g.boards[game.current_player]);
                        sample_sink.as_mut().unwrap().push((features, target));
                    }
                }
                tops.into_iter().next().map(|(mv, avg)| {
                    cascadia_ai::eval::ScoredMove { score: avg.round() as u16, ..mv }
                })
            } else {
                pick_move(&game, strategy, &cards, &mut search_rng)
            }
        } else {
            pick_move(&game, strategy, &cards, &mut search_rng)
        };

        match mv {
            Some(mv) => {
                if !execute_scored_move(&mut game, &mv) { break; }
            }
            None => break,
        }
    }

    // Return both base score and score with habitat bonuses
    let base = cascadia_core::scoring::ScoreBreakdown::compute(
        &mut game.boards[0],
        &game.scoring_cards,
    );
    let with_bonus = cascadia_core::scoring::ScoreBreakdown::compute_with_bonuses(
        &mut game.boards,
        &game.scoring_cards,
        0,
    );
    (base, with_bonus)
}

struct BenchResult {
    strategy: String,
    scores: Vec<u16>,          // base scores (no habitat bonus)
    scores_with_bonus: Vec<u16>, // scores with habitat majority bonus
    elapsed: std::time::Duration,
    avg_habitat: [f64; 5],
    avg_wildlife: [f64; 5],
    avg_tokens: f64,
    avg_habitat_bonus: f64,
}

fn run_benchmark(strategy: &Strategy, num_games: usize) -> BenchResult {
    let start = Instant::now();
    let mut scores: Vec<u16> = Vec::with_capacity(num_games);
    let mut scores_with_bonus: Vec<u16> = Vec::with_capacity(num_games);
    // Track average score breakdown
    let mut total_habitat = [0u64; 5];
    let mut total_wildlife = [0u64; 5];
    let mut total_tokens = 0u64;
    let mut total_habitat_bonus = 0u64;

    // For MCE strategy, automatically collect training samples as a side effect
    let is_mce = matches!(strategy, Strategy::MCE { .. });
    let samples_path = std::path::Path::new("mce_policy_samples.bin");
    let mut total_samples = 0usize;

    // Support seed offset for distributed benchmarking (env var CASCADIA_SEED_OFFSET)
    let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    for i in 0..num_games {
        let mut rng = StdRng::seed_from_u64(i as u64 + seed_offset);
        let (base, with_bonus) = if is_mce {
            let mut game_samples = Vec::new();
            let result = simulate_game_inner(&mut rng, strategy, Some(&mut game_samples));
            if !game_samples.is_empty() {
                total_samples += game_samples.len();
                let _ = cascadia_ai::nnue_train::append_mce_samples(samples_path, &game_samples);
            }
            result
        } else {
            simulate_game(&mut rng, strategy)
        };
        scores.push(base.total);
        scores_with_bonus.push(with_bonus.total);
        for t in 0..5 {
            total_habitat[t] += base.habitat[t] as u64;
            total_wildlife[t] += base.wildlife[t] as u64;
        }
        total_tokens += base.nature_tokens as u64;
        total_habitat_bonus += with_bonus.habitat_bonus.iter().map(|&b| b as u64).sum::<u64>();
    }

    if total_samples > 0 {
        eprintln!("  [Collected {} MCE policy samples → {}]", total_samples, samples_path.display());
    }

    let n = num_games as f64;
    scores.sort();
    scores_with_bonus.sort();
    BenchResult {
        strategy: strategy.to_string(),
        scores,
        scores_with_bonus,
        elapsed: start.elapsed(),
        avg_habitat: [
            total_habitat[0] as f64 / n,
            total_habitat[1] as f64 / n,
            total_habitat[2] as f64 / n,
            total_habitat[3] as f64 / n,
            total_habitat[4] as f64 / n,
        ],
        avg_wildlife: [
            total_wildlife[0] as f64 / n,
            total_wildlife[1] as f64 / n,
            total_wildlife[2] as f64 / n,
            total_wildlife[3] as f64 / n,
            total_wildlife[4] as f64 / n,
        ],
        avg_tokens: total_tokens as f64 / n,
        avg_habitat_bonus: total_habitat_bonus as f64 / n,
    }
}

fn print_result(r: &BenchResult) {
    let n = r.scores.len();
    let sum: u64 = r.scores.iter().map(|&s| s as u64).sum();
    let mean = sum as f64 / n as f64;
    let min = r.scores[0];
    let max = r.scores[n - 1];
    let median = r.scores[n / 2];
    let p10 = r.scores[n / 10];
    let p25 = r.scores[n / 4];
    let p75 = r.scores[3 * n / 4];
    let p90 = r.scores[9 * n / 10];

    let variance: f64 = r.scores.iter().map(|&s| {
        let diff = s as f64 - mean;
        diff * diff
    }).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    // With-bonus stats
    let sum_b: u64 = r.scores_with_bonus.iter().map(|&s| s as u64).sum();
    let mean_b = sum_b as f64 / n as f64;
    let median_b = r.scores_with_bonus[n / 2];
    let p10_b = r.scores_with_bonus[n / 10];
    let p90_b = r.scores_with_bonus[9 * n / 10];

    println!("Results ({} games in {:.1?}, strategy={}):", n, r.elapsed, r.strategy);
    println!("  Base Score (no habitat bonus):");
    println!("    Mean:    {:.1}", mean);
    println!("    Median:  {}", median);
    println!("    P10:     {}", p10);
    println!("    P90:     {}", p90);
    println!("    Min/Max: {}/{}", min, max);
    println!("  With Habitat Bonus:");
    println!("    Mean:    {:.1} (+{:.1} avg bonus)", mean_b, r.avg_habitat_bonus);
    println!("    Median:  {}", median_b);
    println!("    P10:     {}", p10_b);
    println!("    P90:     {}", p90_b);
    println!();
    let terrains = ["Forest", "Prairie", "Wetland", "Mountain", "River"];
    let wildlife = ["Bear", "Elk", "Salmon", "Hawk", "Fox"];
    println!("  Score Breakdown (averages):");
    let hab_total: f64 = r.avg_habitat.iter().sum();
    let wl_total: f64 = r.avg_wildlife.iter().sum();
    println!("    Habitat:  {:.1} total (+{:.1} bonus)", hab_total, r.avg_habitat_bonus);
    for (i, name) in terrains.iter().enumerate() {
        println!("      {:<10} {:.1}", name, r.avg_habitat[i]);
    }
    println!("    Wildlife: {:.1} total", wl_total);
    for (i, name) in wildlife.iter().enumerate() {
        println!("      {:<10} {:.1}", name, r.avg_wildlife[i]);
    }
    println!("    Tokens:   {:.1}", r.avg_tokens);
    println!();

    let bucket_size = 5;
    let min_bucket = (min as usize / bucket_size) * bucket_size;
    let max_bucket = (max as usize / bucket_size + 1) * bucket_size;
    println!("  Score Distribution:");
    let mut bucket = min_bucket;
    while bucket < max_bucket {
        let count = r.scores.iter().filter(|&&s| {
            (s as usize) >= bucket && (s as usize) < bucket + bucket_size
        }).count();
        let bar_len = (count * 60) / n.max(1);
        let bar: String = "█".repeat(bar_len);
        println!("  {:3}-{:3}: {:5} {}", bucket, bucket + bucket_size - 1, count, bar);
        bucket += bucket_size;
    }
    println!();
}

fn print_comparison(results: &[BenchResult]) {
    println!("╔══════════════════════════╦════════╦════════╦════════╦════════╦════════╦═══════════╗");
    println!("║ Strategy                 ║  Mean  ║ Median ║  P10   ║  P90   ║ StdDev ║ Time      ║");
    println!("╠══════════════════════════╬════════╬════════╬════════╬════════╬════════╬═══════════╣");
    for r in results {
        let n = r.scores.len();
        let sum: u64 = r.scores.iter().map(|&s| s as u64).sum();
        let mean = sum as f64 / n as f64;
        let median = r.scores[n / 2];
        let p10 = r.scores[n / 10];
        let p90 = r.scores[9 * n / 10];
        let variance: f64 = r.scores.iter().map(|&s| {
            let diff = s as f64 - mean;
            diff * diff
        }).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        println!(
            "║ {:<24} ║ {:>6.1} ║ {:>6} ║ {:>6} ║ {:>6} ║ {:>6.1} ║ {:>9.1?} ║",
            r.strategy, mean, median, p10, p90, std_dev, r.elapsed,
        );
    }
    println!("╚══════════════════════════╩════════╩════════╩════════╩════════╩════════╩═══════════╝");
}

fn run_gym_server(weights_path: &str) {
    use std::io::{BufRead, Write};

    let net = Arc::new(
        cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
            .expect("Failed to load NNUE weights")
    );

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut game: Option<GameState> = None;
    let mut rng = StdRng::from_entropy();
    let mut candidates: Vec<cascadia_ai::eval::ScoredMove> = Vec::new();
    let mut candidate_features: Vec<Vec<u16>> = Vec::new();
    let mut prev_score: u16 = 0;

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() { continue; }

        match parts[0] {
            "reset" => {
                let cards = ScoringCards::all_a();
                let mut g = GameState::new(4, cards, &mut rng);
                // Advance past opponents until player 0's turn
                while !g.is_game_over() && g.current_player != 0 {
                    let mv = cascadia_ai::nnue_train::pick_best_move_nnue(&g, &net)
                        .or_else(|| greedy_move(&g));
                    match mv {
                        Some(mv) => { if !execute_scored_move(&mut g, &mv) { break; } }
                        None => break,
                    }
                }
                // Generate candidates
                candidates = cascadia_ai::search::candidate_moves_pub(&g);
                let bag_info = cascadia_ai::nnue::BagInfo::from_game(&g);
                candidate_features = candidates.iter().map(|mv| {
                    let mut gc = g.clone();
                    if cascadia_ai::search::execute_scored_move(&mut gc, mv) {
                        cascadia_ai::nnue::extract_features_with_bag(&gc.boards[0], Some(&bag_info))
                    } else {
                        vec![]
                    }
                }).collect();

                let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut g.boards[0].clone(), &g.scoring_cards,
                ).total;
                let n_cands = candidates.len();
                let done = g.is_game_over();
                prev_score = current_score;

                game = Some(g);
                writeln!(out, "{{\"n_candidates\":{},\"current_score\":{},\"done\":{}}}", n_cands, current_score, done).unwrap();
                out.flush().unwrap();
            }
            "obs" => {
                // Return current board features + per-candidate ESTIMATED FINAL SCORES
                // (actual_score_after_move + nnue_remaining_estimate)
                if let Some(ref g) = game {
                    let bag_info = cascadia_ai::nnue::BagInfo::from_game(g);
                    let board_features = cascadia_ai::nnue::extract_features_with_bag(
                        &g.boards[0], Some(&bag_info));

                    // Compute estimated final score for each candidate
                    let scores: Vec<f32> = candidates.iter().enumerate().map(|(i, mv)| {
                        let mut gc = g.clone();
                        if !cascadia_ai::search::execute_scored_move(&mut gc, mv) { return 0.0; }
                        let actual = cascadia_core::scoring::ScoreBreakdown::compute(
                            &mut gc.boards[0], &gc.scoring_cards,
                        ).total as f32;
                        let features = &candidate_features[i];
                        let remaining = if features.is_empty() { 0.0 } else { net.forward(features) };
                        actual + remaining
                    }).collect();

                    let json = format!("{{\"board_features\":{:?},\"candidate_scores\":{:?}}}",
                        board_features, scores);
                    writeln!(out, "{}", json).unwrap();
                } else {
                    writeln!(out, "{{\"board_features\":[],\"candidate_scores\":[]}}").unwrap();
                }
                out.flush().unwrap();
            }
            "step" => {
                let action: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                if let Some(ref mut g) = game {
                    let reward;
                    let done;

                    // Execute chosen candidate
                    if action < candidates.len() {
                        execute_scored_move(g, &candidates[action]);
                    }

                    // Advance opponents until player 0's turn again
                    while !g.is_game_over() && g.current_player != 0 {
                        let mv = cascadia_ai::nnue_train::pick_best_move_nnue(g, &net)
                            .or_else(|| greedy_move(g));
                        match mv {
                            Some(mv) => { if !execute_scored_move(g, &mv) { break; } }
                            None => break,
                        }
                    }

                    done = g.is_game_over();

                    // Compute current score and per-step delta reward
                    let new_score = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut g.boards[0].clone(), &g.scoring_cards,
                    ).total;
                    reward = (new_score as i32 - prev_score as i32) as f32;
                    prev_score = new_score;

                    if done {
                        candidates.clear();
                        candidate_features.clear();
                    } else {
                        // Generate new candidates
                        candidates = cascadia_ai::search::candidate_moves_pub(g);
                        let bag_info = cascadia_ai::nnue::BagInfo::from_game(g);
                        candidate_features = candidates.iter().map(|mv| {
                            let mut gc = g.clone();
                            if cascadia_ai::search::execute_scored_move(&mut gc, mv) {
                                cascadia_ai::nnue::extract_features_with_bag(&gc.boards[0], Some(&bag_info))
                            } else {
                                vec![]
                            }
                        }).collect();
                    }

                    let current_score = new_score;

                    let n_cands = candidates.len();
                    writeln!(out, "{{\"reward\":{},\"done\":{},\"n_candidates\":{},\"current_score\":{}}}", reward, done, n_cands, current_score).unwrap();
                    out.flush().unwrap();
                }
            }
            "quit" => break,
            _ => {
                writeln!(out, "{{\"error\":\"unknown command: {}\"}}", parts[0]).unwrap();
                out.flush().unwrap();
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Gym server mode
    if args.iter().any(|a| a == "--gym") {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        run_gym_server(weights_path);
        return;
    }

    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    let run_all = args.iter().any(|a| a == "--all");
    let run_train = args.iter().any(|a| a == "--train");
    let run_nnue_train = args.iter().any(|a| a == "--nnue-train");
    let run_cache_train = args.iter().any(|a| a == "--cache-train");
    let run_collect_mce = args.iter().any(|a| a == "--collect-mce");
    let run_collect_policy = args.iter().any(|a| a == "--collect-policy");
    let run_collect_mcts = args.iter().any(|a| a == "--collect-mcts");
    let run_collect_mce_policy = args.iter().any(|a| a == "--collect-mce-policy");
    let run_train_mce_policy = args.iter().any(|a| a == "--train-mce-policy");
    let run_export_pytorch = args.iter().any(|a| a == "--export-pytorch");
    let run_self_play = args.iter().any(|a| a == "--self-play");

    let run_mce_selfplay = args.iter().any(|a| a == "--mce-selfplay");
    let run_exact_selfplay = args.iter().any(|a| a == "--exact-selfplay");

    if run_exact_selfplay {
        // Play full games with exact expectimax, record value + policy data
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("exact_value_samples.bin");
        let policy_out = args.iter().position(|a| a == "--policy-out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("exact_policy_samples.bin");
        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xC0DE_C0DE + seed_offset)
        };

        println!("Exact expectimax self-play: {} games, weights={}", num_games, weights_path);
        println!("  Value samples → {}", out_path);
        println!("  Policy samples → {}", policy_out);
        let start = Instant::now();
        let mut all_value_samples: Vec<(Vec<u16>, f32)> = Vec::new();
        let mut all_policy_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
        let mut total_final_score = 0u64;

        for game_i in 0..num_games {
            let mut rng = StdRng::seed_from_u64(entropy_rng.gen());
            let cards = ScoringCards::all_a();
            let mut game = GameState::new(4, cards, &mut rng);

            let mut turn_value_records: Vec<(Vec<u16>, f32)> = Vec::new();
            let mut turn_policy_records: Vec<(Vec<(Vec<u16>, f32)>, f32)> = Vec::new();

            while !game.is_game_over() {
                if game.current_player != 0 {
                    let opp_mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                        .or_else(|| greedy_move(&game));
                    match opp_mv {
                        Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                        None => break,
                    }
                    continue;
                }

                let current = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0].clone(), &game.scoring_cards,
                ).total as f32;

                // Get all scored candidates with expectimax (for policy data)
                let results = cascadia_ai::mce::score_all_candidates_expectimax(&game, &net);
                if results.is_empty() { break; }

                // Policy data: all candidates with expectimax scores
                let policy_candidates: Vec<(Vec<u16>, f32)> = results.iter()
                    .map(|(_, features, score)| (features.clone(), *score as f32))
                    .collect();
                turn_policy_records.push((policy_candidates, current));

                // Play best move (highest expectimax score)
                let best_mv = results.iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .map(|(mv, _, _)| *mv)
                    .unwrap();
                if !execute_scored_move(&mut game, &best_mv) { break; }

                // Value data: afterstate of chosen move
                let bag_info = cascadia_ai::nnue::BagInfo::from_game(&game);
                let features = cascadia_ai::nnue::extract_features_with_bag(
                    &game.boards[0], Some(&bag_info));
                turn_value_records.push((features, current));
            }

            let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0], &game.scoring_cards,
            ).total as f32;
            total_final_score += final_score as u64;

            for (features, current) in turn_value_records {
                let delta = (final_score - current).max(0.0);
                all_value_samples.push((features, delta));
            }

            for (candidates, current) in turn_policy_records {
                all_policy_groups.push(cascadia_ai::nnue_train::PolicyGroup {
                    candidates,
                    value_target: (final_score - current).max(0.0),
                });
            }

            let avg_so_far = total_final_score as f64 / (game_i + 1) as f64;
            eprint!("\r  Game {}/{} — final={:.0}, avg={:.1}, v={}, p={}    ",
                    game_i + 1, num_games, final_score, avg_so_far,
                    all_value_samples.len(), all_policy_groups.len());
        }
        eprintln!();

        cascadia_ai::nnue_train::append_mce_samples(
            std::path::Path::new(out_path), &all_value_samples,
        ).expect("Failed to write value samples");

        cascadia_ai::nnue_train::save_policy_data(
            std::path::Path::new(policy_out), &all_policy_groups,
        ).expect("Failed to write policy samples");

        let elapsed = start.elapsed();
        let avg_score = total_final_score as f64 / num_games as f64;
        println!("Done in {:.1?}. {} games (avg {:.1})", elapsed, num_games, avg_score);
        println!("  Value: {} samples → {}", all_value_samples.len(), out_path);
        println!("  Policy: {} groups → {}", all_policy_groups.len(), policy_out);
        return;
    } else if run_mce_selfplay {
        // Play full games with MCE for player 0, record afterstate delta labels
        // (actual_final_score - current_score) for value network training
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_value_samples.bin");
        let rollouts: usize = args.iter().position(|a| a == "--rollouts")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(300);
        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xC0DE_C0DE + seed_offset)
        };

        let policy_out = args.iter().position(|a| a == "--policy-out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_selfplay_policy.bin");

        println!("MCE value self-play: {} games, rollouts={}, weights={}", num_games, rollouts, weights_path);
        println!("  Value samples → {}", out_path);
        println!("  Policy samples → {}", policy_out);
        let start = Instant::now();
        let mut all_value_samples: Vec<(Vec<u16>, f32)> = Vec::new();
        let mut all_policy_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
        let mut total_final_score = 0u64;

        for game_i in 0..num_games {
            let mut rng = StdRng::seed_from_u64(entropy_rng.gen());
            let cards = ScoringCards::all_a();
            let mut game = GameState::new(4, cards, &mut rng);
            let mut search_rng = StdRng::seed_from_u64(rng.gen());

            // Per-turn records: (chosen_afterstate_features, current_score)
            let mut turn_value_records: Vec<(Vec<u16>, f32)> = Vec::new();
            // Per-turn policy records: (all_candidate_features_and_scores)
            let mut turn_policy_records: Vec<(Vec<(Vec<u16>, f32)>, f32)> = Vec::new();

            while !game.is_game_over() {
                if game.current_player != 0 {
                    let opp_mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                        .or_else(|| greedy_move(&game));
                    match opp_mv {
                        Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                        None => break,
                    }
                    continue;
                }

                let current = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0].clone(), &game.scoring_cards,
                ).total as f32;

                // Get all scored candidates (for policy data) and play the best
                let results = cascadia_ai::mce::mce_candidates_with_features(
                    &game, &net, rollouts, &mut search_rng,
                );
                if results.is_empty() { break; }

                // Policy data: all candidates with MCE scores
                let policy_candidates: Vec<(Vec<u16>, f32)> = results.iter()
                    .map(|(_, features, score)| (features.clone(), *score))
                    .collect();
                turn_policy_records.push((policy_candidates, current));

                // Play best move (first in sorted results)
                let best_mv = results[0].0;
                if !execute_scored_move(&mut game, &best_mv) { break; }

                // Value data: afterstate of chosen move
                let bag_info = cascadia_ai::nnue::BagInfo::from_game(&game);
                let features = cascadia_ai::nnue::extract_features_with_bag(
                    &game.boards[0], Some(&bag_info));
                turn_value_records.push((features, current));
            }

            let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0], &game.scoring_cards,
            ).total as f32;
            total_final_score += final_score as u64;

            // Value samples: delta labels
            for (features, current) in turn_value_records {
                let delta = (final_score - current).max(0.0);
                all_value_samples.push((features, delta));
            }

            // Policy samples: grouped with value target
            for (candidates, current) in turn_policy_records {
                all_policy_groups.push(cascadia_ai::nnue_train::PolicyGroup {
                    candidates,
                    value_target: (final_score - current).max(0.0),
                });
            }

            let avg_so_far = total_final_score as f64 / (game_i + 1) as f64;
            eprint!("\r  Game {}/{} — final={:.0}, avg={:.1}, v_samples={}, p_groups={}    ",
                    game_i + 1, num_games, final_score, avg_so_far,
                    all_value_samples.len(), all_policy_groups.len());
        }
        eprintln!();

        // Write value samples (MCEP format)
        cascadia_ai::nnue_train::append_mce_samples(
            std::path::Path::new(out_path), &all_value_samples,
        ).expect("Failed to write value samples");

        // Write policy samples (MCP2 format)
        cascadia_ai::nnue_train::save_policy_data(
            std::path::Path::new(policy_out), &all_policy_groups,
        ).expect("Failed to write policy samples");

        let elapsed = start.elapsed();
        let avg_score = total_final_score as f64 / num_games as f64;
        println!("Done in {:.1?}. {} games (avg {:.1})", elapsed, num_games, avg_score);
        println!("  Value: {} samples → {}", all_value_samples.len(), out_path);
        println!("  Policy: {} groups → {}", all_policy_groups.len(), policy_out);
        return;
    } else if run_self_play {
        // Generate NNUE self-play games and write to MCEP format
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("self_play_samples.bin");
        let epsilon: f32 = args.iter().position(|a| a == "--epsilon")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.1);

        let top_pct: f32 = args.iter().position(|a| a == "--top-pct")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(100.0);

        let net = weights_path.and_then(|p| {
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(p)).ok()
        });
        let strategy = if net.is_some() { "NNUE" } else { "greedy" };
        let filter_str = if top_pct < 100.0 { format!(", top {}%", top_pct) } else { String::new() };
        println!("Generating {} self-play games ({}, epsilon={}{}, out={})",
            num_games, strategy, epsilon, filter_str, out_path);

        let start = Instant::now();
        let seed = rand::random::<u64>();

        let samples = if top_pct < 100.0 {
            // Generate per-game results, sort by score, keep top %
            let mut games = cascadia_ai::nnue_train::generate_games(
                num_games, seed, net.as_ref(), epsilon, 4,
            );
            games.sort_by(|a, b| b.final_score.cmp(&a.final_score));
            let keep = ((games.len() as f32 * top_pct / 100.0).ceil() as usize).max(1);
            let cutoff = games[keep - 1].final_score;
            println!("  Top {}%: keeping {} games (score >= {})", top_pct, keep, cutoff);
            let avg_score: f64 = games[..keep].iter().map(|g| g.final_score as f64).sum::<f64>() / keep as f64;
            println!("  Avg score of kept games: {:.1}", avg_score);
            games.truncate(keep);
            games.into_iter().flat_map(|g| g.samples).collect::<Vec<_>>()
        } else {
            cascadia_ai::nnue_train::generate_samples(
                num_games, seed, net.as_ref(), epsilon, 4,
            )
        };

        // Write as MCEP format
        let mcep_samples: Vec<(Vec<u16>, f32)> = samples.iter()
            .map(|s| (s.features.clone(), s.target))
            .collect();
        cascadia_ai::nnue_train::append_mce_samples(
            std::path::Path::new(out_path), &mcep_samples,
        ).expect("Failed to write samples");

        println!("Generated {} samples from {} games in {:.1?}",
            samples.len(), num_games, start.elapsed());
        return;
    } else if run_export_pytorch {
        // Load MCE samples, augment with rotations+translations, export as raw binary
        // for PyTorch training. Format: header (u32 num_samples, u32 num_features),
        // then for each sample: bit-packed features (ceil(num_features/8) bytes) + f32 target.
        let samples_path = args.iter().position(|a| a == "--samples")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_policy_samples.bin");
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("training_data.bin");

        println!("Exporting augmented training data for PyTorch...");
        let start = Instant::now();

        eprint!("  Loading samples...");
        let raw = cascadia_ai::nnue_train::load_mce_samples(
            std::path::Path::new(samples_path)).expect("Failed to load samples");
        eprintln!(" {} raw samples", raw.len());

        eprint!("  Augmenting...");
        // Use the same augmentation as Rust training
        let samples = cascadia_ai::nnue_train::augment_samples_pub(&raw);
        eprintln!(" {} augmented samples", samples.len());

        let num_features = cascadia_ai::nnue::NUM_FEATURES as u32;
        let packed_width = ((num_features + 7) / 8) as usize;

        eprint!("  Writing bit-packed to {}...", out_path);
        use std::io::Write;
        let mut file = std::fs::File::create(out_path).expect("Failed to create output");
        // Header
        file.write_all(&(samples.len() as u32).to_le_bytes()).unwrap();
        file.write_all(&num_features.to_le_bytes()).unwrap();
        // Samples: packed features + target
        let mut packed = vec![0u8; packed_width];
        for sample in &samples {
            packed.fill(0);
            for &fi in &sample.features {
                let fi = fi as usize;
                if fi < num_features as usize {
                    packed[fi >> 3] |= 1 << (fi & 7);
                }
            }
            file.write_all(&packed).unwrap();
            file.write_all(&sample.target.to_le_bytes()).unwrap();
        }
        eprintln!(" done");
        println!("Exported {} samples ({} features, {:.1} MB) in {:.1?}",
            samples.len(), num_features,
            (samples.len() * (packed_width + 4)) as f64 / 1e6,
            start.elapsed());
        return;
    } else if run_collect_mce {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let rollouts: usize = args.iter().position(|a| a == "--rollouts")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_policy_samples.bin");
        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        // Entropy source for random seeds
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xC0DE_C0DE)
        };

        println!("Collecting MCE samples: {} games, rollouts={}, weights={}, out={}, seed={}",
                 num_games, rollouts, weights_path, out_path,
                 if use_random_seed { "random" } else { "deterministic" });
        let start = Instant::now();
        let mut total_samples = 0usize;
        let mut total_final_score = 0u64;
        for game_i in 0..num_games {
            let mut rng = StdRng::seed_from_u64(entropy_rng.gen());
            let cards = ScoringCards::all_a();
            let mut game = GameState::new(4, cards, &mut rng);
            let mut search_rng = StdRng::seed_from_u64(rng.gen());
            let mut game_samples: Vec<(Vec<u16>, f32)> = Vec::new();

            while !game.is_game_over() {
                if game.current_player != 0 {
                    let opp_mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                        .or_else(|| greedy_move(&game));
                    match opp_mv {
                        Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                        None => break,
                    }
                    continue;
                }
                // AI turn: collect samples + play MCE move in one pass
                let tops = cascadia_ai::mce::top_moves_mce(&game, &net, rollouts, &mut search_rng, 15);
                for (mv, avg) in &tops {
                    let mut g = game.clone();
                    if cascadia_ai::search::execute_scored_move(&mut g, mv) {
                        let current = cascadia_core::scoring::ScoreBreakdown::compute(
                            &mut g.boards[game.current_player], &g.scoring_cards,
                        ).total as f32;
                        let target = (*avg as f32 - current).max(0.0);
                        let bag_info = cascadia_ai::nnue::BagInfo::from_game(&g);
                        let features = cascadia_ai::nnue::extract_features_with_bag(
                            &g.boards[game.current_player], Some(&bag_info));
                        game_samples.push((features, target));
                    }
                }
                let mv = tops.into_iter().next().map(|(mv, avg)| {
                    cascadia_ai::eval::ScoredMove { score: avg.round() as u16, ..mv }
                });
                match mv {
                    Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                    None => break,
                }
            }

            let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0], &game.scoring_cards,
            ).total;
            total_final_score += final_score as u64;
            total_samples += game_samples.len();
            cascadia_ai::nnue_train::append_mce_samples(
                std::path::Path::new(out_path), &game_samples,
            ).expect("Failed to append samples");

            let avg_so_far = total_final_score as f64 / (game_i + 1) as f64;
            eprint!("\r  Game {}/{} — final={}, avg={:.1}, samples={}    ",
                    game_i + 1, num_games, final_score, avg_so_far, total_samples);
        }
        eprintln!();
        println!("Done in {:.1?}. {} samples written to {}", start.elapsed(), total_samples, out_path);
        return;
    } else if run_collect_policy {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("policy_data.bin");
        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xC0DE_C0DE)
        };

        println!("Collecting policy data: {} games, weights={}, out={}", num_games, weights_path, out_path);
        let start = Instant::now();
        let mut total_groups = 0usize;
        let mut total_final_score = 0u64;

        for game_i in 0..num_games {
            let mut rng = StdRng::seed_from_u64(entropy_rng.gen());
            let cards = ScoringCards::all_a();
            let mut game = GameState::new(4, cards, &mut rng);
            let mut game_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
            let mut move_scores: Vec<(usize, f64)> = Vec::new(); // (group_idx, current_score)

            while !game.is_game_over() {
                if game.current_player != 0 {
                    let opp_mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                        .or_else(|| greedy_move(&game));
                    match opp_mv {
                        Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                        None => break,
                    }
                    continue;
                }
                // AI turn: score all candidates with expectimax, record for policy training
                let scored = cascadia_ai::mce::score_all_candidates_expectimax(&game, &net);
                if scored.is_empty() { break; }

                let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0].clone(), &game.scoring_cards,
                ).total as f64;

                let group_idx = game_groups.len();
                let candidates: Vec<(Vec<u16>, f32)> = scored.iter()
                    .map(|(_, features, score)| (features.clone(), *score as f32))
                    .collect();
                game_groups.push(cascadia_ai::nnue_train::PolicyGroup {
                    candidates,
                    value_target: 0.0, // filled in after game ends
                });
                move_scores.push((group_idx, current_score));

                // Play the best move
                let best_mv = scored.iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .map(|(mv, _, _)| *mv)
                    .unwrap();
                if !execute_scored_move(&mut game, &best_mv) { break; }
            }

            let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0], &game.scoring_cards,
            ).total as f64;
            total_final_score += final_score as u64;

            // Fill in value targets: final_score - current_score
            for (group_idx, current_score) in &move_scores {
                game_groups[*group_idx].value_target = (final_score - current_score) as f32;
            }

            total_groups += game_groups.len();
            cascadia_ai::nnue_train::save_policy_data(
                std::path::Path::new(&format!("{}.{}", out_path, game_i)),
                &game_groups,
            ).expect("Failed to save policy data");

            // Append to main file
            if game_i == 0 {
                cascadia_ai::nnue_train::save_policy_data(
                    std::path::Path::new(out_path),
                    &game_groups,
                ).expect("Failed to save policy data");
            } else {
                // Append without re-writing header
                use std::io::Write;
                let mut buf: Vec<u8> = Vec::new();
                for group in &game_groups {
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
                let mut file = std::fs::OpenOptions::new()
                    .append(true).open(out_path)
                    .expect("Failed to open policy data for append");
                file.write_all(&buf).expect("Failed to append policy data");
            }

            // Clean up per-game temp file
            let _ = std::fs::remove_file(format!("{}.{}", out_path, game_i));

            let avg_so_far = total_final_score as f64 / (game_i + 1) as f64;
            eprint!("\r  Game {}/{} — final={:.0}, avg={:.1}, groups={}    ",
                    game_i + 1, num_games, final_score, avg_so_far, total_groups);
        }
        eprintln!();
        println!("Done in {:.1?}. {} position groups written to {}", start.elapsed(), total_groups, out_path);
        return;
    } else if run_collect_mcts {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mcts_selfplay.bin");
        let simulations: usize = args.iter().position(|a| a == "--simulations")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(100);
        let temperature: f32 = args.iter().position(|a| a == "--temperature")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xC0DE_C0DE)
        };

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        println!("MCTS self-play: {} games, sims={}, temp={}, weights={}, out={}, threads={}",
                 num_games, simulations, temperature, weights_path, out_path, num_threads);
        let start = Instant::now();

        // Pre-generate seeds for all games
        let seeds: Vec<u64> = (0..num_games).map(|_| entropy_rng.gen()).collect();

        // Parallel game execution
        let games_done = std::sync::atomic::AtomicUsize::new(0);
        let score_sum = std::sync::atomic::AtomicU64::new(0);
        let games_done_ref = &games_done;
        let score_sum_ref = &score_sum;

        let chunk_size = (num_games + num_threads - 1) / num_threads;
        let handles: Vec<_> = seeds.chunks(chunk_size).map(|chunk| {
            let chunk_seeds = chunk.to_vec();
            let net = Arc::clone(&net);
            std::thread::spawn(move || {
                let mut thread_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
                let mut thread_scores: Vec<u64> = Vec::new();

                for &seed in &chunk_seeds {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let cards = ScoringCards::all_a();
                    let mut game = GameState::new(4, cards, &mut rng);
                    let mut game_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
                    let mut move_scores: Vec<(usize, f64)> = Vec::new();
                    let mut turn_count = 0usize;

                    while !game.is_game_over() {
                        if game.current_player != 0 {
                            let opp_mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                                .or_else(|| greedy_move(&game));
                            match opp_mv {
                                Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                                None => break,
                            }
                            continue;
                        }

                        let temp = if turn_count < 8 { temperature } else { temperature * 0.1 };
                        let result = cascadia_ai::mcts::mcts_search_with_features(
                            &game, &net, simulations, temp,
                        );

                        match result {
                            Some((best_mv, candidates)) => {
                                let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                                    &mut game.boards[0].clone(), &game.scoring_cards,
                                ).total as f64;

                                let group_idx = game_groups.len();
                                game_groups.push(cascadia_ai::nnue_train::PolicyGroup {
                                    candidates,
                                    value_target: 0.0,
                                });
                                move_scores.push((group_idx, current_score));

                                if !execute_scored_move(&mut game, &best_mv) { break; }
                            }
                            None => break,
                        }
                        turn_count += 1;
                    }

                    let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut game.boards[0], &game.scoring_cards,
                    ).total as f64;

                    for (group_idx, current_score) in &move_scores {
                        game_groups[*group_idx].value_target = (final_score - current_score) as f32;
                    }

                    thread_groups.extend(game_groups);
                    thread_scores.push(final_score as u64);
                }

                (thread_groups, thread_scores)
            })
        }).collect();

        // Collect results from all threads
        let mut all_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
        let mut total_final_score = 0u64;
        let mut total_games = 0usize;

        for handle in handles {
            let (groups, scores) = handle.join().unwrap();
            all_groups.extend(groups);
            for &s in &scores {
                total_final_score += s;
                total_games += 1;
                let avg = total_final_score as f64 / total_games as f64;
                eprint!("\r  {}/{} games done, avg={:.1}    ", total_games, num_games, avg);
            }
        }
        eprintln!();

        let total_groups = all_groups.len();

        // Write all groups
        cascadia_ai::nnue_train::save_policy_data(
            std::path::Path::new(out_path), &all_groups,
        ).expect("Failed to save self-play data");

        let elapsed = start.elapsed();
        let avg_score = total_final_score as f64 / num_games as f64;
        println!("Done in {:.1?}. {} groups from {} games (avg {:.1}), written to {}",
                 elapsed, total_groups, num_games, avg_score, out_path);
        println!("  {:.1}s/game wall, {:.0} groups/game, {} threads",
                 elapsed.as_secs_f64() / num_games as f64,
                 total_groups as f64 / num_games as f64, num_threads);
        return;
    } else if run_collect_mce_policy {
        // Collect MCE-scored candidates in MCP2 format for policy training.
        // Each position: all candidates with MCE scores + value target.
        // Parallelized across games.
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_policy_grouped.bin");
        let rollouts: usize = args.iter().position(|a| a == "--rollouts")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(300);
        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xC0DE_C0DE)
        };

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        println!("MCE policy collection: {} games, rollouts={}, weights={}, out={}, threads={}",
                 num_games, rollouts, weights_path, out_path, num_threads);
        let start = Instant::now();

        let seeds: Vec<u64> = (0..num_games).map(|_| entropy_rng.gen()).collect();
        let chunk_size = (num_games + num_threads - 1) / num_threads;

        let handles: Vec<_> = seeds.chunks(chunk_size).map(|chunk| {
            let chunk_seeds = chunk.to_vec();
            let net = Arc::clone(&net);
            std::thread::spawn(move || {
                let mut thread_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
                let mut thread_scores: Vec<u64> = Vec::new();

                for &seed in &chunk_seeds {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let cards = ScoringCards::all_a();
                    let mut game = GameState::new(4, cards, &mut rng);
                    let mut search_rng = StdRng::seed_from_u64(rng.gen());
                    let mut game_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
                    let mut move_scores: Vec<(usize, f64)> = Vec::new();

                    while !game.is_game_over() {
                        if game.current_player != 0 {
                            let opp_mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                                .or_else(|| greedy_move(&game));
                            match opp_mv {
                                Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                                None => break,
                            }
                            continue;
                        }

                        // Score all candidates with MCE
                        let results = cascadia_ai::mce::mce_candidates_with_features(
                            &game, &net, rollouts, &mut search_rng,
                        );
                        if results.is_empty() { break; }

                        let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                            &mut game.boards[0].clone(), &game.scoring_cards,
                        ).total as f64;

                        // Best move is first (results sorted by MCE score)
                        let best_mv = results[0].0;

                        let candidates: Vec<(Vec<u16>, f32)> = results.iter()
                            .map(|(_, features, score)| (features.clone(), *score))
                            .collect();

                        let group_idx = game_groups.len();
                        game_groups.push(cascadia_ai::nnue_train::PolicyGroup {
                            candidates,
                            value_target: 0.0,
                        });
                        move_scores.push((group_idx, current_score));

                        if !execute_scored_move(&mut game, &best_mv) { break; }
                    }

                    let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut game.boards[0], &game.scoring_cards,
                    ).total as f64;

                    for (group_idx, current_score) in &move_scores {
                        game_groups[*group_idx].value_target = (final_score - current_score) as f32;
                    }

                    thread_groups.extend(game_groups);
                    thread_scores.push(final_score as u64);
                }

                (thread_groups, thread_scores)
            })
        }).collect();

        let mut all_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
        let mut total_final_score = 0u64;
        let mut total_games = 0usize;

        for handle in handles {
            let (groups, scores) = handle.join().unwrap();
            all_groups.extend(groups);
            for &s in &scores {
                total_final_score += s;
                total_games += 1;
                eprint!("\r  {}/{} games done, avg={:.1}    ",
                        total_games, num_games, total_final_score as f64 / total_games as f64);
            }
        }
        eprintln!();

        let total_groups = all_groups.len();
        cascadia_ai::nnue_train::save_policy_data(
            std::path::Path::new(out_path), &all_groups,
        ).expect("Failed to save policy data");

        let elapsed = start.elapsed();
        let avg_score = total_final_score as f64 / num_games as f64;
        println!("Done in {:.1?}. {} groups from {} games (avg {:.1}), written to {}",
                 elapsed, total_groups, num_games, avg_score, out_path);
        println!("  {:.1}s/game wall, {} threads",
                 elapsed.as_secs_f64() / num_games as f64, num_threads);
        return;
    } else if run_train_mce_policy {
        let epochs: usize = args.iter().position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(15);
        let lr: f32 = args.iter().position(|a| a == "--lr")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.0001);
        let samples_path = args.iter().position(|a| a == "--samples")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_policy_samples.bin");
        let weights_in = args.iter().position(|a| a == "--init-weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let weights_out = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights_policy.bin");

        let mut net = if let Some(path) = weights_in {
            println!("Loading initial weights from {}...", path);
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path))
                .expect("Failed to load init weights")
        } else {
            println!("Starting with fresh weights");
            cascadia_ai::nnue::NNUENetwork::new()
        };

        let freeze_legacy = args.iter().any(|a| a == "--freeze-legacy");
        let freeze_below = if freeze_legacy { cascadia_ai::nnue::NUM_FEATURES_LEGACY } else { 0 };

        println!("Training from MCE samples: samples={}, epochs={}, lr={}", samples_path, epochs, lr);
        if freeze_legacy {
            println!("  FROZEN: only training features >= {} (new features only)", freeze_below);
        }
        println!("  Checkpoint: saving after every epoch to {}", weights_out);
        let start = Instant::now();
        let stats = cascadia_ai::nnue_train::train_from_mce_samples_with_checkpoint(
            &mut net, std::path::Path::new(samples_path), epochs, lr,
            Some(std::path::Path::new(weights_out)),
            freeze_below,
        ).expect("Training failed");
        println!("Training complete in {:.1?}", start.elapsed());
        println!("  Samples:    {}", stats.num_samples);
        println!("  Final RMSE: {:.2}", stats.final_rmse);
        net.save(std::path::Path::new(weights_out)).expect("Failed to save weights");
        println!("  Weights saved to {}", weights_out);
        return;
    }

    if run_cache_train {
        let epochs: usize = args.iter().position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(30);
        let lr: f32 = args.iter().position(|a| a == "--lr")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.0001);
        let cache_path = args.iter().position(|a| a == "--cache")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("training_cache_90plus.bin");
        let weights_in = args.iter().position(|a| a == "--init-weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let weights_out = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");

        let mut net = if let Some(path) = weights_in {
            println!("Loading initial weights from {}...", path);
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path))
                .expect("Failed to load init weights")
        } else {
            println!("Starting with fresh weights");
            cascadia_ai::nnue::NNUENetwork::new()
        };

        println!("Cache training: cache={}, epochs={}, lr={}", cache_path, epochs, lr);
        let start = Instant::now();
        let stats = cascadia_ai::nnue_train::train_from_cache(
            &mut net, std::path::Path::new(cache_path), epochs, lr,
        ).expect("Cache training failed");
        let elapsed = start.elapsed();

        println!("Cache training complete in {:.1?}", elapsed);
        println!("  Samples:    {}", stats.num_samples);
        println!("  Final RMSE: {:.2}", stats.final_rmse);

        net.save(std::path::Path::new(weights_out)).expect("Failed to save");
        println!("  Weights saved to {}", weights_out);
        return;
    } else if run_nnue_train {
        let train_games = num_games;
        let lr: f32 = args.iter().position(|a| a == "--lr")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.0001);
        let epochs: usize = args.iter().position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(10);
        let weights_path = std::path::PathBuf::from(
            args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin")
        );

        let mut net = if weights_path.exists() {
            println!("Loading NNUE weights from {:?}...", weights_path);
            cascadia_ai::nnue::NNUENetwork::load(&weights_path).unwrap_or_else(|e| {
                eprintln!("Failed to load: {}, starting fresh", e);
                cascadia_ai::nnue::NNUENetwork::new()
            })
        } else {
            println!("Starting with fresh NNUE weights");
            cascadia_ai::nnue::NNUENetwork::new()
        };

        println!("Training NNUE: {} games, {} epochs, lr={}, weights={:?}",
            train_games, epochs, lr, weights_path);
        let start = Instant::now();
        let stats = cascadia_ai::nnue_train::train_nnue(&mut net, train_games, epochs, lr, 42);
        let elapsed = start.elapsed();

        println!("Training complete in {:.1?}", elapsed);
        println!("  Samples:    {}", stats.num_samples);
        println!("  Final RMSE: {:.2}", stats.final_rmse);

        net.save(&weights_path).expect("Failed to save NNUE weights");
        println!("  Weights saved to {:?}", weights_path);
        return;
    } else if run_train {
        let train_games = num_games;
        let alpha: f32 = args.iter().position(|a| a == "--alpha")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.001);
        let weights_path = std::path::PathBuf::from(
            args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("ntuple_weights.bin")
        );

        // Load existing weights or start fresh
        let mut net = if weights_path.exists() {
            println!("Loading weights from {:?}...", weights_path);
            cascadia_ai::ntuple::NTupleNetwork::load(&weights_path).unwrap_or_else(|e| {
                eprintln!("Failed to load weights: {}, starting fresh", e);
                cascadia_ai::ntuple::NTupleNetwork::new()
            })
        } else {
            println!("Starting with fresh weights");
            cascadia_ai::ntuple::NTupleNetwork::new()
        };

        println!("Training N-tuple network: {} games, alpha={}, weights={:?}", train_games, alpha, weights_path);
        let start = Instant::now();
        let stats = cascadia_ai::train::train(&mut net, train_games, alpha, 42);
        let elapsed = start.elapsed();

        let avg = stats.total_score as f64 / stats.games as f64;
        println!("\nTraining complete in {:.1?}", elapsed);
        println!("  Games:     {}", stats.games);
        println!("  Avg score: {:.1}", avg);

        // Save weights
        net.save(&weights_path).expect("Failed to save weights");
        println!("  Weights saved to {:?}", weights_path);
        return;
    } else if run_all {
        let strategies = vec![
            Strategy::Greedy,
            Strategy::Lookahead1,
            Strategy::Beam { width: 5, depth: 3 },
            Strategy::MonteCarlo { rollouts: 20 },
        ];

        println!("Benchmarking {} strategies with {} games each (Card A)...\n", strategies.len(), num_games);

        let handles: Vec<_> = strategies
            .into_iter()
            .map(|s| {
                thread::spawn(move || run_benchmark(&s, num_games))
            })
            .collect();

        let results: Vec<BenchResult> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        for r in &results {
            print_result(r);
        }

        println!();
        print_comparison(&results);
    } else {
        let strategy = if args.iter().any(|a| a == "--beam") {
            let width = args.iter().position(|a| a == "--width")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(5);
            let depth = args.iter().position(|a| a == "--depth")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(3);
            Strategy::Beam { width, depth }
        } else if args.iter().any(|a| a == "--mcts" || a == "--monte-carlo") {
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(20);
            Strategy::MonteCarlo { rollouts }
        } else if args.iter().any(|a| a == "--lookahead" || a == "-l") {
            Strategy::Lookahead1
        } else if args.iter().any(|a| a == "--expectimax") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let samples = args.iter().position(|a| a == "--samples")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(20);
            let depth = args.iter().position(|a| a == "--depth")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);
            let branching = args.iter().position(|a| a == "--branching")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(5);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights");
            Strategy::Expectimax { net: Arc::new(net), samples, depth, branching }
        } else if args.iter().any(|a| a == "--hybrid") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let top_k: usize = std::env::var("HYBRID_TOP_K")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights");
            Strategy::Hybrid { net: Arc::new(net), rollouts, top_k }
        } else if args.iter().any(|a| a == "--mcts-search") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let simulations: usize = args.iter().position(|a| a == "--simulations")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(200);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for MCTS");
            Strategy::MCTS { net: Arc::new(net), simulations }
        } else if args.iter().any(|a| a == "--exact") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for exact expectimax");
            Strategy::ExactExpectimax { net: Arc::new(net) }
        } else if args.iter().any(|a| a == "--policy-mce") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let policy_path = args.iter().position(|a| a == "--policy-weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("policy_net_v1.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let top_k: usize = args.iter().position(|a| a == "--top-k")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(5);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights");
            let policy = cascadia_ai::nnue::PolicyNetwork::load(std::path::Path::new(policy_path))
                .expect("Failed to load policy weights");
            Strategy::PolicyMCE { net: Arc::new(net), policy: Arc::new(policy), rollouts, top_k }
        } else if args.iter().any(|a| a == "--mce") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for MCE");
            Strategy::MCE { net: Arc::new(net), rollouts }
        } else if args.iter().any(|a| a == "--nnue") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights. Train first with --nnue-train");
            Strategy::NNUE { net: Arc::new(net) }
        } else if args.iter().any(|a| a == "--ntuple") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("ntuple_weights.bin");
            let net = NTupleNetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load N-tuple weights. Train first with --train");
            Strategy::NTuple { net: Arc::new(net) }
        } else {
            Strategy::Greedy
        };

        println!("Simulating {} games with {} AI (Card A)...", num_games, strategy);
        let result = run_benchmark(&strategy, num_games);
        println!();
        print_result(&result);

        // Print MCE diagnostics if any were collected
        let diags = cascadia_ai::mce::take_diagnostics();
        if !diags.is_empty() {
            println!("  MCE Candidate Diagnostics ({} decisions):", diags.len());
            let mut wins_candidate = 0usize;
            let mut wins_greedy = 0usize;
            let mut wins_strategic = 0usize;
            let mut rank_sum = 0usize;
            let mut corr_sum = 0.0f64;
            let mut rank_hist = [0usize; 15];

            for d in &diags {
                match d.winner_source {
                    Some(cascadia_ai::mce::CandidateSource::CandidateMoves) => wins_candidate += 1,
                    Some(cascadia_ai::mce::CandidateSource::Greedy) => wins_greedy += 1,
                    Some(cascadia_ai::mce::CandidateSource::Strategic) => wins_strategic += 1,
                    None => {}
                }
                rank_sum += d.winner_pre_rank;
                corr_sum += d.rank_correlation;
                if d.winner_pre_rank < 15 {
                    rank_hist[d.winner_pre_rank] += 1;
                }
            }
            let n = diags.len() as f64;
            println!("    Winner source:");
            println!("      candidate_moves: {:>5} ({:.1}%)", wins_candidate, wins_candidate as f64 / n * 100.0);
            println!("      greedy:          {:>5} ({:.1}%)", wins_greedy, wins_greedy as f64 / n * 100.0);
            println!("      strategic:       {:>5} ({:.1}%)", wins_strategic, wins_strategic as f64 / n * 100.0);
            println!("    Avg pre-MCE rank of winner: {:.2} (0=eval agreed, higher=MCE reranked)", rank_sum as f64 / n);
            println!("    Avg Spearman correlation (eval vs MCE rank): {:.3}", corr_sum / n);
            println!("    Winner was eval-rank #N:");
            for i in 0..15 {
                if rank_hist[i] > 0 {
                    let bar_len = (rank_hist[i] * 40) / diags.len().max(1);
                    let bar: String = "█".repeat(bar_len);
                    println!("      #{:<2}: {:>5} ({:>5.1}%) {}", i, rank_hist[i], rank_hist[i] as f64 / n * 100.0, bar);
                }
            }
            println!();
        }
    }
}
