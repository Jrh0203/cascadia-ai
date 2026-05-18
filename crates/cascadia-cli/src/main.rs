use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use cascadia_core::game::GameState;
use cascadia_core::types::ScoringCards;
use cascadia_ai::eval::{best_move_with_potential, best_move_lookahead};
use cascadia_ai::ntuple::NTupleNetwork;
use cascadia_ai::search::{best_move_beam, best_move_mcts, execute_scored_move, greedy_move};

/// Optional separate NNUE for opponents (players 1..N).
/// If set via --opp-weights, opponents use this network instead of player 0's net.
/// Used for "AI vs other AI" experiments (e.g., new model vs previous model).
static OPPONENT_NET: OnceLock<Arc<cascadia_ai::nnue::NNUENetwork>> = OnceLock::new();

fn opponent_net() -> Option<&'static Arc<cascadia_ai::nnue::NNUENetwork>> {
    OPPONENT_NET.get()
}

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
    NRPA { net: Arc<cascadia_ai::nnue::NNUENetwork>, level: usize, n: usize },
    OpenLoopMCTS { net: Arc<cascadia_ai::nnue::NNUENetwork>, rollouts: usize },
    GumbelMCTS { net: Arc<cascadia_ai::nnue::NNUENetwork>, rollouts: usize, m: usize },
    GreedyMCE { rollouts: usize, alloc: cascadia_ai::mce::GreedyMceAlloc, expanded: bool },
    NnueRolloutMCE { net: Arc<cascadia_ai::nnue::NNUENetwork>, rollouts: usize, alloc: cascadia_ai::mce::GreedyMceAlloc, expanded: bool, prefilter_k: usize, exact_endgame: usize },
    UctMcts { simulations: usize, parallel: bool },
    /// Cross-turn persistent open-loop UCT MCTS (Exp #6). Tree state persists
    /// across an AI's turns within a single game; the chosen edge's child
    /// subtree is promoted to root after each move so search budget compounds.
    /// `parallel` enables root-parallelism (independent trees across cores).
    /// `net` enables NNUE-guided rollouts (AI uses NNUE-greedy, opponents use
    /// plain greedy). When `net` is None the rollout policy is pure greedy
    /// (matching `uct_mcts.rs`).
    MctsTree {
        simulations: usize,
        parallel: bool,
        net: Option<Arc<cascadia_ai::nnue::NNUENetwork>>,
    },
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
            Strategy::NRPA { level, n, .. } => write!(f, "nrpa(L={},N={})", level, n),
            Strategy::OpenLoopMCTS { rollouts, .. } => write!(f, "ol-mcts(n={})", rollouts),
            Strategy::GumbelMCTS { rollouts, m, .. } => write!(f, "gumbel-mcts(n={},m={})", rollouts, m),
            Strategy::GreedyMCE { rollouts, alloc, expanded } => {
                let a = match alloc {
                    cascadia_ai::mce::GreedyMceAlloc::Uniform => "uniform",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalving => "halving",
                    cascadia_ai::mce::GreedyMceAlloc::Ucb => "ucb",
                    cascadia_ai::mce::GreedyMceAlloc::UniformCRN => "crn",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCRN => "halving-crn",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingEarlyTerm => "halving-et",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCI => "halving-ci",
                    cascadia_ai::mce::GreedyMceAlloc::SuccessiveRejects => "sr",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingPW => "halving-pw",
                    cascadia_ai::mce::GreedyMceAlloc::ThompsonSampling => "thompson",
                    cascadia_ai::mce::GreedyMceAlloc::MctsPW => "mcts-pw",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingHetero => "halving-hetero",
                    cascadia_ai::mce::GreedyMceAlloc::Puct => "puct",
                };
                let e = if *expanded { ",expanded" } else { "" };
                write!(f, "greedy-mce(n={},{}{})", rollouts, a, e)
            }
            Strategy::NnueRolloutMCE { rollouts, alloc, expanded, prefilter_k, exact_endgame, .. } => {
                let a = match alloc {
                    cascadia_ai::mce::GreedyMceAlloc::Uniform => "uniform",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalving => "halving",
                    cascadia_ai::mce::GreedyMceAlloc::Ucb => "ucb",
                    cascadia_ai::mce::GreedyMceAlloc::UniformCRN => "crn",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCRN => "halving-crn",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingEarlyTerm => "halving-et",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCI => "halving-ci",
                    cascadia_ai::mce::GreedyMceAlloc::SuccessiveRejects => "sr",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingPW => "halving-pw",
                    cascadia_ai::mce::GreedyMceAlloc::ThompsonSampling => "thompson",
                    cascadia_ai::mce::GreedyMceAlloc::MctsPW => "mcts-pw",
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalvingHetero => "halving-hetero",
                    cascadia_ai::mce::GreedyMceAlloc::Puct => "puct",
                };
                let e = if *expanded { ",expanded" } else { "" };
                let p = if *prefilter_k > 0 { format!(",pfk={}", prefilter_k) } else { String::new() };
                let g = if *exact_endgame > 0 { format!(",eg={}", exact_endgame) } else { String::new() };
                write!(f, "nnue-rollout-mce(n={},{}{}{}{})", rollouts, a, e, p, g)
            }
            Strategy::UctMcts { simulations, parallel } => {
                write!(f, "uct-mcts(n={}{})", simulations, if *parallel { ",parallel" } else { "" })
            }
            Strategy::MctsTree { simulations, parallel, net } => {
                let n = if net.is_some() { ",nnue" } else { "" };
                write!(f, "mcts-tree(n={}{}{})", simulations,
                    if *parallel { ",parallel" } else { "" }, n)
            }
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
        Strategy::NRPA { net, .. } => {
            cascadia_ai::nrpa::best_move_nrpa(game, net, search_rng)
        }
        Strategy::OpenLoopMCTS { net, rollouts } => {
            cascadia_ai::ol_mcts::best_move_ol_mcts(game, net, *rollouts, search_rng)
        }
        Strategy::GumbelMCTS { net, rollouts, m } => {
            cascadia_ai::gumbel_mcts::best_move_gumbel_mcts(game, net, *rollouts, *m, search_rng)
        }
        Strategy::GreedyMCE { rollouts, alloc, expanded } => {
            let candidates = if *expanded {
                cascadia_ai::mce::expanded_candidates(game)
            } else {
                cascadia_ai::mce::default_greedy_mce_candidates(game)
            };
            cascadia_ai::mce::best_move_greedy_mce_v2(game, *rollouts, *alloc, candidates, search_rng)
        }
        Strategy::NnueRolloutMCE { net, rollouts, alloc, expanded, prefilter_k, exact_endgame } => {
            // Exact-endgame switch: when turns_remaining is small, use exact expectimax
            // (terminal scoring is exact when game ends after the move).
            if *exact_endgame > 0 && game.turns_remaining as usize <= *exact_endgame {
                let depth = (*exact_endgame).saturating_sub(1).max(1);  // depth=remaining-1
                return cascadia_ai::mce::best_move_expectimax_nply(game, net, depth);
            }
            let mut candidates = if *expanded {
                cascadia_ai::mce::expanded_candidates(game)
            } else {
                cascadia_ai::mce::default_greedy_mce_candidates(game)
            };
            if *prefilter_k > 0 && candidates.len() > *prefilter_k {
                // MCE_MUTATE_EXPAND=N: keep N extra near-miss candidates beyond
                // prefilter-K. These are the next-best that prefilter would drop.
                // SHA handles the larger pool automatically. Tests whether the
                // prefilter's NNUE ranking missed a candidate that MCE rollouts
                // would have found superior.
                let expand: usize = std::env::var("MCE_MUTATE_EXPAND").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
                let k = *prefilter_k + expand;
                candidates = cascadia_ai::mce::nnue_prefilter_candidates(game, net, candidates, k);
            }
            cascadia_ai::mce::best_move_nnue_rollout_mce(game, net, *rollouts, *alloc, candidates, search_rng)
        }
        Strategy::UctMcts { simulations, parallel } => {
            if *parallel {
                cascadia_ai::uct_mcts::best_move_uct_mcts_parallel(game, *simulations, search_rng)
            } else {
                cascadia_ai::uct_mcts::best_move_uct_mcts(game, *simulations, search_rng)
            }
        }
        Strategy::MctsTree { simulations, parallel, net } => {
            // One-shot fallback when no persistent forest is threaded through
            // (e.g., opponent-side dispatch under CASCADIA_OPPONENTS_SAME).
            // Player 0's bench loop installs a persistent forest and bypasses
            // pick_move for MctsTree decisions.
            let n = if *parallel {
                thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
            } else { 1 };
            let seed = search_rng.gen::<u64>();
            let mut forest = cascadia_ai::mcts_tree::PersistentUctForest::new_with_net(
                game.current_player, n, seed, net.as_ref().map(Arc::clone),
            );
            forest.pick_move(game, *simulations)
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
            | Strategy::PolicyMCE { ref net, .. }
            | Strategy::NRPA { ref net, .. }
            | Strategy::OpenLoopMCTS { ref net, .. }
            | Strategy::GumbelMCTS { ref net, .. } => Some(net.clone()),
        Strategy::MctsTree { net: Some(n), .. } => Some(n.clone()),
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

/// Temporarily set env vars, run f(), then restore. Safe for sequential callers
/// because env var reads inside MCE threads all happen between the set/restore.
fn pick_with_env<F, T>(vars: &[(&str, &str)], f: F) -> T
where F: FnOnce() -> T,
{
    let prev: Vec<(String, Option<String>)> = vars.iter()
        .map(|(k, _)| (k.to_string(), std::env::var(k).ok()))
        .collect();
    for (k, v) in vars {
        std::env::set_var(k, v);
    }
    let result = f();
    for (k, prev_v) in &prev {
        match prev_v {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
    result
}

/// Dispatch a move by strategy tag. Used for per-seat strategies in head-to-head mode.
fn pick_move_by_tag(
    game: &GameState,
    tag: &str,
    net: Option<&Arc<cascadia_ai::nnue::NNUENetwork>>,
    search_rng: &mut StdRng,
) -> Option<cascadia_ai::eval::ScoredMove> {
    match tag {
        "greedy" => cascadia_ai::search::greedy_move(game),
        "nnue" => match net {
            Some(n) => cascadia_ai::nnue_train::pick_best_move_nnue(game, n)
                .or_else(|| cascadia_ai::search::greedy_move(game)),
            None => cascadia_ai::search::greedy_move(game),
        }
        "mce" | "mce_base" | "mce_200r" => match net {
            Some(n) => {
                let mut cands = cascadia_ai::mce::expanded_candidates(game);
                if cands.len() > 8 {
                    cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 8);
                }
                cascadia_ai::mce::best_move_nnue_rollout_mce(
                    game, n, 200,
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                    cands, search_rng,
                )
            }
            None => cascadia_ai::search::greedy_move(game),
        }
        "mce_100r" => match net {
            Some(n) => {
                let mut cands = cascadia_ai::mce::expanded_candidates(game);
                if cands.len() > 8 {
                    cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 8);
                }
                cascadia_ai::mce::best_move_nnue_rollout_mce(
                    game, n, 100,
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                    cands, search_rng,
                )
            }
            None => cascadia_ai::search::greedy_move(game),
        }
        "mce_sr" => match net {
            Some(n) => {
                let mut cands = cascadia_ai::mce::expanded_candidates(game);
                if cands.len() > 8 {
                    cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 8);
                }
                cascadia_ai::mce::best_move_nnue_rollout_mce(
                    game, n, 200,
                    cascadia_ai::mce::GreedyMceAlloc::SuccessiveRejects,
                    cands, search_rng,
                )
            }
            None => cascadia_ai::search::greedy_move(game),
        }
        "mce_default" => match net {
            Some(n) => {
                let cands = cascadia_ai::mce::default_greedy_mce_candidates(game);
                cascadia_ai::mce::best_move_nnue_rollout_mce(
                    game, n, 200,
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                    cands, search_rng,
                )
            }
            None => cascadia_ai::search::greedy_move(game),
        }
        // Feature-specific variants: set env vars for duration of MCE call.
        // Safe because pick_move_by_tag is called sequentially and MCE's
        // threads all join before the call returns.
        "mce_strategy" | "mce_strat" => pick_with_env(
            &[("MCE_STRATEGY_BIAS", "1")],
            || pick_move_by_tag(game, "mce", net, search_rng),
        ),
        "mce_cv" | "mce_cv_0.85" => pick_with_env(
            &[("MCE_CV_ALPHA", "0.85")],
            || pick_move_by_tag(game, "mce", net, search_rng),
        ),
        "mce_lmr" => pick_with_env(
            &[("MCE_LMR", "1")],
            || pick_move_by_tag(game, "mce", net, search_rng),
        ),
        "mce_full" => pick_with_env(
            &[("MCE_CV_ALPHA", "0.85"), ("MCE_LMR", "1"), ("MCE_STRATEGY_BIAS", "1")],
            || pick_move_by_tag(game, "mce", net, search_rng),
        ),
        // mce_wide_v1: wider prefilter (K=32) + larger halving budget (R=600).
        // Empirically beats champion by +2.0 pts mean (20g local, seed=42).
        // K=8 prefilter coverage: 58.9% → 84.3% (simulation, matches Rust halving budget).
        "mce_wide_v1" => pick_with_env(
            &[("MCE_LMR", "1"), ("MCE_DIVERSE_PREFILTER", "1")],
            || match net {
                Some(n) => {
                    let mut cands = cascadia_ai::mce::expanded_candidates(game);
                    if cands.len() > 32 {
                        cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 32);
                    }
                    cascadia_ai::mce::best_move_nnue_rollout_mce(
                        game, n, 600,
                        cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                        cands, search_rng,
                    )
                }
                None => cascadia_ai::search::greedy_move(game),
            },
        ),
        // mce_wide_v1_b/c/d: identical to mce_wide_v1. Distinct tags exist so a
        // 4-player head-to-head tournament can put 4 different weight files in 4
        // different seats while running the same algorithm — fair multi-model
        // round-robin without confounding by R or allocator differences.
        "mce_wide_v1_b" | "mce_wide_v1_c" | "mce_wide_v1_d" => pick_with_env(
            &[("MCE_LMR", "1"), ("MCE_DIVERSE_PREFILTER", "1")],
            || match net {
                Some(n) => {
                    let mut cands = cascadia_ai::mce::expanded_candidates(game);
                    if cands.len() > 32 {
                        cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 32);
                    }
                    cascadia_ai::mce::best_move_nnue_rollout_mce(
                        game, n, 600,
                        cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                        cands, search_rng,
                    )
                }
                None => cascadia_ai::search::greedy_move(game),
            },
        ),
        // mce_wide_ens_v1: mce_wide_v1 + diverse prefilter ensemble. Paths
        // come from env var CASCADIA_ENS_PATHS (comma-separated) so callers
        // can point the ensemble at /weights/... in Modal, or relative paths
        // locally. If CASCADIA_ENS_PATHS is unset, behaves identically to
        // mce_wide_v1 (ensemble stays empty).
        "mce_wide_ens_v1" => {
            let ens = std::env::var("CASCADIA_ENS_PATHS").unwrap_or_default();
            let ens_str: &str = Box::leak(ens.into_boxed_str());
            pick_with_env(
                &[
                    ("MCE_LMR", "1"),
                    ("MCE_DIVERSE_PREFILTER", "1"),
                    ("MCE_PREFILTER_ENSEMBLE", ens_str),
                ],
                || match net {
                    Some(n) => {
                        let mut cands = cascadia_ai::mce::expanded_candidates(game);
                        if cands.len() > 32 {
                            cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 32);
                        }
                        cascadia_ai::mce::best_move_nnue_rollout_mce(
                            game, n, 600,
                            cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                            cands, search_rng,
                        )
                    }
                    None => cascadia_ai::search::greedy_move(game),
                },
            )
        },
        // mce_wide_v2: R=800 variant (sim-predicted 85.5% K=8 coverage).
        // Within noise of v1 empirically but costs more.
        "mce_wide_v2" => pick_with_env(
            &[("MCE_LMR", "1"), ("MCE_DIVERSE_PREFILTER", "1")],
            || match net {
                Some(n) => {
                    let mut cands = cascadia_ai::mce::expanded_candidates(game);
                    if cands.len() > 32 {
                        cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 32);
                    }
                    cascadia_ai::mce::best_move_nnue_rollout_mce(
                        game, n, 800,
                        cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                        cands, search_rng,
                    )
                }
                None => cascadia_ai::search::greedy_move(game),
            },
        ),
        // Wildcard: any tag starting "mce_" not matched above = "mce" behavior
        // (used for NNUE-model head-to-head with per-seat weights)
        t if t.starts_with("mce_") => pick_move_by_tag(game, "mce", net, search_rng),
        // Wildcard: any "nnue_*" = "nnue" behavior
        t if t.starts_with("nnue_") => pick_move_by_tag(game, "nnue", net, search_rng),
        "mce_500r" => match net {
            Some(n) => {
                let mut cands = cascadia_ai::mce::expanded_candidates(game);
                if cands.len() > 8 {
                    cands = cascadia_ai::mce::nnue_prefilter_candidates(game, n, cands, 8);
                }
                cascadia_ai::mce::best_move_nnue_rollout_mce(
                    game, n, 500,
                    cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                    cands, search_rng,
                )
            }
            None => cascadia_ai::search::greedy_move(game),
        }
        _ => {
            eprintln!("Unknown strategy tag '{}', falling back to greedy", tag);
            cascadia_ai::search::greedy_move(game)
        }
    }
}

fn simulate_game(rng: &mut StdRng, strategy: &Strategy) -> (cascadia_core::scoring::ScoreBreakdown, cascadia_core::scoring::ScoreBreakdown) {
    simulate_game_inner(rng, strategy, None)
}

/// Parse "A,B,C,A,D" → ScoringCards (Bear, Elk, Salmon, Hawk, Fox order).
fn parse_scoring_cards(s: &str) -> Option<ScoringCards> {
    use cascadia_core::types::ScoringCardVariant;
    let parts: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
    if parts.len() != 5 { return None; }
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

fn simulate_game_inner(
    rng: &mut StdRng,
    strategy: &Strategy,
    mut sample_sink: Option<&mut Vec<cascadia_ai::nnue_train::Sample>>,
) -> (cascadia_core::scoring::ScoreBreakdown, cascadia_core::scoring::ScoreBreakdown) {
    // CASCADIA_SCORING_CARDS="A,B,C,A,D" — override scoring cards for [Bear, Elk, Salmon, Hawk, Fox].
    // Defaults to all-A. Useful for benching against alternate rule sets.
    let cards = std::env::var("CASCADIA_SCORING_CARDS").ok()
        .and_then(|s| parse_scoring_cards(&s))
        .unwrap_or_else(ScoringCards::all_a);
    let mut game = GameState::new(4, cards, rng);
    let mut search_rng = StdRng::seed_from_u64(rng.gen());

    // CASCADIA_OPPONENTS_SAME=1: make all 4 players use the same strategy as player 0.
    // Makes the benchmark "self-play style" — use to test if the strategy holds up
    // against equivalent competition (no free ride from weak greedy opponents).
    let opponents_same = std::env::var("CASCADIA_OPPONENTS_SAME").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);

    // CASCADIA_SEAT_STRATEGIES="tag:tag:tag:tag" — head-to-head with different
    // strategies per seat. Tags: "greedy", "nnue", "mce", "mce_sr", "mce_100r".
    // Requires that the `strategy` passed in has an NNUE net (for mce/nnue tags).
    let seat_tags: Option<Vec<String>> = std::env::var("CASCADIA_SEAT_STRATEGIES").ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.split(':').map(|t| t.to_string()).collect());
    // Extract NNUE net from strategy for tag-based dispatch.
    let seat_net: Option<Arc<cascadia_ai::nnue::NNUENetwork>> = match strategy {
        Strategy::NNUE { net } | Strategy::MCE { net, .. }
            | Strategy::Hybrid { net, .. } | Strategy::ExactExpectimax { net }
            | Strategy::MCTS { net, .. } | Strategy::PolicyMCE { net, .. }
            | Strategy::NRPA { net, .. } | Strategy::OpenLoopMCTS { net, .. }
            | Strategy::GumbelMCTS { net, .. } | Strategy::NnueRolloutMCE { net, .. }
            | Strategy::Expectimax { net, .. } => Some(net.clone()),
        Strategy::MctsTree { net: Some(n), .. } => Some(n.clone()),
        _ => None,
    };
    // CASCADIA_SEAT_WEIGHTS="path1:path2:path3:path4" — per-seat NNUE weights.
    // When set with SEAT_STRATEGIES, each seat uses its own weights for MCE/NNUE
    // decisions. Enables head-to-head between different NNUE models.
    let seat_nets: Option<Vec<Arc<cascadia_ai::nnue::NNUENetwork>>> =
        std::env::var("CASCADIA_SEAT_WEIGHTS").ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.split(':').map(|path| {
                    Arc::new(cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path))
                        .unwrap_or_else(|_| panic!("failed to load seat weights: {}", path)))
                }).collect()
            });

    // Per-seat preference vectors for the "preference" draft opponent.
    // Sampled once per game per seat and held constant across all 20 turns.
    // Only touched when a seat's tag is "preference".
    let num_seats = game.num_players;
    let mut seat_preferences: Vec<Option<[f32; 5]>> = vec![None; num_seats];

    // Per-seat persistent UCT MCTS forests for cross-turn tree reuse (Exp #6).
    // A forest is allocated for a seat iff that seat actually plays MctsTree:
    //   - Strategy::MctsTree: seat 0 always; seats 1..N when CASCADIA_OPPONENTS_SAME=1.
    //   - seat_tags mode: any seat whose tag == "mcts_tree".
    // Forest seeds are derived from search_rng so the result is reproducible
    // game-by-game given the outer RNG.
    let mcts_tree_tag_sims: usize = std::env::var("MCTS_TREE_TAG_SIMS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(600);
    let mcts_tree_tag_parallel: bool = std::env::var("MCTS_TREE_TAG_PARALLEL").ok()
        .map(|s| !s.is_empty() && s != "0" && s.to_ascii_lowercase() != "false")
        .unwrap_or(false);
    let init_forest_for = |p: usize, parallel: bool, seed: u64,
                            net: Option<Arc<cascadia_ai::nnue::NNUENetwork>>| {
        let n = if parallel {
            thread::available_parallelism().map(|nn| nn.get()).unwrap_or(4)
        } else { 1 };
        cascadia_ai::mcts_tree::PersistentUctForest::new_with_net(p, n, seed, net)
    };
    let mut mcts_forests: Vec<Option<cascadia_ai::mcts_tree::PersistentUctForest>> =
        (0..num_seats).map(|p| {
            // Tag-mode allocation: seat tag == "mcts_tree". Use the strategy's
            // NNUE net (if any) for NNUE-guided rollouts, mirroring how MCE
            // tags consume the strategy's net.
            if let Some(ref tags) = seat_tags {
                if tags.get(p).map(|t| t == "mcts_tree").unwrap_or(false) {
                    let n = seat_nets.as_ref().and_then(|nets| nets.get(p))
                        .or(seat_net.as_ref())
                        .map(Arc::clone);
                    return Some(init_forest_for(p, mcts_tree_tag_parallel, search_rng.gen(), n));
                }
            }
            // Strategy-mode allocation.
            match strategy {
                Strategy::MctsTree { parallel, net, .. } => {
                    if p == 0 || opponents_same {
                        Some(init_forest_for(p, *parallel, search_rng.gen(),
                            net.as_ref().map(Arc::clone)))
                    } else { None }
                }
                _ => None,
            }
        }).collect();

    while !game.is_game_over() {
        // HEAD-TO-HEAD MODE: if seat_tags set, dispatch each player's move by tag.
        if let Some(ref tags) = seat_tags {
            let p = game.current_player;
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let tag = tags.get(p).map(|s| s.as_str()).unwrap_or("greedy");
            // If per-seat NNUE weights provided, use this seat's net.
            // Otherwise fall back to the strategy's net.
            let net_for_seat: Option<&Arc<cascadia_ai::nnue::NNUENetwork>> =
                seat_nets.as_ref().and_then(|nets| nets.get(p))
                    .or(seat_net.as_ref());
            let mv = if let Some(forest) = mcts_forests[p].as_mut() {
                // Tag mode: persistent MCTS-tree decision for this seat.
                forest.pick_move(&game, mcts_tree_tag_sims)
            } else {
                match tag {
                    "random" => cascadia_ai::draft_opponents::random_draft_move(&game, &mut search_rng),
                    "scarcity" => cascadia_ai::draft_opponents::scarcity_draft_move(&game, &mut search_rng),
                    "preference" => {
                        if seat_preferences[p].is_none() {
                            seat_preferences[p] = Some(cascadia_ai::draft_opponents::sample_preferences(&mut search_rng));
                        }
                        let prefs = seat_preferences[p].as_ref().unwrap();
                        cascadia_ai::draft_opponents::preference_draft_move(&game, prefs, &mut search_rng)
                    }
                    _ => pick_move_by_tag(&game, tag, net_for_seat, &mut search_rng),
                }
            };
            match mv {
                Some(m) => {
                    if !execute_scored_move(&mut game, &m) { break; }
                    if let Some(forest) = mcts_forests[p].as_mut() {
                        forest.advance(&m);
                    }
                }
                None => break,
            }
            continue;
        }

        // Player 0 is the AI; players 1-3 use NNUE if available, otherwise greedy.
        // If --opp-weights was set via OPPONENT_NET, opponents use that network
        // instead of player 0's net (for "v3 vs v1" style head-to-head experiments).
        if game.current_player != 0 {
            let p = game.current_player;
            // Opponents ALWAYS take the free 3-of-a-kind replacement if available.
            // This is strictly an improvement over the current market state and
            // any rational player would take it. Without this, benchmark opponents
            // were systematically weaker than real play, inflating player 0's scores.
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }
            let opp_mv = if let Some(forest) = mcts_forests[p].as_mut() {
                // Persistent MCTS-tree opponent (opponents_same + Strategy::MctsTree).
                // Run pre-move optimization for fairness (matches the equivalent
                // pick_move dispatch behavior under opponents_same).
                if opponents_same {
                    pre_move_optimize(&mut game, strategy, &cards, &mut search_rng);
                }
                let sims = match strategy {
                    Strategy::MctsTree { simulations, .. } => *simulations,
                    _ => mcts_tree_tag_sims,
                };
                forest.pick_move(&game, sims)
            } else if opponents_same {
                // Run the FULL strategy (including MCE, prefilter, etc) for this opponent.
                // Also needs pre-move optimization for fairness.
                pre_move_optimize(&mut game, strategy, &cards, &mut search_rng);
                pick_move(&game, strategy, &cards, &mut search_rng)
            } else if let Some(opp) = opponent_net() {
                cascadia_ai::nnue_train::pick_best_move_nnue(&game, opp)
                    .or_else(|| greedy_move(&game))
            } else {
                match strategy {
                    Strategy::NNUE { ref net } | Strategy::MCE { ref net, .. }
                        | Strategy::Hybrid { ref net, .. } | Strategy::ExactExpectimax { ref net }
                        | Strategy::MCTS { ref net, .. }
                        | Strategy::PolicyMCE { ref net, .. }
                        | Strategy::NRPA { ref net, .. }
                        | Strategy::OpenLoopMCTS { ref net, .. }
                        | Strategy::GumbelMCTS { ref net, .. }
                        | Strategy::NnueRolloutMCE { ref net, .. } => {
                        cascadia_ai::nnue_train::pick_best_move_nnue(&game, net)
                            .or_else(|| greedy_move(&game))
                    }
                    _ => greedy_move(&game),
                }
            };
            match opp_mv {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) { break; }
                    if let Some(forest) = mcts_forests[p].as_mut() {
                        forest.advance(&mv);
                    }
                }
                None => break,
            }
            continue;
        }

        // Pre-move: decide whether to replace 3-of-a-kind (free) or mulligan (costs token).
        // Iterative: keep applying pre-move actions as long as expected value improves.
        pre_move_optimize(&mut game, strategy, &cards, &mut search_rng);

        // Persistent MCTS-tree decision for player 0 (Exp #6 default mode).
        // Bypasses the stateless pick_move path so cross-turn tree state is
        // preserved. Falls through to the existing logic when no forest exists.
        let mv = if let Some(forest) = mcts_forests[0].as_mut() {
            let sims = match strategy {
                Strategy::MctsTree { simulations, .. } => *simulations,
                _ => mcts_tree_tag_sims,
            };
            forest.pick_move(&game, sims)
        } else if sample_sink.is_some() {
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
                        sample_sink.as_mut().unwrap().push(cascadia_ai::nnue_train::Sample {
                            features,
                            target,
                            aux_bear: 0.0,
                            aux_salmon: 0.0,
                            target_wildlife: 0.0,
                            subscore_targets: [0.0; cascadia_ai::nnue::NUM_HEADS],
                        });
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
                if let Some(forest) = mcts_forests[0].as_mut() {
                    forest.advance(&mv);
                }
            }
            None => break,
        }
    }

    // Return both base score and score with habitat bonuses.
    //
    // When CASCADIA_OPPONENTS_SAME=1, also dump per-player breakdowns to stderr
    // (parsed by symmetric-bench script for self-play stats).
    let base = cascadia_core::scoring::ScoreBreakdown::compute(
        &mut game.boards[0],
        &game.scoring_cards,
    );
    let with_bonus = cascadia_core::scoring::ScoreBreakdown::compute_with_bonuses(
        &mut game.boards,
        &game.scoring_cards,
        0,
    );
    if opponents_same || seat_tags.is_some() {
        // Dump all 4 player scores in a parseable format (shared between
        // symmetric and head-to-head modes). Head-to-head adds seat-tag labels.
        let tags_str = seat_tags.as_ref().map(|t| t.join(":"))
            .unwrap_or_else(|| "self:self:self:self".to_string());
        eprintln!("===SYMMETRIC_GAME_BEGIN=== tags={}", tags_str);
        for p in 0..game.num_players {
            let bd = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[p], &game.scoring_cards,
            );
            let bd_bonus = cascadia_core::scoring::ScoreBreakdown::compute_with_bonuses(
                &mut game.boards, &game.scoring_cards, p,
            );
            let hab: u16 = bd.habitat.iter().sum();
            let wl: u16 = bd.wildlife.iter().sum();
            let tokens = bd.nature_tokens;
            let b = bd.wildlife[0];
            let e = bd.wildlife[1];
            let s = bd.wildlife[2];
            let h = bd.wildlife[3];
            let f = bd.wildlife[4];
            eprintln!("SYMPLAYER p={} base={} bonus={} hab={} wl={} tok={} bear={} elk={} salmon={} hawk={} fox={}",
                p, bd.total, bd_bonus.total, hab, wl, tokens, b, e, s, h, f);
        }
        eprintln!("===SYMMETRIC_GAME_END===");
    }
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
    // Per-game per-category scores for distribution stats
    habitat_per_game: [Vec<u16>; 5],
    wildlife_per_game: [Vec<u16>; 5],
    tokens_per_game: Vec<u16>,
    habitat_total_per_game: Vec<u16>,
    wildlife_total_per_game: Vec<u16>,
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
    // Per-game per-category for distribution stats
    let mut habitat_per_game: [Vec<u16>; 5] = Default::default();
    let mut wildlife_per_game: [Vec<u16>; 5] = Default::default();
    let mut tokens_per_game: Vec<u16> = Vec::with_capacity(num_games);
    let mut habitat_total_per_game: Vec<u16> = Vec::with_capacity(num_games);
    let mut wildlife_total_per_game: Vec<u16> = Vec::with_capacity(num_games);
    for v in habitat_per_game.iter_mut() { v.reserve(num_games); }
    for v in wildlife_per_game.iter_mut() { v.reserve(num_games); }

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
            let mut game_samples: Vec<cascadia_ai::nnue_train::Sample> = Vec::new();
            let result = simulate_game_inner(&mut rng, strategy, Some(&mut game_samples));
            if !game_samples.is_empty() {
                total_samples += game_samples.len();
                // MCV3 format: aux+wildlife fields are zero for MCE cache samples, but
                // the format is consistent so v3-aware trainings can load them alongside
                // self-play data without format gymnastics.
                let _ = cascadia_ai::nnue_train::append_mce_samples_v3(samples_path, &game_samples);
            }
            result
        } else {
            simulate_game(&mut rng, strategy)
        };
        scores.push(base.total);
        scores_with_bonus.push(with_bonus.total);
        let mut hab_sum: u16 = 0;
        let mut wl_sum: u16 = 0;
        for t in 0..5 {
            total_habitat[t] += base.habitat[t] as u64;
            total_wildlife[t] += base.wildlife[t] as u64;
            habitat_per_game[t].push(base.habitat[t]);
            wildlife_per_game[t].push(base.wildlife[t]);
            hab_sum += base.habitat[t];
            wl_sum += base.wildlife[t];
        }
        habitat_total_per_game.push(hab_sum);
        wildlife_total_per_game.push(wl_sum);
        tokens_per_game.push(base.nature_tokens);
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
        habitat_per_game,
        wildlife_per_game,
        tokens_per_game,
        habitat_total_per_game,
        wildlife_total_per_game,
    }
}

/// Returns (mean, p10, median, p90, max)
fn dist_stats(v: &mut Vec<u16>) -> (f64, u16, u16, u16, u16) {
    if v.is_empty() { return (0.0, 0, 0, 0, 0); }
    let n = v.len();
    let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    v.sort();
    let p10 = v[n / 10];
    let median = v[n / 2];
    let p90 = v[9 * n / 10];
    let max = v[n - 1];
    (mean, p10, median, p90, max)
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
    println!("  Score Breakdown (mean | P10 | median | P90 | max):");
    let hab_total: f64 = r.avg_habitat.iter().sum();
    let wl_total: f64 = r.avg_wildlife.iter().sum();
    let mut hab_total_v = r.habitat_total_per_game.clone();
    let (_, hab_p10, hab_med, hab_p90, hab_max) = dist_stats(&mut hab_total_v);
    println!("    Habitat:  {:5.1} | {:3} | {:3} | {:3} | {:3}    (+{:.1} bonus)",
             hab_total, hab_p10, hab_med, hab_p90, hab_max, r.avg_habitat_bonus);
    for (i, name) in terrains.iter().enumerate() {
        let mut v = r.habitat_per_game[i].clone();
        let (m, p10, med, p90, mx) = dist_stats(&mut v);
        println!("      {:<10} {:5.1} | {:3} | {:3} | {:3} | {:3}", name, m, p10, med, p90, mx);
    }
    let mut wl_total_v = r.wildlife_total_per_game.clone();
    let (_, wl_p10, wl_med, wl_p90, wl_max) = dist_stats(&mut wl_total_v);
    println!("    Wildlife: {:5.1} | {:3} | {:3} | {:3} | {:3}", wl_total, wl_p10, wl_med, wl_p90, wl_max);
    for (i, name) in wildlife.iter().enumerate() {
        let mut v = r.wildlife_per_game[i].clone();
        let (m, p10, med, p90, mx) = dist_stats(&mut v);
        println!("      {:<10} {:5.1} | {:3} | {:3} | {:3} | {:3}", name, m, p10, med, p90, mx);
    }
    let mut tok_v = r.tokens_per_game.clone();
    let (tok_m, tok_p10, tok_med, tok_p90, tok_max) = dist_stats(&mut tok_v);
    println!("    Tokens:   {:5.1} | {:3} | {:3} | {:3} | {:3}", tok_m, tok_p10, tok_med, tok_p90, tok_max);
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

    // Cross-binary daemon mode: reads text commands from stdin, maintains one
    // GameState internally, replies on stdout. See run_daemon() for protocol.
    if args.iter().any(|a| a == "--daemon") {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        run_daemon(weights_path);
        return;
    }

    // Optional opponent NNUE: load and stash in OPPONENT_NET so simulate_game_inner
    // uses it for players 1..3 instead of player 0's net.
    if let Some(opp_path) = args.iter().position(|a| a == "--opp-weights")
        .and_then(|i| args.get(i + 1))
    {
        match cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(opp_path)) {
            Ok(net) => {
                let _ = OPPONENT_NET.set(Arc::new(net));
                eprintln!("✓ Opponent NNUE loaded from {}", opp_path);
            }
            Err(e) => {
                eprintln!("⚠ Failed to load --opp-weights {}: {}", opp_path, e);
            }
        }
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
    let run_train_pairwise = args.iter().any(|a| a == "--train-pairwise");
    let run_export_pytorch = args.iter().any(|a| a == "--export-pytorch");
    let run_self_play = args.iter().any(|a| a == "--self-play");
    let run_selfplay_pool = args.iter().any(|a| a == "--selfplay-pool");

    let run_mce_selfplay = args.iter().any(|a| a == "--mce-selfplay");
    let run_exact_selfplay = args.iter().any(|a| a == "--exact-selfplay");
    let run_tile_token_selfplay = args.iter().any(|a| a == "--tile-token-selfplay");
    let run_rich_tile_token_selfplay = args.iter().any(|a| a == "--rich-tile-token-selfplay");
    let run_external_eval = args.iter().any(|a| a == "--external-eval");
    let run_gnn_mce_bench = args.iter().any(|a| a == "--gnn-mce-bench");

    // --greedy-ub: print the greedy upper bound for various move counts.
    // Useful for sanity-checking the upper bound oracle from the CLI.
    if args.iter().any(|a| a == "--greedy-ub") {
        let r: usize = args.iter().position(|a| a == "--moves")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(20);
        println!("Greedy upper bound for {} remaining moves:", r);
        for moves in 0..=r {
            let ub = cascadia_ai::greedy_ub::greedy_upper_bound(moves);
            println!("  R={:2} → UB={}", moves, ub);
        }
        return;
    }

    // --train-policy: collect MCE-scored candidates and train the policy head.
    // Plays N games, at each AI turn scores top-K candidates with MCE,
    // then trains w3_policy via cross-entropy on softmax(MCE_scores/τ).
    if args.iter().any(|a| a == "--train-policy") {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let init_path = args.iter().position(|a| a == "--init-weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let load_path = init_path.unwrap_or(weights_path);
        let mut net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(load_path))
            .unwrap_or_else(|e| { eprintln!("Failed to load {}: {}", load_path, e); std::process::exit(1); });

        let n_games: usize = num_games.max(1);
        let rollouts_per_cand: usize = args.iter().position(|a| a == "--rollouts-per-cand")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(100);
        let max_cands: usize = args.iter().position(|a| a == "--max-candidates")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(50);
        let temperature: f32 = args.iter().position(|a| a == "--temperature")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2.0);
        let policy_lr: f32 = args.iter().position(|a| a == "--policy-lr")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.01);
        let epochs: usize = args.iter().position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(20);
        let seed_base: u64 = std::env::var("CASCADIA_SEED_OFFSET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0);

        eprintln!("Policy head training: {} games, {} rollouts/cand, max {} cands, τ={}, lr={}, epochs={}",
            n_games, rollouts_per_cand, max_cands, temperature, policy_lr, epochs);
        eprintln!("  Weights: {} → {}", load_path, weights_path);

        // Try loading pre-collected data (from Modal) instead of collecting
        let load_data_path = args.iter().position(|a| a == "--load-policy-data")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));

        if let Some(path) = load_data_path {
            use std::io::Read;
            let mut f = std::fs::File::open(path).expect("Failed to open policy data");
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic).unwrap();
            assert_eq!(&magic, b"PDAT", "Bad policy data magic");
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4).unwrap();
            let n_pos = u32::from_le_bytes(buf4) as usize;

            let mut all_positions: Vec<cascadia_ai::nnue::PositionPolicyData> = Vec::with_capacity(n_pos);
            let mut buf2 = [0u8; 2];
            for _ in 0..n_pos {
                // Base features
                f.read_exact(&mut buf2).unwrap();
                let nb = u16::from_le_bytes(buf2) as usize;
                let mut base = Vec::with_capacity(nb);
                for _ in 0..nb { f.read_exact(&mut buf2).unwrap(); base.push(u16::from_le_bytes(buf2)); }
                // Candidates
                f.read_exact(&mut buf4).unwrap();
                let nc = u32::from_le_bytes(buf4) as usize;
                let mut cands = Vec::with_capacity(nc);
                for _ in 0..nc {
                    f.read_exact(&mut buf2).unwrap();
                    let nf = u16::from_le_bytes(buf2) as usize;
                    let mut feats = Vec::with_capacity(nf);
                    for _ in 0..nf { f.read_exact(&mut buf2).unwrap(); feats.push(u16::from_le_bytes(buf2)); }
                    f.read_exact(&mut buf4).unwrap();
                    let score = f32::from_le_bytes(buf4);
                    cands.push((feats, score));
                }
                all_positions.push(cascadia_ai::nnue::PositionPolicyData { base_features: base, candidates: cands });
            }
            eprintln!("  Loaded {} positions from {}", all_positions.len(), path);

            // Train
            let mut policy_net = cascadia_ai::policy_net::PolicyNetwork::from_nnue(&net);
            eprintln!("  PolicyNetwork: {} → {} → {} → 1",
                cascadia_ai::nnue::NUM_FEATURES, 512, 256);
            let train_start = Instant::now();
            for epoch in 0..epochs {
                let mut eloss = 0.0f64;
                let mut eagree = 0usize;
                let mut en = 0usize;
                for pos in &all_positions {
                    let (l, c) = policy_net.train_ranking(pos, policy_lr, temperature);
                    eloss += l as f64; if c { eagree += 1; } en += 1;
                }
                if (epoch+1) % 10 == 0 || epoch == 0 || epoch == epochs-1 {
                    eprint!("\r  Epoch {}/{}: loss={:.4} agree={:.1}%",
                        epoch+1, epochs, eloss/en.max(1) as f64, 100.0*eagree as f64/en.max(1) as f64);
                }
            }
            eprintln!("\n  Training: {} epochs in {:.1?}", epochs, train_start.elapsed());
            let mut fa = 0usize; let mut fn_ = 0usize; let mut fl = 0.0f64;
            for pos in &all_positions {
                let (l, c) = policy_net.train_ranking(pos, 0.0, temperature);
                fl += l as f64; if c { fa += 1; } fn_ += 1;
            }
            eprintln!("  Final: loss={:.4} agree={:.1}%", fl/fn_.max(1) as f64, 100.0*fa as f64/fn_.max(1) as f64);
            policy_net.save(std::path::Path::new(weights_path)).expect("save");
            eprintln!("  Saved to {}", weights_path);
            return;
        }

        // Phase 1: Collect data (play games, score candidates with MCE)
        let start = Instant::now();
        let mut all_positions: Vec<cascadia_ai::nnue::PositionPolicyData> = Vec::new();
        let cards = ScoringCards::all_a();

        for g in 0..n_games {
            let mut rng = StdRng::seed_from_u64(seed_base + g as u64 * 1000 + 42);
            let mut game = GameState::new(4, cards, &mut rng);
            let mut search_rng = StdRng::seed_from_u64(rng.gen());

            while !game.is_game_over() {
                if game.current_player != 0 {
                    if game.can_replace_overflow().is_some() { game.replace_overflow(); }
                    let mv = cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                        .or_else(|| greedy_move(&game));
                    match mv { Some(m) => { if !execute_scored_move(&mut game, &m) { break; } } None => break }
                    continue;
                }
                if game.can_replace_overflow().is_some() { game.replace_overflow(); }

                let candidates = cascadia_ai::mce::expanded_candidates(&game);
                if candidates.is_empty() { break; }
                let (nnue_sorted, _) = cascadia_ai::mce::nnue_prefilter_with_priors(
                    &game, &net, candidates, max_cands);

                let mce_results = cascadia_ai::mce::rank_all_candidates_mce(
                    &game, &net, rollouts_per_cand, &nnue_sorted, &mut search_rng);

                // Collect base features + (afterstate_features, mce_score) for each candidate
                let player = game.current_player;
                let bag_before = cascadia_ai::nnue::BagInfo::from_game_for_player(&game, player);
                let base_features = cascadia_ai::nnue::extract_features_with_bag(
                    &game.boards[player], Some(&bag_before));

                let mut cand_data: Vec<(Vec<u16>, f32)> = Vec::new();
                for (mv, stat) in &mce_results {
                    let mut gs = game.clone();
                    if execute_scored_move(&mut gs, mv) {
                        let bag = cascadia_ai::nnue::BagInfo::from_game_for_player(&gs, player);
                        let features = cascadia_ai::nnue::extract_features_with_bag(&gs.boards[player], Some(&bag));
                        cand_data.push((features, stat.mean as f32));
                    }
                }
                all_positions.push(cascadia_ai::nnue::PositionPolicyData {
                    base_features,
                    candidates: cand_data,
                });

                // Play MCE-best move
                if let Some((best, _)) = mce_results.first() {
                    if !execute_scored_move(&mut game, best) { break; }
                } else { break; }
            }
            eprint!("\r  Game {}/{}: {} positions collected", g+1, n_games, all_positions.len());
        }
        let collect_time = start.elapsed();
        eprintln!("\n  Collection: {} positions in {:.1?}", all_positions.len(), collect_time);

        // Save collected data to binary for offline training
        let data_path = args.iter().position(|a| a == "--save-policy-data")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        if let Some(path) = data_path {
            use std::io::Write;
            let mut f = std::fs::File::create(path).expect("Failed to create policy data file");
            f.write_all(b"PDAT").unwrap();
            let n_pos = all_positions.len() as u32;
            f.write_all(&n_pos.to_le_bytes()).unwrap();
            for pos in &all_positions {
                // Base features
                let nb = pos.base_features.len() as u16;
                f.write_all(&nb.to_le_bytes()).unwrap();
                for &fi in &pos.base_features { f.write_all(&fi.to_le_bytes()).unwrap(); }
                // Candidates
                let nc = pos.candidates.len() as u32;
                f.write_all(&nc.to_le_bytes()).unwrap();
                for (feats, score) in &pos.candidates {
                    let nf = feats.len() as u16;
                    f.write_all(&nf.to_le_bytes()).unwrap();
                    for &fi in feats { f.write_all(&fi.to_le_bytes()).unwrap(); }
                    f.write_all(&score.to_le_bytes()).unwrap();
                }
            }
            eprintln!("  Policy data saved to {}", path);
            if epochs == 0 { return; }
        }

        // Phase 2: Train wide policy network (initialized from value NNUE's first layer)
        let mut policy_net = cascadia_ai::policy_net::PolicyNetwork::from_nnue(&net);
        eprintln!("  PolicyNetwork: {} → {} → {} → 1 ({:.1}M params)",
            cascadia_ai::nnue::NUM_FEATURES, 512, 256,
            (cascadia_ai::nnue::NUM_FEATURES * 512 + 512 * 256 + 256) as f64 / 1e6);

        let train_start = Instant::now();
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f64;
            let mut epoch_agree = 0usize;
            let mut epoch_n = 0usize;
            for pos in &all_positions {
                let (loss, correct) = policy_net.train_ranking(pos, policy_lr, temperature);
                epoch_loss += loss as f64;
                if correct { epoch_agree += 1; }
                epoch_n += 1;
            }
            if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1 {
                let avg_loss = epoch_loss / epoch_n.max(1) as f64;
                let agree_pct = 100.0 * epoch_agree as f64 / epoch_n.max(1) as f64;
                eprint!("\r  Epoch {}/{}: loss={:.4} agree={:.1}%", epoch+1, epochs, avg_loss, agree_pct);
            }
        }
        let train_time = train_start.elapsed();
        // Final eval pass (lr=0)
        let mut final_agree = 0usize;
        let mut final_n = 0usize;
        let mut final_loss_sum = 0.0f64;
        for pos in &all_positions {
            let (loss, correct) = policy_net.train_ranking(pos, 0.0, temperature);
            final_loss_sum += loss as f64;
            if correct { final_agree += 1; }
            final_n += 1;
        }
        eprintln!("\n  Training: {} epochs in {:.1?}", epochs, train_time);
        eprintln!("  Final: loss={:.4} top-1 agree={:.1}% ({}/{})",
            final_loss_sum / final_n.max(1) as f64,
            100.0 * final_agree as f64 / final_n.max(1) as f64,
            final_agree, final_n);

        // Save policy network
        policy_net.save(std::path::Path::new(weights_path)).expect("Failed to save");
        eprintln!("  Saved policy net to {}", weights_path);
        return;
    }

    // --rank-correlation: measure NNUE prefilter vs MCE ground-truth ranking.
    // For one game, at each AI turn:
    //   1. Generate expanded candidates
    //   2. Score all with NNUE (prefilter scoring)
    //   3. Score all with MCE (uniform, N rollouts per candidate)
    //   4. Print per-candidate comparison lines
    if args.iter().any(|a| a == "--rank-correlation") {
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights.bin");
        let rollouts_per_cand: usize = args.iter().position(|a| a == "--rollouts-per-cand")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(100);

        let net = Arc::new(
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for rank-correlation")
        );

        let use_random_seed = args.iter().any(|a| a == "--random-seed");
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut entropy_rng = if use_random_seed {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(0xCAFE_BEEF + seed_offset)
        };

        let cards = ScoringCards::all_a();
        let mut game = GameState::new(4, cards, &mut entropy_rng);
        let mut search_rng = StdRng::seed_from_u64(entropy_rng.gen());

        eprintln!("rank-correlation: seed_offset={}, rollouts_per_cand={}, weights={}",
                  seed_offset, rollouts_per_cand, weights_path);

        let mut ai_turn = 0u32;
        let mce_opponents = args.iter().any(|a| a == "--mce-opponents");
        if mce_opponents {
            eprintln!("  Opponents: MCE(50) with prefilter-k 8");
        }
        let net_arc = std::sync::Arc::new(net.clone());

        while !game.is_game_over() {
            if game.current_player != 0 {
                if game.can_replace_overflow().is_some() {
                    game.replace_overflow();
                }
                let opp_mv = if mce_opponents {
                    let mut cands = cascadia_ai::mce::expanded_candidates(&game);
                    if cands.len() > 8 {
                        cands = cascadia_ai::mce::nnue_prefilter_candidates(&game, &net, cands, 8);
                    }
                    cascadia_ai::mce::best_move_nnue_rollout_mce(
                        &game, &net, 50,
                        cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                        cands, &mut search_rng,
                    )
                } else {
                    cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net)
                        .or_else(|| greedy_move(&game))
                };
                match opp_mv {
                    Some(mv) => { if !execute_scored_move(&mut game, &mv) { break; } }
                    None => break,
                }
                continue;
            }

            ai_turn += 1;

            // Free overflow replacement
            if game.can_replace_overflow().is_some() {
                game.replace_overflow();
            }

            // 1. Generate expanded candidates, cap to --max-candidates
            let max_cands: usize = args.iter().position(|a| a == "--max-candidates")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(999);
            let candidates = cascadia_ai::mce::expanded_candidates(&game);
            let n_cands = candidates.len();
            if n_cands == 0 {
                eprintln!("  turn {} — no candidates, skipping", ai_turn);
                break;
            }

            // 2. Score all with NNUE prefilter (returns sorted by NNUE score desc),
            //    capped to max_candidates so MCE budget is bounded.
            let (nnue_sorted, nnue_priors) = cascadia_ai::mce::nnue_prefilter_with_priors(
                &game, &net, candidates.clone(), max_cands,
            );

            // Build a map from candidate key to (nnue_rank, nnue_score)
            // Key includes wildlife_market_index to distinguish nature-token drafts
            type CandKey = (usize, Option<usize>, i8, i8, u8, Option<i8>, Option<i8>);
            let cand_key = |mv: &cascadia_ai::eval::ScoredMove| -> CandKey {
                (mv.market_index, mv.wildlife_market_index, mv.tile_q, mv.tile_r, mv.rotation, mv.wildlife_q, mv.wildlife_r)
            };
            let mut nnue_map: std::collections::HashMap<CandKey, (usize, f32)> =
                std::collections::HashMap::new();
            for (rank, (mv, &score)) in nnue_sorted.iter().zip(nnue_priors.iter()).enumerate() {
                nnue_map.insert(cand_key(mv), (rank, score));
            }

            // 3. Score NNUE's top candidates with MCE (uniform rollouts, no elimination).
            //    Both NNUE and MCE evaluate the SAME candidate set (NNUE top-K).
            let mce_results = cascadia_ai::mce::rank_all_candidates_mce(
                &game, &net, rollouts_per_cand, &nnue_sorted, &mut search_rng,
            );

            // Build MCE rank map
            let mut mce_map: std::collections::HashMap<CandKey, (usize, cascadia_ai::mce::MceStat)> =
                std::collections::HashMap::new();
            for (rank, (mv, stat)) in mce_results.iter().enumerate() {
                mce_map.insert(cand_key(mv), (rank, stat.clone()));
            }

            // Resolve wildlife type names from market
            let wl_name = |mv: &cascadia_ai::eval::ScoredMove| -> &'static str {
                let wi = mv.wildlife_market_index.unwrap_or(mv.market_index);
                match game.market.pairs.get(wi).and_then(|p| p.as_ref()) {
                    Some(p) => match p.wildlife as u8 {
                        0 => "bear", 1 => "elk", 2 => "salmon", 3 => "hawk", 4 => "fox", _ => "?"
                    },
                    None => "?",
                }
            };

            // Board state summary for this turn
            let board = &game.boards[0];
            let hab_score: u16 = board.largest_group.iter().sum();
            let wl_counts: Vec<usize> = (0..5).map(|w| board.wildlife_positions[w].len()).collect();
            let tokens = board.nature_tokens;
            let turns_left = game.turns_remaining;

            // 4. Print JSONL per-candidate
            let mut n_common = 0usize;
            for (nnue_rank, mv) in nnue_sorted.iter().enumerate() {
                let key = cand_key(mv);
                let (mce_rank, stat) = match mce_map.get(&key) {
                    Some((r, s)) => (*r, s.clone()),
                    None => continue,
                };
                n_common += 1;
                let nnue_score = nnue_priors.get(nnue_rank).copied().unwrap_or(0.0);
                let is_independent = mv.wildlife_market_index.is_some();
                let wildlife = wl_name(mv);
                println!(
                    "{{\"turn\":{},\"n_cands\":{},\"nnue_rank\":{},\"mce_rank\":{},\
\"nnue_score\":{:.2},\"mce_mean\":{:.2},\"mce_std\":{:.2},\"mce_min\":{},\"mce_max\":{},\"mce_median\":{:.1},\
\"market\":{},\"wildlife\":\"{}\",\"independent\":{},\
\"tile_q\":{},\"tile_r\":{},\"rot\":{},\"wl_q\":{},\"wl_r\":{},\
\"hab\":{},\"wl_bear\":{},\"wl_elk\":{},\"wl_salmon\":{},\"wl_hawk\":{},\"wl_fox\":{},\"tokens\":{},\"turns_left\":{}}}",
                    ai_turn, nnue_sorted.len(),
                    nnue_rank, mce_rank,
                    nnue_score, stat.mean, stat.std, stat.min, stat.max, stat.median,
                    mv.market_index, wildlife, is_independent,
                    mv.tile_q, mv.tile_r, mv.rotation,
                    mv.wildlife_q.unwrap_or(-99), mv.wildlife_r.unwrap_or(-99),
                    hab_score, wl_counts[0], wl_counts[1], wl_counts[2], wl_counts[3], wl_counts[4],
                    tokens, turns_left,
                );
            }

            eprintln!("  turn {} — {} expanded, {} NNUE top-{}, {} MCE-scored, {} common",
                      ai_turn, n_cands, nnue_sorted.len(), max_cands, mce_results.len(), n_common);

            // Play the MCE-best move to continue the game
            if let Some((best_mv, _)) = mce_results.first() {
                if !execute_scored_move(&mut game, best_mv) { break; }
            } else {
                break;
            }
        }

        // Print final scores for all players (same SYMPLAYER format used by HH)
        for p in 0..game.num_players {
            let bd = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[p], &game.scoring_cards,
            );
            let bd_bonus = cascadia_core::scoring::ScoreBreakdown::compute_with_bonuses(
                &mut game.boards, &game.scoring_cards, p,
            );
            let hab: u16 = bd.habitat.iter().sum();
            let wl: u16 = bd.wildlife.iter().sum();
            println!(
                "{{\"type\":\"score\",\"player\":{},\"base\":{},\"bonus\":{},\"hab\":{},\"wl\":{},\"tok\":{},\"bear\":{},\"elk\":{},\"salmon\":{},\"hawk\":{},\"fox\":{}}}",
                p, bd.total, bd_bonus.total, hab, wl, bd.nature_tokens,
                bd.wildlife[0], bd.wildlife[1], bd.wildlife[2], bd.wildlife[3], bd.wildlife[4]);
        }
        eprintln!("rank-correlation: done, {} AI turns", ai_turn);
        return;
    }

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
        println!("  Value samples (MCV3) → {}", out_path);
        println!("  Policy samples → {}", policy_out);
        let start = Instant::now();
        let mut all_value_samples: Vec<cascadia_ai::nnue_train::Sample> = Vec::new();
        let mut all_policy_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
        let mut total_final_score = 0u64;

        for game_i in 0..num_games {
            let mut rng = StdRng::seed_from_u64(entropy_rng.gen());
            let cards = ScoringCards::all_a();
            let mut game = GameState::new(4, cards, &mut rng);

            // Each turn record: (features, current_total, current_wildlife)
            let mut turn_value_records: Vec<(Vec<u16>, f32, f32)> = Vec::new();
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

                let cur_bd = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0].clone(), &game.scoring_cards,
                );
                let current = cur_bd.total as f32;

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

                // Value data: afterstate of chosen move, with current total + wildlife
                let after_bd = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0], &game.scoring_cards,
                );
                let bag_info = cascadia_ai::nnue::BagInfo::from_game(&game);
                let features = cascadia_ai::nnue::extract_features_with_bag(
                    &game.boards[0], Some(&bag_info));
                turn_value_records.push((features, after_bd.total as f32, after_bd.wildlife_total() as f32));
            }

            let final_bd = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0], &game.scoring_cards,
            );
            let final_score = final_bd.total as f32;
            let final_wildlife = final_bd.wildlife_total() as f32;
            let final_bear_pairs = cascadia_ai::nnue_train::count_bear_pairs(&game.boards[0]) as f32;
            let final_salmon_chain = cascadia_ai::nnue_train::longest_salmon_chain(&game.boards[0]) as f32;
            total_final_score += final_score as u64;

            for (features, current, current_wildlife) in turn_value_records {
                let delta = (final_score - current).max(0.0);
                let delta_wildlife = (final_wildlife - current_wildlife).max(0.0);
                all_value_samples.push(cascadia_ai::nnue_train::Sample {
                    features,
                    target: delta,
                    aux_bear: final_bear_pairs,
                    aux_salmon: final_salmon_chain,
                    target_wildlife: delta_wildlife,
                    subscore_targets: [0.0; cascadia_ai::nnue::NUM_HEADS],
                });
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

        cascadia_ai::nnue_train::append_mce_samples_v3(
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
        println!("  Value samples (MCV3) → {}", out_path);
        println!("  Policy samples → {}", policy_out);
        let start = Instant::now();
        let mut all_value_samples: Vec<cascadia_ai::nnue_train::Sample> = Vec::new();
        let mut all_policy_groups: Vec<cascadia_ai::nnue_train::PolicyGroup> = Vec::new();
        let mut total_final_score = 0u64;

        for game_i in 0..num_games {
            let mut rng = StdRng::seed_from_u64(entropy_rng.gen());
            let cards = ScoringCards::all_a();
            let mut game = GameState::new(4, cards, &mut rng);
            let mut search_rng = StdRng::seed_from_u64(rng.gen());

            // Per-turn records: (chosen_afterstate_features, current_total, current_wildlife)
            let mut turn_value_records: Vec<(Vec<u16>, f32, f32)> = Vec::new();
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

                let cur_bd = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0].clone(), &game.scoring_cards,
                );
                let current = cur_bd.total as f32;

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

                // Value data: afterstate of chosen move, with current total + wildlife
                let after_bd = cascadia_core::scoring::ScoreBreakdown::compute(
                    &mut game.boards[0], &game.scoring_cards,
                );
                let bag_info = cascadia_ai::nnue::BagInfo::from_game(&game);
                let features = cascadia_ai::nnue::extract_features_with_bag(
                    &game.boards[0], Some(&bag_info));
                turn_value_records.push((features, after_bd.total as f32, after_bd.wildlife_total() as f32));
            }

            let final_bd = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0], &game.scoring_cards,
            );
            let final_score = final_bd.total as f32;
            let final_wildlife = final_bd.wildlife_total() as f32;
            let final_bear_pairs = cascadia_ai::nnue_train::count_bear_pairs(&game.boards[0]) as f32;
            let final_salmon_chain = cascadia_ai::nnue_train::longest_salmon_chain(&game.boards[0]) as f32;
            total_final_score += final_score as u64;

            // Value samples: delta labels with aux + wildlife targets
            for (features, current, current_wildlife) in turn_value_records {
                let delta = (final_score - current).max(0.0);
                let delta_wildlife = (final_wildlife - current_wildlife).max(0.0);
                all_value_samples.push(cascadia_ai::nnue_train::Sample {
                    features,
                    target: delta,
                    aux_bear: final_bear_pairs,
                    aux_salmon: final_salmon_chain,
                    target_wildlife: delta_wildlife,
                    subscore_targets: [0.0; cascadia_ai::nnue::NUM_HEADS],
                });
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

        // Write value samples (MCV3 format)
        cascadia_ai::nnue_train::append_mce_samples_v3(
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
    } else if run_selfplay_pool {
        // FSP self-play with opponent pool from CASCADIA_TRAIN_OPP_POOL env.
        // Used by Modal parallel data generation — each worker produces a shard
        // that gets concatenated into one training cache.
        //
        // Usage:
        //   CASCADIA_TRAIN_OPP_POOL="random,scarcity,mce93.bin,iter10.bin" \
        //   CASCADIA_TRAIN_SEED=<seed> \
        //   cascadia-cli N --selfplay-pool --init-weights W_IN --out shard.bin \
        //     [--epsilon 0.1] [--temperature 2.0]
        let init_path = args.iter().position(|a| a == "--init-weights")
            .or_else(|| args.iter().position(|a| a == "--weights"))
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("selfplay_pool_samples.bin");
        let epsilon: f32 = args.iter().position(|a| a == "--epsilon")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.1);
        let temperature: Option<f32> = args.iter().position(|a| a == "--temperature")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok());
        let seed: u64 = std::env::var("CASCADIA_TRAIN_SEED").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(42);

        let net = init_path.and_then(|p| {
            match cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(p)) {
                Ok(n) => { eprintln!("[selfplay-pool] loaded weights from {}", p); Some(n) }
                Err(e) => { eprintln!("[selfplay-pool] failed to load {}: {}", p, e); None }
            }
        });

        let mode = if let Some(t) = temperature {
            cascadia_ai::nnue_train::SamplingMode::Softmax(t)
        } else {
            cascadia_ai::nnue_train::SamplingMode::EpsilonGreedy(epsilon)
        };

        eprintln!("[selfplay-pool] {} games, seed={}, out={}",
                  num_games, seed, out_path);
        let start = Instant::now();
        let samples = cascadia_ai::nnue_train::generate_samples_with_mode(
            num_games, seed, net.as_ref(), mode, 4,
        );
        let elapsed = start.elapsed();

        // v5-feat builds write MCV4 (with per-subscore deltas for split-head training);
        // older builds write MCV3 (target_wildlife only).
        #[cfg(feature = "v5-feat")]
        cascadia_ai::nnue_train::append_mce_samples_v4(
            std::path::Path::new(out_path), &samples,
        ).expect("Failed to write shard");
        #[cfg(not(feature = "v5-feat"))]
        cascadia_ai::nnue_train::append_mce_samples_v3(
            std::path::Path::new(out_path), &samples,
        ).expect("Failed to write shard");
        let fmt_label = if cfg!(feature = "v5-feat") { "MCV4" } else { "MCV3" };
        eprintln!("[selfplay-pool] {} samples from {} games in {:.1?} → {} ({})",
                  samples.len(), num_games, elapsed, out_path, fmt_label);
        println!("SAMPLES={}", samples.len());
        println!("GAMES={}", num_games);
        println!("ELAPSED_SEC={}", elapsed.as_secs_f64());
        println!("FORMAT={}", fmt_label);
        return;
    } else if run_self_play {
        // Generate NNUE self-play games and write to MCV3 format
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("self_play_samples.bin");
        let epsilon: f32 = args.iter().position(|a| a == "--epsilon")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.1);
        // Optional simulated-annealing temperature — if present, replaces ε-greedy
        // with softmax sampling over NNUE-scored candidates.
        let temperature: Option<f32> = args.iter().position(|a| a == "--temperature")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok());

        let top_pct: f32 = args.iter().position(|a| a == "--top-pct")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(100.0);

        let net = weights_path.and_then(|p| {
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(p)).ok()
        });
        let strategy = if net.is_some() { "NNUE" } else { "greedy" };
        let filter_str = if top_pct < 100.0 { format!(", top {}%", top_pct) } else { String::new() };
        let mode_str = if let Some(t) = temperature {
            format!("softmax(T={})", t)
        } else {
            format!("epsilon={}", epsilon)
        };
        println!("Generating {} self-play games ({}, {}{}, out={})",
            num_games, strategy, mode_str, filter_str, out_path);

        let start = Instant::now();
        let seed = rand::random::<u64>();

        let mode = if let Some(t) = temperature {
            cascadia_ai::nnue_train::SamplingMode::Softmax(t)
        } else {
            cascadia_ai::nnue_train::SamplingMode::EpsilonGreedy(epsilon)
        };

        // Always use generate_games so we get per-game final_scores for free.
        let mut games = cascadia_ai::nnue_train::generate_games_with_mode(
            num_games, seed, net.as_ref(), mode, 4,
        );

        // Compute self-play score distribution stats over ALL games (before any filtering).
        // This gives a near-noise-free measure of the AI's play quality at this iteration.
        {
            let mut scores: Vec<u16> = games.iter().map(|g| g.final_score).collect();
            scores.sort();
            let n = scores.len();
            if n > 0 {
                let mean: f64 = scores.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
                let p10 = scores[n / 10];
                let median = scores[n / 2];
                let p90 = scores[9 * n / 10];
                let min = scores[0];
                let max = scores[n - 1];
                let std_dev: f64 = (scores.iter().map(|&s| {
                    let d = s as f64 - mean;
                    d * d
                }).sum::<f64>() / n as f64).sqrt();
                let stderr = std_dev / (n as f64).sqrt();
                println!("  Self-play score distribution (n={}, ε={}):", n, epsilon);
                println!("    Mean:   {:.2} (±{:.3} stderr)", mean, stderr);
                println!("    Median: {}", median);
                println!("    P10:    {}", p10);
                println!("    P90:    {}", p90);
                println!("    Min/Max: {}/{}", min, max);
                println!("    StdDev: {:.2}", std_dev);
                // Histogram of buckets of size 5
                let bucket = 5usize;
                let min_b = (min as usize / bucket) * bucket;
                let max_b = (max as usize / bucket + 1) * bucket;
                println!("    Distribution:");
                let mut b = min_b;
                while b < max_b {
                    let count = scores.iter().filter(|&&s| {
                        (s as usize) >= b && (s as usize) < b + bucket
                    }).count();
                    let bar_len = (count * 60) / n.max(1);
                    let bar: String = "█".repeat(bar_len);
                    println!("    {:3}-{:3}: {:6} {}", b, b + bucket - 1, count, bar);
                    b += bucket;
                }
            }
        }

        // Optional top-pct filtering (e.g., elite games only)
        let samples = if top_pct < 100.0 {
            games.sort_by(|a, b| b.final_score.cmp(&a.final_score));
            let keep = ((games.len() as f32 * top_pct / 100.0).ceil() as usize).max(1);
            let cutoff = games[keep - 1].final_score;
            println!("  Top {}%: keeping {} games (score >= {})", top_pct, keep, cutoff);
            let avg_score: f64 = games[..keep].iter().map(|g| g.final_score as f64).sum::<f64>() / keep as f64;
            println!("  Avg score of kept games: {:.1}", avg_score);
            games.truncate(keep);
            games.into_iter().flat_map(|g| g.samples).collect::<Vec<_>>()
        } else {
            games.into_iter().flat_map(|g| g.samples).collect::<Vec<_>>()
        };

        // Write samples: default to MCV3 format (includes aux_bear, aux_salmon, target_wildlife).
        // MCV3 is the required format going forward — all new caches and training data must have
        // aux fields and wildlife targets populated so downstream trainings can use any head.
        // Legacy MCEP and MCV2 are still READABLE for loading old data, but we never WRITE them.
        let _ = args.iter().any(|a| a == "--aux-targets"); // flag retained for back-compat but now always on
        cascadia_ai::nnue_train::append_mce_samples_v3(
            std::path::Path::new(out_path), &samples,
        ).expect("Failed to write v3 samples");
        println!("Generated {} samples from {} games in {:.1?} (MCV3 format: aux targets + target_wildlife)",
            samples.len(), num_games, start.elapsed());

        // Also write tile-token format alongside MCV3 if requested
        if args.iter().any(|a| a == "--tile-token-out") {
            println!("  NOTE: Use --tile-token-selfplay for dedicated tile-token generation.");
            println!("  The standard --self-play pipeline only saves sparse features, not board states.");
        }
        return;
    } else if run_tile_token_selfplay {
        // Generate self-play games and write tile-token format for transformer/GNN training.
        // This captures full board states (including tile rotations) at each position.
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("tile_tokens.bin");
        let epsilon: f32 = args.iter().position(|a| a == "--epsilon")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.1);

        let net = weights_path.and_then(|p| {
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(p)).ok()
        });
        let strategy = if net.is_some() { "NNUE" } else { "greedy" };
        println!("Generating {} tile-token self-play games ({}, epsilon={}, out={})",
            num_games, strategy, epsilon, out_path);

        let start = Instant::now();
        use rand::SeedableRng;
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0);
        let base_seed: u64 = if args.iter().any(|a| a == "--random-seed") {
            rand::random()
        } else { 42 };

        let num_players = 4;
        // Parallelize across threads: each thread plays its share of games and collects samples.
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(4);
        let games_per_thread = (num_games + num_threads - 1) / num_threads;
        println!("  Using {} threads ({} games/thread)", num_threads, games_per_thread);

        type TTSample = (Vec<cascadia_ai::nnue::TileToken>, cascadia_ai::nnue::GlobalFeatures, f32);
        let net_arc = std::sync::Arc::new(net);

        let handles: Vec<_> = (0..num_threads).map(|t| {
            let n_this = if t < num_threads - 1 {
                games_per_thread.min(num_games.saturating_sub(t * games_per_thread))
            } else {
                num_games.saturating_sub(t * games_per_thread)
            };
            let thread_seed = base_seed
                .wrapping_add(seed_offset)
                .wrapping_add((t as u64) * 1_000_000);
            let net = std::sync::Arc::clone(&net_arc);
            std::thread::spawn(move || {
                let mut results: Vec<TTSample> = Vec::new();
                let mut total_score: u64 = 0;
                for game_idx in 0..n_this {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(thread_seed.wrapping_add(game_idx as u64));
                    let cards = cascadia_core::types::ScoringCards::all_a();
                    let mut game = cascadia_core::game::GameState::new(num_players, cards, &mut rng);

                    let mut snapshots: Vec<(cascadia_core::board::Board, cascadia_ai::nnue::BagInfo)> = Vec::new();
                    while !game.is_game_over() {
                        if game.current_player == 0 {
                            if game.can_replace_overflow().is_some() {
                                game.replace_overflow();
                            }
                            let bag = cascadia_ai::nnue::BagInfo::from_game(&game);
                            snapshots.push((game.boards[0].clone(), bag));

                            let mv = if let Some(ref n) = *net {
                                let cands = cascadia_ai::search::candidate_moves_decomposed(&game, n);
                                if cands.is_empty() {
                                    cascadia_ai::eval::best_move_with_potential(
                                        &mut game.boards[0].clone(),
                                        &game.market.available().map(|(i, p)| (i, p.tile, p.wildlife)).collect::<Vec<_>>(),
                                        &game.scoring_cards, game.turns_remaining,
                                    )
                                } else if rng.gen::<f32>() < epsilon {
                                    use rand::seq::SliceRandom;
                                    Some(*cands.choose(&mut rng).unwrap())
                                } else {
                                    Some(cands[0])
                                }
                            } else {
                                cascadia_ai::search::greedy_move(&game)
                            };
                            if let Some(mv) = mv {
                                if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; }
                            } else { break; }
                        } else {
                            if game.can_replace_overflow().is_some() {
                                game.replace_overflow();
                            }
                            match cascadia_ai::search::greedy_move(&game) {
                                Some(mv) => { if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; } }
                                None => break,
                            }
                        }
                    }

                    let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut game.boards[0].clone(), &game.scoring_cards,
                    ).total;
                    total_score += final_score as u64;

                    for (board, bag) in &snapshots {
                        let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                            &mut board.clone(), &game.scoring_cards,
                        ).total;
                        let delta = (final_score as f32) - (current_score as f32);
                        let (tokens, global) = cascadia_ai::nnue::extract_tile_tokens(board, Some(bag));
                        results.push((tokens, global, delta));
                    }
                }
                (results, total_score, n_this)
            })
        }).collect();

        let mut all_samples: Vec<TTSample> = Vec::new();
        let mut total_score = 0u64;
        for h in handles {
            let (results, ts, _n) = h.join().unwrap();
            all_samples.extend(results);
            total_score += ts;
        }

        cascadia_ai::nnue::write_tile_token_samples(out_path, &all_samples)
            .expect("Failed to write tile-token samples");
        let avg = total_score as f64 / num_games as f64;
        println!("Generated {} tile-token samples from {} games in {:.1?} (avg score {:.1})",
            all_samples.len(), num_games, start.elapsed(), avg);
        return;
    } else if run_rich_tile_token_selfplay {
        // Generate self-play games and write RICH tile-token format (TIL2) for transformer training.
        // Each tile carries per-cell adjacency info (6 dirs × wildlife + terrain) — NNUE-v3-level
        // richness per tile, encoded as a sparse one-hot bag of features inside the transformer.
        let weights_path = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let out_path = args.iter().position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("rich_tile_tokens.bin");
        let epsilon: f32 = args.iter().position(|a| a == "--epsilon")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.1);

        let net = weights_path.and_then(|p| {
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(p)).ok()
        });
        let strategy = if net.is_some() { "NNUE" } else { "greedy" };
        println!("Generating {} RICH tile-token self-play games ({}, epsilon={}, out={})",
            num_games, strategy, epsilon, out_path);

        let start = Instant::now();
        use rand::SeedableRng;
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0);
        let base_seed: u64 = if args.iter().any(|a| a == "--random-seed") {
            rand::random()
        } else { 42 };

        let num_players = 4;
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(4);
        let games_per_thread = (num_games + num_threads - 1) / num_threads;
        println!("  Using {} threads ({} games/thread)", num_threads, games_per_thread);

        type RichSample = (Vec<cascadia_ai::nnue::RichTileToken>, cascadia_ai::nnue::GlobalFeatures, f32);
        let net_arc = std::sync::Arc::new(net);

        let handles: Vec<_> = (0..num_threads).map(|t| {
            let n_this = if t < num_threads - 1 {
                games_per_thread.min(num_games.saturating_sub(t * games_per_thread))
            } else {
                num_games.saturating_sub(t * games_per_thread)
            };
            let thread_seed = base_seed
                .wrapping_add(seed_offset)
                .wrapping_add((t as u64) * 1_000_000);
            let net = std::sync::Arc::clone(&net_arc);
            std::thread::spawn(move || {
                let mut results: Vec<RichSample> = Vec::new();
                let mut total_score: u64 = 0;
                for game_idx in 0..n_this {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(thread_seed.wrapping_add(game_idx as u64));
                    let cards = cascadia_core::types::ScoringCards::all_a();
                    let mut game = cascadia_core::game::GameState::new(num_players, cards, &mut rng);

                    let mut snapshots: Vec<(cascadia_core::board::Board, cascadia_ai::nnue::BagInfo)> = Vec::new();
                    while !game.is_game_over() {
                        if game.current_player == 0 {
                            if game.can_replace_overflow().is_some() {
                                game.replace_overflow();
                            }
                            let bag = cascadia_ai::nnue::BagInfo::from_game(&game);
                            snapshots.push((game.boards[0].clone(), bag));

                            let mv = if let Some(ref n) = *net {
                                let cands = cascadia_ai::search::candidate_moves_decomposed(&game, n);
                                if cands.is_empty() {
                                    cascadia_ai::eval::best_move_with_potential(
                                        &mut game.boards[0].clone(),
                                        &game.market.available().map(|(i, p)| (i, p.tile, p.wildlife)).collect::<Vec<_>>(),
                                        &game.scoring_cards, game.turns_remaining,
                                    )
                                } else if rng.gen::<f32>() < epsilon {
                                    use rand::seq::SliceRandom;
                                    Some(*cands.choose(&mut rng).unwrap())
                                } else {
                                    Some(cands[0])
                                }
                            } else {
                                cascadia_ai::search::greedy_move(&game)
                            };
                            if let Some(mv) = mv {
                                if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; }
                            } else { break; }
                        } else {
                            if game.can_replace_overflow().is_some() {
                                game.replace_overflow();
                            }
                            match cascadia_ai::search::greedy_move(&game) {
                                Some(mv) => { if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; } }
                                None => break,
                            }
                        }
                    }

                    let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut game.boards[0].clone(), &game.scoring_cards,
                    ).total;
                    total_score += final_score as u64;

                    for (board, bag) in &snapshots {
                        let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                            &mut board.clone(), &game.scoring_cards,
                        ).total;
                        let delta = (final_score as f32) - (current_score as f32);
                        let (tokens, global) = cascadia_ai::nnue::extract_rich_tile_tokens(board, Some(bag));
                        results.push((tokens, global, delta));
                    }
                }
                (results, total_score, n_this)
            })
        }).collect();

        let mut all_samples: Vec<RichSample> = Vec::new();
        let mut total_score = 0u64;
        for h in handles {
            let (results, ts, _n) = h.join().unwrap();
            all_samples.extend(results);
            total_score += ts;
        }

        cascadia_ai::nnue::write_rich_tile_token_samples(out_path, &all_samples)
            .expect("Failed to write rich tile-token samples");
        let avg = total_score as f64 / num_games as f64;
        println!("Generated {} rich tile-token samples from {} games in {:.1?} (avg score {:.1})",
            all_samples.len(), num_games, start.elapsed(), avg);
        return;
    } else if run_external_eval {
        // Play N games where player 0 uses an external evaluator (via stdin/stdout).
        //
        // Protocol (all messages framed as: u8 type + u32 LE length + payload):
        //   Rust → Python:
        //     0x01 EVAL:  u8 num_candidates,
        //                 for each: u8 num_tiles,
        //                           (11 * num_tiles) tile bytes,
        //                           45 global bytes,
        //                           f32 current_score (LE)
        //     0x02 DONE:  u16 final_score (LE)
        //     0x03 FINAL: empty
        //   Python → Rust:
        //     0x10 PICK:  u8 chosen_idx
        //
        // All logging goes to stderr. stdout is protocol-only.
        use std::io::{Read, Write};

        let num_players = 4;
        use rand::SeedableRng;
        let base_seed: u64 = if args.iter().any(|a| a == "--random-seed") {
            rand::random()
        } else { 42 };
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0);

        let stdout = std::io::stdout();
        let stdin = std::io::stdin();
        let mut stdout = stdout.lock();
        let mut stdin = stdin.lock();

        eprintln!("[external-eval] Playing {} games with external evaluator", num_games);

        let mut all_scores: Vec<u16> = Vec::with_capacity(num_games);
        let eval_start = Instant::now();

        for game_idx in 0..num_games {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                base_seed.wrapping_add(seed_offset + game_idx as u64));
            let cards = cascadia_core::types::ScoringCards::all_a();
            let mut game = cascadia_core::game::GameState::new(num_players, cards, &mut rng);

            while !game.is_game_over() {
                if game.current_player != 0 {
                    if game.can_replace_overflow().is_some() {
                        game.replace_overflow();
                    }
                    match cascadia_ai::search::greedy_move(&game) {
                        Some(mv) => { if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; } }
                        None => break,
                    }
                    continue;
                }

                // Player 0's turn
                if game.can_replace_overflow().is_some() {
                    game.replace_overflow();
                }

                // Generate candidates (use a lightweight enumeration — all decomposed
                // candidate moves, same as in self-play when no NNUE is loaded).
                // We reuse candidate_moves_decomposed with a dummy net if available.
                let candidates: Vec<cascadia_ai::eval::ScoredMove> = {
                    let mp: Vec<_> = game.market.available()
                        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
                    if mp.is_empty() {
                        Vec::new()
                    } else {
                        let mut c = Vec::new();
                        // Simple: for each market combo (single-slot draft), get the
                        // best move for that combo via existing eval. Also try all
                        // independent drafts if nature tokens are available.
                        let cards = game.scoring_cards;
                        let turns = game.turns_remaining;
                        let board = &game.boards[0];
                        for &(idx, tile, wl) in &mp {
                            let restricted = vec![(idx, tile, wl)];
                            let mut b = board.clone();
                            if let Some(mv) = cascadia_ai::eval::best_move_with_potential(
                                &mut b, &restricted, &cards, turns) {
                                c.push(mv);
                            }
                        }
                        if board.nature_tokens > 0 && mp.len() >= 2 {
                            for &(ti, tile, _) in &mp {
                                for &(wi, _, wl) in &mp {
                                    if ti == wi { continue; }
                                    let restricted = vec![(ti, tile, wl)];
                                    let mut b = board.clone();
                                    if let Some(mut mv) = cascadia_ai::eval::best_move_with_potential(
                                        &mut b, &restricted, &cards, turns) {
                                        mv.wildlife_market_index = Some(wi);
                                        c.push(mv);
                                    }
                                }
                            }
                        }
                        c
                    }
                };

                if candidates.is_empty() {
                    break;
                }

                // Build tile tokens for each candidate's afterstate + current_score
                let mut payload: Vec<u8> = Vec::new();
                payload.push(candidates.len() as u8);
                for mv in &candidates {
                    let mut g = game.clone();
                    if !cascadia_ai::search::execute_scored_move(&mut g, mv) {
                        // shouldn't happen, but fill with minimal data
                        payload.push(0);
                        payload.extend_from_slice(&[0u8; 45]);
                        payload.extend_from_slice(&0f32.to_le_bytes());
                        continue;
                    }
                    let bag = cascadia_ai::nnue::BagInfo::from_game(&g);
                    let (tokens, global) = cascadia_ai::nnue::extract_tile_tokens(&g.boards[0], Some(&bag));
                    let current_score = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut g.boards[0].clone(), &g.scoring_cards,
                    ).total as f32;

                    payload.push(tokens.len() as u8);
                    for t in &tokens {
                        payload.extend_from_slice(&t.terrain_triangles);
                        payload.push(t.wildlife);
                        payload.push(t.allowed_mask);
                        let flags = (t.keystone as u8) | ((t.has_wildlife as u8) << 1);
                        payload.push(flags);
                        payload.push(t.q as u8);
                        payload.push(t.r as u8);
                    }
                    // Globals (45 bytes)
                    payload.push(global.turn);
                    payload.push(global.nature_tokens);
                    payload.extend_from_slice(&global.wildlife_counts);
                    payload.extend_from_slice(&global.largest_habitat);
                    payload.extend_from_slice(&global.bag_remaining);
                    payload.extend_from_slice(&global.opp_habitat);
                    payload.extend_from_slice(&global.market_terrain1);
                    payload.extend_from_slice(&global.market_terrain2);
                    payload.extend_from_slice(&global.market_wildlife);
                    payload.extend_from_slice(&global.tbag_terrain);
                    payload.extend_from_slice(&global.tbag_wildlife);
                    payload.push(global.overflow_used as u8);
                    // current_score as f32 LE
                    payload.extend_from_slice(&current_score.to_le_bytes());
                }

                // Write EVAL frame
                stdout.write_all(&[0x01u8]).expect("stdout write");
                stdout.write_all(&(payload.len() as u32).to_le_bytes()).expect("stdout write");
                stdout.write_all(&payload).expect("stdout write");
                stdout.flush().expect("stdout flush");

                // Read PICK
                let mut header = [0u8; 5]; // type(1) + length(4)
                stdin.read_exact(&mut header).expect("stdin read");
                if header[0] != 0x10 {
                    eprintln!("[external-eval] FATAL: expected PICK (0x10), got {:#x}", header[0]);
                    std::process::exit(1);
                }
                let pick_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
                let mut pick_data = vec![0u8; pick_len];
                stdin.read_exact(&mut pick_data).expect("stdin read pick");
                let chosen_idx = pick_data[0] as usize;
                if chosen_idx >= candidates.len() {
                    eprintln!("[external-eval] FATAL: chosen index {} >= num candidates {}", chosen_idx, candidates.len());
                    std::process::exit(1);
                }

                let chosen = candidates[chosen_idx];
                if !cascadia_ai::search::execute_scored_move(&mut game, &chosen) { break; }
            }

            let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0].clone(), &game.scoring_cards,
            ).total;
            all_scores.push(final_score);

            // Write DONE frame
            stdout.write_all(&[0x02u8]).expect("stdout write");
            stdout.write_all(&(2u32).to_le_bytes()).expect("stdout write");
            stdout.write_all(&final_score.to_le_bytes()).expect("stdout write");
            stdout.flush().expect("stdout flush");

            eprintln!("[external-eval] Game {}/{}: {} pts ({:.1}s total)",
                game_idx + 1, num_games, final_score, eval_start.elapsed().as_secs_f64());
        }

        // FINAL
        stdout.write_all(&[0x03u8]).expect("stdout write");
        stdout.write_all(&(0u32).to_le_bytes()).expect("stdout write");
        stdout.flush().expect("stdout flush");

        let mean = all_scores.iter().map(|&s| s as f64).sum::<f64>() / all_scores.len() as f64;
        eprintln!("[external-eval] DONE: mean {:.1} across {} games", mean, all_scores.len());
        return;
    } else if run_gnn_mce_bench {
        // GNN MCE benchmark: at each player-0 decision, do N rollouts per candidate
        // (greedy policy for all players during rollout), collect leaf tile-tokens,
        // send to Python for batched GNN evaluation. Python responds with chosen_idx.
        //
        // Protocol (framed as: u8 type + u32 LE length + payload):
        //   Rust → Python:
        //     0x04 MCE_EVAL:
        //         u8 num_candidates
        //         u16 LE rollouts_per_candidate (R)
        //         for each of (num_candidates * R):
        //             u8 num_tiles,
        //             (11 * num_tiles) tile bytes,
        //             45 global bytes,
        //             f32 LE leaf_current_score
        //     0x02 DONE:  u16 final_score
        //     0x03 FINAL: empty
        //   Python → Rust:
        //     0x10 PICK:  u8 chosen_idx
        use std::io::{Read, Write};
        use rand::SeedableRng;

        let num_players = 4;
        let rollouts: usize = args.iter().position(|a| a == "--rollouts")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(50);
        let depth: usize = args.iter().position(|a| a == "--depth")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(6);
        let base_seed: u64 = if args.iter().any(|a| a == "--random-seed") {
            rand::random()
        } else { 42 };
        let seed_offset: u64 = std::env::var("CASCADIA_SEED_OFFSET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0);

        let stdout = std::io::stdout();
        let stdin = std::io::stdin();
        let mut stdout = stdout.lock();
        let mut stdin = stdin.lock();

        eprintln!("[gnn-mce-bench] {} games, rollouts={}, depth={}",
            num_games, rollouts, depth);

        let mut all_scores: Vec<u16> = Vec::with_capacity(num_games);
        let bench_start = Instant::now();

        for game_idx in 0..num_games {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                base_seed.wrapping_add(seed_offset + game_idx as u64));
            let cards = cascadia_core::types::ScoringCards::all_a();
            let mut game = cascadia_core::game::GameState::new(num_players, cards, &mut rng);

            while !game.is_game_over() {
                if game.current_player != 0 {
                    if game.can_replace_overflow().is_some() {
                        game.replace_overflow();
                    }
                    match cascadia_ai::search::greedy_move(&game) {
                        Some(mv) => { if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; } }
                        None => break,
                    }
                    continue;
                }

                // Player 0 turn
                if game.can_replace_overflow().is_some() {
                    game.replace_overflow();
                }

                // Generate candidate moves (same logic as external-eval)
                let candidates: Vec<cascadia_ai::eval::ScoredMove> = {
                    let mp: Vec<_> = game.market.available()
                        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
                    if mp.is_empty() {
                        Vec::new()
                    } else {
                        let mut c = Vec::new();
                        let cards = game.scoring_cards;
                        let turns = game.turns_remaining;
                        let board = &game.boards[0];
                        for &(idx, tile, wl) in &mp {
                            let restricted = vec![(idx, tile, wl)];
                            let mut b = board.clone();
                            if let Some(mv) = cascadia_ai::eval::best_move_with_potential(
                                &mut b, &restricted, &cards, turns) {
                                c.push(mv);
                            }
                        }
                        if board.nature_tokens > 0 && mp.len() >= 2 {
                            for &(ti, tile, _) in &mp {
                                for &(wi, _, wl) in &mp {
                                    if ti == wi { continue; }
                                    let restricted = vec![(ti, tile, wl)];
                                    let mut b = board.clone();
                                    if let Some(mut mv) = cascadia_ai::eval::best_move_with_potential(
                                        &mut b, &restricted, &cards, turns) {
                                        mv.wildlife_market_index = Some(wi);
                                        c.push(mv);
                                    }
                                }
                            }
                        }
                        c
                    }
                };

                if candidates.is_empty() {
                    break;
                }

                // Do rollouts in parallel: for each candidate × R rollouts, play
                // greedy for both sides from the afterstate for `depth` AI turns,
                // then capture the leaf (board, bag, current_score).
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get()).unwrap_or(8);
                type Leaf = (Vec<cascadia_ai::nnue::TileToken>, cascadia_ai::nnue::GlobalFeatures, f32);
                let mut work_items: Vec<(usize, u64)> = Vec::new();
                for (ci, _mv) in candidates.iter().enumerate() {
                    for _ in 0..rollouts {
                        work_items.push((ci, rng.gen()));
                    }
                }
                let chunk_size = ((work_items.len() + num_threads - 1) / num_threads).max(1);
                let game_arc = std::sync::Arc::new(game.clone());
                let candidates_arc = std::sync::Arc::new(candidates.clone());

                let handles: Vec<_> = work_items.chunks(chunk_size).map(|chunk| {
                    let work = chunk.to_vec();
                    let g = std::sync::Arc::clone(&game_arc);
                    let c = std::sync::Arc::clone(&candidates_arc);
                    std::thread::spawn(move || {
                        let mut results: Vec<(usize, Leaf)> = Vec::with_capacity(work.len());
                        for &(ci, seed) in &work {
                            let mut gs = (*g).clone();
                            let mut rr = rand::rngs::StdRng::seed_from_u64(seed);
                            gs.shuffle_bags(&mut rr);
                            if !cascadia_ai::search::execute_scored_move(&mut gs, &c[ci]) {
                                continue;
                            }
                            let mut ai_turns = 0usize;
                            while !gs.is_game_over() {
                                if gs.current_player != 0 {
                                    if gs.can_replace_overflow().is_some() {
                                        gs.replace_overflow();
                                    }
                                    match cascadia_ai::search::greedy_move(&gs) {
                                        Some(mv) => { if !cascadia_ai::search::execute_scored_move(&mut gs, &mv) { break; } }
                                        None => break,
                                    }
                                    continue;
                                }
                                ai_turns += 1;
                                if ai_turns > depth { break; }
                                if gs.can_replace_overflow().is_some() {
                                    gs.replace_overflow();
                                }
                                match cascadia_ai::search::greedy_move(&gs) {
                                    Some(mv) => { if !cascadia_ai::search::execute_scored_move(&mut gs, &mv) { break; } }
                                    None => break,
                                }
                            }
                            // Capture leaf
                            let bag = cascadia_ai::nnue::BagInfo::from_game(&gs);
                            let (tokens, global) = cascadia_ai::nnue::extract_tile_tokens(&gs.boards[0], Some(&bag));
                            let leaf_score = cascadia_core::scoring::ScoreBreakdown::compute(
                                &mut gs.boards[0].clone(), &gs.scoring_cards,
                            ).total as f32;
                            results.push((ci, (tokens, global, leaf_score)));
                        }
                        results
                    })
                }).collect();

                // Collect per-candidate leaves in order
                let mut leaves_by_ci: Vec<Vec<Leaf>> = (0..candidates.len())
                    .map(|_| Vec::with_capacity(rollouts)).collect();
                for h in handles {
                    for (ci, leaf) in h.join().unwrap() {
                        leaves_by_ci[ci].push(leaf);
                    }
                }

                // Serialize MCE_EVAL payload
                let mut payload: Vec<u8> = Vec::new();
                payload.push(candidates.len() as u8);
                payload.extend_from_slice(&(rollouts as u16).to_le_bytes());
                for ci in 0..candidates.len() {
                    for r in 0..rollouts {
                        if r < leaves_by_ci[ci].len() {
                            let (tokens, global, leaf_score) = &leaves_by_ci[ci][r];
                            payload.push(tokens.len() as u8);
                            for t in tokens {
                                payload.extend_from_slice(&t.terrain_triangles);
                                payload.push(t.wildlife);
                                payload.push(t.allowed_mask);
                                let flags = (t.keystone as u8) | ((t.has_wildlife as u8) << 1);
                                payload.push(flags);
                                payload.push(t.q as u8);
                                payload.push(t.r as u8);
                            }
                            payload.push(global.turn);
                            payload.push(global.nature_tokens);
                            payload.extend_from_slice(&global.wildlife_counts);
                            payload.extend_from_slice(&global.largest_habitat);
                            payload.extend_from_slice(&global.bag_remaining);
                            payload.extend_from_slice(&global.opp_habitat);
                            payload.extend_from_slice(&global.market_terrain1);
                            payload.extend_from_slice(&global.market_terrain2);
                            payload.extend_from_slice(&global.market_wildlife);
                            payload.extend_from_slice(&global.tbag_terrain);
                            payload.extend_from_slice(&global.tbag_wildlife);
                            payload.push(global.overflow_used as u8);
                            payload.extend_from_slice(&leaf_score.to_le_bytes());
                        } else {
                            // Rollout failed: emit empty-ish leaf (0 tiles, score=0)
                            payload.push(0);
                            payload.extend_from_slice(&[0u8; 45]);
                            payload.extend_from_slice(&0f32.to_le_bytes());
                        }
                    }
                }

                stdout.write_all(&[0x04u8]).expect("stdout write");
                stdout.write_all(&(payload.len() as u32).to_le_bytes()).expect("stdout write");
                stdout.write_all(&payload).expect("stdout write");
                stdout.flush().expect("stdout flush");

                let mut header = [0u8; 5];
                stdin.read_exact(&mut header).expect("stdin read");
                if header[0] != 0x10 {
                    eprintln!("[gnn-mce-bench] FATAL: expected PICK, got {:#x}", header[0]);
                    std::process::exit(1);
                }
                let pick_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
                let mut pick_data = vec![0u8; pick_len];
                stdin.read_exact(&mut pick_data).expect("stdin read pick");
                let chosen_idx = pick_data[0] as usize;
                if chosen_idx >= candidates.len() {
                    eprintln!("[gnn-mce-bench] FATAL: chosen idx {} >= num candidates {}", chosen_idx, candidates.len());
                    std::process::exit(1);
                }
                let chosen = candidates[chosen_idx];
                if !cascadia_ai::search::execute_scored_move(&mut game, &chosen) { break; }
            }

            let final_score = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game.boards[0].clone(), &game.scoring_cards,
            ).total;
            all_scores.push(final_score);

            stdout.write_all(&[0x02u8]).expect("stdout write");
            stdout.write_all(&(2u32).to_le_bytes()).expect("stdout write");
            stdout.write_all(&final_score.to_le_bytes()).expect("stdout write");
            stdout.flush().expect("stdout flush");

            eprintln!("[gnn-mce-bench] Game {}/{}: {} pts ({:.1}s total)",
                game_idx + 1, num_games, final_score, bench_start.elapsed().as_secs_f64());
        }

        stdout.write_all(&[0x03u8]).expect("stdout write");
        stdout.write_all(&(0u32).to_le_bytes()).expect("stdout write");
        stdout.flush().expect("stdout flush");

        let mean = all_scores.iter().map(|&s| s as f64).sum::<f64>() / all_scores.len() as f64;
        eprintln!("[gnn-mce-bench] DONE: mean {:.1} across {} games", mean, all_scores.len());
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
            let mut game_samples: Vec<cascadia_ai::nnue_train::Sample> = Vec::new();

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
                // AI turn: collect samples + play MCE move in one pass.
                // Each candidate evaluation is a hypothetical afterstate so aux and
                // target_wildlife are 0 (see note in simulate_game_inner).
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
                        game_samples.push(cascadia_ai::nnue_train::Sample {
                            features,
                            target,
                            aux_bear: 0.0,
                            aux_salmon: 0.0,
                            target_wildlife: 0.0,
                            subscore_targets: [0.0; cascadia_ai::nnue::NUM_HEADS],
                        });
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
            cascadia_ai::nnue_train::append_mce_samples_v3(
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
    } else if run_train_pairwise {
        // Exp #4: train value head with pairwise ranking loss on MCP2 grouped data.
        let epochs: usize = args.iter().position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(10);
        let lr: f32 = args.iter().position(|a| a == "--lr")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.00003);
        let groups_path = args.iter().position(|a| a == "--groups")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("mce_grouped.bin");
        let weights_in = args.iter().position(|a| a == "--init-weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
        let weights_out = args.iter().position(|a| a == "--weights")
            .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
            .unwrap_or("nnue_weights_pairwise.bin");
        let pairwise_weight: f32 = args.iter().position(|a| a == "--pairwise-weight")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0.5);
        let margin: f32 = args.iter().position(|a| a == "--margin")
            .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let mut net = if let Some(path) = weights_in {
            println!("Loading initial weights from {}...", path);
            cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path))
                .expect("Failed to load init weights")
        } else {
            println!("Starting with fresh weights");
            cascadia_ai::nnue::NNUENetwork::new()
        };
        println!("Pairwise training: groups={}, epochs={}, lr={}, alpha={}, margin={}",
                 groups_path, epochs, lr, pairwise_weight, margin);
        let start = Instant::now();
        let stats = cascadia_ai::nnue_train::train_from_mcp2_pairwise(
            &mut net, std::path::Path::new(groups_path), epochs, lr, pairwise_weight, margin,
        ).expect("Pairwise training failed");
        println!("Training complete in {:.1?}", start.elapsed());
        println!("  Total samples: {}", stats.num_samples);
        println!("  Final RMSE:    {:.2}", stats.final_rmse);
        net.save(std::path::Path::new(weights_out)).expect("Failed to save");
        println!("  Weights saved to {}", weights_out);
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
        // --init-weights <path>: load initial weights from this path instead of
        // --weights. Useful when you want input and output to be different files
        // (checkpoint-style training: train iter N from iter N-1, save as iter N).
        let init_weights_path: Option<std::path::PathBuf> = args.iter().position(|a| a == "--init-weights")
            .and_then(|i| args.get(i + 1))
            .map(|s| std::path::PathBuf::from(s));

        let load_path = init_weights_path.as_ref().unwrap_or(&weights_path);
        let mut net = if load_path.exists() {
            println!("Loading NNUE weights from {:?}...", load_path);
            cascadia_ai::nnue::NNUENetwork::load(load_path).unwrap_or_else(|e| {
                eprintln!("Failed to load: {}, starting fresh", e);
                cascadia_ai::nnue::NNUENetwork::new()
            })
        } else {
            println!("Starting with fresh NNUE weights");
            cascadia_ai::nnue::NNUENetwork::new()
        };

        // CASCADIA_TRAIN_SEED overrides the default seed (42). Useful for
        // iterative training where each iter should explore different data.
        let seed: u64 = std::env::var("CASCADIA_TRAIN_SEED").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(42);
        println!("Training NNUE: {} games, {} epochs, lr={}, seed={}, weights={:?}",
            train_games, epochs, lr, seed, weights_path);
        let start = Instant::now();
        let stats = cascadia_ai::nnue_train::train_nnue(&mut net, train_games, epochs, lr, seed);
        let elapsed = start.elapsed();

        println!("Training complete in {:.1?}", elapsed);
        println!("  Samples:    {}", stats.num_samples);
        println!("  Final RMSE: {:.2}", stats.final_rmse);

        if !stats.final_rmse.is_finite() {
            eprintln!("  [ABORTED save: final RMSE non-finite ({}), weights file left untouched]", stats.final_rmse);
            std::process::exit(2);
        }
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
        } else if args.iter().any(|a| a == "--nrpa") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let level: usize = args.iter().position(|a| a == "--level")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2);
            let n: usize = args.iter().position(|a| a == "--n")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(30);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for NRPA");
            // Honor env vars too (NRPA_LEVEL, NRPA_N) — set to override CLI
            std::env::set_var("NRPA_LEVEL", level.to_string());
            std::env::set_var("NRPA_N", n.to_string());
            Strategy::NRPA { net: Arc::new(net), level, n }
        } else if args.iter().any(|a| a == "--ol-mcts") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for OL-MCTS");
            Strategy::OpenLoopMCTS { net: Arc::new(net), rollouts }
        } else if args.iter().any(|a| a == "--gumbel-mcts") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let m: usize = args.iter().position(|a| a == "--m")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(15);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for Gumbel-MCTS");
            Strategy::GumbelMCTS { net: Arc::new(net), rollouts, m }
        } else if args.iter().any(|a| a == "--mce") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for MCE");
            Strategy::MCE { net: Arc::new(net), rollouts }
        } else if args.iter().any(|a| a == "--greedy-mce") {
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let alloc = args.iter().position(|a| a == "--alloc")
                .and_then(|i| args.get(i + 1))
                .map(|s| match s.as_str() {
                    "halving" | "seq-halving" => cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                    "ucb" => cascadia_ai::mce::GreedyMceAlloc::Ucb,
                    "crn" | "uniform-crn" => cascadia_ai::mce::GreedyMceAlloc::UniformCRN,
                    "halving-crn" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCRN,
                    "halving-et" | "halving-early-term" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingEarlyTerm,
                    "halving-ci" | "halving-conf" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCI,
                    "sr" | "successive-rejects" => cascadia_ai::mce::GreedyMceAlloc::SuccessiveRejects,
                    "halving-pw" | "halving-progressive" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingPW,
                    "thompson" | "ts" => cascadia_ai::mce::GreedyMceAlloc::ThompsonSampling,
                    "mcts-pw" | "mcts" => cascadia_ai::mce::GreedyMceAlloc::MctsPW,
                    "halving-hetero" | "hetero" | "ocba" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingHetero,
                    "puct" | "alphazero" => cascadia_ai::mce::GreedyMceAlloc::Puct,
                    _ => cascadia_ai::mce::GreedyMceAlloc::Uniform,
                }).unwrap_or(cascadia_ai::mce::GreedyMceAlloc::Uniform);
            let expanded = args.iter().position(|a| a == "--candidates")
                .and_then(|i| args.get(i + 1))
                .map(|s| s == "expanded").unwrap_or(false);
            Strategy::GreedyMCE { rollouts, alloc, expanded }
        } else if args.iter().any(|a| a == "--uct-mcts") {
            let simulations = args.iter().position(|a| a == "--simulations")
                .or_else(|| args.iter().position(|a| a == "--rollouts"))
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let parallel = args.iter().any(|a| a == "--parallel");
            Strategy::UctMcts { simulations, parallel }
        } else if args.iter().any(|a| a == "--mcts-tree") {
            let simulations = args.iter().position(|a| a == "--simulations")
                .or_else(|| args.iter().position(|a| a == "--rollouts"))
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let parallel = args.iter().any(|a| a == "--parallel");
            // --weights enables NNUE-guided rollouts (optional; without it the
            // tree uses greedy rollouts, matching `uct_mcts.rs`).
            let net = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .and_then(|path| {
                    cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path))
                        .ok().map(|n| Arc::new(n))
                });
            Strategy::MctsTree { simulations, parallel, net }
        } else if args.iter().any(|a| a == "--nnue-rollout-mce") {
            let weights_path = args.iter().position(|a| a == "--weights")
                .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
                .unwrap_or("nnue_weights.bin");
            let rollouts = args.iter().position(|a| a == "--rollouts")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(750);
            let alloc = args.iter().position(|a| a == "--alloc")
                .and_then(|i| args.get(i + 1))
                .map(|s| match s.as_str() {
                    "halving" | "seq-halving" => cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                    "ucb" => cascadia_ai::mce::GreedyMceAlloc::Ucb,
                    "crn" | "uniform-crn" => cascadia_ai::mce::GreedyMceAlloc::UniformCRN,
                    "halving-crn" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCRN,
                    "halving-et" | "halving-early-term" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingEarlyTerm,
                    "halving-ci" | "halving-conf" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingCI,
                    "sr" | "successive-rejects" => cascadia_ai::mce::GreedyMceAlloc::SuccessiveRejects,
                    "halving-pw" | "halving-progressive" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingPW,
                    "thompson" | "ts" => cascadia_ai::mce::GreedyMceAlloc::ThompsonSampling,
                    "mcts-pw" | "mcts" => cascadia_ai::mce::GreedyMceAlloc::MctsPW,
                    "halving-hetero" | "hetero" | "ocba" => cascadia_ai::mce::GreedyMceAlloc::SeqHalvingHetero,
                    "puct" | "alphazero" => cascadia_ai::mce::GreedyMceAlloc::Puct,
                    _ => cascadia_ai::mce::GreedyMceAlloc::Uniform,
                }).unwrap_or(cascadia_ai::mce::GreedyMceAlloc::Uniform);
            let expanded = args.iter().position(|a| a == "--candidates")
                .and_then(|i| args.get(i + 1))
                .map(|s| s == "expanded").unwrap_or(false);
            let prefilter_k: usize = args.iter().position(|a| a == "--prefilter-k")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0);
            let exact_endgame: usize = args.iter().position(|a| a == "--exact-endgame")
                .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0);
            let net = cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(weights_path))
                .expect("Failed to load NNUE weights for nnue-rollout-mce");
            Strategy::NnueRolloutMCE { net: Arc::new(net), rollouts, alloc, expanded, prefilter_k, exact_endgame }
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

// =====================================================================
// Cross-binary daemon mode
//
// Two daemons (one per cascadia-cli binary, each loaded with its own NNUE
// weights) cooperate with a Python coordinator to play a head-to-head game.
// Each daemon maintains its own GameState, kept in sync by replaying the
// same action sequence via APPLY commands.
//
// Protocol (line-based text on stdin/stdout):
//   READY                     — emitted on startup once weights are loaded
//   INIT <seed:u64>           — reset to fresh game with `seed` (both daemons
//                               must use the same seed). Replies: OK <hash>
//   PICK                      — daemon picks a move for the CURRENT player
//                               using mce_wide_v1, applies it (and any pre-move
//                               actions) to ITS state, and returns the action
//                               sequence so the other daemon can replay.
//                               Replies: ACTIONS <a1>;<a2>;...   (or "ACTIONS"
//                               if no moves; or "ERROR ...")
//   APPLY <action>            — apply one action to this daemon's state.
//                               Action grammar:
//                                  O                                 (free overflow replace)
//                                  M                                 (mulligan, spends a token)
//                                  P <market> <q> <r> <rot> <wq> <wr>     (-100 = no wildlife)
//                                  I <tile_m> <wl_m> <q> <r> <rot> <wq> <wr>
//                               Replies: OK <hash>  or  ERROR <msg>
//   HASH                      — sync verify; replies: HASH <u64>
//   GAMEOVER                  — replies: YES or NO
//   CURPLAYER                 — replies: CUR <player_idx>
//   SCORES                    — final scores; replies: SCORES s0 s1 s2 s3
//   BREAKDOWN                 — full per-player breakdown
//   QUIT                      — exit cleanly
// =====================================================================

fn run_daemon(weights_path: &str) {
    use std::io::{BufRead, Write};
    use cascadia_ai::nnue::NNUENetwork;
    use cascadia_ai::eval::ScoredMove;

    let net = match NNUENetwork::load(std::path::Path::new(weights_path)) {
        Ok(n) => Arc::new(n),
        Err(e) => {
            eprintln!("DAEMON_ERROR failed to load weights {}: {}", weights_path, e);
            std::process::exit(1);
        }
    };
    eprintln!("DAEMON_LOADED weights={}", weights_path);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut game: Option<GameState> = None;
    let mut search_rng = StdRng::seed_from_u64(0);

    writeln!(out, "READY").ok();
    out.flush().ok();

    let mut buf = String::new();
    let mut handle = stdin.lock();
    loop {
        buf.clear();
        let n = match handle.read_line(&mut buf) {
            Ok(0) => break, // EOF
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 { break; }
        let line = buf.trim_end();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }

        let resp = match parts[0] {
            "INIT" => {
                let seed: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
                let mut init_rng = StdRng::seed_from_u64(seed);
                let cards = ScoringCards::all_a();
                let g = GameState::new(4, cards, &mut init_rng);
                let h = state_hash(&g);
                game = Some(g);
                search_rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                format!("OK {}", h)
            }
            "PICK" => {
                let g = game.as_mut().expect("INIT first");
                if g.is_game_over() {
                    "ERROR game_over".to_string()
                } else {
                    let actions = daemon_pick(g, &net, &mut search_rng);
                    if actions.is_empty() {
                        "ACTIONS".to_string()
                    } else {
                        format!("ACTIONS {}", actions.join(";"))
                    }
                }
            }
            "APPLY" => {
                let g = game.as_mut().expect("INIT first");
                let action_str = parts[1..].join(" ");
                match daemon_apply(g, &action_str) {
                    Ok(()) => format!("OK {}", state_hash(g)),
                    Err(e) => format!("ERROR {}", e),
                }
            }
            "HASH" => {
                let g = game.as_ref().expect("INIT first");
                format!("HASH {}", state_hash(g))
            }
            "GAMEOVER" => {
                let g = game.as_ref().expect("INIT first");
                if g.is_game_over() { "YES".to_string() } else { "NO".to_string() }
            }
            "CURPLAYER" => {
                let g = game.as_ref().expect("INIT first");
                format!("CUR {}", g.current_player)
            }
            "SCORES" => {
                let g = game.as_mut().expect("INIT first");
                let mut scores = Vec::with_capacity(g.num_players);
                for p in 0..g.num_players {
                    let bd = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut g.boards[p], &g.scoring_cards);
                    scores.push(bd.total);
                }
                format!("SCORES {}", scores.iter().map(|s| s.to_string())
                    .collect::<Vec<_>>().join(" "))
            }
            "BREAKDOWN" => {
                let g = game.as_mut().expect("INIT first");
                let n = g.num_players;
                // Compute base breakdowns per player first to avoid repeated borrow conflicts.
                let mut bases = Vec::with_capacity(n);
                for p in 0..n {
                    let bd = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut g.boards[p], &g.scoring_cards);
                    bases.push(bd);
                }
                // Now compute with-bonus for each player.
                let mut lines = Vec::with_capacity(n);
                for p in 0..n {
                    let wbd = cascadia_core::scoring::ScoreBreakdown::compute_with_bonuses(
                        &mut g.boards, &g.scoring_cards, p);
                    let bd = bases[p];
                    let hab: u16 = bd.habitat.iter().sum();
                    let wl: u16 = bd.wildlife.iter().sum();
                    let bonus: u16 = wbd.habitat_bonus.iter().map(|&b| b as u16).sum();
                    lines.push(format!(
                        "p={} base={} bonus={} hab={} wl={} tok={} bear={} elk={} salmon={} hawk={} fox={}",
                        p, bd.total, bonus, hab, wl, bd.nature_tokens,
                        bd.wildlife[0], bd.wildlife[1], bd.wildlife[2],
                        bd.wildlife[3], bd.wildlife[4]
                    ));
                }
                format!("BREAKDOWN {}", lines.join("|"))
            }
            "QUIT" => {
                writeln!(out, "BYE").ok();
                out.flush().ok();
                return;
            }
            _ => format!("ERROR unknown_cmd {}", parts[0]),
        };

        writeln!(out, "{}", resp).ok();
        out.flush().ok();
    }
}

/// Compute a deterministic state hash for sync verification across binaries.
/// Hashes board cell raw words + market positions + bag sizes + game state ints.
fn state_hash(game: &GameState) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();

    game.num_players.hash(&mut h);
    game.current_player.hash(&mut h);
    game.turns_remaining.hash(&mut h);
    game.overflow_used_this_turn.hash(&mut h);

    for board in &game.boards {
        // Hash a compact projection of each cell.
        for cell in board.grid.cells.iter() {
            // Cell is Copy + small; hash its bit-pattern via Debug-derived Hash if available.
            // Otherwise project to u32: presence | terrain bits | wildlife bits.
            let projection = cell_projection(*cell);
            projection.hash(&mut h);
        }
        board.nature_tokens.hash(&mut h);
        for &g in &board.largest_group {
            g.hash(&mut h);
        }
    }

    // Market: hash each pair (tile + wildlife) compactly.
    for (i, slot) in game.market.pairs.iter().enumerate() {
        i.hash(&mut h);
        let proj = market_slot_projection(slot);
        proj.hash(&mut h);
    }

    h.finish()
}

#[inline]
fn cell_projection(cell: cascadia_core::types::Cell) -> u32 {
    // Cell wraps a u16 with all of presence/terrain/wildlife packed in.
    cell.0 as u32
}

#[inline]
fn market_slot_projection(slot: &Option<cascadia_core::market::MarketPair>) -> u32 {
    // Project the slot (tile + wildlife) into a u32.
    // Use Debug + FNV-1a — both daemons compile from identical cascadia-core
    // source, so the Debug format is byte-stable across binaries.
    let s = format!("{:?}", slot);
    let mut h = 2166136261u32;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Run pre-move optimisation + main pick on `game` using `net` (mce_wide_v1).
/// Mutates the game state and returns the sequence of actions taken so the
/// peer daemon can replay them.
fn daemon_pick(
    game: &mut GameState,
    net: &Arc<cascadia_ai::nnue::NNUENetwork>,
    search_rng: &mut StdRng,
) -> Vec<String> {
    use cascadia_ai::eval::ScoredMove;

    let mut actions = Vec::with_capacity(4);
    const MAX_MULLIGANS: usize = 5;
    let mut mulligans_used = 0;

    // Pre-move optimisation loop — mirrors pre_move_optimize() but records actions.
    loop {
        let analysis = cascadia_ai::mce::analyze_mulligan_fast(game, net);

        if game.can_replace_overflow().is_some() {
            let mut test = game.clone();
            test.replace_overflow();
            let post = cascadia_ai::mce::analyze_mulligan_fast(&test, net);
            if post.current_best > analysis.current_best {
                game.replace_overflow();
                actions.push("O".to_string());
                continue;
            }
        }

        if mulligans_used < MAX_MULLIGANS && analysis.should_mulligan {
            if game.mulligan_wildlife() {
                mulligans_used += 1;
                actions.push("M".to_string());
                continue;
            }
        }
        if mulligans_used < MAX_MULLIGANS && analysis.should_mulligan_pinecone {
            if game.mulligan_wildlife() {
                mulligans_used += 1;
                actions.push("M".to_string());
                continue;
            }
        }
        break;
    }

    // Main move via mce_wide_v1.
    let mv: Option<ScoredMove> = pick_with_env(
        &[("MCE_LMR", "1"), ("MCE_DIVERSE_PREFILTER", "1")],
        || {
            let mut cands = cascadia_ai::mce::expanded_candidates(game);
            if cands.len() > 32 {
                cands = cascadia_ai::mce::nnue_prefilter_candidates(game, net, cands, 32);
            }
            cascadia_ai::mce::best_move_nnue_rollout_mce(
                game, net, 600,
                cascadia_ai::mce::GreedyMceAlloc::SeqHalving,
                cands, search_rng,
            )
        },
    );

    if let Some(mv) = mv {
        let wq = mv.wildlife_q.map(|x| x as i32).unwrap_or(-100);
        let wr = mv.wildlife_r.map(|x| x as i32).unwrap_or(-100);
        let action_str = if let Some(wmi) = mv.wildlife_market_index {
            format!("I {} {} {} {} {} {} {}",
                mv.market_index, wmi, mv.tile_q, mv.tile_r, mv.rotation, wq, wr)
        } else {
            format!("P {} {} {} {} {} {}",
                mv.market_index, mv.tile_q, mv.tile_r, mv.rotation, wq, wr)
        };
        if cascadia_ai::search::execute_scored_move(game, &mv) {
            actions.push(action_str);
        }
    }

    actions
}

fn daemon_apply(game: &mut GameState, action: &str) -> Result<(), String> {
    use cascadia_core::game::PlayerMove;
    use cascadia_core::hex::HexCoord;

    let parts: Vec<&str> = action.split_whitespace().collect();
    if parts.is_empty() {
        return Err("empty_action".to_string());
    }
    match parts[0] {
        "O" => {
            if !game.replace_overflow() {
                return Err("overflow_failed".to_string());
            }
            Ok(())
        }
        "M" => {
            if !game.mulligan_wildlife() {
                return Err("mulligan_failed".to_string());
            }
            Ok(())
        }
        "P" => {
            // P market q r rot wq wr
            if parts.len() != 7 {
                return Err(format!("P_bad_arity {}", parts.len()));
            }
            let market: usize = parts[1].parse().map_err(|_| "P_market".to_string())?;
            let q: i8 = parts[2].parse().map_err(|_| "P_q".to_string())?;
            let r: i8 = parts[3].parse().map_err(|_| "P_r".to_string())?;
            let rot: u8 = parts[4].parse().map_err(|_| "P_rot".to_string())?;
            let wq: i32 = parts[5].parse().map_err(|_| "P_wq".to_string())?;
            let wr: i32 = parts[6].parse().map_err(|_| "P_wr".to_string())?;
            let placement = if wq == -100 || wr == -100 {
                None
            } else {
                HexCoord::new(wq as i8, wr as i8).to_index()
            };
            let pm = PlayerMove {
                market_index: market,
                tile_coord: HexCoord::new(q, r),
                rotation: rot,
                wildlife_placement: placement,
            };
            if !game.execute_move(pm) {
                return Err("P_execute_failed".to_string());
            }
            Ok(())
        }
        "I" => {
            // I tile_m wl_m q r rot wq wr
            if parts.len() != 8 {
                return Err(format!("I_bad_arity {}", parts.len()));
            }
            let tile_m: usize = parts[1].parse().map_err(|_| "I_tm".to_string())?;
            let wl_m: usize = parts[2].parse().map_err(|_| "I_wm".to_string())?;
            let q: i8 = parts[3].parse().map_err(|_| "I_q".to_string())?;
            let r: i8 = parts[4].parse().map_err(|_| "I_r".to_string())?;
            let rot: u8 = parts[5].parse().map_err(|_| "I_rot".to_string())?;
            let wq: i32 = parts[6].parse().map_err(|_| "I_wq".to_string())?;
            let wr: i32 = parts[7].parse().map_err(|_| "I_wr".to_string())?;
            let placement = if wq == -100 || wr == -100 {
                None
            } else {
                HexCoord::new(wq as i8, wr as i8).to_index()
            };
            if !game.execute_independent_move(tile_m, wl_m, HexCoord::new(q, r), rot, placement) {
                return Err("I_execute_failed".to_string());
            }
            Ok(())
        }
        other => Err(format!("unknown_action {}", other)),
    }
}
