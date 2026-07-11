//! Gumbel AlphaZero-style root search with neural-network leaf values.
//!
//! Replaces the flat one-ply greedy-rollout teacher. Key properties:
//!
//! - Root candidates come from the FULL legal action set (or a model-side cap),
//!   never a greedy-ranked truncation, so non-greedy plans stay reachable.
//! - Root action selection is Gumbel top-m + sequential halving over model
//!   policy logits, with completed-Q values from determinized simulations.
//! - Optional market refresh is a public-state decision valued over sampled
//!   hidden orders. Only after acceptance is fixed does search reveal the real
//!   replacement market and choose a draft.
//! - Every simulation then runs on a hidden-redeterminized clone before its
//!   root draft, so future refills never expose the true tile/bag order.
//! - Interior plies advance every seat by argmax of its own derived final Q
//!   (`exact_afterstate_score_active + predicted score-to-go`), which is max^n
//!   play under the model's value estimates.
//! - Leaf value for the root seat is `w * max-Q bootstrap + (1-w) * sampled
//!   greedy terminal rollout`; `w` ramps toward 1.0 as the value model earns
//!   trust. Simulations advance in lockstep so every ply of every live
//!   simulation lands in one batched model evaluation.

use anyhow::{Context, Result, bail};
use cascadia_game::{GameSeed, GameState, MarketPrelude, score_game};
use cascadia_sim::rank_greedy_actions;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::{CandidateAfterstate, candidate_afterstates, complete_with_sampled_greedy};

const GUMBEL_NOISE_SALT: u64 = 0x6a3b_1e55_9d2c_4f01;
const DETERMINIZATION_STREAM_SALT: u64 = 0x0d5e_ed12_37ab_44c9;
const ROLLOUT_STREAM_SALT: u64 = 0x77c0_ffee_4bad_5eed;
const MARKET_DECISION_STREAM_SALT: u64 = 0x3f62_9a17_c4d8_05be;

/// Normalization applied to the Q values entering `sigma` (both the
/// sequential-halving comparisons and the improved-policy logits). Every
/// scheme is (weakly) monotone in Q, which is all the Gumbel policy-
/// improvement argument requires.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SigmaNormalization {
    /// `(q - min) / (max - min)` over the compared set (incumbent default).
    /// One terrible candidate stretches the denominator and compresses the
    /// contenders' gaps.
    MinMax,
    /// `(q - min) / scale` with a fixed score-point scale: sigma differences
    /// track raw point differences regardless of menu spread.
    FixedScale(f64),
    /// `(q - mean) / std` over the compared set.
    ZScore,
    /// Min-max over the top-k values only; everything below the top-k window
    /// clamps to 0. Immune to menu-tail outliers by construction.
    TopKRange(usize),
}

#[derive(Debug, Clone)]
pub struct GumbelConfig {
    /// Total simulation budget across all root actions.
    pub n_simulations: usize,
    /// Number of root actions sampled without replacement via Gumbel top-m.
    pub top_m: usize,
    /// Optional cap on root candidates (model-ranked by prior after the root
    /// eval, NOT greedy-ranked). `None` keeps the full legal set.
    pub max_root_actions: Option<usize>,
    /// How many times the root seat re-enters before a simulation resolves to
    /// a leaf value. 1 = apply root action, walk opponent plies, value the
    /// state where the root seat moves again.
    pub depth_rounds: usize,
    /// Distinct hidden-order determinizations cycled across each action's
    /// simulations (common random numbers across actions).
    pub determinization_samples: usize,
    /// Hidden-order samples used only for the optional three-of-a-kind market
    /// decision. Kept separate from search worlds so high-d serving does not
    /// multiply every eligible root by the full search determinization count.
    pub market_decision_samples: usize,
    /// Use exact afterstate scores instead of the evaluator/search when the
    /// active player has this many personal turns remaining. The implemented
    /// exact frontier is 1: after that move, the player's own final score is
    /// fully determined even though other seats may still act.
    pub exact_endgame_turns: usize,
    /// Leaf value = w * value bootstrap + (1-w) * sampled greedy terminal
    /// rollout. w=1.0 disables CPU rollouts entirely.
    pub rollout_blend_weight: f64,
    /// Resolve independent terminal greedy rollouts through the global Rayon
    /// pool after each batched model step. This is a pure execution change:
    /// every simulation retains its own deterministic RNG stream and results
    /// are committed in simulation order.
    pub parallel_leaf_rollouts: bool,
    /// Sampled-greedy rollout parameters for the (1-w) branch.
    pub rollout_max_actions: usize,
    pub rollout_top_k: usize,
    /// Interior (non-root) ply menu cap.
    pub k_interior: usize,
    /// Gumbel exploration noise at the root (self-play data generation on,
    /// deterministic evaluation off).
    pub exploration: bool,
    /// sigma(q) = (c_visit + max_visits) * c_scale * normalize(q).
    pub c_visit: f64,
    pub c_scale: f64,
    /// Normalization scheme inside sigma. Default `MinMax` is the incumbent
    /// behavior.
    pub sigma_normalization: SigmaNormalization,
    /// Share leaf-rollout RNG streams across root actions at equal visit
    /// index (common random numbers). The incumbent seeds rollouts with the
    /// action index, adding independent rollout noise to exactly the
    /// pairwise comparisons sequential halving makes; pairing converts that
    /// into common-mode noise that cancels in comparisons. Paired streams
    /// derive from the determinization stream, so pinned-seed paired runs
    /// share rollout randomness across arms as well as across actions.
    pub paired_rollouts: bool,
    pub search_seed: u64,
    /// Hidden-determinization stream seed. `None` derives it from
    /// `search_seed`; pin it for common-random-number paired comparisons.
    pub determinization_seed: Option<u64>,
    /// ORACLE MODE: simulate on the true hidden state instead of
    /// redeterminizing. Leaks hidden information — only valid for
    /// measuring the information ceiling, never for honest gates or
    /// training labels.
    pub peek_true_hidden: bool,
    /// Value simulations by the TABLE TOTAL (sum of all four seats' final
    /// scores) instead of the root seat's own score. Gate-aligned
    /// cooperative objective: the formal gate averages every seat of our
    /// own self-play, so denial moves that buy own-seat rank by burning
    /// other seats' points lower the gate metric. Interior plies remain
    /// selfish argmax (an approximation of the other searchers).
    pub table_total: bool,
    /// The model's q head already predicts TABLE-scale score-to-go (a
    /// cycle trained on table-total selfplay labels). Search then values
    /// simulations by the table (terminals, rollouts) like `table_total`,
    /// but derived Q needs no value-vector shift — and interior plies
    /// become natively cooperative (argmax table-Q). Mutually exclusive
    /// with `table_total` (which is for own-Q models).
    pub table_native_q: bool,
    /// Leaf bootstrap aggregation temperature. `None` keeps the classic
    /// max-Q bootstrap. `Some(tau)` replaces it with a softmax(q/tau)-
    /// weighted mean over the leaf menu: the max of N noisy estimates is
    /// upward-biased and high-variance, and eval noise is the measured
    /// binding constraint; a softened mean trades a little policy
    /// optimality for lower bias and variance. Interior advance stays
    /// argmax either way.
    pub leaf_softmix_temp: Option<f64>,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            n_simulations: 64,
            top_m: 16,
            max_root_actions: None,
            depth_rounds: 1,
            determinization_samples: 4,
            market_decision_samples: 8,
            exact_endgame_turns: 0,
            rollout_blend_weight: 0.5,
            parallel_leaf_rollouts: false,
            rollout_max_actions: 8,
            rollout_top_k: 4,
            k_interior: 16,
            exploration: false,
            c_visit: 50.0,
            c_scale: 1.0,
            sigma_normalization: SigmaNormalization::MinMax,
            paired_rollouts: false,
            search_seed: 0,
            determinization_seed: None,
            peek_true_hidden: false,
            table_total: false,
            table_native_q: false,
            leaf_softmix_temp: None,
        }
    }
}

/// One state whose legal-action menu needs a model evaluation.
pub struct EvalRow {
    pub staged: GameState,
    pub prelude: MarketPrelude,
    pub afterstates: Vec<CandidateAfterstate>,
}

/// Per-row model outputs, action-aligned with `EvalRow::afterstates`.
#[derive(Debug, Clone)]
pub struct EvalOut {
    /// Normalized policy priors.
    pub priors: Vec<f64>,
    /// Derived final Q per action: exact afterstate score + score-to-go.
    pub derived_final_q: Vec<f64>,
    /// Predicted final score per seat (absolute seat order), when the
    /// model's value head is available. Used only by table-total search.
    pub value_vector: Option<Vec<f64>>,
}

pub trait LeafEvaluator {
    fn evaluate_batch(&mut self, rows: &[EvalRow]) -> Result<Vec<EvalOut>>;
}

#[derive(Debug, Clone)]
pub struct GumbelSearchResult {
    /// Per root action: simulation-averaged value for visited actions, model
    /// derived final Q for unvisited ones. Aligned with the root menu.
    pub completed_q: Vec<f64>,
    /// Population variance of simulation values per action (0 when unvisited).
    pub value_variance: Vec<f64>,
    /// softmax(logits + sigma(completed_q)) over the full root menu.
    pub improved_policy: Vec<f64>,
    pub visit_counts: Vec<u32>,
    /// Model root priors (normalized), aligned with the root menu.
    pub root_priors: Vec<f64>,
    /// Improved-policy-weighted mean of completed Q.
    pub root_value: f64,
    pub chosen_index: usize,
    pub simulations_run: usize,
}

/// Complete policy decision for one turn. The optional refresh is valued over
/// hidden-order samples before the real replacement draw is exposed. If the
/// policy accepts, the returned row is then built from the newly revealed real
/// market and the draft is searched there.
pub struct GumbelTurnDecision {
    pub row: EvalRow,
    pub result: GumbelSearchResult,
    pub exact_endgame: bool,
    pub market_branches_searched: usize,
    pub market_chance_samples: usize,
    pub total_simulations_run: usize,
}

/// Builds one model/search row for every legal free three-of-a-kind choice.
/// The default/decline row is first; an accept row follows when the market has
/// exactly three matching tokens and the replacement is feasible.
#[cfg(test)]
pub fn eval_rows_for_state(game: &GameState, menu_limit: Option<usize>) -> Result<Vec<EvalRow>> {
    if game.is_game_over() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for prelude in game.free_three_of_a_kind_choices()? {
        if let Some(row) = eval_row_for_prelude(game, prelude, menu_limit)? {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn sampled_market_state(game: &GameState, cfg: &GumbelConfig, sample_index: usize) -> GameState {
    let public_hash = game.public_state().canonical_hash();
    let public_prefix = u64::from_le_bytes(
        public_hash.as_bytes()[..8]
            .try_into()
            .expect("public hash prefix has eight bytes"),
    );
    let stream = cfg.determinization_seed.unwrap_or(cfg.search_seed);
    let seed = splitmix64(
        stream ^ MARKET_DECISION_STREAM_SALT ^ public_prefix ^ splitmix64(sample_index as u64),
    );
    let mut sampled = game.clone();
    sampled.redeterminize_hidden(GameSeed::from_u64(seed));
    sampled
}

fn selected_completed_q(result: &GumbelSearchResult) -> Result<f64> {
    result
        .completed_q
        .get(result.chosen_index)
        .copied()
        .context("chosen gumbel action is absent from completed Q")
}

/// Exact one-ply solver for a player's final personal turn. Every action's
/// afterstate already contains that player's exact final score, so model
/// score-to-go and simulations can only add noise. Strict comparison keeps
/// the first legal action on ties, matching the engine's deterministic menu
/// order.
fn exact_final_turn_result(row: &EvalRow) -> Result<GumbelSearchResult> {
    let completed_q: Vec<f64> = row
        .afterstates
        .iter()
        .map(|afterstate| afterstate.exact_score_active)
        .collect();
    if completed_q.is_empty() {
        bail!("exact final-turn row has no legal actions");
    }
    let mut chosen_index = 0usize;
    for index in 1..completed_q.len() {
        if completed_q[index] > completed_q[chosen_index] {
            chosen_index = index;
        }
    }

    let action_count = completed_q.len();
    let mut improved_policy = vec![0.0; action_count];
    improved_policy[chosen_index] = 1.0;
    let mut visit_counts = vec![0; action_count];
    visit_counts[chosen_index] = 1;
    let root_value = completed_q[chosen_index];
    Ok(GumbelSearchResult {
        completed_q,
        value_variance: vec![0.0; action_count],
        improved_policy,
        visit_counts,
        root_priors: vec![1.0 / action_count as f64; action_count],
        root_value,
        chosen_index,
        simulations_run: 0,
    })
}

pub(crate) fn eval_row_for_prelude(
    game: &GameState,
    prelude: MarketPrelude,
    menu_limit: Option<usize>,
) -> Result<Option<EvalRow>> {
    if game.is_game_over() {
        return Ok(None);
    }
    let staged = game.preview_market_prelude(&prelude)?;
    let candidates = match rank_greedy_actions(&staged, &MarketPrelude::default(), menu_limit) {
        Ok(candidates) => candidates,
        Err(cascadia_sim::SimulationError::Rules(error))
            if crate::is_rollout_truncation_rule_error(&error) =>
        {
            // Empty bag/stack truncation: value the state as its own terminal.
            return Ok(None);
        }
        Err(error) => return Err(error).context("ranking gumbel menu actions"),
    };
    if candidates.is_empty() {
        bail!(
            "no legal candidates for non-terminal state at turn {}",
            game.completed_turns()
        );
    }
    let active_seat = staged.current_player();
    let afterstates = candidate_afterstates(&staged, &candidates, active_seat)?;
    Ok(Some(EvalRow {
        staged,
        prelude,
        afterstates,
    }))
}

/// Single-row helper retained for focused low-level tests. Policy-facing code
/// must use [`eval_rows_for_state`] or [`gumbel_search_for_state`] so it does
/// not discard the optional market branch.
#[cfg(test)]
pub fn eval_row_for_state(game: &GameState, menu_limit: Option<usize>) -> Result<Option<EvalRow>> {
    Ok(eval_rows_for_state(game, menu_limit)?.into_iter().next())
}

struct Simulation {
    /// Root-menu index of the action this simulation credits.
    action_index: usize,
    state: GameState,
    rounds_left: usize,
    rollout_rng: ChaCha8Rng,
    value: Option<f64>,
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn gumbel_noise(rng: &mut ChaCha8Rng) -> f64 {
    let uniform: f64 = rng.gen_range(f64::EPSILON..1.0);
    -(-uniform.ln()).ln()
}

fn sigma(q_normalized: f64, max_visits: u32, c_visit: f64, c_scale: f64) -> f64 {
    (c_visit + f64::from(max_visits)) * c_scale * q_normalized
}

fn minmax_normalize(values: &[f64]) -> Vec<f64> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
    }
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-12 {
        return vec![0.0; values.len()];
    }
    values
        .iter()
        .map(|value| (value - min) / (max - min))
        .collect()
}

/// Applies the configured sigma normalization. Degenerate inputs (empty,
/// non-finite, zero spread) normalize to all-zeros in every scheme, matching
/// the incumbent min-max fallback.
fn normalize_for_sigma(values: &[f64], scheme: SigmaNormalization) -> Vec<f64> {
    match scheme {
        SigmaNormalization::MinMax => minmax_normalize(values),
        SigmaNormalization::FixedScale(scale) => {
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            if !min.is_finite() || !(scale > 0.0) {
                return vec![0.0; values.len()];
            }
            values.iter().map(|value| (value - min) / scale).collect()
        }
        SigmaNormalization::ZScore => {
            if values.is_empty() {
                return Vec::new();
            }
            let count = values.len() as f64;
            let mean = values.iter().sum::<f64>() / count;
            let variance = values
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f64>()
                / count;
            let std = variance.sqrt();
            if !std.is_finite() || std < 1e-9 {
                return vec![0.0; values.len()];
            }
            values.iter().map(|value| (value - mean) / std).collect()
        }
        SigmaNormalization::TopKRange(k) => {
            let mut sorted: Vec<f64> = values.iter().cloned().filter(|v| v.is_finite()).collect();
            if sorted.is_empty() {
                return vec![0.0; values.len()];
            }
            sorted.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));
            let k = k.max(2).min(sorted.len());
            let max = sorted[0];
            let min = sorted[k - 1];
            if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-12 {
                return vec![0.0; values.len()];
            }
            values
                .iter()
                .map(|value| ((value - min) / (max - min)).clamp(0.0, 1.0))
                .collect()
        }
    }
}

/// Deterministic rollout stream seed for one (action, visit) simulation.
/// Unpaired (incumbent): a distinct stream per action, so rollout noise is
/// independent exactly across the comparisons halving makes. Paired: the
/// stream depends only on (determinization stream, visit index), so every
/// action's k-th visit shares one rollout randomness realization.
fn rollout_stream_seed(
    cfg: &GumbelConfig,
    det_stream: u64,
    action_index: usize,
    visit_index: usize,
) -> u64 {
    if cfg.paired_rollouts {
        splitmix64(det_stream ^ ROLLOUT_STREAM_SALT ^ splitmix64(0x1_0000 + visit_index as u64))
    } else {
        splitmix64(
            cfg.search_seed
                ^ ROLLOUT_STREAM_SALT
                ^ splitmix64(action_index as u64)
                ^ splitmix64(0x1_0000 + visit_index as u64),
        )
    }
}

/// Leaf bootstrap over the menu's derived final Q: classic max, or a
/// softmax(q/tau)-weighted mean when a softmix temperature is set.
fn leaf_bootstrap_value(derived_final_q: &[f64], softmix_temp: Option<f64>) -> f64 {
    let max = derived_final_q
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let Some(temp) = softmix_temp else {
        return max;
    };
    let temp = temp.max(1e-6);
    let mut weight_sum = 0.0_f64;
    let mut value_sum = 0.0_f64;
    for &q in derived_final_q {
        let weight = ((q - max) / temp).exp();
        weight_sum += weight;
        value_sum += weight * q;
    }
    value_sum / weight_sum
}

/// Terminal simulation value: root seat's score, or the table sum in
/// table-total mode.
fn terminal_simulation_value(state: &GameState, root_seat: usize, table_total: bool) -> f64 {
    let scores = score_game(state);
    if table_total {
        scores.iter().map(|score| f64::from(score.total)).sum()
    } else {
        f64::from(scores[root_seat].total)
    }
}

/// Σ of the other seats' predicted final scores (value head, absolute seat
/// order). Falls back to their exact current scores when the head is
/// unavailable (uniform-fallback bridges) — a grounded underestimate.
fn other_seats_final_estimate(eval: &EvalOut, state: &GameState, active_seat: usize) -> f64 {
    if let Some(values) = &eval.value_vector {
        if values.len() > active_seat {
            return values
                .iter()
                .enumerate()
                .filter(|(seat, _)| *seat != active_seat)
                .map(|(_, value)| *value)
                .sum();
        }
    }
    score_game(state)
        .iter()
        .enumerate()
        .filter(|(seat, _)| *seat != active_seat)
        .map(|(_, score)| f64::from(score.total))
        .sum()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|exp| exp / sum).collect()
}

struct InteriorRowGroup {
    sim_index: usize,
    decline_index: usize,
    accept_sample_start: usize,
    accept_sample_end: usize,
    actual_accept_index: Option<usize>,
}

struct LeafRolloutTask {
    simulation_index: usize,
    state: GameState,
    rollout_rng: ChaCha8Rng,
    bootstrap: f64,
}

fn resolve_leaf_rollout(
    mut task: LeafRolloutTask,
    root_seat: usize,
    table_value: bool,
    rollout_max_actions: usize,
    rollout_top_k: usize,
    blend_weight: f64,
) -> (usize, Result<f64>) {
    let result = complete_with_sampled_greedy(
        task.state,
        rollout_max_actions,
        rollout_top_k,
        &mut task.rollout_rng,
        None,
    )
    .map(|(terminal, _truncated)| {
        let rollout = terminal_simulation_value(&terminal, root_seat, table_value);
        blend_weight * task.bootstrap + (1.0 - blend_weight) * rollout
    });
    (task.simulation_index, result)
}

/// Advances every live simulation by one ply (batched model eval), resolving
/// simulations that hit terminal states or leaf conditions.
fn advance_simulations(
    simulations: &mut [Simulation],
    root_seat: usize,
    root_table_shift: f64,
    evaluator: &mut dyn LeafEvaluator,
    cfg: &GumbelConfig,
) -> Result<()> {
    loop {
        let mut model_rows = Vec::new();
        let mut row_groups = Vec::new();
        for sim_index in 0..simulations.len() {
            if simulations[sim_index].value.is_some() {
                continue;
            }
            let state = &simulations[sim_index].state;
            let Some(decline_row) =
                eval_row_for_prelude(state, MarketPrelude::default(), Some(cfg.k_interior))?
            else {
                simulations[sim_index].value = Some(terminal_simulation_value(
                    state,
                    root_seat,
                    cfg.table_total || cfg.table_native_q,
                ));
                continue;
            };
            let decline_index = model_rows.len();
            model_rows.push(decline_row);

            let accept = state
                .free_three_of_a_kind_choices()?
                .into_iter()
                .find(|choice| choice.replace_three_of_a_kind);
            let accept_sample_start = model_rows.len();
            let mut actual_accept_index = None;
            if let Some(accept) = accept {
                for sample_index in 0..cfg.market_decision_samples.max(1) {
                    let sampled = sampled_market_state(state, cfg, sample_index);
                    let row = eval_row_for_prelude(&sampled, accept.clone(), Some(cfg.k_interior))?
                        .context("sampled interior accepted market produced no row")?;
                    model_rows.push(row);
                }
                actual_accept_index = Some(model_rows.len());
                model_rows.push(
                    eval_row_for_prelude(state, accept, Some(cfg.k_interior))?
                        .context("actual interior accepted market produced no row")?,
                );
            }
            let accept_sample_end = actual_accept_index.unwrap_or(accept_sample_start);
            row_groups.push(InteriorRowGroup {
                sim_index,
                decline_index,
                accept_sample_start,
                accept_sample_end,
                actual_accept_index,
            });
        }
        if model_rows.is_empty() && row_groups.is_empty() {
            if simulations
                .iter()
                .all(|simulation| simulation.value.is_some())
            {
                return Ok(());
            }
            continue;
        }
        if row_groups.is_empty() {
            return Ok(());
        }

        let evals = evaluator.evaluate_batch(&model_rows)?;
        if evals.len() != model_rows.len() {
            bail!(
                "leaf evaluator returned {} results for {} rows",
                evals.len(),
                model_rows.len()
            );
        }
        let mut evaluated = model_rows
            .into_iter()
            .zip(evals)
            .map(Some)
            .collect::<Vec<_>>();
        let mut leaf_rollouts = Vec::new();
        for group in row_groups {
            let branch_value = |row_index: usize| -> Result<f64> {
                let Some((row, eval)) = evaluated[row_index].as_ref() else {
                    unreachable!("each interior model row is consumed at most once");
                };
                if eval.derived_final_q.len() != row.afterstates.len() {
                    bail!("evaluator derived_final_q length mismatch");
                }
                Ok(leaf_bootstrap_value(
                    &eval.derived_final_q,
                    cfg.leaf_softmix_temp,
                ))
            };
            let decline_value = branch_value(group.decline_index)?;
            let choose_accept = if let Some(_) = group.actual_accept_index {
                let accept_total = (group.accept_sample_start..group.accept_sample_end)
                    .map(&branch_value)
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .sum::<f64>();
                let accept_count = group.accept_sample_end - group.accept_sample_start;
                accept_total / accept_count as f64 > decline_value
            } else {
                false
            };
            let chosen_row_index = if choose_accept {
                group
                    .actual_accept_index
                    .expect("accepted interior branch has an actual row")
            } else {
                group.decline_index
            };
            let (row, eval) = evaluated[chosen_row_index]
                .take()
                .expect("chosen market branch exists");
            if eval.derived_final_q.len() != row.afterstates.len() {
                bail!("evaluator derived_final_q length mismatch");
            }
            let active_seat = row.staged.current_player();
            let simulation = &mut simulations[group.sim_index];
            let best_index = eval
                .derived_final_q
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index)
                .context("empty derived_final_q")?;
            if active_seat == root_seat {
                simulation.rounds_left = simulation.rounds_left.saturating_sub(1);
                if simulation.rounds_left == 0 {
                    // Leaf: blend a Q bootstrap (max or softmix aggregation)
                    // with an optional terminal rollout. Table-total mode
                    // keeps the exact-grounded own-seat Q and adds the other
                    // seats' predicted finals as a CONSTANT root-level shift:
                    // per-leaf value-head estimates re-introduce eval noise
                    // into the across-action comparison (measured CI− on
                    // 2026-07-08), while within one search the other seats'
                    // expected finals barely move. The rollout branch scores
                    // the whole terminal table exactly.
                    let own_bootstrap =
                        leaf_bootstrap_value(&eval.derived_final_q, cfg.leaf_softmix_temp);
                    let bootstrap = if cfg.table_total {
                        own_bootstrap + root_table_shift
                    } else {
                        own_bootstrap
                    };
                    let w = cfg.rollout_blend_weight.clamp(0.0, 1.0);
                    if w >= 1.0 {
                        simulation.value = Some(bootstrap);
                    } else {
                        leaf_rollouts.push(LeafRolloutTask {
                            simulation_index: group.sim_index,
                            state: row.staged,
                            rollout_rng: simulation.rollout_rng.clone(),
                            bootstrap,
                        });
                    }
                    continue;
                }
            }
            // Advance by the active seat's own argmax derived final Q. The
            // afterstate clone is reused as the next simulation state.
            let mut afterstates = row.afterstates;
            let chosen = afterstates.swap_remove(best_index);
            simulation.state = chosen.state;
        }
        if !leaf_rollouts.is_empty() {
            let resolve = |task| {
                resolve_leaf_rollout(
                    task,
                    root_seat,
                    cfg.table_total || cfg.table_native_q,
                    cfg.rollout_max_actions,
                    cfg.rollout_top_k,
                    cfg.rollout_blend_weight.clamp(0.0, 1.0),
                )
            };
            // Indexed Rayon collection preserves task order. Errors are then
            // surfaced sequentially, so enabling parallel execution does not
            // make failure selection or committed simulation state depend on
            // worker scheduling.
            let resolved = if cfg.parallel_leaf_rollouts {
                leaf_rollouts
                    .into_par_iter()
                    .map(resolve)
                    .collect::<Vec<_>>()
            } else {
                leaf_rollouts.into_iter().map(resolve).collect::<Vec<_>>()
            };
            for (simulation_index, value) in resolved {
                simulations[simulation_index].value = Some(value?);
            }
        }
    }
}

pub fn gumbel_search(
    root: &EvalRow,
    evaluator: &mut dyn LeafEvaluator,
    cfg: &GumbelConfig,
) -> Result<GumbelSearchResult> {
    if root.afterstates.is_empty() {
        bail!("gumbel_search requires at least one root action");
    }
    let root_seat = root.staged.current_player();
    if cfg.table_total && cfg.table_native_q {
        bail!("table_total and table_native_q are mutually exclusive");
    }

    // Root model evaluation: priors + initial per-action Q estimates.
    let root_eval = evaluator
        .evaluate_batch(std::slice::from_ref(root))?
        .into_iter()
        .next()
        .context("evaluator returned no root output")?;
    if root_eval.priors.len() != root.afterstates.len()
        || root_eval.derived_final_q.len() != root.afterstates.len()
    {
        bail!("root evaluator output misaligned with root menu");
    }
    let action_count = root.afterstates.len();
    // Table-total mode: model derived Q is own-seat scale; simulation values
    // are table scale. Shift every model-Q fallback by the other seats'
    // predicted finals so visited and unvisited actions are comparable
    // (constant additive shift — ranking among fallbacks is unchanged).
    let root_q_shift = if cfg.table_total {
        other_seats_final_estimate(&root_eval, &root.staged, root_seat)
    } else {
        0.0
    };
    let logits: Vec<f64> = root_eval
        .priors
        .iter()
        .map(|prior| prior.max(1e-12).ln())
        .collect();

    // Optional model-ranked root cap (by prior, never greedy rank).
    let mut candidate_indexes: Vec<usize> = (0..action_count).collect();
    if let Some(cap) = cfg.max_root_actions {
        candidate_indexes.sort_by(|&left, &right| {
            root_eval.priors[right]
                .partial_cmp(&root_eval.priors[left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidate_indexes.truncate(cap.max(1));
    }

    // Gumbel top-m over logits (noise-free when exploration is off).
    let mut gumbel_rng = ChaCha8Rng::seed_from_u64(cfg.search_seed ^ GUMBEL_NOISE_SALT);
    let mut gumbels = vec![0.0_f64; action_count];
    if cfg.exploration {
        for index in 0..action_count {
            gumbels[index] = gumbel_noise(&mut gumbel_rng);
        }
    }
    let top_m = cfg.top_m.max(1).min(candidate_indexes.len());
    candidate_indexes.sort_by(|&left, &right| {
        let left_score = gumbels[left] + logits[left];
        let right_score = gumbels[right] + logits[right];
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut survivors: Vec<usize> = candidate_indexes.into_iter().take(top_m).collect();

    let mut visit_counts = vec![0_u32; action_count];
    let mut value_sums = vec![0.0_f64; action_count];
    let mut value_sq_sums = vec![0.0_f64; action_count];
    let mut simulations_run = 0_usize;

    let phase_count = (top_m.max(2) as f64).log2().ceil() as usize;
    let budget = cfg.n_simulations.max(top_m);

    while survivors.len() > 1 || (simulations_run == 0 && !survivors.is_empty()) {
        let per_action = (budget / (phase_count.max(1) * survivors.len())).max(1);
        let mut simulations = Vec::with_capacity(per_action * survivors.len());
        for &action_index in &survivors {
            for _ in 0..per_action {
                if simulations_run + simulations.len() >= budget && visit_counts[action_index] > 0 {
                    break;
                }
                let visit_index = visit_counts[action_index] as usize
                    + simulations
                        .iter()
                        .filter(|s: &&Simulation| s.action_index == action_index)
                        .count();
                let det_index = (visit_index % cfg.determinization_samples.max(1)) as u64;
                let det_stream = cfg.determinization_seed.unwrap_or(cfg.search_seed);
                let det_seed =
                    splitmix64(det_stream ^ DETERMINIZATION_STREAM_SALT ^ splitmix64(det_index));
                let mut state = root.staged.clone();
                if !cfg.peek_true_hidden {
                    state.redeterminize_hidden(GameSeed::from_u64(det_seed));
                }
                let action = &root.afterstates[action_index].candidate.action;
                let rollout_seed = rollout_stream_seed(cfg, det_stream, action_index, visit_index);
                let mut simulation = Simulation {
                    action_index,
                    state,
                    rounds_left: cfg.depth_rounds.max(1),
                    rollout_rng: ChaCha8Rng::seed_from_u64(rollout_seed),
                    value: None,
                };
                if let Err(error) = simulation.state.apply(action) {
                    if crate::is_rollout_truncation_rule_error(&error) {
                        simulation.value = Some(terminal_simulation_value(
                            &simulation.state,
                            root_seat,
                            cfg.table_total || cfg.table_native_q,
                        ));
                    } else {
                        return Err(error).context("applying root action in gumbel simulation");
                    }
                }
                simulations.push(simulation);
            }
        }
        advance_simulations(&mut simulations, root_seat, root_q_shift, evaluator, cfg)?;
        for simulation in &simulations {
            let value = simulation
                .value
                .context("simulation resolved without a value")?;
            visit_counts[simulation.action_index] += 1;
            value_sums[simulation.action_index] += value;
            value_sq_sums[simulation.action_index] += value * value;
        }
        simulations_run += simulations.len();

        // Halve survivors by g + logits + sigma(mean value).
        if survivors.len() == 1 {
            break;
        }
        let mean_values: Vec<f64> = survivors
            .iter()
            .map(|&action_index| {
                if visit_counts[action_index] == 0 {
                    root_eval.derived_final_q[action_index] + root_q_shift
                } else {
                    value_sums[action_index] / f64::from(visit_counts[action_index])
                }
            })
            .collect();
        let normalized = normalize_for_sigma(&mean_values, cfg.sigma_normalization);
        let max_visits = survivors
            .iter()
            .map(|&action_index| visit_counts[action_index])
            .max()
            .unwrap_or(0);
        let mut scored: Vec<(usize, f64)> = survivors
            .iter()
            .enumerate()
            .map(|(position, &action_index)| {
                let score = gumbels[action_index]
                    + logits[action_index]
                    + sigma(normalized[position], max_visits, cfg.c_visit, cfg.c_scale);
                (action_index, score)
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = (survivors.len() + 1) / 2;
        survivors = scored
            .into_iter()
            .take(keep)
            .map(|(index, _)| index)
            .collect();
    }

    let chosen_index = *survivors.first().context("no surviving root action")?;

    // Completed Q over the full root menu.
    let completed_q: Vec<f64> = (0..action_count)
        .map(|action_index| {
            if visit_counts[action_index] == 0 {
                root_eval.derived_final_q[action_index] + root_q_shift
            } else {
                value_sums[action_index] / f64::from(visit_counts[action_index])
            }
        })
        .collect();
    let value_variance: Vec<f64> = (0..action_count)
        .map(|action_index| {
            let visits = visit_counts[action_index];
            if visits == 0 {
                0.0
            } else {
                let count = f64::from(visits);
                let mean = value_sums[action_index] / count;
                (value_sq_sums[action_index] / count - mean * mean).max(0.0)
            }
        })
        .collect();
    let normalized_q = normalize_for_sigma(&completed_q, cfg.sigma_normalization);
    let max_visits = visit_counts.iter().cloned().max().unwrap_or(0);
    let improved_logits: Vec<f64> = (0..action_count)
        .map(|action_index| {
            logits[action_index]
                + sigma(
                    normalized_q[action_index],
                    max_visits,
                    cfg.c_visit,
                    cfg.c_scale,
                )
        })
        .collect();
    let improved_policy = softmax(&improved_logits);
    let root_value = improved_policy
        .iter()
        .zip(completed_q.iter())
        .map(|(weight, q)| weight * q)
        .sum();
    let root_priors = root_eval
        .priors
        .iter()
        .map(|prior| prior.max(0.0))
        .collect();

    Ok(GumbelSearchResult {
        completed_q,
        value_variance,
        improved_policy,
        visit_counts,
        root_priors,
        root_value,
        chosen_index,
        simulations_run,
    })
}

/// Makes the optional refresh decision before observing its chance outcome.
///
/// Decline is searched once. Accept is valued by averaging searches over
/// public-hash-derived hidden-order samples. Only after accept wins is the real
/// replacement market revealed and searched for the draft. Exact ties decline.
pub fn gumbel_search_for_state(
    game: &GameState,
    menu_limit: Option<usize>,
    evaluator: &mut dyn LeafEvaluator,
    cfg: &GumbelConfig,
) -> Result<Option<GumbelTurnDecision>> {
    if game.is_game_over() {
        return Ok(None);
    }
    if cfg.exact_endgame_turns > 1 {
        bail!("exact_endgame_turns currently supports only 0 or 1");
    }
    if cfg.exact_endgame_turns > 0 && (cfg.table_total || cfg.table_native_q) {
        bail!("exact final-personal-turn solving is incompatible with table-total objectives");
    }
    let exact_endgame = cfg.exact_endgame_turns == 1
        && game.turns_remaining_for_player(game.current_player()) == 1;
    // The normal root-menu cap is a performance pre-filter. Exact means the
    // complete legal menu: even a score-ranked cap could become unsound if a
    // future rules/scoring change makes that ranking differ from final score.
    let effective_menu_limit = if exact_endgame { None } else { menu_limit };
    let mut search_row = |row: &EvalRow| {
        if exact_endgame {
            exact_final_turn_result(row)
        } else {
            gumbel_search(row, evaluator, cfg)
        }
    };
    let choices = game.free_three_of_a_kind_choices()?;
    let accept = choices
        .iter()
        .find(|choice| choice.replace_three_of_a_kind)
        .cloned();
    let decline = MarketPrelude::default();
    let Some(decline_row) = eval_row_for_prelude(game, decline, effective_menu_limit)? else {
        return Ok(None);
    };
    let decline_result = search_row(&decline_row)?;
    let decline_value = selected_completed_q(&decline_result)?;
    let mut total_simulations_run = 0usize;
    total_simulations_run += decline_result.simulations_run;

    let Some(accept) = accept else {
        return Ok(Some(GumbelTurnDecision {
            row: decline_row,
            result: decline_result,
            exact_endgame,
            market_branches_searched: 1,
            market_chance_samples: 0,
            total_simulations_run,
        }));
    };

    let market_chance_samples = cfg.market_decision_samples.max(1);
    let mut accept_total = 0.0;
    for sample_index in 0..market_chance_samples {
        let sampled = sampled_market_state(game, cfg, sample_index);
        let sampled_row = eval_row_for_prelude(&sampled, accept.clone(), effective_menu_limit)?
            .context("sampled accepted market produced no gumbel row")?;
        let sampled_result = search_row(&sampled_row)?;
        total_simulations_run += sampled_result.simulations_run;
        accept_total += selected_completed_q(&sampled_result)?;
    }
    let accept_value = accept_total / market_chance_samples as f64;

    if accept_value <= decline_value {
        return Ok(Some(GumbelTurnDecision {
            row: decline_row,
            result: decline_result,
            exact_endgame,
            market_branches_searched: 2,
            market_chance_samples,
            total_simulations_run,
        }));
    }

    // The decision is now committed. Reveal the real replacement market and
    // search the downstream draft without reusing any sampled chance outcome.
    let actual_accept_row = eval_row_for_prelude(game, accept, effective_menu_limit)?
        .context("accepted real market produced no gumbel row")?;
    let actual_accept_result = search_row(&actual_accept_row)?;
    total_simulations_run += actual_accept_result.simulations_run;
    Ok(Some(GumbelTurnDecision {
        row: actual_accept_row,
        result: actual_accept_result,
        exact_endgame,
        market_branches_searched: 2,
        market_chance_samples,
        total_simulations_run,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::{GameConfig, Market};

    /// Deterministic evaluator: priors proportional to softmax of exact
    /// afterstate scores; per-action Q = exact afterstate score + 1.
    struct MockEvaluator {
        calls: usize,
        rows_seen: usize,
        value_vector: Option<Vec<f64>>,
    }

    impl MockEvaluator {
        fn new() -> Self {
            Self {
                calls: 0,
                rows_seen: 0,
                value_vector: None,
            }
        }
    }

    impl LeafEvaluator for MockEvaluator {
        fn evaluate_batch(&mut self, rows: &[EvalRow]) -> Result<Vec<EvalOut>> {
            self.calls += 1;
            self.rows_seen += rows.len();
            Ok(rows
                .iter()
                .map(|row| {
                    // Scale-consistent final-score proxy: exact afterstate
                    // score plus a per-remaining-turn allowance, mirroring how
                    // the real q head predicts *final* score, not immediate.
                    let remaining = row.staged.turns_remaining() as f64;
                    let exact: Vec<f64> = row
                        .afterstates
                        .iter()
                        .map(|afterstate| afterstate.exact_score_active)
                        .collect();
                    let priors =
                        softmax(&exact.iter().map(|value| value / 4.0).collect::<Vec<_>>());
                    let derived_final_q = exact
                        .iter()
                        .map(|value| value + remaining.max(0.0))
                        .collect::<Vec<_>>();
                    EvalOut {
                        priors,
                        derived_final_q,
                        value_vector: self.value_vector.clone(),
                    }
                })
                .collect())
        }
    }

    struct MarketBranchEvaluator {
        root_turn: u16,
        prefer_accept: bool,
        current_accept_branch: bool,
    }

    struct ActualReplacementTrapEvaluator {
        target_market: Market,
    }

    impl LeafEvaluator for ActualReplacementTrapEvaluator {
        fn evaluate_batch(&mut self, rows: &[EvalRow]) -> Result<Vec<EvalOut>> {
            Ok(rows
                .iter()
                .map(|row| {
                    let value = if !row.prelude.replace_three_of_a_kind {
                        0.0
                    } else if row.staged.market() == &self.target_market {
                        100.0
                    } else {
                        -100.0
                    };
                    let action_count = row.afterstates.len();
                    EvalOut {
                        priors: vec![1.0 / action_count as f64; action_count],
                        derived_final_q: vec![value; action_count],
                        value_vector: None,
                    }
                })
                .collect())
        }
    }

    impl LeafEvaluator for MarketBranchEvaluator {
        fn evaluate_batch(&mut self, rows: &[EvalRow]) -> Result<Vec<EvalOut>> {
            Ok(rows
                .iter()
                .map(|row| {
                    if row.staged.completed_turns() == self.root_turn {
                        self.current_accept_branch = row.prelude.replace_three_of_a_kind;
                    }
                    let preferred = self.current_accept_branch == self.prefer_accept;
                    let value = if preferred { 100.0 } else { 0.0 };
                    let action_count = row.afterstates.len();
                    EvalOut {
                        priors: vec![1.0 / action_count as f64; action_count],
                        derived_final_q: vec![value; action_count],
                        value_vector: None,
                    }
                })
                .collect())
        }
    }

    fn test_state(seed_u64: u64, plies: usize) -> GameState {
        let config = GameConfig::research_aaaaa(4).expect("4p config");
        let mut game = GameState::new(config, GameSeed::from_u64(seed_u64)).expect("game");
        let mut rng = ChaCha8Rng::seed_from_u64(seed_u64 ^ 0x1234);
        for _ in 0..plies {
            if game.is_game_over() {
                break;
            }
            let (next, _) =
                complete_with_sampled_greedy(game, 4, 2, &mut rng, Some(1)).expect("advance");
            game = next;
        }
        game
    }

    fn state_with_three_of_a_kind() -> GameState {
        let config = GameConfig::research_aaaaa(4).expect("4p config");
        (0..10_000)
            .map(|seed| GameState::new(config, GameSeed::from_u64(seed)).expect("game setup"))
            .find(|game| game.market().three_of_a_kind().is_some())
            .expect("seed search must find a three-of-a-kind market")
    }

    fn test_config(seed: u64) -> GumbelConfig {
        GumbelConfig {
            n_simulations: 16,
            top_m: 4,
            determinization_samples: 2,
            rollout_blend_weight: 1.0,
            k_interior: 6,
            search_seed: seed,
            ..GumbelConfig::default()
        }
    }

    #[test]
    fn search_budget_and_outputs_are_well_formed() {
        let game = test_state(2_026_070_100, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let action_count = root.afterstates.len();
        let mut evaluator = MockEvaluator::new();
        let result =
            gumbel_search(&root, &mut evaluator, &test_config(7)).expect("search completes");

        assert_eq!(result.completed_q.len(), action_count);
        assert_eq!(result.improved_policy.len(), action_count);
        assert_eq!(result.visit_counts.len(), action_count);
        assert!(result.chosen_index < action_count);
        assert!(result.simulations_run >= 1);
        // Budget overrun is bounded by one per-action allotment per phase.
        assert!(result.simulations_run <= 16 + 4);
        let policy_sum: f64 = result.improved_policy.iter().sum();
        assert!((policy_sum - 1.0).abs() < 1e-9, "policy sums to 1");
        assert!(result.visit_counts[result.chosen_index] > 0);
        // improved[a] is proportional to prior[a] * exp(sigma(q_norm[a])), so
        // the relative upweighting improved/prior must peak exactly at the
        // action with the highest completed Q.
        let best_completed = (0..action_count)
            .max_by(|&left, &right| {
                result.completed_q[left]
                    .partial_cmp(&result.completed_q[right])
                    .unwrap()
            })
            .unwrap();
        let best_ratio = (0..action_count)
            .max_by(|&left, &right| {
                let left_ratio = result.improved_policy[left] / result.root_priors[left].max(1e-12);
                let right_ratio =
                    result.improved_policy[right] / result.root_priors[right].max(1e-12);
                left_ratio.partial_cmp(&right_ratio).unwrap()
            })
            .unwrap();
        assert_eq!(
            best_ratio, best_completed,
            "improved policy must upweight the highest completed-Q action most"
        );
    }

    #[test]
    fn search_is_deterministic_given_seed() {
        let game = test_state(2_026_070_200, 5);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |seed: u64| {
            let mut evaluator = MockEvaluator::new();
            gumbel_search(&root, &mut evaluator, &test_config(seed)).expect("search")
        };
        let first = run(42);
        let second = run(42);
        assert_eq!(first.chosen_index, second.chosen_index);
        assert_eq!(first.visit_counts, second.visit_counts);
        assert_eq!(first.completed_q, second.completed_q);
        assert_eq!(first.improved_policy, second.improved_policy);
    }

    #[test]
    fn gumbel_policy_can_accept_or_decline_free_three_of_a_kind() {
        let game = state_with_three_of_a_kind();
        let choices = eval_rows_for_state(&game, Some(8)).expect("market branches");
        assert_eq!(choices.len(), 2);
        assert!(!choices[0].prelude.replace_three_of_a_kind);
        assert!(choices[1].prelude.replace_three_of_a_kind);

        for prefer_accept in [false, true] {
            let mut evaluator = MarketBranchEvaluator {
                root_turn: game.completed_turns(),
                prefer_accept,
                current_accept_branch: false,
            };
            let cfg = GumbelConfig {
                n_simulations: 4,
                top_m: 1,
                max_root_actions: Some(1),
                depth_rounds: 1,
                rollout_blend_weight: 1.0,
                k_interior: 2,
                ..test_config(17)
            };
            let decision = gumbel_search_for_state(&game, Some(8), &mut evaluator, &cfg)
                .expect("search")
                .expect("non-terminal decision");
            assert_eq!(decision.row.prelude.replace_three_of_a_kind, prefer_accept);
        }
    }

    #[test]
    fn exact_final_personal_turn_bypasses_model_and_search() {
        let game = test_state(2_026_071_001, 76);
        assert_eq!(
            game.turns_remaining_for_player(game.current_player()),
            1,
            "76 completed plies leave the active seat one personal turn"
        );
        let mut evaluator = MockEvaluator::new();
        let cfg = GumbelConfig {
            exact_endgame_turns: 1,
            market_decision_samples: 3,
            ..test_config(0x1eaf)
        };
        let decision = gumbel_search_for_state(&game, Some(1), &mut evaluator, &cfg)
            .expect("exact decision")
            .expect("non-terminal decision");

        assert!(decision.exact_endgame);
        assert!(
            decision.row.afterstates.len() > 1,
            "exact frontier must ignore the normal root-menu cap"
        );
        assert_eq!(decision.result.simulations_run, 0);
        assert_eq!(decision.total_simulations_run, 0);
        assert_eq!(evaluator.calls, 0, "exact frontier must not invoke the model");
        let best_exact = decision
            .row
            .afterstates
            .iter()
            .map(|afterstate| afterstate.exact_score_active)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            decision.result.completed_q[decision.result.chosen_index],
            best_exact
        );
        assert_eq!(decision.result.root_value, best_exact);
        assert_eq!(
            decision.result.improved_policy[decision.result.chosen_index],
            1.0
        );
    }

    #[test]
    fn exact_final_turn_refresh_decision_respects_hidden_chance_boundary() {
        let cfg = GumbelConfig {
            exact_endgame_turns: 1,
            market_decision_samples: 4,
            ..test_config(0xdec1de)
        };
        let row_best = |row: &EvalRow| {
            row.afterstates
                .iter()
                .map(|afterstate| afterstate.exact_score_active)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let (game, accept, decline_value, accept_value) = (0..200)
            .find_map(|offset| {
                let game = test_state(2_026_072_000 + offset, 76);
                let accept = game
                    .free_three_of_a_kind_choices()
                    .ok()?
                    .into_iter()
                    .find(|choice| choice.replace_three_of_a_kind)?;
                let decline_row = eval_row_for_prelude(&game, MarketPrelude::default(), None)
                    .ok()??;
                let decline_value = row_best(&decline_row);
                let accept_total: f64 = (0..cfg.market_decision_samples)
                    .map(|sample_index| {
                        let sampled = sampled_market_state(&game, &cfg, sample_index);
                        let row = eval_row_for_prelude(&sampled, accept.clone(), None)
                            .expect("sampled accept row")
                            .expect("sampled accept remains playable");
                        row_best(&row)
                    })
                    .sum();
                let accept_value = accept_total / cfg.market_decision_samples as f64;
                (accept_value > decline_value)
                    .then_some((game, accept, decline_value, accept_value))
            })
            .expect("seed search must find a profitable final-turn refresh");

        let mut evaluator = MockEvaluator::new();
        let decision = gumbel_search_for_state(&game, Some(1), &mut evaluator, &cfg)
            .expect("exact refresh decision")
            .expect("non-terminal decision");
        assert!(
            accept_value > decline_value,
            "fixture must require accepting before the real replacement is known"
        );
        assert_eq!(decision.row.prelude, accept);
        assert!(decision.exact_endgame);
        assert_eq!(decision.market_chance_samples, cfg.market_decision_samples);
        assert_eq!(decision.total_simulations_run, 0);
        assert_eq!(evaluator.calls, 0);
        assert_eq!(
            selected_completed_q(&decision.result).expect("chosen exact score"),
            row_best(&decision.row),
            "after accepting, the actual revealed market must be solved exactly"
        );
    }

    #[test]
    fn exact_final_turn_rejects_unsupported_frontiers_and_objectives() {
        let game = test_state(2_026_071_002, 76);
        let mut evaluator = MockEvaluator::new();
        let mut cfg = test_config(1);
        cfg.exact_endgame_turns = 2;
        let error = gumbel_search_for_state(&game, Some(8), &mut evaluator, &cfg)
            .err()
            .expect("frontier two must fail");
        assert!(error.to_string().contains("only 0 or 1"));

        cfg.exact_endgame_turns = 1;
        cfg.table_total = true;
        let error = gumbel_search_for_state(&game, Some(8), &mut evaluator, &cfg)
            .err()
            .expect("table-total exact personal-turn mode must fail");
        assert!(error.to_string().contains("incompatible with table-total"));
    }

    #[test]
    fn triple_root_market_decision_cannot_observe_actual_replacement_order() {
        let game = state_with_three_of_a_kind();
        let accept = game
            .free_three_of_a_kind_choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.replace_three_of_a_kind)
            .expect("accept branch");
        let target_market = game
            .preview_market_prelude(&accept)
            .expect("actual accepted market")
            .market()
            .clone();
        let redetermined = (1..10_000)
            .find_map(|seed| {
                let mut candidate = game.clone();
                candidate.redeterminize_hidden(GameSeed::from_u64(seed));
                let market = candidate
                    .preview_market_prelude(&accept)
                    .ok()?
                    .market()
                    .clone();
                (market != target_market).then_some(candidate)
            })
            .expect("a different hidden order must reveal a different accepted market");

        let cfg = GumbelConfig {
            n_simulations: 4,
            top_m: 1,
            max_root_actions: Some(1),
            depth_rounds: 1,
            determinization_samples: 2,
            market_decision_samples: 2,
            rollout_blend_weight: 1.0,
            k_interior: 2,
            ..test_config(0x5151)
        };
        let mut left_evaluator = ActualReplacementTrapEvaluator {
            target_market: target_market.clone(),
        };
        let mut right_evaluator = ActualReplacementTrapEvaluator { target_market };
        let left = gumbel_search_for_state(&game, Some(8), &mut left_evaluator, &cfg)
            .expect("left search")
            .expect("left decision");
        let right = gumbel_search_for_state(&redetermined, Some(8), &mut right_evaluator, &cfg)
            .expect("right search")
            .expect("right decision");

        assert_eq!(left.row.prelude, right.row.prelude);
        assert!(!left.row.prelude.replace_three_of_a_kind);
        assert_eq!(left.market_chance_samples, 2);
        assert_eq!(right.market_chance_samples, 2);
    }

    #[test]
    fn search_never_observes_true_hidden_order() {
        let game = test_state(2_026_070_300, 6);
        let staged = game
            .preview_market_prelude(&MarketPrelude::default())
            .expect("staged");
        let mut permuted_game = staged.clone();
        permuted_game.redeterminize_hidden(GameSeed::from_u64(0xfeed_f00d));
        assert_eq!(
            staged.public_state().canonical_hash(),
            permuted_game.public_state().canonical_hash(),
            "redeterminization must preserve public state"
        );

        let run = |state: &GameState| {
            let root = eval_row_for_state(state, None)
                .expect("root row")
                .expect("non-terminal root");
            let mut evaluator = MockEvaluator::new();
            gumbel_search(&root, &mut evaluator, &test_config(9)).expect("search")
        };
        // Both a blended and a pure-bootstrap config must be invariant.
        let original = run(&staged);
        let permuted = run(&permuted_game);
        assert_eq!(original.chosen_index, permuted.chosen_index);
        assert_eq!(original.completed_q, permuted.completed_q);
        assert_eq!(original.improved_policy, permuted.improved_policy);

        let run_blended = |state: &GameState| {
            let root = eval_row_for_state(state, None)
                .expect("root row")
                .expect("non-terminal root");
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(9);
            cfg.rollout_blend_weight = 0.5;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let original_blended = run_blended(&staged);
        let permuted_blended = run_blended(&permuted_game);
        assert_eq!(original_blended.completed_q, permuted_blended.completed_q);
    }

    #[test]
    fn peek_mode_observes_true_hidden_order() {
        // Inverse of the no-peek invariance test: with peek_true_hidden the
        // rollout leaves run on the true hidden order, so permuting it must
        // change the search result. Guards against the flag silently becoming
        // a no-op (which would invalidate ceiling measurements).
        let game = test_state(2_026_070_300, 6);
        let staged = game
            .preview_market_prelude(&MarketPrelude::default())
            .expect("staged");
        let mut permuted_game = staged.clone();
        permuted_game.redeterminize_hidden(GameSeed::from_u64(0xfeed_f00d));

        let run_peek = |state: &GameState| {
            let root = eval_row_for_state(state, None)
                .expect("root row")
                .expect("non-terminal root");
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(9);
            cfg.rollout_blend_weight = 0.0; // leaf = pure rollout on the (peeked) state
            cfg.peek_true_hidden = true;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let original = run_peek(&staged);
        let permuted = run_peek(&permuted_game);
        assert_ne!(
            original.completed_q, permuted.completed_q,
            "peek search must depend on the true hidden order"
        );
    }

    #[test]
    fn table_native_q_scores_tables_without_shift() {
        // Native mode must match table_total's terminal/rollout scale but
        // apply NO fallback shift: with w=0.0 (pure rollout, bootstrap
        // unused) and top_m=1, unvisited fallbacks stay at raw model Q
        // while the visited action's rollout value matches table mode.
        let game = test_state(2_026_070_800, 8);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |table_total: bool, native: bool| {
            let mut evaluator = MockEvaluator::new();
            evaluator.value_vector = Some(vec![10.0, 20.0, 30.0, 40.0]);
            let mut cfg = test_config(19);
            cfg.top_m = 1;
            cfg.rollout_blend_weight = 0.0;
            cfg.table_total = table_total;
            cfg.table_native_q = native;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let own = run(false, false);
        let table = run(true, false);
        let native = run(false, true);
        assert_eq!(own.visit_counts, native.visit_counts);
        for index in 0..own.completed_q.len() {
            if own.visit_counts[index] == 0 {
                // No shift in native mode; table mode shifts.
                assert_eq!(native.completed_q[index], own.completed_q[index]);
                assert!(table.completed_q[index] > own.completed_q[index]);
            } else {
                // Identical rollout streams: table-scale values match.
                assert!((native.completed_q[index] - table.completed_q[index]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn table_flags_are_mutually_exclusive() {
        let game = test_state(2_026_070_800, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let mut evaluator = MockEvaluator::new();
        let mut cfg = test_config(21);
        cfg.table_total = true;
        cfg.table_native_q = true;
        assert!(gumbel_search(&root, &mut evaluator, &cfg).is_err());
    }

    #[test]
    fn leaf_softmix_bounds_and_limits() {
        let q = [10.0, 8.0, 4.0, -2.0];
        let max = leaf_bootstrap_value(&q, None);
        assert_eq!(max, 10.0);
        // Tiny temperature converges to the max.
        let sharp = leaf_bootstrap_value(&q, Some(1e-4));
        assert!((sharp - 10.0).abs() < 1e-6);
        // Softer temperatures move monotonically from max toward the mean,
        // never past either bound.
        let mean = q.iter().sum::<f64>() / q.len() as f64;
        let mut previous = max;
        for temp in [0.5, 2.0, 8.0, 64.0] {
            let value = leaf_bootstrap_value(&q, Some(temp));
            assert!(value <= previous + 1e-12, "monotone in temperature");
            assert!(value > mean - 1e-12, "never below the uniform mean");
            previous = value;
        }
    }

    #[test]
    fn leaf_softmix_changes_search_values() {
        let game = test_state(2_026_070_700, 5);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |softmix: Option<f64>| {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(17);
            cfg.leaf_softmix_temp = softmix;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let max_run = run(None);
        let mix_run = run(Some(4.0));
        // Same seeds, same interior play; softened leaves must lower (or at
        // the degenerate single-action leaf, preserve) visited values.
        let mut lowered = 0;
        for index in 0..max_run.completed_q.len() {
            if max_run.visit_counts[index] > 0 && mix_run.visit_counts[index] > 0 {
                assert!(mix_run.completed_q[index] <= max_run.completed_q[index] + 1e-9);
                if mix_run.completed_q[index] < max_run.completed_q[index] - 1e-9 {
                    lowered += 1;
                }
            }
        }
        assert!(lowered > 0, "softmix must actually soften some leaf");
    }

    #[test]
    fn table_total_shifts_model_fallbacks_onto_table_scale() {
        // With w=1.0 (no rollouts) and no terminals in reach, every value in
        // table mode is the own-seat value plus the other seats' predicted
        // finals — for unvisited fallbacks by construction, for visited leaf
        // bootstraps because the evaluator is deterministic across runs.
        let game = test_state(2_026_070_500, 5);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let root_seat = root.staged.current_player();
        let values = vec![10.0, 20.0, 30.0, 40.0];
        let shift: f64 = values
            .iter()
            .enumerate()
            .filter(|(seat, _)| *seat != root_seat)
            .map(|(_, value)| *value)
            .sum();

        let run = |table_total: bool| {
            let mut evaluator = MockEvaluator::new();
            evaluator.value_vector = Some(values.clone());
            let mut cfg = test_config(11);
            cfg.top_m = 1;
            cfg.table_total = table_total;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let own = run(false);
        let table = run(true);
        assert_eq!(own.visit_counts, table.visit_counts);
        assert_eq!(own.chosen_index, table.chosen_index);
        let mut unvisited_checked = 0;
        for index in 0..own.completed_q.len() {
            let expected = own.completed_q[index] + shift;
            assert!(
                (table.completed_q[index] - expected).abs() < 1e-9,
                "action {index}: table Q {} must equal own Q {} + shift {shift}",
                table.completed_q[index],
                own.completed_q[index]
            );
            if own.visit_counts[index] == 0 {
                unvisited_checked += 1;
            }
        }
        assert!(unvisited_checked > 0, "test requires unvisited fallbacks");
    }

    #[test]
    fn table_total_rollout_values_score_the_whole_table() {
        // Pure-rollout leaves (w=0.0) in table mode score every seat's
        // terminal total, so visited Q strictly exceeds the own-seat variant
        // (identical rollout rng streams → identical terminal states).
        let game = test_state(2_026_070_600, 8);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |table_total: bool| {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(13);
            cfg.rollout_blend_weight = 0.0;
            cfg.table_total = table_total;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let own = run(false);
        let table = run(true);
        // Table mode may allocate visits differently (the objective changed;
        // rollout table totals are not a constant shift of own-seat totals),
        // so only compare actions visited under both objectives: each rollout
        // terminal's table sum strictly exceeds the root seat's component.
        let mut visited_checked = 0;
        for index in 0..own.completed_q.len() {
            if own.visit_counts[index] > 0 && table.visit_counts[index] > 0 {
                assert!(
                    table.completed_q[index] > own.completed_q[index],
                    "visited action {index}: table Q must exceed own-seat Q"
                );
                visited_checked += 1;
            }
        }
        assert!(
            visited_checked > 0,
            "test requires commonly visited actions"
        );
    }

    #[test]
    fn parallel_leaf_rollouts_are_bit_identical() {
        let game = test_state(2_026_070_650, 8);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |parallel_leaf_rollouts: bool| {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(0x5eed);
            cfg.n_simulations = 32;
            cfg.top_m = 8;
            cfg.rollout_blend_weight = 0.5;
            cfg.parallel_leaf_rollouts = parallel_leaf_rollouts;
            let result = gumbel_search(&root, &mut evaluator, &cfg).expect("search");
            (result, evaluator.calls, evaluator.rows_seen)
        };
        let (serial, serial_calls, serial_rows) = run(false);
        let (parallel, parallel_calls, parallel_rows) = run(true);

        assert_eq!(serial_calls, parallel_calls);
        assert_eq!(serial_rows, parallel_rows);
        assert_eq!(serial.completed_q, parallel.completed_q);
        assert_eq!(serial.value_variance, parallel.value_variance);
        assert_eq!(serial.improved_policy, parallel.improved_policy);
        assert_eq!(serial.visit_counts, parallel.visit_counts);
        assert_eq!(serial.root_priors, parallel.root_priors);
        assert_eq!(serial.root_value, parallel.root_value);
        assert_eq!(serial.chosen_index, parallel.chosen_index);
        assert_eq!(serial.simulations_run, parallel.simulations_run);
    }

    #[test]
    fn exploration_off_consumes_no_gumbel_noise() {
        let game = test_state(2_026_070_400, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |search_seed: u64, exploration: bool| {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(search_seed);
            cfg.exploration = exploration;
            // Pin the determinization and rollout inputs so the only varying
            // stream across search seeds is the Gumbel noise itself.
            cfg.determinization_seed = Some(99);
            cfg.rollout_blend_weight = 1.0;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        // With exploration off, different search seeds produce identical
        // results because the noise stream is never sampled.
        let first = run(1, false);
        let second = run(2, false);
        assert_eq!(first.chosen_index, second.chosen_index);
        assert_eq!(first.visit_counts, second.visit_counts);
        assert_eq!(first.completed_q, second.completed_q);
        assert_eq!(first.improved_policy, second.improved_policy);
    }

    #[test]
    fn sigma_normalization_schemes_behave() {
        let values = [90.0, 96.0, 100.0, 40.0];

        // MinMax matches the incumbent helper exactly.
        assert_eq!(
            normalize_for_sigma(&values, SigmaNormalization::MinMax),
            minmax_normalize(&values)
        );

        // FixedScale: differences track raw point differences / scale,
        // independent of the menu tail.
        let fixed = normalize_for_sigma(&values, SigmaNormalization::FixedScale(10.0));
        assert!((fixed[2] - fixed[1] - 0.4).abs() < 1e-12);
        assert!((fixed[1] - fixed[0] - 0.6).abs() < 1e-12);

        // ZScore: mean 0, unit variance, monotone.
        let zscore = normalize_for_sigma(&values, SigmaNormalization::ZScore);
        let mean: f64 = zscore.iter().sum::<f64>() / zscore.len() as f64;
        assert!(mean.abs() < 1e-9);
        assert!(zscore[2] > zscore[1] && zscore[1] > zscore[0] && zscore[0] > zscore[3]);

        // TopKRange(3): the outlier 40 no longer stretches the denominator —
        // the top-3 window [90, 100] maps to [0, 1] and the outlier clamps.
        let topk = normalize_for_sigma(&values, SigmaNormalization::TopKRange(3));
        assert!((topk[2] - 1.0).abs() < 1e-12);
        assert!((topk[0] - 0.0).abs() < 1e-12);
        assert!((topk[1] - 0.6).abs() < 1e-12);
        assert_eq!(topk[3], 0.0, "tail outlier clamps to the window floor");
        // Under MinMax the same top-2 gap is compressed by the outlier.
        let minmax = normalize_for_sigma(&values, SigmaNormalization::MinMax);
        assert!(
            (topk[2] - topk[1]) > (minmax[2] - minmax[1]),
            "top-k range must widen contender gaps relative to min-max"
        );

        // Degenerate spread normalizes to zeros in every scheme.
        for scheme in [
            SigmaNormalization::MinMax,
            SigmaNormalization::FixedScale(10.0),
            SigmaNormalization::ZScore,
            SigmaNormalization::TopKRange(3),
        ] {
            let flat = normalize_for_sigma(&[5.0, 5.0, 5.0], scheme);
            assert!(flat.iter().all(|v| v.abs() < 1e-12), "{scheme:?}");
        }
    }

    #[test]
    fn rollout_stream_seed_contracts() {
        let mut cfg = test_config(0xabc);
        cfg.determinization_seed = Some(0xd5);
        let det_stream = cfg.determinization_seed.unwrap_or(cfg.search_seed);

        // Unpaired (default) reproduces the legacy formula bit-for-bit and
        // separates streams across actions.
        assert!(!cfg.paired_rollouts);
        let legacy = |action_index: usize, visit_index: usize| {
            splitmix64(
                cfg.search_seed
                    ^ ROLLOUT_STREAM_SALT
                    ^ splitmix64(action_index as u64)
                    ^ splitmix64(0x1_0000 + visit_index as u64),
            )
        };
        for action_index in 0..4 {
            for visit_index in 0..3 {
                assert_eq!(
                    rollout_stream_seed(&cfg, det_stream, action_index, visit_index),
                    legacy(action_index, visit_index)
                );
            }
        }
        assert_ne!(
            rollout_stream_seed(&cfg, det_stream, 0, 0),
            rollout_stream_seed(&cfg, det_stream, 1, 0)
        );

        // Paired: equal (det stream, visit index) shares one stream across
        // every action; distinct visits still get distinct streams; and the
        // stream follows the determinization stream, not the search seed.
        cfg.paired_rollouts = true;
        assert_eq!(
            rollout_stream_seed(&cfg, det_stream, 0, 0),
            rollout_stream_seed(&cfg, det_stream, 7, 0)
        );
        assert_ne!(
            rollout_stream_seed(&cfg, det_stream, 0, 0),
            rollout_stream_seed(&cfg, det_stream, 0, 1)
        );
        let mut other_search_seed = cfg.clone();
        other_search_seed.search_seed = 0xdef;
        assert_eq!(
            rollout_stream_seed(&other_search_seed, det_stream, 3, 2),
            rollout_stream_seed(&cfg, det_stream, 3, 2),
            "paired streams must be search-seed independent so pinned-seed \
             paired comparisons share rollout randomness across arms"
        );
    }

    #[test]
    fn paired_rollouts_search_is_deterministic_and_well_formed() {
        let game = test_state(2_026_070_650, 8);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = || {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(0x5eed);
            cfg.n_simulations = 32;
            cfg.top_m = 8;
            cfg.rollout_blend_weight = 0.5;
            cfg.paired_rollouts = true;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let first = run();
        let second = run();
        assert_eq!(first.completed_q, second.completed_q);
        assert_eq!(first.chosen_index, second.chosen_index);
        assert_eq!(first.visit_counts, second.visit_counts);
        assert!(first.chosen_index < root.afterstates.len());
    }

    #[test]
    fn sigma_norm_variants_search_end_to_end() {
        let game = test_state(2_026_070_100, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        for scheme in [
            SigmaNormalization::MinMax,
            SigmaNormalization::FixedScale(20.0),
            SigmaNormalization::ZScore,
            SigmaNormalization::TopKRange(4),
        ] {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(7);
            cfg.sigma_normalization = scheme;
            let result = gumbel_search(&root, &mut evaluator, &cfg).expect("search completes");
            assert!(result.chosen_index < root.afterstates.len(), "{scheme:?}");
            let policy_sum: f64 = result.improved_policy.iter().sum();
            assert!((policy_sum - 1.0).abs() < 1e-9, "{scheme:?} policy sums to 1");
        }
    }

    #[test]
    fn smaller_c_scale_trusts_noisy_q_less() {
        let game = test_state(2_026_070_100, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |c_scale: f64| {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(7);
            cfg.c_scale = c_scale;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search completes")
        };
        let sharp = run(1.0);
        let soft = run(0.01);
        let l1 = |policy: &[f64], priors: &[f64]| -> f64 {
            policy
                .iter()
                .zip(priors.iter())
                .map(|(p, q)| (p - q).abs())
                .sum()
        };
        // sigma shrinks with c_scale, so the improved policy must sit closer
        // to the raw priors when c_scale is small.
        assert!(
            l1(&soft.improved_policy, &soft.root_priors)
                < l1(&sharp.improved_policy, &sharp.root_priors)
        );
    }

    #[test]
    fn near_terminal_roots_still_resolve() {
        // Advance deep into the game, then search close to the end.
        let mut game = test_state(2_026_070_500, 70);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        while !game.is_game_over() && game.turns_remaining() > 2 {
            let (next, _) =
                complete_with_sampled_greedy(game, 4, 1, &mut rng, Some(1)).expect("advance");
            game = next;
        }
        if game.is_game_over() {
            return; // seed ended early; nothing to assert
        }
        let root = eval_row_for_state(&game, None).expect("root row");
        let Some(root) = root else {
            return;
        };
        let mut evaluator = MockEvaluator::new();
        let mut cfg = test_config(11);
        cfg.rollout_blend_weight = 0.5;
        let result = gumbel_search(&root, &mut evaluator, &cfg).expect("search near terminal");
        assert!(result.chosen_index < root.afterstates.len());
        assert!(result.simulations_run >= 1);
    }
}
