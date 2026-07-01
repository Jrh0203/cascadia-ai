use blake3::Hasher;
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, score_board};
use cascadia_sim::{
    MatchResult, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, SimulationError,
    play_match_with_selector, play_pattern_plies, rank_wildlife_diverse_pattern_frontier_actions,
    select_pattern_action, strategy_rng,
};
use rand_chacha::ChaCha8Rng;

use crate::{SearchError, rollout_rng};

pub const PUBLIC_FOCAL_OPEN_LOOP_TREE_STRATEGY_ID: &str = "public-focal-open-loop-tree-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicFocalOpenLoopTreeConfig {
    pub blueprint: PatternAwareConfig,
    pub wildlife_candidate_limit: usize,
    pub root_candidate_limit: usize,
    pub simulations: usize,
    pub exploration_milli: u16,
    pub final_personal_turns: u16,
}

impl PublicFocalOpenLoopTreeConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        if self.wildlife_candidate_limit == 0 || self.root_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "public focal tree candidate limits must be positive",
            ));
        }
        if self.simulations < self.root_candidate_limit {
            return Err(SearchError::InvalidConfig(
                "public focal tree simulations must cover every retained root candidate",
            ));
        }
        if self.exploration_milli == 0 {
            return Err(SearchError::InvalidConfig(
                "public focal tree exploration must be positive",
            ));
        }
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "public focal tree requires 1 to 20 final personal turns",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{PUBLIC_FOCAL_OPEN_LOOP_TREE_STRATEGY_ID}-t{}-s{}-r{}-u{}-k{}-h{}-b{}-w{}-m{}",
            self.final_personal_turns,
            self.simulations,
            self.root_candidate_limit,
            self.exploration_milli,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.wildlife_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }

    fn exploration(self) -> f64 {
        f64::from(self.exploration_milli) / 1_000.0
    }
}

impl Default for PublicFocalOpenLoopTreeConfig {
    fn default() -> Self {
        Self {
            blueprint: PatternAwareConfig::default(),
            wildlife_candidate_limit: 2,
            root_candidate_limit: 16,
            simulations: 128,
            exploration_milli: 2_000,
            final_personal_turns: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PublicTreeRootEvaluation {
    pub action: TurnAction,
    pub visits: usize,
    pub mean_terminal_score: f64,
    pub terminal_score_stddev: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PublicTreeAnalysis {
    pub selected_action: TurnAction,
    pub simulations: usize,
    pub node_count: usize,
    pub maximum_focal_depth: usize,
    pub root: Vec<PublicTreeRootEvaluation>,
}

#[derive(Debug, Clone, Default)]
struct TreeEdge {
    visits: usize,
    score_sum: f64,
    score_square_sum: f64,
    child: Option<usize>,
}

impl TreeEdge {
    fn mean(&self) -> f64 {
        self.score_sum / self.visits as f64
    }

    fn standard_deviation(&self) -> f64 {
        if self.visits < 2 {
            return 0.0;
        }
        let mean = self.mean();
        ((self.score_square_sum - self.visits as f64 * mean * mean) / (self.visits - 1) as f64)
            .max(0.0)
            .sqrt()
    }
}

#[derive(Debug, Clone, Default)]
struct TreeNode {
    visits: usize,
    edges: Vec<TreeEdge>,
}

impl TreeNode {
    fn ensure_edges(&mut self, count: usize) {
        self.edges.resize_with(count, TreeEdge::default);
    }
}

pub struct PublicFocalOpenLoopTreeStrategy {
    config: PublicFocalOpenLoopTreeConfig,
    strategy_id: String,
}

impl PublicFocalOpenLoopTreeStrategy {
    pub fn new(config: PublicFocalOpenLoopTreeConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_tree(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn analyze_deterministic(
        &self,
        game: &GameState,
    ) -> Result<PublicTreeAnalysis, SearchError> {
        if game.is_game_over() {
            return Err(SearchError::NoLegalActions);
        }
        if game.market().three_of_a_kind().is_some() {
            return Err(SearchError::InvalidConfig(
                "public focal tree analysis requires an already observable root market",
            ));
        }

        let root_candidates = rank_wildlife_diverse_pattern_frontier_actions(
            game,
            &MarketPrelude::default(),
            self.config.blueprint,
            self.config.wildlife_candidate_limit,
        )?;
        if root_candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        let root_actions = root_candidates
            .into_iter()
            .take(self.config.root_candidate_limit)
            .map(|candidate| candidate.action)
            .collect::<Vec<_>>();
        let focal_seat = game.current_player();
        let cards = game.config().scoring_cards;
        let public_hash = game.public_state().canonical_hash();
        let mut nodes = vec![TreeNode::default()];
        nodes[0].ensure_edges(root_actions.len());
        let mut maximum_focal_depth = 0;

        for simulation in 0..self.config.simulations {
            let sample_seed = simulation_seed(public_hash.as_bytes(), simulation);
            let mut sample = game.clone();
            sample.redeterminize_hidden(sample_seed);
            let mut policy_rng = rollout_rng(sample_seed);
            let mut node_index = 0;
            let mut path = Vec::new();

            loop {
                let actions = if node_index == 0 {
                    &root_actions
                } else {
                    let ranked = rank_wildlife_diverse_pattern_frontier_actions(
                        &sample,
                        &current_prelude(&sample),
                        self.config.blueprint,
                        self.config.wildlife_candidate_limit,
                    )?;
                    let action_count = ranked.len();
                    nodes[node_index].ensure_edges(action_count);
                    let owned = ranked
                        .into_iter()
                        .map(|candidate| candidate.action)
                        .collect::<Vec<_>>();
                    let edge_index = select_edge(
                        &nodes[node_index],
                        owned.len(),
                        false,
                        self.config.exploration(),
                    )?;
                    sample.apply(&owned[edge_index])?;
                    path.push((node_index, edge_index));
                    advance_opponents(
                        &mut sample,
                        focal_seat,
                        self.config.blueprint,
                        &mut policy_rng,
                    )?;
                    if sample.is_game_over() {
                        break;
                    }
                    let first_visit = nodes[node_index].edges[edge_index].visits == 0;
                    let child = child_node(&mut nodes, node_index, edge_index);
                    if first_visit {
                        break;
                    }
                    node_index = child;
                    continue;
                };

                let edge_index = select_edge(
                    &nodes[node_index],
                    actions.len(),
                    true,
                    self.config.exploration(),
                )?;
                sample.apply(&actions[edge_index])?;
                path.push((node_index, edge_index));
                advance_opponents(
                    &mut sample,
                    focal_seat,
                    self.config.blueprint,
                    &mut policy_rng,
                )?;
                if sample.is_game_over() {
                    break;
                }
                let child = child_node(&mut nodes, node_index, edge_index);
                if nodes[node_index].edges[edge_index].visits == 0 {
                    break;
                }
                node_index = child;
            }

            maximum_focal_depth = maximum_focal_depth.max(path.len());
            if !sample.is_game_over() {
                let remaining = usize::from(sample.turns_remaining());
                play_pattern_plies(
                    &mut sample,
                    remaining,
                    self.config.blueprint,
                    &mut policy_rng,
                )?;
            }
            let score = f64::from(score_board(&sample.boards()[focal_seat], cards).base_total);
            for (visited_node, visited_edge) in path {
                nodes[visited_node].visits += 1;
                let edge = &mut nodes[visited_node].edges[visited_edge];
                edge.visits += 1;
                edge.score_sum += score;
                edge.score_square_sum += score * score;
            }
        }

        let root_node = &nodes[0];
        debug_assert_eq!(root_node.visits, self.config.simulations);
        let selected_index = robust_root_child(root_node)?;
        let root = root_actions
            .into_iter()
            .zip(&root_node.edges)
            .map(|(action, edge)| PublicTreeRootEvaluation {
                action,
                visits: edge.visits,
                mean_terminal_score: edge.mean(),
                terminal_score_stddev: edge.standard_deviation(),
            })
            .collect::<Vec<_>>();
        Ok(PublicTreeAnalysis {
            selected_action: root[selected_index].action.clone(),
            simulations: root_node.visits,
            node_count: nodes.len(),
            maximum_focal_depth,
            root,
        })
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        let prelude = current_prelude(game);
        let anchor = select_pattern_action(game, &prelude, self.config.blueprint, blueprint_rng)?;
        if !self.uses_tree(game) || prelude.replace_three_of_a_kind {
            return Ok(anchor);
        }
        Ok(self.analyze_deterministic(game)?.selected_action)
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

fn current_prelude(game: &GameState) -> MarketPrelude {
    MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    }
}

fn simulation_seed(public_hash: &[u8; 32], simulation: usize) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2/public-focal-open-loop-tree-v1");
    hasher.update(public_hash);
    hasher.update(&(simulation as u64).to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn progressive_width(visits: usize, action_count: usize) -> usize {
    let width = ((visits + 1) as f64).sqrt().ceil() as usize;
    width.clamp(1, action_count)
}

fn select_edge(
    node: &TreeNode,
    action_count: usize,
    root: bool,
    exploration: f64,
) -> Result<usize, SearchError> {
    if action_count == 0 {
        return Err(SearchError::NoLegalActions);
    }
    let width = if root {
        action_count
    } else {
        progressive_width(node.visits, action_count)
    };
    if let Some(unvisited) = node.edges[..width].iter().position(|edge| edge.visits == 0) {
        return Ok(unvisited);
    }
    let log_parent = (node.visits.max(1) as f64).ln();
    node.edges[..width]
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            let left_value = left.mean() + exploration * (log_parent / left.visits as f64).sqrt();
            let right_value =
                right.mean() + exploration * (log_parent / right.visits as f64).sqrt();
            left_value
                .total_cmp(&right_value)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .ok_or(SearchError::NoLegalActions)
}

fn child_node(nodes: &mut Vec<TreeNode>, node_index: usize, edge_index: usize) -> usize {
    if let Some(child) = nodes[node_index].edges[edge_index].child {
        return child;
    }
    let child = nodes.len();
    nodes.push(TreeNode::default());
    nodes[node_index].edges[edge_index].child = Some(child);
    child
}

fn robust_root_child(root: &TreeNode) -> Result<usize, SearchError> {
    root.edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.visits > 0)
        .max_by(|(left_index, left), (right_index, right)| {
            left.visits
                .cmp(&right.visits)
                .then_with(|| left.mean().total_cmp(&right.mean()))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .ok_or(SearchError::NoLegalActions)
}

fn advance_opponents(
    game: &mut GameState,
    focal_seat: usize,
    blueprint: PatternAwareConfig,
    rng: &mut ChaCha8Rng,
) -> Result<(), SearchError> {
    while !game.is_game_over() && game.current_player() != focal_seat {
        let action = select_pattern_action(game, &current_prelude(game), blueprint, rng)?;
        game.apply(&action)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};
    use cascadia_sim::{play_pattern_plies, strategy_rng};

    use super::*;

    fn tiny_config() -> PublicFocalOpenLoopTreeConfig {
        PublicFocalOpenLoopTreeConfig {
            blueprint: PatternAwareConfig {
                immediate_candidate_limit: 2,
                habitat_candidate_limit: 1,
                bear_candidate_limit: 1,
                future_market_draws: 1,
            },
            wildlife_candidate_limit: 1,
            root_candidate_limit: 2,
            simulations: 4,
            exploration_milli: 2_000,
            final_personal_turns: 2,
        }
    }

    fn late_game(seed: u64) -> GameState {
        let game_seed = GameSeed::from_u64(seed);
        let mut game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), game_seed).unwrap();
        let mut rng = strategy_rng(game_seed, 0, "public-tree-test");
        while game.turns_remaining_for_player(game.current_player()) > 2
            || game.market().three_of_a_kind().is_some()
        {
            play_pattern_plies(&mut game, 1, tiny_config().blueprint, &mut rng).unwrap();
        }
        game
    }

    #[test]
    fn config_rejects_invalid_work() {
        let mut config = tiny_config();
        config.simulations = 1;
        assert!(config.validate().is_err());
        config = tiny_config();
        config.exploration_milli = 0;
        assert!(config.validate().is_err());
        config = tiny_config();
        config.final_personal_turns = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn analysis_is_deterministic_legal_and_accounts_for_budget() {
        let game = late_game(701);
        let strategy = PublicFocalOpenLoopTreeStrategy::new(tiny_config()).unwrap();
        let left = strategy.analyze_deterministic(&game).unwrap();
        let right = strategy.analyze_deterministic(&game).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.simulations, tiny_config().simulations);
        assert_eq!(
            left.root.iter().map(|edge| edge.visits).sum::<usize>(),
            tiny_config().simulations
        );
        assert!(left.root.iter().all(|edge| edge.visits > 0));
        assert!(left.node_count > 1);
        assert!(left.maximum_focal_depth >= 1);
        game.transition(&left.selected_action).unwrap();
    }

    #[test]
    fn analysis_ignores_actual_hidden_order() {
        let game = late_game(702);
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(703));
        let strategy = PublicFocalOpenLoopTreeStrategy::new(tiny_config()).unwrap();

        assert_eq!(
            strategy.analyze_deterministic(&game).unwrap(),
            strategy.analyze_deterministic(&redetermined).unwrap()
        );
    }

    #[test]
    fn complete_match_is_deterministic_legal_and_replayable() {
        let config = tiny_config();
        let strategy = PublicFocalOpenLoopTreeStrategy::new(config).unwrap();
        let game_config = GameConfig::research_aaaaa(2).unwrap();
        let seed = GameSeed::from_u64(704);
        let left = strategy.play_match(game_config, seed).unwrap();
        let right = strategy.play_match(game_config, seed).unwrap();

        assert_eq!(left.scores, right.scores);
        assert_eq!(left.replay, right.replay);
        left.replay.play().unwrap();
    }
}
