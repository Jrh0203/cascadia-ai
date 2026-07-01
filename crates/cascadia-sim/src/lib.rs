//! Deterministic match simulation built exclusively on `cascadia-game`.

mod pattern;

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use blake3::Hasher;
use cascadia_game::{
    Board, GameConfig, GameSeed, GameState, MarketPrelude, Replay, RuleError, ScoreBreakdown,
    TurnAction, Wildlife, rescore_after_tile_with_habitat_analysis,
    rescore_after_wildlife_placement, rescore_with_wildlife_scores, score_board, score_game,
};
#[cfg(test)]
use cascadia_game::{rescore_after_placement, rescore_after_placement_with_habitat_analysis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use pattern::{
    PATTERN_AWARE_STRATEGY_ID, PATTERN_COMMITMENT_STRATEGY_ID, PATTERN_COMPETITION_STRATEGY_ID,
    PATTERN_PORTFOLIO_STRATEGY_ID, PATTERN_POTENTIAL_STRATEGY_ID, PatternAwareConfig,
    PatternCandidate, PatternPotentialConfig, PatternPotentialStrategy,
    best_pattern_heuristic_value, future_market_opportunity, future_wildlife_opportunity,
    play_pattern_plies, rank_pattern_actions, rank_pattern_commitment_actions,
    rank_pattern_competition_actions, rank_pattern_frontier_actions,
    rank_pattern_portfolio_actions, rank_pattern_potential_actions,
    rank_wildlife_diverse_pattern_frontier_actions, rank_wildlife_focused_pattern_frontier_actions,
    select_pattern_action, select_pattern_commitment_action, select_pattern_competition_action,
    select_pattern_portfolio_action, select_pattern_potential_action, wildlife_marginal_gains,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyKind {
    Random,
    Greedy,
    PatternAware,
    PatternCommitment,
    PatternCompetition,
    PatternPortfolio,
}

impl StrategyKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Random => "random-v1",
            Self::Greedy => "greedy-v1",
            Self::PatternAware => PATTERN_AWARE_STRATEGY_ID,
            Self::PatternCommitment => PATTERN_COMMITMENT_STRATEGY_ID,
            Self::PatternCompetition => PATTERN_COMPETITION_STRATEGY_ID,
            Self::PatternPortfolio => PATTERN_PORTFOLIO_STRATEGY_ID,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchConfig {
    pub game: GameConfig,
    pub seed: GameSeed,
    pub seats: Vec<StrategyKind>,
}

impl MatchConfig {
    pub fn symmetric(game: GameConfig, seed: GameSeed, strategy: StrategyKind) -> Self {
        Self {
            game,
            seed,
            seats: vec![strategy; usize::from(game.player_count)],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchResult {
    pub seed: GameSeed,
    pub strategies: Vec<String>,
    pub scores: Vec<ScoreBreakdown>,
    pub turns: u16,
    #[serde(default)]
    pub decision_seconds: Vec<f64>,
    pub elapsed_seconds: f64,
    pub replay: Replay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GreedyCandidate {
    pub action: TurnAction,
    pub resulting_base_score: u16,
    pub immediate_rank: usize,
}

fn is_dominated_same_slot_independent(action: &TurnAction) -> bool {
    matches!(
        action.draft,
        cascadia_game::DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } if tile_slot == wildlife_slot
    )
}

fn rescore_after_cached_wildlife_placement(
    board: &Board,
    cards: cascadia_game::ScoringCards,
    after_tile: ScoreBreakdown,
    placed_wildlife: (Wildlife, cascadia_game::HexCoord),
    cache: &mut HashMap<(Wildlife, cascadia_game::HexCoord), [u16; 5]>,
) -> ScoreBreakdown {
    let wildlife_scores = *cache.entry(placed_wildlife).or_insert_with(|| {
        rescore_after_wildlife_placement(board, cards, after_tile, placed_wildlife.0).wildlife
    });
    rescore_with_wildlife_scores(board, after_tile, wildlife_scores)
}

pub fn rank_greedy_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    limit: Option<usize>,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    let cards = game.config().scoring_cards;
    let active_board = &game.boards()[game.current_player()];
    let baseline = score_board(active_board, cards);
    let habitat = active_board.habitat_analysis();
    let mut wildlife_score_cache = HashMap::new();
    let mut candidates: Vec<_> = game
        .evaluate_legal_turn_actions_with_tile_context(
            prelude,
            |board, placement, tile| {
                rescore_after_tile_with_habitat_analysis(
                    board, cards, baseline, &habitat, placement, tile,
                )
            },
            |board, after_tile, placed_wildlife| {
                placed_wildlife
                    .map_or(*after_tile, |placed_wildlife| {
                        rescore_after_cached_wildlife_placement(
                            board,
                            cards,
                            *after_tile,
                            placed_wildlife,
                            &mut wildlife_score_cache,
                        )
                    })
                    .base_total
            },
        )?
        .into_iter()
        .filter(|(action, _)| !is_dominated_same_slot_independent(action))
        .map(|(action, resulting_base_score)| GreedyCandidate {
            action,
            resulting_base_score,
            immediate_rank: 0,
        })
        .collect();
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.resulting_base_score));
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.immediate_rank = index + 1;
    }
    if let Some(limit) = limit {
        candidates.truncate(limit);
    }
    Ok(candidates)
}

pub fn rank_bear_setup_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    limit: Option<usize>,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    let cards = game.config().scoring_cards;
    let active_board = &game.boards()[game.current_player()];
    let baseline = score_board(active_board, cards);
    let habitat = active_board.habitat_analysis();
    let mut wildlife_score_cache = HashMap::new();
    let mut candidates: Vec<_> = game
        .evaluate_legal_turn_actions_with_tile_context(
            prelude,
            |board, placement, tile| {
                rescore_after_tile_with_habitat_analysis(
                    board, cards, baseline, &habitat, placement, tile,
                )
            },
            |board, after_tile, placed_wildlife| {
                let score = placed_wildlife.map_or(*after_tile, |placed_wildlife| {
                    rescore_after_cached_wildlife_placement(
                        board,
                        cards,
                        *after_tile,
                        placed_wildlife,
                        &mut wildlife_score_cache,
                    )
                });
                (
                    score.base_total,
                    score.wildlife[Wildlife::Bear as usize],
                    bear_pair_ready_slots(board),
                )
            },
        )?
        .into_iter()
        .filter(|(action, _)| !is_dominated_same_slot_independent(action))
        .map(
            |(action, (resulting_base_score, bear_score, pair_ready_slots))| {
                (
                    GreedyCandidate {
                        action,
                        resulting_base_score,
                        immediate_rank: 0,
                    },
                    bear_score,
                    pair_ready_slots,
                )
            },
        )
        .collect();
    let mut immediate_order: Vec<_> = (0..candidates.len()).collect();
    immediate_order.sort_by(|left, right| {
        candidates[*right]
            .0
            .resulting_base_score
            .cmp(&candidates[*left].0.resulting_base_score)
    });
    for (rank, index) in immediate_order.into_iter().enumerate() {
        candidates[index].0.immediate_rank = rank + 1;
    }
    candidates.sort_by(
        |(left, left_bear, left_ready), (right, right_bear, right_ready)| {
            right_bear
                .cmp(left_bear)
                .then_with(|| right_ready.cmp(left_ready))
                .then_with(|| right.resulting_base_score.cmp(&left.resulting_base_score))
        },
    );
    if let Some(limit) = limit {
        candidates.truncate(limit);
    }
    Ok(candidates
        .into_iter()
        .map(|(candidate, _, _)| candidate)
        .collect())
}

pub fn rank_habitat_setup_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    limit: Option<usize>,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    let cards = game.config().scoring_cards;
    let active_board = &game.boards()[game.current_player()];
    let baseline = score_board(active_board, cards);
    let habitat = active_board.habitat_analysis();
    let mut wildlife_score_cache = HashMap::new();
    let mut candidates: Vec<_> = game
        .evaluate_legal_turn_actions_with_tile_context(
            prelude,
            |board, placement, tile| {
                let score = rescore_after_tile_with_habitat_analysis(
                    board, cards, baseline, &habitat, placement, tile,
                );
                (
                    score,
                    habitat.matching_edges_after_tile(
                        board,
                        placement.coord,
                        tile,
                        placement.rotation,
                    ),
                    score.habitat.iter().sum::<u16>(),
                )
            },
            |board, &(after_tile, matching_edges, habitat_score), placed_wildlife| {
                let score = placed_wildlife.map_or(after_tile, |placed_wildlife| {
                    rescore_after_cached_wildlife_placement(
                        board,
                        cards,
                        after_tile,
                        placed_wildlife,
                        &mut wildlife_score_cache,
                    )
                });
                (score.base_total, matching_edges, habitat_score)
            },
        )?
        .into_iter()
        .filter(|(action, _)| !is_dominated_same_slot_independent(action))
        .map(
            |(action, (resulting_base_score, matching_edges, habitat_score))| {
                (
                    GreedyCandidate {
                        action,
                        resulting_base_score,
                        immediate_rank: 0,
                    },
                    matching_edges,
                    habitat_score,
                )
            },
        )
        .collect();
    let mut immediate_order: Vec<_> = (0..candidates.len()).collect();
    immediate_order.sort_by(|left, right| {
        candidates[*right]
            .0
            .resulting_base_score
            .cmp(&candidates[*left].0.resulting_base_score)
    });
    for (rank, index) in immediate_order.into_iter().enumerate() {
        candidates[index].0.immediate_rank = rank + 1;
    }
    candidates.sort_by(
        |(left, left_edges, left_habitat), (right, right_edges, right_habitat)| {
            right_edges
                .cmp(left_edges)
                .then_with(|| right_habitat.cmp(left_habitat))
                .then_with(|| right.resulting_base_score.cmp(&left.resulting_base_score))
        },
    );

    let mut distinct = Vec::new();
    for (candidate, _, _) in candidates {
        if distinct.iter().any(|retained: &GreedyCandidate| {
            retained.action.draft == candidate.action.draft
                && retained.action.tile == candidate.action.tile
        }) {
            continue;
        }
        distinct.push(candidate);
    }
    if let Some(limit) = limit {
        distinct.truncate(limit);
    }
    Ok(distinct)
}

fn bear_pair_ready_slots(board: &Board) -> u16 {
    let mut ready = 0u16;
    for (coord, placed) in board.placed_tiles() {
        if placed.wildlife.is_some() || !placed.tile.wildlife.contains(Wildlife::Bear) {
            continue;
        }
        let mut adjacent_bear = None;
        let mut has_multiple_adjacent_bears = false;
        for neighbor in coord.neighbors() {
            if board.wildlife_at(neighbor) == Some(Wildlife::Bear)
                && adjacent_bear.replace(neighbor).is_some()
            {
                has_multiple_adjacent_bears = true;
                break;
            }
        }
        let Some(adjacent_bear) = adjacent_bear else {
            continue;
        };
        if !has_multiple_adjacent_bears
            && adjacent_bear
                .neighbors()
                .into_iter()
                .all(|neighbor| board.wildlife_at(neighbor) != Some(Wildlife::Bear))
        {
            ready += 1;
        }
    }
    ready
}

#[cfg(test)]
fn bear_pair_ready_slots_reference(board: &Board) -> u16 {
    board
        .wildlife_placements(Wildlife::Bear)
        .into_iter()
        .filter(|coord| {
            let adjacent_bears: Vec<_> = coord
                .neighbors()
                .into_iter()
                .filter(|neighbor| board.wildlife_at(*neighbor) == Some(Wildlife::Bear))
                .collect();
            adjacent_bears.len() == 1
                && adjacent_bears[0]
                    .neighbors()
                    .into_iter()
                    .all(|neighbor| board.wildlife_at(neighbor) != Some(Wildlife::Bear))
        })
        .count() as u16
}

#[cfg(test)]
fn matching_habitat_edges(board: &Board) -> u16 {
    let doubled_matches = board
        .placed_tiles()
        .map(|(coord, placed)| {
            (0..6)
                .filter(|edge| {
                    board
                        .tile_at(coord.neighbor(*edge))
                        .is_some_and(|neighbor| {
                            placed.tile.terrain_on_edge(placed.rotation, *edge)
                                == neighbor
                                    .tile
                                    .terrain_on_edge(neighbor.rotation, (*edge + 3) % 6)
                        })
                })
                .count() as u16
        })
        .sum::<u16>();
    doubled_matches / 2
}

pub fn play_match(config: &MatchConfig) -> Result<MatchResult, SimulationError> {
    play_match_observed(config, |_, _| {})
}

pub fn play_match_observed(
    config: &MatchConfig,
    mut observe: impl FnMut(&GameState, &TurnAction),
) -> Result<MatchResult, SimulationError> {
    if config.seats.len() != usize::from(config.game.player_count) {
        return Err(SimulationError::SeatCount {
            expected: usize::from(config.game.player_count),
            actual: config.seats.len(),
        });
    }

    let started = Instant::now();
    let mut game = GameState::new(config.game, config.seed)?;
    let mut replay = Replay::new(config.game, config.seed);
    let mut strategies: Vec<_> = config
        .seats
        .iter()
        .enumerate()
        .map(|(seat, kind)| Strategy::new(*kind, config.seed, seat))
        .collect();
    let mut decision_seconds = Vec::with_capacity(usize::from(game.total_turns()));

    while !game.is_game_over() {
        let player = game.current_player();
        let decision_started = Instant::now();
        let action = strategies[player].select_action(&game)?;
        decision_seconds.push(decision_started.elapsed().as_secs_f64());
        observe(&game, &action);
        game.apply(&action)?;
        replay.turns.push(action);
    }
    replay.final_state_hash = Some(*game.canonical_hash().as_bytes());
    let scores = score_game(&game);
    Ok(MatchResult {
        seed: config.seed,
        strategies: config
            .seats
            .iter()
            .map(|kind| kind.id().to_owned())
            .collect(),
        scores,
        turns: game.completed_turns(),
        decision_seconds,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        replay,
    })
}

pub fn play_match_with_selector(
    game_config: GameConfig,
    seed: GameSeed,
    strategy_id: &str,
    select_action: impl FnMut(usize, &GameState) -> Result<TurnAction, SimulationError>,
) -> Result<MatchResult, SimulationError> {
    let strategy_ids = vec![strategy_id.to_owned(); usize::from(game_config.player_count)];
    play_match_with_seat_selector(game_config, seed, &strategy_ids, select_action)
}

/// Plays a match with an externally supplied selector and an exact identity for
/// every seat.
///
/// This is the serving boundary for heterogeneous model pools.  The callback
/// receives the active seat and may dispatch to a different frozen local model
/// for each seat.  The identities are copied into the result verbatim so a
/// trajectory can prove which policy controlled every action.
pub fn play_match_with_seat_selector(
    game_config: GameConfig,
    seed: GameSeed,
    strategy_ids: &[String],
    mut select_action: impl FnMut(usize, &GameState) -> Result<TurnAction, SimulationError>,
) -> Result<MatchResult, SimulationError> {
    let expected = usize::from(game_config.player_count);
    if strategy_ids.len() != expected {
        return Err(SimulationError::SeatCount {
            expected,
            actual: strategy_ids.len(),
        });
    }
    if strategy_ids
        .iter()
        .any(|identity| identity.trim().is_empty())
    {
        return Err(SimulationError::InvalidStrategyIdentity);
    }
    let started = Instant::now();
    let mut game = GameState::new(game_config, seed)?;
    let mut replay = Replay::new(game_config, seed);
    let mut decision_seconds = Vec::with_capacity(usize::from(game.total_turns()));
    while !game.is_game_over() {
        let player = game.current_player();
        let decision_started = Instant::now();
        let action = select_action(player, &game)?;
        decision_seconds.push(decision_started.elapsed().as_secs_f64());
        game.apply(&action)?;
        replay.turns.push(action);
    }
    replay.final_state_hash = Some(*game.canonical_hash().as_bytes());
    Ok(MatchResult {
        seed,
        strategies: strategy_ids.to_vec(),
        scores: score_game(&game),
        turns: game.completed_turns(),
        decision_seconds,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        replay,
    })
}

struct Strategy {
    kind: StrategyKind,
    rng: ChaCha8Rng,
}

impl Strategy {
    fn new(kind: StrategyKind, game_seed: GameSeed, seat: usize) -> Self {
        Self {
            kind,
            rng: strategy_rng(game_seed, seat, kind.id()),
        }
    }

    fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SimulationError> {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        match self.kind {
            StrategyKind::Random => {
                let actions = game.legal_turn_actions(&prelude)?;
                if actions.is_empty() {
                    return Err(SimulationError::NoLegalActions);
                }
                let index = self.rng.gen_range(0..actions.len());
                Ok(actions[index].clone())
            }
            StrategyKind::Greedy => self.select_greedy(game, &prelude),
            StrategyKind::PatternAware => {
                select_pattern_action(game, &prelude, PatternAwareConfig::default(), &mut self.rng)
            }
            StrategyKind::PatternCommitment => select_pattern_commitment_action(
                game,
                &prelude,
                PatternAwareConfig::default(),
                &mut self.rng,
            ),
            StrategyKind::PatternCompetition => select_pattern_competition_action(
                game,
                &prelude,
                PatternAwareConfig::default(),
                &mut self.rng,
            ),
            StrategyKind::PatternPortfolio => select_pattern_portfolio_action(
                game,
                &prelude,
                PatternAwareConfig::default(),
                &mut self.rng,
            ),
        }
    }

    fn select_greedy(
        &mut self,
        game: &GameState,
        prelude: &MarketPrelude,
    ) -> Result<TurnAction, SimulationError> {
        select_greedy_action(game, prelude, &mut self.rng)
    }
}

pub fn strategy_rng(game_seed: GameSeed, seat: usize, strategy_id: &str) -> ChaCha8Rng {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-strategy-rng");
    hasher.update(&game_seed.0);
    hasher.update(&(seat as u64).to_le_bytes());
    hasher.update(strategy_id.as_bytes());
    ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
}

pub fn select_greedy_action(
    game: &GameState,
    prelude: &MarketPrelude,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SimulationError> {
    let candidates = rank_greedy_actions(game, prelude, None)?;
    let Some(best_score) = candidates
        .first()
        .map(|candidate| candidate.resulting_base_score)
    else {
        return Err(SimulationError::NoLegalActions);
    };
    let tied = candidates
        .iter()
        .take_while(|candidate| candidate.resulting_base_score == best_score)
        .count();
    let index = rng.gen_range(0..tied);
    Ok(candidates[index].action.clone())
}

pub fn play_greedy_plies(
    game: &mut GameState,
    plies: usize,
    rng: &mut ChaCha8Rng,
) -> Result<(), SimulationError> {
    for _ in 0..plies {
        if game.is_game_over() {
            break;
        }
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let action = select_greedy_action(game, &prelude, rng)?;
        game.apply(&action)?;
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum SimulationError {
    #[error("match needs {expected} seat strategies but received {actual}")]
    SeatCount { expected: usize, actual: usize },
    #[error("strategy identities must be non-empty")]
    InvalidStrategyIdentity,
    #[error("strategy found no legal action")]
    NoLegalActions,
    #[error("strategy failed: {0}")]
    Strategy(String),
    #[error(transparent)]
    Rules(#[from] RuleError),
}

pub fn duration_per_game(results: &[MatchResult]) -> Duration {
    Duration::from_secs_f64(
        results
            .iter()
            .map(|result| result.elapsed_seconds)
            .sum::<f64>()
            / results.len().max(1) as f64,
    )
}

#[cfg(test)]
mod tests {
    use cascadia_game::{ScoringCards, ScoringVariant, WildlifeWipe};

    use super::*;

    #[test]
    fn symmetric_random_match_is_reproducible() {
        let game = GameConfig::research_aaaaa(2).unwrap();
        let config = MatchConfig::symmetric(game, GameSeed::from_u64(1), StrategyKind::Random);
        let left = play_match(&config).unwrap();
        let right = play_match(&config).unwrap();

        assert_eq!(left.scores, right.scores);
        assert_eq!(left.replay.final_state_hash, right.replay.final_state_hash);
        assert_eq!(left.turns, 40);
    }

    #[test]
    fn solo_random_match_completes_twenty_turns() {
        let config = MatchConfig::symmetric(
            GameConfig::solo(ScoringCards::AAAAA),
            GameSeed::from_u64(2),
            StrategyKind::Random,
        );
        let result = play_match(&config).unwrap();
        assert_eq!(result.turns, 20);
        assert_eq!(result.scores.len(), 1);
    }

    #[test]
    fn external_selector_preserves_each_seat_identity() {
        let game_config = GameConfig::research_aaaaa(4).unwrap();
        let seed = GameSeed::from_u64(3);
        let identities = vec![
            "r2-map-newest".to_owned(),
            "greedy-v1".to_owned(),
            "r2-map-c0".to_owned(),
            "r2-map-c1".to_owned(),
        ];
        let mut rngs: Vec<_> = identities
            .iter()
            .enumerate()
            .map(|(seat, identity)| strategy_rng(seed, seat, identity))
            .collect();
        let result = play_match_with_seat_selector(game_config, seed, &identities, |seat, game| {
            let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
            select_greedy_action(game, &prelude, &mut rngs[seat])
        })
        .unwrap();

        assert_eq!(result.strategies, identities);
        assert_eq!(result.turns, 80);
        result.replay.play().unwrap();
    }

    #[test]
    fn external_selector_rejects_missing_or_blank_seat_identity() {
        let game_config = GameConfig::research_aaaaa(4).unwrap();
        let seed = GameSeed::from_u64(4);
        let selector = |_: usize, _: &GameState| Err(SimulationError::NoLegalActions);
        assert!(matches!(
            play_match_with_seat_selector(game_config, seed, &["only-one".to_owned()], selector,),
            Err(SimulationError::SeatCount {
                expected: 4,
                actual: 1
            })
        ));
        assert!(matches!(
            play_match_with_seat_selector(
                game_config,
                seed,
                &[
                    "a".to_owned(),
                    "b".to_owned(),
                    "".to_owned(),
                    "d".to_owned(),
                ],
                selector,
            ),
            Err(SimulationError::InvalidStrategyIdentity)
        ));
    }

    #[test]
    fn ranked_greedy_candidates_are_descending_and_legal() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(10),
        )
        .unwrap();
        let prelude = MarketPrelude::default();
        let candidates = rank_greedy_actions(&game, &prelude, Some(8)).unwrap();

        assert_eq!(candidates.len(), 8);
        assert!(
            candidates
                .windows(2)
                .all(|pair| pair[0].resulting_base_score >= pair[1].resulting_base_score)
        );
        for candidate in candidates {
            game.transition(&candidate.action).unwrap();
        }
    }

    #[test]
    fn bear_setup_candidates_are_legal_and_prioritize_pair_ready_boards() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(10),
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        play_greedy_plies(&mut game, 20, &mut rng).unwrap();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let candidates = rank_bear_setup_actions(&game, &prelude, Some(8)).unwrap();

        assert!(!candidates.is_empty());
        for candidate in &candidates {
            game.transition(&candidate.action).unwrap();
        }
        let readiness: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                let board = game.preview_active_board(&candidate.action).unwrap();
                let score = score_board(&board, game.config().scoring_cards);
                (
                    score.wildlife[Wildlife::Bear as usize],
                    bear_pair_ready_slots(&board),
                    candidate.resulting_base_score,
                )
            })
            .collect();
        assert!(readiness.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn allocation_free_bear_readiness_matches_reference_on_generated_afterstates() {
        for seed in 20..24 {
            let mut game = GameState::new(
                GameConfig::research_aaaaa(4).unwrap(),
                GameSeed::from_u64(seed),
            )
            .unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed + 100);
            for _ in 0..40 {
                let prelude = MarketPrelude {
                    replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                    wildlife_wipes: Vec::new(),
                };
                let candidates = rank_greedy_actions(&game, &prelude, Some(12)).unwrap();
                for candidate in &candidates {
                    let board = game.preview_active_board(&candidate.action).unwrap();
                    assert_eq!(
                        bear_pair_ready_slots(&board),
                        bear_pair_ready_slots_reference(&board),
                    );
                }
                let action = candidates[rng.gen_range(0..candidates.len())]
                    .action
                    .clone();
                game.apply(&action).unwrap();
            }
        }
    }

    #[test]
    fn habitat_setup_candidates_are_legal_cohesive_and_tile_distinct() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(11),
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(18);
        play_greedy_plies(&mut game, 20, &mut rng).unwrap();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let candidates = rank_habitat_setup_actions(&game, &prelude, Some(8)).unwrap();

        assert!(!candidates.is_empty());
        for (index, candidate) in candidates.iter().enumerate() {
            game.transition(&candidate.action).unwrap();
            assert!(candidates[..index].iter().all(|earlier| {
                earlier.action.draft != candidate.action.draft
                    || earlier.action.tile != candidate.action.tile
            }));
        }
        let cohesion: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                let board = game.preview_active_board(&candidate.action).unwrap();
                let score = score_board(&board, game.config().scoring_cards);
                (
                    matching_habitat_edges(&board),
                    score.habitat.iter().sum::<u16>(),
                    candidate.resulting_base_score,
                )
            })
            .collect();
        assert!(cohesion.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn ranked_strategy_frontiers_exclude_dominated_same_slot_token_spends() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(11_001),
        )
        .unwrap();
        while !game.is_game_over() && game.boards()[game.current_player()].nature_tokens() == 0 {
            let prelude = MarketPrelude::default();
            let before_tokens = game.boards()[game.current_player()].nature_tokens();
            let actions = game.legal_turn_actions(&prelude).unwrap();
            let action = actions
                .iter()
                .find(|action| {
                    game.preview_active_board(action)
                        .is_ok_and(|board| board.nature_tokens() > before_tokens)
                })
                .unwrap_or(&actions[0])
                .clone();
            game.apply(&action).unwrap();
        }
        assert!(game.boards()[game.current_player()].nature_tokens() > 0);
        let prelude = MarketPrelude::default();
        let frontiers = [
            rank_greedy_actions(&game, &prelude, None).unwrap(),
            rank_bear_setup_actions(&game, &prelude, None).unwrap(),
            rank_habitat_setup_actions(&game, &prelude, None).unwrap(),
            rank_pattern_frontier_actions(&game, &prelude, PatternAwareConfig::default()).unwrap(),
        ];

        for frontier in frontiers {
            assert!(!frontier.is_empty());
            assert!(
                frontier
                    .iter()
                    .all(|candidate| !is_dominated_same_slot_independent(&candidate.action))
            );
        }
    }

    #[test]
    fn delta_rescoring_matches_full_scoring_for_every_card_family() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(12),
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        play_greedy_plies(&mut game, 16, &mut rng).unwrap();

        for variant in [
            ScoringVariant::A,
            ScoringVariant::B,
            ScoringVariant::C,
            ScoringVariant::D,
        ] {
            let cards = ScoringCards {
                bear: variant,
                elk: variant,
                salmon: variant,
                hawk: variant,
                fox: variant,
            };
            assert_delta_scores(&game, &MarketPrelude::default(), cards);
        }

        if game.boards()[game.current_player()].nature_tokens() > 0 {
            assert_delta_scores(
                &game,
                &MarketPrelude {
                    replace_three_of_a_kind: false,
                    wildlife_wipes: vec![WildlifeWipe {
                        slots: vec![cascadia_game::MarketSlot::ZERO],
                    }],
                },
                ScoringCards::AAAAA,
            );
        }
    }

    fn assert_delta_scores(game: &GameState, prelude: &MarketPrelude, cards: ScoringCards) {
        let active_board = &game.boards()[game.current_player()];
        let baseline = score_board(active_board, cards);
        let habitat = active_board.habitat_analysis();
        let mut wildlife_score_cache = HashMap::new();
        let evaluations = game
            .evaluate_legal_turn_actions_with_tile_context(
                prelude,
                |board, placement, tile| {
                    (
                        placement,
                        tile,
                        rescore_after_tile_with_habitat_analysis(
                            board, cards, baseline, &habitat, placement, tile,
                        ),
                        habitat.matching_edges_after_tile(
                            board,
                            placement.coord,
                            tile,
                            placement.rotation,
                        ),
                    )
                },
                |board, &(placement, tile, after_tile, analyzed_edges), placed_wildlife| {
                    let wildlife = placed_wildlife.map(|(wildlife, _)| wildlife);
                    (
                        rescore_after_placement(board, cards, baseline, tile, wildlife),
                        rescore_after_placement_with_habitat_analysis(
                            board, cards, baseline, &habitat, placement, tile, wildlife,
                        ),
                        wildlife.map_or(after_tile, |wildlife| {
                            rescore_after_wildlife_placement(board, cards, after_tile, wildlife)
                        }),
                        placed_wildlife.map_or(after_tile, |placed_wildlife| {
                            rescore_after_cached_wildlife_placement(
                                board,
                                cards,
                                after_tile,
                                placed_wildlife,
                                &mut wildlife_score_cache,
                            )
                        }),
                        score_board(board, cards),
                        analyzed_edges,
                        matching_habitat_edges(board),
                    )
                },
            )
            .unwrap();
        assert!(!evaluations.is_empty());
        for (_, (incremental, analyzed, split, cached, full, analyzed_edges, full_edges)) in
            evaluations
        {
            assert_eq!(incremental, full);
            assert_eq!(analyzed, full);
            assert_eq!(split, full);
            assert_eq!(cached, full);
            assert_eq!(analyzed_edges, full_edges);
        }
    }
}
