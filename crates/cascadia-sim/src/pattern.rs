use cascadia_game::{
    DraftChoice, GameState, Market, MarketPrelude, ScoringCards, TurnAction, Wildlife,
    rescore_after_tile_with_habitat_analysis, score_board,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::{
    GreedyCandidate, MatchResult, SimulationError, WildlifeScoreCache, bear_pair_ready_slots,
    is_dominated_same_slot_independent, play_match_with_selector,
    rescore_after_cached_wildlife_placement,
};

pub const PATTERN_AWARE_STRATEGY_ID: &str = "pattern-aware-v1-k8-h6-b8-m4";
pub const PATTERN_COMMITMENT_STRATEGY_ID: &str =
    "pattern-commitment-v2-k8-h6-b8-m4-t2-phase-capped";
pub const PATTERN_COMPETITION_STRATEGY_ID: &str =
    "pattern-competition-v1-k8-h6-b8-m4-t2-first-rotation";
pub const PATTERN_PORTFOLIO_STRATEGY_ID: &str =
    "pattern-portfolio-v1-k8-h6-b8-m4-t2-conditioned-premium";
pub const PATTERN_POTENTIAL_STRATEGY_ID: &str = "pattern-potential-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternAwareConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub bear_candidate_limit: usize,
    pub future_market_draws: usize,
}

impl PatternAwareConfig {
    pub fn validate(self) -> Result<Self, SimulationError> {
        if self.immediate_candidate_limit == 0
            || self.habitat_candidate_limit == 0
            || self.bear_candidate_limit == 0
        {
            return Err(SimulationError::Strategy(
                "pattern-aware candidate limits must be positive".to_owned(),
            ));
        }
        if self.future_market_draws == 0 {
            return Err(SimulationError::Strategy(
                "pattern-aware market draw count must be positive".to_owned(),
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "pattern-aware-v1-k{}-h{}-b{}-m{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.bear_candidate_limit,
            self.future_market_draws,
        )
    }
}

impl Default for PatternAwareConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
            bear_candidate_limit: 8,
            future_market_draws: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternCandidate {
    pub action: TurnAction,
    pub resulting_base_score: u16,
    pub immediate_rank: usize,
    pub future_market_opportunity: f64,
    pub matching_habitat_edge_delta: i16,
    pub bear_pair_ready_delta: i16,
    pub personal_turns_remaining: u16,
    pub heuristic_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternPotentialConfig {
    pub blueprint: PatternAwareConfig,
    pub opportunity_quarters: u8,
    pub habitat_quarters: u8,
    pub bear_quarters: u8,
}

impl PatternPotentialConfig {
    pub fn from_weights(
        blueprint: PatternAwareConfig,
        opportunity_weight: f64,
        habitat_weight: f64,
        bear_weight: f64,
    ) -> Result<Self, SimulationError> {
        Ok(Self {
            blueprint,
            opportunity_quarters: quarter_units(opportunity_weight, 2, 6, "opportunity weight")?,
            habitat_quarters: quarter_units(habitat_weight, 0, 4, "habitat weight")?,
            bear_quarters: quarter_units(bear_weight, 0, 4, "Bear weight")?,
        })
    }

    pub fn validate(self) -> Result<Self, SimulationError> {
        self.blueprint.validate()?;
        if !(2..=6).contains(&self.opportunity_quarters) {
            return Err(SimulationError::Strategy(
                "pattern-potential opportunity weight must be 0.50 to 1.50 in 0.25 increments"
                    .to_owned(),
            ));
        }
        if self.habitat_quarters > 4 || self.bear_quarters > 4 {
            return Err(SimulationError::Strategy(
                "pattern-potential structural weights must be 0.00 to 1.00 in 0.25 increments"
                    .to_owned(),
            ));
        }
        Ok(self)
    }

    pub fn opportunity_weight(self) -> f64 {
        f64::from(self.opportunity_quarters) / 4.0
    }

    pub fn habitat_weight(self) -> f64 {
        f64::from(self.habitat_quarters) / 4.0
    }

    pub fn bear_weight(self) -> f64 {
        f64::from(self.bear_quarters) / 4.0
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{PATTERN_POTENTIAL_STRATEGY_ID}-k{}-h{}-b{}-m{}-a{:03}-h{:03}-b{:03}",
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
            self.opportunity_quarters * 25,
            self.habitat_quarters * 25,
            self.bear_quarters * 25,
        )
    }
}

impl Default for PatternPotentialConfig {
    fn default() -> Self {
        Self {
            blueprint: PatternAwareConfig::default(),
            opportunity_quarters: 4,
            habitat_quarters: 0,
            bear_quarters: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternPotentialStrategy {
    config: PatternPotentialConfig,
    strategy_id: String,
}

impl PatternPotentialStrategy {
    pub fn new(config: PatternPotentialConfig) -> Result<Self, SimulationError> {
        let config = config.validate()?;
        Ok(Self {
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn select_action(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SimulationError> {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        select_pattern_potential_action(game, &prelude, self.config, rng)
    }

    pub fn play_match(
        &self,
        game_config: cascadia_game::GameConfig,
        seed: cascadia_game::GameSeed,
    ) -> Result<MatchResult, SimulationError> {
        let mut rngs = (0..usize::from(game_config.player_count))
            .map(|seat| super::strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        play_match_with_selector(game_config, seed, &self.strategy_id, |player, game| {
            self.select_action(game, &mut rngs[player])
        })
    }
}

fn quarter_units(value: f64, minimum: u8, maximum: u8, name: &str) -> Result<u8, SimulationError> {
    if !value.is_finite() {
        return Err(SimulationError::Strategy(format!(
            "pattern-potential {name} must be finite"
        )));
    }
    let scaled = value * 4.0;
    let rounded = scaled.round();
    if (scaled - rounded).abs() > 1e-9
        || rounded < f64::from(minimum)
        || rounded > f64::from(maximum)
    {
        return Err(SimulationError::Strategy(format!(
            "pattern-potential {name} is outside the registered quarter-point grid"
        )));
    }
    Ok(rounded as u8)
}

pub fn rank_pattern_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    rank_pattern_actions_with_turns(game, prelude, config, 1)
}

pub fn rank_pattern_commitment_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    rank_pattern_actions_with_turns(game, prelude, config, 2)
}

pub fn rank_pattern_competition_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    rank_pattern_actions_with_model(
        game,
        prelude,
        config,
        2,
        OpportunityModel::OpponentConditioned,
    )
}

pub fn rank_pattern_portfolio_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    rank_pattern_actions_with_model(
        game,
        prelude,
        config,
        2,
        OpportunityModel::ConditionedPremium,
    )
}

fn rank_pattern_actions_with_turns(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    future_turns: usize,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    rank_pattern_actions_with_model(
        game,
        prelude,
        config,
        future_turns,
        OpportunityModel::Optimistic,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpportunityModel {
    Optimistic,
    OpponentConditioned,
    ConditionedPremium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpportunitySpec {
    future_market_draws: usize,
    future_turns: usize,
    model: OpportunityModel,
}

#[derive(Debug, Clone, Copy)]
struct PatternEvaluationContext {
    acting_seat: usize,
    cards: ScoringCards,
    opportunity: OpportunitySpec,
    baseline_matching_habitat_edges: u16,
    baseline_bear_pair_ready_slots: u16,
}

fn rank_pattern_actions_with_model(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    future_turns: usize,
    opportunity_model: OpportunityModel,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    let mut ranked = evaluate_pattern_actions_with_model(
        game,
        prelude,
        config,
        future_turns,
        opportunity_model,
    )?;
    sort_pattern_candidates(&mut ranked);
    Ok(ranked)
}

fn evaluate_pattern_actions_with_model(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    future_turns: usize,
    opportunity_model: OpportunityModel,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    let config = config.validate()?;
    let acting_seat = game.current_player();
    let staged = game.preview_market_prelude(prelude)?;
    let baseline_board = &staged.boards()[acting_seat];
    let baseline_matching_habitat_edges = baseline_board.habitat_analysis().matching_edges();
    let baseline_bear_pair_ready_slots = bear_pair_ready_slots(baseline_board);
    let candidates = rank_pattern_frontier_actions(&staged, &MarketPrelude::default(), config)?;

    let cards = game.config().scoring_cards;
    let opportunity = OpportunitySpec {
        future_market_draws: config.future_market_draws,
        future_turns,
        model: opportunity_model,
    };
    let context = PatternEvaluationContext {
        acting_seat,
        cards,
        opportunity,
        baseline_matching_habitat_edges,
        baseline_bear_pair_ready_slots,
    };
    let mut competition = matches!(
        opportunity_model,
        OpportunityModel::OpponentConditioned | OpportunityModel::ConditionedPremium
    )
    .then(|| OpponentConditionedOpportunity::new(&staged, acting_seat, cards));
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        ranked.push(evaluate_pattern_candidate(
            &staged,
            prelude,
            context,
            candidate,
            competition.as_mut(),
        )?);
    }
    Ok(ranked)
}

fn sort_pattern_candidates(ranked: &mut [PatternCandidate]) {
    ranked.sort_by(|left, right| {
        right
            .heuristic_value
            .total_cmp(&left.heuristic_value)
            .then_with(|| right.resulting_base_score.cmp(&left.resulting_base_score))
            .then_with(|| left.immediate_rank.cmp(&right.immediate_rank))
    });
}

pub fn best_pattern_heuristic_value(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
) -> Result<Option<f64>, SimulationError> {
    Ok(
        evaluate_pattern_actions_with_model(
            game,
            prelude,
            config,
            1,
            OpportunityModel::Optimistic,
        )?
        .into_iter()
        .map(|candidate| candidate.heuristic_value)
        .max_by(f64::total_cmp),
    )
}

#[derive(Debug, Clone)]
struct PatternFrontierRecord {
    candidate: GreedyCandidate,
    drafted_wildlife: Wildlife,
    wildlife_scores: [u16; 5],
    bear_pair_ready_slots: u16,
    matching_habitat_edges: u16,
    habitat_score: u16,
}

pub fn rank_pattern_frontier_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    rank_pattern_frontier_actions_with_wildlife_coverage(game, prelude, config, None)
}

pub fn rank_wildlife_diverse_pattern_frontier_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    wildlife_candidate_limit: usize,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    if wildlife_candidate_limit == 0 {
        return Err(SimulationError::Strategy(
            "wildlife-diverse candidate limit must be positive".to_owned(),
        ));
    }
    rank_pattern_frontier_actions_with_wildlife_coverage(
        game,
        prelude,
        config,
        Some(WildlifeCoverage::All {
            limit: wildlife_candidate_limit,
        }),
    )
}

pub fn rank_wildlife_focused_pattern_frontier_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    wildlife: Wildlife,
    wildlife_candidate_limit: usize,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    if wildlife_candidate_limit == 0 {
        return Err(SimulationError::Strategy(
            "wildlife-focused candidate limit must be positive".to_owned(),
        ));
    }
    rank_pattern_frontier_actions_with_wildlife_coverage(
        game,
        prelude,
        config,
        Some(WildlifeCoverage::Focused {
            wildlife,
            limit: wildlife_candidate_limit,
        }),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WildlifeCoverage {
    All { limit: usize },
    Focused { wildlife: Wildlife, limit: usize },
}

fn rank_pattern_frontier_actions_with_wildlife_coverage(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    wildlife_coverage: Option<WildlifeCoverage>,
) -> Result<Vec<GreedyCandidate>, SimulationError> {
    let config = config.validate()?;
    if game.is_game_over() {
        return Ok(Vec::new());
    }
    let cards = game.config().scoring_cards;
    let active_board = &game.boards()[game.current_player()];
    let baseline = score_board(active_board, cards);
    let habitat = active_board.habitat_analysis();
    let staged_market = game.preview_market_prelude(prelude)?.market().clone();
    let mut wildlife_score_cache = WildlifeScoreCache::default();
    let mut records = game
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
            |board, &(after_tile, matching_habitat_edges, habitat_score), placed_wildlife| {
                let score = placed_wildlife.map_or(after_tile, |placed_wildlife| {
                    rescore_after_cached_wildlife_placement(
                        board,
                        cards,
                        after_tile,
                        placed_wildlife,
                        &mut wildlife_score_cache,
                    )
                });
                (
                    score.base_total,
                    score.wildlife,
                    bear_pair_ready_slots(board),
                    matching_habitat_edges,
                    habitat_score,
                )
            },
        )?
        .into_iter()
        .filter(|(action, _)| !is_dominated_same_slot_independent(action))
        .map(
            |(
                action,
                (
                    resulting_base_score,
                    wildlife_scores,
                    bear_pair_ready_slots,
                    matching_habitat_edges,
                    habitat_score,
                ),
            )| PatternFrontierRecord {
                drafted_wildlife: drafted_wildlife(&staged_market, action.draft)
                    .expect("evaluated legal draft has a visible wildlife token"),
                candidate: GreedyCandidate {
                    action,
                    resulting_base_score,
                    immediate_rank: 0,
                },
                wildlife_scores,
                bear_pair_ready_slots,
                matching_habitat_edges,
                habitat_score,
            },
        )
        .collect::<Vec<_>>();

    let mut immediate_order = (0..records.len()).collect::<Vec<_>>();
    immediate_order.sort_by(|left, right| {
        records[*right]
            .candidate
            .resulting_base_score
            .cmp(&records[*left].candidate.resulting_base_score)
    });
    for (rank, index) in immediate_order.into_iter().enumerate() {
        records[index].candidate.immediate_rank = rank + 1;
    }

    let mut immediate = (0..records.len()).collect::<Vec<_>>();
    immediate.sort_by(|left, right| {
        records[*right]
            .candidate
            .resulting_base_score
            .cmp(&records[*left].candidate.resulting_base_score)
    });
    immediate.truncate(config.immediate_candidate_limit);

    let mut habitat = (0..records.len()).collect::<Vec<_>>();
    habitat.sort_by(|left, right| {
        records[*right]
            .matching_habitat_edges
            .cmp(&records[*left].matching_habitat_edges)
            .then_with(|| {
                records[*right]
                    .habitat_score
                    .cmp(&records[*left].habitat_score)
            })
            .then_with(|| {
                records[*right]
                    .candidate
                    .resulting_base_score
                    .cmp(&records[*left].candidate.resulting_base_score)
            })
    });
    let mut distinct_habitat = Vec::with_capacity(config.habitat_candidate_limit);
    for index in habitat {
        let candidate = &records[index].candidate;
        if distinct_habitat.iter().any(|retained: &GreedyCandidate| {
            retained.action.draft == candidate.action.draft
                && retained.action.tile == candidate.action.tile
        }) {
            continue;
        }
        distinct_habitat.push(candidate.clone());
        if distinct_habitat.len() == config.habitat_candidate_limit {
            break;
        }
    }

    let mut bear = (0..records.len()).collect::<Vec<_>>();
    bear.sort_by(|left, right| {
        records[*right].wildlife_scores[Wildlife::Bear as usize]
            .cmp(&records[*left].wildlife_scores[Wildlife::Bear as usize])
            .then_with(|| {
                records[*right]
                    .bear_pair_ready_slots
                    .cmp(&records[*left].bear_pair_ready_slots)
            })
            .then_with(|| {
                records[*right]
                    .candidate
                    .resulting_base_score
                    .cmp(&records[*left].candidate.resulting_base_score)
            })
    });
    bear.truncate(config.bear_candidate_limit);

    let mut retained = immediate
        .into_iter()
        .map(|index| records[index].candidate.clone())
        .collect::<Vec<_>>();
    merge_unique(&mut retained, distinct_habitat);
    merge_unique(
        &mut retained,
        bear.into_iter()
            .map(|index| records[index].candidate.clone()),
    );
    if let Some(coverage) = wildlife_coverage {
        let limit = match coverage {
            WildlifeCoverage::All { limit } | WildlifeCoverage::Focused { limit, .. } => limit,
        };
        for wildlife in Wildlife::ALL.into_iter().filter(|wildlife| {
            matches!(coverage, WildlifeCoverage::All { .. })
                || matches!(
                    coverage,
                    WildlifeCoverage::Focused {
                        wildlife: focused,
                        ..
                    } if focused == *wildlife
                )
        }) {
            let mut species = records
                .iter()
                .enumerate()
                .filter(|(_, record)| record.drafted_wildlife == wildlife)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            species.sort_by(|left, right| {
                records[*right].wildlife_scores[wildlife as usize]
                    .cmp(&records[*left].wildlife_scores[wildlife as usize])
                    .then_with(|| {
                        records[*right]
                            .candidate
                            .resulting_base_score
                            .cmp(&records[*left].candidate.resulting_base_score)
                    })
                    .then_with(|| {
                        records[*left]
                            .candidate
                            .immediate_rank
                            .cmp(&records[*right].candidate.immediate_rank)
                    })
            });
            let mut distinct = Vec::with_capacity(limit);
            for index in species {
                let candidate = &records[index].candidate;
                if distinct.iter().any(|retained: &GreedyCandidate| {
                    retained.action.draft == candidate.action.draft
                        && retained.action.tile == candidate.action.tile
                }) {
                    continue;
                }
                distinct.push(candidate.clone());
                if distinct.len() == limit {
                    break;
                }
            }
            merge_unique(&mut retained, distinct);
        }
    }
    Ok(retained)
}

fn drafted_wildlife(market: &Market, draft: DraftChoice) -> Option<Wildlife> {
    match draft {
        DraftChoice::Paired { slot } => market.paired(slot).map(|(_, wildlife)| wildlife),
        DraftChoice::Independent { wildlife_slot, .. } => market.wildlife[wildlife_slot.index()],
    }
}

fn merge_unique(
    retained: &mut Vec<GreedyCandidate>,
    additional: impl IntoIterator<Item = GreedyCandidate>,
) {
    for candidate in additional {
        if !retained
            .iter()
            .any(|existing| existing.action == candidate.action)
        {
            retained.push(candidate);
        }
    }
}

pub fn select_pattern_action(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SimulationError> {
    let evaluated = evaluate_pattern_actions_with_model(
        game,
        prelude,
        config,
        1,
        OpportunityModel::Optimistic,
    )?;
    let Some(best_value) = evaluated
        .iter()
        .map(|candidate| candidate.heuristic_value)
        .max_by(f64::total_cmp)
    else {
        return Err(SimulationError::NoLegalActions);
    };
    let mut tied = evaluated
        .into_iter()
        .filter(|candidate| candidate.heuristic_value == best_value)
        .collect::<Vec<_>>();
    sort_pattern_candidates(&mut tied);
    Ok(tied[rng.gen_range(0..tied.len())].action.clone())
}

pub fn rank_pattern_potential_actions(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternPotentialConfig,
) -> Result<Vec<PatternCandidate>, SimulationError> {
    let config = config.validate()?;
    let mut evaluated = evaluate_pattern_actions_with_model(
        game,
        prelude,
        config.blueprint,
        1,
        OpportunityModel::Optimistic,
    )?;
    for candidate in &mut evaluated {
        candidate.heuristic_value = pattern_potential_value(candidate, config);
    }
    sort_pattern_candidates(&mut evaluated);
    Ok(evaluated)
}

pub fn select_pattern_potential_action(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternPotentialConfig,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SimulationError> {
    let ranked = rank_pattern_potential_actions(game, prelude, config)?;
    let Some(best) = ranked.first() else {
        return Err(SimulationError::NoLegalActions);
    };
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.heuristic_value == best.heuristic_value)
        .count();
    Ok(ranked[rng.gen_range(0..tied)].action.clone())
}

fn pattern_potential_value(candidate: &PatternCandidate, config: PatternPotentialConfig) -> f64 {
    let phase = f64::from(candidate.personal_turns_remaining) / 19.0;
    f64::from(candidate.resulting_base_score)
        + config.opportunity_weight() * candidate.future_market_opportunity
        + phase
            * (config.habitat_weight() * f64::from(candidate.matching_habitat_edge_delta)
                + config.bear_weight() * f64::from(candidate.bear_pair_ready_delta))
}

pub fn select_pattern_commitment_action(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SimulationError> {
    let ranked = rank_pattern_commitment_actions(game, prelude, config)?;
    let Some(best) = ranked.first() else {
        return Err(SimulationError::NoLegalActions);
    };
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.heuristic_value == best.heuristic_value)
        .count();
    Ok(ranked[rng.gen_range(0..tied)].action.clone())
}

pub fn select_pattern_competition_action(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SimulationError> {
    let ranked = rank_pattern_competition_actions(game, prelude, config)?;
    let Some(best) = ranked.first() else {
        return Err(SimulationError::NoLegalActions);
    };
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.heuristic_value == best.heuristic_value)
        .count();
    Ok(ranked[rng.gen_range(0..tied)].action.clone())
}

pub fn select_pattern_portfolio_action(
    game: &GameState,
    prelude: &MarketPrelude,
    config: PatternAwareConfig,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SimulationError> {
    let ranked = rank_pattern_portfolio_actions(game, prelude, config)?;
    let Some(best) = ranked.first() else {
        return Err(SimulationError::NoLegalActions);
    };
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.heuristic_value == best.heuristic_value)
        .count();
    Ok(ranked[rng.gen_range(0..tied)].action.clone())
}

pub fn play_pattern_plies(
    game: &mut GameState,
    plies: usize,
    config: PatternAwareConfig,
    rng: &mut ChaCha8Rng,
) -> Result<(), SimulationError> {
    for _ in 0..plies {
        if game.is_game_over() {
            break;
        }
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let action = select_pattern_action(game, &prelude, config, rng)?;
        game.apply(&action)?;
    }
    Ok(())
}

fn evaluate_pattern_candidate(
    staged: &GameState,
    prelude: &MarketPrelude,
    context: PatternEvaluationContext,
    candidate: GreedyCandidate,
    competition: Option<&mut OpponentConditionedOpportunity>,
) -> Result<PatternCandidate, SimulationError> {
    let public_afterstate = staged.preview_public_afterstate(&candidate.action)?;
    let remaining_turns =
        usize::from(public_afterstate.turns_remaining_for_player(context.acting_seat));
    let after_board = &public_afterstate.boards()[context.acting_seat];
    let matching_habitat_edge_delta = after_board
        .habitat_analysis()
        .matching_edges()
        .cast_signed()
        - context.baseline_matching_habitat_edges.cast_signed();
    let bear_pair_ready_delta = bear_pair_ready_slots(after_board).cast_signed()
        - context.baseline_bear_pair_ready_slots.cast_signed();
    let realizable_future_turns = context.opportunity.future_turns.min(remaining_turns);
    let future_market_opportunity = if realizable_future_turns == 0 {
        0.0
    } else {
        match context.opportunity.model {
            OpportunityModel::Optimistic => future_wildlife_opportunity(
                after_board,
                context.cards,
                public_afterstate.unplaced_wildlife_counts(),
                context.opportunity.future_market_draws,
                realizable_future_turns,
            ),
            OpportunityModel::OpponentConditioned => competition
                .expect("opponent-conditioned ranking creates an opportunity evaluator")
                .evaluate(
                    &public_afterstate,
                    context.acting_seat,
                    context.cards,
                    context.opportunity.future_market_draws,
                    realizable_future_turns,
                )?,
            OpportunityModel::ConditionedPremium => competition
                .expect("conditioned-premium ranking creates an opportunity evaluator")
                .evaluate_premium(
                    &public_afterstate,
                    context.acting_seat,
                    context.cards,
                    context.opportunity.future_market_draws,
                    realizable_future_turns,
                )?,
        }
    };
    let mut action = candidate.action;
    action.replace_three_of_a_kind = prelude.replace_three_of_a_kind;
    action.wildlife_wipes = prelude.wildlife_wipes.clone();
    Ok(PatternCandidate {
        action,
        resulting_base_score: candidate.resulting_base_score,
        immediate_rank: candidate.immediate_rank,
        future_market_opportunity,
        matching_habitat_edge_delta,
        bear_pair_ready_delta,
        personal_turns_remaining: remaining_turns as u16,
        heuristic_value: f64::from(candidate.resulting_base_score) + future_market_opportunity,
    })
}

#[path = "pattern_opportunity.rs"]
mod opportunity;

use opportunity::OpponentConditionedOpportunity;
#[cfg(test)]
use opportunity::{
    ReplacementKernel, WildlifeMarketState, draw_allocations, expected_market_value,
    expected_max_without_replacement, market_token_total, opponent_draft_choices,
    terminal_market_distribution,
};
pub use opportunity::{
    future_market_opportunity, future_wildlife_opportunity, wildlife_marginal_gains,
};

#[cfg(test)]
#[path = "pattern_tests.rs"]
mod tests;
