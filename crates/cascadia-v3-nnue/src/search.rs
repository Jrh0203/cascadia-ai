use std::{
    cmp::Ordering,
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    time::Instant,
};

use cascadia_data::OpportunityGraphBuildContext;
use cascadia_game::{
    DraftChoice, GameSeed, GameState, MarketPrelude, MarketSlot, TilePlacement, TurnAction,
    score_board, score_game,
};
use cascadia_sim::{select_greedy_action, strategy_rng};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    InferenceBackend, PreparedOpportunityEvaluation, QuantizedV3Model, Result, V3AccumulatorStack,
    V3Error, V3FeatureContext, encode_public_features,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum V3SearchBudget {
    Direct,
    K32R64,
    K32R600,
    SequentialHalving { candidates: usize, rollouts: usize },
}

impl V3SearchBudget {
    pub fn parameters(self) -> Option<(usize, usize)> {
        match self {
            Self::Direct => None,
            Self::K32R64 => Some((32, 64)),
            Self::K32R600 => Some((32, 600)),
            Self::SequentialHalving {
                candidates,
                rollouts,
            } => Some((candidates, rollouts)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalRolloutConfig {
    pub model_guided: bool,
    pub maximum_plies: Option<u16>,
}

impl Default for TerminalRolloutConfig {
    fn default() -> Self {
        Self {
            model_guided: true,
            maximum_plies: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedV3Action {
    pub action: TurnAction,
    pub direct_raw_units: i32,
    pub direct_score: f32,
    pub rollout_mean: Option<f64>,
    pub rollout_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3TeacherCandidateEstimate {
    pub action: TurnAction,
    pub direct_raw_units: i32,
    pub rollout_mean: f64,
    pub rollout_variance: f64,
    pub rollout_count: u32,
    pub rank: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3TeacherRootLabel {
    pub state_blake3: [u8; 32],
    pub focal_seat: u8,
    pub phase_bucket: u8,
    pub candidate_limit: u8,
    pub rollout_budget: u16,
    pub selected_action: TurnAction,
    pub candidates: Vec<V3TeacherCandidateEstimate>,
    pub rng_domain: String,
}

impl V3TeacherRootLabel {
    pub fn validate(&self) -> Result<()> {
        if self.focal_seat >= 4
            || self.phase_bucket >= 8
            || self.candidates.is_empty()
            || self.candidates.len() > usize::from(self.candidate_limit)
            || self
                .candidates
                .iter()
                .map(|candidate| candidate.rollout_count as usize)
                .sum::<usize>()
                != usize::from(self.rollout_budget)
            || self.candidates.iter().any(|candidate| {
                candidate.rollout_count == 0
                    || !candidate.rollout_mean.is_finite()
                    || !candidate.rollout_variance.is_finite()
                    || candidate.rollout_variance < 0.0
                    || candidate.rank == 0
                    || usize::from(candidate.rank) > self.candidates.len()
            })
            || self
                .candidates
                .iter()
                .filter(|candidate| candidate.rank == 1)
                .count()
                != 1
            || self
                .candidates
                .iter()
                .find(|candidate| candidate.rank == 1)
                .is_none_or(|candidate| candidate.action != self.selected_action)
        {
            return Err(V3Error::InvalidTraining(
                "teacher root label is internally inconsistent".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3RankProfile {
    pub legal_action_count: usize,
    pub legal_action_enumeration_seconds: f64,
    pub public_context_seconds: f64,
    pub board_apply_undo_seconds: f64,
    pub habitat_prepare_seconds: f64,
    pub wildlife_prepare_seconds: f64,
    pub matching_prepare_seconds: f64,
    pub anchor_prepare_seconds: f64,
    pub field_accumulator_prepare_seconds: f64,
    pub own_accumulator_prepare_seconds: f64,
    pub candidate_scoring_wall_seconds: f64,
    pub feature_and_inference_wall_seconds: f64,
    pub summed_board_copy_cpu_seconds: f64,
    pub summed_feature_delta_cpu_seconds: f64,
    pub summed_accumulator_cpu_seconds: f64,
    pub summed_dense_evaluation_cpu_seconds: f64,
    pub top32_sort_seconds: f64,
    pub total_seconds: f64,
}

pub struct V3SearchPolicy<'a> {
    model: &'a QuantizedV3Model,
    backend: InferenceBackend,
}

impl<'a> V3SearchPolicy<'a> {
    pub fn new(model: &'a QuantizedV3Model, backend: InferenceBackend) -> Result<Self> {
        model.validate()?;
        Ok(Self { model, backend })
    }

    pub fn rank_legal_actions(&self, game: &GameState) -> Result<Vec<RankedV3Action>> {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        self.rank_legal_actions_with_prelude(game, &prelude)
    }

    pub fn rank_legal_actions_with_prelude(
        &self,
        game: &GameState,
        prelude: &MarketPrelude,
    ) -> Result<Vec<RankedV3Action>> {
        Ok(self
            .rank_legal_actions_with_prelude_internal(game, prelude, false)?
            .0)
    }

    pub fn rank_legal_actions_profiled(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RankedV3Action>, V3RankProfile)> {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        self.rank_legal_actions_with_prelude_internal(game, &prelude, true)
    }

    fn rank_legal_actions_with_prelude_internal(
        &self,
        game: &GameState,
        prelude: &MarketPrelude,
        collect_profile: bool,
    ) -> Result<(Vec<RankedV3Action>, V3RankProfile)> {
        let profile_started = Instant::now();
        let focal = game.current_player();
        let stage_started = Instant::now();
        let actions = game.legal_turn_actions(prelude)?;
        let legal_seconds = stage_started.elapsed().as_secs_f64();
        if actions.is_empty() {
            return Err(V3Error::InvalidFeature(
                "V3 policy received a nonterminal state with no legal actions".to_owned(),
            ));
        }
        let staged = game.preview_market_prelude(prelude)?;
        let context = V3FeatureContext::new(&staged.public_state(), focal)?;
        let stage_started = Instant::now();
        let public_representatives = actions
            .iter()
            .map(|action| ((action.draft, action.wildlife.is_some()), action.clone()))
            .collect::<HashMap<_, _>>();
        let prepared_public = public_representatives
            .into_par_iter()
            .map(|(key, action)| -> Result<_> {
                let afterstate = game.preview_public_afterstate(&action)?;
                Ok((key, afterstate))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut public_template_by_draft = HashMap::with_capacity(prepared_public.len());
        for (key, afterstate) in prepared_public {
            public_template_by_draft.insert(key, afterstate);
        }
        let field_representatives = actions
            .iter()
            .map(|action| {
                (
                    draft_market_key(action.draft),
                    (action.draft, action.wildlife.is_some()),
                )
            })
            .collect::<HashMap<_, _>>();
        let prepared_market = field_representatives
            .into_par_iter()
            .map(|(market_key, context_key)| -> Result<_> {
                let afterstate = public_template_by_draft.get(&context_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing field representative".to_owned())
                })?;
                Ok((
                    market_key,
                    context.field_opportunities(afterstate)?,
                    OpportunityGraphBuildContext::new(afterstate, focal)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut field_by_market = HashMap::with_capacity(prepared_market.len());
        let mut opportunity_context_by_market = HashMap::with_capacity(prepared_market.len());
        for (market_key, field, opportunity_context) in prepared_market {
            field_by_market.insert(market_key, field);
            opportunity_context_by_market.insert(market_key, opportunity_context);
        }
        let template_seconds = stage_started.elapsed().as_secs_f64();
        let drafts = actions
            .iter()
            .map(|action| action.draft)
            .collect::<std::collections::HashSet<_>>();
        let mut boards_by_action = HashMap::with_capacity(actions.len());
        let mut wildlife_by_draft = HashMap::with_capacity(drafts.len());
        let stage_started = Instant::now();
        for draft in drafts {
            let wildlife_slot = match draft {
                DraftChoice::Paired { slot } => slot,
                DraftChoice::Independent { wildlife_slot, .. } => wildlife_slot,
            };
            let wildlife = staged.market().wildlife[wildlife_slot.index()].ok_or_else(|| {
                V3Error::InvalidFeature("legal draft is missing its wildlife token".to_owned())
            })?;
            wildlife_by_draft.insert(draft, wildlife);
            for (action, board) in
                game.evaluate_legal_draft_actions(prelude, draft, |board| board.clone())?
            {
                boards_by_action.insert(action, board);
            }
        }
        if boards_by_action.len() != actions.len() {
            return Err(V3Error::InvalidFeature(
                "in-place legal-action traversal disagrees with exhaustive action set".to_owned(),
            ));
        }
        let board_enumeration_seconds = stage_started.elapsed().as_secs_f64();
        let stage_started = Instant::now();
        let print_profile = std::env::var_os("CASCADIA_V3_PROFILE_STAGES").is_some();
        let detailed_profile = collect_profile || print_profile;
        let replace_ns = AtomicU64::new(0);
        let feature_ns = AtomicU64::new(0);
        let accumulator_ns = AtomicU64::new(0);
        let evaluate_ns = AtomicU64::new(0);
        let mut tile_sibling_has_wildlife = HashMap::new();
        for action in &actions {
            tile_sibling_has_wildlife
                .entry(tile_sibling_key(action))
                .and_modify(|has_wildlife| *has_wildlife |= action.wildlife.is_some())
                .or_insert(action.wildlife.is_some());
        }
        let tile_sibling_keys = tile_sibling_has_wildlife
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let substage_started = Instant::now();
        let habitat_representatives = tile_sibling_keys
            .iter()
            .map(|key| (habitat_sibling_key(key), key.clone()))
            .collect::<HashMap<_, _>>();
        let habitat_by_tile = habitat_representatives
            .par_iter()
            .map(|(habitat_key, representative)| -> Result<_> {
                let board = boards_by_action.get(representative).ok_or_else(|| {
                    V3Error::InvalidFeature("missing tile-sibling parent board".to_owned())
                })?;
                let opportunity_context = opportunity_context_by_market
                    .get(&draft_market_key(representative.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing tile opportunity context".to_owned())
                    })?;
                let habitat =
                    context.habitat_opportunity_graph_with_context(board, opportunity_context)?;
                Ok((*habitat_key, habitat))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let habitat_prepare_seconds = substage_started.elapsed().as_secs_f64();
        let substage_started = Instant::now();
        let wildlife_demands_by_tile = habitat_by_tile
            .par_iter()
            .map(|(habitat_key, _)| -> Result<_> {
                let representative = habitat_representatives.get(habitat_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing habitat representative".to_owned())
                })?;
                let board = boards_by_action.get(representative).ok_or_else(|| {
                    V3Error::InvalidFeature("missing wildlife-demand tile board".to_owned())
                })?;
                let opportunity_context = opportunity_context_by_market
                    .get(&draft_market_key(representative.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing wildlife-demand context".to_owned())
                    })?;
                Ok((
                    *habitat_key,
                    opportunity_context.prepare_wildlife_demands(board)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let wildlife_prepare_seconds = substage_started.elapsed().as_secs_f64();
        let substage_started = Instant::now();
        let matching_by_habitat = habitat_by_tile
            .par_iter()
            .map(|(habitat_key, habitat)| -> Result<_> {
                let representative = habitat_representatives.get(habitat_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing matching representative".to_owned())
                })?;
                let opportunity_context = opportunity_context_by_market
                    .get(&draft_market_key(representative.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing habitat matching context".to_owned())
                    })?;
                Ok((
                    *habitat_key,
                    opportunity_context.prepare_habitat_matching_frontier(habitat)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let matching_by_wildlife = tile_sibling_keys
            .par_iter()
            .map(|key| -> Result<_> {
                let habitat_key = habitat_sibling_key(key);
                let opportunity_context = opportunity_context_by_market
                    .get(&draft_market_key(key.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing wildlife matching context".to_owned())
                    })?;
                let board = boards_by_action.get(key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing wildlife matching board".to_owned())
                })?;
                Ok((
                    key.clone(),
                    opportunity_context.prepare_wildlife_matching_frontiers(
                        board,
                        wildlife_demands_by_tile.get(&habitat_key).ok_or_else(|| {
                            V3Error::InvalidFeature("missing wildlife matching demands".to_owned())
                        })?,
                    )?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let matching_prepare_seconds = substage_started.elapsed().as_secs_f64();
        let substage_started = Instant::now();
        let anchor_key = tile_sibling_keys.iter().next().ok_or_else(|| {
            V3Error::InvalidFeature("legal action set has no tile sibling".to_owned())
        })?;
        let anchor_board = boards_by_action
            .get(anchor_key)
            .ok_or_else(|| V3Error::InvalidFeature("missing anchor afterstate board".to_owned()))?;
        let anchor_context_key = (
            anchor_key.draft,
            *tile_sibling_has_wildlife
                .get(anchor_key)
                .ok_or_else(|| V3Error::InvalidFeature("missing anchor sibling kind".to_owned()))?,
        );
        let anchor_template = public_template_by_draft
            .get(&anchor_context_key)
            .ok_or_else(|| V3Error::InvalidFeature("missing anchor template".to_owned()))?;
        let anchor_afterstate = anchor_template.with_replaced_board(focal, anchor_board.clone())?;
        let anchor_habitat_key = habitat_sibling_key(anchor_key);
        let anchor_features = context.encode_afterstate_board_with_cached_habitat(
            &anchor_afterstate,
            anchor_board,
            PreparedOpportunityEvaluation::new(
                opportunity_context_by_market
                    .get(&draft_market_key(anchor_key.draft))
                    .ok_or_else(|| V3Error::InvalidFeature("missing anchor context".to_owned()))?,
                habitat_by_tile
                    .get(&anchor_habitat_key)
                    .ok_or_else(|| V3Error::InvalidFeature("missing anchor habitat".to_owned()))?,
                wildlife_demands_by_tile
                    .get(&anchor_habitat_key)
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing anchor wildlife demands".to_owned())
                    })?,
                matching_by_habitat
                    .get(&anchor_habitat_key)
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing anchor habitat matching".to_owned())
                    })?,
                matching_by_wildlife.get(anchor_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing anchor wildlife matching".to_owned())
                })?,
            ),
            None,
            field_by_market
                .get(&draft_market_key(anchor_key.draft))
                .cloned()
                .ok_or_else(|| V3Error::InvalidFeature("missing anchor field".to_owned()))?,
        )?;
        let root_accumulator = V3AccumulatorStack::new(self.model, anchor_features, self.backend)?;
        let anchor_prepare_seconds = substage_started.elapsed().as_secs_f64();
        let substage_started = Instant::now();
        let field_accumulator_by_market = field_by_market
            .iter()
            .map(|(key, opportunities)| {
                Ok((
                    *key,
                    root_accumulator.prepare_field_fork(self.model, opportunities, self.backend)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let field_accumulator_prepare_seconds = substage_started.elapsed().as_secs_f64();
        let substage_started = Instant::now();
        let prepared_own_by_tile = tile_sibling_keys
            .par_iter()
            .map(|key| -> Result<_> {
                let board = boards_by_action.get(key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing prepared tile board".to_owned())
                })?;
                let context_key = (
                    key.draft,
                    *tile_sibling_has_wildlife.get(key).ok_or_else(|| {
                        V3Error::InvalidFeature("missing prepared sibling kind".to_owned())
                    })?,
                );
                let template = public_template_by_draft.get(&context_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing prepared tile template".to_owned())
                })?;
                let afterstate = template.with_replaced_board(focal, board.clone())?;
                let opportunity_context = opportunity_context_by_market
                    .get(&draft_market_key(key.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing prepared tile context".to_owned())
                    })?;
                let field = field_by_market
                    .get(&draft_market_key(key.draft))
                    .cloned()
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing prepared tile field".to_owned())
                    })?;
                let habitat_key = habitat_sibling_key(key);
                let habitat = habitat_by_tile.get(&habitat_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing prepared habitat graph".to_owned())
                })?;
                let features = context.encode_afterstate_board_with_cached_habitat(
                    &afterstate,
                    board,
                    PreparedOpportunityEvaluation::new(
                        opportunity_context,
                        habitat,
                        wildlife_demands_by_tile.get(&habitat_key).ok_or_else(|| {
                            V3Error::InvalidFeature("missing prepared wildlife demands".to_owned())
                        })?,
                        matching_by_habitat.get(&habitat_key).ok_or_else(|| {
                            V3Error::InvalidFeature("missing prepared habitat matching".to_owned())
                        })?,
                        matching_by_wildlife.get(key).ok_or_else(|| {
                            V3Error::InvalidFeature("missing prepared wildlife matching".to_owned())
                        })?,
                    ),
                    None,
                    field,
                )?;
                let prepared =
                    root_accumulator.prepare_own_fork(self.model, &features, self.backend)?;
                Ok((key.clone(), prepared))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let own_accumulator_prepare_seconds = substage_started.elapsed().as_secs_f64();
        let candidate_started = Instant::now();
        let mut ranked = actions
            .into_par_iter()
            .map(|action| -> Result<RankedV3Action> {
                let detail_started = Instant::now();
                let key = (action.draft, action.wildlife.is_some());
                let board = boards_by_action.get(&action).ok_or_else(|| {
                    V3Error::InvalidFeature("missing in-place afterstate board".to_owned())
                })?;
                let template = public_template_by_draft.get(&key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing public afterstate template".to_owned())
                })?;
                if detailed_profile {
                    replace_ns.fetch_add(
                        detail_started.elapsed().as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                }
                let opportunity_context = opportunity_context_by_market
                    .get(&draft_market_key(action.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing opportunity build context".to_owned())
                    })?;
                let detail_started = Instant::now();
                let sibling_key = tile_sibling_key(&action);
                let prepared_own = prepared_own_by_tile.get(&sibling_key).ok_or_else(|| {
                    V3Error::InvalidFeature("missing prepared own accumulator".to_owned())
                })?;
                let features = if let Some(wildlife_coord) = action.wildlife {
                    let habitat_key = habitat_sibling_key(&action);
                    let habitat = habitat_by_tile.get(&habitat_key).ok_or_else(|| {
                        V3Error::InvalidFeature("missing cached habitat opportunities".to_owned())
                    })?;
                    let placed_wildlife = (
                        *wildlife_by_draft
                            .get(&action.draft)
                            .expect("all legal drafts have wildlife"),
                        wildlife_coord,
                    );
                    Some(context.encode_wildlife_sibling_own_with_cached_habitat(
                        template,
                        boards_by_action.get(&sibling_key).ok_or_else(|| {
                            V3Error::InvalidFeature("missing wildlife sibling board".to_owned())
                        })?,
                        board,
                        prepared_own.own_base_features(),
                        PreparedOpportunityEvaluation::new(
                            opportunity_context,
                            habitat,
                            wildlife_demands_by_tile.get(&habitat_key).ok_or_else(|| {
                                V3Error::InvalidFeature(
                                    "missing prepared wildlife demands".to_owned(),
                                )
                            })?,
                            matching_by_habitat.get(&habitat_key).ok_or_else(|| {
                                V3Error::InvalidFeature(
                                    "missing prepared habitat matching".to_owned(),
                                )
                            })?,
                            matching_by_wildlife.get(&sibling_key).ok_or_else(|| {
                                V3Error::InvalidFeature(
                                    "missing prepared wildlife matching".to_owned(),
                                )
                            })?,
                        ),
                        placed_wildlife,
                    )?)
                } else {
                    None
                };
                if detailed_profile {
                    feature_ns.fetch_add(
                        detail_started.elapsed().as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                }
                let detail_started = Instant::now();
                let prepared_field = field_accumulator_by_market
                    .get(&draft_market_key(action.draft))
                    .ok_or_else(|| {
                        V3Error::InvalidFeature("missing prepared field accumulator".to_owned())
                    })?;
                let evaluation = features.as_ref().map_or_else(
                    || prepared_own.evaluate(self.model, prepared_field),
                    |features| {
                        prepared_own.evaluate_fork(
                            self.model,
                            features,
                            prepared_field,
                            self.backend,
                        )
                    },
                )?;
                if detailed_profile {
                    accumulator_ns.fetch_add(
                        detail_started.elapsed().as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                }
                let detail_started = Instant::now();
                let immediate =
                    i32::from(score_board(board, template.config().scoring_cards).base_total);
                let direct_raw_units = evaluation
                    .raw_output_units
                    .checked_add(immediate * self.model.scales.output)
                    .ok_or(V3Error::AccumulatorOverflow)?;
                if detailed_profile {
                    evaluate_ns.fetch_add(
                        detail_started.elapsed().as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                }
                Ok(RankedV3Action {
                    action,
                    direct_raw_units,
                    direct_score: direct_raw_units as f32 / self.model.scales.output as f32,
                    rollout_mean: None,
                    rollout_count: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let candidate_scoring_wall_seconds = candidate_started.elapsed().as_secs_f64();
        let feature_inference_seconds = stage_started.elapsed().as_secs_f64();
        let stage_started = Instant::now();
        ranked.sort_by(canonical_rank_order);
        let sort_seconds = stage_started.elapsed().as_secs_f64();
        let profile = V3RankProfile {
            legal_action_count: ranked.len(),
            legal_action_enumeration_seconds: legal_seconds,
            public_context_seconds: template_seconds,
            board_apply_undo_seconds: board_enumeration_seconds,
            habitat_prepare_seconds,
            wildlife_prepare_seconds,
            matching_prepare_seconds,
            anchor_prepare_seconds,
            field_accumulator_prepare_seconds,
            own_accumulator_prepare_seconds,
            candidate_scoring_wall_seconds,
            feature_and_inference_wall_seconds: feature_inference_seconds,
            summed_board_copy_cpu_seconds: replace_ns.load(AtomicOrdering::Relaxed) as f64 / 1e9,
            summed_feature_delta_cpu_seconds: feature_ns.load(AtomicOrdering::Relaxed) as f64 / 1e9,
            summed_accumulator_cpu_seconds: accumulator_ns.load(AtomicOrdering::Relaxed) as f64
                / 1e9,
            summed_dense_evaluation_cpu_seconds: evaluate_ns.load(AtomicOrdering::Relaxed) as f64
                / 1e9,
            top32_sort_seconds: sort_seconds,
            total_seconds: profile_started.elapsed().as_secs_f64(),
        };
        if print_profile {
            eprintln!(
                "V3_STAGE_PROFILE {}",
                serde_json::json!({
                    "profile": &profile,
                })
            );
        }
        Ok((ranked, profile))
    }

    /// Correct intentionally unoptimized baseline retained for performance
    /// regression measurement. It rebuilds every public perspective for each
    /// action and must remain output-identical to the production ranker.
    pub fn rank_legal_actions_reference(&self, game: &GameState) -> Result<Vec<RankedV3Action>> {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let focal = game.current_player();
        let actions = game.legal_turn_actions(&prelude)?;
        let mut ranked = Vec::with_capacity(actions.len());
        for action in actions {
            let afterstate = game.preview_public_afterstate(&action)?;
            let features = encode_public_features(&afterstate, focal)?;
            let accumulator = V3AccumulatorStack::new(self.model, features, self.backend)?;
            let evaluation = accumulator.evaluate(self.model)?;
            let immediate = i32::from(
                score_board(
                    &afterstate.boards()[focal],
                    afterstate.config().scoring_cards,
                )
                .base_total,
            );
            let direct_raw_units = evaluation
                .raw_output_units
                .checked_add(immediate * self.model.scales.output)
                .ok_or(V3Error::AccumulatorOverflow)?;
            ranked.push(RankedV3Action {
                action,
                direct_raw_units,
                direct_score: direct_raw_units as f32 / self.model.scales.output as f32,
                rollout_mean: None,
                rollout_count: 0,
            });
        }
        ranked.sort_by(canonical_rank_order);
        Ok(ranked)
    }

    pub fn select_action(
        &self,
        game: &GameState,
        budget: V3SearchBudget,
        rollout: TerminalRolloutConfig,
    ) -> Result<TurnAction> {
        let ranked = self.rank_legal_actions(game)?;
        let Some((candidate_count, rollout_count)) = budget.parameters() else {
            return Ok(ranked[0].action.clone());
        };
        self.select_terminal_parallel(game, ranked, candidate_count, rollout_count, rollout)
            .map(|candidate| candidate.action)
    }

    /// Produce a replayable teacher packet containing every top-K candidate's
    /// mean, sample count, variance, and final rank. Sequential halving still
    /// spends exactly the declared shared budget; eliminated candidates remain
    /// in the packet instead of disappearing from the training provenance.
    pub fn label_teacher_root(
        &self,
        game: &GameState,
        budget: V3SearchBudget,
        config: TerminalRolloutConfig,
    ) -> Result<V3TeacherRootLabel> {
        let (candidate_limit, rollout_budget) = budget.parameters().ok_or_else(|| {
            V3Error::InvalidTraining("teacher labeling requires a rollout budget".to_owned())
        })?;
        if candidate_limit == 0
            || candidate_limit > 32
            || rollout_budget == 0
            || rollout_budget < candidate_limit
            || rollout_budget > u16::MAX as usize
        {
            return Err(V3Error::InvalidTraining(
                "teacher K/R budget is outside the V3 contract".to_owned(),
            ));
        }
        let mut ranked = self.rank_legal_actions(game)?;
        ranked.truncate(candidate_limit.min(ranked.len()));
        let mut sums = vec![0.0f64; ranked.len()];
        let mut sums_squared = vec![0.0f64; ranked.len()];
        let mut counts = vec![0u32; ranked.len()];
        let mut active = (0..ranked.len()).collect::<Vec<_>>();
        let mut sample = 0u64;
        let mut remaining = rollout_budget;
        while active.len() > 1 && remaining > 0 {
            let rounds_left = active.len().next_power_of_two().ilog2().max(1) as usize;
            let per_candidate = (remaining / (active.len() * rounds_left)).max(1);
            let mut tasks = Vec::with_capacity((active.len() * per_candidate).min(remaining));
            for &index in &active {
                for _ in 0..per_candidate {
                    if tasks.len() >= remaining {
                        break;
                    }
                    tasks.push((index, sample));
                    sample += 1;
                }
            }
            let values = tasks
                .par_iter()
                .map(|(index, sample)| {
                    self.terminal_rollout(game, &ranked[*index].action, *sample, config)
                })
                .collect::<Result<Vec<_>>>()?;
            for ((index, _), value) in tasks.iter().zip(values) {
                sums[*index] += value;
                sums_squared[*index] += value * value;
                counts[*index] += 1;
            }
            remaining -= tasks.len();
            active.sort_by(|&left, &right| {
                mean(sums[right], counts[right])
                    .total_cmp(&mean(sums[left], counts[left]))
                    .then_with(|| canonical_rank_order(&ranked[left], &ranked[right]))
            });
            active.truncate(active.len().div_ceil(2));
        }
        if remaining > 0 {
            let winner = active[0];
            let start = sample;
            let values = (0..remaining)
                .into_par_iter()
                .map(|offset| {
                    self.terminal_rollout(
                        game,
                        &ranked[winner].action,
                        start + offset as u64,
                        config,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            for value in values {
                sums[winner] += value;
                sums_squared[winner] += value * value;
                counts[winner] += 1;
            }
        }
        let mut order = (0..ranked.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            mean(sums[right], counts[right])
                .total_cmp(&mean(sums[left], counts[left]))
                .then_with(|| canonical_rank_order(&ranked[left], &ranked[right]))
        });
        let mut final_rank = vec![0u8; ranked.len()];
        for (position, index) in order.iter().copied().enumerate() {
            final_rank[index] = (position + 1) as u8;
        }
        let candidates = ranked
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let count = counts[index];
                let variance = if count > 1 {
                    ((sums_squared[index] - sums[index] * sums[index] / f64::from(count))
                        / f64::from(count - 1))
                    .max(0.0)
                } else {
                    0.0
                };
                V3TeacherCandidateEstimate {
                    action: candidate.action.clone(),
                    direct_raw_units: candidate.direct_raw_units,
                    rollout_mean: mean(sums[index], count),
                    rollout_variance: variance,
                    rollout_count: count,
                    rank: final_rank[index],
                }
            })
            .collect::<Vec<_>>();
        let focal = game.current_player();
        let completed = game.boards()[focal].tile_count().saturating_sub(3).min(20);
        let label = V3TeacherRootLabel {
            state_blake3: *game.public_state().canonical_hash().as_bytes(),
            focal_seat: focal as u8,
            phase_bucket: ((8 * completed) / 20).min(7) as u8,
            candidate_limit: candidate_limit as u8,
            rollout_budget: rollout_budget as u16,
            selected_action: ranked[order[0]].action.clone(),
            candidates,
            rng_domain: "cascadia-v3-terminal-rollout-v1".to_owned(),
        };
        label.validate()?;
        Ok(label)
    }

    fn select_terminal_parallel(
        &self,
        game: &GameState,
        mut ranked: Vec<RankedV3Action>,
        candidate_count: usize,
        rollout_budget: usize,
        config: TerminalRolloutConfig,
    ) -> Result<RankedV3Action> {
        if candidate_count == 0 || rollout_budget == 0 {
            return Err(V3Error::InvalidFeature(
                "sequential halving needs positive candidates and rollouts".to_owned(),
            ));
        }
        ranked.truncate(candidate_count.min(ranked.len()));
        let mut sums = vec![0.0f64; ranked.len()];
        let mut counts = vec![0u32; ranked.len()];
        let mut sample = 0u64;
        let mut remaining = rollout_budget;
        while ranked.len() > 1 && remaining > 0 {
            let rounds_left = ranked.len().next_power_of_two().ilog2().max(1) as usize;
            let per_candidate = (remaining / (ranked.len() * rounds_left)).max(1);
            let mut tasks = Vec::with_capacity((ranked.len() * per_candidate).min(remaining));
            for index in 0..ranked.len() {
                for _ in 0..per_candidate {
                    if tasks.len() >= remaining {
                        break;
                    }
                    tasks.push((index, sample));
                    sample += 1;
                }
            }
            let values = tasks
                .par_iter()
                .map(|(index, sample)| {
                    self.terminal_rollout(game, &ranked[*index].action, *sample, config)
                })
                .collect::<Result<Vec<_>>>()?;
            for ((index, _), value) in tasks.iter().zip(values) {
                sums[*index] += value;
                counts[*index] += 1;
            }
            remaining -= tasks.len();
            let mut order = (0..ranked.len()).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                mean(sums[right], counts[right])
                    .total_cmp(&mean(sums[left], counts[left]))
                    .then_with(|| canonical_rank_order(&ranked[left], &ranked[right]))
            });
            let retained = ranked.len().div_ceil(2);
            let mut next_ranked = Vec::with_capacity(retained);
            let mut next_sums = Vec::with_capacity(retained);
            let mut next_counts = Vec::with_capacity(retained);
            for index in order.into_iter().take(retained) {
                next_ranked.push(ranked[index].clone());
                next_sums.push(sums[index]);
                next_counts.push(counts[index]);
            }
            ranked = next_ranked;
            sums = next_sums;
            counts = next_counts;
        }
        if remaining > 0 {
            let start = sample;
            let values = (0..remaining)
                .into_par_iter()
                .map(|offset| {
                    self.terminal_rollout(game, &ranked[0].action, start + offset as u64, config)
                })
                .collect::<Result<Vec<_>>>()?;
            sums[0] += values.iter().sum::<f64>();
            counts[0] += values.len() as u32;
        }
        ranked[0].rollout_mean = Some(mean(sums[0], counts[0]));
        ranked[0].rollout_count = counts[0];
        Ok(ranked.remove(0))
    }

    pub fn select_with_evaluator(
        &self,
        mut ranked: Vec<RankedV3Action>,
        candidate_count: usize,
        rollout_budget: usize,
        mut evaluate: impl FnMut(&TurnAction, u64) -> Result<f64>,
    ) -> Result<RankedV3Action> {
        if candidate_count == 0 || rollout_budget == 0 {
            return Err(V3Error::InvalidFeature(
                "sequential halving needs positive candidates and rollouts".to_owned(),
            ));
        }
        ranked.truncate(candidate_count.min(ranked.len()));
        let mut sums = vec![0.0f64; ranked.len()];
        let mut counts = vec![0u32; ranked.len()];
        let mut sample = 0u64;
        let mut remaining = rollout_budget;
        while ranked.len() > 1 && remaining > 0 {
            let rounds_left = ranked.len().next_power_of_two().ilog2().max(1) as usize;
            let per_candidate = (remaining / (ranked.len() * rounds_left)).max(1);
            let mut spent = 0usize;
            for index in 0..ranked.len() {
                for _ in 0..per_candidate {
                    if spent >= remaining {
                        break;
                    }
                    sums[index] += evaluate(&ranked[index].action, sample)?;
                    counts[index] += 1;
                    sample += 1;
                    spent += 1;
                }
            }
            remaining = remaining.saturating_sub(spent);
            let mut order = (0..ranked.len()).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                mean(sums[right], counts[right])
                    .total_cmp(&mean(sums[left], counts[left]))
                    .then_with(|| canonical_rank_order(&ranked[left], &ranked[right]))
            });
            let retained = ranked.len().div_ceil(2);
            let mut next_ranked = Vec::with_capacity(retained);
            let mut next_sums = Vec::with_capacity(retained);
            let mut next_counts = Vec::with_capacity(retained);
            for index in order.into_iter().take(retained) {
                next_ranked.push(ranked[index].clone());
                next_sums.push(sums[index]);
                next_counts.push(counts[index]);
            }
            ranked = next_ranked;
            sums = next_sums;
            counts = next_counts;
        }
        while remaining > 0 {
            sums[0] += evaluate(&ranked[0].action, sample)?;
            counts[0] += 1;
            sample += 1;
            remaining -= 1;
        }
        ranked[0].rollout_mean = Some(mean(sums[0], counts[0]));
        ranked[0].rollout_count = counts[0];
        Ok(ranked.remove(0))
    }

    fn terminal_rollout(
        &self,
        root: &GameState,
        action: &TurnAction,
        sample: u64,
        config: TerminalRolloutConfig,
    ) -> Result<f64> {
        let focal = root.current_player();
        let base_seed = rollout_seed(root, action, sample)?;
        let mut post_action = root.clone();
        // The selected action is legal in the observed root market, including
        // any bundled public prelude. Apply it before resampling the remaining
        // hidden order; resampling first can change a prelude reveal and make
        // the already-selected wildlife placement spuriously illegal.
        post_action.apply(action)?;
        retry_conditioned_terminal_rollout(base_seed, |seed| {
            self.terminal_rollout_from_post_action(&post_action, focal, seed, config)
        })
    }

    fn terminal_rollout_from_post_action(
        &self,
        post_action: &GameState,
        focal: usize,
        seed: GameSeed,
        config: TerminalRolloutConfig,
    ) -> Result<f64> {
        let mut game = post_action.clone();
        game.redeterminize_hidden(seed);
        let mut plies = 0u16;
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "cascadia-v3-terminal-rollout-v1"))
            .collect::<Vec<_>>();
        while !game.is_game_over() && config.maximum_plies.is_none_or(|limit| plies < limit) {
            let seat = game.current_player();
            let selected = if config.model_guided {
                self.rank_legal_actions(&game)?[0].action.clone()
            } else {
                let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                select_greedy_action(&game, &prelude, &mut rngs[seat])?
            };
            game.apply(&selected)?;
            plies += 1;
        }
        if game.is_game_over() {
            Ok(f64::from(score_game(&game)[focal].base_total))
        } else {
            let public = game.public_state();
            let features = encode_public_features(&public, focal)?;
            let stack = V3AccumulatorStack::new(self.model, features, self.backend)?;
            let realized =
                score_board(&game.boards()[focal], game.config().scoring_cards).base_total;
            Ok(f64::from(realized) + f64::from(stack.evaluate(self.model)?.score))
        }
    }
}

fn tile_sibling_key(action: &TurnAction) -> TurnAction {
    let mut key = action.clone();
    key.wildlife = None;
    key
}

fn habitat_sibling_key(action: &TurnAction) -> (MarketSlot, TilePlacement) {
    let tile_slot = match action.draft {
        DraftChoice::Paired { slot } => slot,
        DraftChoice::Independent { tile_slot, .. } => tile_slot,
    };
    (tile_slot, action.tile)
}

fn draft_market_key(draft: DraftChoice) -> (MarketSlot, MarketSlot) {
    match draft {
        DraftChoice::Paired { slot } => (slot, slot),
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (tile_slot, wildlife_slot),
    }
}

fn mean(sum: f64, count: u32) -> f64 {
    if count == 0 {
        f64::NEG_INFINITY
    } else {
        sum / f64::from(count)
    }
}

fn canonical_rank_order(left: &RankedV3Action, right: &RankedV3Action) -> Ordering {
    right
        .direct_raw_units
        .cmp(&left.direct_raw_units)
        .then_with(|| {
            let left = postcard::to_allocvec(&left.action).expect("actions are serializable");
            let right = postcard::to_allocvec(&right.action).expect("actions are serializable");
            left.cmp(&right)
        })
}

fn rollout_seed(root: &GameState, action: &TurnAction, sample: u64) -> Result<GameSeed> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v3-terminal-rollout-seed-v1");
    hasher.update(root.canonical_hash().as_bytes());
    hasher.update(&postcard::to_allocvec(action)?);
    hasher.update(&sample.to_le_bytes());
    Ok(GameSeed(*hasher.finalize().as_bytes()))
}

fn conditioned_terminal_rollout_seed(base_seed: GameSeed, attempt: u64) -> GameSeed {
    if attempt == 0 {
        return base_seed;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v3-terminal-rollout-conditioning-v1");
    hasher.update(&base_seed.0);
    hasher.update(&attempt.to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn retry_conditioned_terminal_rollout<T>(
    base_seed: GameSeed,
    mut run: impl FnMut(GameSeed) -> Result<T>,
) -> Result<T> {
    let mut attempt = 0u64;
    loop {
        let seed = conditioned_terminal_rollout_seed(base_seed, attempt);
        match run(seed) {
            Err(error) if is_unstable_market_exhaustion(&error) => {
                attempt = attempt.checked_add(1).ok_or_else(|| {
                    V3Error::InvalidFeature(
                        "terminal rollout conditioning attempt overflow".to_owned(),
                    )
                })?;
            }
            result => return result,
        }
    }
}

fn is_unstable_market_exhaustion(error: &V3Error) -> bool {
    matches!(
        error,
        V3Error::Rules(cascadia_game::RuleError::WildlifeBagEmpty)
            | V3Error::Simulation(cascadia_sim::SimulationError::Rules(
                cascadia_game::RuleError::WildlifeBagEmpty
            ))
    )
}

pub fn select_boltzmann_top32(
    ranked: &[RankedV3Action],
    epsilon: f64,
    temperature: f64,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction> {
    if ranked.is_empty()
        || !(0.0..=1.0).contains(&epsilon)
        || !temperature.is_finite()
        || temperature <= 0.0
    {
        return Err(V3Error::InvalidFeature(
            "invalid Boltzmann exploration request".to_owned(),
        ));
    }
    if rng.r#gen::<f64>() >= epsilon {
        return Ok(ranked[0].action.clone());
    }
    let candidates = &ranked[..ranked.len().min(32)];
    let best = f64::from(candidates[0].direct_score);
    let weights = candidates
        .iter()
        .map(|candidate| ((f64::from(candidate.direct_score) - best) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let mut draw = rng.r#gen::<f64>() * total;
    for (candidate, weight) in candidates.iter().zip(weights) {
        if draw <= weight {
            return Ok(candidate.action.clone());
        }
        draw -= weight;
    }
    Ok(candidates
        .last()
        .expect("candidate slice is nonempty")
        .action
        .clone())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState, RuleError, score_board};
    use cascadia_sim::{SimulationError, play_greedy_plies, strategy_rng};

    use super::*;

    #[test]
    fn direct_policy_scores_every_legal_action_once() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(100),
        )
        .unwrap();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let expected = game.legal_turn_actions(&prelude).unwrap().len();
        let model = QuantizedV3Model::zeroed();
        let policy = V3SearchPolicy::new(&model, InferenceBackend::Scalar).unwrap();
        let ranked = policy
            .rank_legal_actions_with_prelude(&game, &prelude)
            .unwrap();
        assert_eq!(ranked.len(), expected);
        for candidate in ranked {
            let afterstate = game.preview_public_afterstate(&candidate.action).unwrap();
            let immediate = score_board(
                &afterstate.boards()[game.current_player()],
                afterstate.config().scoring_cards,
            )
            .base_total;
            assert_eq!(candidate.direct_score, f32::from(immediate));
            assert_eq!(
                candidate.direct_raw_units,
                i32::from(immediate) * model.scales.output
            );
        }
    }

    #[test]
    fn optimized_ranker_is_bit_identical_to_reference() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(102),
        )
        .unwrap();
        let model = QuantizedV3Model::engineering_smoke(103);
        let policy = V3SearchPolicy::new(&model, InferenceBackend::Scalar).unwrap();
        let optimized = policy.rank_legal_actions(&game).unwrap();
        let reference = policy.rank_legal_actions_reference(&game).unwrap();
        assert_eq!(optimized, reference);
    }

    #[test]
    fn optimized_ranker_matches_reference_across_game_phases() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x600d),
        )
        .unwrap();
        let mut rng = strategy_rng(game.seed(), 0, "v3-ranker-parity");
        let model = QuantizedV3Model::engineering_smoke(0x600e);
        let policy = V3SearchPolicy::new(&model, InferenceBackend::Neon).unwrap();
        let mut completed = 0usize;
        for target in [16usize, 44, 72] {
            play_greedy_plies(&mut game, target - completed, &mut rng).unwrap();
            completed = target;
            let optimized = policy.rank_legal_actions(&game).unwrap();
            let reference = policy.rank_legal_actions_reference(&game).unwrap();
            assert_eq!(optimized, reference, "ranker mismatch after ply {target}");
        }
    }

    #[test]
    fn sequential_halving_spends_exact_budget_and_selects_best() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(101),
        )
        .unwrap();
        let model = QuantizedV3Model::zeroed();
        let policy = V3SearchPolicy::new(&model, InferenceBackend::Scalar).unwrap();
        let ranked = policy.rank_legal_actions(&game).unwrap();
        let mut calls = 0usize;
        let selected = policy
            .select_with_evaluator(ranked, 8, 64, |action, _| {
                calls += 1;
                Ok(f64::from(action.tile.coord.q))
            })
            .unwrap();
        assert_eq!(calls, 64);
        assert!(selected.rollout_count as usize <= 64);
    }

    #[test]
    fn teacher_packet_preserves_all_candidate_statistics_and_budget() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(104),
        )
        .unwrap();
        let model = QuantizedV3Model::engineering_smoke(105);
        let policy = V3SearchPolicy::new(&model, InferenceBackend::Scalar).unwrap();
        let label = policy
            .label_teacher_root(
                &game,
                V3SearchBudget::SequentialHalving {
                    candidates: 4,
                    rollouts: 16,
                },
                TerminalRolloutConfig {
                    model_guided: false,
                    maximum_plies: Some(1),
                },
            )
            .unwrap();
        label.validate().unwrap();
        assert_eq!(label.candidates.len(), 4);
        assert_eq!(
            label
                .candidates
                .iter()
                .map(|candidate| candidate.rollout_count)
                .sum::<u32>(),
            16
        );
        assert_eq!(
            label
                .candidates
                .iter()
                .find(|candidate| candidate.rank == 1)
                .unwrap()
                .action,
            label.selected_action
        );
    }

    #[test]
    fn terminal_rollout_conditioning_retries_only_market_exhaustion() {
        let base = GameSeed::from_u64(0xabc);
        let mut attempts = Vec::new();
        let selected = retry_conditioned_terminal_rollout(base, |seed| {
            attempts.push(seed);
            match attempts.len() {
                1 => Err(V3Error::Rules(RuleError::WildlifeBagEmpty)),
                2 => Err(V3Error::Simulation(SimulationError::Rules(
                    RuleError::WildlifeBagEmpty,
                ))),
                _ => Ok(seed),
            }
        })
        .unwrap();
        assert_eq!(attempts[0], base);
        assert_eq!(attempts[1], conditioned_terminal_rollout_seed(base, 1));
        assert_eq!(attempts[2], conditioned_terminal_rollout_seed(base, 2));
        assert_eq!(selected, attempts[2]);
        assert_ne!(attempts[0], attempts[1]);
        assert_ne!(attempts[1], attempts[2]);

        let error = retry_conditioned_terminal_rollout(base, |_| -> Result<()> {
            Err(V3Error::Rules(RuleError::TileStackEmpty))
        })
        .unwrap_err();
        assert!(matches!(error, V3Error::Rules(RuleError::TileStackEmpty)));
    }
}
