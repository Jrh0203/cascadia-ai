//! Local heterogeneous R2-MAP game execution for iterative trajectory collection.

use std::sync::Mutex;

use cascadia_data::{
    R2MapCollectionKind, R2MapCollectorError, R2MapExplorationDecisionIdentity,
    R2MapExplorationDraw, R2MapExplorationRngContext, R2MapGameRequest, R2MapGameRunner,
    R2MapPlayedGame, R2MapPolicyRole, r2_map_action_draw_u64, r2_map_explore_draw_u64,
    reconstruct_r2_map_public_turns,
};
use cascadia_game::{GameConfig, MarketDecisionStage};
use cascadia_model::{R2MapModelProcess, R2MapServingBundle};
use cascadia_sim::{
    SimulationError, play_match_with_seat_selector, select_greedy_action, strategy_rng,
};

use crate::{
    R2MapExplorationChoice, R2MapPredictor, R2MapTurnDecisionContext, R2MapTurnDecisionKind,
    select_r2_map_turn,
};

/// Executes games against four frozen local policy identities in one verified
/// multi-checkpoint service. The process lock deliberately spans a whole game:
/// one request stream has one owner and cannot interleave framed messages.
pub struct R2MapLocalGameRunner<P = R2MapModelProcess> {
    bundle: R2MapServingBundle,
    process: Mutex<P>,
}

impl<P> R2MapLocalGameRunner<P> {
    pub fn new(
        bundle: R2MapServingBundle,
        process: P,
    ) -> Result<Self, cascadia_model::R2MapModelError> {
        bundle.validate()?;
        Ok(Self {
            bundle,
            process: Mutex::new(process),
        })
    }
}

impl R2MapLocalGameRunner<R2MapModelProcess> {
    pub fn from_verified_bundle(
        bundle_path: impl AsRef<std::path::Path>,
        process: R2MapModelProcess,
    ) -> Result<Self, cascadia_model::R2MapModelError> {
        Ok(Self {
            bundle: R2MapServingBundle::read_verified(bundle_path)?,
            process: Mutex::new(process),
        })
    }

    pub fn restart_service(&self) -> Result<(), cascadia_model::R2MapModelError> {
        self.process
            .lock()
            .map_err(|_| cascadia_model::R2MapModelError::ProcessLockPoisoned)?
            .restart()
    }

    pub fn shutdown_service(self) -> Result<(), cascadia_model::R2MapModelError> {
        self.process
            .into_inner()
            .map_err(|_| cascadia_model::R2MapModelError::ProcessLockPoisoned)?
            .shutdown()
    }
}

impl<P> R2MapLocalGameRunner<P> {
    fn validate_request(&self, request: &R2MapGameRequest) -> Result<(), R2MapCollectorError> {
        if request.context.seats.len() != 4
            || usize::from(request.context.focal_seat) >= request.context.seats.len()
        {
            return Err(R2MapCollectorError::RunnerContract(
                "local R2-MAP runner requires exactly four seats and a valid focal seat",
            ));
        }
        if request.context.protocols != self.bundle.protocols {
            return Err(R2MapCollectorError::RunnerContract(
                "local R2-MAP runner refuses stale collector, source, or serving protocol identity",
            ));
        }
        match request.context.collection_kind {
            R2MapCollectionKind::IterativeTraining
                if request.context.exploration.enabled
                    && request.context.seats[usize::from(request.context.focal_seat)].role
                        == R2MapPolicyRole::Newest
                    && request
                        .context
                        .seats
                        .iter()
                        .filter(|policy| policy.role == R2MapPolicyRole::Newest)
                        .count()
                        == 1 => {}
            R2MapCollectionKind::Benchmark
                if request.context.exploration
                    == cascadia_data::R2MapExplorationIdentity::disabled()
                    && request
                        .context
                        .seats
                        .iter()
                        .enumerate()
                        .all(|(seat, policy)| {
                            seat == usize::from(request.context.focal_seat)
                                || policy.role != R2MapPolicyRole::Newest
                        }) => {}
            _ => {
                return Err(R2MapCollectorError::RunnerContract(
                    "local R2-MAP runner requires iterative exploration or disabled benchmark exploration",
                ));
            }
        }
        for policy in &request.context.seats {
            policy.validate()?;
            if let Some(hash) = policy.checkpoint_hash {
                self.bundle.model_for_manifest_identity(hash).map_err(|_| {
                    R2MapCollectorError::RunnerContract(
                        "local serving bundle is missing a requested checkpoint",
                    )
                })?;
            }
        }
        Ok(())
    }
}

impl<P: R2MapPredictor + Send> R2MapGameRunner for R2MapLocalGameRunner<P> {
    fn play_game(
        &self,
        request: &R2MapGameRequest,
    ) -> Result<R2MapPlayedGame, R2MapCollectorError> {
        self.play_game_inner(request, false)
    }
}

impl<P: R2MapPredictor + Send> R2MapLocalGameRunner<P> {
    fn play_game_inner(
        &self,
        request: &R2MapGameRequest,
        verify_reference_preparation: bool,
    ) -> Result<R2MapPlayedGame, R2MapCollectorError> {
        self.validate_request(request)?;
        let mut process = self
            .process
            .lock()
            .map_err(|_| R2MapCollectorError::RunnerContract("model process lock was poisoned"))?;
        let identities = request
            .context
            .seats
            .iter()
            .map(|policy| policy.policy_id.clone())
            .collect::<Vec<_>>();
        let mut greedy_rngs = request
            .context
            .seats
            .iter()
            .enumerate()
            .map(|(seat, policy)| strategy_rng(request.seed, seat, &policy.policy_id))
            .collect::<Vec<_>>();
        let mut exploration_draws = Vec::with_capacity(20);
        let exploration_rng_context = R2MapExplorationRngContext::new(
            &request.context.identity,
            request.seed,
            request.context.focal_seat,
        );
        let result = play_match_with_seat_selector(
            GameConfig::research_aaaaa(4)?,
            request.seed,
            &identities,
            |seat, game| {
                let policy = &request.context.seats[seat];
                if policy.role == R2MapPolicyRole::Greedy {
                    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                    return select_greedy_action(game, &prelude, &mut greedy_rngs[seat]);
                }
                let checkpoint_hash = policy.checkpoint_hash.ok_or_else(|| {
                    SimulationError::Strategy("checkpoint policy omitted its compact hash".into())
                })?;
                let model = self
                    .bundle
                    .model_for_manifest_identity(checkpoint_hash)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))?;
                let is_focal_newest = request.context.collection_kind
                    == R2MapCollectionKind::IterativeTraining
                    && seat == usize::from(request.context.focal_seat)
                    && policy.role == R2MapPolicyRole::Newest;
                let turn_index = game.completed_turns();
                let mut exploration_for = |context: &R2MapTurnDecisionContext| {
                    if !is_focal_newest {
                        return Ok(None);
                    }
                    let stage = context_stage(context);
                    let decision_identity = R2MapExplorationDecisionIdentity::new(
                        turn_index,
                        context.ordinal,
                        stage,
                        context.decision_id,
                        context.parent_public_hash,
                    );
                    let gate_draw =
                        r2_map_explore_draw_u64(exploration_rng_context, decision_identity);
                    let action_draws_u64 = context
                        .action_ids
                        .iter()
                        .map(|action_id| {
                            r2_map_action_draw_u64(
                                exploration_rng_context,
                                decision_identity,
                                *action_id,
                            )
                        })
                        .collect();
                    Ok(Some(R2MapExplorationChoice {
                        gate_draw_u64: gate_draw,
                        epsilon_parts_per_million: request
                            .context
                            .exploration
                            .epsilon_parts_per_million,
                        temperature_parts_per_million: request
                            .context
                            .exploration
                            .temperature_parts_per_million,
                        action_draws_u64,
                    }))
                };
                #[cfg(test)]
                let selected = if verify_reference_preparation {
                    crate::r2_map_direct::select_r2_map_turn_with_reference_parity(
                        &mut *process,
                        game,
                        request.context.identity.global_game_index,
                        model,
                        &mut exploration_for,
                    )
                } else {
                    select_r2_map_turn(
                        &mut *process,
                        game,
                        request.context.identity.global_game_index,
                        model,
                        &mut exploration_for,
                    )
                }
                .map_err(|error| SimulationError::Strategy(error.to_string()))?;
                #[cfg(not(test))]
                let selected = {
                    debug_assert!(!verify_reference_preparation);
                    select_r2_map_turn(
                        &mut *process,
                        game,
                        request.context.identity.global_game_index,
                        model,
                        &mut exploration_for,
                    )
                }
                .map_err(|error| SimulationError::Strategy(error.to_string()))?;
                if is_focal_newest {
                    for market in &selected.market_decisions {
                        exploration_draws.push(exploration_draw_for(
                            exploration_rng_context,
                            turn_index,
                            &market.context,
                            market.selected_index,
                            market.explored,
                        ));
                    }
                    exploration_draws.push(exploration_draw_for(
                        exploration_rng_context,
                        turn_index,
                        &selected.draft_context,
                        selected.draft.selected_index,
                        selected.draft.explored,
                    ));
                }
                Ok(selected.action)
            },
        )?;
        let public_turns = reconstruct_r2_map_public_turns(&result.replay)?;
        Ok(R2MapPlayedGame {
            result,
            exploration_draws,
            public_turns,
        })
    }

    #[cfg(test)]
    fn play_game_with_reference_preparation_parity(
        &self,
        request: &R2MapGameRequest,
    ) -> Result<R2MapPlayedGame, R2MapCollectorError> {
        self.play_game_inner(request, true)
    }
}

fn context_stage(context: &R2MapTurnDecisionContext) -> MarketDecisionStage {
    match context.kind {
        R2MapTurnDecisionKind::Market(stage) => stage,
        R2MapTurnDecisionKind::Draft => MarketDecisionStage::Draft,
    }
}

fn exploration_draw_for(
    rng_context: R2MapExplorationRngContext<'_>,
    turn_index: u16,
    context: &R2MapTurnDecisionContext,
    selected_index: usize,
    explored: bool,
) -> R2MapExplorationDraw {
    let stage = context_stage(context);
    let selected_action_id = context.action_ids[selected_index];
    let decision_identity = R2MapExplorationDecisionIdentity::new(
        turn_index,
        context.ordinal,
        stage,
        context.decision_id,
        context.parent_public_hash,
    );
    let explore_draw_u64 = r2_map_explore_draw_u64(rng_context, decision_identity);
    R2MapExplorationDraw {
        turn_index,
        ordinal: context.ordinal,
        stage,
        decision_id: context.decision_id,
        parent_public_hash: context.parent_public_hash,
        selected_action_id,
        explore_draw_u64,
        action_draw_u64: explored
            .then(|| r2_map_action_draw_u64(rng_context, decision_identity, selected_action_id)),
        explored,
    }
}

#[cfg(test)]
mod tests {
    use cascadia_data::{
        R2MapExplorationIdentity, R2MapGameIdentity, R2MapPolicyIdentity, R2MapProtocolIdentity,
        R2MapRecordContext, R2MapRngIdentity, R2MapSeedPurpose, focal_seat_for_game,
        r2_map_game_seed,
    };
    use cascadia_model::{
        R2_MAP_SERVING_BUNDLE_SCHEMA, R2MapInferenceGroup, R2MapMarketInferenceGroup,
        R2MapMarketPredictionGroup, R2MapModelError, R2MapModelIdentity, R2MapPredictionGroup,
        R2MapServingBundleEntry,
    };

    use super::*;

    struct ExactFakePredictor;

    impl crate::R2MapPredictor for ExactFakePredictor {
        fn score_r2_map_groups(
            &mut self,
            groups: &[R2MapInferenceGroup],
        ) -> Result<Vec<R2MapPredictionGroup>, R2MapModelError> {
            Ok(groups
                .iter()
                .map(|group| {
                    let action_scores = group
                        .candidates
                        .iter()
                        .map(|candidate| candidate.exact_afterstate_score)
                        .collect::<Vec<_>>();
                    R2MapPredictionGroup {
                        group_id: group.group_id,
                        decision_id: group.decision_id,
                        action_ids: group
                            .candidates
                            .iter()
                            .map(|candidate| candidate.action_id)
                            .collect(),
                        predicted_score_to_go: vec![0.0; action_scores.len()],
                        predicted_score_components_to_go: vec![[0.0; 11]; action_scores.len()],
                        bootstrap_policy_logits: vec![0.0; action_scores.len()],
                        action_scores,
                    }
                })
                .collect())
        }

        fn score_r2_map_market_groups(
            &mut self,
            groups: &[R2MapMarketInferenceGroup],
        ) -> Result<Vec<R2MapMarketPredictionGroup>, R2MapModelError> {
            Ok(groups
                .iter()
                .map(|group| R2MapMarketPredictionGroup {
                    group_id: group.group_id,
                    decision_id: group.decision_id,
                    action_ids: group
                        .candidates
                        .iter()
                        .map(|candidate| candidate.action_id)
                        .collect(),
                    action_scores: vec![0.0; group.candidates.len()],
                    predicted_score_to_go: vec![0.0; group.candidates.len()],
                })
                .collect())
        }
    }

    fn model(checkpoint: &str, digit: char) -> R2MapModelIdentity {
        R2MapModelIdentity {
            checkpoint_id: checkpoint.into(),
            checkpoint_manifest_blake3: digit.to_string().repeat(64),
            model_config_blake3: "1".repeat(64),
            model_weights_blake3: "2".repeat(64),
            verification_id: "3".repeat(64),
        }
    }

    #[test]
    #[ignore = "canonical full-game sequential-versus-parallel preparation parity gate"]
    fn heterogeneous_four_model_game_completes_with_exact_focal_exploration_trace() {
        let temp = std::env::temp_dir();
        let specs = [
            ('a', "newest"),
            ('b', "history-1"),
            ('c', "history-2"),
            ('d', "history-3"),
        ];
        let bundle = R2MapServingBundle {
            schema_version: 2,
            schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.into(),
            protocols: R2MapProtocolIdentity {
                collector_hash: [1; 32],
                source_hash: [2; 32],
                serving_protocol_hash: [3; 32],
            },
            entries: specs
                .iter()
                .map(|(digit, checkpoint)| R2MapServingBundleEntry {
                    manifest_identity_blake3: digit.to_string().repeat(64),
                    run_dir: temp.clone(),
                    checkpoint_path: temp.join(checkpoint),
                    model: model(checkpoint, *digit),
                    pinned: true,
                })
                .collect(),
        };
        let game_index = 0;
        let campaign = "r2-map-runner-smoke";
        let seed = r2_map_game_seed(campaign, R2MapSeedPurpose::Generation, 0, game_index);
        let identity = R2MapGameIdentity::new(campaign, 0, "john2", game_index, seed);
        let focal = focal_seat_for_game(game_index);
        let mut seats = vec![
            R2MapPolicyIdentity::historical("history-1", [0xbb; 32]),
            R2MapPolicyIdentity::historical("history-2", [0xcc; 32]),
            R2MapPolicyIdentity::historical("history-3", [0xdd; 32]),
            R2MapPolicyIdentity::historical("history-1", [0xbb; 32]),
        ];
        seats[usize::from(focal)] = R2MapPolicyIdentity::newest("newest", [0xaa; 32]);
        let request = R2MapGameRequest {
            seed,
            context: R2MapRecordContext {
                collection_kind: R2MapCollectionKind::IterativeTraining,
                identity,
                seed_purpose: R2MapSeedPurpose::Generation,
                focal_seat: focal,
                seats: seats.clone(),
                rng: R2MapRngIdentity::default(),
                exploration: R2MapExplorationIdentity::training(0, 1_000_000),
                protocols: R2MapProtocolIdentity {
                    collector_hash: [1; 32],
                    source_hash: [2; 32],
                    serving_protocol_hash: [3; 32],
                },
            },
        };
        let runner = R2MapLocalGameRunner::new(bundle, ExactFakePredictor).unwrap();
        let mut stale = request.clone();
        stale.context.protocols.source_hash = [9; 32];
        assert!(matches!(
            runner.validate_request(&stale),
            Err(R2MapCollectorError::RunnerContract(message))
                if message.contains("refuses stale")
        ));
        crate::r2_map_direct::reset_reference_parity_timing();
        let played = runner
            .play_game_with_reference_preparation_parity(&request)
            .unwrap();
        let (reference_seconds, rayon_cache_seconds, incremental_seconds) =
            crate::r2_map_direct::reference_parity_timing_seconds();
        let projected_incremental_game_seconds =
            played.result.elapsed_seconds - reference_seconds - rayon_cache_seconds;
        eprintln!(
            "r2-map-preparation-parity-timing reference_seconds={reference_seconds:.6} rayon_cache_seconds={rayon_cache_seconds:.6} incremental_seconds={incremental_seconds:.6} projected_incremental_game_seconds={projected_incremental_game_seconds:.6}"
        );
        assert!(reference_seconds > 0.0);
        assert!(rayon_cache_seconds > 0.0);
        assert!(incremental_seconds > 0.0);
        assert!(projected_incremental_game_seconds > 0.0);
        assert_eq!(played.result.turns, 80);
        assert_eq!(
            played.result.strategies,
            seats
                .iter()
                .map(|seat| seat.policy_id.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            played
                .exploration_draws
                .iter()
                .filter(|draw| draw.stage == MarketDecisionStage::Draft)
                .count(),
            20
        );
        assert!(played.exploration_draws.len() >= 40);
        assert!(
            played
                .exploration_draws
                .iter()
                .all(|draw| { draw.explored == draw.action_draw_u64.is_some() })
        );
        assert_eq!(played.public_turns.len(), 80);
        let replayed = played.result.replay.play().unwrap();
        assert_eq!(
            played.result.replay.final_state_hash,
            Some(*replayed.canonical_hash().as_bytes())
        );
    }
}
