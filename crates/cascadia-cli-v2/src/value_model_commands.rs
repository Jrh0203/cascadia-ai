use std::time::Instant;

use cascadia_data::{DatasetSplit, PositionRecord};
use cascadia_eval::{summarize_match_results, summarize_paired_match_results};
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude};
use cascadia_model::ModelProcess;
use cascadia_search::{
    HabitatCandidateLookaheadConfig, HabitatCandidateLookaheadStrategy,
    LateConservativeBasePolicyImprovementConfig, LateConservativeBasePolicyImprovementStrategy,
    MlxPublicBeamValueConfig, MlxPublicBeamValueStrategy, MlxValueConfig,
    MlxValueLeafLookaheadConfig, MlxValueLeafLookaheadStrategy, MlxValueStrategy,
    TerminalPolicyImprovementConfig,
};
use cascadia_sim::{
    MatchConfig, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, StrategyKind, play_match,
    select_pattern_action, strategy_rng,
};

use crate::cli::Command;
use crate::report::{ReportContext, model_source_args, write_report};

pub fn run(
    command: Command,
    report_context: &ReportContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::ModelSmoke {
            run_dir,
            model_dir,
            server,
        } => {
            let source_args = model_source_args(run_dir, model_dir)?;
            let state = GameState::new(
                GameConfig::research_aaaaa(4)?,
                DatasetSplit::Validation.game_seed(0),
            )?;
            let record = PositionRecord::observe(&state, 0);
            let mut model = ModelProcess::spawn(server, source_args)?;
            let predictions = model.predict(&[record])?;
            model.shutdown()?;
            let total: f32 = predictions[0].iter().sum();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "components": predictions[0],
                    "predicted_total": total,
                }))?
            );
        }
        Command::ModelBenchmark {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            prefilter_k,
            output,
        } => {
            if games == 0 {
                return Err("model-benchmark requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let mut strategy = MlxValueStrategy::spawn_with_config(
                server,
                source_args,
                MlxValueConfig {
                    greedy_prefilter: prefilter_k,
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    strategy.play_match(game_config, GameSeed::from_u64(first_seed + index as u64))
                })
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = strategy.shutdown();
            let matches = result?;
            shutdown?;
            let strategy_id = strategy_id(prefilter_k);
            let report =
                summarize_match_results(&strategy_id, games, first_seed, &matches, elapsed_seconds);
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::ModelCompare {
            games,
            first_seed,
            baseline,
            run_dir,
            model_dir,
            server,
            prefilter_k,
            output,
        } => {
            if games == 0 {
                return Err("model-compare requires at least one game".into());
            }
            let baseline: StrategyKind = baseline.into();
            let source_args = model_source_args(run_dir, model_dir)?;
            let mut strategy = MlxValueStrategy::spawn_with_config(
                server,
                source_args,
                MlxValueConfig {
                    greedy_prefilter: prefilter_k,
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_match(&MatchConfig::symmetric(game_config, seed, baseline))?;
                    let treatment_result = strategy.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = strategy.shutdown();
            let pairs = result?;
            shutdown?;
            let report = summarize_paired_match_results(
                baseline.id(),
                &strategy_id(prefilter_k),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::PublicBeamValueModelSmoke {
            run_dir,
            model_dir,
            server,
        } => {
            let source_args = model_source_args(run_dir, model_dir)?;
            let config = MlxPublicBeamValueConfig::default();
            let mut strategy = MlxPublicBeamValueStrategy::spawn(server, source_args, config)?;
            let seed = GameSeed::from_u64(31_000);
            let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
            let mut blueprint_rngs = (0..usize::from(game.config().player_count))
                .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
                .collect::<Vec<_>>();
            while game.turns_remaining_for_player(game.current_player()) > 5 {
                let player = game.current_player();
                let prelude = MarketPrelude {
                    replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                    wildlife_wipes: Vec::new(),
                };
                let action = select_pattern_action(
                    &game,
                    &prelude,
                    config.blueprint,
                    &mut blueprint_rngs[player],
                )?;
                game.apply(&action)?;
            }
            let ranked = strategy.rank_terminal_actions(&game)?;
            let selected = strategy.select_action(&game)?;
            strategy.shutdown()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "strategy": config.strategy_id(),
                    "turn": game.completed_turns(),
                    "active_seat": game.current_player(),
                    "personal_turns_remaining": game.turns_remaining_for_player(game.current_player()),
                    "candidates": ranked.len(),
                    "top_predicted_final_score": ranked[0].mean_leaf_score,
                    "selected_action": selected,
                }))?
            );
        }
        Command::PublicBeamValueModelCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            output,
        } => {
            if games == 0 {
                return Err("public-beam-value-model-compare requires at least one game".into());
            }
            let blueprint = PatternAwareConfig::default();
            let baseline = LateConservativeBasePolicyImprovementStrategy::new(
                LateConservativeBasePolicyImprovementConfig {
                    final_personal_turns: 5,
                    terminal: TerminalPolicyImprovementConfig {
                        determinizations: 8,
                        blueprint,
                    },
                },
            )?;
            let config = MlxPublicBeamValueConfig::default();
            let source_args = model_source_args(run_dir, model_dir)?;
            let mut treatment = MlxPublicBeamValueStrategy::spawn(server, source_args, config)?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = treatment.shutdown();
            let pairs = result?;
            shutdown?;
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                &config.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::ValueLeafCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            candidates,
            habitat_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("value-leaf-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let treatment_config = MlxValueLeafLookaheadConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
                determinizations,
                greedy_plies,
            };
            let mut treatment =
                MlxValueLeafLookaheadStrategy::spawn(server, source_args, treatment_config)?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = treatment.shutdown();
            let pairs = result?;
            shutdown?;
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                &treatment_config.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        _ => unreachable!("value-model dispatcher received a different command family"),
    }
    Ok(())
}

fn strategy_id(prefilter_k: Option<usize>) -> String {
    MlxValueConfig {
        greedy_prefilter: prefilter_k,
    }
    .strategy_id()
}
