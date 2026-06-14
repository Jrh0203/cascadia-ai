use std::time::Instant;

use cascadia_eval::{
    EvaluationBlock, summarize_paired_evaluation_blocks, summarize_paired_match_results,
};
use cascadia_game::{GameConfig, GameSeed, MarketPrelude};
use cascadia_search::{
    LateConservativeBasePolicyImprovementConfig, LateConservativeBasePolicyImprovementStrategy,
    LateConservativeFocalBeamConfig, LateConservativeFocalBeamStrategy,
    PerfectInformationFocalBeamConfig, PerfectInformationFocalBeamStrategy,
    PerfectInformationPatternOracleConfig, PerfectInformationPatternOracleStrategy,
    PerfectInformationPortfolioBeamConfig, PerfectInformationPortfolioBeamStrategy,
    PerfectInformationRootDiverseBeamConfig, PerfectInformationRootDiverseBeamStrategy,
    PublicFocalOpenLoopTreeConfig, PublicFocalOpenLoopTreeStrategy,
    TerminalPolicyImprovementConfig,
};
use cascadia_sim::{
    MatchConfig, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, StrategyKind, play_match,
    play_match_with_selector, select_pattern_action, strategy_rng,
};
use rayon::prelude::*;

use crate::cli::Command;
use crate::report::{ReportContext, write_report};

pub fn run(
    command: Command,
    report_context: &ReportContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::PerfectInformationOracleCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            output,
        } => {
            if games == 0 {
                return Err("perfect-information-oracle-compare requires at least one game".into());
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let treatment = PerfectInformationPatternOracleStrategy::new(
                PerfectInformationPatternOracleConfig {
                    blueprint,
                    wildlife_candidate_limit: None,
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = play_match(&MatchConfig::symmetric(
                        game_config,
                        seed,
                        StrategyKind::PatternAware,
                    ))?;
                    let treatment_result =
                        play_focal_oracle_block(game_config, seed, &treatment, blueprint)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        EvaluationBlock::from(&baseline_result),
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                StrategyKind::PatternAware.id(),
                treatment.strategy_id(),
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
        Command::PerfectInformationOracleFrontierCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            output,
        } => {
            if games == 0 {
                return Err(
                    "perfect-information-oracle-frontier-compare requires at least one game".into(),
                );
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline = PerfectInformationPatternOracleStrategy::new(
                PerfectInformationPatternOracleConfig {
                    blueprint,
                    wildlife_candidate_limit: None,
                },
            )?;
            let treatment = PerfectInformationPatternOracleStrategy::new(
                PerfectInformationPatternOracleConfig {
                    blueprint,
                    wildlife_candidate_limit: Some(wildlife_candidates),
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_focal_oracle_block(game_config, seed, &baseline, blueprint)?;
                    let treatment_result =
                        play_focal_oracle_block(game_config, seed, &treatment, blueprint)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PerfectInformationFocalBeamCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            beam_width,
            terminal_turns,
            output,
        } => {
            if games == 0 {
                return Err(
                    "perfect-information-focal-beam-compare requires at least one game".into(),
                );
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline = PerfectInformationPatternOracleStrategy::new(
                PerfectInformationPatternOracleConfig {
                    blueprint,
                    wildlife_candidate_limit: Some(wildlife_candidates),
                },
            )?;
            let treatment =
                PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_focal_oracle_block(game_config, seed, &baseline, blueprint)?;
                    let treatment_result =
                        play_focal_beam_block(game_config, seed, &treatment, blueprint)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PerfectInformationFocalFrontierCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            baseline_wildlife_candidates,
            treatment_wildlife_candidates,
            policy_market_draws,
            beam_width,
            terminal_turns,
            output,
        } => {
            if games == 0 {
                return Err(
                    "perfect-information-focal-frontier-compare requires at least one game".into(),
                );
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline =
                PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: baseline_wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                })?;
            let treatment =
                PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: treatment_wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_focal_beam_block(game_config, seed, &baseline, blueprint)?;
                    let treatment_result =
                        play_focal_beam_block(game_config, seed, &treatment, blueprint)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PerfectInformationBeamCapacityCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            baseline_beam_width,
            treatment_beam_width,
            terminal_turns,
            output,
        } => {
            if games == 0 {
                return Err(
                    "perfect-information-beam-capacity-compare requires at least one game".into(),
                );
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline =
                PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: wildlife_candidates,
                    beam_width: baseline_beam_width,
                    final_personal_turns: terminal_turns,
                })?;
            let treatment =
                PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: wildlife_candidates,
                    beam_width: treatment_beam_width,
                    final_personal_turns: terminal_turns,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_focal_beam_block(game_config, seed, &baseline, blueprint)?;
                    let treatment_result =
                        play_focal_beam_block(game_config, seed, &treatment, blueprint)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PerfectInformationRootDiverseBeamCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            baseline_root_wildlife_candidates,
            treatment_root_wildlife_candidates,
            future_wildlife_candidates,
            policy_market_draws,
            beam_width,
            terminal_turns,
            output,
        } => {
            if games == 0 {
                return Err(
                    "perfect-information-root-diverse-beam-compare requires at least one game"
                        .into(),
                );
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline = PerfectInformationRootDiverseBeamStrategy::new(
                PerfectInformationRootDiverseBeamConfig {
                    blueprint,
                    root_wildlife_candidate_limit: baseline_root_wildlife_candidates,
                    future_wildlife_candidate_limit: future_wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                },
            )?;
            let treatment = PerfectInformationRootDiverseBeamStrategy::new(
                PerfectInformationRootDiverseBeamConfig {
                    blueprint,
                    root_wildlife_candidate_limit: treatment_root_wildlife_candidates,
                    future_wildlife_candidate_limit: future_wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = play_focal_root_diverse_beam_block(
                        game_config,
                        seed,
                        &baseline,
                        blueprint,
                    )?;
                    let treatment_result = play_focal_root_diverse_beam_block(
                        game_config,
                        seed,
                        &treatment,
                        blueprint,
                    )?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PerfectInformationPortfolioBeamCompare {
            games,
            first_seed,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            beam_width,
            terminal_turns,
            output,
        } => {
            if games == 0 {
                return Err(
                    "perfect-information-portfolio-beam-compare requires at least one game".into(),
                );
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline =
                PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                })?;
            let treatment = PerfectInformationPortfolioBeamStrategy::new(
                PerfectInformationPortfolioBeamConfig {
                    blueprint,
                    wildlife_candidate_limit: wildlife_candidates,
                    beam_width,
                    final_personal_turns: terminal_turns,
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_focal_beam_block(game_config, seed, &baseline, blueprint)?;
                    let treatment_result =
                        play_focal_portfolio_beam_block(game_config, seed, &treatment, blueprint)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_evaluation_blocks(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PublicFocalBeamCompare {
            games,
            first_seed,
            terminal_turns,
            determinizations,
            beam_width,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err("public-focal-beam-compare requires at least one game".into());
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline = LateConservativeBasePolicyImprovementStrategy::new(
                LateConservativeBasePolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: TerminalPolicyImprovementConfig {
                        determinizations: 8,
                        blueprint,
                    },
                },
            )?;
            let treatment =
                LateConservativeFocalBeamStrategy::new(LateConservativeFocalBeamConfig {
                    final_personal_turns: terminal_turns,
                    determinizations,
                    beam_width,
                    wildlife_candidate_limit: wildlife_candidates,
                    blueprint,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = baseline
                    .play_match(game_config, seed)
                    .map_err(|error| error.to_string())?;
                let treatment_result = treatment
                    .play_match(game_config, seed)
                    .map_err(|error| error.to_string())?;
                Ok::<_, String>((numeric_seed, baseline_result, treatment_result))
            };
            let started = Instant::now();
            let pairs = if sequential {
                (0..games).map(run_pair).collect::<Result<Vec<_>, _>>()
            } else {
                (0..games)
                    .into_par_iter()
                    .map(run_pair)
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(std::io::Error::other)?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        Command::PublicFocalTreeCompare {
            games,
            first_seed,
            terminal_turns,
            simulations,
            root_candidates,
            exploration_milli,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err("public-focal-tree-compare requires at least one game".into());
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let baseline = LateConservativeBasePolicyImprovementStrategy::new(
                LateConservativeBasePolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: TerminalPolicyImprovementConfig {
                        determinizations: 8,
                        blueprint,
                    },
                },
            )?;
            let treatment = PublicFocalOpenLoopTreeStrategy::new(PublicFocalOpenLoopTreeConfig {
                blueprint,
                wildlife_candidate_limit: wildlife_candidates,
                root_candidate_limit: root_candidates,
                simulations,
                exploration_milli,
                final_personal_turns: terminal_turns,
            })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = baseline
                    .play_match(game_config, seed)
                    .map_err(|error| error.to_string())?;
                let treatment_result = treatment
                    .play_match(game_config, seed)
                    .map_err(|error| error.to_string())?;
                Ok::<_, String>((numeric_seed, baseline_result, treatment_result))
            };
            let started = Instant::now();
            let pairs = if sequential {
                (0..games).map(run_pair).collect::<Result<Vec<_>, _>>()
            } else {
                (0..games)
                    .into_par_iter()
                    .map(run_pair)
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(std::io::Error::other)?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
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
        _ => unreachable!("oracle dispatcher received a non-oracle command"),
    }
    Ok(())
}

fn play_focal_oracle_block(
    game_config: GameConfig,
    seed: GameSeed,
    oracle: &PerfectInformationPatternOracleStrategy,
    blueprint: PatternAwareConfig,
) -> Result<EvaluationBlock, Box<dyn std::error::Error>> {
    let player_count = usize::from(game_config.player_count);
    let mut scores = Vec::with_capacity(player_count);
    let mut decision_seconds = Vec::new();
    let mut elapsed_seconds = 0.0;

    for focal_seat in 0..player_count {
        let mut pattern_rngs = (0..player_count)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let result =
            play_match_with_selector(game_config, seed, oracle.strategy_id(), |player, game| {
                if player == focal_seat {
                    let started = Instant::now();
                    let action = oracle.select_action_deterministic(game).map_err(|error| {
                        cascadia_sim::SimulationError::Strategy(error.to_string())
                    })?;
                    decision_seconds.push(started.elapsed().as_secs_f64());
                    Ok(action)
                } else {
                    select_pattern_action(
                        game,
                        &MarketPrelude {
                            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                            wildlife_wipes: Vec::new(),
                        },
                        blueprint,
                        &mut pattern_rngs[player],
                    )
                }
            })?;
        scores.push(result.scores[focal_seat]);
        elapsed_seconds += result.elapsed_seconds;
    }

    Ok(EvaluationBlock {
        scores,
        decision_seconds,
        elapsed_seconds,
    })
}

fn play_focal_beam_block(
    game_config: GameConfig,
    seed: GameSeed,
    beam: &PerfectInformationFocalBeamStrategy,
    blueprint: PatternAwareConfig,
) -> Result<EvaluationBlock, Box<dyn std::error::Error>> {
    let player_count = usize::from(game_config.player_count);
    let mut scores = Vec::with_capacity(player_count);
    let mut decision_seconds = Vec::new();
    let mut elapsed_seconds = 0.0;

    for focal_seat in 0..player_count {
        let mut pattern_rngs = (0..player_count)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let result =
            play_match_with_selector(game_config, seed, beam.strategy_id(), |player, game| {
                if player == focal_seat {
                    let started = Instant::now();
                    let action = beam.select_action_deterministic(game).map_err(|error| {
                        cascadia_sim::SimulationError::Strategy(error.to_string())
                    })?;
                    decision_seconds.push(started.elapsed().as_secs_f64());
                    Ok(action)
                } else {
                    select_pattern_action(
                        game,
                        &MarketPrelude {
                            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                            wildlife_wipes: Vec::new(),
                        },
                        blueprint,
                        &mut pattern_rngs[player],
                    )
                }
            })?;
        scores.push(result.scores[focal_seat]);
        elapsed_seconds += result.elapsed_seconds;
    }

    Ok(EvaluationBlock {
        scores,
        decision_seconds,
        elapsed_seconds,
    })
}

fn play_focal_portfolio_beam_block(
    game_config: GameConfig,
    seed: GameSeed,
    beam: &PerfectInformationPortfolioBeamStrategy,
    blueprint: PatternAwareConfig,
) -> Result<EvaluationBlock, Box<dyn std::error::Error>> {
    let player_count = usize::from(game_config.player_count);
    let mut scores = Vec::with_capacity(player_count);
    let mut decision_seconds = Vec::new();
    let mut elapsed_seconds = 0.0;

    for focal_seat in 0..player_count {
        let mut pattern_rngs = (0..player_count)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let result =
            play_match_with_selector(game_config, seed, beam.strategy_id(), |player, game| {
                if player == focal_seat {
                    let started = Instant::now();
                    let action = beam.select_action_deterministic(game).map_err(|error| {
                        cascadia_sim::SimulationError::Strategy(error.to_string())
                    })?;
                    decision_seconds.push(started.elapsed().as_secs_f64());
                    Ok(action)
                } else {
                    select_pattern_action(
                        game,
                        &MarketPrelude {
                            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                            wildlife_wipes: Vec::new(),
                        },
                        blueprint,
                        &mut pattern_rngs[player],
                    )
                }
            })?;
        scores.push(result.scores[focal_seat]);
        elapsed_seconds += result.elapsed_seconds;
    }

    Ok(EvaluationBlock {
        scores,
        decision_seconds,
        elapsed_seconds,
    })
}

fn play_focal_root_diverse_beam_block(
    game_config: GameConfig,
    seed: GameSeed,
    beam: &PerfectInformationRootDiverseBeamStrategy,
    blueprint: PatternAwareConfig,
) -> Result<EvaluationBlock, Box<dyn std::error::Error>> {
    let player_count = usize::from(game_config.player_count);
    let mut scores = Vec::with_capacity(player_count);
    let mut decision_seconds = Vec::new();
    let mut elapsed_seconds = 0.0;

    for focal_seat in 0..player_count {
        let mut pattern_rngs = (0..player_count)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let result =
            play_match_with_selector(game_config, seed, beam.strategy_id(), |player, game| {
                if player == focal_seat {
                    let started = Instant::now();
                    let action = beam.select_action_deterministic(game).map_err(|error| {
                        cascadia_sim::SimulationError::Strategy(error.to_string())
                    })?;
                    decision_seconds.push(started.elapsed().as_secs_f64());
                    Ok(action)
                } else {
                    select_pattern_action(
                        game,
                        &MarketPrelude {
                            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                            wildlife_wipes: Vec::new(),
                        },
                        blueprint,
                        &mut pattern_rngs[player],
                    )
                }
            })?;
        scores.push(result.scores[focal_seat]);
        elapsed_seconds += result.elapsed_seconds;
    }

    Ok(EvaluationBlock {
        scores,
        decision_seconds,
        elapsed_seconds,
    })
}
