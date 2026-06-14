use std::time::Instant;

use cascadia_eval::summarize_paired_match_results;
use cascadia_game::{GameConfig, GameSeed};
use cascadia_search::{
    LateConservativeBasePolicyImprovementConfig, LateConservativeBasePolicyImprovementStrategy,
    LateConservativePolicyImprovementConfig, LateConservativePolicyImprovementStrategy,
    LateConservativeWildlifeFocusedPolicyImprovementConfig,
    LateConservativeWildlifeFocusedPolicyImprovementStrategy, LateTerminalPolicyImprovementConfig,
    LateTerminalPolicyImprovementStrategy, LateWildlifeDiversePolicyImprovementConfig,
    LateWildlifeDiversePolicyImprovementStrategy, TerminalPolicyImprovementConfig,
    TerminalPolicyImprovementStrategy, WildlifeDiverseTerminalPolicyImprovementConfig,
    WildlifeFocusedTerminalPolicyImprovementConfig,
};
use cascadia_sim::{MatchConfig, PatternAwareConfig, StrategyKind, play_match};
use rayon::prelude::*;

use crate::cli::Command;
use crate::report::{ReportContext, write_report};

pub fn run(
    command: Command,
    report_context: &ReportContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::TerminalPolicyImprovementCompare {
            games,
            first_seed,
            determinizations,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            output,
        } => {
            if games == 0 {
                return Err(
                    "terminal-policy-improvement-compare requires at least one game".into(),
                );
            }
            let treatment =
                TerminalPolicyImprovementStrategy::new(TerminalPolicyImprovementConfig {
                    determinizations,
                    blueprint: PatternAwareConfig {
                        immediate_candidate_limit: policy_candidates,
                        habitat_candidate_limit: policy_habitat_candidates,
                        bear_candidate_limit: policy_bear_candidates,
                        future_market_draws: policy_market_draws,
                    },
                })?;
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
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
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
        Command::LateTerminalPolicyImprovementCompare {
            games,
            first_seed,
            terminal_turns,
            determinizations,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err(
                    "late-terminal-policy-improvement-compare requires at least one game".into(),
                );
            }
            let treatment =
                LateTerminalPolicyImprovementStrategy::new(LateTerminalPolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: TerminalPolicyImprovementConfig {
                        determinizations,
                        blueprint: PatternAwareConfig {
                            immediate_candidate_limit: policy_candidates,
                            habitat_candidate_limit: policy_habitat_candidates,
                            bear_candidate_limit: policy_bear_candidates,
                            future_market_draws: policy_market_draws,
                        },
                    },
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = play_match(&MatchConfig::symmetric(
                    game_config,
                    seed,
                    StrategyKind::PatternAware,
                ))
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
        Command::LateWildlifeDiversePolicyImprovementCompare {
            games,
            first_seed,
            terminal_turns,
            determinizations,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err(
                    "late-wildlife-diverse-policy-improvement-compare requires at least one game"
                        .into(),
                );
            }
            let treatment = LateWildlifeDiversePolicyImprovementStrategy::new(
                LateWildlifeDiversePolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: WildlifeDiverseTerminalPolicyImprovementConfig {
                        wildlife_candidate_limit: wildlife_candidates,
                        terminal: TerminalPolicyImprovementConfig {
                            determinizations,
                            blueprint: PatternAwareConfig {
                                immediate_candidate_limit: policy_candidates,
                                habitat_candidate_limit: policy_habitat_candidates,
                                bear_candidate_limit: policy_bear_candidates,
                                future_market_draws: policy_market_draws,
                            },
                        },
                    },
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = play_match(&MatchConfig::symmetric(
                    game_config,
                    seed,
                    StrategyKind::PatternAware,
                ))
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
        Command::LateConservativePolicyImprovementCompare {
            games,
            first_seed,
            terminal_turns,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err(
                    "late-conservative-policy-improvement-compare requires at least one game"
                        .into(),
                );
            }
            let treatment = LateConservativePolicyImprovementStrategy::new(
                LateConservativePolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: WildlifeDiverseTerminalPolicyImprovementConfig {
                        wildlife_candidate_limit: wildlife_candidates,
                        terminal: TerminalPolicyImprovementConfig {
                            determinizations: 8,
                            blueprint: PatternAwareConfig {
                                immediate_candidate_limit: policy_candidates,
                                habitat_candidate_limit: policy_habitat_candidates,
                                bear_candidate_limit: policy_bear_candidates,
                                future_market_draws: policy_market_draws,
                            },
                        },
                    },
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = play_match(&MatchConfig::symmetric(
                    game_config,
                    seed,
                    StrategyKind::PatternAware,
                ))
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
        Command::LateConservativeBasePolicyImprovementCompare {
            games,
            first_seed,
            terminal_turns,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err(
                    "late-conservative-base-policy-improvement-compare requires at least one game"
                        .into(),
                );
            }
            let treatment = LateConservativeBasePolicyImprovementStrategy::new(
                LateConservativeBasePolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: TerminalPolicyImprovementConfig {
                        determinizations: 8,
                        blueprint: PatternAwareConfig {
                            immediate_candidate_limit: policy_candidates,
                            habitat_candidate_limit: policy_habitat_candidates,
                            bear_candidate_limit: policy_bear_candidates,
                            future_market_draws: policy_market_draws,
                        },
                    },
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = play_match(&MatchConfig::symmetric(
                    game_config,
                    seed,
                    StrategyKind::PatternAware,
                ))
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
        Command::LateConservativeWildlifeFocusedPolicyImprovementCompare {
            games,
            first_seed,
            terminal_turns,
            determinizations,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            wildlife,
            wildlife_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err(
                    "late-conservative-wildlife-focused-policy-improvement-compare requires at least one game"
                        .into(),
                );
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
                        determinizations,
                        blueprint,
                    },
                },
            )?;
            let treatment = LateConservativeWildlifeFocusedPolicyImprovementStrategy::new(
                LateConservativeWildlifeFocusedPolicyImprovementConfig {
                    final_personal_turns: terminal_turns,
                    terminal: WildlifeFocusedTerminalPolicyImprovementConfig {
                        wildlife: wildlife.into(),
                        wildlife_candidate_limit: wildlife_candidates,
                        terminal: TerminalPolicyImprovementConfig {
                            determinizations,
                            blueprint,
                        },
                    },
                },
            )?;
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
        Command::ConservativeSampleCountCompare {
            games,
            first_seed,
            terminal_turns,
            baseline_determinizations,
            treatment_determinizations,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err("conservative-sample-count-compare requires at least one game".into());
            }
            if baseline_determinizations == treatment_determinizations {
                return Err("conservative sample counts must differ".into());
            }
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let strategy = |determinizations| {
                LateConservativeBasePolicyImprovementStrategy::new(
                    LateConservativeBasePolicyImprovementConfig {
                        final_personal_turns: terminal_turns,
                        terminal: TerminalPolicyImprovementConfig {
                            determinizations,
                            blueprint,
                        },
                    },
                )
            };
            let baseline = strategy(baseline_determinizations)?;
            let treatment = strategy(treatment_determinizations)?;
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
        _ => unreachable!("policy dispatcher received a non-policy command"),
    }
    Ok(())
}
