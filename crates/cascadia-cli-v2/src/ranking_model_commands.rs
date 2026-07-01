use std::time::Instant;

use cascadia_eval::{summarize_match_results, summarize_paired_match_results};
use cascadia_game::{GameConfig, GameSeed};
use cascadia_provenance::checksum_file;
use cascadia_search::{
    BearCandidateLookaheadConfig, BearCandidateLookaheadStrategy, DeterminizedLookaheadConfig,
    DeterminizedLookaheadStrategy, HabitatCandidateLookaheadConfig,
    HabitatCandidateLookaheadStrategy, MlxActionDeltaRankingConfig, MlxActionDeltaRankingStrategy,
    MlxFullActionImitationStrategy, MlxHabitatPrefilteredLookaheadConfig,
    MlxHabitatPrefilteredLookaheadStrategy, MlxHabitatRankingConfig, MlxHabitatRankingStrategy,
    MlxHabitatRolloutLookaheadConfig, MlxHabitatRolloutLookaheadStrategy, MlxPatternRankingConfig,
    MlxPatternRankingStrategy, MlxPrefilteredLookaheadConfig, MlxPrefilteredLookaheadStrategy,
    MlxRankingConfig, MlxRankingStrategy, MlxSelfRolloutLookaheadConfig,
    MlxSelfRolloutLookaheadStrategy,
};
use cascadia_sim::{MatchConfig, PatternAwareConfig, StrategyKind, play_match};

use crate::cli::{Command, HabitatRankingBaselineArg, RankingBaselineArg};
use crate::report::{ReportContext, model_source_args, write_report};

pub fn run(
    command: Command,
    report_context: &ReportContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::RankingModelBenchmark {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            candidates,
            bear_candidates,
            output,
        } => {
            if games == 0 {
                return Err("ranking-model-benchmark requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let mut strategy = MlxRankingStrategy::spawn(
                server,
                source_args,
                MlxRankingConfig {
                    immediate_candidate_limit: candidates,
                    bear_candidate_limit: bear_candidates,
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
            let report = summarize_match_results(
                &MlxRankingConfig {
                    immediate_candidate_limit: candidates,
                    bear_candidate_limit: bear_candidates,
                }
                .strategy_id(),
                games,
                first_seed,
                &matches,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::RankingModelCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            baseline,
            candidates,
            bear_candidates,
            output,
        } => {
            if games == 0 {
                return Err("ranking-model-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let action_config = DeterminizedLookaheadConfig {
                candidate_limit: 8,
                determinizations: 4,
                greedy_plies: 4,
            };
            let k8 = DeterminizedLookaheadStrategy::new(action_config)?;
            let teacher = BearCandidateLookaheadStrategy::new(BearCandidateLookaheadConfig {
                immediate_candidate_limit: candidates,
                bear_candidate_limit: bear_candidates,
                determinizations: 4,
                greedy_plies: 4,
            })?;
            let mut treatment = MlxRankingStrategy::spawn(
                server,
                source_args,
                MlxRankingConfig {
                    immediate_candidate_limit: candidates,
                    bear_candidate_limit: bear_candidates,
                },
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = match baseline {
                        RankingBaselineArg::K8 => k8.play_match(game_config, seed)?,
                        RankingBaselineArg::BearTeacher => teacher.play_match(game_config, seed)?,
                    };
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
            let baseline_id = match baseline {
                RankingBaselineArg::K8 => k8.strategy_id(),
                RankingBaselineArg::BearTeacher => teacher.strategy_id(),
            };
            let treatment_id = MlxRankingConfig {
                immediate_candidate_limit: candidates,
                bear_candidate_limit: bear_candidates,
            }
            .strategy_id();
            let report = summarize_paired_match_results(
                baseline_id,
                &treatment_id,
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
        Command::HabitatRankingModelBenchmark {
            games,
            first_seed,
            model_dir,
            server,
            candidates,
            habitat_candidates,
            output,
        } => {
            if games == 0 {
                return Err("habitat-ranking-model-benchmark requires at least one game".into());
            }
            let config = MlxHabitatRankingConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
            };
            let mut strategy = MlxHabitatRankingStrategy::spawn(
                server,
                [
                    std::ffi::OsString::from("--model-dir"),
                    model_dir.into_os_string(),
                ],
                config,
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
            let report = summarize_match_results(
                &config.strategy_id(),
                games,
                first_seed,
                &matches,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::HabitatRankingModelCompare {
            games,
            first_seed,
            model_dir,
            server,
            baseline,
            candidates,
            habitat_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("habitat-ranking-model-compare requires at least one game".into());
            }
            let config = MlxHabitatRankingConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
            };
            let teacher =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let mut treatment = MlxHabitatRankingStrategy::spawn(
                server,
                [
                    std::ffi::OsString::from("--model-dir"),
                    model_dir.into_os_string(),
                ],
                config,
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = match baseline {
                        HabitatRankingBaselineArg::PatternAware => play_match(
                            &MatchConfig::symmetric(game_config, seed, StrategyKind::PatternAware),
                        )?,
                        HabitatRankingBaselineArg::HabitatTeacher => {
                            teacher.play_match(game_config, seed)?
                        }
                    };
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
            let baseline_id = match baseline {
                HabitatRankingBaselineArg::PatternAware => StrategyKind::PatternAware.id(),
                HabitatRankingBaselineArg::HabitatTeacher => teacher.strategy_id(),
            };
            let report = summarize_paired_match_results(
                baseline_id,
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
        Command::PatternRankingModelCompare {
            games,
            first_seed,
            model_dir,
            server,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            output,
        } => {
            if games == 0 {
                return Err("pattern-ranking-model-compare requires at least one game".into());
            }
            let config = MlxPatternRankingConfig {
                blueprint: PatternAwareConfig {
                    immediate_candidate_limit: policy_candidates,
                    habitat_candidate_limit: policy_habitat_candidates,
                    bear_candidate_limit: policy_bear_candidates,
                    future_market_draws: policy_market_draws,
                },
            };
            let mut treatment = MlxPatternRankingStrategy::spawn(
                server,
                [
                    std::ffi::OsString::from("--model-dir"),
                    model_dir.into_os_string(),
                ],
                config,
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
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
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = treatment.shutdown();
            let pairs = result?;
            shutdown?;
            let report = summarize_paired_match_results(
                StrategyKind::PatternAware.id(),
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
        Command::ActionRankingModelCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            output,
        } => {
            if games == 0 {
                return Err("action-ranking-model-compare requires at least one game".into());
            }
            let config = MlxActionDeltaRankingConfig {
                blueprint: PatternAwareConfig {
                    immediate_candidate_limit: policy_candidates,
                    habitat_candidate_limit: policy_habitat_candidates,
                    bear_candidate_limit: policy_bear_candidates,
                    future_market_draws: policy_market_draws,
                },
            };
            let source_args = model_source_args(run_dir, model_dir)?;
            let mut treatment = MlxActionDeltaRankingStrategy::spawn(server, source_args, config)?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
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
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = treatment.shutdown();
            let pairs = result?;
            shutdown?;
            let report = summarize_paired_match_results(
                StrategyKind::PatternAware.id(),
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
        Command::FullActionImitationCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            output,
        } => {
            if games == 0 {
                return Err("full-action-imitation-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let mut treatment = MlxFullActionImitationStrategy::spawn(server, source_args)?;
            let treatment_id = treatment.strategy_id();
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
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
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let shutdown = treatment.shutdown();
            let pairs = result?;
            shutdown?;
            let report = summarize_paired_match_results(
                StrategyKind::PatternAware.id(),
                treatment_id,
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
        Command::HabitatRankingModelH2h {
            games,
            first_seed,
            baseline_model_dir,
            treatment_model_dir,
            server,
            candidates,
            habitat_candidates,
            output,
        } => {
            if games == 0 {
                return Err("habitat-ranking-model-h2h requires at least one game".into());
            }
            let config = MlxHabitatRankingConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
            };
            let baseline_manifest = baseline_model_dir.join("model.json");
            let treatment_manifest = treatment_model_dir.join("model.json");
            let baseline_digest = checksum_file(&baseline_manifest)?;
            let treatment_digest = checksum_file(&treatment_manifest)?;
            let mut baseline_strategy = MlxHabitatRankingStrategy::spawn(
                server.clone(),
                [
                    std::ffi::OsString::from("--model-dir"),
                    baseline_model_dir.into_os_string(),
                ],
                config,
            )?;
            let mut treatment_strategy = MlxHabitatRankingStrategy::spawn(
                server,
                [
                    std::ffi::OsString::from("--model-dir"),
                    treatment_model_dir.into_os_string(),
                ],
                config,
            )?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let result = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline_strategy.play_match(game_config, seed)?;
                    let treatment_result = treatment_strategy.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>();
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let baseline_shutdown = baseline_strategy.shutdown();
            let treatment_shutdown = treatment_strategy.shutdown();
            let pairs = result?;
            baseline_shutdown?;
            treatment_shutdown?;
            let report = summarize_paired_match_results(
                &format!("{}-{}", config.strategy_id(), &baseline_digest[..12]),
                &format!("{}-{}", config.strategy_id(), &treatment_digest[..12]),
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
        Command::RankingPrefilterCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            candidates,
            bear_candidates,
            immediate_anchors,
            prefilter_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("ranking-prefilter-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let baseline = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: candidates,
                determinizations,
                greedy_plies,
            })?;
            let treatment_config = MlxPrefilteredLookaheadConfig {
                immediate_candidate_limit: candidates,
                bear_candidate_limit: bear_candidates,
                immediate_anchor_limit: immediate_anchors,
                prefilter_candidate_limit: prefilter_candidates,
                determinizations,
                greedy_plies,
            };
            let mut treatment =
                MlxPrefilteredLookaheadStrategy::spawn(server, source_args, treatment_config)?;
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
        Command::RankingHabitatPrefilterCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            baseline_candidates,
            baseline_habitat_candidates,
            candidates,
            habitat_candidates,
            immediate_anchors,
            prefilter_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("ranking-habitat-prefilter-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: baseline_candidates,
                    habitat_candidate_limit: baseline_habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let treatment_config = MlxHabitatPrefilteredLookaheadConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
                immediate_anchor_limit: immediate_anchors,
                prefilter_candidate_limit: prefilter_candidates,
                determinizations,
                greedy_plies,
            };
            let mut treatment = MlxHabitatPrefilteredLookaheadStrategy::spawn(
                server,
                source_args,
                treatment_config,
            )?;
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
        Command::RankingHabitatRolloutCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            candidates,
            habitat_candidates,
            determinizations,
            rollout_plies,
            rollout_candidates,
            rollout_habitat_candidates,
            output,
        } => {
            if games == 0 {
                return Err("ranking-habitat-rollout-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies: rollout_plies,
                })?;
            let treatment_config = MlxHabitatRolloutLookaheadConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
                determinizations,
                rollout_plies,
                rollout_immediate_candidate_limit: rollout_candidates,
                rollout_habitat_candidate_limit: rollout_habitat_candidates,
            };
            let mut treatment =
                MlxHabitatRolloutLookaheadStrategy::spawn(server, source_args, treatment_config)?;
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
        Command::RankingSelfRolloutCompare {
            games,
            first_seed,
            run_dir,
            model_dir,
            server,
            candidates,
            habitat_candidates,
            determinizations,
            rollout_plies,
            policy_candidates,
            policy_habitat_candidates,
            output,
        } => {
            if games == 0 {
                return Err("ranking-self-rollout-compare requires at least one game".into());
            }
            let source_args = model_source_args(run_dir, model_dir)?;
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies: rollout_plies,
                })?;
            let treatment_config = MlxSelfRolloutLookaheadConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
                determinizations,
                rollout_plies,
                policy_immediate_candidate_limit: policy_candidates,
                policy_habitat_candidate_limit: policy_habitat_candidates,
            };
            let mut treatment =
                MlxSelfRolloutLookaheadStrategy::spawn(server, source_args, treatment_config)?;
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
        _ => unreachable!("ranking-model dispatcher received a different command family"),
    }
    Ok(())
}
