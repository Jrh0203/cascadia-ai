use cascadia_data::{
    ActionRankingDatasetManifest, ConservativeAdvantageDatasetConfig,
    ConservativeAdvantageDatasetManifest, ConservativeAdvantageDatasetWriter,
    ConservativeAdvantageTeacherConfig, DatasetSplit, RankingCandidateFamily, RankingDatasetConfig,
    RankingDatasetManifest, RankingDatasetWriter, RankingTeacherConfig, RankingTrajectoryConfig,
    validate_action_ranking_dataset, validate_conservative_advantage_dataset,
    validate_ranking_dataset,
};
use cascadia_provenance::checksum_file;
use cascadia_search::{
    BearCandidateLookaheadConfig, BearCandidateLookaheadStrategy, HabitatCandidateLookaheadConfig,
    HabitatCandidateLookaheadStrategy, LateConservativeBasePolicyImprovementConfig,
    LateConservativeBasePolicyImprovementStrategy, MlxHabitatRankingConfig,
    MlxHabitatRankingStrategy, TerminalPolicyImprovementConfig, TerminalPolicyImprovementStrategy,
};
use cascadia_sim::PatternAwareConfig;
use rayon::prelude::*;

use crate::cli::{Command, RankingTeacherArg};
use crate::ranking_data::{
    RankingTeacherStrategy, collect_conservative_advantage_game, collect_ranking_game,
    collect_ranking_iteration_game, collect_terminal_ranking_game, enrich_action_ranking_dataset,
};

pub fn run(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::CollectRanking {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
            teacher,
            candidates,
            bear_candidates,
            habitat_candidates,
            determinizations,
            greedy_plies,
        } => {
            if shard_games == 0 {
                return Err("collect-ranking requires a positive shard size".into());
            }
            let split: DatasetSplit = split.into();
            let (teacher, candidate_family, active_bear_candidates, active_habitat_candidates) =
                match teacher {
                    RankingTeacherArg::Bear => (
                        RankingTeacherStrategy::Bear(BearCandidateLookaheadStrategy::new(
                            BearCandidateLookaheadConfig {
                                immediate_candidate_limit: candidates,
                                bear_candidate_limit: bear_candidates,
                                determinizations,
                                greedy_plies,
                            },
                        )?),
                        RankingCandidateFamily::Bear,
                        bear_candidates,
                        0,
                    ),
                    RankingTeacherArg::Habitat => (
                        RankingTeacherStrategy::Habitat(HabitatCandidateLookaheadStrategy::new(
                            HabitatCandidateLookaheadConfig {
                                immediate_candidate_limit: candidates,
                                habitat_candidate_limit: habitat_candidates,
                                determinizations,
                                greedy_plies,
                            },
                        )?),
                        RankingCandidateFamily::Habitat,
                        0,
                        habitat_candidates,
                    ),
                };
            let teacher_config = RankingTeacherConfig {
                strategy_id: teacher.strategy_id().to_owned(),
                immediate_candidates: candidates,
                candidate_family,
                bear_candidates: active_bear_candidates,
                habitat_candidates: active_habitat_candidates,
                determinizations,
                greedy_plies,
                terminal_continuation_strategy_id: None,
            };
            let mut writer = RankingDatasetWriter::open(&RankingDatasetConfig {
                output,
                split,
                first_game_index,
                games,
                teacher: teacher_config,
                trajectory: None,
                resume,
            })?;
            while writer.manifest().completed_games < games {
                let completed = writer.manifest().completed_games;
                let game_count = shard_games.min(games - completed);
                let shard_first = first_game_index + completed as u64;
                let mut games_with_records = (shard_first..shard_first + game_count as u64)
                    .into_par_iter()
                    .map(|game_index| {
                        collect_ranking_game(&teacher, split, game_index)
                            .map(|records| (game_index, records))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(std::io::Error::other)?;
                games_with_records.sort_unstable_by_key(|(game_index, _)| *game_index);
                let records = games_with_records
                    .into_iter()
                    .flat_map(|(_, records)| records)
                    .collect::<Vec<_>>();
                writer.append_shard(shard_first, game_count, &records)?;
                eprintln!(
                    "ranking dataset: {}/{} games, {} groups, {} candidates",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_groups,
                    writer.manifest().total_records,
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::CollectTerminalRanking {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
            determinizations,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
        } => {
            if games == 0 || shard_games == 0 {
                return Err(
                    "collect-terminal-ranking requires positive game and shard counts".into(),
                );
            }
            let split: DatasetSplit = split.into();
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let teacher =
                TerminalPolicyImprovementStrategy::new(TerminalPolicyImprovementConfig {
                    determinizations,
                    blueprint,
                })?;
            let mut writer = RankingDatasetWriter::open(&RankingDatasetConfig {
                output,
                split,
                first_game_index,
                games,
                teacher: RankingTeacherConfig {
                    strategy_id: teacher.strategy_id().to_owned(),
                    immediate_candidates: policy_candidates,
                    candidate_family: RankingCandidateFamily::Pattern,
                    bear_candidates: policy_bear_candidates,
                    habitat_candidates: policy_habitat_candidates,
                    determinizations,
                    greedy_plies: 0,
                    terminal_continuation_strategy_id: Some(blueprint.strategy_id()),
                },
                trajectory: None,
                resume,
            })?;
            while writer.manifest().completed_games < games {
                let completed = writer.manifest().completed_games;
                let game_count = shard_games.min(games - completed);
                let shard_first = first_game_index + completed as u64;
                let mut games_with_records = (shard_first..shard_first + game_count as u64)
                    .into_par_iter()
                    .map(|game_index| {
                        collect_terminal_ranking_game(&teacher, split, game_index)
                            .map(|records| (game_index, records))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(std::io::Error::other)?;
                games_with_records.sort_unstable_by_key(|(game_index, _)| *game_index);
                let records = games_with_records
                    .into_iter()
                    .flat_map(|(_, records)| records)
                    .collect::<Vec<_>>();
                writer.append_shard(shard_first, game_count, &records)?;
                eprintln!(
                    "terminal ranking dataset: {}/{} games, {} groups, {} candidates",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_groups,
                    writer.manifest().total_records,
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::CollectConservativeAdvantage {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
            terminal_turns,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
        } => {
            if games == 0 || shard_games == 0 {
                return Err(
                    "collect-conservative-advantage requires positive game and shard counts".into(),
                );
            }
            let split: DatasetSplit = split.into();
            let blueprint = PatternAwareConfig {
                immediate_candidate_limit: policy_candidates,
                habitat_candidate_limit: policy_habitat_candidates,
                bear_candidate_limit: policy_bear_candidates,
                future_market_draws: policy_market_draws,
            };
            let config = LateConservativeBasePolicyImprovementConfig {
                final_personal_turns: terminal_turns,
                terminal: TerminalPolicyImprovementConfig {
                    determinizations: 8,
                    blueprint,
                },
            };
            let teacher = LateConservativeBasePolicyImprovementStrategy::new(config)?;
            let mut writer =
                ConservativeAdvantageDatasetWriter::open(&ConservativeAdvantageDatasetConfig {
                    output,
                    split,
                    first_game_index,
                    games,
                    teacher: ConservativeAdvantageTeacherConfig {
                        strategy_id: teacher.strategy_id().to_owned(),
                        final_personal_turns: terminal_turns,
                        determinizations: 8,
                        immediate_candidates: policy_candidates,
                        habitat_candidates: policy_habitat_candidates,
                        bear_candidates: policy_bear_candidates,
                        future_market_draws: policy_market_draws,
                        confidence_percent: 90,
                        anchor_strategy_id: blueprint.strategy_id(),
                        continuation_strategy_id: blueprint.strategy_id(),
                    },
                    resume,
                })?;
            while writer.manifest().completed_games < games {
                let completed = writer.manifest().completed_games;
                let game_count = shard_games.min(games - completed);
                let shard_first = first_game_index + completed as u64;
                let mut games_with_records = (shard_first..shard_first + game_count as u64)
                    .into_par_iter()
                    .map(|game_index| {
                        collect_conservative_advantage_game(&teacher, blueprint, split, game_index)
                            .map(|records| (game_index, records))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(std::io::Error::other)?;
                games_with_records.sort_unstable_by_key(|(game_index, _)| *game_index);
                let records = games_with_records
                    .into_iter()
                    .flat_map(|(_, records)| records)
                    .collect::<Vec<_>>();
                writer.append_shard(shard_first, game_count, &records)?;
                eprintln!(
                    "conservative advantage dataset: {}/{} games, {} groups, {} challengers",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_groups,
                    writer.manifest().total_records,
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::CollectRankingIteration {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
            model_dir,
            server,
            candidates,
            habitat_candidates,
            determinizations,
            greedy_plies,
        } => {
            if games == 0 || shard_games == 0 {
                return Err(
                    "collect-ranking-iteration requires positive game and shard counts".into(),
                );
            }
            let split: DatasetSplit = split.into();
            let teacher =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let apprentice_config = MlxHabitatRankingConfig {
                immediate_candidate_limit: candidates,
                habitat_candidate_limit: habitat_candidates,
            };
            let model_manifest = model_dir.join("model.json").canonicalize()?;
            let trajectory = RankingTrajectoryConfig {
                strategy_id: apprentice_config.strategy_id(),
                model_manifest: model_manifest.display().to_string(),
                model_manifest_blake3: checksum_file(&model_manifest)?,
            };
            let mut apprentice = MlxHabitatRankingStrategy::spawn(
                server,
                [
                    std::ffi::OsString::from("--model-dir"),
                    model_dir.into_os_string(),
                ],
                apprentice_config,
            )?;
            let result = (|| -> Result<_, Box<dyn std::error::Error>> {
                let mut writer = RankingDatasetWriter::open(&RankingDatasetConfig {
                    output,
                    split,
                    first_game_index,
                    games,
                    teacher: RankingTeacherConfig {
                        strategy_id: teacher.strategy_id().to_owned(),
                        immediate_candidates: candidates,
                        candidate_family: RankingCandidateFamily::Habitat,
                        bear_candidates: 0,
                        habitat_candidates,
                        determinizations,
                        greedy_plies,
                        terminal_continuation_strategy_id: None,
                    },
                    trajectory: Some(trajectory),
                    resume,
                })?;
                while writer.manifest().completed_games < games {
                    let completed = writer.manifest().completed_games;
                    let game_count = shard_games.min(games - completed);
                    let shard_first = first_game_index + completed as u64;
                    let mut records = Vec::new();
                    for game_index in shard_first..shard_first + game_count as u64 {
                        records.extend(collect_ranking_iteration_game(
                            &teacher,
                            &mut apprentice,
                            split,
                            game_index,
                        )?);
                    }
                    writer.append_shard(shard_first, game_count, &records)?;
                    eprintln!(
                        "ranking iteration dataset: {}/{} games, {} groups, {} candidates",
                        writer.manifest().completed_games,
                        games,
                        writer.manifest().total_groups,
                        writer.manifest().total_records,
                    );
                }
                Ok(writer.manifest().clone())
            })();
            let shutdown = apprentice.shutdown();
            let manifest = result?;
            shutdown?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::ValidateRankingDataset { dataset } => {
            let manifest: RankingDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_ranking_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} groups, {} candidates, {} shards",
                manifest.completed_games,
                manifest.total_groups,
                manifest.total_records,
                manifest.shards.len()
            );
        }
        Command::EnrichActionRanking {
            source_dataset,
            output,
            resume,
            policy_market_draws,
        } => {
            let manifest = enrich_action_ranking_dataset(
                &source_dataset,
                output,
                resume,
                policy_market_draws,
            )?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::ValidateActionRankingDataset { dataset } => {
            let manifest: ActionRankingDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_action_ranking_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} groups, {} candidates, {} shards",
                manifest.completed_games,
                manifest.total_groups,
                manifest.total_records,
                manifest.shards.len()
            );
        }
        Command::ValidateConservativeAdvantageDataset { dataset } => {
            let manifest: ConservativeAdvantageDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_conservative_advantage_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} groups, {} challengers, {} shards",
                manifest.completed_games,
                manifest.total_groups,
                manifest.total_records,
                manifest.shards.len()
            );
        }
        _ => unreachable!("ranking-data dispatcher received a different command family"),
    }
    Ok(())
}
