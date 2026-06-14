use std::time::Instant;

use cascadia_data::{
    COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING, CollectConfig,
    CounterfactualAdvantageDatasetConfig, CounterfactualAdvantageDatasetManifest,
    CounterfactualAdvantageDatasetWriter, CounterfactualAdvantageTeacherConfig,
    CounterfactualValueDatasetConfig, CounterfactualValueDatasetManifest,
    CounterfactualValueDatasetWriter, CounterfactualValueTeacherConfig, DatasetManifest,
    DatasetSplit, DatasetWriter, DatasetWriterConfig, PositionRecord, ScoreToGoDatasetConfig,
    ScoreToGoDatasetManifest, ScoreToGoDatasetWriter, ScoreToGoRecord, ScoreToGoTeacherConfig,
    collect_dataset, validate_counterfactual_advantage_dataset,
    validate_counterfactual_value_dataset, validate_dataset, validate_score_to_go_dataset,
};
use cascadia_game::{GameConfig, GameState, score_board, score_game};
use cascadia_search::{HabitatCandidateLookaheadConfig, HabitatCandidateLookaheadStrategy};
use rayon::prelude::*;

use crate::cli::Command;
use crate::counterfactual::{
    audit_counterfactual_value_dataset, collect_counterfactual_value_game, write_json_atomic,
};
use crate::counterfactual_advantage::{
    CounterfactualCandidateSelectionArg, audit_counterfactual_advantage_dataset,
    collect_counterfactual_advantage_game, render_counterfactual_advantage_markdown,
    write_text_atomic,
};

pub fn run(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::Collect {
            output,
            games,
            first_game_index,
            split,
            strategy,
            shard_games,
            resume,
        } => {
            let manifest = collect_dataset(&CollectConfig {
                output,
                split: split.into(),
                first_game_index,
                games,
                shard_games,
                strategy: strategy.into(),
                resume,
            })?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::CollectSearch {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
            candidates,
            habitat_candidates,
            determinizations,
            greedy_plies,
        } => {
            if games == 0 || shard_games == 0 {
                return Err("collect-search requires positive game and shard counts".into());
            }
            let split: DatasetSplit = split.into();
            let strategy =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let mut writer = DatasetWriter::open(&DatasetWriterConfig {
                output,
                split,
                first_game_index,
                games,
                strategy_id: strategy.strategy_id().to_owned(),
                resume,
            })?;
            while writer.manifest().completed_games < games {
                let game_count = shard_games.min(games - writer.manifest().completed_games);
                let shard_first = first_game_index + writer.manifest().completed_games as u64;
                let mut games_with_records = (shard_first..shard_first + game_count as u64)
                    .into_par_iter()
                    .map(|game_index| {
                        collect_search_game(&strategy, split, game_index)
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
                    "collected {} / {} H6 games",
                    writer.manifest().completed_games,
                    games
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::CollectScoreToGo {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
        } => {
            if games == 0 || shard_games == 0 {
                return Err("collect-score-to-go requires positive game and shard counts".into());
            }
            let split: DatasetSplit = split.into();
            let strategy =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: 8,
                    habitat_candidate_limit: 6,
                    determinizations: 4,
                    greedy_plies: 4,
                })?;
            let teacher = ScoreToGoTeacherConfig {
                strategy_id: strategy.strategy_id().to_owned(),
                immediate_candidates: 8,
                habitat_candidates: 6,
                determinizations: 4,
                greedy_plies: 4,
            };
            let mut writer = ScoreToGoDatasetWriter::open(&ScoreToGoDatasetConfig {
                output,
                split,
                first_game_index,
                games,
                teacher,
                resume,
            })?;
            while writer.manifest().completed_games < games {
                let remaining_games = games - writer.manifest().completed_games;
                let collection_batch_games = shard_games
                    .saturating_mul(rayon::current_num_threads())
                    .min(remaining_games);
                let batch_first = first_game_index + writer.manifest().completed_games as u64;
                let mut games_with_records = (batch_first
                    ..batch_first + collection_batch_games as u64)
                    .into_par_iter()
                    .map(|game_index| {
                        collect_score_to_go_game(&strategy, split, game_index)
                            .map(|records| (game_index, records))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(std::io::Error::other)?;
                games_with_records.sort_unstable_by_key(|(game_index, _)| *game_index);
                let mut games_with_records = games_with_records.into_iter();
                loop {
                    let shard_games_with_records = games_with_records
                        .by_ref()
                        .take(shard_games)
                        .collect::<Vec<_>>();
                    if shard_games_with_records.is_empty() {
                        break;
                    }
                    let shard_first = shard_games_with_records[0].0;
                    let game_count = shard_games_with_records.len();
                    let records = shard_games_with_records
                        .into_iter()
                        .flat_map(|(_, records)| records)
                        .collect::<Vec<_>>();
                    writer.append_shard(shard_first, game_count, &records)?;
                    eprintln!(
                        "score-to-go dataset: {}/{} games, {} records",
                        writer.manifest().completed_games,
                        games,
                        writer.manifest().total_records,
                    );
                }
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::CollectCounterfactualValue {
            output,
            games,
            first_game_index,
            split,
            samples_per_state,
            resume,
        } => {
            if games == 0 || !(1..=16).contains(&samples_per_state) {
                return Err(
                    "collect-counterfactual-value requires positive games and 1-16 samples".into(),
                );
            }
            let split: DatasetSplit = split.into();
            let strategy =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: 8,
                    habitat_candidate_limit: 6,
                    determinizations: 4,
                    greedy_plies: 4,
                })?;
            let teacher = CounterfactualValueTeacherConfig {
                strategy_id: strategy.strategy_id().to_owned(),
                immediate_candidates: 8,
                habitat_candidates: 6,
                determinizations: 4,
                greedy_plies: 4,
                samples_per_state,
                sample_seed_domain: "cascadia-v2-counterfactual-value-v1".to_owned(),
            };
            let mut writer =
                CounterfactualValueDatasetWriter::open(&CounterfactualValueDatasetConfig {
                    output,
                    split,
                    first_game_index,
                    games,
                    teacher,
                    resume,
                })?;
            let previous_milliseconds = writer.manifest().collection_milliseconds;
            let started = Instant::now();
            while writer.manifest().completed_games < games {
                let game_index = first_game_index + writer.manifest().completed_games as u64;
                let records = collect_counterfactual_value_game(
                    &strategy,
                    split,
                    game_index,
                    samples_per_state,
                )?;
                writer.append_game(game_index, &records)?;
                writer.set_collection_milliseconds(previous_milliseconds.saturating_add(
                    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                ))?;
                eprintln!(
                    "counterfactual-value dataset: {}/{} games, {} states, {} continuations, {:.1}s",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_records,
                    writer.manifest().total_continuations,
                    writer.manifest().collection_milliseconds as f64 / 1000.0,
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::CollectCounterfactualAdvantage {
            output,
            games,
            first_game_index,
            split,
            groups_per_game,
            samples_per_candidate,
            candidate_selection,
            resume,
        } => {
            if games == 0
                || groups_per_game == 0
                || groups_per_game > 80
                || !80usize.is_multiple_of(groups_per_game)
                || !(1..=16).contains(&samples_per_candidate)
            {
                return Err(
                    "collect-counterfactual-advantage requires positive games, a group count dividing 80, and 1-16 samples"
                        .into(),
                );
            }
            let split: DatasetSplit = split.into();
            let strategy =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: 8,
                    habitat_candidate_limit: 6,
                    determinizations: 4,
                    greedy_plies: 4,
                })?;
            let teacher = CounterfactualAdvantageTeacherConfig {
                strategy_id: strategy.strategy_id().to_owned(),
                immediate_candidates: 8,
                habitat_candidates: 6,
                determinizations: 4,
                greedy_plies: 4,
                candidate_count: 4,
                groups_per_game,
                samples_per_candidate,
                sample_seed_domain: "cascadia-v2-counterfactual-advantage-v1".to_owned(),
                candidate_selection: (candidate_selection
                    != CounterfactualCandidateSelectionArg::Nearest)
                    .then(|| candidate_selection.id().to_owned()),
                stabilization_conditioning: Some(
                    COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING.to_owned(),
                ),
            };
            let mut writer = CounterfactualAdvantageDatasetWriter::open(
                &CounterfactualAdvantageDatasetConfig {
                    output,
                    split,
                    first_game_index,
                    games,
                    teacher,
                    resume,
                },
            )?;
            let previous_milliseconds = writer.manifest().collection_milliseconds;
            let started = Instant::now();
            while writer.manifest().completed_games < games {
                let game_index = first_game_index + writer.manifest().completed_games as u64;
                let records = collect_counterfactual_advantage_game(
                    &strategy,
                    split,
                    game_index,
                    groups_per_game,
                    samples_per_candidate,
                    candidate_selection,
                )?;
                writer.append_game(game_index, &records)?;
                writer.set_collection_milliseconds(previous_milliseconds.saturating_add(
                    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                ))?;
                eprintln!(
                    "counterfactual-advantage dataset: {}/{} games, {} groups, {} candidates, {} continuations, {:.1}s",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_groups,
                    writer.manifest().total_candidates,
                    writer.manifest().total_continuations,
                    writer.manifest().collection_milliseconds as f64 / 1000.0,
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::ValidateDataset { dataset } => {
            let manifest: DatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} records, {} shards",
                manifest.completed_games,
                manifest.total_records,
                manifest.shards.len()
            );
        }
        Command::ValidateScoreToGoDataset { dataset } => {
            let manifest: ScoreToGoDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_score_to_go_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} score-to-go records, {} shards",
                manifest.completed_games,
                manifest.total_records,
                manifest.shards.len()
            );
        }
        Command::ValidateCounterfactualValueDataset { dataset } => {
            let manifest: CounterfactualValueDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_counterfactual_value_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} public states, {} continuations, {} shards",
                manifest.completed_games,
                manifest.total_records,
                manifest.total_continuations,
                manifest.shards.len()
            );
        }
        Command::ValidateCounterfactualAdvantageDataset { dataset } => {
            let manifest: CounterfactualAdvantageDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_counterfactual_advantage_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} groups, {} candidates, {} continuations, {} shards",
                manifest.completed_games,
                manifest.total_groups,
                manifest.total_candidates,
                manifest.total_continuations,
                manifest.shards.len()
            );
        }
        Command::AuditCounterfactualValueDataset { dataset, output } => {
            let manifest: CounterfactualValueDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            let report = audit_counterfactual_value_dataset(&dataset, &manifest)?;
            write_json_atomic(&output, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::AuditCounterfactualAdvantageDataset {
            dataset,
            output,
            markdown_output,
            estimator_samples,
        } => {
            let manifest: CounterfactualAdvantageDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            let report =
                audit_counterfactual_advantage_dataset(&dataset, &manifest, estimator_samples)?;
            write_json_atomic(&output, &report)?;
            write_text_atomic(
                &markdown_output,
                &render_counterfactual_advantage_markdown(&report),
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        _ => unreachable!("basic-data dispatcher received a different command family"),
    }
    Ok(())
}

fn collect_search_game(
    strategy: &HabitatCandidateLookaheadStrategy,
    split: DatasetSplit,
    game_index: u64,
) -> Result<Vec<PositionRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut records = Vec::with_capacity(80);
    while !game.is_game_over() {
        records.push(PositionRecord::observe(&game, game_index));
        let action = strategy.select_action_deterministic(&game)?;
        game.apply(&action)?;
    }
    let scores = score_game(&game);
    for record in &mut records {
        record.set_target(scores[usize::from(record.active_seat)]);
    }
    Ok(records)
}

fn collect_score_to_go_game(
    strategy: &HabitatCandidateLookaheadStrategy,
    split: DatasetSplit,
    game_index: u64,
) -> Result<Vec<ScoreToGoRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut positions = Vec::with_capacity(80);
    while !game.is_game_over() {
        let active_seat = game.current_player();
        positions.push((
            PositionRecord::observe(&game, game_index),
            score_board(&game.boards()[active_seat], game.config().scoring_cards),
        ));
        let action = strategy.select_action_deterministic(&game)?;
        game.apply(&action)?;
    }
    let scores = score_game(&game);
    positions
        .into_iter()
        .map(|(position, current)| {
            let final_score = scores[usize::from(position.active_seat)];
            ScoreToGoRecord::new(position, current, final_score)
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>)
        })
        .collect()
}
