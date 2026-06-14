use std::time::Instant;

use cascadia_data::{
    DatasetSplit, PublicBeamValueDatasetConfig, PublicBeamValueDatasetManifest,
    PublicBeamValueDatasetWriter, read_public_beam_value_shard_records,
    validate_public_beam_value_dataset,
};
use cascadia_sim::PatternAwareConfig;

use crate::cli::Command;
use crate::public_beam_probe::{
    collect_public_beam_value_game, public_beam_value_probe_config, public_beam_value_teacher,
    summarize_public_beam_value_probe,
};
use crate::report::{ReportContext, write_report};

pub fn run(
    command: Command,
    report_context: &ReportContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::CollectPublicBeamValue {
            output,
            games,
            first_game_index,
            split,
            resume,
        } => {
            if games == 0 {
                return Err("collect-public-beam-value requires a positive game count".into());
            }
            let split: DatasetSplit = split.into();
            let blueprint = PatternAwareConfig::default();
            let probe_config = public_beam_value_probe_config(blueprint)?;
            let mut writer = PublicBeamValueDatasetWriter::open(&PublicBeamValueDatasetConfig {
                output,
                split,
                first_game_index,
                games,
                teacher: public_beam_value_teacher(blueprint),
                resume,
            })?;
            while writer.manifest().completed_games < games {
                let game_index = first_game_index + writer.manifest().completed_games as u64;
                let records =
                    collect_public_beam_value_game(split, game_index, blueprint, probe_config)?;
                writer.append_shard(game_index, 1, &records)?;
                eprintln!(
                    "public beam-value dataset: {}/{} games, {} groups, {} candidates",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_groups,
                    writer.manifest().total_records,
                );
            }
            println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        }
        Command::PublicBeamValueProbe {
            output,
            first_game_index,
            games,
            resume,
            report,
        } => {
            if first_game_index != 40_000 || games != 2 {
                return Err(
                    "the frozen public beam-value probe is exactly train games 40000-40001".into(),
                );
            }
            let blueprint = PatternAwareConfig::default();
            let probe_config = public_beam_value_probe_config(blueprint)?;
            let teacher = public_beam_value_teacher(blueprint);
            let output_root = output.clone();
            let mut writer = PublicBeamValueDatasetWriter::open(&PublicBeamValueDatasetConfig {
                output,
                split: DatasetSplit::Train,
                first_game_index,
                games,
                teacher,
                resume,
            })?;
            let started = Instant::now();
            while writer.manifest().completed_games < games {
                let game_index = first_game_index + writer.manifest().completed_games as u64;
                let records = collect_public_beam_value_game(
                    DatasetSplit::Train,
                    game_index,
                    blueprint,
                    probe_config,
                )?;
                writer.append_shard(game_index, 1, &records)?;
                eprintln!(
                    "public beam-value probe: {}/{} games, {} groups, {} candidates",
                    writer.manifest().completed_games,
                    games,
                    writer.manifest().total_groups,
                    writer.manifest().total_records,
                );
            }
            let manifest = writer.manifest().clone();
            validate_public_beam_value_dataset(&output_root, &manifest)?;
            let mut records = Vec::with_capacity(manifest.total_records);
            for shard in &manifest.shards {
                records.extend(read_public_beam_value_shard_records(
                    &output_root,
                    manifest.split,
                    shard,
                )?);
            }
            let probe_report = summarize_public_beam_value_probe(
                &manifest,
                &records,
                started.elapsed().as_secs_f64(),
            )?;
            let json = report_context.to_json(&probe_report)?;
            if let Some(path) = report {
                write_report(&path, &json, &probe_report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::ValidatePublicBeamValueDataset { dataset } => {
            let manifest: PublicBeamValueDatasetManifest =
                serde_json::from_reader(std::fs::File::open(dataset.join("dataset.json"))?)?;
            validate_public_beam_value_dataset(&dataset, &manifest)?;
            println!(
                "validated {} games, {} groups, {} candidates, {} shards",
                manifest.completed_games,
                manifest.total_groups,
                manifest.total_records,
                manifest.shards.len()
            );
        }
        _ => unreachable!("public-beam data dispatcher received a different command family"),
    }
    Ok(())
}
