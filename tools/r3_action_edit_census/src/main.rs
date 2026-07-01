use std::{
    io::{self, Write},
    path::{Path, PathBuf},
};

use cascadia_game::{GameConfig, GameSeed, GameState};
use clap::{Parser, Subcommand};
use r3_action_edit_census::{
    CensusConfig, CorpusContract, PublicStateTrunk, R3_CENSUS_PROTOCOL_ID, R3_EXPERIMENT_ID,
    aggregate_census_files, capture_runtime_identity_checked, prove_aggregate_order, run_census,
    write_json_atomic,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "R3 exact action-centric local-patch/global-edit foundation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        turns: u16,
        #[arg(long, default_value_t = 0)]
        action_index: usize,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        packed_trunk: Option<PathBuf>,
        #[arg(long)]
        packed_edit: Option<PathBuf>,
    },
    Census {
        #[arg(long, default_value_t = 3_300_000)]
        train_first_seed: u64,
        #[arg(long, default_value_t = 16)]
        train_games: u32,
        #[arg(long, default_value_t = 3_400_000)]
        validation_first_seed: u64,
        #[arg(long, default_value_t = 4)]
        validation_games: u32,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        paid_wipe_sentinels: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        d6_sentinel_per_position: bool,
        #[arg(long)]
        shard_index: usize,
        #[arg(long, default_value_t = 4)]
        shard_count: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Aggregate {
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    ProveOrder {
        #[arg(long)]
        forward: PathBuf,
        #[arg(long)]
        reverse: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Identity {
        #[arg(long)]
        expected_source_bundle_blake3: Option<String>,
        #[arg(long)]
        expected_executable_blake3: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Inspect {
            seed,
            turns,
            action_index,
            output,
            packed_trunk,
            packed_edit,
        } => inspect(seed, turns, action_index, output, packed_trunk, packed_edit)?,
        Command::Census {
            train_first_seed,
            train_games,
            validation_first_seed,
            validation_games,
            paid_wipe_sentinels,
            d6_sentinel_per_position,
            shard_index,
            shard_count,
            output,
        } => {
            let report = run_census(&CensusConfig {
                experiment_id: R3_EXPERIMENT_ID.to_owned(),
                protocol_id: R3_CENSUS_PROTOCOL_ID.to_owned(),
                corpus: CorpusContract {
                    train_first_seed,
                    train_games,
                    validation_first_seed,
                    validation_games,
                    include_paid_wipe_sentinels: paid_wipe_sentinels,
                    d6_sentinel_per_position,
                },
                shard_index,
                shard_count,
            })?;
            write_json_atomic(&output, &report)?;
            println!(
                "shard {shard_index}/{shard_count}: {}",
                report.scientific_blake3
            );
        }
        Command::Aggregate { inputs, output } => {
            let report = aggregate_census_files(&inputs)?;
            write_json_atomic(&output, &report)?;
            println!("{}", report.scientific_blake3);
        }
        Command::ProveOrder {
            forward,
            reverse,
            output,
        } => {
            let proof = prove_aggregate_order(&forward, &reverse)?;
            write_json_atomic(&output, &proof)?;
            println!("{}", proof.scientific_blake3);
        }
        Command::Identity {
            expected_source_bundle_blake3,
            expected_executable_blake3,
            output,
        } => {
            let identity = capture_runtime_identity_checked(
                expected_source_bundle_blake3.as_deref(),
                expected_executable_blake3.as_deref(),
            )?;
            write_json_or_stdout(output.as_deref(), &identity)?;
        }
    }
    Ok(())
}

fn write_json_or_stdout(
    path: Option<&Path>,
    value: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = path {
        write_json_atomic(path, value)?;
    } else {
        let mut stdout = io::stdout().lock();
        serde_json::to_writer_pretty(&mut stdout, value)?;
        stdout.write_all(b"\n")?;
    }
    Ok(())
}

fn inspect(
    seed: u64,
    turns: u16,
    action_index: usize,
    output: Option<PathBuf>,
    packed_trunk: Option<PathBuf>,
    packed_edit: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(seed))?;
    for _ in 0..turns {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let actions = game.legal_turn_actions(&prelude)?;
        let index = deterministic_index(seed, game.completed_turns(), actions.len());
        game.apply(&actions[index])?;
    }
    let game_index = seed * 100 + u64::from(game.completed_turns());
    let trunk = PublicStateTrunk::observe(&game, game_index)?;
    let prepared = trunk.prepare_action_edits()?;
    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
    let observed = prepared.observe_legal_actions(&game, &prelude)?;
    let (_, edit) = observed.get(action_index).ok_or_else(|| {
        format!(
            "action index {action_index} is outside the legal set of {} actions",
            observed.len()
        )
    })?;
    prepared.apply(edit)?;
    if let Some(path) = output {
        std::fs::write(path, serde_json::to_vec_pretty(&edit)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&edit)?);
    }
    if let Some(path) = packed_trunk {
        std::fs::write(path, prepared.packed_bytes())?;
    }
    if let Some(path) = packed_edit {
        std::fs::write(path, edit.to_packed_bytes()?)?;
    }
    Ok(())
}

fn deterministic_index(seed: u64, turn: u16, len: usize) -> usize {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-r3-inspect-action-v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    (u64::from_le_bytes(bytes) % len as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preregistered_boolean_values_parse_explicitly() {
        let cli = Cli::try_parse_from([
            "r3-action-edit-census",
            "census",
            "--paid-wipe-sentinels",
            "true",
            "--d6-sentinel-per-position",
            "true",
            "--shard-index",
            "2",
            "--output",
            "/tmp/r3.json",
        ])
        .unwrap();
        match cli.command {
            Command::Census {
                paid_wipe_sentinels,
                d6_sentinel_per_position,
                shard_index,
                shard_count,
                ..
            } => {
                assert!(paid_wipe_sentinels);
                assert!(d6_sentinel_per_position);
                assert_eq!(shard_index, 2);
                assert_eq!(shard_count, 4);
            }
            _ => panic!("parsed the wrong subcommand"),
        }
    }

    #[test]
    fn production_census_requires_an_explicit_shard_index() {
        assert!(
            Cli::try_parse_from([
                "r3-action-edit-census",
                "census",
                "--output",
                "/tmp/r3.json",
            ])
            .is_err()
        );
    }

    #[test]
    fn identity_accepts_fail_closed_expected_hashes() {
        let cli = Cli::try_parse_from([
            "r3-action-edit-census",
            "identity",
            "--expected-source-bundle-blake3",
            &"a".repeat(64),
            "--expected-executable-blake3",
            &"b".repeat(64),
            "--output",
            "/tmp/r3-identity.json",
        ])
        .unwrap();
        match cli.command {
            Command::Identity {
                expected_source_bundle_blake3,
                expected_executable_blake3,
                output,
            } => {
                assert_eq!(expected_source_bundle_blake3, Some("a".repeat(64)));
                assert_eq!(expected_executable_blake3, Some("b".repeat(64)));
                assert_eq!(output, Some(PathBuf::from("/tmp/r3-identity.json")));
            }
            _ => panic!("parsed the wrong subcommand"),
        }
    }
}
