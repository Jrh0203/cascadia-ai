use std::path::PathBuf;

use clap::{Parser, Subcommand};
use f5_corrected_tail_activation_census::{
    GenerateShardConfig, aggregate_reports, census_shard, generate_shard,
    verify_reports_byte_identical,
};

#[derive(Debug, Parser)]
#[command(
    name = "f5-corrected-tail-activation-census",
    about = "Source-frozen corrected historical mid-tail activation census"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    GenerateShard {
        #[arg(long)]
        output_root: PathBuf,
        #[arg(long)]
        shard_index: usize,
        #[arg(long, default_value_t = 4)]
        shard_count: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, default_value_t = 1024)]
        total_games: usize,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    CensusShard {
        #[arg(long)]
        corpus_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Aggregate {
        #[arg(long = "report", required = true)]
        reports: Vec<PathBuf>,
        #[arg(long, default_value_t = 4)]
        require_shards: usize,
        #[arg(long)]
        output: PathBuf,
    },
    VerifyOrder {
        #[arg(long)]
        left: PathBuf,
        #[arg(long)]
        right: PathBuf,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("corrected-tail activation census: {error}");
        std::process::exit(2);
    }
}

fn run() -> f5_corrected_tail_activation_census::Result<()> {
    match Cli::parse().command {
        Command::GenerateShard {
            output_root,
            shard_index,
            shard_count,
            first_game_index,
            total_games,
            threads,
        } => {
            let manifest = generate_shard(&GenerateShardConfig {
                output_root,
                shard_index,
                shard_count,
                first_game_index,
                total_games,
                threads,
            })?;
            println!("{}", manifest.scientific_blake3);
        }
        Command::CensusShard {
            corpus_root,
            output,
        } => {
            let report = census_shard(&corpus_root, &output)?;
            println!("{}", report.scientific_blake3);
        }
        Command::Aggregate {
            reports,
            require_shards,
            output,
        } => {
            let report = aggregate_reports(&reports, require_shards, &output)?;
            println!("{}", report.scientific_blake3);
        }
        Command::VerifyOrder { left, right } => {
            verify_reports_byte_identical(&left, &right)?;
            println!("byte-identical");
        }
    }
    Ok(())
}
