use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use relational_feature_census::{
    CommonConfig, ExperimentLane, run_r5, run_r6, run_s3, run_s5, run_s6, write_json_atomic,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Lane {
    R5,
    R6,
    S3,
    S5,
    S6,
}

impl From<Lane> for ExperimentLane {
    fn from(value: Lane) -> Self {
        match value {
            Lane::R5 => Self::R5Quotient,
            Lane::R6 => Self::R6Incremental,
            Lane::S3 => Self::S3ComponentMotif,
            Lane::S5 => Self::S5Derivatives,
            Lane::S6 => Self::S6Topology,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "relational-feature-census",
    about = "Run one exact R5/R6/S3/S5/S6 Cascadia relational-feature lane"
)]
struct Cli {
    #[arg(long, value_enum)]
    lane: Lane,
    #[arg(long)]
    first_seed: u64,
    #[arg(long)]
    games: u32,
    #[arg(long)]
    source_bundle_id: String,
    #[arg(long)]
    host: String,
    #[arg(long, default_value_t = 8)]
    rayon_threads: usize,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    let config = CommonConfig {
        lane: args.lane.into(),
        first_seed: args.first_seed,
        games: args.games,
        source_bundle_id: args.source_bundle_id,
        host: args.host,
        rayon_threads: args.rayon_threads,
    };
    let receipt = match args.lane {
        Lane::R5 => {
            let report = run_r5(config)?;
            let receipt = serde_json::json!({
                "experiment_id": report.scientific.experiment_id,
                "scientific_blake3": report.scientific_blake3,
                "passed": report.scientific.passed,
                "classification": report.scientific.classification,
                "output": args.output,
            });
            write_json_atomic(&args.output, &report)?;
            receipt
        }
        Lane::R6 => {
            let report = run_r6(config)?;
            let receipt = serde_json::json!({
                "experiment_id": report.scientific.experiment_id,
                "scientific_blake3": report.scientific_blake3,
                "passed": report.scientific.passed,
                "classification": report.scientific.classification,
                "output": args.output,
            });
            write_json_atomic(&args.output, &report)?;
            receipt
        }
        Lane::S3 => {
            let report = run_s3(config)?;
            let receipt = serde_json::json!({
                "experiment_id": report.scientific.experiment_id,
                "scientific_blake3": report.scientific_blake3,
                "passed": report.scientific.passed,
                "classification": report.scientific.classification,
                "output": args.output,
            });
            write_json_atomic(&args.output, &report)?;
            receipt
        }
        Lane::S5 => {
            let report = run_s5(config)?;
            let receipt = serde_json::json!({
                "experiment_id": report.scientific.experiment_id,
                "scientific_blake3": report.scientific_blake3,
                "passed": report.scientific.passed,
                "classification": report.scientific.classification,
                "output": args.output,
            });
            write_json_atomic(&args.output, &report)?;
            receipt
        }
        Lane::S6 => {
            let report = run_s6(config)?;
            let receipt = serde_json::json!({
                "experiment_id": report.scientific.experiment_id,
                "scientific_blake3": report.scientific_blake3,
                "passed": report.scientific.passed,
                "classification": report.scientific.classification,
                "output": args.output,
            });
            write_json_atomic(&args.output, &report)?;
            receipt
        }
    };
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
