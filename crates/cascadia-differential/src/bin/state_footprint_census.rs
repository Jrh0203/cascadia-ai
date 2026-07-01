use std::path::PathBuf;

use cascadia_differential::state_footprint::{
    PREREGISTERED_FIRST_SEED, PREREGISTERED_GAMES, StateFootprintConfig,
    merge_state_footprint_report_files, run_state_footprint_census,
    write_state_footprint_report_atomic,
};
use cascadia_sim::StrategyKind;
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "state-footprint-census",
    about = "Deterministic compact-state footprint census and exact report merger"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run(RunArgs),
    Merge(MergeArgs),
}

#[derive(Debug, Args)]
struct RunArgs {
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = PREREGISTERED_FIRST_SEED)]
    first_seed: u64,

    #[arg(long, default_value_t = PREREGISTERED_GAMES)]
    games: usize,

    #[arg(long, value_enum, default_value_t = StrategyArg::PatternAware)]
    strategy: StrategyArg,

    #[arg(long = "position-dataset-root")]
    position_dataset_roots: Vec<PathBuf>,

    #[arg(long = "graded-dataset-root")]
    graded_dataset_roots: Vec<PathBuf>,

    #[arg(long, default_value_t = 1_000_000)]
    outlier_cap: usize,
}

#[derive(Debug, Args)]
struct MergeArgs {
    #[arg(long = "input", required = true, num_args = 2..)]
    inputs: Vec<PathBuf>,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum StrategyArg {
    Random,
    Greedy,
    PatternAware,
}

impl From<StrategyArg> for StrategyKind {
    fn from(value: StrategyArg) -> Self {
        match value {
            StrategyArg::Random => Self::Random,
            StrategyArg::Greedy => Self::Greedy,
            StrategyArg::PatternAware => Self::PatternAware,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Run(args) => {
            let report = run_state_footprint_census(
                &StateFootprintConfig {
                    first_seed: args.first_seed,
                    games: args.games,
                    strategy: args.strategy.into(),
                    position_dataset_roots: args.position_dataset_roots,
                    graded_dataset_roots: args.graded_dataset_roots,
                    outlier_cap: args.outlier_cap,
                },
                &args.output,
            )?;
            write_state_footprint_report_atomic(&args.output, &report)?;
            println!(
                "wrote {} ({})",
                args.output.display(),
                report.scientific_hash
            );
        }
        Command::Merge(args) => {
            let report = merge_state_footprint_report_files(&args.inputs, &args.output)?;
            write_state_footprint_report_atomic(&args.output, &report)?;
            println!(
                "wrote {} ({})",
                args.output.display(),
                report.scientific_hash
            );
        }
    }
    Ok(())
}
