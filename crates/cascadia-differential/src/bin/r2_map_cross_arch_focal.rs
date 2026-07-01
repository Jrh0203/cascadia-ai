use std::{fs::File, io::BufReader, path::PathBuf, time::Instant};

use cascadia_differential::r2_map_cross_arch_focal::CrossArchitectureFocalExecutor;
use cascadia_eval::focal_campaign::{
    FocalBenchmarkContract, FocalCampaignLayout, load_work_item_summary, run_focal_work_item,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Run one resumable R2-MAP versus exact-NNUE focal pair")]
struct Cli {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    work_item: String,
    #[arg(long)]
    r2_bundle: PathBuf,
    #[arg(long)]
    r2_backend_parity_receipt: PathBuf,
    #[arg(long, default_value = ".venv/bin/python")]
    r2_python: PathBuf,
    #[arg(long, default_value = "python")]
    r2_python_path: PathBuf,
    #[arg(long)]
    exact_weights: PathBuf,
    #[arg(
        long,
        default_value = "/opt/cascadia/repo/docs/v2/reports/legacy-nnue-v4opp-mlx-exact-rollout-wave-v1.json"
    )]
    exact_rollout_parity_report: PathBuf,
    #[arg(long, default_value_t = 600)]
    exact_rollouts: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let layout = FocalCampaignLayout::new(&cli.root);
    let contract: FocalBenchmarkContract =
        serde_json::from_reader(BufReader::new(File::open(layout.contract_path())?))?;
    let started = Instant::now();
    let mut executor = CrossArchitectureFocalExecutor::spawn(
        contract.candidate_checkpoint_id.clone(),
        contract.control_checkpoint_id.clone(),
        contract.implementation_binding,
        &cli.r2_bundle,
        &cli.r2_backend_parity_receipt,
        &cli.r2_python,
        &cli.r2_python_path,
        &cli.exact_weights,
        &cli.exact_rollout_parity_report,
        cli.exact_rollouts,
    )?;
    let outcome = run_focal_work_item(&layout, &cli.work_item, &mut executor)?;
    let summary = load_work_item_summary(&layout, &cli.work_item)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "work_item": cli.work_item,
            "assigned_pairs": outcome.assigned_pairs,
            "executed_pairs": outcome.executed_pairs,
            "resumed_pairs": outcome.resumed_pairs,
            "physical_games": summary.physical_games,
            "peak_rss_bytes": summary.peak_rss_bytes,
            "maximum_swap_delta_bytes": summary.maximum_swap_delta_bytes,
            "all_clean_shutdowns": summary.all_clean_shutdowns,
            "wall_seconds": started.elapsed().as_secs_f64(),
        }))?
    );
    Ok(())
}
