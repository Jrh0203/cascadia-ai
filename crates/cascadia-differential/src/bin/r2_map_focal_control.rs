//! Canonical-image control plane for distributed focal campaigns.

use std::{fs::File, io::BufReader, path::PathBuf};

use cascadia_eval::{
    focal::{FOCAL_MAX_RSS_BYTES, PromotionGates},
    focal_campaign::{
        FocalBenchmarkContract, FocalCampaignLayout, OpponentFieldManifest,
        aggregate_focal_campaign, initialize_focal_campaign, load_all_work_item_summaries,
    },
};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(about = "Initialize or aggregate a distributed R2-MAP focal campaign")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate immutable inputs and create the canonical campaign layout.
    Initialize {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        opponent_field: PathBuf,
    },
    /// Validate every scheduler-managed pair and publish the final report.
    Aggregate {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        wall_seconds: f64,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        resource_gates_pass: bool,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        preregistered_guardrails_pass: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Initialize {
            root,
            contract,
            opponent_field,
        } => {
            let contract: FocalBenchmarkContract =
                serde_json::from_reader(BufReader::new(File::open(contract)?))?;
            let field: OpponentFieldManifest =
                serde_json::from_reader(BufReader::new(File::open(opponent_field)?))?;
            let layout = initialize_focal_campaign(root, &contract, &field)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "root": layout.root(),
                    "benchmark_id": contract.benchmark_id,
                    "pairs": contract.pair_count,
                    "stage": contract.stage,
                }))?
            );
        }
        Command::Aggregate {
            root,
            wall_seconds,
            resource_gates_pass,
            preregistered_guardrails_pass,
        } => {
            if !wall_seconds.is_finite() || wall_seconds <= 0.0 {
                return Err("wall-seconds must be finite and positive".into());
            }
            let layout = FocalCampaignLayout::new(root);
            let work_items = load_all_work_item_summaries(&layout)?;
            let observed_resource_gates_pass = work_items.iter().all(|summary| {
                summary.peak_rss_bytes <= FOCAL_MAX_RSS_BYTES
                    && summary.maximum_swap_delta_bytes <= 0
                    && summary.all_clean_shutdowns
                    && summary.all_pinecone_conservation_checks_passed
            });
            let (report, artifacts) = aggregate_focal_campaign(
                &layout,
                wall_seconds,
                PromotionGates {
                    resource_gates_pass: resource_gates_pass && observed_resource_gates_pass,
                    preregistered_guardrails_pass,
                },
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "benchmark_id": report.benchmark_id,
                    "report_json": artifacts.json,
                    "report_markdown": artifacts.markdown,
                    "dashboard_projection": layout.dashboard_input_path(),
                    "ledger_projection": layout.ledger_feed_path(),
                    "observed_resource_gates_pass": observed_resource_gates_pass,
                }))?
            );
        }
    }
    Ok(())
}
