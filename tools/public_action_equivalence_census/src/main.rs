use std::{error::Error, path::PathBuf};

use clap::{Parser, Subcommand};
use public_action_equivalence_census::{
    AggregateReport, CensusConfig, CensusReport, aggregate_reports_with_order_proof, read_json,
    run_adversarial_suite, run_census, run_duplicate_accounting_smoke, write_json_atomic,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "public-action-equivalence-census",
    about = "Exact S7 public-action equivalence census over open graded-oracle data"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Exhaustively verify the key contract on deterministic synthetic states.
    Adversarial {
        #[arg(long)]
        output: PathBuf,
    },
    /// Exercise production class accounting with a duplicated real legal action.
    DuplicateSmoke {
        #[arg(long)]
        dataset_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Census one disjoint modulo shard of the open train and validation data.
    Census {
        #[arg(long = "dataset-root", required = true)]
        dataset_roots: Vec<PathBuf>,
        #[arg(long)]
        shard_index: u8,
        #[arg(long, default_value_t = 3)]
        shard_count: u8,
        #[arg(long)]
        source_bundle_blake3: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        maximum_selected_groups: Option<usize>,
    },
    /// Merge complete disjoint shards in a report-order-invariant way.
    Aggregate {
        #[arg(long = "report", required = true)]
        reports: Vec<PathBuf>,
        #[arg(long)]
        adversarial_report: PathBuf,
        #[arg(long)]
        forward_output: PathBuf,
        #[arg(long)]
        reverse_output: PathBuf,
        #[arg(long)]
        order_proof_output: PathBuf,
    },
}

#[derive(Debug, Serialize)]
struct Receipt {
    artifact: String,
    report_id: String,
    classification: Option<String>,
    passed: Option<bool>,
}

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Adversarial { output } => {
            let report = run_adversarial_suite()?;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                report_id: report.report_id,
                classification: None,
                passed: Some(report.scientific.passed),
            })?;
        }
        Command::DuplicateSmoke {
            dataset_root,
            output,
        } => {
            let report = run_duplicate_accounting_smoke(&dataset_root)?;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                report_id: report.report_id,
                classification: None,
                passed: Some(report.scientific.passed),
            })?;
        }
        Command::Census {
            dataset_roots,
            shard_index,
            shard_count,
            source_bundle_blake3,
            output,
            maximum_selected_groups,
        } => {
            let report = run_census(&CensusConfig {
                dataset_roots,
                shard_index,
                shard_count,
                source_bundle_blake3,
                maximum_selected_groups,
            })?;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                report_id: report.report_id,
                classification: None,
                passed: Some(report.scientific.checks.invariant_failures == 0),
            })?;
        }
        Command::Aggregate {
            reports,
            adversarial_report,
            forward_output,
            reverse_output,
            order_proof_output,
        } => {
            let forward_inputs = reports
                .iter()
                .map(read_json::<CensusReport>)
                .collect::<Result<Vec<_>, _>>()?;
            let adversarial = read_json(&adversarial_report)?;
            let (forward, reverse, order_proof) =
                aggregate_reports_with_order_proof(&forward_inputs, &adversarial)?;
            if !order_proof.scientific.byte_identical {
                return Err("S7 aggregate classification depends on report order".into());
            }
            write_json_atomic(&forward_output, &forward)?;
            write_json_atomic(&reverse_output, &reverse)?;
            write_json_atomic(&order_proof_output, &order_proof)?;
            print_aggregate_receipt(&forward_output, &forward)?;
        }
    }
    Ok(())
}

fn print_aggregate_receipt(
    path: &std::path::Path,
    report: &AggregateReport,
) -> Result<(), Box<dyn Error>> {
    print_receipt(Receipt {
        artifact: path.display().to_string(),
        report_id: report.report_id.clone(),
        classification: Some(report.scientific.classification.clone()),
        passed: Some(report.scientific.valid),
    })
}

fn print_receipt(receipt: Receipt) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(std::io::stdout().lock(), &receipt)?;
    println!();
    Ok(())
}
