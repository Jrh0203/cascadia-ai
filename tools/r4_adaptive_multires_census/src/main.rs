use std::{
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use r4_adaptive_multires_census::{
    AdaptiveMultiResolutionState, AdversarialReport, ShardReport,
    aggregate_reports_with_order_proof, census_datasets, compare_adversarial_reports,
    run_adversarial_suite, validate_adversarial_report, write_json_atomic,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "r4-adaptive-multires-census",
    about = "Exact 61/91-cell focal views with ablated far-field topology"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the frozen legal-state collision and retention matrix.
    Adversarial {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        require_pass: bool,
    },
    /// Verify that independently produced adversarial reports are identical.
    VerifyAdversarial {
        #[arg(long = "report", required = true)]
        reports: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        require_pass: bool,
    },
    /// Validate and census one unique train/validation corpus part.
    Census {
        #[arg(long = "dataset-root", required = true)]
        dataset_roots: Vec<PathBuf>,
        #[arg(long)]
        shard_index: u8,
        #[arg(long, default_value_t = 4)]
        shard_count: u8,
        #[arg(long)]
        require_frozen: bool,
        #[arg(long)]
        output: PathBuf,
    },
    /// Build forward/reverse aggregates and a byte-order proof.
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
    /// Decode and validate one canonical CSR4AM1 exact state.
    Decode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    artifact: String,
    scientific_blake3: String,
    passed: Option<bool>,
    classification: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct DecodeOutput {
    packed_bytes: usize,
    packed_blake3: String,
    state: AdaptiveMultiResolutionState,
}

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Adversarial {
            output,
            require_pass,
        } => {
            let report = run_adversarial_suite()?;
            let passed = report.scientific.passed;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: report.scientific_blake3,
                passed: Some(passed),
                classification: None,
            })?;
            if require_pass && !passed {
                return Err("production adversarial suite did not pass".into());
            }
        }
        Command::Census {
            dataset_roots,
            shard_index,
            shard_count,
            require_frozen,
            output,
        } => {
            let report = census_datasets(&dataset_roots, shard_index, shard_count, require_frozen)?;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: report.scientific_blake3,
                passed: None,
                classification: None,
            })?;
        }
        Command::VerifyAdversarial {
            reports,
            output,
            require_pass,
        } => {
            let reports = reports
                .iter()
                .map(read_json::<AdversarialReport>)
                .collect::<Result<Vec<_>, _>>()?;
            let parity = compare_adversarial_reports(&reports)?;
            let passed = parity.scientific.all_scientific_reports_identical
                && parity.scientific.all_suites_passed;
            write_json_atomic(&output, &parity)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: parity.scientific_blake3,
                passed: Some(passed),
                classification: None,
            })?;
            if require_pass && !passed {
                return Err("adversarial reports failed cross-host parity".into());
            }
        }
        Command::Aggregate {
            reports,
            adversarial_report,
            forward_output,
            reverse_output,
            order_proof_output,
        } => {
            let shards = reports
                .iter()
                .map(read_json::<ShardReport>)
                .collect::<Result<Vec<_>, _>>()?;
            let adversarial = read_json::<AdversarialReport>(&adversarial_report)?;
            validate_adversarial_report(&adversarial)?;
            let (forward, reverse, order_proof) = aggregate_reports_with_order_proof(
                &shards,
                Some(adversarial.scientific_blake3),
                adversarial.scientific.passed,
            )?;
            write_json_atomic(&forward_output, &forward)?;
            write_json_atomic(&reverse_output, &reverse)?;
            write_json_atomic(&order_proof_output, &order_proof)?;
            if !order_proof.scientific.byte_identical {
                return Err("forward and reverse aggregate documents differ".into());
            }
            print_receipt(Receipt {
                artifact: forward_output.display().to_string(),
                scientific_blake3: forward.scientific_blake3,
                passed: Some(order_proof.scientific.byte_identical),
                classification: Some(format!("{:?}", forward.scientific.classification)),
            })?;
        }
        Command::Decode { input, output } => {
            let mut bytes = Vec::new();
            BufReader::new(File::open(&input)?).read_to_end(&mut bytes)?;
            let state = AdaptiveMultiResolutionState::from_packed_bytes(&bytes)?;
            let result = DecodeOutput {
                packed_bytes: bytes.len(),
                packed_blake3: blake3::hash(&bytes).to_hex().to_string(),
                state,
            };
            write_json_atomic(&output, &result)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: result.packed_blake3,
                passed: Some(true),
                classification: None,
            })?;
        }
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn print_receipt(receipt: Receipt) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(std::io::stdout().lock(), &receipt)?;
    println!();
    Ok(())
}
