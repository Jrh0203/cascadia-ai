use std::{
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use r4_adaptive_multires_census::{
    BoundedAdversarialReport, BoundedArm, BoundedArmReport, BoundedFeatureView,
    aggregate_bounded_reports_with_order_proof, census_bounded_arm,
    compare_bounded_adversarial_reports, run_bounded_adversarial_suite, write_json_atomic,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "r4-bounded-far-quotient-census",
    about = "Evaluate hard-bounded radius-four far-field quotient hypotheses"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run all four arms through the frozen seven-pair proof matrix.
    Adversarial {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        require_pass: bool,
    },
    /// Verify byte-identical adversarial reports from independent hosts.
    VerifyAdversarial {
        #[arg(long = "report", required = true)]
        reports: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        require_pass: bool,
    },
    /// Census one arm over the complete accepted 60,000-position corpus.
    Census {
        #[arg(long = "dataset-root", required = true)]
        dataset_roots: Vec<PathBuf>,
        #[arg(long)]
        arm: String,
        #[arg(long)]
        require_frozen: bool,
        #[arg(long)]
        output: PathBuf,
    },
    /// Classify all four arms and prove input-order independence.
    Aggregate {
        #[arg(long = "report", required = true)]
        reports: Vec<PathBuf>,
        #[arg(long)]
        adversarial_report: PathBuf,
        #[arg(long)]
        adversarial_parity_report: PathBuf,
        #[arg(long)]
        forward_output: PathBuf,
        #[arg(long)]
        reverse_output: PathBuf,
        #[arg(long)]
        order_proof_output: PathBuf,
    },
    /// Decode and validate one canonical CSR4BQ1 bounded envelope.
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
    canonical_bytes: usize,
    canonical_blake3: String,
    view: BoundedFeatureView,
}

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Adversarial {
            output,
            require_pass,
        } => {
            let report = run_bounded_adversarial_suite()?;
            let passed = report.scientific.passed;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: report.scientific_blake3,
                passed: Some(passed),
                classification: None,
            })?;
            if require_pass && !passed {
                return Err("bounded adversarial suite did not pass".into());
            }
        }
        Command::VerifyAdversarial {
            reports,
            output,
            require_pass,
        } => {
            let reports = reports
                .iter()
                .map(read_json::<BoundedAdversarialReport>)
                .collect::<Result<Vec<_>, _>>()?;
            let parity = compare_bounded_adversarial_reports(&reports)?;
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
                return Err("bounded adversarial reports failed cross-host parity".into());
            }
        }
        Command::Census {
            dataset_roots,
            arm,
            require_frozen,
            output,
        } => {
            let arm =
                BoundedArm::from_id(&arm).ok_or_else(|| format!("unknown bounded arm {arm}"))?;
            let report = census_bounded_arm(&dataset_roots, arm, require_frozen)?;
            write_json_atomic(&output, &report)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: report.scientific_blake3,
                passed: None,
                classification: Some(arm.id().to_owned()),
            })?;
        }
        Command::Aggregate {
            reports,
            adversarial_report,
            adversarial_parity_report,
            forward_output,
            reverse_output,
            order_proof_output,
        } => {
            let reports = reports
                .iter()
                .map(read_json::<BoundedArmReport>)
                .collect::<Result<Vec<_>, _>>()?;
            let adversarial = read_json(&adversarial_report)?;
            let parity = read_json(&adversarial_parity_report)?;
            let (forward, reverse, order_proof) =
                aggregate_bounded_reports_with_order_proof(&reports, &adversarial, &parity)?;
            write_json_atomic(&forward_output, &forward)?;
            write_json_atomic(&reverse_output, &reverse)?;
            write_json_atomic(&order_proof_output, &order_proof)?;
            if !order_proof.scientific.byte_identical {
                return Err("bounded forward and reverse aggregates differ".into());
            }
            print_receipt(Receipt {
                artifact: forward_output.display().to_string(),
                scientific_blake3: forward.scientific_blake3,
                passed: Some(order_proof.scientific.byte_identical),
                classification: serde_json::to_value(forward.scientific.classification)?
                    .as_str()
                    .map(str::to_owned),
            })?;
        }
        Command::Decode { input, output } => {
            let mut bytes = Vec::new();
            BufReader::new(File::open(&input)?).read_to_end(&mut bytes)?;
            let view = BoundedFeatureView::from_canonical_bytes(&bytes)?;
            let result = DecodeOutput {
                canonical_bytes: bytes.len(),
                canonical_blake3: blake3::hash(&bytes).to_hex().to_string(),
                view,
            };
            write_json_atomic(&output, &result)?;
            print_receipt(Receipt {
                artifact: output.display().to_string(),
                scientific_blake3: result.canonical_blake3,
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
