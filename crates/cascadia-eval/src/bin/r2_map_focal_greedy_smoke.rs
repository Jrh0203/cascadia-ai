//! Topology-free strength-blinded focal benchmark smoke using frozen greedy seats.

use std::{collections::BTreeSet, path::PathBuf};

use cascadia_eval::{
    focal::{OpponentIdentity, PromotionGates},
    focal_campaign::{
        FocalBenchmarkContract, FocalBenchmarkIdentities, FocalPairAssignment,
        OpponentFieldManifest, aggregate_focal_campaign, initialize_focal_campaign,
        load_work_item_summary, run_focal_work_item,
    },
    focal_gameplay::{
        FocalGameplayExecutor, LocalGreedyStrategyFactory, MacOsRuntimeResourceProbe,
    },
    r2_map_binding::R2MapImplementationBinding,
};
use cascadia_game::GameSeed;
use clap::Parser;
use serde_json::json;

const CANDIDATE_ID: &str = "greedy-smoke-candidate-v1";
const CONTROL_ID: &str = "greedy-smoke-control-v1";
const FIELD_ID: &str = "greedy-smoke-opponent-field-v1";
const INFERENCE_ID: &str = "deterministic-greedy-shared-focal-rng-v1";

#[derive(Debug, Parser)]
#[command(about = "Run the R2-MAP strength-blinded all-greedy focal smoke")]
struct Args {
    /// Empty or previously identical campaign directory in this execution.
    #[arg(long)]
    root: PathBuf,
    /// First numeric seed in a protected, benchmark-only 20-seed interval.
    #[arg(long)]
    first_seed: u64,
    #[arg(long, default_value = "r2-map-greedy-integrity-smoke-v1")]
    benchmark_id: String,
    /// Exact implementation binding derived from the registered W0 v1.1 artifact.
    #[arg(long)]
    implementation_binding: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let implementation_binding: R2MapImplementationBinding =
        serde_json::from_slice(&std::fs::read(&args.implementation_binding)?)?;
    implementation_binding.validate()?;
    let contract = FocalBenchmarkContract::new(
        0,
        cascadia_eval::focal::BenchmarkStage::StrengthBlindedSmoke,
        FocalBenchmarkIdentities::new(
            "r2-map-expert-iteration-v1",
            &args.benchmark_id,
            CANDIDATE_ID,
            CONTROL_ID,
            FIELD_ID,
            INFERENCE_ID,
        ),
        implementation_binding,
    );
    let assignments = (0..contract.pair_count)
        .map(|pair_index| {
            let focal_seat = (pair_index % 4) as u8;
            FocalPairAssignment {
                pair_index,
                game_seed: GameSeed::from_u64(args.first_seed + pair_index as u64),
                seed_domain_id: format!(
                    "{}:benchmark-only:{}",
                    args.benchmark_id,
                    args.first_seed + pair_index as u64
                ),
                focal_seat,
                opponents: (0..4)
                    .filter(|seat| *seat != focal_seat)
                    .map(|seat| OpponentIdentity {
                        seat,
                        checkpoint_id: format!("greedy-smoke-opponent-seat-{seat}-v1"),
                    })
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let field = OpponentFieldManifest::new(FIELD_ID, assignments);
    let layout = initialize_focal_campaign(&args.root, &contract, &field)?;
    let mut allowed = BTreeSet::from([CANDIDATE_ID.to_owned(), CONTROL_ID.to_owned()]);
    for assignment in &field.assignments {
        allowed.extend(
            assignment
                .opponents
                .iter()
                .map(|opponent| opponent.checkpoint_id.clone()),
        );
    }
    let factory = LocalGreedyStrategyFactory::new(allowed)?;
    let probe = MacOsRuntimeResourceProbe::new()?;
    let mut executor = FocalGameplayExecutor::with_resource_probe(factory, probe);
    let mut executed_pairs = 0;
    let mut resumed_pairs = 0;
    let mut complete_runtime = 0.0;
    let mut peak_rss_bytes = 0;
    let mut maximum_swap_delta_bytes = i64::MIN;
    let mut all_clean_shutdowns = true;
    let mut pinecone_conservation = true;
    for pair_index in 0..contract.pair_count {
        let work_item = format!("pair-{pair_index:04}");
        let outcome = run_focal_work_item(&layout, &work_item, &mut executor)?;
        let summary = load_work_item_summary(&layout, &work_item)?;
        executed_pairs += outcome.executed_pairs;
        resumed_pairs += outcome.resumed_pairs;
        complete_runtime += summary.summed_game_seconds + summary.summed_checkpoint_load_seconds;
        peak_rss_bytes = peak_rss_bytes.max(summary.peak_rss_bytes);
        maximum_swap_delta_bytes = maximum_swap_delta_bytes.max(summary.maximum_swap_delta_bytes);
        all_clean_shutdowns &= summary.all_clean_shutdowns;
        pinecone_conservation &= summary.all_pinecone_conservation_checks_passed;
    }
    let (report, artifacts) =
        aggregate_focal_campaign(&layout, complete_runtime, PromotionGates::default())?;
    let report_bytes = std::fs::read(&artifacts.json)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark_id": report.benchmark_id,
            "strength_outputs_blinded": true,
            "assigned_pairs": contract.pair_count,
            "executed_pairs": executed_pairs,
            "resumed_pairs": resumed_pairs,
            "physical_games": contract.pair_count * 2,
            "complete_runtime_seconds": complete_runtime,
            "report_json": artifacts.json,
            "report_markdown": artifacts.markdown,
            "report_blake3": blake3::hash(&report_bytes).to_hex().to_string(),
            "peak_rss_bytes": peak_rss_bytes,
            "maximum_swap_delta_bytes": maximum_swap_delta_bytes,
            "all_clean_shutdowns": all_clean_shutdowns,
            "pinecone_conservation": pinecone_conservation,
        }))?
    );
    Ok(())
}
