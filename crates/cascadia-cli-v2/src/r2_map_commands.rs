use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write, stdin, stdout},
    path::{Component, Path, PathBuf},
    time::Instant,
};

use cascadia_data::{
    R2MapCollectorConfig, R2MapCollectorManifest, R2MapPolicyIdentity, R2MapProtocolIdentity,
    R2MapSeedLease, R2MapSeedPurpose, collect_r2_map_bootstrap, collect_r2_map_with_runner,
    validate_r2_map_collector_dataset,
};
use cascadia_eval::{
    focal::PromotionGates,
    focal_campaign::{
        FocalBenchmarkContract, FocalCampaignLayout, OpponentFieldManifest,
        aggregate_focal_campaign, initialize_focal_campaign, load_work_item_summary,
        run_focal_work_item,
    },
    focal_gameplay::MacOsRuntimeResourceProbe,
    longitudinal::{
        BenchmarkEvidenceClass, LongitudinalBenchmarkContract, LongitudinalBenchmarkPurpose,
        LongitudinalFieldManifest, LongitudinalGameAssignment, LongitudinalLayout,
        aggregate_longitudinal_campaign, initialize_longitudinal_campaign,
        run_longitudinal_work_item,
    },
    r2_map_binding::R2MapImplementationBinding,
    r2_map_gameplay::{R2MapBenchmarkExecutionConfig, R2MapBenchmarkGameplayExecutor},
};
use cascadia_game::GameSeed;
use cascadia_model::{
    R2_MAP_SERVING_BUNDLE_SCHEMA, R2MapModelIdentity, R2MapModelProcess, R2MapServingBundle,
    R2MapServingBundleEntry,
};
use cascadia_r2::{
    R2MapDatasetMode, R2MapDatasetStreamConfig, R2MapPackedBatchProducerConfig,
    build_r2_map_compact_index_metadata, build_r2_map_dataset_manifest,
    serve_r2_map_packed_batches, stream_r2_map_dataset,
    stream_r2_map_dataset_after_semantic_validation,
};
use cascadia_search::{R2MapExactScoreReferencePredictor, R2MapLocalGameRunner};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::cli::{Command, R2MapDatasetModeArg};

const JOHN1_CAMPAIGN_ROOT: &str = "/Users/johnherrick/cascadia-bench/r2-map-v1";
const JOHN2_WORKER_ROOT: &str = "/Users/john2/cascadia-bench/r2-map-worker-v1";
const JOHN3_WORKER_ROOT: &str = "/Users/john3/cascadia-bench/r2-map-worker-v1";
const CONTAINER_OUTPUT_ROOT: &str = "/output";
const W0_V1_1_MANIFEST_SCHEMA_ID: &str = "cascadia.r2-map.reference-panel-manifest.v1.1";
const W0_V1_1_REGISTRATION_SCHEMA_ID: &str = "cascadia.r2-map.w0-preregistration-registration.v1.1";
const W0_V1_REGISTRATION_SHA256: &str =
    "7d0336714a1e520c9c99f0d488e48577848f6c0b336ca6257ae987f2548e0d51";
const W0_V1_1_CONTRACT_REVISION: &str = "sequential-public-market-v1.1";
const OPEN_PANEL_REQUIRED_SOURCE_BINDINGS: &[&str] = &[
    "crates/cascadia-game/src/game.rs",
    "crates/cascadia-data/src/r2_map_experience.rs",
    "crates/cascadia-r2/src/r2_map_runtime.rs",
    "crates/cascadia-model/src/r2_map.rs",
    "crates/cascadia-search/src/r2_map_direct.rs",
    "crates/cascadia-search/src/r2_map_runner.rs",
    "crates/cascadia-eval/src/focal.rs",
    "crates/cascadia-eval/src/focal_campaign.rs",
    "crates/cascadia-eval/src/longitudinal.rs",
    "crates/cascadia-eval/src/r2_map_binding.rs",
    "crates/cascadia-eval/src/r2_map_gameplay.rs",
    "crates/cascadia-cli-v2/src/r2_map_commands.rs",
    "python/cascadia_mlx/r2_map_model.py",
    "python/cascadia_mlx/r2_map_serve.py",
    "tests/fixtures/r2_map/public-market-decision-protocol-v3.json",
    "tools/r2_map_reference_panels.py",
];

pub fn run(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::CollectR2MapBootstrap {
            output,
            campaign_id,
            iteration,
            host,
            first_game_index,
            games,
            shard_games,
            collector_hash,
            source_hash,
            serving_protocol_hash,
            resume,
        } => {
            validate_authoritative_output_boundary(&host, &output)?;
            let protocols = R2MapProtocolIdentity {
                collector_hash: parse_hash("collector", &collector_hash)?,
                source_hash: parse_hash("source", &source_hash)?,
                serving_protocol_hash: parse_hash("serving protocol", &serving_protocol_hash)?,
            };
            let config = R2MapCollectorConfig::bootstrap(
                output,
                R2MapSeedLease {
                    campaign_id,
                    iteration,
                    purpose: R2MapSeedPurpose::Bootstrap,
                    host_id: host,
                    first_game_index,
                    game_count: games,
                },
                shard_games,
                resume,
                protocols,
            );
            let manifest = collect_r2_map_bootstrap(&config)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::CollectR2MapIteration {
            output,
            campaign_id,
            iteration,
            host,
            first_game_index,
            games,
            shard_games,
            temperature_parts_per_million,
            collector_hash,
            source_hash,
            serving_protocol_hash,
            bundle,
            newest_manifest_identity,
            historical_manifest_identities,
            exact_score_reference,
            python,
            python_path,
            resume,
        } => {
            validate_authoritative_output_boundary(&host, &output)?;
            let protocols = R2MapProtocolIdentity {
                collector_hash: parse_hash("collector", &collector_hash)?,
                source_hash: parse_hash("source", &source_hash)?,
                serving_protocol_hash: parse_hash("serving protocol", &serving_protocol_hash)?,
            };
            let lease = R2MapSeedLease {
                campaign_id,
                iteration,
                purpose: R2MapSeedPurpose::Generation,
                host_id: host,
                first_game_index,
                game_count: games,
            };
            let (manifest, load_seconds) = if exact_score_reference {
                if bundle.is_some()
                    || newest_manifest_identity.is_some()
                    || !historical_manifest_identities.is_empty()
                {
                    return Err(
                        "exact-score reference mode forbids serving-bundle identities".into(),
                    );
                }
                let reference_hash = [0xee; 32];
                let reference_model = R2MapModelIdentity {
                    checkpoint_id: "exact-score-reference-v1".to_owned(),
                    checkpoint_manifest_blake3: "a".repeat(64),
                    model_config_blake3: "b".repeat(64),
                    model_weights_blake3: "c".repeat(64),
                    verification_id: "d".repeat(64),
                };
                let reference_bundle = R2MapServingBundle {
                    schema_version: 2,
                    schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.to_owned(),
                    protocols: protocols.clone(),
                    entries: vec![R2MapServingBundleEntry {
                        manifest_identity_blake3: hex_hash(reference_hash),
                        run_dir: PathBuf::from("/reference"),
                        checkpoint_path: PathBuf::from("/reference/checkpoint"),
                        model: reference_model,
                        pinned: true,
                    }],
                };
                let runner =
                    R2MapLocalGameRunner::new(reference_bundle, R2MapExactScoreReferencePredictor)?;
                let config = R2MapCollectorConfig::iterative(
                    output,
                    lease,
                    shard_games,
                    resume,
                    protocols,
                    R2MapPolicyIdentity::newest("exact-score-reference-v1", reference_hash),
                    vec![],
                    temperature_parts_per_million,
                );
                (collect_r2_map_with_runner(&config, &runner)?, 0.0)
            } else {
                let bundle = bundle.ok_or("iteration collection requires --bundle")?;
                let newest_manifest_identity = newest_manifest_identity
                    .ok_or("iteration collection requires --newest-manifest-identity")?;
                let (verified_bundle, runner, load_seconds) =
                    spawn_verified_runner(&bundle, &python, &python_path)?;
                let (newest, opponent_pool) = iterative_policy_field(
                    &verified_bundle,
                    &newest_manifest_identity,
                    &historical_manifest_identities,
                )?;
                let config = R2MapCollectorConfig::iterative(
                    output,
                    lease,
                    shard_games,
                    resume,
                    protocols,
                    newest,
                    opponent_pool,
                    temperature_parts_per_million,
                );
                let manifest = collect_r2_map_with_runner(&config, &runner)?;
                runner.shutdown_service()?;
                (manifest, load_seconds)
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "manifest": manifest,
                    "model_load_seconds": load_seconds,
                }))?
            );
        }
        Command::ValidateR2MapCollector { dataset } => {
            let manifest_path = dataset.join("dataset.json");
            let manifest: R2MapCollectorManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_r2_map_collector_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::InspectR2MapIndexMetadata { shards } => {
            let metadata = build_r2_map_compact_index_metadata(&shards)?;
            println!("{}", serde_json::to_string(&metadata)?);
        }
        Command::ExportR2MapDataset {
            shards,
            manifest,
            stream,
            mode,
            epoch,
            sampler_seed,
            fixed_panel_games,
            game_indices,
            validated_aggregate_receipt,
            validated_compact_index,
            validated_packing_receipt,
        } => {
            validate_primary_storage_output(&manifest)?;
            validate_primary_storage_output(&stream)?;
            let validated_binding = match (
                validated_aggregate_receipt.as_deref(),
                validated_compact_index.as_deref(),
                validated_packing_receipt.as_deref(),
            ) {
                (Some(aggregate), Some(index), Some(packing)) => Some((aggregate, index, packing)),
                (None, None, None) => None,
                _ => return Err("validated export requires all three receipt/index paths".into()),
            };
            let dataset = if validated_binding.is_some() {
                build_r2_map_compact_index_metadata(&shards)?.dataset_manifest
            } else {
                build_r2_map_dataset_manifest(&shards)?
            };
            if let Some((aggregate, index, packing)) = validated_binding {
                validate_semantic_export_binding(&shards, &dataset, aggregate, index, packing)?;
            }
            let config = R2MapDatasetStreamConfig {
                mode: match mode {
                    R2MapDatasetModeArg::Train => R2MapDatasetMode::Train,
                    R2MapDatasetModeArg::Validation => R2MapDatasetMode::Validation,
                    R2MapDatasetModeArg::FixedPanel => R2MapDatasetMode::FixedPanel,
                },
                epoch,
                sampler_seed,
                fixed_panel_games,
                game_indices,
            };
            let stream_temp = stream.with_extension("r2map.tmp");
            if let Some(parent) = stream.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut stream_writer = BufWriter::new(File::create(&stream_temp)?);
            let receipt = if validated_binding.is_some() {
                stream_r2_map_dataset_after_semantic_validation(
                    &shards,
                    &dataset,
                    &config,
                    &mut stream_writer,
                )?
            } else {
                stream_r2_map_dataset(&shards, &dataset, &config, &mut stream_writer)?
            };
            stream_writer.flush()?;
            stream_writer.get_ref().sync_all()?;
            fs::rename(&stream_temp, &stream)?;

            if let Some(parent) = manifest.parent() {
                fs::create_dir_all(parent)?;
            }
            let manifest_temp = manifest.with_extension("json.tmp");
            let mut manifest_writer = BufWriter::new(File::create(&manifest_temp)?);
            serde_json::to_writer_pretty(&mut manifest_writer, &dataset)?;
            manifest_writer.write_all(b"\n")?;
            manifest_writer.flush()?;
            manifest_writer.get_ref().sync_all()?;
            fs::rename(&manifest_temp, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Command::ServeR2MapPackedBatches {
            shard,
            mode,
            epoch,
            sampler_seed,
            group_batch_size,
            maximum_candidates_per_batch,
            bootstrap_value_only,
            ordered_game_indices,
            start_game_offset,
            start_turn_offset,
            start_batch_index,
            validated_aggregate_receipt,
            validated_compact_index,
            validated_packing_receipt,
        } => {
            let shards = vec![shard.clone()];
            let dataset = build_r2_map_compact_index_metadata(&shards)?.dataset_manifest;
            validate_semantic_export_binding(
                &shards,
                &dataset,
                &validated_aggregate_receipt,
                &validated_compact_index,
                &validated_packing_receipt,
            )?;
            let config = R2MapPackedBatchProducerConfig {
                mode: match mode {
                    R2MapDatasetModeArg::Train => R2MapDatasetMode::Train,
                    R2MapDatasetModeArg::Validation => R2MapDatasetMode::Validation,
                    R2MapDatasetModeArg::FixedPanel => R2MapDatasetMode::FixedPanel,
                },
                epoch,
                sampler_seed,
                group_batch_size,
                maximum_candidates_per_batch,
                bootstrap_value_only,
                ordered_game_indices,
                start_game_offset,
                start_turn_offset,
                start_batch_index,
            };
            let input = stdin();
            let output = stdout();
            serve_r2_map_packed_batches(
                &shard,
                &config,
                input.lock(),
                BufWriter::new(output.lock()),
            )?;
        }
        Command::PrepareR2MapServingBundle {
            host,
            output,
            collector_hash,
            source_hash,
            serving_protocol_hash,
            checkpoints,
        } => {
            validate_authoritative_output_boundary(&host, &output)?;
            let bundle = prepare_serving_bundle(
                &checkpoints,
                R2MapProtocolIdentity {
                    collector_hash: parse_hash("collector", &collector_hash)?,
                    source_hash: parse_hash("source", &source_hash)?,
                    serving_protocol_hash: parse_hash("serving protocol", &serving_protocol_hash)?,
                },
            )?;
            write_immutable_json(&output, &bundle)?;
            let verified = R2MapServingBundle::read_verified(&output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "bundle": output,
                    "entries": verified.entries.len(),
                    "verified": true,
                }))?
            );
        }
        Command::InitR2MapLongitudinalOpenPanel {
            root,
            reference_panel_manifest,
            reference_panel_registration,
            campaign_id,
            benchmark_id,
            iteration,
            focal_checkpoint_id,
            field_manifest_id,
            historical_checkpoints,
            inference_settings_id,
        } => {
            validate_primary_storage_output(&root)?;
            if historical_checkpoints.is_empty() {
                return Err("open longitudinal panel requires a historical checkpoint".into());
            }
            let (panel, implementation_binding) = read_open_performance_panel(
                &reference_panel_manifest,
                &reference_panel_registration,
            )?;
            let seed_domain_id = panel
                .get("seed_domain")
                .and_then(Value::as_str)
                .ok_or("open performance panel omitted its seed domain")?
                .to_owned();
            let seeds = panel
                .get("seeds")
                .and_then(Value::as_array)
                .ok_or("open performance panel omitted seeds")?;
            let contract = LongitudinalBenchmarkContract::new(
                campaign_id,
                benchmark_id,
                iteration,
                focal_checkpoint_id,
                field_manifest_id.clone(),
                historical_checkpoints.clone(),
                inference_settings_id,
                seed_domain_id,
                LongitudinalBenchmarkPurpose::OpenPerformanceReferenceOnly,
                BenchmarkEvidenceClass::RealOpenCheckpointPerformanceOnly,
                implementation_binding,
            );
            let assignments = seeds
                .iter()
                .enumerate()
                .map(|(game_index, raw_seed)| {
                    let seed = raw_seed
                        .as_u64()
                        .ok_or("open performance seed is not u64")?;
                    let focal_seat = (game_index % 4) as u8;
                    Ok(LongitudinalGameAssignment {
                        game_index,
                        game_seed: GameSeed::from_u64(seed),
                        focal_seat,
                        opponents: (0..4)
                            .filter(|seat| *seat != focal_seat)
                            .map(|seat| cascadia_eval::focal::OpponentIdentity {
                                seat,
                                checkpoint_id: historical_checkpoints[(game_index
                                    + usize::from(seat))
                                    % historical_checkpoints.len()]
                                .clone(),
                            })
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let field = LongitudinalFieldManifest::new(field_manifest_id, assignments);
            let layout = initialize_longitudinal_campaign(root, &contract, &field)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "root": layout.root(),
                    "games": contract.game_count,
                    "evidence_class": contract.evidence_class,
                    "strength_claim_authorized": false,
                }))?
            );
        }
        Command::InitR2MapLongitudinalCampaign {
            root,
            contract,
            historical_field,
        } => {
            validate_primary_storage_output(&root)?;
            let contract: LongitudinalBenchmarkContract = read_json(&contract)?;
            let field: LongitudinalFieldManifest = read_json(&historical_field)?;
            let layout = initialize_longitudinal_campaign(root, &contract, &field)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "root": layout.root(),
                    "benchmark_id": contract.benchmark_id,
                    "purpose": contract.purpose,
                    "games": contract.game_count,
                    "strength_claim_authorized": false,
                }))?
            );
        }
        Command::InitR2MapFocalCampaign {
            root,
            contract,
            opponent_field,
        } => {
            validate_primary_storage_output(&root)?;
            let contract: FocalBenchmarkContract = read_json(&contract)?;
            let field: OpponentFieldManifest = read_json(&opponent_field)?;
            let layout = initialize_focal_campaign(root, &contract, &field)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "root": layout.root(),
                    "benchmark_id": contract.benchmark_id,
                    "stage": contract.stage,
                    "pairs": contract.pair_count,
                }))?
            );
        }
        Command::RunR2MapLongitudinalWorkItem {
            root,
            work_item,
            bundle,
            python,
            python_path,
        } => {
            let layout = LongitudinalLayout::new(&root);
            let contract: LongitudinalBenchmarkContract = read_json(&layout.contract_path())?;
            let (verified_bundle, runner, load_seconds) =
                spawn_verified_runner(&bundle, &python, &python_path)?;
            let probe = MacOsRuntimeResourceProbe::new()?;
            let config = R2MapBenchmarkExecutionConfig::longitudinal(
                contract.campaign_id.clone(),
                contract.iteration,
                contract.implementation_binding.clone(),
            );
            let mut executor = R2MapBenchmarkGameplayExecutor::new(
                &verified_bundle,
                runner,
                probe,
                config,
                load_seconds,
            )?;
            let outcome = run_longitudinal_work_item(&layout, &work_item, &mut executor)?;
            let restarts = executor.service_restarts();
            executor.into_runner().shutdown_service()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "work_item": work_item,
                    "assigned_games": outcome.assigned_games,
                    "executed_games": outcome.executed_games,
                    "resumed_games": outcome.resumed_games,
                    "service_restarts": restarts,
                    "clean_service_shutdown": true,
                }))?
            );
        }
        Command::AggregateR2MapLongitudinal { root, wall_seconds } => {
            validate_primary_storage_output(&root)?;
            let layout = LongitudinalLayout::new(root);
            let report = aggregate_longitudinal_campaign(&layout, wall_seconds)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::RunR2MapFocalWorkItem {
            root,
            work_item,
            bundle,
            python,
            python_path,
        } => {
            let layout = FocalCampaignLayout::new(&root);
            let contract: FocalBenchmarkContract = read_json(&layout.contract_path())?;
            let (verified_bundle, runner, load_seconds) =
                spawn_verified_runner(&bundle, &python, &python_path)?;
            let probe = MacOsRuntimeResourceProbe::new()?;
            let config = R2MapBenchmarkExecutionConfig::candidate_gate(
                contract.campaign_id.clone(),
                contract.iteration,
                contract.implementation_binding.clone(),
            );
            let mut executor = R2MapBenchmarkGameplayExecutor::new(
                &verified_bundle,
                runner,
                probe,
                config,
                load_seconds,
            )?;
            let outcome = run_focal_work_item(&layout, &work_item, &mut executor)?;
            let restarts = executor.service_restarts();
            executor.into_runner().shutdown_service()?;
            let summary = load_work_item_summary(&layout, &work_item)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "work_item": work_item,
                    "assigned_pairs": outcome.assigned_pairs,
                    "executed_pairs": outcome.executed_pairs,
                    "resumed_pairs": outcome.resumed_pairs,
                    "physical_games": summary.physical_games,
                    "peak_rss_bytes": summary.peak_rss_bytes,
                    "maximum_swap_delta_bytes": summary.maximum_swap_delta_bytes,
                    "service_restarts": restarts,
                    "clean_service_shutdown": true,
                }))?
            );
        }
        Command::AggregateR2MapFocal { root, wall_seconds } => {
            validate_primary_storage_output(&root)?;
            let layout = FocalCampaignLayout::new(root);
            let (report, artifacts) =
                aggregate_focal_campaign(&layout, wall_seconds, PromotionGates::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "report": report,
                    "json": artifacts.json,
                    "markdown": artifacts.markdown,
                    "dashboard_input": layout.dashboard_input_path(),
                    "ledger_feed": layout.ledger_feed_path(),
                }))?
            );
        }
        _ => return Err("r2-map command dispatcher received an unrelated command".into()),
    }
    Ok(())
}

fn spawn_verified_runner(
    bundle_path: &Path,
    python: &Path,
    python_path: &Path,
) -> Result<
    (
        R2MapServingBundle,
        R2MapLocalGameRunner<R2MapModelProcess>,
        f64,
    ),
    Box<dyn std::error::Error>,
> {
    let started = Instant::now();
    let bundle = R2MapServingBundle::read_verified(bundle_path)?;
    let args = vec![
        OsString::from("PYTHONDONTWRITEBYTECODE=1"),
        OsString::from(format!("PYTHONPATH={}", python_path.display())),
        python.as_os_str().to_owned(),
        OsString::from("-m"),
        OsString::from("cascadia_mlx.r2_map_serve"),
        OsString::from("--bundle"),
        bundle_path.as_os_str().to_owned(),
    ];
    let process = R2MapModelProcess::spawn("/usr/bin/env", args)?;
    let runner = R2MapLocalGameRunner::from_verified_bundle(bundle_path, process)?;
    Ok((bundle, runner, started.elapsed().as_secs_f64()))
}

fn iterative_policy_field(
    bundle: &R2MapServingBundle,
    newest_manifest_identity: &str,
    historical_manifest_identities: &[String],
) -> Result<(R2MapPolicyIdentity, Vec<R2MapPolicyIdentity>), Box<dyn std::error::Error>> {
    let newest_hash = parse_hash("newest manifest identity", newest_manifest_identity)?;
    let newest_model = bundle.model_for_manifest_identity(newest_hash)?;
    let newest = R2MapPolicyIdentity::newest(newest_model.checkpoint_id, newest_hash);
    let mut historical = Vec::with_capacity(historical_manifest_identities.len());
    for identity in historical_manifest_identities {
        let hash = parse_hash("historical manifest identity", identity)?;
        if hash == newest_hash {
            return Err("newest manifest identity cannot appear in the historical field".into());
        }
        let model = bundle.model_for_manifest_identity(hash)?;
        historical.push(R2MapPolicyIdentity::historical(model.checkpoint_id, hash));
    }
    Ok((newest, historical))
}

fn prepare_serving_bundle(
    checkpoint_paths: &[PathBuf],
    protocols: R2MapProtocolIdentity,
) -> Result<R2MapServingBundle, Box<dyn std::error::Error>> {
    if checkpoint_paths.is_empty() {
        return Err("serving bundle requires at least one checkpoint".into());
    }
    let mut entries = Vec::with_capacity(checkpoint_paths.len());
    for raw_path in checkpoint_paths {
        let checkpoint_path = raw_path.canonicalize()?;
        let checkpoints_directory = checkpoint_path
            .parent()
            .ok_or("checkpoint path has no checkpoints directory")?;
        let run_dir = checkpoints_directory
            .parent()
            .ok_or("checkpoint path has no run directory")?
            .to_owned();
        if checkpoints_directory
            .file_name()
            .and_then(|name| name.to_str())
            != Some("checkpoints")
        {
            return Err("checkpoint path must be RUN/checkpoints/CHECKPOINT".into());
        }
        let manifest_path = checkpoint_path.join("checkpoint.json");
        let manifest_bytes = fs::read(&manifest_path)?;
        let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
        let checkpoint_id = manifest
            .get("checkpoint_id")
            .and_then(Value::as_str)
            .ok_or("checkpoint manifest omitted checkpoint_id")?;
        if checkpoint_path.file_name().and_then(|name| name.to_str()) != Some(checkpoint_id) {
            return Err("checkpoint directory and manifest identity differ".into());
        }
        let compact = manifest
            .get("manifest_identity_blake3")
            .and_then(Value::as_str)
            .ok_or("checkpoint manifest omitted manifest_identity_blake3")?;
        let model_config_blake3 = manifest
            .pointer("/identity/model_config_blake3")
            .and_then(Value::as_str)
            .ok_or("checkpoint manifest omitted model config identity")?;
        let model_weights_blake3 = manifest
            .pointer("/files/model.safetensors/blake3")
            .and_then(Value::as_str)
            .ok_or("checkpoint manifest omitted model weights identity")?;
        let verification_path = run_dir
            .join("verifications")
            .join(format!("{checkpoint_id}.json"));
        let verification: Value = read_json(&verification_path)?;
        let verification_id = verification
            .get("verification_id")
            .and_then(Value::as_str)
            .ok_or("checkpoint verification omitted verification_id")?;
        entries.push(R2MapServingBundleEntry {
            manifest_identity_blake3: compact.to_owned(),
            run_dir,
            checkpoint_path,
            model: R2MapModelIdentity {
                checkpoint_id: checkpoint_id.to_owned(),
                checkpoint_manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
                model_config_blake3: model_config_blake3.to_owned(),
                model_weights_blake3: model_weights_blake3.to_owned(),
                verification_id: verification_id.to_owned(),
            },
            pinned: true,
        });
    }
    let bundle = R2MapServingBundle {
        schema_version: 2,
        schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.to_owned(),
        protocols,
        entries,
    };
    bundle.validate()?;
    Ok(bundle)
}

fn read_open_performance_panel(
    path: &Path,
    registration_path: &Path,
) -> Result<(Value, R2MapImplementationBinding), Box<dyn std::error::Error>> {
    verify_reference_panel_registration(path, registration_path)?;
    let manifest: Value = read_json(path)?;
    if manifest.get("schema_id").and_then(Value::as_str) != Some(W0_V1_1_MANIFEST_SCHEMA_ID)
        || manifest.get("campaign_id").and_then(Value::as_str) != Some("r2-map-expert-iteration-v1")
        || manifest.get("contract_revision").and_then(Value::as_str)
            != Some(W0_V1_1_CONTRACT_REVISION)
        || manifest.get("status").and_then(Value::as_str)
            != Some("frozen-open-reference-panels-v1.1")
    {
        return Err(
            "reference manifest has the wrong v1.1 schema, campaign, contract, or status".into(),
        );
    }
    if manifest
        .pointer("/predecessor/open_panel_outcomes_opened")
        .and_then(Value::as_bool)
        != Some(false)
        || manifest
            .pointer("/predecessor/open_seed_domain_reused_by_successor")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("W0 v1.1 did not preserve the unopened v1 open-seed contract".into());
    }
    verify_embedded_canonical_sha256(&manifest, "manifest_sha256")?;
    let matching = manifest
        .get("panels")
        .and_then(Value::as_array)
        .ok_or("reference manifest omitted panels")?
        .iter()
        .filter(|panel| {
            panel.get("panel_id").and_then(Value::as_str) == Some("open-performance-100")
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(
            "reference manifest must contain exactly one open-performance-100 panel".into(),
        );
    }
    let panel_entry = matching[0];
    verify_embedded_canonical_sha256(panel_entry, "panel_sha256")?;
    verify_open_panel_source_bindings(panel_entry)?;
    let panel = panel_entry
        .get("definition")
        .cloned()
        .ok_or("open performance panel omitted its definition")?;
    if panel.get("game_count").and_then(Value::as_u64) != Some(100)
        || panel.get("protected_domain").and_then(Value::as_bool) != Some(false)
        || panel
            .get("strength_claim_authorized")
            .and_then(Value::as_bool)
            != Some(false)
        || panel
            .get("predecessor_outcomes_opened")
            .and_then(Value::as_bool)
            != Some(false)
        || panel.get("seed_domain_changed").and_then(Value::as_bool) != Some(false)
    {
        return Err("open performance panel lost its fixed non-protected identity".into());
    }
    let identity = manifest
        .get("implementation_identity")
        .and_then(Value::as_object)
        .ok_or("W0 v1.1 manifest omitted implementation identity")?;
    let identity_string = |field: &str| -> Result<String, Box<dyn std::error::Error>> {
        Ok(identity
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("W0 v1.1 implementation identity omitted {field}"))?
            .to_owned())
    };
    let binding = R2MapImplementationBinding::new(
        sha256_hex(&fs::read(registration_path)?),
        manifest
            .get("manifest_sha256")
            .and_then(Value::as_str)
            .ok_or("W0 v1.1 manifest omitted canonical identity")?
            .to_owned(),
        identity_string("maximum_width_panel_sha256")?,
        identity_string("replay_pinecone_panel_sha256")?,
        identity_string("source_bundle_sha256")?,
        identity_string("serving_protocol_schema_sha256")?,
        identity_string("market_action_schema_blake3")?,
        identity_string("request_schema_blake3")?,
        identity_string("response_schema_blake3")?,
        identity_string("protocol_fixture_canonical_blake3")?,
        identity_string("protocol_fixture_file_blake3")?,
        identity_string("model_schema_sha256")?,
        identity_string("open_reference_seed_domain_id")?,
    )?;
    if panel.get("seed_domain").and_then(Value::as_str)
        != Some(binding.open_reference_seed_domain_id.as_str())
    {
        return Err("open panel seed domain differs from its W0 v1.1 binding".into());
    }
    Ok((panel, binding))
}

fn verify_reference_panel_registration(
    manifest_path: &Path,
    registration_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let control_root = Path::new(JOHN1_CAMPAIGN_ROOT)
        .join("control")
        .canonicalize()?;
    let registration_path = registration_path.canonicalize()?;
    if !registration_path.starts_with(&control_root) || !registration_path.is_file() {
        return Err(
            "W0 registration must be a regular file below the campaign control root".into(),
        );
    }
    let registration: Value = read_json(&registration_path)?;
    if registration.get("campaign_id").and_then(Value::as_str) != Some("r2-map-expert-iteration-v1")
        || registration.get("schema_id").and_then(Value::as_str)
            != Some(W0_V1_1_REGISTRATION_SCHEMA_ID)
        || registration
            .get("contract_revision")
            .and_then(Value::as_str)
            != Some(W0_V1_1_CONTRACT_REVISION)
        || registration
            .get("protected_seed_values_opened")
            .and_then(Value::as_bool)
            != Some(false)
        || registration.get("john4_used").and_then(Value::as_bool) != Some(false)
    {
        return Err(
            "W0 registration lost its v1.1 schema, campaign, protected-domain, or host boundary"
                .into(),
        );
    }
    let verification = registration
        .get("independent_verification")
        .and_then(Value::as_object)
        .ok_or("W0 v1.1 registration omitted independent verification")?;
    for required in [
        "python_exact_regeneration_required",
        "rust_source_rehash_required",
        "rust_initializer_must_reject_v1",
        "all_live_source_bindings_required",
    ] {
        if verification.get(required).and_then(Value::as_bool) != Some(true) {
            return Err(format!("W0 v1.1 registration weakened verifier: {required}").into());
        }
    }
    let predecessor_path = registration
        .pointer("/append_only_predecessor/path")
        .and_then(Value::as_str)
        .ok_or("W0 v1.1 registration omitted its predecessor path")?;
    let predecessor_sha256 = registration
        .pointer("/append_only_predecessor/formatted_file_sha256")
        .and_then(Value::as_str)
        .ok_or("W0 v1.1 registration omitted its predecessor identity")?;
    if registration
        .pointer("/append_only_predecessor/execution_status")
        .and_then(Value::as_str)
        != Some("immutable-stale-negative")
        || predecessor_sha256 != W0_V1_REGISTRATION_SHA256
    {
        return Err("W0 v1.1 predecessor chain changed".into());
    }
    let expected_predecessor = control_root.join("w0-preregistration/registration.json");
    let predecessor = Path::new(predecessor_path).canonicalize()?;
    if predecessor != expected_predecessor.canonicalize()?
        || sha256_hex(&fs::read(predecessor)?) != W0_V1_REGISTRATION_SHA256
    {
        return Err("W0 v1 predecessor registration is not byte-immutable".into());
    }
    let manifest_bytes = fs::read(manifest_path)?;
    let formatted_expected = registration
        .pointer("/artifacts/reference_panels/formatted_file_sha256")
        .and_then(Value::as_str)
        .ok_or("W0 registration omitted the formatted manifest SHA-256")?;
    if sha256_hex(&manifest_bytes) != formatted_expected {
        return Err("W0 registration does not bind the exact formatted manifest bytes".into());
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if registration.get("implementation_identity") != manifest.get("implementation_identity") {
        return Err("W0 v1.1 registration implementation identity differs from manifest".into());
    }
    let canonical_expected = registration
        .pointer("/artifacts/reference_panels/canonical_manifest_sha256")
        .and_then(Value::as_str)
        .ok_or("W0 registration omitted the canonical manifest SHA-256")?;
    if manifest.get("manifest_sha256").and_then(Value::as_str) != Some(canonical_expected) {
        return Err("W0 registration and manifest canonical identities differ".into());
    }
    let registered_path = registration
        .pointer("/artifacts/reference_panels/ssd_path")
        .and_then(Value::as_str)
        .ok_or("W0 registration omitted the canonical John2 manifest path")?;
    if Path::new(registered_path).canonicalize()? != manifest_path.canonicalize()? {
        return Err("W0 initializer was not given the manifest registered on John2".into());
    }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let repository_manifest = registration
        .pointer("/artifacts/reference_panels/repository_path")
        .and_then(Value::as_str)
        .ok_or("W0 v1.1 registration omitted the repository manifest path")?;
    let expected_repository_manifest =
        repository.join("docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json");
    if Path::new(repository_manifest).canonicalize()?
        != expected_repository_manifest.canonicalize()?
        || fs::read(expected_repository_manifest)? != manifest_bytes
    {
        return Err(
            "W0 repository and canonical John2 v1.1 manifests are not identical registered bytes"
                .into(),
        );
    }
    Ok(())
}

fn verify_embedded_canonical_sha256(
    value: &Value,
    field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !json_strings_are_ascii(value) {
        return Err("reference manifest canonical JSON must be ASCII".into());
    }
    let mut canonical = value.clone();
    let object = canonical
        .as_object_mut()
        .ok_or("reference manifest hash boundary must be an object")?;
    let expected = object
        .remove(field)
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or("reference manifest omitted its canonical SHA-256")?;
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("reference manifest contains a malformed canonical SHA-256".into());
    }
    let observed = sha256_hex(&serde_json::to_vec(&canonical)?);
    if observed != expected {
        return Err(format!("reference manifest canonical hash drifted: {field}").into());
    }
    Ok(())
}

fn validate_semantic_export_binding(
    shards: &[PathBuf],
    subset: &cascadia_r2::R2MapDatasetManifest,
    aggregate_path: &Path,
    compact_index_path: &Path,
    packing_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if shards.is_empty() {
        return Err("receipt-bound export requires at least one shard".into());
    }
    let aggregate_bytes = fs::read(aggregate_path)?;
    let compact_index_bytes = fs::read(compact_index_path)?;
    let packing_bytes = fs::read(packing_path)?;
    let aggregate: Value = serde_json::from_slice(&aggregate_bytes)?;
    let compact_index: Value = serde_json::from_slice(&compact_index_bytes)?;
    let packing: Value = serde_json::from_slice(&packing_bytes)?;
    let aggregate_sha256 = sha256_hex(&aggregate_bytes);
    let compact_index_sha256 = sha256_hex(&compact_index_bytes);
    verify_embedded_canonical_sha256(&packing, "receipt_sha256")?;
    if json_text(&packing, "schema_id") != Some("cascadia.r2-map.bootstrap-packing-selection.v1")
        || json_text(&packing, "result") != Some("pass")
        || json_text(&packing, "aggregate_receipt_sha256") != Some(aggregate_sha256.as_str())
        || json_text(&packing, "index_sha256") != Some(compact_index_sha256.as_str())
        || json_u64(&packing, "games") != Some(100_000)
        || json_u64(&packing, "examples") != Some(8_000_000)
        || json_u64(&packing, "replay_shards") != Some(420)
    {
        return Err("packing receipt does not bind the aggregate receipt and compact index".into());
    }
    if json_text(&aggregate, "schema_id") != Some("cascadia.r2-map.bootstrap-aggregate.v1")
        || json_text(&aggregate, "result") != Some("pass")
        || json_u64(&aggregate, "games") != Some(100_000)
        || json_u64(&aggregate, "primary_example_count") != Some(8_000_000)
        || json_u64(&aggregate, "worker_datasets") != Some(30)
        || json_u64(&aggregate, "replay_shards") != Some(420)
        || json_u64(&aggregate, "completion_audits") != Some(30)
        || aggregate
            .get("require_worker_validation")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "aggregate receipt does not prove the complete semantic-validation gate".into(),
        );
    }
    if json_text(&compact_index, "protocol_id") != Some("r2-map-compact-index-v4") {
        return Err("receipt-bound export requires the v4 bounded-window compact index".into());
    }
    let full: cascadia_r2::R2MapDatasetManifest = serde_json::from_value(
        compact_index
            .get("dataset_manifest")
            .cloned()
            .ok_or("compact index omitted its dataset manifest")?,
    )?;
    if json_text(&packing, "dataset_blake3") != Some(full.dataset_blake3.as_str())
        || full.game_count != 100_000
        || full.example_count != 8_000_000
        || full.sources.len() != 420
    {
        return Err("packing receipt dataset identity differs from the compact index".into());
    }
    for equal in [
        subset.schema_version == full.schema_version,
        subset.protocol_id == full.protocol_id,
        subset.feature_schema == full.feature_schema,
        subset.target_schema == full.target_schema,
        subset.split_schema == full.split_schema,
        subset.d6_schema == full.d6_schema,
        subset.imitation_subset_schema == full.imitation_subset_schema,
        subset.imitation_subset_parts_per_million == full.imitation_subset_parts_per_million,
        subset.round == full.round,
    ] {
        if !equal {
            return Err("receipt-bound shard contract differs from the compact index".into());
        }
    }
    let full_sources = full
        .sources
        .iter()
        .map(|source| (source.file_name.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let aggregate_shards = aggregate
        .get("shards")
        .and_then(Value::as_array)
        .ok_or("aggregate receipt omitted replay shards")?;
    if aggregate_shards.len() != full_sources.len() {
        return Err("aggregate receipt and compact index shard counts differ".into());
    }
    let mut aggregate_sources = BTreeMap::new();
    for shard in aggregate_shards {
        let name = json_text(shard, "file_name").ok_or("aggregate shard omitted its file name")?;
        if aggregate_sources.insert(name, shard).is_some() {
            return Err("aggregate receipt repeats a replay shard".into());
        }
    }
    for (name, source) in &full_sources {
        let aggregate_source = aggregate_sources
            .get(name)
            .ok_or("compact-index source is absent from the aggregate receipt")?;
        if json_u64(aggregate_source, "bytes") != Some(source.bytes)
            || json_text(aggregate_source, "blake3") != Some(source.blake3.as_str())
            || json_u64(aggregate_source, "first_game_index") != Some(source.first_game_index)
            || json_u64(aggregate_source, "game_count") != Some(u64::try_from(source.game_count)?)
        {
            return Err("aggregate receipt replay shard differs from compact index source".into());
        }
    }
    if subset.sources.len() != shards.len() {
        return Err("receipt-bound shard paths and subset manifest cardinality differ".into());
    }
    let subset_sources = subset
        .sources
        .iter()
        .map(|source| (source.file_name.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    for shard in shards {
        let name = shard
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("receipt-bound shard file name is not UTF-8")?;
        let source = subset_sources
            .get(name)
            .ok_or("receipt-bound shard is absent from its subset manifest")?;
        let observed_blake3 = blake3::hash(&fs::read(shard)?).to_hex().to_string();
        if full_sources.get(name).copied() != Some(*source)
            || fs::metadata(shard)?.len() != source.bytes
            || observed_blake3 != source.blake3
        {
            return Err(
                "receipt-bound shard bytes or identity differ from the compact index".into(),
            );
        }
    }
    Ok(())
}

fn json_text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn verify_open_panel_source_bindings(panel: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let bindings = panel
        .get("source_bindings")
        .and_then(Value::as_array)
        .ok_or("open performance panel omitted source bindings")?;
    let mut observed_paths = std::collections::BTreeSet::new();
    for binding in bindings {
        let relative = binding
            .get("path")
            .and_then(Value::as_str)
            .ok_or("open performance source binding omitted path")?;
        let expected = binding
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or("open performance source binding omitted SHA-256")?;
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
            || !observed_paths.insert(relative.to_owned())
        {
            return Err("open performance source binding path is unsafe or duplicated".into());
        }
        let source = repository.join(relative_path).canonicalize()?;
        if !source.starts_with(&repository) || !source.is_file() {
            return Err("open performance source binding escaped the repository".into());
        }
        let observed = sha256_hex(&fs::read(source)?);
        if observed != expected {
            return Err(format!("open performance source binding drifted: {relative}").into());
        }
    }
    for required in OPEN_PANEL_REQUIRED_SOURCE_BINDINGS {
        if !observed_paths.contains(*required) {
            return Err(
                format!("open performance panel omitted required binding: {required}").into(),
            );
        }
    }
    Ok(())
}

fn json_strings_are_ascii(value: &Value) -> bool {
    match value {
        Value::String(value) => value.is_ascii(),
        Value::Array(values) => values.iter().all(json_strings_are_ascii),
        Value::Object(values) => values
            .iter()
            .all(|(key, value)| key.is_ascii() && json_strings_are_ascii(value)),
        Value::Null | Value::Bool(_) | Value::Number(_) => true,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn write_immutable_json<T: Serialize + DeserializeOwned + PartialEq>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        let existing: T = read_json(path)?;
        if &existing == value {
            return Ok(());
        }
        return Err(format!("immutable artifact differs: {}", path.display()).into());
    }
    let parent = path.parent().ok_or("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let mut writer = BufWriter::new(file);
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    })();
    drop(writer);
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn validate_primary_storage_output(output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    validate_authoritative_output_boundary("john1", output)
}

fn parse_hash(label: &str, value: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} hash must contain exactly 64 hexadecimal characters").into());
    }
    let mut decoded = [0u8; 32];
    for (index, output) in decoded.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)?;
    }
    if decoded == [0; 32] {
        return Err(format!("{label} hash cannot be all zeroes").into());
    }
    Ok(decoded)
}

fn hex_hash(value: [u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_authoritative_output_boundary(
    host: &str,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if !output.is_absolute()
        || output
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("R2-MAP output must be an absolute path without '..'".into());
    }
    match host {
        "john1" | "john2" | "john3" => {}
        _ => return Err(format!("R2-MAP host must be john1, john2, or john3: {host}").into()),
    }
    let boundary = if output.exists() {
        output.canonicalize()?
    } else {
        let parent = output
            .parent()
            .ok_or("R2-MAP output must have a parent directory")?;
        parent.canonicalize()?.join(
            output
                .file_name()
                .ok_or("R2-MAP output must have a final path component")?,
        )
    };
    let roots = host_output_roots(host)
        .iter()
        .filter_map(|root| PathBuf::from(root).canonicalize().ok())
        .collect::<Vec<_>>();
    if roots.is_empty() || !roots.iter().any(|root| boundary.starts_with(root)) {
        return Err(format!(
            "R2-MAP output attributed to {host} must remain below one of its configured roots: {}",
            host_output_roots(host).join(", ")
        )
        .into());
    }
    Ok(())
}

fn host_output_roots(host: &str) -> [&'static str; 2] {
    let host_root = match host {
        "john1" => JOHN1_CAMPAIGN_ROOT,
        "john2" => JOHN2_WORKER_ROOT,
        "john3" => JOHN3_WORKER_ROOT,
        _ => unreachable!("host is validated before roots are requested"),
    };
    [host_root, CONTAINER_OUTPUT_ROOT]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_parser_is_exact_and_rejects_missing_identity() {
        assert_eq!(parse_hash("source", &"ab".repeat(32)).unwrap(), [0xab; 32]);
        assert!(parse_hash("source", "ab").is_err());
        assert!(parse_hash("source", &"00".repeat(32)).is_err());
        assert!(parse_hash("source", &"gg".repeat(32)).is_err());
    }

    #[test]
    fn authoritative_output_boundary_rejects_local_disks_parent_escape_and_john4() {
        assert!(
            validate_authoritative_output_boundary(
                "john1",
                Path::new("/Users/johnherrick/cascadia/r2-map-output")
            )
            .is_err()
        );
        assert!(
            validate_authoritative_output_boundary(
                "john1",
                Path::new("/Volumes/John_1/cascadia-cluster/r2-map-v1/datasets/../outside")
            )
            .is_err()
        );
        assert!(
            validate_authoritative_output_boundary(
                "john4",
                Path::new("/Users/john4/cascadia-bench/r2-map-output")
            )
            .is_err()
        );
    }

    #[test]
    fn output_roots_use_john1_primary_and_host_local_worker_staging() {
        assert_eq!(
            host_output_roots("john1"),
            [JOHN1_CAMPAIGN_ROOT, CONTAINER_OUTPUT_ROOT]
        );
        assert_eq!(
            host_output_roots("john2"),
            [JOHN2_WORKER_ROOT, CONTAINER_OUTPUT_ROOT]
        );
        assert_eq!(
            host_output_roots("john3"),
            [JOHN3_WORKER_ROOT, CONTAINER_OUTPUT_ROOT]
        );
    }

    #[test]
    fn iterative_field_binds_one_newest_and_distinct_historical_models() {
        let entry = |digit: char, checkpoint: &str| R2MapServingBundleEntry {
            manifest_identity_blake3: digit.to_string().repeat(64),
            run_dir: PathBuf::from(format!("/runs/{checkpoint}")),
            checkpoint_path: PathBuf::from(format!("/runs/{checkpoint}/checkpoints/{checkpoint}")),
            model: R2MapModelIdentity {
                checkpoint_id: checkpoint.to_owned(),
                checkpoint_manifest_blake3: "a".repeat(64),
                model_config_blake3: "b".repeat(64),
                model_weights_blake3: "c".repeat(64),
                verification_id: "d".repeat(64),
            },
            pinned: true,
        };
        let bundle = R2MapServingBundle {
            schema_version: 2,
            schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.to_owned(),
            protocols: R2MapProtocolIdentity {
                collector_hash: [1; 32],
                source_hash: [2; 32],
                serving_protocol_hash: [3; 32],
            },
            entries: vec![entry('1', "newest"), entry('2', "history")],
        };
        bundle.validate().unwrap();

        let (newest, historical) =
            iterative_policy_field(&bundle, &"1".repeat(64), &["2".repeat(64)]).unwrap();
        assert_eq!(newest.policy_id, "newest");
        assert_eq!(newest.checkpoint_hash, Some([0x11; 32]));
        assert_eq!(historical.len(), 1);
        assert_eq!(historical[0].policy_id, "history");
        assert_eq!(historical[0].checkpoint_hash, Some([0x22; 32]));
        assert!(iterative_policy_field(&bundle, &"1".repeat(64), &["1".repeat(64)]).is_err());
    }

    #[test]
    fn superseded_w0_manifest_cannot_launch_after_source_drift() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.json");
        let manifest: Value = read_json(&path).unwrap();
        verify_embedded_canonical_sha256(&manifest, "manifest_sha256").unwrap();
        let panel = manifest["panels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|panel| panel["panel_id"] == "open-performance-100")
            .unwrap();
        let error = verify_open_panel_source_bindings(panel)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("source binding drifted") || error.contains("omitted required binding")
        );
    }

    #[test]
    fn embedded_canonical_sha256_rejects_tamper() {
        let mut value = serde_json::json!({"alpha": 1, "beta": "two"});
        let digest = sha256_hex(&serde_json::to_vec(&value).unwrap());
        value["sha256"] = Value::String(digest);
        verify_embedded_canonical_sha256(&value, "sha256").unwrap();
        value["alpha"] = Value::from(2);
        assert!(verify_embedded_canonical_sha256(&value, "sha256").is_err());
    }
}
