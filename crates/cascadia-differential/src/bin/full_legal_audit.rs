use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_data::{
    DatasetSplit, GRADED_SOURCE_BEST_CHAMPION_FRONTIER, GRADED_SOURCE_CHAMPION_FRONTIER,
    GRADED_SOURCE_CHAMPION_SELECTED, GRADED_SOURCE_COMPLETE_LEGAL, GRADED_SOURCE_R600,
    GRADED_SOURCE_R1200, GRADED_SOURCE_R4800, GRADED_SOURCE_SENTINEL,
    GRADED_SOURCE_SUBSTANTIAL_TOP, GRADED_SOURCE_TOP_SCREEN, GradedOracleAuditInput,
    GradedOracleCandidate, GradedOracleDatasetConfig, GradedOracleDatasetManifest,
    GradedOracleDatasetWriter, GradedOracleEstimate, GradedOracleGroup, PositionRecord,
    merge_graded_oracle_datasets, validate_graded_oracle_dataset,
};
use cascadia_differential::{
    full_legal_audit::{
        ActionSources, AuditProvenance, DecisionPhase, FullLegalActionRecord, FullLegalAuditConfig,
        FullLegalAuditShard, FullLegalGameAudit, RolloutEstimateRecord,
        SerializableBatchDiagnostics, SignedScoreBreakdown, collect_full_legal_audit_shard,
        merge_full_legal_audit_shards, qualify_exact_root_evaluation,
        qualify_paid_wipe_hidden_invariance, read_full_legal_audit_merged,
        read_full_legal_audit_shard, select_full_legal_oracle_action, unix_seconds,
        write_full_legal_audit_merged, write_full_legal_audit_shard,
    },
    legacy_teacher::{BridgeDiagnostics, ExactMlxLegacyTeacher, spawn_exact_mlx_process},
};
use cascadia_eval::{ComparisonReport, summarize_paired_match_results};
use cascadia_game::{GameConfig, GameSeed, GameState, score_board, score_game};
use cascadia_provenance::{checksum_file, source_provenance};
use cascadia_sim::{MatchResult, play_match_with_selector};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

const ORACLE_EXPERIMENT_ID: &str = "full-legal-public-oracle-v1";
const CHAMPION_STRATEGY_ID: &str = "exact-mlx-k32-r600-champion-v1";
const ORACLE_STRATEGY_ID: &str = "full-legal-public-oracle-v1";

#[derive(Debug, Parser)]
#[command(about = "Full-legal decision-regret audit for the exact MLX Cascadia champion")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DatasetSplitArg {
    Train,
    Validation,
    Test,
}

impl From<DatasetSplitArg> for DatasetSplit {
    fn from(value: DatasetSplitArg) -> Self {
        match value {
            DatasetSplitArg::Train => Self::Train,
            DatasetSplitArg::Validation => Self::Validation,
            DatasetSplitArg::Test => Self::Test,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    Collect {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        first_seed: u64,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        worker: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 600)]
        champion_rollouts: usize,
        #[arg(long, default_value_t = 64)]
        screen_limit: usize,
        #[arg(long, default_value_t = 16)]
        sentinel_count: usize,
        #[arg(long, default_value_t = 1_200)]
        substantial_rollouts: usize,
        #[arg(long, default_value_t = 8)]
        high_confidence_limit: usize,
        #[arg(long, default_value_t = 4_800)]
        high_confidence_rollouts: usize,
        #[arg(long, value_delimiter = ',')]
        audited_completed_turns: Vec<u16>,
        #[arg(long, value_delimiter = ',', default_value = "12,39,66")]
        realized_hidden_completed_turns: Vec<u16>,
        #[arg(long)]
        skip_realized_hidden_diagnostics: bool,
        #[arg(long, default_value_t = 8)]
        paid_wipe_determinizations: usize,
        #[arg(long, default_value_t = 2)]
        paid_wipe_followup_determinizations: usize,
        #[arg(long, default_value_t = 3)]
        paid_wipe_followup_width: usize,
        #[arg(long)]
        skip_paid_wipe_diagnostics: bool,
    },
    Validate {
        #[arg(long)]
        input: PathBuf,
    },
    Merge {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    ValidateMerged {
        #[arg(long)]
        input: PathBuf,
    },
    ExportGradedOracle {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum)]
        split: DatasetSplitArg,
        #[arg(long)]
        resume: bool,
    },
    ValidateGradedOracle {
        #[arg(long)]
        dataset: PathBuf,
    },
    MergeGradedOracle {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    Qualify {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        completed_turns: u16,
        #[arg(long, default_value_t = 600)]
        champion_rollouts: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    QualifyPaidWipe {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        completed_turns: Option<u16>,
        #[arg(long, default_value_t = 1)]
        minimum_nature_tokens: u8,
        #[arg(long, default_value_t = 600)]
        champion_rollouts: usize,
        #[arg(long, default_value_t = 8)]
        paid_wipe_determinizations: usize,
        #[arg(long, default_value_t = 2)]
        paid_wipe_followup_determinizations: usize,
        #[arg(long, default_value_t = 3)]
        paid_wipe_followup_width: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    OracleCompare {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long)]
        worker: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 600)]
        champion_rollouts: usize,
        #[arg(long, default_value_t = 64)]
        screen_limit: usize,
        #[arg(long, default_value_t = 16)]
        sentinel_count: usize,
        #[arg(long, default_value_t = 1_200)]
        substantial_rollouts: usize,
        #[arg(long, default_value_t = 8)]
        high_confidence_limit: usize,
        #[arg(long, default_value_t = 4_800)]
        high_confidence_rollouts: usize,
    },
    OracleMerge {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        expected_first_seed: u64,
        #[arg(long)]
        expected_games: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        markdown_output: PathBuf,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OracleGameRecord {
    raw_seed: u64,
    baseline: MatchResult,
    treatment: MatchResult,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct OracleDecisionSummary {
    decisions: usize,
    changed_actions: usize,
    top_screen_recalled_winners: usize,
    actions_screened: usize,
    champion_frontier_actions: usize,
    substantial_actions: usize,
    high_confidence_actions: usize,
    champion_regret_sum: f64,
    champion_seconds: f64,
    enumeration_seconds: f64,
    screening_seconds: f64,
    substantial_seconds: f64,
    high_confidence_seconds: f64,
    total_seconds: f64,
    phase_decisions: [usize; 3],
    phase_changed_actions: [usize; 3],
    phase_champion_regret_sum: [f64; 3],
}

impl OracleDecisionSummary {
    fn record(
        &mut self,
        decision: &cascadia_differential::full_legal_audit::FullLegalOracleDecision,
        completed_turns: u16,
    ) {
        let phase = match completed_turns / 4 + 1 {
            1..=7 => 0,
            8..=14 => 1,
            _ => 2,
        };
        self.decisions += 1;
        self.changed_actions += usize::from(decision.action != decision.champion_action);
        self.top_screen_recalled_winners += usize::from(decision.top_screen_recalled_winner);
        self.actions_screened += decision.action_count;
        self.champion_frontier_actions += decision.champion_frontier_count;
        self.substantial_actions += decision.substantial_count;
        self.high_confidence_actions += decision.high_confidence_count;
        self.champion_regret_sum += decision.champion_regret.points;
        self.champion_seconds += decision.champion_seconds;
        self.enumeration_seconds += decision.enumeration_seconds;
        self.screening_seconds += decision.screening_seconds;
        self.substantial_seconds += decision.substantial_seconds;
        self.high_confidence_seconds += decision.high_confidence_seconds;
        self.total_seconds += decision.total_seconds;
        self.phase_decisions[phase] += 1;
        self.phase_changed_actions[phase] +=
            usize::from(decision.action != decision.champion_action);
        self.phase_champion_regret_sum[phase] += decision.champion_regret.points;
    }

    fn absorb(&mut self, other: &Self) {
        self.decisions += other.decisions;
        self.changed_actions += other.changed_actions;
        self.top_screen_recalled_winners += other.top_screen_recalled_winners;
        self.actions_screened += other.actions_screened;
        self.champion_frontier_actions += other.champion_frontier_actions;
        self.substantial_actions += other.substantial_actions;
        self.high_confidence_actions += other.high_confidence_actions;
        self.champion_regret_sum += other.champion_regret_sum;
        self.champion_seconds += other.champion_seconds;
        self.enumeration_seconds += other.enumeration_seconds;
        self.screening_seconds += other.screening_seconds;
        self.substantial_seconds += other.substantial_seconds;
        self.high_confidence_seconds += other.high_confidence_seconds;
        self.total_seconds += other.total_seconds;
        for phase in 0..3 {
            self.phase_decisions[phase] += other.phase_decisions[phase];
            self.phase_changed_actions[phase] += other.phase_changed_actions[phase];
            self.phase_champion_regret_sum[phase] += other.phase_champion_regret_sum[phase];
        }
    }

    fn mean_champion_regret(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.champion_regret_sum / self.decisions as f64
        }
    }

    fn action_change_rate(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.changed_actions as f64 / self.decisions as f64
        }
    }

    fn top_screen_recall(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.top_screen_recalled_winners as f64 / self.decisions as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OracleCompareReport {
    schema_version: u32,
    experiment_id: String,
    status: String,
    worker: String,
    first_seed: u64,
    games: usize,
    config: FullLegalAuditConfig,
    comparison: ComparisonReport,
    decision_summary: OracleDecisionSummary,
    mean_champion_regret: f64,
    action_change_rate: f64,
    top_screen_recall: f64,
    baseline_diagnostics: BridgeDiagnostics,
    treatment_diagnostics: BridgeDiagnostics,
    baseline_batch_diagnostics: SerializableBatchDiagnostics,
    treatment_batch_diagnostics: SerializableBatchDiagnostics,
    baseline_clean_shutdown: bool,
    treatment_clean_shutdown: bool,
    elapsed_seconds: f64,
    model_json_blake3: String,
    model_safetensors_blake3: String,
    executable_blake3: String,
    source: cascadia_provenance::SourceProvenance,
    game_records: Vec<OracleGameRecord>,
    completed_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleHostSummary {
    worker: String,
    first_seed: u64,
    games: usize,
    baseline_mean: f64,
    treatment_mean: f64,
    mean_paired_delta: f64,
    confidence_95: [f64; 2],
    elapsed_seconds: f64,
}

#[derive(Debug, Serialize)]
struct OracleMergedReport {
    schema_version: u32,
    experiment_id: String,
    status: String,
    stage: String,
    expected_first_seed: u64,
    expected_games: usize,
    config: FullLegalAuditConfig,
    comparison: ComparisonReport,
    decision_summary: OracleDecisionSummary,
    mean_champion_regret: f64,
    action_change_rate: f64,
    top_screen_recall: f64,
    phase_mean_champion_regret: [f64; 3],
    phase_action_change_rate: [f64; 3],
    hosts: Vec<OracleHostSummary>,
    model_json_blake3: String,
    model_safetensors_blake3: String,
    executable_blake3: String,
    v2_source_blake3: String,
    pilot_gates: OraclePilotGates,
    input_reports: Vec<String>,
    completed_unix_seconds: u64,
}

#[derive(Debug, Serialize)]
struct OraclePilotGates {
    all_integrity_gates_passed: bool,
    treatment_mean_at_least_100: bool,
    paired_delta_at_least_3: bool,
    every_host_nonnegative: bool,
    complete_phase_coverage: bool,
    passed: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Collect {
            server_program,
            model_dir,
            first_seed,
            games,
            worker,
            output,
            champion_rollouts,
            screen_limit,
            sentinel_count,
            substantial_rollouts,
            high_confidence_limit,
            high_confidence_rollouts,
            audited_completed_turns,
            realized_hidden_completed_turns,
            skip_realized_hidden_diagnostics,
            paid_wipe_determinizations,
            paid_wipe_followup_determinizations,
            paid_wipe_followup_width,
            skip_paid_wipe_diagnostics,
        } => {
            let config = FullLegalAuditConfig {
                champion_rollouts,
                screen_limit,
                sentinel_count,
                substantial_rollouts,
                high_confidence_limit,
                high_confidence_rollouts,
                audited_completed_turns: (!audited_completed_turns.is_empty())
                    .then_some(audited_completed_turns),
                realized_hidden_completed_turns: if skip_realized_hidden_diagnostics {
                    Vec::new()
                } else {
                    realized_hidden_completed_turns
                },
                paid_wipe_determinizations: if skip_paid_wipe_diagnostics {
                    0
                } else {
                    paid_wipe_determinizations
                },
                paid_wipe_followup_determinizations,
                paid_wipe_followup_width,
                ..FullLegalAuditConfig::default()
            };
            config.validate()?;
            let executable = std::env::current_exe()?;
            let provenance = AuditProvenance {
                worker,
                source: source_provenance()?,
                executable_blake3: checksum_file(&executable)?,
                model_json_blake3: checksum_file(&model_dir.join("model.json"))?,
                model_safetensors_blake3: checksum_file(&model_dir.join("model.safetensors"))?,
                started_unix_seconds: unix_seconds(),
            };
            let process = spawn_exact_mlx_process(&server_program, &model_dir)?;
            let mut teacher = ExactMlxLegacyTeacher::new(process, champion_rollouts)?;
            let collection =
                collect_full_legal_audit_shard(&mut teacher, config, provenance, first_seed, games);
            let shutdown = teacher.shutdown();
            let shard = collection?;
            shutdown?;
            write_full_legal_audit_shard(&output, &shard)?;
            println!("{}", serde_json::to_string_pretty(&shard.summary)?);
        }
        Command::Validate { input } => {
            let shard = read_full_legal_audit_shard(&input)?;
            println!("{}", serde_json::to_string_pretty(&shard.summary)?);
        }
        Command::Merge { input, output } => {
            let shards = input
                .iter()
                .map(|path| read_full_legal_audit_shard(path))
                .collect::<Result<Vec<_>, _>>()?;
            let merged = merge_full_legal_audit_shards(shards)?;
            write_full_legal_audit_merged(&output, &merged)?;
            println!("{}", serde_json::to_string_pretty(&merged.summary)?);
        }
        Command::ValidateMerged { input } => {
            let merged = read_full_legal_audit_merged(&input)?;
            println!("{}", serde_json::to_string_pretty(&merged.summary)?);
        }
        Command::ExportGradedOracle {
            input,
            output,
            split,
            resume,
        } => export_graded_oracle_dataset(&input, &output, split.into(), resume)?,
        Command::ValidateGradedOracle { dataset } => {
            let manifest: GradedOracleDatasetManifest = serde_json::from_reader(
                std::io::BufReader::new(std::fs::File::open(dataset.join("dataset.json"))?),
            )?;
            validate_graded_oracle_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::MergeGradedOracle { input, output } => {
            let manifest = merge_graded_oracle_datasets(&input, &output)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Qualify {
            server_program,
            model_dir,
            seed,
            completed_turns,
            champion_rollouts,
            output,
        } => {
            let process = spawn_exact_mlx_process(&server_program, &model_dir)?;
            let mut teacher = ExactMlxLegacyTeacher::new(process, champion_rollouts)?;
            let qualification = qualify_exact_root_evaluation(
                &mut teacher,
                seed,
                completed_turns,
                champion_rollouts,
            );
            let shutdown = teacher.shutdown();
            let qualification = qualification?;
            shutdown?;
            let json = serde_json::to_string_pretty(&qualification)?;
            if let Some(output) = output {
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(output, format!("{json}\n"))?;
            }
            println!("{json}");
        }
        Command::QualifyPaidWipe {
            server_program,
            model_dir,
            seed,
            completed_turns,
            minimum_nature_tokens,
            champion_rollouts,
            paid_wipe_determinizations,
            paid_wipe_followup_determinizations,
            paid_wipe_followup_width,
            output,
        } => {
            let config = FullLegalAuditConfig {
                champion_rollouts,
                paid_wipe_determinizations,
                paid_wipe_followup_determinizations,
                paid_wipe_followup_width,
                ..FullLegalAuditConfig::default()
            };
            config.validate()?;
            let process = spawn_exact_mlx_process(&server_program, &model_dir)?;
            let mut teacher = ExactMlxLegacyTeacher::new(process, champion_rollouts)?;
            let qualification = qualify_paid_wipe_hidden_invariance(
                &mut teacher,
                &config,
                seed,
                completed_turns,
                minimum_nature_tokens,
            );
            let shutdown = teacher.shutdown();
            let qualification = qualification?;
            shutdown?;
            let json = serde_json::to_string_pretty(&qualification)?;
            if let Some(output) = output {
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(output, format!("{json}\n"))?;
            }
            println!("{json}");
        }
        Command::OracleCompare {
            server_program,
            model_dir,
            games,
            first_seed,
            worker,
            output,
            champion_rollouts,
            screen_limit,
            sentinel_count,
            substantial_rollouts,
            high_confidence_limit,
            high_confidence_rollouts,
        } => run_oracle_compare(
            &server_program,
            &model_dir,
            games,
            first_seed,
            worker,
            &output,
            FullLegalAuditConfig {
                champion_rollouts,
                screen_limit,
                sentinel_count,
                substantial_rollouts,
                high_confidence_limit,
                high_confidence_rollouts,
                realized_hidden_completed_turns: Vec::new(),
                paid_wipe_determinizations: 0,
                ..FullLegalAuditConfig::default()
            },
        )?,
        Command::OracleMerge {
            input,
            expected_first_seed,
            expected_games,
            output,
            markdown_output,
        } => run_oracle_merge(
            &input,
            expected_first_seed,
            expected_games,
            &output,
            &markdown_output,
        )?,
    }
    Ok(())
}

fn export_graded_oracle_dataset(
    inputs: &[PathBuf],
    output: &Path,
    split: DatasetSplit,
    resume: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err("graded-oracle export requires at least one audit input".into());
    }
    let mut ordered = inputs
        .iter()
        .map(|path| Ok((audit_seed_from_path(path)?, path.canonicalize()?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    ordered.sort_by_key(|(seed, _)| *seed);
    if ordered.windows(2).any(|pair| pair[1].0 <= pair[0].0) {
        return Err("graded-oracle audit inputs must cover unique ordered seeds".into());
    }

    let mut first_shard = Some(read_full_legal_audit_shard(&ordered[0].1)?);
    let frozen = first_shard.as_ref().expect("first shard is present");
    validate_export_source(frozen, ordered[0].0, None)?;
    let config_blake3 = blake3::hash(&serde_json::to_vec(&frozen.config)?)
        .to_hex()
        .to_string();
    let source = frozen.provenance.source.v2_source_blake3.clone();
    let executable = frozen.provenance.executable_blake3.clone();
    let model_json = frozen.provenance.model_json_blake3.clone();
    let model_weights = frozen.provenance.model_safetensors_blake3.clone();
    let protocol = frozen.config.protocol_id.clone();
    let audit_inputs = ordered
        .iter()
        .map(|(raw_seed, path)| {
            Ok(GradedOracleAuditInput {
                path: portable_audit_path(path)?,
                blake3: checksum_file(path)?,
                raw_seed: *raw_seed,
                audit_protocol_id: protocol.clone(),
                audit_config_blake3: config_blake3.clone(),
                source_blake3: source.clone(),
                executable_blake3: executable.clone(),
                model_json_blake3: model_json.clone(),
                model_safetensors_blake3: model_weights.clone(),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let mut writer = GradedOracleDatasetWriter::open(&GradedOracleDatasetConfig {
        output: output.to_path_buf(),
        split,
        audit_inputs,
        resume,
    })?;
    let completed = writer.manifest().completed_games;
    for (index, (raw_seed, path)) in ordered.iter().enumerate().skip(completed) {
        let shard = if index == 0 {
            first_shard
                .take()
                .expect("first shard has not been consumed")
        } else {
            read_full_legal_audit_shard(path)?
        };
        validate_export_source(&shard, *raw_seed, Some(frozen_identity(&writer)))?;
        let game = shard
            .games
            .into_iter()
            .next()
            .ok_or("graded-oracle audit shard contains no game")?;
        let groups = export_graded_oracle_game(game)?;
        writer.append_game(*raw_seed, &groups)?;
        eprintln!(
            "graded-oracle export: {}/{} games, {} groups, {} candidates",
            writer.manifest().completed_games,
            writer.manifest().requested_games,
            writer.manifest().total_groups,
            writer.manifest().total_records,
        );
    }
    validate_graded_oracle_dataset(writer.root(), writer.manifest())?;
    println!("{}", serde_json::to_string_pretty(writer.manifest())?);
    Ok(())
}

fn portable_audit_path(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let canonical = path.canonicalize()?;
    let current = std::env::current_dir()?.canonicalize()?;
    Ok(canonical
        .strip_prefix(current)
        .unwrap_or(&canonical)
        .display()
        .to_string())
}

fn audit_seed_from_path(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or("audit input has no UTF-8 file stem")?;
    let raw = stem
        .strip_prefix("seed-")
        .ok_or("audit input must be named seed-N.json")?;
    Ok(raw.parse()?)
}

fn frozen_identity(writer: &GradedOracleDatasetWriter) -> (&str, &str, &str, &str, &str, &str) {
    let input = &writer.manifest().audit_inputs[0];
    (
        &input.audit_protocol_id,
        &input.audit_config_blake3,
        &input.source_blake3,
        &input.executable_blake3,
        &input.model_json_blake3,
        &input.model_safetensors_blake3,
    )
}

fn validate_export_source(
    shard: &FullLegalAuditShard,
    raw_seed: u64,
    frozen: Option<(&str, &str, &str, &str, &str, &str)>,
) -> Result<(), Box<dyn std::error::Error>> {
    if shard.first_seed != raw_seed
        || shard.games_requested != 1
        || shard.games.len() != 1
        || shard.games[0].raw_seed != raw_seed
        || shard.games[0].decisions.len() != 80
    {
        return Err("graded-oracle audit input is not one complete 80-decision game".into());
    }
    if let Some((protocol, config, source, executable, model_json, model_weights)) = frozen {
        let actual_config = blake3::hash(&serde_json::to_vec(&shard.config)?)
            .to_hex()
            .to_string();
        if shard.config.protocol_id != protocol
            || actual_config != config
            || shard.provenance.source.v2_source_blake3 != source
            || shard.provenance.executable_blake3 != executable
            || shard.provenance.model_json_blake3 != model_json
            || shard.provenance.model_safetensors_blake3 != model_weights
        {
            return Err("graded-oracle audit input identity drifted".into());
        }
    }
    Ok(())
}

fn export_graded_oracle_game(
    audit: FullLegalGameAudit,
) -> Result<Vec<GradedOracleGroup>, Box<dyn std::error::Error>> {
    let raw_seed = audit.raw_seed;
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(raw_seed))?;
    let mut groups = Vec::with_capacity(audit.decisions.len());
    for decision in audit.decisions {
        if game.completed_turns() != decision.completed_turns
            || game.current_player() != decision.current_player
            || game.public_state().canonical_hash().to_hex().to_string()
                != decision.public_state_blake3
            || score_board(
                &game.boards()[game.current_player()],
                game.config().scoring_cards,
            ) != decision.current_score
            || game.public_supply() != decision.public_supply
        {
            return Err(format!(
                "audit replay drifted at seed {raw_seed} turn {}",
                decision.completed_turns
            )
            .into());
        }
        let staged = game.preview_market_prelude(&decision.prelude)?;
        if staged.public_state().canonical_hash().to_hex().to_string()
            != decision.staged_public_state_blake3
        {
            return Err(format!(
                "audit staged state drifted at seed {raw_seed} turn {}",
                decision.completed_turns
            )
            .into());
        }
        let selected_hash = parse_action_hash(&decision.best_complete_screen_hash)?;
        let champion_hash = parse_action_hash(&decision.champion_action_hash)?;
        let position = PositionRecord::observe(&game, raw_seed);
        let group_id = graded_group_id(raw_seed, decision.completed_turns, decision.current_player);
        let mut source_selected_index = None;
        let mut champion_index = None;
        let mut champion_action = None;
        let candidate_count = decision.actions.len();
        let mut candidates = Vec::with_capacity(candidate_count);
        for (index, action) in decision.actions.into_iter().enumerate() {
            if action.canonical_index != index {
                return Err("audit canonical action indices are not contiguous".into());
            }
            let action_hash = parse_action_hash(&action.canonical_hash)?;
            if graded_canonical_action_hash(&action.action)? != action_hash {
                return Err("audit canonical action hash does not match its payload".into());
            }
            if action_hash == selected_hash {
                source_selected_index = Some(index);
            }
            if action_hash == champion_hash {
                champion_index = Some(u16::try_from(index)?);
                champion_action = Some(action.action.clone());
            }
            let source_flags = graded_source_flags(&action);
            let r600 = graded_estimate(action.champion_frontier_r600)?;
            let r1200 = graded_estimate(action.substantial_r1200)?;
            let r4800 = graded_estimate(action.high_confidence_r4800)?;
            let candidate = GradedOracleCandidate::observe(
                &game,
                &action.action,
                action_hash,
                u16::try_from(action.canonical_index)?,
                u16::try_from(action.screen_rank)?,
                source_flags,
                action.model_immediate_score,
                action.model_remaining_value,
                action.screen_value,
                action.uniform_market_survival_proxy as f32,
                action.visible_wildlife_count,
                action.public_bag_wildlife_count,
                r600,
                r1200,
                r4800,
            )?;
            if candidate.action.immediate_score != action.exact_resulting_score.base_total
                || candidate.action.immediate_deltas
                    != signed_score_deltas(action.exact_score_delta)
            {
                return Err("audit exact action deltas changed during export".into());
            }
            candidates.push(candidate);
        }
        if candidates.len() != decision.action_count {
            return Err("audit action count changed during export".into());
        }
        let source_selected_index = source_selected_index.ok_or("audit R4800 winner is absent")?;
        let best_r4800 = candidates
            .iter()
            .filter(|candidate| candidate.r4800.samples > 0)
            .map(|candidate| candidate.r4800.mean)
            .fold(f32::NEG_INFINITY, f32::max);
        if (candidates[source_selected_index].r4800.mean - best_r4800).abs() > f32::EPSILON {
            return Err("audit R4800 winner is not a maximum".into());
        }
        let selected_index = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.r4800.samples > 0
                    && (candidate.r4800.mean - best_r4800).abs() <= f32::EPSILON
            })
            .min_by_key(|(_, candidate)| candidate.action_hash)
            .map(|(index, _)| u16::try_from(index))
            .transpose()?
            .ok_or("audit has no R4800 maximum")?;
        groups.push(GradedOracleGroup {
            group_id,
            raw_seed,
            completed_turns: decision.completed_turns,
            current_player: u8::try_from(decision.current_player)?,
            personal_turn: u8::try_from(decision.personal_turn)?,
            phase: match decision.phase {
                DecisionPhase::Early => 0,
                DecisionPhase::Middle => 1,
                DecisionPhase::Late => 2,
            },
            selected_index,
            champion_index: champion_index.ok_or("audit champion action is absent")?,
            public_state_hash: parse_action_hash(&decision.public_state_blake3)?,
            public_supply: decision.public_supply,
            position,
            candidates,
        });
        game.apply(&champion_action.ok_or("audit champion action is absent")?)?;
    }
    if !game.is_game_over()
        || score_game(&game) != audit.final_scores
        || game.canonical_hash().to_hex().to_string() != audit.final_state_blake3
    {
        return Err(format!("audit terminal replay drifted for seed {raw_seed}").into());
    }
    Ok(groups)
}

fn signed_score_deltas(delta: SignedScoreBreakdown) -> [i16; 11] {
    let mut values = [0; 11];
    values[..5].copy_from_slice(&delta.habitat);
    values[5..10].copy_from_slice(&delta.wildlife);
    values[10] = delta.nature_tokens;
    values
}

fn parse_action_hash(value: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    Ok(*value.parse::<blake3::Hash>()?.as_bytes())
}

fn graded_canonical_action_hash(
    action: &cascadia_game::TurnAction,
) -> Result<[u8; 32], serde_json::Error> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-full-legal-action-v1");
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

fn graded_estimate(
    value: Option<RolloutEstimateRecord>,
) -> Result<GradedOracleEstimate, Box<dyn std::error::Error>> {
    match value {
        Some(estimate) => Ok(GradedOracleEstimate {
            mean: estimate.mean as f32,
            stddev: estimate.stddev as f32,
            samples: u16::try_from(estimate.samples)?,
        }),
        None => Ok(GradedOracleEstimate::default()),
    }
}

fn graded_source_flags(action: &FullLegalActionRecord) -> u16 {
    let ActionSources {
        top_complete_screen,
        champion_frontier,
        champion_selected,
        rank_stratified_sentinel,
        substantial_top,
        best_champion_frontier,
    } = action.sources;
    GRADED_SOURCE_COMPLETE_LEGAL
        | if top_complete_screen {
            GRADED_SOURCE_TOP_SCREEN
        } else {
            0
        }
        | if champion_frontier {
            GRADED_SOURCE_CHAMPION_FRONTIER
        } else {
            0
        }
        | if champion_selected {
            GRADED_SOURCE_CHAMPION_SELECTED
        } else {
            0
        }
        | if rank_stratified_sentinel {
            GRADED_SOURCE_SENTINEL
        } else {
            0
        }
        | if substantial_top {
            GRADED_SOURCE_SUBSTANTIAL_TOP
        } else {
            0
        }
        | if best_champion_frontier {
            GRADED_SOURCE_BEST_CHAMPION_FRONTIER
        } else {
            0
        }
        | if action.champion_frontier_r600.is_some() {
            GRADED_SOURCE_R600
        } else {
            0
        }
        | if action.substantial_r1200.is_some() {
            GRADED_SOURCE_R1200
        } else {
            0
        }
        | if action.high_confidence_r4800.is_some() {
            GRADED_SOURCE_R4800
        } else {
            0
        }
}

fn graded_group_id(raw_seed: u64, completed_turns: u16, current_player: usize) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-graded-oracle-group");
    hasher.update(&raw_seed.to_le_bytes());
    hasher.update(&completed_turns.to_le_bytes());
    hasher.update(&(current_player as u64).to_le_bytes());
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 output contains eight bytes"),
    )
}

fn run_oracle_compare(
    server_program: &str,
    model_dir: &Path,
    games: usize,
    first_seed: u64,
    worker: String,
    output: &Path,
    config: FullLegalAuditConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 {
        return Err("oracle comparison requires at least one game".into());
    }
    config.validate()?;
    let started = Instant::now();
    let baseline_process = spawn_exact_mlx_process(server_program, model_dir)?;
    let treatment_process = spawn_exact_mlx_process(server_program, model_dir)?;
    let mut baseline = ExactMlxLegacyTeacher::new(baseline_process, config.champion_rollouts)?;
    let mut treatment = ExactMlxLegacyTeacher::new(treatment_process, config.champion_rollouts)?;
    let game_config = GameConfig::research_aaaaa(4)?;
    let mut decision_summary = OracleDecisionSummary::default();
    let mut game_records = Vec::with_capacity(games);

    for offset in 0..games {
        let raw_seed = first_seed + offset as u64;
        let seed = GameSeed::from_u64(raw_seed);
        let baseline_match =
            play_match_with_selector(game_config, seed, CHAMPION_STRATEGY_ID, |_player, game| {
                baseline
                    .select_action(game)
                    .map_err(|error| cascadia_sim::SimulationError::Strategy(error.to_string()))
            })?;
        let treatment_match =
            play_match_with_selector(game_config, seed, ORACLE_STRATEGY_ID, |_player, game| {
                let decision = select_full_legal_oracle_action(&mut treatment, &config, game)
                    .map_err(|error| cascadia_sim::SimulationError::Strategy(error.to_string()))?;
                decision_summary.record(&decision, game.completed_turns());
                Ok(decision.action)
            })?;
        game_records.push(OracleGameRecord {
            raw_seed,
            baseline: baseline_match,
            treatment: treatment_match,
        });
        eprintln!(
            "full-legal oracle {worker}: {}/{} pairs complete, {:.1}s elapsed",
            offset + 1,
            games,
            started.elapsed().as_secs_f64()
        );
    }

    let paired = game_records
        .iter()
        .map(|record| {
            (
                record.raw_seed,
                record.baseline.clone(),
                record.treatment.clone(),
            )
        })
        .collect::<Vec<_>>();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let comparison = summarize_paired_match_results(
        CHAMPION_STRATEGY_ID,
        ORACLE_STRATEGY_ID,
        first_seed,
        &paired,
        elapsed_seconds,
    );
    let baseline_diagnostics = baseline.diagnostics.clone();
    let treatment_diagnostics = treatment.diagnostics.clone();
    let baseline_batch_diagnostics = SerializableBatchDiagnostics::from(baseline.batch_diagnostics);
    let treatment_batch_diagnostics =
        SerializableBatchDiagnostics::from(treatment.batch_diagnostics);
    let baseline_clean_shutdown = baseline.shutdown().is_ok();
    let treatment_clean_shutdown = treatment.shutdown().is_ok();
    let complete = decision_summary.decisions == games * 80
        && baseline_diagnostics.fallbacks == 0
        && treatment_diagnostics.fallbacks == 0
        && baseline_batch_diagnostics.policy_fallbacks == 0
        && treatment_batch_diagnostics.policy_fallbacks == 0
        && baseline_batch_diagnostics.bootstrapped_samples == 0
        && treatment_batch_diagnostics.bootstrapped_samples == 0
        && baseline_clean_shutdown
        && treatment_clean_shutdown;
    let executable = std::env::current_exe()?;
    let mean_champion_regret = decision_summary.mean_champion_regret();
    let action_change_rate = decision_summary.action_change_rate();
    let top_screen_recall = decision_summary.top_screen_recall();
    let report = OracleCompareReport {
        schema_version: 1,
        experiment_id: ORACLE_EXPERIMENT_ID.to_owned(),
        status: if complete { "complete" } else { "invalid" }.to_owned(),
        worker,
        first_seed,
        games,
        config,
        comparison,
        decision_summary,
        mean_champion_regret,
        action_change_rate,
        top_screen_recall,
        baseline_diagnostics,
        treatment_diagnostics,
        baseline_batch_diagnostics,
        treatment_batch_diagnostics,
        baseline_clean_shutdown,
        treatment_clean_shutdown,
        elapsed_seconds,
        model_json_blake3: checksum_file(&model_dir.join("model.json"))?,
        model_safetensors_blake3: checksum_file(&model_dir.join("model.safetensors"))?,
        executable_blake3: checksum_file(&executable)?,
        source: source_provenance()?,
        game_records,
        completed_unix_seconds: unix_seconds(),
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report.comparison)?);
    if !complete {
        return Err("full-legal oracle comparison failed an integrity gate".into());
    }
    Ok(())
}

fn run_oracle_merge(
    inputs: &[PathBuf],
    expected_first_seed: u64,
    expected_games: usize,
    output: &Path,
    markdown_output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() || expected_games == 0 {
        return Err("oracle merge requires inputs and a positive expected game count".into());
    }
    let reports = inputs
        .iter()
        .map(|path| {
            let bytes = fs::read(path)?;
            Ok(serde_json::from_slice::<OracleCompareReport>(&bytes)?)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let reference = &reports[0];
    let mut records = Vec::with_capacity(expected_games);
    let mut seeds = std::collections::BTreeSet::new();
    let mut decision_summary = OracleDecisionSummary::default();
    let mut hosts = Vec::with_capacity(reports.len());
    let mut campaign_wall_seconds = 0.0_f64;

    for report in &reports {
        report.config.validate()?;
        if report.schema_version != 1
            || report.experiment_id != ORACLE_EXPERIMENT_ID
            || report.status != "complete"
            || report.games == 0
            || report.game_records.len() != report.games
            || report.decision_summary.decisions != report.games * 80
            || report.config != reference.config
            || report.model_json_blake3 != reference.model_json_blake3
            || report.model_safetensors_blake3 != reference.model_safetensors_blake3
            || report.executable_blake3 != reference.executable_blake3
            || report.source.v2_source_blake3 != reference.source.v2_source_blake3
            || !report.baseline_clean_shutdown
            || !report.treatment_clean_shutdown
            || report.baseline_diagnostics.fallbacks != 0
            || report.treatment_diagnostics.fallbacks != 0
            || report.baseline_batch_diagnostics.policy_fallbacks != 0
            || report.treatment_batch_diagnostics.policy_fallbacks != 0
            || report.baseline_batch_diagnostics.bootstrapped_samples != 0
            || report.treatment_batch_diagnostics.bootstrapped_samples != 0
        {
            return Err(format!(
                "oracle report from {} failed identity or integrity validation",
                report.worker
            )
            .into());
        }
        for (offset, record) in report.game_records.iter().enumerate() {
            let expected_seed = report.first_seed + offset as u64;
            if record.raw_seed != expected_seed || !seeds.insert(record.raw_seed) {
                return Err(format!(
                    "oracle report from {} has duplicate or non-contiguous seed {}",
                    report.worker, record.raw_seed
                )
                .into());
            }
            records.push(record.clone());
        }
        decision_summary.absorb(&report.decision_summary);
        campaign_wall_seconds = campaign_wall_seconds.max(report.elapsed_seconds);
        hosts.push(OracleHostSummary {
            worker: report.worker.clone(),
            first_seed: report.first_seed,
            games: report.games,
            baseline_mean: report.comparison.baseline_mean,
            treatment_mean: report.comparison.treatment_mean,
            mean_paired_delta: report.comparison.mean_paired_delta,
            confidence_95: report.comparison.confidence_95,
            elapsed_seconds: report.elapsed_seconds,
        });
    }

    records.sort_by_key(|record| record.raw_seed);
    let expected_seeds =
        (expected_first_seed..expected_first_seed + expected_games as u64).collect::<Vec<_>>();
    let actual_seeds = records
        .iter()
        .map(|record| record.raw_seed)
        .collect::<Vec<_>>();
    if actual_seeds != expected_seeds {
        return Err(format!(
            "oracle seed coverage mismatch: expected {expected_seeds:?}, found {actual_seeds:?}"
        )
        .into());
    }
    let paired = records
        .iter()
        .map(|record| {
            (
                record.raw_seed,
                record.baseline.clone(),
                record.treatment.clone(),
            )
        })
        .collect::<Vec<_>>();
    let comparison = summarize_paired_match_results(
        CHAMPION_STRATEGY_ID,
        ORACLE_STRATEGY_ID,
        expected_first_seed,
        &paired,
        campaign_wall_seconds,
    );
    let complete_phase_coverage = decision_summary.phase_decisions
        == [
            expected_games * 28,
            expected_games * 28,
            expected_games * 24,
        ];
    let treatment_mean_at_least_100 = comparison.treatment_mean >= 100.0;
    let paired_delta_at_least_3 = comparison.mean_paired_delta >= 3.0;
    let every_host_nonnegative = hosts.iter().all(|host| host.mean_paired_delta >= 0.0);
    let passed = treatment_mean_at_least_100
        && paired_delta_at_least_3
        && every_host_nonnegative
        && complete_phase_coverage;
    let phase_mean_champion_regret = std::array::from_fn(|phase| {
        if decision_summary.phase_decisions[phase] == 0 {
            0.0
        } else {
            decision_summary.phase_champion_regret_sum[phase]
                / decision_summary.phase_decisions[phase] as f64
        }
    });
    let phase_action_change_rate = std::array::from_fn(|phase| {
        if decision_summary.phase_decisions[phase] == 0 {
            0.0
        } else {
            decision_summary.phase_changed_actions[phase] as f64
                / decision_summary.phase_decisions[phase] as f64
        }
    });
    let merged = OracleMergedReport {
        schema_version: 1,
        experiment_id: ORACLE_EXPERIMENT_ID.to_owned(),
        status: if passed {
            "pilot_passed".to_owned()
        } else {
            "pilot_failed".to_owned()
        },
        stage: "pilot".to_owned(),
        expected_first_seed,
        expected_games,
        config: reference.config.clone(),
        mean_champion_regret: decision_summary.mean_champion_regret(),
        action_change_rate: decision_summary.action_change_rate(),
        top_screen_recall: decision_summary.top_screen_recall(),
        phase_mean_champion_regret,
        phase_action_change_rate,
        decision_summary,
        hosts,
        model_json_blake3: reference.model_json_blake3.clone(),
        model_safetensors_blake3: reference.model_safetensors_blake3.clone(),
        executable_blake3: reference.executable_blake3.clone(),
        v2_source_blake3: reference.source.v2_source_blake3.clone(),
        pilot_gates: OraclePilotGates {
            all_integrity_gates_passed: true,
            treatment_mean_at_least_100,
            paired_delta_at_least_3,
            every_host_nonnegative,
            complete_phase_coverage,
            passed,
        },
        input_reports: inputs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        completed_unix_seconds: unix_seconds(),
        comparison,
    };
    write_json_atomic(output, &merged)?;
    if let Some(parent) = markdown_output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(markdown_output, oracle_merged_markdown(&merged))?;
    println!("{}", serde_json::to_string_pretty(&merged.comparison)?);
    Ok(())
}

fn oracle_merged_markdown(report: &OracleMergedReport) -> String {
    let mut output = format!(
        "# Full-Legal Public Oracle V1 Pilot\n\n\
         - Status: **{}**\n\
         - Games: {}\n\
         - Baseline mean: {:.3}\n\
         - Treatment mean: {:.3}\n\
         - Paired delta: **{:+.3}**\n\
         - Paired 95% CI: [{:+.3}, {:+.3}]\n\
         - Mean local champion regret: {:.3}\n\
         - Action change rate: {:.3}%\n\
         - Top-64 winner rate: {:.3}%\n\
         - Pilot gates passed: `{}`\n\n\
         ## Hosts\n\n\
         | Host | Games | Baseline | Treatment | Delta | 95% CI |\n\
         |---|---:|---:|---:|---:|---:|\n",
        report.status,
        report.expected_games,
        report.comparison.baseline_mean,
        report.comparison.treatment_mean,
        report.comparison.mean_paired_delta,
        report.comparison.confidence_95[0],
        report.comparison.confidence_95[1],
        report.mean_champion_regret,
        report.action_change_rate * 100.0,
        report.top_screen_recall * 100.0,
        report.pilot_gates.passed,
    );
    for host in &report.hosts {
        output.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:+.3} | [{:+.3}, {:+.3}] |\n",
            host.worker,
            host.games,
            host.baseline_mean,
            host.treatment_mean,
            host.mean_paired_delta,
            host.confidence_95[0],
            host.confidence_95[1],
        ));
    }
    output.push_str("\n## Phase Diagnostics\n\n");
    output.push_str("| Phase | Decisions | Mean regret | Action change |\n");
    output.push_str("|---|---:|---:|---:|\n");
    for (phase, label) in ["Early", "Middle", "Late"].iter().enumerate() {
        output.push_str(&format!(
            "| {label} | {} | {:.3} | {:.3}% |\n",
            report.decision_summary.phase_decisions[phase],
            report.phase_mean_champion_regret[phase],
            report.phase_action_change_rate[phase] * 100.0,
        ));
    }
    output
}

fn write_json_atomic<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    fs::write(&temporary, serde_json::to_vec(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
