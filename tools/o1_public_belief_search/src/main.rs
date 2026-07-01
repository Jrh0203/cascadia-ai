use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use blake3::Hasher;
use cascadia_ai::nnue::{BagInfo, extract_features_with_bag};
use cascadia_data::{
    DatasetSplit, GradedOracleCandidate, GradedOracleDatasetManifest, GradedOracleGroup,
    PositionRecord, read_graded_oracle_shard, validate_graded_oracle_dataset,
};
use cascadia_differential::legacy_teacher::translate_public_state_allowing_legacy_elk_undercount;
use cascadia_game::{
    DraftChoice, GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, score_board,
};
use cascadia_model::{DEFAULT_SPARSE_NNUE_SHARED_MEMORY_BYTES, LEGACY_NNUE_FEATURES, ModelProcess};
use cascadia_sim::{PatternAwareConfig, rank_pattern_actions};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type AnyError = Box<dyn Error + Send + Sync>;

const EXPERIMENT_ID: &str = "o1-public-belief-one-rotation-search-v1";
const PROTOCOL_ID: &str = "o1-public-belief-one-rotation-high-regret-v2";
const COHORT_WIDTH: usize = 64;
const INTENT_WIDTH: usize = 81;
const ROOT_STAGE_ADDITIONAL_SAMPLES: [usize; 4] = [4, 4, 8, 16];
const ROOT_STAGE_RETAIN: [usize; 4] = [32, 16, 8, 1];
const TRAJECTORIES_PER_GROUP: usize = 640;
const CONTROL_TEMPERATURE: f64 = 1.0;
const PROBABILITY_FLOOR: f64 = 1e-9;
const MODEL_BATCH_ROWS: usize = 4_096;
const ACTION_HASH_DOMAIN: &[u8] = b"cascadia-v2-full-legal-action-v1";
const DETERMINIZATION_DOMAIN: &[u8] =
    b"cascadia-v2-o1-public-belief-search-post-root-determinization-v2";
const OPPONENT_UNIFORM_DOMAIN: &[u8] = b"cascadia-v2-o1-public-belief-search-opponent-uniform-v1";
const TRACE_DOMAIN: &[u8] = b"cascadia-v2-o1-public-belief-search-trace-v1";

#[derive(Debug, Parser)]
#[command(about = "Run the preregistered O1 one-rotation public-belief search")]
struct Args {
    #[arg(long)]
    dataset_root: PathBuf,
    #[arg(long)]
    cohort_root: PathBuf,
    #[arg(long)]
    intent_root: PathBuf,
    #[arg(long)]
    panel: PathBuf,
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    python: PathBuf,
    #[arg(long)]
    authorization: PathBuf,
    #[arg(long)]
    bundle_id: String,
    #[arg(long)]
    role: String,
    #[arg(long)]
    host: String,
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    maximum_groups: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum SearchArm {
    #[value(name = "c0-pattern-prior")]
    C0PatternPrior,
    #[value(name = "a0-public-state-intent")]
    A0PublicStateIntent,
    #[value(name = "a2-history-intent")]
    A2HistoryIntent,
    #[value(name = "s3-shuffled-history-intent")]
    S3ShuffledHistoryIntent,
}

impl SearchArm {
    fn as_str(self) -> &'static str {
        match self {
            Self::C0PatternPrior => "c0-pattern-prior",
            Self::A0PublicStateIntent => "a0-public-state-intent",
            Self::A2HistoryIntent => "a2-history-intent",
            Self::S3ShuffledHistoryIntent => "s3-shuffled-history-intent",
        }
    }

    fn parse(value: &str) -> Result<Self, AnyError> {
        match value {
            "c0-pattern-prior" => Ok(Self::C0PatternPrior),
            "a0-public-state-intent" => Ok(Self::A0PublicStateIntent),
            "a2-history-intent" => Ok(Self::A2HistoryIntent),
            "s3-shuffled-history-intent" => Ok(Self::S3ShuffledHistoryIntent),
            _ => Err(format!("unknown search arm {value}").into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FrozenProtocol {
    root_candidates: usize,
    stage_additional_samples: Vec<usize>,
    stage_retain: Vec<usize>,
    trajectories_per_group: usize,
    opponent_turns: usize,
    control_temperature: f64,
    pattern_config: FrozenPatternConfig,
    leaf_model: String,
    leaf_value: String,
    root_chance_policy: String,
    hidden_order_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FrozenPatternConfig {
    immediate_candidate_limit: usize,
    habitat_candidate_limit: usize,
    bear_candidate_limit: usize,
    future_market_draws: usize,
}

impl FrozenProtocol {
    fn expected() -> Self {
        let pattern = PatternAwareConfig::default();
        Self {
            root_candidates: COHORT_WIDTH,
            stage_additional_samples: ROOT_STAGE_ADDITIONAL_SAMPLES.to_vec(),
            stage_retain: ROOT_STAGE_RETAIN.to_vec(),
            trajectories_per_group: TRAJECTORIES_PER_GROUP,
            opponent_turns: 3,
            control_temperature: CONTROL_TEMPERATURE,
            pattern_config: FrozenPatternConfig {
                immediate_candidate_limit: pattern.immediate_candidate_limit,
                habitat_candidate_limit: pattern.habitat_candidate_limit,
                bear_candidate_limit: pattern.bear_candidate_limit,
                future_market_draws: pattern.future_market_draws,
            },
            leaf_model: "qualified-legacy-v4opp-exact-mlx-v1".to_owned(),
            leaf_value: "v2-current-base-score-plus-legacy-nnue-remaining-value".to_owned(),
            root_chance_policy: "condition-on-frozen-complete-turn-staged-prelude-context"
                .to_owned(),
            hidden_order_policy:
                "sort-and-redeterminize-after-frozen-root-before-opponent-rotation".to_owned(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Authorization {
    schema_version: u16,
    experiment_id: String,
    protocol_id: String,
    authorization_id: String,
    bundle_id: String,
    roles: BTreeMap<String, String>,
    protocol: FrozenProtocol,
    inputs: AuthorizationInputs,
}

#[derive(Debug, Deserialize)]
struct AuthorizationInputs {
    dataset_id: String,
    dataset_manifest_blake3: String,
    cohort_id: String,
    cohort_manifest_blake3: String,
    intent_id: String,
    intent_manifest_blake3: String,
    panel_id: String,
    panel_blake3: String,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Panel {
    schema_version: u16,
    experiment_id: String,
    panel_id: String,
    source_experiment_id: String,
    source_report_blake3: String,
    threshold: f64,
    split: String,
    groups: Vec<PanelGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PanelGroup {
    row: usize,
    group_id: u64,
    game_index: u64,
    turn: u16,
    control_regret: f64,
}

#[derive(Debug)]
struct CohortData {
    cache_id: String,
    dataset_id: String,
    groups: usize,
    group_ids: Vec<u64>,
    source_candidate_indices: Vec<u16>,
    action_hashes: Vec<[u8; 32]>,
}

#[derive(Debug)]
struct IntentData {
    cache_id: String,
    groups: usize,
    a0: Vec<f32>,
    a2: Vec<f32>,
    shuffle_sources: Vec<u32>,
}

#[derive(Debug, Clone)]
struct RootCandidate {
    source_index: u16,
    action_hash: [u8; 32],
    action: TurnAction,
    intent: [f32; INTENT_WIDTH],
    r600_mean: Option<f32>,
    r1200_mean: Option<f32>,
    r4800_mean: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct DraftKey {
    kind: u8,
    tile_slot: u8,
    wildlife_slot: u8,
}

impl DraftKey {
    fn from_choice(choice: DraftChoice) -> Self {
        match choice {
            DraftChoice::Paired { slot } => Self {
                kind: 0,
                tile_slot: slot.index() as u8,
                wildlife_slot: slot.index() as u8,
            },
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => Self {
                kind: 1,
                tile_slot: tile_slot.index() as u8,
                wildlife_slot: wildlife_slot.index() as u8,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct DraftOption {
    key: DraftKey,
    action: TurnAction,
    action_hash: [u8; 32],
    heuristic_value: f64,
    drafted_wildlife: usize,
}

#[derive(Debug, Clone)]
struct LeafSample {
    actual_score: f64,
    sparse_features: Option<Vec<u16>>,
    trace_hash: [u8; 32],
    public_leaf_hash: [u8; 32],
    opponent_decisions: usize,
    opponent_options: usize,
}

#[derive(Debug, Clone, Default)]
struct RunningMoments {
    samples: usize,
    sum: f64,
    square_sum: f64,
}

impl RunningMoments {
    fn add(&mut self, value: f64) {
        self.samples += 1;
        self.sum += value;
        self.square_sum += value * value;
    }

    fn mean(&self) -> f64 {
        self.sum / self.samples as f64
    }

    fn stddev(&self) -> f64 {
        if self.samples < 2 {
            return 0.0;
        }
        let mean = self.mean();
        ((self.square_sum - self.samples as f64 * mean * mean) / (self.samples - 1) as f64)
            .max(0.0)
            .sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateResult {
    cohort_index: usize,
    source_index: u16,
    action_hash: String,
    search_mean: f64,
    search_stddev: f64,
    samples: usize,
    eliminated_stage: Option<usize>,
    r600_mean: Option<f32>,
    r1200_mean: Option<f32>,
    r4800_mean: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupDiagnostics {
    root_candidates: usize,
    trajectories: usize,
    leaf_model_rows: usize,
    terminal_leaves: usize,
    opponent_decisions: usize,
    opponent_options: usize,
    hidden_order_invariance_checks: usize,
    candidate_hash_checks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupResult {
    panel_row: usize,
    cohort_row: usize,
    group_id: u64,
    game_index: u64,
    completed_turns: u16,
    current_player: u8,
    public_state_hash: String,
    arm: String,
    selected_cohort_index: usize,
    selected_source_index: u16,
    selected_action_hash: String,
    selected_search_mean: f64,
    selected_search_stddev: f64,
    candidates: Vec<CandidateResult>,
    diagnostics: GroupDiagnostics,
    group_result_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RunIdentity {
    schema_version: u16,
    experiment_id: String,
    protocol_id: String,
    authorization_id: String,
    bundle_id: String,
    role: String,
    arm: String,
    host: String,
    production: bool,
    maximum_groups: Option<usize>,
    inputs: BTreeMap<String, String>,
    protocol: FrozenProtocol,
}

#[derive(Debug, Serialize)]
struct FinalReport {
    schema_version: u16,
    experiment_id: String,
    protocol_id: String,
    authorization_id: String,
    bundle_id: String,
    role: String,
    arm: String,
    host: String,
    production: bool,
    inputs: BTreeMap<String, String>,
    protocol: FrozenProtocol,
    groups_expected: usize,
    groups_completed: usize,
    root_candidates: usize,
    trajectories: usize,
    leaf_model_rows: usize,
    terminal_leaves: usize,
    opponent_decisions: usize,
    opponent_options: usize,
    hidden_order_invariance_checks: usize,
    candidate_hash_checks: usize,
    wall_seconds: f64,
    groups: Vec<GroupResult>,
    scientific_result_id: String,
    report_id: String,
}

fn main() -> Result<(), AnyError> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<(), AnyError> {
    let started = Instant::now();
    let authorization = read_json::<Authorization>(&args.authorization)?;
    validate_authorization(&authorization, &args)?;
    let arm = SearchArm::parse(
        authorization
            .roles
            .get(&args.role)
            .ok_or("authorization omitted requested role")?,
    )?;
    let panel = read_json::<Panel>(&args.panel)?;
    validate_panel(&panel)?;
    let cohort = load_cohort(&args.cohort_root)?;
    let intent = load_intent(&args.intent_root, cohort.groups)?;
    let dataset_manifest =
        read_json::<GradedOracleDatasetManifest>(&args.dataset_root.join("dataset.json"))?;
    validate_graded_oracle_dataset(&args.dataset_root, &dataset_manifest)?;
    validate_inputs(
        &authorization,
        &args,
        &panel,
        &cohort,
        &intent,
        &dataset_manifest,
    )?;

    let production = args.maximum_groups.is_none();
    let selected_panel_groups = args
        .maximum_groups
        .map_or(panel.groups.len(), |limit| limit.min(panel.groups.len()));
    if selected_panel_groups == 0 {
        return Err("maximum-groups selected no work".into());
    }
    let active_panel = Panel {
        groups: panel.groups[..selected_panel_groups].to_vec(),
        ..panel.clone()
    };
    let panel_by_group = active_panel
        .groups
        .iter()
        .enumerate()
        .map(|(panel_row, group)| (group.group_id, (panel_row, group)))
        .collect::<HashMap<_, _>>();
    if panel_by_group.len() != active_panel.groups.len() {
        return Err("panel repeats a group".into());
    }

    fs::create_dir_all(args.run_dir.join("groups"))?;
    let run_identity = build_run_identity(
        &authorization,
        &args,
        arm,
        production,
        &dataset_manifest,
        &cohort,
        &intent,
        &panel,
    )?;
    freeze_run_identity(&args.run_dir.join("run.json"), &run_identity)?;

    let process = ModelProcess::spawn_with_sparse_nnue_shared_memory(
        &args.python,
        [
            "-m",
            "cascadia_mlx.legacy_nnue_serve",
            "--model-dir",
            args.model_dir
                .to_str()
                .ok_or("model path is not valid UTF-8")?,
        ],
        DEFAULT_SPARSE_NNUE_SHARED_MEMORY_BYTES,
    )?;
    let mut evaluator = LeafEvaluator { process };
    let warmup = evaluator
        .process
        .predict_sparse_nnue_csr_exact(&[Vec::new()])?;
    if warmup.len() != 1 || !warmup[0].is_finite() {
        return Err("leaf model warmup failed".into());
    }

    let mut completed = Vec::with_capacity(active_panel.groups.len());
    let mut seen_groups = HashSet::new();
    for shard in &dataset_manifest.shards {
        let groups = read_graded_oracle_shard(&args.dataset_root, DatasetSplit::Validation, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        for group in groups {
            if let Some(&(panel_row, panel_group)) = panel_by_group.get(&group.group_id) {
                if !seen_groups.insert(group.group_id) {
                    return Err(format!("panel group {} replayed twice", group.group_id).into());
                }
                validate_replayed_group(&game, &group, panel_group)?;
                let group_path = args
                    .run_dir
                    .join("groups")
                    .join(format!("row-{panel_row:03}.json"));
                let result = if group_path.exists() {
                    let existing = read_json::<GroupResult>(&group_path)?;
                    validate_resumed_group(&existing, panel_row, &group, arm)?;
                    existing
                } else {
                    let result = evaluate_group(
                        &game,
                        &group,
                        panel_row,
                        panel_group.row,
                        arm,
                        &cohort,
                        &intent,
                        &mut evaluator,
                    )?;
                    write_json_atomic(&group_path, &result)?;
                    result
                };
                completed.push(result);
            }
            let champion = group.candidates[usize::from(group.champion_index)]
                .action
                .to_game_action(&game)?;
            game.apply(&champion)?;
        }
        if !game.is_game_over() {
            return Err(format!(
                "validation game {} did not replay to completion",
                shard.first_game_index
            )
            .into());
        }
    }
    evaluator.process.shutdown()?;

    if seen_groups.len() != active_panel.groups.len() {
        return Err(format!(
            "replay found {} of {} panel groups",
            seen_groups.len(),
            active_panel.groups.len()
        )
        .into());
    }
    completed.sort_by_key(|group| group.panel_row);
    for (expected, group) in completed.iter().enumerate() {
        if group.panel_row != expected {
            return Err("completed group order is incomplete".into());
        }
    }
    let totals = aggregate_diagnostics(&completed);
    let scientific_identity = json!({
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "authorization_id": authorization.authorization_id,
        "bundle_id": authorization.bundle_id,
        "arm": arm.as_str(),
        "production": production,
        "inputs": run_identity.inputs,
        "protocol": run_identity.protocol,
        "groups": completed,
    });
    let scientific_result_id = canonical_blake3(&scientific_identity)?;
    let report_without_id = json!({
        "scientific_result_id": scientific_result_id,
        "role": args.role,
        "host": args.host,
        "wall_seconds": started.elapsed().as_secs_f64(),
    });
    let report_id = canonical_blake3(&report_without_id)?;
    let report = FinalReport {
        schema_version: 1,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        authorization_id: authorization.authorization_id,
        bundle_id: authorization.bundle_id,
        role: args.role,
        arm: arm.as_str().to_owned(),
        host: args.host,
        production,
        inputs: run_identity.inputs,
        protocol: run_identity.protocol,
        groups_expected: active_panel.groups.len(),
        groups_completed: completed.len(),
        root_candidates: totals.root_candidates,
        trajectories: totals.trajectories,
        leaf_model_rows: totals.leaf_model_rows,
        terminal_leaves: totals.terminal_leaves,
        opponent_decisions: totals.opponent_decisions,
        opponent_options: totals.opponent_options,
        hidden_order_invariance_checks: totals.hidden_order_invariance_checks,
        candidate_hash_checks: totals.candidate_hash_checks,
        wall_seconds: started.elapsed().as_secs_f64(),
        groups: completed,
        scientific_result_id,
        report_id,
    };
    write_json_atomic(&args.output, &report)?;
    Ok(())
}

struct LeafEvaluator {
    process: ModelProcess,
}

fn evaluate_group(
    game: &GameState,
    group: &GradedOracleGroup,
    panel_row: usize,
    cohort_row: usize,
    arm: SearchArm,
    cohort: &CohortData,
    intent: &IntentData,
    evaluator: &mut LeafEvaluator,
) -> Result<GroupResult, AnyError> {
    let roots = reconstruct_roots(game, group, cohort_row, arm, cohort, intent)?;
    let focal_player = game.current_player();
    let root_afterstate = apply_frozen_root(game, group.group_id, &roots[0], 0)?;
    let invariance_left = run_post_root_trajectory(
        root_afterstate.clone(),
        focal_player,
        group.group_id,
        &roots[0],
        0,
        arm,
    )?;
    let mut perturbed = root_afterstate;
    perturbed.redeterminize_hidden(GameSeed::from_u64(group.group_id ^ 0xa5a5_5a5a));
    let invariance_right =
        run_post_root_trajectory(perturbed, focal_player, group.group_id, &roots[0], 0, arm)?;
    if !leaf_samples_equivalent(&invariance_left, &invariance_right) {
        return Err(format!(
            "hidden-order invariance failed for group {}",
            group.group_id
        )
        .into());
    }

    let mut moments = vec![RunningMoments::default(); roots.len()];
    let mut eliminated_stage = vec![None; roots.len()];
    let mut active = (0..roots.len()).collect::<Vec<_>>();
    let mut sample_start = 0usize;
    let mut leaf_model_rows = 0usize;
    let mut terminal_leaves = 0usize;
    let mut opponent_options = 0usize;
    let mut opponent_decisions = 0usize;

    for (stage, additional_samples) in ROOT_STAGE_ADDITIONAL_SAMPLES.into_iter().enumerate() {
        let work = active
            .iter()
            .flat_map(|&candidate| {
                (sample_start..sample_start + additional_samples)
                    .map(move |sample| (candidate, sample))
            })
            .collect::<Vec<_>>();
        let generated = work
            .par_iter()
            .map(|&(candidate, sample)| {
                run_trajectory(game, group.group_id, &roots[candidate], sample, arm)
                    .map(|leaf| (candidate, leaf))
            })
            .collect::<Vec<_>>();
        let mut samples = Vec::with_capacity(generated.len());
        for result in generated {
            samples.push(result?);
        }
        let mut feature_rows = Vec::new();
        let mut feature_sample_indices = Vec::new();
        for (sample_index, (_, sample)) in samples.iter().enumerate() {
            opponent_decisions += sample.opponent_decisions;
            opponent_options += sample.opponent_options;
            if let Some(features) = &sample.sparse_features {
                feature_rows.push(features.clone());
                feature_sample_indices.push(sample_index);
            } else {
                terminal_leaves += 1;
            }
        }
        let mut predictions = Vec::with_capacity(feature_rows.len());
        for chunk in feature_rows.chunks(MODEL_BATCH_ROWS) {
            predictions.extend(evaluator.process.predict_sparse_nnue_csr_exact(chunk)?);
        }
        if predictions.len() != feature_rows.len() {
            return Err("leaf evaluator returned the wrong row count".into());
        }
        leaf_model_rows += predictions.len();
        let mut values = samples
            .iter()
            .map(|(_, sample)| sample.actual_score)
            .collect::<Vec<_>>();
        for (&sample_index, &remaining) in feature_sample_indices.iter().zip(&predictions) {
            values[sample_index] += f64::from(remaining);
        }
        for ((candidate, _), value) in samples.into_iter().zip(values) {
            if !value.is_finite() {
                return Err("trajectory produced a non-finite leaf value".into());
            }
            moments[candidate].add(value);
        }
        sample_start += additional_samples;

        active.sort_by(|&left, &right| {
            moments[right]
                .mean()
                .total_cmp(&moments[left].mean())
                .then_with(|| roots[left].action_hash.cmp(&roots[right].action_hash))
        });
        let retained = ROOT_STAGE_RETAIN[stage];
        for &candidate in &active[retained..] {
            eliminated_stage[candidate] = Some(stage + 1);
        }
        active.truncate(retained);
    }
    if active.len() != 1 {
        return Err("sequential halving did not select exactly one root".into());
    }
    let selected = active[0];
    let trajectories = moments.iter().map(|value| value.samples).sum::<usize>();
    if trajectories != TRAJECTORIES_PER_GROUP {
        return Err(format!(
            "group {} used {trajectories} trajectories instead of {TRAJECTORIES_PER_GROUP}",
            group.group_id
        )
        .into());
    }
    let candidates = roots
        .iter()
        .enumerate()
        .map(|(index, root)| CandidateResult {
            cohort_index: index,
            source_index: root.source_index,
            action_hash: hex(&root.action_hash),
            search_mean: moments[index].mean(),
            search_stddev: moments[index].stddev(),
            samples: moments[index].samples,
            eliminated_stage: eliminated_stage[index],
            r600_mean: root.r600_mean,
            r1200_mean: root.r1200_mean,
            r4800_mean: root.r4800_mean,
        })
        .collect::<Vec<_>>();
    let diagnostics = GroupDiagnostics {
        root_candidates: roots.len(),
        trajectories,
        leaf_model_rows,
        terminal_leaves,
        opponent_decisions,
        opponent_options,
        hidden_order_invariance_checks: 1,
        candidate_hash_checks: roots.len(),
    };
    let identity = json!({
        "panel_row": panel_row,
        "cohort_row": cohort_row,
        "group_id": group.group_id,
        "arm": arm.as_str(),
        "selected_cohort_index": selected,
        "candidates": candidates,
        "diagnostics": diagnostics,
    });
    let group_result_id = canonical_blake3(&identity)?;
    Ok(GroupResult {
        panel_row,
        cohort_row,
        group_id: group.group_id,
        game_index: group.raw_seed,
        completed_turns: group.completed_turns,
        current_player: group.current_player,
        public_state_hash: hex(&group.public_state_hash),
        arm: arm.as_str().to_owned(),
        selected_cohort_index: selected,
        selected_source_index: roots[selected].source_index,
        selected_action_hash: hex(&roots[selected].action_hash),
        selected_search_mean: moments[selected].mean(),
        selected_search_stddev: moments[selected].stddev(),
        candidates,
        diagnostics,
        group_result_id,
    })
}

fn reconstruct_roots(
    game: &GameState,
    group: &GradedOracleGroup,
    cohort_row: usize,
    arm: SearchArm,
    cohort: &CohortData,
    intent: &IntentData,
) -> Result<Vec<RootCandidate>, AnyError> {
    if cohort_row >= cohort.groups {
        return Err("panel cohort row is out of range".into());
    }
    if cohort.group_ids[cohort_row] != group.group_id {
        return Err("panel cohort row points at a different group".into());
    }
    let mut roots = Vec::with_capacity(COHORT_WIDTH);
    for cohort_index in 0..COHORT_WIDTH {
        let flat = cohort_row * COHORT_WIDTH + cohort_index;
        let source_index = cohort.source_candidate_indices[flat];
        let candidate = group
            .candidates
            .get(usize::from(source_index))
            .ok_or("cohort source candidate is out of range")?;
        let action = candidate.action.to_game_action(game)?;
        let observed_hash = canonical_action_hash(&action)?;
        if observed_hash != cohort.action_hashes[flat] || observed_hash != candidate.action_hash {
            return Err(format!(
                "root action hash drifted for group {} source {}",
                group.group_id, source_index
            )
            .into());
        }
        let intent = intent_vector(arm, flat, intent)?;
        roots.push(RootCandidate {
            source_index,
            action_hash: observed_hash,
            action,
            intent,
            r600_mean: estimate(
                candidate,
                |candidate| candidate.r600.mean,
                candidate.r600.samples,
            ),
            r1200_mean: estimate(
                candidate,
                |candidate| candidate.r1200.mean,
                candidate.r1200.samples,
            ),
            r4800_mean: estimate(
                candidate,
                |candidate| candidate.r4800.mean,
                candidate.r4800.samples,
            ),
        });
    }
    Ok(roots)
}

fn estimate(
    candidate: &GradedOracleCandidate,
    value: impl FnOnce(&GradedOracleCandidate) -> f32,
    samples: u16,
) -> Option<f32> {
    (samples > 0).then(|| value(candidate))
}

fn intent_vector(
    arm: SearchArm,
    flat: usize,
    intent: &IntentData,
) -> Result<[f32; INTENT_WIDTH], AnyError> {
    let source = match arm {
        SearchArm::C0PatternPrior => return Ok([0.0; INTENT_WIDTH]),
        SearchArm::A0PublicStateIntent => {
            &intent.a0[flat * INTENT_WIDTH..(flat + 1) * INTENT_WIDTH]
        }
        SearchArm::A2HistoryIntent => &intent.a2[flat * INTENT_WIDTH..(flat + 1) * INTENT_WIDTH],
        SearchArm::S3ShuffledHistoryIntent => {
            let donor = usize::try_from(intent.shuffle_sources[flat])?;
            let candidates = intent.groups * COHORT_WIDTH;
            if donor >= candidates {
                return Err("shuffle donor is out of range".into());
            }
            &intent.a2[donor * INTENT_WIDTH..(donor + 1) * INTENT_WIDTH]
        }
    };
    Ok(source.try_into().expect("intent vector width is exact"))
}

fn run_trajectory(
    game: &GameState,
    group_id: u64,
    root: &RootCandidate,
    sample: usize,
    arm: SearchArm,
) -> Result<LeafSample, AnyError> {
    let focal_player = game.current_player();
    let state = apply_frozen_root(game, group_id, root, sample)?;
    run_post_root_trajectory(state, focal_player, group_id, root, sample, arm)
}

fn apply_frozen_root(
    game: &GameState,
    group_id: u64,
    root: &RootCandidate,
    sample: usize,
) -> Result<GameState, AnyError> {
    let mut state = game.clone();
    state.apply(&root.action).map_err(|error| {
        format!(
            "root apply failed: group={group_id} sample={sample} root={} action={} error={error}",
            hex(&root.action_hash),
            serde_json::to_string(&root.action).expect("turn action serializes"),
        )
    })?;
    Ok(state)
}

fn run_post_root_trajectory(
    mut state: GameState,
    focal_player: usize,
    group_id: u64,
    root: &RootCandidate,
    sample: usize,
    arm: SearchArm,
) -> Result<LeafSample, AnyError> {
    state.redeterminize_hidden(determinization_seed(group_id, &root.action_hash, sample));
    let mut trace_hasher = Hasher::new();
    trace_hasher.update(TRACE_DOMAIN);
    trace_hasher.update(&root.action_hash);
    let mut opponent_decisions = 0usize;
    let mut opponent_options = 0usize;

    for opponent_offset in 0..3 {
        if state.is_game_over() {
            break;
        }
        let (action, option_count) = select_opponent_action(
            &state,
            group_id,
            &root.action_hash,
            sample,
            opponent_offset,
            arm,
            &root.intent,
        )?;
        opponent_decisions += 1;
        opponent_options += option_count;
        let action_hash = canonical_action_hash(&action)?;
        trace_hasher.update(&action_hash);
        state.apply(&action).map_err(|error| {
            format!(
                "opponent apply failed: group={group_id} sample={sample} root={} \
                 opponent_offset={opponent_offset} public_state={} action_hash={} \
                 action={} error={error}",
                hex(&root.action_hash),
                state.public_state().canonical_hash().to_hex(),
                hex(&action_hash),
                serde_json::to_string(&action).expect("turn action serializes"),
            )
        })?;
    }
    let score = f64::from(
        score_board(&state.boards()[focal_player], state.config().scoring_cards).base_total,
    );
    let sparse_features = if state.is_game_over() {
        None
    } else {
        if state.current_player() != focal_player {
            return Err("one-rotation leaf did not return to the focal player".into());
        }
        let translated =
            translate_public_state_allowing_legacy_elk_undercount(&state.public_state())?;
        let bag = BagInfo::from_game_for_player(&translated.game, focal_player);
        let features = extract_features_with_bag(&translated.game.boards[focal_player], Some(&bag));
        if features
            .iter()
            .any(|&feature| usize::from(feature) >= LEGACY_NNUE_FEATURES)
        {
            return Err("leaf sparse feature crossed the qualified model width".into());
        }
        Some(features)
    };
    Ok(LeafSample {
        actual_score: score,
        sparse_features,
        trace_hash: *trace_hasher.finalize().as_bytes(),
        public_leaf_hash: *state.public_state().canonical_hash().as_bytes(),
        opponent_decisions,
        opponent_options,
    })
}

fn select_opponent_action(
    state: &GameState,
    group_id: u64,
    root_hash: &[u8; 32],
    sample: usize,
    opponent_offset: usize,
    arm: SearchArm,
    intent: &[f32; INTENT_WIDTH],
) -> Result<(TurnAction, usize), AnyError> {
    let prelude = MarketPrelude {
        replace_three_of_a_kind: state.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let staged = state.preview_market_prelude(&prelude)?;
    let candidates = rank_pattern_actions(state, &prelude, PatternAwareConfig::default())?;
    if candidates.is_empty() {
        return Err("pattern policy produced no opponent actions".into());
    }
    let mut by_draft = BTreeMap::<DraftKey, DraftOption>::new();
    for candidate in candidates {
        let key = DraftKey::from_choice(candidate.action.draft);
        let wildlife = staged.market().wildlife[usize::from(key.wildlife_slot)]
            .ok_or("opponent draft points at an empty wildlife slot")?
            as usize;
        let action_hash = canonical_action_hash(&candidate.action)?;
        let option = DraftOption {
            key,
            action: candidate.action,
            action_hash,
            heuristic_value: candidate.heuristic_value,
            drafted_wildlife: wildlife,
        };
        match by_draft.get(&key) {
            None => {
                by_draft.insert(key, option);
            }
            Some(previous)
                if option.heuristic_value > previous.heuristic_value
                    || (option.heuristic_value == previous.heuristic_value
                        && option.action_hash < previous.action_hash) =>
            {
                by_draft.insert(key, option);
            }
            Some(_) => {}
        }
    }
    let options = by_draft.into_values().collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(options.len());
    match arm {
        SearchArm::C0PatternPrior => {
            let best = options
                .iter()
                .map(|option| option.heuristic_value)
                .max_by(f64::total_cmp)
                .ok_or("control policy has no draft options")?;
            for option in &options {
                weights.push(
                    ((option.heuristic_value - best) / CONTROL_TEMPERATURE)
                        .clamp(-40.0, 0.0)
                        .exp(),
                );
            }
        }
        SearchArm::A0PublicStateIntent
        | SearchArm::A2HistoryIntent
        | SearchArm::S3ShuffledHistoryIntent => {
            for option in &options {
                weights.push(intent_draft_probability(intent, opponent_offset, option)?);
            }
        }
    }
    let selected = weighted_index(
        &weights,
        opponent_uniform(group_id, root_hash, sample, opponent_offset),
    )?;
    Ok((options[selected].action.clone(), options.len()))
}

fn intent_draft_probability(
    intent: &[f32; INTENT_WIDTH],
    opponent_offset: usize,
    option: &DraftOption,
) -> Result<f64, AnyError> {
    if opponent_offset >= 3 || option.drafted_wildlife >= 5 {
        return Err("intent draft lookup is out of range".into());
    }
    let tile = f64::from(intent[36 + opponent_offset * 4 + usize::from(option.key.tile_slot)])
        .max(PROBABILITY_FLOOR);
    let wildlife =
        f64::from(intent[48 + opponent_offset * 4 + usize::from(option.key.wildlife_slot)])
            .max(PROBABILITY_FLOOR);
    let independent =
        f64::from(intent[60 + opponent_offset]).clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR);
    let kind = if option.key.kind == 1 {
        independent
    } else {
        1.0 - independent
    };
    let drafted = f64::from(intent[63 + opponent_offset * 5 + option.drafted_wildlife])
        .max(PROBABILITY_FLOOR);
    let probability = kind * (tile * wildlife * drafted).powf(1.0 / 3.0);
    if !probability.is_finite() || probability <= 0.0 {
        return Err("intent policy emitted an invalid legal-draft probability".into());
    }
    Ok(probability)
}

fn weighted_index(weights: &[f64], uniform: f64) -> Result<usize, AnyError> {
    let total = weights.iter().sum::<f64>();
    if weights.is_empty()
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || !total.is_finite()
        || total <= 0.0
        || !(0.0..1.0).contains(&uniform)
    {
        return Err("weighted opponent selection received invalid inputs".into());
    }
    let target = uniform * total;
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if target < cumulative {
            return Ok(index);
        }
    }
    Ok(weights.len() - 1)
}

fn determinization_seed(group_id: u64, action_hash: &[u8; 32], sample: usize) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(DETERMINIZATION_DOMAIN);
    hasher.update(&group_id.to_le_bytes());
    hasher.update(action_hash);
    hasher.update(&(sample as u64).to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn opponent_uniform(
    group_id: u64,
    action_hash: &[u8; 32],
    sample: usize,
    opponent_offset: usize,
) -> f64 {
    let mut hasher = Hasher::new();
    hasher.update(OPPONENT_UNIFORM_DOMAIN);
    hasher.update(&group_id.to_le_bytes());
    hasher.update(action_hash);
    hasher.update(&(sample as u64).to_le_bytes());
    hasher.update(&(opponent_offset as u64).to_le_bytes());
    let bytes = hasher.finalize();
    let numerator = u64::from_le_bytes(bytes.as_bytes()[..8].try_into().expect("eight bytes"));
    (numerator as f64) / ((u64::MAX as f64) + 1.0)
}

fn leaf_samples_equivalent(left: &LeafSample, right: &LeafSample) -> bool {
    left.actual_score == right.actual_score
        && left.sparse_features == right.sparse_features
        && left.trace_hash == right.trace_hash
        && left.public_leaf_hash == right.public_leaf_hash
        && left.opponent_decisions == right.opponent_decisions
        && left.opponent_options == right.opponent_options
}

fn validate_replayed_group(
    game: &GameState,
    group: &GradedOracleGroup,
    panel: &PanelGroup,
) -> Result<(), AnyError> {
    if game.completed_turns() != group.completed_turns
        || game.current_player() != usize::from(group.current_player)
        || PositionRecord::observe(game, group.raw_seed).to_bytes() != group.position.to_bytes()
        || *game.public_state().canonical_hash().as_bytes() != group.public_state_hash
        || panel.group_id != group.group_id
        || panel.game_index != group.raw_seed
        || panel.turn != group.completed_turns
    {
        return Err(format!("validation replay drifted at group {}", group.group_id).into());
    }
    Ok(())
}

fn validate_resumed_group(
    result: &GroupResult,
    panel_row: usize,
    group: &GradedOracleGroup,
    arm: SearchArm,
) -> Result<(), AnyError> {
    if result.panel_row != panel_row
        || result.group_id != group.group_id
        || result.arm != arm.as_str()
        || result.candidates.len() != COHORT_WIDTH
        || result.diagnostics.trajectories != TRAJECTORIES_PER_GROUP
    {
        return Err("resumed group artifact does not match this run".into());
    }
    Ok(())
}

fn aggregate_diagnostics(groups: &[GroupResult]) -> GroupDiagnostics {
    groups.iter().fold(
        GroupDiagnostics {
            root_candidates: 0,
            trajectories: 0,
            leaf_model_rows: 0,
            terminal_leaves: 0,
            opponent_decisions: 0,
            opponent_options: 0,
            hidden_order_invariance_checks: 0,
            candidate_hash_checks: 0,
        },
        |mut total, group| {
            total.root_candidates += group.diagnostics.root_candidates;
            total.trajectories += group.diagnostics.trajectories;
            total.leaf_model_rows += group.diagnostics.leaf_model_rows;
            total.terminal_leaves += group.diagnostics.terminal_leaves;
            total.opponent_decisions += group.diagnostics.opponent_decisions;
            total.opponent_options += group.diagnostics.opponent_options;
            total.hidden_order_invariance_checks +=
                group.diagnostics.hidden_order_invariance_checks;
            total.candidate_hash_checks += group.diagnostics.candidate_hash_checks;
            total
        },
    )
}

fn validate_authorization(authorization: &Authorization, args: &Args) -> Result<(), AnyError> {
    if authorization.schema_version != 1
        || authorization.experiment_id != EXPERIMENT_ID
        || authorization.protocol_id != PROTOCOL_ID
        || authorization.bundle_id != args.bundle_id
        || authorization.protocol != FrozenProtocol::expected()
        || !is_digest(&authorization.authorization_id)
        || !authorization.roles.contains_key(&args.role)
    {
        return Err("search authorization does not match the frozen protocol".into());
    }
    Ok(())
}

fn validate_panel(panel: &Panel) -> Result<(), AnyError> {
    if panel.schema_version != 1
        || panel.experiment_id != EXPERIMENT_ID
        || panel.source_experiment_id != "o1-high-regret-draft-ranking-integration-v1"
        || panel.split != "validation"
        || panel.threshold != 0.5
        || panel.groups.len() != 99
        || !is_digest(&panel.panel_id)
        || !is_digest(&panel.source_report_blake3)
    {
        return Err("high-regret panel does not match the preregistered contract".into());
    }
    if panel
        .groups
        .iter()
        .any(|group| group.control_regret < panel.threshold)
    {
        return Err("high-regret panel contains an ineligible group".into());
    }
    Ok(())
}

fn validate_inputs(
    authorization: &Authorization,
    args: &Args,
    panel: &Panel,
    cohort: &CohortData,
    intent: &IntentData,
    dataset: &GradedOracleDatasetManifest,
) -> Result<(), AnyError> {
    let inputs = &authorization.inputs;
    if dataset.split != DatasetSplit::Validation
        || dataset.dataset_id != inputs.dataset_id
        || dataset.dataset_id != cohort.dataset_id
        || checksum(&args.dataset_root.join("dataset.json"))? != inputs.dataset_manifest_blake3
        || cohort.cache_id != inputs.cohort_id
        || checksum(&args.cohort_root.join("cache.json"))? != inputs.cohort_manifest_blake3
        || intent.cache_id != inputs.intent_id
        || checksum(&args.intent_root.join("cache.json"))? != inputs.intent_manifest_blake3
        || panel.panel_id != inputs.panel_id
        || checksum(&args.panel)? != inputs.panel_blake3
        || checksum(&args.model_dir.join("model.json"))? != inputs.model_manifest_blake3
        || checksum(&args.model_dir.join("model.safetensors"))? != inputs.model_safetensors_blake3
    {
        return Err("runtime input bytes differ from the frozen authorization".into());
    }
    Ok(())
}

fn build_run_identity(
    authorization: &Authorization,
    args: &Args,
    arm: SearchArm,
    production: bool,
    dataset: &GradedOracleDatasetManifest,
    cohort: &CohortData,
    intent: &IntentData,
    panel: &Panel,
) -> Result<RunIdentity, AnyError> {
    let mut inputs = BTreeMap::new();
    inputs.insert("dataset_id".to_owned(), dataset.dataset_id.clone());
    inputs.insert("cohort_id".to_owned(), cohort.cache_id.clone());
    inputs.insert("intent_id".to_owned(), intent.cache_id.clone());
    inputs.insert("panel_id".to_owned(), panel.panel_id.clone());
    inputs.insert(
        "model_manifest_blake3".to_owned(),
        checksum(&args.model_dir.join("model.json"))?,
    );
    inputs.insert(
        "model_safetensors_blake3".to_owned(),
        checksum(&args.model_dir.join("model.safetensors"))?,
    );
    Ok(RunIdentity {
        schema_version: 1,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        authorization_id: authorization.authorization_id.clone(),
        bundle_id: authorization.bundle_id.clone(),
        role: args.role.clone(),
        arm: arm.as_str().to_owned(),
        host: args.host.clone(),
        production,
        maximum_groups: args.maximum_groups,
        inputs,
        protocol: FrozenProtocol::expected(),
    })
}

fn freeze_run_identity(path: &Path, expected: &RunIdentity) -> Result<(), AnyError> {
    if path.exists() {
        let observed = read_json::<RunIdentity>(path)?;
        if &observed != expected {
            return Err("existing run identity differs from this invocation".into());
        }
    } else {
        write_json_atomic(path, expected)?;
    }
    Ok(())
}

fn load_cohort(root: &Path) -> Result<CohortData, AnyError> {
    let manifest = read_json::<Value>(&root.join("cache.json"))?;
    let validation = &manifest["splits"]["validation"];
    let groups = usize_value(&validation["groups"], "cohort validation groups")?;
    if groups != 240 {
        return Err("cohort validation group count drifted".into());
    }
    let files = &validation["files"];
    let group_ids = read_u64_file(root.join(string_value(
        &files["group_ids"]["file"],
        "cohort group file",
    )?))?;
    let source_candidate_indices = read_u16_file(root.join(string_value(
        &files["source_candidate_indices"]["file"],
        "cohort source file",
    )?))?;
    let action_bytes = fs::read(root.join(string_value(
        &files["action_hashes"]["file"],
        "cohort action-hash file",
    )?))?;
    if group_ids.len() != groups
        || source_candidate_indices.len() != groups * COHORT_WIDTH
        || action_bytes.len() != groups * COHORT_WIDTH * 32
    {
        return Err("cohort tensor sizes drifted".into());
    }
    let action_hashes = action_bytes
        .chunks_exact(32)
        .map(|bytes| bytes.try_into().expect("hash width is exact"))
        .collect::<Vec<_>>();
    Ok(CohortData {
        cache_id: string_value(&manifest["cache_id"], "cohort ID")?.to_owned(),
        dataset_id: string_value(&validation["dataset_id"], "cohort dataset ID")?.to_owned(),
        groups,
        group_ids,
        source_candidate_indices,
        action_hashes,
    })
}

fn load_intent(root: &Path, expected_groups: usize) -> Result<IntentData, AnyError> {
    let manifest = read_json::<Value>(&root.join("cache.json"))?;
    let validation = &manifest["splits"]["validation"];
    let groups = usize_value(&validation["groups"], "intent validation groups")?;
    if groups != expected_groups {
        return Err("intent and cohort group counts differ".into());
    }
    let files = &validation["files"];
    let a0 = read_f32_file(root.join(string_value(
        &files["a0_features"]["file"],
        "A0 intent file",
    )?))?;
    let a2 = read_f32_file(root.join(string_value(
        &files["a2_features"]["file"],
        "A2 intent file",
    )?))?;
    let shuffle_sources = read_u32_file(root.join(string_value(
        &files["shuffle_source_indices"]["file"],
        "shuffle source file",
    )?))?;
    let candidates = groups * COHORT_WIDTH;
    if a0.len() != candidates * INTENT_WIDTH
        || a2.len() != candidates * INTENT_WIDTH
        || shuffle_sources.len() != candidates
        || a0
            .iter()
            .chain(&a2)
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("intent tensor sizes or probabilities drifted".into());
    }
    Ok(IntentData {
        cache_id: string_value(&manifest["cache_id"], "intent ID")?.to_owned(),
        groups,
        a0,
        a2,
        shuffle_sources,
    })
}

fn read_u16_file(path: PathBuf) -> Result<Vec<u16>, AnyError> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err("u16 tensor has an odd byte count".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two bytes")))
        .collect())
}

fn read_u32_file(path: PathBuf) -> Result<Vec<u32>, AnyError> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("u32 tensor byte count is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn read_u64_file(path: PathBuf) -> Result<Vec<u64>, AnyError> {
    let bytes = fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err("u64 tensor byte count is not divisible by eight".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect())
}

fn read_f32_file(path: PathBuf) -> Result<Vec<f32>, AnyError> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("f32 tensor byte count is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn canonical_action_hash(action: &TurnAction) -> Result<[u8; 32], AnyError> {
    let mut hasher = Hasher::new();
    hasher.update(ACTION_HASH_DOMAIN);
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

fn checksum(path: &Path) -> Result<String, AnyError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn canonical_blake3(value: &Value) -> Result<String, AnyError> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, AnyError> {
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), AnyError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
    ));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn string_value<'a>(value: &'a Value, field: &str) -> Result<&'a str, AnyError> {
    value
        .as_str()
        .ok_or_else(|| format!("{field} is not a string").into())
}

fn usize_value(value: &Value, field: &str) -> Result<usize, AnyError> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| format!("{field} is not an unsigned integer"))?,
    )
    .map_err(Into::into)
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::MarketSlot;

    #[test]
    fn protocol_has_exact_sequential_halving_budget() {
        let active = [64usize, 32, 16, 8];
        let total = active
            .into_iter()
            .zip(ROOT_STAGE_ADDITIONAL_SAMPLES)
            .map(|(candidates, samples)| candidates * samples)
            .sum::<usize>();
        assert_eq!(total, TRAJECTORIES_PER_GROUP);
        assert_eq!(FrozenProtocol::expected().trajectories_per_group, 640);
        assert_eq!(
            FrozenProtocol::expected().root_chance_policy,
            "condition-on-frozen-complete-turn-staged-prelude-context"
        );
        assert_eq!(
            FrozenProtocol::expected().hidden_order_policy,
            "sort-and-redeterminize-after-frozen-root-before-opponent-rotation"
        );
    }

    #[test]
    fn frozen_complete_root_is_applied_before_future_hidden_redeterminization() {
        let mut witness = None;
        'seeds: for seed in 0..4_096 {
            let game = GameState::new(
                GameConfig::research_aaaaa(4).unwrap(),
                GameSeed::from_u64(seed),
            )
            .unwrap();
            if game.market().three_of_a_kind().is_none() {
                continue;
            }
            let prelude = MarketPrelude {
                replace_three_of_a_kind: true,
                wildlife_wipes: Vec::new(),
            };
            for action in game.legal_turn_actions(&prelude).unwrap() {
                if action.wildlife.is_none() {
                    continue;
                }
                for redetermination in 0..32 {
                    let mut perturbed = game.clone();
                    perturbed.redeterminize_hidden(GameSeed::from_u64(redetermination));
                    if perturbed.transition(&action).is_err() {
                        witness = Some((game, action));
                        break 'seeds;
                    }
                }
            }
        }
        let (game, action) =
            witness.expect("a staged root action should depend on its observed prelude draw");
        let root = RootCandidate {
            source_index: 0,
            action_hash: canonical_action_hash(&action).unwrap(),
            action,
            intent: [0.0; INTENT_WIDTH],
            r600_mean: None,
            r1200_mean: None,
            r4800_mean: None,
        };
        let afterstate = apply_frozen_root(&game, 1, &root, 0).unwrap();
        assert_eq!(afterstate.completed_turns(), game.completed_turns() + 1);
    }

    #[test]
    fn weighted_selection_is_stable_at_boundaries() {
        assert_eq!(weighted_index(&[1.0, 1.0], 0.0).unwrap(), 0);
        assert_eq!(weighted_index(&[1.0, 1.0], 0.49).unwrap(), 0);
        assert_eq!(weighted_index(&[1.0, 1.0], 0.50).unwrap(), 1);
        assert_eq!(weighted_index(&[1.0, 1.0], 0.99).unwrap(), 1);
    }

    #[test]
    fn intent_layout_uses_all_registered_next_draft_heads() {
        let mut intent = [0.0f32; INTENT_WIDTH];
        for opponent in 0..3 {
            for slot in 0..4 {
                intent[36 + opponent * 4 + slot] = 0.25;
                intent[48 + opponent * 4 + slot] = 0.25;
            }
            intent[60 + opponent] = 0.2;
            for wildlife in 0..5 {
                intent[63 + opponent * 5 + wildlife] = 0.2;
            }
        }
        let option = DraftOption {
            key: DraftKey {
                kind: 0,
                tile_slot: 2,
                wildlife_slot: 2,
            },
            action: TurnAction::paired(
                MarketSlot::TWO,
                cascadia_game::HexCoord::new(0, 0),
                cascadia_game::Rotation::ZERO,
            ),
            action_hash: [0; 32],
            heuristic_value: 0.0,
            drafted_wildlife: 4,
        };
        let expected = 0.8 * (0.25 * 0.25 * 0.2f64).powf(1.0 / 3.0);
        assert!((intent_draft_probability(&intent, 1, &option).unwrap() - expected).abs() < 1e-7);
    }

    #[test]
    fn common_random_numbers_ignore_arm_identity() {
        let hash = [7; 32];
        assert_eq!(
            opponent_uniform(41, &hash, 9, 2),
            opponent_uniform(41, &hash, 9, 2)
        );
        assert_ne!(
            opponent_uniform(41, &hash, 9, 1),
            opponent_uniform(41, &hash, 9, 2)
        );
    }
}
