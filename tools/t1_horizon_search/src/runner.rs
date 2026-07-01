use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::Path,
    time::Instant,
};

use cascadia_ai::nnue::{BagInfo, extract_features_with_bag};
use cascadia_data::{
    DatasetSplit, GradedOracleCandidate, GradedOracleDatasetManifest, GradedOracleGroup,
    PositionRecord, read_graded_oracle_shard, validate_graded_oracle_dataset,
};
use cascadia_differential::legacy_teacher::translate_public_state_allowing_legacy_elk_undercount;
use cascadia_game::{GameConfig, GameSeed, GameState, TurnAction, score_board};
use cascadia_model::{DEFAULT_SPARSE_NNUE_SHARED_MEMORY_BYTES, LEGACY_NNUE_FEATURES, ModelProcess};
use cascadia_search::{
    HalvingRootStatistics, HorizonPatternPolicyConfig, PublicBeliefTrajectory, SequentialHalving,
    canonical_complete_action_hash, simulate_pattern_prior_horizon,
    simulate_pattern_prior_post_root_horizon,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::contract::{
    AnyError, Args, Authorization, COHORT_WIDTH, CohortData, EXPECTED_GROUPS, EXPERIMENT_ID,
    FrozenProtocol, HorizonArm, MODEL_BATCH_ROWS, PROTOCOL_ID, SEARCH_TRAJECTORIES_PER_GROUP,
    canonical_blake3, checksum, load_cohort, read_json, read_json_value, search_schedule,
    validate_authorization, write_json_atomic,
};

#[derive(Debug, Clone)]
struct RootCandidate {
    source_index: u16,
    action_hash: [u8; 32],
    action: TurnAction,
    direct_rank: u16,
    direct_score: f32,
    r600_mean: Option<f32>,
    r1200_mean: Option<f32>,
    r4800_mean: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawLeaf {
    actual_score: u16,
    sparse_features: Option<Vec<u16>>,
    trace_hash: [u8; 32],
    public_leaf_hash: [u8; 32],
    opponent_action_hashes: Vec<[u8; 32]>,
    opponent_decisions: usize,
    opponent_options: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateResult {
    cohort_index: usize,
    source_index: u16,
    action_hash: String,
    direct_rank: u16,
    direct_score: f32,
    search_mean: f64,
    search_stddev: f64,
    samples: usize,
    eliminated_stage: Option<usize>,
    r600_mean: Option<f32>,
    r1200_mean: Option<f32>,
    r4800_mean: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct GroupDiagnostics {
    root_candidates: usize,
    trajectories: usize,
    leaf_model_rows: usize,
    terminal_leaves: usize,
    opponent_decisions: usize,
    opponent_options: usize,
    hidden_order_invariance_checks: usize,
    prefix_coupling_checks: usize,
    candidate_hash_checks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupResult {
    cohort_row: usize,
    group_id: u64,
    game_index: u64,
    completed_turns: u16,
    current_player: u8,
    public_state_hash: String,
    arm: String,
    opponent_turns: usize,
    selected_cohort_index: usize,
    selected_source_index: u16,
    selected_action_hash: String,
    selected_search_mean: f64,
    selected_search_stddev: f64,
    direct_cohort_index: usize,
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
    opponent_turns: usize,
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
    prefix_coupling_checks: usize,
    candidate_hash_checks: usize,
    wall_seconds: f64,
    groups: Vec<GroupResult>,
    scientific_result_id: String,
    report_id: String,
}

struct LeafEvaluator {
    process: ModelProcess,
}

pub fn run(args: Args) -> Result<(), AnyError> {
    let started = Instant::now();
    let authorization = read_json::<Authorization>(&args.authorization)?;
    validate_authorization(&authorization, &args)?;
    let arm = HorizonArm::parse(
        authorization
            .roles
            .get(&args.role)
            .ok_or("authorization omitted the requested T1 role")?,
    )?;
    let cohort = load_cohort(&args.cohort_root)?;
    let dataset_manifest =
        read_json::<GradedOracleDatasetManifest>(&args.dataset_root.join("dataset.json"))?;
    validate_graded_oracle_dataset(&args.dataset_root, &dataset_manifest)?;
    validate_inputs(&authorization, &args, &cohort, &dataset_manifest)?;

    let production = args.maximum_groups.is_none();
    let groups_expected = args
        .maximum_groups
        .map_or(cohort.groups, |limit| limit.min(cohort.groups));
    if groups_expected == 0 {
        return Err("maximum-groups selected no T1 work".into());
    }
    let active_by_group = cohort.group_ids[..groups_expected]
        .iter()
        .enumerate()
        .map(|(row, &group_id)| (group_id, row))
        .collect::<HashMap<_, _>>();
    if active_by_group.len() != groups_expected {
        return Err("T1 cohort repeats a group ID".into());
    }

    fs::create_dir_all(args.run_dir.join("groups"))?;
    let run_identity = build_run_identity(
        &authorization,
        &args,
        arm,
        production,
        &dataset_manifest,
        &cohort,
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
        return Err("T1 leaf-model warmup failed".into());
    }

    let mut completed = Vec::with_capacity(groups_expected);
    let mut seen_groups = HashSet::new();
    for shard in &dataset_manifest.shards {
        let groups = read_graded_oracle_shard(&args.dataset_root, DatasetSplit::Train, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        for group in groups {
            if let Some(&cohort_row) = active_by_group.get(&group.group_id) {
                if !seen_groups.insert(group.group_id) {
                    return Err(format!("T1 group {} replayed twice", group.group_id).into());
                }
                validate_replayed_group(&game, &group, cohort_row, &cohort)?;
                let group_path = args
                    .run_dir
                    .join("groups")
                    .join(format!("row-{cohort_row:03}.json"));
                let result = if group_path.exists() {
                    let existing = read_json::<GroupResult>(&group_path)?;
                    validate_resumed_group(&existing, cohort_row, &group, arm)?;
                    existing
                } else {
                    let result =
                        evaluate_group(&game, &group, cohort_row, arm, &cohort, &mut evaluator)?;
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
                "T1 train game {} did not replay to completion",
                shard.first_game_index
            )
            .into());
        }
    }
    evaluator.process.shutdown()?;

    if seen_groups.len() != groups_expected {
        return Err(format!(
            "T1 replay found {} of {groups_expected} groups",
            seen_groups.len()
        )
        .into());
    }
    completed.sort_by_key(|group| group.cohort_row);
    for (expected, group) in completed.iter().enumerate() {
        if group.cohort_row != expected {
            return Err("T1 completed group order is incomplete".into());
        }
    }
    let totals = aggregate_diagnostics(&completed);
    let scientific_identity = json!({
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "authorization_id": authorization.authorization_id,
        "bundle_id": authorization.bundle_id,
        "arm": arm.as_str(),
        "opponent_turns": arm.opponent_turns(),
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
        opponent_turns: arm.opponent_turns(),
        host: args.host,
        production,
        inputs: run_identity.inputs,
        protocol: run_identity.protocol,
        groups_expected,
        groups_completed: completed.len(),
        root_candidates: totals.root_candidates,
        trajectories: totals.trajectories,
        leaf_model_rows: totals.leaf_model_rows,
        terminal_leaves: totals.terminal_leaves,
        opponent_decisions: totals.opponent_decisions,
        opponent_options: totals.opponent_options,
        hidden_order_invariance_checks: totals.hidden_order_invariance_checks,
        prefix_coupling_checks: totals.prefix_coupling_checks,
        candidate_hash_checks: totals.candidate_hash_checks,
        wall_seconds: started.elapsed().as_secs_f64(),
        groups: completed,
        scientific_result_id,
        report_id,
    };
    write_json_atomic(&args.output, &report)?;
    Ok(())
}

fn evaluate_group(
    game: &GameState,
    group: &GradedOracleGroup,
    cohort_row: usize,
    arm: HorizonArm,
    cohort: &CohortData,
    evaluator: &mut LeafEvaluator,
) -> Result<GroupResult, AnyError> {
    let roots = reconstruct_roots(game, group, cohort_row, cohort)?;
    let (hidden_order_invariance_checks, prefix_coupling_checks) =
        verify_group_probes(game, group.group_id, &roots[0], arm)?;
    let (statistics, selected, mut diagnostics) = if arm == HorizonArm::H0RootLeaf {
        evaluate_h0(game, group.group_id, &roots, evaluator)?
    } else {
        evaluate_searched_horizon(game, group.group_id, &roots, arm, evaluator)?
    };
    diagnostics.hidden_order_invariance_checks = hidden_order_invariance_checks;
    diagnostics.prefix_coupling_checks = prefix_coupling_checks;
    diagnostics.candidate_hash_checks = roots.len();
    let candidates = roots
        .iter()
        .zip(&statistics)
        .enumerate()
        .map(|(cohort_index, (root, stats))| CandidateResult {
            cohort_index,
            source_index: root.source_index,
            action_hash: hex(&root.action_hash),
            direct_rank: root.direct_rank,
            direct_score: root.direct_score,
            search_mean: stats.mean,
            search_stddev: stats.standard_deviation,
            samples: stats.samples,
            eliminated_stage: stats.eliminated_stage,
            r600_mean: root.r600_mean,
            r1200_mean: root.r1200_mean,
            r4800_mean: root.r4800_mean,
        })
        .collect::<Vec<_>>();
    let direct = usize::from(cohort.direct_cohort_indices[cohort_row]);
    let identity = json!({
        "cohort_row": cohort_row,
        "group_id": group.group_id,
        "arm": arm.as_str(),
        "opponent_turns": arm.opponent_turns(),
        "selected_cohort_index": selected,
        "direct_cohort_index": direct,
        "candidates": candidates,
        "diagnostics": diagnostics,
    });
    let group_result_id = canonical_blake3(&identity)?;
    Ok(GroupResult {
        cohort_row,
        group_id: group.group_id,
        game_index: group.raw_seed,
        completed_turns: group.completed_turns,
        current_player: group.current_player,
        public_state_hash: hex(&group.public_state_hash),
        arm: arm.as_str().to_owned(),
        opponent_turns: arm.opponent_turns(),
        selected_cohort_index: selected,
        selected_source_index: roots[selected].source_index,
        selected_action_hash: hex(&roots[selected].action_hash),
        selected_search_mean: statistics[selected].mean,
        selected_search_stddev: statistics[selected].standard_deviation,
        direct_cohort_index: direct,
        candidates,
        diagnostics,
        group_result_id,
    })
}

fn evaluate_h0(
    game: &GameState,
    group_id: u64,
    roots: &[RootCandidate],
    evaluator: &mut LeafEvaluator,
) -> Result<(Vec<HalvingRootStatistics>, usize, GroupDiagnostics), AnyError> {
    let generated = roots
        .par_iter()
        .map(|root| {
            simulate_pattern_prior_horizon(
                game,
                group_id,
                &root.action,
                &root.action_hash,
                0,
                0,
                HorizonPatternPolicyConfig::default(),
            )
            .map_err(Into::into)
            .and_then(prepare_leaf)
        })
        .collect::<Vec<Result<RawLeaf, AnyError>>>();
    let leaves = collect_results(generated)?;
    let (values, leaf_model_rows, terminal_leaves, opponent_decisions, opponent_options) =
        score_leaves(&leaves, evaluator)?;
    let selected = select_best(&values, roots)?;
    let statistics = values
        .into_iter()
        .map(|mean| HalvingRootStatistics {
            samples: 1,
            mean,
            standard_deviation: 0.0,
            eliminated_stage: None,
        })
        .collect();
    Ok((
        statistics,
        selected,
        GroupDiagnostics {
            root_candidates: roots.len(),
            trajectories: roots.len(),
            leaf_model_rows,
            terminal_leaves,
            opponent_decisions,
            opponent_options,
            ..GroupDiagnostics::default()
        },
    ))
}

fn evaluate_searched_horizon(
    game: &GameState,
    group_id: u64,
    roots: &[RootCandidate],
    arm: HorizonArm,
    evaluator: &mut LeafEvaluator,
) -> Result<(Vec<HalvingRootStatistics>, usize, GroupDiagnostics), AnyError> {
    let mut halving = SequentialHalving::new(
        roots.iter().map(|root| root.action_hash).collect(),
        search_schedule(),
    )?;
    let mut diagnostics = GroupDiagnostics {
        root_candidates: roots.len(),
        ..GroupDiagnostics::default()
    };
    while !halving.is_complete() {
        let work = halving.work()?;
        let generated = work
            .par_iter()
            .map(|item| {
                let root = &roots[item.root_index];
                simulate_pattern_prior_horizon(
                    game,
                    group_id,
                    &root.action,
                    &root.action_hash,
                    item.sample_index,
                    arm.opponent_turns(),
                    HorizonPatternPolicyConfig::default(),
                )
                .map_err(Into::into)
                .and_then(prepare_leaf)
            })
            .collect::<Vec<Result<RawLeaf, AnyError>>>();
        let leaves = collect_results(generated)?;
        let (values, leaf_rows, terminals, decisions, options) = score_leaves(&leaves, evaluator)?;
        diagnostics.trajectories += values.len();
        diagnostics.leaf_model_rows += leaf_rows;
        diagnostics.terminal_leaves += terminals;
        diagnostics.opponent_decisions += decisions;
        diagnostics.opponent_options += options;
        halving.complete_stage(&values)?;
    }
    let result = halving.finish()?;
    if result.total_evaluations != SEARCH_TRAJECTORIES_PER_GROUP {
        return Err(format!(
            "T1 group {group_id} used {} trajectories instead of {}",
            result.total_evaluations, SEARCH_TRAJECTORIES_PER_GROUP
        )
        .into());
    }
    Ok((result.roots, result.selected_root, diagnostics))
}

fn verify_group_probes(
    game: &GameState,
    group_id: u64,
    root: &RootCandidate,
    arm: HorizonArm,
) -> Result<(usize, usize), AnyError> {
    let focal_player = game.current_player();
    let afterstate = game.transition(&root.action)?;
    let left = simulate_pattern_prior_post_root_horizon(
        afterstate.clone(),
        focal_player,
        group_id,
        &root.action_hash,
        0,
        arm.opponent_turns(),
        HorizonPatternPolicyConfig::default(),
    )?;
    let mut perturbed = afterstate;
    perturbed.redeterminize_hidden(GameSeed::from_u64(group_id ^ 0xa5a5_5a5a));
    let right = simulate_pattern_prior_post_root_horizon(
        perturbed,
        focal_player,
        group_id,
        &root.action_hash,
        0,
        arm.opponent_turns(),
        HorizonPatternPolicyConfig::default(),
    )?;
    if prepare_leaf(left)? != prepare_leaf(right)? {
        return Err(format!("T1 hidden-order invariance failed for group {group_id}").into());
    }

    let h1 = simulate_pattern_prior_post_root_horizon(
        game.transition(&root.action)?,
        focal_player,
        group_id,
        &root.action_hash,
        1,
        1,
        HorizonPatternPolicyConfig::default(),
    )?;
    let h2 = simulate_pattern_prior_post_root_horizon(
        game.transition(&root.action)?,
        focal_player,
        group_id,
        &root.action_hash,
        1,
        2,
        HorizonPatternPolicyConfig::default(),
    )?;
    let h3 = simulate_pattern_prior_post_root_horizon(
        game.transition(&root.action)?,
        focal_player,
        group_id,
        &root.action_hash,
        1,
        3,
        HorizonPatternPolicyConfig::default(),
    )?;
    if h1.opponent_action_hashes != h2.opponent_action_hashes[..h1.opponent_decisions]
        || h1.opponent_action_hashes != h3.opponent_action_hashes[..h1.opponent_decisions]
        || h2.opponent_action_hashes != h3.opponent_action_hashes[..h2.opponent_decisions]
    {
        return Err(format!("T1 prefix coupling failed for group {group_id}").into());
    }
    Ok((1, 1))
}

fn prepare_leaf(trajectory: PublicBeliefTrajectory) -> Result<RawLeaf, AnyError> {
    let score = score_board(
        &trajectory.state.boards()[trajectory.focal_player],
        trajectory.state.config().scoring_cards,
    )
    .base_total;
    let sparse_features = if trajectory.state.is_game_over() {
        None
    } else {
        let translated = translate_public_state_allowing_legacy_elk_undercount(
            &trajectory.state.public_state(),
        )?;
        let bag = BagInfo::from_game_for_player(&translated.game, trajectory.focal_player);
        let features =
            extract_features_with_bag(&translated.game.boards[trajectory.focal_player], Some(&bag));
        if features
            .iter()
            .any(|&feature| usize::from(feature) >= LEGACY_NNUE_FEATURES)
        {
            return Err("T1 leaf sparse feature crossed the qualified model width".into());
        }
        Some(features)
    };
    Ok(RawLeaf {
        actual_score: score,
        sparse_features,
        trace_hash: trajectory.trace_hash,
        public_leaf_hash: trajectory.public_leaf_hash,
        opponent_action_hashes: trajectory.opponent_action_hashes,
        opponent_decisions: trajectory.opponent_decisions,
        opponent_options: trajectory.opponent_options,
    })
}

fn score_leaves(
    leaves: &[RawLeaf],
    evaluator: &mut LeafEvaluator,
) -> Result<(Vec<f64>, usize, usize, usize, usize), AnyError> {
    let mut feature_rows = Vec::new();
    let mut feature_leaf_indices = Vec::new();
    let mut terminal_leaves = 0usize;
    let mut opponent_decisions = 0usize;
    let mut opponent_options = 0usize;
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        opponent_decisions += leaf.opponent_decisions;
        opponent_options += leaf.opponent_options;
        if let Some(features) = &leaf.sparse_features {
            feature_rows.push(features.clone());
            feature_leaf_indices.push(leaf_index);
        } else {
            terminal_leaves += 1;
        }
    }
    let mut predictions = Vec::with_capacity(feature_rows.len());
    for chunk in feature_rows.chunks(MODEL_BATCH_ROWS) {
        predictions.extend(evaluator.process.predict_sparse_nnue_csr_exact(chunk)?);
    }
    if predictions.len() != feature_rows.len() {
        return Err("T1 leaf evaluator returned the wrong row count".into());
    }
    let mut values = leaves
        .iter()
        .map(|leaf| f64::from(leaf.actual_score))
        .collect::<Vec<_>>();
    for (&leaf_index, &remaining) in feature_leaf_indices.iter().zip(&predictions) {
        values[leaf_index] += f64::from(remaining);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("T1 trajectory produced a non-finite leaf value".into());
    }
    Ok((
        values,
        predictions.len(),
        terminal_leaves,
        opponent_decisions,
        opponent_options,
    ))
}

fn collect_results(values: Vec<Result<RawLeaf, AnyError>>) -> Result<Vec<RawLeaf>, AnyError> {
    values.into_iter().collect()
}

fn select_best(values: &[f64], roots: &[RootCandidate]) -> Result<usize, AnyError> {
    if values.len() != roots.len() || values.is_empty() {
        return Err("T1 root values do not align with candidates".into());
    }
    Ok((0..values.len())
        .min_by(|&left, &right| {
            values[right]
                .total_cmp(&values[left])
                .then_with(|| roots[left].action_hash.cmp(&roots[right].action_hash))
        })
        .expect("nonempty roots were checked"))
}

fn reconstruct_roots(
    game: &GameState,
    group: &GradedOracleGroup,
    cohort_row: usize,
    cohort: &CohortData,
) -> Result<Vec<RootCandidate>, AnyError> {
    if cohort_row >= cohort.groups || cohort.group_ids[cohort_row] != group.group_id {
        return Err("T1 cohort row points at a different group".into());
    }
    let mut roots = Vec::with_capacity(COHORT_WIDTH);
    for cohort_index in 0..COHORT_WIDTH {
        let flat = cohort_row * COHORT_WIDTH + cohort_index;
        let source_index = cohort.source_candidate_indices[flat];
        let candidate = group
            .candidates
            .get(usize::from(source_index))
            .ok_or("T1 cohort source candidate is out of range")?;
        let action = candidate.action.to_game_action(game)?;
        let observed_hash = canonical_complete_action_hash(&action)?;
        if observed_hash != cohort.action_hashes[flat] || observed_hash != candidate.action_hash {
            return Err(format!(
                "T1 root action hash drifted for group {} source {}",
                group.group_id, source_index
            )
            .into());
        }
        roots.push(RootCandidate {
            source_index,
            action_hash: observed_hash,
            action,
            direct_rank: cohort.base_ranks[flat],
            direct_score: cohort.base_scores[flat],
            r600_mean: estimate(candidate, candidate.r600.mean, candidate.r600.samples),
            r1200_mean: estimate(candidate, candidate.r1200.mean, candidate.r1200.samples),
            r4800_mean: estimate(candidate, candidate.r4800.mean, candidate.r4800.samples),
        });
    }
    Ok(roots)
}

fn estimate(_candidate: &GradedOracleCandidate, value: f32, samples: u16) -> Option<f32> {
    (samples > 0).then_some(value)
}

fn validate_replayed_group(
    game: &GameState,
    group: &GradedOracleGroup,
    cohort_row: usize,
    cohort: &CohortData,
) -> Result<(), AnyError> {
    if game.completed_turns() != group.completed_turns
        || game.current_player() != usize::from(group.current_player)
        || PositionRecord::observe(game, group.raw_seed).to_bytes() != group.position.to_bytes()
        || *game.public_state().canonical_hash().as_bytes() != group.public_state_hash
        || cohort.group_ids[cohort_row] != group.group_id
        || cohort.game_indices[cohort_row] != group.raw_seed
        || u16::from(cohort.turns[cohort_row]) != group.completed_turns
        || cohort.current_players[cohort_row] != group.current_player
    {
        return Err(format!("T1 train replay drifted at group {}", group.group_id).into());
    }
    Ok(())
}

fn validate_resumed_group(
    result: &GroupResult,
    cohort_row: usize,
    group: &GradedOracleGroup,
    arm: HorizonArm,
) -> Result<(), AnyError> {
    if result.cohort_row != cohort_row
        || result.group_id != group.group_id
        || result.arm != arm.as_str()
        || result.opponent_turns != arm.opponent_turns()
        || result.candidates.len() != COHORT_WIDTH
        || result.diagnostics.trajectories != arm.expected_evaluations()
        || result.diagnostics.candidate_hash_checks != COHORT_WIDTH
        || result.diagnostics.hidden_order_invariance_checks != 1
        || result.diagnostics.prefix_coupling_checks != 1
    {
        return Err("resumed T1 group artifact does not match this run".into());
    }
    let identity = json!({
        "cohort_row": result.cohort_row,
        "group_id": result.group_id,
        "arm": result.arm,
        "opponent_turns": result.opponent_turns,
        "selected_cohort_index": result.selected_cohort_index,
        "direct_cohort_index": result.direct_cohort_index,
        "candidates": result.candidates,
        "diagnostics": result.diagnostics,
    });
    if result.group_result_id != canonical_blake3(&identity)? {
        return Err("resumed T1 group scientific identity drifted".into());
    }
    Ok(())
}

fn aggregate_diagnostics(groups: &[GroupResult]) -> GroupDiagnostics {
    groups
        .iter()
        .fold(GroupDiagnostics::default(), |mut total, group| {
            total.root_candidates += group.diagnostics.root_candidates;
            total.trajectories += group.diagnostics.trajectories;
            total.leaf_model_rows += group.diagnostics.leaf_model_rows;
            total.terminal_leaves += group.diagnostics.terminal_leaves;
            total.opponent_decisions += group.diagnostics.opponent_decisions;
            total.opponent_options += group.diagnostics.opponent_options;
            total.hidden_order_invariance_checks +=
                group.diagnostics.hidden_order_invariance_checks;
            total.prefix_coupling_checks += group.diagnostics.prefix_coupling_checks;
            total.candidate_hash_checks += group.diagnostics.candidate_hash_checks;
            total
        })
}

fn validate_inputs(
    authorization: &Authorization,
    args: &Args,
    cohort: &CohortData,
    dataset: &GradedOracleDatasetManifest,
) -> Result<(), AnyError> {
    let inputs = &authorization.inputs;
    let model = read_json_value(&args.model_dir.join("model.json"))?;
    if dataset.split != DatasetSplit::Train
        || dataset.total_groups != EXPECTED_GROUPS
        || dataset.dataset_id != inputs.dataset_id
        || dataset.dataset_id != cohort.dataset_id
        || checksum(&args.dataset_root.join("dataset.json"))? != inputs.dataset_manifest_blake3
        || cohort.cohort_id != inputs.cohort_id
        || checksum(&args.cohort_root.join("cohort.json"))? != inputs.cohort_manifest_blake3
        || checksum(&args.model_dir.join("model.json"))? != inputs.model_manifest_blake3
        || checksum(&args.model_dir.join("model.safetensors"))? != inputs.model_safetensors_blake3
        || model["architecture"].as_str() != Some("legacy-sparse-nnue-v4opp-mlx-v1")
        || model["dimensions"]["features"].as_u64() != Some(11_231)
    {
        return Err("T1 runtime input bytes differ from the frozen authorization".into());
    }
    Ok(())
}

fn build_run_identity(
    authorization: &Authorization,
    args: &Args,
    arm: HorizonArm,
    production: bool,
    dataset: &GradedOracleDatasetManifest,
    cohort: &CohortData,
) -> Result<RunIdentity, AnyError> {
    let mut inputs = BTreeMap::new();
    inputs.insert("dataset_id".to_owned(), dataset.dataset_id.clone());
    inputs.insert("cohort_id".to_owned(), cohort.cohort_id.clone());
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
            return Err("existing T1 run identity differs from this invocation".into());
        }
    } else {
        write_json_atomic(path, expected)?;
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_budget_and_horizons_are_exact() {
        let protocol = FrozenProtocol::expected();
        assert_eq!(
            search_schedule().total_evaluations(COHORT_WIDTH).unwrap(),
            SEARCH_TRAJECTORIES_PER_GROUP
        );
        assert_eq!(
            protocol,
            serde_json::from_value(serde_json::to_value(&protocol).unwrap()).unwrap()
        );
        assert_eq!(HorizonArm::H0RootLeaf.expected_evaluations(), 64);
        assert_eq!(HorizonArm::H3FullRotation.opponent_turns(), 3);
    }

    #[test]
    fn best_selection_uses_action_hash_ties() {
        let root = |hash: u8| RootCandidate {
            source_index: 0,
            action_hash: [hash; 32],
            action: TurnAction::paired(
                cascadia_game::MarketSlot::ZERO,
                cascadia_game::HexCoord::new(0, 0),
                cascadia_game::Rotation::ZERO,
            ),
            direct_rank: 0,
            direct_score: 0.0,
            r600_mean: None,
            r1200_mean: None,
            r4800_mean: None,
        };
        assert_eq!(
            select_best(&[4.0, 4.0, 3.0], &[root(2), root(1), root(0)]).unwrap(),
            1
        );
    }
}
