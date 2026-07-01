use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_data::{
    DatasetSplit, GradedOracleDatasetManifest, GradedOracleGroup, OpportunityGraphV1,
    OpportunityMatchingSummary, PositionRecord, read_graded_oracle_shard,
    validate_graded_oracle_dataset,
};
use cascadia_game::{D6Transform, GameConfig, GameSeed, GameState};
use cascadia_search::canonical_complete_action_hash;
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

type AnyError = Box<dyn std::error::Error + Send + Sync>;

const EXPERIMENT_ID: &str = "o2-exact-opportunity-matching-v1";
const PROTOCOL_ID: &str = "o2-strict-train-top64-foundation-identifiability-v1";
const COHORT_ID: &str = "aac7a480bd3f73bf15fa09b2314c8efa80cbae01a4ce09f8cf342845c2808512";
const DATASET_MANIFEST_BLAKE3: &str =
    "7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99";
const COHORT_WIDTH: usize = 64;
const EXPECTED_GROUPS: usize = 560;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    dataset_root: PathBuf,
    #[arg(long)]
    cohort_root: PathBuf,
    #[arg(long)]
    output_root: PathBuf,
    #[arg(long)]
    source_id: String,
    #[arg(long)]
    maximum_groups: Option<usize>,
}

#[derive(Debug)]
struct Cohort {
    groups: usize,
    group_ids: Vec<u64>,
    game_indices: Vec<u64>,
    turns: Vec<u8>,
    current_players: Vec<u8>,
    source_candidate_indices: Vec<u16>,
    base_ranks: Vec<u16>,
    base_scores: Vec<f32>,
    action_hashes: Vec<[u8; 32]>,
    direct_cohort_indices: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateRow {
    cohort_index: usize,
    source_index: u16,
    action_hash: String,
    direct_rank: u16,
    direct_score: f32,
    r4800_mean: Option<f32>,
    r4800_stddev: Option<f32>,
    r4800_samples: u16,
    draft_kind: u8,
    drafted_wildlife: u8,
    graph_hash: String,
    matching_hash: String,
    demand_count: u32,
    supply_count: u32,
    edge_count: u32,
    matched_demands: u32,
    unmatched_demands: u32,
    wildlife_matches: u32,
    habitat_matches: u32,
    market_matches: u32,
    unseen_matches: u32,
    exact_completion_value: u64,
    teacher_value_micros: i64,
    matched_demand_fraction: f64,
    market_match_fraction: f64,
    wildlife_match_fraction: f64,
    mean_matched_exposure: f64,
    graph_bytes: u64,
    matching_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroupReport {
    schema_version: u16,
    experiment_id: String,
    protocol_id: String,
    cohort_row: usize,
    group_id: u64,
    game_index: u64,
    completed_turns: u16,
    personal_turn: u8,
    current_player: u8,
    nature_tokens: u8,
    public_state_hash: String,
    hidden_order_invariance_checks: usize,
    replay_construction_checks: usize,
    codec_round_trip_checks: usize,
    d6_covariance_checks: usize,
    candidate_hash_checks: usize,
    rows: Vec<CandidateRow>,
    group_result_id: String,
}

#[derive(Debug, Serialize)]
struct ExportReport {
    schema_version: u16,
    experiment_id: String,
    protocol_id: String,
    production: bool,
    host: String,
    inputs: BTreeMap<String, String>,
    groups_expected: usize,
    groups_completed: usize,
    candidate_rows: usize,
    unique_action_hashes: usize,
    hidden_order_invariance_checks: usize,
    replay_construction_checks: usize,
    codec_round_trip_checks: usize,
    d6_covariance_checks: usize,
    candidate_hash_checks: usize,
    graph_bytes: u64,
    matching_bytes: u64,
    wall_seconds: f64,
    rows_per_second: f64,
    peak_rss_bytes: u64,
    swap_delta_bytes: i64,
    zero_swap_growth: bool,
    validation_opened: bool,
    sealed_test_opened: bool,
    gameplay_run: bool,
    group_result_ids: Vec<String>,
    scientific_result_id: String,
}

fn main() -> Result<(), AnyError> {
    let args = Args::parse();
    if args.source_id.len() != 64
        || !args
            .source_id
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("O2 source-id must be one lowercase 64-hex digest".into());
    }
    let started = Instant::now();
    let swap_before = swap_used_bytes()?;
    fs::create_dir_all(args.output_root.join("groups"))?;
    fs::create_dir_all(args.output_root.join("fixtures"))?;

    let dataset_manifest_path = args.dataset_root.join("dataset.json");
    if checksum(&dataset_manifest_path)? != DATASET_MANIFEST_BLAKE3 {
        return Err("O2 graded-oracle dataset manifest digest drifted".into());
    }
    let dataset: GradedOracleDatasetManifest = read_json(&dataset_manifest_path)?;
    if dataset.split != DatasetSplit::Train
        || dataset.total_groups != EXPECTED_GROUPS
        || dataset.seeds != vec![61000, 61001, 61002, 61005, 61006, 61009, 61010]
    {
        return Err("O2 graded-oracle train domain drifted".into());
    }
    validate_graded_oracle_dataset(&args.dataset_root, &dataset)?;
    let cohort = load_cohort(&args.cohort_root)?;
    let groups_expected = args
        .maximum_groups
        .unwrap_or(cohort.groups)
        .min(cohort.groups);
    if groups_expected == 0 {
        return Err("O2 maximum-groups selected no work".into());
    }
    let active = cohort.group_ids[..groups_expected]
        .iter()
        .enumerate()
        .map(|(row, &group_id)| (group_id, row))
        .collect::<HashMap<_, _>>();
    if active.len() != groups_expected {
        return Err("O2 cohort group IDs are not unique".into());
    }

    let mut completed = Vec::with_capacity(groups_expected);
    let mut seen = HashSet::new();
    let mut fixture_written = args.output_root.join("fixtures/fixture-v1.json").is_file();
    for shard in &dataset.shards {
        let groups = read_graded_oracle_shard(&args.dataset_root, DatasetSplit::Train, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        for group in groups {
            if let Some(&cohort_row) = active.get(&group.group_id) {
                if !seen.insert(group.group_id) {
                    return Err(format!("O2 group {} replayed twice", group.group_id).into());
                }
                validate_replayed_group(&game, &group, cohort_row, &cohort)?;
                let path = args
                    .output_root
                    .join("groups")
                    .join(format!("row-{cohort_row:03}.json"));
                let report = if path.exists() {
                    let existing: GroupReport = read_json(&path)?;
                    validate_resumed_group(&existing, cohort_row, &group)?;
                    existing
                } else {
                    let report = evaluate_group(
                        &game,
                        &group,
                        cohort_row,
                        &cohort,
                        &args.output_root,
                        !fixture_written,
                    )?;
                    write_json_atomic(&path, &report)?;
                    fixture_written = true;
                    report
                };
                completed.push(report);
            }
            let champion = group.candidates[usize::from(group.champion_index)]
                .action
                .to_game_action(&game)?;
            game.apply(&champion)?;
        }
        if !game.is_game_over() {
            return Err(format!("O2 game {} did not replay fully", shard.first_game_index).into());
        }
    }
    if seen.len() != groups_expected {
        return Err(format!("O2 found {} of {groups_expected} groups", seen.len()).into());
    }
    completed.sort_by_key(|group| group.cohort_row);
    if completed
        .iter()
        .enumerate()
        .any(|(expected, group)| expected != group.cohort_row)
    {
        return Err("O2 completed group order is incomplete".into());
    }

    let rows = completed
        .iter()
        .map(|group| group.rows.len())
        .sum::<usize>();
    let unique_action_hashes = completed
        .iter()
        .flat_map(|group| group.rows.iter().map(|row| &row.action_hash))
        .collect::<HashSet<_>>()
        .len();
    if rows != groups_expected * COHORT_WIDTH || unique_action_hashes != rows {
        return Err("O2 candidate accounting or uniqueness failed".into());
    }
    let group_result_ids = completed
        .iter()
        .map(|group| group.group_result_id.clone())
        .collect::<Vec<_>>();
    let scientific_identity = serde_json::json!({
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "production": args.maximum_groups.is_none(),
        "groups_expected": groups_expected,
        "group_result_ids": group_result_ids,
    });
    let scientific_result_id = canonical_blake3(&scientific_identity)?;
    let wall_seconds = started.elapsed().as_secs_f64();
    let swap_after = swap_used_bytes()?;
    let report = ExportReport {
        schema_version: 1,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        production: args.maximum_groups.is_none(),
        host: "john1".to_owned(),
        inputs: BTreeMap::from([
            ("cohort_id".to_owned(), COHORT_ID.to_owned()),
            (
                "dataset_manifest_blake3".to_owned(),
                DATASET_MANIFEST_BLAKE3.to_owned(),
            ),
            (
                "cohort_manifest_blake3".to_owned(),
                checksum(&args.cohort_root.join("cohort.json"))?,
            ),
            ("source_id".to_owned(), args.source_id),
        ]),
        groups_expected,
        groups_completed: completed.len(),
        candidate_rows: rows,
        unique_action_hashes,
        hidden_order_invariance_checks: completed
            .iter()
            .map(|group| group.hidden_order_invariance_checks)
            .sum(),
        replay_construction_checks: completed
            .iter()
            .map(|group| group.replay_construction_checks)
            .sum(),
        codec_round_trip_checks: completed
            .iter()
            .map(|group| group.codec_round_trip_checks)
            .sum(),
        d6_covariance_checks: completed
            .iter()
            .map(|group| group.d6_covariance_checks)
            .sum(),
        candidate_hash_checks: completed
            .iter()
            .map(|group| group.candidate_hash_checks)
            .sum(),
        graph_bytes: completed
            .iter()
            .flat_map(|group| &group.rows)
            .map(|row| row.graph_bytes)
            .sum(),
        matching_bytes: completed
            .iter()
            .flat_map(|group| &group.rows)
            .map(|row| row.matching_bytes)
            .sum(),
        wall_seconds,
        rows_per_second: rows as f64 / wall_seconds,
        peak_rss_bytes: peak_rss_bytes(),
        swap_delta_bytes: swap_after as i64 - swap_before as i64,
        zero_swap_growth: swap_after <= swap_before,
        validation_opened: false,
        sealed_test_opened: false,
        gameplay_run: false,
        group_result_ids,
        scientific_result_id,
    };
    write_json_atomic(&args.output_root.join("export-report.json"), &report)?;
    Ok(())
}

fn evaluate_group(
    game: &GameState,
    group: &GradedOracleGroup,
    cohort_row: usize,
    cohort: &Cohort,
    output_root: &Path,
    write_fixture: bool,
) -> Result<GroupReport, AnyError> {
    let public = game.public_state();
    let pre_graph = OpportunityGraphV1::from_public_state(&public, game.current_player())?;
    let pre_bytes = pre_graph.canonical_bytes()?;
    let mut redeterminized = game.clone();
    redeterminized.redeterminize_hidden(GameSeed::from_u64(group.group_id ^ 0x02a2_5eed_5a5a_c3c3));
    if redeterminized.public_state().canonical_bytes() != public.canonical_bytes()
        || OpportunityGraphV1::from_public_state(
            &redeterminized.public_state(),
            game.current_player(),
        )?
        .canonical_bytes()?
            != pre_bytes
    {
        return Err(format!("O2 hidden-order invariance failed for {}", group.group_id).into());
    }

    let direct = usize::from(cohort.direct_cohort_indices[cohort_row]);
    type EvaluatedCandidate = (CandidateRow, usize, Option<(Vec<u8>, Vec<u8>)>);
    let evaluated = (0..COHORT_WIDTH)
        .into_par_iter()
        .map(|cohort_index| -> Result<EvaluatedCandidate, AnyError> {
            let flat = cohort_row * COHORT_WIDTH + cohort_index;
            let source_index = cohort.source_candidate_indices[flat];
            let candidate = group
                .candidates
                .get(usize::from(source_index))
                .ok_or("O2 source candidate is out of range")?;
            let action = candidate.action.to_game_action(game)?;
            let action_hash = canonical_complete_action_hash(&action)?;
            if action_hash != cohort.action_hashes[flat] || action_hash != candidate.action_hash {
                return Err(format!("O2 action hash drifted in group {}", group.group_id).into());
            }
            let after = game.preview_public_afterstate(&action)?;
            let graph = OpportunityGraphV1::from_public_state(&after, game.current_player())?;
            let graph_bytes = graph.canonical_bytes()?;
            if OpportunityGraphV1::from_canonical_bytes(&graph_bytes)? != graph
                || OpportunityGraphV1::from_public_state(&after, game.current_player())?
                    .canonical_bytes()?
                    != graph_bytes
            {
                return Err(format!("O2 graph replay/codec failed in {}", group.group_id).into());
            }
            let matching = graph.solve_matching()?;
            let matching_bytes = matching.canonical_bytes()?;
            if OpportunityMatchingSummary::from_canonical_bytes(&matching_bytes)? != matching {
                return Err(format!("O2 matching codec failed in {}", group.group_id).into());
            }
            let mut d6_checks = 0usize;
            if cohort_index == direct {
                for transform in D6Transform::ALL {
                    let transformed_public = after.transformed(transform)?;
                    let transformed = OpportunityGraphV1::from_public_state(
                        &transformed_public,
                        game.current_player(),
                    )?;
                    graph.verify_d6_covariance(&transformed, transform)?;
                    d6_checks += 1;
                }
            }
            let graph_byte_count = graph_bytes.len() as u64;
            let matching_byte_count = matching_bytes.len() as u64;
            let fixture =
                (write_fixture && cohort_index == direct).then_some((graph_bytes, matching_bytes));
            Ok((
                candidate_row(
                    cohort_index,
                    source_index,
                    action_hash,
                    cohort.base_ranks[flat],
                    cohort.base_scores[flat],
                    candidate,
                    &graph,
                    &matching,
                    graph_byte_count,
                    matching_byte_count,
                )?,
                d6_checks,
                fixture,
            ))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let d6_covariance_checks = evaluated.iter().map(|(_, checks, _)| *checks).sum();
    if let Some((row, _, Some((graph_bytes, matching_bytes)))) = evaluated
        .iter()
        .find(|(row, _, _)| row.cohort_index == direct)
    {
        fs::write(output_root.join("fixtures/graph-v1.bin"), graph_bytes)?;
        fs::write(output_root.join("fixtures/matching-v1.bin"), matching_bytes)?;
        let fixture = serde_json::json!({
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "group_id": group.group_id,
            "action_hash": row.action_hash,
            "graph_blake3": blake3::hash(graph_bytes).to_hex().to_string(),
            "matching_blake3": blake3::hash(matching_bytes).to_hex().to_string(),
        });
        write_json_atomic(&output_root.join("fixtures/fixture-v1.json"), &fixture)?;
    }
    let rows = evaluated
        .into_iter()
        .map(|(row, _, _)| row)
        .collect::<Vec<_>>();
    let identity = serde_json::json!({
        "cohort_row": cohort_row,
        "group_id": group.group_id,
        "rows": rows,
        "hidden_order_invariance_checks": 1,
        "replay_construction_checks": COHORT_WIDTH,
        "codec_round_trip_checks": COHORT_WIDTH * 2,
        "d6_covariance_checks": d6_covariance_checks,
        "candidate_hash_checks": COHORT_WIDTH,
    });
    Ok(GroupReport {
        schema_version: 1,
        experiment_id: EXPERIMENT_ID.to_owned(),
        protocol_id: PROTOCOL_ID.to_owned(),
        cohort_row,
        group_id: group.group_id,
        game_index: group.raw_seed,
        completed_turns: group.completed_turns,
        personal_turn: group.personal_turn,
        current_player: group.current_player,
        nature_tokens: game.boards()[game.current_player()].nature_tokens(),
        public_state_hash: hex(&group.public_state_hash),
        hidden_order_invariance_checks: 1,
        replay_construction_checks: COHORT_WIDTH,
        codec_round_trip_checks: COHORT_WIDTH * 2,
        d6_covariance_checks,
        candidate_hash_checks: COHORT_WIDTH,
        rows,
        group_result_id: canonical_blake3(&identity)?,
    })
}

fn candidate_row(
    cohort_index: usize,
    source_index: u16,
    action_hash: [u8; 32],
    direct_rank: u16,
    direct_score: f32,
    candidate: &cascadia_data::GradedOracleCandidate,
    graph: &OpportunityGraphV1,
    matching: &OpportunityMatchingSummary,
    graph_bytes: u64,
    matching_bytes: u64,
) -> Result<CandidateRow, AnyError> {
    let supply_by_id = graph
        .supplies
        .iter()
        .map(|supply| (supply.id, supply))
        .collect::<BTreeMap<_, _>>();
    let mean_matched_exposure = if matching.assignments.is_empty() {
        0.0
    } else {
        matching
            .assignments
            .iter()
            .map(|assignment| {
                let supply = supply_by_id[&assignment.supply];
                f64::from(1 + supply.access_delay_turns + supply.opponents_before_access)
            })
            .sum::<f64>()
            / matching.assignments.len() as f64
    };
    Ok(CandidateRow {
        cohort_index,
        source_index,
        action_hash: hex(&action_hash),
        direct_rank,
        direct_score,
        r4800_mean: (candidate.r4800.samples > 0).then_some(candidate.r4800.mean),
        r4800_stddev: (candidate.r4800.samples > 0).then_some(candidate.r4800.stddev),
        r4800_samples: candidate.r4800.samples,
        draft_kind: candidate.action.draft_kind,
        drafted_wildlife: candidate.action.drafted_wildlife,
        graph_hash: graph.canonical_hash()?.to_hex().to_string(),
        matching_hash: matching.canonical_hash()?.to_hex().to_string(),
        demand_count: matching.demand_count,
        supply_count: matching.supply_count,
        edge_count: matching.edge_count,
        matched_demands: matching.matched_demands,
        unmatched_demands: matching.unmatched_demands,
        wildlife_matches: matching.wildlife_matches,
        habitat_matches: matching.habitat_matches,
        market_matches: matching.market_matches,
        unseen_matches: matching.unseen_matches,
        exact_completion_value: matching.exact_completion_value,
        teacher_value_micros: matching.teacher_value_micros,
        matched_demand_fraction: ratio(matching.matched_demands, matching.demand_count),
        market_match_fraction: ratio(matching.market_matches, matching.matched_demands),
        wildlife_match_fraction: ratio(matching.wildlife_matches, matching.matched_demands),
        mean_matched_exposure,
        graph_bytes,
        matching_bytes,
    })
}

fn validate_replayed_group(
    game: &GameState,
    group: &GradedOracleGroup,
    cohort_row: usize,
    cohort: &Cohort,
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
        return Err(format!("O2 replay drifted at group {}", group.group_id).into());
    }
    Ok(())
}

fn validate_resumed_group(
    report: &GroupReport,
    cohort_row: usize,
    group: &GradedOracleGroup,
) -> Result<(), AnyError> {
    let identity = serde_json::json!({
        "cohort_row": report.cohort_row,
        "group_id": report.group_id,
        "rows": report.rows,
        "hidden_order_invariance_checks": report.hidden_order_invariance_checks,
        "replay_construction_checks": report.replay_construction_checks,
        "codec_round_trip_checks": report.codec_round_trip_checks,
        "d6_covariance_checks": report.d6_covariance_checks,
        "candidate_hash_checks": report.candidate_hash_checks,
    });
    if report.schema_version != 1
        || report.experiment_id != EXPERIMENT_ID
        || report.protocol_id != PROTOCOL_ID
        || report.cohort_row != cohort_row
        || report.group_id != group.group_id
        || report.rows.len() != COHORT_WIDTH
        || report.hidden_order_invariance_checks != 1
        || report.replay_construction_checks != COHORT_WIDTH
        || report.codec_round_trip_checks != COHORT_WIDTH * 2
        || report.d6_covariance_checks != D6Transform::ALL.len()
        || report.candidate_hash_checks != COHORT_WIDTH
        || report.group_result_id != canonical_blake3(&identity)?
    {
        return Err("resumed O2 group report violates the frozen contract".into());
    }
    Ok(())
}

fn load_cohort(root: &Path) -> Result<Cohort, AnyError> {
    let manifest: Value = read_json(&root.join("cohort.json"))?;
    let groups = usize_value(&manifest["groups"], "cohort groups")?;
    if manifest["cohort_id"].as_str() != Some(COHORT_ID)
        || manifest["protocol_id"].as_str() != Some("t1-strict-train-top64-cohort-v1")
        || manifest["cohort_schema"].as_str() != Some("t1-strict-exact-r2-top64-cohort-v1")
        || manifest["complete_train_corpus"].as_bool() != Some(true)
        || groups != EXPECTED_GROUPS
    {
        return Err("O2 cohort manifest violates the frozen contract".into());
    }
    let files = manifest["files"]
        .as_object()
        .ok_or("cohort files missing")?;
    for (name, spec) in files {
        verify_tensor_file(root, name, spec)?;
    }
    let group_ids = read_u64_tensor(root, files, "group_ids")?;
    let game_indices = read_u64_tensor(root, files, "game_indices")?;
    let turns = read_tensor_bytes(root, files, "turns")?;
    let current_players = read_tensor_bytes(root, files, "current_players")?;
    let source_candidate_indices = read_u16_tensor(root, files, "source_candidate_indices")?;
    let base_ranks = read_u16_tensor(root, files, "base_ranks")?;
    let base_scores = read_f32_tensor(root, files, "base_scores")?;
    let direct_cohort_indices = read_u16_tensor(root, files, "direct_cohort_indices")?;
    let action_bytes = read_tensor_bytes(root, files, "action_hashes")?;
    if group_ids.len() != groups
        || game_indices.len() != groups
        || turns.len() != groups
        || current_players.len() != groups
        || source_candidate_indices.len() != groups * COHORT_WIDTH
        || base_ranks.len() != groups * COHORT_WIDTH
        || base_scores.len() != groups * COHORT_WIDTH
        || direct_cohort_indices.len() != groups
        || action_bytes.len() != groups * COHORT_WIDTH * 32
    {
        return Err("O2 cohort tensor dimensions drifted".into());
    }
    let action_hashes = action_bytes
        .chunks_exact(32)
        .map(|bytes| bytes.try_into().expect("exact action hash width"))
        .collect();
    Ok(Cohort {
        groups,
        group_ids,
        game_indices,
        turns,
        current_players,
        source_candidate_indices,
        base_ranks,
        base_scores,
        action_hashes,
        direct_cohort_indices,
    })
}

fn verify_tensor_file(root: &Path, name: &str, spec: &Value) -> Result<(), AnyError> {
    let file = spec["file"].as_str().ok_or("tensor filename missing")?;
    let relative = Path::new(file);
    if relative.components().count() != 1 {
        return Err(format!("O2 tensor {name} escapes its root").into());
    }
    let path = root.join(relative);
    let expected_bytes = spec["bytes"].as_u64().ok_or("tensor size missing")?;
    let expected_blake3 = spec["blake3"].as_str().ok_or("tensor digest missing")?;
    if !path.is_file()
        || path.metadata()?.len() != expected_bytes
        || checksum(&path)? != expected_blake3
    {
        return Err(format!("O2 tensor {name} failed integrity").into());
    }
    Ok(())
}

fn tensor_path(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<PathBuf, AnyError> {
    Ok(root.join(files[name]["file"].as_str().ok_or("tensor path missing")?))
}

fn read_tensor_bytes(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u8>, AnyError> {
    Ok(fs::read(tensor_path(root, files, name)?)?)
}

fn read_u16_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u16>, AnyError> {
    let bytes = read_tensor_bytes(root, files, name)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("O2 tensor {name} has odd byte length").into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn read_u64_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<u64>, AnyError> {
    let bytes = read_tensor_bytes(root, files, name)?;
    if bytes.len() % 8 != 0 {
        return Err(format!("O2 tensor {name} has invalid u64 length").into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn read_f32_tensor(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Vec<f32>, AnyError> {
    let bytes = read_tensor_bytes(root, files, name)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("O2 tensor {name} has invalid f32 length").into());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("O2 tensor {name} contains non-finite values").into());
    }
    Ok(values)
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        f64::from(numerator) / f64::from(denominator)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, AnyError> {
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), AnyError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn checksum(path: &Path) -> Result<String, AnyError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
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

fn usize_value(value: &Value, label: &str) -> Result<usize, AnyError> {
    usize::try_from(value.as_u64().ok_or_else(|| format!("{label} missing"))?).map_err(Into::into)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the provided rusage struct for RUSAGE_SELF.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    // SAFETY: a zero return from getrusage guarantees initialization.
    let usage = unsafe { usage.assume_init() };
    u64::try_from(usage.ru_maxrss).unwrap_or(0)
}

fn swap_used_bytes() -> Result<u64, AnyError> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "vm.swapusage"])
        .output()?;
    if !output.status.success() {
        return Err("sysctl vm.swapusage failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let used = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(3)
        .find_map(|parts| (parts[0] == "used" && parts[1] == "=").then_some(parts[2]))
        .ok_or("vm.swapusage omitted used bytes")?;
    let mebibytes = used.trim_end_matches('M').parse::<f64>()?;
    Ok((mebibytes * 1024.0 * 1024.0).round() as u64)
}
