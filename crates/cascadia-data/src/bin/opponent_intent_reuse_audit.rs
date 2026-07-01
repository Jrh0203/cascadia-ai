use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use blake3::Hasher;
use cascadia_data::{
    DatasetSplit, ImitationDatasetManifest, ImitationRecord, PositionRecord,
    read_imitation_shard_records, validate_imitation_dataset,
};
use cascadia_game::{DraftChoice, GameConfig, GameState, Tile};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use rayon::prelude::*;
use serde::Serialize;

const SCHEMA_VERSION: u16 = 2;
const EXPERIMENT_ID: &str = "o1-opponent-intent-corpus-reuse-audit-v1";
const EXPECTED_TURNS_PER_GAME: usize = 80;
const HELP: &str = concat!(
    "Usage: opponent_intent_reuse_audit \\\n",
    "  --dataset-root PATH --dataset-root PATH [--dataset-root PATH ...] \\\n",
    "  --output PATH\n\n",
    "Every canonical imitation game is rebuilt from its split seed. The audit checks\n",
    "every position and candidate action, replays the selected trajectory, recovers\n",
    "unique tile identity, derives post-action opponent survival windows, and proves\n",
    "cross-dataset non-overlap. It does not promote a single-policy corpus as final\n",
    "policy-held-out O1 evidence."
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    dataset_roots: Vec<PathBuf>,
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    classification: &'static str,
    datasets: Vec<DatasetAudit>,
    cross_dataset_overlaps: Vec<CrossDatasetOverlap>,
    recoverability: Recoverability,
    claim_boundary: ClaimBoundary,
    provenance: ExecutionProvenance,
    scientific_blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetAudit {
    dataset_id: String,
    split: String,
    teacher_strategy_id: String,
    teacher_weights_blake3: String,
    manifest_blake3: String,
    games: usize,
    positions: usize,
    candidates: usize,
    exact_checks: ExactChecks,
    action_counts: ActionCounts,
    identity_recovery: IdentityRecovery,
    survival_windows: SurvivalWindows,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ExactChecks {
    manifest_and_shard_checksums: usize,
    exact_turn_order: usize,
    exact_active_seat: usize,
    exact_position_bytes: usize,
    exact_candidate_action_hashes: usize,
    exactly_one_selected_action: usize,
    exact_state_transitions: usize,
    terminal_games: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ActionCounts {
    paired_drafts: usize,
    independent_drafts: usize,
    free_three_of_a_kind_replacements: usize,
    paid_wildlife_wipes: usize,
    wildlife_placements: usize,
    wildlife_returns: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
struct IdentityRecovery {
    positions_with_four_unique_tile_ids: usize,
    positions_with_duplicate_tile_semantics: usize,
    duplicate_tile_semantic_occurrences: usize,
    positions_with_duplicate_pair_semantics: usize,
    duplicate_pair_semantic_occurrences: usize,
    selected_tile_ids_recovered: usize,
    selected_wildlife_species_recovered: usize,
    post_action_markets_recovered: usize,
    recent_draft_histories_recoverable: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SurvivalWindows {
    focal_post_action_windows: usize,
    market_tile_labels: usize,
    exact_tile_survivors_after_opponent_1: usize,
    exact_tile_survivors_after_opponent_2: usize,
    exact_tile_survivors_after_opponent_3: usize,
    exact_tile_consumed_by_opponent_1: usize,
    exact_tile_consumed_by_opponent_2: usize,
    exact_tile_consumed_by_opponent_3: usize,
    exact_tile_survived_to_next_focal_access: usize,
    exact_tile_survival_rate_to_next_focal_access: f64,
    tile_exact_wildlife_semantic_pair_survivors: usize,
    tile_exact_wildlife_semantic_pair_survival_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CrossDatasetOverlap {
    left_dataset_id: String,
    right_dataset_id: String,
    group_id_overlap: usize,
    position_record_overlap: usize,
    public_state_overlap: usize,
    initial_hidden_state_overlap: usize,
}

#[derive(Debug, Serialize)]
struct Recoverability {
    exact_sequential_replay: bool,
    exact_candidate_action_reconstruction: bool,
    exact_selected_action_labels: bool,
    exact_unique_tile_identity: bool,
    exact_post_action_tile_survival: bool,
    exact_next_pick_slots_and_species: bool,
    exact_nature_token_action: bool,
    public_recent_draft_history: bool,
    wildlife_token_physical_identity: bool,
    wildlife_identity_note: &'static str,
}

#[derive(Debug, Serialize)]
struct ClaimBoundary {
    foundation_reuse_authorized: bool,
    final_o1_training_corpus_authorized: bool,
    policy_held_out_evaluation_available: bool,
    checkpoint_identity_shortcut_testable: bool,
    strategy_switch_target_available: bool,
    required_successor: &'static str,
}

#[derive(Debug, Serialize)]
struct ExecutionProvenance {
    source: SourceProvenance,
    executable_blake3: String,
    hostname: String,
    logical_parallelism: usize,
    dataset_roots: Vec<String>,
}

#[derive(Debug)]
struct AuditedDataset {
    report: DatasetAudit,
    group_ids: BTreeSet<u64>,
    position_hashes: BTreeSet<String>,
    public_state_hashes: BTreeSet<String>,
    initial_hidden_state_hashes: BTreeSet<String>,
}

#[derive(Debug)]
struct GameAudit {
    game_index: u64,
    positions: usize,
    candidates: usize,
    exact_checks: ExactChecks,
    action_counts: ActionCounts,
    identity_recovery: IdentityRecovery,
    survival_windows: SurvivalWindows,
    group_ids: Vec<u64>,
    position_hashes: Vec<String>,
    public_state_hashes: Vec<String>,
    initial_hidden_state_hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TileSemanticKey {
    terrain_a: u8,
    terrain_b: u8,
    wildlife_mask: u8,
    keystone: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PairSemanticKey {
    tile: TileSemanticKey,
    wildlife: u8,
}

#[derive(Debug, Clone)]
struct TurnSnapshot {
    tile_ids: [u8; 4],
    wildlife: [u8; 4],
    selected_tile_id: u8,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let dataset_roots = args
        .dataset_roots
        .iter()
        .map(|root| root.display().to_string())
        .collect();
    let mut audited = args
        .dataset_roots
        .iter()
        .map(|root| audit_dataset(root))
        .collect::<Result<Vec<_>, _>>()?;
    audited.sort_unstable_by(|left, right| {
        left.report
            .split
            .cmp(&right.report.split)
            .then_with(|| left.report.dataset_id.cmp(&right.report.dataset_id))
    });

    let overlaps = cross_dataset_overlaps(&audited);
    if overlaps.iter().any(|overlap| {
        overlap.group_id_overlap != 0
            || overlap.position_record_overlap != 0
            || overlap.public_state_overlap != 0
            || overlap.initial_hidden_state_overlap != 0
    }) {
        return Err("canonical imitation datasets overlap across declared splits".into());
    }

    let datasets = audited
        .into_iter()
        .map(|dataset| dataset.report)
        .collect::<Vec<_>>();
    let scientific_blake3 = scientific_digest(&datasets, &overlaps)?;
    let report = AuditReport {
        schema_version: SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID,
        status: "complete",
        classification: "exact_replay_foundation_reusable_policy_holdout_required",
        datasets,
        cross_dataset_overlaps: overlaps,
        recoverability: Recoverability {
            exact_sequential_replay: true,
            exact_candidate_action_reconstruction: true,
            exact_selected_action_labels: true,
            exact_unique_tile_identity: true,
            exact_post_action_tile_survival: true,
            exact_next_pick_slots_and_species: true,
            exact_nature_token_action: true,
            public_recent_draft_history: true,
            wildlife_token_physical_identity: false,
            wildlife_identity_note: "Wildlife tokens of one species are rules-equivalent; the exact public target is species and slot continuity, not an artificial physical token ID.",
        },
        claim_boundary: ClaimBoundary {
            foundation_reuse_authorized: true,
            final_o1_training_corpus_authorized: false,
            policy_held_out_evaluation_available: false,
            checkpoint_identity_shortcut_testable: false,
            strategy_switch_target_available: false,
            required_successor: "Collect exact sequential trajectories from multiple held-out opponent policy families, preserving policy only as provenance and never as an observable feature.",
        },
        provenance: ExecutionProvenance {
            source: source_provenance()?,
            executable_blake3: checksum_file(&env::current_exe()?)?,
            hostname: command_output("hostname", &[]).unwrap_or_else(|| "unknown".to_owned()),
            logical_parallelism: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            dataset_roots,
        },
        scientific_blake3,
    };
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    write_atomically(&args.output, &bytes)?;
    println!(
        "{}",
        serde_json::json!({
            "classification": report.classification,
            "datasets": report.datasets.len(),
            "games": report.datasets.iter().map(|dataset| dataset.games).sum::<usize>(),
            "positions": report.datasets.iter().map(|dataset| dataset.positions).sum::<usize>(),
            "candidates": report.datasets.iter().map(|dataset| dataset.candidates).sum::<usize>(),
            "output": args.output,
            "scientific_blake3": report.scientific_blake3,
        })
    );
    Ok(())
}

fn parse_args(
    arguments: impl IntoIterator<Item = impl Into<std::ffi::OsString>>,
) -> Result<Args, Box<dyn Error>> {
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .into_iter();
    let mut dataset_roots = Vec::new();
    let mut output = None;
    while let Some(argument) = arguments.next() {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        match argument {
            "--dataset-root" => dataset_roots.push(PathBuf::from(
                arguments.next().ok_or("--dataset-root requires a path")?,
            )),
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ));
            }
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n\n{HELP}").into()),
        }
    }
    if dataset_roots.len() < 2 {
        return Err("at least two --dataset-root values are required".into());
    }
    let mut unique = BTreeSet::new();
    for root in &dataset_roots {
        if !unique.insert(root.clone()) {
            return Err(format!("duplicate dataset root {}", root.display()).into());
        }
    }
    Ok(Args {
        dataset_roots,
        output: output.ok_or("--output is required")?,
    })
}

fn audit_dataset(root: &Path) -> Result<AuditedDataset, Box<dyn Error>> {
    let manifest_path = root.join("dataset.json");
    let manifest: ImitationDatasetManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    validate_imitation_dataset(root, &manifest)?;
    let game_results = manifest
        .shards
        .par_iter()
        .map(|shard| audit_shard(root, manifest.split, shard))
        .collect::<Vec<_>>();
    let mut games = Vec::new();
    for result in game_results {
        games.extend(result.map_err(std::io::Error::other)?);
    }
    games.sort_unstable_by_key(|game| game.game_index);
    if games.len() != manifest.completed_games {
        return Err(format!(
            "{} replayed {} games but manifest declares {}",
            root.display(),
            games.len(),
            manifest.completed_games
        )
        .into());
    }

    let mut exact_checks = ExactChecks {
        manifest_and_shard_checksums: manifest.shards.len() + 1,
        ..ExactChecks::default()
    };
    let mut action_counts = ActionCounts::default();
    let mut identity_recovery = IdentityRecovery::default();
    let mut survival_windows = SurvivalWindows::default();
    let mut positions = 0usize;
    let mut candidates = 0usize;
    let mut group_ids = BTreeSet::new();
    let mut position_hashes = BTreeSet::new();
    let mut public_state_hashes = BTreeSet::new();
    let mut initial_hidden_state_hashes = BTreeSet::new();
    for game in games {
        positions += game.positions;
        candidates += game.candidates;
        merge_exact_checks(&mut exact_checks, &game.exact_checks);
        merge_action_counts(&mut action_counts, &game.action_counts);
        merge_identity_recovery(&mut identity_recovery, &game.identity_recovery);
        merge_survival_windows(&mut survival_windows, &game.survival_windows);
        for group_id in game.group_ids {
            if !group_ids.insert(group_id) {
                return Err(format!(
                    "{} contains duplicate imitation group ID {group_id}",
                    root.display()
                )
                .into());
            }
        }
        position_hashes.extend(game.position_hashes);
        public_state_hashes.extend(game.public_state_hashes);
        if !initial_hidden_state_hashes.insert(game.initial_hidden_state_hash) {
            return Err(format!(
                "{} contains duplicate initial hidden game states",
                root.display()
            )
            .into());
        }
    }
    finalize_survival_rates(&mut survival_windows);

    if positions != manifest.total_groups || candidates != manifest.total_records {
        return Err(format!(
            "{} replay totals disagree with manifest groups or candidates",
            root.display()
        )
        .into());
    }
    Ok(AuditedDataset {
        report: DatasetAudit {
            dataset_id: manifest.dataset_id,
            split: manifest.split.id().to_owned(),
            teacher_strategy_id: manifest.teacher.strategy_id,
            teacher_weights_blake3: manifest.teacher.weights_blake3,
            manifest_blake3: checksum_file(&manifest_path)?,
            games: manifest.completed_games,
            positions,
            candidates,
            exact_checks,
            action_counts,
            identity_recovery,
            survival_windows,
        },
        group_ids,
        position_hashes,
        public_state_hashes,
        initial_hidden_state_hashes,
    })
}

fn audit_shard(
    root: &Path,
    split: DatasetSplit,
    shard: &cascadia_data::RankingShardManifest,
) -> Result<Vec<GameAudit>, String> {
    let records =
        read_imitation_shard_records(root, split, shard).map_err(|error| error.to_string())?;
    let mut games = Vec::new();
    let mut offset = 0usize;
    while offset < records.len() {
        let game_index = records[offset].input.position.game_index;
        let start = offset;
        while offset < records.len() && records[offset].input.position.game_index == game_index {
            let count = usize::from(records[offset].candidate_count);
            if count < 2 || offset + count > records.len() {
                return Err(format!(
                    "game {game_index} contains a truncated candidate group"
                ));
            }
            if records[offset..offset + count]
                .iter()
                .any(|record| record.input.position.game_index != game_index)
            {
                return Err(format!(
                    "game {game_index} crosses a candidate group boundary"
                ));
            }
            offset += count;
        }
        games.push(audit_game(split, game_index, &records[start..offset])?);
    }
    if games.len() != shard.game_count {
        return Err(format!(
            "shard {} replays {} games but declares {}",
            shard.file,
            games.len(),
            shard.game_count
        ));
    }
    Ok(games)
}

fn audit_game(
    split: DatasetSplit,
    game_index: u64,
    records: &[ImitationRecord],
) -> Result<GameAudit, String> {
    let mut state = GameState::new(
        GameConfig::research_aaaaa(4).map_err(|error| error.to_string())?,
        split.game_seed(game_index),
    )
    .map_err(|error| error.to_string())?;
    let initial_hidden_state_hash = state.canonical_hash().to_hex().to_string();
    let mut offset = 0usize;
    let mut turn = 0usize;
    let mut candidates = 0usize;
    let mut exact_checks = ExactChecks::default();
    let mut action_counts = ActionCounts::default();
    let mut identity_recovery = IdentityRecovery::default();
    let mut group_ids = Vec::new();
    let mut position_hashes = Vec::new();
    let mut public_state_hashes = Vec::new();
    let mut snapshots = Vec::with_capacity(EXPECTED_TURNS_PER_GAME);
    while offset < records.len() {
        let count = usize::from(records[offset].candidate_count);
        let end = offset + count;
        if count < 2 || end > records.len() {
            return Err(format!("game {game_index} contains a truncated group"));
        }
        let group = &records[offset..end];
        let expected_group_id = group[0].group_id;
        if group.iter().enumerate().any(|(candidate_index, record)| {
            record.group_id != expected_group_id
                || usize::from(record.candidate_index) != candidate_index
                || usize::from(record.candidate_count) != count
        }) {
            return Err(format!(
                "game {game_index} group {expected_group_id} is misaligned"
            ));
        }
        if state.completed_turns() as usize != turn || group[0].input.position.turn as usize != turn
        {
            return Err(format!("game {game_index} is not ordered at turn {turn}"));
        }
        exact_checks.exact_turn_order += 1;
        if state.current_player() != turn % 4
            || usize::from(group[0].input.position.active_seat) != state.current_player()
        {
            return Err(format!(
                "game {game_index} has the wrong active seat at turn {turn}"
            ));
        }
        exact_checks.exact_active_seat += 1;

        let expected_position = PositionRecord::observe(&state, game_index);
        if group
            .iter()
            .any(|record| record.input.position.to_bytes() != expected_position.to_bytes())
        {
            return Err(format!(
                "game {game_index} position bytes differ at turn {turn}"
            ));
        }
        exact_checks.exact_position_bytes += 1;
        position_hashes.push(
            blake3::hash(&expected_position.to_bytes())
                .to_hex()
                .to_string(),
        );
        public_state_hashes.push(state.public_state().canonical_hash().to_hex().to_string());
        group_ids.push(expected_group_id);

        let market = state.market();
        let tile_ids = market.tiles.map(|tile| {
            tile.expect("active standard game market has four tiles")
                .id
                .0
        });
        let wildlife = market
            .wildlife
            .map(|wildlife| wildlife.expect("active standard game market has four wildlife") as u8);
        if tile_ids.into_iter().collect::<BTreeSet<_>>().len() != 4 {
            return Err(format!(
                "game {game_index} repeats a unique tile ID at turn {turn}"
            ));
        }
        identity_recovery.positions_with_four_unique_tile_ids += 1;
        let tile_semantics = market
            .tiles
            .map(|tile| tile_semantic_key(tile.expect("market is full")));
        let tile_duplicates = duplicate_occurrences(&tile_semantics);
        if tile_duplicates != 0 {
            identity_recovery.positions_with_duplicate_tile_semantics += 1;
            identity_recovery.duplicate_tile_semantic_occurrences += tile_duplicates;
        }
        let pair_semantics: [PairSemanticKey; 4] = std::array::from_fn(|slot| PairSemanticKey {
            tile: tile_semantics[slot],
            wildlife: wildlife[slot],
        });
        let pair_duplicates = duplicate_occurrences(&pair_semantics);
        if pair_duplicates != 0 {
            identity_recovery.positions_with_duplicate_pair_semantics += 1;
            identity_recovery.duplicate_pair_semantic_occurrences += pair_duplicates;
        }

        let mut selected = None;
        for record in group {
            let action = record
                .input
                .action
                .to_game_action(&state)
                .map_err(|error| error.to_string())?;
            let hash =
                *blake3::hash(&serde_json::to_vec(&action).map_err(|error| error.to_string())?)
                    .as_bytes();
            if hash != record.action_hash {
                return Err(format!(
                    "game {game_index} candidate hash differs at turn {turn}"
                ));
            }
            exact_checks.exact_candidate_action_hashes += 1;
            if record.teacher_mean == 1.0 {
                if selected.replace(action).is_some() {
                    return Err(format!(
                        "game {game_index} has multiple selected actions at turn {turn}"
                    ));
                }
            } else if record.teacher_mean != 0.0 {
                return Err(format!(
                    "game {game_index} has a non-binary selected label at turn {turn}"
                ));
            }
        }
        let selected = selected
            .ok_or_else(|| format!("game {game_index} has no selected action at turn {turn}"))?;
        exact_checks.exactly_one_selected_action += 1;
        let (tile_slot, wildlife_slot) = match selected.draft {
            DraftChoice::Paired { slot } => {
                action_counts.paired_drafts += 1;
                (slot.index(), slot.index())
            }
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => {
                action_counts.independent_drafts += 1;
                (tile_slot.index(), wildlife_slot.index())
            }
        };
        action_counts.free_three_of_a_kind_replacements +=
            usize::from(selected.replace_three_of_a_kind);
        action_counts.paid_wildlife_wipes += selected.wildlife_wipes.len();
        action_counts.wildlife_placements += usize::from(selected.wildlife.is_some());
        action_counts.wildlife_returns += usize::from(selected.wildlife.is_none());
        identity_recovery.selected_tile_ids_recovered += 1;
        identity_recovery.selected_wildlife_species_recovered += 1;
        snapshots.push(TurnSnapshot {
            tile_ids,
            wildlife,
            selected_tile_id: tile_ids[tile_slot],
        });
        if wildlife[wildlife_slot] > 4 {
            return Err(format!(
                "game {game_index} selected an invalid wildlife code at turn {turn}"
            ));
        }

        state.apply(&selected).map_err(|error| error.to_string())?;
        exact_checks.exact_state_transitions += 1;
        identity_recovery.post_action_markets_recovered += usize::from(!state.is_game_over());
        identity_recovery.recent_draft_histories_recoverable += 1;
        candidates += count;
        turn += 1;
        offset = end;
    }
    if turn != EXPECTED_TURNS_PER_GAME || !state.is_game_over() {
        return Err(format!(
            "game {game_index} ended after {turn} turns instead of {EXPECTED_TURNS_PER_GAME}"
        ));
    }
    exact_checks.terminal_games = 1;
    let survival_windows = summarize_survival_windows(&snapshots)?;
    Ok(GameAudit {
        game_index,
        positions: turn,
        candidates,
        exact_checks,
        action_counts,
        identity_recovery,
        survival_windows,
        group_ids,
        position_hashes,
        public_state_hashes,
        initial_hidden_state_hash,
    })
}

fn summarize_survival_windows(snapshots: &[TurnSnapshot]) -> Result<SurvivalWindows, String> {
    if snapshots.len() != EXPECTED_TURNS_PER_GAME {
        return Err("survival summary requires one snapshot per game turn".to_owned());
    }
    let mut summary = SurvivalWindows::default();
    for focal_turn in 0..=(EXPECTED_TURNS_PER_GAME - 5) {
        let post_action = &snapshots[focal_turn + 1];
        let after_opponent_1 = &snapshots[focal_turn + 2];
        let after_opponent_2 = &snapshots[focal_turn + 3];
        let next_focal_access = &snapshots[focal_turn + 4];
        summary.focal_post_action_windows += 1;
        for slot in 0..4 {
            let tile_id = post_action.tile_ids[slot];
            let wildlife = post_action.wildlife[slot];
            summary.market_tile_labels += 1;
            summary.exact_tile_survivors_after_opponent_1 +=
                usize::from(after_opponent_1.tile_ids.contains(&tile_id));
            summary.exact_tile_survivors_after_opponent_2 +=
                usize::from(after_opponent_2.tile_ids.contains(&tile_id));
            summary.exact_tile_survivors_after_opponent_3 +=
                usize::from(next_focal_access.tile_ids.contains(&tile_id));

            if snapshots[focal_turn + 1].selected_tile_id == tile_id {
                summary.exact_tile_consumed_by_opponent_1 += 1;
            } else if snapshots[focal_turn + 2].selected_tile_id == tile_id {
                summary.exact_tile_consumed_by_opponent_2 += 1;
            } else if snapshots[focal_turn + 3].selected_tile_id == tile_id {
                summary.exact_tile_consumed_by_opponent_3 += 1;
            }

            if let Some(next_slot) = next_focal_access
                .tile_ids
                .iter()
                .position(|candidate| *candidate == tile_id)
            {
                summary.exact_tile_survived_to_next_focal_access += 1;
                summary.tile_exact_wildlife_semantic_pair_survivors +=
                    usize::from(next_focal_access.wildlife[next_slot] == wildlife);
            }
        }
    }
    finalize_survival_rates(&mut summary);
    Ok(summary)
}

fn tile_semantic_key(tile: Tile) -> TileSemanticKey {
    TileSemanticKey {
        terrain_a: tile.terrain_a as u8,
        terrain_b: tile.terrain_b.map_or(u8::MAX, |terrain| terrain as u8),
        wildlife_mask: tile.wildlife.bits(),
        keystone: tile.keystone,
    }
}

fn duplicate_occurrences<T: Ord + Copy>(values: &[T; 4]) -> usize {
    let mut counts = BTreeMap::new();
    for value in values {
        *counts.entry(*value).or_insert(0usize) += 1;
    }
    counts.values().filter(|count| **count > 1).copied().sum()
}

fn cross_dataset_overlaps(datasets: &[AuditedDataset]) -> Vec<CrossDatasetOverlap> {
    let mut overlaps = Vec::new();
    for left_index in 0..datasets.len() {
        for right_index in left_index + 1..datasets.len() {
            let left = &datasets[left_index];
            let right = &datasets[right_index];
            overlaps.push(CrossDatasetOverlap {
                left_dataset_id: left.report.dataset_id.clone(),
                right_dataset_id: right.report.dataset_id.clone(),
                group_id_overlap: intersection_count(&left.group_ids, &right.group_ids),
                position_record_overlap: intersection_count(
                    &left.position_hashes,
                    &right.position_hashes,
                ),
                public_state_overlap: intersection_count(
                    &left.public_state_hashes,
                    &right.public_state_hashes,
                ),
                initial_hidden_state_overlap: intersection_count(
                    &left.initial_hidden_state_hashes,
                    &right.initial_hidden_state_hashes,
                ),
            });
        }
    }
    overlaps
}

fn intersection_count<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> usize {
    left.intersection(right).count()
}

fn merge_exact_checks(target: &mut ExactChecks, source: &ExactChecks) {
    target.manifest_and_shard_checksums += source.manifest_and_shard_checksums;
    target.exact_turn_order += source.exact_turn_order;
    target.exact_active_seat += source.exact_active_seat;
    target.exact_position_bytes += source.exact_position_bytes;
    target.exact_candidate_action_hashes += source.exact_candidate_action_hashes;
    target.exactly_one_selected_action += source.exactly_one_selected_action;
    target.exact_state_transitions += source.exact_state_transitions;
    target.terminal_games += source.terminal_games;
}

fn merge_action_counts(target: &mut ActionCounts, source: &ActionCounts) {
    target.paired_drafts += source.paired_drafts;
    target.independent_drafts += source.independent_drafts;
    target.free_three_of_a_kind_replacements += source.free_three_of_a_kind_replacements;
    target.paid_wildlife_wipes += source.paid_wildlife_wipes;
    target.wildlife_placements += source.wildlife_placements;
    target.wildlife_returns += source.wildlife_returns;
}

fn merge_identity_recovery(target: &mut IdentityRecovery, source: &IdentityRecovery) {
    target.positions_with_four_unique_tile_ids += source.positions_with_four_unique_tile_ids;
    target.positions_with_duplicate_tile_semantics +=
        source.positions_with_duplicate_tile_semantics;
    target.duplicate_tile_semantic_occurrences += source.duplicate_tile_semantic_occurrences;
    target.positions_with_duplicate_pair_semantics +=
        source.positions_with_duplicate_pair_semantics;
    target.duplicate_pair_semantic_occurrences += source.duplicate_pair_semantic_occurrences;
    target.selected_tile_ids_recovered += source.selected_tile_ids_recovered;
    target.selected_wildlife_species_recovered += source.selected_wildlife_species_recovered;
    target.post_action_markets_recovered += source.post_action_markets_recovered;
    target.recent_draft_histories_recoverable += source.recent_draft_histories_recoverable;
}

fn merge_survival_windows(target: &mut SurvivalWindows, source: &SurvivalWindows) {
    target.focal_post_action_windows += source.focal_post_action_windows;
    target.market_tile_labels += source.market_tile_labels;
    target.exact_tile_survivors_after_opponent_1 += source.exact_tile_survivors_after_opponent_1;
    target.exact_tile_survivors_after_opponent_2 += source.exact_tile_survivors_after_opponent_2;
    target.exact_tile_survivors_after_opponent_3 += source.exact_tile_survivors_after_opponent_3;
    target.exact_tile_consumed_by_opponent_1 += source.exact_tile_consumed_by_opponent_1;
    target.exact_tile_consumed_by_opponent_2 += source.exact_tile_consumed_by_opponent_2;
    target.exact_tile_consumed_by_opponent_3 += source.exact_tile_consumed_by_opponent_3;
    target.exact_tile_survived_to_next_focal_access +=
        source.exact_tile_survived_to_next_focal_access;
    target.tile_exact_wildlife_semantic_pair_survivors +=
        source.tile_exact_wildlife_semantic_pair_survivors;
}

fn finalize_survival_rates(summary: &mut SurvivalWindows) {
    summary.exact_tile_survival_rate_to_next_focal_access = ratio(
        summary.exact_tile_survived_to_next_focal_access,
        summary.market_tile_labels,
    );
    summary.tile_exact_wildlife_semantic_pair_survival_rate = ratio(
        summary.tile_exact_wildlife_semantic_pair_survivors,
        summary.market_tile_labels,
    );
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn scientific_digest(
    datasets: &[DatasetAudit],
    overlaps: &[CrossDatasetOverlap],
) -> Result<String, serde_json::Error> {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-o1-opponent-intent-corpus-reuse-audit-v1");
    hasher.update(&serde_json::to_vec(datasets)?);
    hasher.update(&serde_json::to_vec(overlaps)?);
    Ok(hasher.finalize().to_hex().to_string())
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use cascadia_data::{ImitationRecord, ProposalPositionRecord};
    use cascadia_game::{GameSeed, MarketSlot, Rotation, TurnAction, score_board};

    use super::*;

    fn sample_group(game: &GameState, game_index: u64) -> Vec<ImitationRecord> {
        game.legal_turn_actions(&Default::default())
            .unwrap()
            .into_iter()
            .take(2)
            .enumerate()
            .map(|(index, action)| {
                let score = score_board(
                    &game.preview_active_board(&action).unwrap(),
                    game.config().scoring_cards,
                )
                .base_total;
                ImitationRecord {
                    group_id: 99,
                    candidate_index: index as u16,
                    candidate_count: 2,
                    immediate_rank: index as u16 + 1,
                    immediate_score: score,
                    teacher_mean: f32::from(index == 0),
                    teacher_stddev: 0.0,
                    action_hash: *blake3::hash(&serde_json::to_vec(&action).unwrap()).as_bytes(),
                    input: ProposalPositionRecord::observe(
                        game,
                        &action,
                        game_index,
                        index as u16 + 1,
                        score,
                    )
                    .unwrap(),
                }
            })
            .collect()
    }

    #[test]
    fn semantic_duplicate_counter_counts_all_ambiguous_occurrences() {
        assert_eq!(duplicate_occurrences(&[1, 1, 2, 3]), 2);
        assert_eq!(duplicate_occurrences(&[1, 1, 2, 2]), 4);
        assert_eq!(duplicate_occurrences(&[1, 2, 3, 4]), 0);
    }

    #[test]
    fn exact_candidate_actions_reconstruct_from_a_recorded_group() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(7),
        )
        .unwrap();
        let records = sample_group(&game, 90_000);
        for record in records {
            let action = record.input.action.to_game_action(&game).unwrap();
            assert_eq!(
                *blake3::hash(&serde_json::to_vec(&action).unwrap()).as_bytes(),
                record.action_hash
            );
        }
    }

    #[test]
    fn survival_windows_follow_post_action_opponent_access_order() {
        let snapshots = (0..80)
            .map(|turn| TurnSnapshot {
                tile_ids: [
                    (turn * 4) as u8,
                    (turn * 4 + 1) as u8,
                    (turn * 4 + 2) as u8,
                    (turn * 4 + 3) as u8,
                ],
                wildlife: [0, 1, 2, 3],
                selected_tile_id: (turn * 4) as u8,
            })
            .collect::<Vec<_>>();
        let summary = summarize_survival_windows(&snapshots).unwrap();
        assert_eq!(summary.focal_post_action_windows, 76);
        assert_eq!(summary.market_tile_labels, 304);
        assert_eq!(summary.exact_tile_survived_to_next_focal_access, 0);
    }

    #[test]
    fn compact_paired_action_metadata_names_the_selected_market_tile() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(8),
        )
        .unwrap();
        let action = TurnAction::paired(
            MarketSlot::ZERO,
            game.boards()[0].frontier()[0],
            Rotation::ZERO,
        );
        let score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        let record = ProposalPositionRecord::observe(&game, &action, 1, 1, score).unwrap();
        assert_eq!(record.action.tile_slot, 0);
        assert_eq!(record.action.wildlife_slot, 0);
        assert_eq!(
            game.market().tiles[0].unwrap().id.0,
            game.market().tiles[usize::from(record.action.tile_slot)]
                .unwrap()
                .id
                .0
        );
    }
}
