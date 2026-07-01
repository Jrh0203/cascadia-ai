use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_data::{
    DatasetSplit, GradedOracleDatasetManifest, GradedOracleGroup, read_graded_oracle_shard,
    validate_graded_oracle_dataset,
};
use cascadia_game::{DraftChoice, GameConfig, GameSeed, GameState, TurnAction};
use clap::Parser;
use r3_action_edit_census::{ActionEdit, PublicStateTrunk};
use relational_feature_census::{IncrementalSparseAccumulator, write_json_atomic};
use serde::{Deserialize, Serialize};
use serde_json::json;

const CACHE_SCHEMA: &str = "relational-substrate-mlx-cache-v1";
const EXPERIMENT_ID: &str = "relational-substrate-mlx-tournament-v1";
const PROTOCOL_ID: &str = "r5-s3-s5-matched-mlx-v1";

#[derive(Debug, Parser)]
#[command(about = "Replay exact ADR 0161 validation actions through R6 apply/undo")]
struct Args {
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    relational_cache: PathBuf,
    #[arg(long)]
    rows: String,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CacheManifest {
    cache_schema: String,
    experiment_id: String,
    protocol_id: String,
    complete_open_corpus: bool,
    splits: BTreeMap<String, SplitManifest>,
}

#[derive(Debug, Deserialize)]
struct SplitManifest {
    groups: usize,
    dataset_id: String,
    dataset_manifest_blake3: String,
    files: BTreeMap<String, FileSpec>,
}

#[derive(Debug, Deserialize)]
struct FileSpec {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

struct CacheBinding {
    groups: usize,
    group_ids: Vec<u64>,
    candidate_offsets: Vec<u64>,
    source_candidate_indices: Vec<u16>,
    action_hashes: Vec<[u8; 32]>,
    dataset_id: String,
    dataset_manifest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DraftBatchKey {
    replace_three_of_a_kind: bool,
    wipe_masks: Vec<u8>,
    draft_kind: u8,
    tile_slot: u8,
    wildlife_slot: u8,
}

#[derive(Default)]
struct ReplayMetrics {
    groups: usize,
    actions: usize,
    action_hash_checks: usize,
    grouped_action_matches: usize,
    afterstate_hash_checks: usize,
    apply_checks: usize,
    apply_failures: usize,
    undo_checks: usize,
    undo_failures: usize,
    apply_undo_ns: u128,
    samples: Vec<(usize, usize, u64)>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let rows = parse_rows(&args.rows)?;
    let cache = load_cache(&args.relational_cache)?;
    if rows.iter().any(|row| *row >= cache.groups) {
        return Err("R6 replay row is outside the validation cache".into());
    }
    let manifest_path = args.dataset.join("dataset.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_graded_oracle_dataset(&args.dataset, &manifest)?;
    if manifest.split != DatasetSplit::Validation
        || manifest.dataset_id != cache.dataset_id
        || blake3::hash(&manifest_bytes).to_hex().as_str() != cache.dataset_manifest_blake3
    {
        return Err("R6 replay dataset identity differs from the relational cache".into());
    }

    let wanted = rows
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut metrics = ReplayMetrics::default();
    let mut global_row = 0usize;
    'shards: for shard in &manifest.shards {
        let groups = read_graded_oracle_shard(&args.dataset, DatasetSplit::Validation, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        for group in groups {
            if global_row >= cache.groups || group.group_id != cache.group_ids[global_row] {
                return Err(format!("R6 replay group identity drifted at row {global_row}").into());
            }
            if wanted.contains(&global_row) {
                replay_group(global_row, &game, &group, &cache, &mut metrics)?;
            }
            let champion = group.candidates[usize::from(group.champion_index)]
                .action
                .to_game_action(&game)?;
            game.apply(&champion)?;
            global_row += 1;
            if metrics.groups == rows.len() && global_row > *rows.last().expect("nonempty rows") {
                break 'shards;
            }
        }
        if !game.is_game_over() {
            return Err(format!(
                "R6 replay source game {} is incomplete",
                shard.first_game_index
            )
            .into());
        }
    }
    if metrics.groups != rows.len()
        || metrics.actions == 0
        || metrics.action_hash_checks != metrics.actions
        || metrics.grouped_action_matches != metrics.actions
        || metrics.afterstate_hash_checks != metrics.actions
        || metrics.apply_checks != metrics.actions
        || metrics.undo_checks != metrics.actions
        || metrics.apply_failures != 0
        || metrics.undo_failures != 0
    {
        return Err("R6 replay did not produce complete exact evidence".into());
    }
    let latencies = metrics
        .samples
        .iter()
        .map(|(_, _, ns)| *ns)
        .collect::<Vec<_>>();
    let identity = json!({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": manifest.dataset_id,
        "relational_cache": args.relational_cache,
        "rows": rows,
        "groups": metrics.groups,
        "actions": metrics.actions,
        "action_hash_checks": metrics.action_hash_checks,
        "grouped_action_matches": metrics.grouped_action_matches,
        "afterstate_hash_checks": metrics.afterstate_hash_checks,
        "apply_checks": metrics.apply_checks,
        "apply_failures": metrics.apply_failures,
        "undo_checks": metrics.undo_checks,
        "undo_failures": metrics.undo_failures,
        "apply_undo_ns": u64::try_from(metrics.apply_undo_ns)?,
        "action_apply_undo_per_second": metrics.actions as f64 * 1_000_000_000.0
            / metrics.apply_undo_ns.max(1) as f64,
        "latency_milliseconds": latency_summary(&latencies),
        "samples": metrics.samples.iter().map(|(row, actions, ns)| json!({
            "row": row,
            "actions": actions,
            "nanoseconds": ns,
        })).collect::<Vec<_>>(),
        "exact_parity_pass": true,
    });
    let report = json!({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "report_id": canonical_blake3(&identity)?,
        "scientific_identity": identity,
    });
    write_json_atomic(&args.output, &report)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn replay_group(
    row: usize,
    game: &GameState,
    group: &GradedOracleGroup,
    cache: &CacheBinding,
    metrics: &mut ReplayMetrics,
) -> Result<(), Box<dyn Error>> {
    let start = usize::try_from(cache.candidate_offsets[row])?;
    let end = usize::try_from(cache.candidate_offsets[row + 1])?;
    if start >= end || end > cache.source_candidate_indices.len() {
        return Err(format!("R6 candidate offsets drifted at row {row}").into());
    }
    let sources = &cache.source_candidate_indices[start..end];
    let hashes = &cache.action_hashes[start..end];
    let trunk = PublicStateTrunk::observe(game, group.raw_seed)?;
    let prepared = trunk.prepare_action_edits()?;
    let mut batches = BTreeMap::<DraftBatchKey, Vec<(usize, TurnAction)>>::new();
    for (retained, source_index) in sources.iter().copied().enumerate() {
        let source = usize::from(source_index);
        let candidate = group
            .candidates
            .get(source)
            .ok_or("R6 retained source index is outside the graded group")?;
        if candidate.action_hash != hashes[retained] {
            return Err(format!("R6 action hash drifted at row {row} source {source}").into());
        }
        metrics.action_hash_checks += 1;
        let action = candidate.action.to_game_action(game)?;
        batches
            .entry(draft_batch_key(&action))
            .or_default()
            .push((retained, action));
    }
    let mut edits = vec![None; sources.len()];
    for targets in batches.into_values() {
        let prelude = targets[0].1.prelude();
        let draft = targets[0].1.draft;
        let positions = targets
            .into_iter()
            .map(|(position, action)| (action, position))
            .collect::<HashMap<_, _>>();
        for (action, edit) in prepared.observe_draft_actions(game, &prelude, draft)? {
            if let Some(position) = positions.get(&action).copied() {
                if edits[position].replace(edit).is_some() {
                    return Err("R6 grouped enumeration emitted a duplicate".into());
                }
                metrics.grouped_action_matches += 1;
            }
        }
    }
    let edits = edits
        .into_iter()
        .map(|edit| edit.ok_or("R6 grouped enumeration missed a retained action"))
        .collect::<Result<Vec<ActionEdit>, _>>()?;
    let mut accumulator = IncrementalSparseAccumulator::from_prepared(&prepared)?;
    let parent_digest = accumulator.canonical_blake3()?;

    let started = Instant::now();
    for edit in &edits {
        let journal = accumulator.apply(edit)?;
        black_box(accumulator.canonical_blake3()?);
        accumulator.undo(edit, journal)?;
    }
    let elapsed = started.elapsed().as_nanos();
    metrics.apply_undo_ns += elapsed;
    metrics
        .samples
        .push((row, edits.len(), u64::try_from(elapsed)?));

    for edit in &edits {
        let authoritative = prepared.apply(edit)?;
        if authoritative.canonical_record_hash() != edit.expected_public_afterstate_blake3 {
            return Err("R6 authoritative afterstate hash drifted".into());
        }
        metrics.afterstate_hash_checks += 1;
        let journal = accumulator.apply(edit)?;
        metrics.apply_checks += 1;
        if !accumulator.matches_authoritative(&authoritative, edit, &trunk)? {
            metrics.apply_failures += 1;
        }
        accumulator.undo(edit, journal)?;
        metrics.undo_checks += 1;
        if accumulator.canonical_blake3()? != parent_digest {
            metrics.undo_failures += 1;
        }
    }
    metrics.groups += 1;
    metrics.actions += edits.len();
    Ok(())
}

fn load_cache(root: &Path) -> Result<CacheBinding, Box<dyn Error>> {
    let bytes = fs::read(root.join("cache.json"))?;
    let manifest: CacheManifest = serde_json::from_slice(&bytes)?;
    if manifest.cache_schema != CACHE_SCHEMA
        || manifest.experiment_id != EXPERIMENT_ID
        || manifest.protocol_id != PROTOCOL_ID
    {
        return Err("R6 replay relational cache envelope drifted".into());
    }
    let split = manifest
        .splits
        .get("validation")
        .ok_or("R6 replay cache lacks validation")?;
    let group_ids = read_u64(root, required_file(split, "group_ids")?, split.groups)?;
    let candidate_offsets = read_u64(
        root,
        required_file(split, "candidate_offsets")?,
        split.groups + 1,
    )?;
    let candidates = usize::try_from(*candidate_offsets.last().ok_or("empty offsets")?)?;
    let source_candidate_indices = read_u16(
        root,
        required_file(split, "source_candidate_indices")?,
        candidates,
    )?;
    let action_hashes = read_hashes(root, required_file(split, "action_hashes")?, candidates)?;
    if candidate_offsets[0] != 0
        || candidate_offsets.windows(2).any(|pair| pair[0] >= pair[1])
        || manifest.complete_open_corpus && split.groups != 240
    {
        return Err("R6 replay cache candidate accounting drifted".into());
    }
    Ok(CacheBinding {
        groups: split.groups,
        group_ids,
        candidate_offsets,
        source_candidate_indices,
        action_hashes,
        dataset_id: split.dataset_id.clone(),
        dataset_manifest_blake3: split.dataset_manifest_blake3.clone(),
    })
}

fn required_file<'a>(split: &'a SplitManifest, name: &str) -> Result<&'a FileSpec, Box<dyn Error>> {
    split
        .files
        .get(name)
        .ok_or_else(|| format!("R6 cache tensor {name} is absent").into())
}

fn read_u16(root: &Path, spec: &FileSpec, count: usize) -> Result<Vec<u16>, Box<dyn Error>> {
    let bytes = read_tensor(root, spec, "<u2", &[count])?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_u64(root: &Path, spec: &FileSpec, count: usize) -> Result<Vec<u64>, Box<dyn Error>> {
    let bytes = read_tensor(root, spec, "<u8", &[count])?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn read_hashes(
    root: &Path,
    spec: &FileSpec,
    count: usize,
) -> Result<Vec<[u8; 32]>, Box<dyn Error>> {
    let bytes = read_tensor(root, spec, "|u1", &[count, 32])?;
    Ok(bytes
        .chunks_exact(32)
        .map(|chunk| {
            let mut value = [0; 32];
            value.copy_from_slice(chunk);
            value
        })
        .collect())
}

fn read_tensor(
    root: &Path,
    spec: &FileSpec,
    dtype: &str,
    shape: &[usize],
) -> Result<Vec<u8>, Box<dyn Error>> {
    if spec.dtype != dtype || spec.shape != shape {
        return Err("R6 tensor dtype or shape drifted".into());
    }
    let path = root.join(&spec.file);
    if path.parent() != Some(root) || !path.is_file() {
        return Err("R6 tensor path escapes or is absent".into());
    }
    let bytes = fs::read(&path)?;
    if bytes.len() as u64 != spec.bytes || blake3::hash(&bytes).to_hex().as_str() != spec.blake3 {
        return Err("R6 tensor byte count or checksum drifted".into());
    }
    Ok(bytes)
}

fn parse_rows(raw: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut rows = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    rows.sort_unstable();
    rows.dedup();
    if rows.is_empty() {
        return Err("R6 replay rows are empty".into());
    }
    Ok(rows)
}

fn draft_batch_key(action: &TurnAction) -> DraftBatchKey {
    let wipe_masks = action
        .wildlife_wipes
        .iter()
        .map(|wipe| {
            wipe.slots
                .iter()
                .fold(0u8, |mask, slot| mask | (1 << slot.index()))
        })
        .collect();
    let (draft_kind, tile_slot, wildlife_slot) = match action.draft {
        DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8),
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (1, tile_slot.index() as u8, wildlife_slot.index() as u8),
    };
    DraftBatchKey {
        replace_three_of_a_kind: action.replace_three_of_a_kind,
        wipe_masks,
        draft_kind,
        tile_slot,
        wildlife_slot,
    }
}

fn latency_summary(values: &[u64]) -> serde_json::Value {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let select = |numerator: usize, denominator: usize| {
        let index = (sorted.len() - 1) * numerator / denominator;
        sorted[index] as f64 / 1_000_000.0
    };
    json!({
        "p50": select(50, 100),
        "p95": select(95, 100),
        "p99": select(99, 100),
    })
}

fn canonical_blake3(value: &impl Serialize) -> Result<String, Box<dyn Error>> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}
