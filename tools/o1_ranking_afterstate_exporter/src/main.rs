use std::{
    collections::{BTreeMap, HashMap},
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use blake3::Hasher;
use cascadia_data::{
    DatasetSplit, GradedOracleDatasetManifest, GradedOracleGroup, OPPONENT_INTENT_HISTORY_LENGTH,
    OPPONENT_INTENT_RECORD_SIZE, OpponentActionTarget, OpponentIntentHistoryEntry,
    OpponentIntentRecord, PositionRecord, PublicActionRecord, TileSurvivalTarget,
    read_graded_oracle_shard, validate_graded_oracle_dataset,
};
use cascadia_game::{GameConfig, GameSeed, GameState, TurnAction};
use cascadia_provenance::{checksum_file, source_provenance};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const EXPERIMENT_ID: &str = "o1-high-regret-draft-ranking-integration-v1";
const PROTOCOL_ID: &str = "o1-intent-conditioned-exact-r2-reranker-v1";
const ADR_ID: &str = "0188";
const CACHE_SCHEMA_VERSION: u16 = 1;
const CACHE_SCHEMA: &str = "o1-ranking-public-afterstate-cache-v1";
const COHORT_SCHEMA: &str = "o1-ranking-exact-r2-top64-cohort-v1";
const COHORT_WIDTH: usize = 64;
const TRAIN_GROUPS: usize = 560;
const VALIDATION_GROUPS: usize = 240;
const NONE: u8 = u8::MAX;
const INPUT_HASH_DOMAIN: &[u8] = b"cascadia-v2-o1-ranking-model-input-v1";

#[derive(Debug, Parser)]
#[command(about = "Export authoritative public candidate afterstates for ADR 0188")]
struct Args {
    #[arg(long)]
    train_dataset: PathBuf,
    #[arg(long)]
    validation_dataset: PathBuf,
    #[arg(long)]
    cohort: PathBuf,
    #[arg(long)]
    output_root: PathBuf,
    #[arg(long)]
    receipt: PathBuf,
    #[arg(long)]
    max_groups_per_split: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct CohortManifest {
    schema_version: u16,
    cache_schema: String,
    experiment_id: String,
    protocol_id: String,
    adr: String,
    cache_id: String,
    complete_open_corpus: bool,
    splits: BTreeMap<String, CohortSplitManifest>,
}

#[derive(Debug, Clone, Deserialize)]
struct CohortSplitManifest {
    dataset_id: String,
    groups: usize,
    files: BTreeMap<String, InputFileSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct InputFileSpec {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

#[derive(Debug, Clone)]
struct CohortSplit {
    dataset_id: String,
    groups: usize,
    source_candidate_indices: Vec<u16>,
    action_hashes: Vec<u8>,
    group_rows: HashMap<u64, usize>,
}

impl CohortSplit {
    fn sources(&self, row: usize) -> &[u16] {
        &self.source_candidate_indices[row * COHORT_WIDTH..(row + 1) * COHORT_WIDTH]
    }

    fn hashes(&self, row: usize) -> &[u8] {
        &self.action_hashes[row * COHORT_WIDTH * 32..(row + 1) * COHORT_WIDTH * 32]
    }
}

#[derive(Debug, Clone, Serialize)]
struct FileSpec {
    file: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SplitChecks {
    groups_replayed: usize,
    position_record_checks: usize,
    public_state_hash_checks: usize,
    action_reconstructions: usize,
    action_hash_checks: usize,
    afterstate_records: usize,
    afterstate_turn_checks: usize,
    afterstate_perspective_checks: usize,
    depleted_tile_checks: usize,
    depleted_wildlife_checks: usize,
    history_records: usize,
    history_age_checks: usize,
    model_input_hashes: usize,
    champion_actions_applied: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    split: &'static str,
    dataset_id: String,
    dataset_manifest_blake3: String,
    cohort_groups: usize,
    groups: usize,
    candidates: usize,
    complete_open_split: bool,
    files: BTreeMap<String, FileSpec>,
    checks: SplitChecks,
}

#[derive(Debug, Clone, Serialize)]
struct ExporterIdentity {
    executable_blake3: String,
    source: cascadia_provenance::SourceProvenance,
}

#[derive(Debug, Clone, Serialize)]
struct HiddenInformationBoundary {
    public_candidate_afterstate_only: bool,
    champion_history_only: bool,
    target_fields_zeroed: bool,
    policy_identity_exported: bool,
    hidden_post_draft_refill_exported: bool,
    hidden_stack_order_exported: bool,
    hidden_bag_order_exported: bool,
    sealed_test_opened: bool,
    gameplay_run: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CacheManifest {
    schema_version: u16,
    cache_schema: &'static str,
    experiment_id: &'static str,
    protocol_id: &'static str,
    adr: &'static str,
    cache_id: String,
    complete_open_corpus: bool,
    cohort_id: String,
    cohort_manifest_blake3: String,
    exporter: ExporterIdentity,
    hidden_information: HiddenInformationBoundary,
    splits: BTreeMap<String, SplitManifest>,
    scientific_identity: Value,
}

#[derive(Debug, Clone, Copy)]
struct HistoricalAction {
    seat: usize,
    action: PublicActionRecord,
}

struct HashedWriter {
    file_name: String,
    writer: BufWriter<File>,
    hasher: Hasher,
    bytes: u64,
}

impl HashedWriter {
    fn create(root: &Path, file_name: impl Into<String>) -> Result<Self, Box<dyn Error>> {
        let file_name = file_name.into();
        Ok(Self {
            writer: BufWriter::new(File::create(root.join(&file_name))?),
            file_name,
            hasher: Hasher::new(),
            bytes: 0,
        })
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        self.writer.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    fn write_u16(&mut self, value: u16) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_u64(&mut self, value: u64) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn finish(
        mut self,
        dtype: &'static str,
        shape: Vec<usize>,
    ) -> Result<FileSpec, Box<dyn Error>> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(FileSpec {
            file: self.file_name,
            dtype,
            shape,
            bytes: self.bytes,
            blake3: self.hasher.finalize().to_hex().to_string(),
        })
    }
}

struct SplitWriters {
    group_ids: HashedWriter,
    source_candidate_indices: HashedWriter,
    action_hashes: HashedWriter,
    records: HashedWriter,
    model_input_hashes: HashedWriter,
}

impl SplitWriters {
    fn create(root: &Path, split: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            group_ids: HashedWriter::create(root, format!("{split}-group-ids.bin"))?,
            source_candidate_indices: HashedWriter::create(
                root,
                format!("{split}-source-candidate-indices.bin"),
            )?,
            action_hashes: HashedWriter::create(root, format!("{split}-action-hashes.bin"))?,
            records: HashedWriter::create(root, format!("{split}-records.bin"))?,
            model_input_hashes: HashedWriter::create(
                root,
                format!("{split}-model-input-hashes.bin"),
            )?,
        })
    }

    fn write_group_identity(
        &mut self,
        group_id: u64,
        sources: &[u16],
        hashes: &[u8],
    ) -> Result<(), Box<dyn Error>> {
        self.group_ids.write_u64(group_id)?;
        for source in sources {
            self.source_candidate_indices.write_u16(*source)?;
        }
        self.action_hashes.write_bytes(hashes)?;
        Ok(())
    }

    fn write_record(&mut self, record: &OpponentIntentRecord) -> Result<(), Box<dyn Error>> {
        self.records.write_bytes(&record.to_bytes())?;
        let mut hasher = Hasher::new();
        hasher.update(INPUT_HASH_DOMAIN);
        hasher.update(&record.model_input_bytes());
        self.model_input_hashes
            .write_bytes(hasher.finalize().as_bytes())?;
        Ok(())
    }

    fn finish(self, groups: usize) -> Result<BTreeMap<String, FileSpec>, Box<dyn Error>> {
        let candidates = groups * COHORT_WIDTH;
        Ok(BTreeMap::from([
            (
                "group_ids".to_owned(),
                self.group_ids.finish("<u8", vec![groups])?,
            ),
            (
                "source_candidate_indices".to_owned(),
                self.source_candidate_indices
                    .finish("<u2", vec![groups, COHORT_WIDTH])?,
            ),
            (
                "action_hashes".to_owned(),
                self.action_hashes
                    .finish("|u1", vec![groups, COHORT_WIDTH, 32])?,
            ),
            (
                "records".to_owned(),
                self.records.finish(
                    "|u1",
                    vec![groups, COHORT_WIDTH, OPPONENT_INTENT_RECORD_SIZE],
                )?,
            ),
            (
                "model_input_hashes".to_owned(),
                self.model_input_hashes
                    .finish("|u1", vec![groups, COHORT_WIDTH, 32])?,
            ),
        ]))
        .inspect(|files| {
            debug_assert_eq!(
                files["records"].bytes,
                (candidates * OPPONENT_INTENT_RECORD_SIZE) as u64
            );
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.max_groups_per_split == Some(0) {
        return Err("--max-groups-per-split must be positive".into());
    }
    let (cohort_manifest, cohort_splits) = load_cohort(&args.cohort)?;
    fs::create_dir_all(&args.output_root)?;
    let temporary = args.output_root.join(format!(
        ".tmp-o1-ranking-afterstates-{}-{}",
        std::process::id(),
        unix_millis()?
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir(&temporary)?;

    let result = (|| -> Result<(PathBuf, CacheManifest), Box<dyn Error>> {
        let train = export_split(
            &args.train_dataset,
            DatasetSplit::Train,
            cohort_splits
                .get("train")
                .ok_or("cohort is missing train")?,
            &temporary,
            args.max_groups_per_split,
        )?;
        let validation = export_split(
            &args.validation_dataset,
            DatasetSplit::Validation,
            cohort_splits
                .get("validation")
                .ok_or("cohort is missing validation")?,
            &temporary,
            args.max_groups_per_split,
        )?;
        let splits = BTreeMap::from([
            ("train".to_owned(), train),
            ("validation".to_owned(), validation),
        ]);
        let complete_open_corpus = args.max_groups_per_split.is_none()
            && cohort_manifest.complete_open_corpus
            && splits.values().all(|split| split.complete_open_split);
        let exporter = ExporterIdentity {
            executable_blake3: checksum_file(&env::current_exe()?)?,
            source: source_provenance()?,
        };
        let hidden_information = HiddenInformationBoundary {
            public_candidate_afterstate_only: true,
            champion_history_only: true,
            target_fields_zeroed: true,
            policy_identity_exported: false,
            hidden_post_draft_refill_exported: false,
            hidden_stack_order_exported: false,
            hidden_bag_order_exported: false,
            sealed_test_opened: false,
            gameplay_run: false,
        };
        let cohort_manifest_blake3 = checksum_file(&args.cohort.join("cache.json"))?;
        let scientific_identity = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_schema": CACHE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "complete_open_corpus": complete_open_corpus,
            "cohort_id": cohort_manifest.cache_id,
            "cohort_manifest_blake3": cohort_manifest_blake3,
            "exporter": exporter,
            "hidden_information": hidden_information,
            "splits": splits,
        });
        let cache_id = canonical_blake3(&scientific_identity)?;
        let manifest = CacheManifest {
            schema_version: CACHE_SCHEMA_VERSION,
            cache_schema: CACHE_SCHEMA,
            experiment_id: EXPERIMENT_ID,
            protocol_id: PROTOCOL_ID,
            adr: ADR_ID,
            cache_id: cache_id.clone(),
            complete_open_corpus,
            cohort_id: cohort_manifest.cache_id.clone(),
            cohort_manifest_blake3,
            exporter,
            hidden_information,
            splits,
            scientific_identity,
        };
        write_json_atomic(&temporary.join("cache.json"), &manifest)?;
        let final_root = args.output_root.join(&cache_id);
        if final_root.exists() {
            if fs::read(final_root.join("cache.json"))? != fs::read(temporary.join("cache.json"))? {
                return Err(format!(
                    "afterstate content-address collision at {}",
                    final_root.display()
                )
                .into());
            }
            fs::remove_dir_all(&temporary)?;
        } else {
            fs::rename(&temporary, &final_root)?;
        }
        Ok((final_root, manifest))
    })();

    match result {
        Ok((cache_root, manifest)) => {
            let receipt = json!({
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "cache_id": manifest.cache_id,
                "cache_root": cache_root,
                "cohort_id": manifest.cohort_id,
                "complete_open_corpus": manifest.complete_open_corpus,
                "train_groups": manifest.splits["train"].groups,
                "validation_groups": manifest.splits["validation"].groups,
                "candidates": manifest.splits.values().map(|split| split.candidates).sum::<usize>(),
            });
            write_json_atomic(&args.receipt, &receipt)?;
            println!("{}", serde_json::to_string(&receipt)?);
            Ok(())
        }
        Err(error) => {
            fs::remove_dir_all(&temporary).ok();
            Err(error)
        }
    }
}

fn load_cohort(
    root: &Path,
) -> Result<(CohortManifest, BTreeMap<String, CohortSplit>), Box<dyn Error>> {
    let manifest_path = root.join("cache.json");
    let manifest: CohortManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.schema_version != 1
        || manifest.cache_schema != COHORT_SCHEMA
        || manifest.experiment_id != EXPERIMENT_ID
        || manifest.protocol_id != PROTOCOL_ID
        || manifest.adr != ADR_ID
        || root.file_name().and_then(|value| value.to_str()) != Some(&manifest.cache_id)
        || !manifest.complete_open_corpus
        || manifest
            .splits
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != vec!["train", "validation"]
    {
        return Err("cohort manifest does not satisfy ADR 0188".into());
    }
    let mut splits = BTreeMap::new();
    for (name, raw) in &manifest.splits {
        let expected_groups = if name == "train" {
            TRAIN_GROUPS
        } else {
            VALIDATION_GROUPS
        };
        if raw.groups != expected_groups {
            return Err(format!("{name} cohort group count drifted").into());
        }
        let group_ids =
            read_u64_tensor(root, required_file(raw, "group_ids", "<u8", &[raw.groups])?)?;
        let source_candidate_indices = read_u16_tensor(
            root,
            required_file(
                raw,
                "source_candidate_indices",
                "<u2",
                &[raw.groups, COHORT_WIDTH],
            )?,
        )?;
        let action_hashes = read_u8_tensor(
            root,
            required_file(raw, "action_hashes", "|u1", &[raw.groups, COHORT_WIDTH, 32])?,
        )?;
        if group_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != raw.groups
        {
            return Err(format!("{name} cohort group IDs are duplicated").into());
        }
        for sources in source_candidate_indices.chunks_exact(COHORT_WIDTH) {
            if sources.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(format!("{name} cohort source indices are not ordered").into());
            }
        }
        let group_rows = group_ids
            .iter()
            .enumerate()
            .map(|(row, group_id)| (*group_id, row))
            .collect();
        splits.insert(
            name.clone(),
            CohortSplit {
                dataset_id: raw.dataset_id.clone(),
                groups: raw.groups,
                source_candidate_indices,
                action_hashes,
                group_rows,
            },
        );
    }
    Ok((manifest, splits))
}

fn required_file<'a>(
    split: &'a CohortSplitManifest,
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> Result<&'a InputFileSpec, Box<dyn Error>> {
    let spec = split
        .files
        .get(name)
        .ok_or_else(|| format!("cohort tensor is missing: {name}"))?;
    if spec.dtype != dtype || spec.shape != shape {
        return Err(format!("cohort tensor contract drifted: {name}").into());
    }
    Ok(spec)
}

fn read_tensor_bytes(root: &Path, spec: &InputFileSpec) -> Result<Vec<u8>, Box<dyn Error>> {
    let relative = Path::new(&spec.file);
    if relative.components().count() != 1 {
        return Err("cohort tensor path is not a plain file name".into());
    }
    let path = root.join(relative);
    let bytes = fs::read(&path)?;
    if bytes.len() as u64 != spec.bytes || checksum_file(&path)? != spec.blake3 {
        return Err(format!("cohort tensor failed integrity: {}", spec.file).into());
    }
    Ok(bytes)
}

fn read_u8_tensor(root: &Path, spec: &InputFileSpec) -> Result<Vec<u8>, Box<dyn Error>> {
    read_tensor_bytes(root, spec)
}

fn read_u16_tensor(root: &Path, spec: &InputFileSpec) -> Result<Vec<u16>, Box<dyn Error>> {
    let bytes = read_tensor_bytes(root, spec)?;
    if bytes.len() % 2 != 0 {
        return Err("u16 cohort tensor byte count is odd".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_u64_tensor(root: &Path, spec: &InputFileSpec) -> Result<Vec<u64>, Box<dyn Error>> {
    let bytes = read_tensor_bytes(root, spec)?;
    if bytes.len() % 8 != 0 {
        return Err("u64 cohort tensor byte count is not divisible by eight".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunk width is eight")))
        .collect())
}

fn export_split(
    root: &Path,
    expected_split: DatasetSplit,
    cohort: &CohortSplit,
    output: &Path,
    maximum_groups: Option<usize>,
) -> Result<SplitManifest, Box<dyn Error>> {
    let manifest_path = root.join("dataset.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_graded_oracle_dataset(root, &manifest)?;
    if manifest.split != expected_split || manifest.dataset_id != cohort.dataset_id {
        return Err("graded dataset identity disagrees with the cohort".into());
    }
    let split_name = match expected_split {
        DatasetSplit::Train => "train",
        DatasetSplit::Validation => "validation",
        DatasetSplit::Test | DatasetSplit::Final => {
            return Err("afterstate exporter prohibits sealed and final splits".into());
        }
    };
    let mut writers = SplitWriters::create(output, split_name)?;
    let mut checks = SplitChecks::default();
    let mut groups = 0usize;
    let mut seen = vec![false; cohort.groups];
    let mut stop = false;

    for shard in &manifest.shards {
        if stop {
            break;
        }
        let shard_groups = read_graded_oracle_shard(root, expected_split, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        let mut history = Vec::<HistoricalAction>::with_capacity(80);
        for group in shard_groups {
            if maximum_groups.is_some_and(|maximum| groups >= maximum) {
                stop = true;
                break;
            }
            let row = *cohort
                .group_rows
                .get(&group.group_id)
                .ok_or_else(|| format!("cohort omitted group {}", group.group_id))?;
            if seen[row] {
                return Err(format!("cohort group {} replayed twice", group.group_id).into());
            }
            process_group(
                &mut game,
                &history,
                &group,
                row,
                cohort,
                &mut writers,
                &mut checks,
            )?;
            seen[row] = true;
            let champion = group.candidates[usize::from(group.champion_index)]
                .action
                .to_game_action(&game)?;
            let champion_record = PublicActionRecord::observe(&game, &champion)?;
            let champion_seat = game.current_player();
            game.apply(&champion)?;
            history.push(HistoricalAction {
                seat: champion_seat,
                action: champion_record,
            });
            checks.champion_actions_applied += 1;
            groups += 1;
        }
        if !stop && !game.is_game_over() {
            return Err(format!(
                "source game {} did not contain 80 decisions",
                shard.first_game_index
            )
            .into());
        }
    }

    let complete_open_split =
        maximum_groups.is_none() && groups == cohort.groups && seen.iter().all(|value| *value);
    if maximum_groups.is_none() && !complete_open_split {
        return Err(format!("{split_name} afterstate export did not cover the cohort").into());
    }
    let files = writers.finish(groups)?;
    Ok(SplitManifest {
        split: split_name,
        dataset_id: manifest.dataset_id,
        dataset_manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        cohort_groups: cohort.groups,
        groups,
        candidates: groups * COHORT_WIDTH,
        complete_open_split,
        files,
        checks,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_group(
    game: &mut GameState,
    history: &[HistoricalAction],
    group: &GradedOracleGroup,
    cohort_row: usize,
    cohort: &CohortSplit,
    writers: &mut SplitWriters,
    checks: &mut SplitChecks,
) -> Result<(), Box<dyn Error>> {
    if game.completed_turns() != group.completed_turns
        || game.current_player() != usize::from(group.current_player)
        || PositionRecord::observe(game, group.raw_seed).to_bytes() != group.position.to_bytes()
    {
        return Err(format!("graded replay drifted at group {}", group.group_id).into());
    }
    checks.position_record_checks += 1;
    if *game.public_state().canonical_hash().as_bytes() != group.public_state_hash {
        return Err(format!("public-state hash drifted at group {}", group.group_id).into());
    }
    checks.public_state_hash_checks += 1;

    let sources = cohort.sources(cohort_row);
    let expected_hashes = cohort.hashes(cohort_row);
    writers.write_group_identity(group.group_id, sources, expected_hashes)?;
    for (cohort_position, source_index) in sources.iter().copied().enumerate() {
        let candidate = group
            .candidates
            .get(usize::from(source_index))
            .ok_or_else(|| {
                format!(
                    "cohort source {} is outside group {}",
                    source_index, group.group_id
                )
            })?;
        let action = candidate.action.to_game_action(game)?;
        checks.action_reconstructions += 1;
        let action_hash = canonical_action_hash(&action)?;
        let hash_start = cohort_position * 32;
        if action_hash != candidate.action_hash
            || action_hash.as_slice() != &expected_hashes[hash_start..hash_start + 32]
        {
            return Err(format!(
                "candidate action hash drifted at group {} source {}",
                group.group_id, source_index
            )
            .into());
        }
        checks.action_hash_checks += 1;
        let public_action = PublicActionRecord::observe(game, &action)?;
        let position = PositionRecord::observable_afterstate(game, &action, group.raw_seed)?;
        if position.turn != group.position.turn + 1 {
            return Err("candidate afterstate turn did not advance exactly once".into());
        }
        checks.afterstate_turn_checks += 1;
        if usize::from(position.active_seat) != game.current_player() {
            return Err("candidate afterstate perspective is not the focal player".into());
        }
        checks.afterstate_perspective_checks += 1;
        validate_depleted_market(&position, public_action)?;
        checks.depleted_tile_checks += 1;
        checks.depleted_wildlife_checks += 1;
        let record = build_input_record(
            group.raw_seed,
            group.completed_turns,
            game.current_player(),
            position,
            history,
            public_action,
        )?;
        validate_history(&record)?;
        checks.history_records += 1;
        checks.history_age_checks += usize::from(record.history_count);
        writers.write_record(&record)?;
        checks.model_input_hashes += 1;
        checks.afterstate_records += 1;
    }
    checks.groups_replayed += 1;
    Ok(())
}

fn build_input_record(
    game_index: u64,
    focal_turn: u16,
    focal_seat: usize,
    position: PositionRecord,
    prior: &[HistoricalAction],
    candidate: PublicActionRecord,
) -> Result<OpponentIntentRecord, Box<dyn Error>> {
    let prior_count = prior.len().min(OPPONENT_INTENT_HISTORY_LENGTH - 1);
    let prior_start = prior.len() - prior_count;
    let history_count = prior_count + 1;
    let mut history = [OpponentIntentHistoryEntry::default(); OPPONENT_INTENT_HISTORY_LENGTH];
    for (slot, historical) in prior[prior_start..].iter().enumerate() {
        history[slot] = OpponentIntentHistoryEntry {
            valid: 1,
            age: u8::try_from(history_count - 1 - slot)?,
            relative_seat: u8::try_from((historical.seat + 4 - focal_seat) % 4)?,
            action: historical.action,
        };
    }
    history[history_count - 1] = OpponentIntentHistoryEntry {
        valid: 1,
        age: 0,
        relative_seat: 0,
        action: candidate,
    };
    Ok(OpponentIntentRecord {
        game_index,
        focal_turn: u8::try_from(focal_turn)?,
        focal_seat: u8::try_from(focal_seat)?,
        seat_policy_codes: [NONE; 4],
        position,
        history_count: u8::try_from(history_count)?,
        history,
        opponent_targets: [OpponentActionTarget::default(); 3],
        survival_targets: [TileSurvivalTarget::default(); 4],
        final_scores: [0; 4],
    })
}

fn validate_history(record: &OpponentIntentRecord) -> Result<(), Box<dyn Error>> {
    let count = usize::from(record.history_count);
    if count == 0
        || count > OPPONENT_INTENT_HISTORY_LENGTH
        || record.history[..count]
            .iter()
            .enumerate()
            .any(|(index, entry)| entry.valid != 1 || usize::from(entry.age) != count - 1 - index)
        || record.history[count - 1].relative_seat != 0
        || record.history[count - 1].age != 0
        || record.history[count..]
            .iter()
            .any(|entry| *entry != Default::default())
    {
        return Err("candidate O1 history is invalid".into());
    }
    Ok(())
}

fn validate_depleted_market(
    position: &PositionRecord,
    action: PublicActionRecord,
) -> Result<(), Box<dyn Error>> {
    let missing_tiles = position
        .market_entities
        .iter()
        .enumerate()
        .filter_map(|(slot, entity)| (entity[0] == NONE).then_some(slot))
        .collect::<Vec<_>>();
    let missing_wildlife = position
        .market_entities
        .iter()
        .enumerate()
        .filter_map(|(slot, entity)| (entity[3] == NONE).then_some(slot))
        .collect::<Vec<_>>();
    if missing_tiles != [usize::from(action.tile_slot)]
        || missing_wildlife != [usize::from(action.wildlife_slot)]
    {
        return Err("candidate afterstate depleted unexpected market components".into());
    }
    Ok(())
}

fn canonical_action_hash(action: &TurnAction) -> Result<[u8; 32], Box<dyn Error>> {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-full-legal-action-v1");
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

fn canonical_blake3(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("json")
    ));
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn unix_millis() -> Result<u128, Box<dyn Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(value: u8) -> PublicActionRecord {
        PublicActionRecord {
            tile_slot: value % 4,
            wildlife_slot: value % 4,
            drafted_wildlife: value % 5,
            ..PublicActionRecord::default()
        }
    }

    fn position() -> PositionRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(1),
        )
        .unwrap();
        PositionRecord::observe(&game, 1)
    }

    #[test]
    fn history_keeps_eleven_predecessors_and_candidate_at_age_zero() {
        let prior = (0..20)
            .map(|index| HistoricalAction {
                seat: index % 4,
                action: action(index as u8),
            })
            .collect::<Vec<_>>();
        let record = build_input_record(7, 20, 0, position(), &prior, action(99)).unwrap();

        assert_eq!(record.history_count, 12);
        assert_eq!(record.history[0].age, 11);
        assert_eq!(record.history[0].action, action(9));
        assert_eq!(record.history[11].age, 0);
        assert_eq!(record.history[11].relative_seat, 0);
        assert_eq!(record.history[11].action, action(99));
        validate_history(&record).unwrap();
    }

    #[test]
    fn early_history_preserves_relative_seats_and_ages() {
        let prior = vec![
            HistoricalAction {
                seat: 2,
                action: action(1),
            },
            HistoricalAction {
                seat: 3,
                action: action(2),
            },
        ];
        let record = build_input_record(8, 2, 0, position(), &prior, action(3)).unwrap();

        assert_eq!(record.history_count, 3);
        assert_eq!(
            record.history[..3]
                .iter()
                .map(|entry| (entry.age, entry.relative_seat))
                .collect::<Vec<_>>(),
            vec![(2, 2), (1, 3), (0, 0)]
        );
        validate_history(&record).unwrap();
    }
}
