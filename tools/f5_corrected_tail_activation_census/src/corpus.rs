use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use blake3::Hasher;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    Result, canonical_blake3, hash_file, invalid,
    model::{
        ARTIFACT_SCHEMA_VERSION, DATASET_ID, EXPERIMENT_ID, FEATURE_SCHEMA,
        PRODUCTION_FIRST_GAME_INDEX, PRODUCTION_SHARD_COUNT, PRODUCTION_TOTAL_GAMES,
        PublicStateRecord, ROWS_PER_GAME, RecordKind, TrajectoryPolicy,
        generate_reachable_overflow_witness, generate_representative_game, owned_game_indices,
        replay_record, validate_compiled_contract, validate_generation_environment,
    },
    source::{SourceIdentity, capture_source_identity},
    strict_json, update_framed, write_json_atomic,
};

#[derive(Debug, Clone)]
pub struct GenerateShardConfig {
    pub output_root: PathBuf,
    pub shard_index: usize,
    pub shard_count: usize,
    pub first_game_index: u64,
    pub total_games: usize,
    pub threads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedSchedule {
    pub algorithm: String,
    pub game_domain: String,
    pub policy_domain: String,
    pub overflow_witness_domain: String,
    pub raw_seed_serialized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusContract {
    pub experiment_id: String,
    pub dataset_id: String,
    pub feature_schema: String,
    pub players: usize,
    pub ruleset: String,
    pub state_timing: String,
    pub representative_rows_per_game: usize,
    pub policy_assignment: String,
    pub policies: Vec<String>,
    pub shard_assignment: String,
    pub public_tile_supply_reconstruction: String,
    pub extractor: String,
    pub source_freeze: String,
    pub forbidden_record_fields: Vec<String>,
    pub seed_schedule: SeedSchedule,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardOwnership {
    pub shard_index: usize,
    pub shard_count: usize,
    pub first_game_index: u64,
    pub total_games: usize,
    pub owned_game_indices: Vec<u64>,
    pub owned_game_indices_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadFile {
    pub file: String,
    pub bytes: u64,
    pub blake3: String,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverflowWitnessFile {
    pub payload: PayloadFile,
    pub fixture_search_counter: u64,
    pub public_state_blake3: String,
    pub corrected_tail_blake3: String,
    pub overflow_feature_index: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicCorpusStatistics {
    pub games: usize,
    pub rows: usize,
    pub rows_by_phase: BTreeMap<String, u64>,
    pub rows_by_focal_seat: BTreeMap<String, u64>,
    pub rows_by_policy: BTreeMap<String, u64>,
    pub rows_by_overflow_used: BTreeMap<String, u64>,
    pub raw_feature_emissions: u64,
    pub normalized_feature_activations: u64,
    pub corrected_tail_activations: u64,
    pub public_state_stream_blake3: String,
    pub normalized_feature_stream_blake3: String,
    pub corrected_tail_stream_blake3: String,
    pub representative_record_identity_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifestScientific {
    pub schema_version: u32,
    pub contract: CorpusContract,
    pub source: SourceIdentity,
    pub ownership: ShardOwnership,
    pub records: PayloadFile,
    pub overflow_witness: Option<OverflowWitnessFile>,
    pub statistics: PublicCorpusStatistics,
    pub aggregate_eligible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutableIdentity {
    pub file_name: String,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifestOperational {
    pub executable: ExecutableIdentity,
    pub requested_threads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifest {
    pub scientific: CorpusManifestScientific,
    pub scientific_blake3: String,
    pub operational: CorpusManifestOperational,
}

#[derive(Default)]
struct MutableStatistics {
    games: usize,
    rows: usize,
    rows_by_phase: BTreeMap<String, u64>,
    rows_by_focal_seat: BTreeMap<String, u64>,
    rows_by_policy: BTreeMap<String, u64>,
    rows_by_overflow_used: BTreeMap<String, u64>,
    raw_feature_emissions: u64,
    normalized_feature_activations: u64,
    corrected_tail_activations: u64,
    public_state_hasher: Hasher,
    normalized_feature_hasher: Hasher,
    corrected_tail_hasher: Hasher,
    identity_hasher: Hasher,
}

impl MutableStatistics {
    fn new() -> Self {
        let mut value = Self::default();
        value
            .public_state_hasher
            .update(b"corrected-mid-tail-activation-census-v1/public-state-stream/v1");
        value
            .normalized_feature_hasher
            .update(b"corrected-mid-tail-activation-census-v1/feature-stream/v1");
        value
            .corrected_tail_hasher
            .update(b"corrected-mid-tail-activation-census-v1/tail-stream/v1");
        value
            .identity_hasher
            .update(b"corrected-mid-tail-activation-census-v1/record-identity-stream/v1");
        value
    }

    fn observe(&mut self, record: &PublicStateRecord) -> Result<()> {
        if record.provenance.kind != RecordKind::Representative {
            return Err(invalid(
                "representative corpus statistics received a fixture record",
            ));
        }
        self.rows += 1;
        increment(
            &mut self.rows_by_phase,
            record.state.phase.as_str().to_owned(),
        );
        increment(
            &mut self.rows_by_focal_seat,
            record.state.focal_seat.to_string(),
        );
        increment(
            &mut self.rows_by_policy,
            record.provenance.policy.as_str().to_owned(),
        );
        increment(
            &mut self.rows_by_overflow_used,
            if record.state.overflow_used_this_turn {
                "used"
            } else {
                "not_used"
            }
            .to_owned(),
        );
        self.raw_feature_emissions += u64::from(record.raw_feature_emissions);
        self.normalized_feature_activations += u64::from(record.normalized_feature_activations);
        self.corrected_tail_activations += u64::try_from(record.corrected_tail_features.len())?;
        update_framed(
            &mut self.public_state_hasher,
            record.public_state_blake3.as_bytes(),
        );
        update_framed(
            &mut self.normalized_feature_hasher,
            record.normalized_features_blake3.as_bytes(),
        );
        update_framed(
            &mut self.corrected_tail_hasher,
            record.corrected_tail_blake3.as_bytes(),
        );
        update_framed(
            &mut self.identity_hasher,
            &serde_json::to_vec(&record.provenance)?,
        );
        Ok(())
    }

    fn finish(self) -> PublicCorpusStatistics {
        PublicCorpusStatistics {
            games: self.games,
            rows: self.rows,
            rows_by_phase: self.rows_by_phase,
            rows_by_focal_seat: self.rows_by_focal_seat,
            rows_by_policy: self.rows_by_policy,
            rows_by_overflow_used: self.rows_by_overflow_used,
            raw_feature_emissions: self.raw_feature_emissions,
            normalized_feature_activations: self.normalized_feature_activations,
            corrected_tail_activations: self.corrected_tail_activations,
            public_state_stream_blake3: self.public_state_hasher.finalize().to_hex().to_string(),
            normalized_feature_stream_blake3: self
                .normalized_feature_hasher
                .finalize()
                .to_hex()
                .to_string(),
            corrected_tail_stream_blake3: self
                .corrected_tail_hasher
                .finalize()
                .to_hex()
                .to_string(),
            representative_record_identity_blake3: self
                .identity_hasher
                .finalize()
                .to_hex()
                .to_string(),
        }
    }
}

struct TemporaryRoot {
    path: PathBuf,
    published: bool,
}

impl Drop for TemporaryRoot {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

pub fn generate_shard(config: &GenerateShardConfig) -> Result<CorpusManifest> {
    validate_compiled_contract()?;
    validate_generation_environment()?;
    validate_generate_config(config)?;
    if config.output_root.exists() {
        return Err(invalid(format!(
            "output root already exists: {}",
            config.output_root.display()
        )));
    }
    let owned = owned_game_indices(
        config.first_game_index,
        config.total_games,
        config.shard_index,
        config.shard_count,
    )?;
    if owned.is_empty() {
        return Err(invalid("shard owns no representative games"));
    }
    let source_before = capture_source_identity()?;
    let parent = config
        .output_root
        .parent()
        .ok_or_else(|| invalid("output root requires a parent directory"))?;
    fs::create_dir_all(parent)?;
    let name = config
        .output_root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("output root file name must be UTF-8"))?;
    let temporary_path = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary_path.exists() {
        return Err(invalid(format!(
            "temporary output root already exists: {}",
            temporary_path.display()
        )));
    }
    fs::create_dir(&temporary_path)?;
    let mut temporary = TemporaryRoot {
        path: temporary_path,
        published: false,
    };

    let thread_count = if config.threads == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        config.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()?;
    let mut games = pool.install(|| {
        owned
            .par_iter()
            .map(|game_index| {
                generate_representative_game(*game_index).map(|records| (*game_index, records))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    games.sort_by_key(|(game_index, _)| *game_index);

    let records_path = temporary.path.join("records.jsonl");
    let mut writer = BufWriter::new(File::create(&records_path)?);
    let mut statistics = MutableStatistics::new();
    for (game_index, records) in &games {
        if records.len() != ROWS_PER_GAME {
            return Err(invalid(format!(
                "game {game_index} emitted {} records instead of {ROWS_PER_GAME}",
                records.len()
            )));
        }
        for record in records {
            replay_record(record)?;
            statistics.observe(record)?;
            serde_json::to_writer(&mut writer, record)?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    statistics.games = games.len();
    let statistics = statistics.finish();
    validate_statistics_shape(&statistics, &owned)?;
    let records = PayloadFile {
        file: "records.jsonl".to_owned(),
        bytes: fs::metadata(&records_path)?.len(),
        blake3: hash_file(&records_path)?,
        rows: statistics.rows,
    };

    let overflow_witness = if config.shard_index == 0 {
        let witness = generate_reachable_overflow_witness()?;
        replay_record(&witness)?;
        let path = temporary.path.join("overflow-witness.json");
        write_json_atomic(&path, &witness)?;
        Some(OverflowWitnessFile {
            payload: PayloadFile {
                file: "overflow-witness.json".to_owned(),
                bytes: fs::metadata(&path)?.len(),
                blake3: hash_file(&path)?,
                rows: 1,
            },
            fixture_search_counter: witness
                .provenance
                .fixture_search_counter
                .ok_or_else(|| invalid("overflow witness has no search counter"))?,
            public_state_blake3: witness.public_state_blake3,
            corrected_tail_blake3: witness.corrected_tail_blake3,
            overflow_feature_index: 11_230,
        })
    } else {
        None
    };

    let source_after = capture_source_identity()?;
    if source_before != source_after {
        return Err(invalid(
            "source bundle changed while corrected-record corpus was generated",
        ));
    }
    let ownership = ShardOwnership {
        shard_index: config.shard_index,
        shard_count: config.shard_count,
        first_game_index: config.first_game_index,
        total_games: config.total_games,
        owned_game_indices_blake3: canonical_blake3(&owned)?,
        owned_game_indices: owned,
    };
    let scientific = CorpusManifestScientific {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        contract: corpus_contract(),
        source: source_before,
        ownership,
        records,
        overflow_witness,
        statistics,
        aggregate_eligible: is_production_contract(config),
    };
    let manifest = CorpusManifest {
        scientific_blake3: canonical_blake3(&scientific)?,
        scientific,
        operational: CorpusManifestOperational {
            executable: executable_identity()?,
            requested_threads: config.threads,
        },
    };
    write_json_atomic(&temporary.path.join("manifest.json"), &manifest)?;
    validate_corpus_manifest(&temporary.path)?;
    fs::rename(&temporary.path, &config.output_root)?;
    temporary.published = true;
    Ok(manifest)
}

pub fn read_manifest(root: &Path) -> Result<CorpusManifest> {
    Ok(strict_json::from_reader(BufReader::new(File::open(
        root.join("manifest.json"),
    )?))?)
}

pub fn validate_corpus_manifest(root: &Path) -> Result<CorpusManifest> {
    validate_compiled_contract()?;
    let manifest = read_manifest(root)?;
    if manifest.scientific.schema_version != ARTIFACT_SCHEMA_VERSION
        || manifest.scientific.contract != corpus_contract()
        || manifest.scientific_blake3 != canonical_blake3(&manifest.scientific)?
    {
        return Err(invalid(
            "corrected-record manifest scientific contract drifted",
        ));
    }
    let current_source = capture_source_identity()?;
    if current_source != manifest.scientific.source {
        return Err(invalid(
            "corrected-record manifest source bundle differs from current source",
        ));
    }
    let ownership = &manifest.scientific.ownership;
    let expected_owned = owned_game_indices(
        ownership.first_game_index,
        ownership.total_games,
        ownership.shard_index,
        ownership.shard_count,
    )?;
    if ownership.owned_game_indices != expected_owned
        || ownership.owned_game_indices_blake3 != canonical_blake3(&expected_owned)?
        || manifest.scientific.aggregate_eligible
            != (ownership.first_game_index == PRODUCTION_FIRST_GAME_INDEX
                && ownership.total_games == PRODUCTION_TOTAL_GAMES
                && ownership.shard_count == PRODUCTION_SHARD_COUNT)
    {
        return Err(invalid("corrected-record shard ownership drifted"));
    }
    validate_payload(root, &manifest.scientific.records)?;
    if manifest.scientific.records.rows != expected_owned.len() * ROWS_PER_GAME
        || manifest.scientific.statistics.rows != manifest.scientific.records.rows
        || manifest.scientific.statistics.games != expected_owned.len()
    {
        return Err(invalid(
            "corrected-record manifest row or game count drifted",
        ));
    }
    match (
        ownership.shard_index == 0,
        manifest.scientific.overflow_witness.as_ref(),
    ) {
        (true, Some(witness)) => {
            validate_payload(root, &witness.payload)?;
            if witness.payload.rows != 1 || witness.overflow_feature_index != 11_230 {
                return Err(invalid("overflow witness manifest contract drifted"));
            }
        }
        (false, None) => {}
        _ => {
            return Err(invalid(
                "exactly shard zero must carry the reachable overflow witness",
            ));
        }
    }
    let expected_files = {
        let mut files = BTreeSet::from([
            "manifest.json".to_owned(),
            manifest.scientific.records.file.clone(),
        ]);
        if let Some(witness) = &manifest.scientific.overflow_witness {
            files.insert(witness.payload.file.clone());
        }
        files
    };
    let actual_files = fs::read_dir(root)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<BTreeSet<_>>>()?;
    if actual_files != expected_files {
        return Err(invalid(
            "corrected-record corpus file set differs from its manifest",
        ));
    }
    Ok(manifest)
}

pub(crate) fn read_representative_records(
    root: &Path,
    manifest: &CorpusManifest,
) -> Result<Vec<PublicStateRecord>> {
    let path = root.join(&manifest.scientific.records.file);
    let reader = BufReader::new(File::open(path)?);
    let mut records = Vec::with_capacity(manifest.scientific.records.rows);
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            return Err(invalid(format!(
                "representative record line {} is empty",
                line_index + 1
            )));
        }
        records.push(strict_json::from_str(&line).map_err(|error| {
            invalid(format!(
                "representative record line {} is malformed: {error}",
                line_index + 1
            ))
        })?);
    }
    if records.len() != manifest.scientific.records.rows {
        return Err(invalid(format!(
            "representative payload has {} rows, expected {}",
            records.len(),
            manifest.scientific.records.rows
        )));
    }
    Ok(records)
}

pub(crate) fn read_overflow_witness(
    root: &Path,
    manifest: &CorpusManifest,
) -> Result<Option<PublicStateRecord>> {
    manifest
        .scientific
        .overflow_witness
        .as_ref()
        .map(|witness| {
            Ok(strict_json::from_reader(BufReader::new(File::open(
                root.join(&witness.payload.file),
            )?))?)
        })
        .transpose()
}

pub(crate) fn statistics_for_records(
    records: &[PublicStateRecord],
    games: usize,
) -> Result<PublicCorpusStatistics> {
    let mut statistics = MutableStatistics::new();
    for record in records {
        statistics.observe(record)?;
    }
    statistics.games = games;
    Ok(statistics.finish())
}

fn validate_generate_config(config: &GenerateShardConfig) -> Result<()> {
    if config.total_games == 0
        || config.first_game_index != PRODUCTION_FIRST_GAME_INDEX
        || config.shard_count != PRODUCTION_SHARD_COUNT
        || config.shard_index >= config.shard_count
    {
        return Err(invalid(
            "generation requires first game 0, exactly four shards, positive games, and an in-range shard index",
        ));
    }
    let _ = config
        .first_game_index
        .checked_add(u64::try_from(config.total_games)?)
        .ok_or_else(|| invalid("generation game range overflows u64"))?;
    Ok(())
}

fn validate_statistics_shape(statistics: &PublicCorpusStatistics, owned: &[u64]) -> Result<()> {
    if statistics.games != owned.len() || statistics.rows != owned.len() * ROWS_PER_GAME {
        return Err(invalid(
            "representative statistics game or row count drifted",
        ));
    }
    let games = u64::try_from(owned.len())?;
    let expected_phase = BTreeMap::from([
        ("opening".to_owned(), games * 4),
        ("early".to_owned(), games * 16),
        ("middle".to_owned(), games * 32),
        ("late".to_owned(), games * 28),
    ]);
    if statistics.rows_by_phase != expected_phase {
        return Err(invalid("representative phase coverage drifted"));
    }
    let rows = u64::try_from(statistics.rows)?;
    if statistics.rows_by_focal_seat.values().sum::<u64>() != rows
        || statistics.rows_by_policy.values().sum::<u64>() != rows
        || statistics.rows_by_overflow_used.values().sum::<u64>() != rows
    {
        return Err(invalid("representative slice counts do not sum to rows"));
    }
    Ok(())
}

fn validate_payload(root: &Path, payload: &PayloadFile) -> Result<()> {
    if payload.file.contains('/')
        || payload.file.contains('\\')
        || payload.file == "manifest.json"
        || payload.rows == 0
    {
        return Err(invalid("payload file declaration is invalid"));
    }
    let path = root.join(&payload.file);
    if fs::metadata(&path)?.len() != payload.bytes || hash_file(&path)? != payload.blake3 {
        return Err(invalid(format!(
            "payload bytes or BLAKE3 drifted: {}",
            payload.file
        )));
    }
    Ok(())
}

fn corpus_contract() -> CorpusContract {
    CorpusContract {
        experiment_id: EXPERIMENT_ID.to_owned(),
        dataset_id: DATASET_ID.to_owned(),
        feature_schema: FEATURE_SCHEMA.to_owned(),
        players: 4,
        ruleset: "AAAAA; habitat bonus labels and score outputs excluded".to_owned(),
        state_timing: ("pre-draft after applying the legal free three-of-a-kind replacement once")
            .to_owned(),
        representative_rows_per_game: ROWS_PER_GAME,
        policy_assignment: "(game_index + absolute_seat) mod 4".to_owned(),
        policies: TrajectoryPolicy::ALL
            .into_iter()
            .map(|policy| policy.as_str().to_owned())
            .collect(),
        shard_assignment: "(game_index - first_game_index) mod shard_count".to_owned(),
        public_tile_supply_reconstruction:
            ("standard 85-tile multiset minus public non-starter board tiles and visible market")
                .to_owned(),
        extractor: ("cascadia_ai::nnue::BagInfo::from_game_for_player plus \
             cascadia_ai::nnue::extract_features_with_bag compiled with \
             legacy-mid-v4-fixed-v1")
            .to_owned(),
        source_freeze:
            ("sorted content hashes over this experiment, ADR/preregistration, legacy AI, and \
             legacy core sources")
                .to_owned(),
        forbidden_record_fields: vec![
            "tile bag order".to_owned(),
            "wildlife bag order".to_owned(),
            "future refill realization".to_owned(),
            "future action".to_owned(),
            "score or terminal target".to_owned(),
            "teacher output".to_owned(),
            "habitat bonus label".to_owned(),
            "raw RNG seed".to_owned(),
        ],
        seed_schedule: SeedSchedule {
            algorithm: "u64 little-endian prefix of BLAKE3(domain || values_le_u64)".to_owned(),
            game_domain: String::from_utf8_lossy(
                b"corrected-mid-tail-activation-census-v1/game-seed/v1",
            )
            .into_owned(),
            policy_domain: String::from_utf8_lossy(
                b"corrected-mid-tail-activation-census-v1/policy-seed/v1",
            )
            .into_owned(),
            overflow_witness_domain: String::from_utf8_lossy(
                b"corrected-mid-tail-activation-census-v1/reachable-overflow-witness/v1",
            )
            .into_owned(),
            raw_seed_serialized: false,
        },
    }
}

fn is_production_contract(config: &GenerateShardConfig) -> bool {
    config.first_game_index == PRODUCTION_FIRST_GAME_INDEX
        && config.total_games == PRODUCTION_TOTAL_GAMES
        && config.shard_count == PRODUCTION_SHARD_COUNT
}

fn executable_identity() -> Result<ExecutableIdentity> {
    let path = std::env::current_exe()?;
    Ok(ExecutableIdentity {
        file_name: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("f5-corrected-tail-activation-census")
            .to_owned(),
        bytes: fs::metadata(&path)?.len(),
        blake3: hash_file(&path)?,
    })
}

fn increment(map: &mut BTreeMap<String, u64>, key: String) {
    *map.entry(key).or_default() += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_contract_is_exactly_four_disjoint_shards() {
        for shard_index in 0..4 {
            let config = GenerateShardConfig {
                output_root: PathBuf::from("/tmp/unused"),
                shard_index,
                shard_count: 4,
                first_game_index: 0,
                total_games: 1_024,
                threads: 1,
            };
            assert!(is_production_contract(&config));
            assert_eq!(
                owned_game_indices(0, 1_024, shard_index, 4).unwrap().len(),
                256
            );
        }
    }

    #[test]
    fn generation_contract_rejects_non_four_shard_or_shifted_ranges() {
        let base = GenerateShardConfig {
            output_root: PathBuf::from("/tmp/unused"),
            shard_index: 0,
            shard_count: 4,
            first_game_index: 0,
            total_games: 4,
            threads: 1,
        };
        validate_generate_config(&base).unwrap();

        let mut three_shards = base.clone();
        three_shards.shard_count = 3;
        assert!(validate_generate_config(&three_shards).is_err());

        let mut shifted = base;
        shifted.first_game_index = 1;
        assert!(validate_generate_config(&shifted).is_err());
    }

    #[test]
    fn corpus_contract_explicitly_excludes_hidden_information() {
        let contract = corpus_contract();
        assert!(
            contract
                .forbidden_record_fields
                .contains(&"tile bag order".to_owned())
        );
        assert!(
            contract
                .forbidden_record_fields
                .contains(&"raw RNG seed".to_owned())
        );
        assert!(!contract.seed_schedule.raw_seed_serialized);
    }
}
