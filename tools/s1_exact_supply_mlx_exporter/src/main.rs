use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use blake3::Hasher;
use cascadia_data::{
    CanonicalTileArchetype, DatasetSplit, ExactSemanticSupply, FrontierTerrainRequirements,
    GradedOracleDatasetManifest, GradedOracleGroup, PositionRecord, SemanticArchetypeCatalog,
    standard_semantic_archetype_catalog, validate_graded_oracle_dataset,
};
use cascadia_game::{
    HexCoord, PublicSupply, Rotation, STANDARD_TILES, Terrain, Tile, TileId, Wildlife, WildlifeMask,
};
use cascadia_provenance::{checksum_file, source_provenance};
use serde::Serialize;
use serde_json::{Value, json};

const CACHE_SCHEMA_VERSION: u16 = 1;
const CACHE_SCHEMA: &str = "s1-exact-supply-mlx-cache-v1";
const EXPERIMENT_ID: &str = "exact-semantic-supply-learned-comparison-v1";
const PROTOCOL_ID: &str = "s1-exact-semantic-supply-mlx-comparison-v1";
const ADR_ID: &str = "0147";
const ARCHETYPE_COUNT: usize = 75;
const EXACT_VALUE_COUNT: usize = 83;
const LEGACY_SUPPLY_COUNT: usize = 30;
const NONE: u8 = u8::MAX;
const FRONTIER_NONE: u8 = 5;
const SUPPLY_MAGIC: &[u8; 8] = b"CSSSUP1\0";
const TRAIN_GROUPS: usize = 560;
const TRAIN_CANDIDATES: usize = 2_135_111;
const VALIDATION_GROUPS: usize = 240;
const VALIDATION_CANDIDATES: usize = 860_203;

const HELP: &str = concat!(
    "Usage: s1_exact_supply_mlx_exporter \\\n",
    "  --train-dataset PATH --validation-dataset PATH \\\n",
    "  --output-root PATH --receipt PATH \\\n",
    "  [--max-groups-per-split N]\n\n",
    "The optional group bound is smoke-only. A bounded cache is marked incomplete\n",
    "and is rejected by ADR 0147 production preflight."
);

#[derive(Debug, Clone)]
struct Args {
    train_dataset: PathBuf,
    validation_dataset: PathBuf,
    output_root: PathBuf,
    receipt: PathBuf,
    max_groups_per_split: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FileSpec {
    file: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    split: &'static str,
    dataset_root: String,
    dataset_id: String,
    dataset_manifest_blake3: String,
    source_v2_blake3: String,
    groups: usize,
    candidates: usize,
    complete_open_split: bool,
    files: BTreeMap<String, FileSpec>,
    checks: SplitChecks,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SplitChecks {
    csssup_round_trips: usize,
    legacy_parity_groups: usize,
    wildlife_parity_groups: usize,
    tile_count_conservation_groups: usize,
    drawable_conservation_groups: usize,
    hidden_exclusion_count_groups: usize,
    staged_legacy_tile_parity_candidates: usize,
    staged_wildlife_parity_candidates: usize,
    market_tile_identity_candidates: usize,
    frontier_compatibility_candidates: usize,
    hidden_order_fields_read: usize,
    excluded_tile_identity_fields_read: usize,
    future_refill_fields_read: usize,
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
    catalog_blake3: String,
    catalog: Vec<cascadia_data::SemanticArchetypeDefinition>,
    collision_witness: Value,
    exporter: ExporterIdentity,
    splits: BTreeMap<String, SplitManifest>,
    hidden_information: HiddenInformationBoundary,
    scientific_identity: Value,
}

#[derive(Debug, Clone, Serialize)]
struct ExporterIdentity {
    executable_blake3: String,
    source: cascadia_provenance::SourceProvenance,
}

#[derive(Debug, Clone, Serialize)]
struct HiddenInformationBoundary {
    public_position_records_only: bool,
    public_supply_only: bool,
    hidden_stack_order_read: bool,
    hidden_wildlife_order_read: bool,
    excluded_tile_identities_read: bool,
    future_refills_read: bool,
    sealed_test_opened: bool,
    gameplay_opened: bool,
}

#[derive(Debug, Clone)]
struct DerivedSupply {
    exact_values: [u8; EXACT_VALUE_COUNT],
    canonical_hash: [u8; 32],
}

#[derive(Debug, Clone)]
struct SplitBuffers {
    group_ids: Vec<u8>,
    public_state_hashes: Vec<u8>,
    exact_supply_values: Vec<u8>,
    exact_supply_hashes: Vec<u8>,
    candidate_offsets: Vec<u8>,
    staged_wildlife_counts: Vec<u8>,
    selected_archetype_ids: Vec<u8>,
    frontier_requirements: Vec<u8>,
    selected_compatibility: Vec<u8>,
    candidate_identity_hashes: Vec<u8>,
    groups: usize,
    candidates: usize,
    checks: SplitChecks,
}

impl SplitBuffers {
    fn new() -> Self {
        let mut candidate_offsets = Vec::with_capacity(8);
        candidate_offsets.extend_from_slice(&0u64.to_le_bytes());
        Self {
            group_ids: Vec::new(),
            public_state_hashes: Vec::new(),
            exact_supply_values: Vec::new(),
            exact_supply_hashes: Vec::new(),
            candidate_offsets,
            staged_wildlife_counts: Vec::new(),
            selected_archetype_ids: Vec::new(),
            frontier_requirements: Vec::new(),
            selected_compatibility: Vec::new(),
            candidate_identity_hashes: Vec::new(),
            groups: 0,
            candidates: 0,
            checks: SplitChecks::default(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let catalog = standard_semantic_archetype_catalog();
    if catalog.len() != ARCHETYPE_COUNT {
        return Err(format!(
            "semantic catalog has {} archetypes; expected {ARCHETYPE_COUNT}",
            catalog.len()
        )
        .into());
    }

    let temporary = args.output_root.join(format!(
        ".tmp-s1-exact-supply-{}-{}",
        std::process::id(),
        unix_millis()?
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir_all(&temporary)?;

    let result = (|| -> Result<(PathBuf, CacheManifest), Box<dyn Error>> {
        let train = export_split(
            &args.train_dataset,
            DatasetSplit::Train,
            &temporary,
            args.max_groups_per_split,
            catalog,
        )?;
        let validation = export_split(
            &args.validation_dataset,
            DatasetSplit::Validation,
            &temporary,
            args.max_groups_per_split,
            catalog,
        )?;
        let complete_open_corpus = train.complete_open_split && validation.complete_open_split;
        let executable_blake3 = checksum_file(&env::current_exe()?)?;
        let source = source_provenance()?;
        let collision_witness = build_collision_witness(catalog)?;
        let splits = BTreeMap::from([
            ("train".to_owned(), train),
            ("validation".to_owned(), validation),
        ]);
        let scientific_identity = scientific_identity(
            complete_open_corpus,
            catalog,
            &executable_blake3,
            &source,
            &splits,
            &collision_witness,
        )?;
        let cache_id = canonical_blake3(&scientific_identity)?;
        let manifest = CacheManifest {
            schema_version: CACHE_SCHEMA_VERSION,
            cache_schema: CACHE_SCHEMA,
            experiment_id: EXPERIMENT_ID,
            protocol_id: PROTOCOL_ID,
            adr: ADR_ID,
            cache_id: cache_id.clone(),
            complete_open_corpus,
            catalog_blake3: catalog.canonical_blake3().to_hex().to_string(),
            catalog: catalog.definitions().to_vec(),
            collision_witness,
            exporter: ExporterIdentity {
                executable_blake3,
                source,
            },
            splits,
            hidden_information: HiddenInformationBoundary {
                public_position_records_only: true,
                public_supply_only: true,
                hidden_stack_order_read: false,
                hidden_wildlife_order_read: false,
                excluded_tile_identities_read: false,
                future_refills_read: false,
                sealed_test_opened: false,
                gameplay_opened: false,
            },
            scientific_identity,
        };
        write_json_atomic(&temporary.join("cache.json"), &manifest)?;
        let destination = args.output_root.join(&cache_id);
        if destination.exists() {
            let existing = fs::read(destination.join("cache.json"))?;
            let generated = fs::read(temporary.join("cache.json"))?;
            if existing != generated {
                return Err("existing cache directory has different manifest bytes".into());
            }
            fs::remove_dir_all(&temporary)?;
        } else {
            fs::rename(&temporary, &destination)?;
        }
        Ok((destination, manifest))
    })();

    if result.is_err() && temporary.exists() {
        fs::remove_dir_all(&temporary).ok();
    }
    let (cache_root, manifest) = result?;
    let receipt = json!({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "cache_id": manifest.cache_id,
        "cache_root": cache_root,
        "complete_open_corpus": manifest.complete_open_corpus,
        "train_groups": manifest.splits["train"].groups,
        "train_candidates": manifest.splits["train"].candidates,
        "validation_groups": manifest.splits["validation"].groups,
        "validation_candidates": manifest.splits["validation"].candidates,
    });
    write_json_atomic(&args.receipt, &receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn export_split(
    root: &Path,
    expected_split: DatasetSplit,
    output: &Path,
    max_groups: Option<usize>,
    catalog: &SemanticArchetypeCatalog,
) -> Result<SplitManifest, Box<dyn Error>> {
    let manifest_path = root.join("dataset.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.split != expected_split {
        return Err(format!(
            "dataset {} has split {}; expected {}",
            root.display(),
            manifest.split.id(),
            expected_split.id()
        )
        .into());
    }
    if !matches!(
        expected_split,
        DatasetSplit::Train | DatasetSplit::Validation
    ) {
        return Err("S1 exporter accepts only open train and validation splits".into());
    }
    validate_graded_oracle_dataset(root, &manifest)?;
    validate_ruleset(&manifest)?;

    let mut buffers = SplitBuffers::new();
    'shards: for shard in &manifest.shards {
        let groups = cascadia_data::read_graded_oracle_shard(root, expected_split, shard)?;
        for group in groups {
            if max_groups.is_some_and(|limit| buffers.groups >= limit) {
                break 'shards;
            }
            append_group(&mut buffers, &group, catalog)?;
        }
    }

    let (expected_groups, expected_candidates) = match expected_split {
        DatasetSplit::Train => (TRAIN_GROUPS, TRAIN_CANDIDATES),
        DatasetSplit::Validation => (VALIDATION_GROUPS, VALIDATION_CANDIDATES),
        _ => unreachable!(),
    };
    let complete_open_split = max_groups.is_none()
        && buffers.groups == expected_groups
        && buffers.candidates == expected_candidates
        && buffers.groups == manifest.total_groups
        && buffers.candidates == manifest.total_records;
    if max_groups.is_none() && !complete_open_split {
        return Err(format!(
            "{} export coverage drifted: groups {}/{}, candidates {}/{}",
            expected_split.id(),
            buffers.groups,
            expected_groups,
            buffers.candidates,
            expected_candidates
        )
        .into());
    }
    if buffers.groups == 0 || buffers.candidates == 0 {
        return Err(format!("{} export is empty", expected_split.id()).into());
    }
    if buffers.checks.hidden_order_fields_read != 0
        || buffers.checks.excluded_tile_identity_fields_read != 0
        || buffers.checks.future_refill_fields_read != 0
    {
        return Err("hidden-information counters must remain zero".into());
    }

    let split = expected_split.id();
    let files = write_split_files(output, split, &buffers)?;
    Ok(SplitManifest {
        split,
        dataset_root: root
            .canonicalize()
            .unwrap_or_else(|_| root.to_path_buf())
            .display()
            .to_string(),
        dataset_id: manifest.dataset_id,
        dataset_manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        source_v2_blake3: manifest.provenance.v2_source_blake3,
        groups: buffers.groups,
        candidates: buffers.candidates,
        complete_open_split,
        files,
        checks: buffers.checks,
    })
}

fn validate_ruleset(manifest: &GradedOracleDatasetManifest) -> Result<(), Box<dyn Error>> {
    let game = manifest.game;
    if game.player_count != 4
        || game.habitat_bonuses
        || game.scoring_cards.bear != cascadia_game::ScoringVariant::A
        || game.scoring_cards.elk != cascadia_game::ScoringVariant::A
        || game.scoring_cards.salmon != cascadia_game::ScoringVariant::A
        || game.scoring_cards.hawk != cascadia_game::ScoringVariant::A
        || game.scoring_cards.fox != cascadia_game::ScoringVariant::A
    {
        return Err("graded-oracle dataset is not four-player AAAAA without bonuses".into());
    }
    Ok(())
}

fn append_group(
    buffers: &mut SplitBuffers,
    group: &GradedOracleGroup,
    catalog: &SemanticArchetypeCatalog,
) -> Result<(), Box<dyn Error>> {
    let derived = derive_supply(group, catalog)?;
    buffers
        .group_ids
        .extend_from_slice(&group.group_id.to_le_bytes());
    buffers
        .public_state_hashes
        .extend_from_slice(&group.public_state_hash);
    buffers
        .exact_supply_values
        .extend_from_slice(&derived.exact_values);
    buffers
        .exact_supply_hashes
        .extend_from_slice(&derived.canonical_hash);

    let parent_legacy = legacy_supply_bytes(group.public_supply);
    let active_board = active_board_tiles(&group.position)?;
    let mut candidate_hasher = Hasher::new();
    candidate_hasher.update(b"S1MLXCAND1\0");
    candidate_hasher.update(&group.group_id.to_le_bytes());
    candidate_hasher.update(&(group.candidates.len() as u64).to_le_bytes());

    for candidate in &group.candidates {
        let staged = candidate.action.staged_public_supply;
        if staged[5..] != parent_legacy[5..] {
            return Err(format!(
                "group {} candidate changed hidden tile marginals during a wildlife-only prelude",
                group.group_id
            )
            .into());
        }
        buffers.checks.staged_legacy_tile_parity_candidates += 1;
        let staged_wildlife: [u8; 5] = staged[..5].try_into()?;
        validate_staged_wildlife(
            group,
            &candidate.action.staged_market_entities,
            staged_wildlife,
        )?;
        buffers.checks.staged_wildlife_parity_candidates += 1;

        let tile_id = usize::from(candidate.action.tile_id);
        let tile = *STANDARD_TILES.get(tile_id).ok_or_else(|| {
            format!(
                "group {} candidate names invalid standard tile ID {}",
                group.group_id, tile_id
            )
        })?;
        validate_candidate_tile(candidate, tile)?;
        buffers.checks.market_tile_identity_candidates += 1;
        let reference = catalog.reference_for_tile(tile)?;
        let requirements = frontier_requirements(
            &active_board,
            HexCoord::new(candidate.action.tile_q, candidate.action.tile_r),
        );
        let compatibility = reference.frontier_compatibility(tile, requirements)?;
        buffers.checks.frontier_compatibility_candidates += 1;

        let requirement_bytes = requirements
            .neighbor_facing_terrains
            .map(|terrain| terrain.map_or(FRONTIER_NONE, |value| value as u8));
        let compatibility_bytes = [
            compatibility.matching_edges_by_rotation[0],
            compatibility.matching_edges_by_rotation[1],
            compatibility.matching_edges_by_rotation[2],
            compatibility.matching_edges_by_rotation[3],
            compatibility.matching_edges_by_rotation[4],
            compatibility.matching_edges_by_rotation[5],
            compatibility.all_present_match_rotation_mask,
            compatibility.best_matching_edges,
        ];
        buffers
            .staged_wildlife_counts
            .extend_from_slice(&staged_wildlife);
        buffers
            .selected_archetype_ids
            .push(reference.archetype_id.code() as u8);
        buffers
            .frontier_requirements
            .extend_from_slice(&requirement_bytes);
        buffers
            .selected_compatibility
            .extend_from_slice(&compatibility_bytes);
        candidate_hasher.update(&candidate.action_hash);
        candidate_hasher.update(&staged_wildlife);
        candidate_hasher.update(&[reference.archetype_id.code() as u8]);
        candidate_hasher.update(&requirement_bytes);
        candidate_hasher.update(&compatibility_bytes);
    }
    buffers
        .candidate_identity_hashes
        .extend_from_slice(candidate_hasher.finalize().as_bytes());
    buffers.groups += 1;
    buffers.candidates += group.candidates.len();
    buffers
        .candidate_offsets
        .extend_from_slice(&(buffers.candidates as u64).to_le_bytes());
    buffers.checks.csssup_round_trips += 1;
    buffers.checks.legacy_parity_groups += 1;
    buffers.checks.wildlife_parity_groups += 1;
    buffers.checks.tile_count_conservation_groups += 1;
    buffers.checks.drawable_conservation_groups += 1;
    buffers.checks.hidden_exclusion_count_groups += 1;
    Ok(())
}

fn derive_supply(
    group: &GradedOracleGroup,
    catalog: &SemanticArchetypeCatalog,
) -> Result<DerivedSupply, Box<dyn Error>> {
    let record = &group.position;
    if record.player_count != 4 || record.total_turns != 80 || record.habitat_bonuses {
        return Err(format!(
            "group {} has unsupported public position metadata",
            group.group_id
        )
        .into());
    }
    let mut counts = catalog
        .definitions()
        .iter()
        .map(|definition| definition.standard_tile_count)
        .collect::<Vec<_>>();
    let starter_coords = BTreeSet::from([
        HexCoord::new(0, 0),
        HexCoord::new(0, 1),
        HexCoord::new(1, 0),
    ]);
    let mut standard_board_tiles = 0usize;
    for board in 0..4 {
        let count = usize::from(record.board_counts[board]);
        if count < 3 || count > record.board_entities[board].len() {
            return Err(format!("group {} has invalid board count", group.group_id).into());
        }
        let mut seen = BTreeSet::new();
        for entity in &record.board_entities[board][..count] {
            let coord = HexCoord::new(entity[0] as i8, entity[1] as i8);
            if !seen.insert(coord) {
                return Err(
                    format!("group {} duplicates a board coordinate", group.group_id).into(),
                );
            }
            if starter_coords.contains(&coord) {
                continue;
            }
            remove_semantic_tile(&mut counts, tile_from_board_entity(*entity)?, catalog)?;
            standard_board_tiles += 1;
        }
        if !starter_coords.iter().all(|coord| seen.contains(coord)) {
            return Err(format!("group {} is missing a starter coordinate", group.group_id).into());
        }
    }
    if standard_board_tiles != usize::from(record.turn) {
        return Err(format!(
            "group {} has {} standard board tiles at turn {}",
            group.group_id, standard_board_tiles, record.turn
        )
        .into());
    }

    let mut market_tiles = 0usize;
    for entity in &record.market_entities {
        if entity[0] == NONE {
            continue;
        }
        remove_semantic_tile(&mut counts, tile_from_market_entity(*entity)?, catalog)?;
        market_tiles += 1;
    }
    let unseen: u16 = counts.iter().sum();
    let expected_unseen = 85usize
        .checked_sub(standard_board_tiles + market_tiles)
        .ok_or("visible tile count exceeds the official inventory")?;
    if usize::from(unseen) != expected_unseen {
        return Err(format!("group {} exact unseen count drifted", group.group_id).into());
    }
    let visible_drawable = standard_board_tiles + market_tiles;
    let drawable = usize::from(record.total_turns) + 3;
    let drawable = drawable
        .checked_sub(visible_drawable)
        .ok_or("drawable tile count underflow")? as u16;
    let excluded = unseen
        .checked_sub(drawable)
        .ok_or("drawable count exceeds unseen count")?;
    if excluded != 2 {
        return Err(format!(
            "group {} inferred {excluded} hidden exclusions",
            group.group_id
        )
        .into());
    }
    let wildlife = public_wildlife_counts(record)?;
    if wildlife.map(u16::from) != group.public_supply.wildlife_bag.map(u16::from) {
        return Err(format!("group {} wildlife supply parity failed", group.group_id).into());
    }

    let canonical = canonical_supply_bytes(catalog, wildlife.map(u16::from), &counts, drawable);
    let exact = ExactSemanticSupply::from_canonical_bytes(&canonical)?;
    if exact.to_legacy_public_supply() != group.public_supply {
        return Err(format!("group {} legacy supply parity failed", group.group_id).into());
    }
    if exact.excluded_tile_count() != 2
        || exact.unseen_tile_count() != unseen
        || exact.drawable_tile_count() != drawable
    {
        return Err(format!("group {} canonical supply totals drifted", group.group_id).into());
    }
    let mut exact_values = [0u8; EXACT_VALUE_COUNT];
    exact_values[..5].copy_from_slice(&wildlife);
    for (target, count) in exact_values[5..80].iter_mut().zip(&counts) {
        *target = u8::try_from(*count)?;
    }
    exact_values[80] = u8::try_from(unseen)?;
    exact_values[81] = u8::try_from(drawable)?;
    exact_values[82] = u8::try_from(excluded)?;
    Ok(DerivedSupply {
        exact_values,
        canonical_hash: *exact.canonical_hash().as_bytes(),
    })
}

fn remove_semantic_tile(
    counts: &mut [u16],
    tile: Tile,
    catalog: &SemanticArchetypeCatalog,
) -> Result<(), Box<dyn Error>> {
    let archetype = CanonicalTileArchetype::from_tile(tile);
    let id = catalog
        .id_for_archetype(archetype)
        .ok_or("public tile semantics are outside the official semantic catalog")?;
    counts[id.index()] = counts[id.index()]
        .checked_sub(1)
        .ok_or("public semantic archetype count underflow")?;
    Ok(())
}

fn tile_from_board_entity(entity: [u8; 8]) -> Result<Tile, Box<dyn Error>> {
    if entity[4] >= 6 || entity[5] & !0b1_1111 != 0 {
        return Err("board entity has invalid orientation or wildlife mask".into());
    }
    let terrain_a = decode_terrain(entity[2])?;
    let terrain_b = decode_optional_terrain(entity[3])?;
    Ok(Tile {
        id: TileId(NONE),
        terrain_a,
        terrain_b,
        wildlife: WildlifeMask::from_bits(entity[5]),
        keystone: entity[7] != 0,
    })
}

fn tile_from_market_entity(entity: [u8; 8]) -> Result<Tile, Box<dyn Error>> {
    let terrain_a = decode_terrain(entity[0])?;
    let terrain_b = decode_optional_terrain(entity[1])?;
    if entity[2] & !0b1_1111 != 0 {
        return Err("market entity has invalid wildlife mask".into());
    }
    Ok(Tile {
        id: TileId(NONE),
        terrain_a,
        terrain_b,
        wildlife: WildlifeMask::from_bits(entity[2]),
        keystone: entity[4] != 0,
    })
}

fn validate_candidate_tile(
    candidate: &cascadia_data::GradedOracleCandidate,
    tile: Tile,
) -> Result<(), Box<dyn Error>> {
    let action = &candidate.action;
    if tile.terrain_a as u8 != action.tile_terrain_a
        || tile.terrain_b.map_or(NONE, |terrain| terrain as u8) != action.tile_terrain_b
        || tile.wildlife.bits() != action.tile_wildlife_mask
        || u8::from(tile.keystone) != action.tile_keystone
        || action.rotation >= 6
    {
        return Err("candidate tile identity disagrees with the official catalog".into());
    }
    Ok(())
}

fn public_wildlife_counts(record: &PositionRecord) -> Result<[u8; 5], Box<dyn Error>> {
    let mut counts = [20u8; 5];
    for board in 0..4 {
        for wildlife in Wildlife::ALL {
            counts[wildlife as usize] = counts[wildlife as usize]
                .checked_sub(record.wildlife_counts[board][wildlife as usize])
                .ok_or("placed wildlife exceeds official inventory")?;
        }
    }
    for entity in &record.market_entities {
        if entity[3] == NONE {
            continue;
        }
        let wildlife = decode_wildlife(entity[3])?;
        counts[wildlife as usize] = counts[wildlife as usize]
            .checked_sub(1)
            .ok_or("market wildlife exceeds official inventory")?;
    }
    Ok(counts)
}

fn validate_staged_wildlife(
    group: &GradedOracleGroup,
    staged_market: &[[u8; 8]; 4],
    staged_counts: [u8; 5],
) -> Result<(), Box<dyn Error>> {
    let mut expected = [20u8; 5];
    for board in 0..4 {
        for wildlife in Wildlife::ALL {
            expected[wildlife as usize] = expected[wildlife as usize]
                .checked_sub(group.position.wildlife_counts[board][wildlife as usize])
                .ok_or("staged placed wildlife exceeds official inventory")?;
        }
    }
    for entity in staged_market {
        if entity[3] == NONE {
            continue;
        }
        let wildlife = decode_wildlife(entity[3])?;
        expected[wildlife as usize] = expected[wildlife as usize]
            .checked_sub(1)
            .ok_or("staged market wildlife exceeds official inventory")?;
    }
    if expected != staged_counts {
        return Err(format!("group {} staged wildlife parity failed", group.group_id).into());
    }
    Ok(())
}

fn active_board_tiles(
    record: &PositionRecord,
) -> Result<BTreeMap<HexCoord, (Tile, Rotation)>, Box<dyn Error>> {
    let count = usize::from(record.board_counts[0]);
    let mut board = BTreeMap::new();
    for entity in &record.board_entities[0][..count] {
        let coord = HexCoord::new(entity[0] as i8, entity[1] as i8);
        let rotation =
            Rotation::new(entity[4]).ok_or("board rotation is outside zero through five")?;
        if board
            .insert(coord, (tile_from_board_entity(*entity)?, rotation))
            .is_some()
        {
            return Err("active board duplicates a coordinate".into());
        }
    }
    Ok(board)
}

fn frontier_requirements(
    board: &BTreeMap<HexCoord, (Tile, Rotation)>,
    target: HexCoord,
) -> FrontierTerrainRequirements {
    FrontierTerrainRequirements::new(std::array::from_fn(|edge| {
        board
            .get(&target.neighbor(edge))
            .map(|(tile, rotation)| tile.terrain_on_edge(*rotation, (edge + 3) % 6))
    }))
}

fn canonical_supply_bytes(
    catalog: &SemanticArchetypeCatalog,
    wildlife: [u16; 5],
    counts: &[u16],
    drawable: u16,
) -> Vec<u8> {
    let unseen: u16 = counts.iter().sum();
    let mut bytes = Vec::with_capacity(8 + 2 + 32 + 10 + 6 + counts.len() * 2);
    bytes.extend_from_slice(SUPPLY_MAGIC);
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(catalog.canonical_blake3().as_bytes());
    for count in wildlife {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    bytes.extend_from_slice(&unseen.to_le_bytes());
    bytes.extend_from_slice(&drawable.to_le_bytes());
    bytes.extend_from_slice(&(counts.len() as u16).to_le_bytes());
    for count in counts {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    bytes
}

fn legacy_supply_bytes(supply: PublicSupply) -> [u8; LEGACY_SUPPLY_COUNT] {
    let mut values = [0u8; LEGACY_SUPPLY_COUNT];
    let mut offset = 0;
    for slice in [
        supply.wildlife_bag.as_slice(),
        supply.unseen_tile_terrain_capacity.as_slice(),
        supply.unseen_tile_wildlife_capacity.as_slice(),
        supply.unseen_keystones_by_terrain.as_slice(),
        supply.unseen_dual_terrain_pairs.as_slice(),
    ] {
        values[offset..offset + slice.len()].copy_from_slice(slice);
        offset += slice.len();
    }
    values
}

fn decode_terrain(value: u8) -> Result<Terrain, Box<dyn Error>> {
    Terrain::ALL
        .into_iter()
        .find(|terrain| *terrain as u8 == value)
        .ok_or_else(|| format!("invalid terrain code {value}").into())
}

fn decode_optional_terrain(value: u8) -> Result<Option<Terrain>, Box<dyn Error>> {
    if value == NONE {
        Ok(None)
    } else {
        decode_terrain(value).map(Some)
    }
}

fn decode_wildlife(value: u8) -> Result<Wildlife, Box<dyn Error>> {
    Wildlife::ALL
        .into_iter()
        .find(|wildlife| *wildlife as u8 == value)
        .ok_or_else(|| format!("invalid wildlife code {value}").into())
}

fn write_split_files(
    root: &Path,
    split: &str,
    buffers: &SplitBuffers,
) -> Result<BTreeMap<String, FileSpec>, Box<dyn Error>> {
    let specifications = [
        (
            "group_ids",
            format!("{split}-group-ids.u64"),
            "<u8",
            vec![buffers.groups],
            buffers.group_ids.as_slice(),
        ),
        (
            "public_state_hashes",
            format!("{split}-public-state-hashes.u8"),
            "|u1",
            vec![buffers.groups, 32],
            buffers.public_state_hashes.as_slice(),
        ),
        (
            "exact_supply_values",
            format!("{split}-exact-supply-values.u8"),
            "|u1",
            vec![buffers.groups, EXACT_VALUE_COUNT],
            buffers.exact_supply_values.as_slice(),
        ),
        (
            "exact_supply_hashes",
            format!("{split}-exact-supply-hashes.u8"),
            "|u1",
            vec![buffers.groups, 32],
            buffers.exact_supply_hashes.as_slice(),
        ),
        (
            "candidate_offsets",
            format!("{split}-candidate-offsets.u64"),
            "<u8",
            vec![buffers.groups + 1],
            buffers.candidate_offsets.as_slice(),
        ),
        (
            "staged_wildlife_counts",
            format!("{split}-staged-wildlife-counts.u8"),
            "|u1",
            vec![buffers.candidates, 5],
            buffers.staged_wildlife_counts.as_slice(),
        ),
        (
            "selected_archetype_ids",
            format!("{split}-selected-archetype-ids.u8"),
            "|u1",
            vec![buffers.candidates],
            buffers.selected_archetype_ids.as_slice(),
        ),
        (
            "frontier_requirements",
            format!("{split}-frontier-requirements.u8"),
            "|u1",
            vec![buffers.candidates, 6],
            buffers.frontier_requirements.as_slice(),
        ),
        (
            "selected_compatibility",
            format!("{split}-selected-compatibility.u8"),
            "|u1",
            vec![buffers.candidates, 8],
            buffers.selected_compatibility.as_slice(),
        ),
        (
            "candidate_identity_hashes",
            format!("{split}-candidate-identity-hashes.u8"),
            "|u1",
            vec![buffers.groups, 32],
            buffers.candidate_identity_hashes.as_slice(),
        ),
    ];
    let mut files = BTreeMap::new();
    for (name, file_name, dtype, shape, bytes) in specifications {
        let path = root.join(&file_name);
        write_bytes_atomic(&path, bytes)?;
        files.insert(
            name.to_owned(),
            FileSpec {
                file: file_name,
                dtype,
                shape,
                bytes: bytes.len() as u64,
                blake3: blake3::hash(bytes).to_hex().to_string(),
            },
        );
    }
    Ok(files)
}

fn scientific_identity(
    complete_open_corpus: bool,
    catalog: &SemanticArchetypeCatalog,
    executable_blake3: &str,
    source: &cascadia_provenance::SourceProvenance,
    splits: &BTreeMap<String, SplitManifest>,
    collision_witness: &Value,
) -> Result<Value, Box<dyn Error>> {
    let files = splits
        .iter()
        .map(|(split, manifest)| {
            (
                split.clone(),
                manifest
                    .files
                    .iter()
                    .map(|(name, specification)| {
                        (
                            name.clone(),
                            json!({
                                "file": specification.file,
                                "dtype": specification.dtype,
                                "shape": specification.shape,
                                "bytes": specification.bytes,
                                "blake3": specification.blake3,
                            }),
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    Ok(json!({
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "complete_open_corpus": complete_open_corpus,
        "catalog_blake3": catalog.canonical_blake3().to_hex().to_string(),
        "collision_witness": collision_witness,
        "exporter_executable_blake3": executable_blake3,
        "exporter_source_v2_blake3": source.v2_source_blake3,
        "datasets": splits.iter().map(|(split, manifest)| {
            (split.clone(), json!({
                "dataset_id": manifest.dataset_id,
                "manifest_blake3": manifest.dataset_manifest_blake3,
                "source_v2_blake3": manifest.source_v2_blake3,
                "groups": manifest.groups,
                "candidates": manifest.candidates,
                "complete_open_split": manifest.complete_open_split,
            }))
        }).collect::<BTreeMap<_, _>>(),
        "files": files,
        "hidden_information": {
            "hidden_stack_order_read": false,
            "hidden_wildlife_order_read": false,
            "excluded_tile_identities_read": false,
            "future_refills_read": false,
            "sealed_test_opened": false,
            "gameplay_opened": false,
        },
    }))
}

fn build_collision_witness(catalog: &SemanticArchetypeCatalog) -> Result<Value, Box<dyn Error>> {
    let left_tile_ids = [0usize, 23usize];
    let right_tile_ids = [2usize, 20usize];
    let exact_for = |tile_ids: [usize; 2]| -> Result<ExactSemanticSupply, Box<dyn Error>> {
        let mut counts = vec![0u16; catalog.len()];
        for tile_id in tile_ids {
            let tile = *STANDARD_TILES
                .get(tile_id)
                .ok_or("collision witness tile ID is outside the standard catalog")?;
            let reference = catalog.reference_for_tile(tile)?;
            counts[reference.archetype_id.index()] += 1;
        }
        let bytes = canonical_supply_bytes(catalog, [0; 5], &counts, 2);
        Ok(ExactSemanticSupply::from_canonical_bytes(&bytes)?)
    };
    let left = exact_for(left_tile_ids)?;
    let right = exact_for(right_tile_ids)?;
    let left_legacy = legacy_supply_bytes(left.to_legacy_public_supply());
    let right_legacy = legacy_supply_bytes(right.to_legacy_public_supply());
    if left_legacy != right_legacy || left.archetype_counts() == right.archetype_counts() {
        return Err("ADR 0143 collision witness no longer separates exact refill laws".into());
    }
    let archetype_ids = |supply: &ExactSemanticSupply| {
        supply
            .archetype_counts()
            .iter()
            .enumerate()
            .flat_map(|(id, count)| std::iter::repeat_n(id as u8, usize::from(*count)))
            .collect::<Vec<_>>()
    };
    let identity = json!({
        "schema": "adr-0143-factual-legacy-collision-v1",
        "left_physical_tile_ids": left_tile_ids,
        "right_physical_tile_ids": right_tile_ids,
        "left_archetype_ids": archetype_ids(&left),
        "right_archetype_ids": archetype_ids(&right),
        "legacy_supply_values": left_legacy,
        "refill_denominator": 2,
        "left_refill_numerators": left.archetype_counts(),
        "right_refill_numerators": right.archetype_counts(),
        "left_supply_blake3": left.canonical_hash().to_hex().to_string(),
        "right_supply_blake3": right.canonical_hash().to_hex().to_string(),
        "legacy_marginals_equal": true,
        "refill_laws_differ": true,
    });
    Ok(json!({
        "witness_id": canonical_blake3(&identity)?,
        "identity": identity,
    }))
}

fn canonical_blake3(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn parse_args(
    arguments: impl IntoIterator<Item = impl Into<std::ffi::OsString>>,
) -> Result<Args, Box<dyn Error>> {
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .into_iter();
    let mut train_dataset = None;
    let mut validation_dataset = None;
    let mut output_root = None;
    let mut receipt = None;
    let mut max_groups_per_split = None;
    while let Some(argument) = arguments.next() {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        match argument {
            "--train-dataset" => {
                train_dataset = Some(PathBuf::from(
                    arguments.next().ok_or("--train-dataset requires a path")?,
                ));
            }
            "--validation-dataset" => {
                validation_dataset = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or("--validation-dataset requires a path")?,
                ));
            }
            "--output-root" => {
                output_root = Some(PathBuf::from(
                    arguments.next().ok_or("--output-root requires a path")?,
                ));
            }
            "--receipt" => {
                receipt = Some(PathBuf::from(
                    arguments.next().ok_or("--receipt requires a path")?,
                ));
            }
            "--max-groups-per-split" => {
                let raw = arguments
                    .next()
                    .ok_or("--max-groups-per-split requires a value")?;
                let value = raw
                    .to_str()
                    .ok_or("--max-groups-per-split must be valid UTF-8")?
                    .parse::<usize>()?;
                if value == 0 {
                    return Err("--max-groups-per-split must be positive".into());
                }
                max_groups_per_split = Some(value);
            }
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}").into()),
        }
    }
    Ok(Args {
        train_dataset: train_dataset.ok_or("--train-dataset is required")?,
        validation_dataset: validation_dataset.ok_or("--validation-dataset is required")?,
        output_root: output_root.ok_or("--output-root is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
        max_groups_per_split,
    })
}

fn unix_millis() -> Result<u128, Box<dyn Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
    ));
    {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        writer.write_all(bytes)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_bytes_atomic(path, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::{GameConfig, GameSeed, GameState};

    #[test]
    fn canonical_supply_bytes_round_trip_the_accepted_schema() {
        let catalog = standard_semantic_archetype_catalog();
        let state = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(71),
        )
        .unwrap();
        let exact = ExactSemanticSupply::from_game(&state).unwrap();
        let bytes = canonical_supply_bytes(
            catalog,
            exact.wildlife_bag_counts(),
            exact.archetype_counts(),
            exact.drawable_tile_count(),
        );
        assert_eq!(
            ExactSemanticSupply::from_canonical_bytes(&bytes).unwrap(),
            exact
        );
    }

    #[test]
    fn legacy_bytes_preserve_the_frozen_field_order() {
        let supply = PublicSupply {
            wildlife_bag: [1, 2, 3, 4, 5],
            unseen_tile_terrain_capacity: [6, 7, 8, 9, 10],
            unseen_tile_wildlife_capacity: [11, 12, 13, 14, 15],
            unseen_keystones_by_terrain: [16, 17, 18, 19, 20],
            unseen_dual_terrain_pairs: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        };
        assert_eq!(
            legacy_supply_bytes(supply),
            std::array::from_fn(|index| index as u8 + 1)
        );
    }

    #[test]
    fn frontier_requirements_use_neighbor_facing_edges() {
        let tile = STANDARD_TILES[25];
        let board = BTreeMap::from([(HexCoord::new(1, 0), (tile, Rotation::new(2).unwrap()))]);
        let requirements = frontier_requirements(&board, HexCoord::ORIGIN);
        assert_eq!(
            requirements.neighbor_facing_terrains[0],
            Some(tile.terrain_on_edge(Rotation::new(2).unwrap(), 3))
        );
        assert!(
            requirements.neighbor_facing_terrains[1..]
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn parser_rejects_zero_smoke_bound() {
        let arguments = [
            "--train-dataset",
            "train",
            "--validation-dataset",
            "validation",
            "--output-root",
            "out",
            "--receipt",
            "receipt",
            "--max-groups-per-split",
            "0",
        ];
        assert!(parse_args(arguments).is_err());
    }

    #[test]
    fn collision_witness_preserves_legacy_alias_and_exact_separation() {
        let witness = build_collision_witness(standard_semantic_archetype_catalog()).unwrap();
        let identity = &witness["identity"];
        assert_eq!(identity["left_archetype_ids"], json!([26, 72]));
        assert_eq!(identity["right_archetype_ids"], json!([24, 74]));
        assert_eq!(identity["legacy_marginals_equal"], json!(true));
        assert_eq!(identity["refill_laws_differ"], json!(true));
        assert_ne!(
            identity["left_refill_numerators"],
            identity["right_refill_numerators"]
        );
    }
}
