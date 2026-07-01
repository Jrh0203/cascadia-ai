use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    CANONICAL_TILE_ARCHETYPE_SCHEMA, DatasetSplit, EXACT_SEMANTIC_SUPPLY_SCHEMA,
    EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION, ExactRefillDistribution, ExactSemanticSupply,
    SemanticArchetypeDefinition, standard_semantic_archetype_catalog,
};
use cascadia_game::{
    D6Transform, GameConfig, GameSeed, GameState, STANDARD_TILES, Terrain, Tile, Wildlife,
};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use cascadia_sim::{MatchConfig, StrategyKind, play_match_observed};
use rayon::prelude::*;
use serde::Serialize;

const CENSUS_SCHEMA_VERSION: u16 = 1;
const EXPERIMENT_ID: &str = "exact-semantic-supply-v1";
const HELP: &str = concat!(
    "Usage: exact_semantic_supply_census \\\n",
    "  --output PATH --games N \\\n",
    "  [--first-game-index N] [--split train|validation] \\\n",
    "  [--strategy random|greedy|pattern-aware|pattern-commitment|pattern-competition|pattern-portfolio] \\\n",
    "  [--shard-index I --shard-count N]\n\n",
    "The global interval is [first_game_index, first_game_index + games). A shard owns\n",
    "exactly the offsets satisfying offset % shard_count == shard_index. Test and final\n",
    "splits are intentionally unavailable to this open foundation census."
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    output: PathBuf,
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    strategy: StrategyKind,
    shard_index: usize,
    shard_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CensusShard {
    schema_version: u16,
    experiment_id: &'static str,
    semantic_supply_schema_version: u16,
    semantic_supply_schema: &'static str,
    archetype_schema: &'static str,
    catalog_blake3: String,
    request: CensusRequest,
    shard: ShardIdentity,
    provenance: CensusProvenance,
    catalog: Vec<SemanticArchetypeDefinition>,
    legacy_collision_witness: LegacyCollisionWitness,
    summary: CensusSummary,
    records: Vec<CensusRecord>,
    scientific_blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct CensusRequest {
    split: String,
    strategy: String,
    first_game_index: u64,
    requested_games: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ShardIdentity {
    shard_index: usize,
    shard_count: usize,
    partition_rule: &'static str,
    selected_game_indices: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct CensusProvenance {
    source: SourceProvenance,
    executable_blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct CensusRecord {
    game_index: u64,
    turn: u16,
    active_player: usize,
    public_state_blake3: String,
    semantic_supply_blake3: String,
    semantic_supply_bytes_hex: String,
    unseen_tile_count: u16,
    drawable_tile_count: u16,
    excluded_tile_count: u16,
    wildlife_bag_counts: [u16; 5],
    archetype_counts: Vec<u16>,
    market_archetype_ids: [Option<u16>; 4],
    refill_distribution_blake3_by_slots: [Option<String>; 4],
}

#[derive(Debug, Clone, Serialize)]
struct CensusSummary {
    selected_games: usize,
    positions: usize,
    expected_positions_per_game: usize,
    unique_supply_states: usize,
    minimum_unseen_tiles: u16,
    maximum_unseen_tiles: u16,
    minimum_drawable_tiles: u16,
    maximum_drawable_tiles: u16,
    exact_checks: ExactCheckSummary,
    archetypes: Vec<ArchetypeCensus>,
}

#[derive(Debug, Clone, Serialize)]
struct ExactCheckSummary {
    count_conservation_positions: usize,
    legacy_marginal_parity_positions: usize,
    hidden_order_invariance_positions: usize,
    d6_invariance_positions: usize,
    probability_normalization_positions: usize,
    serialization_round_trip_positions: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ArchetypeCensus {
    archetype_id: u16,
    standard_tile_count: u16,
    minimum_unseen_count: u16,
    maximum_unseen_count: u16,
    zero_count_positions: usize,
    total_unseen_occurrences: u64,
}

#[derive(Debug, Clone, Serialize)]
struct LegacyCollisionWitness {
    left_standard_tile_ids: [u8; 2],
    right_standard_tile_ids: [u8; 2],
    left_archetype_ids: [u16; 2],
    right_archetype_ids: [u16; 2],
    shared_legacy_tile_marginals: [u8; 25],
    exact_archetype_multisets_differ: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let selected_game_indices = selected_game_indices(&args)?;
    let game_results: Vec<Result<Vec<CensusRecord>, String>> = selected_game_indices
        .par_iter()
        .map(|game_index| collect_game(&args, *game_index))
        .collect();
    let mut records = Vec::with_capacity(selected_game_indices.len() * 80);
    for result in game_results {
        records.extend(result.map_err(std::io::Error::other)?);
    }
    records.sort_unstable_by_key(|record| (record.game_index, record.turn));
    validate_record_coverage(&records, &selected_game_indices)?;

    let catalog = standard_semantic_archetype_catalog();
    let scientific_blake3 = scientific_digest(
        catalog.canonical_blake3().to_hex().as_str(),
        args.split.id(),
        args.strategy.id(),
        args.first_game_index,
        args.games,
        args.shard_index,
        args.shard_count,
        &records,
    );
    let shard = CensusShard {
        schema_version: CENSUS_SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID,
        semantic_supply_schema_version: EXACT_SEMANTIC_SUPPLY_SCHEMA_VERSION,
        semantic_supply_schema: EXACT_SEMANTIC_SUPPLY_SCHEMA,
        archetype_schema: CANONICAL_TILE_ARCHETYPE_SCHEMA,
        catalog_blake3: catalog.canonical_blake3().to_hex().to_string(),
        request: CensusRequest {
            split: args.split.id().to_owned(),
            strategy: args.strategy.id().to_owned(),
            first_game_index: args.first_game_index,
            requested_games: args.games,
        },
        shard: ShardIdentity {
            shard_index: args.shard_index,
            shard_count: args.shard_count,
            partition_rule: "(game_index - first_game_index) % shard_count == shard_index",
            selected_game_indices,
        },
        provenance: CensusProvenance {
            source: source_provenance()?,
            executable_blake3: checksum_file(&env::current_exe()?)?,
        },
        catalog: catalog.definitions().to_vec(),
        legacy_collision_witness: legacy_collision_witness()?,
        summary: summarize(&records),
        records,
        scientific_blake3,
    };
    let mut encoded = serde_json::to_vec_pretty(&shard)?;
    encoded.push(b'\n');
    write_atomically(&args.output, &encoded)?;
    println!(
        "{}",
        serde_json::json!({
            "output": args.output,
            "positions": shard.summary.positions,
            "scientific_blake3": shard.scientific_blake3,
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
    let mut output = None;
    let mut games = None;
    let mut first_game_index = 0u64;
    let mut split = DatasetSplit::Train;
    let mut strategy = StrategyKind::PatternAware;
    let mut shard_index = 0usize;
    let mut shard_count = 1usize;

    while let Some(argument) = arguments.next() {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        match argument {
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ));
            }
            "--games" => {
                games = Some(parse_value(
                    arguments.next().ok_or("--games requires a value")?,
                    "--games",
                )?);
            }
            "--first-game-index" => {
                first_game_index = parse_value(
                    arguments
                        .next()
                        .ok_or("--first-game-index requires a value")?,
                    "--first-game-index",
                )?;
            }
            "--split" => {
                split = parse_split(
                    arguments
                        .next()
                        .ok_or("--split requires train or validation")?
                        .to_str()
                        .ok_or("--split must be valid UTF-8")?,
                )?;
            }
            "--strategy" => {
                strategy = parse_strategy(
                    arguments
                        .next()
                        .ok_or("--strategy requires a value")?
                        .to_str()
                        .ok_or("--strategy must be valid UTF-8")?,
                )?;
            }
            "--shard-index" => {
                shard_index = parse_value(
                    arguments.next().ok_or("--shard-index requires a value")?,
                    "--shard-index",
                )?;
            }
            "--shard-count" => {
                shard_count = parse_value(
                    arguments.next().ok_or("--shard-count requires a value")?,
                    "--shard-count",
                )?;
            }
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}").into()),
        }
    }

    let output = output.ok_or("--output is required")?;
    let games = games.ok_or("--games is required")?;
    if games == 0 {
        return Err("--games must be positive".into());
    }
    if shard_count == 0 {
        return Err("--shard-count must be positive".into());
    }
    if shard_index >= shard_count {
        return Err(format!(
            "--shard-index {shard_index} must be less than --shard-count {shard_count}"
        )
        .into());
    }
    if games <= shard_index {
        return Err("the selected shard would contain no games".into());
    }
    first_game_index
        .checked_add(games as u64)
        .ok_or("game interval exceeds u64")?;

    Ok(Args {
        output,
        games,
        first_game_index,
        split,
        strategy,
        shard_index,
        shard_count,
    })
}

fn parse_value<T: std::str::FromStr>(
    value: std::ffi::OsString,
    flag: &str,
) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    Ok(value
        .to_str()
        .ok_or_else(|| format!("{flag} must be valid UTF-8"))?
        .parse()?)
}

fn parse_split(value: &str) -> Result<DatasetSplit, Box<dyn Error>> {
    match value {
        "train" => Ok(DatasetSplit::Train),
        "validation" => Ok(DatasetSplit::Validation),
        _ => Err("--split must be train or validation".into()),
    }
}

fn parse_strategy(value: &str) -> Result<StrategyKind, Box<dyn Error>> {
    match value {
        "random" | "random-v1" => Ok(StrategyKind::Random),
        "greedy" | "greedy-v1" => Ok(StrategyKind::Greedy),
        "pattern-aware" | "pattern-aware-v1" => Ok(StrategyKind::PatternAware),
        "pattern-commitment" | "pattern-commitment-v1" => Ok(StrategyKind::PatternCommitment),
        "pattern-competition" | "pattern-competition-v1" => Ok(StrategyKind::PatternCompetition),
        "pattern-portfolio" | "pattern-portfolio-v1" => Ok(StrategyKind::PatternPortfolio),
        _ => Err(format!("unknown strategy: {value}").into()),
    }
}

fn selected_game_indices(args: &Args) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut selected = Vec::new();
    for offset in 0..args.games {
        if offset % args.shard_count == args.shard_index {
            selected.push(
                args.first_game_index
                    .checked_add(offset as u64)
                    .ok_or("game index exceeds u64")?,
            );
        }
    }
    if selected.is_empty() {
        return Err("the selected shard contains no games".into());
    }
    Ok(selected)
}

fn collect_game(args: &Args, game_index: u64) -> Result<Vec<CensusRecord>, String> {
    let config = GameConfig::research_aaaaa(4).map_err(|error| error.to_string())?;
    let match_config =
        MatchConfig::symmetric(config, args.split.game_seed(game_index), args.strategy);
    let mut records = Vec::with_capacity(80);
    let mut collection_error = None;
    play_match_observed(&match_config, |state, _| {
        if collection_error.is_some() {
            return;
        }
        match observe_state(state, game_index) {
            Ok(record) => records.push(record),
            Err(error) => collection_error = Some(error.to_string()),
        }
    })
    .map_err(|error| error.to_string())?;
    if let Some(error) = collection_error {
        return Err(error);
    }
    if records.len() != 80 {
        return Err(format!(
            "game {game_index} produced {} positions; expected 80",
            records.len()
        ));
    }
    Ok(records)
}

fn observe_state(state: &GameState, game_index: u64) -> Result<CensusRecord, Box<dyn Error>> {
    let supply = ExactSemanticSupply::from_game(state)?;
    if supply.to_legacy_public_supply() != state.public_supply() {
        return Err("exact semantic supply does not reproduce legacy public marginals".into());
    }
    if supply.archetype_counts().iter().sum::<u16>() != supply.unseen_tile_count() {
        return Err("exact semantic supply count conservation failed".into());
    }
    if supply.drawable_tile_count() + supply.excluded_tile_count() != supply.unseen_tile_count() {
        return Err("drawable and excluded tile counts do not conserve unseen supply".into());
    }
    if ExactSemanticSupply::from_canonical_bytes(&supply.canonical_bytes())? != supply {
        return Err("exact semantic supply serialization round trip failed".into());
    }

    let mut refill_distribution_blake3_by_slots: [Option<String>; 4] =
        std::array::from_fn(|_| None);
    for slots in 1..=4u8.min(supply.drawable_tile_count() as u8) {
        let distribution = supply.refill_distribution(slots)?;
        validate_refill_normalization(&distribution)?;
        if ExactRefillDistribution::from_canonical_bytes(&distribution.canonical_bytes())?
            != distribution
        {
            return Err("exact refill distribution serialization round trip failed".into());
        }
        refill_distribution_blake3_by_slots[usize::from(slots - 1)] =
            Some(distribution.canonical_hash().to_hex().to_string());
    }

    let redeterminization_seed =
        GameSeed::from_u64(game_index.rotate_left(17) ^ u64::from(state.completed_turns()));
    let mut redetermined = state.clone();
    redetermined.redeterminize_hidden(redeterminization_seed);
    let redetermined_supply = ExactSemanticSupply::from_game(&redetermined)?;
    if redetermined_supply != supply {
        return Err("semantic supply depends on hidden tile or wildlife order".into());
    }
    for slots in 1..=4u8.min(supply.drawable_tile_count() as u8) {
        if redetermined_supply
            .refill_distribution(slots)?
            .canonical_hash()
            != supply.refill_distribution(slots)?.canonical_hash()
        {
            return Err("refill distribution depends on hidden future order".into());
        }
    }

    for transform in D6Transform::ALL {
        let transformed = state.transformed(transform)?;
        if ExactSemanticSupply::from_game(&transformed)? != supply {
            return Err(format!(
                "semantic supply changed under D6 transform {}",
                transform.id()
            )
            .into());
        }
    }

    let links = supply.market_links(state.market())?;
    let market_archetype_ids = links.map(|link| link.map(|link| link.tile.archetype_id.code()));
    let canonical_bytes = supply.canonical_bytes();
    Ok(CensusRecord {
        game_index,
        turn: state.completed_turns(),
        active_player: state.current_player(),
        public_state_blake3: state.public_state().canonical_hash().to_hex().to_string(),
        semantic_supply_blake3: supply.canonical_hash().to_hex().to_string(),
        semantic_supply_bytes_hex: encode_hex(&canonical_bytes),
        unseen_tile_count: supply.unseen_tile_count(),
        drawable_tile_count: supply.drawable_tile_count(),
        excluded_tile_count: supply.excluded_tile_count(),
        wildlife_bag_counts: supply.wildlife_bag_counts(),
        archetype_counts: supply.archetype_counts().to_vec(),
        market_archetype_ids,
        refill_distribution_blake3_by_slots,
    })
}

fn validate_refill_normalization(
    distribution: &ExactRefillDistribution,
) -> Result<(), Box<dyn Error>> {
    let one_slot_mass: u64 = distribution
        .one_slot_probabilities()
        .iter()
        .map(|entry| {
            let probability = entry.probability;
            probability.numerator
                * (u64::from(distribution.total_unseen()) / probability.denominator)
        })
        .sum();
    if one_slot_mass != u64::from(distribution.total_unseen()) {
        return Err("one-slot refill probability mass does not normalize".into());
    }
    if distribution.ordered_denominator() == 0 {
        return Err("multi-slot refill denominator is zero".into());
    }
    Ok(())
}

fn validate_record_coverage(
    records: &[CensusRecord],
    selected_game_indices: &[u64],
) -> Result<(), Box<dyn Error>> {
    if records.len() != selected_game_indices.len() * 80 {
        return Err("census record count does not match selected games".into());
    }
    for (game_offset, game_index) in selected_game_indices.iter().enumerate() {
        let rows = &records[game_offset * 80..(game_offset + 1) * 80];
        if rows.iter().any(|record| record.game_index != *game_index) {
            return Err(format!("census rows are not grouped for game {game_index}").into());
        }
        if rows
            .iter()
            .enumerate()
            .any(|(turn, record)| record.turn != turn as u16)
        {
            return Err(format!("game {game_index} does not contain exact turns 0..79").into());
        }
    }
    Ok(())
}

fn summarize(records: &[CensusRecord]) -> CensusSummary {
    let catalog = standard_semantic_archetype_catalog();
    let mut archetypes: Vec<_> = catalog
        .definitions()
        .iter()
        .map(|definition| ArchetypeCensus {
            archetype_id: definition.id.code(),
            standard_tile_count: definition.standard_tile_count,
            minimum_unseen_count: u16::MAX,
            maximum_unseen_count: 0,
            zero_count_positions: 0,
            total_unseen_occurrences: 0,
        })
        .collect();
    let mut unique_supply_states = BTreeSet::new();
    let mut minimum_unseen_tiles = u16::MAX;
    let mut maximum_unseen_tiles = 0u16;
    let mut minimum_drawable_tiles = u16::MAX;
    let mut maximum_drawable_tiles = 0u16;
    for record in records {
        unique_supply_states.insert(record.semantic_supply_blake3.clone());
        minimum_unseen_tiles = minimum_unseen_tiles.min(record.unseen_tile_count);
        maximum_unseen_tiles = maximum_unseen_tiles.max(record.unseen_tile_count);
        minimum_drawable_tiles = minimum_drawable_tiles.min(record.drawable_tile_count);
        maximum_drawable_tiles = maximum_drawable_tiles.max(record.drawable_tile_count);
        for (census, count) in archetypes.iter_mut().zip(&record.archetype_counts) {
            census.minimum_unseen_count = census.minimum_unseen_count.min(*count);
            census.maximum_unseen_count = census.maximum_unseen_count.max(*count);
            census.zero_count_positions += usize::from(*count == 0);
            census.total_unseen_occurrences += u64::from(*count);
        }
    }
    CensusSummary {
        selected_games: records.len() / 80,
        positions: records.len(),
        expected_positions_per_game: 80,
        unique_supply_states: unique_supply_states.len(),
        minimum_unseen_tiles,
        maximum_unseen_tiles,
        minimum_drawable_tiles,
        maximum_drawable_tiles,
        exact_checks: ExactCheckSummary {
            count_conservation_positions: records.len(),
            legacy_marginal_parity_positions: records.len(),
            hidden_order_invariance_positions: records.len(),
            d6_invariance_positions: records.len(),
            probability_normalization_positions: records.len(),
            serialization_round_trip_positions: records.len(),
        },
        archetypes,
    }
}

fn legacy_collision_witness() -> Result<LegacyCollisionWitness, Box<dyn Error>> {
    let left_standard_tile_ids = [0, 23];
    let right_standard_tile_ids = [2, 20];
    let catalog = standard_semantic_archetype_catalog();
    let left_tiles = left_standard_tile_ids.map(|id| STANDARD_TILES[usize::from(id)]);
    let right_tiles = right_standard_tile_ids.map(|id| STANDARD_TILES[usize::from(id)]);
    let left_archetype_ids = [
        catalog
            .reference_for_tile(left_tiles[0])?
            .archetype_id
            .code(),
        catalog
            .reference_for_tile(left_tiles[1])?
            .archetype_id
            .code(),
    ];
    let right_archetype_ids = [
        catalog
            .reference_for_tile(right_tiles[0])?
            .archetype_id
            .code(),
        catalog
            .reference_for_tile(right_tiles[1])?
            .archetype_id
            .code(),
    ];
    let left_marginals = legacy_tile_marginals(left_tiles);
    let right_marginals = legacy_tile_marginals(right_tiles);
    if left_marginals != right_marginals {
        return Err("frozen semantic supply collision witness no longer aliases marginals".into());
    }
    let mut left_sorted = left_archetype_ids;
    let mut right_sorted = right_archetype_ids;
    left_sorted.sort_unstable();
    right_sorted.sort_unstable();
    if left_sorted == right_sorted {
        return Err("frozen semantic supply collision witness no longer differs exactly".into());
    }
    Ok(LegacyCollisionWitness {
        left_standard_tile_ids,
        right_standard_tile_ids,
        left_archetype_ids,
        right_archetype_ids,
        shared_legacy_tile_marginals: left_marginals,
        exact_archetype_multisets_differ: true,
    })
}

fn legacy_tile_marginals(tiles: [Tile; 2]) -> [u8; 25] {
    let mut marginals = [0u8; 25];
    for tile in tiles {
        for terrain in Terrain::ALL {
            if tile.contains_terrain(terrain) {
                marginals[terrain as usize] += 1;
            }
        }
        for wildlife in Wildlife::ALL {
            if tile.wildlife.contains(wildlife) {
                marginals[5 + wildlife as usize] += 1;
            }
        }
        if tile.keystone {
            marginals[10 + tile.terrain_a as usize] += 1;
        } else if let Some(secondary) = tile.terrain_b {
            marginals[15 + terrain_pair_index(tile.terrain_a, secondary)] += 1;
        }
    }
    marginals
}

fn terrain_pair_index(left: Terrain, right: Terrain) -> usize {
    let (low, high) = if (left as u8) < (right as u8) {
        (left as usize, right as usize)
    } else {
        (right as usize, left as usize)
    };
    let mut index = 0;
    for first in 0..5 {
        for second in first + 1..5 {
            if (first, second) == (low, high) {
                return index;
            }
            index += 1;
        }
    }
    unreachable!("official dual-terrain tiles contain distinct terrains")
}

#[allow(clippy::too_many_arguments)]
fn scientific_digest(
    catalog_blake3: &str,
    split: &str,
    strategy: &str,
    first_game_index: u64,
    requested_games: usize,
    shard_index: usize,
    shard_count: usize,
    records: &[CensusRecord],
) -> String {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-exact-semantic-supply-census-v1\0");
    update_len_prefixed(&mut hasher, catalog_blake3.as_bytes());
    update_len_prefixed(&mut hasher, split.as_bytes());
    update_len_prefixed(&mut hasher, strategy.as_bytes());
    hasher.update(&first_game_index.to_le_bytes());
    hasher.update(&(requested_games as u64).to_le_bytes());
    hasher.update(&(shard_index as u64).to_le_bytes());
    hasher.update(&(shard_count as u64).to_le_bytes());
    hasher.update(&(records.len() as u64).to_le_bytes());
    for record in records {
        hasher.update(&record.game_index.to_le_bytes());
        hasher.update(&record.turn.to_le_bytes());
        hasher.update(&(record.active_player as u64).to_le_bytes());
        update_len_prefixed(&mut hasher, record.public_state_blake3.as_bytes());
        update_len_prefixed(&mut hasher, record.semantic_supply_blake3.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn update_len_prefixed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("output path requires a UTF-8 file name")?;
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_open_sharded_requests_and_canonical_strategy_ids() {
        let args = parse_args([
            "--output",
            "shard.json",
            "--games",
            "20",
            "--first-game-index",
            "300000",
            "--split",
            "validation",
            "--strategy",
            "pattern-aware",
            "--shard-index",
            "2",
            "--shard-count",
            "4",
        ])
        .unwrap();
        assert_eq!(args.output, PathBuf::from("shard.json"));
        assert_eq!(args.games, 20);
        assert_eq!(args.first_game_index, 300_000);
        assert_eq!(args.split, DatasetSplit::Validation);
        assert_eq!(args.strategy, StrategyKind::PatternAware);
        assert_eq!(args.shard_index, 2);
        assert_eq!(args.shard_count, 4);
    }

    #[test]
    fn parser_rejects_sealed_splits_empty_shards_and_invalid_counts() {
        assert!(parse_args(["--output", "x", "--games", "4", "--split", "test",]).is_err());
        assert!(
            parse_args([
                "--output",
                "x",
                "--games",
                "2",
                "--shard-index",
                "2",
                "--shard-count",
                "4",
            ])
            .is_err()
        );
        assert!(parse_args(["--output", "x", "--games", "1", "--shard-count", "0",]).is_err());
    }

    #[test]
    fn modulo_partitions_are_disjoint_and_cover_the_requested_interval() {
        let mut partitions = Vec::new();
        for shard_index in 0..4 {
            let args = Args {
                output: PathBuf::from("unused"),
                games: 17,
                first_game_index: 300_000,
                split: DatasetSplit::Train,
                strategy: StrategyKind::Random,
                shard_index,
                shard_count: 4,
            };
            partitions.push(selected_game_indices(&args).unwrap());
        }
        let flattened: Vec<_> = partitions.iter().flatten().copied().collect();
        let unique: BTreeSet<_> = flattened.iter().copied().collect();
        assert_eq!(flattened.len(), 17);
        assert_eq!(unique.len(), 17);
        assert_eq!(unique, (300_000..300_017).collect::<BTreeSet<_>>());
    }

    #[test]
    fn frozen_collision_witness_aliases_only_the_legacy_boundary() {
        let witness = legacy_collision_witness().unwrap();
        assert_eq!(witness.left_standard_tile_ids, [0, 23]);
        assert_eq!(witness.right_standard_tile_ids, [2, 20]);
        assert!(witness.exact_archetype_multisets_differ);
        assert_ne!(witness.left_archetype_ids, witness.right_archetype_ids);
    }

    #[test]
    fn one_game_smoke_export_is_complete_and_deterministic() {
        let args = Args {
            output: PathBuf::from("unused"),
            games: 1,
            first_game_index: 399_999,
            split: DatasetSplit::Train,
            strategy: StrategyKind::Random,
            shard_index: 0,
            shard_count: 1,
        };
        let first = collect_game(&args, 399_999).unwrap();
        let second = collect_game(&args, 399_999).unwrap();
        validate_record_coverage(&first, &[399_999]).unwrap();
        assert_eq!(first.len(), 80);
        assert_eq!(
            first
                .iter()
                .map(|record| &record.semantic_supply_blake3)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|record| &record.semantic_supply_blake3)
                .collect::<Vec<_>>()
        );
        assert_eq!(first[0].unseen_tile_count, 81);
        assert_eq!(first[0].drawable_tile_count, 79);
        assert_eq!(first[0].excluded_tile_count, 2);
        assert_eq!(first[79].unseen_tile_count, 2);
        assert_eq!(first[79].drawable_tile_count, 0);
        assert_eq!(first[79].excluded_tile_count, 2);
        assert!(
            first[79]
                .refill_distribution_blake3_by_slots
                .iter()
                .all(Option::is_none)
        );
    }
}
