use std::{
    collections::VecDeque,
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
};

use cascadia_game::D6Transform;
use cascadia_v3_nnue::{
    ActiveFeature, BASE_FEATURE_ROWS, FullOpportunitiesCatalog, GAME_SHARD_MAGIC,
    LABELED_TEACHER_ROOT_SHARD_MAGIC, TRAINING_SHARD_MAGIC, V3CampaignState, V3GameRecord,
    V3GameShardReader, V3LabeledTeacherRoot, V3LabeledTeacherRootShardReader, V3TrainingEntry,
    V3TrainingShardReader, labeled_root_training_entries, transform_feature_set,
};
use clap::Parser;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

const STREAM_MAGIC: &[u8; 8] = b"CSV3BT1\0";
const STREAM_VERSION: u16 = 2;
const FRAME_MAGIC: &[u8; 4] = b"BCH1";
const DEFAULT_EXPANSION_THREADS: usize = 4;
const EXPANSION_CHUNK_PER_THREAD: usize = 4;
const BALANCE_QUEUE_CAPACITY_PER_STRATUM: usize = 64;

#[derive(Debug, Parser)]
#[command(about = "Stream whole V3 CSR batches from native Rust")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    batch_size: usize,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = false)]
    allow_scientific_data: bool,
    #[arg(long, default_value_t = false)]
    d6_cycle: bool,
    #[arg(long)]
    campaign_state: Option<PathBuf>,
    #[arg(long)]
    cycle: Option<u8>,
    /// Override the teacher/realized blend at stream time. This keeps the
    /// immutable teacher corpus reusable across the bootstrap annealing
    /// schedule while binding the exact lambda into each training run.
    #[arg(long)]
    teacher_lambda: Option<f32>,
    /// Emit exactly this many entries, failing if the inputs cannot supply
    /// them within the declared epoch count.
    #[arg(long)]
    max_examples: Option<usize>,
    /// Balance every phase bucket exactly in groups of eight.
    #[arg(long, default_value_t = false)]
    uniform_phase: bool,
    /// Offset the deterministic online D6 cycle between schedule blocks.
    #[arg(long, default_value_t = 0)]
    d6_offset: usize,
    /// CPU threads used to reconstruct compact games and teacher roots. Two
    /// source streams run concurrently during MLX training, so four threads
    /// per stream consume eight of John1's nine authorized CPU slots.
    #[arg(long, default_value_t = DEFAULT_EXPANSION_THREADS)]
    expansion_threads: usize,
    /// Empirical score-to-go quartile boundaries. When present, each emitted
    /// group of 32 contains one row from every phase x score-quartile cell.
    #[arg(long, value_delimiter = ',', num_args = 24)]
    score_quantile_boundaries: Vec<f32>,
}

fn expand_games(
    pool: &ThreadPool,
    records: Vec<V3GameRecord>,
    teacher_lambda: Option<f32>,
    d6_transform: Option<D6Transform>,
) -> cascadia_v3_nnue::Result<Vec<Vec<V3TrainingEntry>>> {
    pool.install(|| {
        records
            .into_par_iter()
            .map(|record| {
                let mut entries = record.training_entries()?;
                prepare_entries(&mut entries, teacher_lambda, d6_transform)?;
                Ok(entries)
            })
            .collect()
    })
}

fn expand_labeled_roots(
    pool: &ThreadPool,
    roots: Vec<V3LabeledTeacherRoot>,
    teacher_lambda: Option<f32>,
    d6_transform: Option<D6Transform>,
) -> cascadia_v3_nnue::Result<Vec<Vec<V3TrainingEntry>>> {
    pool.install(|| {
        roots
            .into_par_iter()
            .map(|root| {
                let mut entries = labeled_root_training_entries(&root)?;
                prepare_entries(&mut entries, teacher_lambda, d6_transform)?;
                Ok(entries)
            })
            .collect()
    })
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T) {
    if queue.len() < BALANCE_QUEUE_CAPACITY_PER_STRATUM {
        queue.push_back(value);
    }
}

fn write_u16(output: &mut impl Write, value: u16) -> io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_u32(output: &mut impl Write, value: usize) -> io::Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "CSR value exceeds u32"))?;
    output.write_all(&value.to_le_bytes())
}

fn write_sparse(
    output: &mut impl Write,
    entries: &[V3TrainingEntry],
    select: impl Fn(&V3TrainingEntry) -> &[ActiveFeature],
) -> io::Result<()> {
    let mut offset = 0usize;
    write_u32(output, 0)?;
    for entry in entries {
        offset = offset
            .checked_add(select(entry).len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "CSR offset overflow"))?;
        write_u32(output, offset)?;
    }
    for entry in entries {
        for feature in select(entry) {
            output.write_all(&feature.index.to_le_bytes())?;
        }
    }
    for entry in entries {
        for feature in select(entry) {
            write_u16(output, feature.count)?;
        }
    }
    Ok(())
}

fn expanded_training_factors(
    features: &[ActiveFeature],
    counts: &mut [u32],
    touched: &mut Vec<u32>,
) -> io::Result<Vec<ActiveFeature>> {
    let catalog = FullOpportunitiesCatalog::global();
    if counts.len() != catalog.training_factor_len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "training-factor scratch width drifted",
        ));
    }
    touched.clear();
    for feature in features {
        let factors = catalog
            .training_factors_for_row(feature.index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "opportunity row {} has no training-factor map",
                        feature.index
                    ),
                )
            })?;
        for factor in factors {
            let slot = &mut counts[*factor as usize];
            if *slot == 0 {
                touched.push(*factor);
            }
            *slot = slot.checked_add(u32::from(feature.count)).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "factor count overflow")
            })?;
        }
    }
    touched.sort_unstable();
    let result = touched
        .iter()
        .copied()
        .map(|index| {
            let count = counts[index as usize];
            Ok(ActiveFeature {
                index,
                count: u16::try_from(count).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "factor count exceeds u16")
                })?,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    for index in touched.iter().copied() {
        counts[index as usize] = 0;
    }
    Ok(result)
}

fn write_training_factors(
    output: &mut impl Write,
    entries: &[V3TrainingEntry],
    select: impl Fn(&V3TrainingEntry) -> &[ActiveFeature],
) -> io::Result<()> {
    let catalog = FullOpportunitiesCatalog::global();
    let mut counts = vec![0u32; catalog.training_factor_len()];
    let mut touched = Vec::new();
    let rows = entries
        .iter()
        .map(|entry| expanded_training_factors(select(entry), &mut counts, &mut touched))
        .collect::<io::Result<Vec<_>>>()?;
    let mut offset = 0usize;
    write_u32(output, 0)?;
    for row in &rows {
        offset = offset
            .checked_add(row.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "CSR offset overflow"))?;
        write_u32(output, offset)?;
    }
    for feature in rows.iter().flatten() {
        output.write_all(&feature.index.to_le_bytes())?;
    }
    for feature in rows.iter().flatten() {
        write_u16(output, feature.count)?;
    }
    Ok(())
}

fn write_batch(output: &mut impl Write, entries: &[V3TrainingEntry]) -> io::Result<()> {
    output.write_all(FRAME_MAGIC)?;
    write_u32(output, entries.len())?;
    write_sparse(output, entries, |entry| &entry.features.own_base)?;
    write_sparse(output, entries, |entry| &entry.features.field_base)?;
    write_sparse(output, entries, |entry| &entry.features.own_opportunities)?;
    write_sparse(output, entries, |entry| &entry.features.field_opportunities)?;
    write_training_factors(output, entries, |entry| &entry.features.own_opportunities)?;
    write_training_factors(output, entries, |entry| &entry.features.field_opportunities)?;
    for entry in entries {
        output.write_all(&[entry.features.phase_bucket])?;
    }
    for entry in entries {
        output.write_all(&entry.target_score_to_go.to_le_bytes())?;
    }
    for entry in entries {
        output.write_all(&entry.confidence_weight().to_le_bytes())?;
    }
    Ok(())
}

fn apply_teacher_lambda(
    entry: &mut V3TrainingEntry,
    teacher_lambda: Option<f32>,
) -> cascadia_v3_nnue::Result<()> {
    if let (Some(lambda), Some(teacher)) = (teacher_lambda, entry.teacher_score_to_go) {
        entry.lambda = lambda;
        entry.target_score_to_go = lambda * teacher + (1.0 - lambda) * entry.realized_score_to_go;
        entry.validate()?;
    }
    Ok(())
}

fn prepare_entry(
    entry: &mut V3TrainingEntry,
    teacher_lambda: Option<f32>,
    d6_transform: Option<D6Transform>,
) -> cascadia_v3_nnue::Result<()> {
    apply_teacher_lambda(entry, teacher_lambda)?;
    if let Some(transform) = d6_transform {
        entry.features = transform_feature_set(&entry.features, transform)?;
    }
    Ok(())
}

fn prepare_entries(
    entries: &mut [V3TrainingEntry],
    teacher_lambda: Option<f32>,
    d6_transform: Option<D6Transform>,
) -> cascadia_v3_nnue::Result<()> {
    for entry in entries {
        prepare_entry(entry, teacher_lambda, d6_transform)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.batch_size == 0 || args.epochs == 0 || args.expansion_threads == 0 {
        return Err("batch size, epochs, and expansion threads must be positive".into());
    }
    if args.max_examples == Some(0) {
        return Err("max examples must be positive".into());
    }
    if args.uniform_phase && args.max_examples.is_some_and(|examples| examples % 8 != 0) {
        return Err("uniform-phase max examples must be divisible by eight".into());
    }
    if args.uniform_phase && args.batch_size % 8 != 0 {
        return Err("uniform-phase batch size must be divisible by eight".into());
    }
    if !args.score_quantile_boundaries.is_empty()
        && (args.score_quantile_boundaries.len() != 24
            || args
                .score_quantile_boundaries
                .iter()
                .any(|value| !value.is_finite())
            || args
                .score_quantile_boundaries
                .chunks_exact(3)
                .any(|values| values[0] >= values[1] || values[1] >= values[2]))
    {
        return Err(
            "score quantile boundaries must be eight finite increasing phase triples".into(),
        );
    }
    if !args.score_quantile_boundaries.is_empty()
        && (args.max_examples.is_some_and(|examples| examples % 32 != 0)
            || args.batch_size % 32 != 0)
    {
        return Err("phase x score-quantile balancing requires multiples of 32".into());
    }
    if args
        .teacher_lambda
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err("teacher lambda must be finite and within [0, 1]".into());
    }
    if args.allow_scientific_data {
        let state = V3CampaignState::load_verified(
            args.campaign_state
                .as_deref()
                .ok_or("scientific streaming requires --campaign-state")?,
        )?;
        let expected_phase = args.cycle.map_or_else(
            || "bootstrap_training".to_owned(),
            |cycle| format!("cycle-{cycle:02}-training"),
        );
        if state.part != 2
            || !state.phase2_authorized
            || state.phase != expected_phase
            || state.protected_seed_values_opened
            || state.approved_readiness_sha256.is_none()
            || state.approved_readiness_sha256 != state.readiness_sha256
        {
            return Err("scientific streaming is not authorized for this V3 phase".into());
        }
    } else if args.campaign_state.is_some() || args.cycle.is_some() {
        return Err("campaign authorization is only valid for scientific streaming".into());
    }
    let stdout = io::stdout().lock();
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, stdout);
    output.write_all(STREAM_MAGIC)?;
    write_u16(&mut output, STREAM_VERSION)?;
    write_u16(&mut output, u16::from(args.d6_cycle))?;
    output.write_all(&(BASE_FEATURE_ROWS as u32).to_le_bytes())?;
    output.write_all(&(FullOpportunitiesCatalog::global().len() as u32).to_le_bytes())?;
    output.write_all(
        &(FullOpportunitiesCatalog::global().training_factor_len() as u32).to_le_bytes(),
    )?;
    // The Python client needs only this fixed header to launch the second
    // source stream. Without an explicit flush it waited for the first huge
    // CSR batch, accidentally serializing broad and teacher preprocessing.
    output.flush()?;

    let expansion_pool = ThreadPoolBuilder::new()
        .num_threads(args.expansion_threads)
        .thread_name(|index| format!("v3-expand-{index}"))
        .build()?;
    let expansion_chunk = args.expansion_threads * EXPANSION_CHUNK_PER_THREAD;

    let mut batch = Vec::with_capacity(args.batch_size);
    let mut phase_queues: [VecDeque<V3TrainingEntry>; 8] = std::array::from_fn(|_| VecDeque::new());
    let mut phase_score_queues: [[VecDeque<V3TrainingEntry>; 4]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| VecDeque::new()));
    let mut emitted = 0usize;
    'epochs: for epoch in 0..args.epochs {
        let d6_transform = args
            .d6_cycle
            .then(|| D6Transform::ALL[(epoch + args.d6_offset) % D6Transform::ALL.len()]);
        for path in &args.input {
            let mut input = File::open(path)?;
            let mut magic = [0u8; 8];
            std::io::Read::read_exact(&mut input, &mut magic)?;
            drop(input);
            let mut emit = |entry: V3TrainingEntry| -> Result<bool, Box<dyn std::error::Error>> {
                if entry.provenance.scientific_eligible() && !args.allow_scientific_data {
                    return Err(format!(
                        "{} contains scientific data but --allow-scientific-data is absent",
                        path.display()
                    )
                    .into());
                }
                if !args.score_quantile_boundaries.is_empty() {
                    let phase = usize::from(entry.features.phase_bucket);
                    let boundaries = &args.score_quantile_boundaries[phase * 3..phase * 3 + 3];
                    let score_bin = boundaries
                        .iter()
                        .position(|boundary| entry.target_score_to_go <= *boundary)
                        .unwrap_or(3);
                    push_bounded(&mut phase_score_queues[phase][score_bin], entry);
                    while phase_score_queues
                        .iter()
                        .flatten()
                        .all(|queue| !queue.is_empty())
                    {
                        for phase_queues in &mut phase_score_queues {
                            for queue in phase_queues {
                                batch.push(
                                    queue.pop_front().expect("all stratum queues were nonempty"),
                                );
                                emitted += 1;
                            }
                        }
                        if batch.len() >= args.batch_size {
                            write_batch(&mut output, &batch)?;
                            batch.clear();
                        }
                        if args.max_examples == Some(emitted) {
                            return Ok(true);
                        }
                    }
                } else if args.uniform_phase {
                    let phase = usize::from(entry.features.phase_bucket);
                    push_bounded(&mut phase_queues[phase], entry);
                    while phase_queues.iter().all(|queue| !queue.is_empty()) {
                        for queue in &mut phase_queues {
                            batch.push(queue.pop_front().expect("all phase queues were nonempty"));
                            emitted += 1;
                        }
                        if batch.len() >= args.batch_size {
                            write_batch(&mut output, &batch)?;
                            batch.clear();
                        }
                        if args.max_examples == Some(emitted) {
                            return Ok(true);
                        }
                    }
                } else {
                    batch.push(entry);
                    emitted += 1;
                    if batch.len() == args.batch_size {
                        write_batch(&mut output, &batch)?;
                        batch.clear();
                    }
                    if args.max_examples == Some(emitted) {
                        return Ok(true);
                    }
                }
                Ok(false)
            };
            let mut stop = false;
            if &magic == TRAINING_SHARD_MAGIC {
                let mut reader = V3TrainingShardReader::open(path)?;
                while let Some(mut entry) = reader.next_entry()? {
                    prepare_entry(&mut entry, args.teacher_lambda, d6_transform)?;
                    if emit(entry)? {
                        stop = true;
                        break;
                    }
                }
            } else if &magic == GAME_SHARD_MAGIC {
                let mut reader = V3GameShardReader::open(path)?;
                loop {
                    let mut records = Vec::with_capacity(expansion_chunk);
                    while records.len() < expansion_chunk {
                        let Some(record) = reader.next_record()? else {
                            break;
                        };
                        records.push(record);
                    }
                    if records.is_empty() {
                        break;
                    }
                    // Rayon indexed collection preserves source-record order;
                    // entries within each record retain replay order. Parallel
                    // expansion is therefore byte-identical to one thread.
                    for entries in
                        expand_games(&expansion_pool, records, args.teacher_lambda, d6_transform)?
                    {
                        for entry in entries {
                            if emit(entry)? {
                                stop = true;
                                break;
                            }
                        }
                        if stop {
                            break;
                        }
                    }
                    if stop {
                        break;
                    }
                }
            } else if &magic == LABELED_TEACHER_ROOT_SHARD_MAGIC {
                let mut reader = V3LabeledTeacherRootShardReader::open(path)?;
                loop {
                    let mut roots = Vec::with_capacity(expansion_chunk);
                    while roots.len() < expansion_chunk {
                        let Some(root) = reader.next_labeled_root()? else {
                            break;
                        };
                        roots.push(root);
                    }
                    if roots.is_empty() {
                        break;
                    }
                    for entries in expand_labeled_roots(
                        &expansion_pool,
                        roots,
                        args.teacher_lambda,
                        d6_transform,
                    )? {
                        for entry in entries {
                            if emit(entry)? {
                                stop = true;
                                break;
                            }
                        }
                        if stop {
                            break;
                        }
                    }
                    if stop {
                        break;
                    }
                }
            } else {
                return Err(format!(
                    "{} is not a V3 training, compact game, or labeled-root shard",
                    path.display()
                )
                .into());
            }
            if stop {
                break 'epochs;
            }
        }
    }
    if let Some(expected) = args.max_examples
        && emitted != expected
    {
        return Err(format!("stream emitted {emitted} examples, expected {expected}").into());
    }
    if !batch.is_empty() {
        write_batch(&mut output, &batch)?;
    }
    output.write_all(FRAME_MAGIC)?;
    write_u32(&mut output, 0)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{apply_teacher_lambda, expand_games, push_bounded};
    use cascadia_game::{GameConfig, GameSeed, GameState, Replay};
    use cascadia_sim::{select_greedy_action, strategy_rng};
    use cascadia_v3_nnue::{
        V3GameRecord, V3TrainingEntry, V3TrainingProvenance, encode_public_features,
    };
    use rayon::ThreadPoolBuilder;

    fn entry(teacher: Option<f32>) -> V3TrainingEntry {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(991),
        )
        .unwrap();
        V3TrainingEntry {
            state_blake3: *game.public_state().canonical_hash().as_bytes(),
            game_index: 0,
            decision_index: 0,
            focal_seat: 0,
            features: encode_public_features(&game.public_state(), 0).unwrap(),
            realized_score_to_go: 20.0,
            teacher_score_to_go: teacher,
            teacher_variance: teacher.map(|_| 1.0),
            teacher_sample_count: u32::from(teacher.is_some()),
            lambda: if teacher.is_some() { 1.0 } else { 0.0 },
            target_score_to_go: teacher.unwrap_or(20.0),
            provenance: V3TrainingProvenance::EngineeringSmoke,
        }
    }

    #[test]
    fn teacher_lambda_recomputes_only_teacher_targets() {
        let mut labeled = entry(Some(100.0));
        apply_teacher_lambda(&mut labeled, Some(0.75)).unwrap();
        assert_eq!(labeled.lambda, 0.75);
        assert_eq!(labeled.target_score_to_go, 80.0);

        let mut realized = entry(None);
        apply_teacher_lambda(&mut realized, Some(0.75)).unwrap();
        assert_eq!(realized.lambda, 0.0);
        assert_eq!(realized.target_score_to_go, 20.0);
    }

    fn game_record(index: u64) -> V3GameRecord {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let seed = GameSeed::from_u64(10_000 + index);
        let mut game = GameState::new(config, seed).unwrap();
        let mut replay = Replay::new(config, seed);
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "v3-batch-parallel-test-v1"))
            .collect::<Vec<_>>();
        while !game.is_game_over() {
            let seat = game.current_player();
            let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
            let action = select_greedy_action(&game, &prelude, &mut rngs[seat]).unwrap();
            game.apply(&action).unwrap();
            replay.turns.push(action);
        }
        replay.seal().unwrap();
        V3GameRecord {
            game_index: index,
            replay,
            seat_policy_ids: std::array::from_fn(|_| "greedy".to_owned()),
            newest_model_id: None,
            focal_training_seat: None,
            exploration_epsilon: 0.0,
            provenance: V3TrainingProvenance::EngineeringSmoke,
        }
    }

    #[test]
    fn parallel_game_expansion_preserves_serial_order_and_values() {
        let records = (0..4).map(game_record).collect::<Vec<_>>();
        let serial = records
            .iter()
            .map(V3GameRecord::training_entries)
            .collect::<cascadia_v3_nnue::Result<Vec<_>>>()
            .unwrap();
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let parallel = expand_games(&pool, records, None, None).unwrap();
        assert_eq!(parallel, serial);
    }

    #[test]
    fn balancing_queues_are_bounded_and_keep_earliest_rows() {
        let mut queue = std::collections::VecDeque::new();
        for value in 0..100 {
            push_bounded(&mut queue, value);
        }
        assert_eq!(queue.len(), 64);
        assert_eq!(queue.front(), Some(&0));
        assert_eq!(queue.back(), Some(&63));
    }
}
