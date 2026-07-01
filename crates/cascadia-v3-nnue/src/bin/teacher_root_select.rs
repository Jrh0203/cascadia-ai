use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap},
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_game::{GameState, score_board, score_game};
use cascadia_v3_nnue::{
    V3GameRecord, V3GameShardReader, V3TeacherRoot, V3TeacherRootShardWriter, V3TeacherSplit,
    V3TeacherStratum,
};
use clap::Parser;
use serde::Serialize;

const RESERVOIR_PER_STRATUM: usize = 64;

#[derive(Debug, Parser)]
#[command(about = "Select deterministic stratified V3 teacher and validation roots")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    teacher_output: PathBuf,
    #[arg(long)]
    validation_output: PathBuf,
    #[arg(long)]
    receipt: PathBuf,
    #[arg(long, default_value_t = 100_000)]
    teacher_roots: usize,
    #[arg(long, default_value_t = 20_000)]
    validation_roots: usize,
    #[arg(long, default_value_t = 150)]
    oversample_permyriad: u16,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Candidate {
    priority: [u8; 32],
    input_index: usize,
    record_ordinal: u64,
    turn_index: u8,
    stratum: V3TeacherStratum,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.input_index.cmp(&other.input_index))
            .then_with(|| self.record_ordinal.cmp(&other.record_ordinal))
            .then_with(|| self.turn_index.cmp(&other.turn_index))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct Selected {
    candidate: Candidate,
    split: V3TeacherSplit,
}

#[derive(Debug, Serialize)]
struct SelectionReceipt {
    schema_id: &'static str,
    passed: bool,
    scientific_eligible: bool,
    selection_domain: &'static str,
    inputs: Vec<InputReceipt>,
    games: u64,
    positions_considered: u64,
    oversampled_positions: u64,
    strata_observed: usize,
    teacher_roots: u64,
    validation_roots: u64,
    teacher_output: String,
    teacher_bytes: u64,
    teacher_blake3: String,
    validation_output: String,
    validation_bytes: u64,
    validation_blake3: String,
    elapsed_seconds: f64,
}

#[derive(Debug, Serialize)]
struct InputReceipt {
    path: String,
    bytes: u64,
    blake3: String,
}

fn hash(domain: &[u8], game_index: u64, turn_index: u8) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&game_index.to_le_bytes());
    hasher.update(&[turn_index]);
    *hasher.finalize().as_bytes()
}

fn sampled(game_index: u64, turn_index: u8, permyriad: u16) -> bool {
    let digest = hash(
        b"cascadia-v3-teacher-root-oversample-v1",
        game_index,
        turn_index,
    );
    u16::from_le_bytes([digest[0], digest[1]]) % 10_000 < permyriad
}

fn checksum(path: &Path) -> Result<String, std::io::Error> {
    let mut input = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn market_signature(game: &GameState) -> u8 {
    let mut wildlife = [0u8; 5];
    let mut terrain = [0u8; 5];
    for value in game.market().wildlife.iter().flatten() {
        wildlife[*value as usize] += 1;
    }
    for tile in game.market().tiles.iter().flatten() {
        terrain[tile.terrain_a as usize] += 1;
        if let Some(other) = tile.terrain_b {
            terrain[other as usize] += 1;
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v3-market-stratum-v1");
    hasher.update(&wildlife);
    hasher.update(&terrain);
    let digest = hasher.finalize();
    digest.as_bytes()[0] % 16
}

fn nature_bin(tokens: u8) -> u8 {
    tokens.min(2)
}

fn legal_width_bin(width: usize) -> u8 {
    match width {
        0..=32 => 0,
        33..=128 => 1,
        129..=512 => 2,
        _ => 3,
    }
}

fn score_to_go_bin(value: i32) -> u8 {
    match value {
        i32::MIN..=19 => 0,
        20..=39 => 1,
        40..=59 => 2,
        60..=79 => 3,
        _ => 4,
    }
}

fn candidate_for_root(
    game: &GameState,
    final_scores: &[cascadia_game::ScoreBreakdown],
    input_index: usize,
    record_ordinal: u64,
    record: &V3GameRecord,
    turn_index: u8,
) -> Result<Candidate, Box<dyn std::error::Error>> {
    let focal = game.current_player();
    let completed = game.boards()[focal].tile_count().saturating_sub(3).min(20);
    let phase_bucket = ((8 * completed) / 20).min(7) as u8;
    let current = score_board(&game.boards()[focal], record.replay.config.scoring_cards).base_total;
    let score_to_go = i32::from(final_scores[focal].base_total) - i32::from(current);
    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
    let legal_width = game.legal_turn_actions(&prelude)?.len();
    Ok(Candidate {
        priority: hash(
            b"cascadia-v3-teacher-root-priority-v1",
            record.game_index,
            turn_index,
        ),
        input_index,
        record_ordinal,
        turn_index,
        stratum: V3TeacherStratum {
            focal_seat: focal as u8,
            phase_bucket,
            nature_token_bin: nature_bin(game.boards()[focal].nature_tokens()),
            legal_width_bin: legal_width_bin(legal_width),
            score_to_go_bin: score_to_go_bin(score_to_go),
            market_signature_bin: market_signature(game),
        },
    })
}

fn select(args: &Args) -> Result<SelectionReceipt, Box<dyn std::error::Error>> {
    if args.teacher_roots == 0
        || args.oversample_permyriad == 0
        || args.oversample_permyriad > 10_000
    {
        return Err("root counts and oversampling rate must be positive and bounded".into());
    }
    let started = Instant::now();
    let mut reservoirs: BTreeMap<V3TeacherStratum, BinaryHeap<Candidate>> = BTreeMap::new();
    let mut games = 0u64;
    let mut positions_considered = 0u64;
    let mut oversampled_positions = 0u64;
    for (input_index, path) in args.input.iter().enumerate() {
        let mut reader = V3GameShardReader::open(path)?;
        let mut record_ordinal = 0u64;
        while let Some(record) = reader.next_record()? {
            games += 1;
            let eligible_turns = (0u8..80)
                .filter(|turn| {
                    record
                        .focal_training_seat
                        .is_none_or(|seat| *turn % 4 == seat)
                })
                .collect::<Vec<_>>();
            let sampled_turns = eligible_turns
                .iter()
                .copied()
                .filter(|turn| sampled(record.game_index, *turn, args.oversample_permyriad))
                .collect::<Vec<_>>();
            positions_considered += eligible_turns.len() as u64;
            oversampled_positions += sampled_turns.len() as u64;
            if !sampled_turns.is_empty() {
                let terminal = record.replay.play()?;
                let final_scores = score_game(&terminal);
                let mut game = GameState::new(record.replay.config, record.replay.seed)?;
                let mut next_sample = 0usize;
                for (turn, action) in record.replay.turns.iter().enumerate() {
                    if next_sample < sampled_turns.len()
                        && usize::from(sampled_turns[next_sample]) == turn
                    {
                        let candidate = candidate_for_root(
                            &game,
                            &final_scores,
                            input_index,
                            record_ordinal,
                            &record,
                            turn as u8,
                        )?;
                        let heap = reservoirs.entry(candidate.stratum).or_default();
                        if heap.len() < RESERVOIR_PER_STRATUM {
                            heap.push(candidate);
                        } else if candidate < *heap.peek().unwrap() {
                            heap.pop();
                            heap.push(candidate);
                        }
                        next_sample += 1;
                    }
                    game.apply(action)?;
                }
            }
            record_ordinal += 1;
        }
    }

    let required = args.teacher_roots + args.validation_roots;
    let strata_observed = reservoirs.len();
    let mut groups = reservoirs
        .into_values()
        .map(|heap| {
            let mut values = heap.into_vec();
            values.sort();
            values
        })
        .collect::<Vec<_>>();
    let mut balanced = Vec::with_capacity(required);
    for depth in 0..RESERVOIR_PER_STRATUM {
        for group in &mut groups {
            if let Some(candidate) = group.get(depth) {
                balanced.push(candidate.clone());
                if balanced.len() == required {
                    break;
                }
            }
        }
        if balanced.len() == required {
            break;
        }
    }
    if balanced.len() != required {
        return Err(format!(
            "stratified oversample produced {} roots, need {required}",
            balanced.len()
        )
        .into());
    }
    let validation_stride = if args.validation_roots == 0 {
        None
    } else {
        let stride = required / args.validation_roots;
        if stride == 0 || !required.is_multiple_of(args.validation_roots) {
            return Err("teacher/validation counts must define an exact validation stride".into());
        }
        Some(stride)
    };
    let selected = balanced
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| Selected {
            candidate,
            split: if validation_stride.is_some_and(|stride| index % stride == 0) {
                V3TeacherSplit::Validation
            } else {
                V3TeacherSplit::Teacher
            },
        })
        .collect::<Vec<_>>();
    if selected
        .iter()
        .filter(|value| value.split == V3TeacherSplit::Validation)
        .count()
        != args.validation_roots
    {
        return Err("validation stride did not produce the registered root count".into());
    }

    let mut by_record: BTreeMap<(usize, u64), Vec<Selected>> = BTreeMap::new();
    for value in selected {
        by_record
            .entry((value.candidate.input_index, value.candidate.record_ordinal))
            .or_default()
            .push(value);
    }
    if let Some(parent) = args.teacher_output.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args.validation_output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut teacher_writer = V3TeacherRootShardWriter::create(&args.teacher_output)?;
    let mut validation_writer = V3TeacherRootShardWriter::create(&args.validation_output)?;
    for (input_index, path) in args.input.iter().enumerate() {
        let mut reader = V3GameShardReader::open(path)?;
        let mut record_ordinal = 0u64;
        while let Some(record) = reader.next_record()? {
            if let Some(mut roots) = by_record.remove(&(input_index, record_ordinal)) {
                roots.sort_by_key(|value| value.candidate.turn_index);
                let mut game = GameState::new(record.replay.config, record.replay.seed)?;
                let mut cursor = 0usize;
                for value in roots {
                    let turn = usize::from(value.candidate.turn_index);
                    while cursor < turn {
                        game.apply(&record.replay.turns[cursor])?;
                        cursor += 1;
                    }
                    let root = V3TeacherRoot {
                        record: record.clone(),
                        turn_index: value.candidate.turn_index,
                        state_blake3: *game.public_state().canonical_hash().as_bytes(),
                        split: value.split,
                        stratum: value.candidate.stratum,
                    };
                    match value.split {
                        V3TeacherSplit::Teacher => teacher_writer.append(&root)?,
                        V3TeacherSplit::Validation => validation_writer.append(&root)?,
                    }
                }
            }
            record_ordinal += 1;
        }
    }
    if !by_record.is_empty() {
        return Err("selected root locators were not found in the second pass".into());
    }
    let teacher_roots = teacher_writer.finish()?;
    let validation_roots = validation_writer.finish()?;
    if teacher_roots != args.teacher_roots as u64
        || validation_roots != args.validation_roots as u64
    {
        return Err("written root counts differ from the registered selection".into());
    }
    Ok(SelectionReceipt {
        schema_id: "cascadia-v3-teacher-root-selection-v1",
        passed: true,
        scientific_eligible: true,
        selection_domain: "balanced-bottom-hash-strata-v1",
        inputs: args
            .input
            .iter()
            .map(|path| {
                Ok(InputReceipt {
                    path: path.display().to_string(),
                    bytes: path.metadata()?.len(),
                    blake3: checksum(path)?,
                })
            })
            .collect::<Result<Vec<_>, std::io::Error>>()?,
        games,
        positions_considered,
        oversampled_positions,
        strata_observed,
        teacher_roots,
        validation_roots,
        teacher_output: args.teacher_output.display().to_string(),
        teacher_bytes: args.teacher_output.metadata()?.len(),
        teacher_blake3: checksum(&args.teacher_output)?,
        validation_output: args.validation_output.display().to_string(),
        validation_bytes: args.validation_output.metadata()?.len(),
        validation_blake3: checksum(&args.validation_output)?,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let receipt = select(&args)?;
    if let Some(parent) = args.receipt.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = args
        .receipt
        .with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&receipt)?)?;
    fs::rename(temporary, &args.receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
