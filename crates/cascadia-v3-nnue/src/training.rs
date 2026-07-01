use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use cascadia_game::{GameConfig, GameState, Replay, score_board, score_game};

use crate::{Result, V3Error, V3FeatureSet, encode_public_features};

pub const TRAINING_SHARD_MAGIC: &[u8; 8] = b"CSV3TR1\0";
pub const TRAINING_SHARD_VERSION: u16 = 1;
pub const GAME_SHARD_MAGIC: &[u8; 8] = b"CSV3GM1\0";
pub const GAME_SHARD_VERSION: u16 = 1;
const HEADER_SIZE: u64 = 8 + 2 + 2 + 8;

pub fn signed_score_to_go(final_score: u16, current_score: u16) -> f32 {
    f32::from(final_score) - f32::from(current_score)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum V3TrainingProvenance {
    EngineeringSmoke,
    Bootstrap { component: String },
    ExpertIteration { cycle: u8 },
}

impl V3TrainingProvenance {
    pub fn scientific_eligible(&self) -> bool {
        !matches!(self, Self::EngineeringSmoke)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3TrainingEntry {
    pub state_blake3: [u8; 32],
    pub game_index: u64,
    pub decision_index: u8,
    pub focal_seat: u8,
    pub features: V3FeatureSet,
    pub realized_score_to_go: f32,
    pub teacher_score_to_go: Option<f32>,
    pub teacher_variance: Option<f32>,
    pub teacher_sample_count: u32,
    pub lambda: f32,
    pub target_score_to_go: f32,
    pub provenance: V3TrainingProvenance,
}

impl V3TrainingEntry {
    pub fn validate(&self) -> Result<()> {
        self.features.validate()?;
        if self.focal_seat >= 4
            || self.decision_index >= 20
            || !self.realized_score_to_go.is_finite()
            || !self.target_score_to_go.is_finite()
            || !self.lambda.is_finite()
            || !(0.0..=1.0).contains(&self.lambda)
            || self
                .teacher_score_to_go
                .is_some_and(|value| !value.is_finite())
            || self
                .teacher_variance
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            || self.teacher_score_to_go.is_some() != self.teacher_variance.is_some()
            || (self.teacher_score_to_go.is_none() && self.teacher_sample_count != 0)
        {
            return Err(V3Error::InvalidTraining(
                "training entry metadata or target is invalid".to_owned(),
            ));
        }
        if let Some(teacher) = self.teacher_score_to_go {
            if self.teacher_sample_count == 0 {
                return Err(V3Error::InvalidTraining(
                    "teacher target has no samples".to_owned(),
                ));
            }
            let expected = self.lambda * teacher + (1.0 - self.lambda) * self.realized_score_to_go;
            if (expected - self.target_score_to_go).abs() > 1e-4 {
                return Err(V3Error::InvalidTraining(
                    "blended training target is inconsistent".to_owned(),
                ));
            }
        } else if (self.target_score_to_go - self.realized_score_to_go).abs() > 1e-4 {
            return Err(V3Error::InvalidTraining(
                "unlabeled target must equal realized score-to-go".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn confidence_weight(&self) -> f32 {
        self.teacher_variance.map_or(1.0, |variance| {
            let squared_standard_error = variance / self.teacher_sample_count as f32;
            (1.0 / (0.25 + squared_standard_error)).clamp(0.25, 4.0)
        })
    }
}

/// Replay-authoritative campaign record. Scientific collections store this
/// compact form and reconstruct sparse afterstates only in the native loader,
/// keeping the 40-million-position bootstrap well below the 40 GiB ceiling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3GameRecord {
    pub game_index: u64,
    pub replay: Replay,
    pub seat_policy_ids: [String; 4],
    pub newest_model_id: Option<String>,
    pub focal_training_seat: Option<u8>,
    pub exploration_epsilon: f32,
    pub provenance: V3TrainingProvenance,
}

impl V3GameRecord {
    fn validate_metadata(&self) -> Result<()> {
        if self.replay.config != GameConfig::research_aaaaa(4)?
            || self.replay.turns.len() != 80
            || self.replay.final_state_hash.is_none()
            || self.seat_policy_ids.iter().any(String::is_empty)
            || !self.exploration_epsilon.is_finite()
            || !(0.0..=1.0).contains(&self.exploration_epsilon)
            || self.focal_training_seat.is_some_and(|seat| seat >= 4)
        {
            return Err(V3Error::InvalidTraining(
                "compact V3 game record metadata is invalid".to_owned(),
            ));
        }
        match &self.provenance {
            V3TrainingProvenance::ExpertIteration { cycle } => {
                let focal = self.focal_training_seat.ok_or_else(|| {
                    V3Error::InvalidTraining("expert game lacks a focal training seat".to_owned())
                })?;
                if !(1..=10).contains(cycle) {
                    return Err(V3Error::InvalidTraining(
                        "expert iteration cycle is outside 1..=10".to_owned(),
                    ));
                }
                let newest = self.newest_model_id.as_ref().ok_or_else(|| {
                    V3Error::InvalidTraining("expert game lacks newest model identity".to_owned())
                })?;
                let seats = self
                    .seat_policy_ids
                    .iter()
                    .enumerate()
                    .filter(|(_, policy)| *policy == newest)
                    .map(|(seat, _)| seat)
                    .collect::<Vec<_>>();
                if seats != [usize::from(focal)] {
                    return Err(V3Error::InvalidTraining(
                        "newest expert model must occupy exactly the focal seat".to_owned(),
                    ));
                }
            }
            V3TrainingProvenance::Bootstrap { .. } => {
                if self.focal_training_seat.is_some() {
                    return Err(V3Error::InvalidTraining(
                        "bootstrap games must record all four seats".to_owned(),
                    ));
                }
            }
            V3TrainingProvenance::EngineeringSmoke => {}
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_metadata()?;
        self.replay
            .play()
            .map_err(|error| V3Error::InvalidTraining(error.to_string()))?;
        Ok(())
    }

    pub fn training_entries(&self) -> Result<Vec<V3TrainingEntry>> {
        // `Replay::play` below is both the terminal-score source and the full
        // replay-integrity check. Re-running it through `validate` first made
        // every compact game expansion replay all 80 turns twice.
        self.validate_metadata()?;
        let terminal = self
            .replay
            .play()
            .map_err(|error| V3Error::InvalidTraining(error.to_string()))?;
        let final_scores = score_game(&terminal);
        let mut game = GameState::new(self.replay.config, self.replay.seed)?;
        let mut entries = Vec::with_capacity(if self.focal_training_seat.is_some() {
            20
        } else {
            80
        });
        for action in &self.replay.turns {
            let focal = game.current_player();
            let decision_index = game.boards()[focal].tile_count().saturating_sub(3) as u8;
            game.apply(action)?;
            if self
                .focal_training_seat
                .is_none_or(|seat| usize::from(seat) == focal)
            {
                let public = game.public_state();
                let features = encode_public_features(&public, focal)?;
                let current =
                    score_board(&game.boards()[focal], self.replay.config.scoring_cards).base_total;
                let score_to_go = signed_score_to_go(final_scores[focal].base_total, current);
                entries.push(V3TrainingEntry {
                    state_blake3: *public.canonical_hash().as_bytes(),
                    game_index: self.game_index,
                    decision_index,
                    focal_seat: focal as u8,
                    features,
                    realized_score_to_go: score_to_go,
                    teacher_score_to_go: None,
                    teacher_variance: None,
                    teacher_sample_count: 0,
                    lambda: 0.0,
                    target_score_to_go: score_to_go,
                    provenance: self.provenance.clone(),
                });
            }
        }
        if game.canonical_hash().as_bytes() != self.replay.final_state_hash.as_ref().unwrap() {
            return Err(V3Error::InvalidTraining(
                "compact V3 game replay changed during expansion".to_owned(),
            ));
        }
        Ok(entries)
    }
}

pub struct V3GameShardWriter {
    output: BufWriter<File>,
    count: u64,
    finished: bool,
}

impl V3GameShardWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(GAME_SHARD_MAGIC)?;
        output.write_all(&GAME_SHARD_VERSION.to_le_bytes())?;
        output.write_all(&0u16.to_le_bytes())?;
        output.write_all(&0u64.to_le_bytes())?;
        Ok(Self {
            output,
            count: 0,
            finished: false,
        })
    }

    pub fn append(&mut self, record: &V3GameRecord) -> Result<()> {
        if self.finished {
            return Err(V3Error::InvalidTraining(
                "cannot append to a finished game shard".to_owned(),
            ));
        }
        record.validate()?;
        let bytes = postcard::to_allocvec(record)?;
        let length = u32::try_from(bytes.len()).map_err(|_| {
            V3Error::InvalidTraining("compact game record exceeds u32 length".to_owned())
        })?;
        self.output.write_all(&length.to_le_bytes())?;
        self.output.write_all(&bytes)?;
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| V3Error::InvalidTraining("game row count overflow".to_owned()))?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.output.flush()?;
        let file = self.output.get_mut();
        file.seek(SeekFrom::Start(12))?;
        file.write_all(&self.count.to_le_bytes())?;
        file.sync_all()?;
        self.finished = true;
        Ok(self.count)
    }
}

pub struct V3GameShardReader {
    input: BufReader<File>,
    remaining: u64,
    total: u64,
}

impl V3GameShardReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        let version = read_u16(&mut input)?;
        let reserved = read_u16(&mut input)?;
        let total = read_u64(&mut input)?;
        if &magic != GAME_SHARD_MAGIC || version != GAME_SHARD_VERSION || reserved != 0 {
            return Err(V3Error::InvalidTraining(
                "compact game shard header is invalid".to_owned(),
            ));
        }
        Ok(Self {
            input,
            remaining: total,
            total,
        })
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn next_record(&mut self) -> Result<Option<V3GameRecord>> {
        if self.remaining == 0 {
            let mut trailing = [0u8; 1];
            if self.input.read(&mut trailing)? != 0 {
                return Err(V3Error::InvalidTraining(
                    "compact game shard contains trailing bytes".to_owned(),
                ));
            }
            return Ok(None);
        }
        let length = read_u32(&mut self.input)? as usize;
        if length == 0 || length > 16 * 1024 * 1024 {
            return Err(V3Error::InvalidTraining(
                "compact game record length is invalid".to_owned(),
            ));
        }
        let mut bytes = vec![0u8; length];
        self.input.read_exact(&mut bytes)?;
        let record: V3GameRecord = postcard::from_bytes(&bytes)?;
        record.validate()?;
        self.remaining -= 1;
        Ok(Some(record))
    }
}

pub struct V3TrainingShardWriter {
    output: BufWriter<File>,
    count: u64,
    finished: bool,
}

impl V3TrainingShardWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(TRAINING_SHARD_MAGIC)?;
        output.write_all(&TRAINING_SHARD_VERSION.to_le_bytes())?;
        output.write_all(&0u16.to_le_bytes())?;
        output.write_all(&0u64.to_le_bytes())?;
        Ok(Self {
            output,
            count: 0,
            finished: false,
        })
    }

    pub fn append(&mut self, entry: &V3TrainingEntry) -> Result<()> {
        if self.finished {
            return Err(V3Error::InvalidTraining(
                "cannot append to a finished training shard".to_owned(),
            ));
        }
        entry.validate()?;
        let bytes = postcard::to_allocvec(entry)?;
        let length = u32::try_from(bytes.len()).map_err(|_| {
            V3Error::InvalidTraining("training entry exceeds u32 length".to_owned())
        })?;
        self.output.write_all(&length.to_le_bytes())?;
        self.output.write_all(&bytes)?;
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| V3Error::InvalidTraining("training row count overflow".to_owned()))?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.output.flush()?;
        let file = self.output.get_mut();
        file.seek(SeekFrom::Start(12))?;
        file.write_all(&self.count.to_le_bytes())?;
        file.sync_all()?;
        self.finished = true;
        Ok(self.count)
    }
}

pub struct V3TrainingShardReader {
    input: BufReader<File>,
    remaining: u64,
    total: u64,
}

impl V3TrainingShardReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        let version = read_u16(&mut input)?;
        let reserved = read_u16(&mut input)?;
        let total = read_u64(&mut input)?;
        if &magic != TRAINING_SHARD_MAGIC || version != TRAINING_SHARD_VERSION || reserved != 0 {
            return Err(V3Error::InvalidTraining(
                "training shard header is invalid".to_owned(),
            ));
        }
        debug_assert_eq!(input.stream_position()?, HEADER_SIZE);
        Ok(Self {
            input,
            remaining: total,
            total,
        })
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn next_entry(&mut self) -> Result<Option<V3TrainingEntry>> {
        if self.remaining == 0 {
            let mut trailing = [0u8; 1];
            if self.input.read(&mut trailing)? != 0 {
                return Err(V3Error::InvalidTraining(
                    "training shard contains trailing bytes".to_owned(),
                ));
            }
            return Ok(None);
        }
        let length = read_u32(&mut self.input)? as usize;
        if length == 0 || length > 64 * 1024 * 1024 {
            return Err(V3Error::InvalidTraining(
                "training entry length is invalid".to_owned(),
            ));
        }
        let mut bytes = vec![0u8; length];
        self.input.read_exact(&mut bytes)?;
        let entry: V3TrainingEntry = postcard::from_bytes(&bytes)?;
        entry.validate()?;
        self.remaining -= 1;
        Ok(Some(entry))
    }
}

fn read_u16(input: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    input.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use cascadia_game::{GameConfig, GameSeed, GameState};
    use cascadia_sim::{select_greedy_action, strategy_rng};

    use crate::encode_public_features;

    use super::*;

    #[test]
    fn shard_round_trip_preserves_variable_sparse_rows() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(22),
        )
        .unwrap();
        let entry = V3TrainingEntry {
            state_blake3: *game.public_state().canonical_hash().as_bytes(),
            game_index: 7,
            decision_index: 0,
            focal_seat: 0,
            features: encode_public_features(&game.public_state(), 0).unwrap(),
            realized_score_to_go: 75.0,
            teacher_score_to_go: Some(80.0),
            teacher_variance: Some(1.5),
            teacher_sample_count: 600,
            lambda: 0.75,
            target_score_to_go: 78.75,
            provenance: V3TrainingProvenance::EngineeringSmoke,
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("v3-training-{nonce}.bin"));
        let mut writer = V3TrainingShardWriter::create(&path).unwrap();
        writer.append(&entry).unwrap();
        assert_eq!(writer.finish().unwrap(), 1);
        let mut reader = V3TrainingShardReader::open(&path).unwrap();
        assert_eq!(reader.next_entry().unwrap(), Some(entry));
        assert_eq!(reader.next_entry().unwrap(), None);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn engineering_smoke_is_never_scientific_data() {
        assert!(!V3TrainingProvenance::EngineeringSmoke.scientific_eligible());
        assert!(V3TrainingProvenance::ExpertIteration { cycle: 1 }.scientific_eligible());
    }

    #[test]
    fn teacher_confidence_uses_squared_standard_error() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(23),
        )
        .unwrap();
        let entry = V3TrainingEntry {
            state_blake3: *game.public_state().canonical_hash().as_bytes(),
            game_index: 8,
            decision_index: 0,
            focal_seat: 0,
            features: encode_public_features(&game.public_state(), 0).unwrap(),
            realized_score_to_go: 75.0,
            teacher_score_to_go: Some(80.0),
            teacher_variance: Some(4.0),
            teacher_sample_count: 16,
            lambda: 1.0,
            target_score_to_go: 80.0,
            provenance: V3TrainingProvenance::EngineeringSmoke,
        };
        assert!((entry.confidence_weight() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn score_to_go_preserves_negative_motif_changes() {
        assert_eq!(signed_score_to_go(73, 76), -3.0);
    }

    #[test]
    fn compact_game_shard_replays_to_eighty_exact_entries() {
        let seed = GameSeed::from_u64(24);
        let config = GameConfig::research_aaaaa(4).unwrap();
        let mut game = GameState::new(config, seed).unwrap();
        let mut replay = Replay::new(config, seed);
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "v3-compact-game-test-v1"))
            .collect::<Vec<_>>();
        while !game.is_game_over() {
            let seat = game.current_player();
            let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
            let action = select_greedy_action(&game, &prelude, &mut rngs[seat]).unwrap();
            game.apply(&action).unwrap();
            replay.turns.push(action);
        }
        replay.seal().unwrap();
        let record = V3GameRecord {
            game_index: 24,
            replay,
            seat_policy_ids: std::array::from_fn(|_| "greedy-v1".to_owned()),
            newest_model_id: None,
            focal_training_seat: None,
            exploration_epsilon: 0.1,
            provenance: V3TrainingProvenance::EngineeringSmoke,
        };
        let encoded = postcard::to_allocvec(&record).unwrap();
        assert!(encoded.len() < 16 * 1024);
        assert_eq!(record.training_entries().unwrap().len(), 80);

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("v3-games-{nonce}.bin"));
        let mut writer = V3GameShardWriter::create(&path).unwrap();
        writer.append(&record).unwrap();
        assert_eq!(writer.finish().unwrap(), 1);
        let mut reader = V3GameShardReader::open(&path).unwrap();
        assert_eq!(reader.next_record().unwrap(), Some(record));
        assert_eq!(reader.next_record().unwrap(), None);
        std::fs::remove_file(path).unwrap();
    }
}
