use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use cascadia_game::{GameState, score_board, score_game};
use serde::{Deserialize, Serialize};

use crate::{
    Result, V3Error, V3GameRecord, V3TeacherRootLabel, V3TrainingEntry, encode_public_features,
    signed_score_to_go,
};

pub const TEACHER_ROOT_SHARD_MAGIC: &[u8; 8] = b"CSV3RT1\0";
pub const LABELED_TEACHER_ROOT_SHARD_MAGIC: &[u8; 8] = b"CSV3LB1\0";
const TEACHER_ROOT_SHARD_VERSION: u16 = 1;
const LABELED_TEACHER_ROOT_SHARD_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum V3TeacherSplit {
    Teacher,
    Validation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct V3TeacherStratum {
    pub focal_seat: u8,
    pub phase_bucket: u8,
    pub nature_token_bin: u8,
    pub legal_width_bin: u8,
    pub score_to_go_bin: u8,
    pub market_signature_bin: u8,
}

impl V3TeacherStratum {
    pub fn validate(self) -> Result<()> {
        if self.focal_seat >= 4
            || self.phase_bucket >= 8
            || self.nature_token_bin >= 3
            || self.legal_width_bin >= 4
            || self.score_to_go_bin >= 5
            || self.market_signature_bin >= 16
        {
            return Err(V3Error::InvalidTraining(
                "teacher-root stratum is outside its registered bins".to_owned(),
            ));
        }
        Ok(())
    }
}

/// A replay-authoritative decision root. The full compact game is retained so
/// labeling never depends on mutable corpus paths or hidden-state snapshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3TeacherRoot {
    pub record: V3GameRecord,
    pub turn_index: u8,
    pub state_blake3: [u8; 32],
    pub split: V3TeacherSplit,
    pub stratum: V3TeacherStratum,
}

impl V3TeacherRoot {
    pub fn reconstruct(&self) -> Result<GameState> {
        self.record.validate()?;
        if usize::from(self.turn_index) >= self.record.replay.turns.len() {
            return Err(V3Error::InvalidTraining(
                "teacher root turn index is outside the replay".to_owned(),
            ));
        }
        let mut game = GameState::new(self.record.replay.config, self.record.replay.seed)?;
        for action in self
            .record
            .replay
            .turns
            .iter()
            .take(usize::from(self.turn_index))
        {
            game.apply(action)?;
        }
        Ok(game)
    }

    pub fn validate(&self) -> Result<()> {
        self.stratum.validate()?;
        let game = self.reconstruct()?;
        let focal = game.current_player();
        let completed = game.boards()[focal].tile_count().saturating_sub(3).min(20);
        let phase = ((8 * completed) / 20).min(7) as u8;
        if focal as u8 != self.stratum.focal_seat
            || phase != self.stratum.phase_bucket
            || game.public_state().canonical_hash().as_bytes() != &self.state_blake3
        {
            return Err(V3Error::InvalidTraining(
                "teacher root metadata differs from replay reconstruction".to_owned(),
            ));
        }
        Ok(())
    }
}

pub struct V3TeacherRootShardWriter {
    output: BufWriter<File>,
    count: u64,
    finished: bool,
}

impl V3TeacherRootShardWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(TEACHER_ROOT_SHARD_MAGIC)?;
        output.write_all(&TEACHER_ROOT_SHARD_VERSION.to_le_bytes())?;
        output.write_all(&0u16.to_le_bytes())?;
        output.write_all(&0u64.to_le_bytes())?;
        Ok(Self {
            output,
            count: 0,
            finished: false,
        })
    }

    pub fn append(&mut self, root: &V3TeacherRoot) -> Result<()> {
        if self.finished {
            return Err(V3Error::InvalidTraining(
                "cannot append to a finished teacher-root shard".to_owned(),
            ));
        }
        root.validate()?;
        let bytes = postcard::to_allocvec(root)?;
        let length = u32::try_from(bytes.len())
            .map_err(|_| V3Error::InvalidTraining("teacher root exceeds u32 length".to_owned()))?;
        self.output.write_all(&length.to_le_bytes())?;
        self.output.write_all(&bytes)?;
        self.count += 1;
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

pub struct V3TeacherRootShardReader {
    input: BufReader<File>,
    remaining: u64,
    total: u64,
}

impl V3TeacherRootShardReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        let version = read_u16(&mut input)?;
        let reserved = read_u16(&mut input)?;
        let total = read_u64(&mut input)?;
        if &magic != TEACHER_ROOT_SHARD_MAGIC
            || version != TEACHER_ROOT_SHARD_VERSION
            || reserved != 0
        {
            return Err(V3Error::InvalidTraining(
                "teacher-root shard header is invalid".to_owned(),
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

    pub fn next_root(&mut self) -> Result<Option<V3TeacherRoot>> {
        if self.remaining == 0 {
            let mut trailing = [0u8; 1];
            if self.input.read(&mut trailing)? != 0 {
                return Err(V3Error::InvalidTraining(
                    "teacher-root shard contains trailing bytes".to_owned(),
                ));
            }
            return Ok(None);
        }
        let length = read_u32(&mut self.input)? as usize;
        if length == 0 || length > 16 * 1024 * 1024 {
            return Err(V3Error::InvalidTraining(
                "teacher-root record length is invalid".to_owned(),
            ));
        }
        let mut bytes = vec![0u8; length];
        self.input.read_exact(&mut bytes)?;
        let root: V3TeacherRoot = postcard::from_bytes(&bytes)?;
        root.validate()?;
        self.remaining -= 1;
        Ok(Some(root))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3LabeledTeacherRoot {
    pub root: V3TeacherRoot,
    pub teacher_id: String,
    pub label: V3TeacherRootLabel,
}

impl V3LabeledTeacherRoot {
    pub fn validate(&self) -> Result<()> {
        self.root.validate()?;
        self.label.validate()?;
        if self.teacher_id.is_empty()
            || self.label.state_blake3 != self.root.state_blake3
            || self.label.focal_seat != self.root.stratum.focal_seat
            || self.label.phase_bucket != self.root.stratum.phase_bucket
        {
            return Err(V3Error::InvalidTraining(
                "labeled teacher root identity is inconsistent".to_owned(),
            ));
        }
        let game = self.root.reconstruct()?;
        for candidate in &self.label.candidates {
            let mut after = game.clone();
            after.apply(&candidate.action).map_err(|error| {
                V3Error::InvalidTraining(format!("teacher candidate is illegal: {error}"))
            })?;
        }
        Ok(())
    }
}

/// Expand one compact labeled root on demand. Scientific training streams use
/// this path directly so the 120K labels remain compact instead of becoming a
/// roughly 10-GiB materialized sparse corpus.
pub fn labeled_root_training_entries(value: &V3LabeledTeacherRoot) -> Result<Vec<V3TrainingEntry>> {
    value.validate()?;
    let game = value.root.reconstruct()?;
    let focal = game.current_player();
    let decision_index = game.boards()[focal].tile_count().saturating_sub(3) as u8;
    let terminal = value.root.record.replay.play().map_err(|error| {
        V3Error::InvalidTraining(format!(
            "teacher replay failed to reach terminal state: {error}"
        ))
    })?;
    let final_score = score_game(&terminal)[focal].base_total;
    let realized_action = &value.root.record.replay.turns[usize::from(value.root.turn_index)];
    value
        .label
        .candidates
        .iter()
        .map(|candidate| {
            let mut after = game.clone();
            after.apply(&candidate.action)?;
            let current = score_board(
                &after.boards()[focal],
                value.root.record.replay.config.scoring_cards,
            )
            .base_total;
            let teacher_score_to_go = candidate.rollout_mean as f32 - f32::from(current);
            let realized_score_to_go = if &candidate.action == realized_action {
                signed_score_to_go(final_score, current)
            } else {
                // Counterfactuals have no observed terminal. Equal anchors make
                // lambda annealing affect only the replay-realized action.
                teacher_score_to_go
            };
            let features = encode_public_features(&after.public_state(), focal)?;
            let entry = V3TrainingEntry {
                state_blake3: *after.public_state().canonical_hash().as_bytes(),
                game_index: value.root.record.game_index,
                decision_index,
                focal_seat: focal as u8,
                features,
                realized_score_to_go,
                teacher_score_to_go: Some(teacher_score_to_go),
                teacher_variance: Some(candidate.rollout_variance as f32),
                teacher_sample_count: candidate.rollout_count,
                lambda: 1.0,
                target_score_to_go: teacher_score_to_go,
                provenance: value.root.record.provenance.clone(),
            };
            entry.validate()?;
            Ok(entry)
        })
        .collect()
}

pub struct V3LabeledTeacherRootShardWriter {
    output: BufWriter<File>,
    count: u64,
    finished: bool,
}

impl V3LabeledTeacherRootShardWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(LABELED_TEACHER_ROOT_SHARD_MAGIC)?;
        output.write_all(&LABELED_TEACHER_ROOT_SHARD_VERSION.to_le_bytes())?;
        output.write_all(&0u16.to_le_bytes())?;
        output.write_all(&0u64.to_le_bytes())?;
        Ok(Self {
            output,
            count: 0,
            finished: false,
        })
    }

    pub fn append(&mut self, value: &V3LabeledTeacherRoot) -> Result<()> {
        if self.finished {
            return Err(V3Error::InvalidTraining(
                "cannot append to a finished labeled-root shard".to_owned(),
            ));
        }
        value.validate()?;
        let bytes = postcard::to_allocvec(value)?;
        let length = u32::try_from(bytes.len()).map_err(|_| {
            V3Error::InvalidTraining("labeled teacher root exceeds u32 length".to_owned())
        })?;
        self.output.write_all(&length.to_le_bytes())?;
        self.output.write_all(&bytes)?;
        self.count += 1;
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

pub struct V3LabeledTeacherRootShardReader {
    input: BufReader<File>,
    remaining: u64,
    total: u64,
}

impl V3LabeledTeacherRootShardReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        let version = read_u16(&mut input)?;
        let reserved = read_u16(&mut input)?;
        let total = read_u64(&mut input)?;
        if &magic != LABELED_TEACHER_ROOT_SHARD_MAGIC
            || version != LABELED_TEACHER_ROOT_SHARD_VERSION
            || reserved != 0
        {
            return Err(V3Error::InvalidTraining(
                "labeled teacher-root shard header is invalid".to_owned(),
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

    pub fn next_labeled_root(&mut self) -> Result<Option<V3LabeledTeacherRoot>> {
        if self.remaining == 0 {
            let mut trailing = [0u8; 1];
            if self.input.read(&mut trailing)? != 0 {
                return Err(V3Error::InvalidTraining(
                    "labeled teacher-root shard contains trailing bytes".to_owned(),
                ));
            }
            return Ok(None);
        }
        let length = read_u32(&mut self.input)? as usize;
        if length == 0 || length > 64 * 1024 * 1024 {
            return Err(V3Error::InvalidTraining(
                "labeled teacher-root record length is invalid".to_owned(),
            ));
        }
        let mut bytes = vec![0u8; length];
        self.input.read_exact(&mut bytes)?;
        let value: V3LabeledTeacherRoot = postcard::from_bytes(&bytes)?;
        value.validate()?;
        self.remaining -= 1;
        Ok(Some(value))
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

    use cascadia_game::{GameConfig, GameSeed, GameState, Replay};
    use cascadia_sim::{select_greedy_action, strategy_rng};

    use crate::V3TrainingProvenance;

    use super::*;

    fn record() -> V3GameRecord {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let seed = GameSeed::from_u64(91_337);
        let mut game = GameState::new(config, seed).unwrap();
        let mut replay = Replay::new(config, seed);
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, "teacher-root-roundtrip-v1"))
            .collect::<Vec<_>>();
        while !game.is_game_over() {
            let focal = game.current_player();
            let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
            let action = select_greedy_action(&game, &prelude, &mut rngs[focal]).unwrap();
            game.apply(&action).unwrap();
            replay.turns.push(action);
        }
        replay.seal().unwrap();
        V3GameRecord {
            game_index: 91_337,
            replay,
            seat_policy_ids: std::array::from_fn(|_| "greedy".to_owned()),
            newest_model_id: None,
            focal_training_seat: None,
            exploration_epsilon: 0.0,
            provenance: V3TrainingProvenance::Bootstrap {
                component: "greedy".to_owned(),
            },
        }
    }

    #[test]
    fn root_shard_round_trip_is_replay_authoritative() {
        let record = record();
        let game = GameState::new(record.replay.config, record.replay.seed).unwrap();
        let root = V3TeacherRoot {
            record,
            turn_index: 0,
            state_blake3: *game.public_state().canonical_hash().as_bytes(),
            split: V3TeacherSplit::Teacher,
            stratum: V3TeacherStratum {
                focal_seat: 0,
                phase_bucket: 0,
                nature_token_bin: 0,
                legal_width_bin: 0,
                score_to_go_bin: 4,
                market_signature_bin: 0,
            },
        };
        root.validate().unwrap();
        let path = std::env::temp_dir().join(format!(
            "v3-teacher-root-{}.bin",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut writer = V3TeacherRootShardWriter::create(&path).unwrap();
        writer.append(&root).unwrap();
        assert_eq!(writer.finish().unwrap(), 1);
        let mut reader = V3TeacherRootShardReader::open(&path).unwrap();
        assert_eq!(reader.len(), 1);
        assert_eq!(reader.next_root().unwrap(), Some(root));
        assert!(reader.next_root().unwrap().is_none());
        std::fs::remove_file(path).unwrap();
    }
}
