//! Typed client for the long-lived local MLX inference process.

mod r2_map;

pub use r2_map::{
    R2_MAP_MARKET_REQUEST_SCHEMA, R2_MAP_MARKET_REQUEST_SCHEMA_BLAKE3,
    R2_MAP_MARKET_RESPONSE_SCHEMA, R2_MAP_MARKET_RESPONSE_SCHEMA_BLAKE3,
    R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP, R2_MAP_SERVING_BUNDLE_SCHEMA,
    R2MapInferenceCandidate, R2MapInferenceGroup, R2MapMarketInferenceCandidate,
    R2MapMarketInferenceGroup, R2MapMarketPredictionGroup, R2MapModelError, R2MapModelIdentity,
    R2MapModelProcess, R2MapPredictionGroup, R2MapServingBundle, R2MapServingBundleEntry,
    ordered_market_action_ids_blake3, ordered_r2_map_action_ids_blake3,
};

use std::{
    ffi::{OsStr, OsString},
    io::{Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{Ordering, compiler_fence},
};

use cascadia_data::{ActionPositionRecord, PositionRecord, ProposalActionFeatures, TARGET_DIM};
use memmap2::{MmapMut, MmapOptions};
use tempfile::NamedTempFile;
use thiserror::Error;
use zerocopy::IntoBytes;

pub const PROTOCOL_MAGIC: [u8; 4] = *b"CMLX";
pub const PROTOCOL_VERSION: u16 = 1;
pub const MESSAGE_PREDICT: u16 = 1;
pub const MESSAGE_SHUTDOWN: u16 = 2;
pub const MESSAGE_PREDICT_ACTION_RANKING: u16 = 3;
pub const MESSAGE_PREDICT_IMITATION: u16 = 4;
pub const MESSAGE_PREDICT_SPARSE_NNUE: u16 = 5;
pub const MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT: u16 = 6;
pub const MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN: u16 = 7;
pub const MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED: u16 = 8;
pub const MESSAGE_PREDICTION: u16 = 0x8001;
pub const MESSAGE_RANKING_PREDICTION: u16 = 0x8002;
pub const MESSAGE_ACTION_RANKING_PREDICTION: u16 = 0x8003;
pub const MESSAGE_IMITATION_PREDICTION: u16 = 0x8004;
pub const MESSAGE_SPARSE_NNUE_PREDICTION: u16 = 0x8005;
pub const MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION: u16 = 0x8006;
pub const MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION: u16 = 0x8007;
pub const MESSAGE_SPARSE_NNUE_CSR_EXACT_SHARED_PREDICTION: u16 = 0x8008;
pub const MESSAGE_ERROR: u16 = 0xffff;
pub const FRAME_HEADER_SIZE: usize = 16;
pub const MAX_BATCH: usize = 65_536;
pub const LEGACY_NNUE_FEATURES: usize = 11_231;
pub const LEGACY_NNUE_HIDDEN2: usize = 64;
pub const MAX_SPARSE_FEATURES_PER_ROW: usize = 4_096;
pub const DEFAULT_SPARSE_NNUE_SHARED_MEMORY_BYTES: usize = 8 * 1024 * 1024;

const SPARSE_SHARED_MAGIC: [u8; 4] = *b"CSHM";
const SPARSE_SHARED_VERSION: u32 = 1;
const SPARSE_SHARED_HEADER_SIZE: usize = 16;

pub type Prediction = [f32; TARGET_DIM];

#[derive(Debug, Clone, PartialEq)]
pub struct ExactNnueHiddenPrediction {
    pub hidden: [f32; LEGACY_NNUE_HIDDEN2],
    pub value: f32,
}

pub struct ModelProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    next_request_id: u32,
    request_buffer: Vec<u8>,
    sparse_shared_memory: Option<SparseNnueSharedMemory>,
}

struct SparseNnueSharedMemory {
    _file: NamedTempFile,
    mapping: MmapMut,
}

impl ModelProcess {
    pub fn spawn<I, S>(program: impl AsRef<OsStr>, args: I) -> Result<Self, ModelError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().ok_or(ModelError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(ModelError::MissingPipe("stdout"))?;
        Ok(Self {
            child,
            stdin,
            stdout,
            next_request_id: 1,
            request_buffer: Vec::new(),
            sparse_shared_memory: None,
        })
    }

    pub fn spawn_with_sparse_nnue_shared_memory<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        capacity: usize,
    ) -> Result<Self, ModelError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        if capacity < SPARSE_SHARED_HEADER_SIZE + 8 {
            return Err(ModelError::SharedMemoryTooSmall {
                required: SPARSE_SHARED_HEADER_SIZE + 8,
                capacity,
            });
        }
        let file = NamedTempFile::new()?;
        file.as_file().set_len(capacity as u64)?;
        let mapping = unsafe { MmapOptions::new().len(capacity).map_mut(file.as_file())? };
        let mut args = args
            .into_iter()
            .map(|arg| arg.as_ref().to_os_string())
            .collect::<Vec<OsString>>();
        args.push(OsString::from("--shared-memory"));
        args.push(file.path().as_os_str().to_os_string());
        let mut process = Self::spawn(program, args)?;
        process.sparse_shared_memory = Some(SparseNnueSharedMemory {
            _file: file,
            mapping,
        });
        Ok(process)
    }

    pub fn predict(&mut self, records: &[PositionRecord]) -> Result<Vec<Prediction>, ModelError> {
        let values = self.predict_values(
            records.len(),
            MESSAGE_PREDICT,
            MESSAGE_PREDICTION,
            TARGET_DIM,
            |stdin| {
                for record in records {
                    stdin.write_all(&record.to_bytes())?;
                }
                Ok(())
            },
        )?;
        Ok(values
            .chunks_exact(TARGET_DIM)
            .map(|values| values.try_into().expect("prediction width is exact"))
            .collect())
    }

    pub fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, ModelError> {
        self.predict_values(
            records.len(),
            MESSAGE_PREDICT,
            MESSAGE_RANKING_PREDICTION,
            1,
            |stdin| {
                for record in records {
                    stdin.write_all(&record.to_bytes())?;
                }
                Ok(())
            },
        )
    }

    pub fn predict_action_scores(
        &mut self,
        records: &[ActionPositionRecord],
    ) -> Result<Vec<f32>, ModelError> {
        self.predict_values(
            records.len(),
            MESSAGE_PREDICT_ACTION_RANKING,
            MESSAGE_ACTION_RANKING_PREDICTION,
            1,
            |stdin| {
                for record in records {
                    stdin.write_all(&record.to_bytes())?;
                }
                Ok(())
            },
        )
    }

    pub fn predict_imitation_scores(
        &mut self,
        position: &PositionRecord,
        actions: &[ProposalActionFeatures],
    ) -> Result<Vec<f32>, ModelError> {
        self.predict_values(
            actions.len(),
            MESSAGE_PREDICT_IMITATION,
            MESSAGE_IMITATION_PREDICTION,
            1,
            |stdin| {
                stdin.write_all(&position.to_bytes())?;
                for action in actions {
                    stdin.write_all(&action.to_bytes())?;
                }
                Ok(())
            },
        )
    }

    pub fn predict_sparse_nnue(
        &mut self,
        feature_sets: &[Vec<u16>],
    ) -> Result<Vec<f32>, ModelError> {
        validate_sparse_features(feature_sets)?;
        self.predict_values(
            feature_sets.len(),
            MESSAGE_PREDICT_SPARSE_NNUE,
            MESSAGE_SPARSE_NNUE_PREDICTION,
            1,
            |stdin| write_sparse_payload(stdin, feature_sets),
        )
    }

    pub fn predict_sparse_nnue_csr_exact(
        &mut self,
        feature_sets: &[Vec<u16>],
    ) -> Result<Vec<f32>, ModelError> {
        validate_sparse_features(feature_sets)?;
        if self.sparse_shared_memory.is_some() {
            return self.predict_sparse_nnue_csr_exact_shared(feature_sets);
        }
        let mut payload = std::mem::take(&mut self.request_buffer);
        encode_sparse_csr_payload(&mut payload, feature_sets);
        let result = self.predict_values(
            feature_sets.len(),
            MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
            MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION,
            1,
            |stdin| stdin.write_all(&payload),
        );
        payload.clear();
        self.request_buffer = payload;
        result
    }

    fn predict_sparse_nnue_csr_exact_shared(
        &mut self,
        feature_sets: &[Vec<u16>],
    ) -> Result<Vec<f32>, ModelError> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        let response_offset = encode_sparse_csr_shared(
            &mut self
                .sparse_shared_memory
                .as_mut()
                .expect("shared sparse transport presence was checked")
                .mapping,
            request_id,
            feature_sets,
        )?;
        compiler_fence(Ordering::Release);
        self.stdin.write_all(&encode_header(FrameHeader {
            message_type: MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED,
            request_id,
            count: feature_sets.len() as u32,
        }))?;
        self.stdin.flush()?;

        let response = read_header(&mut self.stdout)?;
        if response.request_id != request_id {
            return Err(ModelError::RequestIdMismatch {
                expected: request_id,
                actual: response.request_id,
            });
        }
        if response.message_type == MESSAGE_ERROR {
            let mut message = vec![0; response.count as usize];
            self.stdout.read_exact(&mut message)?;
            return Err(ModelError::Service(
                String::from_utf8_lossy(&message).into_owned(),
            ));
        }
        if response.message_type != MESSAGE_SPARSE_NNUE_CSR_EXACT_SHARED_PREDICTION
            || response.count as usize != feature_sets.len()
        {
            return Err(ModelError::InvalidResponse);
        }

        compiler_fence(Ordering::Acquire);
        let prediction_bytes = &self
            .sparse_shared_memory
            .as_ref()
            .expect("shared sparse transport presence was checked")
            .mapping[response_offset..response_offset + feature_sets.len() * 4];
        let predictions = prediction_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        if let Some((index, _)) = predictions
            .iter()
            .enumerate()
            .find(|(_, prediction)| !prediction.is_finite())
        {
            return Err(ModelError::NonFinitePrediction(index));
        }
        Ok(predictions)
    }

    pub fn predict_sparse_nnue_csr_exact_hidden(
        &mut self,
        feature_sets: &[Vec<u16>],
    ) -> Result<Vec<ExactNnueHiddenPrediction>, ModelError> {
        validate_sparse_features(feature_sets)?;
        let mut payload = std::mem::take(&mut self.request_buffer);
        encode_sparse_csr_payload(&mut payload, feature_sets);
        let values = self.predict_values(
            feature_sets.len(),
            MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN,
            MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION,
            LEGACY_NNUE_HIDDEN2 + 1,
            |stdin| stdin.write_all(&payload),
        );
        payload.clear();
        self.request_buffer = payload;
        let values = values?;
        Ok(values
            .chunks_exact(LEGACY_NNUE_HIDDEN2 + 1)
            .map(|row| ExactNnueHiddenPrediction {
                hidden: row[..LEGACY_NNUE_HIDDEN2]
                    .try_into()
                    .expect("hidden prediction width is exact"),
                value: row[LEGACY_NNUE_HIDDEN2],
            })
            .collect())
    }

    fn predict_values(
        &mut self,
        count: usize,
        request_message_type: u16,
        expected_message_type: u16,
        output_width: usize,
        write_payload: impl FnOnce(&mut ChildStdin) -> std::io::Result<()>,
    ) -> Result<Vec<f32>, ModelError> {
        if count == 0 || count > MAX_BATCH {
            return Err(ModelError::InvalidBatchSize(count));
        }
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        self.stdin.write_all(&encode_header(FrameHeader {
            message_type: request_message_type,
            request_id,
            count: count as u32,
        }))?;
        write_payload(&mut self.stdin)?;
        self.stdin.flush()?;

        let response = read_header(&mut self.stdout)?;
        if response.request_id != request_id {
            return Err(ModelError::RequestIdMismatch {
                expected: request_id,
                actual: response.request_id,
            });
        }
        if response.message_type == MESSAGE_ERROR {
            let mut message = vec![0; response.count as usize];
            self.stdout.read_exact(&mut message)?;
            return Err(ModelError::Service(
                String::from_utf8_lossy(&message).into_owned(),
            ));
        }
        if response.message_type != expected_message_type || response.count as usize != count {
            return Err(ModelError::InvalidResponse);
        }

        let mut predictions = vec![0.0f32; count * output_width];
        #[cfg(target_endian = "little")]
        self.stdout
            .read_exact(predictions.as_mut_slice().as_mut_bytes())?;
        #[cfg(target_endian = "big")]
        {
            let mut prediction_bytes = vec![0u8; count * output_width * 4];
            self.stdout.read_exact(&mut prediction_bytes)?;
            for (prediction, bytes) in predictions.iter_mut().zip(prediction_bytes.chunks_exact(4))
            {
                *prediction = f32::from_le_bytes(bytes.try_into().expect("four-byte float"));
            }
        }
        if let Some((index, _)) = predictions
            .iter()
            .enumerate()
            .find(|(_, prediction)| !prediction.is_finite())
        {
            return Err(ModelError::NonFinitePrediction(index));
        }
        Ok(predictions)
    }

    pub fn shutdown(mut self) -> Result<(), ModelError> {
        self.stdin.write_all(&encode_header(FrameHeader {
            message_type: MESSAGE_SHUTDOWN,
            request_id: self.next_request_id,
            count: 0,
        }))?;
        self.stdin.flush()?;
        drop(self.stdin);
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(ModelError::ProcessExit(status.code()))
        }
    }
}

fn validate_sparse_features(feature_sets: &[Vec<u16>]) -> Result<(), ModelError> {
    if feature_sets.is_empty() || feature_sets.len() > MAX_BATCH {
        return Err(ModelError::InvalidBatchSize(feature_sets.len()));
    }
    for (row, features) in feature_sets.iter().enumerate() {
        if features.len() > MAX_SPARSE_FEATURES_PER_ROW {
            return Err(ModelError::SparseRowTooWide {
                row,
                features: features.len(),
            });
        }
        if let Some(&feature) = features
            .iter()
            .find(|&&feature| feature as usize >= LEGACY_NNUE_FEATURES)
        {
            return Err(ModelError::SparseFeatureOutOfRange { row, feature });
        }
    }
    Ok(())
}

fn write_sparse_payload(writer: &mut impl Write, feature_sets: &[Vec<u16>]) -> std::io::Result<()> {
    let total_features = feature_sets.iter().map(Vec::len).sum::<usize>();
    let mut payload = Vec::with_capacity(feature_sets.len() * 2 + total_features * 2);
    for features in feature_sets {
        payload.extend_from_slice(&(features.len() as u16).to_le_bytes());
        for &feature in features {
            payload.extend_from_slice(&feature.to_le_bytes());
        }
    }
    writer.write_all(&payload)
}

#[cfg(test)]
fn write_sparse_csr_payload(
    writer: &mut impl Write,
    feature_sets: &[Vec<u16>],
) -> std::io::Result<()> {
    let mut payload = Vec::new();
    encode_sparse_csr_payload(&mut payload, feature_sets);
    writer.write_all(&payload)
}

fn encode_sparse_csr_payload(payload: &mut Vec<u8>, feature_sets: &[Vec<u16>]) {
    let total_features = feature_sets.iter().map(Vec::len).sum::<usize>();
    let payload_bytes = 4 + (feature_sets.len() + 1) * 4 + total_features * 2;
    payload.clear();
    payload.reserve(payload_bytes);
    payload.extend_from_slice(&(total_features as u32).to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    let mut offset = 0u32;
    for features in feature_sets {
        offset += features.len() as u32;
        payload.extend_from_slice(&offset.to_le_bytes());
    }
    for features in feature_sets {
        #[cfg(target_endian = "little")]
        payload.extend_from_slice(features.as_slice().as_bytes());
        #[cfg(target_endian = "big")]
        for &feature in features {
            payload.extend_from_slice(&feature.to_le_bytes());
        }
    }
    debug_assert_eq!(payload.len(), payload_bytes);
}

fn encode_sparse_csr_shared(
    mapping: &mut [u8],
    request_id: u32,
    feature_sets: &[Vec<u16>],
) -> Result<usize, ModelError> {
    let total_features = feature_sets.iter().map(Vec::len).sum::<usize>();
    let offsets_start = SPARSE_SHARED_HEADER_SIZE;
    let features_start = offsets_start + (feature_sets.len() + 1) * 4;
    let features_end = features_start + total_features * 2;
    let response_offset = align_to_four(features_end);
    let required = response_offset + feature_sets.len() * 4;
    if required > mapping.len() {
        return Err(ModelError::SharedMemoryTooSmall {
            required,
            capacity: mapping.len(),
        });
    }
    write_sparse_csr_shared_input(
        mapping,
        request_id,
        feature_sets,
        total_features,
        offsets_start,
        features_start,
    );
    mapping[features_end..response_offset].fill(0);
    Ok(response_offset)
}

fn write_sparse_csr_shared_input(
    mapping: &mut [u8],
    request_id: u32,
    feature_sets: &[Vec<u16>],
    total_features: usize,
    offsets_start: usize,
    features_start: usize,
) {
    mapping[..4].copy_from_slice(&SPARSE_SHARED_MAGIC);
    mapping[4..8].copy_from_slice(&SPARSE_SHARED_VERSION.to_le_bytes());
    mapping[8..12].copy_from_slice(&request_id.to_le_bytes());
    mapping[12..16].copy_from_slice(&(total_features as u32).to_le_bytes());
    let mut offset = 0_u32;
    mapping[offsets_start..offsets_start + 4].copy_from_slice(&offset.to_le_bytes());
    for (row, features) in feature_sets.iter().enumerate() {
        offset += features.len() as u32;
        let start = offsets_start + (row + 1) * 4;
        mapping[start..start + 4].copy_from_slice(&offset.to_le_bytes());
    }
    let mut feature_offset = features_start;
    for features in feature_sets {
        #[cfg(target_endian = "little")]
        {
            let bytes = features.as_bytes();
            mapping[feature_offset..feature_offset + bytes.len()].copy_from_slice(bytes);
            feature_offset += bytes.len();
        }
        #[cfg(target_endian = "big")]
        for &feature in features {
            mapping[feature_offset..feature_offset + 2].copy_from_slice(&feature.to_le_bytes());
            feature_offset += 2;
        }
    }
    debug_assert_eq!(feature_offset, features_start + total_features * 2);
}

const fn align_to_four(value: usize) -> usize {
    (value + 3) & !3
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameHeader {
    message_type: u16,
    request_id: u32,
    count: u32,
}

fn encode_header(header: FrameHeader) -> [u8; FRAME_HEADER_SIZE] {
    let mut bytes = [0u8; FRAME_HEADER_SIZE];
    bytes[..4].copy_from_slice(&PROTOCOL_MAGIC);
    bytes[4..6].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    bytes[6..8].copy_from_slice(&header.message_type.to_le_bytes());
    bytes[8..12].copy_from_slice(&header.request_id.to_le_bytes());
    bytes[12..16].copy_from_slice(&header.count.to_le_bytes());
    bytes
}

fn read_header(reader: &mut impl Read) -> Result<FrameHeader, ModelError> {
    let mut bytes = [0u8; FRAME_HEADER_SIZE];
    reader.read_exact(&mut bytes)?;
    if bytes[..4] != PROTOCOL_MAGIC
        || u16::from_le_bytes(bytes[4..6].try_into().expect("fixed header")) != PROTOCOL_VERSION
    {
        return Err(ModelError::ProtocolMismatch);
    }
    Ok(FrameHeader {
        message_type: u16::from_le_bytes(bytes[6..8].try_into().expect("fixed header")),
        request_id: u32::from_le_bytes(bytes[8..12].try_into().expect("fixed header")),
        count: u32::from_le_bytes(bytes[12..16].try_into().expect("fixed header")),
    })
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model batch size {0} is invalid")]
    InvalidBatchSize(usize),
    #[error("spawned model process did not expose {0}")]
    MissingPipe(&'static str),
    #[error("model protocol version or magic did not match")]
    ProtocolMismatch,
    #[error("model response did not match the request")]
    InvalidResponse,
    #[error("model response request id {actual} did not match {expected}")]
    RequestIdMismatch { expected: u32, actual: u32 },
    #[error("sparse NNUE row {row} has {features} features, exceeding the protocol maximum")]
    SparseRowTooWide { row: usize, features: usize },
    #[error("sparse NNUE row {row} contains out-of-range feature {feature}")]
    SparseFeatureOutOfRange { row: usize, feature: u16 },
    #[error("sparse NNUE shared memory requires {required} bytes but capacity is {capacity} bytes")]
    SharedMemoryTooSmall { required: usize, capacity: usize },
    #[error("model prediction {0} was not finite")]
    NonFinitePrediction(usize),
    #[error("MLX model service rejected the request: {0}")]
    Service(String),
    #[error("MLX model process exited unsuccessfully with code {0:?}")]
    ProcessExit(Option<i32>),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_round_trip_is_little_endian_and_fixed_width() {
        let expected = FrameHeader {
            message_type: MESSAGE_PREDICT,
            request_id: 0x1020_3040,
            count: 513,
        };
        let bytes = encode_header(expected);
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
        assert_eq!(&bytes[..4], b"CMLX");
        assert_eq!(read_header(&mut bytes.as_slice()).unwrap(), expected);
    }

    #[test]
    fn malformed_header_is_rejected() {
        let mut bytes = encode_header(FrameHeader {
            message_type: MESSAGE_PREDICT,
            request_id: 1,
            count: 1,
        });
        bytes[0] = b'X';
        assert!(matches!(
            read_header(&mut bytes.as_slice()),
            Err(ModelError::ProtocolMismatch)
        ));
    }

    #[test]
    fn sparse_payload_is_little_endian_and_preserves_duplicates() {
        let features = vec![vec![], vec![1, 1, 11_230]];
        validate_sparse_features(&features).unwrap();
        let mut bytes = Vec::new();
        write_sparse_payload(&mut bytes, &features).unwrap();
        assert_eq!(
            bytes,
            vec![
                0, 0, // empty first row
                3, 0, // three features
                1, 0, 1, 0, 0xde, 0x2b,
            ]
        );
    }

    #[test]
    fn sparse_payload_rejects_out_of_range_features() {
        let error = validate_sparse_features(&[vec![11_231]]).unwrap_err();
        assert!(matches!(
            error,
            ModelError::SparseFeatureOutOfRange {
                row: 0,
                feature: 11_231
            }
        ));
    }

    #[test]
    fn exact_sparse_csr_payload_is_little_endian_and_preserves_duplicates() {
        let features = vec![vec![], vec![1, 1, 11_230]];
        validate_sparse_features(&features).unwrap();
        let mut bytes = Vec::new();
        write_sparse_csr_payload(&mut bytes, &features).unwrap();
        assert_eq!(
            bytes,
            vec![
                3, 0, 0, 0, // total features
                0, 0, 0, 0, // row zero start
                0, 0, 0, 0, // row one start
                3, 0, 0, 0, // payload end
                1, 0, 1, 0, 0xde, 0x2b,
            ]
        );
    }

    #[test]
    fn exact_sparse_shared_layout_is_bounded_and_little_endian() {
        let features = vec![vec![], vec![1, 1, 11_230]];
        let mut mapping = vec![0_u8; 128];
        let response_offset = encode_sparse_csr_shared(&mut mapping, 41, &features).unwrap();

        assert_eq!(&mapping[..4], b"CSHM");
        assert_eq!(u32::from_le_bytes(mapping[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(mapping[8..12].try_into().unwrap()), 41);
        assert_eq!(u32::from_le_bytes(mapping[12..16].try_into().unwrap()), 3);
        assert_eq!(
            &mapping[16..response_offset],
            &[
                0, 0, 0, 0, // row zero start
                0, 0, 0, 0, // row one start
                3, 0, 0, 0, // payload end
                1, 0, 1, 0, 0xde, 0x2b, // sparse features
                0, 0, // alignment before float output
            ]
        );
        assert_eq!(response_offset, 36);

        let error = encode_sparse_csr_shared(&mut mapping[..35], 41, &features).unwrap_err();
        assert!(matches!(
            error,
            ModelError::SharedMemoryTooSmall {
                required: 44,
                capacity: 35
            }
        ));
    }
}
