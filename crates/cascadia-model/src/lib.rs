//! Typed client for the long-lived local MLX inference process.

use std::{
    ffi::OsStr,
    io::{Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use cascadia_data::{ActionPositionRecord, PositionRecord, ProposalActionFeatures, TARGET_DIM};
use thiserror::Error;

pub const PROTOCOL_MAGIC: [u8; 4] = *b"CMLX";
pub const PROTOCOL_VERSION: u16 = 1;
pub const MESSAGE_PREDICT: u16 = 1;
pub const MESSAGE_SHUTDOWN: u16 = 2;
pub const MESSAGE_PREDICT_ACTION_RANKING: u16 = 3;
pub const MESSAGE_PREDICT_IMITATION: u16 = 4;
pub const MESSAGE_PREDICT_SPARSE_NNUE: u16 = 5;
pub const MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT: u16 = 6;
pub const MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN: u16 = 7;
pub const MESSAGE_PREDICTION: u16 = 0x8001;
pub const MESSAGE_RANKING_PREDICTION: u16 = 0x8002;
pub const MESSAGE_ACTION_RANKING_PREDICTION: u16 = 0x8003;
pub const MESSAGE_IMITATION_PREDICTION: u16 = 0x8004;
pub const MESSAGE_SPARSE_NNUE_PREDICTION: u16 = 0x8005;
pub const MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION: u16 = 0x8006;
pub const MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION: u16 = 0x8007;
pub const MESSAGE_ERROR: u16 = 0xffff;
pub const FRAME_HEADER_SIZE: usize = 16;
pub const MAX_BATCH: usize = 65_536;
pub const LEGACY_NNUE_FEATURES: usize = 11_231;
pub const LEGACY_NNUE_HIDDEN2: usize = 64;
pub const MAX_SPARSE_FEATURES_PER_ROW: usize = 4_096;

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
        })
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
        self.predict_values(
            feature_sets.len(),
            MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
            MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION,
            1,
            |stdin| write_sparse_csr_payload(stdin, feature_sets),
        )
    }

    pub fn predict_sparse_nnue_csr_exact_hidden(
        &mut self,
        feature_sets: &[Vec<u16>],
    ) -> Result<Vec<ExactNnueHiddenPrediction>, ModelError> {
        validate_sparse_features(feature_sets)?;
        let values = self.predict_values(
            feature_sets.len(),
            MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN,
            MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION,
            LEGACY_NNUE_HIDDEN2 + 1,
            |stdin| write_sparse_csr_payload(stdin, feature_sets),
        )?;
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

        let mut prediction_bytes = vec![0u8; count * output_width * 4];
        self.stdout.read_exact(&mut prediction_bytes)?;
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

fn write_sparse_csr_payload(
    writer: &mut impl Write,
    feature_sets: &[Vec<u16>],
) -> std::io::Result<()> {
    let total_features = feature_sets.iter().map(Vec::len).sum::<usize>();
    let mut payload = Vec::with_capacity(4 + (feature_sets.len() + 1) * 4 + total_features * 2);
    payload.extend_from_slice(&(total_features as u32).to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    let mut offset = 0u32;
    for features in feature_sets {
        offset += features.len() as u32;
        payload.extend_from_slice(&offset.to_le_bytes());
    }
    for features in feature_sets {
        for &feature in features {
            payload.extend_from_slice(&feature.to_le_bytes());
        }
    }
    writer.write_all(&payload)
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
}
