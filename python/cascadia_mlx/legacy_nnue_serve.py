"""Long-lived sparse inference service for the qualified legacy NNUE."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import BinaryIO

import mlx.core as mx
import numpy as np

from cascadia_mlx.legacy_nnue import (
    LEGACY_NNUE_FEATURES,
    LEGACY_NNUE_HIDDEN2,
    LegacyRustExactSparseNnue,
    LegacySparseNnue,
    pack_sparse_features,
)
from cascadia_mlx.serve import (
    FRAME_HEADER,
    MAX_BATCH,
    MESSAGE_ERROR,
    MESSAGE_SHUTDOWN,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ProtocolError,
    _read_exact,
    _read_exact_or_eof,
)

MESSAGE_PREDICT_SPARSE_NNUE = 5
MESSAGE_SPARSE_NNUE_PREDICTION = 0x8005
MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT = 6
MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION = 0x8006
MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN = 7
MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION = 0x8007
MAX_SPARSE_FEATURES_PER_ROW = 4_096
ROW_LENGTH = struct.Struct("<H")
TOTAL_FEATURES = struct.Struct("<I")


def serve_legacy_nnue(
    model: LegacySparseNnue,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    exact_model: LegacyRustExactSparseNnue | None = None,
) -> None:
    """Serve ordered variable-length sparse batches until shutdown or EOF."""
    while True:
        header = _read_exact_or_eof(input_stream, FRAME_HEADER.size)
        if header is None:
            return
        magic, version, message_type, request_id, count = FRAME_HEADER.unpack(header)
        try:
            if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
                raise ProtocolError("incompatible protocol header")
            if message_type == MESSAGE_SHUTDOWN:
                if count:
                    raise ProtocolError("shutdown frame cannot contain records")
                return
            if message_type not in (
                MESSAGE_PREDICT_SPARSE_NNUE,
                MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
                MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN,
            ):
                raise ProtocolError(f"unsupported message type {message_type}")
            if count == 0 or count > MAX_BATCH:
                raise ProtocolError(f"invalid prediction batch size {count}")

            if message_type == MESSAGE_PREDICT_SPARSE_NNUE:
                feature_sets = []
                for row in range(count):
                    feature_count = ROW_LENGTH.unpack(_read_exact(input_stream, ROW_LENGTH.size))[0]
                    if feature_count > MAX_SPARSE_FEATURES_PER_ROW:
                        raise ProtocolError(
                            f"sparse row {row} has {feature_count} features, "
                            f"maximum is {MAX_SPARSE_FEATURES_PER_ROW}"
                        )
                    payload = _read_exact(input_stream, feature_count * 2)
                    features = np.frombuffer(payload, dtype="<u2").astype(np.int32).tolist()
                    if any(index >= LEGACY_NNUE_FEATURES for index in features):
                        raise ProtocolError(f"sparse row {row} contains an out-of-range index")
                    feature_sets.append(features)
                indices, mask = pack_sparse_features(feature_sets)
                predictions = model(indices, mask)
                response_type = MESSAGE_SPARSE_NNUE_PREDICTION
            else:
                if exact_model is None:
                    raise ProtocolError("exact sparse NNUE operation is unavailable")
                total_features = TOTAL_FEATURES.unpack(
                    _read_exact(input_stream, TOTAL_FEATURES.size)
                )[0]
                if total_features > count * MAX_SPARSE_FEATURES_PER_ROW:
                    raise ProtocolError(
                        f"exact sparse batch has {total_features} features, "
                        f"maximum is {count * MAX_SPARSE_FEATURES_PER_ROW}"
                    )
                offsets = np.frombuffer(
                    _read_exact(input_stream, (count + 1) * 4),
                    dtype="<u4",
                )
                if offsets[0] != 0 or offsets[-1] != total_features:
                    raise ProtocolError("exact sparse offsets do not span the payload")
                widths = np.diff(offsets.astype(np.int64))
                if np.any(widths < 0) or np.any(widths > MAX_SPARSE_FEATURES_PER_ROW):
                    raise ProtocolError("exact sparse offsets contain an invalid row width")
                features = np.frombuffer(
                    _read_exact(input_stream, total_features * 2),
                    dtype="<u2",
                )
                if np.any(features >= LEGACY_NNUE_FEATURES):
                    raise ProtocolError("exact sparse batch contains an out-of-range index")
                exact_offsets = mx.array(offsets.astype(np.int32))
                exact_features = mx.array(features.astype(np.int32))
                if message_type == MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT:
                    predictions = exact_model(exact_offsets, exact_features)
                    response_type = MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION
                else:
                    hidden, values = exact_model.hidden_and_output(exact_offsets, exact_features)
                    predictions = mx.concatenate([hidden, values[:, None]], axis=1)
                    response_type = MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION
            mx.eval(predictions)
            values = np.asarray(predictions, dtype=np.float32)
            expected_shape = (
                (count, LEGACY_NNUE_HIDDEN2 + 1)
                if message_type == MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN
                else (count,)
            )
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise ProtocolError("sparse NNUE returned an invalid prediction tensor")
            output_stream.write(
                FRAME_HEADER.pack(
                    PROTOCOL_MAGIC,
                    PROTOCOL_VERSION,
                    response_type,
                    request_id,
                    count,
                )
            )
            output_stream.write(values.astype("<f4", copy=False).tobytes(order="C"))
            output_stream.flush()
        except Exception as error:
            message = str(error).encode()
            output_stream.write(
                FRAME_HEADER.pack(
                    PROTOCOL_MAGIC,
                    PROTOCOL_VERSION,
                    MESSAGE_ERROR,
                    request_id,
                    len(message),
                )
            )
            output_stream.write(message)
            output_stream.flush()
            if isinstance(error, (EOFError, ProtocolError)):
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    model = LegacySparseNnue.load(args.model_dir)
    exact_model = LegacyRustExactSparseNnue(model.tensors)
    print(
        f"cascadia-mlx legacy NNUE serving {args.model_dir.name} on {mx.default_device()}",
        file=sys.stderr,
    )
    serve_legacy_nnue(model, sys.stdin.buffer, sys.stdout.buffer, exact_model)


if __name__ == "__main__":
    main()
