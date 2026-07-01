"""Long-lived binary batch-inference service for the Rust engine."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import BinaryIO

import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import load_latest_checkpoint
from cascadia_mlx.dataset import RECORD_SIZE, decode_record_bytes
from cascadia_mlx.promote import load_promoted_model

PROTOCOL_MAGIC = b"CMLX"
PROTOCOL_VERSION = 1
MESSAGE_PREDICT = 1
MESSAGE_SHUTDOWN = 2
MESSAGE_PREDICTION = 0x8001
MESSAGE_ERROR = 0xFFFF
MAX_BATCH = 65_536
FRAME_HEADER = struct.Struct("<4sHHII")


class ProtocolError(ValueError):
    """Raised for a malformed local inference frame."""


def serve(
    model: object,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
) -> None:
    """Serve requests until a clean shutdown frame or EOF."""
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
            if message_type != MESSAGE_PREDICT:
                raise ProtocolError(f"unsupported message type {message_type}")
            if count == 0 or count > MAX_BATCH:
                raise ProtocolError(f"invalid prediction batch size {count}")
            payload = _read_exact(input_stream, count * RECORD_SIZE)
            batch = decode_record_bytes(payload, count)
            predictions = model.predict_components(
                batch.board_entities,
                batch.board_mask,
                batch.market_entities,
                batch.market_mask,
                batch.global_features,
            )
            mx.eval(predictions)
            response = np.asarray(predictions, dtype="<f4").tobytes(order="C")
            output_stream.write(
                FRAME_HEADER.pack(
                    PROTOCOL_MAGIC,
                    PROTOCOL_VERSION,
                    MESSAGE_PREDICTION,
                    request_id,
                    count,
                )
            )
            output_stream.write(response)
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
    """Load one checkpoint and serve binary requests on stdin/stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--model-dir", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if args.model_dir is not None:
        model = load_promoted_model(args.model_dir)
        source_name = args.model_dir.name
    else:
        model, _optimizer, _state, checkpoint = load_latest_checkpoint(
            args.run_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model.eval()
        source_name = checkpoint.name
    print(f"cascadia-mlx serving {source_name} on {mx.default_device()}", file=sys.stderr)
    serve(model, sys.stdin.buffer, sys.stdout.buffer)


def _read_exact_or_eof(stream: BinaryIO, size: int) -> bytes | None:
    first = stream.read(size)
    if not first:
        return None
    if len(first) == size:
        return first
    return first + _read_exact(stream, size - len(first))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    first = stream.read(size)
    if len(first) == size:
        return first
    if not first:
        raise EOFError("inference stream ended inside a frame")
    chunks = bytearray(first)
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("inference stream ended inside a frame")
        chunks.extend(chunk)
    return bytes(chunks)


if __name__ == "__main__":
    main()
