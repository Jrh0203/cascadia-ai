"""Long-lived grouped inference service for the public beam set ranker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import BinaryIO

import mlx.core as mx
import numpy as np

from cascadia_mlx.action_ranking_dataset import (
    _ACTION_POSITION_DTYPE,
    ACTION_POSITION_RECORD_SIZE,
    decode_action_position_bytes,
)
from cascadia_mlx.action_ranking_serve import (
    MESSAGE_ACTION_RANKING_PREDICTION,
    MESSAGE_PREDICT_ACTION_RANKING,
)
from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.public_beam_set_model import (
    PublicBeamSetModelConfig,
    PublicBeamSetRanker,
)
from cascadia_mlx.public_beam_set_promote import load_promoted_public_beam_set_model
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


def serve_public_beam_set(
    model: PublicBeamSetRanker,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
) -> None:
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
            if message_type != MESSAGE_PREDICT_ACTION_RANKING:
                raise ProtocolError(f"unsupported message type {message_type}")
            if count == 0 or count > MAX_BATCH:
                raise ProtocolError(f"invalid prediction batch size {count}")
            payload = _read_exact(input_stream, count * ACTION_POSITION_RECORD_SIZE)
            batch, action_features = decode_action_position_bytes(payload, count)
            records = np.frombuffer(payload, dtype=_ACTION_POSITION_DTYPE, count=count)
            immediate_score = mx.array(
                records["action"]["immediate_score"].astype(np.float32)[None, :]
            )
            candidate_mask = mx.ones((1, count), dtype=mx.bool_)
            scores = model(
                batch.board_entities[None, ...],
                batch.board_mask[None, ...],
                batch.market_entities[None, ...],
                batch.market_mask[None, ...],
                batch.global_features[None, ...],
                action_features[None, ...],
                immediate_score,
                candidate_mask,
            ).reshape(count)
            mx.eval(scores)
            response = np.asarray(scores, dtype="<f4").tobytes(order="C")
            output_stream.write(
                FRAME_HEADER.pack(
                    PROTOCOL_MAGIC,
                    PROTOCOL_VERSION,
                    MESSAGE_ACTION_RANKING_PREDICTION,
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
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if args.model_dir is not None:
        model = load_promoted_public_beam_set_model(args.model_dir)
        source_name = args.model_dir.name
    else:
        model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
            args.run_dir,
            pointer=args.checkpoint,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_factory=lambda values: PublicBeamSetRanker(
                PublicBeamSetModelConfig.from_dict(values)
            ),
        )
        model.eval()
        source_name = checkpoint.name
    print(
        f"cascadia-mlx public beam set service {source_name} on {mx.default_device()}",
        file=sys.stderr,
    )
    serve_public_beam_set(model, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
